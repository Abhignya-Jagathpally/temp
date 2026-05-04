"""S5g v11: Sharpen the v11.5 vs mmSYGNAL TIE finding (Caveat #1).

Caveat #1 from `r-2026-05-03-v11s5-cox-with-programs`: the TIE was reported
with a paired bootstrap CI [-0.024, +0.026] at B=1000. A reviewer can ask
two reasonable questions:

    (a) Is the bootstrap CI tight enough at B=10000 to distinguish a
        narrow LOSS / WIN from a true TIE?
    (b) Is there a *test* (not just an interval) that gives a P-value
        for "C(A) = C(B)" against "C(A) ≠ C(B)"?

For PAIRED ROC AUC, the canonical answer is DeLong-Pearson 1988. For
paired SURVIVAL C-index there is no single canonical test, but two
recognized constructions exist:

    (1) Kang/Tian/Cai 2015 (Stat Med): U-statistic decomposition of
        Harrell's C-index with a paired covariance estimator giving a
        normal-approximation z-test.
    (2) High-B paired percentile bootstrap with a proper two-sided
        empirical P-value: P_emp = 2 * min(P(Δ ≤ 0), P(Δ ≥ 0)).

This script implements BOTH and reports them side-by-side, plus
per-stratum versions, on the corrected 6-submodel mmSYGNAL reference
from `r-2026-05-03-v11s5-mmsygnal-complete-routing`.

Approach (1) — U-statistic z-test for paired C-index difference:

    Let U(i,j) = 1{t_i < t_j AND event_i = 1} indicate i is a comparable
    pair-leader (a usable pair).
    Let φ_A(i,j) = 1{η_A(i) > η_A(j)} - 0.5 * 1{η_A(i) = η_A(j)} on
    comparable pairs only (defined 0 elsewhere).
    Then 2*(C_A - C_B) is a U-statistic of pairs (i,j).

    The Hájek projection onto patient i gives the IF (influence
    function) component for patient i. The paired variance is then
    Var(C_A - C_B) ≈ (4/N) * Var_i(IF_i^A - IF_i^B) under standard
    U-statistic theory, where Var_i is the empirical variance over
    patients i.

Approach (2) — B=10000 paired percentile bootstrap:

    Resample patients with replacement (marginal and within-stratum),
    compute Δ = C_A_b - C_B_b on the SAME resampled indices, and
    report 95% CI + two-sided percentile P-value at Δ = 0.

Output: paper/v8_artifacts/v11_sprint5/paired_cindex_test.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.utils import resample

warnings.filterwarnings("ignore", category=Warning)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT5_V11 = ROOT / "paper" / "v8_artifacts" / "v11_sprint5"

sys.path.insert(0, str(ROOT))
from resistancemap.observability.v11_stage_decorator import v11_stage  # noqa: E402


# ----------------- C-index influence functions -----------------

def cindex_concordance_arrays(eta: np.ndarray, t_days: np.ndarray,
                              event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-patient (numerator, denominator) accumulators that sum to
    Harrell's C-index = sum(num) / sum(den). For each ordered pair (i,j)
    with i an event-leader (event_i=1, t_i<t_j or t_j event-after-i):
        denom contributes 1 (counted once toward i)
        numer contributes 1 if eta_i > eta_j, 0.5 if tied
    We assign each pair's contribution to its event-leader i. This makes
    the arrays of length N where C = sum_i num_i / sum_i den_i.
    """
    N = len(eta)
    num = np.zeros(N, dtype=np.float64)
    den = np.zeros(N, dtype=np.float64)
    # Sort by time ascending
    order = np.argsort(t_days, kind="stable")
    t_s = t_days[order]
    e_s = event.astype(np.int64)[order]
    eta_s = eta[order]
    for k in range(N):
        if e_s[k] != 1:
            continue
        # j strictly after k
        if k + 1 >= N:
            continue
        # all j with t_j > t_k (regardless of event_j)
        future = np.where(t_s[k + 1:] > t_s[k])[0] + (k + 1)
        if len(future) == 0:
            continue
        eta_k = eta_s[k]
        eta_j = eta_s[future]
        contrib = np.where(eta_k > eta_j, 1.0,
                           np.where(eta_k == eta_j, 0.5, 0.0))
        # accumulate to leader k (in unsorted index)
        unsorted_k = order[k]
        num[unsorted_k] += contrib.sum()
        den[unsorted_k] += float(len(future))
    return num, den


def paired_cindex_z_test(
    risk_a: np.ndarray, risk_b: np.ndarray,
    t_days: np.ndarray, event: np.ndarray,
) -> dict:
    """U-statistic-based paired z-test for C-index difference.

    Builds per-patient influence-function components for C_A and C_B
    using the same comparable-pair set, then computes:
        Δ_hat = C_A - C_B
        Var(Δ_hat) via empirical variance of (IF_i^A - IF_i^B)
        z = Δ_hat / sqrt(Var)
        two-sided P = 2*(1 - Φ(|z|))
    """
    n = len(risk_a)
    # Numerator/denominator per leader for each model
    num_a, den_a = cindex_concordance_arrays(risk_a, t_days, event)
    num_b, den_b = cindex_concordance_arrays(risk_b, t_days, event)
    # Pairs are comparable for both models simultaneously by construction
    # (same event/time set), so denominators match by patient.
    assert np.allclose(den_a, den_b), "denominators must match"

    D_total = den_a.sum()
    if D_total <= 0:
        return {"error": "no comparable pairs"}
    C_A = num_a.sum() / D_total
    C_B = num_b.sum() / D_total
    delta_hat = C_A - C_B

    # Influence-function approximation (per-patient pseudo-value):
    # IF_i ≈ (num_i - C * den_i) / mean(den)
    mean_den = D_total / n
    if_a = (num_a - C_A * den_a) / mean_den
    if_b = (num_b - C_B * den_b) / mean_den
    diff_if = if_a - if_b

    var_delta = float(np.var(diff_if, ddof=1) / n)
    se_delta = float(np.sqrt(var_delta))
    if se_delta == 0:
        z, p_two = float("nan"), float("nan")
    else:
        z = float(delta_hat / se_delta)
        # Use scipy if available; otherwise erf-based
        try:
            from scipy.stats import norm
            p_two = float(2.0 * (1.0 - norm.cdf(abs(z))))
        except ImportError:
            from math import erf, sqrt
            p_two = float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2)))))

    return {
        "C_A": float(C_A),
        "C_B": float(C_B),
        "delta_hat": float(delta_hat),
        "se_delta": se_delta,
        "z": z,
        "p_two_sided": p_two,
        "n_patients": int(n),
        "n_comparable_pairs": int(D_total),
    }


# ----------------- High-B paired bootstrap with empirical P -----------------

def paired_bootstrap_high_b(
    risk_a: np.ndarray, risk_b: np.ndarray,
    t_days: np.ndarray, event: np.ndarray,
    strata: np.ndarray | None = None,
    n_boot: int = 10000, seed: int = 20260503,
) -> dict:
    """B=10000 paired bootstrap with two-sided empirical P-value at Δ=0."""
    n = len(risk_a)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        if strata is None:
            idx = resample(np.arange(n), n_samples=n, replace=True,
                           random_state=seed + b)
        else:
            chunks = []
            for s in np.unique(strata):
                in_s = np.where(strata == s)[0]
                chunks.append(resample(in_s, n_samples=len(in_s), replace=True,
                                        random_state=seed + b * 13 + (hash(str(s)) % 997)))
            idx = np.concatenate(chunks)
        ca = float(concordance_index(t_days[idx], -risk_a[idx], event[idx]))
        cb = float(concordance_index(t_days[idx], -risk_b[idx], event[idx]))
        deltas[b] = ca - cb
    p_left = float((deltas <= 0).mean())
    p_right = float((deltas >= 0).mean())
    p_two = float(2.0 * min(p_left, p_right))
    return {
        "delta_mean": float(deltas.mean()),
        "delta_median": float(np.median(deltas)),
        "ci95": [float(np.percentile(deltas, 2.5)),
                 float(np.percentile(deltas, 97.5))],
        "ci99": [float(np.percentile(deltas, 0.5)),
                 float(np.percentile(deltas, 99.5))],
        "p_a_beats_b": float((deltas > 0).mean()),
        "p_two_sided_empirical": p_two,
        "n_boot": n_boot,
        "stratified": strata is not None,
    }


# ----------------- Main pipeline -----------------

@v11_stage("s5g_paired_cindex_test")
def main() -> None:
    print("=== S5g v11: paired C-index test (DeLong-style + B=10000 bootstrap) ===")

    # Load v11.5 per-patient log-hazards
    npz = np.load(SPRINT5_V11 / "cox_per_patient_log_hazards.npz", allow_pickle=True)
    sids = list(npz["submitter_ids"])
    t_days = npz["t_days"].astype(np.float64) + 1e-3
    event = npz["event"].astype(np.int64)
    strata = np.array(npz["strata"], dtype=object)
    risk_v115 = npz["Cox_v11_routed_mmsygnal"].astype(np.float64)
    risk_richer = npz["Cox_v11_richer"].astype(np.float64)
    risk_mm4 = npz["mmsygnal_routed"].astype(np.float64)
    print(f"  loaded v11.5 log-hazards for N={len(sids)} patients")

    # Reconstruct 6-submodel mmSYGNAL reference (matches s5f result of 0.6957)
    df = pd.read_csv(PROC / "mmrf_sprint4_analysis.tsv", sep="\t")
    df = df.dropna(subset=["tt2L_days", "had_2L"]).reset_index(drop=True)
    df_ids = df["submitter_id"].tolist()
    assert df_ids == sids, "submitter_id alignment mismatch"

    sc = pd.read_csv(PROC / "mmrf_mmsygnal_per_model_scores.csv").set_index("patient_id")
    df["mm_id"] = df["submitter_id"].str.replace(r"_\d+_\w+$", "", regex=True)
    sc_aligned = sc.loc[df["mm_id"].values].reset_index(drop=True)

    # Re-derive del(1p36) and FGFR3 proxies
    from scripts.v11.s5f_mmsygnal_complete_routing import (
        derive_del1p36, derive_fgfr3_high, route_mmsygnal_with_completion,
    )
    del1p_score = derive_del1p36(sids).values
    fgfr3_z = derive_fgfr3_high(sids).values
    del1p_thresh = float(np.nanquantile(del1p_score, 0.10))
    fgfr3_thresh = float(np.nanquantile(fgfr3_z, 0.90))
    del1p_call = (del1p_score < del1p_thresh).astype(int)
    del1p_call[np.isnan(del1p_score)] = 0
    fgfr3_call = (fgfr3_z > fgfr3_thresh).astype(int)
    fgfr3_call[np.isnan(fgfr3_z)] = 0
    risk_mm6, _ = route_mmsygnal_with_completion(sc_aligned, df, del1p_call, fgfr3_call)
    print(f"  mmSYGNAL 6-submodel re-routed: n_del1p={del1p_call.sum()}, "
          f"n_fgfr3={fgfr3_call.sum()}")

    # Sanity: marginal C of v11.5 and 6-submodel mmSYGNAL
    c_v115 = float(concordance_index(t_days, -risk_v115, event))
    c_mm6 = float(concordance_index(t_days, -risk_mm6, event))
    print(f"  C(v11.5) = {c_v115:.4f}, C(mm6) = {c_mm6:.4f}, Δ = {c_v115 - c_mm6:+.4f}")

    # ===== Approach (1): U-statistic paired z-test =====
    print("\n[1/2] U-statistic paired z-test (Kang/Tian/Cai-style)...")
    z_marginal = paired_cindex_z_test(risk_v115, risk_mm6, t_days, event)
    print(f"  z = {z_marginal['z']:+.3f}, p = {z_marginal['p_two_sided']:.4f}, "
          f"SE = {z_marginal['se_delta']:.5f}")

    z_per_stratum = {}
    for s in sorted(np.unique(strata)):
        in_s = strata == s
        if in_s.sum() < 20:
            continue
        z_per_stratum[str(s)] = paired_cindex_z_test(
            risk_v115[in_s], risk_mm6[in_s], t_days[in_s], event[in_s],
        )
        z_per_stratum[str(s)]["n"] = int(in_s.sum())

    # ===== Approach (2): B=10000 paired bootstrap =====
    print("\n[2/2] B=10000 paired bootstrap...")
    t0 = time.time()
    boot_marginal = paired_bootstrap_high_b(
        risk_v115, risk_mm6, t_days, event, strata=None, n_boot=10000)
    print(f"  marginal: Δ={boot_marginal['delta_mean']:+.4f} "
          f"95%CI={boot_marginal['ci95']} "
          f"99%CI={boot_marginal['ci99']} "
          f"p_two={boot_marginal['p_two_sided_empirical']:.4f} "
          f"({time.time() - t0:.0f}s)")

    t0 = time.time()
    boot_strat = paired_bootstrap_high_b(
        risk_v115, risk_mm6, t_days, event, strata=strata, n_boot=10000)
    print(f"  stratified: Δ={boot_strat['delta_mean']:+.4f} "
          f"95%CI={boot_strat['ci95']} "
          f"99%CI={boot_strat['ci99']} "
          f"p_two={boot_strat['p_two_sided_empirical']:.4f} "
          f"({time.time() - t0:.0f}s)")

    # ===== Combined verdict =====
    verdict = "TIE" if (z_marginal["p_two_sided"] >= 0.05
                        and boot_marginal["p_two_sided_empirical"] >= 0.05) else "REJECT_TIE"
    print(f"\nverdict (TIE iff both p ≥ 0.05): {verdict}")

    # Save
    SPRINT5_V11.mkdir(parents=True, exist_ok=True)
    out = {
        "design": (
            "Paired C-index test for v11.5 (Cox_v11_routed_mmsygnal) vs the "
            "Caveat-#3-corrected 6-submodel mmSYGNAL reference. Two methods: "
            "(1) U-statistic-based normal-approximation z-test using per-patient "
            "Hájek-projection influence functions; (2) B=10000 paired percentile "
            "bootstrap with two-sided empirical P-value at Δ=0, marginal + "
            "stratified."
        ),
        "n_patients": len(sids),
        "n_events": int(event.sum()),
        "C_v11_5_marginal": c_v115,
        "C_mmsygnal_6submodel_marginal": c_mm6,
        "delta_marginal": c_v115 - c_mm6,
        "u_statistic_test": {
            "marginal": z_marginal,
            "per_stratum": z_per_stratum,
        },
        "paired_bootstrap_B10000": {
            "marginal": boot_marginal,
            "stratified": boot_strat,
        },
        "verdict_marginal_tie": verdict,
        "honest_caveats": [
            "U-statistic z-test uses Hájek-projection IFs; it is asymptotically "
            "valid but does not adjust for the discrete distribution of small "
            "C-index differences in finite samples.",
            "Both methods test marginal Δ_C only. Per-stratum z-tests are "
            "reported but small-stratum power is limited.",
            "All comparisons hold N=787 patients, same event/time, same "
            "submitter_id alignment as r-2026-05-03-v11s5-mmsygnal-complete-routing.",
        ],
    }
    out_path = SPRINT5_V11 / "paired_cindex_test.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
