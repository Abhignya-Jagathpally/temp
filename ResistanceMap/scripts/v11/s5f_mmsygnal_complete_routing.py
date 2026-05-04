"""S5f v11: Complete the mmSYGNAL routing with derived del(1p36) and FGFR3 calls.

The §3.7 head-to-head benchmark (`r-2026-05-03-v11s5-mmsygnal`) and the
v11.5 §3.8 follow-up (`r-2026-05-03-v11s5-cox-with-programs`) both
applied only 4 of mmSYGNAL's 6 submodels because:
    - del(1p): MMRF cytogenetics.tsv lacks a 1p36 column.
    - FGFR3:   no RNA-seq subtype call available.

Both omissions UNDERESTIMATE mmSYGNAL — i.e., the 0.694 reference and the
v11.5 paired Δ = +0.002 [TIE] are measured against an undercredited
SOTA. This script derives both subtype calls from data we already have:

    del(1p36): mean Segment_Mean over chr1 positions 1-30 Mb from
               data/raw/mmrf_commpass/copy_number.tsv. Bottom-decile
               threshold over the 787-patient cohort → del(1p36)+ call
               (target ~10% prevalence per Manier 2017 / Walker 2018).
    FGFR3+:    top-decile FGFR3 baseline-expression z-score from
               data/processed/mmrf_baseline_expression.parquet
               (Wall 2021 RNA-seq proxy for the IHC-defined subtype).

The mmSYGNAL per-model scores already exist for ALL 6 submodels at
data/processed/mmrf_mmsygnal_per_model_scores.csv (Agent 2 ran the R
fitter on all of them; only routing was missing). We re-route using
the corrected subtype calls, recompute the Harrell C-index, and run a
PAIRED bootstrap Δ vs the v11.5 best (Cox_v11_routed_mmsygnal at C=0.6955)
on the SAME 787 patients via SAME-resampled-indices through both risks.

Honest caveats baked into the design:
    - Both new calls are V11-derived proxies, NOT the official MMRF FISH-
      based subtype labels. Reported with explicit "_proxy" suffix.
    - Bottom-decile / top-decile is a cohort-relative threshold, not an
      absolute clinical cutoff. Sensitivity analysis sweeps {5%, 10%, 15%}.
    - If the rerouted mmSYGNAL C-index INCREASES, the Δ vs v11+routed
      may shift from TIE to LOSS. We report whatever happens.

Output: paper/v8_artifacts/v11_sprint5/mmsygnal_complete_routing.json
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
RAW = ROOT / "data" / "raw" / "mmrf_commpass"
SPRINT5_V11 = ROOT / "paper" / "v8_artifacts" / "v11_sprint5"


def derive_del1p36(submitter_ids: list[str]) -> pd.Series:
    """Per-patient mean log2-Segment_Mean over chr1 positions 1-30 Mb.
    Lower = more deletion. Returned as a continuous score (caller thresholds).
    """
    print("[del1p36] Loading copy_number.tsv...")
    cn = pd.read_csv(RAW / "copy_number.tsv", sep="\t", low_memory=False)
    chr1 = cn[cn["Chromosome"].astype(str).isin(["chr1", "1"])].copy()
    chr1["Start"] = chr1["Start"].astype(np.int64)
    chr1["End"] = chr1["End"].astype(np.int64)
    p36 = chr1[(chr1["End"] >= 1) & (chr1["Start"] <= 30_000_000)].copy()
    p36["bp_in_p36"] = (
        np.minimum(p36["End"], 30_000_000) - np.maximum(p36["Start"], 1)
    ).clip(lower=0)
    p36 = p36[p36["bp_in_p36"] > 0]
    # weighted mean Segment_Mean over the 1p36 region per patient
    p36["weighted"] = p36["Segment_Mean"] * p36["bp_in_p36"]
    grp = p36.groupby("submitter_id").agg(
        bp=("bp_in_p36", "sum"),
        wsum=("weighted", "sum"),
    )
    grp["mean_segmean_1p36"] = grp["wsum"] / grp["bp"].replace(0, np.nan)
    print(f"[del1p36] {grp.shape[0]} patients with chr1 1p36 coverage; "
          f"mean={grp['mean_segmean_1p36'].mean():+.3f}, "
          f"std={grp['mean_segmean_1p36'].std():.3f}")

    # patient ID alignment (analysis ID may be like MMRF_2240_1_BM; CNV file
    # likely uses MMRF_2240). We do prefix match below.
    # build short_id -> mean_segmean dict
    short_to_score = {}
    for sid, row in grp.iterrows():
        short = "_".join(str(sid).split("_")[:2])
        if short not in short_to_score or pd.isna(short_to_score[short]):
            short_to_score[short] = float(row["mean_segmean_1p36"])
    out = pd.Series(
        [short_to_score.get("_".join(s.split("_")[:2]), np.nan)
         for s in submitter_ids],
        index=submitter_ids, name="mean_segmean_1p36",
    )
    n_have = int(out.notna().sum())
    print(f"[del1p36] mapped to analysis cohort: {n_have}/{len(submitter_ids)}")
    return out


def derive_fgfr3_high(submitter_ids: list[str]) -> pd.Series:
    """Baseline FGFR3 expression z-score per patient. Higher = FGFR3+ proxy."""
    print("[FGFR3] Loading baseline expression...")
    expr = pd.read_parquet(PROC / "mmrf_baseline_expression.parquet")
    sym_map = pd.read_csv(PROC / "mmrf_ensembl_to_symbol.tsv", sep="\t")
    sym_to_ens = dict(zip(sym_map["gene_name"], sym_map["ensembl_base"]))
    fgfr3_ens = sym_to_ens.get("FGFR3")
    if fgfr3_ens is None or fgfr3_ens not in expr.columns:
        raise RuntimeError("FGFR3 ENSG not found in expression matrix")
    fgfr3 = expr[fgfr3_ens]
    z = (fgfr3 - fgfr3.mean()) / (fgfr3.std(ddof=0) + 1e-9)
    z_dict = dict(zip(expr.index, z.values))
    out = pd.Series(
        [z_dict.get(s, np.nan) for s in submitter_ids],
        index=submitter_ids, name="fgfr3_zscore",
    )
    print(f"[FGFR3] z mean={out.mean():+.3f} std={out.std():.3f} "
          f"min={out.min():+.3f} max={out.max():+.3f}")
    return out


def route_mmsygnal_with_completion(
    scores_df: pd.DataFrame, df: pd.DataFrame,
    del1p_call: np.ndarray, fgfr3_call: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Tutorial routing A>B>C grade, mean within grade, with del1p and FGFR3 added.

    Grade A: t(4;14), FGFR3 (RNA-derived proxy)
    Grade B: amp1q, del13, del1p (CNV-derived proxy)
    Grade C: agnostic (default)
    """
    n = len(scores_df)
    routed = np.zeros(n, dtype=np.float64)
    routing_log = {"A_only": 0, "B_only": 0, "C_default": 0,
                   "A_or_B_via_proxy": 0, "no_change": 0}
    for i in range(n):
        row = scores_df.iloc[i]
        clin = df.iloc[i]
        a_scores, b_scores = [], []
        had_proxy = False
        if int(clin["cyto_t_4_14"]) == 1:
            a_scores.append(float(row["t4_14"]))
        if fgfr3_call[i] == 1:
            a_scores.append(float(row["FGFR3"]))
            had_proxy = True
        if int(clin["cyto_chr1q21_gain"]) == 1:
            b_scores.append(float(row["amp1q"]))
        if int(clin["cyto_del13q"]) == 1:
            b_scores.append(float(row["del13"]))
        if del1p_call[i] == 1:
            b_scores.append(float(row["del1p"]))
            had_proxy = True
        if a_scores:
            routed[i] = float(np.mean(a_scores))
            routing_log["A_only"] += 1
        elif b_scores:
            routed[i] = float(np.mean(b_scores))
            routing_log["B_only"] += 1
        else:
            routed[i] = float(row["agnostic"])
            routing_log["C_default"] += 1
        if had_proxy:
            routing_log["A_or_B_via_proxy"] += 1
    return routed, routing_log


def discrimination(risk: np.ndarray, t_days: np.ndarray, event: np.ndarray,
                   strata: np.ndarray) -> dict:
    out = {"per_stratum": {}, "marginal": {}}
    for s in sorted(np.unique(strata)):
        in_s = strata == s
        if in_s.sum() < 10:
            continue
        c_s = float(concordance_index(t_days[in_s], -risk[in_s], event[in_s]))
        out["per_stratum"][s] = {"n": int(in_s.sum()), "c_index": c_s}
    out["marginal"]["c_index"] = float(concordance_index(t_days, -risk, event))
    return out


def paired_bootstrap_delta(
    risk_a: np.ndarray, risk_b: np.ndarray,
    t_days: np.ndarray, event: np.ndarray,
    strata: np.ndarray | None = None,
    n_boot: int = 1000, seed: int = 20260503,
) -> dict:
    """Paired bootstrap of Δ_C = C(a) - C(b) using sklearn.utils.resample.

    If `strata` is given, resample within each stratum (stratified bootstrap).
    """
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
                                        random_state=seed + b * 13 + hash(s) % 1000))
            idx = np.concatenate(chunks)
        ca = float(concordance_index(t_days[idx], -risk_a[idx], event[idx]))
        cb = float(concordance_index(t_days[idx], -risk_b[idx], event[idx]))
        deltas[b] = ca - cb
    return {
        "delta_mean": float(deltas.mean()),
        "delta_median": float(np.median(deltas)),
        "ci95": [float(np.percentile(deltas, 2.5)),
                 float(np.percentile(deltas, 97.5))],
        "p_a_beats_b": float((deltas > 0).mean()),
        "n_boot": n_boot,
        "stratified": strata is not None,
    }


def main() -> None:
    print("=== S5f v11: mmSYGNAL complete routing with derived del(1p36) + FGFR3 ===")

    df = pd.read_csv(PROC / "mmrf_sprint4_analysis.tsv", sep="\t")
    df = df.dropna(subset=["tt2L_days", "had_2L"]).reset_index(drop=True)

    # Cyto strata reuse (assign_strata is in s5_v11_conformal_waddington_features)
    sys.path.insert(0, str(ROOT))
    from scripts.v11.s5_v11_conformal_waddington_features import assign_strata
    strata = assign_strata(df)
    t_days = df["tt2L_days"].values.astype(np.float64) + 1e-3
    event = df["had_2L"].values.astype(np.int64)

    submitter_ids = df["submitter_id"].tolist()

    # Derive subtype proxies
    del1p_score = derive_del1p36(submitter_ids).values
    fgfr3_z     = derive_fgfr3_high(submitter_ids).values

    # mmSYGNAL per-model scores
    sc = pd.read_csv(PROC / "mmrf_mmsygnal_per_model_scores.csv").set_index("patient_id")
    df["mm_id"] = df["submitter_id"].str.replace(r"_\d+_\w+$", "", regex=True)
    sc_aligned = sc.loc[df["mm_id"].values].reset_index(drop=True)
    print(f"  per-model scores aligned: {sc_aligned.shape}")

    # Pre-existing v11.5 routed (4 submodels) for baseline comparison
    sys.path.insert(0, str(ROOT))
    from scripts.v11.s5e_cox_with_programs import route_mmsygnal as route_4submodels
    routed_4 = route_4submodels(sc_aligned, df)
    c4 = float(concordance_index(t_days, -routed_4, event))
    print(f"\nbaseline 4-submodel routing: marginal C = {c4:.4f}")

    # Sweep cutoff thresholds for the new proxies
    sweep_results = []
    for q_del1p, q_fgfr3 in [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85), (0.20, 0.80)]:
        del1p_thresh = float(np.nanquantile(del1p_score, q_del1p))
        fgfr3_thresh = float(np.nanquantile(fgfr3_z, q_fgfr3))
        del1p_call = (del1p_score < del1p_thresh).astype(int)
        # NaN handling: if missing, treat as 0
        del1p_call[np.isnan(del1p_score)] = 0
        fgfr3_call = (fgfr3_z > fgfr3_thresh).astype(int)
        n_del1p = int(del1p_call.sum())
        n_fgfr3 = int(fgfr3_call.sum())

        routed_6, log = route_mmsygnal_with_completion(
            sc_aligned, df, del1p_call, fgfr3_call,
        )
        disc = discrimination(routed_6, t_days, event, strata)
        sweep_results.append({
            "q_del1p_bottom_quantile": q_del1p,
            "q_fgfr3_top_quantile":     q_fgfr3,
            "n_del1p_proxy_positive": n_del1p,
            "n_fgfr3_proxy_positive": n_fgfr3,
            "routing_log": log,
            "marginal_c_index": disc["marginal"]["c_index"],
            "per_stratum_c_index": disc["per_stratum"],
        })
        print(f"\n  proxy-cutoffs: del1p<q{q_del1p*100:.0f}, fgfr3>q{q_fgfr3*100:.0f}: "
              f"n_del1p={n_del1p}, n_fgfr3={n_fgfr3}, routing={log}, "
              f"marginal C={disc['marginal']['c_index']:.4f}")

    # Pick the proxies at the literature-prevalence point (10% / 10%)
    chosen = next(r for r in sweep_results
                  if r["q_del1p_bottom_quantile"] == 0.10
                  and r["q_fgfr3_top_quantile"] == 0.90)
    print(f"\n*** chosen cutoff (lit-prevalence): del1p bot-10%, fgfr3 top-10% ***")
    print(f"    marginal C = {chosen['marginal_c_index']:.4f}  "
          f"(was {c4:.4f} with 4 submodels → Δ={chosen['marginal_c_index']-c4:+.4f})")

    # Re-do routing for the chosen cutoff (returns risk vector for paired bootstrap)
    del1p_thresh = float(np.nanquantile(del1p_score, 0.10))
    fgfr3_thresh = float(np.nanquantile(fgfr3_z, 0.90))
    del1p_call = (del1p_score < del1p_thresh).astype(int)
    del1p_call[np.isnan(del1p_score)] = 0
    fgfr3_call = (fgfr3_z > fgfr3_thresh).astype(int)
    routed_6, _ = route_mmsygnal_with_completion(
        sc_aligned, df, del1p_call, fgfr3_call,
    )

    # Paired bootstrap: 6-submodel routing vs v11.5 best (Cox_v11_routed_mmsygnal)
    print("\n=== Paired bootstrap: 6-submodel mmSYGNAL vs v11.5 Cox_v11_routed_mmsygnal ===")
    log_hr = np.load(SPRINT5_V11 / "cox_per_patient_log_hazards.npz")
    cox_v11_plus = log_hr["Cox_v11_routed_mmsygnal"]
    # Δ = C(v11.5) - C(mmSYGNAL_6submodel); positive = v11.5 wins
    b_marg = paired_bootstrap_delta(cox_v11_plus, routed_6, t_days, event,
                                     strata=None, n_boot=1000)
    b_strat = paired_bootstrap_delta(cox_v11_plus, routed_6, t_days, event,
                                     strata=strata, n_boot=1000)

    print(f"  marginal:        Δ_mean={b_marg['delta_mean']:+.4f}  "
          f"95%CI=[{b_marg['ci95'][0]:+.4f}, {b_marg['ci95'][1]:+.4f}]  "
          f"P(v11.5 better)={b_marg['p_a_beats_b']:.3f}")
    print(f"  stratified boot: Δ_mean={b_strat['delta_mean']:+.4f}  "
          f"95%CI=[{b_strat['ci95'][0]:+.4f}, {b_strat['ci95'][1]:+.4f}]  "
          f"P(v11.5 better)={b_strat['p_a_beats_b']:.3f}")

    # Save
    out = {
        "design": ("Complete mmSYGNAL routing by adding derived del(1p36) "
                   "and FGFR3 subtype calls. del(1p36) from chr1 1-30Mb "
                   "weighted mean Segment_Mean (CNV); FGFR3+ from baseline "
                   "FGFR3 z-score (RNA-seq Wall 2021 proxy). Cutoff sweep "
                   "{5,10,15,20}% per literature-prevalence range."),
        "n_total": int(len(df)),
        "baseline_4_submodel_routing_c": c4,
        "sweep": sweep_results,
        "chosen_cutoff": {
            "q_del1p_bottom": 0.10,
            "q_fgfr3_top":    0.90,
            "marginal_c_index": chosen["marginal_c_index"],
            "delta_vs_4submodel": chosen["marginal_c_index"] - c4,
        },
        "v11_5_paired_bootstrap_marginal":   b_marg,
        "v11_5_paired_bootstrap_stratified": b_strat,
        "honest_caveats": [
            "del(1p36) and FGFR3 calls are V11-derived PROXIES, not official MMRF FISH labels.",
            "Bottom-decile / top-decile is a cohort-relative threshold, not an absolute clinical cutoff.",
            "If mmSYGNAL improves under complete routing, the v11.5 head-to-head TIE may shift.",
        ],
    }
    out_path = SPRINT5_V11 / "mmsygnal_complete_routing.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    from resistancemap.observability.v11_stage_decorator import v11_stage
    with v11_stage("s5f_mmsygnal_complete_routing"):
        main()
