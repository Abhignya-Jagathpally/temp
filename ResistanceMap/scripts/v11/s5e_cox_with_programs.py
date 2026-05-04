"""S5e v11: Close the +0.040 mmSYGNAL gap by adding transcriptional-program features to v11 Cox.

The Sprint 5 head-to-head (`r-2026-05-03-v11s5-mmsygnal`) showed mmSYGNAL
beats v11 Cox best by Δ = +0.040 marginal C-index on the same MMRF N=787
cohort (mmSYGNAL 0.694 [0.658-0.729] vs Cox_v11_richer 0.654). The agent
explicitly noted the residual gap is feature-set, not loss-function or
architecture.

This script tests three feature-augmentation paths to close that gap:

    Cox_v11_richer                — baseline reference (existing
                                    11 + 4 Waddington + 8 z-PCs = 23 feats)
    Cox_v11_routed_mmsygnal       — v11_richer + mmSYGNAL routed risk score
                                    (single-feature drop-in, 24 feats)
    Cox_v11_six_mmsygnal_scores   — v11_richer + 6 raw mmSYGNAL submodel
                                    scores (agnostic, amp1q, del13, del1p,
                                    t4_14, FGFR3) → 29 feats
    Cox_only_six_mmsygnal_scores  — Cox on the 6 mmSYGNAL scores ONLY
                                    (no v11 features) → upper bound for
                                    "what the program-activity-derived risk
                                    alone is worth"
    Cox_v11_program_activity_pcs  — v11_richer + top-10 PCs of the IA12
                                    program-activity matrix (141 programs)
                                    → 33 feats (programs as raw feature axis)

Each runs LOO Cox with lifelines CoxPHFitter penalizer=0.1; per-patient
log-hazards are SAVED so the next sprint can compute paired bootstrap
Δ-CI vs mmSYGNAL on the same patients. Output: per-stratum + marginal
C-index + AUROC@12mo / 24mo for each variant; head-to-head Δ vs
mmSYGNAL = 0.694.

Bootstrap resampling uses sklearn.utils.resample on real patient indices
(no synthetic data generation; matches `paper/v8_artifacts/v11_sprint5/mmsygnal/bootstrap_ci.py`
zero-trust pattern).

Inputs (real data, no fabrication):
    data/processed/mmrf_sprint4_analysis.tsv      — outcome + cyto
    data/processed/mmrf_baseline_expression.parquet — proteasome mediator
    data/processed/mmrf_z64.npy                    — Sprint 1 latents
    data/processed/mmrf_mmsygnal_program_activity.csv (787 × 141 programs)
    data/processed/mmrf_mmsygnal_per_model_scores.csv (787 × 6 models)
    checkpoints/u_theta_v10s1.pt                   — for Waddington features

Output:
    paper/v8_artifacts/v11_sprint5/cox_with_programs.json
    paper/v8_artifacts/v11_sprint5/cox_per_patient_log_hazards.npz
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

warnings.filterwarnings("ignore", category=Warning)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT5_V11 = ROOT / "paper" / "v8_artifacts" / "v11_sprint5"
SPRINT5_V11.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from scripts.v11.s5_v11_conformal_waddington_features import (  # noqa: E402
    CONFOUNDERS_V10, assign_strata, compute_waddington_features,
)


def cox_loo(df_features: pd.DataFrame, t_col: str, e_col: str,
            penalizer: float = 0.1) -> np.ndarray:
    n = len(df_features)
    feature_cols = [c for c in df_features.columns if c not in (t_col, e_col)]
    log_hr = np.empty(n, dtype=np.float64)
    n_failed = 0
    for i in range(n):
        train = df_features.drop(index=i)
        test = df_features.loc[[i]]
        cph = CoxPHFitter(penalizer=penalizer)
        try:
            cph.fit(train, duration_col=t_col, event_col=e_col, show_progress=False)
            log_hr[i] = float(cph.predict_log_partial_hazard(test[feature_cols]).iloc[0])
        except Exception:
            log_hr[i] = 0.0
            n_failed += 1
        if (i + 1) % 100 == 0:
            print(f"    Cox LOO {i+1}/{n}  (failed: {n_failed})")
    return log_hr


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
    for label_h, horizon in [("AUROC_12mo", 365), ("AUROC_24mo", 730)]:
        usable = (event == 1) | (t_days >= horizon)
        if usable.sum() < 30 or (event[usable] == 1).sum() < 5:
            out["marginal"][label_h] = None
            continue
        y_bin = ((t_days <= horizon) & (event == 1)).astype(int)[usable]
        if len(np.unique(y_bin)) < 2:
            out["marginal"][label_h] = None
            continue
        out["marginal"][label_h] = float(roc_auc_score(y_bin, risk[usable]))
    return out


def paired_bootstrap_delta_c(
    risk_a: np.ndarray, risk_b: np.ndarray,
    t_days: np.ndarray, event: np.ndarray,
    n_boot: int = 1000, seed: int = 20260503,
) -> dict:
    """Paired bootstrap of Δ_C = C(a) − C(b) using sklearn.utils.resample
    on real patient indices. Both risk vectors are evaluated on the SAME
    resampled patients per draw, so the Δ is paired (vs the unpaired
    bootstrap_ci.py from the mmSYGNAL agent run)."""
    n = len(risk_a)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = resample(np.arange(n), n_samples=n, replace=True, random_state=seed + b)
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
    }


def route_mmsygnal(scores_df: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """Tutorial routing A>B>C grade, mean within grade.

    Per Murie 2025 supplement and the v11s5-mmsygnal head-to-head:
    Grade A: t(4;14) and FGFR3 (high-risk subtype)
    Grade B: amp1q, del13, del1p (intermediate)
    Grade C: agnostic (default)
    Apply only the models matching the patient's actual cyto status; if
    none of A or B applies, fall back to agnostic.
    """
    n = len(scores_df)
    routed = np.zeros(n, dtype=np.float64)
    for i in range(n):
        row = scores_df.iloc[i]
        clin = df.iloc[i]
        a_scores, b_scores = [], []
        if int(clin["cyto_t_4_14"]) == 1:
            a_scores.append(float(row["t4_14"]))
        # FGFR3 model: skipped per agent (no RNA-seq subtype call available)
        if int(clin["cyto_chr1q21_gain"]) == 1:
            b_scores.append(float(row["amp1q"]))
        if int(clin["cyto_del13q"]) == 1:
            b_scores.append(float(row["del13"]))
        # del1p: MMRF panel lacks 1p36 column; skipped per agent.
        if a_scores:
            routed[i] = float(np.mean(a_scores))
        elif b_scores:
            routed[i] = float(np.mean(b_scores))
        else:
            routed[i] = float(row["agnostic"])
    return routed


def main() -> None:
    print("=== S5e v11: Cox with mmSYGNAL transcriptional-program features ===")
    df = pd.read_csv(PROC / "mmrf_sprint4_analysis.tsv", sep="\t")
    df = df.dropna(subset=["tt2L_days", "had_2L"]).reset_index(drop=True)

    expr = pd.read_parquet(PROC / "mmrf_baseline_expression.parquet")
    sym_map = pd.read_csv(PROC / "mmrf_ensembl_to_symbol.tsv", sep="\t")
    sym_to_ens = dict(zip(sym_map["gene_name"], sym_map["ensembl_base"]))
    seed3_ens = [sym_to_ens[g] for g in ("PSMB5", "PSMB1", "PSMB2")]
    sub = expr[seed3_ens].copy()
    sub = (sub - sub.mean()) / (sub.std(ddof=0) + 1e-9)
    M_seed3 = sub.mean(axis=1)
    df = df.merge(
        pd.DataFrame({"submitter_id": expr.index, "M_seed3": M_seed3.values}),
        on="submitter_id", how="inner",
    ).reset_index(drop=True)

    Z = np.load(PROC / "mmrf_z64.npy")
    ids = json.loads((PROC / "mmrf_z64_sample_ids.json").read_text())
    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    z_idx = np.array([id_to_idx[sid] for sid in df["submitter_id"]], dtype=np.int64)
    z0_pat = Z[z_idx]

    # Load mmSYGNAL features
    pa = pd.read_csv(PROC / "mmrf_mmsygnal_program_activity.csv")
    sc = pd.read_csv(PROC / "mmrf_mmsygnal_per_model_scores.csv")
    df_short_id = df["submitter_id"].str.replace(r"_\d+_\w+$", "", regex=True)
    df["mm_id"] = df_short_id
    pa = pa.set_index("patient_id")
    sc = sc.set_index("patient_id")
    miss_pa = sum(1 for x in df["mm_id"] if x not in pa.index)
    miss_sc = sum(1 for x in df["mm_id"] if x not in sc.index)
    print(f"mmSYGNAL alignment: program-activity missing {miss_pa}/{len(df)}, "
          f"per-model-scores missing {miss_sc}/{len(df)}")

    pa_aligned = pa.loc[df["mm_id"].values].reset_index(drop=True)
    sc_aligned = sc.loc[df["mm_id"].values].reset_index(drop=True)
    print(f"  pa_aligned {pa_aligned.shape}, sc_aligned {sc_aligned.shape}")

    strata = assign_strata(df)
    t_days = df["tt2L_days"].values.astype(np.float64) + 1e-3
    event = df["had_2L"].values.astype(np.int64)
    print(f"\nN={len(df)}; events={int(event.sum())} ({100*event.mean():.1f}%); "
          f"median tt2L={np.median(t_days):.0f}d")

    # Build feature matrices
    base_v10 = pd.DataFrame({
        "bort_1L": df["bort_1L"].astype(float),
        "M_seed3": df["M_seed3"].astype(float),
        "bort_x_M": (df["bort_1L"] * df["M_seed3"]).astype(float),
        **{c: df[c].astype(float) for c in CONFOUNDERS_V10},
    })
    X_wadd = compute_waddington_features(z0_pat, T=1.0)
    X_wadd_std = (X_wadd - X_wadd.mean(axis=0)) / (X_wadd.std(axis=0, ddof=0) + 1e-9)
    wadd = pd.DataFrame(X_wadd_std, columns=["U_z0", "U_zT", "grad_norm", "displacement"])
    Zc = z0_pat - z0_pat.mean(axis=0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    z_pcs = (Zc @ Vt[:8].T)
    z_pcs = (z_pcs - z_pcs.mean(axis=0)) / (z_pcs.std(axis=0, ddof=0) + 1e-9)
    z_pc_df = pd.DataFrame(z_pcs, columns=[f"z_pc{k+1}" for k in range(8)])

    df_v11_richer = pd.concat([base_v10, wadd, z_pc_df], axis=1)
    print(f"v11_richer feature count: {df_v11_richer.shape[1]}")

    # mmSYGNAL routed score
    routed = route_mmsygnal(sc_aligned, df)
    routed_std = (routed - routed.mean()) / (routed.std(ddof=0) + 1e-9)
    routed_feat = pd.DataFrame({"mmsygnal_routed": routed_std})

    # 6 raw mmSYGNAL scores standardized
    six = sc_aligned.copy()
    six = (six - six.mean()) / (six.std(ddof=0) + 1e-9)
    six.columns = [f"mmsygnal_{c}" for c in six.columns]

    # IA12 program-activity PCs (top-10)
    pa_mat = pa_aligned.values.astype(np.float64)
    pa_centered = pa_mat - pa_mat.mean(axis=0)
    _, _, Vt_pa = np.linalg.svd(pa_centered, full_matrices=False)
    pa_pcs = pa_centered @ Vt_pa[:10].T
    pa_pcs = (pa_pcs - pa_pcs.mean(axis=0)) / (pa_pcs.std(axis=0, ddof=0) + 1e-9)
    pa_pc_df = pd.DataFrame(pa_pcs, columns=[f"pa_pc{k+1}" for k in range(10)])

    # Five Cox variants
    variants = {}
    variants["Cox_v11_richer"] = pd.concat(
        [df_v11_richer, pd.DataFrame({"t": t_days, "e": event})], axis=1)
    variants["Cox_v11_routed_mmsygnal"] = pd.concat(
        [df_v11_richer, routed_feat, pd.DataFrame({"t": t_days, "e": event})], axis=1)
    variants["Cox_v11_six_mmsygnal_scores"] = pd.concat(
        [df_v11_richer, six, pd.DataFrame({"t": t_days, "e": event})], axis=1)
    variants["Cox_only_six_mmsygnal_scores"] = pd.concat(
        [six, pd.DataFrame({"t": t_days, "e": event})], axis=1)
    variants["Cox_v11_program_activity_pcs"] = pd.concat(
        [df_v11_richer, pa_pc_df, pd.DataFrame({"t": t_days, "e": event})], axis=1)

    summary = {}
    log_hr_per_variant = {}
    for label, df_x in variants.items():
        n_feat = df_x.shape[1] - 2
        print(f"\n=== {label} ({n_feat} feats, LOO Cox PH) ===")
        t0 = time.time()
        log_hr = cox_loo(df_x, "t", "e", penalizer=0.1)
        print(f"  done in {time.time()-t0:.1f}s")
        summary[label] = discrimination(log_hr, t_days, event, strata)
        summary[label]["n_features"] = int(n_feat)
        log_hr_per_variant[label] = log_hr
        m = summary[label]["marginal"]
        print(f"  C={m['c_index']:.4f}  AUC12={m.get('AUROC_12mo')}  AUC24={m.get('AUROC_24mo')}")

    # Paired bootstrap Δ vs mmSYGNAL routed
    print("\n=== Paired bootstrap Δ-C vs mmSYGNAL routed ===")
    risk_mmsygnal = routed
    deltas_vs_mmsygnal = {}
    for label, log_hr in log_hr_per_variant.items():
        b = paired_bootstrap_delta_c(log_hr, risk_mmsygnal, t_days, event, n_boot=1000)
        deltas_vs_mmsygnal[label] = b
        print(f"  {label:35s}  Δ_mean={b['delta_mean']:+.4f}  "
              f"95%CI=[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  "
              f"P(better)={b['p_a_beats_b']:.3f}")

    # Print comparison table
    print("\n" + "=" * 100)
    print(f"{'Learner':36s} | {'feats':>6s} | {'marg C':>7s} | "
          f"{'S1':>7s} {'S2':>7s} {'S3':>7s} {'S4':>7s} {'S5':>7s} | "
          f"{'AUC12':>6s} {'AUC24':>6s}")
    print("-" * 100)
    for label, _ in variants.items():
        s = summary[label]
        m = s["marginal"]
        ps = s["per_stratum"]
        def _f(d, k): return f"{d[k]:.3f}" if k in d and d[k] is not None else "  -- "
        print(f"{label:36s} | {s['n_features']:>6d} | {m['c_index']:>7.4f} | "
              f"{_f(ps.get('S1_del17p',{}), 'c_index'):>7s} "
              f"{_f(ps.get('S2_t_4_14',{}), 'c_index'):>7s} "
              f"{_f(ps.get('S3_1q21',{}), 'c_index'):>7s} "
              f"{_f(ps.get('S4_t_11_14',{}), 'c_index'):>7s} "
              f"{_f(ps.get('S5_other',{}), 'c_index'):>7s} | "
              f"{_f(m, 'AUROC_12mo'):>6s} {_f(m, 'AUROC_24mo'):>6s}")
    print("=" * 100)
    print(f"\nReference — mmSYGNAL routed (head-to-head, r-2026-05-03-v11s5-mmsygnal):")
    print(f"  Marginal C-index = 0.694 [95% CI 0.658-0.729]")

    # Save outputs
    out = {
        "design": ("Cox PH LOO with progressive feature augmentation: "
                   "v11_richer baseline + mmSYGNAL transcriptional-program "
                   "features (routed risk score, 6 raw model scores, "
                   "10 PCs of IA12 program activity matrix). Per-patient "
                   "log-hazards saved for paired Δ-CI vs mmSYGNAL."),
        "n_total": int(len(df)),
        "n_events": int(event.sum()),
        "median_tt2L_days": float(np.median(t_days)),
        "results": summary,
        "paired_bootstrap_delta_vs_mmsygnal_routed": deltas_vs_mmsygnal,
        "mmsygnal_reference_marginal_c_index": 0.694,
        "mmsygnal_reference_ci95": [0.658, 0.729],
    }
    out_path = SPRINT5_V11 / "cox_with_programs.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved → {out_path}")

    # Save per-patient log-hazards for downstream paired analyses
    np.savez(
        SPRINT5_V11 / "cox_per_patient_log_hazards.npz",
        submitter_ids=df["submitter_id"].values,
        t_days=t_days,
        event=event,
        strata=strata,
        mmsygnal_routed=routed,
        **{label: lh for label, lh in log_hr_per_variant.items()},
    )
    print(f"saved → {SPRINT5_V11 / 'cox_per_patient_log_hazards.npz'}")


if __name__ == "__main__":
    from resistancemap.observability.v11_stage_decorator import v11_stage
    with v11_stage("s5e_cox_with_programs"):
        main()
