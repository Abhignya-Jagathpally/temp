"""S5d v11: Cox PH base learner — survival-aware discrimination.

The v10 / v11 conformal pipeline regresses log(tt2L+1) under MSE; this is
naive about right-censoring. mmSYGNAL ([PMID 40169765](https://pubmed.ncbi.nlm.nih.gov/40169765/),
Murie/Baliga 2025) uses Cox proportional hazards as its survival-aware
backbone, reporting MM-PFS C-index 0.65–0.75 across 5 cohorts on 1,367
patients. v11's Sprint 5c discrimination addendum reported 0.566 best
under MSE — still in the weakly-discriminating range.

This script tests: is the v11-vs-SOTA discrimination gap explained by the
*loss function* (MSE-on-log-time vs Cox partial likelihood) or by the
*features* (cyto+clinical+Waddington vs mmSYGNAL's transcriptional-program
activities)?

Three Cox PH variants under leave-one-out:

    Cox_v10           — Cox on the 11 v10 features (treatment + mediator +
                        interaction + ISS + age + gender + 5 cyto)
    Cox_v11features   — Cox on 11 + 4 Waddington features
    Cox_v11_richer    — Cox on 11 + 4 Waddington + 8 expression PCs
                        (top-8 PCs of the MMRF z₀ baseline encoder, which
                        carry the broad transcriptomic signature beyond
                        the focused proteasome mediator)

For each variant: per-patient LOO log-hazard prediction, then Harrell C-index
per stratum + marginal + AUROC at 12mo / 24mo.

Output: paper/v8_artifacts/v11_sprint5/cox_discrimination.json
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

warnings.filterwarnings("ignore", category=Warning)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT5_V11 = ROOT / "paper" / "v8_artifacts" / "v11_sprint5"
SPRINT5_V11.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from scripts.v11.s5_v11_conformal_waddington_features import (  # noqa: E402
    CONFOUNDERS_V10, assign_strata, compute_waddington_features,
)


def cox_loo_partial_hazards(
    df_features: pd.DataFrame, t_col: str, e_col: str, penalizer: float = 0.1,
) -> np.ndarray:
    """LOO Cox PH: returns per-patient log-partial-hazard fit on the held-out N-1.

    df_features must have only feature columns + t_col + e_col.
    """
    n = len(df_features)
    feature_cols = [c for c in df_features.columns if c not in (t_col, e_col)]
    log_hr = np.empty(n, dtype=np.float64)
    for i in range(n):
        train = df_features.drop(index=i)
        test = df_features.loc[[i]]
        cph = CoxPHFitter(penalizer=penalizer)
        try:
            cph.fit(train, duration_col=t_col, event_col=e_col, show_progress=False)
            # log-hazard ratio = X·β; lifelines returns linear-predictor via
            # `predict_log_partial_hazard` (without baseline)
            lp = float(cph.predict_log_partial_hazard(test[feature_cols]).iloc[0])
            log_hr[i] = lp
        except Exception as e:
            log_hr[i] = 0.0
        if (i + 1) % 100 == 0:
            print(f"    Cox LOO {i+1}/{n}")
    return log_hr


def discrimination(risk: np.ndarray, t_days: np.ndarray, event: np.ndarray,
                   strata: np.ndarray) -> dict:
    """C-index (per-stratum + marginal) + AUROC at 12mo / 24mo. Risk = log-hazard."""
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


def main() -> None:
    print("=== S5d v11: Cox PH discrimination — survival-aware base learner ===")
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

    strata = assign_strata(df)
    t_days = df["tt2L_days"].values.astype(np.float64) + 1e-3  # avoid t=0
    event = df["had_2L"].values.astype(np.int64)
    print(f"N={len(df)}; events={int(event.sum())} ({100*event.mean():.1f}%); "
          f"median tt2L={np.median(t_days):.0f}d")

    # v10 11-feature matrix
    base_features_v10 = pd.DataFrame({
        "bort_1L": df["bort_1L"].astype(float),
        "M_seed3": df["M_seed3"].astype(float),
        "bort_x_M": (df["bort_1L"] * df["M_seed3"]).astype(float),
        **{c: df[c].astype(float) for c in CONFOUNDERS_V10},
    })

    # Standardize Waddington features (Cox is regularized; scale matters)
    X_wadd = compute_waddington_features(z0_pat, T=1.0)
    X_wadd_std = (X_wadd - X_wadd.mean(axis=0)) / (X_wadd.std(axis=0, ddof=0) + 1e-9)
    wadd_features = pd.DataFrame(X_wadd_std,
                                  columns=["U_z0", "U_zT", "grad_norm", "displacement"])

    # 8 PCs of the z₀ latent (PCA on 64-dim → 8-dim) — additional broad-
    # transcriptomic signal beyond the focused proteasome mediator
    Zc = z0_pat - z0_pat.mean(axis=0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    z_pcs = (Zc @ Vt[:8].T)
    z_pcs = (z_pcs - z_pcs.mean(axis=0)) / (z_pcs.std(axis=0, ddof=0) + 1e-9)
    pc_features = pd.DataFrame(z_pcs, columns=[f"z_pc{k+1}" for k in range(8)])

    # Three Cox variants
    df_v10 = pd.concat([base_features_v10,
                       pd.DataFrame({"t": t_days, "e": event})], axis=1)
    df_v11 = pd.concat([base_features_v10, wadd_features,
                       pd.DataFrame({"t": t_days, "e": event})], axis=1)
    df_v11_richer = pd.concat([base_features_v10, wadd_features, pc_features,
                              pd.DataFrame({"t": t_days, "e": event})], axis=1)

    print(f"Cox feature counts: v10={base_features_v10.shape[1]}, "
          f"v11features={base_features_v10.shape[1]+wadd_features.shape[1]}, "
          f"v11_richer={df_v11_richer.shape[1]-2}")

    summary = {}
    for label, df_x in [("Cox_v10", df_v10),
                        ("Cox_v11features", df_v11),
                        ("Cox_v11_richer", df_v11_richer)]:
        print(f"\n=== {label} (LOO Cox PH) ===")
        t0 = time.time()
        log_hr = cox_loo_partial_hazards(df_x, "t", "e", penalizer=0.1)
        print(f"  done in {time.time()-t0:.1f}s")
        summary[label] = discrimination(log_hr, t_days, event, strata)
        m = summary[label]["marginal"]
        print(f"  Marginal: C={m['c_index']:.3f}  AUC12={m.get('AUROC_12mo')}  AUC24={m.get('AUROC_24mo')}")

    # Print comparison
    print("\n" + "=" * 92)
    print(f"{'Learner':22s} | {'marg C':>7s} | "
          f"{'S1':>7s} {'S2':>7s} {'S3':>7s} {'S4':>7s} {'S5':>7s} | "
          f"{'AUC12':>6s} {'AUC24':>6s}")
    print("-" * 92)
    for label in ("Cox_v10", "Cox_v11features", "Cox_v11_richer"):
        s = summary[label]
        m = s["marginal"]
        ps = s["per_stratum"]
        def _f(d, k): return f"{d[k]:.3f}" if k in d and d[k] is not None else "  -- "
        print(f"{label:22s} | {m['c_index']:>7.3f} | "
              f"{_f(ps.get('S1_del17p',{}), 'c_index'):>7s} "
              f"{_f(ps.get('S2_t_4_14',{}), 'c_index'):>7s} "
              f"{_f(ps.get('S3_1q21',{}), 'c_index'):>7s} "
              f"{_f(ps.get('S4_t_11_14',{}), 'c_index'):>7s} "
              f"{_f(ps.get('S5_other',{}), 'c_index'):>7s} | "
              f"{_f(m, 'AUROC_12mo'):>6s} {_f(m, 'AUROC_24mo'):>6s}")
    print("=" * 92)

    print("\nReference (Sprint 5c MSE-loss base learners):")
    print("  Ridge_v10              C=0.537  AUC12=0.530  AUC24=0.592")
    print("  Ridge_v11features      C=0.556  AUC12=0.593  AUC24=0.610  ← best calibrated")
    print("  GBM_v10                C=0.546  AUC12=0.533  AUC24=0.581")
    print("  GBM_v11features        C=0.566  AUC12=0.623  AUC24=0.598  ← best discrimination")
    print("\nReference (mmSYGNAL — PMID 40169765, Murie/Baliga 2025 BJC):")
    print("  MM-PFS C-index 0.65–0.75 across 5 cohorts (1,367 patients).")

    out = {
        "design": ("Cox PH LOO log-partial-hazard predictions, lifelines "
                   "CoxPHFitter penalizer=0.1; 3 feature variants; risk "
                   "score = log-hazard; Harrell C-index per stratum + marginal."),
        "n_total": int(len(df)),
        "n_events": int(event.sum()),
        "median_tt2L_days": float(np.median(t_days)),
        "feature_counts": {
            "v10": int(base_features_v10.shape[1]),
            "v11features": int(base_features_v10.shape[1] + wadd_features.shape[1]),
            "v11_richer": int(df_v11_richer.shape[1] - 2),
        },
        "results": summary,
        "v11_mse_reference_c_index_marginal": {
            "Ridge_v10": 0.537, "Ridge_v11features": 0.556,
            "GBM_v10": 0.546, "GBM_v11features": 0.566,
        },
        "mmSYGNAL_reference_c_index_range": [0.65, 0.75],
    }
    out_path = SPRINT5_V11 / "cox_discrimination.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    from resistancemap.observability.v11_stage_decorator import v11_stage  # noqa: E402

    with v11_stage("s5d_cox_discrimination"):
        main()
