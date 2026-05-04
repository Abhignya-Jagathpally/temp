"""S5c v11: Discrimination metrics (C-index, AUROC at 12mo / 24mo) on top of
v11 Sprint 5 jackknife+ point predictions.

Sprint 5 v10/v11 reports calibrated COVERAGE (F_S5: 5/5 strata within ±3% of
nominal 0.90). Coverage is calibration; it does NOT measure discrimination.
A perfectly-calibrated model can still be useless for clinical decision-making
if its risk ordering is no better than chance.

mmSYGNAL ([PMID 40169765](https://pubmed.ncbi.nlm.nih.gov/40169765/),
Murie/Baliga 2025 BJC) — the MM-specific multi-omic ML model identified as
the v11 head-to-head benchmark per `paper/v8_artifacts/sota_comparison.md`
§0.2 — reports PFS C-index across 5 cohorts. To position v11 in the same
metric space, we add discrimination evaluation here.

Three base learners under the same Mondrian jackknife+ wrapper as
`s5_v11_conformal_waddington_features.py`. For each, we record the per-
patient leave-one-out point prediction ŷ_{−i}(x_i) (log time-to-2L) and
compute:

    C-index_Harrell  : concordance(−ŷ, t_observed, event); higher = better
    AUROC_12mo       : binary progression by day 365 — restrict to
                       patients with follow-up ≥ 365d OR event before 365d
    AUROC_24mo       : same at day 730

The C-index is the canonical metric for survival-prediction discrimination
on right-censored data (Harrell 1996); it is what mmSYGNAL reports.

Output: paper/v8_artifacts/v11_sprint5/discrimination_metrics.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT5_V11 = ROOT / "paper" / "v8_artifacts" / "v11_sprint5"
SPRINT5_V11.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.scalar_potential import (  # noqa: E402
    ScalarPotential, ScalarPotentialConfig,
)
from resistancemap.landscape.neural_ode_flow import (  # noqa: E402
    NeuralODEFlow, NeuralODEFlowConfig,
)
from scripts.v11.s5_v11_conformal_waddington_features import (  # noqa: E402
    CONFOUNDERS_V10, assign_strata, compute_waddington_features,
)


def jackknife_predictions(
    X: np.ndarray, y: np.ndarray, model_factory,
) -> np.ndarray:
    """Return ŷ_{−i}(x_i) — per-train-point leave-one-out point prediction.

    This is the LOO point estimate (NOT the conformal interval); we use it
    for survival-discrimination metrics that take a per-patient risk score.
    """
    n = len(X)
    yhat = np.empty(n, dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        m = model_factory()
        m.fit(X[mask], y[mask])
        yhat[i] = float(m.predict(X[i:i+1])[0])
        if (i + 1) % 200 == 0:
            print(f"    LOO {i+1}/{n}")
    return yhat


def discrimination_summary(
    yhat_log: np.ndarray, t_obs_days: np.ndarray, event: np.ndarray,
    strata: np.ndarray, label: str,
) -> dict:
    """C-index per stratum + marginal; AUROC at 12mo / 24mo.

    Risk score = −ŷ_log; higher = sooner progression.
    """
    risk = -yhat_log
    out = {"label": label, "per_stratum": {}, "marginal": {}}

    # C-index per stratum
    for s in sorted(np.unique(strata)):
        in_s = strata == s
        if in_s.sum() < 10:
            continue
        c_s = float(concordance_index(t_obs_days[in_s], -risk[in_s], event[in_s]))
        out["per_stratum"][s] = {"n": int(in_s.sum()), "c_index": c_s}
    out["marginal"]["c_index"] = float(concordance_index(t_obs_days, -risk, event))

    # AUROC at 12mo (365d) and 24mo (730d). Patients censored before the
    # horizon are excluded for that horizon (standard cumulative-incidence
    # binary AUC).
    for label_h, horizon in [("AUROC_12mo", 365), ("AUROC_24mo", 730)]:
        usable = (event == 1) | (t_obs_days >= horizon)
        if usable.sum() < 30 or (event[usable] == 1).sum() < 5:
            out["marginal"][label_h] = None
            continue
        y_bin = ((t_obs_days <= horizon) & (event == 1)).astype(int)[usable]
        if len(np.unique(y_bin)) < 2:
            out["marginal"][label_h] = None
            continue
        out["marginal"][label_h] = float(roc_auc_score(y_bin, risk[usable]))
    return out


def main() -> None:
    print("=== S5c v11: Discrimination metrics (C-index, AUROC) on jackknife+ point preds ===")
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
    t_days = df["tt2L_days"].values.astype(np.float64)
    event = df["had_2L"].values.astype(np.int64)
    y = np.log(t_days + 1.0)

    print(f"N={len(df)}; events={int(event.sum())} ({100*event.mean():.1f}% had 2L); "
          f"median tt2L={np.median(t_days):.0f}d")

    X_v10 = np.column_stack([
        df["bort_1L"].values, df["M_seed3"].values,
        df["bort_1L"].values * df["M_seed3"].values,
        *[df[c].values for c in CONFOUNDERS_V10],
    ]).astype(np.float64)
    X_wadd = compute_waddington_features(z0_pat, T=1.0)
    X_wadd_std = (X_wadd - X_wadd.mean(axis=0)) / (X_wadd.std(axis=0, ddof=0) + 1e-9)
    X_v11 = np.column_stack([X_v10, X_wadd_std])

    print(f"Feature dims: v10={X_v10.shape[1]}, v11={X_v11.shape[1]}\n")

    runs = []
    print("=== JK+ Ridge_v10 (point predictions) ===")
    t0 = time.time()
    yhat_a = jackknife_predictions(X_v10, y, lambda: Ridge(alpha=1.0))
    print(f"  done in {time.time()-t0:.1f}s")
    runs.append(("Ridge_v10", yhat_a))

    print("\n=== JK+ Ridge_v11features (point predictions) ===")
    t0 = time.time()
    yhat_b = jackknife_predictions(X_v11, y, lambda: Ridge(alpha=1.0))
    print(f"  done in {time.time()-t0:.1f}s")
    runs.append(("Ridge_v11features", yhat_b))

    print("\n=== JK+ GBM_v10 (point predictions) ===")
    t0 = time.time()
    yhat_c = jackknife_predictions(X_v10, y, lambda: HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=20, random_state=0,
    ))
    print(f"  done in {time.time()-t0:.1f}s")
    runs.append(("GBM_v10", yhat_c))

    print("\n=== JK+ GBM_v11features (Waddington + flexible base) ===")
    t0 = time.time()
    yhat_d = jackknife_predictions(X_v11, y, lambda: HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=20, random_state=0,
    ))
    print(f"  done in {time.time()-t0:.1f}s")
    runs.append(("GBM_v11features", yhat_d))

    # Discrimination summary
    summary = {}
    for label, yhat in runs:
        summary[label] = discrimination_summary(yhat, t_days, event, strata, label)

    # Print table
    print("\n" + "=" * 92)
    print(f"{'Learner':22s} | {'marg C':>7s} | "
          f"{'S1':>7s} {'S2':>7s} {'S3':>7s} {'S4':>7s} {'S5':>7s} | "
          f"{'AUC12':>6s} {'AUC24':>6s}")
    print("-" * 92)
    for label, _ in runs:
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

    out = {
        "design": ("Discrimination metrics (Harrell C-index, AUROC at 12 / 24 "
                   "months) on jackknife+ leave-one-out point predictions for "
                   "the 4 base learners (Ridge_v10, Ridge_v11features, "
                   "GBM_v10, GBM_v11features). Risk = −ŷ_log."),
        "n_total": int(len(df)),
        "n_events": int(event.sum()),
        "event_rate": float(event.mean()),
        "median_tt2L_days": float(np.median(t_days)),
        "strata_sizes": {s: int(np.sum(strata == s)) for s in np.unique(strata)},
        "learners": summary,
    }
    out_path = SPRINT5_V11 / "discrimination_metrics.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved → {out_path}")

    print("\nReference comparator: mmSYGNAL (PMID 40169765, Murie/Baliga 2025 BJC):")
    print("  reports MM-PFS C-index across 5 independent cohorts (1,367 patients).")
    print("  Direct head-to-head requires running mmSYGNAL on overlapping MMRF")
    print("  patients; this script does NOT execute that head-to-head, only")
    print("  produces v11 discrimination numbers in the comparable metric space.")


if __name__ == "__main__":
    from resistancemap.observability.v11_stage_decorator import v11_stage  # noqa: E402

    with v11_stage("s5c_discrimination_metrics"):
        main()
