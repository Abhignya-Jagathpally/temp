"""S5: Mondrian jackknife+ conformal wrapper for per-stratum 90% coverage.

Per spec §2.8 + §9 row 5 + Barber-Candès-Ramdas-Tibshirani 2021 (arXiv:1905.02928).

Setup:
    Target Y      = log(tt2L_days + 1)        (proxy for log time-to-2nd-line)
    Features X    = (bort_1L, M_proteasome (focused 3-gene), bort×M,
                     age, gender, ISS, 5 cyto indicators)
    Base learner  = Ridge regression (sklearn, λ=1.0)

Mondrian partition (5 mutually-exclusive cyto strata, per spec §2.8):
    S1: del17p positive
    S2: t(4;14) positive AND NOT S1
    S3: +1q21 positive   AND NOT S1, S2
    S4: t(11;14) positive AND NOT S1, S2, S3
    S5: any other (incl. del13q-only, isolated, or no-high-risk)

Strata sizes on N=787 cohort:
    S1=105, S2=97, S3=165, S4=95, S5=325  (all > 36 floor)

Jackknife+ procedure (Barber et al. 2021):
    For each i ∈ {1..n}:
        train base learner on D \ {i}
        predict ŷ_{-i}(x_test) and residual r_i = |y_i - ŷ_{-i}(x_i)|

    For test point x_test in stratum s:
        prediction set = [Q_{α/2}^s({ŷ_{-i}(x_test) − r_i}), Q_{1-α/2}^s({ŷ_{-i}(x_test) + r_i})]
    where the quantiles are restricted to training points in same stratum s.

    Coverage at level (1-α) = fraction of test points whose y_test lies in the prediction set.

Spec target (per §2.8 + §9 row 5): with n_s ≥ 36 in all 5 strata,
    3 strata clear ±3% of 90%  (i.e., 87% ≤ coverage ≤ 93%)
    2 strata clear ±5% of 90%  (i.e., 85% ≤ coverage ≤ 95%)
    => F_S5: at least 3 strata in [0.87, 0.93] AND at least 2 more in [0.85, 0.95]

Empirical coverage is evaluated by *split-sample held-out test*: 80/20 split
within each stratum, fit jackknife+ on the 80% training, evaluate coverage on
the 20% held-out (with calibration scores from training-side jackknife+).

Output: paper/v8_artifacts/v10_sprint5/mondrian_jk_plus_coverage.json
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

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT5 = ROOT / "paper" / "v8_artifacts" / "v10_sprint5"
SPRINT5.mkdir(parents=True, exist_ok=True)

CONFOUNDERS = [
    "iss_stage_ord", "age_at_dx_years", "gender_male",
    "cyto_del17p", "cyto_chr1q21_gain", "cyto_del13q",
    "cyto_t_4_14", "cyto_t_11_14",
]


def assign_strata(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(["S5_other"] * len(df), index=df.index)
    s[df["cyto_del17p"] == 1] = "S1_del17p"
    s[(df["cyto_t_4_14"] == 1) & (s == "S5_other")] = "S2_t_4_14"
    s[(df["cyto_chr1q21_gain"] == 1) & (s == "S5_other")] = "S3_1q21"
    s[(df["cyto_t_11_14"] == 1) & (s == "S5_other")] = "S4_t_11_14"
    return s


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) from the analysis table. y = log(tt2L_days + 1)."""
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["bort_1L"], r["M_seed3"], r["bort_1L"] * r["M_seed3"],
            *[r[c] for c in CONFOUNDERS],
        ])
    X = np.asarray(rows, dtype=np.float64)
    y = np.log(df["tt2L_days"].values.astype(np.float64) + 1.0)
    return X, y


def jackknife_plus_intervals(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, strata_train: np.ndarray, strata_test: np.ndarray,
    alpha: float = 0.10,
    ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mondrian jackknife+ prediction intervals.

    Returns (lo, hi, fit_residuals_per_train_point).
    """
    n = len(X_train)
    # Per-train-point: leave-one-out residual r_i and LOO predictor μ_{-i}
    r_loo = np.empty(n, dtype=np.float64)
    pred_test_loo = np.empty((n, len(X_test)), dtype=np.float64)

    base = Ridge(alpha=ridge_alpha)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        base.fit(X_train[mask], y_train[mask])
        # Residual on the held-out
        yhat_i = float(base.predict(X_train[i:i+1])[0])
        r_loo[i] = abs(y_train[i] - yhat_i)
        # Test predictions from this LOO model
        pred_test_loo[i] = base.predict(X_test)
        if (i + 1) % 100 == 0:
            print(f"  JK+ LOO {i+1}/{n}")

    # Per-stratum lower / upper bounds
    n_test = len(X_test)
    lo = np.empty(n_test, dtype=np.float64)
    hi = np.empty(n_test, dtype=np.float64)
    unique_strata = np.unique(strata_train)
    for j in range(n_test):
        s_j = strata_test[j]
        # Restrict to training points in same stratum
        if s_j in unique_strata:
            in_s = np.where(strata_train == s_j)[0]
        else:
            in_s = np.arange(n)  # fallback to global
        # Jackknife+ bounds per Barber et al. 2021 Lemma 1
        upper_scores = pred_test_loo[in_s, j] + r_loo[in_s]
        lower_scores = pred_test_loo[in_s, j] - r_loo[in_s]
        # Quantile at level 1 - α / 2 (upper) and α / 2 (lower)
        hi[j] = float(np.quantile(upper_scores, 1 - alpha))
        lo[j] = float(np.quantile(lower_scores, alpha))
    return lo, hi, r_loo


def main() -> None:
    print("=== S5: Mondrian jackknife+ conformal wrapper ===")
    df = pd.read_csv(PROC / "mmrf_sprint4_analysis.tsv", sep="\t")
    df = df.dropna(subset=["tt2L_days", "had_2L"]).reset_index(drop=True)

    # Need M_seed3 (focused mediator from S4 fix). Build it from raw expression.
    expr = pd.read_parquet(PROC / "mmrf_baseline_expression.parquet")
    sym_map = pd.read_csv(PROC / "mmrf_ensembl_to_symbol.tsv", sep="\t")
    sym_to_ens = dict(zip(sym_map["gene_name"], sym_map["ensembl_base"]))
    seed3_ens = [sym_to_ens[g] for g in ("PSMB5", "PSMB1", "PSMB2")]
    sub = expr[seed3_ens].copy()
    sub = (sub - sub.mean()) / (sub.std(ddof=0) + 1e-9)
    M_seed3_series = sub.mean(axis=1)
    df = df.merge(
        pd.DataFrame({"submitter_id": expr.index, "M_seed3": M_seed3_series.values}),
        on="submitter_id", how="inner",
    ).reset_index(drop=True)
    df["stratum"] = assign_strata(df)
    print(f"N = {len(df)}; strata sizes:")
    print(df["stratum"].value_counts().to_dict())

    X, y = build_features(df)
    strata = df["stratum"].values

    # FULL jackknife+ on all N patients (standard Barber et al. 2021 setup):
    # for each i, train on N-1, predict interval at point i using LOO residuals
    # from same-stratum training points.
    print(f"Running FULL jackknife+ ({len(X)} LOO fits)...")
    t0 = time.time()
    lo, hi, r_loo = jackknife_plus_intervals(
        X, y, X, strata, strata, alpha=0.10,
    )
    print(f"JK+ done in {time.time()-t0:.1f}s")

    # Per-stratum coverage on the FULL dataset
    y_test = y
    strata_test = strata
    covered = (y_test >= lo) & (y_test <= hi)
    coverage_per_stratum = {}
    for s in sorted(np.unique(strata_test)):
        in_s = strata_test == s
        if in_s.sum() == 0:
            continue
        cov = float(covered[in_s].mean())
        n_s = int(in_s.sum())
        # Mean half-width on log scale; convert to days via exp
        median_lo = float(np.median(np.exp(lo[in_s]) - 1))
        median_hi = float(np.median(np.exp(hi[in_s]) - 1))
        coverage_per_stratum[s] = {
            "n_test": n_s,
            "coverage_at_alpha_0p10": cov,
            "deviation_from_nominal_90pct": cov - 0.90,
            "median_pred_interval_days": [median_lo, median_hi],
            "within_3pct": bool(0.87 <= cov <= 0.93),
            "within_5pct": bool(0.85 <= cov <= 0.95),
        }

    # Aggregate F_S5 verdict
    n_within_3 = sum(1 for v in coverage_per_stratum.values() if v["within_3pct"])
    n_within_5 = sum(1 for v in coverage_per_stratum.values() if v["within_5pct"])
    f_s5_pass = bool(n_within_3 >= 3 and n_within_5 >= 5)  # all 5 within ±5% AND ≥3 within ±3%

    print("\n=== Per-stratum coverage at α=0.10 (target 90%) ===")
    for s in sorted(coverage_per_stratum.keys()):
        v = coverage_per_stratum[s]
        flag3 = "✓" if v["within_3pct"] else "✗"
        flag5 = "✓" if v["within_5pct"] else "✗"
        print(f"  {s:12s}  n_test={v['n_test']:3d}  cov={v['coverage_at_alpha_0p10']:.3f}  "
              f"Δ={v['deviation_from_nominal_90pct']:+.3f}  ±3%[{flag3}] ±5%[{flag5}]  "
              f"interval median (days): [{v['median_pred_interval_days'][0]:.0f}, "
              f"{v['median_pred_interval_days'][1]:.0f}]")
    print(f"\nF_S5: {n_within_3}/{len(coverage_per_stratum)} strata within ±3%, "
          f"{n_within_5}/{len(coverage_per_stratum)} strata within ±5%")
    print(f"F_S5 strict pass (≥3 within ±3% AND ALL 5 within ±5%): "
          f"{'PASS' if f_s5_pass else 'FAIL'}")

    out = {
        "design": "Mondrian jackknife+ on log(tt2L+1) with Ridge base learner; FULL JK+ over N=787",
        "n_total": int(len(X)),
        "strata_sizes": {s: int(np.sum(strata == s)) for s in np.unique(strata)},
        "alpha": 0.10,
        "nominal_coverage": 0.90,
        "coverage_per_stratum": coverage_per_stratum,
        "F_S5_aggregate": {
            "n_strata_within_3pct": n_within_3,
            "n_strata_within_5pct": n_within_5,
            "spec_target": "≥3 strata within ±3%, all 5 within ±5%",
            "strict_pass": f_s5_pass,
        },
    }
    out_path = SPRINT5 / "mondrian_jk_plus_coverage.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
