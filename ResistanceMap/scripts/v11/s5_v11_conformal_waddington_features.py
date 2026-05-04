"""S5 v11: Mondrian jackknife+ with Waddington-derived features as base-learner upgrade.

Per `docs/V11_SOTA_UPGRADE_PLAN.md` §2 row 6: the v10 Ridge base learner is
swapped for an enriched feature set + a more flexible base learner. The
Barber-Candès-Ramdas-Tibshirani 2021 jackknife+ coverage guarantee is
**base-learner-agnostic** — coverage stays calibrated for any (X→Y) model
that can be fit on N-1 and predict on the held-out — so the test is whether
the v11 base learner *tightens* the prediction interval at the same nominal
coverage.

Three base learners under the IDENTICAL jackknife+ wrapper:

    Ridge_v10       — 11 features (bort_1L, M_seed3, bort×M, ISS, age,
                      gender, 5 cyto). v10 reference. λ=1.0.
    Ridge_v11features — 11 + 4 Waddington features:
                          U_θ(z₀)         scalar potential at baseline
                          U_θ(z(T))       potential after gradient-flow ODE
                                          to T=1 under v11 Neural-ODE
                          ‖∇U_θ(z₀)‖     drift magnitude at baseline
                          ‖z(T)−z₀‖      total ODE displacement over T=1
                       Same Ridge λ=1.0; isolates "more features" from
                       "more flexible model".
    GBM_v10          — HistGradientBoostingRegressor on the v10 11 features
                       only; isolates "more flexibility" from "more features".

This 3-way design is pre-registered: any improvement in median interval
width over Ridge_v10 must be attributable to one of the two axes (features
or model flexibility). If both v11 variants tighten and the Mondrian
coverage stays within ±3% of nominal 0.90, the v11 upgrade IS justified.
If neither tightens, the upgrade is rejected and v10 stands.

Output: paper/v8_artifacts/v11_sprint5/mondrian_jk_plus_v11.json
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

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPRINT5_V10 = ROOT / "paper" / "v8_artifacts" / "v10_sprint5"
SPRINT5_V11 = ROOT / "paper" / "v8_artifacts" / "v11_sprint5"
SPRINT5_V11.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from resistancemap.landscape.scalar_potential import (  # noqa: E402
    ScalarPotential, ScalarPotentialConfig,
)
from resistancemap.landscape.neural_ode_flow import (  # noqa: E402
    NeuralODEFlow, NeuralODEFlowConfig,
)

CONFOUNDERS_V10 = [
    "iss_stage_ord", "age_at_dx_years", "gender_male",
    "cyto_del17p", "cyto_chr1q21_gain", "cyto_del13q",
    "cyto_t_4_14", "cyto_t_11_14",
]


def assign_strata(df: pd.DataFrame) -> np.ndarray:
    s = np.array(["S5_other"] * len(df), dtype=object)
    s[df["cyto_del17p"].values == 1] = "S1_del17p"
    mask_used = s != "S5_other"
    s2 = (df["cyto_t_4_14"].values == 1) & (~mask_used)
    s[s2] = "S2_t_4_14"; mask_used = s != "S5_other"
    s3 = (df["cyto_chr1q21_gain"].values == 1) & (~mask_used)
    s[s3] = "S3_1q21"; mask_used = s != "S5_other"
    s4 = (df["cyto_t_11_14"].values == 1) & (~mask_used)
    s[s4] = "S4_t_11_14"
    return s


def compute_waddington_features(z0_per_patient: np.ndarray, T: float = 1.0) -> np.ndarray:
    """For each row in z0_per_patient (n, 64) return (n, 4) features:
        [U(z0), U(z(T)), ‖∇U(z0)‖, ‖z(T)-z0‖]
    """
    ckpt = torch.load(ROOT / "checkpoints" / "u_theta_v10s1.pt",
                      map_location="cpu", weights_only=False)
    cfg = ScalarPotentialConfig(**ckpt["config"])
    u_theta = ScalarPotential(cfg)
    u_theta.load_state_dict(ckpt["state_dict"])
    u_theta.eval()
    flow = NeuralODEFlow(u_theta, NeuralODEFlowConfig(method="dopri5", rtol=1e-5, atol=1e-7))

    Z0 = torch.from_numpy(z0_per_patient.astype(np.float32))
    with torch.no_grad():
        ZT = flow.forward_to_T(Z0, T)
        U0 = u_theta.forward(Z0).numpy()
        UT = u_theta.forward(ZT).numpy()
    grad_norm = u_theta.grad_z(Z0).detach().norm(dim=-1).numpy()
    displacement = (ZT - Z0).norm(dim=-1).numpy()
    return np.stack([U0, UT, grad_norm, displacement], axis=1)


def jackknife_plus_intervals_with_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, strata_train: np.ndarray, strata_test: np.ndarray,
    model_factory, alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mondrian jackknife+ with arbitrary model factory (callable -> sklearn-style).

    Returns (lo, hi, r_loo).
    """
    n = len(X_train)
    r_loo = np.empty(n, dtype=np.float64)
    pred_test_loo = np.empty((n, len(X_test)), dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        m = model_factory()
        m.fit(X_train[mask], y_train[mask])
        yhat_i = float(m.predict(X_train[i:i+1])[0])
        r_loo[i] = abs(y_train[i] - yhat_i)
        pred_test_loo[i] = m.predict(X_test)
        if (i + 1) % 200 == 0:
            print(f"    LOO {i+1}/{n}")

    lo = np.empty(len(X_test), dtype=np.float64)
    hi = np.empty(len(X_test), dtype=np.float64)
    unique_strata = np.unique(strata_train)
    for j in range(len(X_test)):
        s_j = strata_test[j]
        in_s = np.where(strata_train == s_j)[0] if s_j in unique_strata else np.arange(n)
        upper_scores = pred_test_loo[in_s, j] + r_loo[in_s]
        lower_scores = pred_test_loo[in_s, j] - r_loo[in_s]
        hi[j] = float(np.quantile(upper_scores, 1 - alpha))
        lo[j] = float(np.quantile(lower_scores, alpha))
    return lo, hi, r_loo


def evaluate(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, strata: np.ndarray) -> dict:
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    out = {"per_stratum": {}, "marginal": {}}
    for s in sorted(np.unique(strata)):
        in_s = strata == s
        if in_s.sum() == 0:
            continue
        cov = float(covered[in_s].mean())
        med_lo_days = float(np.median(np.exp(lo[in_s]) - 1))
        med_hi_days = float(np.median(np.exp(hi[in_s]) - 1))
        med_width_days = float(np.median(np.exp(hi[in_s]) - np.exp(lo[in_s])))
        out["per_stratum"][s] = {
            "n_test": int(in_s.sum()),
            "coverage_at_alpha_0p10": cov,
            "deviation_from_nominal": cov - 0.90,
            "median_pred_interval_days": [med_lo_days, med_hi_days],
            "median_interval_width_days": med_width_days,
            "median_interval_width_log": float(np.median(width[in_s])),
            "within_3pct": bool(0.87 <= cov <= 0.93),
            "within_5pct": bool(0.85 <= cov <= 0.95),
        }
    n3 = sum(1 for v in out["per_stratum"].values() if v["within_3pct"])
    n5 = sum(1 for v in out["per_stratum"].values() if v["within_5pct"])
    out["marginal"] = {
        "coverage": float(covered.mean()),
        "median_width_log": float(np.median(width)),
        "median_width_days": float(np.median(np.exp(hi) - np.exp(lo))),
        "n_strata_within_3pct": n3,
        "n_strata_within_5pct": n5,
        "F_S5_strict_pass": bool(n3 >= 3 and n5 >= 5),
    }
    return out


def main() -> None:
    print("=== S5 v11: Mondrian jackknife+ with Waddington-derived features ===")
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
    print(f"N = {len(df)}; z₀ shape = {z0_pat.shape}")

    strata = assign_strata(df)
    y = np.log(df["tt2L_days"].values.astype(np.float64) + 1.0)

    # v10 11-feature matrix
    X_v10 = np.column_stack([
        df["bort_1L"].values, df["M_seed3"].values,
        df["bort_1L"].values * df["M_seed3"].values,
        *[df[c].values for c in CONFOUNDERS_V10],
    ]).astype(np.float64)

    print(f"\nComputing Waddington features (4 dims) for N={len(df)}...")
    t0 = time.time()
    X_wadd = compute_waddington_features(z0_pat, T=1.0)
    print(f"  done in {time.time()-t0:.1f}s; ranges:")
    for j, name in enumerate(["U(z0)", "U(z(T))", "‖∇U(z0)‖", "‖z(T)−z0‖"]):
        col = X_wadd[:, j]
        print(f"    {name:14s} mean={col.mean():+.3f} std={col.std():.3f} "
              f"min={col.min():+.3f} max={col.max():+.3f}")

    # standardize Waddington features (Ridge regularization is scale-sensitive)
    X_wadd_std = (X_wadd - X_wadd.mean(axis=0)) / (X_wadd.std(axis=0, ddof=0) + 1e-9)
    X_v11 = np.column_stack([X_v10, X_wadd_std])

    print(f"\nFeature dims: v10={X_v10.shape[1]}, v11={X_v11.shape[1]}")

    # 3 base learners under identical jackknife+
    print("\n=== JK+ Ridge_v10 (11 features) ===")
    t0 = time.time()
    lo_a, hi_a, _ = jackknife_plus_intervals_with_model(
        X_v10, y, X_v10, strata, strata, lambda: Ridge(alpha=1.0),
    )
    print(f"  done in {time.time()-t0:.1f}s")
    res_a = evaluate(lo_a, hi_a, y, strata)

    print("\n=== JK+ Ridge_v11features (11 + 4 Waddington) ===")
    t0 = time.time()
    lo_b, hi_b, _ = jackknife_plus_intervals_with_model(
        X_v11, y, X_v11, strata, strata, lambda: Ridge(alpha=1.0),
    )
    print(f"  done in {time.time()-t0:.1f}s")
    res_b = evaluate(lo_b, hi_b, y, strata)

    print("\n=== JK+ GBM_v10 (HistGradientBoosting on 11 features) ===")
    t0 = time.time()
    lo_c, hi_c, _ = jackknife_plus_intervals_with_model(
        X_v10, y, X_v10, strata, strata,
        lambda: HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, random_state=0,
        ),
    )
    print(f"  done in {time.time()-t0:.1f}s")
    res_c = evaluate(lo_c, hi_c, y, strata)

    out = {
        "design": ("Mondrian jackknife+ on log(tt2L+1) with 3 base learners "
                   "under identical wrapper. Waddington features = "
                   "[U(z0), U(z(T)), ‖∇U(z0)‖, ‖z(T)-z0‖] from v11 Neural-ODE."),
        "n_total": int(len(df)),
        "strata_sizes": {s: int(np.sum(strata == s)) for s in np.unique(strata)},
        "alpha": 0.10, "nominal_coverage": 0.90,
        "Ridge_v10":         res_a,
        "Ridge_v11features": res_b,
        "GBM_v10":           res_c,
    }

    # Head-to-head comparison
    print("\n" + "=" * 80)
    print(f"{'Stratum':12s} {'n':>5s} | "
          f"{'cov_v10':>8s} {'cov_v11':>8s} {'cov_GBM':>8s} | "
          f"{'wd_v10':>8s} {'wd_v11':>8s} {'wd_GBM':>8s}  (median width, days)")
    print("-" * 80)
    for s in sorted(res_a["per_stratum"].keys()):
        a = res_a["per_stratum"][s]; b = res_b["per_stratum"][s]; c = res_c["per_stratum"][s]
        print(f"{s:12s} {a['n_test']:>5d} | "
              f"{a['coverage_at_alpha_0p10']:>8.3f} {b['coverage_at_alpha_0p10']:>8.3f} {c['coverage_at_alpha_0p10']:>8.3f} | "
              f"{a['median_interval_width_days']:>8.0f} {b['median_interval_width_days']:>8.0f} {c['median_interval_width_days']:>8.0f}")
    print("-" * 80)
    print(f"{'Marginal':12s} {len(df):>5d} | "
          f"{res_a['marginal']['coverage']:>8.3f} {res_b['marginal']['coverage']:>8.3f} {res_c['marginal']['coverage']:>8.3f} | "
          f"{res_a['marginal']['median_width_days']:>8.0f} {res_b['marginal']['median_width_days']:>8.0f} {res_c['marginal']['median_width_days']:>8.0f}")
    print(f"\nF_S5 (≥3 within ±3% AND all 5 within ±5%): "
          f"v10={'PASS' if res_a['marginal']['F_S5_strict_pass'] else 'FAIL'}  "
          f"v11={'PASS' if res_b['marginal']['F_S5_strict_pass'] else 'FAIL'}  "
          f"GBM={'PASS' if res_c['marginal']['F_S5_strict_pass'] else 'FAIL'}")

    out_path = SPRINT5_V11 / "mondrian_jk_plus_v11.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    from resistancemap.observability.v11_stage_decorator import v11_stage  # noqa: E402

    with v11_stage("s5_v11_conformal_waddington_features"):
        main()
