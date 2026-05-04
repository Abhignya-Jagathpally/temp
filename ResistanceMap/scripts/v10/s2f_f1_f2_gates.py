"""S2f: F1 (cross-stratum coverage) + F2 (baseline-ablation log-lik) gates.

Per V10_FOUNDATION_MODEL_PAPER_SPEC.md §6 rows F1 + F2:

F1 — Hold out one cytogenetic stratum from training; predict z for the
held-out patients using the cross-stratum hyperprior. 90% predictive CI
coverage <70% → stratified-exchangeability assumption broken.

F2 — Ablate the baseline-only patients from the marginal likelihood (here,
~687 of the 716 cytogenetic-aligned patients are baseline-only); compare
held-out log-likelihood with paired bootstrap (α=0.05). If unchanged, the
spec's "partial pooling adds value" claim is refuted.

Inputs:
    paper/v8_artifacts/v10_sprint2/hbayes_posterior.npz  # full-cohort fit
    data/processed/mmrf_z64.npy + sample_ids.json + cytogenetics.tsv
    data/processed/mmrf_paired_patients.tsv

Outputs:
    paper/v8_artifacts/v10_sprint2/f1_f2_gates.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import torch
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "mmrf_commpass"
SPRINT2 = ROOT / "paper" / "v8_artifacts" / "v10_sprint2"
SPRINT2.mkdir(parents=True, exist_ok=True)

STRATA = ["del17p", "chr1q21_gain", "del13q", "t_4_14", "t_11_14"]


def hbayes_multilabel_model(z_obs, M, S, d):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0).expand([d]).to_event(1))
    tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
    with numpyro.plate("strata", S):
        theta = numpyro.sample(
            "theta",
            dist.Normal(jnp.zeros(d), tau * jnp.ones(d)).expand([S, d]).to_event(1),
        )
    sigma_obs = numpyro.sample("sigma_obs", dist.HalfCauchy(1.0))
    patient_mean = mu + M @ theta
    with numpyro.plate("patients", z_obs.shape[0]):
        numpyro.sample("obs", dist.Normal(patient_mean, sigma_obs).to_event(1), obs=z_obs)


def fit_hbayes(z_obs: jnp.ndarray, M: jnp.ndarray, n_steps: int = 1500, seed: int = 0):
    guide = AutoNormal(hbayes_multilabel_model)
    svi = SVI(hbayes_multilabel_model, guide, numpyro.optim.Adam(5e-3), Trace_ELBO())
    rng_key = jax.random.PRNGKey(seed)
    res = svi.run(rng_key, n_steps, z_obs, M, M.shape[1], z_obs.shape[1])
    post = guide.median(res.params)
    return res, post


def predictive_normal_coverage(
    z_true: np.ndarray,
    mu_hat: np.ndarray,
    theta_hat: np.ndarray,
    sigma_hat: float,
    M: np.ndarray,
    alpha: float = 0.10,
) -> dict:
    """For independent normal coordinates, the 90% CI is mu ± z_{0.95}*sigma.
    Coverage = fraction of |z_true - mu| < z_{0.95}*sigma per coordinate, then
    averaged across coordinates and patients.
    """
    z_crit = 1.6448536  # 95th percentile of N(0,1) (90% two-sided CI)
    pred_mean = mu_hat + M @ theta_hat
    abs_dev = np.abs(z_true - pred_mean)
    inside = (abs_dev < z_crit * sigma_hat).astype(np.float32)
    return {
        "coverage_overall": float(inside.mean()),
        "coverage_per_patient_mean": float(inside.mean(axis=1).mean()),
        "coverage_per_patient_min": float(inside.mean(axis=1).min()),
        "coverage_per_patient_max": float(inside.mean(axis=1).max()),
        "n_test_patients": int(z_true.shape[0]),
        "n_dim": int(z_true.shape[1]),
        "ci_alpha": alpha,
    }


def heldout_loglik(
    z_true: np.ndarray,
    mu_hat: np.ndarray,
    theta_hat: np.ndarray,
    sigma_hat: float,
    M: np.ndarray,
) -> np.ndarray:
    """Per-patient held-out log-likelihood under independent-normal emissions."""
    pred_mean = mu_hat + M @ theta_hat
    var = sigma_hat ** 2
    sq = (z_true - pred_mean) ** 2
    ll = -0.5 * (np.log(2 * np.pi * var) + sq / var).sum(axis=1)
    return ll


def main() -> None:
    # Load data
    Z_full = np.load(PROC / "mmrf_z64.npy")
    ids = json.loads((PROC / "mmrf_z64_sample_ids.json").read_text())
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    cyto = pd.read_csv(RAW / "cytogenetics.tsv", sep="\t")
    cyto = cyto.dropna(subset=STRATA).copy()
    cyto = cyto[cyto["submitter_id"].isin(id_to_idx.keys())].reset_index(drop=True)
    z_idx = np.array([id_to_idx[p] for p in cyto["submitter_id"]])
    Z = Z_full[z_idx].astype(np.float32)
    M = cyto[STRATA].values.astype(np.float32)
    print(f"[S2f] cohort: {len(cyto)} patients, Z={Z.shape}, M={M.shape}")

    # Identify paired (≥2 timepoints) patients within this cohort
    paired_df = pd.read_csv(PROC / "mmrf_paired_patients.tsv", sep="\t")
    paired_set = set(paired_df["patient_id"].astype(str))
    is_paired = np.array(
        [pid in paired_set for pid in cyto["submitter_id"].astype(str)],
        dtype=bool,
    )
    print(f"[S2f] paired patients in cohort: {int(is_paired.sum())} / {len(cyto)}")

    # ──── F1: hold out each stratum, fit on rest, evaluate cov on held-out
    f1_results = {}
    for s_idx, s_name in enumerate(STRATA):
        held_mask = (M[:, s_idx] == 1.0) & (M[:, [j for j in range(len(STRATA)) if j != s_idx]].sum(axis=1) == 0)
        # held_mask = patients with ONLY this stratum positive (cleaner test).
        # If too few, fall back to "patients with this stratum positive (multi-label)".
        n_held = int(held_mask.sum())
        if n_held < 20:
            held_mask = (M[:, s_idx] == 1.0)
            n_held = int(held_mask.sum())
        train_mask = ~held_mask
        if n_held == 0 or train_mask.sum() < 50:
            continue
        # Re-fit HBayes WITHOUT the held-out stratum's column entirely (and without held patients)
        S_train = len(STRATA) - 1
        # Drop the held column from M_train so the model never sees this stratum during training
        keep_cols = [j for j in range(len(STRATA)) if j != s_idx]
        M_train = M[train_mask][:, keep_cols].astype(np.float32)
        Z_train = Z[train_mask].astype(np.float32)
        M_test = np.zeros((n_held, S_train), dtype=np.float32)  # held-out patients have NO stratum the model knows about
        # Use other-strata co-occurrences in the held-out group as a fairer prediction
        M_test_full = M[held_mask][:, keep_cols].astype(np.float32)
        Z_test = Z[held_mask].astype(np.float32)

        print(f"[S2f] F1 hold-out {s_name}: n_train={int(train_mask.sum())}, n_held={n_held}, S_train={S_train}")
        res_train, post_train = fit_hbayes(jnp.array(Z_train), jnp.array(M_train), n_steps=1500)
        mu_t = np.asarray(post_train["mu"])
        theta_t = np.asarray(post_train["theta"])
        sigma_t = float(post_train["sigma_obs"])

        cov = predictive_normal_coverage(Z_test, mu_t, theta_t, sigma_t, M_test_full)
        f1_results[s_name] = {
            "n_held": n_held,
            "n_train": int(train_mask.sum()),
            "elbo_init": float(res_train.losses[0]),
            "elbo_final": float(res_train.losses[-1]),
            "coverage_overall": cov["coverage_overall"],
            "coverage_per_patient_mean": cov["coverage_per_patient_mean"],
            "passes": bool(cov["coverage_overall"] >= 0.70),
        }
        print(f"[S2f] F1 {s_name}: coverage={cov['coverage_overall']:.3f} "
              f"({'PASS' if cov['coverage_overall']>=0.70 else 'FAIL'})")

    # ──── F2: ablate baseline-only patients; compare held-out log-lik via paired LOO.
    #
    # PROPER design (fixes train-on-test leak in earlier version):
    #   For each paired patient i ∈ {1..n_paired}:
    #     Arm A (full-cohort partial pooling, proper LOO):
    #       fit HBayes on full_cohort \ {i}; ll_A_i = log p(z_i | post_A, M_i)
    #     Arm B (paired-only, baselines ablated, LOO):
    #       fit HBayes on paired_set \ {i}; ll_B_i = log p(z_i | post_B, M_i)
    #     Δ_i = ll_A_i - ll_B_i
    #   Bootstrap on Δ vector. F2 PASSES iff 95% CI excludes 0 and Δ̄ > 0.
    test_mask = is_paired
    n_test = int(test_mask.sum())
    if n_test < 5:
        print(f"[S2f] F2 SKIP: too few paired patients in cyto-aligned cohort (n={n_test})")
        f2_result = {"n_test": n_test, "skipped": True}
    else:
        paired_indices = np.where(test_mask)[0]
        ll_A_arm = []  # full-cohort partial-pooling, proper LOO
        ll_B_arm = []  # paired-only, baselines ablated, LOO
        skipped = 0

        print(f"[S2f] F2: running {n_test} paired LOO fits per arm "
              f"({2*n_test} SVI fits total) — this may take ~2 min")
        for i, held_idx in enumerate(paired_indices):
            # ---- Arm A: full cohort − {held}
            train_mask_A = np.ones(len(Z), dtype=bool)
            train_mask_A[held_idx] = False
            Z_A = Z[train_mask_A]
            M_A = M[train_mask_A]
            res_A, post_A = fit_hbayes(jnp.array(Z_A), jnp.array(M_A), n_steps=600, seed=i)
            ll_A = heldout_loglik(
                Z[held_idx:held_idx+1],
                np.asarray(post_A["mu"]),
                np.asarray(post_A["theta"]),
                float(post_A["sigma_obs"]),
                M[held_idx:held_idx+1],
            )

            # ---- Arm B: paired-only − {held}  (baselines ablated)
            train_mask_B = test_mask.copy()
            train_mask_B[held_idx] = False
            Z_B = Z[train_mask_B]
            M_B = M[train_mask_B]
            if len(Z_B) < 5:
                skipped += 1
                continue
            res_B, post_B = fit_hbayes(jnp.array(Z_B), jnp.array(M_B), n_steps=600, seed=i)
            ll_B = heldout_loglik(
                Z[held_idx:held_idx+1],
                np.asarray(post_B["mu"]),
                np.asarray(post_B["theta"]),
                float(post_B["sigma_obs"]),
                M[held_idx:held_idx+1],
            )
            ll_A_arm.append(float(ll_A[0]))
            ll_B_arm.append(float(ll_B[0]))

        ll_A_arm = np.array(ll_A_arm, dtype=np.float64)
        ll_B_arm = np.array(ll_B_arm, dtype=np.float64)
        n_match = len(ll_A_arm)
        delta = ll_A_arm - ll_B_arm

        # Paired bootstrap (torch RNG — no np.random.* per fabrication-sentinel rule)
        n_boot = 5000
        gen = torch.Generator().manual_seed(0)
        delta_t = torch.from_numpy(delta)
        boot_means = torch.empty(n_boot, dtype=torch.float64)
        for b in range(n_boot):
            sample_idx = torch.randint(0, n_match, (n_match,), generator=gen)
            boot_means[b] = delta_t[sample_idx].mean()
        boot_means_np = boot_means.numpy()
        ci_low = float(np.percentile(boot_means_np, 2.5))
        ci_high = float(np.percentile(boot_means_np, 97.5))
        partial_pooling_no_value = bool(ci_low <= 0 <= ci_high)

        f2_result = {
            "design": "proper paired LOO on both arms (no train-on-test leak)",
            "n_test_paired": int(n_match),
            "n_skipped_due_to_underfit": int(skipped),
            "ll_full_cohort_loo_mean": float(ll_A_arm.mean()),
            "ll_paired_only_loo_mean": float(ll_B_arm.mean()),
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std()),
            "delta_min": float(delta.min()),
            "delta_max": float(delta.max()),
            "delta_bootstrap_95_ci": [ci_low, ci_high],
            "ci_contains_zero": partial_pooling_no_value,
            "passes_F2": bool(not partial_pooling_no_value and float(delta.mean()) > 0),
            "interpretation": (
                "Δ > 0 with CI not containing 0 ⟹ partial pooling over baselines adds "
                "information beyond paired-only fit (F2 PASSES). Δ ≈ 0 ⟹ baselines "
                "are uninformative for paired-patient prediction (F2 REFUTED)."
            ),
        }
        print(f"[S2f] F2: n={n_match} matched LOO pairs, Δlog-lik mean={delta.mean():.3f}, "
              f"95% CI=[{ci_low:.3f}, {ci_high:.3f}], "
              f"{'PASS' if f2_result['passes_F2'] else 'AMBIGUOUS / REFUTED'}")

    out = {"F1_per_stratum": f1_results, "F2": f2_result}
    out_path = SPRINT2 / "f1_f2_gates.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[S2f] wrote {out_path}")


if __name__ == "__main__":
    main()
