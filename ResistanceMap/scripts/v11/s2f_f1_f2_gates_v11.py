"""S2f v11 — F1 (cross-stratum coverage) + F2 (baseline-ablation log-lik) gates
re-run on the v11 Geneformer-derived latent (mmrf_z64_v11.npy).

PURPOSE: zero-trust leak check per V11_SOTA_UPGRADE_PLAN.md §2 row 1.
At v10, F2 is REFUTED (Δ̄=24.2 with 95% CI [-43.1, 99.5] containing 0; see
paper/v8_artifacts/v10_sprint2/f1_f2_gates.json). The v11 encoder MUST keep
F2 REFUTED. If F2 flips to PASS at v11, the foundation-model adapter is
leaking paired-patient identity into the latent → reject and revert.

This script is a faithful re-run of scripts/v10/s2f_f1_f2_gates.py with a
SINGLE difference: load mmrf_z64_v11.npy instead of mmrf_z64.npy. No other
modeling change. v10 file is NOT modified.
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
ART = ROOT / "paper" / "v8_artifacts" / "v11_sprint1"
ART.mkdir(parents=True, exist_ok=True)

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


def predictive_normal_coverage(z_true, mu_hat, theta_hat, sigma_hat, M, alpha=0.10):
    z_crit = 1.6448536
    pred_mean = mu_hat + M @ theta_hat
    abs_dev = np.abs(z_true - pred_mean)
    inside = (abs_dev < z_crit * sigma_hat).astype(np.float32)
    return {
        "coverage_overall": float(inside.mean()),
        "coverage_per_patient_mean": float(inside.mean(axis=1).mean()),
        "n_test_patients": int(z_true.shape[0]),
        "n_dim": int(z_true.shape[1]),
    }


def heldout_loglik(z_true, mu_hat, theta_hat, sigma_hat, M):
    pred_mean = mu_hat + M @ theta_hat
    var = sigma_hat ** 2
    sq = (z_true - pred_mean) ** 2
    ll = -0.5 * (np.log(2 * np.pi * var) + sq / var).sum(axis=1)
    return ll


def main() -> None:
    # ── Load v11 latents (the only difference from v10 s2f) ──
    Z_full = np.load(PROC / "mmrf_z64_v11.npy")
    ids = json.loads((PROC / "mmrf_z64_v11_sample_ids.json").read_text())
    id_to_idx = {pid: i for i, pid in enumerate(ids)}

    cyto = pd.read_csv(RAW / "cytogenetics.tsv", sep="\t")
    cyto = cyto.dropna(subset=STRATA).copy()
    cyto = cyto[cyto["submitter_id"].isin(id_to_idx.keys())].reset_index(drop=True)
    z_idx = np.array([id_to_idx[p] for p in cyto["submitter_id"]])
    Z = Z_full[z_idx].astype(np.float32)
    M = cyto[STRATA].values.astype(np.float32)
    sys.stdout.write(f"[v11-s2f] cohort: {len(cyto)} patients, Z={Z.shape}, M={M.shape}\n")

    paired_df = pd.read_csv(PROC / "mmrf_paired_patients.tsv", sep="\t")
    paired_set = set(paired_df["patient_id"].astype(str))
    is_paired = np.array(
        [pid in paired_set for pid in cyto["submitter_id"].astype(str)],
        dtype=bool,
    )
    sys.stdout.write(f"[v11-s2f] paired patients in cohort: {int(is_paired.sum())} / {len(cyto)}\n")

    # ── F1 ──
    f1_results = {}
    for s_idx, s_name in enumerate(STRATA):
        held_mask = (M[:, s_idx] == 1.0) & (
            M[:, [j for j in range(len(STRATA)) if j != s_idx]].sum(axis=1) == 0
        )
        n_held = int(held_mask.sum())
        if n_held < 20:
            held_mask = M[:, s_idx] == 1.0
            n_held = int(held_mask.sum())
        train_mask = ~held_mask
        if n_held == 0 or train_mask.sum() < 50:
            continue
        keep_cols = [j for j in range(len(STRATA)) if j != s_idx]
        M_train = M[train_mask][:, keep_cols].astype(np.float32)
        Z_train = Z[train_mask].astype(np.float32)
        M_test_full = M[held_mask][:, keep_cols].astype(np.float32)
        Z_test = Z[held_mask].astype(np.float32)

        sys.stdout.write(
            f"[v11-s2f] F1 hold-out {s_name}: n_train={int(train_mask.sum())}, n_held={n_held}\n"
        )
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
            "passes": bool(cov["coverage_overall"] >= 0.70),
        }
        sys.stdout.write(
            f"[v11-s2f] F1 {s_name}: coverage={cov['coverage_overall']:.3f} "
            f"({'PASS' if cov['coverage_overall']>=0.70 else 'FAIL'})\n"
        )

    # ── F2: paired LOO on both arms (no leak) ──
    test_mask = is_paired
    n_test = int(test_mask.sum())
    if n_test < 5:
        f2_result = {"n_test": n_test, "skipped": True}
    else:
        paired_indices = np.where(test_mask)[0]
        ll_A_arm, ll_B_arm = [], []
        skipped = 0
        sys.stdout.write(
            f"[v11-s2f] F2: running {n_test} paired LOO fits per arm "
            f"({2*n_test} SVI fits total)\n"
        )
        for i, held_idx in enumerate(paired_indices):
            train_mask_A = np.ones(len(Z), dtype=bool)
            train_mask_A[held_idx] = False
            res_A, post_A = fit_hbayes(
                jnp.array(Z[train_mask_A]), jnp.array(M[train_mask_A]), n_steps=600, seed=i
            )
            ll_A = heldout_loglik(
                Z[held_idx:held_idx + 1],
                np.asarray(post_A["mu"]),
                np.asarray(post_A["theta"]),
                float(post_A["sigma_obs"]),
                M[held_idx:held_idx + 1],
            )

            train_mask_B = test_mask.copy()
            train_mask_B[held_idx] = False
            if int(train_mask_B.sum()) < 5:
                skipped += 1
                continue
            res_B, post_B = fit_hbayes(
                jnp.array(Z[train_mask_B]), jnp.array(M[train_mask_B]), n_steps=600, seed=i
            )
            ll_B = heldout_loglik(
                Z[held_idx:held_idx + 1],
                np.asarray(post_B["mu"]),
                np.asarray(post_B["theta"]),
                float(post_B["sigma_obs"]),
                M[held_idx:held_idx + 1],
            )
            ll_A_arm.append(float(ll_A[0]))
            ll_B_arm.append(float(ll_B[0]))

        ll_A_arm = np.array(ll_A_arm, dtype=np.float64)
        ll_B_arm = np.array(ll_B_arm, dtype=np.float64)
        n_match = len(ll_A_arm)
        delta = ll_A_arm - ll_B_arm

        n_boot = 5000
        gen = torch.Generator().manual_seed(0)
        delta_t = torch.from_numpy(delta)
        boot_means = torch.empty(n_boot, dtype=torch.float64)
        for b in range(n_boot):
            sample_idx = torch.randint(0, n_match, (n_match,), generator=gen)
            boot_means[b] = delta_t[sample_idx].mean()
        boot_np = boot_means.numpy()
        ci_low = float(np.percentile(boot_np, 2.5))
        ci_high = float(np.percentile(boot_np, 97.5))
        partial_pooling_no_value = bool(ci_low <= 0 <= ci_high)
        passes = bool(not partial_pooling_no_value and float(delta.mean()) > 0)

        f2_result = {
            "encoder": "v11_geneformer_v1_10m",
            "design": "proper paired LOO on both arms (no train-on-test leak)",
            "n_test_paired": int(n_match),
            "n_skipped_due_to_underfit": int(skipped),
            "ll_full_cohort_loo_mean": float(ll_A_arm.mean()),
            "ll_paired_only_loo_mean": float(ll_B_arm.mean()),
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std()),
            "delta_bootstrap_95_ci": [ci_low, ci_high],
            "ci_contains_zero": partial_pooling_no_value,
            "passes_F2": passes,
            "verdict": (
                "F2 PASS at v11 — encoder may be leaking paired-patient identity "
                "(reject + revert per V11 plan §2 row 1 zero-trust check)"
                if passes else
                "F2 REFUTED at v11 (matches v10 verdict — encoder upgrade is honest)"
            ),
        }
        sys.stdout.write(
            f"[v11-s2f] F2: n={n_match}, Δlog-lik mean={delta.mean():.3f}, "
            f"95% CI=[{ci_low:.3f}, {ci_high:.3f}], "
            f"{'PASS (LEAK?)' if passes else 'REFUTED (honest)'}\n"
        )

    # Compare to v10 verdict for explicit zero-trust diff
    v10_path = ROOT / "paper" / "v8_artifacts" / "v10_sprint2" / "f1_f2_gates.json"
    v10_v = json.loads(v10_path.read_text()) if v10_path.exists() else None
    out = {
        "F1_per_stratum_v11": f1_results,
        "F2_v11": f2_result,
        "v10_F2_passes": v10_v["F2"]["passes_F2"] if v10_v else None,
        "v10_F2_delta_mean": v10_v["F2"]["delta_mean"] if v10_v else None,
        "v10_F2_ci": v10_v["F2"]["delta_bootstrap_95_ci"] if v10_v else None,
        "leak_check_passed": (
            (not f2_result.get("passes_F2", False))
            if "passes_F2" in f2_result else None
        ),
    }
    out_path = ART / "f1_f2_gates_v11.json"
    out_path.write_text(json.dumps(out, indent=2))
    sys.stdout.write(f"[v11-s2f] wrote {out_path}\n")


if __name__ == "__main__":
    main()
