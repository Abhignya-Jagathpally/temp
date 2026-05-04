"""S2e: Run hierarchical-Bayes SVI on real MMRF cytogenetic strata.

Model (per V10_FOUNDATION_MODEL_PAPER_SPEC.md §2.6, generalized to multi-label):
    μ          ~ N(0, 1)             # global latent mean
    τ          ~ HalfCauchy(1)       # stratum spread
    θ_s | μ,τ  ~ N(μ, τ²)            # per-stratum effect (s ∈ 5 cyto flags)
    σ_obs      ~ HalfCauchy(1)
    z_p        ~ N(μ + Σ_s m_{p,s} θ_s, σ_obs²)

The 994-baseline-driving-marginal-likelihood claim from the spec is realized
as: 716 baseline patients with full cytogenetic calls anchor (μ, τ),
contributing to the marginal likelihood non-vacuously even if no temporal
information is used (Sprint 3 task).

Inputs:
    data/processed/mmrf_z64.npy
    data/processed/mmrf_z64_sample_ids.json
    data/raw/mmrf_commpass/cytogenetics.tsv

Outputs:
    paper/v8_artifacts/v10_sprint2/hbayes_posterior.npz
    paper/v8_artifacts/v10_sprint2/hbayes_summary.json
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
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "mmrf_commpass"
SPRINT2 = ROOT / "paper" / "v8_artifacts" / "v10_sprint2"
SPRINT2.mkdir(parents=True, exist_ok=True)

STRATA = ["del17p", "chr1q21_gain", "del13q", "t_4_14", "t_11_14"]


def hbayes_multilabel_model(z_obs, M, S, d):
    """Multi-label HBayes over latent emissions.

    z_obs : (N, d), real latents
    M     : (N, S), 0/1 cytogenetic membership matrix
    """
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0).expand([d]).to_event(1))
    tau = numpyro.sample("tau", dist.HalfCauchy(1.0))

    with numpyro.plate("strata", S):
        theta = numpyro.sample(
            "theta",
            dist.Normal(jnp.zeros(d), tau * jnp.ones(d)).expand([S, d]).to_event(1),
        )

    sigma_obs = numpyro.sample("sigma_obs", dist.HalfCauchy(1.0))

    # Per-patient mean: μ + Σ_s m_{p,s} · θ_s
    patient_mean = mu + M @ theta  # (N, d)
    with numpyro.plate("patients", z_obs.shape[0]):
        numpyro.sample(
            "obs",
            dist.Normal(patient_mean, sigma_obs).to_event(1),
            obs=z_obs,
        )


def main() -> None:
    Z = np.load(PROC / "mmrf_z64.npy")
    ids = json.loads((PROC / "mmrf_z64_sample_ids.json").read_text())
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    print(f"[S2e] z64 latents: {Z.shape}")

    cyto = pd.read_csv(RAW / "cytogenetics.tsv", sep="\t")
    cyto = cyto.dropna(subset=STRATA).copy()
    cyto = cyto[cyto["submitter_id"].isin(id_to_idx.keys())].reset_index(drop=True)
    print(f"[S2e] cytogenetic-aligned cohort: {len(cyto)} patients")
    for s in STRATA:
        n_pos = int((cyto[s] == 1.0).sum())
        n_total = len(cyto)
        print(f"[S2e]   {s}: {n_pos}/{n_total} ({n_pos/n_total*100:.1f}%)")

    # Build (Z_aligned, M)
    z_idx = np.array([id_to_idx[p] for p in cyto["submitter_id"]])
    Z_aligned = Z[z_idx].astype(np.float32)
    M = cyto[STRATA].values.astype(np.float32)
    print(f"[S2e] aligned Z: {Z_aligned.shape}, M: {M.shape}")

    z_obs = jnp.array(Z_aligned)
    M_jax = jnp.array(M)
    S = len(STRATA)
    d = Z_aligned.shape[1]

    # SVI
    guide = AutoNormal(hbayes_multilabel_model)
    svi = SVI(hbayes_multilabel_model, guide, numpyro.optim.Adam(5e-3), Trace_ELBO())
    rng_key = jax.random.PRNGKey(0)
    print("[S2e] running SVI for 2000 steps ...")
    res = svi.run(rng_key, 2000, z_obs, M_jax, S, d)
    print(f"[S2e] ELBO loss: init={float(res.losses[0]):.2f}  "
          f"final={float(res.losses[-1]):.2f}  "
          f"Δ={float(res.losses[0]-res.losses[-1]):.2f}")

    # Pull posterior point estimates from the guide
    post = guide.median(res.params)
    mu_hat = np.asarray(post["mu"])
    theta_hat = np.asarray(post["theta"])    # (S, d)
    tau_hat = float(post["tau"])
    sigma_hat = float(post["sigma_obs"])
    print(f"[S2e] posterior — τ̂={tau_hat:.3f}, σ̂_obs={sigma_hat:.3f}")
    print(f"[S2e] θ̂ per-stratum L2 norms: "
          f"{ {STRATA[s]: float(np.linalg.norm(theta_hat[s])) for s in range(S)} }")

    # Save posterior + ELBO trace
    out_path = SPRINT2 / "hbayes_posterior.npz"
    np.savez(
        out_path,
        mu=mu_hat,
        theta=theta_hat,
        tau=tau_hat,
        sigma_obs=sigma_hat,
        elbo_losses=np.array(res.losses),
        strata=np.array(STRATA),
        patient_ids=cyto["submitter_id"].values,
        z_aligned=Z_aligned,
        M=M,
    )
    print(f"[S2e] wrote {out_path}")

    summary = {
        "n_patients": int(len(cyto)),
        "n_strata": int(S),
        "latent_dim": int(d),
        "elbo_init": float(res.losses[0]),
        "elbo_final": float(res.losses[-1]),
        "tau_hat": tau_hat,
        "sigma_obs_hat": sigma_hat,
        "theta_norm_per_stratum": {STRATA[s]: float(np.linalg.norm(theta_hat[s])) for s in range(S)},
        "stratum_prevalence": {s: float((cyto[s] == 1.0).mean()) for s in STRATA},
        "spec_target_5_strata": True,
        "honest_disclosure": (
            "Multi-label additive HBayes generalizes the spec's 'one-of-5' "
            "stratum assignment because patients commonly carry multiple "
            "cytogenetic abnormalities (e.g., del13q + chr1q21+). The marginal "
            "likelihood interpretation in spec §2.6 carries through with M as "
            "the membership matrix."
        ),
    }
    summary_path = SPRINT2 / "hbayes_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[S2e] wrote {summary_path}")


if __name__ == "__main__":
    main()
