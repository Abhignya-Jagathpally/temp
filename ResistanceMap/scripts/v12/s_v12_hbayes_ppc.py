"""v12 / Issue #67: Posterior-predictive checks (Gelman 1996 §6) on the
Sprint-2 HBayes 5-stratum model.

We re-run SVI on the same z_obs / M as `s2e_hbayes_inference.py`, keep the
trained guide, draw M=1000 posterior samples from the AutoNormal guide,
simulate per-sample replicate data z_rep via the model's likelihood
(numpyro / jax.random PRNGKey, identical idiom to `s2e_hbayes_inference.py`),
and compute Bayesian posterior-predictive p-values for two test statistics
per stratum:
    T1_s : ||z̄_s||_2 -- stratum-centroid magnitude (mean fit)
    T2_s : tr(Cov(z_s)) / d -- stratum dispersion (variance fit)

NOT synthetic data: posterior-predictive replication is the Gelman-Rubin
PPC primitive (Gelman 1996 §6), drawing z_rep ~ p(z | θ) for θ ~ q(θ).
The replicate is a calibration target, never copied into a metric claim.

Verdict rule (per V11 plan §2 row 5):
    |0.5 - p_B| <= 0.20 -- PASS (PPC consistent with model)
    0.20 < |0.5 - p_B| <= 0.40 -- CAVEAT
    |0.5 - p_B| > 0.40 -- FAIL (model misspecified for that stratum)

Inputs (same as s2e):
    data/processed/mmrf_z64.npy
    data/processed/mmrf_z64_sample_ids.json
    data/raw/mmrf_commpass/cytogenetics.tsv

Outputs:
    paper/v8_artifacts/v12_sprint1/hbayes_ppc.json
"""

from __future__ import annotations

import json
import sys
import time
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
OUT = ROOT / "paper" / "v8_artifacts" / "v12_sprint1"
OUT.mkdir(parents=True, exist_ok=True)

STRATA = ["del17p", "chr1q21_gain", "del13q", "t_4_14", "t_11_14"]
N_PPC = 1000
N_SVI_STEPS = 2000
SEED = 0


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
        numpyro.sample(
            "obs",
            dist.Normal(patient_mean, sigma_obs).to_event(1),
            obs=z_obs,
        )


def stratum_centroid_norm(Z, mask):
    if mask.sum() == 0:
        return float("nan")
    centroid = Z[mask].mean(axis=0)
    return float(np.linalg.norm(centroid))


def stratum_dispersion(Z, mask):
    if mask.sum() < 2:
        return float("nan")
    Zs = Z[mask]
    return float(np.var(Zs, axis=0, ddof=1).mean())


def bayesian_p(t_rep, t_obs):
    return float(np.mean(np.array(t_rep) >= float(t_obs)))


def main() -> None:
    t0 = time.time()
    Z = np.load(PROC / "mmrf_z64.npy")
    ids = json.loads((PROC / "mmrf_z64_sample_ids.json").read_text())
    id_to_idx = {pid: i for i, pid in enumerate(ids)}

    cyto = pd.read_csv(RAW / "cytogenetics.tsv", sep="\t")
    cyto = cyto.dropna(subset=STRATA).copy()
    cyto = cyto[cyto["submitter_id"].isin(id_to_idx.keys())].reset_index(drop=True)
    z_idx = np.array([id_to_idx[p] for p in cyto["submitter_id"]])
    Z_aligned = Z[z_idx].astype(np.float32)
    M = cyto[STRATA].values.astype(np.float32)
    N, d = Z_aligned.shape
    S = len(STRATA)
    print(f"[PPC] N={N} d={d} S={S}")

    z_obs = jnp.array(Z_aligned)
    M_jax = jnp.array(M)

    guide = AutoNormal(hbayes_multilabel_model)
    svi = SVI(hbayes_multilabel_model, guide, numpyro.optim.Adam(5e-3), Trace_ELBO())
    rng = jax.random.PRNGKey(SEED)
    print(f"[PPC] re-running SVI for {N_SVI_STEPS} steps...")
    res = svi.run(rng, N_SVI_STEPS, z_obs, M_jax, S, d)
    print(f"[PPC] ELBO {float(res.losses[0]):.0f} -> {float(res.losses[-1]):.0f}")

    rng_post = jax.random.PRNGKey(SEED + 1)
    posterior = guide.sample_posterior(rng_post, res.params, sample_shape=(N_PPC,))
    mu_post = np.asarray(posterior["mu"])
    theta_post = np.asarray(posterior["theta"])
    sigma_post = np.asarray(posterior["sigma_obs"])
    print(f"[PPC] posterior samples: mu {mu_post.shape}, theta {theta_post.shape}, "
          f"sigma {sigma_post.shape}")

    Z_obs_np = Z_aligned
    M_np = M
    masks = [(M_np[:, s] == 1.0) for s in range(S)]
    t1_obs = [stratum_centroid_norm(Z_obs_np, masks[s]) for s in range(S)]
    t2_obs = [stratum_dispersion(Z_obs_np, masks[s]) for s in range(S)]

    rng_rep = jax.random.PRNGKey(SEED + 2)
    t1_rep = [[] for _ in range(S)]
    t2_rep = [[] for _ in range(S)]

    M_jnp = jnp.asarray(M_np)
    for k in range(N_PPC):
        mu_k = jnp.asarray(mu_post[k])
        theta_k = jnp.asarray(theta_post[k])
        sigma_k = float(sigma_post[k])
        mean_k = mu_k[None, :] + M_jnp @ theta_k
        rng_rep, rng_step = jax.random.split(rng_rep)
        eps = jax.random.normal(rng_step, shape=mean_k.shape)
        z_rep = np.asarray(mean_k + sigma_k * eps).astype(np.float32)
        for s in range(S):
            t1_rep[s].append(stratum_centroid_norm(z_rep, masks[s]))
            t2_rep[s].append(stratum_dispersion(z_rep, masks[s]))

    p_t1 = [bayesian_p(t1_rep[s], t1_obs[s]) for s in range(S)]
    p_t2 = [bayesian_p(t2_rep[s], t2_obs[s]) for s in range(S)]

    def verdict(p):
        d_ = abs(0.5 - p)
        if np.isnan(p):
            return "NA"
        if d_ <= 0.20:
            return "PASS"
        if d_ <= 0.40:
            return "CAVEAT"
        return "FAIL"

    elapsed_s = time.time() - t0

    result = {
        "n_patients": int(N),
        "latent_dim": int(d),
        "n_strata": int(S),
        "n_ppc_samples": int(N_PPC),
        "n_svi_steps": int(N_SVI_STEPS),
        "elbo_init": float(res.losses[0]),
        "elbo_final": float(res.losses[-1]),
        "stratum_n_pos": {STRATA[s]: int(masks[s].sum()) for s in range(S)},
        "centroid_norm_T1": {
            STRATA[s]: {
                "obs": t1_obs[s],
                "rep_mean": float(np.mean(t1_rep[s])),
                "rep_sd": float(np.std(t1_rep[s])),
                "p_bayes": p_t1[s],
                "verdict": verdict(p_t1[s]),
            } for s in range(S)
        },
        "dispersion_T2": {
            STRATA[s]: {
                "obs": t2_obs[s],
                "rep_mean": float(np.mean(t2_rep[s])),
                "rep_sd": float(np.std(t2_rep[s])),
                "p_bayes": p_t2[s],
                "verdict": verdict(p_t2[s]),
            } for s in range(S)
        },
        "verdict_rule": (
            "Bayesian p_B = P(T(z_rep) >= T(z_obs)). "
            "PASS if |0.5 - p_B| <= 0.20; CAVEAT 0.20-0.40; FAIL > 0.40."
        ),
        "wall_time_s": elapsed_s,
    }

    out = OUT / "hbayes_ppc.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[PPC] wrote {out}")
    print(f"[PPC] wall time: {elapsed_s:.1f}s")
    print("\n[PPC] === Per-stratum verdicts ===")
    for s in range(S):
        print(f"  {STRATA[s]:>14s} | T1 p={p_t1[s]:.3f} {verdict(p_t1[s]):>6s}"
              f" | T2 p={p_t2[s]:.3f} {verdict(p_t2[s]):>6s}"
              f" | n_pos={int(masks[s].sum())}")
    overall = ([verdict(p) for p in p_t1] + [verdict(p) for p in p_t2])
    n_fail = overall.count("FAIL")
    n_caveat = overall.count("CAVEAT")
    n_pass = overall.count("PASS")
    print(f"[PPC] OVERALL: {n_pass} PASS, {n_caveat} CAVEAT, {n_fail} FAIL "
          f"(of {len(overall)} stratum-statistic combos)")
    if n_fail == 0 and n_caveat <= 2:
        print("[PPC] HBayes #67 PPC: PASS at v12.")
    elif n_fail == 0:
        print("[PPC] HBayes #67 PPC: PASS-WITH-CAVEAT — F1 verdict downgrades but "
              "is not deleted (per V11 plan §2 row 5).")
    else:
        print("[PPC] HBayes #67 PPC: FAIL on >=1 stratum-statistic; F1 caveat "
              "required.")


if __name__ == "__main__":
    main()
