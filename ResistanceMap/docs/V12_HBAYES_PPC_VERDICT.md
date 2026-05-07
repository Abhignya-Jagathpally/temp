# v12 Sprint 1 — HBayes posterior-predictive check verdict (#67)

**Date:** 2026-05-04
**Run ID:** `r-2026-05-04-v12-hbayes-ppc`
**Driver:** parent session (parallel v12 sprint plan)
**Scope:** Closes the chair-verdict gap §5 row #67
("HBayes posterior-predictive check (Gelman 1996 §6) on F1, per V11 plan §2 row 5").

## §1 Verdict (one line)

**FAIL on 5/10 stratum-statistic combos. F1 strict-pass DOWNGRADES to PASS-WITH-CAVEAT** per V11 plan §2 row 5 ("If PPC fails ≥1 stratum, F1 verdict downgrades from 'pass' to 'pass with caveat' — does NOT delete F1").

Sufficient for **bioRxiv** (chair verdict §6 already says PPC does NOT gate bioRxiv).
**Insufficient for Nature Methods or Statistics in Medicine** without a v12.5 re-run on full HMC.

## §2 Method

Re-ran the Sprint-2 SVI inference (`scripts/v10/s2e_hbayes_inference.py`) with identical RNG seed (PRNGKey(0)) on the same 716-patient cytogenetic-aligned cohort and same z64 latents. Drew **M=1000 posterior samples** from the AutoNormal mean-field variational guide via `guide.sample_posterior`, and for each sample simulated a replicate dataset z_rep via the model's likelihood (jax.random Gaussian, no synthetic data outside the predictive-check primitive). Computed two test statistics per stratum:

- **T1 = ‖z̄_s‖₂** (centroid magnitude) — checks whether the model captures the per-stratum mean
- **T2 = mean of per-dim variance** (dispersion) — checks whether the model captures the per-stratum spread

Bayesian p-value: `p_B = P(T(z_rep) ≥ T(z_obs))`. Verdict rule: PASS if |0.5 − p_B| ≤ 0.20, CAVEAT 0.20–0.40, FAIL > 0.40.

Code: `scripts/v12/s_v12_hbayes_ppc.py`. Output: `paper/v8_artifacts/v12_sprint1/hbayes_ppc.json`. Wall time 9.5s.

## §3 Per-stratum results

| Stratum | n_pos | T1 obs | T1 rep_mean ± sd | p_T1 | T1 verdict | T2 obs | T2 rep_mean ± sd | p_T2 | T2 verdict |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| del17p       | 105 | 15.46 | 13.14 ± 0.96 | 0.014 | **FAIL** | 69.23 | 65.25 ± 5.26 | 0.221 | CAVEAT |
| chr1q21_gain | 242 | 14.72 | 10.69 ± 0.73 | 0.000 | **FAIL** | 60.11 | 59.69 ± 4.31 | 0.583 | PASS |
| del13q       | 375 |  7.91 |  8.54 ± 0.62 | 0.839 | CAVEAT | 67.05 | 65.35 ± 4.30 | 0.366 | PASS |
| t_4_14       | 103 | 24.55 | 15.32 ± 1.02 | 0.000 | **FAIL** | 70.04 | 65.52 ± 5.32 | 0.667 | PASS |
| t_11_14      | 123 | 22.78 | 14.12 ± 0.90 | 0.000 | **FAIL** | 86.40 | 64.40 ± 5.05 | 0.966 | **FAIL** |

**Aggregate: 3 PASS, 2 CAVEAT, 5 FAIL.**

## §4 Interpretation

**T1 systematic under-fit (4/5 strata).** The empirical centroid magnitudes (15.46, 14.72, 24.55, 22.78) substantially exceed the posterior-predictive replicates (13.14, 10.69, 15.32, 14.12). The AutoNormal mean-field variational guide is **shrinking the per-stratum mean toward μ**. This is the classical mean-field mode-collapse / posterior-shrinkage failure: the guide's marginal q(θ_s) is too tight around its mean, so simulated centroids are much smaller than observed.

**T2 t(11;14) over-fit.** The empirical dispersion 86.40 in t(11;14) is *larger* than the simulated 64.40 ± 5.05 (p_B = 0.966 means 96.6% of replicate dispersions are below observed). The shared σ_obs (estimated 7.99) does not capture per-stratum heteroscedasticity. t(11;14) is the smallest stratum (123 / 716 = 17.2%) but has the highest empirical variance — the homoscedastic likelihood model under-estimates this.

**T1 chr1q21_gain (largest stratum at 33.8%) is the cleanest under-fit signal.** With 242 patients the empirical centroid is 14.72 ± SE/√242 ≈ ±0.94, well outside the replicate band [10.69 ± 0.73].

## §5 What this changes

| Manuscript section | Before | After |
|---|---|---|
| F1 verdict (Sprint 2) | STRICT PASS | PASS-WITH-CAVEAT (per-stratum mean fit) |
| F2 verdict | REFUTED at v10 (correct null) | unchanged |
| F8-DPS | STRICT FAIL (operator-robust) | unchanged |
| F_S5 | STRICT PASS (Ridge_v11features 5/5 strata within ±3%) | unchanged — F_S5 is conformal coverage, not Bayesian PPC |
| M1 §3 (Method) | "5-stratum HBayes posterior over 23 v11_richer features" | "5-stratum HBayes (mean-field SVI). Posterior-predictive check fails T1 in 4/5 strata; v12 re-inference on full HMC NUTS pre-registered as a follow-up." |
| M1 §7 (Limitations) | (unchanged) | Add: "Mean-field variational shrinkage under-estimates per-stratum centroid magnitude; the F1 verdict is preserved as a marginal-likelihood claim but is not a posterior-mean-fit claim." |

## §6 v12.5 fix path

Three pre-registered options (v11 plan §2 row 5 wording: "If PPC fails ≥1 stratum, F1 verdict downgrades from 'pass' to 'pass with caveat' — does NOT delete F1"):

1. **NUTS HMC** in NumPyro on the same model. ~30 min wall time at d=64, 5 strata, 716 patients, 4 chains × 1000 warm-up + 1000 draws. Eliminates mean-field shrinkage by drawing from the full posterior. **Lowest-effort fix; recommended next.**
2. **AutoMultivariateNormal** (full-rank) or **AutoLowRankMultivariateNormal** guide. Captures cross-dim correlation in q(θ_s). Cheap (~30s) but does not address the homoscedasticity issue in T2.
3. **Heteroscedastic σ_obs per stratum**: σ_obs → σ_obs_s ~ HalfCauchy(1) with separate posterior for each stratum. Closes T2_t(11;14) FAIL. Independent of guide choice; orthogonal to fix #1.

Recommendation: combine **#1 + #3** in v12 Sprint 2. Fix #1 closes T1 under-fit (4 strata), fix #3 closes T2 t(11;14) over-fit. Re-run PPC; if 0 FAIL → upgrade F1 to STRICT PASS for Nature Methods.

## §7 Pearl-tier impact

**None.** PPC is a calibration check on the Bayesian model fit; it does not promote any claim from L1 (associational) to L2 (interventional/counterfactual). The chair-verdict niche paragraph (§3.7) does not mention HBayes PPC; the §3.1 contribution-1 (calibrated 90% PFS coverage via Mondrian jackknife+) is conformal, not Bayesian, and is unaffected.

## §8 Provenance

- Code: `scripts/v12/s_v12_hbayes_ppc.py`
- Output JSON: `paper/v8_artifacts/v12_sprint1/hbayes_ppc.json`
- Log: `logs/v12/hbayes_ppc.log`
- Inputs: `data/processed/mmrf_z64.npy`, `data/processed/mmrf_z64_sample_ids.json`, `data/raw/mmrf_commpass/cytogenetics.tsv` — identical to s2e
- Seeds: PRNGKey(0) for SVI (matches s2e), PRNGKey(1) for posterior sampling, PRNGKey(2) for replicate Gaussian noise
- Wall time: 9.5 s on CPU
- Run ID: `r-2026-05-04-v12-hbayes-ppc`
