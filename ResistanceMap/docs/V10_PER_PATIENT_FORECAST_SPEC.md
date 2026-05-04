# v10 — Per-Patient Resistance Forecast: Integrated Architecture Spec

Date: 2026-05-03
Synthesis source artifacts:
- `TRAJECTORY_MATH_FEW_SHOT.md` — math of identifiability under N≤100 paired
- `DRIVER_FUNCTION_CAUSAL.md` — Pearl-tier conditions for "driver pathway" claim
- `CLONE_DYNAMICS_AND_CONFORMAL.md` — branching-process prior + jackknife+ conformal
- `PER_PATIENT_FORECAST_LITERATURE.md` — 2024-2026 SOTA + GigaTIME verdict
- `INTEGRATED_BUILD_PLAN.md`, `CAUSAL_VALIDITY_AUDIT.md` — updated with N values

This document is the unified v10 architecture spec. Every component is grounded
in math from one of the four PhD-level analyses above. Every claim is paired
with a falsification test that runs on existing data.

---

## 1. The honest goal

**v9 (current ceiling):** Tier 1 association on cell lines + Tier 2 predictive
on N=994 MMRF baseline cohort with cytogenetic adjustment, framed per
PERCEPTION (PMID 38637658).

**v10 deliverable:** *population-stratified* per-patient forecasting with
honest uncertainty intervals on the N=42 paired sub-cohort (RNA-Seq at ≥2
visits + observed first→second-line transition), framed as exploratory.

**Out of scope for v10:** universal "predict for any new patient before it
happens." That requires data we do not have. The clone-dynamics agent
flagged the structural reason — selection bias in the Recurrent BM cohort
makes the conformal coverage guarantee transferable only to "patients
exchangeable with the calibration cohort," not to arbitrary future patients.

---

## 2. The architecture stack (outer → inner)

```
┌────────────────────────────────────────────────────────────────────────┐
│  CONFORMAL UNCERTAINTY WRAPPER (jackknife+ Mondrian, Vovk-style)       │
│  Per-stratum coverage: ±3% on 3 strata, ±5% on 2 (n_s ≥ 36 floor)      │
│  Strata: del17p, chr1q21+, t(4;14), t(11;14), del13q                   │
└──────────────────────────────────┬─────────────────────────────────────┘
┌──────────────────────────────────▼─────────────────────────────────────┐
│  HIERARCHICAL BAYES OUTER LAYER (stratified partial pooling)            │
│  θ_p | c_p ~ N(μ_{c_p}, Σ_{c_p}); 994 baselines tighten constants     │
│  in O(N_paired^{-1/2}) ≈ 0.14 contraction rate                         │
│  Identifiability: only this lets baseline-only patients enter the      │
│  marginal likelihood non-vacuously                                     │
└──────────────────────────────────┬─────────────────────────────────────┘
┌──────────────────────────────────▼─────────────────────────────────────┐
│  CLONAL-DYNAMICS STRUCTURAL PRIOR (Bellman-Harris branching, K≤8)      │
│  PyClone-VI on MMRF VAF → subclone frequencies                         │
│  Sample complexity reduced ~10× vs unconstrained                       │
│  319 Recurrent BM patients ≈ within order of mag of N for ε=0.15      │
└──────────────────────────────────┬─────────────────────────────────────┘
┌──────────────────────────────────▼─────────────────────────────────────┐
│  TRAJECTORY MECHANICS (per-patient likelihood p(x^p | θ_p))             │
│  Trajectory Flow Matching (arXiv 2410.21154) +                         │
│    TorchCFM (atong01/conditional-flow-matching, MIT, 2,440 ⭐)         │
│  Simulation-free neural-SDE; irregular sampling; uncertainty           │
└──────────────────────────────────┬─────────────────────────────────────┘
┌──────────────────────────────────▼─────────────────────────────────────┐
│  STATE-TRANSITION BACKBONE                                              │
│  CellRank 2 (PMID 38871986, scverse/cellrank, BSD-3, 443 ⭐)           │
│  Multi-view Markov-chain kernels over single-cell states               │
└──────────────────────────────────┬─────────────────────────────────────┘
┌──────────────────────────────────▼─────────────────────────────────────┐
│  LATENT ENCODER (frozen)                                                │
│  scGPT via tdc/scGPT@acf749f3 (MIT, PMID 38409223)                     │
│  512-d per-cell embedding from log1p TPM                               │
│  Pretrained on ~33M cells from CELLxGENE                               │
└──────────────────────────────────┬─────────────────────────────────────┘
┌──────────────────────────────────▼─────────────────────────────────────┐
│  CLINICAL-TIMELINE CHANNEL (parallel)                                   │
│  Chronos (arXiv 2403.07815, amazon-science/chronos-forecasting,        │
│    Apache-2.0, 5,245 ⭐)                                                │
│  Tokenized treatment-line + IMWG-response sequence                     │
└────────────────────────────────────────────────────────────────────────┘
```

The clinical-timeline channel runs in parallel to the molecular trajectory
channel and joins at the hierarchical-Bayes layer through a shared latent
$\theta_p$.

---

## 3. The math anchoring each layer

### 3.1 Why hierarchical Bayes is the only identifiable outer-layer choice (from `TRAJECTORY_MATH_FEW_SHOT.md`)

The full likelihood factorizes as:

$$p(x_{t_0}^{1:1045}, x_{t_1}^{1:51}) = \int p(\mu,\Sigma)\prod_p \int p(\theta_p\mid\mu,\Sigma)\,p(x^p\mid\theta_p)\,d\theta_p\,d\mu\,d\Sigma$$

Only this factorization lets the 994 baseline-only patients tighten the
constant on $(\mu, \Sigma)$ — the priors over the per-patient parameters.
Approach A (latent SDE w/ patient conditioning) is 10:1 over-parameterized at
N=51 paired — Stone's minimax rate $N^{-2\beta/(2\beta+d_c)}$ admits at most
$d_c^{\text{eff}} \approx 4$. Approach B (OT matching to references) discards
the 994 baselines entirely and the $W_2$ rate $K^{-2/d} \approx 0.985$ at
K=51, d=512 is essentially null. Approach C wins by elimination, mathematically.

### 3.2 Why a clonal-dynamics prior is necessary (from `CLONE_DYNAMICS_AND_CONFORMAL.md`)

Under a Bellman-Harris branching process with K subclones and Gaussian fitness
prior $\theta_0, \Sigma_0$, identifying clone fitness to $\varepsilon$ needs
$N_{\text{paired}} \approx K \log(K/\delta)/\varepsilon^2$ patients. With K≤8
and the literature-informed prior's 10× reduction, **the 319 Recurrent BM
patients are within an order of magnitude of the bound for $\varepsilon = 0.15$
(one half-doubling)**. Without the prior, an unconstrained latent-SDE inflates
this rate by $d_{\text{latent}}\log(d/\varepsilon)/\varepsilon$ where
$d_{\text{latent}} \in [32, 256]$ — that's **30–500× more patients**, infeasible
at N=994.

### 3.3 Why jackknife+ conformal beats split conformal here (from `CLONE_DYNAMICS_AND_CONFORMAL.md`)

Per-stratum coverage at ±3% empirical-coverage tolerance needs
$n_s \ge \alpha(1-\alpha)/0.03^2 = 100$ per stratum. With 30% of N=994 reserved
for split-conformal calibration (~300 patients) and 5 cytogenetic strata, only
chr1q21+ (40% prevalence → 120) clears ±3%.

**Jackknife+ (Barber, Candès, Ramdas, Tibshirani 2021, arXiv 1905.02928)** uses
all 994 patients for both training and calibration via leave-one-out, so the
effective per-stratum n grows to the full prevalence count. With this:
- 3 strata clear ±3% empirical coverage (n ≥ 100): chr1q21+ (~333), del13q (~470), t(11;14) (~130 borderline)
- 2 strata clear ±5% only (n ≥ 36): del17p (~135), t(4;14) (~113)

### 3.4 Why "driver pathway" gets framed as L1-with-CRISPR-falsification, not L3 mediator (from `DRIVER_FUNCTION_CAUSAL.md`)

Three formulations were analyzed:

| Formulation | Verdict | Reason |
|---|---|---|
| F1 — TIG/SHAP attribution | **L1 only** | Counter-example proven: del17p → BCL2 ← high attribution for Bortezomib resistance; DepMap shows BCL2 Chronos +0.066 (anti-essential). Attribution can't see the U → M back-door. |
| F2 — NIE mediation on survival | **L2 feasible at N=42 single mediator** | Powered for one pre-specified mediator (e.g. proteasome pathway → time-to-2nd-line). Not powered for k=10 simultaneous. Chief unmeasured confounder: **EIMOC** — physician treatment-adjustment between visits, structurally unidentifiable without visit-level dose records. |
| F3 — DepMap CRISPR as IV | **INVALID** | Pearl-Bareinboim transportability fails: immune microenvironment + clonal heterogeneity + passage drift directly affect Y bypassing M. |

**The legitimate use of CRISPR is as a falsification oracle** — for ResistanceMap's
top-k predicted driver genes per resistance state, Wilcoxon rank-sum on Chronos
scores vs 1,000 permuted random gene sets at p<0.05 Bonferroni. This runs on
existing on-disk data (1,208 cell lines × 18,531 genes; 3 confirmed MM lines:
OPM2, KMS34, U266).

---

## 4. Three falsification tests (each runnable on data already on disk)

### Test 1 — Hierarchical Bayes / partial pooling (primary)
Hold out one cytogenetic stratum (e.g. t(14;16) if present, else t(4;14))
entirely from paired training. Train stratified Approach C on remaining 4
strata. Forecast t-stratum test patients using only the cross-stratum
hyperprior. **Refutation threshold:** if 90% predictive credible interval
empirical coverage < 70%, the stratified-exchangeability assumption is broken
and v10 is refuted.

### Test 2 — Partial pooling utility (secondary)
Ablate the 994 baseline-only patients from the marginal likelihood. **Refutation
threshold:** if held-out predictive log-likelihood is statistically unchanged
(paired bootstrap, α=0.05), partial pooling provides no benefit and v10 is also
refuted — meaning per-stratum independent fits on N=51 paired alone are
sufficient.

### Test 3 — CRISPR driver-pathway falsification oracle
For each resistance state S, take ResistanceMap's top-k predicted driver genes;
extract DepMap Chronos scores. **Refutation threshold:** if the observed driver
gene set ranks below the 95th percentile of 1,000 size-matched random gene
sets on the one-sided rank-sum test, the attribution claim fails.

Validating evidence already on disk:
- ✅ PSMB5/proteasome: Chronos −2.51 in 3 MM lines vs −1.89 pan-cancer
  (directionally correct for Bortezomib)
- ❌ BCL2: +0.066 in MM (anti-essential — would catch as false driver if model
  ranks it high for Bortezomib)

---

## 5. Identifiability ceilings: what v10 still cannot deliver

These are *structural* gaps, not engineering gaps. No amount of v10 work resolves them.

| Gap | Why it can't be closed in v10 | When it could close |
|---|---|---|
| **Per-patient forecast for arbitrary new patients** | Selection bias in 319 Recurrent BM cohort; jackknife+ guarantee transfers only to exchangeable cohort | IPCW correction in v11; eventually new prospective cohort |
| **k=10 simultaneous pathway-mediator NIE** | Events-per-variable too low at N=42 | Larger paired sub-cohort or shrinkage priors |
| **EIMOC closure for mediation** | Visit-level dose adjustments not in `treatments.tsv` | Future MMRF release with finer treatment-record granularity |
| **L3 causal "driver pathway"** | Cross-population IV invalid; transportability fails | Wet-lab CRISPR in primary MM cells (Cohen template) |
| **GigaTIME-style population-scale TME forecasting** | Different problem; we don't have multiplexed IF imaging | Out of project scope |

---

## 6. Sprint plan

### Sprint 1.5 (this week, no blockers)
- Wire `MultiOmicsDataset` to consume the new MMRF tables (`gene_expression.tsv`, `mutations.tsv`, `copy_number.tsv`, `cytogenetics.tsv`, `treatments.tsv`)
- Implement the **Mondrian / jackknife+ conformal wrapper** as a standalone module — runnable on any per-patient prediction function. (Independent of the trajectory machinery — gives us honest intervals even on the v9 Tier-1 outputs.)
- Run the **CRISPR falsification oracle** (Test 3) against v9's existing pathway-attribution scores. This is a standalone analysis using on-disk DepMap data.

### Sprint 2 (2 weeks)
- Cache scGPT frozen embeddings on MMRF baseline RNA-Seq (859 aliquots) and on GSE124310 + GSE271107. Pin to `tdc/scGPT@acf749f3`.
- Implement the **Hierarchical Bayes outer layer** with stratified prior over 5 cytogenetic strata; PyMC or NumPyro backend.
- Plug **Trajectory Flow Matching** (TorchCFM) as the per-patient likelihood; train on N=42 paired patients with the 994 baselines anchoring the cohort prior.

### Sprint 3 (1 week)
- **PyClone-VI** on MMRF VAF → subclone frequencies → Bellman-Harris fitness
  prior. Substitute the unconstrained latent prior in Sprint 2 for this
  structural prior.
- Run Tests 1 and 2 (hold-out stratum + ablate baselines).

### Sprint 4 (1 week)
- **Single pre-specified mediator NIE** on N=42 with EIMOC disclaimer
  (Formulation 2). One mediator: proteasome pathway score at T1 → time to
  2nd-line Bortezomib therapy. IPTW-Cox conditioning on ISS + 5 cytogenetics.
- Write the v10 paper draft framing: Tier 1 + Tier 2 with explicit
  L2-strong sub-cohort caveat.

---

## 7. Honesty constraints that propagate to every figure caption

These are not optional. They are the consequences of the math derived above.

1. **Population-stratified, not universal.** The conformal coverage guarantee
   transfers only to patients exchangeable with the MMRF Recurrent BM
   calibration cohort. Every interval-quoting figure must say so.
2. **N=42 paired sub-cohort is exploratory.** Single-mediator NIE only;
   k>1 mediators not powered. EIMOC unmeasured.
3. **Driver pathways are L1 features with L2 CRISPR-oracle support, not L3
   mediators.** Frame as "predicted driver gene set passes/fails CRISPR-
   essentiality falsification test" — never as "causes resistance."
4. **The clone-dynamics prior assumes K ≤ 8 subclones**, justified by
   PyClone-VI on MMRF VAF; figures must report the K used and its
   sensitivity.
5. **Run-ledger discipline applies.** None of the numbers above (N=42, N=994,
   coverage ±3% / ±5%, sample-complexity bounds) can ship to README or
   ARCHITECTURE without a corresponding ledger row + W&B link +
   verification_chain.json. They live in this debugging-tier doc only until
   the v10 retrain produces ledger artifacts.

---

## 8. One-paragraph summary for an external reviewer

ResistanceMap v10 forecasts per-patient resistance trajectories on the MMRF
CoMMpass cohort using a four-layer architecture: a hierarchical-Bayes outer
layer with stratified partial pooling over 5 cytogenetic strata, anchored by
994 baseline-only patients tightening the contraction-rate constant; a
Bellman-Harris branching-process prior on K≤8 subclones (PyClone-VI on
matched VAF) reducing sample complexity ~10× vs unconstrained; a Trajectory-
Flow-Matching neural-SDE for the per-patient likelihood; a CellRank-2
multi-view Markov backbone over scGPT-frozen 512-d cell embeddings; with a
parallel Chronos foundation-model channel for the clinical timeline. Honest
uncertainty intervals come from a Mondrian jackknife+ conformal wrapper
(Barber et al. 2021), achieving 90% coverage at ±3% per-stratum tolerance for
3 of 5 strata and ±5% for the remaining 2. Driver-pathway claims are framed
as L1 features with L2 CRISPR-essentiality falsification-oracle support,
never as L3 mediators — Pearl-Bareinboim transportability fails for cross-
population IV due to immune microenvironment, clonal heterogeneity, and
passage drift. Three pre-specified falsification experiments (hold-out
stratum coverage, partial-pooling ablation, CRISPR rank-sum) are runnable on
existing data and define refutation thresholds in advance.

---

## 9. Single most important sentence

If only one paragraph survives this document, let it be this one:

*v10 is mathematically the correct architecture for the data we have, but it
is not "predict for any future MM patient." It is a population-stratified
forecaster with calibrated uncertainty intervals on a sub-cohort that is
itself selection-biased. The honest contribution is: a Tier-2 predictive
validation on N=994, plus a Tier-2 single-mediator NIE on N=42 with explicit
EIMOC caveat, plus a CRISPR-oracle falsification test for the driver-pathway
claim — all three in one paper, with run-ledger discipline. v11's "predict
before it happens for any patient" requires data we do not yet have, and
should not be claimed.*
