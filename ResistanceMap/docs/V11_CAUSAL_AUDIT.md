# Causal-Inference Audit — ResistanceMap v11.5 SOTA-Tying Discrimination Claim

**Date:** 2026-05-03
**Auditor:** Causal-inference audit agent (Pearl/Rubin tradition)
**Claim under review:** "Cox-PH ensemble with v11 Waddington features + mmSYGNAL routed score TIES Murie/Baliga 2025 6-submodel mmSYGNAL on N=787 MMRF (marginal C=0.6955 vs 0.6957; paired z-test p=0.994; B=10000 bootstrap p=0.989). F_S5 calibrated 90% coverage strict pass with v11 features. Pearl-tier ceiling: L1-with-structural-prior."
**Evidence base:** `docs/V11_SPRINT5_CONFORMAL_VERDICT.md` §3.6–§3.11; `paper/v8_artifacts/v11_sprint5/cox_with_programs.json`; `paper/v8_artifacts/v11_sprint5/paired_cindex_test.json`; `paper/v8_artifacts/v11_sprint5/mmsygnal_complete_routing.json`; `paper/v8_artifacts/v10_sprint7/f8_energy_distance.json`; `paper/v8_artifacts/v11_sprint7/f8_neural_ode.json`; `docs/CAUSAL_VALIDITY_AUDIT.md`; `RUNS.md`

---

## 1. Implied causal graph for the v11.5 Cox ensemble

The v11.5 model takes as input a baseline (time-0) snapshot of:
- Patient cytogenetic risk indicators (del17p, t(4;14), +1q21, t(11;14)) from MMRF cytogenetics.tsv
- Baseline expression principal components (PCA-64 from MMRF STAR RNA-seq)
- Four Waddington-landscape scalars derived from the v11 Neural-ODE forward pass on the baseline latent: U(z0), U(z(T)), ||nabla U(z0)||, ||z(T) - z0||
- The mmSYGNAL routed risk score: one scalar produced by applying the Murie/Baliga 2025 pre-trained subtype-specific glmnet models to the IA12 program-activity matrix and routing by subtype grade priority (A > B > C)

Output: Cox partial-likelihood log-hazard at baseline; evaluated against time-to-next-line-therapy (TT2L) outcome.

**DAG (text form):**

```
U (cytogenetic risk, partly measured) ---+
                                         |
X_expression (baseline RNA, measured) ---> eta_v11 (Cox log-hazard) --> C-index on TT2L
                                         |
X_program (IA12 activity, from mmSYGNAL) +
X_Waddington (ODE features, from v11) --+

U -----> TT2L (direct path; cytogenetics also directly determine PFS)
U -----> treatment_assignment (back-door: physicians prescribe by cytogenetic risk)
treatment_assignment ----> TT2L
```

**Back-door paths still open in the training data:**
1. U (unmeasured physician preference, ECOG, center effects) -> treatment -> TT2L
2. The mmSYGNAL program-activity features are pre-trained on CoMMpass IA12 data that overlaps with the same cohort used for evaluation (see §3 below).

The model does NOT include a do-operator. It conditions on observed baseline features; it does not simulate intervention.

---

## 2. Causal assumptions required for each claim

| Claim | Required assumption | Holds in training data? | Evidence / RUNS.md row |
|---|---|---|---|
| "C=0.6955 ties SOTA" (L1 discrimination) | Consistency of C-index estimand: same patients, same outcome, same time origin across both models | YES — both models evaluated on identical N=787, same TT2L, same strata; paired test removes any between-cohort confounding | `r-2026-05-03-v11s5-paired-cindex-test`; `paired_cindex_test.json` delta=-0.00014, SE=0.01384, p=0.994 |
| "C=0.6955 predicts PFS" (ranking patients by baseline features) | L1 only: exchangeability-for-ranking (patients with the same covariate profile have the same expected concordance ordering). NOT unconfoundedness in the causal sense | PARTIAL — cytogenetic back-doors are measured and included as covariates; physician-assignment confounding and ECOG remain unmeasured | `docs/CAUSAL_VALIDITY_AUDIT.md` §3.1-§3.4 |
| "F_S5: 90% calibrated prediction intervals" (conformal coverage) | Exchangeability of calibration and test scores (jackknife+ BCRT 2021 guarantee is base-learner-agnostic and requires only exchangeable residuals) | YES — same MMRF cohort, LOO protocol, same outcome; marginal coverage 0.898 (nominal 0.900) | `r-2026-05-03-v11s5-conformal`; `mondrian_jk_plus_v11.json` |
| "v11 Waddington features add +0.010" (incremental attribution) | Features carry independent signal from the outcome: neural-ODE scalars derived from U_theta are not functions of TT2L itself | YES — U_theta was trained via DSM on cross-sectional baseline latents only, with no exposure to TT2L labels; ODE scalars are a deterministic function of baseline z0 and the frozen checkpoint | `r-2026-05-03-v11s1-ode-diag`; `r-2026-05-03-v11s5-cox` |
| "Waddington landscape position predicts where the patient transitions" (interpretive) | Causal identification of potential energy landscape with future disease state: requires that gradient descent in latent space corresponds to observable biological transitions | NOT HELD — F8 DPS strict-fail (0/16 cells) proves the U_theta prior drift does not beat constant prediction for the 29 paired patients; the landscape is structurally well-defined but dynamically uninformative at N_paired=29 | `r-2026-05-03-v10s7`; `r-2026-05-03-v11s7-ode-dps`; both F8_strict_pass=false |
| "Pearl-tier ceiling: L1-with-structural-prior" | Acknowledgment that the ensemble ranks patients under observational conditioning, not under do-calculus intervention | HELD — no L2 claim is made; the structural prior (Helmholtz-Hodge, PPI Tikhonov) constrains the model class but does not enable identification of interventional effects | Every sprint verdict `docs/V11_SPRINT5_CONFORMAL_VERDICT.md` §3.11.4, §3.8.7 |

---

## 3. The five audit questions — findings

### Question 1: Is the paper claim language consistent with L1 vs L2?

**Finding: YES for the core C-index claim. ONE latent overclaim risk identified.**

The stated claim reads: "TIES Murie/Baliga 2025 6-submodel mmSYGNAL on N=787 MMRF (marginal C=0.6955 vs 0.6957)." This is unambiguously L1 language — it quantifies concordance between a predicted score and an observed ranking. It names the cohort (N=787 MMRF), the comparator (mmSYGNAL), and the test (paired z-test p=0.994, bootstrap p=0.989). No do-operator is implied.

**Latent overclaim risk:** The phrase "Pearl-tier ceiling: L1-with-structural-prior" in the claim line is correct but should appear explicitly in the manuscript abstract and intro, not only in internal audit documents. If absent from the manuscript, a reader who observes the Waddington-landscape framing (potential energy, gradient flow, drift-diffusion) might infer causal or interventional content that the data do not support.

**Recommended abstract language:**

> "We report associative (Pearl L1) prognostic discrimination of TT2L on MMRF CoMMpass (N=787) using a Cox-PH ensemble combining v11 Waddington-landscape features with the mmSYGNAL transcriptional-program risk score. The marginal concordance index (C=0.6955) is statistically indistinguishable from the Murie/Baliga 2025 mmSYGNAL benchmark (C=0.6957; paired U-statistic z-test p=0.994; B=10,000 bootstrap p=0.989). All claims are observational; no interventional (do-calculus) interpretation is supported."

**Recommended intro language:**

> "The Pearl-tier ceiling of v11.5 is L1-with-structural-prior: the model ranks patients by expected TT2L concordance from observational baseline features, and the structural prior (Helmholtz-Hodge decomposition of the learned vector field; PPI Tikhonov regularization) constrains the model class without enabling causal identification. Per-patient longitudinal predictions at the interventional tier (L2) are explicitly pre-registered as beyond the current scope, as demonstrated by the F8 diffusion-posterior strict fail (both v10 single-Euler and v11 Neural-ODE forward operators; 0/16 hyperparameter cells; N_paired=29 < Stone minimax bound for d=64)."

**Verdict: PASS with required language addition (see §6).**

---

### Question 2: Does F8-DPS refutation correctly bound the Pearl ceiling?

**Finding: YES, the F8 refutation correctly bounds the ceiling. No accidental counterfactual claim is made in the stated claim line.**

The F8 results across both operators are unambiguous:

- **v10 single-Euler** (`r-2026-05-03-v10s7`; `f8_energy_distance.json`): 0/16 cells pass; best p_vs_const=0.432, p_vs_pop=0.050 (not jointly significant). ED_prior approx ED_const across the full (eta, sigma) grid.
- **v11 Neural-ODE dopri5** (`r-2026-05-03-v11s7-ode-dps`; `f8_neural_ode.json`): 0/16 cells pass; best p_vs_const=0.402, p_vs_pop=0.064. Marginally tighter than v10 (corrected drift direction), but still fails.

The consistency is structural: at N_paired=29 the Stone minimax rate bounds the identifiable conditioning dimension at approximately 4, versus the 64-dimensional latent space. This is not a model-engineering failure — the v11 Neural-ODE forward operator is a strictly more correct integration of the same drift field, yet the result is identical (F8 fail, operator-robust). The scientific conclusion is that **the identifiability failure is in the data, not the model.**

**Critical check:** Does the claim "predicts PFS ranking from baseline features" (L1, which the v11.5 ensemble does achieve) accidentally imply "predicts where the patient's disease state will evolve" (L2, which F8 refutes)?

The claim line does not use language like "forecasts", "trajectory", or "dynamic prediction." It uses "discrimination" and "C-index", which are explicitly concordance-ordering concepts at a fixed time horizon. The Waddington language in the model *description* (potential energy, gradient flow) describes the regularization architecture, not a claim about future state prediction. This distinction must be explicit in the manuscript.

**Latent confusion point:** The manuscript describes Waddington features as "derived from the Neural-ODE forward pass at T=1." A reader could interpret "T=1" as predicting the patient's future state at time T=1 (years). This is not what the features represent — they are baseline-level scalars characterizing the gradient landscape at the time-0 latent, not forecasted future positions. The distinction must be stated explicitly:

> "The four Waddington-derived covariates (U(z0), U(z(T)), ||nabla U(z0)||, ||z(T)-z0||) characterize the geometry of the learned potential landscape at the patient's baseline latent position. They are not predictions of future disease states; the F8 diffusion-posterior test demonstrates that the U_theta prior does not carry temporally predictive content at N_paired=29 (Sprint 7, both operators, 0/16 cells pass)."

**Verdict: PASS — ceiling is correctly bounded. Required clarifying language in Methods.**

---

### Question 3: Does borrowing the mmSYGNAL routed score introduce confounding in the v11.5 ensemble's stated scope?

**Finding: YES — one identification-level issue and one scope issue, both manageable with explicit disclosure.**

**Issue 3a: The mmSYGNAL score is pre-trained on the same cohort used for evaluation.**

The Murie/Baliga 2025 mmSYGNAL models (six caret/glmnet .Rds files at `baliga-lab/mmSYGNAL-risk-prediction-models`) were trained on CoMMpass IA12 data. The v11.5 evaluation cohort is MMRF CoMMpass IA22 (N=787 patients, all IA22 identifiers). IA22 is an update of IA12 and includes overlapping patients.

The degree of patient overlap between mmSYGNAL's IA12 training set and the IA22 evaluation N=787 is not precisely quantified in any sprint artifact. If substantial overlap exists, the mmSYGNAL routed score is partially trained-on-test, and its C-index contribution in the v11.5 ensemble is inflated by overfitting, not generalizable signal. The v11.5 C-index then inherits that inflation.

This does NOT affect the **paired test** conclusion (TIE is still valid — both models face the same cohort on the same patients), but it does affect the **absolute C-index claim** and any claim about generalizing C=0.6955 to a new cohort.

**Disclosure required:**

> "The mmSYGNAL pre-trained models were developed on CoMMpass IA12; our evaluation cohort is IA22 (N=787). Patient overlap between IA12 and IA22 is not zero; the mmSYGNAL routed score may therefore be partially evaluated on patients it was exposed to during training, which could inflate its C-index contribution in the v11.5 ensemble. External validation on an independent MM cohort (e.g., HOVON-65/GMMG-HD4, NCT00416195; or IFM 2009, NCT01191060) is required before the C=0.6955 figure can be generalized beyond this MMRF sample."

**Issue 3b: mmSYGNAL is a TF-network activity model routed by clinical subtype.**

The mmSYGNAL score is not a simple gene expression summary — it embeds SYGNAL network inference (transcription-factor regulon activity) plus caret/glmnet per-subtype Cox-risk weighting. When this score is added to the v11.5 Cox model as a single scalar covariate, the Cox coefficient on that scalar absorbs all the mmSYGNAL feature engineering as a black-box signal. The Cox model cannot disentangle which part of the mmSYGNAL score's prognostic value comes from TF activity versus clinical subtype routing versus the glmnet regularization.

This does not violate L1 scope — the combined model is still an associative ranking tool. But it means attribution of the C=0.6955 number to "v11 architectural features" versus "mmSYGNAL feature engineering" is not achievable from the Cox coefficients alone. The §3.11 decomposition in `V11_SPRINT5_CONFORMAL_VERDICT.md` addresses this honestly (adding mmSYGNAL routed score to v11_richer: delta_C = +0.042; adding v11_richer to mmSYGNAL: delta_C = -0.0002). The decomposition confirms that ~95% of the SOTA-tying lift is from mmSYGNAL, not from v11 architecture.

**This is Caveat #2 from the pre-registered context — already acknowledged as closed.** The decomposition is in the evidence base. The manuscript must report it. Specifically:

> "Of the +0.042 C-index gain over the pure v11 Cox baseline (Cox_v11_richer = 0.6538), approximately all is attributable to adding the mmSYGNAL routed risk score. Adding the v11_richer feature set on top of the raw six mmSYGNAL submodel scores adds +0.010 C-index (0.6771 to 0.6875); adding it on top of the routed score adds -0.0002. The SOTA-tying C=0.6955 cannot be attributed to v11 architectural novelty; it is a demonstration that v11 and mmSYGNAL carry complementary signal, and that the combination ties the strongest published MM-PFS discriminator on this cohort."

**Verdict: HEDGE-NEEDED on absolute C-index generalizability (Issue 3a); PASS on L1 scope consistency (Issue 3b is disclosed via §3.11 decomposition, requires manuscript presence).**

---

### Question 4: What clinical actions can a reader correctly take from C=0.6955 PFS discrimination at L1?

**Finding: Three L1-appropriate clinical uses; two uses that are NOT supported.**

**Appropriate uses (L1, observational):**

1. **Stratified trial enrollment:** C=0.6955 discrimination on baseline features can be used to enrich a clinical trial for high-risk patients (e.g., enroll the top-tercile predicted-risk patients into an intensification arm). This is a ranking application — the clinician uses the score to order patients, not to predict what will happen if a specific drug is given.

2. **Population-level prognostic benchmarking:** Reporting C=0.6955 alongside the mmSYGNAL benchmark (C=0.6957) establishes that the v11.5 ensemble is competitive with the state of the art for MM-PFS prognostication from baseline multi-omics. This is a fair characterization of the model's discriminative power.

3. **Calibrated interval reporting for individual patients:** The F_S5 Mondrian jackknife+ 90% coverage pass (5/5 cytogenetic strata within +/- 3% of nominal 0.90) supports reporting calibrated prediction intervals for individual patients' TT2L. These intervals are calibrated in the conformal sense — they contain the true TT2L with at least 90% marginal frequency within each cytogenetic stratum, regardless of the underlying model. A clinician can correctly say "for this del17p patient, the model's 90% prediction interval for TT2L is [X, Y] days." This does NOT imply the point prediction is causal.

**Uses NOT supported at L1:**

4. **Treatment selection:** C=0.6955 tells a clinician who will progress sooner (on the drugs they received). It does NOT tell the clinician which drug would prevent or delay progression for a specific patient. Treatment selection requires P(TT2L | do(drug=D), baseline) — an L2 quantity. The F8 DPS fail (0/16 cells) directly refutes any dynamic treatment-selection claim. The Sprint 4 NIE finding (proteasome mediator CI [+0.029, +0.152]) offers supporting evidence for a proteasome-pathway role, but the E-value at F10 (1.20 < pre-registered threshold of 1.5) precludes claiming robustness to unmeasured confounding.

5. **Per-patient trajectory prediction:** Telling a patient "your disease will transition through pathway X at time T" requires L2 (interventional) or L3 (counterfactual) identification. F8 DPS fail rules this out at the current N_paired=29.

**Recommended manuscript language for clinical implications section:**

> "The v11.5 C-index (0.6955) supports population-level prognostic stratification from baseline multi-omics. Calibrated 90% prediction intervals (F_S5 strict pass, 5/5 cytogenetic strata) support patient-level interval reporting. These are observational (Pearl L1) applications. The model does not support treatment selection decisions — predicting the effect of administering a specific drug requires interventional identification (Pearl L2), which is not achievable from this observational cohort (F8 DPS strict fail at N_paired=29 < Stone minimax bound for d=64, robust across v10 and v11 forward operators)."

**Verdict: PASS for prognostic stratification and interval reporting; FAIL for treatment selection or trajectory prediction claims.**

---

### Question 5: Is the v11.5 claim defensible at L1-with-structural-prior, or does the paper need additional language hedges?

**Finding: The claim is defensible at L1-with-structural-prior subject to four specific language hedges. Without those hedges, the Waddington framing creates L2 appearance at L1 substance.**

**Evidence for defensibility:**

- The paired statistical tests are rigorous: U-statistic z-test (p=0.994) and B=10,000 paired percentile bootstrap (marginal p=0.989, stratified p=0.979) are both two-sided, use the unbiased 6-submodel mmSYGNAL reference (C=0.6957, corrected from 4-submodel C=0.6938 in Caveat #3), and the 99% CI of [-0.035, +0.036] rules out any advantage or disadvantage larger than 3.6 C-index points at the 1% level.
- The pre-registered ceiling (L1 + structural prior, never L2 without N_paired > Stone bound for d=64) is documented in the spec and consistently enforced across all seven v10 sprints and all three v11 sprints.
- The decomposition (Caveat #2, §3.11) honestly attributes ~95% of the SOTA-tying lift to mmSYGNAL features, not v11 architecture.
- The F8 DPS fail is documented under both operators and correctly bounds the L2 ceiling.

**Required language hedges (four specific items):**

**Hedge H1 — Abstract: Observational scope declaration.**
Current risk: the abstract may not include the word "observational" or "Pearl L1." Required: the abstract must state the Pearl-tier ceiling explicitly (see recommended language in §3, Question 1 above).

**Hedge H2 — Methods: Waddington features are geometry, not forecast.**
Current risk: "derived from the Neural-ODE forward pass at T=1" sounds like a time-T forecast. Required: clarify that U(z(T)), ||z(T)-z0||, etc. are baseline-latent-geometry descriptors, not predictions of future state (see recommended language in §3, Question 2 above).

**Hedge H3 — Results or Discussion: mmSYGNAL cohort overlap disclosure.**
Current risk: C=0.6955 may be partially inflated if IA12 training patients overlap with IA22 evaluation cohort. Required: disclose the overlap risk and state that external validation is needed before generalizing the absolute C-index figure (see recommended language in §3, Question 3, Issue 3a above).

**Hedge H4 — Clinical implications: Separation of prognostic from treatment-selection use.**
Current risk: a clinical reader may interpret C=0.6955 discrimination as supporting drug selection. Required: the clinical implications section must explicitly state that treatment selection is not supported at L1 (see recommended language in §3, Question 4 above).

---

## 4. Verdict table

| Claim element | Pearl tier | Evidence | Verdict | Required action |
|---|---|---|---|---|
| Marginal C=0.6955 ties mmSYGNAL C=0.6957 on N=787 MMRF | L1 (concordance ordering) | `paired_cindex_test.json`: z=-0.008, p=0.994; bootstrap p=0.989 | PASS | None for the statistical claim itself |
| F_S5 90% calibrated coverage strict pass | L1 (conformal, base-learner-agnostic) | `mondrian_jk_plus_v11.json`: 5/5 strata within +/-3%; `r-2026-05-03-v11s5-conformal` | PASS | None |
| Waddington features add +0.010 incremental C-index over raw mmSYGNAL scores | L1 (incremental attribution in Cox regression) | `cox_with_programs.json`: Cox_v11_richer+6raw=0.6875 vs Cox_only_6=0.6771, delta=+0.0104 | PASS | Report the decomposition table (§3.11) in manuscript |
| Pearl-tier ceiling L1-with-structural-prior correctly declared | L1 + structural prior | All sprint verdicts; F8 DPS fail (0/16) both operators | PASS | Add explicit ceiling language to abstract and intro (Hedge H1, H2) |
| F8 DPS refutation bounds L2 ceiling, both operators | Identification bound | `f8_energy_distance.json` and `f8_neural_ode.json`: F8_strict_pass=false, 0/16 cells, operator-robust | PASS | Reference both run IDs (`r-2026-05-03-v10s7`, `r-2026-05-03-v11s7-ode-dps`) in manuscript |
| mmSYGNAL score as input feature does not introduce L2 scope | L1 — score is a conditioning variable, not an intervention | Feature is a scalar covariate in an observational Cox; does not simulate do(program_activity) | PASS | Disclose cohort-overlap risk (Hedge H3); report §3.11 decomposition (Hedge disclosure) |
| C=0.6955 supports treatment selection for individual patients | L2 required | Not supported; F8 fail; no randomized assignment; F10 E-value < 1.5 | FAIL | Hedge H4: clinical implications section must explicitly exclude this use case |
| C=0.6955 generalizes to independent external cohorts | L1 but requires external validation | mmSYGNAL IA12/IA22 cohort overlap unquantified; only one cohort evaluated | CONDITIONAL | Hedge H3 + pre-register external validation (HOVON-65, IFM 2009, or equivalent) |

---

## 5. RUNS.md row IDs cited

| Run ID | What it provides to this audit |
|---|---|
| `r-2026-05-03-v11s5-paired-cindex-test` | Primary statistical evidence for TIE (z-test p=0.994, B=10000 bootstrap p=0.989) |
| `r-2026-05-03-v11s5-cox-with-programs` | C-index = 0.6955 for Cox_v11_routed_mmsygnal; decomposition anchors |
| `r-2026-05-03-v11s5-mmsygnal-complete-routing` | Unbiased 6-submodel mmSYGNAL reference C=0.6957; Caveat #3 closure |
| `r-2026-05-03-v11s5-conformal` | F_S5 strict pass for Ridge_v11features; GBM rejected |
| `r-2026-05-03-v10s7` | F8 DPS strict fail (v10 single-Euler forward operator, 0/16 cells) |
| `r-2026-05-03-v11s7-ode-dps` | F8 DPS strict fail (v11 Neural-ODE dopri5, 0/16 cells); operator-robustness confirmed |
| `r-2026-05-03-v11s5-mmsygnal` | Head-to-head establishing +0.040 pure-v11-vs-mmSYGNAL gap; motivates v11.5 |
| `r-2026-05-03-v11s5-cox` | Cox loss vs MSE lift: +0.116 C-index from same features |

---

## 6. Required changes — priority-ordered

**Change 1 (Hedge H1, HIGH PRIORITY): Add Pearl-tier ceiling declaration to abstract.**
Current state: not in stated claim line (appears only in internal sprint verdict docs and RUNS.md).
Required text: see Question 1 recommended abstract language above.
Without this: reviewers familiar with Waddington landscape literature may interpret the model as a biological dynamical system capable of L2 inference.

**Change 2 (Hedge H2, HIGH PRIORITY): Clarify Waddington features as geometry descriptors, not forecasts.**
Current state: the sprint verdicts correctly note "baseline-level scalars," but the paper methods section has not been audited for this framing.
Required text: see Question 2 recommended Methods language above.
Without this: "derived from the Neural-ODE forward pass at T=1" reads as a future-state prediction.

**Change 3 (Hedge H3, MEDIUM PRIORITY): Disclose IA12/IA22 cohort overlap risk.**
Current state: not disclosed in any public-facing document.
Required text: see Question 3, Issue 3a recommended language above.
Without this: C=0.6955 may be slightly inflated and not reproducible on a fully independent cohort.

**Change 4 (Hedge H4, HIGH PRIORITY): Separate prognostic from treatment-selection claims in clinical implications.**
Current state: implicit — the sprint verdicts correctly confine claims to L1, but this is not explicitly stated in the clinical framing.
Required text: see Question 4 recommended clinical implications language above.
Without this: a clinical reader may misinterpret prognostic discrimination as supporting drug selection, which the data and F8 fail do not support.

**Change 5 (MEDIUM PRIORITY): Report the §3.11 decomposition table in the manuscript Results section.**
Current state: in `V11_SPRINT5_CONFORMAL_VERDICT.md` §3.11 and `cox_with_programs.json`, but not yet surfaced as a manuscript table.
Required: decomposition table showing pure mmSYGNAL (C=0.6957), pure v11_richer (C=0.6538), combined v11.5 (C=0.6955), and the incremental attribution of +0.042 to mmSYGNAL vs +0.010 to v11 on top of raw 6 scores.
Without this: the SOTA-tying claim overstates the v11 architectural contribution.

**Change 6 (LOW PRIORITY, v12 pre-registration): Pre-register external validation cohort.**
Recommended: HOVON-65/GMMG-HD4 (NCT00416195) or IFM 2009 (NCT01191060) or CoMMpass IA23 held-out. This converts the absolute C=0.6955 from a single-cohort finding to an independently replicated one.

---

## 7. Aggregate verdict

| Scope | Verdict |
|---|---|
| L1 discrimination claim (C=0.6955 TIE) | PASS |
| L1 conformal calibration claim (F_S5) | PASS |
| L1 attribution decomposition | PASS (disclosure required in manuscript) |
| Pearl-tier ceiling enforcement | PASS (language hedges H1, H2 required) |
| Treatment-selection clinical use | FAIL (excluded by L1 scope and F8 DPS fail) |
| Per-patient trajectory prediction | FAIL (excluded by F8 DPS fail, operator-robust) |
| Absolute C-index generalizability to external cohorts | CONDITIONAL (cohort overlap disclosure required, external validation recommended) |

**Overall verdict: CONDITIONAL PASS.** The v11.5 claim is defensible at L1-with-structural-prior. It is not defensible without Hedges H1, H2, H4 in the manuscript text and the §3.11 decomposition table in Results. With those additions the claim is scientifically honest and consistent with the pre-registered ceiling.
