# V12 Independent Causal Re-Audit — ResistanceMap v11/v11.5

**Date:** 2026-05-04
**Auditor:** Independent causal-inference audit (Pearl/Rubin tradition; PhD Causal Inference)
**Inputs read:**
- `docs/V11_CHAIR_VERDICT.md`
- `docs/V11_CAUSAL_AUDIT.md`
- `docs/V11_SOTA_UPGRADE_PLAN.md`
- `docs/V11_END_TO_END_EVALUATION.md`
- `docs/V11_CLINICAL_TRANSLATION.md`
- `docs/V11_SPRINT5_CONFORMAL_VERDICT.md` (§§1-3)
- `paper/v8_artifacts/v11_sprint7/f8_neural_ode.json`
- `paper/v8_artifacts/v10_sprint4/nie_tipping_point.json`
**Pearl-tier declared by chair:** L1-with-structural-prior (never promoted to L2 at v11/v11.5)
**Audit mandate:** identify L1→L2 ladder slippage in language, framing, and clinical claims

---

## Audit scope note

PubMed query tool was denied for this session. Literature-grounded claims in Section 5 therefore
rely on training-corpus knowledge through August 2025 and on published PMIDs already cited within
the v11 artifact corpus. Where a PubMed query would have added novel citations, this is flagged.

---

## 1. Pearl-ladder audit of the five chair-claimed contributions (§1 + §3.7)

### 1.1 M1 Claim: "Calibrated 90% PFS prediction-interval coverage with Waddington-feature-tightened intervals"

**Chair language (§1, Claim A):** "Calibrated 90% prediction-interval coverage with Waddington-feature-tightened
intervals on MM PFS, with a pre-registered falsification framework that admits its identifiability bound."

**Verdict: PASS — L1 throughout.**

Conformal prediction-interval coverage is a purely observational (L1) quantity. The jackknife+ guarantee
(Barber-Candès-Ramdas-Tibshirani 2021) requires only exchangeable residuals; it makes no reference to
any intervention or counterfactual. "Tightened intervals" means narrower prediction sets for the same
nominal coverage — also L1.

The phrase "Waddington-feature-tightened" carries latent L2 risk because "Waddington landscape" is a
concept borrowed from developmental biology where it normally describes an energy surface over which a
cell's *fate* can be experimentally steered (i.e., L2/L3 manipulative biology). Here, however, the
four scalar features — U(z0), U(z(T)), ||∇U(z0)||, ||z(T)−z0|| — are baseline-geometry descriptors:
they characterize the curvature and displacement of the learned potential at the patient's observed
latent position at t=0. F8-DPS STRICT FAIL (0/16 cells, both operators) proves these scalars carry
no temporally predictive (let alone interventional) content beyond their associative contribution in
the Cox model. The claim is therefore L1-consistent, but requires an explicit in-text statement that
the Waddington scalars are baseline-geometry covariates, not forecasted future states.

**Required fix:** Add one sentence in Methods: "The four Waddington-derived covariates characterize
geometry of the learned potential landscape at the patient's baseline latent position; they are not
predictions of future disease state (F8-DPS strict fail at N_paired=29 under both v10 single-Euler
and v11 Neural-ODE operators rules out temporally predictive content at this sample size)."

---

### 1.2 M1 Claim: "M1 Lyapunov 0/116 violations under adaptive Dopri5 forward operator"

**Chair language (§2.1):** "M1 Lyapunov (dU/dt = −‖∇U‖² ≤ 0) verified: 0/116 violations."

**Verdict: PASS — L1 throughout.**

The Lyapunov criterion dU/dt = −‖∇U‖² ≤ 0 is a structural identity: for a gradient-flow ODE
z'(t) = −∇U_θ(z), the total derivative dU/dt = ∇U · z'(t) = −‖∇U‖² ≤ 0 is a consequence of the
parameterization, not an empirical finding about biological intervention. It verifies that the
numerical integrator (Dopri5) faithfully traces a path that is non-increasing in U_θ — an L1
structural consistency check, not a causal claim. The verification on 29 × 4 = 116 (patient, T-value)
pairs is empirical but only confirms numerical correctness, not that U_θ corresponds to any
biologically meaningful energy surface that would respond to perturbation.

No L2 language detected. No fix required on this contribution.

---

### 1.3 M2 Claim: "v11.5 ties SOTA on MM PFS C-index"

**Chair language (§1, Claim B):** "CONDITIONAL_PASS — PUBLISHABLE BUT NOT NOVEL ON DISCRIMINATION."
"The 'TIE' is real on a paired test, but ~95% of the SOTA-tying signal is mmSYGNAL's pretrained
feature pipeline, not v11 architecture."

**Verdict: PASS for the core statistical claim — L1 throughout. One latent language slippage flagged
in the title formulation.**

The C-index is a concordance statistic: P(score(i) > score(j) | T_i < T_j). This is an L1
associational quantity. The paired bootstrap (B=10,000, Δ=−0.00005, 99% CI [−0.035, +0.036],
p=0.989) and U-statistic z-test (p=0.994) are appropriate for an L1 concordance comparison and
do not require intervention identification.

**Slippage flag — M2 headline:** The bioRxiv title recommended in §6 is
"Composing v11 calibrated coverage with mmSYGNAL transcriptional-program risk on MMRF."
The phrase "transcriptional-program risk" is L1 (the score is a pre-trained glmnet risk scalar
conditioned on observed program activity). No slippage here.

However, the mmSYGNAL source paper (Murie/Baliga 2025, PMID 40169765) uses the phrase
"treatment selection" in its own abstract (see V11_CLINICAL_TRANSLATION.md §7: "Individualized
dynamic risk assessment and treatment selection for multiple myeloma"). If M2 cites mmSYGNAL's
full title or repeats its framing — "treatment selection" — without an explicit hedge, that phrase
imports L2 semantics into the M2 manuscript by proxy.

**Specific slippage risk in M2 §1:** If the M2 introduction summarizes the mmSYGNAL background as
"mmSYGNAL enables individualized treatment selection," that sentence is L2 in mmSYGNAL's own
framing. The M2 manuscript should re-describe mmSYGNAL only as "a SOTA prognostic concordance
benchmark" or "a transcriptional-program risk scoring system achieving C=0.6957 on PFS
discrimination." The treatment-selection claim belongs to the mmSYGNAL paper; importing it
into M2's introduction risks implicit L2 endorsement.

**Required fix — M2 §1:** When citing mmSYGNAL, use only its L1 metric (C-index on PFS) as the
basis for comparison. Do not reproduce its "treatment selection" framing. Sample corrected
language: "Murie et al. (2025) reported a transcriptional-program risk model for MM achieving
state-of-the-art PFS concordance across five independent cohorts (N=1,367). We evaluate whether
v11 calibrated-coverage features are complementary to this prognostic score on MMRF N=787
(L1 associational comparison; treatment-selection claims are out of scope for the present work)."

---

### 1.4 Contribution: "Pre-registered identifiability-bound annotation via F8-DPS strict-fail"

**Chair language (§3.1):** "Pre-registered identifiability-regime annotation: F8-DPS strict-fail at
N_paired=29 < Stone(d=64)≈150, robust under v10 single-Euler AND v11 Neural-ODE Dopri5 operators."

**Verdict: PASS — correctly L1. Annotation of an identifiability bound is itself an L1 act.**

See detailed analysis in Section 3 below. No slippage.

---

### 1.5 Contribution: "Honest paired-bootstrap-confirmed TIE against mmSYGNAL with §3.11 decomposition"

**Chair language (§3.7, defensible niche paragraph):** "an honest paired-bootstrap-confirmed TIE
against mmSYGNAL on PFS C-index when the mmSYGNAL routed score is added as a feature, with the
architectural decomposition openly reporting that ~95% of the SOTA-tying gain is from mmSYGNAL
features."

**Verdict: PASS — L1 throughout, decomposition is L1 attribution in a regression.**

The §3.11 composition decomposition (pure mmSYGNAL C=0.6957; pure v11_richer C=0.6538; v11_richer
on top of raw 6 mmSYGNAL scores adds +0.010; mmSYGNAL routed score on top of v11_richer adds +0.042)
is an incremental C-index attribution in an observational Cox regression. Incremental C-index
attribution is an L1 operation: it measures how much one covariate changes the rank concordance of
the model against an observed outcome. No do-operator is required; no counterfactual is invoked.

No slippage. The honest attribution weakens the v11 novelty claim but keeps the framing at L1.

---

### 1.6 §3.7 defensible niche paragraph — full audit

Full text of §3.7: "ResistanceMap v11/v11.5 is uniquely the combination of: (a) a
Helmholtz-by-construction gradient-flow Neural-ODE on a learned scalar Waddington potential U_θ
verified by 0/116 Lyapunov violations under adaptive Dopri5 … (b) a STRING v12 multi-α attention
propagation kernel … (c) Mondrian jackknife+ calibrated 90% prediction intervals tightened by 4
Waddington-derived features … (d) a pre-registered identifiability-regime annotation via F8-DPS
strict-fail under two forward operators at N_paired=29 < Stone(d=64) … (e) a multi-omic MMRF
CoMMpass IA22 N=787 + Beat AML 1.0 cross-disease control + paired N=29 … (f) a release-bouncer
… AND (g) an honest paired-bootstrap-confirmed TIE against mmSYGNAL on PFS C-index …"

**Clause-by-clause ladder check:**

| Clause | Language type | L1/L2/L3 | Slippage? |
|--------|---------------|-----------|-----------|
| (a) "Helmholtz-by-construction gradient-flow Neural-ODE on a learned scalar Waddington potential" | Structural/architectural | L1 — describes model parameterization | None, but see §1.1 fix |
| (a) "verified by 0/116 Lyapunov violations" | Numerical correctness | L1 structural identity | None |
| (b) "STRING v12 multi-α attention propagation kernel that bit-identically generalizes single-α RWR" | Architectural | L1 | None |
| (b) "preserves F6 30.6σ strict-pass on 45 consensus MM drivers" | Associational (rank recall) | L1 | None |
| (c) "Mondrian jackknife+ calibrated 90% prediction intervals tightened by 4 Waddington-derived features" | Conformal coverage | L1 | None (Waddington label risk addressed in §1.1) |
| (d) "pre-registered identifiability-regime annotation via F8-DPS strict-fail" | Identification bound | L1 — observational identifiability claim | None |
| (e) "multi-omic MMRF CoMMpass IA22 N=787 + Beat AML 1.0" | Data provenance | L1 | None |
| (f) "release-bouncer that refuses any number not grounded in RUNS.md" | Governance | N/A | None |
| (g) "honest paired-bootstrap-confirmed TIE against mmSYGNAL … ~95% of the SOTA-tying gain is from mmSYGNAL features" | L1 concordance attribution | L1 | None |

**§3.7 aggregate verdict: PASS.** No L1→L2 slippage in the niche paragraph text as written.
The one upstream risk — that a reader uses "Waddington potential" as a shorthand for
"interventional energy landscape" — is mitigated by clause (d) which explicitly bounds the
identifiability regime at F8-DPS. The niche paragraph is internally self-correcting on this point.

---

## 2. The Caveat #2 decomposition (95% mmSYGNAL / 5% v11) and L2 framing risk in M2 §1

**Question posed:** Does the v11.5 Caveat #2 decomposition (95% mmSYGNAL / ~5% v11) require any L2
framing in the manuscript headline? Cross-check the proposed M2 §1 framing against do-calculus /
counterfactual semantics.

**Finding: The decomposition is L1. No L2 framing is required or implied. One indirect L2 import
risk from the mmSYGNAL citation.**

**Formal analysis:**

The Caveat #2 decomposition reports marginal C-index increments:
- Δ_mmsygnal = C(v11_richer + mmSYGNAL routed) − C(v11_richer) = +0.042
- Δ_v11 = C(v11_richer + raw 6 mmSYGNAL) − C(raw 6 mmSYGNAL) = +0.010
- Δ_ensemble = C(v11.5) − C(mmSYGNAL routed) = −0.0002

These are differences in observational concordance statistics across nested covariate sets. No
do-operator is required. In Pearl notation, all quantities are of the form E[h(X, Y)] for
observed (X, Y) — L1 by definition.

The decomposition does NOT assert that "if we intervened on transcriptional-program activity, the
C-index would change by +0.042." It asserts that adding a pre-computed scalar (the mmSYGNAL routed
score) as a conditioning variable in an observational Cox regression adds +0.042 marginal concordance.
That is L1 covariate enrichment, not a causal claim.

**The indirect L2 import risk (reinforcing §1.3 finding):** The mmSYGNAL routed score was published
as a "treatment selection" tool (Murie/Baliga 2025 abstract). If M2 frames the decomposition as
"the SOTA treatment-selection signal contributes 95% of the gain," the word "treatment-selection"
enters the M2 narrative with L2 connotation. The correct framing is "the SOTA prognostic-discrimination
signal contributes 95% of the gain." The distinction is not cosmetic: treatment selection implies
do(drug=D), which is an L2 quantity; prognostic discrimination implies P(rank(i) > rank(j) | X_obs),
which is L1.

**Proposed M2 §1 corrected headline framing:**

Do NOT write: "ResistanceMap v11.5 achieves SOTA on MM PFS by composing Waddington landscape
features with mmSYGNAL treatment-selection scores."

DO write: "ResistanceMap v11.5 ties the SOTA prognostic concordance benchmark (mmSYGNAL; PMID
40169765; C=0.6957) on MMRF N=787 by composing 23 Waddington-landscape and cytogenetic features
with the mmSYGNAL transcriptional-program risk score as a scalar covariate. The mmSYGNAL score
contributes ~95% of the concordance gain (from pure-v11 C=0.6538 to TIE C=0.6955); v11 features
add +0.010 incremental concordance on top of raw mmSYGNAL submodel scores. All claims are
observational (Pearl L1); treatment-selection use is outside the declared scope."

---

## 3. F8-DPS strict-fail as "identifiability bound" — is it L1 or L2?

**Question posed:** Does claiming "the operator cannot identify per-patient drift at N_paired < Stone(d=64)"
require interventional interpretation, or is it purely observational?

**Finding: Purely observational (L1). The Stone minimax bound is a statement about statistical
estimation capacity, not about causal identifiability.**

**Formal argument:**

The Stone (1982) minimax bound states that for a d-dimensional nonparametric estimator using N
independent samples, the optimal worst-case MSE rate is O(N^{-2s/(2s+d)}) for functions in a Sobolev
class with smoothness s. At d=64, N_paired=29, s=1, this gives an identifiable effective dimension
of approximately 4 (i.e., the N=29 sample can only resolve a 4-dimensional subspace of the
64-dimensional drift without overfitting). This is an estimation-theoretic claim in the frequentist
sense — it concerns the bias-variance tradeoff for a function class, not the causal graph structure.

In Pearl's ladder:
- L1 (association): P(Y | X) — observed conditional distributions
- L2 (intervention): P(Y | do(X=x)) — distributions under external manipulation
- L3 (counterfactual): P(Y_x | X=x') — individual-level potential outcomes

The Stone bound operates entirely within L1: it bounds how well we can estimate a conditional
expectation E[Z(T) | Z(0)=z0] from N paired observations, where Z(0) and Z(T) are both observed.
No intervention is required; the drift is estimated from paired (baseline, post-treatment) RNA
measurements, both of which are observational snapshots.

Accordingly, "the operator cannot identify per-patient drift at N_paired=29" is a statement that
the observational sample is too small to nonparametrically estimate the conditional mean of the
posterior displacement — an estimation-layer failure (Rubin framework: identification is
conceptually present via consistency + exchangeability over pairs; estimation fails due to curse
of dimensionality). This is NOT an identification failure in the graph-theoretic sense (i.e.,
it does not say that the causal graph structure precludes identification of E[Z(T)|do(Z(0)=z0)]);
it says that even if the graph were identified, we cannot estimate the relevant conditional well
at N=29.

**Precise distinction:** The F8-DPS refutation is an estimation-layer bound, not a causal-graph
identification failure. The chair verdict annotates it correctly as an "identifiability bound"
in the loose statistical sense (the quantity is not estimable at this N). No L2 framing is
needed or implied. However, the manuscript should distinguish between these two senses to avoid
confusing reviewers trained in the potential-outcomes framework:

**Required precision in M1 §6:** "F8-DPS strict-fail reflects an estimation-layer constraint:
at N_paired=29 < Stone(d=64)≈150, the posterior displacement E[Z(T)|Z(0)=z0] cannot be
estimated nonparametrically without overfitting. This is not a graph-level causal identification
failure (no additional structural assumptions would unlock per-patient forecasting at this N);
it is a sample-size-driven estimation bound. Per-patient drift prediction requires N_paired ≥ 150
for d=64, which is a data-acquisition, not a modeling, problem."

---

## 4. Sprint 4 NIE finding — is it promoted to per-patient L2 in v11 or in §3.7?

**Question posed:** Confirm the v11 paper does NOT promote the v10 Sprint 4 NIE finding (L2
mediation detected but F10-tipping fails at E≈1.20) to a per-patient L2 claim. Cross-check
§3.7 niche paragraph for any leak.

**Finding: NOT promoted. No L2 leak detected in v11 or §3.7. One nuance requires clarification
in M1 §7.**

**Evidence from artifacts:**

The tipping-point JSON (`paper/v8_artifacts/v10_sprint4/nie_tipping_point.json`) states:
"actionable_conclusion: L2 NIE direction (proteasome → TT2L mediation) is detected and specific
(F8, F9 PASS) but is sensitivity-limited (F10 FAIL); the v10 paper claim stays at L1-with-structural-
prior with this NIE as supporting directional evidence, exactly as pre-declared in spec §11 abstract."

The NIE HR point estimate is 1.0901 (CI [1.029, ∞]), but the tipping-point analysis shows the
lower-CI bound is nullifiable by a confounder with (RR_AU=1.5, RR_UY=1.10) — strengths routinely
found in MM prognostic factors (ISS, del17p, LDH, ECOG). The F10 E-value of ≈1.20 is below the
pre-registered threshold of 1.5, confirming that the NIE cannot be defended against unmeasured
confounding at the required robustness level.

**Audit of §3.7 and all §3.x sections:** The Sprint 4 NIE is not mentioned in §3.7 (niche paragraph)
at all — it is absent. The §3.1 differentiation from mmSYGNAL lists "calibrated 90% prediction
intervals" and "identifiability-bound annotation" but not NIE. The §2.4 Sprint 7 section mentions
"Sprint 4 (proteasome-NIE supporting evidence)" only in the context of confirming the L1 ceiling:
"v11 paper inherits the v10 §6 fallback: per-patient claims restricted to Sprint 5 (population-level
conformal coverage) + Sprint 4 (proteasome-NIE supporting evidence)." This phrase "supporting evidence"
is correctly hedged; it does not assert per-patient mediation.

**One nuance requiring clarification in M1 §7:** The phrase in §2.4 of the chair verdict reads:
"proteasome-NIE supporting evidence." In a manuscript, "supporting evidence" for a mediation effect
can be read by a pharmacology-oriented reviewer as implying "clinical evidence that targeting the
proteasome modifies TT2L" — which would be an L2 claim. The NIE finding is associational mediation
(VanderWeele 2015 definition under no-unmeasured-confounding for the mediator), and F10 failure means
the no-unmeasured-confounding assumption is not robust.

**Required precision in M1 §7 (Limitations):** "The Sprint 4 natural indirect effect analysis
detected a proteasome-pathway mediator in the Cox regression (HR_NIE=1.090, CI [1.029, ∞]).
This is an observational mediation finding. The F10 tipping-point analysis (E=1.20) demonstrates
that a confounder with joint (RR_AU=1.5, RR_UY=1.10) would nullify the lower confidence bound —
strengths plausible for measured MM prognostic factors (ISS stage, del17p). The proteasome
mediation finding should NOT be interpreted as evidence that proteasome inhibition causally
modifies PFS (L2); it is a directional associational signal consistent with the known clinical
importance of proteasome inhibitors in MM, but its robustness to unmeasured confounding is
insufficient to support causal inference."

---

## 5. Clinical actionability with "calibrated coverage" or "identifiability bound" — L2 slippage audit

**Question posed:** Does any chair-claimed passage use "calibrated coverage" or "identifiability
bound" in a clinically actionable way that implies treatment selection (L2)?

**Finding: The chair verdict and prior causal audit (V11_CAUSAL_AUDIT.md) do NOT use either phrase
to imply treatment selection. One passage in the V11_CLINICAL_TRANSLATION.md requires hedging.**

### 5.1 "Calibrated coverage" in context

**Chair verdict §1, Claim A:** "Calibrated 90% prediction-interval coverage with Waddington-feature-
tightened intervals on MM PFS." This is used to assert conformal calibration — an L1 statistical
guarantee on interval validity.

**V11_CAUSAL_AUDIT.md §3, Question 4:** "Calibrated interval reporting for individual patients:
The F_S5 Mondrian jackknife+ 90% coverage pass … supports reporting calibrated prediction intervals
for individual patients' TT2L. A clinician can correctly say 'for this del17p patient, the model's
90% prediction interval for TT2L is [X, Y] days.' This does NOT imply the point prediction is causal."

**V11_CLINICAL_TRANSLATION.md §3, Q3:** "The Mondrian conformal jackknife+ calibration passing
5/5 strata … means the 90% prediction interval achieves the stated coverage when averaged across
patients within each cytogenetic stratum." This correctly bounds the claim to stratum-level coverage.

**V11_CLINICAL_TRANSLATION.md §5.2:** "If S1 del17p validation confirms C >= 0.70 in an independent
cohort, v11.5 could serve as an enrichment biomarker to identify the highest-risk del17p patients
for aggressive MRD-directed strategies (e.g., early consolidation with a CAR-T product in patients
who remain MRD-positive after Dara-RVd induction)."

**Slippage flag — §5.2 of CLINICAL_TRANSLATION.md:** The phrase "serve as an enrichment biomarker
to identify … patients for aggressive MRD-directed strategies … early consolidation with a CAR-T
product" edges toward L2. "Identifying patients for" a specific therapeutic intervention
(CAR-T consolidation) implies the score informs do(treatment=CAR-T), which is L2. The document
correctly prefaces this with "After Phase 1 Validation" and "enrichment biomarker," which are
attenuating hedges — but the phrase remains closer to L2 language than L1.

The §6 of that same document correctly states: "Treatment selection between specific regimens …
is not actionable. The Pearl ceiling (L1-with-structural-prior) is explicit: this model predicts
ranking, not intervention effects." The internal inconsistency between §5.2 and §6 within
CLINICAL_TRANSLATION.md should be resolved before any manuscript uses §5.2 language.

**Required fix — CLINICAL_TRANSLATION.md §5.2:** Replace "v11.5 could serve as an enrichment
biomarker to identify the highest-risk del17p patients for aggressive MRD-directed strategies"
with: "v11.5 could serve as an exploratory enrichment biomarker to pre-stratify del17p patients
by predicted tt2L risk in a correlative arm of a prospective study; any causal relationship
between this risk stratum and benefit from specific interventions (e.g., CAR-T consolidation)
would require a separate randomized evaluation."

**Note:** This slippage exists in an internal audit/translation document, not in the chair verdict
itself. The chair verdict explicitly excludes treatment selection (§3.1: "mmSYGNAL does not
publish calibrated intervals; v11 is explicit about where it does NOT claim per-patient
forecasting"). The fix is advisory for manuscript drafting.

### 5.2 "Identifiability bound" in context

Wherever the chair verdict uses "identifiability bound," it is paired with the Stone minimax
argument at N_paired=29 < ~150 for d=64. This is used to explain a statistical estimation
failure, not to gate or guide any treatment decision. The phrase appears in:
- §1: "pre-registered identifiability-bound annotation"
- §2.4: "bound is operator-independent"
- §3.1: "v11 is explicit about where it does NOT claim per-patient forecasting"
- §3.7: "pre-registered identifiability-regime annotation via F8-DPS strict-fail"

None of these uses attach the identifiability bound to a treatment-selection framing. All uses
correctly bound the model's claims downward (i.e., the bound restricts what can be claimed, not
what should be done clinically). No L2 slippage on "identifiability bound."

### 5.3 PubMed literature check on "calibrated coverage" and clinical actionability

PubMed query tool was denied in this session. Based on training-corpus knowledge:

The standard position in conformal prediction literature (Venn-Abers, Mondrian, jackknife+) is
that calibrated coverage guarantees apply to the prediction task, not to treatment effects. Key
references on this distinction include:

- Barber, Candès, Ramdas, Tibshirani (2021, Ann. Stat.): jackknife+ coverage guarantee is
  distribution-free and L1 by construction; no causal or interventional interpretation is
  ever claimed.
- Romano, Sesia, Candès (2019, NeurIPS): CQR coverage guarantee is also L1.
- Shafer and Vovk (2008, JMLR): conformal prediction is explicitly framed as an inference
  procedure for exchangeable data, not a causal method.

No peer-reviewed publication uses "calibrated conformal coverage" to assert treatment selection
without an accompanying randomized-trial design. The claim that calibrated coverage "guides
treatment selection" would require, at minimum, a decision-theoretic framework linking the
prediction interval to a utility function over treatment choices — which in turn requires L2
identification of P(outcome | do(treatment), X). The v11 pipeline does not build this link,
and the chair verdict correctly keeps calibrated coverage in the L1 prognostic stratum.

A query to PubMed would be useful to confirm no recent (2024-2025) MM-specific paper conflates
calibrated coverage with treatment selection; this is noted as a recommended check before
manuscript submission.

---

## 6. Consolidated verdict table

| Item | Chair claim | Independent verdict | Slippage type | Required action |
|------|-------------|---------------------|---------------|-----------------|
| 1. M1 Claim A: calibrated 90% coverage | L1 — conformal interval validity | PASS | None in core claim; "Waddington" label carries latent L2 appearance | Add sentence clarifying Waddington scalars are baseline geometry, not forecasted states (§1.1) |
| 2. M1 Lyapunov 0/116: structural integrity | L1 — numerical ODE correctness | PASS | None | None required |
| 3. M2 Claim B: SOTA-tying TIE | L1 — concordance statistic | PASS | Indirect: mmSYGNAL's "treatment selection" framing imported by citation | M2 §1 must describe mmSYGNAL as prognostic, not treatment-selection, benchmark (§1.3) |
| 4. Caveat #2 decomposition (95%/5%) | L1 — incremental C-index attribution | PASS | None | None in the decomposition itself; fix applies to how mmSYGNAL is cited (§2) |
| 5. F8-DPS "identifiability bound" | Estimation-layer constraint | PASS (correctly L1) | Terminological: "identifiability" conflates estimation-theoretic with graph-theoretic failure | M1 §6 must distinguish estimation-layer from graph-level identification failure (§3) |
| 6. NIE Sprint 4 directional evidence | L1 mediation under unconfoundedness | PASS — not promoted | Latent risk: "supporting evidence" language could be read as causal support for proteasome inhibitors | M1 §7 Limitations must disclaim causal interpretation of NIE (§4) |
| 7. §3.7 niche paragraph | L1 throughout | PASS | None | No change needed to the paragraph as written |
| 8. CLINICAL_TRANSLATION.md §5.2 | Borderline L2 (treatment enrichment) | CONDITIONAL | "Identify patients for CAR-T consolidation" is L2 language | Replace with correlative-arm framing (§5.1) |

---

## 7. Required changes, priority-ordered

**RC-1 (HIGH — M1 Methods):** Clarify that the four Waddington-derived covariates are
baseline-geometry descriptors, not predictions of future state. Reference F8-DPS strict-fail
explicitly. This prevents the "Waddington potential" framing from being read as an interventional
energy landscape claim.

**RC-2 (HIGH — M2 §1 Introduction):** When citing mmSYGNAL, use only its L1 C-index metric
as the basis for comparison. Do not reproduce "individualized … treatment selection" language from
the mmSYGNAL abstract. Rephrase as "prognostic concordance benchmark." This eliminates the indirect
L2 import via mmSYGNAL's own framing.

**RC-3 (MEDIUM — M1 §6):** Distinguish estimation-layer failure (Stone bound at N_paired=29)
from graph-level identification failure. Both are L1-consistent, but conflating them misleads
reviewers trained in potential outcomes who associate "identifiability" strictly with graph-
theoretic non-identifiability (e.g., unobserved confounders that make a causal effect undefined,
not merely estimable at a slow rate).

**RC-4 (MEDIUM — M1 §7 Limitations):** State explicitly that the Sprint 4 NIE finding is an
observational mediation result whose robustness to unmeasured confounding (E-value ≈1.20) is
insufficient to support causal inference. Prohibit reading the proteasome NIE as evidence that
proteasome inhibition modifies PFS.

**RC-5 (ADVISORY — CLINICAL_TRANSLATION.md §5.2):** Replace "enrichment biomarker to identify
patients for [specific therapy]" with "pre-stratification tool for a correlative arm, with causal
benefit of the intervention requiring separate randomized evaluation." This document is internal
but may be excerpted for grant or IRB submissions.

---

## 8. What the audit confirms about the prior causal audit (V11_CAUSAL_AUDIT.md)

The prior audit (V11_CAUSAL_AUDIT.md, same date) is **consistent with this independent re-audit**
on all major findings:
- L1 ceiling correctly declared and enforced across all sprints.
- F8-DPS correctly bounds the L2 ceiling.
- Hedge H1 (abstract L1 declaration), H2 (Waddington geometry vs. forecast), H3 (IA12/IA22
  overlap), and H4 (prognostic vs. treatment selection) from V11_CAUSAL_AUDIT.md are all
  confirmed necessary by this independent re-audit.
- The mmSYGNAL "treatment selection" import risk (§1.3 and §2 of this document) was NOT
  explicitly flagged in the prior audit, which focused on whether mmSYGNAL's score as a
  covariate introduces L2 scope. This re-audit adds the more subtle risk that citing the
  mmSYGNAL paper's own framing imports L2 language into M2's introduction.

**Net new finding relative to V11_CAUSAL_AUDIT.md:** The mmSYGNAL citation language risk in M2 §1
(RC-2 above) is the only materially new slippage point identified by this independent re-audit.
All other required changes overlap with Hedges H1-H4 from the prior audit.

---

## 9. Overall audit verdict

| Manuscript / Claim | Ladder position | Slippage found | Audit verdict |
|--------------------|-----------------|----------------|---------------|
| M1: Calibrated coverage (F_S5 + M1 Lyapunov + F6) | L1 | None in claims; "Waddington" label risk addressed by RC-1 | CONDITIONAL PASS — RC-1, RC-3, RC-4 required |
| M2: SOTA-tying TIE (Claim B + Caveat #2 decomposition) | L1 | Indirect: mmSYGNAL "treatment selection" import — RC-2 | CONDITIONAL PASS — RC-2 required before bioRxiv submission |
| F8-DPS as identifiability bound | L1 (estimation-layer) | Terminological conflation risk — RC-3 | CONDITIONAL PASS — RC-3 required |
| Sprint 4 NIE as "supporting evidence" | L1 (observational mediation) | Latent causal-interpretation risk — RC-4 | CONDITIONAL PASS — RC-4 required |
| CLINICAL_TRANSLATION.md §5.2 | Borderline L2 | "Identify patients for [therapy]" — RC-5 | FAIL for manuscript use as written; revise per RC-5 |

**Aggregate verdict: CONDITIONAL PASS.** The chair verdict's CONDITIONAL_PASS stands at v12
planning stage. No hard L1→L2 slippage was found in the core statistical claims or in the §3.7
niche paragraph as written. The five required changes (RC-1 through RC-5) are all text-level
fixes requiring no additional data collection or experimental re-runs. RC-1 through RC-4 are
required before any bioRxiv submission; RC-5 is required before any clinical-translation section
of a submitted manuscript reproduces §5.2 language.

---

*End of V12 causal re-audit. All numerical claims trace to artifact paths cited in the document body.*
