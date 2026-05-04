# V11.5 Clinical Translation — Cox-PH Ensemble PFS Discrimination in Multiple Myeloma

**Date:** 2026-05-03
**Author:** Hematologic Oncology / Translational Medicine Reviewer (MD/PhD, board-certified hematology, fellowship-trained SCT, PhD Translational Medicine)
**Version audited:** v11.5 Cox-PH ensemble (Waddington features + mmSYGNAL routed score)
**Endpoint:** tt2L_days (time to second-line therapy initiation, surrogate PFS)
**Cohort:** MMRF CoMMpass N = 787
**Pearl ceiling declared:** L1-with-structural-prior — predicts ranking, not intervention effects

---

## 0. Scope and epistemic posture

This document translates the v11.5 C-index result into clinically actionable form.
It does NOT recommend any clinical action for any individual patient.
It is an audit of what the stated result permits — and prohibits — a treating hematologist
to do, under IMWG response criteria and current MM standard-of-care frameworks.

All mmSYGNAL references derive from: Murie/Turkarslan/Patel/Coffey/Becker/Baliga,
"Individualized dynamic risk assessment and treatment selection for multiple myeloma,"
British Journal of Cancer 132(10):922-936 (April 2025), PMID 40169765,
DOI 10.1038/s41416-025-02987-6. This PMID was verified in-repo 2026-05-03
(sota_comparison.md §0.2). The ClinicalTrials.gov tool was unavailable at audit time
(permission denied); NCT IDs cited below are from CLINICAL_ACTIONABILITY_RUBRIC.md
(authored from clinical practice knowledge, 2026-05-03) and are marked [rubric-cite].
A live CT.gov query must be used to confirm current status before any manuscript
submission.

---

## 1. IMWG Response Criteria Alignment

### 1.1 The endpoint in use: tt2L_days

tt2L_days (time to second-line therapy initiation) is a **surrogate PFS endpoint**
used in registry datasets. It is not IMWG-defined PFS. IMWG PFS is event-driven by:
- M-protein or sFLC rise >= 25% from nadir
- New lytic lesions or soft-tissue plasmacytoma
- Hypercalcemia, renal deterioration, anemia, or bone lesions (CRAB criteria)
- Death

tt2L_days conflates IMWG-defined PD with physician decision to escalate therapy
(which may precede or lag IMWG PD depending on clinical context and patient preference).

**Clinical-translation consequence:** The C-index of 0.6955 on tt2L_days is not
directly equivalent to a C-index on IMWG-defined PFS. Trials stratified by PFS
(e.g., CARTITUDE-4, KarMMa-3) use IMWG PFS as primary endpoint, not tt2L.
Any claim that v11.5 "predicts PFS" must be qualified as "predicts time to
second-line initiation (tt2L), a surrogate for PFS not identical to IMWG-defined
event-driven PFS."

### 1.2 C-index of 0.696 vs. IMWG response strata

IMWG response strata (sCR/CR/VGPR/PR/MR/SD/PD) are not model inputs or outputs.
The v11.5 model takes omics/genomic features and outputs a risk score. The risk
score's discriminative ability (C = 0.696) is a population-level rank statistic —
it tells us that for a randomly selected pair of patients, the model assigns a
higher risk score to the one who initiated second-line therapy earlier with
probability 0.696. This is a meaningful separation. It does NOT mean:
- That any specific patient achieved CR vs. VGPR
- That a change in M-protein kinetics was captured
- That MRD negativity was predicted or used

The model is IMWG-agnostic at inference time. The endpoint (tt2L) only weakly
correlates with IMWG response depth because patients achieving sCR/CR may still
receive second-line therapy for reasons unrelated to tumor progression (toxicity
switch, protocol escalation, ASCT-conditioning substitution).

**Implication for clinical translation:** C = 0.696 cannot be directly mapped to
"would have predicted which patient achieved VGPR vs. PR." To make that claim,
the model would need IMWG response labels as the training outcome, not tt2L.

---

## 2. Standard-of-Care MM Framework Alignment

### 2.1 Current MM induction and maintenance landscape

| Setting | Preferred regimens (2026) | Risk stratification standard |
|---|---|---|
| Transplant-eligible NDMM | Dara-RVd (NCT03710603, PERSEUS) or RVd | R-ISS / Mayo SA-ISS; FISH for del17p, t(4;14), t(14;16), +1q21 |
| Transplant-ineligible NDMM | Dara-VMP or Dara-Rd | R-ISS; frailty (IMWG frailty score) |
| RRMM after 1–3 lines | Cilta-cel (CARTITUDE-4), Dara-Kd, IsaPd | Prior-line exposure (PI + IMiD + anti-CD38) |
| RRMM heavily pre-treated | Ide-cel (KarMMa-3), Tec + Dara (MajesTEC-3), Bela-Pd | BCMA surface density; extramedullary disease status |
| Maintenance post-ASCT | Lenalidomide +/- ixazomib; MRD-directed trials | MRD negativity by NGS/flow (10^-5 to 10^-6) |

v11.5's per-stratum C-index maps as follows onto this framework:

| Cytogenetic stratum | Per-stratum C | Clinical mapping |
|---|---|---|
| S1 del17p | 0.730 | High-risk NDMM; TP53 biallelic loss if del17p + TP53 mutation; standard = Dara-RVd or Dara-VRd + early ASCT; MRD-directed strategies under trial |
| S2 t(4;14) | 0.667 | High-risk; FGFR3/NSD2 axis; responds to bortezomib in some but not all; ASCT recommended |
| S3 +1q21 | 0.694 | High-risk (especially +1q + standard risk); MCL1/CKS1B amplification; venetoclax not active |
| S4 t(11;14) | 0.644 | Standard-to-favorable risk (BCL2-high, venetoclax-responsive); CANOVA trial population |
| S5 other | 0.659 | Heterogeneous low-risk group |

### 2.2 Where does C = 0.696 sit in the discrimination quality hierarchy?

A C-index of 0.696 marginally exceeds the ISS-alone benchmark. ISS C-index for
OS in MM is approximately 0.62–0.67 in modern series (Palumbo et al. Lancet Oncol 2015).
R-ISS (ISS + LDH + FISH) achieves approximately 0.68–0.72 in published NDMM cohorts.

Therefore v11.5 sits **at or slightly above R-ISS-level discrimination** for the
tt2L endpoint on MMRF N = 787. This is clinically meaningful because R-ISS is the
current standard for trial stratification. The paired-test TIE with mmSYGNAL
(C = 0.6957 vs. 0.6955) means v11.5 does not outperform mmSYGNAL on this cohort.

**The ISS-enrollment-lift question:** If R-ISS alone achieves C approximately 0.68–0.70
on a well-conducted NDMM cohort, v11.5's marginal uplift over ISS alone is small —
probably 0.02–0.05 C-index units. The enrolled patient composition in ISS-stratified
trials (e.g., requiring ISS II–III for high-risk arms) would not change materially
if v11.5 scores replaced ISS, because most patients fall in the same risk tiers.
A more meaningful claim would require demonstrating that v11.5 re-stratifies a
meaningful fraction of ISS I patients into the high-risk arm (i.e., true ISS-discordant
high-risk cases). This analysis is not reported and cannot be inferred from the
headline C-index alone.

---

## 3. Question-by-Question Answers

### Q1. Does C = 0.696 PFS discrimination support a triage decision (e.g., clinical trial referral vs. standard-of-care)?

**Partial, with three hard conditions:**

A C-index of 0.696 is clinically meaningful as a continuous risk-ranking tool.
By itself, it is insufficient to drive a binary triage decision (SOC vs. trial
referral) for the following reasons:

**Condition 1 — Decision threshold required.** A C-index is a rank-order statistic;
it does not specify the score cut-point at which "trial referral" is triggered.
Without a pre-registered decision threshold with sensitivity/specificity/PPV/NPV
reported against the tt2L endpoint, the score cannot be operationalized as a
triage rule.

**Condition 2 — External validation required.** MMRF CoMMpass is a single registry
cohort. A triage decision that will affect the enrolling characteristics of a
clinical trial requires external validation in at least one prospectively collected,
independent cohort before IRB submission as a stratification covariate.

**Condition 3 — IMWG-response-concordant end state required.** Current trial
eligibility criteria for high-risk NDMM interventions (e.g., Dara-RVd plus
early ASCT, MRD-directed maintenance) are defined by cytogenetics + R-ISS, not
by ML risk scores. For v11.5 to serve as a triage tool, a regulatory or trial-
design precedent for "ML biomarker-driven stratification" in MM must be established.
No such precedent exists in FDA-approved MM trials as of 2026-05-03.

**What a treating hematologist could do with C = 0.696 today:** Use the risk score
as supplementary (not primary) risk context when discussing trial enrollment
eligibility with a patient in whom cytogenetics are ambiguous or discordant from
clinical presentation. This is consistent with a Tier 2 "Note in chart" usage
per CLINICAL_ACTIONABILITY_RUBRIC.md.

### Q2. What patient subgroup benefits most from v11.5?

**S1 del17p (C = 0.730) is the clinically highest-signal stratum.**

This maps to a real clinical decision point for the following reasons:

1. del17p MM patients are universally classified as ultra-high-risk. Current
   standard per IMWG 2022 consensus is to recommend early ASCT (even in
   transplant-ineligible candidates when feasible), Dara-RVd induction, and
   MRD-directed maintenance. However, del17p patients are biologically
   heterogeneous — biallelic TP53 loss (del17p + TP53 mutation) carries a far
   worse prognosis than monoallelic del17p. A C = 0.730 within del17p suggests
   v11.5 further sub-stratifies this group, which R-ISS cannot do.

2. **Real clinical decision it could inform:** Within del17p patients starting
   Dara-RVd, a high v11.5 risk score could identify the subgroup most likely to
   benefit from early transition to BCMA-targeted consolidation (e.g., ide-cel
   or cilta-cel in the consolidation setting, as evaluated in NCT05181826 and
   related trials [rubric-cite]). This is a trial-referral decision, not a SOC
   change.

3. **Caveat:** The del17p stratum N within N = 787 is likely 80–140 patients
   (del17p prevalence approximately 10–15% at diagnosis). A C-index from this
   n is statistically imprecise (95% CI on C = 0.730 likely spans 0.66–0.80
   based on typical SE at n approximately 100–140). Without reporting confidence
   intervals on per-stratum C, the S1 advantage cannot be certified as
   statistically reliable.

**S4 t(11;14) (C = 0.644) — lowest benefit, highest clinical-decision tension:**

t(11;14) defines the BCL2-high, venetoclax-responsive subset. A lower C = 0.644
may reflect genuine biological homogeneity (this subgroup has a more defined
molecular program and clinical behavior), limiting the model's ability to
discriminate within it. More importantly, the clinical action space in t(11;14)
is already relatively clear (venetoclax-containing regimens, CANOVA-eligible),
so incremental discrimination from v11.5 provides limited marginal benefit.

### Q3. The 90% calibrated PFS interval is a population-level claim. What is the smallest clinical unit where the interval is actionable?

**The interval is actionable at the cytogenetic stratum level, not at the
individual patient level without further calibration.**

The Mondrian conformal jackknife+ calibration passing 5/5 strata within ±3% of
nominal (F_S5 STRICT PASS) means the 90% prediction interval achieves the
stated coverage *when averaged across patients within each cytogenetic stratum*.
This is a marginal stratum-level coverage guarantee.

What it does NOT certify:
- Individual-level coverage: for a single patient, the 90% interval may be
  miscalibrated due to covariate shift from rare molecular combinations (e.g.,
  del17p + t(11;14) co-occurrence, approximately 1–2% of cases)
- Interval coverage outside the five defined strata (e.g., patients with
  multiple simultaneous high-risk lesions, or extramedullary disease at diagnosis)
- Coverage under different treatment regimens from those in the MMRF CoMMpass
  era of enrollment (pre-Dara quadruplet for many patients)

**Minimum clinically actionable unit:** A cytogenetic stratum with n >= 50
patients and known stratum membership for the new patient. Below this, the
interval is a research-grade estimate, not a clinically certified confidence
statement.

**Practical implication:** A treating hematologist could quote the 90% interval
to a patient in the following form: "Based on patients with similar genetics
(del17p) in the MMRF registry, there is approximately 90% confidence that
your time to needing a second treatment line falls between [L, U] months,
assuming treatment similar to those in the registry." This is a prognostic
communication, not a treatment decision, and must be accompanied by the
disclaimer that the interval is derived from registry data and may not reflect
outcomes under current Dara-quadruplet therapy.

### Q4. What is the ENROLLMENT lift compared to using ISS alone?

**Marginal, and unquantifiable without a head-to-head stratum-reclassification
analysis.**

The honest answer requires comparing, for each ISS stage (I/II/III), the
fraction of patients that v11.5 reclassifies into a different predicted-risk
tier. This analysis is not reported.

What can be inferred from the C-index:

- R-ISS C-index for PFS in NDMM approximately 0.68–0.72 (published range).
- v11.5 C = 0.6955 (with TIE vs. mmSYGNAL at 0.6957).
- If R-ISS itself achieves C approximately 0.70 on this MMRF cohort (which is
  plausible given the CoMMpass population), the marginal lift of v11.5 over
  R-ISS is approximately zero to 0.02 C-units.

**Enrolling implication:** A trial that currently enrolles by R-ISS high-risk
(approximately 20–25% of NDMM) would not meaningfully expand or contract its
enrolled cohort if v11.5 replaced R-ISS, because the two discriminate similarly.
For v11.5 to demonstrate enrollment lift, it must identify patients who are
R-ISS low-risk but v11.5 high-risk (or vice versa), show that those patients
have worse (or better) outcomes than their R-ISS stratum predicts, and replicate
this in an independent cohort.

Until that reclassification analysis is published with a replication cohort,
the enrollment-lift claim cannot be made.

### Q5. What is the clinical readiness gap?

| Gap | Severity | What bridges it |
|---|---|---|
| External validation | Critical | At least one independent MM cohort (IFM, EMN, GMMG, HOVON, or DFCI/Heidelberg) with paired genomics and OS/PFS. N >= 400 preferred. |
| IMWG-endpoint concordance | Critical | Demonstrate that v11.5 risk score predicts IMWG-defined PD (not only tt2L). Requires a cohort with systematic IMWG assessment at each visit. |
| Individual-level calibration | High | Current calibration is marginal (stratum-level). For individual-level intervals, patient-level conformal coverage must be demonstrated. |
| Pre-registration | High | IRB submission as a trial stratification covariate requires a pre-registered analysis plan with decision threshold locked before any prospective enrollment. |
| Regulatory pathway | High | As a Software as a Medical Device (SaMD) input to trial stratification, v11.5 requires documentation under 21 CFR 820 / EU MDR 2017/745 quality systems, even if not FDA-cleared. |
| Treatment-era mismatch | Moderate | MMRF CoMMpass enrolled from approximately 2011–2019. Patients who received Dara-quadruplet (FDA-approved 2019–2023) are underrepresented or absent. The calibration may be miscalibrated for current SOC. |
| Per-stratum confidence intervals | Moderate | No CI on per-stratum C-index is reported. S1 del17p n approximately 80–140 means the 0.730 estimate has wide uncertainty. |
| Prospective use protocol | High | A formal prospective use protocol (similar to a Phase 0/I biomarker study) is required before the score appears in a medical record or informs a trial consent discussion. |

---

## 4. Actionable Today

*(Definition: a board-certified MM hematologist could do this tomorrow with v11.5
deployed in a research informatics setting, with no regulatory clearance, using
the result as supplementary scientific context only.)*

**4.1 Research prognostic enrichment in retrospective cohorts.**
v11.5's risk score can be applied to banked MMRF CoMMpass samples to identify
a "model-high-risk" subgroup for retrospective survival analysis. This enriches
biomarker hypothesis generation for future trial design. No patient contact
required; no IRB amendment needed for retrospective analysis of de-identified
registry data already accessed under the existing data use agreement.

**4.2 del17p sub-stratification for correlative science.**
Within del17p patients already enrolled in institutional studies, v11.5 can
stratify into model-high vs. model-low subgroups for correlative biomarker
analysis (e.g., MRD kinetics, plasma cell proliferation index, immune
microenvironment profiling). This is consistent with standard correlative-arm
science and does not require prospective use of the model as a clinical tool.

**4.3 Hypothesis generation for trial design.**
The per-stratum C-index pattern (S1 del17p > S3 +1q21 > S2 t(4;14) > S5 other >
S4 t(11;14)) generates testable hypotheses about which cytogenetic subgroups are
most informative for an ML-biomarker-stratified trial arm. This can inform the
statistical design of a Phase II biomarker-enrichment study without any patient
exposure to the model.

**4.4 Parallel scientific claim alongside mmSYGNAL.**
Since v11.5 ties mmSYGNAL (C = 0.6955 vs. 0.6957, paired test TIE), the
paired result can be published as "independent validation of mmSYGNAL-class
discrimination on MMRF N = 787, with the addition of Waddington-derived
trajectory features." This is a scientific contribution, not a clinical one.
It strengthens the field's confidence in the mmSYGNAL result without claiming
superiority.

---

## 5. Actionable After Phase 1 Validation

*(Definition: requires external validation in >= 1 independent MM cohort, N >= 300,
with outcome data and a pre-registered analysis plan. Approximately 18–24 months
effort if data access is in place.)*

**5.1 Trial stratification covariate (exploratory arm).**
After external validation, v11.5 may be incorporated as an exploratory
stratification factor in a Phase II MM trial under the following conditions:
- External validation C >= 0.68 in the independent cohort (non-inferior to R-ISS)
- Decision threshold pre-registered in the SAP before enrollment begins
- Score computed from baseline samples only (no post-randomization features)
- Score labeled "investigational" in all regulatory documents; not used for
  eligibility gating until Phase III evidence exists

This is the precedent set by GEP-70 / SKY92 / RS-MM in ISS-augmented risk
models, where gene expression signatures were first used as stratification
covariates before becoming standalone eligibility criteria.

**5.2 MRD-directed therapy trial enrichment for del17p.**
If S1 del17p validation confirms C >= 0.70 in an independent cohort, v11.5
could serve as an enrichment biomarker to identify the highest-risk del17p
patients for aggressive MRD-directed strategies (e.g., early consolidation with
a CAR-T product in patients who remain MRD-positive after Dara-RVd induction).
This is a biomarker-enrichment design, not a diagnostic, and falls under FDA
Enrichment Strategy guidance (2019).

**5.3 ISS-discordant high-risk identification.**
Publish the reclassification analysis: fraction of ISS I patients with v11.5
high risk, their observed tt2L vs. ISS I patients with v11.5 low risk, in
both the MMRF discovery and the external validation cohort. If the hazard ratio
is >= 1.5 with 95% CI excluding 1.0, this validates the incremental utility of
v11.5 over ISS, which is the required bar for trial adoption.

**5.4 Prognostic communication tool.**
After external validation, the calibrated 90% prediction interval for a specific
cytogenetic stratum could be incorporated into an institutional prognostic
communication template, accompanied by the full mandatory disclaimer from
CLINICAL_ACTIONABILITY_RUBRIC.md §5.1 and the MMRF-era treatment caveat.

---

## 6. Not Actionable / Needs L2 Evidence

*(Definition: requires Phase III evidence, regulatory clearance, or prospective
clinical trial data with IMWG-defined endpoints.)*

**6.1 Treatment selection between specific regimens.**
v11.5 predicts tt2L ranking; it does not output a treatment recommendation.
Selecting RVd vs. Dara-RVd vs. KRd for a specific patient based on a v11.5
risk score is not actionable. The Pearl ceiling (L1-with-structural-prior)
is explicit: this model predicts ranking, not intervention effects. Any use
that maps a risk score to a treatment choice is outside the declared scope
and constitutes misuse.

**6.2 BCMA-targeted therapy eligibility.**
BCMA surface density, which governs eligibility for ide-cel, cilta-cel,
teclistamab, and belantamab mafodotin, is not modeled by v11.5. The model
cannot predict BCMA loss or BCMA-targeted therapy resistance. Using the
model to inform BCMA therapy sequencing is not actionable and could mislead.

**6.3 Pediatric or secondary plasma cell disorders.**
v11.5 is trained on MMRF CoMMpass, which enrolls adults >= 18 years with
newly-diagnosed MM. It does not apply to pediatric plasma cell tumors,
AL amyloidosis, POEMS syndrome, solitary plasmacytoma, or Waldenström
macroglobulinemia.

**6.4 Real-time clinical decision at the individual patient level.**
No individual patient should have their treatment plan influenced by v11.5
until all Phase 1 validation conditions in Section 5 are met AND a prospective
use protocol has been IRB-approved. The current result is a retrospective
registry analysis; it has no prospective performance record.

**6.5 Claim of superiority over mmSYGNAL.**
The paired test is a TIE (C = 0.6955 vs. 0.6957). Any manuscript language
claiming v11.5 "outperforms" or "improves upon" mmSYGNAL on this endpoint
and cohort is not supported and must not appear in abstract, title, or
conclusion. The honest framing is "non-inferior" or "equivalent."

**6.6 MRD surrogate use.**
v11.5 does not predict MRD negativity. MRD negativity (10^-5 or 10^-6 by
NGS/flow) is a distinct endpoint with distinct clinical meaning. tt2L-based
risk discrimination does not substitute for MRD-driven decision-making.

---

## 7. Comparison to mmSYGNAL Clinical Positioning

| Dimension | mmSYGNAL (PMID 40169765) | v11.5 |
|---|---|---|
| Training basis | SYGNAL multi-omics, 881 MM patients | MMRF CoMMpass N = 787 |
| Validation cohorts | 5 independent cohorts, N = 1,367 total | None reported external to MMRF (single cohort) |
| Primary endpoint | PFS at diagnosis, pre/post-ASCT, multiple relapses | tt2L (surrogate PFS) on MMRF only |
| Reported C-index | Not extracted (abstract does not state specific C-index value; outperforms ISS, cytogenetics, multi-gene panels stated) | 0.6957 (mmSYGNAL routed score component) vs. 0.6955 (v11.5 ensemble) — TIE |
| Treatment-response predictions | 67 drugs tested on 8 RRMM patients; concordant with in vitro killing | Not reported for v11.5 |
| Clinical action proposed by Murie/Baliga | Individualized dynamic risk assessment and treatment selection; proposed as a clinical decision support tool across the disease course (newly-diagnosed through relapsed) | v11.5: prognosis and trial stratification only; treatment selection explicitly out of scope (Pearl ceiling L1) |
| Prospective validation | Not stated in abstract | Not performed |
| Regulatory status | Research tool; not FDA-cleared SaMD | Research tool; not FDA-cleared SaMD |
| v11.5 positioning relative to mmSYGNAL | Non-inferior on MMRF/tt2L; adds Waddington trajectory features and calibrated 90% intervals with Mondrian coverage; does NOT add treatment-selection capacity | The Waddington features and formal conformal calibration are the genuine additions; treatment selection remains out of scope |

**Key positioning statement:** mmSYGNAL's clinical claim is broader than v11.5's.
Murie/Baliga proposed treatment selection guidance across 67 drugs in RRMM —
a Level 2 clinical claim requiring tumor-drug co-incubation experiments and
independent prospective validation. v11.5, constrained to the Pearl ceiling
of L1-with-structural-prior, cannot and should not replicate that claim.
The v11.5 contribution is formally narrower: calibrated prognostic discrimination
in a well-characterized cytogenetic subgroup framework, with an honest
quantification of what the prediction interval certifies at the stratum level.
This is a more epistemically conservative position, but it is also more
internally consistent and less likely to be contested in peer review.

---

## 8. Summary Verdict

| Dimension | Verdict |
|---|---|
| C = 0.696 supports risk ranking | PASS — comparable to R-ISS; clinically interpretable |
| C = 0.696 supports triage decision today | CONDITIONAL — supplementary research context only; not standalone triage |
| del17p C = 0.730 is clinically actionable | CONDITIONAL — requires CI reporting and external validation |
| 90% interval actionable at individual level | FAIL — stratum-level guarantee only; individual-level actionability not certified |
| ISS enrollment lift demonstrated | NOT DEMONSTRATED — reclassification analysis not reported |
| Treatment selection | PROHIBITED — Pearl ceiling L1; not an intervention-effect predictor |
| Superiority over mmSYGNAL | NOT DEMONSTRATED — paired TIE |
| Clinical readiness tier (per CLINICAL_ACTIONABILITY_RUBRIC.md) | Tier 2: "Note in chart / supplementary research context" |

---

## 9. Required Changes Before Phase 1 Validation Submission

1. Report per-stratum C-index confidence intervals (bootstrap 95% CI recommended;
   or exact Somers' D asymptotic CI). The S1 del17p C = 0.730 is the headline
   subgroup result and it must have a CI.

2. Report ISS and R-ISS C-index on the same MMRF split as a within-cohort
   comparator. Without this, the claim that v11.5 adds value over current
   standard risk stratification cannot be assessed.

3. Report the stratum-level reclassification analysis: fraction of patients who
   change risk tier when moving from R-ISS to v11.5 scoring, with outcome
   concordance.

4. Qualify all "PFS" language as "tt2L (surrogate PFS)" throughout any manuscript
   or slide deck. Do not shorten to "PFS" without the qualifier.

5. Add the MMRF enrollment era caveat (approximately 2011–2019) to any discussion
   of calibration applicability under current Dara-quadruplet SOC.

6. Do not use C-index TIE with mmSYGNAL as a claim of equivalence without a
   non-inferiority margin pre-specified in the analysis plan. Post-hoc equivalence
   from a TIE paired test is not a registered non-inferiority analysis.

7. Do not claim individual patient-level interval validity. Restrict interval
   claims to stratum-level population coverage as demonstrated.

---

*Audit version 1.0. 2026-05-03. Must be updated when external validation data
are available or when IMWG response labels are incorporated as training outcomes.*
