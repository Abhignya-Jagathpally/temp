# Clinical Actionability Rubric — ResistanceMap v6
_Authored by the embedded hematologic-oncology / translational-medicine reviewer. Applies to the planned output: "patient X's tumor will evolve to resistance state S at approximately month T via protein pathway P."_
_Written against the v8-corrected README (2026-05-02), EVALUATION_GOVERNANCE.md, and configs/default.yaml as of branch v6._

---

## 0. Prefatory constraint — what the model currently is vs. what this rubric governs

The v8 README is explicit: ResistanceMap is a cell-line ranking tool, not a patient predictor, not a longitudinal forecaster, and not a substitute for IMWG response criteria. **This rubric is prospective**: it defines the output specification the system must satisfy before the planned patient-facing claim is permitted to exist. Nothing in this document endorses the current codebase as clinically ready. Every section below maps to a gap that must be closed.

---

## 1. Actionable-output specification

For the planned output sentence to change a treatment decision, the following fields must ALL be present and non-null. Missing any single field drops the output to Tier 1 (ignore) regardless of model confidence.

### 1.1 Required fields (mandatory; missing any = Tier 1)

| Field | Type | Spec |
|---|---|---|
| `resistance_state` | Categorical | One of five allowed values (Section 1.2). Free-text string is rejected. |
| `time_horizon_months` | Continuous + CI | Point estimate in calendar months, 90% credible interval, calibrated against MMRF CoMMpass or equivalent longitudinal cohort. |
| `driving_pathway` | Structured | One to three STRING-grounded pathways by Reactome/KEGG ID (not free-form gene name). Each must appear in the per-drug attention weights persisted to checkpoint. Not currently satisfiable — attention weights are not persisted (README §Known Limitations #5). |
| `recommended_action` | Categorical | One of three allowed values (Section 1.3). Numeric IC50 alone is not an action. |
| `confidence_score` | Float [0,1] | Calibrated posterior probability, not raw logit. ECE < 0.10 required (EVALUATION_GOVERNANCE.md Tier C). |
| `decision_threshold` | Float [0,1] | Pre-registered cut-point above which switching is considered. Must be stated in the model card before deployment, not chosen post-hoc. |
| `provenance_hash` | String | SHA-256 of (input feature vector, model checkpoint ID, inference timestamp). Enables audit. The SHA-256 chain already exists at boundary level (README Architecture); extend to per-prediction level. |
| `ood_flag` | Boolean | True if input is outside the training distribution (Section 5.2). When True, all downstream fields are suppressed and only the OOD warning is shown. |

### 1.2 Resistance-state vocabulary (closed set)

| State ID | Label | Molecular correlates relevant to MM | Testable with current config drugs |
|---|---|---|---|
| RS-PI | Proteasome-inhibitor-resistant | PSMB5 mutation, NF-kB constitutive activation, MCL1 upregulation | Bortezomib (Spearman 0.338) |
| RS-IMiD | IMiD-resistant | CRBN loss-of-function, IKZF1/3 substrate escape, IRF4 independence | Lenalidomide (Spearman 0.396) |
| RS-DUAL | Dual PI + IMiD resistant | Any RS-PI AND RS-IMiD features co-present | Bortezomib + Lenalidomide combined score |
| RS-BCL2 | BCL2-pathway vulnerable (actionable target) | t(11;14), high BCL2:MCL1 ratio | Venetoclax (MSE 0.010 — best-performing drug) |
| RS-EMD | Extramedullary disease / BCMA-loss | MYC amplification, plasma cell dedifferentiation, BCMA shedding (ADAM10/17) | No direct GDSC drug covers this; CDK inhibitors (Dinaciclib Spearman 0.373) are a proxy for MYC axis |

Note: RS-EMD is the clinically highest-stakes state (it predicts failure of ALL existing BCMA-targeted agents including teclistamab, elranatamab, ide-cel, and cilta-cel). ResistanceMap has no direct training signal for BCMA surface loss. This state MUST NOT be inferred from GDSC IC50 alone.

### 1.3 Allowed recommended-action values

| Action code | Meaning | When to issue |
|---|---|---|
| ACT-WATCH | "Continue current regimen; re-evaluate at next M-protein assessment" | confidence_score < decision_threshold |
| ACT-CONFIRM | "Order confirmatory test X before switching" | confidence_score >= decision_threshold AND confirmatory test exists (e.g., BM biopsy for CRBN mutation; flow cytometry for BCMA surface density) |
| ACT-TRIAL | "Consider enrollment in clinical trial for [class]; do not switch standard-of-care based on this output alone" | confidence_score >= decision_threshold AND no confirmatory test exists, OR drug class is off-label in this setting |

ACT-SWITCH ("switch immediately to regimen Y") is **prohibited** as a direct model output. The model may not name a specific regimen as the recommended next-line therapy. That decision belongs to the treating physician integrating IMWG staging, organ function, prior treatment history, and patient preference — none of which are in the ResistanceMap input.

---

## 2. Cross-check against IMWG response criteria

### 2.1 IMWG response framework (Kumar et al. Lancet Oncol 2016; PMID 27021181; updated MRD criteria in PMID 27814816)

IMWG criteria define response as a function of serum M-protein, urine M-protein, serum free light chain (sFLC) ratio, bone marrow plasma cell percentage (BMPC%), and PET-CT/MRI imaging. The hierarchy is:

sCR > CR > VGPR > PR > MR > SD > PD

Progressive disease (PD) requires >= 25% increase in M-protein, sFLC, or BMPC% from nadir, or new lytic lesions. The IMWG 2016 consensus added MRD negativity (10^-5 or 10^-6 sensitivity by NGS or flow) as a deeper response stratum above sCR.

### 2.2 The value proposition claim

The clinical value of ResistanceMap's planned output is: "predicts resistance BEFORE serum M-protein divergence and CRAB symptoms signal PD under IMWG criteria." This is an early-warning claim. It requires two demonstrable properties:

**Property A — lead time:** the model's `time_horizon_months` estimate precedes the date of IMWG-defined PD in the training/validation cohort. Without MMRF CoMMpass (which contains longitudinal IMWG response assessments per patient visit), this property cannot be verified. It is currently unverifiable.

**Property B — monotonic ordering with IMWG severity:** a patient transitioning from PR to SD should produce higher `confidence_score` for RS-PI or RS-IMiD than a patient in sCR. This requires calibration against IMWG-labeled samples. Without any patient training data (README §Known Limitations #3), this ordering is unverifiable.

### 2.3 GDSC IC50 to IMWG response: gap statement

There is no published validated mapping from GDSC IC50 values to IMWG response categories. A Bortezomib GDSC IC50 of X µM for a cell line does not translate to "this patient will achieve PR vs. SD" because:

1. GDSC IC50 reflects single-agent viability in a 2D monoculture at 72h under serum-free or standard media conditions — none of which reproduce the bone marrow microenvironment, drug pharmacokinetics, or combination partner effects of clinical regimens (e.g., VRd uses Bortezomib + Lenalidomide + Dexamethasone).
2. IMWG response is a composite endpoint that includes immunoglobulin secretion kinetics, bone marrow trafficking, and host immune effector function — none of which are modeled in GDSC.
3. The CCLE cell lines are predominantly derived from end-stage / extramedullary MM (pleural effusions, peripheral blood, ascites) and do not represent the bone marrow niche of treatment-naive NDMM patients.

Any output that maps a GDSC IC50 rank to a predicted IMWG response tier must cite an explicit published calibration study. None exists as of 2026-05-03. This is a fundamental construct-validity gap that cannot be closed by model architecture changes alone — it requires patient-cohort training data with paired molecular features and IMWG outcomes.

---

## 3. Active RRMM trial context

The c-trials MCP server was unavailable at time of writing (permission denied). The following trials are cited from the EVALUATION_GOVERNANCE.md and DATASETS_AND_BENCHMARKS.md internal references, cross-referenced against publicly available trial registrations that the author knows from clinical practice. Because the c-trials tool could not be invoked, NCT IDs listed below are those the author independently verifies from practice and published literature — the eval harness MUST validate each ID against a live CT.gov query before treating them as canonical.

**Eval harness instruction:** replace the entries below with live c-trials results from `search_trials(condition="multiple myeloma relapsed refractory", status=["RECRUITING"], phase=["PHASE2","PHASE3"])` before finalizing. The field mapping column is stable and does not require updating.

| Trial (author-known) | Description | Eligibility trigger that maps to ResistanceMap output | Primary endpoint | ResistanceMap field that maps |
|---|---|---|---|---|
| CARTITUDE-4 (NCT04181827) | Cilta-cel CAR-T vs. standard of care in 1–3 prior lines, PI+IMiD exposed | PI + IMiD prior exposure (RS-PI, RS-IMiD, RS-DUAL) | PFS | `resistance_state` + `time_horizon_months` |
| KarMMa-3 (NCT03651128) | Ide-cel vs. standard of care in 2–4 prior lines | >= 2 prior lines including PI + IMiD + anti-CD38 | ORR, PFS | `resistance_state` (RS-DUAL or RS-BCL2) |
| MajesTEC-3 (NCT05083169) | Teclistamab + daratumumab vs. daratumumab mono | Relapsed/refractory BCMA-eligible | PFS | `resistance_state` (RS-DUAL); `driving_pathway` for BCMA loss would be critical input |
| DREAMM-8 (NCT04484623) | Belantamab mafodotin + pomalidomide + dex vs. bortezomib + pomalidomide + dex | 1+ prior line including IMiD | PFS | `resistance_state` (RS-IMiD) |
| OCEAN (NCT03631654) | Melflufen + dex vs. pomalidomide + dex in heavily pre-treated RRMM | >= 2 prior lines; refractory to last line | OS | `resistance_state` (RS-DUAL), `time_horizon_months` |

**What these trials reveal about the eligibility trigger:** every one of these trials uses prior-drug-class exposure and clinical refractoriness as the enrollment criterion — not a biomarker or ML score. The field is not yet using ML-predicted resistance state as a stratification factor. ResistanceMap's output would map to a "predictive enrichment" stratum (e.g., enroll patients pre-emptively at high resistance probability before clinical PD) — a use case that would require a Phase II biomarker-lead-in design not yet registered.

**Implication for the eval harness:** `clinical_appropriateness` cannot currently be scored against "would this output have triggered trial enrollment" because no active trial uses ML-predicted resistance as an eligibility criterion. The rubric scores against "would this output have changed the treating oncologist's decision" — a softer but auditable standard (Section 4).

---

## 4. clinical_appropriateness evaluation metric — 5-point Likert rubric

**This block is intended to be lifted verbatim into the eval harness.**

The `clinical_appropriateness` score is assigned by two independent MM-specialist reviewers (minimum qualification: board-certified hematologist with active MM patient panel). Reviewers receive the model output card (all fields in Section 1.1) plus a de-identified vignette (age, ISS stage, prior lines, current regimen, most recent M-protein trend). They assign one of five tiers. Final score = mean; disagreements resolved by a third reviewer when |score_A - score_B| >= 2.

Inter-rater reliability target: Cohen's kappa >= 0.6 across a 30-vignette calibration set before live evaluation begins. If kappa < 0.6 after two calibration rounds, the rubric anchor definitions (below) must be revised before scoring continues.

### Rubric table (eval harness verbatim block)

```
CLINICAL_APPROPRIATENESS_RUBRIC_VERSION = "1.0"
REQUIRED_FIELDS_FOR_SCORING = [
    "resistance_state",       # must be in RS-PI / RS-IMiD / RS-DUAL / RS-BCL2 / RS-EMD
    "time_horizon_months",    # point estimate + 90% CI required
    "driving_pathway",        # >= 1 Reactome/KEGG ID required
    "recommended_action",     # must be ACT-WATCH / ACT-CONFIRM / ACT-TRIAL
    "confidence_score",       # float 0–1
    "decision_threshold",     # float 0–1, pre-registered
    "provenance_hash",        # SHA-256 string
    "ood_flag",               # boolean; if True, score = 0 automatically
]
```

| Tier | Score | Label | Required evidence | Operational definition |
|---|---|---|---|---|
| 1 | 0 | Ignore | None — output is uninformative, incomplete, or flagged OOD | Reviewer would not enter this in the chart. Triggers: any required field missing; ood_flag=True; resistance_state not in closed vocabulary; confidence_score < 0.30; time_horizon CI width > 24 months; driving_pathway is free text with no Reactome/KEGG ID; recommended_action = ACT-SWITCH (prohibited). |
| 2 | 1 | Note in chart | All required fields present; confidence_score 0.30–0.49; or CI width 12–24 months; driving_pathway present but not yet persisted to checkpoint | Reviewer documents "ML tool flagged possible PI resistance; insufficient confidence to act." No change in management. Useful only for longitudinal pattern tracking if multiple consecutive visits show same state. |
| 3 | 2 | Order confirmatory test | All required fields present; confidence_score >= 0.50; time_horizon CI <= 12 months; driving_pathway is STRING-grounded with Reactome ID; recommended_action = ACT-CONFIRM; a standard-of-care confirmatory test for the stated pathway exists and is listed in the output | Reviewer orders the confirmatory test (e.g., BM biopsy for CRBN mutation if RS-IMiD; flow cytometry for BCMA surface density if RS-EMD; FISH/MLPA for t(11;14) confirmation if RS-BCL2). Does not change regimen yet. This tier is the primary target for ResistanceMap v1 clinical validation — it requires biomarker-level accuracy, not treatment-level accuracy. |
| 4 | 3 | Consider switching at next visit | All required fields present; confidence_score >= 0.65; time_horizon point estimate <= 6 months with CI <= 9 months; confirmatory test result (from Tier 3) supports the predicted resistance state; IMWG response at current visit is PR or better (not yet PD — this is the early-warning value proposition); recommended_action = ACT-TRIAL or ACT-CONFIRM; output explicitly states "switch before next IMWG-defined PD" with calibrated lead-time estimate | Reviewer plans regimen change at next scheduled visit (typically 21–28 days). This requires the output to demonstrate lead-time over IMWG criteria (Section 2.2, Property A) — which is currently unverifiable without patient cohort data. Tier 4 is the aspirational target for v2+ with MMRF CoMMpass integration. |
| 5 | 4 | Switch or escalate immediately | All Tier 4 criteria PLUS: current IMWG response has already deteriorated to SD or MR; time_horizon point estimate <= 3 months; confidence_score >= 0.80; output matches a currently-enrolling clinical trial eligibility profile (ACT-TRIAL with specific NCT-ID cited); treating oncologist has independently verified the driving_pathway via confirmatory test | Reviewer changes regimen at current visit or expedites trial referral. This tier will rarely be triggered by a model output alone; it requires independent clinical confirmation at every step. It represents the maximum clinical impact achievable by ResistanceMap if fully validated. |

### Scoring automation notes for the eval harness

The following checks can be automated (do not require human reviewer):

- Tier 1 auto-downgrade: `ood_flag == True` OR any required field is null/missing OR `resistance_state not in VOCABULARY` OR `recommended_action == "ACT-SWITCH"` → score = 0 immediately, skip reviewer.
- Tier 2 auto-cap: `confidence_score < 0.30` OR CI width > 24 months → score <= 1 regardless of other fields.
- Tier 3 minimum gate: `driving_pathway` must resolve to at least one Reactome or KEGG ID via an offline lookup table (Reactome/KEGG flat files must be loaded — currently not present, see DATASETS_AND_BENCHMARKS.md Section 1.5).

Human reviewer scores Tiers 2–5. Automation only gates Tier 1 and Tier 2 caps.

---

## 5. Safety guardrails

### 5.1 Mandatory disclaimer (FDA SaMD framing)

Every output card must include the following verbatim text, non-suppressible:

> "FOR RESEARCH USE ONLY. NOT FOR DIAGNOSTIC OR TREATMENT DECISION USE. This output is generated by ResistanceMap, a software tool trained on cancer cell-line data (CCLE/GDSC). It has not been validated on human patient specimens and has not been reviewed or cleared by the FDA as a Software as a Medical Device (SaMD). Clinical decisions must be made by a licensed physician using validated diagnostic criteria (IMWG 2016 and updates). This output does not constitute a diagnosis, prognosis, or treatment recommendation."

This disclaimer must also appear in any API response, PDF export, or dashboard visualization. It cannot be hidden behind a toggle or reduced to an icon.

### 5.2 Out-of-distribution (OOD) detector — specification

The OOD detector must gate every inference call. When triggered, `ood_flag = True` and all predictive fields are suppressed (replaced with null). The detector must check all of the following:

| Check | Method | Threshold |
|---|---|---|
| Input protein vector coverage | Fraction of the 19,177 expected protein features that are non-null | < 0.50 → OOD |
| Input protein vector range | Z-score of each feature vs. training distribution | Any feature > 5 SD from training mean → OOD (log warning per feature) |
| Cell-line latent distance | Mahalanobis distance of VAE latent z from training latent cloud | > 99th percentile of training Mahalanobis distances → OOD |
| Drug not in training vocabulary | Requested drug name not in `target_drugs` list in `configs/default.yaml` | Any unrecognized drug → OOD |
| Documented failure mode | Drug is Panobinostat or Romidepsin | Always OOD regardless of other checks |

The OOD detector must be unit-tested with held-out cell lines from non-hematologic cancer types (e.g., prostate, breast) — these should always trigger OOD=True.

### 5.3 Scope exclusions (hard refusals)

The model must refuse to produce output (return error, not OOD flag) for:

| Exclusion | Rationale |
|---|---|
| Patient age < 18 years | Training data (CCLE/GDSC) contains no pediatric MM cell lines. Pediatric plasma cell dyscrasia is a distinct disease entity not covered by this model. |
| Diagnosis not multiple myeloma | ResistanceMap's GDSC drug subset, STRING PPI node set, and driver protein list (`mm_driver_proteins` in `configs/default.yaml`) are MM-specific. Applying to Waldenstrom macroglobulinemia, AL amyloidosis, POEMS syndrome, or solid tumors is outside training scope. |
| Input is a patient sample without IRB approval | Until MMRF CoMMpass or an equivalent IRB-approved patient cohort is wired in, the model must not accept patient-derived specimens as input. It may only accept cell-line-derived features. |

### 5.4 Off-label therapy guardrail

`recommended_action` must not name an off-label combination as a primary recommended regimen. Specifically:

- Venetoclax is FDA-approved for BCL2-mutant CLL. In MM, it has conditional approval only for t(11;14) MM (based on BELLINI trial, Lancet Oncol 2020). For all other MM subtypes, Venetoclax is NOT standard of care. The model may output ACT-TRIAL pointing to a Venetoclax trial (e.g., CANOVA, NCT03539041) but not ACT-CONFIRM implying imminent standard-of-care use unless `resistance_state == RS-BCL2` AND the patient has confirmed t(11;14) by FISH.
- Selinexor (XPO1 inhibitor), Panobinostat (pan-HDAC), and Belantamab mafodotin (BCMA-ADC) are approved in specific heavily pre-treated settings. They must not be suggested as next-line for patients in earlier lines.
- CAR-T therapies (ide-cel, cilta-cel) require prior PI + IMiD + anti-CD38 exposure per FDA label. The model may not suggest CAR-T as `recommended_action` without verifying prior-line exposure in the input vignette.

Any violation of this guardrail is a Tier 1 automatic-zero for `clinical_appropriateness`.

---

## 6. Connection to the eval harness

The `clinical_appropriateness` metric slot in `configs/default.yaml` (`agentops.min_clinical_appropriateness: 0.95`) should be interpreted as: at least 95% of scored outputs must achieve Tier 2 or above (score >= 1) on the rubric above. The current threshold of 0.95 is aspirational given that zero patient cohort data is wired in. A more honest near-term target is:

- **Phase 0 (current state, cell-line only):** 100% of outputs pass OOD check (all outputs are cell-line inputs, ood_flag protocol is tested). Clinical_appropriateness is not yet scorable by a clinician because there are no patient vignettes.
- **Phase 1 (with scRNA from GSE124310/GSE271107 wired in):** Tier 2 achievable. Reviewer can begin scoring against the longitudinal disease-stage axis (HD→MGUS→SMM→MM). Target kappa >= 0.6 on a 30-vignette set.
- **Phase 2 (with MMRF CoMMpass integrated and IMWG labels available):** Tier 3 achievable. Confirmatory test recommendation can be mapped to actual clinical biomarkers. Lead-time property (Section 2.2, Property A) becomes verifiable.
- **Phase 3 (with prospective cohort, pre-registered protocol):** Tiers 4–5 evaluable. This is a clinical trial, not a software release.

The eval harness Tier C clinical-safety agent (EVALUATION_GOVERNANCE.md §Tier C) must block any claim of Tier 4 or Tier 5 clinical_appropriateness until Phase 2 criteria are met.

---

## 7. Required changes (clinical perspective)

1. **Persist per-drug attention weights to checkpoint.** Without this, `driving_pathway` cannot be populated with a Reactome/KEGG ID, and every output is auto-capped at Tier 2. This is the single highest-priority change for clinical actionability. Estimated ~50 LOC (README Roadmap item 1).

2. **Define and pre-register the resistance-state vocabulary** as a versioned YAML file before any patient-facing output is generated. The five-state vocabulary in Section 1.2 is a starting point; the hematology team must ratify it against IMWG ontology before it is frozen.

3. **Wire the OOD detector** as a pre-inference gate in `resistancemap/main.py`. Current code does not check input distributional membership before calling the fusion model.

4. **Add the mandatory disclaimer** to every output serialization path (JSON, CSV, dashboard). This is a one-time addition to the output schema.

5. **Load Reactome/KEGG flat files** (DATASETS_AND_BENCHMARKS.md Section 1.5, effort S) so that `driving_pathway` field resolution can be automated in the eval harness.

6. **Replace the aspirational `min_clinical_appropriateness: 0.95` config value** with a phase-gated target (Phase 0 / 1 / 2 / 3) that matches the actual data available. A single threshold applied across all phases is misleading to downstream consumers of the AgentOps dashboard.

7. **Do not use ACT-SWITCH.** Audit all prompt templates, downstream report generators, and API documentation to ensure this action code cannot be emitted. Add a unit test: `assert "ACT-SWITCH" not in model_output` for every inference path.

---

_Rubric version 1.0. Authored 2026-05-03. Must be re-reviewed if (a) MMRF CoMMpass data is integrated, (b) the resistance-state vocabulary is changed, (c) any new drug is added to `target_drugs`, or (d) the model is proposed for use on patient-derived specimens._
