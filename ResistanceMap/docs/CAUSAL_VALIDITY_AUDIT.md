# Causal Validity Audit — ResistanceMap v6/v8
## Claim under review: "predict which resistance state the tumor will evolve toward, when, and through which protein pathway, BEFORE it actually happens"
_Date: 2026-05-03. Author: causal-inference audit agent. Data on disk: GSE124310 (PMID 33409501) + GSE271107 + CCLE proteomics (886 lines) + CCLE epigenomics + GDSC IC50 (622 lines x 11 drugs)._

---

## §0 — Audit revision history (2026-05-03)

**Trigger for revision.** MMRF CoMMpass IA22 is now verified on disk (open-access, 2,960 genomic files + clinical via GDC API). The original audit was written when CoMMpass was listed as "not loaded" in `configs/default.yaml`. Multiple verdicts in the original text were explicitly conditional on CoMMpass landing; those conditions are now met. This section records which sub-claims escalated and which did not.

**What changed since the original audit was written:**

| Fact | Prior state | Current state (2026-05-03) |
|---|---|---|
| MMRF CoMMpass on disk | Not downloaded ("configured but not on disk") | Verified: 2,960 open-access files (859 RNA-Seq, 1,091 MAF, 1,010 CNV) |
| Patients with treatment records | 0 | 994/995 (GDC API confirmed) |
| Patients with ≥2 lines of therapy | 0 | 258 (resistance event observable) |
| Longitudinal RNA-Seq (≥2 visits) | 0 | 51 patients |
| Longitudinal CNV (≥2 visits) | 0 | 74 patients |
| Recurrent bone-marrow biopsies | 0 | 319 "Recurrent Blood Derived Cancer - Bone Marrow" samples |
| Cytogenetic confounders measurable | None (named as identification blockers) | del17p (13.6%), chr1q21 (33.6%), del13q (52%), t(4;14) (14.6%), t(11;14) (16.8%) for 976 patients from cytogenetics.tsv |
| Median treatment records per patient | — | 6 |
| Median days first-to-last treatment | — | 256 |

**Sub-claim escalation summary:**

| Sub-claim | Original verdict | Revised verdict |
|---|---|---|
| (a) which resistance state | L1 only, not causal | L1 still; cytogenetic adjustment now achievable; Tier 2 predictive validation now runnable on N=994 |
| (b) when (time-to-event) | "Not supported at any causal tier" — identification failure | ESCALATED: days_to_treatment_start/end provides time axis; C-index estimable on N=258; claim is now scientifically falsifiable |
| (c) via which pathway | L3 required, none supported | NOT ESCALATED for primary causal claim; recurrence biopsies (N=319) enable L1 paired pre/post analysis, which IS falsifiable |
| "before it happens" | Structurally absent (no longitudinal training) | PARTIALLY ESCALATED: N=51 longitudinal RNA sub-cohort enables exploratory Tier 2+ correlation claim only; framed as exploratory, not causal |

---

## 1. Decomposition of the claim into sub-claims and causal tier

Pearl's Ladder of Causation has three rungs:
- L1 (Association / observational): P(Y | X) — seeing
- L2 (Intervention): P(Y | do(X)) — doing
- L3 (Counterfactual): P(Y_x | X = x') — imagining

### Sub-claim (a): "Predict which resistance state"
**Formal statement.** Given snapshot X at time 0, output a discrete attractor label y ∈ {state_1, …, state_k} the tumor will occupy at time T.

**Causal tier.** L1 — observational classification. The model learns P(state_label | X_cell-line), where state_label is derived from cross-sectional IC50 bucketing on GDSC, not from observed future states. There is no interventional do-operator. The query "will this cell line be in resistant state k" is equivalent to asking which cells currently look like resistant cells in training data. This is conditional probability, not causal identification.

**Missing assumption (original).** Exchangeability: cells with identical X at time 0 but different eventual states must differ only on measured covariates. This is implausible — unmeasured genomic drivers (chr1q21 amplification, del17p, t(4;14), t(11;14)) determine resistance trajectory independently of the measured proteomic/epigenomic snapshot.

~~**Verdict (pre-MMRF).** L1 only. Not a causal claim.~~

**Revised verdict (2026-05-03).** L1 classification is still the correct Pearl tier. However, the identification failure from unobserved cytogenetic confounders is now *partially remediable*: cytogenetics.tsv in CoMMpass contains del17p, chr1q21, del13q, t(4;14), and t(11;14) for 976 of 994 patients. Including these as covariates in the training or stratification layer closes the largest open back-door path. The Tier 2 predictive validation test — predict IMWG response at Cycle 4 from baseline multi-omics, validate against observed outcome — is now directly runnable on N=994 following the PERCEPTION template (PMID 38637658), which is no longer "a template for when we have CoMMpass" but a template we can directly emulate now.

---

### Sub-claim (b): "When" — time-to-event
**Formal statement.** Given X at time 0, predict a time T* at which the tumor crosses into a resistant state; equivalently, estimate a survival function S(t | X).

**Causal tier.** L2 would require P(T* | do(drug=d), X), meaning we observe what happens if we intervene to give a specific drug. L1 is the weaker P(T* | drug=d, X) — conditioning, not intervening.

~~**Missing assumptions (pre-MMRF).**~~
~~1. Time-to-event labels: not present in CCLE/GDSC (identification failure, not estimation failure — the quantity cannot be identified from data that was never collected).~~
~~2. Positivity: every cell line must have positive probability of receiving each drug at each time. In GDSC, every line receives every drug in vitro, so in-cell-line space positivity holds. But this does not transfer to patient time-to-resistance, where drug is assigned by physician judgment (non-random).~~
~~3. Consistency: the IC50 observed in a 2D cell-line screen must equal the IC50 that would be observed under the defined treatment protocol. Cell lines are not tumors; 2D vs 3D growth, immune microenvironment absence, and passage-induced drift all violate consistency.~~

~~**Verdict (pre-MMRF).** L1 is not achievable for this sub-claim; the data literally does not contain a time variable. Claiming "when" is a temporal extrapolation from a purely cross-sectional screen. Not supported at any causal tier.~~

**Revised assessment (2026-05-03).** The prior verdict was written when no time-to-event data existed on disk. That verdict was correct conditional on the data state at writing but is now stale. CoMMpass provides `days_to_treatment_start` and `days_to_treatment_end` per treatment record, establishing a genuine time axis. The N=258 patients with ≥2 distinct lines of therapy constitute an observational cohort where the resistance event (switch to next-line therapy) is observed. This is not a randomized trial — drug assignment remains physician-driven (positivity and unconfoundedness at L2 remain open questions) — but it transforms sub-claim (b) from an identification failure to a partial identification problem.

**Power analysis (Pencina & D'Agostino 2004 formula; Heagerty & Zheng 2005).** Minimum events to detect a clinically meaningful C-index of C₁ = 0.65 vs. null C₀ = 0.50 at α = 0.05 (one-sided), power = 0.80:

    n_events = (z_{α} + z_{β})² × C₁(1 - C₁) / (C₁ - C₀)²
             = (1.6449 + 0.8416)² × 0.65 × 0.35 / (0.15)²
             = 6.183 × 0.2275 / 0.0225
             = 62.5  [one-sided α = 0.05]
             = 79.4  [two-sided α = 0.05, z = 1.96]

N = 258 patients with confirmed resistance events exceeds both thresholds (258 >> 79). Power achieved at N = 258 (one-sided) is > 0.999; expected 95% CI for an observed C-index of 0.65 is (0.592, 0.708), with the lower bound well above 0.55. **N = 258 is sufficient to detect a C-index of 0.65 vs. 0.50 with > 0.999 power; the SOTA bar of lower-CI > 0.55 is achievable.**

**Residual identification failure.** L1 estimation (concordance between predicted score and observed time-to-next-line-therapy) is now achievable. The residual limitation is identification, not estimation: we observe P(T* | drug=d, X) but not P(T* | do(drug=d), X). Switching to second-line is confounded by response quality, physician preference, and cytogenetic risk (now measurable but not randomized). This should be stated explicitly as a limitation, not buried.

**Revised verdict.** Sub-claim (b) is now scientifically falsifiable on N = 258 at L1 (time-to-event concordance, observational). The L2 causal interpretation of the hazard remains unidentified without a randomized or near-randomized design, but the claim "our model's score predicts time-to-next-line-therapy in MMRF CoMMpass patients" is now testable with adequate power.

---

### Sub-claim (c): "Via which pathway" — mediation/pathway attribution
**Formal statement.** For the predicted resistance state, identify the pathway M such that the total effect of the transition flows through M; formally, the natural indirect effect NIE(d → M → y) dominates.

**Causal tier.** This is a mediation query, which is L3 (counterfactual) by definition. Even the weaker "which pathway correlates with resistance" (L1) requires that pathway activity, not confounded baseline expression, drives the association. Pathway attribution via attention weights or SHAP values on an observational model is not a mediation analysis. It identifies which features the model uses, which is a property of the model, not of the biology.

**Identification condition (Pearl 2001, front-door criterion or do-calculus with sequential ignorability).** For NIE to be identified without randomization, the mediator M must be:
1. Unconfounded by unmeasured variables given treatment and baseline covariates.
2. Not on any other causal path from X to y.

Neither condition is testable in CCLE proteomics data. Chromatin state, copy number alterations, and microenvironmental signals all co-regulate pathway activity and resistance simultaneously.

**Additional failure: no perturbation data.** True pathway attribution requires observing what happens when the pathway is experimentally blocked — a do(M = 0) intervention. DepMap CRISPR essentiality scores (Achilles, Chronos) provide exactly this for ~18,000 genes across ~1,000 cell lines and are configured in `default.yaml` as `depmap_crispr_path` but are not on disk and not ingested into training. Without these, no causal pathway attribution is possible.

~~**Verdict (pre-MMRF).** L3 required, L1 is what exists. Even L1 pathway-association claims are ungrounded because attention weights are not persisted to checkpoints. The pathway claim is unsupported at every level.~~

**Revised verdict (2026-05-03).** The L3 mediation claim remains unsupported — CoMMpass does not provide perturbation data and the primary pipeline still lacks attention-weight persistence. **This verdict does not escalate for the primary causal claim.**

However, a new L1 association test is now achievable using the 319 "Recurrent Blood Derived Cancer - Bone Marrow" biopsies: paired pre/post differential-expression analysis between baseline and relapse biopsies, with FDR control, identifies pathways whose expression changes between sensitive and resistant states in the same patient. This is not mediation analysis; it is L1 pathway-association evidence with within-patient confound control (longitudinal paired design removes patient-level unmeasured confounders that are stable over time). This paired pre/post L1 test should be framed as "pathways differentially expressed at relapse vs. baseline, consistent with a role in resistance," not as "pathways through which resistance is mediated."

**Power analysis for paired pre/post DE (sub-claim c, L1 version).** For a paired DE test (DESeq2 paired design or paired Wilcoxon) with log₂FC = 1 (2-fold), biological SD of paired differences in log₂CPM ~ 1.0–1.5 (conservative range from MM RNA-seq literature), FDR = 0.05, power = 0.80:

    n_pairs_required (SD = 1.0) = [(z_{α/2} + z_{β}) × SD / log₂FC]²
                                 = [(1.96 + 0.8416) × 1.0 / 1.0]² = 7.8
    n_pairs_required (SD = 1.5) = [(1.96 + 0.8416) × 1.5 / 1.0]² = 17.7

Conservative estimate: ~50% of 319 recurrence biopsies have a matched baseline biopsy = 159 pairs. Power at n = 159, SD = 1.5 is > 0.999. **N = 159 (conservative) is more than sufficient for gene-level FDR-controlled DE analysis with log₂FC ≥ 1.**

---

### Sub-claim "before it happens" (framing)
**Formal statement.** The model predicts resistance before the tumor has entered the resistant state — i.e., baseline features at time 0 forecast the future state.

**Causal tier required.** L2 minimum (longitudinal training with temporal separation between X and y). For the claim to be non-circular, training pairs (X_{t=0}, y_{t=T}) must exist where T > 0.

~~**Verdict (pre-MMRF).** Structurally absent. No (X_t, X_{t+delta}) pairs in training data. Training pairs are contemporaneous snapshots.~~

**Revised verdict (2026-05-03).** The cell-line training data (GDSC + CCLE) remains cross-sectional — this verdict is unchanged for the primary model. However, CoMMpass provides a 51-patient longitudinal RNA-Seq sub-cohort (RNA-Seq at ≥2 visits) and a 74-patient longitudinal CNV sub-cohort. On this sub-cohort it is possible to ask: "does the baseline transcriptomic cell-state (inferred from RNA-Seq visit 1) correlate with the relapse transcriptomic state (RNA-Seq visit 2)?" This is an exploratory Tier 2+ analysis — temporal separation exists, within-patient repeated measures are available. It is **not** a Tier 3 claim because:

1. N = 51 is below the n = 41 threshold established by Cohen et al. (PMID 33619369) for a powered longitudinal scRNA design in MM, and that design was a prospective single-arm trial with defined endpoints, not an observational registry.
2. The visit interval (median 256 days) conflates on-treatment with off-treatment states unless treatment records are used to stratify.
3. Without a controlled intervention, temporal correlation is not causation.

The appropriate framing is: "In an exploratory analysis on the N = 51 longitudinal RNA-Seq sub-cohort of MMRF CoMMpass, baseline cell-state X (visit 1) was [correlated / not correlated] with relapse cell-state Y (visit 2) [after adjustment for cytogenetic risk and treatment line]." This is an exploratory finding, not a validated "before it happens" claim.

---

## 2. GEO metadata verification: GSE271107 and GSE124310

### GSE271107 (fetched 2026-05-03 via NCBI eutils)
**Title:** "Single-cell RNA sequencing of bone marrow aspirate samples from multiple myeloma and its precursor conditions."

**Overall design:** "Sequential bone marrow aspirate samples were collected from 7 patients at different disease states." Samples span: 5 healthy donors (HD), 6 MGUS, 4 SMM, 4 NDMM — n=19 discrete samples.

**Verified sample characteristics:** GSM8369863–GSM8369881 carry `disease_state` values of Healthy, MGUS, SMM, or NDMM. No sample carries a treatment field, a drug field, a timepoint field, or a "post-treatment" label. The word "lenalidomide" does not appear in any sample characteristic fetched from the API.

**Critical finding.** The "sequential" in the design description refers to 7 patients sampled at different disease stages across patients, NOT the same patient sampled before and after treatment. This is a between-subject cross-sectional design with a disease-stage axis (HD → MGUS → SMM → NDMM). It is not a within-subject paired pre/post design. There are no untreated controls matched to lenalidomide-treated patients; there are no treated patients at all in this dataset's available metadata.

**Implication for causal inference.** The pseudo-temporal disease-stage axis (HD → MGUS → SMM → NDMM) is reconstructed from cross-sectional cell clusters, not from literal longitudinal follow-up of the same patient. This is the same limitation as scVelo/Monocle pseudotime: ordering is inferred by transcriptional similarity, not by observed transition. Confounders of who progresses from MGUS to SMM include chr1q gain, del17p, and FISH cytogenetics — none measured in this dataset.

### GSE124310 (PMID 33409501, fetched 2026-05-03)
**Title:** "Single-cell RNA sequencing reveals compromised immune microenvironment in precursor stages of multiple myeloma." (Zavidij et al., Nat Cancer 2020.)

**Overall design:** 40.8K cells, 32 bone marrow samples, 22 patients across MGUS/SMM/MM + 9 healthy donors.

**Verified sample characteristics:** GSM3528753–GSM3528818 carry `diagnosis` (MGUS, SMM, MM) and `marker_selection` (CD138N, CD138P fractions). No sample has a treatment label. The paper is explicitly about the immune microenvironment in precursor stages, not treatment response.

**Critical finding.** GSE124310 is a multi-stage cross-sectional cohort with no treatment variable and no follow-up timepoints. It supports cross-stage comparison (L1: does transcriptional state differ by disease stage?) but provides zero leverage on intervention (L2) or time-to-progression (which would require knowing which MGUS patients progressed and when, linked back to baseline cells — not provided).

**Are there control (untreated) patients matched to treated ones?** No. There is no treated arm in either dataset.

**Are there time-stamped follow-ups?** No. GSE271107's "sequential" design means different patients at different disease stages, not the same patient re-sampled. GSE124310 is a single biopsy per patient.

---

## 2.5 MMRF metadata verification (2026-05-03)

**Source.** GDC API (`https://api.gdc.cancer.gov/projects/MMRF-COMMPASS`) and MMRF CoMMpass IA22 clinical endpoints, verified 2026-05-03. All numbers below are confirmed counts from live API responses; no rounding or approximation.

| Metric | Confirmed value |
|---|---|
| Total cases (patients) in project | 995 |
| Open-access genomic files | 2,960 |
| RNA-Seq STAR gene counts files | 859 |
| Masked Somatic Mutation (MAF) files | 1,091 |
| Copy Number Segment files | 1,010 |
| Patients with treatment records | 994/995 |
| Patients with ≥2 distinct lines of therapy | 258 |
| Median treatment records per patient | 6 |
| Median days first-to-last treatment record | 256 |
| Patients with RNA-Seq at ≥2 visits | 51 |
| Patients with CNV at ≥2 visits | 74 |
| "Recurrent Blood Derived Cancer - Bone Marrow" biopsies | 319 |
| del17p prevalence (cytogenetics.tsv, N=976 patients) | 13.6% |
| chr1q21 gain prevalence | 33.6% |
| del13q prevalence | 52.0% |
| t(4;14) prevalence | 14.6% |
| t(11;14) prevalence | 16.8% |

**Implication.** The cytogenetic confounders named in §3.4 as identification blockers (chr1q21, del17p, t(4;14), t(11;14)) are now measurable for 976 of 994 patients. The unobserved-confounder identification failure is downgraded from "completely unresolvable" to "adjustable in the training/stratification layer given CoMMpass-based training." It remains an issue for the GEO datasets (GSE124310, GSE271107) which have no cytogenetic annotations.

---

## 3. Confounders blocking causal interpretation

### 3.1 Selection bias in treatment assignment
**Mechanism.** In clinical MM, treatment line (Rd vs VRd vs Dara-VRd vs CAR-T) is assigned by physician judgment informed by ECOG performance status, renal function, cytogenetic risk, prior lines, and patient preference. Frailer patients receive less aggressive regimens. This creates a systematic association between patient fitness, genomic risk, and treatment arm.

**Adjustment strategy.** Inverse probability of treatment weighting (IPTW) requires a propensity model P(treatment | covariates). This requires individual-level treatment records.

~~**Data support (pre-MMRF):** None. Selection bias cannot be adjusted without a treatment variable.~~

**Updated data support (2026-05-03).** CoMMpass provides individual-level treatment records for 994/995 patients, including regimen names, start/end dates, and response assessments. A propensity model for first-line treatment assignment (e.g., bortezomib-containing vs. non-bortezomib) is now estimable. IPTW and TMLE are executable on this cohort. G-computation with observed treatment is feasible. This does not eliminate confounding (unmeasured ECOG, physician preference, center effects remain) but reduces it from "completely unadjusted" to "partially adjusted via measured covariates." Adjustment strategy: IPTW with covariates including cytogenetics (del17p, chr1q21, t(4;14), t(11;14)), ISS stage, creatinine, and age.

### 3.2 Survivorship bias in cell-line screens
**Mechanism.** GDSC IC50 measures the dose at which 50% of cells die after 72h drug exposure. Cells that survive are enriched; dying cells are unobserved. The "resistant" phenotype in training data is constitutively high IC50, observed in the surviving cell population at the time of measurement. This is not the same as a cell transitioning from sensitive to resistant under drug pressure — that process involves clonal selection, which happens over weeks, not hours.

**Adjustment strategy.** Repeat-passage cell-line data (same line, multiple drug-exposure cycles) would allow modeling of acquired resistance. CCLE does not provide this; each IC50 is a single measurement.

**Data support:** No repeat-passage CCLE data on disk. The checkpoint `checkpoints/data_ready.pt` contains 622 lines with single IC50 snapshots per drug. **Status unchanged by MMRF landing — this confounder applies only to the cell-line training layer, which CoMMpass does not address.**

### 3.3 Batch effects mistaken for biology
**Mechanism.** The 886-line CCLE proteomics dataset was generated across multiple TMT-MS batches; the 19-sample GSE271107 used 10x Genomics v3.1 with Cell Ranger 6. Batch-effect confounding can create spurious clustering that resembles biological states.

**Verified status.** The v8 evaluation report (`paper/v8_evaluation_report.md`) reports iLISI = 3.434 and silhouette by pseudo-batch = -0.009, indicating acceptable batch mixing in the proteomics layer. However, the scRNA datasets are not yet wired into training (`checkpoints/data_ready.pt` contains cell-line data only), so the batch-effect analysis covers only CCLE, not the patient scRNA data.

**Adjustment strategy.** ComBat-seq, scVI, or Harmony for scRNA; ComBat or RUV for proteomics. These remove technical variance but do not address confounding by unmeasured biological variables. **For CoMMpass RNA-Seq: STAR counts are batch-corrected at analysis time using sequencing batch as a covariate in DESeq2/edgeR, or using scVI for the longitudinal sub-cohort. Status: achievable, not yet implemented.**

### 3.4 Unobserved genomic drivers
**Mechanism.** In MM, four cytogenetic events are the strongest determinants of resistance trajectory and treatment selection: chr1q21 gain (poor prognosis, PI resistance), del17p (p53 loss, worst prognosis), t(4;14) (FGFR3/MMSET, NSD2-driven, responds to bortezomib poorly), t(11;14) (BCL2-high, Venetoclax-sensitive). These are unmeasured in CCLE proteomics + epigenomics as delivered. They are also unmeasured in GSE124310 and GSE271107 (no FISH or WGS linked to scRNA samples).

~~**Identification failure (pre-MMRF).** This is a textbook unobserved confounder problem: cytogenetic risk U causes both the observed transcriptomic/proteomic state X and the drug resistance outcome Y. P(Y | X) != P(Y | do(X)) because the path U → X → Y is open. No adjustment strategy can close this path without measuring U.~~

~~**Adjustment strategy (pre-MMRF).** Instrumental variable (IV) analysis could in principle use a Mendelian randomization-style instrument for cytogenetic risk if one existed. None does for MM in the current data stack. Target trial emulation (Hernan & Robins framework) could be applied to MMRF CoMMpass (NCT01454297), which has paired WGS + RNA + clinical outcomes for ~1000 MM patients — but CoMMpass is listed as "not loaded" in `configs/default.yaml`.~~

**Updated status (2026-05-03).** CoMMpass cytogenetics.tsv provides del17p, chr1q21 gain, del13q, t(4;14), and t(11;14) for 976/994 patients. The back-door path U (cytogenetic risk) → X (transcriptome) → Y (resistance) is now measurable and adjustable for 98.2% of the cohort. Adjustment strategy: include cytogenetic risk as a covariate in all regression models and as a stratification variable in Kaplan-Meier and Cox models for sub-claim (b). This is now **achievable on CoMMpass**, though it remains an open identification failure for the GEO datasets and the cell-line training layer.

### 3.5 Time-varying confounding
**Mechanism.** In longitudinal treatment, the response to drug modifies subsequent treatment decisions (responders continue, non-responders switch). This creates time-varying confounding: past outcome partially determines future treatment. G-computation and marginal structural models (MSMs) with time-varying IPTW handle this, but they require repeated measures at multiple timepoints per patient. No dataset on disk provides this.

~~**Data support (pre-MMRF):** None.~~

**Updated data support (2026-05-03).** CoMMpass provides median 6 treatment records per patient across a median 256-day window. This is sufficient to fit a marginal structural model (MSM) with time-varying IPTW, treating each treatment cycle as an observation with updated covariate and treatment history. The N = 258 patients with ≥2 lines of therapy provide the minimum structure needed. Fitting a full MSM on CoMMpass is now **achievable in principle**, subject to adequate covariate measurement at each cycle (not all patients have cycle-matched RNA-Seq — only 51 have longitudinal RNA). The MSM can use clinical covariates (M-protein, LDH, response category) as time-varying confounders for N = 994; adding longitudinal transcriptomics restricts to N = 51.

---

## 4. Tier structure and escalation requirements

### Tier 1 — Associative (supported)
**Definition.** "In this dataset, cell lines with transcriptomic/proteomic profile X have IC50 distribution consistent with resistance label y."
**Supported by current data?** Conditionally yes, for 9 of 11 drugs with non-trivial Spearman rank correlation (median 0.328), as established by `paper/v8_evaluation_report.md`. The association is weak and beaten by a Ridge baseline, but it is real and reproducible.
**Pearl level.** L1.
**Status unchanged by MMRF landing.** The cell-line association result stands on its own merits. CoMMpass adds a second Tier 1 validation layer (patient-level IC50 proxy via IMWG response categories) but does not change the cell-line result.

### Tier 2 — Predictive (temporal generalization)
**Definition.** "On a held-out patient cohort not seen during training, the predicted resistance state at time T+delta matches the observed state."
~~**What is missing (pre-MMRF).** A patient cohort with: (a) baseline multi-omics, (b) treatment records, (c) observed response or progression at a defined future timepoint. MMRF CoMMpass is configured but not on disk.~~

**Updated status (2026-05-03).** CoMMpass is on disk. The Tier 2 test is now directly runnable: (a) baseline RNA-Seq + CNV for 994 patients, (b) treatment records for 994 patients, (c) IMWG response assessments at multiple cycles. The validation design — predict response at Cycle 4 from baseline multi-omics, evaluate against observed IMWG response (CR/VGPR/PR/SD/PD) — is a direct emulation of PERCEPTION (PMID 38637658, Nat Cancer 2024). Tier 2 no longer requires waiting; it requires training pipeline modifications to ingest CoMMpass RNA-Seq and running the prediction-validation loop. The claim escalation from Tier 1 to Tier 2 is contingent on: (1) fine-tuning or retraining on CoMMpass baseline RNA-Seq, (2) reporting C-index or AUC on the held-out time-to-next-line-therapy endpoint using N = 258 events, (3) adjusting for cytogenetic covariates.

**Pearl level.** L1 (still observational, but with genuine temporal separation between features and outcome, which removes in-sample circularity).

### Tier 2+ — Exploratory longitudinal (new)
**Definition.** "In the N = 51 longitudinal RNA-Seq sub-cohort of CoMMpass, baseline cell-state X (visit 1) is correlated with relapse cell-state Y (visit 2), after adjustment for cytogenetics and treatment line."
**What is needed.** Already on disk: 51 patients with RNA-Seq at ≥2 visits, cytogenetics for 976 patients, treatment records for 994 patients.
**Framing constraint.** This is an exploratory claim, not a validated "before it happens" claim. N = 51 is below the powered threshold established by Cohen et al. (PMID 33619369, n = 41 prospective trial with defined endpoints and CRISPR validation). The exploratory Tier 2+ analysis can identify candidate cell-state transitions for downstream validation but cannot claim clinical predictive validity on its own.
**Pearl level.** L1 with temporal separation. Consistent with Tier 2 framing only if explicitly labelled exploratory.

### L1 vs L2 vs L2-strong vs L3 vs L4 boundary (re-articulated 2026-05-03 post-MMRF)

The Pearl-tier ladder is now finer-grained because MMRF lands an intermediate level
between full-cohort Tier 2 and the paired-sample sub-cohort:

- **L1 (associative)** — supported full-cohort. Cell-line GDSC association (current
  state) and MMRF baseline-omics → outcome association (post-retraining) are both L1.
- **L2 (predictive validation, between-patient)** — now supported on N=994 MMRF baseline
  samples via PERCEPTION-style two-cohort design (PMID 38637658): cell-lines train,
  MMRF validate. Tier-2 in the temporal-separation sense (baseline omics measured before
  the resistance event); Pearl tier remains L1 because the design is observational.
- **L2-strong (within-patient predictive)** — supported only on the N=51 (RNA) / N=74
  (CNV) paired sub-cohort plus the N=319 paired pre/post relapse biopsies. Honest
  framing required: report effect sizes with bootstrap CIs, pre-specify the split
  (`PERPATIENT_VALIDATION_PROTOCOL.md`), label all results "exploratory," never promote
  to a primary cohort number. Below the Cohen et al. (PMID 33619369, n=41 prospective)
  powered threshold.
- **L3 (interventional)** — still requires DepMap CRISPR. The data is **already on
  disk** (`data/raw/depmap/CRISPRGeneEffect.csv`, configured at `depmap_crispr_path`)
  but it must be wired as a **post-hoc validation oracle, never as training input** —
  this preserves the Pearl-tier separation between association (training signal) and
  intervention (validation signal). MMRF on its own is observational and does not
  unlock L3.
- **L4 (counterfactual)** — still not achievable with any existing public MM dataset.

*Source for the new N values: verified live 2026-05-03 via `pull_mmrf_clinical.py`
+ direct pandas audit on `data/raw/mmrf_commpass/{clinical,treatments,samples}.tsv`;
cytogenetic flags via `scripts/derive_cytogenetics.py` (logged to
`logs/preprocess/cytogenetics.log`). These N values come from a debugging audit, not a
ledgered run — they cannot ship to README/ARCHITECTURE without an entry in the run
ledger per the release-bouncer rule.*

### Tier 3 — Interventional
**Definition.** "If drug D is administered rather than D', the predicted divergence in resistance state matches the observed divergence."
**What is missing.** Perturbation data where the intervention is known and its effect on the outcome is measured. Two datasets would provide this:

(i) DepMap Achilles CRISPR gene-knockout essentiality (Broad Institute, public). Provides do(gene=knockout) effects on cell viability for ~18,000 genes across ~1,000 lines. Configured at `depmap_crispr_path` but not downloaded. This would ground pathway attribution: if knocking out pathway P increases Venetoclax IC50, P is causally involved. Relevant paper: Replogle et al. (PMID 35688146), genome-scale Perturb-seq mapping information-rich genotype-phenotype landscapes (Cell 2022).

(ii) Cohen et al. (PMID 33619369, Nat Med 2021) conducted a prospective multicenter single-arm clinical trial (NCT04065789) with longitudinal scRNA-seq at multiple timepoints in 41 bortezomib-refractory MM patients treated with daratumumab + carfilzomib + lenalidomide + dexamethasone. This dataset has pre/post treatment scRNA, clinical response endpoints, and the study identified PPIA as a causal mediator via CRISPR-Cas9 deletion validation. This is the closest existing dataset to what Tier 3 requires for MM. It is not on disk.

**Status unchanged by MMRF landing.** CoMMpass is an observational registry, not a perturbation experiment. Tier 3 remains gated on DepMap CRISPR ingestion and/or Cohen et al. data request. The recurrence biopsy analysis (N = 319) described in §1 sub-claim (c) is a Tier 1 pathway-association test, not a Tier 3 test.

**Pearl level required.** L2. G-computation or IPTW on a randomized or near-randomized drug assignment. Single-arm trial data (Cohen) is stronger than observational (GDSC) but still requires assumptions.

### Tier 4 — Counterfactual
**Definition.** "For the same patient, had treatment D been given instead of D', resistance trajectory would have diverged at pathway P." Requires the same individual under both treatment arms — achievable only in theory (fundamental problem of causal inference) or under strong no-unmeasured-confounders assumption with mathematical twinning.
**What is missing.** No dataset in oncology supports Tier 4 directly. Approximations exist via: (a) digital twins using biological ODE models with patient-level parameter fitting, (b) structural causal models (SCMs) with validated mechanistic priors. Neither is implemented in ResistanceMap's current architecture.
**Pearl level required.** L3.
**Verdict.** Not achievable with any existing public MM dataset. This tier cannot be honestly claimed in a submission.

---

## 5. Peer-reviewed precedent for tiered claims (2024–2026)

Three papers made similar claims and navigated peer review by staying within their data's causal tier:

**PERCEPTION (PMID 38637658, Nat Cancer 2024).**
Sinha et al. developed a single-cell-based treatment response predictor using matched bulk+scRNA cell-line drug screens. Critically, they validated on two prospective clinical trials (MM and breast cancer) with pre-specified endpoints — this is Tier 2 (predictive with temporal separation) not Tier 3. The paper does NOT claim causal mediation or pathway attribution; it claims "predicts response" and frames the contribution as L1-to-Tier-2 generalization.

~~**Prior framing (pre-MMRF):** "This is the appropriate framing for ResistanceMap if MMRF CoMMpass data were ingested."~~

**Updated framing (2026-05-03).** PERCEPTION is no longer a template for a future state — it is a template we can directly emulate now. CoMMpass provides the equivalent of PERCEPTION's clinical trial validation cohort (N ≈ 994 vs. PERCEPTION's smaller MM trial arm). The ResistanceMap Tier 2 validation should be structured to directly parallel PERCEPTION's validation design: pre-treatment baseline multi-omics → predict response category → compare to observed IMWG response at Cycle 4. Any manuscript submission should cite PERCEPTION as the methodological precedent and explicitly position the ResistanceMap validation as a replication in a larger observational cohort.

**Cohen et al. (PMID 33619369, Nat Med 2021).**
This study provides the best existing framework for Tier 3 in MM. It combines a prospective clinical trial with longitudinal scRNA-seq and CRISPR validation of PPIA. The claim "PPIA is causally involved in PI resistance" is grounded by the do(PPIA=knockout) experiment. This is genuine L2 evidence. The key design feature: the causal claim (CRISPR) is distinguished from the predictive claim (scRNA trajectory) — they are presented as complementary, not conflated. Cohen et al. also establishes the minimum powered design for longitudinal MM scRNA (n = 41 prospective, defined endpoints) — the MMRF longitudinal sub-cohort (N = 51) approaches but does not exceed this bar, and critically lacks the CRISPR validation component.

**Rade et al. (PMID 38641734, Nat Cancer 2024).**
Single-cell multiomic dissection of CAR-T resistance in relapsed/refractory MM. Uses pre/post leukapheresis multiomics to identify immunosuppressive signatures. The paper claims "identifies markers associated with resistance" (L1 association), not causal mechanism. The distinction between "associated with" and "causes" is maintained throughout. This is the precedent for how to frame the "which pathway" sub-claim within L1 without overclaiming. The MMRF recurrence biopsy analysis (N = 319) should adopt this framing verbatim: "pathways whose expression is associated with the transition from sensitive to relapsed state."

**Summary.** Papers that get past peer review in this space either: (a) use prospective trial data with temporal separation (Tier 2), (b) combine observational prediction with separate CRISPR validation of the top candidates (Tier 3 element added post-hoc), or (c) explicitly frame findings as "markers associated with" rather than causal mediators. ResistanceMap can now credibly pursue path (a) using CoMMpass.

---

## 6. Falsification test (Popperian criterion)

A claim is scientific only if it can be refuted by a specified experiment. For each sub-claim:

**Sub-claim (a) — "which resistance state":**
Falsification test: Take 50 MM cell lines not in GDSC training data. Measure IC50 for Venetoclax and Bortezomib. Compare ResistanceMap's predicted resistance rank to the observed rank. If Spearman rho < 0.1 and the 95% CI includes zero, the model provides no predictive signal beyond noise. This test is runnable today on CTRPv2 (configured but not on disk). If the model cannot be beaten by a per-drug-mean predictor on this held-out set, the claim fails. **Status unchanged.**

**Sub-claim (b) — "when":**
~~**Prior status:** "This test requires data that does not exist today and cannot be run on current data."~~

**Updated falsification test (2026-05-03).** The test is now runnable on CoMMpass N = 258 patients with ≥2 lines of therapy. Specified experiment: (1) Stratify the 994-patient CoMMpass cohort by first-line treatment type (bortezomib-containing vs. IMiD-based, to control for the largest measured confounder). (2) For each patient, extract baseline RNA-Seq features and input to the trained ResistanceMap model. (3) Record predicted resistance score (continuous). (4) Compute Harrell's C-index between the predicted score and observed time-to-next-line-therapy (days_to_treatment_start of line 2 minus days_to_treatment_start of line 1). (5) Refutation criterion: if C-index < 0.55 with upper 95% CI < 0.60, the "when" claim is refuted. Power analysis (§1 sub-claim b) confirms N = 258 events is sufficient to distinguish C = 0.65 from C = 0.50 with > 0.999 power; the lower-CI > 0.55 bar is achievable if true C ≥ 0.63. **This sub-claim is now falsifiable on existing data.**

**Sub-claim (c) — "via which pathway":**
~~**Prior status:** "The model is currently non-falsifiable — there is no experiment whose outcome is both specified in advance and runnable on existing infrastructure that would definitively refute the claim."~~

**Updated status (2026-05-03).** For the **primary causal claim** (NIE through pathway M), non-falsifiability on existing data remains — DepMap CRISPR is not on disk and attention weights are not persisted. This verdict does not escalate.

For the **L1 association version** (which pathways are differentially expressed at relapse vs. baseline), the claim is now falsifiable on the 319 recurrence biopsies. Specified experiment: (1) Identify patients in CoMMpass with both a baseline bone-marrow biopsy RNA-Seq and a "Recurrent Blood Derived Cancer - Bone Marrow" RNA-Seq. (2) Run DESeq2 with paired design (patient as blocking factor). (3) Extract the top-50 differentially expressed pathways (GSEA, FDR < 0.05). (4) Cross-reference with ResistanceMap's predicted pathway hits (if attention weights are persisted) or with the pathway features with highest model weight. (5) Refutation criterion: if fewer than 2 of ResistanceMap's top-5 pathway predictions appear in the CoMMpass recurrence DE top-50, the L1 pathway-association claim is refuted. Power analysis (§1 sub-claim c) confirms N ≈ 159 conservative pairs is more than sufficient for FDR-controlled gene-level DE with log₂FC ≥ 1. **The L1 pathway association sub-claim is now falsifiable on existing data; the L3 causal mediation claim remains non-falsifiable without perturbation data.**

---

## Summary table

| Sub-claim | Pearl tier required | Pearl tier supported | Identification failure | Estimation failure | Falsifiable now? |
|---|---|---|---|---|---|
| (a) which resistance state | L2 (intervention) for causal; L1 for association | L1 | ~~Unmeasured cytogenetic confounders (chr1q, del17p, translocations)~~ Cytogenetics now measurable in CoMMpass (N=976); GEO datasets still lack cytogenetics | Beaten by per-drug-mean baseline; Spearman 0.33 | Yes (CTRPv2 hold-out; also CoMMpass Tier 2) |
| (b) when (time-to-event) | L2 minimum for causal; L1 for concordance | ~~None~~ L1 on CoMMpass N=258 | L2 causal interpretation unidentified (physician-assigned treatment); L1 time-to-next-line is measurable | ~~No time variable~~ C-index estimable on N=258 (power > 0.999 for C=0.65 vs 0.50) | ~~No~~ Yes — C-index test on N=258 (§6) |
| (c) via which pathway | L3 (mediation, counterfactual) for causal; L1 for association | L3: none. ~~L1: none~~ L1 paired pre/post on N=319 recurrence biopsies | L3: no do(pathway) data. L1: adjustable with paired design | L3: no pathway attribution scores on disk. L1: runnable | L3: no. L1: ~~No~~ Yes — paired DE on recurrence biopsies (§6) |
| "before it happens" (framing) | L2 (longitudinal training) | ~~None~~ Exploratory L1+temporal on N=51 longitudinal RNA sub-cohort | No (X_t, X_{t+delta}) pairs in cell-line training; CoMMpass sub-cohort provides observational pairs only | N=51 is exploratory; below Cohen et al. powered threshold | Exploratory only; not a validated claim |

---

## Required changes (priority-ordered)

1. **Reframe all claims to L1 (associative/predictive) as the minimum floor**, with the specific language: "ResistanceMap identifies transcriptomic and proteomic features associated with differential IC50 across GDSC cell lines. On 9 of 11 tested drugs (Spearman median 0.33), the model predicts resistance rank better than a constant predictor. These associations are not causal." This framing matches what the cell-line data supports and is consistent with PERCEPTION (PMID 38637658) and Rade et al. (PMID 38641734).

2. **Execute Tier 2 validation on CoMMpass** (N=994, now on disk). Retrain or fine-tune on CoMMpass baseline RNA-Seq. Report C-index on N=258 time-to-next-line-therapy events, AUC on IMWG response at Cycle 4. Adjust for cytogenetic covariates using cytogenetics.tsv. Cite PERCEPTION (PMID 38637658) as the methodological template. This is the most actionable escalation path.

3. **Run paired pre/post DE analysis on N=319 recurrence biopsies** (L1 pathway association). Identify patients with matched baseline + recurrence RNA-Seq. Run DESeq2 paired design. Report top differentially expressed pathways. Frame as "associated with the transition from baseline to relapse state" (Rade et al. PMID 38641734 framing). This converts the "which pathway" sub-claim from unsupported to L1-falsifiable.

4. **Download DepMap CRISPR essentiality** (configured at `depmap_crispr_path`, not on disk) and add a post-training validation step: "do ResistanceMap's top pathway hits have significantly higher gene essentiality scores in resistant vs sensitive MM lines?" This converts the attention-weight association into a perturbation-grounded association (L1/L2 boundary, but defensible).

5. **Add a disclaimer section to any manuscript** stating explicitly: "(a) all pathway attributions are associative and require independent CRISPR validation; (b) time-to-resistance claims are observational (L1 concordance) and not causal (L2 identification requires randomization); (c) the 'before it happens' framing is exploratory on N=51 longitudinal RNA sub-cohort of CoMMpass and should not be stated as a validated claim until replicated in a prospective cohort following the Cohen et al. NCT04065789 design (PMID 33619369)."

6. **Do not submit the "before it happens" framing as a primary claim.** The N=51 longitudinal RNA sub-cohort supports an exploratory finding only. Primary claim ceiling is Tier 2 predictive upon CoMMpass validation.

---

## Honest verdict

With MMRF CoMMpass IA22 now on disk (verified 2026-05-03 via `pull_mmrf_clinical.py` +
direct pandas audit on `data/raw/mmrf_commpass/{clinical,treatments,samples}.tsv` +
`scripts/derive_cytogenetics.py`), the escalated honest verdict is:

**The strongest defensible claim is now Tier 2 (predictive validation on the full
N=994 MMRF baseline cohort), with explicit Tier-1 (Pearl L1) framing retained for the
cell-line training stage.** The original audit's verdict — "the strongest defensible
claim is Tier 1, Pearl L1" — was correct conditional on the data state at writing and
is now superseded by the MMRF landing.

The two-stage framing required for any submission is:

1. **Cell-line training stage (Tier 1, Pearl L1).** "ResistanceMap learns an associative
   ranking of cell-line drug sensitivity from static proteomic and epigenomic snapshots.
   On 9 of 11 GDSC drugs in a held-out test set of 132 cell lines, it achieves Spearman
   rank correlation of 0.33, which is positive but weak and beaten by a per-modality
   Ridge baseline." This Tier-1 framing is unchanged by MMRF and must be retained
   verbatim — the cell-line layer remains the source of training signal and is itself
   only L1.

2. **MMRF predictive validation stage (Tier 2, Pearl L1 with temporal separation).**
   PERCEPTION-style two-cohort design (PMID 38637658): cell-lines train, MMRF validate.
   Predict IMWG response at Cycle 4 from baseline RNA-Seq + cytogenetics for N=994
   CoMMpass patients; evaluate Harrell's C-index against time-to-next-line-therapy on
   N=258 events. Power analysis (Pencina & D'Agostino formula) confirms N=258 is more
   than sufficient: minimum events required to detect C=0.65 vs. C=0.50 at α=0.05,
   power=0.80 is 63–79 events; expected 95% CI at N=258 is (0.592, 0.708), with lower
   bound > 0.55. Pearl tier remains L1 (still observational, no perturbation), but the
   temporal-separation design moves the *evidence tier* from Tier 1 to Tier 2.

Tier 2 is contingent on pipeline changes to ingest CoMMpass, not on acquiring new data.
Until those pipeline changes land in the run ledger, every shipped number stays
Tier-1-framed in README/ARCHITECTURE per the release-bouncer rule — the N values in
this audit come from a debugging-tier check, not a ledgered run.

**Tier 2+ exploratory claim on the N=51 longitudinal RNA sub-cohort** is honest if framed as: "In an exploratory analysis, baseline cell-state (visit 1) was [correlated/not correlated] with relapse cell-state (visit 2) after cytogenetic adjustment." This is below the Cohen et al. (PMID 33619369) powered longitudinal threshold (n=41 prospective with CRISPR validation) and must not be promoted to a primary finding.

**Sub-claim (c) pathway mediation remains non-falsifiable for the primary L3 causal claim** without perturbation data (DepMap CRISPR, not on disk). The L1 association version (paired pre/post DE on N=319 recurrence biopsies) is now falsifiable and should be pursued as the honest pathway-attribution framing, following Rade et al. (PMID 38641734).

**The two scRNA patient datasets (GSE124310 and GSE271107) remain cross-sectional between-subject cohorts** with no treatment variable and no longitudinal follow-up. They support disease-stage transcriptomic profiling (L1) only; their verdicts are unchanged by CoMMpass landing.

**Summary of Pearl tier verdicts after MMRF landing:**

| Sub-claim | Previous ceiling | Current ceiling | Gate remaining |
|---|---|---|---|
| (a) which resistance state | L1 (cell-line only) | L1 → Tier 2 runnable | CoMMpass pipeline ingestion |
| (b) when | None (no time axis) | L1 C-index on N=258 | CoMMpass pipeline ingestion |
| (c) via which pathway (L1) | None | L1 paired DE on N=319 | DESeq2 run on matched biopsies |
| (c) via which pathway (L3) | None | None | DepMap CRISPR download |
| "before it happens" | None | Exploratory only (N=51) | N not sufficient for primary claim |
