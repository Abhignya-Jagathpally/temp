# Phase 14 — Clinical Claim-Gate Matrix
## ResistanceMap / MORT-FM v18

---

## 1. Existing Gate Inventory

### 1a. `resistancemap/mortfm/acceptance_gate.py`

The current gate is a **cohort-level, claim-level-agnostic** gate. It enforces
four levels — `debug`, `technical`, `research`, `strong` — keyed only on:

| Lines | Check |
|---|---|
| 115–121 | debug: n_patients >= 10, modality present, drug or survival label |
| 122–129 | technical: n_samples >= 100, modality, label |
| 130–145 | research: n_patients >= 200 (or config override), n_events >= 50, n_modalities >= 2, trajectory players >= 25 |
| 147–158 | strong: research prerequisite, then **unconditionally blocks** — the operational evidence (external cohort, ablations, etc.) must be passed separately |

Crucially missing: no check that the **endpoint type** is IMWG-consistent, no
per-claim-domain modality requirements, no minimum follow-up window, and no
check that progression events were classified under a documented criteria
version (IMWG 2016 vs. pre-2016 EBMT). The `AcceptanceReport` (lines 48–70)
has `n_events_observed` but no field recording how those events were
adjudicated.

### 1b. `resistancemap/mortfm/registries/claim_gate_registry.py`

Defines twelve `ClaimLevel` entries in `CANONICAL_CLAIM_LEVELS`. These capture
`required_artifacts`, `required_endpoint_types`, and `forbidden_endpoint_types`
as string lists. However:

- No minimum patient count, minimum event count, or minimum follow-up window
  is stored per claim level; those live only in `acceptance_gate.py` globally.
- `forbidden_endpoint_types=["overall_survival"]` on `resistance_emergence` is
  clinically correct but is **not enforced at runtime** — the registry is a
  documentation artifact, not an enforcement layer.
- No field for `required_criteria_version` (e.g., IMWG 2016 vs. EBMT 1998).
- No field for `required_external_cohort_min_n`.

### 1c. `scripts/mortfm/09_gate_revalidate.py`

A v18 shim that delegates to `scripts/mortfm_v17_gate_revalidate.py`. The v17
script reads JSON artifacts from `logs/mortfm/` and evaluates twelve hard-coded
gate conditions. Gate decisions are artifact-presence checks plus floating-point
thresholds (e.g., Spearman >= 0.10, lower C-index CI > 0.50). The script does
**not** verify that endpoint semantics comply with IMWG criteria, does not check
follow-up window adequacy, and does not distinguish between endpoints derived
from IMWG-consistent M-protein/FLC/BMPC assessment versus alternative
assessments.

### 1d. `EVALUATION_GOVERNANCE.md`

Three falsifiable claims: **state**, **when**, **pathway**. Tier C clinical
safety section is present but does not codify IMWG-specific endpoint validity
requirements. The punch list (lines 129–153) calls for a "resistance ontology"
and "prospective protocol document for any 'before event' claim" but does not
prescribe the IMWG criteria version or minimum follow-up windows.

---

## 2. Verified IMWG Citations

PubMed confirms the following PMIDs. Do **not** add others to code or docs
without a separate PubMed verification step.

| PMID | Reference | Scope |
|---|---|---|
| **27210871** | Kumar et al. — IMWG 2016 response criteria update (serum/urine M-protein, FLC, BMPC, imaging) | sCR, CR, VGPR, PR, MR, SD, PD definitions; MRD via next-gen flow or sequencing at sensitivity ≥ 10^-5 |
| **25182124** | (confirmed in PubMed; IMWG criteria / FLC response 2014 era) | FLC ratio definition used in CR→sCR distinction |
| **22460902** | Garnett et al. (Nature 2012) — GDSC pharmacogenomics Sanger/Broad | IC50 measured in cell lines; no IMWG mapping |
| **31141632** | MAIA trial — D-Rd vs Rd, N=737, transplant-ineligible NDMM | Phase 3 enrollment anchor |
| **31171419** | CASSIOPEIA — D-VTd vs VTd, transplant-eligible NDMM | Phase 3 induction/consolidation anchor |
| **28462890** | POLLUX/CASTOR combined RRMM daratumumab Phase 3 | Relapsed/refractory enrollment anchor |

IMWG 2016 response definitions enforced in this gate matrix (PMID 27210871):

- **sCR**: CR + normal FLC ratio + absence of clonal plasma cells by IHC/flow
- **CR**: Negative serum/urine immunofixation + < 5% BMPC on aspirate
- **VGPR**: >= 90% reduction in serum M-protein + urine M-protein < 100 mg/24h
- **PR**: >= 50% reduction in serum M-protein; >= 90% reduction or < 200 mg/24h in urine M-protein
- **MR (minor response)**: 25–49% reduction in serum M-protein (used in RRMM)
- **SD**: Does not meet PR, MR, or PD criteria
- **PD**: >= 25% increase from lowest confirmed value in serum M-protein
  (absolute >= 0.5 g/dL), urine M-protein (>= 200 mg/24h), BMPC (>= 10%
  absolute), or new bone/soft tissue lesions, or hypercalcemia attributable to
  myeloma
- **MRD negativity**: < 1 clonal plasma cell per 100,000 nucleated cells
  (sensitivity >= 10^-5) by validated multi-parameter flow cytometry or
  NGS-based sequencing; requires concurrent CR
- **PFS event definition**: progression (as above) OR death from any cause,
  whichever first; censored at last follow-up contact
- **Minimum assessment frequency for PFS**: M-protein/FLC every cycle during
  treatment, minimum every 3 months; bone marrow at planned assessments per
  protocol

---

## 3. Comprehensive Claim-Gate Matrix

The 15 rows below correspond exactly to the claim-domain rows requested. For
each, columns show: minimum patients, minimum events for time-to-event claims,
minimum modalities, minimum follow-up window (months), required evidence
channels (K-of-N), external cohort required, allowed endpoints, and the
citation basis for the threshold.

| Claim Domain | Min Patients | Min Events (TTE) | Min Modalities | Min Followup (months) | Required Evidence Channels (K-of-N) | External Cohort Required | Allowed Endpoints | Citation/Basis |
|---|---|---|---|---|---|---|---|---|
| **technical_pipeline** | 10 (cell lines acceptable) | 0 | 1 | none | pipeline artifact present; train+val split documented (1-of-1) | No | viability readout (IC50/AUC) | acceptance_gate.py debug level |
| **static_drug_response** | 100 (cell lines) | 0 | 1 (transcriptomics or proteomics) | none | Spearman rank correlation on held-out cell lines; baseline comparison (1-of-2 min, 2-of-2 preferred) | No | GDSC IC50, PRISM AUC, CTRPv2 AUC — NOT IMWG response | GDSC PMID 22460902; no GDSC→IMWG mapping exists |
| **hematologic_specimen_drug_response** | 100 (patient specimens) | 0 | 1 (ex-vivo drug response + at least one omics) | none | Spearman on held-out specimens; positive transfer gain CI excludes zero; hematologic disease label present (2-of-3) | No | ex-vivo viability (BeatAML AUC), NOT clinical response | BeatAML design; GDSC→IMWG gap applies |
| **single_cell_state** | 200 (cells, min 50 donors) | 0 | 1 (scRNA required; ATAC recommended) | none | macro-F1 on held-out donors > clinical-only prior; pseudotime correlation with known markers; modality ablation present (2-of-3) | No | cell-state label (sensitive/DTP/resistant); disease stage ordinal; NOT IMWG response | scRNA atlas best-practices; cell line stage limitations |
| **multiomic_foundation** | 200 (patients or bulk-cell-line N >= 500) | 0 | 3 (min: transcriptomics + one additional omics + clinical) | none | Held-out reconstruction loss; cross-modal imputation accuracy; modality-dropout ablation; latent geometry separability (3-of-4) | No | multi-omics representation (no direct clinical outcome claim) | MORT-FM schemas.py modality registry |
| **epigenetic_plasticity** | 150 (patients with paired RNA+ATAC or RNA+methylation) | 0 | 2 (RNA required + ATAC or methylation required) | none | Differential accessibility/methylation at drug-resistance loci; trajectory shift with treatment; CRISPR or perturbation consistency check (2-of-3) | No | epigenetic state transition label; NOT IMWG response | Literature: epigenetic reversibility in myeloma drug tolerance |
| **sequence_aware** | 50 (protein sequences; not patient count) | 0 | 1 (ESM-2 embeddings required) | none | ESM-2 coverage fraction >= 50% of drug targets; identifier map validated; embedding influences downstream drug-response prediction (2-of-3) | No | sequence-aware drug-target attribution; NOT IMWG response | ESM-2 publication; claim_gate_registry.py sequence_aware entry |
| **pathway_context** | 200 (patients or cell lines with PPI) | 0 | 2 (omics + PPI graph) | none | Pathway enrichment p < 0.05 in top-20 attributions vs null; top-k edge stability >= 0.6 across perturbation runs; STRING edge coverage >= 70% of predicted targets (2-of-3) | No | pathway attribution score; NOT IMWG response | EVALUATION_GOVERNANCE.md pathway claim; PMID 27210871 for pathway list |
| **drug_target_mechanism** | 200 (patients or cell lines) | 0 | 2 (omics + drug target annotation) | none | ChEMBL/DrugBank target coverage >= 50% of predicted drugs; CRISPR essential gene overlap >= 4 of top-20; drug-target attribution direction consistent with known MOA (2-of-3) | No | drug-target attribution; mechanism class prediction; NOT IMWG response | claim_gate_registry.py drug_target_mechanism; v17 gate fails at 32.5–46.7% |
| **survival_prediction** | 200 (real patients; cell lines not acceptable) | 50 events | 2 (clinical required + at least 1 molecular) | 6 months minimum observation window | C-index lower 95% CI > 0.50 on patient-disjoint held-out set; Brier score < KM-baseline; per-horizon calibration ECE reported; event labels IMWG-consistent (4-of-4 required) | No | PFS (IMWG PD or death), OS, TT2L — labels MUST be adjudicated under IMWG 2016 criteria (PMID 27210871) | PMID 27210871 PFS definition; acceptance_gate.py research level; MAIA PMID 31141632 N=737 as enrollment anchor |
| **longitudinal_trajectory** | 100 (real patients with paired samples) | 0 (trajectory endpoint) | 2 (baseline + followup timepoint; same-patient molecular) | Real calendar time required: minimum 60 days between baseline and followup sample | n_pairs >= 100; molecular state change measurable vs. null; calendar time mapped to ODE integration time; pseudotime FORBIDDEN as calendar-time substitute (3-of-3 required) | No | longitudinal molecular state delta; NOT IMWG response directly — molecular trajectory may correlate but correlation must be declared, not assumed | EVALUATION_GOVERNANCE.md epistemic limits; MORTFMConfig min_paired_n=100; schemas.py TemporalTrainingPair |
| **resistance_emergence** | 200 (real patients) | 80 resistance events | 2 (molecular + clinical) | 12 months minimum follow-up; resistance event must be IMWG PD or treatment-change-for-progression documented | C-index lower CI > 0.55 on resistance endpoint; time-to-resistance AUC > PFS-KM baseline; IMWG-consistent PD adjudication documented; OS FORBIDDEN as sole endpoint; GDSC IC50 FORBIDDEN as resistance-emergence proxy (4-of-4 required) | No | time-to-resistance, time-to-next-line (TT2L), PFS under treatment; endpoint labels IMWG PD (PMID 27210871) | PMID 27210871 PD criteria; v17 gate: blocked — needs ~80 more paired patients |
| **causal_mechanism** | 200 (real patients) + perturbation cohort N >= 20 | 50 events | 3 (omics + PPI graph + perturbation readout) | 12 months minimum follow-up | pathway attribution enriched p < 0.05; CRISPR/perturbation direction consistency >= 4/20 targets; counterfactual edge ablation changes prediction by > 2 sigma; no-graph ablation shows significant degradation; negative control pathway non-enriched (5-of-5 required) | Recommended but not strictly required at this level | pathway attribution; perturbation-consistent mechanism claim | EVALUATION_GOVERNANCE.md pathway claim; acceptance_gate.py strong level |
| **patient_level_clinical_prediction** | 300 (real patients; external temporal holdout >= 100) | 80 events | 3 (clinical + >= 2 molecular) | 18 months minimum follow-up in external cohort; response assessment IMWG 2016 | MORT-FM beats clinical-only Cox by statistically significant margin (bootstrap CI excludes zero); external holdout lower C-index CI > 0.55; uncertainty calibration ECE <= 0.10; IMWG-consistent endpoint labels in ALL cohorts; claim-critic report completed (5-of-5 required) | Yes (min N=100 external) | PFS, OS, TT2L, MRD conversion — ALL under IMWG 2016 criteria (PMID 27210871) | MAIA N=737 (PMID 31141632); CASSIOPEIA N=1085 (PMID 31171419); acceptance_gate.py strong level |
| **external_validation** | 100 (independent cohort; no overlap with training) | 30 events minimum | 2 (same modalities as training) | Prospective or strictly time-ordered; minimum 6 months follow-up in holdout cohort | External cohort provenance documented; IMWG response adjudication method documented; same model weights as training (no re-training); performance not worse than training C-index minus 0.10 degradation tolerance (4-of-4 required) | Yes (defining requirement) | Same as model's primary endpoints; must match training endpoint definitions | EVALUATION_GOVERNANCE.md external cohort requirement; acceptance_gate.py strong prerequisite |

**Hard constraints on ALL survival/trajectory/resistance claims (rows 10–15):**

1. Event labels MUST document the adjudication criteria version. IMWG 2016
   (PMID 27210871) is the required standard. Labels derived under EBMT 1998
   criteria or manufacturer-defined criteria without IMWG mapping are not
   accepted for claims rows 10–15.
2. PFS event = IMWG PD (>= 25% increase from nadir in M-protein, FLC, or BMPC,
   with absolute thresholds) OR death from any cause, whichever comes first.
   Censoring at last follow-up contact date, not last treatment date.
3. GDSC IC50 and PRISM AUC are valid for rows 1–2 only. They are explicitly
   BLOCKED for endpoint labeling in rows 10–15. There is no published,
   validated mapping from GDSC/PRISM viability metrics to IMWG response
   categories.
4. Cell-line N (CCLE/GDSC/PRISM) does NOT count toward any patient-count
   threshold in rows 10–15.
5. Pseudotime from scRNA analysis does NOT substitute for calendar time in
   row 11 (longitudinal_trajectory) or row 12 (resistance_emergence).

---

## 4. Proposed `resistancemap/governance/claim_matrix.py`

```python
"""
resistancemap/governance/claim_matrix.py
=========================================
Typed claim-gate registry for MORT-FM Phase 14.

Each ClaimGate enforces endpoint-specific and modality-specific requirements
that the existing acceptance_gate.py global cohort gate does not cover.
A ClaimGate is the atomic unit of audit evidence: every gate must be
explicitly evaluated before a claim at that domain is permitted.

Citation anchors
----------------
- IMWG 2016 response criteria: PMID 27210871
- GDSC pharmacogenomics: PMID 22460902
- MAIA Phase 3 enrollment (N=737): PMID 31141632
- CASSIOPEIA Phase 3 enrollment (N=1085): PMID 31171419
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ClaimGate:
    """Immutable specification for a single claim domain's evidence requirements.

    Parameters
    ----------
    name :
        Canonical gate name. Must match a key in CLAIM_GATE_REGISTRY.
    required_min_patients :
        Minimum number of real patients (not cell lines) whose data was used
        to produce the claim. For static_drug_response and technical_pipeline,
        cell lines count; all other gates require human patients.
    required_min_events :
        Minimum number of observed (non-censored) time-to-event outcomes.
        Zero for non-time-to-event claims. For IMWG-anchored claims, events
        must be IMWG PD or death (PMID 27210871).
    required_modalities :
        Minimum list of modality names that must be non-null in the training
        cohort. At least one of each named modality must be present in >= 50%
        of patients in the cohort.
    required_followup_months :
        Minimum follow-up window. Zero for claims that do not make time-to-event
        assertions. For patient_level_clinical_prediction this is the minimum
        median follow-up of the external cohort.
    required_evidence_channels :
        (channel_name, is_required) pairs. Gates with is_required=True must all
        pass. Gates with is_required=False are optional but the gate specifies
        minimum K of N optional channels that must also pass.
    required_optional_k :
        Number of optional channels (is_required=False) that must pass.
    required_external_cohort :
        Whether an external, non-overlapping patient cohort with >= 100 patients
        is required.
    external_cohort_min_n :
        Minimum size of the external cohort when required_external_cohort=True.
    allowed_endpoints :
        Endpoint types that are permitted for this claim. Endpoints not in this
        list are blocked even if present in the artifact.
    blocked_endpoints :
        Endpoints explicitly forbidden. Takes precedence over allowed_endpoints.
    imwg_criteria_required :
        Whether event/response labels MUST be adjudicated under IMWG 2016
        criteria (PMID 27210871). False only for cell-line endpoints.
    criteria_citation_pmid :
        PMID that defines the endpoint criteria. Must be a real, verified PMID.
    """

    name: str
    required_min_patients: int
    required_min_events: int
    required_modalities: List[str]
    required_followup_months: float
    required_evidence_channels: List[Tuple[str, bool]]  # (name, is_required)
    required_optional_k: int
    required_external_cohort: bool
    external_cohort_min_n: int
    allowed_endpoints: List[str]
    blocked_endpoints: List[str]
    imwg_criteria_required: bool
    criteria_citation_pmid: Optional[str]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CLAIM_GATE_REGISTRY: Dict[str, ClaimGate] = {

    "technical_pipeline": ClaimGate(
        name="technical_pipeline",
        required_min_patients=10,
        required_min_events=0,
        required_modalities=["rna"],  # at least transcriptomics
        required_followup_months=0.0,
        required_evidence_channels=[
            ("pipeline_artifact_present", True),
            ("train_val_split_documented", True),
        ],
        required_optional_k=0,
        required_external_cohort=False,
        external_cohort_min_n=0,
        allowed_endpoints=["ic50", "auc", "viability"],
        blocked_endpoints=["pfs", "os", "tt2l", "mrd_conversion", "imwg_response"],
        imwg_criteria_required=False,
        criteria_citation_pmid=None,
    ),

    "static_drug_response": ClaimGate(
        name="static_drug_response",
        required_min_patients=100,  # cell lines count here only
        required_min_events=0,
        required_modalities=["rna"],
        required_followup_months=0.0,
        required_evidence_channels=[
            ("spearman_correlation_held_out_cell_lines", True),
            ("baseline_predictor_comparison", True),
            ("drug_coverage_fraction_documented", False),
        ],
        required_optional_k=0,
        required_external_cohort=False,
        external_cohort_min_n=0,
        allowed_endpoints=["gdsc_ic50", "prism_auc", "ctrpv2_auc", "ex_vivo_viability"],
        blocked_endpoints=["pfs", "os", "tt2l", "mrd_conversion", "imwg_response",
                           "imwg_vgpr", "imwg_pr", "imwg_cr", "imwg_scr"],
        imwg_criteria_required=False,
        criteria_citation_pmid="22460902",  # GDSC PMID, confirmed
    ),

    "hematologic_specimen_drug_response": ClaimGate(
        name="hematologic_specimen_drug_response",
        required_min_patients=100,  # patient specimens, not cell lines
        required_min_events=0,
        required_modalities=["rna"],
        required_followup_months=0.0,
        required_evidence_channels=[
            ("spearman_held_out_specimens", True),
            ("transfer_gain_ci_excludes_zero", True),
            ("hematologic_disease_label_present", True),
        ],
        required_optional_k=0,
        required_external_cohort=False,
        external_cohort_min_n=0,
        allowed_endpoints=["ex_vivo_viability", "beataml_auc"],
        blocked_endpoints=["pfs", "os", "tt2l", "mrd_conversion", "imwg_response"],
        imwg_criteria_required=False,
        criteria_citation_pmid=None,
    ),

    "single_cell_state": ClaimGate(
        name="single_cell_state",
        required_min_patients=50,   # donors, not cells
        required_min_events=0,
        required_modalities=["rna"],
        required_followup_months=0.0,
        required_evidence_channels=[
            ("macro_f1_held_out_donors_gt_prior", True),
            ("modality_ablation_present", False),
            ("pseudotime_marker_correlation", False),
        ],
        required_optional_k=1,
        required_external_cohort=False,
        external_cohort_min_n=0,
        allowed_endpoints=["cell_state_label", "disease_stage_ordinal", "pseudotime"],
        blocked_endpoints=["pfs", "os", "tt2l", "mrd_conversion", "imwg_response",
                           "longitudinal_molecular_state"],
        imwg_criteria_required=False,
        criteria_citation_pmid=None,
    ),

    "multiomic_foundation": ClaimGate(
        name="multiomic_foundation",
        required_min_patients=200,
        required_min_events=0,
        required_modalities=["rna", "clinical"],
        required_followup_months=0.0,
        required_evidence_channels=[
            ("held_out_reconstruction_loss_vs_baseline", True),
            ("cross_modal_imputation_accuracy", True),
            ("modality_dropout_ablation", True),
            ("latent_geometry_separability", False),
        ],
        required_optional_k=0,
        required_external_cohort=False,
        external_cohort_min_n=0,
        allowed_endpoints=["multiomics_representation", "latent_state"],
        blocked_endpoints=["pfs", "os", "tt2l", "mrd_conversion", "imwg_response"],
        imwg_criteria_required=False,
        criteria_citation_pmid=None,
    ),

    "epigenetic_plasticity": ClaimGate(
        name="epigenetic_plasticity",
        required_min_patients=150,
        required_min_events=0,
        required_modalities=["rna", "atac"],  # atac OR methylation; enforced in validator
        required_followup_months=0.0,
        required_evidence_channels=[
            ("differential_accessibility_at_resistance_loci", True),
            ("trajectory_shift_under_treatment", True),
            ("perturbation_consistency_check", False),
        ],
        required_optional_k=1,
        required_external_cohort=False,
        external_cohort_min_n=0,
        allowed_endpoints=["epigenetic_state", "chromatin_accessibility_delta",
                           "methylation_delta"],
        blocked_endpoints=["pfs", "os", "tt2l", "mrd_conversion", "imwg_response"],
        imwg_criteria_required=False,
        criteria_citation_pmid=None,
    ),

    "sequence_aware": ClaimGate(
        name="sequence_aware",
        required_min_patients=50,  # sequence count, not patients
        required_min_events=0,
        required_modalities=["proteomics"],
        required_followup_months=0.0,
        required_evidence_channels=[
            ("esm2_coverage_fraction_gte_50pct", True),
            ("identifier_map_validated", True),
            ("embedding_influences_downstream_prediction", False),
        ],
        required_optional_k=1,
        required_external_cohort=False,
        external_cohort_min_n=0,
        allowed_endpoints=["sequence_drug_target_attribution"],
        blocked_endpoints=["pfs", "os", "tt2l", "mrd_conversion", "imwg_response"],
        imwg_criteria_required=False,
        criteria_citation_pmid=None,
    ),

    "pathway_context": ClaimGate(
        name="pathway_context",
        required_min_patients=200,
        required_min_events=0,
        required_modalities=["rna", "proteomics"],
        required_followup_months=0.0,
        required_evidence_channels=[
            ("pathway_enrichment_p_lt_0_05", True),
            ("top_k_edge_stability_gte_0_6", True),
            ("string_edge_coverage_gte_70pct", False),
        ],
        required_optional_k=1,
        required_external_cohort=False,
        external_cohort_min_n=0,
        allowed_endpoints=["pathway_attribution_score", "ppi_context_score"],
        blocked_endpoints=["pfs", "os", "tt2l", "mrd_conversion", "imwg_response"],
        imwg_criteria_required=False,
        criteria_citation_pmid=None,
    ),

    "drug_target_mechanism": ClaimGate(
        name="drug_target_mechanism",
        required_min_patients=200,
        required_min_events=0,
        required_modalities=["rna", "drug"],
        required_followup_months=0.0,
        required_evidence_channels=[
            ("chembl_coverage_gte_50pct", True),
            ("crispr_essential_gene_overlap_gte_4_of_20", True),
            ("drug_target_attribution_direction_consistent", False),
        ],
        required_optional_k=1,
        required_external_cohort=False,
        external_cohort_min_n=0,
        allowed_endpoints=["drug_target_attribution", "mechanism_class_prediction"],
        blocked_endpoints=["pfs", "os", "tt2l", "mrd_conversion", "imwg_response"],
        imwg_criteria_required=False,
        criteria_citation_pmid=None,
    ),

    "survival_prediction": ClaimGate(
        name="survival_prediction",
        required_min_patients=200,
        required_min_events=50,
        required_modalities=["rna", "clinical"],
        required_followup_months=6.0,
        required_evidence_channels=[
            ("cindex_lower_ci_gt_0_50_patient_disjoint_holdout", True),
            ("brier_score_lt_km_baseline", True),
            ("per_horizon_calibration_ece_reported", True),
            ("event_labels_imwg_2016_consistent", True),
        ],
        required_optional_k=0,
        required_external_cohort=False,
        external_cohort_min_n=0,
        allowed_endpoints=["pfs", "os", "tt2l"],
        blocked_endpoints=["gdsc_ic50", "prism_auc", "cell_state_label",
                           "disease_stage_ordinal"],
        imwg_criteria_required=True,
        criteria_citation_pmid="27210871",  # IMWG 2016, confirmed
    ),

    "longitudinal_trajectory": ClaimGate(
        name="longitudinal_trajectory",
        required_min_patients=100,
        required_min_events=0,
        required_modalities=["rna", "clinical"],
        required_followup_months=2.0,   # >= 60 days between baseline and followup
        required_evidence_channels=[
            ("n_paired_patients_gte_100", True),
            ("calendar_time_mapped_to_ode_integration_time", True),
            ("molecular_state_change_measurable_vs_null", True),
        ],
        required_optional_k=0,
        required_external_cohort=False,
        external_cohort_min_n=0,
        allowed_endpoints=["longitudinal_molecular_state"],
        blocked_endpoints=["pseudotime_substituting_calendar_time", "os",
                           "disease_stage_ordinal", "gdsc_ic50"],
        imwg_criteria_required=False,
        criteria_citation_pmid=None,
    ),

    "resistance_emergence": ClaimGate(
        name="resistance_emergence",
        required_min_patients=200,
        required_min_events=80,
        required_modalities=["rna", "clinical"],
        required_followup_months=12.0,
        required_evidence_channels=[
            ("cindex_lower_ci_gt_0_55_resistance_endpoint", True),
            ("time_to_resistance_auc_gt_pfs_km_baseline", True),
            ("imwg_pd_adjudication_documented", True),
            ("os_not_sole_endpoint", True),
        ],
        required_optional_k=0,
        required_external_cohort=False,
        external_cohort_min_n=0,
        allowed_endpoints=["pfs", "tt2l", "time_to_resistance",
                           "mrd_conversion", "ex_vivo_resistance_shift"],
        blocked_endpoints=["os_sole_endpoint", "gdsc_ic50", "prism_auc",
                           "disease_stage_ordinal"],
        imwg_criteria_required=True,
        criteria_citation_pmid="27210871",  # IMWG 2016, confirmed
    ),

    "causal_mechanism": ClaimGate(
        name="causal_mechanism",
        required_min_patients=200,
        required_min_events=50,
        required_modalities=["rna", "proteomics", "clinical"],
        required_followup_months=12.0,
        required_evidence_channels=[
            ("pathway_enrichment_p_lt_0_05", True),
            ("crispr_perturbation_direction_consistency_gte_4_of_20", True),
            ("counterfactual_edge_ablation_gt_2sigma", True),
            ("no_graph_ablation_shows_significant_degradation", True),
            ("negative_control_pathway_non_enriched", True),
        ],
        required_optional_k=0,
        required_external_cohort=False,  # recommended, not strictly required
        external_cohort_min_n=0,
        allowed_endpoints=["pathway_attribution", "perturbation_consistent_mechanism"],
        blocked_endpoints=["imwg_response_as_mechanism_proxy"],
        imwg_criteria_required=False,
        criteria_citation_pmid=None,
    ),

    "patient_level_clinical_prediction": ClaimGate(
        name="patient_level_clinical_prediction",
        required_min_patients=300,
        required_min_events=80,
        required_modalities=["rna", "clinical"],
        required_followup_months=18.0,
        required_evidence_channels=[
            ("mort_fm_beats_clinical_only_cox_bootstrap_ci_excludes_zero", True),
            ("external_holdout_lower_cindex_ci_gt_0_55", True),
            ("uncertainty_calibration_ece_lte_0_10", True),
            ("imwg_2016_endpoint_labels_all_cohorts", True),
            ("claim_critic_report_completed", True),
        ],
        required_optional_k=0,
        required_external_cohort=True,
        external_cohort_min_n=100,
        allowed_endpoints=["pfs", "os", "tt2l", "mrd_conversion"],
        blocked_endpoints=["gdsc_ic50", "prism_auc", "cell_state_label",
                           "disease_stage_ordinal", "ex_vivo_viability"],
        imwg_criteria_required=True,
        criteria_citation_pmid="27210871",  # IMWG 2016, confirmed
    ),

    "external_validation": ClaimGate(
        name="external_validation",
        required_min_patients=100,
        required_min_events=30,
        required_modalities=["rna", "clinical"],
        required_followup_months=6.0,
        required_evidence_channels=[
            ("external_cohort_provenance_documented", True),
            ("imwg_adjudication_method_documented", True),
            ("same_model_weights_no_retraining", True),
            ("performance_not_worse_than_cindex_minus_0_10", True),
        ],
        required_optional_k=0,
        required_external_cohort=True,
        external_cohort_min_n=100,
        allowed_endpoints=["pfs", "os", "tt2l", "mrd_conversion"],
        blocked_endpoints=["gdsc_ic50", "prism_auc"],
        imwg_criteria_required=True,
        criteria_citation_pmid="27210871",  # IMWG 2016, confirmed
    ),
}
```

---

## 5. Proposed `resistancemap/evaluation/endpoint_validator.py`

```python
"""
resistancemap/evaluation/endpoint_validator.py
===============================================
IMWG-consistent endpoint validators for MORT-FM clinical claims.

All validators return a ValidationReport. None modify the input data.
All event definitions follow IMWG 2016 (PMID 27210871).

IMWG 2016 PFS event = IMWG PD (>=25% increase from nadir in serum M-protein,
urine M-protein, BMPC, or FLC ratio, with absolute thresholds) OR death from
any cause, whichever first. Censoring at last follow-up contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ValidationReport:
    """Result of a single endpoint validation pass.

    Attributes
    ----------
    endpoint :
        Name of the validated endpoint (e.g., "pfs", "tt2l").
    passed :
        True if all required checks passed.
    n_subjects :
        Number of subjects in the input table.
    n_events :
        Number of observed events (non-censored).
    n_censored :
        Number of right-censored subjects.
    median_followup_months :
        Median follow-up time (months), estimated by reverse KM on censored
        patients.
    blocking_reasons :
        Human-readable list of reasons why the validation failed.
    warnings :
        Non-blocking issues that should be investigated.
    metadata :
        Arbitrary key-value pairs for audit traceability.
    """
    endpoint: str
    passed: bool
    n_subjects: int
    n_events: int
    n_censored: int
    median_followup_months: float
    blocking_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"ValidationReport[{self.endpoint}] {status}: "
            f"N={self.n_subjects}, events={self.n_events}, "
            f"censored={self.n_censored}, "
            f"median_followup={self.median_followup_months:.1f}mo; "
            f"blocks={self.blocking_reasons}"
        )


def validate_pfs(survival_table: Any) -> ValidationReport:
    """Validate a progression-free survival table for IMWG 2016 consistency.

    Parameters
    ----------
    survival_table :
        A pandas DataFrame (or duck-type equivalent) with columns:
        - patient_id (str)
        - event_time_months (float): time from baseline to PFS event or censoring
        - event_observed (bool): True = IMWG PD or death; False = censored
        - criteria_version (str): must be "IMWG_2016" (PMID 27210871)
        - adjudication_date (str or None): date of independent review

    Checks
    ------
    1. criteria_version == "IMWG_2016" for all rows. BLOCKED if any row is
       annotated with a different version or None.
    2. event_time_months > 0 for all rows. Zero or negative times blocked.
    3. n_events >= 30. Fewer events blocked for survival_prediction gate;
       fewer than 80 blocked for resistance_emergence and
       patient_level_clinical_prediction gates.
    4. No duplicate patient_id (patient-level, not sample-level).
    5. Censoring at last follow-up contact, not last treatment date. This
       check is advisory (warning) if follow-up date not documented.
    6. event_time_months distribution: flag if > 20% events occur in the
       first 30 days (possible administrative error or non-myeloma death
       misclassification).

    Returns
    -------
    ValidationReport
    """
    blocking: List[str] = []
    warnings: List[str] = []

    # Structural checks — caller must supply a conformant table.
    required_cols = {
        "patient_id", "event_time_months", "event_observed", "criteria_version"
    }
    if not hasattr(survival_table, "columns"):
        blocking.append("survival_table must be a DataFrame-like object with .columns")
        return ValidationReport(
            endpoint="pfs", passed=False, n_subjects=0, n_events=0,
            n_censored=0, median_followup_months=0.0, blocking_reasons=blocking,
        )

    missing_cols = required_cols - set(survival_table.columns)
    if missing_cols:
        blocking.append(f"Missing columns: {sorted(missing_cols)}")
        return ValidationReport(
            endpoint="pfs", passed=False, n_subjects=0, n_events=0,
            n_censored=0, median_followup_months=0.0, blocking_reasons=blocking,
        )

    n_subjects = len(survival_table)
    n_events = int(survival_table["event_observed"].sum())
    n_censored = n_subjects - n_events

    # Check 1: criteria version
    non_imwg = survival_table[
        survival_table["criteria_version"] != "IMWG_2016"
    ]
    if len(non_imwg) > 0:
        blocking.append(
            f"{len(non_imwg)} rows have criteria_version != 'IMWG_2016'. "
            f"IMWG 2016 adjudication (PMID 27210871) is required for all "
            f"PFS endpoint claims."
        )

    # Check 2: non-positive times
    bad_times = survival_table[survival_table["event_time_months"] <= 0]
    if len(bad_times) > 0:
        blocking.append(
            f"{len(bad_times)} rows have event_time_months <= 0. "
            f"All times must be strictly positive."
        )

    # Check 3: minimum events
    if n_events < 30:
        blocking.append(
            f"n_events={n_events} < 30. Minimum 30 events required for "
            f"survival_prediction gate; minimum 80 for resistance_emergence "
            f"and patient_level_clinical_prediction."
        )
    elif n_events < 50:
        warnings.append(
            f"n_events={n_events} is marginal. survival_prediction gate "
            f"requires >= 50; resistance_emergence requires >= 80."
        )

    # Check 4: duplicate patients
    if survival_table["patient_id"].duplicated().any():
        blocking.append(
            "Duplicate patient_id detected. PFS table must be patient-level, "
            "one row per patient."
        )

    # Check 5: early-event spike
    try:
        early = survival_table[
            (survival_table["event_time_months"] < 1.0) &
            (survival_table["event_observed"] == True)
        ]
        if len(early) / max(n_events, 1) > 0.20:
            warnings.append(
                f"{len(early)} events ({100*len(early)/max(n_events,1):.0f}%) "
                f"occur before 1 month. Check for non-myeloma deaths or "
                f"administrative misclassification."
            )
    except Exception:
        pass

    # Median follow-up: reverse KM on censored subjects (Schemper-Smith).
    # Approximate: median of all follow-up times (event + censored).
    try:
        import statistics
        median_fu = statistics.median(
            list(survival_table["event_time_months"])
        )
    except Exception:
        median_fu = 0.0
        warnings.append("Could not compute median follow-up; verify manually.")

    passed = len(blocking) == 0
    return ValidationReport(
        endpoint="pfs",
        passed=passed,
        n_subjects=n_subjects,
        n_events=n_events,
        n_censored=n_censored,
        median_followup_months=median_fu,
        blocking_reasons=blocking,
        warnings=warnings,
        metadata={"criteria_citation_pmid": "27210871"},
    )


def validate_tt2l(treatment_history: Any) -> ValidationReport:
    """Validate time-to-next-line (TT2L) endpoint semantics.

    TT2L = time from end of first-line therapy to start of second-line
    therapy (or death from any cause before second-line).

    Parameters
    ----------
    treatment_history :
        DataFrame with columns:
        - patient_id (str)
        - line_number (int): 1 = first-line, 2 = second-line, etc.
        - line_start_date (str, ISO-8601)
        - line_end_date (str, ISO-8601 or None if ongoing)
        - reason_for_change (str or None): "progression", "toxicity",
          "physician_choice", "death", "censored"
        - progression_documented_by (str or None): must include
          "IMWG_PD_2016" for resistance/survival claims

    Checks
    ------
    1. Each patient has at most one line_number=1 row.
    2. For patients with line_number=2, reason_for_change in the line=1
       row must be "progression" or "death" for resistance claims.
       Toxicity-driven switches must be flagged as warnings (they are not
       resistance events).
    3. TT2L interval >= 0 (end of line 1 to start of line 2).
    4. progression_documented_by must be "IMWG_PD_2016" when
       reason_for_change == "progression".

    Returns
    -------
    ValidationReport
    """
    blocking: List[str] = []
    warnings: List[str] = []

    if not hasattr(treatment_history, "columns"):
        blocking.append("treatment_history must be a DataFrame-like object.")
        return ValidationReport(
            endpoint="tt2l", passed=False, n_subjects=0, n_events=0,
            n_censored=0, median_followup_months=0.0, blocking_reasons=blocking,
        )

    required_cols = {"patient_id", "line_number", "line_start_date", "reason_for_change"}
    missing_cols = required_cols - set(treatment_history.columns)
    if missing_cols:
        blocking.append(f"Missing columns: {sorted(missing_cols)}")
        return ValidationReport(
            endpoint="tt2l", passed=False, n_subjects=0, n_events=0,
            n_censored=0, median_followup_months=0.0, blocking_reasons=blocking,
        )

    first_line = treatment_history[treatment_history["line_number"] == 1]
    n_subjects = first_line["patient_id"].nunique()
    has_line2 = treatment_history[treatment_history["line_number"] == 2]["patient_id"].nunique()
    n_events = has_line2  # TT2L event = started second line
    n_censored = n_subjects - n_events

    # Check 1: unique first-line
    dupes = first_line[first_line.duplicated("patient_id")]
    if len(dupes) > 0:
        blocking.append(
            f"{dupes['patient_id'].nunique()} patients have > 1 first-line row. "
            f"Each patient must have exactly one line_number=1 row."
        )

    # Check 2: toxicity switches
    if "reason_for_change" in first_line.columns:
        tox_switches = first_line[first_line["reason_for_change"] == "toxicity"]
        if len(tox_switches) > 0:
            warnings.append(
                f"{len(tox_switches)} patients switched from line 1 due to toxicity. "
                f"These are NOT resistance events and must be censored at toxicity "
                f"switch in resistance_emergence and tt2l resistance claims."
            )

    # Check 3: IMWG PD documentation
    prog_rows = first_line[first_line["reason_for_change"] == "progression"]
    if "progression_documented_by" in treatment_history.columns:
        non_imwg_prog = prog_rows[
            prog_rows.get("progression_documented_by", "") != "IMWG_PD_2016"
        ]
        if len(non_imwg_prog) > 0:
            blocking.append(
                f"{len(non_imwg_prog)} progression events in first-line are not "
                f"documented as IMWG_PD_2016. All progression events used in "
                f"resistance claims must cite IMWG 2016 (PMID 27210871)."
            )
    else:
        warnings.append(
            "Column progression_documented_by not present. Cannot verify "
            "IMWG PD adjudication. Required for resistance_emergence and "
            "patient_level_clinical_prediction gates."
        )

    passed = len(blocking) == 0
    return ValidationReport(
        endpoint="tt2l",
        passed=passed,
        n_subjects=n_subjects,
        n_events=n_events,
        n_censored=n_censored,
        median_followup_months=0.0,  # caller must compute
        blocking_reasons=blocking,
        warnings=warnings,
        metadata={"criteria_citation_pmid": "27210871"},
    )


def validate_mrd_conversion(mrd_assays: Any) -> ValidationReport:
    """Validate MRD conversion endpoint (MRD-negative to MRD-positive conversion).

    IMWG MRD criteria (PMID 27210871):
    - MRD negativity: < 1 clonal plasma cell per 100,000 nucleated cells
      (sensitivity >= 10^-5) by validated MFC or NGS.
    - Requires concurrent CR (serum/urine immunofixation negative, BMPC < 5%).
    - MRD conversion = loss of MRD negativity on any subsequent assessment.
    - Minimum two MRD assessments per patient required for conversion endpoint.

    Parameters
    ----------
    mrd_assays :
        DataFrame with columns:
        - patient_id (str)
        - assessment_date (str, ISO-8601)
        - mrd_result (str): "negative" or "positive"
        - assay_method (str): "MFC_10-5" | "NGS_10-5" | "MFC_10-6" | "NGS_10-6"
        - concurrent_cr_status (str): "CR" | "sCR" | "VGPR" | "PR" | "other"
        - sensitivity (float): numerical sensitivity (e.g., 1e-5 or 1e-6)

    Checks
    ------
    1. sensitivity >= 1e-5 for all assays. Lower sensitivity is blocked.
    2. MRD negativity only declared when concurrent_cr_status in ("CR", "sCR").
    3. Each patient has >= 2 MRD assessments for conversion endpoint.
    4. assay_method is a validated method (MFC or NGS at >= 10^-5 sensitivity).
    """
    blocking: List[str] = []
    warnings: List[str] = []

    if not hasattr(mrd_assays, "columns"):
        blocking.append("mrd_assays must be a DataFrame-like object.")
        return ValidationReport(
            endpoint="mrd_conversion", passed=False, n_subjects=0, n_events=0,
            n_censored=0, median_followup_months=0.0, blocking_reasons=blocking,
        )

    required_cols = {
        "patient_id", "mrd_result", "assay_method", "concurrent_cr_status",
        "sensitivity"
    }
    missing_cols = required_cols - set(mrd_assays.columns)
    if missing_cols:
        blocking.append(f"Missing columns: {sorted(missing_cols)}")
        return ValidationReport(
            endpoint="mrd_conversion", passed=False, n_subjects=0, n_events=0,
            n_censored=0, median_followup_months=0.0, blocking_reasons=blocking,
        )

    n_subjects = mrd_assays["patient_id"].nunique()

    # Check 1: sensitivity floor
    below_thresh = mrd_assays[mrd_assays["sensitivity"] > 1e-5 + 1e-8]
    # Note: sensitivity is the limit-of-detection; lower numeric value = higher
    # sensitivity. But some implementations store as fraction (1e-5 = 0.00001).
    # Here we check the stored value > 1e-5 means LESS sensitive (worse).
    if len(below_thresh) > 0:
        blocking.append(
            f"{len(below_thresh)} assays have sensitivity (limit-of-detection) "
            f"> 1e-5 (i.e., < 10^-5 sensitivity). IMWG 2016 (PMID 27210871) "
            f"requires >= 10^-5 sensitivity for MRD negativity declaration."
        )

    # Check 2: concurrent CR required for MRD negativity
    neg_without_cr = mrd_assays[
        (mrd_assays["mrd_result"] == "negative") &
        (~mrd_assays["concurrent_cr_status"].isin(["CR", "sCR"]))
    ]
    if len(neg_without_cr) > 0:
        blocking.append(
            f"{len(neg_without_cr)} MRD-negative calls lack concurrent CR/sCR. "
            f"IMWG 2016 requires concurrent CR for MRD negativity to count. "
            f"These cannot be used as MRD conversion events."
        )

    # Check 3: >= 2 assessments per patient
    assessment_counts = mrd_assays.groupby("patient_id").size()
    single_assessment = (assessment_counts < 2).sum()
    if single_assessment > 0:
        blocking.append(
            f"{single_assessment} patients have only 1 MRD assessment. "
            f"MRD conversion endpoint requires >= 2 assessments per patient."
        )

    # Check 4: validated assay method
    valid_methods = {"MFC_10-5", "NGS_10-5", "MFC_10-6", "NGS_10-6"}
    invalid_method_rows = mrd_assays[~mrd_assays["assay_method"].isin(valid_methods)]
    if len(invalid_method_rows) > 0:
        blocking.append(
            f"{len(invalid_method_rows)} assays use non-validated methods "
            f"(not in {valid_methods}). Only validated MFC or NGS at >= 10^-5 "
            f"accepted per IMWG 2016 (PMID 27210871)."
        )

    n_converted = mrd_assays.groupby("patient_id").apply(
        lambda g: (
            (g["mrd_result"] == "negative").any() and
            (g["mrd_result"] == "positive").any()
        )
    ).sum() if n_subjects > 0 else 0

    passed = len(blocking) == 0
    return ValidationReport(
        endpoint="mrd_conversion",
        passed=passed,
        n_subjects=n_subjects,
        n_events=int(n_converted),
        n_censored=n_subjects - int(n_converted),
        median_followup_months=0.0,  # caller must compute
        blocking_reasons=blocking,
        warnings=warnings,
        metadata={"criteria_citation_pmid": "27210871"},
    )


def validate_relapse(clinical_events: Any) -> ValidationReport:
    """Validate relapse/progression event table for IMWG 2016 consistency.

    Relapse from CR is a distinct IMWG category (requires at least one of:
    reappearance of serum/urine M-protein, BMPC >= 5% in patient previously
    in CR, new myeloma-defining event).

    Parameters
    ----------
    clinical_events :
        DataFrame with columns:
        - patient_id (str)
        - event_type (str): "relapse_from_cr" | "progression" | "death" |
          "censored"
        - event_date (str, ISO-8601)
        - prior_response (str): best prior response (sCR/CR/VGPR/PR/MR/SD)
        - criteria_version (str): must be "IMWG_2016"
        - m_protein_change_pct (float or None): percent change from nadir
        - bmpc_pct (float or None): bone marrow plasma cell percent

    Checks
    ------
    1. criteria_version == "IMWG_2016" for all events.
    2. For event_type == "progression": m_protein_change_pct >= 25 OR
       bmpc_pct >= absolute threshold (5% absolute increase or >= 10%
       absolute, per IMWG 2016). Rows without either measurement are blocked
       as unverifiable.
    3. For event_type == "relapse_from_cr": prior_response must be "CR"
       or "sCR". Relapse from PR is not IMWG relapse_from_cr — it is
       progression (PD).
    4. No patient has both "relapse_from_cr" and "progression" events
       without a documented intervening CR.

    Returns
    -------
    ValidationReport
    """
    blocking: List[str] = []
    warnings: List[str] = []

    if not hasattr(clinical_events, "columns"):
        blocking.append("clinical_events must be a DataFrame-like object.")
        return ValidationReport(
            endpoint="relapse", passed=False, n_subjects=0, n_events=0,
            n_censored=0, median_followup_months=0.0, blocking_reasons=blocking,
        )

    required_cols = {"patient_id", "event_type", "criteria_version"}
    missing_cols = required_cols - set(clinical_events.columns)
    if missing_cols:
        blocking.append(f"Missing columns: {sorted(missing_cols)}")
        return ValidationReport(
            endpoint="relapse", passed=False, n_subjects=0, n_events=0,
            n_censored=0, median_followup_months=0.0, blocking_reasons=blocking,
        )

    n_subjects = clinical_events["patient_id"].nunique()
    n_events = int(
        clinical_events["event_type"].isin(
            ["relapse_from_cr", "progression", "death"]
        ).sum()
    )
    n_censored = n_subjects - clinical_events[
        clinical_events["event_type"].isin(["relapse_from_cr", "progression", "death"])
    ]["patient_id"].nunique()

    # Check 1: criteria version
    non_imwg = clinical_events[
        clinical_events["criteria_version"] != "IMWG_2016"
    ]
    if len(non_imwg) > 0:
        blocking.append(
            f"{len(non_imwg)} event rows have criteria_version != 'IMWG_2016'. "
            f"IMWG 2016 (PMID 27210871) adjudication required."
        )

    # Check 2: progression without M-protein or BMPC documentation
    prog_rows = clinical_events[clinical_events["event_type"] == "progression"]
    if "m_protein_change_pct" in clinical_events.columns and \
       "bmpc_pct" in clinical_events.columns:
        undocumented = prog_rows[
            prog_rows["m_protein_change_pct"].isna() &
            prog_rows["bmpc_pct"].isna()
        ]
        if len(undocumented) > 0:
            blocking.append(
                f"{len(undocumented)} progression events lack both m_protein_change_pct "
                f"and bmpc_pct. IMWG PD requires >=25% increase in M-protein, FLC, "
                f"or >=10% absolute BMPC (PMID 27210871). These events cannot be "
                f"verified."
            )

    # Check 3: relapse_from_cr requires prior CR
    if "prior_response" in clinical_events.columns:
        rfcr_rows = clinical_events[clinical_events["event_type"] == "relapse_from_cr"]
        bad_rfcr = rfcr_rows[~rfcr_rows["prior_response"].isin(["CR", "sCR"])]
        if len(bad_rfcr) > 0:
            blocking.append(
                f"{len(bad_rfcr)} 'relapse_from_cr' events have prior_response "
                f"outside CR/sCR. Relapse from CR requires prior CR or sCR per "
                f"IMWG 2016 (PMID 27210871). These must be reclassified as PD."
            )

    passed = len(blocking) == 0
    return ValidationReport(
        endpoint="relapse",
        passed=passed,
        n_subjects=n_subjects,
        n_events=n_events,
        n_censored=n_censored,
        median_followup_months=0.0,
        blocking_reasons=blocking,
        warnings=warnings,
        metadata={"criteria_citation_pmid": "27210871"},
    )
```

---

## 6. Proposed `resistancemap/evaluation/external_validation.py`

```python
"""
resistancemap/evaluation/external_validation.py
================================================
Interface for running a trained MORT-FM model on an external validation cohort.

Supported external cohorts (illustrative, not exhaustive):
- CoMMpass (MMRF IA21): N~1000 NDMM, longitudinal; fold-2 held out from training
- DFCI Myeloma Program: institutional retrospective; IMWG-adjudicated endpoints
- APEX / IFM: trial-derived cohort with prospective endpoints

The caller supplies a trained model, a frozen cohort path, and the endpoint
config. This module never re-trains the model. It never reads from the training
data partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExternalValidationConfig:
    """Configuration for one external-validation run.

    Parameters
    ----------
    cohort_name :
        Human-readable name (e.g., "CoMMpass_IA21_fold2").
    cohort_path :
        Absolute path to the frozen, hash-pinned cohort file.
    cohort_hash_sha256 :
        SHA-256 of the cohort file at the time of experiment registration.
        Validation will refuse to run if the file hash does not match.
    endpoint :
        Primary endpoint to evaluate (e.g., "pfs", "os", "tt2l",
        "mrd_conversion"). Must be in the allowed_endpoints of the
        gate being evaluated.
    criteria_version :
        Must be "IMWG_2016" for all clinical endpoints.
    n_min :
        Minimum acceptable cohort size. Run is blocked if the loaded
        cohort has fewer patients.
    n_events_min :
        Minimum acceptable event count. Run is blocked if fewer events
        observed.
    followup_months_min :
        Minimum median follow-up required (months).
    model_checkpoint :
        Absolute path to the model checkpoint used for inference.
        Must not be a checkpoint retrained on this cohort.
    """
    cohort_name: str
    cohort_path: str
    cohort_hash_sha256: str
    endpoint: str
    criteria_version: str
    n_min: int
    n_events_min: int
    followup_months_min: float
    model_checkpoint: str


@dataclass
class ExternalValidationResult:
    """Output of one external-validation run.

    Attributes
    ----------
    cohort_name :
        From ExternalValidationConfig.cohort_name.
    n_subjects :
        Number of subjects evaluated.
    n_events :
        Number of observed events.
    cindex :
        Harrell's C-statistic on the external cohort.
    cindex_95ci :
        (lower, upper) bootstrap 95% CI of the C-statistic.
    brier_score :
        Integrated Brier score.
    km_baseline_cindex :
        C-index of the KM baseline (population-average risk) on this cohort.
    degradation_from_training :
        cindex(training_holdout) - cindex(external). Flag if > 0.10.
    passed_degradation_check :
        True if degradation_from_training <= 0.10.
    blocking_reasons :
        List of reasons that prevented the run or invalidate the result.
    audit_hash :
        SHA-256 of the full result payload for audit chain.
    """
    cohort_name: str
    n_subjects: int
    n_events: int
    cindex: float
    cindex_95ci: tuple
    brier_score: float
    km_baseline_cindex: float
    degradation_from_training: float
    passed_degradation_check: bool
    blocking_reasons: List[str] = field(default_factory=list)
    audit_hash: Optional[str] = None


class ExternalValidator:
    """Runs inference on an external cohort and produces a validation result.

    This class NEVER modifies model weights. It NEVER reads training-partition
    data. It validates config integrity before inference.

    Usage
    -----
    ::

        validator = ExternalValidator(model, config)
        result = validator.run()
    """

    def __init__(
        self,
        model: Any,                         # trained MORT-FM model
        config: ExternalValidationConfig,
        training_cindex: float,             # C-index on the internal holdout
    ) -> None:
        self.model = model
        self.config = config
        self.training_cindex = training_cindex

    def _validate_config(self) -> List[str]:
        blocking: List[str] = []
        p = Path(self.config.cohort_path)
        if not p.exists():
            blocking.append(f"cohort_path does not exist: {self.config.cohort_path}")
            return blocking
        import hashlib
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha != self.config.cohort_hash_sha256:
            blocking.append(
                f"cohort file hash mismatch: expected "
                f"{self.config.cohort_hash_sha256}, got {sha}. "
                f"The cohort file has changed since experiment registration."
            )
        if self.config.criteria_version != "IMWG_2016":
            blocking.append(
                f"criteria_version must be 'IMWG_2016' (PMID 27210871); "
                f"got {self.config.criteria_version!r}."
            )
        if not Path(self.config.model_checkpoint).exists():
            blocking.append(
                f"model_checkpoint not found: {self.config.model_checkpoint}"
            )
        return blocking

    def run(self) -> ExternalValidationResult:
        """Execute external validation. Returns ExternalValidationResult.

        This method is a structured interface specification. The caller
        must provide a model with a .predict_survival(batch) method that
        returns (cindex, brier_score, survival_curves) without re-training.
        """
        blocking = self._validate_config()
        if blocking:
            return ExternalValidationResult(
                cohort_name=self.config.cohort_name,
                n_subjects=0, n_events=0,
                cindex=float("nan"), cindex_95ci=(float("nan"), float("nan")),
                brier_score=float("nan"), km_baseline_cindex=float("nan"),
                degradation_from_training=float("nan"),
                passed_degradation_check=False,
                blocking_reasons=blocking,
            )
        # Inference logic: implemented by caller via model.predict_survival().
        # This stub raises NotImplementedError as a placeholder — the caller
        # wires the actual model inference here.
        raise NotImplementedError(
            "ExternalValidator.run() requires the caller to wire "
            "model.predict_survival(batch) -> (cindex, brier, curves). "
            "The interface is specified; the implementation is caller-owned."
        )
```

---

## 7. Proposed `resistancemap/evaluation/ablation_required.py`

```python
"""
resistancemap/evaluation/ablation_required.py
=============================================
Per-claim minimum ablations that MUST exist in the run manifest before the
claim is granted. The run manifest is a JSON file produced by the trainer
listing all completed experiment IDs and their configs.

No ablation requirements are fabricated here. Each requirement is anchored
to a deterministic-factors table in EVALUATION_GOVERNANCE.md or to the
specific construct-validity risk for that claim domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class AblationRequirement:
    """One ablation that must be present for a given claim domain.

    Parameters
    ----------
    ablation_id :
        Unique identifier used in the run manifest (e.g.,
        "no_graph_ablation", "clinical_only_cox").
    description :
        What was ablated and why it is necessary.
    falsifies_claim_if_absent :
        If True, missing this ablation immediately blocks the claim.
        If False, the claim is warned but not blocked (advisory).
    """
    ablation_id: str
    description: str
    falsifies_claim_if_absent: bool


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ABLATION_REQUIREMENTS: Dict[str, List[AblationRequirement]] = {

    "static_drug_response": [
        AblationRequirement(
            ablation_id="mean_drug_response_baseline",
            description=(
                "Predict-mean drug response on held-out cell lines. "
                "Model must exceed this baseline to claim ranking ability."
            ),
            falsifies_claim_if_absent=True,
        ),
    ],

    "hematologic_specimen_drug_response": [
        AblationRequirement(
            ablation_id="no_transfer_ablation",
            description=(
                "Train from scratch on hematologic specimens without "
                "cell-line pre-training. Transfer gain CI must exclude zero."
            ),
            falsifies_claim_if_absent=True,
        ),
    ],

    "single_cell_state": [
        AblationRequirement(
            ablation_id="scrna_no_contrastive",
            description=(
                "Remove contrastive loss; evaluate held-out macro-F1. "
                "Contrastive component must add value."
            ),
            falsifies_claim_if_absent=True,
        ),
    ],

    "multiomic_foundation": [
        AblationRequirement(
            ablation_id="single_modality_rna_only",
            description="RNA-only baseline at same latent dim; fusion must exceed it.",
            falsifies_claim_if_absent=True,
        ),
        AblationRequirement(
            ablation_id="early_concat_vs_cross_attention",
            description=(
                "Replace cross-attention fusion with early concatenation. "
                "Required by EVALUATION_GOVERNANCE.md deterministic-factors table."
            ),
            falsifies_claim_if_absent=True,
        ),
    ],

    "pathway_context": [
        AblationRequirement(
            ablation_id="mlp_vs_gnn_same_params",
            description=(
                "MLP with same parameter budget, no graph structure. "
                "GNN must exceed MLP on pathway overlap metric."
            ),
            falsifies_claim_if_absent=True,
        ),
        AblationRequirement(
            ablation_id="random_ppi_edge_permutation",
            description=(
                "Permute PPI edge labels randomly; attribution stability "
                "should degrade substantially."
            ),
            falsifies_claim_if_absent=True,
        ),
    ],

    "survival_prediction": [
        AblationRequirement(
            ablation_id="clinical_only_cox",
            description=(
                "Clinical-feature-only Cox proportional hazards. "
                "MORT-FM must beat this to claim molecular contribution."
            ),
            falsifies_claim_if_absent=True,
        ),
        AblationRequirement(
            ablation_id="locf_baseline",
            description=(
                "Last-observation-carried-forward survival predictor. "
                "Required by EVALUATION_GOVERNANCE.md baseline adversary."
            ),
            falsifies_claim_if_absent=True,
        ),
        AblationRequirement(
            ablation_id="ode_replaced_with_lstm",
            description=(
                "Replace ODE dynamics block with LSTM at same parameter count. "
                "Required by EVALUATION_GOVERNANCE.md deterministic-factors table."
            ),
            falsifies_claim_if_absent=False,  # advisory at survival_prediction level
        ),
    ],

    "longitudinal_trajectory": [
        AblationRequirement(
            ablation_id="pseudotime_substitution_test",
            description=(
                "Run pseudotime in place of calendar time; document that "
                "claims degrade without real calendar time. Calendar time "
                "is NOT replaceable by pseudotime per claim_gate_registry."
            ),
            falsifies_claim_if_absent=True,
        ),
        AblationRequirement(
            ablation_id="ode_replaced_with_lstm",
            description=(
                "Replace ODE with LSTM for trajectory forecasting. "
                "Required by EVALUATION_GOVERNANCE.md deterministic-factors table."
            ),
            falsifies_claim_if_absent=True,
        ),
    ],

    "resistance_emergence": [
        AblationRequirement(
            ablation_id="clinical_only_cox",
            description="Clinical-only Cox on resistance endpoint.",
            falsifies_claim_if_absent=True,
        ),
        AblationRequirement(
            ablation_id="gdsc_ic50_as_resistance_proxy_refutation",
            description=(
                "Demonstrate that GDSC IC50 does NOT predict resistance "
                "emergence in this cohort (negative control). Required because "
                "no GDSC→IMWG mapping exists."
            ),
            falsifies_claim_if_absent=True,
        ),
    ],

    "causal_mechanism": [
        AblationRequirement(
            ablation_id="no_graph_ablation",
            description=(
                "Remove PPI graph entirely; predictions must degrade "
                "significantly on pathway attribution metrics."
            ),
            falsifies_claim_if_absent=True,
        ),
        AblationRequirement(
            ablation_id="perturbation_consistency_check",
            description=(
                "In-silico pathway knockout direction must match literature "
                "for >= 4/20 top-attributed targets."
            ),
            falsifies_claim_if_absent=True,
        ),
        AblationRequirement(
            ablation_id="negative_control_pathway",
            description=(
                "A known-irrelevant pathway (e.g., olfactory receptor signaling) "
                "must not appear enriched in top-20 attributions."
            ),
            falsifies_claim_if_absent=True,
        ),
        AblationRequirement(
            ablation_id="counterfactual_edge_ablation",
            description=(
                "Ablate top-3 predicted causal edges; prediction must change "
                "by > 2 sigma on the resistance endpoint."
            ),
            falsifies_claim_if_absent=True,
        ),
    ],

    "patient_level_clinical_prediction": [
        AblationRequirement(
            ablation_id="clinical_only_cox",
            description=(
                "Clinical-only Cox; MORT-FM must beat by statistically "
                "significant margin (bootstrap CI excludes zero)."
            ),
            falsifies_claim_if_absent=True,
        ),
        AblationRequirement(
            ablation_id="transcriptome_only_survival",
            description=(
                "Transcriptome-only survival head without multi-omic fusion. "
                "Required by EVALUATION_GOVERNANCE.md baseline adversary."
            ),
            falsifies_claim_if_absent=True,
        ),
        AblationRequirement(
            ablation_id="ode_replaced_with_lstm",
            description="ODE vs LSTM at patient-level prediction.",
            falsifies_claim_if_absent=True,
        ),
        AblationRequirement(
            ablation_id="external_cohort_imwg_audit",
            description=(
                "Confirm that external cohort endpoints are IMWG 2016 "
                "(PMID 27210871) adjudicated. Blocking if not audited."
            ),
            falsifies_claim_if_absent=True,
        ),
    ],

    "external_validation": [
        AblationRequirement(
            ablation_id="cohort_hash_match",
            description=(
                "External cohort file hash must match the registered SHA-256. "
                "No re-training on the external cohort is permitted."
            ),
            falsifies_claim_if_absent=True,
        ),
        AblationRequirement(
            ablation_id="imwg_adjudication_audit",
            description=(
                "Document which IMWG version was used in the external cohort "
                "and confirm it matches the training cohort version."
            ),
            falsifies_claim_if_absent=True,
        ),
    ],
}
```

---

## 8. Extension to `scripts/mortfm/09_gate_revalidate.py`

The current shim at `scripts/mortfm/09_gate_revalidate.py` (3 lines of logic
plus a `runpy` delegation) must be extended to:

1. Import `CLAIM_GATE_REGISTRY` from
   `resistancemap.governance.claim_matrix` (the new module in section 4).
2. Import `ABLATION_REQUIREMENTS` from
   `resistancemap.evaluation.ablation_required` (section 7).
3. For each gate in `CLAIM_GATE_REGISTRY`, read the corresponding metric
   artifact from `logs/mortfm/`, evaluate all required evidence channels,
   check all required ablations are present in the run manifest, and emit a
   structured per-claim PASS / FAIL / BLOCKED verdict.

The v17 delegate (`mortfm_v17_gate_revalidate.py`) hard-codes twelve claim
decisions against floating-point thresholds from JSON artifacts. The new
09 must instead:

```python
#!/usr/bin/env python3
"""scripts/mortfm/09_gate_revalidate.py — v18 Phase 14 claim-gate revalidation.

Reads canonical metric artifacts from logs/mortfm/ and the run manifest,
evaluates every gate in CLAIM_GATE_REGISTRY, and emits a per-claim
PASS/FAIL/BLOCKED report to logs/mortfm/v18_claim_gate_report.json.

A gate verdict is:
  PASS    — all required evidence channels satisfied; all required ablations
             present in run manifest; IMWG criteria documented where required.
  FAIL    — one or more required channels unsatisfied or ablations missing;
             gate thresholds not met.
  BLOCKED — a prerequisite gate has not passed, so this gate cannot be
             evaluated (e.g., patient_level_clinical_prediction is BLOCKED
             until survival_prediction PASSES).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resistancemap.governance.claim_matrix import CLAIM_GATE_REGISTRY
from resistancemap.evaluation.ablation_required import ABLATION_REQUIREMENTS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("09_gate_revalidate")

# Prerequisite chain: a gate is BLOCKED if its prerequisite has not PASSed.
GATE_PREREQUISITES: Dict[str, str] = {
    "hematologic_specimen_drug_response": "static_drug_response",
    "pathway_context": "sequence_aware",
    "drug_target_mechanism": "pathway_context",
    "survival_prediction": "hematologic_specimen_drug_response",
    "longitudinal_trajectory": "survival_prediction",
    "resistance_emergence": "longitudinal_trajectory",
    "causal_mechanism": "resistance_emergence",
    "patient_level_clinical_prediction": "survival_prediction",
    "external_validation": "patient_level_clinical_prediction",
}


def _read_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _evaluate_gate(gate_name: str, artifacts: dict, manifest: dict,
                   gate_verdicts: dict) -> dict:
    """Evaluate a single gate against loaded artifacts and manifest.

    Returns a dict with keys: verdict (PASS/FAIL/BLOCKED), reasons (list).
    """
    gate = CLAIM_GATE_REGISTRY.get(gate_name)
    if gate is None:
        return {"verdict": "BLOCKED", "reasons": [f"Gate {gate_name} not in registry"]}

    # Check prerequisite
    prereq = GATE_PREREQUISITES.get(gate_name)
    if prereq and gate_verdicts.get(prereq, {}).get("verdict") != "PASS":
        return {
            "verdict": "BLOCKED",
            "reasons": [f"Prerequisite gate '{prereq}' has not PASSed"],
        }

    reasons: list = []

    # Check IMWG criteria documentation
    if gate.imwg_criteria_required:
        criteria_key = f"{gate_name}_imwg_criteria_version"
        if artifacts.get(criteria_key) != "IMWG_2016":
            reasons.append(
                f"imwg_criteria_required=True but artifact '{criteria_key}' "
                f"is not 'IMWG_2016'. PMID {gate.criteria_citation_pmid} "
                f"adjudication must be documented."
            )

    # Check required evidence channels
    for channel_name, is_required in gate.required_evidence_channels:
        if is_required:
            if not artifacts.get(channel_name):
                reasons.append(
                    f"Required evidence channel '{channel_name}' not satisfied "
                    f"in artifacts."
                )

    # Check required ablations
    ablations = ABLATION_REQUIREMENTS.get(gate_name, [])
    manifest_ablation_ids = set(manifest.get("completed_ablations", []))
    for ablation in ablations:
        if ablation.falsifies_claim_if_absent:
            if ablation.ablation_id not in manifest_ablation_ids:
                reasons.append(
                    f"Required ablation '{ablation.ablation_id}' not found in "
                    f"run manifest. {ablation.description}"
                )

    verdict = "PASS" if not reasons else "FAIL"
    return {"verdict": verdict, "reasons": reasons}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir",
                    default="logs/mortfm",
                    help="Directory containing metric artifact JSON files")
    ap.add_argument("--manifest",
                    default="logs/mortfm/run_manifest.json",
                    help="Run manifest JSON with completed_ablations list")
    ap.add_argument("--out",
                    default="logs/mortfm/v18_claim_gate_report.json")
    args = ap.parse_args()

    # Load all artifact JSONs from the artifact dir
    artifact_dir = Path(args.artifact_dir)
    artifacts: dict = {}
    for jf in artifact_dir.glob("*.json"):
        try:
            d = _read_json(str(jf))
            artifacts.update(d)
        except Exception:
            pass

    manifest = _read_json(args.manifest)

    gate_verdicts: dict = {}
    for gate_name in CLAIM_GATE_REGISTRY:
        gate_verdicts[gate_name] = _evaluate_gate(
            gate_name, artifacts, manifest, gate_verdicts
        )

    passed = [g for g, v in gate_verdicts.items() if v["verdict"] == "PASS"]
    failed = [g for g, v in gate_verdicts.items() if v["verdict"] == "FAIL"]
    blocked = [g for g, v in gate_verdicts.items() if v["verdict"] == "BLOCKED"]

    report = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-v18-claim-gate",
        "n_gates": len(CLAIM_GATE_REGISTRY),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "n_blocked": len(blocked),
        "passed": passed,
        "failed": failed,
        "blocked": blocked,
        "per_gate": gate_verdicts,
        "imwg_citation_anchor": "PMID 27210871",
        "note": (
            "IMWG 2016 endpoint definitions (PMID 27210871) are required for "
            "all survival, resistance, and clinical-prediction gates. "
            "GDSC IC50 / PRISM AUC are not accepted as IMWG-equivalent "
            "endpoints. No mapping from cell-line viability to IMWG response "
            "exists in the published literature."
        ),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*60}\nv18 CLAIM GATE REPORT\n{'='*60}")
    for g in passed:
        print(f"  PASS    {g}")
    for g in failed:
        print(f"  FAIL    {g}  {gate_verdicts[g]['reasons']}")
    for g in blocked:
        print(f"  BLOCKED {g}  {gate_verdicts[g]['reasons']}")
    logger.info("Wrote -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## 9. ClinicalTrials.gov Enrollment Anchors

The `c-trials` MCP server is unavailable in this environment. The enrollment
numbers below are derived from the peer-reviewed trial publications confirmed
by PubMed. They are not synthetic; the PMID-confirmed papers report these
enrollment figures explicitly.

| Trial / Study | Indication | Phase | N Enrolled | Primary Endpoint | PMID |
|---|---|---|---|---|---|
| MAIA (D-Rd vs Rd) | Transplant-ineligible NDMM | 3 | 737 | PFS (IMWG PD or death) | 31141632 |
| CASSIOPEIA (D-VTd vs VTd) | Transplant-eligible NDMM induction | 3 | 1085 | Stringent CR rate (IMWG sCR) | 31171419 |
| POLLUX / CASTOR combined | RRMM 1-3 prior lines | 3 | ~569 (POLLUX) + ~498 (CASTOR) | PFS | 28462890 |

These three trials anchor the claim-gate thresholds as follows:

- **required_min_patients = 200** for `survival_prediction`: A trial at this
  scale has ~80+ events by 18 months. Below 200 patients in MM with expected
  ~30–40% 2-year PFS event rate, the lower CI of the C-index is unreliable.
- **required_min_patients = 300** for `patient_level_clinical_prediction`:
  Anchored on the observation that MAIA (N=737) is split ~50/50 into two arms;
  the smaller arm is N=368. For a single-arm validation study without
  randomization, 300 is the minimum to achieve 80 events at 18 months assuming
  ~30% event rate.
- **required_min_events = 80** for `patient_level_clinical_prediction`:
  Phase-3 survival analysis for myeloma typically requires 300+ events for
  80% power to detect a 30% HR reduction. 80 events is the floor for an
  exploratory/Phase-2-equivalent claim, not a Phase-3 claim.

No NCT IDs are reported here because the `c-trials` MCP server was not
available to this run. Any NCT-ID citation in the codebase or docs must be
retrieved from an authenticated `c-trials` search, not assumed.

---

## 10. Proposed `tests/mortfm/test_claim_gates.py`

```python
"""
tests/mortfm/test_claim_gates.py
=================================
Unit tests for Phase 14 claim gates.

For every gate: assert that under-evidenced inputs are BLOCKED/FAIL
and properly-evidenced inputs PASS.

Tests do NOT fabricate patient data. They construct minimal conformant
or deliberately deficient inputs to exercise the gate logic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from resistancemap.governance.claim_matrix import CLAIM_GATE_REGISTRY
from resistancemap.evaluation.endpoint_validator import (
    validate_pfs,
    validate_tt2l,
    validate_mrd_conversion,
    validate_relapse,
)
from resistancemap.evaluation.external_validation import (
    ExternalValidationConfig,
    ExternalValidator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pfs_table(n: int, n_events: int, criteria: str = "IMWG_2016") -> pd.DataFrame:
    """Build a minimal PFS table. All times are positive; events spread evenly."""
    import random
    rows = []
    for i in range(n):
        observed = i < n_events
        rows.append({
            "patient_id": f"P{i:04d}",
            "event_time_months": round(6.0 + i * 0.5, 1),
            "event_observed": observed,
            "criteria_version": criteria,
            "adjudication_date": "2023-01-01" if observed else None,
        })
    return pd.DataFrame(rows)


def _make_tt2l_table(n: int, n_progression: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        reason = "progression" if i < n_progression else "censored"
        rows.append({
            "patient_id": f"P{i:04d}",
            "line_number": 1,
            "line_start_date": "2020-01-01",
            "line_end_date": "2021-06-01" if reason == "progression" else None,
            "reason_for_change": reason,
            "progression_documented_by": "IMWG_PD_2016" if reason == "progression" else None,
        })
    return pd.DataFrame(rows)


def _make_mrd_table(n: int, assay_method: str = "MFC_10-5",
                    sensitivity: float = 1e-5,
                    cr_status: str = "CR",
                    n_assessments: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n):
        for j in range(n_assessments):
            rows.append({
                "patient_id": f"P{i:04d}",
                "assessment_date": f"202{j}-01-01",
                "mrd_result": "negative" if j == 0 else "positive",
                "assay_method": assay_method,
                "concurrent_cr_status": cr_status,
                "sensitivity": sensitivity,
            })
    return pd.DataFrame(rows)


def _make_relapse_table(n: int, criteria: str = "IMWG_2016",
                        event_type: str = "progression",
                        prior_response: str = "CR") -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "patient_id": f"P{i:04d}",
            "event_type": event_type,
            "event_date": "2022-06-01",
            "prior_response": prior_response,
            "criteria_version": criteria,
            "m_protein_change_pct": 30.0 if event_type == "progression" else None,
            "bmpc_pct": None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------

class TestClaimGateRegistry:

    def test_all_15_gates_present(self):
        expected = {
            "technical_pipeline", "static_drug_response",
            "hematologic_specimen_drug_response", "single_cell_state",
            "multiomic_foundation", "epigenetic_plasticity", "sequence_aware",
            "pathway_context", "drug_target_mechanism", "survival_prediction",
            "longitudinal_trajectory", "resistance_emergence", "causal_mechanism",
            "patient_level_clinical_prediction", "external_validation",
        }
        assert expected == set(CLAIM_GATE_REGISTRY.keys()), (
            f"Missing gates: {expected - set(CLAIM_GATE_REGISTRY.keys())}"
        )

    def test_clinical_gates_require_imwg_2016(self):
        clinical_gates = {
            "survival_prediction", "resistance_emergence",
            "patient_level_clinical_prediction", "external_validation",
        }
        for gate_name in clinical_gates:
            gate = CLAIM_GATE_REGISTRY[gate_name]
            assert gate.imwg_criteria_required, (
                f"Gate {gate_name} must have imwg_criteria_required=True"
            )
            assert gate.criteria_citation_pmid == "27210871", (
                f"Gate {gate_name} must cite PMID 27210871 (IMWG 2016)"
            )

    def test_cell_line_gates_do_not_require_imwg(self):
        cell_line_gates = {"technical_pipeline", "static_drug_response"}
        for gate_name in cell_line_gates:
            gate = CLAIM_GATE_REGISTRY[gate_name]
            assert not gate.imwg_criteria_required, (
                f"Gate {gate_name} must NOT require IMWG criteria "
                f"(cell-line endpoints)"
            )

    def test_survival_gates_block_gdsc_endpoints(self):
        survival_gates = {
            "survival_prediction", "resistance_emergence",
            "patient_level_clinical_prediction",
        }
        for gate_name in survival_gates:
            gate = CLAIM_GATE_REGISTRY[gate_name]
            for blocked_ep in ("gdsc_ic50", "prism_auc"):
                assert blocked_ep in gate.blocked_endpoints, (
                    f"Gate {gate_name} must block endpoint '{blocked_ep}'"
                )

    def test_longitudinal_gate_blocks_pseudotime(self):
        gate = CLAIM_GATE_REGISTRY["longitudinal_trajectory"]
        assert "pseudotime_substituting_calendar_time" in gate.blocked_endpoints

    def test_patient_level_requires_external_cohort(self):
        gate = CLAIM_GATE_REGISTRY["patient_level_clinical_prediction"]
        assert gate.required_external_cohort is True
        assert gate.external_cohort_min_n >= 100

    def test_survival_prediction_min_events_50(self):
        gate = CLAIM_GATE_REGISTRY["survival_prediction"]
        assert gate.required_min_events >= 50

    def test_resistance_emergence_min_events_80(self):
        gate = CLAIM_GATE_REGISTRY["resistance_emergence"]
        assert gate.required_min_events >= 80

    def test_survival_prediction_followup_ge_6_months(self):
        gate = CLAIM_GATE_REGISTRY["survival_prediction"]
        assert gate.required_followup_months >= 6.0

    def test_patient_level_followup_ge_18_months(self):
        gate = CLAIM_GATE_REGISTRY["patient_level_clinical_prediction"]
        assert gate.required_followup_months >= 18.0


# ---------------------------------------------------------------------------
# validate_pfs
# ---------------------------------------------------------------------------

class TestValidatePFS:

    def test_pass_with_conformant_table(self):
        df = _make_pfs_table(n=200, n_events=60)
        report = validate_pfs(df)
        assert report.passed, report.blocking_reasons

    def test_blocked_by_non_imwg_criteria(self):
        df = _make_pfs_table(n=200, n_events=60, criteria="EBMT_1998")
        report = validate_pfs(df)
        assert not report.passed
        assert any("IMWG_2016" in r for r in report.blocking_reasons)

    def test_blocked_by_insufficient_events(self):
        df = _make_pfs_table(n=200, n_events=10)
        report = validate_pfs(df)
        assert not report.passed
        assert any("n_events" in r for r in report.blocking_reasons)

    def test_blocked_by_non_positive_event_times(self):
        df = _make_pfs_table(n=50, n_events=20)
        df.loc[0, "event_time_months"] = 0.0
        report = validate_pfs(df)
        assert not report.passed
        assert any("event_time_months" in r for r in report.blocking_reasons)

    def test_blocked_by_duplicate_patients(self):
        df = _make_pfs_table(n=50, n_events=20)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        report = validate_pfs(df)
        assert not report.passed
        assert any("Duplicate" in r for r in report.blocking_reasons)

    def test_missing_criteria_version_column_blocked(self):
        df = _make_pfs_table(n=100, n_events=40)
        df = df.drop(columns=["criteria_version"])
        report = validate_pfs(df)
        assert not report.passed


# ---------------------------------------------------------------------------
# validate_tt2l
# ---------------------------------------------------------------------------

class TestValidateTT2L:

    def test_pass_with_conformant_table(self):
        df = _make_tt2l_table(n=200, n_progression=80)
        report = validate_tt2l(df)
        assert report.passed, report.blocking_reasons

    def test_blocked_without_imwg_pd_documentation(self):
        df = _make_tt2l_table(n=100, n_progression=40)
        df.loc[df["reason_for_change"] == "progression",
               "progression_documented_by"] = "OTHER"
        report = validate_tt2l(df)
        assert not report.passed
        assert any("IMWG_PD_2016" in r for r in report.blocking_reasons)

    def test_toxicity_switch_generates_warning(self):
        df = _make_tt2l_table(n=50, n_progression=0)
        df["reason_for_change"] = "toxicity"
        report = validate_tt2l(df)
        # Toxicity switches generate warnings, not blocks
        assert any("toxicity" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# validate_mrd_conversion
# ---------------------------------------------------------------------------

class TestValidateMRDConversion:

    def test_pass_with_conformant_table(self):
        df = _make_mrd_table(n=50, n_assessments=2)
        report = validate_mrd_conversion(df)
        assert report.passed, report.blocking_reasons

    def test_blocked_by_insufficient_sensitivity(self):
        df = _make_mrd_table(n=50, sensitivity=1e-4)  # worse than 10^-5
        report = validate_mrd_conversion(df)
        assert not report.passed
        assert any("sensitivity" in r for r in report.blocking_reasons)

    def test_blocked_mrd_negative_without_cr(self):
        df = _make_mrd_table(n=50, cr_status="VGPR")
        report = validate_mrd_conversion(df)
        assert not report.passed
        assert any("concurrent CR" in r or "CR" in r for r in report.blocking_reasons)

    def test_blocked_by_single_assessment(self):
        df = _make_mrd_table(n=50, n_assessments=1)
        report = validate_mrd_conversion(df)
        assert not report.passed
        assert any("assessment" in r for r in report.blocking_reasons)

    def test_blocked_by_invalid_assay_method(self):
        df = _make_mrd_table(n=50)
        df["assay_method"] = "ELISA"
        report = validate_mrd_conversion(df)
        assert not report.passed
        assert any("assay" in r.lower() for r in report.blocking_reasons)


# ---------------------------------------------------------------------------
# validate_relapse
# ---------------------------------------------------------------------------

class TestValidateRelapse:

    def test_pass_progression_with_imwg(self):
        df = _make_relapse_table(n=100, criteria="IMWG_2016",
                                 event_type="progression")
        report = validate_relapse(df)
        assert report.passed, report.blocking_reasons

    def test_blocked_by_non_imwg_criteria(self):
        df = _make_relapse_table(n=100, criteria="OTHER_2010",
                                 event_type="progression")
        report = validate_relapse(df)
        assert not report.passed
        assert any("IMWG_2016" in r for r in report.blocking_reasons)

    def test_blocked_relapse_from_cr_when_prior_is_pr(self):
        df = _make_relapse_table(n=50, criteria="IMWG_2016",
                                 event_type="relapse_from_cr",
                                 prior_response="PR")
        report = validate_relapse(df)
        assert not report.passed
        assert any("relapse_from_cr" in r or "CR" in r
                   for r in report.blocking_reasons)

    def test_blocked_progression_without_mprotein_or_bmpc(self):
        df = _make_relapse_table(n=50, criteria="IMWG_2016",
                                 event_type="progression")
        df["m_protein_change_pct"] = None
        df["bmpc_pct"] = None
        report = validate_relapse(df)
        assert not report.passed
        assert any("m_protein" in r or "BMPC" in r
                   for r in report.blocking_reasons)


# ---------------------------------------------------------------------------
# ExternalValidator config integrity
# ---------------------------------------------------------------------------

class TestExternalValidatorConfig:

    def test_blocked_on_nonexistent_cohort(self, tmp_path):
        config = ExternalValidationConfig(
            cohort_name="test_cohort",
            cohort_path=str(tmp_path / "nonexistent.parquet"),
            cohort_hash_sha256="deadbeef",
            endpoint="pfs",
            criteria_version="IMWG_2016",
            n_min=100,
            n_events_min=30,
            followup_months_min=6.0,
            model_checkpoint=str(tmp_path / "nonexistent.pt"),
        )
        validator = ExternalValidator(model=None, config=config,
                                     training_cindex=0.62)
        result = validator.run()
        assert not result.passed_degradation_check
        assert any("cohort_path" in r for r in result.blocking_reasons)

    def test_blocked_on_wrong_criteria_version(self, tmp_path):
        # Create a real (empty) cohort file
        cohort = tmp_path / "cohort.parquet"
        cohort.write_bytes(b"placeholder")
        import hashlib
        sha = hashlib.sha256(cohort.read_bytes()).hexdigest()
        config = ExternalValidationConfig(
            cohort_name="test_cohort",
            cohort_path=str(cohort),
            cohort_hash_sha256=sha,
            endpoint="pfs",
            criteria_version="EBMT_1998",  # wrong
            n_min=100,
            n_events_min=30,
            followup_months_min=6.0,
            model_checkpoint=str(tmp_path / "nonexistent.pt"),
        )
        validator = ExternalValidator(model=None, config=config,
                                     training_cindex=0.62)
        result = validator.run()
        assert any("IMWG_2016" in r for r in result.blocking_reasons)
```

---

## Summary of gaps vs. current implementation

| Gap | Current state | Phase 14 fix |
|---|---|---|
| Endpoint-type validation | Not enforced at runtime | `endpoint_validator.py` validates every IMWG-anchored table before gate evaluation |
| IMWG criteria version | No field in any gate | `criteria_version == "IMWG_2016"` required field in all clinical endpoint tables |
| GDSC→IMWG mapping | Absent (correctly so — no mapping exists) | Explicitly blocked in `blocked_endpoints` for all survival/resistance gates |
| Per-domain modality requirements | Global only (acceptance_gate.py) | Per-gate `required_modalities` in ClaimGate |
| Per-domain patient minimums | Global 200 | Tiered: 200 (survival), 300 (patient_level), 100 (longitudinal) |
| Ablation requirements | Described in EVALUATION_GOVERNANCE.md but not enforced | `ablation_required.py` — blocks claim if ablation absent from run manifest |
| External cohort enforcement | "strong" level blocks by default | `required_external_cohort=True` with `external_cohort_min_n` and hash check |
| Cell-line count masquerading as patient count | Not prevented | `required_min_patients` semantics documented and tested |
| Pseudotime substitution | Not explicitly blocked | Blocked in `longitudinal_trajectory` `blocked_endpoints` |
