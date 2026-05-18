"""
resistancemap/mortfm/registries/claim_gate_registry.py
======================================================
Twelve canonical claim levels, each with the artifacts and gates it
requires, and the artifacts/endpoints that explicitly block it.

This is the programmatic mirror of ``configs/mortfm_gates.yaml``. The YAML
holds *thresholds*; this registry holds *semantics* — what each claim level
means, what it requires, what it forbids. The two together let the run-time
auditor decide whether a specific run earns a specific claim.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Ordered from weakest to strongest. A claim at level i implicitly subsumes
# everything below it that shares its artifact dependencies.
CLAIM_LEVELS: List[str] = [
    "technical",
    "static_drug_response",
    "hematologic_specimen_drug_response",
    "single_cell_state",
    "sequence_aware",
    "pathway_context",
    "drug_target_mechanism",
    "survival_prediction",
    "patient_level_clinical_prediction",
    "longitudinal_trajectory",
    "resistance_emergence",
    "causal_mechanism",
]


@dataclass
class ClaimLevel:
    name: str
    required_artifacts: List[str] = field(default_factory=list)
    required_gates: List[str] = field(default_factory=list)
    required_endpoint_types: List[str] = field(default_factory=list)
    forbidden_endpoint_types: List[str] = field(default_factory=list)
    blocks_until: List[str] = field(default_factory=list)
    description: str = ""


CANONICAL_CLAIM_LEVELS: Dict[str, ClaimLevel] = {
    "technical": ClaimLevel(
        name="technical",
        description="Pipeline runs end-to-end on real data without fabrication. No biological claim.",
    ),
    "static_drug_response": ClaimLevel(
        name="static_drug_response",
        required_artifacts=["A_cellline_foundation"],
        required_gates=["gate_static_drug_response"],
        required_endpoint_types=["drug_response"],
        forbidden_endpoint_types=["overall_survival"],
        description="Static cell-line drug-response ranking (GDSC/PRISM/CCLE). No patient claim.",
    ),
    "hematologic_specimen_drug_response": ClaimLevel(
        name="hematologic_specimen_drug_response",
        required_artifacts=["A_cellline_foundation", "B_beataml_finetuned"],
        required_gates=["gate_beataml_specimen_drug_response", "gate_static_drug_response"],
        required_endpoint_types=["drug_response"],
        description="Specimen-level ex-vivo drug-response ranking on hematologic cancers (BeatAML).",
    ),
    "single_cell_state": ClaimLevel(
        name="single_cell_state",
        required_artifacts=["C_scrna_state_encoder"],
        required_gates=["gate_single_cell_state_pretraining"],
        required_endpoint_types=["disease_stage_ordinal"],
        forbidden_endpoint_types=["overall_survival", "longitudinal_molecular_state"],
        description="Disease-state representation learning on public scRNA atlases. Pseudotime only.",
    ),
    "sequence_aware": ClaimLevel(
        name="sequence_aware",
        required_artifacts=["D_esm2_embeddings"],
        required_gates=["gate_sequence_aware"],
        description="Protein sequences (ESM-2) consumed by downstream heads.",
    ),
    "pathway_context": ClaimLevel(
        name="pathway_context",
        required_artifacts=["D_esm2_embeddings", "A_cellline_foundation"],
        required_gates=["gate_pathway_context"],
        description=(
            "Reactome pathway membership + STRING PPI + drug-target edges form "
            "biological *context* for predictions. NOT a causal mechanism claim."
        ),
    ),
    "drug_target_mechanism": ClaimLevel(
        name="drug_target_mechanism",
        required_artifacts=["D_esm2_embeddings"],
        required_gates=["gate_drug_target_mechanism", "gate_pathway_mechanism"],
        description=(
            "Drug-target ChEMBL coverage above gate threshold. Permits "
            "drug-class attribution, not causal pathway routing."
        ),
    ),
    "survival_prediction": ClaimLevel(
        name="survival_prediction",
        required_gates=["gate_survival_claim"],
        required_endpoint_types=[
            "progression_free_survival", "relapse", "refractory_status", "mrd_conversion",
            "overall_survival",  # OS allowed at this level for *technical* survival head; resistance claim still blocked
        ],
        description="Patient-level survival head technical validation (C-index, Brier, KM).",
    ),
    "patient_level_clinical_prediction": ClaimLevel(
        name="patient_level_clinical_prediction",
        required_artifacts=["A_cellline_foundation", "B_beataml_finetuned"],
        required_gates=["gate_patient_level_clinical_prediction"],
        required_endpoint_types=[
            "progression_free_survival", "relapse", "refractory_status",
            "mrd_conversion",
        ],
        forbidden_endpoint_types=["disease_stage_ordinal", "drug_response"],
        description=(
            "Full clinical-outcome prediction with external/temporal holdout "
            "AND clinical-only baseline. Requires MORT-FM to add calibrated "
            "value beyond a clinical-only Cox."
        ),
    ),
    "longitudinal_trajectory": ClaimLevel(
        name="longitudinal_trajectory",
        required_artifacts=["E_integrated"],
        required_gates=["gate_longitudinal_trajectory"],
        required_endpoint_types=["longitudinal_molecular_state"],
        forbidden_endpoint_types=["disease_stage_ordinal", "overall_survival"],
        description=(
            "Same-patient baseline→followup molecular pairs with real "
            "calendar-time labels. Pseudotime is forbidden as a substitute."
        ),
    ),
    "resistance_emergence": ClaimLevel(
        name="resistance_emergence",
        required_artifacts=["E_integrated"],
        required_gates=["gate_resistance_claim", "gate_time_to_resistance"],
        required_endpoint_types=[
            "progression_free_survival", "relapse", "refractory_status",
            "mrd_conversion", "ex_vivo_resistance_shift",
            "longitudinal_molecular_state",
        ],
        forbidden_endpoint_types=["overall_survival", "drug_response", "disease_stage_ordinal"],
        description=(
            "Time-to-resistance prediction under treatment pressure. OS is "
            "explicitly forbidden as the sole endpoint."
        ),
    ),
    "causal_mechanism": ClaimLevel(
        name="causal_mechanism",
        required_artifacts=["E_integrated", "D_esm2_embeddings"],
        required_gates=["gate_causal_pathway_mechanism"],
        description=(
            "Pathway attribution + drug-target edges + CRISPR-perturbation "
            "consistency + counterfactual edge ablation + negative controls. "
            "The highest bar in the registry."
        ),
    ),
}


def lookup_claim_level(name: str) -> Optional[ClaimLevel]:
    return CANONICAL_CLAIM_LEVELS.get(name.strip().lower())


def write_claim_gate_registry(
    out_path: str = "logs/mortfm/claim_gate_registry.json",
) -> str:
    payload = {
        "claim_levels_ordered": CLAIM_LEVELS,
        "specs": {n: asdict(c) for n, c in CANONICAL_CLAIM_LEVELS.items()},
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote claim-gate registry (%d entries) -> %s",
                len(CANONICAL_CLAIM_LEVELS), out)
    return str(out)
