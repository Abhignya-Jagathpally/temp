"""
resistancemap/mortfm/registries/endpoint_registry.py
====================================================
Canonical endpoint definitions. Each endpoint is *typed* and carries an
explicit allow-list and forbid-list of claim levels.

This is the registry that prevents:
  * OS being silently reused as a resistance endpoint
  * IC50 being silently reused as a resistance-emergence endpoint
  * pseudotime being silently reused as calendar time

Honest behaviour
----------------
* Every endpoint MUST be looked up via :func:`lookup_endpoint` before any
  claim is allowed against it. If the lookup misses, the caller must refuse
  the claim — there is no "default" semantics.
* The allowed_claims list is non-empty for every endpoint; if a claim level
  is not listed, it is implicitly forbidden.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


EndpointType = Literal[
    "drug_response",
    "overall_survival",
    "progression_free_survival",
    "relapse",
    "refractory_status",
    "mrd_conversion",
    "ex_vivo_resistance_shift",
    "longitudinal_molecular_state",
    "disease_stage_ordinal",  # pseudotime stage only — never calendar time
]


@dataclass
class EndpointSpec:
    endpoint_name: str
    endpoint_type: EndpointType
    event_time_column: Optional[str] = None
    event_observed_column: Optional[str] = None
    allowed_claims: List[str] = field(default_factory=list)
    forbidden_claims: List[str] = field(default_factory=list)
    notes: str = ""


# Canonical endpoints. Names are lowercase; lookup is case-insensitive.
CANONICAL_ENDPOINTS: Dict[str, EndpointSpec] = {
    # --- Drug-response (static / ex-vivo) -----------------------------------
    "ln_ic50": EndpointSpec(
        endpoint_name="ln_ic50",
        endpoint_type="drug_response",
        allowed_claims=["technical", "static_drug_response"],
        forbidden_claims=[
            "longitudinal_trajectory", "resistance_emergence",
            "survival_prediction", "patient_level_clinical_prediction",
        ],
        notes="Static cell-line IC50. NOT a resistance endpoint.",
    ),
    "auc": EndpointSpec(
        endpoint_name="auc",
        endpoint_type="drug_response",
        allowed_claims=[
            "technical", "static_drug_response", "hematologic_specimen_drug_response",
        ],
        forbidden_claims=[
            "longitudinal_trajectory", "resistance_emergence",
            "survival_prediction", "patient_level_clinical_prediction",
        ],
        notes="Ex vivo AUC (BeatAML / PRISM). Specimen-level static response only.",
    ),
    # --- Survival -----------------------------------------------------------
    "overall_survival": EndpointSpec(
        endpoint_name="overall_survival",
        endpoint_type="overall_survival",
        event_time_column="os_time",
        event_observed_column="os_event",
        allowed_claims=["technical", "survival_prediction"],
        forbidden_claims=[
            "longitudinal_trajectory",
            "resistance_emergence",
            "patient_level_clinical_prediction",  # blocked unless a clinical-only baseline is beaten
        ],
        notes=(
            "OS is NOT a resistance endpoint. Useful for survival-head "
            "technical validation only — any time-to-resistance or "
            "resistance-emergence claim built on OS alone is forbidden."
        ),
    ),
    "progression_free_survival": EndpointSpec(
        endpoint_name="progression_free_survival",
        endpoint_type="progression_free_survival",
        event_time_column="pfs_time",
        event_observed_column="pfs_event",
        allowed_claims=[
            "technical", "survival_prediction", "patient_level_clinical_prediction",
            "resistance_emergence",
        ],
        forbidden_claims=["longitudinal_trajectory"],
        notes="PFS supports time-to-resistance under competing-risk framing.",
    ),
    "relapse_time": EndpointSpec(
        endpoint_name="relapse_time",
        endpoint_type="relapse",
        event_time_column="relapse_time",
        event_observed_column="relapse_event",
        allowed_claims=[
            "technical", "survival_prediction", "patient_level_clinical_prediction",
            "resistance_emergence",
        ],
        forbidden_claims=["longitudinal_trajectory"],
    ),
    "refractory_status": EndpointSpec(
        endpoint_name="refractory_status",
        endpoint_type="refractory_status",
        event_time_column="refractory_time",
        event_observed_column="refractory_event",
        allowed_claims=[
            "technical", "patient_level_clinical_prediction", "resistance_emergence",
        ],
        forbidden_claims=["longitudinal_trajectory"],
    ),
    "mrd_conversion": EndpointSpec(
        endpoint_name="mrd_conversion",
        endpoint_type="mrd_conversion",
        event_time_column="mrd_conversion_time",
        event_observed_column="mrd_event",
        allowed_claims=[
            "technical", "patient_level_clinical_prediction", "resistance_emergence",
        ],
        forbidden_claims=["longitudinal_trajectory"],
    ),
    # --- Ex-vivo molecular shift -------------------------------------------
    "ex_vivo_resistance_shift": EndpointSpec(
        endpoint_name="ex_vivo_resistance_shift",
        endpoint_type="ex_vivo_resistance_shift",
        allowed_claims=["technical", "resistance_emergence"],
        forbidden_claims=["longitudinal_trajectory"],
        notes=(
            "Pre→post drug treatment IC50 shift on paired specimens. Allowed "
            "for resistance_emergence when paired ex-vivo measurements exist."
        ),
    ),
    # --- Longitudinal molecular --------------------------------------------
    "longitudinal_molecular_state": EndpointSpec(
        endpoint_name="longitudinal_molecular_state",
        endpoint_type="longitudinal_molecular_state",
        allowed_claims=["technical", "longitudinal_trajectory", "resistance_emergence"],
        forbidden_claims=[],
        notes=(
            "True same-patient baseline-and-followup molecular pairs. "
            "Requires real calendar-time labels — pseudotime is forbidden."
        ),
    ),
    # --- Pseudotime stage (NOT calendar time) ------------------------------
    "disease_stage_pseudotime": EndpointSpec(
        endpoint_name="disease_stage_pseudotime",
        endpoint_type="disease_stage_ordinal",
        allowed_claims=["technical", "single_cell_state"],
        forbidden_claims=[
            "longitudinal_trajectory", "resistance_emergence",
            "survival_prediction", "patient_level_clinical_prediction",
        ],
        notes=(
            "Ordinal disease stage (HD < MGUS < SMM < MM). NEVER use as "
            "calendar time. NEVER source a trajectory claim from this."
        ),
    ),
}


def lookup_endpoint(name: str) -> Optional[EndpointSpec]:
    """Case-insensitive lookup. Returns ``None`` for unknown endpoints; the
    caller MUST refuse the claim in that case (no default semantics)."""
    if not name:
        return None
    return CANONICAL_ENDPOINTS.get(name.strip().lower())


def write_endpoint_registry(
    out_path: str = "logs/mortfm/endpoint_registry.json",
) -> str:
    payload = {n: asdict(e) for n, e in CANONICAL_ENDPOINTS.items()}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote endpoint registry (%d entries) -> %s", len(payload), out)
    return str(out)
