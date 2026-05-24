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
from typing import Dict, List, Literal, Optional, Tuple

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
    requires_temporal_holdout: bool = False
    requires_external_validation: bool = False
    min_events: Optional[int] = None


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
        requires_temporal_holdout=False,
        requires_external_validation=False,
        min_events=None,
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
        requires_temporal_holdout=False,
        requires_external_validation=False,
        min_events=None,
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
        requires_temporal_holdout=True,
        requires_external_validation=True,
        min_events=30,
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
        requires_temporal_holdout=True,
        requires_external_validation=True,
        min_events=25,
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
        requires_temporal_holdout=True,
        requires_external_validation=False,
        min_events=20,
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
        requires_temporal_holdout=False,
        requires_external_validation=True,
        min_events=15,
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
        requires_temporal_holdout=False,
        requires_external_validation=True,
        min_events=20,
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
        requires_temporal_holdout=False,
        requires_external_validation=False,
        min_events=10,
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
        requires_temporal_holdout=True,
        requires_external_validation=False,
        min_events=15,
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
        requires_temporal_holdout=False,
        requires_external_validation=False,
        min_events=None,
    ),
    # --- Time-to-second-line (TT2L) ------------------------------------------
    "time_to_second_line": EndpointSpec(
        endpoint_name="time_to_second_line",
        endpoint_type="relapse",
        event_time_column="tt2l_time",
        event_observed_column="tt2l_event",
        allowed_claims=[
            "technical", "survival_prediction", "patient_level_clinical_prediction",
            "resistance_emergence",
        ],
        forbidden_claims=["longitudinal_trajectory"],
        notes="Time to second line of therapy. Requires >= 20 observed events.",
        requires_temporal_holdout=True,
        requires_external_validation=True,
        min_events=20,
    ),
}


def lookup_endpoint(name: str) -> Optional[EndpointSpec]:
    """Case-insensitive lookup. Returns ``None`` for unknown endpoints; the
    caller MUST refuse the claim in that case (no default semantics)."""
    if not name:
        return None
    return CANONICAL_ENDPOINTS.get(name.strip().lower())


def validate_endpoint_for_claim(
    endpoint_name: str, claim_level: str
) -> Tuple[bool, str]:
    """Check whether *endpoint_name* is allowed for *claim_level*.

    Enforces three layers of policy:

    1. **Existence** -- the endpoint must be registered.
    2. **Explicit allow/forbid lists** from :class:`EndpointSpec`.
    3. **Hard blocks** that override allow-lists:
       - OS is BLOCKED for ``resistance_emergence`` and
         ``patient_level_clinical_prediction`` regardless of allow-list.
       - Pseudotime (``disease_stage_ordinal`` type) is BLOCKED for
         ``longitudinal_trajectory``, ``resistance_emergence``,
         ``survival_prediction``, and ``patient_level_clinical_prediction``.

    Returns
    -------
    (allowed, reason) : Tuple[bool, str]
        ``allowed`` is True when the claim is permitted; ``reason`` is a
        human-readable explanation of the decision.
    """
    spec = lookup_endpoint(endpoint_name)
    if spec is None:
        return False, (
            f"Endpoint '{endpoint_name}' is not registered. "
            "All endpoints must be registered before any claim is allowed."
        )

    claim = claim_level.strip().lower()

    # --- Hard block: OS must never support resistance or patient-clinical ---
    _OS_BLOCKED_CLAIMS = {"resistance_emergence", "patient_level_clinical_prediction"}
    if spec.endpoint_type == "overall_survival" and claim in _OS_BLOCKED_CLAIMS:
        return False, (
            f"HARD BLOCK: Overall survival endpoint '{spec.endpoint_name}' "
            f"is categorically blocked for claim '{claim}'. OS conflates "
            "all-cause mortality with treatment resistance; use PFS, relapse, "
            "or refractory endpoints instead."
        )

    # --- Hard block: pseudotime must never pose as calendar-time claims -----
    _PSEUDOTIME_BLOCKED_CLAIMS = {
        "longitudinal_trajectory",
        "resistance_emergence",
        "survival_prediction",
        "patient_level_clinical_prediction",
    }
    if spec.endpoint_type == "disease_stage_ordinal" and claim in _PSEUDOTIME_BLOCKED_CLAIMS:
        return False, (
            f"HARD BLOCK: Pseudotime endpoint '{spec.endpoint_name}' "
            f"is categorically blocked for claim '{claim}'. Disease-stage "
            "ordinal pseudo-progression is NOT calendar time and must never "
            "be used for trajectory, survival, or clinical-prediction claims."
        )

    # --- Explicit forbid list (checked before allow list) -------------------
    if claim in [c.strip().lower() for c in spec.forbidden_claims]:
        return False, (
            f"Endpoint '{spec.endpoint_name}' explicitly forbids claim "
            f"'{claim}' per its forbidden_claims list."
        )

    # --- Explicit allow list ------------------------------------------------
    if claim not in [c.strip().lower() for c in spec.allowed_claims]:
        return False, (
            f"Claim '{claim}' is not in the allowed_claims list for "
            f"endpoint '{spec.endpoint_name}'. Allowed claims: "
            f"{spec.allowed_claims}."
        )

    return True, (
        f"Endpoint '{spec.endpoint_name}' permits claim '{claim}'."
    )


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
