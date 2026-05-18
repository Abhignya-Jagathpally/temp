"""
resistancemap/governance/endpoint_validator.py
==============================================
Endpoint-semantics validator.

The single most common honesty failure in MM/AML ML is calling an OS-based
model a "resistance predictor" or a "trajectory forecaster". This module
enforces an explicit endpoint taxonomy and refuses claims that conflate
distinct biology.

Allowed claim levels per endpoint
---------------------------------
* ``overall_survival``                   -> survival_head_technical_validation
* ``progression_free_survival``          -> survival_head_technical_validation +
                                            (resistance proxy allowed if explicitly framed)
* ``relapse_time``                       -> resistance_claim_allowed
* ``refractory_status``                  -> resistance_claim_allowed
* ``mrd_conversion``                     -> resistance_claim_allowed
* ``drug_resistance_label``              -> resistance_claim_allowed
* ``ex_vivo_resistance_shift``           -> resistance_claim_allowed

The trainer calls :func:`validate_endpoint_semantics` before writing a RUNS
row. ``release-bouncer`` calls it before any paper claim.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


VALID_ENDPOINTS = (
    "overall_survival",
    "progression_free_survival",
    "relapse_time",
    "refractory_status",
    "mrd_conversion",
    "drug_resistance_label",
    "ex_vivo_resistance_shift",
    "ic50",
    "auc",
    "viability",
    "drug_response_ranking",
)


_RESISTANCE_ALLOWED = {
    "relapse_time",
    "refractory_status",
    "mrd_conversion",
    "drug_resistance_label",
    "ex_vivo_resistance_shift",
}


_TRAJECTORY_REQUIRES_LONGITUDINAL = True


@dataclass
class EndpointSemanticsReport:
    endpoint_name: str
    endpoint_type: str          # one of {clinical_survival, drug_response, resistance_label}
    event_observed_column: str
    event_time_column: str
    claim_level: str            # survival_head_technical_validation | resistance_claim | static_drug_response | none
    resistance_claim_allowed: bool
    trajectory_claim_allowed: bool
    survival_claim_allowed: bool
    drug_response_claim_allowed: bool
    blocking_reasons: list = field(default_factory=list)


def validate_endpoint_semantics(
    *,
    endpoint_name: str,
    event_observed_column: Optional[str] = None,
    event_time_column: Optional[str] = None,
    has_longitudinal_pairs: bool = False,
    n_longitudinal_pairs: int = 0,
    min_longitudinal_pairs: int = 100,
) -> EndpointSemanticsReport:
    """Return a typed semantics report. Raises :class:`ValueError` on unknown endpoint."""
    if endpoint_name not in VALID_ENDPOINTS:
        raise ValueError(
            f"Unknown endpoint {endpoint_name!r}; expected one of {VALID_ENDPOINTS}"
        )

    endpoint_type = _endpoint_type(endpoint_name)
    rep = EndpointSemanticsReport(
        endpoint_name=endpoint_name,
        endpoint_type=endpoint_type,
        event_observed_column=event_observed_column or "",
        event_time_column=event_time_column or "",
        claim_level="none",
        resistance_claim_allowed=False,
        trajectory_claim_allowed=False,
        survival_claim_allowed=False,
        drug_response_claim_allowed=False,
    )

    if endpoint_name == "overall_survival":
        rep.claim_level = "survival_head_technical_validation"
        rep.survival_claim_allowed = True
        rep.blocking_reasons.append(
            "OS is NOT a resistance endpoint; resistance_claim_allowed=False"
        )
    elif endpoint_name == "progression_free_survival":
        rep.claim_level = "survival_head_technical_validation"
        rep.survival_claim_allowed = True
        rep.resistance_claim_allowed = True
        rep.blocking_reasons.append(
            "PFS proxies resistance only when explicitly framed; do not claim 'time-to-resistance' without "
            "validating against an exact resistance/relapse label."
        )
    elif endpoint_name in _RESISTANCE_ALLOWED:
        rep.claim_level = "resistance_claim"
        rep.resistance_claim_allowed = True
        rep.survival_claim_allowed = True
    elif endpoint_name in {"ic50", "auc", "viability", "drug_response_ranking"}:
        rep.claim_level = "static_drug_response"
        rep.drug_response_claim_allowed = True
        rep.blocking_reasons.append(
            "Static drug-response endpoint does NOT support resistance or trajectory claims"
        )

    if _TRAJECTORY_REQUIRES_LONGITUDINAL:
        if has_longitudinal_pairs and n_longitudinal_pairs >= min_longitudinal_pairs:
            rep.trajectory_claim_allowed = rep.resistance_claim_allowed
        elif rep.resistance_claim_allowed:
            rep.blocking_reasons.append(
                f"trajectory_claim requires >= {min_longitudinal_pairs} longitudinal pairs; "
                f"have {n_longitudinal_pairs}."
            )

    logger.info("Endpoint semantics: %s -> claim_level=%s", endpoint_name, rep.claim_level)
    return rep


def _endpoint_type(name: str) -> str:
    if name in {"overall_survival", "progression_free_survival", "relapse_time"}:
        return "clinical_survival"
    if name in {"refractory_status", "drug_resistance_label", "mrd_conversion"}:
        return "resistance_label"
    if name in {"ex_vivo_resistance_shift"}:
        return "resistance_shift"
    return "drug_response"


def write_endpoint_semantics_report(
    report: EndpointSemanticsReport,
    out_json: str = "logs/mortfm/endpoint_semantics_report.json",
) -> str:
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report.__dict__, f, indent=2)
    logger.info("Wrote endpoint semantics report -> %s", out)
    return str(out)


class EndpointSemanticsError(RuntimeError):
    """Raised by helpers that refuse to proceed when an endpoint mismatch is detected."""


def require_resistance_endpoint(report: EndpointSemanticsReport) -> None:
    """Refuse to proceed with a 'resistance' claim if the endpoint doesn't support it."""
    if not report.resistance_claim_allowed:
        raise EndpointSemanticsError(
            f"Endpoint {report.endpoint_name!r} does NOT support a resistance claim "
            f"(claim_level={report.claim_level}; blocking_reasons={report.blocking_reasons})."
        )


def require_trajectory_endpoint(report: EndpointSemanticsReport) -> None:
    """Refuse to proceed with a 'trajectory' claim if longitudinal pairs are insufficient."""
    if not report.trajectory_claim_allowed:
        raise EndpointSemanticsError(
            f"Endpoint {report.endpoint_name!r} does NOT support a trajectory claim "
            f"(blocking_reasons={report.blocking_reasons})."
        )
