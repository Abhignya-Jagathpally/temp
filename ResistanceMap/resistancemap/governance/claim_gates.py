"""
resistancemap/governance/claim_gates.py
=======================================
Top-level claim-gate evaluator. Loads ``configs/mortfm_gates.yaml``, takes a
``RunEvidence`` bundle, and emits a typed report saying which claim levels
are permitted.

Invocation
----------
``release-bouncer`` / the trainer / scripts call ``run_gates(evidence)``
before writing a RUNS.md row.

The gates layer on top of :class:`AcceptanceReport` from
``resistancemap.mortfm.acceptance_gate`` and the
:class:`EndpointSemanticsReport` from
``resistancemap.governance.endpoint_validator``. None of those gates here
override those — they ADD additional data-coverage requirements.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from resistancemap.governance.endpoint_validator import (
    EndpointSemanticsReport,
    validate_endpoint_semantics,
)

logger = logging.getLogger(__name__)


@dataclass
class RunEvidence:
    """Everything the gates need to decide what claims are permitted."""
    endpoint_name: str
    n_patients: int = 0
    n_events: int = 0
    n_longitudinal_pairs: int = 0
    n_unique_patients_with_2_timepoints: int = 0
    has_c_index: bool = False
    has_patient_disjoint_split: bool = False
    has_real_time_units: bool = True
    has_pseudotime_only: bool = False
    has_uniprot_fasta: bool = False
    has_embedding_cache: bool = False
    string_protein_coverage_rate: float = 0.0
    feature_gene_coverage_rate: float = 0.0
    n_models: int = 0
    n_drugs: int = 0
    n_observed_response_rows: int = 0
    model_mapping_rate: float = 0.0
    gene_to_uniprot_coverage: float = 0.0
    drug_to_target_coverage: float = 0.0
    pathway_mapping_coverage: float = 0.0
    has_reactome: bool = False
    has_drug_target_edges: bool = False


@dataclass
class ClaimGateReport:
    endpoint_name: str
    claim_levels_allowed: List[str] = field(default_factory=list)
    claim_levels_blocked: List[str] = field(default_factory=list)
    per_gate: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def load_gates(path: str = "configs/mortfm_gates.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Gates config not found at {p.resolve()}")
    return yaml.safe_load(p.read_text())


def _eval_static_drug_response(g: dict, e: RunEvidence) -> Dict[str, Any]:
    pass_ = (
        e.n_models >= g["min_models"]
        and e.n_drugs >= g["min_drugs"]
        and e.n_observed_response_rows >= g["min_observed_response_rows"]
        and e.model_mapping_rate >= g["min_model_mapping_rate"]
    )
    return {
        "pass": pass_,
        "n_models": e.n_models,
        "n_drugs": e.n_drugs,
        "n_observed_response_rows": e.n_observed_response_rows,
        "model_mapping_rate": e.model_mapping_rate,
        "thresholds": g,
    }


def _eval_pathway_mechanism(g: dict, e: RunEvidence) -> Dict[str, Any]:
    pass_ = (
        e.gene_to_uniprot_coverage >= g["min_gene_to_uniprot_coverage"]
        and e.drug_to_target_coverage >= g["min_drug_to_target_coverage"]
        and e.pathway_mapping_coverage >= g["min_pathway_mapping_coverage"]
        and (e.has_reactome if g["require_reactome"] else True)
        and (e.has_drug_target_edges if g["require_drug_target_edges"] else True)
    )
    return {
        "pass": pass_,
        "gene_to_uniprot_coverage": e.gene_to_uniprot_coverage,
        "drug_to_target_coverage": e.drug_to_target_coverage,
        "pathway_mapping_coverage": e.pathway_mapping_coverage,
        "has_reactome": e.has_reactome,
        "has_drug_target_edges": e.has_drug_target_edges,
        "thresholds": g,
    }


def _eval_survival(g: dict, e: RunEvidence) -> Dict[str, Any]:
    pass_ = (
        e.n_patients >= g["min_patients"]
        and e.n_events >= g["min_events"]
        and (e.has_c_index if g["require_c_index"] else True)
        and (e.has_patient_disjoint_split if g["require_patient_disjoint_split"] else True)
        and (bool(e.endpoint_name) if g["require_endpoint_name"] else True)
    )
    return {
        "pass": pass_,
        "n_patients": e.n_patients,
        "n_events": e.n_events,
        "has_c_index": e.has_c_index,
        "has_patient_disjoint_split": e.has_patient_disjoint_split,
        "endpoint_name": e.endpoint_name,
        "thresholds": g,
    }


def _eval_trajectory(g: dict, e: RunEvidence) -> Dict[str, Any]:
    pass_ = (
        e.n_longitudinal_pairs >= g["min_longitudinal_pairs"]
        and e.n_unique_patients_with_2_timepoints >= g["min_unique_patients_with_2_timepoints"]
        and (e.has_real_time_units if g["require_real_time_units"] else True)
        and (not e.has_pseudotime_only if g["forbid_pseudotime_as_calendar_time"] else True)
    )
    return {
        "pass": pass_,
        "n_longitudinal_pairs": e.n_longitudinal_pairs,
        "n_unique_patients_with_2_timepoints": e.n_unique_patients_with_2_timepoints,
        "has_real_time_units": e.has_real_time_units,
        "has_pseudotime_only": e.has_pseudotime_only,
        "thresholds": g,
    }


def _eval_resistance(g: dict, e: RunEvidence) -> Dict[str, Any]:
    forbid_os = g["forbid_endpoint_only_overall_survival"]
    in_allowed = e.endpoint_name in g["allowed_endpoints"]
    pass_ = in_allowed and not (forbid_os and e.endpoint_name == "overall_survival")
    return {
        "pass": pass_,
        "endpoint_name": e.endpoint_name,
        "in_allowed_endpoints": in_allowed,
        "forbid_os_satisfied": e.endpoint_name != "overall_survival" if forbid_os else True,
        "thresholds": g,
    }


def _eval_sequence_aware(g: dict, e: RunEvidence) -> Dict[str, Any]:
    pass_ = (
        (e.has_uniprot_fasta if g["require_uniprot_fasta"] else True)
        and e.string_protein_coverage_rate >= g["min_string_protein_coverage"]
        and e.feature_gene_coverage_rate >= g["min_feature_gene_coverage"]
        and (e.has_embedding_cache if g["require_embedding_cache"] else True)
    )
    return {
        "pass": pass_,
        "has_uniprot_fasta": e.has_uniprot_fasta,
        "string_protein_coverage_rate": e.string_protein_coverage_rate,
        "feature_gene_coverage_rate": e.feature_gene_coverage_rate,
        "has_embedding_cache": e.has_embedding_cache,
        "thresholds": g,
    }


_EVAL_FUNCS = {
    "gate_static_drug_response": ("static_drug_response", _eval_static_drug_response),
    "gate_pathway_mechanism": ("pathway_mechanism", _eval_pathway_mechanism),
    "gate_survival_claim": ("survival", _eval_survival),
    "gate_trajectory_claim": ("trajectory", _eval_trajectory),
    "gate_resistance_claim": ("resistance", _eval_resistance),
    "gate_sequence_aware_claim": ("sequence_aware", _eval_sequence_aware),
}


def run_gates(
    evidence: RunEvidence,
    *,
    config_path: str = "configs/mortfm_gates.yaml",
    out_json: str = "logs/mortfm/claim_gate_report.json",
) -> ClaimGateReport:
    gates = load_gates(config_path)
    report = ClaimGateReport(endpoint_name=evidence.endpoint_name)
    for key, (claim_name, fn) in _EVAL_FUNCS.items():
        if key not in gates:
            continue
        detail = fn(gates[key], evidence)
        report.per_gate[claim_name] = detail
        if detail["pass"]:
            report.claim_levels_allowed.append(claim_name)
        else:
            report.claim_levels_blocked.append(claim_name)

    # Endpoint validator overlay.
    es = validate_endpoint_semantics(
        endpoint_name=evidence.endpoint_name,
        has_longitudinal_pairs=evidence.n_longitudinal_pairs > 0,
        n_longitudinal_pairs=evidence.n_longitudinal_pairs,
    )
    report.per_gate["endpoint_semantics"] = asdict(es)

    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info(
        "Claim gate report: %d allowed, %d blocked -> %s",
        len(report.claim_levels_allowed), len(report.claim_levels_blocked), out,
    )
    return report
