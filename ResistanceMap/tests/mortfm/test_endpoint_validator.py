"""Tests for resistancemap.governance.endpoint_validator + claim_gates."""

from __future__ import annotations

import pytest

from resistancemap.governance.claim_gates import RunEvidence, run_gates
from resistancemap.governance.endpoint_validator import (
    EndpointSemanticsError,
    VALID_ENDPOINTS,
    require_resistance_endpoint,
    require_trajectory_endpoint,
    validate_endpoint_semantics,
)


def test_unknown_endpoint_raises():
    with pytest.raises(ValueError, match="Unknown endpoint"):
        validate_endpoint_semantics(endpoint_name="time_to_grant_funding")


def test_os_blocks_resistance_and_trajectory():
    rep = validate_endpoint_semantics(endpoint_name="overall_survival",
                                       has_longitudinal_pairs=True, n_longitudinal_pairs=1000)
    assert rep.survival_claim_allowed
    assert not rep.resistance_claim_allowed
    assert not rep.trajectory_claim_allowed
    with pytest.raises(EndpointSemanticsError):
        require_resistance_endpoint(rep)
    with pytest.raises(EndpointSemanticsError):
        require_trajectory_endpoint(rep)


def test_pfs_allows_resistance_with_warning():
    rep = validate_endpoint_semantics(endpoint_name="progression_free_survival",
                                       has_longitudinal_pairs=True, n_longitudinal_pairs=200)
    assert rep.resistance_claim_allowed
    assert rep.trajectory_claim_allowed
    assert any("PFS proxies resistance only when explicitly framed" in r for r in rep.blocking_reasons)


def test_drug_resistance_label_requires_longitudinal_for_trajectory():
    rep_low = validate_endpoint_semantics(endpoint_name="drug_resistance_label",
                                           has_longitudinal_pairs=True, n_longitudinal_pairs=10)
    assert rep_low.resistance_claim_allowed
    assert not rep_low.trajectory_claim_allowed

    rep_hi = validate_endpoint_semantics(endpoint_name="drug_resistance_label",
                                          has_longitudinal_pairs=True, n_longitudinal_pairs=200)
    assert rep_hi.resistance_claim_allowed
    assert rep_hi.trajectory_claim_allowed


def test_auc_is_static_drug_response():
    rep = validate_endpoint_semantics(endpoint_name="auc")
    assert rep.claim_level == "static_drug_response"
    assert rep.drug_response_claim_allowed
    assert not rep.resistance_claim_allowed
    assert not rep.trajectory_claim_allowed


def test_all_valid_endpoints_have_a_claim_level():
    for ep in VALID_ENDPOINTS:
        rep = validate_endpoint_semantics(endpoint_name=ep)
        assert rep.claim_level in {
            "survival_head_technical_validation", "resistance_claim",
            "static_drug_response", "none",
        }


def test_claim_gates_block_os_resistance_combo(tmp_path):
    evidence = RunEvidence(
        endpoint_name="overall_survival",
        n_patients=500, n_events=200, n_longitudinal_pairs=500,
        has_c_index=True, has_patient_disjoint_split=True,
        has_uniprot_fasta=True, has_embedding_cache=True,
        string_protein_coverage_rate=0.9, feature_gene_coverage_rate=0.9,
        has_reactome=True, has_drug_target_edges=True,
        gene_to_uniprot_coverage=0.9, drug_to_target_coverage=0.9, pathway_mapping_coverage=0.9,
        n_models=500, n_drugs=100, n_observed_response_rows=20000, model_mapping_rate=0.95,
    )
    rep = run_gates(evidence, out_json=str(tmp_path / "gate.json"))
    # OS endpoint still blocks resistance even when every other gate would pass.
    assert "resistance" in rep.claim_levels_blocked
