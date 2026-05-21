"""tests/mortfm/test_edge_evidence_kofn.py

Regression test for Bug B13: the causal-mechanism gate previously hard-
AND'd CRISPR essentiality, drug-target overlap, AND Reactome enrichment,
which made the gate unreachable whenever any one channel abstained
(typical for novel targets without an FDA drug).

Phase-8 §9 specifies a k-of-N gate; default ``k=2``. This test exercises
the gate's policy logic via the ``evaluate_edge_evidence_gate`` public
helper for fast, dataframe-free assertions.
"""

from __future__ import annotations

import pytest

from resistancemap.mortfm.causal.edge_evidence_report import (
    DEFAULT_K_OF_N,
    N_EVIDENCE_CHANNELS,
    evaluate_edge_evidence_gate,
)


def test_default_k_is_two():
    assert DEFAULT_K_OF_N == 2
    assert N_EVIDENCE_CHANNELS == 3


def test_two_channels_supporting_passes_at_k2():
    """Exactly 2-of-3 → passes at k=2."""
    result = evaluate_edge_evidence_gate(
        crispr_support=True,
        drug_target_support=True,
        reactome_support=False,
        k=2,
    )
    assert result["gate_passes"] is True
    assert result["n_supporting_channels"] == 2
    assert set(result["channels_supported"]) == {
        "crispr_common_essential",
        "drug_target_either_endpoint",
    }
    assert result["channels_not_supported"] == ["reactome_pathway_enrichment"]


def test_two_channels_supporting_fails_at_k3():
    """Same row, k=3 (hard-AND) → fails."""
    result = evaluate_edge_evidence_gate(
        crispr_support=True,
        drug_target_support=True,
        reactome_support=False,
        k=3,
    )
    assert result["gate_passes"] is False
    assert result["n_supporting_channels"] == 2


def test_one_channel_supporting_fails_at_k2():
    """1-of-3 → fails at default k=2."""
    result = evaluate_edge_evidence_gate(
        crispr_support=False,
        drug_target_support=True,
        reactome_support=False,
        k=2,
    )
    assert result["gate_passes"] is False
    assert result["n_supporting_channels"] == 1


def test_one_channel_supporting_passes_at_k1():
    """1-of-3 passes at the loosest threshold k=1."""
    result = evaluate_edge_evidence_gate(
        crispr_support=False,
        drug_target_support=False,
        reactome_support=True,
        k=1,
    )
    assert result["gate_passes"] is True
    assert result["n_supporting_channels"] == 1


def test_all_three_pass_at_k3():
    result = evaluate_edge_evidence_gate(
        crispr_support=True,
        drug_target_support=True,
        reactome_support=True,
        k=3,
    )
    assert result["gate_passes"] is True
    assert result["n_supporting_channels"] == 3
    assert result["channels_not_supported"] == []


def test_all_three_fail_at_k1():
    """Sanity: with 0 supports, even k=1 fails."""
    result = evaluate_edge_evidence_gate(
        crispr_support=False,
        drug_target_support=False,
        reactome_support=False,
        k=1,
    )
    assert result["gate_passes"] is False
    assert result["n_supporting_channels"] == 0
