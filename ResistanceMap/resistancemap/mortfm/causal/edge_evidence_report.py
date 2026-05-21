"""
resistancemap/mortfm/causal/edge_evidence_report.py
======================================================
Compose the three evidence joiners (CRISPR, drug-target, Reactome) into a
single per-edge evidence table + run the top-k pathway enrichment test.

v19 Bug B13 — the causal-mechanism gate was originally a *hard AND* across
all three channels (CRISPR essential AND drug-target AND Reactome
enrichment). That made the gate impossible to pass when one channel
abstained (e.g. a target with no FDA drug yet), even if the other two
channels strongly supported the edge. Phase-8 §9 specifies a **k-of-N**
policy instead: the gate passes when at least ``k`` of the ``N=3`` channels
support the edge. Default ``k=2`` — two-of-three independent lines of
evidence — and the default is parameterisable so the report can be
re-rendered at any stricter or looser threshold.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from resistancemap.mortfm.causal.crispr_evidence_joiner import join_crispr_essentiality
from resistancemap.mortfm.causal.drug_target_evidence_joiner import join_drug_target_support
from resistancemap.mortfm.causal.pathway_evidence_joiner import (
    join_pathway_support, top_k_pathway_enrichment,
)

logger = logging.getLogger(__name__)


#: Default k for the k-of-N policy; ``2`` matches Phase-8 design §9.
DEFAULT_K_OF_N: int = 2

#: Total number of independent evidence channels (N).
N_EVIDENCE_CHANNELS: int = 3


def _evaluate_k_of_n_gate(
    crispr_support: bool,
    drug_target_support: bool,
    reactome_support: bool,
    k: int,
) -> dict:
    """Return the k-of-N gate verdict plus a per-channel breakdown.

    The returned dict carries:
      * ``n_supporting_channels`` — count of channels that voted "support"
      * ``gate_passes``           — n_supporting_channels >= k
      * ``k``                     — threshold actually applied
      * ``channels_supported``    — list of channel names that supported
      * ``channels_not_supported``— list of channel names that did NOT support
    """
    channel_votes = {
        "crispr_common_essential": bool(crispr_support),
        "drug_target_either_endpoint": bool(drug_target_support),
        "reactome_pathway_enrichment": bool(reactome_support),
    }
    n_supporting = int(sum(channel_votes.values()))
    return {
        "k": int(k),
        "n_supporting_channels": n_supporting,
        "gate_passes": bool(n_supporting >= k),
        "channels_supported": [name for name, v in channel_votes.items() if v],
        "channels_not_supported": [name for name, v in channel_votes.items() if not v],
    }


def evaluate_edge_evidence_gate(
    *,
    crispr_support: bool,
    drug_target_support: bool,
    reactome_support: bool,
    k: int = DEFAULT_K_OF_N,
) -> dict:
    """Public helper — evaluate the k-of-N causal-mechanism gate on a single
    evidence row.

    Exposed separately so tests + downstream code can re-use the same
    policy without re-implementing it. The aggregator below calls this
    with the cohort-level channel verdicts.
    """
    return _evaluate_k_of_n_gate(
        crispr_support=crispr_support,
        drug_target_support=drug_target_support,
        reactome_support=reactome_support,
        k=k,
    )


def build_edge_evidence_report(
    edge_effects: pd.DataFrame,
    *,
    top_k: int = 20,
    k: int = DEFAULT_K_OF_N,
) -> dict:
    """Return ``{table, enrichment, summary}`` — the full v17.8 evidence package.

    The ``k`` parameter controls the k-of-N causal-mechanism gate (default
    ``2`` per Phase-8 §9). Pass ``k=3`` to require ALL channels (the old
    hard-AND behaviour) or ``k=1`` for any-of-three (loosest).
    """
    out = edge_effects.copy()
    out = join_crispr_essentiality(out)
    out = join_drug_target_support(out)
    out = join_pathway_support(out)
    enrichment = top_k_pathway_enrichment(out, k=top_k)

    top = out.sort_values("delta", key=lambda s: s.abs(), ascending=False).head(top_k)

    # Cohort-level channel verdicts (same thresholds as the legacy gate).
    crispr_threshold = max(2, top_k // 5)
    drug_threshold = max(2, top_k // 5)
    crispr_count = int(top["target_is_common_essential"].sum())
    drug_count = int(top["drug_target_either_endpoint"].sum())
    crispr_support = bool(crispr_count >= crispr_threshold)
    drug_support = bool(drug_count >= drug_threshold)
    reactome_support = bool(enrichment.get("enriched_at_p_lt_0_05", False))

    gate = _evaluate_k_of_n_gate(
        crispr_support=crispr_support,
        drug_target_support=drug_support,
        reactome_support=reactome_support,
        k=k,
    )

    summary = {
        "n_edges": int(len(out)),
        "n_target_hgnc_mapped": int(out["target_hgnc"].notna().sum()),
        "top_k": int(top_k),
        "top_k_common_essential_target": crispr_count,
        "top_k_drug_target_either_endpoint": drug_count,
        "top_k_mean_pathway_count": float(top["target_pathway_count"].mean()),
        "pathway_enrichment": enrichment,
        # v19 — k-of-N gate (replaces hard-AND from v17.8).
        "causal_mechanism_gate_k": int(k),
        "causal_mechanism_gate_n_total": int(N_EVIDENCE_CHANNELS),
        "causal_mechanism_gate_n_supporting": gate["n_supporting_channels"],
        "causal_mechanism_gate_pass": gate["gate_passes"],
        "causal_mechanism_gate_channels_supported": gate["channels_supported"],
        "causal_mechanism_gate_channels_not_supported": gate["channels_not_supported"],
    }

    # Build a human-readable verdict string enumerating which channels
    # supported (vs which abstained / did not support). Stored on the
    # summary AND logged at INFO so anyone reading the run log can see
    # exactly why the gate passed or failed.
    verdict_line = (
        f"k-of-N gate (k={k}, N={N_EVIDENCE_CHANNELS}): "
        f"{gate['n_supporting_channels']} supporting channels "
        f"→ {'PASS' if gate['gate_passes'] else 'FAIL'}. "
        f"Supported: {gate['channels_supported'] or 'NONE'}. "
        f"Not supported: {gate['channels_not_supported'] or 'NONE'}."
    )
    summary["causal_mechanism_gate_verdict"] = verdict_line
    logger.info(verdict_line)

    return {"table": out, "enrichment": enrichment, "summary": summary}
