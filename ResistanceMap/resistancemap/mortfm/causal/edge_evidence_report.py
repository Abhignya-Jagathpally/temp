"""
resistancemap/mortfm/causal/edge_evidence_report.py
======================================================
Compose the three evidence joiners (CRISPR, drug-target, Reactome) into a
single per-edge evidence table + run the top-k pathway enrichment test.
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


def build_edge_evidence_report(
    edge_effects: pd.DataFrame,
    *,
    top_k: int = 20,
) -> dict:
    """Return ``{table, enrichment, summary}`` — the full v17.8 evidence package."""
    out = edge_effects.copy()
    out = join_crispr_essentiality(out)
    out = join_drug_target_support(out)
    out = join_pathway_support(out)
    enrichment = top_k_pathway_enrichment(out, k=top_k)

    top = out.sort_values("delta", key=lambda s: s.abs(), ascending=False).head(top_k)
    summary = {
        "n_edges": int(len(out)),
        "n_target_hgnc_mapped": int(out["target_hgnc"].notna().sum()),
        "top_k": int(top_k),
        "top_k_common_essential_target": int(top["target_is_common_essential"].sum()),
        "top_k_drug_target_either_endpoint": int(top["drug_target_either_endpoint"].sum()),
        "top_k_mean_pathway_count": float(top["target_pathway_count"].mean()),
        "pathway_enrichment": enrichment,
    }
    # Causal-mechanism gate (v17 stricter than v16):
    summary["causal_mechanism_gate_pass"] = bool(
        summary["top_k_common_essential_target"] >= max(2, top_k // 5)
        and summary["top_k_drug_target_either_endpoint"] >= max(2, top_k // 5)
        and enrichment.get("enriched_at_p_lt_0_05", False)
    )
    return {"table": out, "enrichment": enrichment, "summary": summary}
