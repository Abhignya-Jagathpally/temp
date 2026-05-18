"""
resistancemap/mortfm/causal/crispr_evidence_joiner.py
========================================================
Join counterfactual edge effects to DepMap CRISPR essentiality.

This is the *evidence-side* of the causal validator. The validator
decides PASS/FAIL given a join report; this module produces the join.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def join_crispr_essentiality(
    edge_effects: pd.DataFrame,
    *,
    crispr_summary_csv: str = "data/processed/crispr/essentiality_summary.csv",
    target_col: str = "target_hgnc",
) -> pd.DataFrame:
    """Annotate every edge with the DepMap CRISPR essentiality of its target."""
    if not Path(crispr_summary_csv).exists():
        logger.warning("CRISPR summary missing at %s", crispr_summary_csv)
        edge_effects = edge_effects.copy()
        edge_effects["depmap_target_gene_effect_median"] = float("nan")
        edge_effects["target_frac_essential"] = float("nan")
        edge_effects["target_is_common_essential"] = False
        return edge_effects
    cr = pd.read_csv(crispr_summary_csv)
    cr["sym"] = cr["gene_symbol"].astype(str).str.upper()
    med = dict(zip(cr["sym"], cr["median_effect"]))
    frac = dict(zip(cr["sym"], cr["frac_essential"]))
    out = edge_effects.copy()
    out["depmap_target_gene_effect_median"] = [
        med.get(str(t).upper()) if pd.notna(t) else None
        for t in out[target_col]
    ]
    out["target_frac_essential"] = [
        frac.get(str(t).upper()) if pd.notna(t) else None
        for t in out[target_col]
    ]
    out["target_is_common_essential"] = (
        out["target_frac_essential"].fillna(0) >= 0.9
    )
    return out
