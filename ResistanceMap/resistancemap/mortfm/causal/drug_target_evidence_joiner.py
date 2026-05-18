"""
resistancemap/mortfm/causal/drug_target_evidence_joiner.py
=============================================================
Join edge effects to ChEMBL drug-target evidence.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def join_drug_target_support(
    edge_effects: pd.DataFrame,
    *,
    drug_targets_csv: str = "data/processed/drugs/drug_targets.csv",
    target_col: str = "target_hgnc",
    source_col: str = "source_hgnc",
) -> pd.DataFrame:
    """Annotate edges with drug-target support (either endpoint).

    Adds:
      drug_target_target_support — True if target_hgnc is an annotated
                                   drug target in ChEMBL
      drug_target_source_support — True if source_hgnc is annotated
      drug_target_either_endpoint — True if either endpoint is annotated
    """
    out = edge_effects.copy()
    if not Path(drug_targets_csv).exists():
        out["drug_target_target_support"] = False
        out["drug_target_source_support"] = False
        out["drug_target_either_endpoint"] = False
        return out
    dt = pd.read_csv(drug_targets_csv)
    if "target_gene" not in dt.columns:
        out["drug_target_target_support"] = False
        out["drug_target_source_support"] = False
        out["drug_target_either_endpoint"] = False
        return out
    targets = set(dt["target_gene"].astype(str).str.upper().dropna())
    out["drug_target_target_support"] = [
        (str(t).upper() in targets) if pd.notna(t) else False
        for t in out[target_col]
    ]
    out["drug_target_source_support"] = [
        (str(s).upper() in targets) if pd.notna(s) else False
        for s in out[source_col]
    ]
    out["drug_target_either_endpoint"] = (
        out["drug_target_target_support"] | out["drug_target_source_support"]
    )
    return out
