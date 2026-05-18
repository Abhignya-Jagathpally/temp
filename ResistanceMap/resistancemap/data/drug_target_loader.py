"""
resistancemap/data/drug_target_loader.py
========================================
Load drug-target edges from DrugBank / DGIdb / ChEMBL CSV dumps.

Emits an edge list ``[(drug_name, target_gene, edge_weight)]`` plus optional
target *protein* identifiers. Used by
:mod:`resistancemap.data.biological_graph_builder` to add ``drug -> protein``
edges to the heterogeneous biological graph.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_drug_target_edges(
    path: str,
    *,
    drug_col: str = "drug_name",
    target_col: str = "gene_symbol",
    weight_col: str = "evidence_score",
    default_weight: float = 1.0,
) -> List[Tuple[str, str, float]]:
    """Load drug-target edges from a tabular file.

    Parameters
    ----------
    path
        CSV/TSV with at minimum ``drug_col`` and ``target_col`` columns.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Drug-target file not found: {p.resolve()}")
    sep = "\t" if p.suffix in {".tsv", ".txt"} else ","
    df = pd.read_csv(p, sep=sep, low_memory=False)
    if drug_col not in df.columns or target_col not in df.columns:
        raise KeyError(
            f"Expected columns {drug_col!r}, {target_col!r} in {p}. "
            f"Available: {list(df.columns)[:20]}"
        )
    edges: List[Tuple[str, str, float]] = []
    for _, row in df.iterrows():
        drug = str(row[drug_col]).strip()
        target = str(row[target_col]).strip()
        if not drug or not target:
            continue
        weight = (
            float(row[weight_col]) if weight_col in df.columns and pd.notna(row[weight_col])
            else default_weight
        )
        edges.append((drug, target, weight))
    logger.info("Loaded %d drug-target edges from %s", len(edges), p.name)
    return edges
