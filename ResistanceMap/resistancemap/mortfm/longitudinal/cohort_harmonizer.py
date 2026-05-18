"""
resistancemap/mortfm/longitudinal/cohort_harmonizer.py
========================================================
Cross-cohort feature-space harmonisation.

Different cohorts use different feature ID types (ENSG, HGNC, UniProt).
The harmoniser routes each cohort's matrix through the canonical
Block-A feature space (rna_block_a_top2000) so a single trainer can
consume MMRF + PADIMAC + BeatAML latents on the same axis.

For ENSG -> HGNC conversion we reuse ``data/processed/mmrf_ensembl_to_symbol.tsv``.
For HGNC -> Block-A feature subset we reuse
:func:`resistancemap.data.feature_alignment.align_to_reference_features`.

This module is *introspective*; it does not write out new parquet files
itself. Use :mod:`scripts.mortfm_cohort_harmonize` to materialise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from resistancemap.data.feature_alignment import align_to_reference_features

logger = logging.getLogger(__name__)


def _ensg_to_hgnc_lookup(
    map_tsv: str = "data/processed/mmrf_ensembl_to_symbol.tsv",
) -> Dict[str, str]:
    p = Path(map_tsv)
    if not p.exists():
        return {}
    df = pd.read_csv(p, sep="\t")
    col_map = {c.lower(): c for c in df.columns}
    ens_col = (col_map.get("ensembl_id") or col_map.get("ensembl_gene_id")
               or col_map.get("ensembl") or col_map.get("ensembl_base"))
    sym_col = (col_map.get("gene_symbol") or col_map.get("symbol")
               or col_map.get("hgnc_symbol") or col_map.get("gene_name"))
    if not (ens_col and sym_col):
        logger.warning("ENSG->HGNC map columns not found in %s; got %s", p, list(df.columns))
        return {}
    df = df.dropna(subset=[ens_col, sym_col])
    df[ens_col] = df[ens_col].astype(str).str.split(".").str[0]
    return dict(zip(df[ens_col], df[sym_col]))


def harmonize_to_block_a(
    cohort_expression: pd.DataFrame,
    cohort_feature_id_type: str,
    block_a_reference_features: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Project a cohort's expression matrix onto the Block-A feature space.

    Returns ``(aligned, mask, report)`` matching
    :func:`align_to_reference_features` semantics.
    """
    expr = cohort_expression
    if cohort_feature_id_type.upper() == "ENSG":
        e2h = _ensg_to_hgnc_lookup()
        if not e2h:
            raise FileNotFoundError(
                "ENSG cohort harmonisation requires "
                "data/processed/mmrf_ensembl_to_symbol.tsv"
            )
        cols = expr.columns.astype(str).str.split(".").str[0]
        mapped = cols.map(e2h)
        # Aggregate duplicates by sum (counts) — drop unmapped.
        expr = expr.T
        expr.index = mapped
        expr = expr.dropna(axis=0)
        expr = expr.groupby(level=0).sum().T
    elif cohort_feature_id_type.upper() in {"HGNC", "SYMBOL"}:
        # Already symbol-keyed; nothing to do.
        pass
    else:
        raise ValueError(f"Unsupported feature_id_type {cohort_feature_id_type!r}")
    return align_to_reference_features(expr, block_a_reference_features)
