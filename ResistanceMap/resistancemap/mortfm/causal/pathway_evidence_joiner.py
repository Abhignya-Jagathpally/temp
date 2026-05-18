"""
resistancemap/mortfm/causal/pathway_evidence_joiner.py
==========================================================
Annotate edges with Reactome pathway membership of source + target.

The Reactome membership table at
``data/processed/graphs/reactome_membership.parquet`` is keyed by
``source_id`` (an internal STRING alias). We bridge through the
:class:`IDHarmonizer` to associate each edge with the **set of pathways**
its target gene participates in.

The top-pathway enrichment test for top-k causal edges is implemented in
:func:`top_k_pathway_enrichment`.
"""

from __future__ import annotations

import logging
import math
import random as _random
from pathlib import Path
from typing import Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


def _build_hgnc_to_pathways(
    reactome_parquet: str = "data/processed/graphs/reactome_membership.parquet",
) -> dict[str, Set[str]]:
    if not Path(reactome_parquet).exists():
        return {}
    r = pd.read_parquet(reactome_parquet)
    if "source_id" not in r.columns or "pathway_name" not in r.columns:
        return {}
    try:
        from resistancemap.mortfm.graph import IDHarmonizer
        h = IDHarmonizer.build_or_load()
    except Exception as exc:
        logger.warning("IDHarmonizer unavailable in pathway evidence joiner: %s", exc)
        return {}
    out: dict[str, Set[str]] = {}
    for _, row in r.iterrows():
        hgnc_list = h.any_alias_to_hgnc([row["source_id"]])
        hgnc = hgnc_list[0]
        if hgnc is None:
            continue
        out.setdefault(hgnc.upper(), set()).add(str(row["pathway_name"]))
    return out


_PATHWAY_CACHE: Optional[dict[str, Set[str]]] = None


def join_pathway_support(
    edge_effects: pd.DataFrame,
    *,
    target_col: str = "target_hgnc",
    reactome_parquet: str = "data/processed/graphs/reactome_membership.parquet",
) -> pd.DataFrame:
    """Add ``target_pathway_count`` + ``target_top_pathway`` columns."""
    global _PATHWAY_CACHE
    if _PATHWAY_CACHE is None:
        _PATHWAY_CACHE = _build_hgnc_to_pathways(reactome_parquet)
    pmap = _PATHWAY_CACHE or {}
    out = edge_effects.copy()
    out["target_pathway_count"] = [
        len(pmap.get(str(t).upper(), set())) if pd.notna(t) else 0
        for t in out[target_col]
    ]
    out["target_top_pathway"] = [
        (sorted(pmap.get(str(t).upper(), set()))[0]
         if pmap.get(str(t).upper(), set()) else None)
        if pd.notna(t) else None
        for t in out[target_col]
    ]
    return out


def top_k_pathway_enrichment(
    edge_effects: pd.DataFrame,
    *,
    k: int = 20,
    n_permutations: int = 500,
    seed: int = 91,
    target_col: str = "target_hgnc",
) -> dict:
    """Empirical pathway-enrichment test: does the top-k mean pathway count
    exceed what we'd expect from random k-sized subsets of the same edges?
    """
    rng = _random.Random(seed)
    counts = edge_effects.get("target_pathway_count")
    if counts is None or len(counts) < 2 * k:
        return {"skipped": True, "reason": "too few edges for enrichment test"}
    counts = counts.dropna().astype(int).tolist()
    top_mean = float(pd.Series(counts).sort_values(ascending=False).head(k).mean())
    null_means: list[float] = []
    for _ in range(n_permutations):
        sample = [counts[rng.randrange(len(counts))] for _ in range(k)]
        null_means.append(sum(sample) / k)
    null_mean = sum(null_means) / len(null_means)
    null_std = math.sqrt(
        sum((x - null_mean) ** 2 for x in null_means) / max(len(null_means) - 1, 1)
    )
    n_ge = sum(1 for x in null_means if x >= top_mean)
    p = (n_ge + 1) / (len(null_means) + 1)
    return {
        "k": k,
        "top_k_mean_pathway_count": top_mean,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": p,
        "enriched_at_p_lt_0_05": p < 0.05,
    }
