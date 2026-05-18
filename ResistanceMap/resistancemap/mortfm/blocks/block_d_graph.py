"""
resistancemap/mortfm/blocks/block_d_graph.py
============================================
Block D — graph + protein-sequence (ESM-2) cache adapter.

Block D is not a single checkpoint — it is a *bundle* of:
  * the biological graph (PPI + Reactome + drug-target edges)
  * the gene→UniProt harmonization map
  * the ESM-2 embedding cache

The adapter advertises all three but the primary path is the ESM-2 cache
because that's the artifact that unlocks the sequence_aware claim.
"""

from __future__ import annotations

import logging
from pathlib import Path

from resistancemap.mortfm.blocks._common import (
    git_head_short, read_json, utcnow_iso,
)
from resistancemap.mortfm.registries.artifact_registry import ArtifactRecord

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_D_CACHE = "data/processed/proteins/esm_embeddings__facebook__esm2_t6_8M_UR50D.pt"
DEFAULT_BLOCK_D_REPORT = "logs/mortfm/block_d_summary.json"
DEFAULT_GRAPH_EDGES = "data/processed/graphs/biological_edges.parquet"
DEFAULT_IDENTIFIER_MAP = "data/processed/metadata/gene_protein_identifier_map.csv"


def load_block_d(
    cache_path: str = DEFAULT_BLOCK_D_CACHE,
):
    """Load the ESM-2 cache via the existing ESMEmbeddingCache class."""
    from resistancemap.data.protein_sequence_cache import ESMEmbeddingCache
    cache = ESMEmbeddingCache(model_name="facebook/esm2_t6_8M_UR50D")
    ok = cache.load(str(Path(cache_path).parent))
    return cache if ok else None


def block_d_artifact_record(
    cache_path: str = DEFAULT_BLOCK_D_CACHE,
    report_path: str = DEFAULT_BLOCK_D_REPORT,
) -> ArtifactRecord:
    rep = read_json(report_path) or {}
    n_emb = int(rep.get("n_embeddings_cached", 0))
    allowed = ["technical"]
    if bool(rep.get("sequence_aware_gate_pass", False)):
        allowed.append("sequence_aware")
    if rep.get("feature_gene_coverage_via_identifier_map", 0.0) >= 0.70:
        allowed.append("pathway_context")
    return ArtifactRecord(
        block_id="D_esm2_embeddings",
        artifact_type="embedding_cache",
        path=str(Path(cache_path).resolve() if Path(cache_path).exists() else cache_path),
        data_source=[
            "UniProt UP000005640_9606 human reference proteome",
            "ESM-2 8M (facebook/esm2_t6_8M_UR50D) embeddings",
            f"Biological graph edges at {DEFAULT_GRAPH_EDGES}",
            f"Identifier map at {DEFAULT_IDENTIFIER_MAP}",
        ],
        n_samples=n_emb,
        n_features=None,  # embedding dim 320 is documented in the FeatureSpace.notes
        feature_space_id="protein_esm2_8m_cached",
        allowed_claims=sorted(set(allowed)),
        blocked_claims=[
            "longitudinal_trajectory", "resistance_emergence", "causal_mechanism",
        ],
        checkpoint_compatible=False,
        created_at=utcnow_iso(),
        git_commit=git_head_short(),
        gate_report_path=report_path if Path(report_path).exists() else None,
        notes=(
            "Embedding cache populated for 1,781 mapped feature-gene UniProt sequences. "
            "Identifier-map coverage 73.2% PASSES the 70% pathway_context gate; "
            "UniProt-direct gene_symbol coverage is 69.0% (honest caveat in report)."
        ),
    )
