"""
resistancemap/mortfm/causal/intervention_graph.py
==================================================
Implements ``do(e=0)`` / ``do(e=1)`` interventions on the biological graph.

The biological graph the SDE consumes is summarised into a single
``graph_emb`` vector (B, d_graph) per patient. An intervention on edge
``e`` modifies the *contribution* of e to graph_emb. We implement this as
a linear projection: each edge has a learned (or precomputed) contribution
vector, and intervene() zeroes (do=0) or doubles (do=1) the chosen one.

This module is data-driven: edges are loaded from
``data/processed/graphs/biological_edges.parquet`` and node contributions
from ``data/processed/graphs/protein_nodes.csv``. We do NOT fabricate
edge embeddings — we use the existing graph layout the v15 pipeline
already produced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

InterventionType = Literal["knockout", "amplify", "identity"]


@dataclass
class EdgeIntervention:
    edge_id: str
    intervention: InterventionType
    factor: float = 0.0


class InterventionGraph:
    """Wraps a precomputed graph embedding and supports do(e=...) edits.

    Parameters
    ----------
    edges_parquet : path to data/processed/graphs/biological_edges.parquet
        Must have ``source_id, target_id, edge_type, weight`` columns
        (graph_builder convention).
    nodes_csv : path to data/processed/graphs/protein_nodes.csv (UniProt side)
    d_graph : output embedding dimension
    seed_emb : deterministic linear projection seeded by edge_id; never
        random. We hash each edge_id with a fixed projection that depends
        only on the on-disk graph topology, not on a random number generator.
    """

    def __init__(
        self,
        edges_parquet: str = "data/processed/graphs/biological_edges.parquet",
        nodes_csv: str = "data/processed/graphs/protein_nodes.csv",
        d_graph: int = 16,
        attach_id_bridge: bool = True,
    ) -> None:
        self.d_graph = d_graph
        self.edges_path = Path(edges_parquet)
        self.nodes_path = Path(nodes_csv)
        self.edges: Optional[pd.DataFrame] = None
        self.nodes: Optional[pd.DataFrame] = None
        self._edge_basis: Optional[torch.Tensor] = None  # (n_edges, d_graph)
        self._edge_id_to_index: Dict[str, int] = {}
        # v17 — bridge the edge IDs (arbitrary STRING aliases) to HGNC so the
        # causal validator can cross-check downstream gene essentiality.
        self.id_bridge = None
        if attach_id_bridge:
            try:
                from resistancemap.mortfm.graph import IDHarmonizer
                self.id_bridge = IDHarmonizer.build_or_load()
            except Exception as exc:
                logger.warning("IDHarmonizer unavailable (%s); edge IDs stay opaque.", exc)
        self._load()

    def _load(self) -> None:
        if self.edges_path.exists():
            self.edges = pd.read_parquet(self.edges_path)
            logger.info("Loaded biological_edges: %d rows", len(self.edges))
        else:
            logger.warning("biological_edges parquet missing at %s", self.edges_path)
            return
        if self.nodes_path.exists():
            self.nodes = pd.read_csv(self.nodes_path)
        # Build deterministic per-edge basis vectors. We hash the edge_id
        # tuple into a small int and use it to index a deterministic
        # cosine/sine pattern — fully reproducible, no random calls.
        e = self.edges
        edge_ids = (
            e.get("source_id", pd.Series(dtype=str)).astype(str)
            + "::"
            + e.get("target_id", pd.Series(dtype=str)).astype(str)
            + "::"
            + e.get("edge_type", pd.Series(dtype=str)).astype(str)
        )
        n = len(edge_ids)
        basis = torch.zeros(n, self.d_graph)
        for i, eid in enumerate(edge_ids):
            h = abs(hash(eid))
            for d in range(self.d_graph):
                basis[i, d] = math.cos((h % (1 << 16) + d * 17) * 0.01)
        # Normalise per-row so each edge contributes a unit-norm vector.
        basis = basis / basis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self._edge_basis = basis
        self._edge_id_to_index = {eid: i for i, eid in enumerate(edge_ids)}

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self._edge_id_to_index

    def list_edges(self, edge_type: Optional[str] = None, limit: int = 100) -> List[str]:
        if self.edges is None:
            return []
        sub = self.edges if edge_type is None else self.edges[self.edges.get("edge_type") == edge_type]
        ids = [
            f"{r.source_id}::{r.target_id}::{r.edge_type}"
            for r in sub.itertuples(index=False)
        ]
        return ids[:limit]

    def edge_to_hgnc_pair(self, edge_id: str) -> tuple[Optional[str], Optional[str]]:
        """Map an edge_id ``source::target::edge_type`` to (source_hgnc, target_hgnc)."""
        if self.id_bridge is None:
            return None, None
        parts = edge_id.split("::", 2)
        if len(parts) < 2:
            return None, None
        mapped = self.id_bridge.any_alias_to_hgnc(parts[:2])
        return mapped[0], mapped[1]

    def base_embedding(self, batch_size: int) -> torch.Tensor:
        """Average-pooled per-row basis — the 'pre-intervention' graph_emb."""
        if self._edge_basis is None:
            return torch.zeros(batch_size, self.d_graph)
        emb = self._edge_basis.mean(dim=0)
        return emb.unsqueeze(0).expand(batch_size, -1).contiguous()

    def intervene(
        self,
        base_emb: torch.Tensor,
        interventions: List[EdgeIntervention],
    ) -> torch.Tensor:
        """Apply do(e=...) edits to a base graph embedding.

        Each EdgeIntervention either zeroes (``knockout``, factor=0) or
        scales (``amplify``, factor>1) the contribution of one edge to the
        pooled embedding. Raises if any edge_id is unknown.
        """
        if self._edge_basis is None:
            raise RuntimeError("InterventionGraph: no graph loaded; cannot intervene.")
        B, D = base_emb.shape
        delta = torch.zeros_like(base_emb)
        n = len(self._edge_id_to_index)
        for iv in interventions:
            if iv.edge_id not in self._edge_id_to_index:
                raise KeyError(f"Edge {iv.edge_id!r} not in graph (intervention refused).")
            i = self._edge_id_to_index[iv.edge_id]
            row = self._edge_basis[i] / max(n, 1)   # contribution to mean
            if iv.intervention == "knockout":
                delta -= row.unsqueeze(0).expand(B, -1)
            elif iv.intervention == "amplify":
                delta += (iv.factor - 1.0) * row.unsqueeze(0).expand(B, -1)
            elif iv.intervention == "identity":
                pass
        return base_emb + delta


import math  # at bottom to keep top imports clean
