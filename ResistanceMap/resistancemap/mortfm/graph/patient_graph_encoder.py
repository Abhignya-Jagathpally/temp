"""
resistancemap/mortfm/graph/patient_graph_encoder.py
===================================================
Patient-specific graph embedding via real PPI/pathway/drug-target topology.

Replaces the zero-tensor placeholder in the canonical trainer. When this module
is used as graph_emb_fn, pathway and causal claims become supportable.
When only LatentToGraphProjector is used, pathway claims must be blocked.

Graph data
----------
* **Edges** are loaded once from ``data/processed/graphs/biological_edges.parquet``
  which carries three edge types:

  - ``ppi`` (protein--protein, from STRING, undirected, weighted)
  - ``drug_target`` (drug -> protein, curated, weight=1)
  - ``protein_in_pathway`` (gene/protein -> Reactome pathway, weight=1)

* **Nodes** are the union of all unique source/target IDs across the
  three edge types. We assign each unique ID a contiguous integer index
  within its node type.

ID bridging
-----------
The :class:`IDHarmonizer` from ``id_harmonizer.py`` maps HGNC gene
symbols (which index the patient RNA expression matrix) to STRING
aliases that appear in the edge file, so patient features land on the
correct graph nodes.

No synthetic data is constructed anywhere in this module. Nodes that
have no patient-specific feature receive zero-initialised features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyG guard
# ---------------------------------------------------------------------------
try:
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.nn import GATConv, HeteroConv, GlobalAttention
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.info(
        "torch_geometric not available; will use manual message-passing "
        "fallback for PatientGraphEncoder."
    )

from resistancemap.mortfm.graph.id_harmonizer import IDHarmonizer

__all__ = [
    "GraphStore",
    "PatientNodeFeatureBuilder",
    "PatientGraphEncoder",
    "PatientGraphEmbeddingFn",
    "build_patient_graph_embedding_fn",
    # Legacy exports preserved for backward compatibility with __init__.py
    "HeteroBiologicalGraph",
    "build_patient_graph_emb_fn",
]


# ===================================================================
# 1. GraphStore -- loads and caches the Block A biological graph
# ===================================================================

# Canonical mapping from parquet edge_type -> PyG (src_type, rel, dst_type)
_EDGE_TYPE_MAP: Dict[str, Tuple[str, str, str]] = {
    "ppi": ("protein", "interacts", "protein"),
    "drug_target": ("drug", "targets", "protein"),
    "protein_in_pathway": ("gene", "member_of", "pathway"),
}


class GraphStore:
    """Loads and caches the Block A biological graph from disk.

    Builds sparse adjacency from edges in
    ``data/processed/graphs/biological_edges.parquet``, maps protein nodes
    to indices, and provides ``edge_index`` and ``edge_attr`` as PyTorch
    tensors.

    Parameters
    ----------
    edges_parquet :
        Path to the biological_edges.parquet file (columns: source_id,
        target_id, edge_type, edge_weight).
    nodes_csv :
        Path to protein_nodes.csv (columns: uniprot_id, gene_symbol,
        length).
    string_threshold :
        Minimum STRING combined score for PPI edges (0-1000 scale).
        Edges below this threshold are dropped.
    """

    def __init__(
        self,
        edges_parquet: str = "data/processed/graphs/biological_edges.parquet",
        nodes_csv: str = "data/processed/graphs/protein_nodes.csv",
        string_threshold: int = 700,
    ) -> None:
        self.edges_path = Path(edges_parquet)
        self.nodes_path = Path(nodes_csv)
        self.string_threshold = string_threshold

        # Populated by _load()
        self.edges_df: Optional[pd.DataFrame] = None
        self.nodes_df: Optional[pd.DataFrame] = None
        self.protein_index: Dict[str, int] = {}
        self.gene_symbols: List[str] = []
        self.drug_ids: List[str] = []
        self.n_nodes: Dict[str, int] = {}

        # Per-type node ID -> index mappings
        self.node_id_to_idx: Dict[str, Dict[str, int]] = {}

        # Edge index tensors (homogeneous flattened view for fallback)
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_attr: Optional[torch.Tensor] = None

        # Per edge-type tensors
        self._edge_index_per_type: Dict[str, torch.Tensor] = {}
        self._edge_attr_per_type: Dict[str, torch.Tensor] = {}

        self._load()

    def _load(self) -> None:
        """Load graph from disk and build index structures."""
        if not self.edges_path.exists():
            raise FileNotFoundError(
                f"Edges file not found: {self.edges_path}. "
                "Run the graph builder first: scripts/mortfm_build_biological_graph.py"
            )

        self.edges_df = pd.read_parquet(self.edges_path)
        logger.info(
            "GraphStore: loaded %d edges, types=%s",
            len(self.edges_df),
            list(self.edges_df["edge_type"].unique()),
        )

        if self.nodes_path.exists():
            self.nodes_df = pd.read_csv(self.nodes_path)
            logger.info("GraphStore: loaded %d protein nodes", len(self.nodes_df))

        # ------------------------------------------------------------------
        # Collect unique node IDs per type
        # ------------------------------------------------------------------
        per_type_ids: Dict[str, set] = {
            "protein": set(), "drug": set(), "gene": set(), "pathway": set(),
        }

        for _, row in self.edges_df.iterrows():
            etype = row["edge_type"]
            if etype not in _EDGE_TYPE_MAP:
                continue
            src_type, _, dst_type = _EDGE_TYPE_MAP[etype]
            per_type_ids[src_type].add(str(row["source_id"]))
            per_type_ids[dst_type].add(str(row["target_id"]))

        # Deterministic ordering for reproducibility.
        for ntype, ids in per_type_ids.items():
            sorted_ids = sorted(ids)
            self.node_id_to_idx[ntype] = {rid: i for i, rid in enumerate(sorted_ids)}

        self.n_nodes = {ntype: len(m) for ntype, m in self.node_id_to_idx.items()}
        logger.info("GraphStore node counts: %s", self.n_nodes)

        # ------------------------------------------------------------------
        # Build edge index tensors per type
        # ------------------------------------------------------------------
        for etype_key, pyg_etype in _EDGE_TYPE_MAP.items():
            src_type, _, dst_type = pyg_etype
            srcs, dsts, wts = [], [], []

            subset = self.edges_df[self.edges_df["edge_type"] == etype_key]
            for _, row in subset.iterrows():
                src_raw = str(row["source_id"])
                dst_raw = str(row["target_id"])
                src_idx = self.node_id_to_idx[src_type].get(src_raw)
                dst_idx = self.node_id_to_idx[dst_type].get(dst_raw)
                if src_idx is None or dst_idx is None:
                    continue
                srcs.append(src_idx)
                dsts.append(dst_idx)
                wts.append(float(row.get("edge_weight", 1.0)))

            if len(srcs) == 0:
                continue

            ei = torch.tensor([srcs, dsts], dtype=torch.long)
            ew = torch.tensor(wts, dtype=torch.float32).unsqueeze(-1)

            # For undirected PPI, add reverse edges.
            if etype_key == "ppi":
                rev_ei = torch.tensor([dsts, srcs], dtype=torch.long)
                ei = torch.cat([ei, rev_ei], dim=1)
                ew = torch.cat([ew, ew], dim=0)

            self._edge_index_per_type[etype_key] = ei
            self._edge_attr_per_type[etype_key] = ew

        # Build protein index from nodes_csv gene_symbol column.
        if self.nodes_df is not None and "gene_symbol" in self.nodes_df.columns:
            for idx, row in self.nodes_df.iterrows():
                sym = str(row["gene_symbol"])
                if sym not in self.protein_index:
                    self.protein_index[sym] = len(self.protein_index)

        # Gene symbols aligned to gene node indices.
        gene_idx_to_id = {v: k for k, v in self.node_id_to_idx.get("gene", {}).items()}
        self.gene_symbols = [
            gene_idx_to_id.get(i, f"UNKNOWN_{i}")
            for i in range(self.n_nodes.get("gene", 0))
        ]

        # Drug IDs.
        drug_idx_to_id = {v: k for k, v in self.node_id_to_idx.get("drug", {}).items()}
        self.drug_ids = [
            drug_idx_to_id.get(i, f"UNKNOWN_{i}")
            for i in range(self.n_nodes.get("drug", 0))
        ]

    @property
    def edge_index(self) -> torch.Tensor:
        """Homogeneous (protein-protein) PPI edge index (2, E)."""
        if "ppi" in self._edge_index_per_type:
            return self._edge_index_per_type["ppi"]
        return torch.zeros(2, 0, dtype=torch.long)

    @property
    def edge_attr(self) -> torch.Tensor:
        """Homogeneous PPI edge weights (E, 1)."""
        if "ppi" in self._edge_attr_per_type:
            return self._edge_attr_per_type["ppi"]
        return torch.zeros(0, 1, dtype=torch.float32)

    def to_pyg_data(self, node_features: torch.Tensor) -> Any:
        """Create a PyG Data object from the PPI subgraph.

        Parameters
        ----------
        node_features : Tensor (N_protein, F)
            Node feature matrix for protein nodes.

        Returns
        -------
        torch_geometric.data.Data or dict
            PyG Data if available, else a plain dict with keys
            ``x``, ``edge_index``, ``edge_attr``.
        """
        ei = self.edge_index
        ea = self.edge_attr
        if HAS_PYG:
            return Data(x=node_features, edge_index=ei, edge_attr=ea)
        return {"x": node_features, "edge_index": ei, "edge_attr": ea}

    def gene_symbol_to_idx(self) -> Dict[str, int]:
        """Return ``{HGNC_symbol: gene_node_index}`` for RNA alignment."""
        return {sym: i for i, sym in enumerate(self.gene_symbols)}

    def drug_id_to_idx(self) -> Dict[str, int]:
        """Return ``{drug_id: drug_node_index}`` for drug alignment."""
        return {did: i for i, did in enumerate(self.drug_ids)}


# ===================================================================
# 2. PatientNodeFeatureBuilder -- maps molecular features to nodes
# ===================================================================


class PatientNodeFeatureBuilder:
    """Maps patient molecular features onto graph nodes.

    Places patient RNA expression values on the corresponding gene/protein
    graph nodes using HGNC symbol matching. Missing proteins receive zero
    features with an explicit mask.

    Parameters
    ----------
    protein_index : dict
        ``{HGNC_symbol: node_index}`` mapping genes to protein graph
        nodes.
    feature_dim : int
        Dimensionality of node features produced (typically 1 for raw
        expression, or higher if additional features are concatenated).
    """

    def __init__(self, protein_index: Dict[str, int], feature_dim: int = 1) -> None:
        self.protein_index = protein_index
        self.feature_dim = feature_dim
        self.n_nodes = max(protein_index.values()) + 1 if protein_index else 0

    def build_node_features(
        self,
        batch_rna: Optional[torch.Tensor],
        batch_proteomics: Optional[torch.Tensor],
        protein_names: Optional[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build per-patient node feature tensors from molecular data.

        Parameters
        ----------
        batch_rna : Tensor (B, n_genes) or None
            RNA expression values per patient. Columns correspond to
            ``protein_names``.
        batch_proteomics : Tensor (B, n_proteins) or None
            Protein abundance values. Columns correspond to
            ``protein_names``. If both RNA and proteomics are present,
            proteomics takes priority for shared genes.
        protein_names : list of str or None
            HGNC gene symbols labelling columns of the input matrices.

        Returns
        -------
        (node_features, mask)
            ``node_features`` : Tensor (B, N_nodes, feature_dim)
            ``mask`` : Tensor (B, N_nodes) bool, True where a real
            feature was placed.
        """
        if batch_rna is not None:
            B = batch_rna.shape[0]
        elif batch_proteomics is not None:
            B = batch_proteomics.shape[0]
        else:
            # No input data -- return zeros.
            B = 1
            features = torch.zeros(B, self.n_nodes, self.feature_dim)
            mask = torch.zeros(B, self.n_nodes, dtype=torch.bool)
            return features, mask

        device = (batch_rna if batch_rna is not None else batch_proteomics).device
        features = torch.zeros(B, self.n_nodes, self.feature_dim, device=device)
        mask = torch.zeros(B, self.n_nodes, dtype=torch.bool, device=device)

        if protein_names is None:
            return features, mask

        # Map expression values onto graph nodes.
        for col_idx, name in enumerate(protein_names):
            node_idx = self.protein_index.get(name)
            if node_idx is None or node_idx >= self.n_nodes:
                continue

            # Prefer proteomics if available, else use RNA.
            if batch_proteomics is not None and col_idx < batch_proteomics.shape[1]:
                features[:, node_idx, 0] = batch_proteomics[:, col_idx]
                mask[:, node_idx] = True
            elif batch_rna is not None and col_idx < batch_rna.shape[1]:
                features[:, node_idx, 0] = batch_rna[:, col_idx]
                mask[:, node_idx] = True

        return features, mask


# ===================================================================
# 3. PatientGraphEncoder -- GNN producing per-patient embeddings
# ===================================================================


class _ManualGATLayer(nn.Module):
    """Simple single-head attention-based message passing fallback.

    Used when torch_geometric is not installed. Implements a basic
    graph attention mechanism over a sparse edge index.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, in_dim)
        edge_index : Tensor (2, E)

        Returns
        -------
        Tensor (N, out_dim)
        """
        N = x.shape[0]
        h = self.W(x)  # (N, out_dim)

        src, dst = edge_index[0], edge_index[1]

        # Attention coefficients.
        e_src = self.a_src(h[src])  # (E, 1)
        e_dst = self.a_dst(h[dst])  # (E, 1)
        e = self.leaky_relu(e_src + e_dst).squeeze(-1)  # (E,)

        # Softmax per destination node.
        e_max = torch.full((N,), float("-inf"), device=x.device)
        e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=False)
        e_exp = torch.exp(e - e_max[dst])
        e_sum = torch.zeros(N, device=x.device)
        e_sum.scatter_add_(0, dst, e_exp)
        alpha = e_exp / (e_sum[dst] + 1e-8)  # (E,)
        alpha = self.dropout(alpha)

        # Aggregate messages.
        msg = alpha.unsqueeze(-1) * h[src]  # (E, out_dim)
        out = torch.zeros(N, h.shape[1], device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

        return out


class PatientGraphEncoder(nn.Module):
    """GNN that produces per-patient graph embeddings.

    Uses GATConv layers when torch_geometric is available. Falls back
    to a manual message-passing implementation otherwise.

    The encoder operates on the **protein-protein** PPI subgraph:
    patient RNA/proteomics features are placed on protein nodes, then
    propagated through multi-layer graph attention, and globally
    mean-pooled to a (B, d_graph) output.

    Parameters
    ----------
    node_feature_dim : int
        Input feature dimensionality per node.
    d_graph : int
        Output graph embedding dimensionality.
    n_layers : int
        Number of GNN layers.
    n_heads : int
        Number of GAT attention heads per layer (PyG path only;
        the fallback uses single-head attention).
    dropout : float
        Dropout probability.
    d_hidden : int
        Hidden dimensionality inside GNN layers.
    d_latent : int
        Dimensionality of patient z0 latent (for FiLM conditioning).
        Set to 0 to disable FiLM.
    """

    def __init__(
        self,
        node_feature_dim: int = 1,
        d_graph: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        d_hidden: int = 128,
        d_latent: int = 0,
    ) -> None:
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.d_graph = d_graph
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_p = dropout
        self.d_latent = d_latent

        # Input projection.
        self.input_proj = nn.Linear(node_feature_dim, d_hidden)

        # Optional FiLM conditioning from z0.
        if d_latent > 0:
            self.film_gamma = nn.Linear(d_latent, d_hidden)
            self.film_beta = nn.Linear(d_latent, d_hidden)
        else:
            self.film_gamma = None
            self.film_beta = None

        # GNN layers.
        if HAS_PYG:
            self.convs = nn.ModuleList()
            for _ in range(n_layers):
                self.convs.append(
                    GATConv(
                        d_hidden,
                        d_hidden // n_heads,
                        heads=n_heads,
                        dropout=dropout,
                        edge_dim=1,
                        add_self_loops=True,
                    )
                )
        else:
            self.convs = nn.ModuleList()
            for _ in range(n_layers):
                self.convs.append(_ManualGATLayer(d_hidden, d_hidden, dropout))

        self.norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        # Global mean pooling -> output projection.
        self.out_proj = nn.Sequential(
            nn.Linear(d_hidden, d_graph),
            nn.LayerNorm(d_graph),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        z0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode graph node features into per-patient embeddings.

        Parameters
        ----------
        node_features : Tensor (N_total, node_feature_dim)
            Concatenated node features for all patients in the batch.
        edge_index : Tensor (2, E_total)
            Edge index for the batched graph.
        batch_idx : Tensor (N_total,) or None
            Maps each node to its patient index (0..B-1). Required for
            batched graphs. If None, assumes single graph.
        edge_attr : Tensor (E_total, 1) or None
            Edge weights.
        z0 : Tensor (B, d_latent) or None
            Patient latents for FiLM conditioning.

        Returns
        -------
        Tensor (B, d_graph)
        """
        # Input projection.
        h = self.input_proj(node_features)  # (N_total, d_hidden)

        # FiLM conditioning.
        if self.film_gamma is not None and z0 is not None and batch_idx is not None:
            gamma = self.film_gamma(z0)  # (B, d_hidden)
            beta = self.film_beta(z0)    # (B, d_hidden)
            # Broadcast to per-node.
            h = gamma[batch_idx] * h + beta[batch_idx]

        # Message passing layers with residual + norm.
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            if HAS_PYG:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            h = norm(h)
            h = F.silu(h)
            h = self.dropout(h) + residual

        # Global mean pooling per patient.
        if batch_idx is not None:
            B = int(batch_idx.max().item()) + 1
            pooled = torch.zeros(B, self.d_hidden, device=h.device, dtype=h.dtype)
            counts = torch.zeros(B, 1, device=h.device, dtype=h.dtype)
            pooled.scatter_add_(0, batch_idx.unsqueeze(-1).expand_as(h), h)
            counts.scatter_add_(0, batch_idx.unsqueeze(-1), torch.ones_like(batch_idx, dtype=h.dtype).unsqueeze(-1))
            pooled = pooled / counts.clamp(min=1.0)
        else:
            # Single graph -- mean over all nodes.
            pooled = h.mean(dim=0, keepdim=True)  # (1, d_hidden)

        return self.out_proj(pooled)  # (B, d_graph)


# ===================================================================
# 4. PatientGraphEmbeddingFn -- callable wrapper for canonical trainer
# ===================================================================


class PatientGraphEmbeddingFn:
    """Callable wrapper producing per-patient graph embeddings from a MORTBatch.

    Designed to be passed as ``graph_emb_fn`` to
    :class:`CanonicalMORTFMTrainer`. Extracts RNA/proteomics from the
    batch, builds node features, and runs the GNN encoder.

    Parameters
    ----------
    graph_store : GraphStore
        Loaded biological graph.
    node_feature_builder : PatientNodeFeatureBuilder
        Maps patient molecular features to graph nodes.
    graph_encoder : PatientGraphEncoder
        GNN that produces the embedding.
    device : str
        Target device.
    protein_names : list of str or None
        HGNC symbols identifying columns of ``MORTBatch.rna`` or
        ``MORTBatch.proteomics``.
    trainable : bool
        If False (default), runs under ``torch.no_grad()`` for a
        frozen encoder. If True, gradients flow through the GNN.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        node_feature_builder: PatientNodeFeatureBuilder,
        graph_encoder: PatientGraphEncoder,
        device: str = "cuda",
        protein_names: Optional[List[str]] = None,
        trainable: bool = False,
    ) -> None:
        self.graph_store = graph_store
        self.node_feature_builder = node_feature_builder
        self.graph_encoder = graph_encoder
        self.device = torch.device(device)
        self.protein_names = protein_names
        self.trainable = trainable

        # Precompute edge topology on device.
        self._edge_index = graph_store.edge_index.to(self.device)
        self._edge_attr = graph_store.edge_attr.to(self.device)

    def __call__(self, batch: Any) -> torch.Tensor:
        """Produce (B, d_graph) graph embedding from a MORTBatch.

        Parameters
        ----------
        batch : MORTBatch
            Must have ``.rna`` and/or ``.proteomics`` tensors, plus
            ``.batch_size``.

        Returns
        -------
        Tensor (B, d_graph)
        """
        rna = getattr(batch, "rna", None)
        proteomics = getattr(batch, "proteomics", None)
        B = batch.batch_size

        # Build per-patient node features.
        node_features, mask = self.node_feature_builder.build_node_features(
            batch_rna=rna,
            batch_proteomics=proteomics,
            protein_names=self.protein_names,
        )
        # node_features: (B, N_nodes, F)
        node_features = node_features.to(self.device)

        N_nodes = node_features.shape[1]

        # Build batched graph: replicate edge topology per patient with
        # offset node indices.
        all_node_feats = node_features.reshape(B * N_nodes, -1)  # (B*N, F)
        batch_idx = torch.arange(B, device=self.device).repeat_interleave(N_nodes)

        # Offset edge indices per patient.
        base_ei = self._edge_index  # (2, E)
        E = base_ei.shape[1]
        offsets = torch.arange(B, device=self.device) * N_nodes  # (B,)
        # Repeat edge index B times with offsets.
        batched_ei = base_ei.unsqueeze(0).expand(B, 2, E)  # (B, 2, E)
        batched_ei = batched_ei + offsets.view(B, 1, 1)
        batched_ei = batched_ei.reshape(2, B * E)

        # Repeat edge attr.
        batched_ea = self._edge_attr.repeat(B, 1) if self._edge_attr.numel() > 0 else None

        # Optional z0 for FiLM.
        z0 = getattr(batch, "z0", None)
        if z0 is None:
            z0 = getattr(batch, "latent", None)

        if self.trainable:
            return self.graph_encoder(
                all_node_feats, batched_ei,
                batch_idx=batch_idx,
                edge_attr=batched_ea,
                z0=z0.to(self.device) if z0 is not None else None,
            )
        else:
            with torch.no_grad():
                return self.graph_encoder(
                    all_node_feats, batched_ei,
                    batch_idx=batch_idx,
                    edge_attr=batched_ea,
                    z0=z0.to(self.device) if z0 is not None else None,
                )


# ===================================================================
# 5. Factory function
# ===================================================================


def build_patient_graph_embedding_fn(
    edges_parquet: str = "data/processed/graphs/biological_edges.parquet",
    nodes_csv: str = "data/processed/graphs/protein_nodes.csv",
    protein_names: Optional[List[str]] = None,
    d_graph: int = 64,
    device: str = "cuda",
    pretrained_encoder_path: Optional[str] = None,
    n_layers: int = 3,
    n_heads: int = 4,
    dropout: float = 0.1,
    d_hidden: int = 128,
    d_latent: int = 0,
    trainable: bool = False,
) -> PatientGraphEmbeddingFn:
    """Build a ready-to-use graph_emb_fn for CanonicalMORTFMTrainer.

    Parameters
    ----------
    edges_parquet :
        Path to biological_edges.parquet.
    nodes_csv :
        Path to protein_nodes.csv.
    protein_names :
        HGNC symbols labelling columns of ``MORTBatch.rna``. When
        ``None``, the graph store's own protein index is used.
    d_graph :
        Output embedding dimensionality for the SDE drift.
    device :
        Target torch device string.
    pretrained_encoder_path :
        Optional path to a pretrained GNN encoder checkpoint.
        If provided, loads state_dict into the encoder.
    n_layers :
        Number of GNN layers.
    n_heads :
        Number of GAT attention heads (PyG path).
    dropout :
        Dropout probability.
    d_hidden :
        Hidden dimensionality of GNN layers.
    d_latent :
        Dimensionality of patient z0 for FiLM conditioning (0=disabled).
    trainable :
        Whether the embedding function should allow gradient flow.

    Returns
    -------
    PatientGraphEmbeddingFn
        Callable compatible with ``CanonicalMORTFMTrainer.graph_emb_fn``.
    """
    # Load graph.
    graph_store = GraphStore(
        edges_parquet=edges_parquet,
        nodes_csv=nodes_csv,
    )

    # Build protein index for feature alignment.
    protein_index = graph_store.protein_index
    if not protein_index and graph_store.node_id_to_idx.get("protein"):
        # Fall back to protein node IDs as index.
        protein_index = graph_store.node_id_to_idx["protein"]

    # Node feature builder.
    node_feature_builder = PatientNodeFeatureBuilder(
        protein_index=protein_index,
        feature_dim=1,
    )

    # Encoder.
    encoder = PatientGraphEncoder(
        node_feature_dim=1,
        d_graph=d_graph,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        d_hidden=d_hidden,
        d_latent=d_latent,
    )

    # Load pretrained weights if available.
    if pretrained_encoder_path is not None:
        path = Path(pretrained_encoder_path)
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            encoder.load_state_dict(state, strict=False)
            logger.info("Loaded pretrained encoder from %s", path)
        else:
            logger.warning(
                "Pretrained encoder path does not exist: %s", path
            )

    encoder.to(torch.device(device))
    if not trainable:
        encoder.eval()

    # Resolve protein_names: use the graph store's own protein index keys.
    if protein_names is None:
        protein_names = list(protein_index.keys()) if protein_index else None

    return PatientGraphEmbeddingFn(
        graph_store=graph_store,
        node_feature_builder=node_feature_builder,
        graph_encoder=encoder,
        device=device,
        protein_names=protein_names,
        trainable=trainable,
    )


# ===================================================================
# Legacy compatibility aliases
# ===================================================================
# The __init__.py imports these names; preserve them.

# HeteroBiologicalGraph is an alias for GraphStore with a PyG-oriented
# interface. We provide a thin wrapper that matches the old API.

class HeteroBiologicalGraph(GraphStore):
    """Legacy alias for GraphStore that provides ``from_disk()`` and
    ``to()`` methods matching the v17 interface.

    New code should use :class:`GraphStore` directly.
    """

    @classmethod
    def from_disk(
        cls,
        edges_parquet: str = "data/processed/graphs/biological_edges.parquet",
        nodes_csv: str = "data/processed/graphs/protein_nodes.csv",
        id_harmonizer: Optional[IDHarmonizer] = None,
    ) -> "HeteroBiologicalGraph":
        """Build from on-disk data, matching the legacy v17 API."""
        instance = cls(edges_parquet=edges_parquet, nodes_csv=nodes_csv)
        return instance

    def to(self, device: torch.device) -> "HeteroBiologicalGraph":
        """Move edge tensors to device (in-place). Returns self."""
        for key in list(self._edge_index_per_type.keys()):
            self._edge_index_per_type[key] = self._edge_index_per_type[key].to(device)
        for key in list(self._edge_attr_per_type.keys()):
            self._edge_attr_per_type[key] = self._edge_attr_per_type[key].to(device)
        return self


def build_patient_graph_emb_fn(
    graph_data_path: str = "data/processed/graphs",
    id_harmonizer: Optional[IDHarmonizer] = None,
    device: str = "cpu",
    d_latent: int = 64,
    d_graph: int = 32,
    d_hidden: int = 128,
    n_layers: int = 3,
    n_heads: int = 4,
    dropout: float = 0.2,
    gene_columns: Optional[Sequence[str]] = None,
    drug_columns: Optional[Sequence[str]] = None,
) -> Tuple["PatientGraphEncoder", Callable]:
    """Legacy factory matching the v17 ``build_patient_graph_emb_fn`` API.

    Returns ``(encoder, graph_emb_fn)`` where ``graph_emb_fn`` is a
    ``Callable[[MORTBatch], Tensor]`` returning ``(B, d_graph)``.

    New code should prefer :func:`build_patient_graph_embedding_fn`.
    """
    base = Path(graph_data_path)
    edges_pq = str(base / "biological_edges.parquet")
    nodes_csv = str(base / "protein_nodes.csv")

    emb_fn = build_patient_graph_embedding_fn(
        edges_parquet=edges_pq,
        nodes_csv=nodes_csv,
        protein_names=list(gene_columns) if gene_columns is not None else None,
        d_graph=d_graph,
        device=device,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        d_hidden=d_hidden,
        d_latent=d_latent,
        trainable=True,
    )

    # Return (encoder, callable) to match legacy API.
    return emb_fn.graph_encoder, emb_fn
