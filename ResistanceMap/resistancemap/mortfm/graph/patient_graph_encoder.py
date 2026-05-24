"""
resistancemap/mortfm/graph/patient_graph_encoder.py
====================================================
Patient-conditioned GNN that propagates patient-specific features over
the biological graph.

Replaces the :class:`LatentToGraphProjector` (which just projects z0
linearly) with a REAL heterogeneous graph neural network that conditions
on each patient's RNA expression and drug exposure, propagates those
signals over the STRING PPI + Reactome pathway + drug-target graph, and
pools them into a per-patient ``(B, d_graph)`` embedding the SDE drift
can consume.

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
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import GATConv, HeteroConv, GlobalAttention
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning(
        "torch_geometric not available; PatientGraphEncoder will not be "
        "usable. Install with: pip install torch_geometric"
    )

from resistancemap.mortfm.graph.id_harmonizer import IDHarmonizer

__all__ = [
    "HeteroBiologicalGraph",
    "PatientGraphEncoder",
    "build_patient_graph_emb_fn",
]


# ===================================================================
# 1. HeteroBiologicalGraph -- builds PyG HeteroData from on-disk data
# ===================================================================

# Canonical mapping from parquet edge_type -> PyG (src_type, rel, dst_type)
_EDGE_TYPE_MAP: Dict[str, Tuple[str, str, str]] = {
    "ppi":                ("protein", "interacts", "protein"),
    "drug_target":        ("drug", "targets", "protein"),
    "protein_in_pathway": ("gene", "member_of", "pathway"),
}


@dataclass
class HeteroBiologicalGraph:
    """Heterogeneous biological graph backed by a PyG :class:`HeteroData`.

    Build once from on-disk parquet/csv via :meth:`from_disk`, then call
    :meth:`to_hetero_data` to get the immutable graph topology that
    :class:`PatientGraphEncoder` clones per-batch (with patient-specific
    node features stamped on).

    Attributes
    ----------
    hetero_data : HeteroData
        PyG heterogeneous graph object with node indices and edge topology.
    node_id_to_idx : dict
        ``{node_type: {raw_id_str: int_index}}`` for every node type.
    gene_symbols : list[str]
        Ordered list of HGNC symbols that correspond to ``gene`` node
        indices (via IDHarmonizer bridging). Used to align patient RNA
        expression columns.
    drug_ids : list[str]
        Ordered list of drug IDs (CHEMBL) that correspond to ``drug``
        node indices.
    n_nodes : dict[str, int]
        ``{node_type: count}`` for quick shape queries.
    """

    hetero_data: Any  # HeteroData (typed as Any for when PYG is missing)
    node_id_to_idx: Dict[str, Dict[str, int]]
    gene_symbols: List[str]
    drug_ids: List[str]
    n_nodes: Dict[str, int]

    # ----------------------------------------------------------------
    # Factory
    # ----------------------------------------------------------------
    @classmethod
    def from_disk(
        cls,
        edges_parquet: str = "data/processed/graphs/biological_edges.parquet",
        nodes_csv: str = "data/processed/graphs/protein_nodes.csv",
        id_harmonizer: Optional[IDHarmonizer] = None,
    ) -> "HeteroBiologicalGraph":
        """Build the heterogeneous graph from on-disk edge + node files.

        Parameters
        ----------
        edges_parquet :
            Path to the biological_edges.parquet file produced by the
            graph builder (columns: source_id, target_id, source_type,
            target_type, edge_type, edge_weight, directed,
            evidence_source).
        nodes_csv :
            Path to the protein_nodes.csv (columns: uniprot_id,
            gene_symbol, length). Used to establish the ``gene``
            node-type vocabulary for RNA expression alignment.
        id_harmonizer :
            Optional :class:`IDHarmonizer` for bridging HGNC symbols to
            STRING alias IDs. When ``None`` a fresh one is built via
            ``IDHarmonizer.build_or_load()`` (which uses the parquet
            cache if available).

        Returns
        -------
        HeteroBiologicalGraph
        """
        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required for HeteroBiologicalGraph. "
                "Install with: pip install torch_geometric"
            )

        edges_path = Path(edges_parquet)
        nodes_path = Path(nodes_csv)
        if not edges_path.exists():
            raise FileNotFoundError(f"Edges file not found: {edges_path}")

        df = pd.read_parquet(edges_path)
        logger.info(
            "Loaded biological edges: %d rows, edge_types=%s",
            len(df), list(df["edge_type"].unique()),
        )

        # Build harmonizer if not provided (for HGNC <-> alias bridging).
        if id_harmonizer is None:
            id_harmonizer = IDHarmonizer.build_or_load()

        # ----------------------------------------------------------
        # Collect unique node IDs per node type
        # ----------------------------------------------------------
        # We must separate the node ID spaces by type because the same
        # raw string might appear as both a protein and a gene alias.
        per_type_ids: Dict[str, set] = {
            "protein": set(),
            "drug": set(),
            "gene": set(),
            "pathway": set(),
        }

        for _, row in df.iterrows():
            etype = row["edge_type"]
            if etype not in _EDGE_TYPE_MAP:
                continue
            src_type, _, dst_type = _EDGE_TYPE_MAP[etype]
            per_type_ids[src_type].add(str(row["source_id"]))
            per_type_ids[dst_type].add(str(row["target_id"]))

        # Deterministic ordering for reproducibility.
        node_id_to_idx: Dict[str, Dict[str, int]] = {}
        for ntype, ids in per_type_ids.items():
            sorted_ids = sorted(ids)
            node_id_to_idx[ntype] = {rid: i for i, rid in enumerate(sorted_ids)}

        n_nodes = {ntype: len(mapping) for ntype, mapping in node_id_to_idx.items()}
        logger.info("Node counts: %s", n_nodes)

        # ----------------------------------------------------------
        # Build HeteroData edge_index tensors
        # ----------------------------------------------------------
        hetero = HeteroData()

        # Placeholder node features (zero-initialised; PatientGraphEncoder
        # will stamp patient-specific features at forward time).
        for ntype, count in n_nodes.items():
            hetero[ntype].num_nodes = count

        edge_lists: Dict[Tuple[str, str, str], Tuple[List[int], List[int], List[float]]] = {}
        for etype_key, pyg_etype in _EDGE_TYPE_MAP.items():
            edge_lists[pyg_etype] = ([], [], [])

        for _, row in df.iterrows():
            etype = row["edge_type"]
            if etype not in _EDGE_TYPE_MAP:
                continue
            pyg_etype = _EDGE_TYPE_MAP[etype]
            src_type, _, dst_type = pyg_etype

            src_raw = str(row["source_id"])
            dst_raw = str(row["target_id"])

            src_idx = node_id_to_idx[src_type].get(src_raw)
            dst_idx = node_id_to_idx[dst_type].get(dst_raw)
            if src_idx is None or dst_idx is None:
                continue

            srcs, dsts, wts = edge_lists[pyg_etype]
            srcs.append(src_idx)
            dsts.append(dst_idx)
            wts.append(float(row["edge_weight"]))

        for pyg_etype, (srcs, dsts, wts) in edge_lists.items():
            if len(srcs) == 0:
                continue
            ei = torch.tensor([srcs, dsts], dtype=torch.long)
            ew = torch.tensor(wts, dtype=torch.float32).unsqueeze(-1)  # (E, 1)
            hetero[pyg_etype].edge_index = ei
            hetero[pyg_etype].edge_attr = ew

            # For undirected PPI, add reverse edges so message-passing
            # is symmetric.
            if pyg_etype == ("protein", "interacts", "protein"):
                rev_ei = torch.tensor([dsts, srcs], dtype=torch.long)
                # Store reverse under the same edge type (PPI is symmetric)
                # by concatenating.
                hetero[pyg_etype].edge_index = torch.cat([ei, rev_ei], dim=1)
                hetero[pyg_etype].edge_attr = torch.cat([ew, ew], dim=0)

        # ----------------------------------------------------------
        # Build gene_symbols list (HGNC) aligned to gene node indices
        # ----------------------------------------------------------
        # The ``gene`` node IDs in the parquet are Entrez-style numeric
        # strings (from Reactome protein_in_pathway edges). We bridge
        # them to HGNC via the IDHarmonizer.
        gene_idx_to_id = {v: k for k, v in node_id_to_idx.get("gene", {}).items()}
        gene_ids_ordered = [gene_idx_to_id[i] for i in range(n_nodes.get("gene", 0))]
        gene_symbols = id_harmonizer.any_alias_to_hgnc(gene_ids_ordered)
        # Replace None with the raw ID so downstream alignment can skip
        # unmapped genes gracefully.
        gene_symbols_clean: List[str] = [
            sym if sym is not None else raw
            for sym, raw in zip(gene_symbols, gene_ids_ordered)
        ]

        # Ordered drug IDs for the drug node type.
        drug_idx_to_id = {v: k for k, v in node_id_to_idx.get("drug", {}).items()}
        drug_ids = [drug_idx_to_id[i] for i in range(n_nodes.get("drug", 0))]

        # Also load protein_nodes.csv for additional metadata (optional).
        if nodes_path.exists():
            _nodes_df = pd.read_csv(nodes_path)
            logger.info(
                "Loaded protein_nodes.csv: %d rows (used for metadata "
                "alignment, not for node creation).",
                len(_nodes_df),
            )

        logger.info(
            "HeteroBiologicalGraph built: %s nodes, %d edge types",
            n_nodes, len(edge_lists),
        )
        return cls(
            hetero_data=hetero,
            node_id_to_idx=node_id_to_idx,
            gene_symbols=gene_symbols_clean,
            drug_ids=drug_ids,
            n_nodes=n_nodes,
        )

    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------
    def to(self, device: torch.device) -> "HeteroBiologicalGraph":
        """Move the underlying HeteroData to ``device`` (in-place).

        Returns ``self`` for chaining.
        """
        self.hetero_data = self.hetero_data.to(device)
        return self

    def gene_symbol_to_idx(self) -> Dict[str, int]:
        """Return ``{HGNC_symbol: gene_node_index}`` for RNA alignment."""
        return {sym: i for i, sym in enumerate(self.gene_symbols)}

    def drug_id_to_idx(self) -> Dict[str, int]:
        """Return ``{CHEMBL_id: drug_node_index}`` for drug alignment."""
        return {did: i for i, did in enumerate(self.drug_ids)}


# ===================================================================
# 2. PatientGraphEncoder -- heterogeneous GAT conditioned on patient
# ===================================================================

class PatientGraphEncoder(nn.Module):
    """Patient-conditioned heterogeneous GNN over the biological graph.

    For each patient in the batch this encoder:

    1. **Stamps** patient-specific node features onto the graph:
       - ``gene`` nodes receive the patient's RNA expression (mapped by
         HGNC symbol alignment).
       - ``drug`` nodes receive the patient's drug exposure vector.
       - ``protein`` and ``pathway`` nodes receive learned embeddings.

    2. **Conditions** all node features on the patient's z0 latent via a
       learned gating mechanism (FiLM-style: ``gamma * x + beta``
       derived from z0).

    3. **Propagates** features via a stack of :class:`HeteroConv` layers
       wrapping :class:`GATConv` for each edge type.

    4. **Pools** node representations to a single ``(d_graph,)`` vector
       via global attention pooling over ``protein`` nodes.

    Parameters
    ----------
    d_latent : int
        Dimensionality of the patient z0 latent (output of the
        foundation encoder).
    d_graph : int
        Output dimensionality of the graph embedding consumed by the SDE
        drift.
    d_hidden : int
        Hidden dimensionality inside the GNN layers.
    n_layers : int
        Number of HeteroConv (GAT) layers. Default 3.
    n_heads : int
        Number of GAT attention heads per layer. ``d_hidden`` must be
        divisible by ``n_heads``.
    dropout : float
        Dropout probability applied after each conv + residual.
    """

    def __init__(
        self,
        d_latent: int,
        d_graph: int,
        d_hidden: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required for PatientGraphEncoder. "
                "Install with: pip install torch_geometric"
            )
        if d_hidden % n_heads != 0:
            raise ValueError(
                f"d_hidden ({d_hidden}) must be divisible by n_heads "
                f"({n_heads}) for multi-head GAT."
            )

        self.d_latent = d_latent
        self.d_graph = d_graph
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_p = dropout

        # ---- Per-node-type input projections --------------------------
        # Gene nodes: 1 feature (expression) -> d_hidden
        self.gene_proj = nn.Linear(1, d_hidden)
        # Drug nodes: 1 feature (exposure flag/dose) -> d_hidden
        self.drug_proj = nn.Linear(1, d_hidden)
        # Protein nodes: learned embedding (no patient-specific input)
        self.protein_emb_dim = d_hidden  # will be set by register_graph
        # Pathway nodes: learned embedding
        self.pathway_emb_dim = d_hidden

        # Protein and pathway embeddings are created lazily in
        # register_graph() because we need the node counts from the
        # HeteroBiologicalGraph.
        self.protein_embedding: Optional[nn.Embedding] = None
        self.pathway_embedding: Optional[nn.Embedding] = None

        # ---- FiLM conditioning from z0 --------------------------------
        # Produces per-node-type (gamma, beta) from z0 for each type.
        self.film_gamma = nn.Linear(d_latent, d_hidden)
        self.film_beta = nn.Linear(d_latent, d_hidden)

        # ---- HeteroConv layers ----------------------------------------
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            conv_dict = {
                ("protein", "interacts", "protein"): GATConv(
                    d_hidden, d_hidden // n_heads,
                    heads=n_heads, dropout=dropout, edge_dim=1,
                    add_self_loops=False,
                ),
                ("drug", "targets", "protein"): GATConv(
                    (d_hidden, d_hidden), d_hidden // n_heads,
                    heads=n_heads, dropout=dropout, edge_dim=1,
                    add_self_loops=False,
                ),
                ("gene", "member_of", "pathway"): GATConv(
                    (d_hidden, d_hidden), d_hidden // n_heads,
                    heads=n_heads, dropout=dropout, edge_dim=1,
                    add_self_loops=False,
                ),
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            # Per-type LayerNorm applied after each conv.
            self.norms.append(nn.ModuleDict({
                "protein": nn.LayerNorm(d_hidden),
                "drug": nn.LayerNorm(d_hidden),
                "gene": nn.LayerNorm(d_hidden),
                "pathway": nn.LayerNorm(d_hidden),
            }))

        self.dropout = nn.Dropout(dropout)

        # ---- Global attention pooling over protein nodes ---------------
        gate_nn = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),
        )
        self.pool = GlobalAttention(gate_nn=gate_nn)

        # ---- Output projection + LayerNorm ----------------------------
        self.out_proj = nn.Sequential(
            nn.Linear(d_hidden, d_graph),
            nn.SiLU(),
        )
        self.out_norm = nn.LayerNorm(d_graph)

        self._registered = False

    # ----------------------------------------------------------------
    # Graph registration (call once after constructing graph)
    # ----------------------------------------------------------------

    def register_graph(self, graph: HeteroBiologicalGraph) -> None:
        """Register the static graph topology and allocate learned
        node embeddings for protein and pathway nodes.

        Must be called once before the first ``forward()`` call.

        Parameters
        ----------
        graph :
            The :class:`HeteroBiologicalGraph` built from on-disk data.
        """
        n_protein = graph.n_nodes.get("protein", 0)
        n_pathway = graph.n_nodes.get("pathway", 0)

        if n_protein > 0:
            self.protein_embedding = nn.Embedding(n_protein, self.protein_emb_dim)
            nn.init.xavier_uniform_(self.protein_embedding.weight)
        if n_pathway > 0:
            self.pathway_embedding = nn.Embedding(n_pathway, self.pathway_emb_dim)
            nn.init.xavier_uniform_(self.pathway_embedding.weight)

        self._n_protein = n_protein
        self._n_pathway = n_pathway
        self._gene_sym_to_idx = graph.gene_symbol_to_idx()
        self._drug_id_to_idx = graph.drug_id_to_idx()
        self._n_gene = graph.n_nodes.get("gene", 0)
        self._n_drug = graph.n_nodes.get("drug", 0)
        self._registered = True
        logger.info(
            "PatientGraphEncoder registered: protein=%d, gene=%d, "
            "drug=%d, pathway=%d",
            n_protein, self._n_gene, self._n_drug, n_pathway,
        )

    # ----------------------------------------------------------------
    # Feature stamping helpers
    # ----------------------------------------------------------------

    def _stamp_node_features(
        self,
        z0_single: torch.Tensor,
        expression: Optional[torch.Tensor],
        drug_exposure: Optional[torch.Tensor],
        gene_columns: Optional[Sequence[str]],
        drug_columns: Optional[Sequence[str]],
        hetero: HeteroData,
    ) -> Dict[str, torch.Tensor]:
        """Build per-node-type feature tensors for a single patient.

        Parameters
        ----------
        z0_single : Tensor (d_latent,)
            Patient latent vector for FiLM conditioning.
        expression : Tensor (n_genes_rna,) or None
            Patient RNA expression values. ``gene_columns`` maps each
            position to an HGNC symbol.
        drug_exposure : Tensor (n_drugs,) or None
            Patient drug exposure vector. ``drug_columns`` maps each
            position to a CHEMBL drug ID.
        gene_columns : sequence of str or None
            HGNC symbols for each column of ``expression``.
        drug_columns : sequence of str or None
            Drug IDs for each column of ``drug_exposure``.
        hetero : HeteroData
            The base graph topology.

        Returns
        -------
        dict mapping node_type -> Tensor (n_nodes_of_type, d_hidden)
        """
        device = z0_single.device
        dtype = z0_single.dtype

        # FiLM parameters from z0 (shared across node types).
        gamma = self.film_gamma(z0_single)  # (d_hidden,)
        beta = self.film_beta(z0_single)    # (d_hidden,)

        features: Dict[str, torch.Tensor] = {}

        # -- Gene nodes: patient RNA expression --------------------------
        gene_feat = torch.zeros(self._n_gene, 1, device=device, dtype=dtype)
        if expression is not None and gene_columns is not None:
            for col_idx, sym in enumerate(gene_columns):
                node_idx = self._gene_sym_to_idx.get(sym)
                if node_idx is not None and col_idx < expression.shape[0]:
                    gene_feat[node_idx, 0] = expression[col_idx]
        gene_h = self.gene_proj(gene_feat)  # (n_gene, d_hidden)
        features["gene"] = gamma.unsqueeze(0) * gene_h + beta.unsqueeze(0)

        # -- Drug nodes: patient drug exposure ----------------------------
        drug_feat = torch.zeros(self._n_drug, 1, device=device, dtype=dtype)
        if drug_exposure is not None and drug_columns is not None:
            for col_idx, did in enumerate(drug_columns):
                node_idx = self._drug_id_to_idx.get(did)
                if node_idx is not None and col_idx < drug_exposure.shape[0]:
                    drug_feat[node_idx, 0] = drug_exposure[col_idx]
        drug_h = self.drug_proj(drug_feat)  # (n_drug, d_hidden)
        features["drug"] = gamma.unsqueeze(0) * drug_h + beta.unsqueeze(0)

        # -- Protein nodes: learned embedding (not patient-specific) ------
        if self.protein_embedding is not None:
            prot_idx = torch.arange(self._n_protein, device=device)
            prot_h = self.protein_embedding(prot_idx)
        else:
            prot_h = torch.zeros(
                self._n_protein, self.d_hidden, device=device, dtype=dtype,
            )
        features["protein"] = gamma.unsqueeze(0) * prot_h + beta.unsqueeze(0)

        # -- Pathway nodes: learned embedding -----------------------------
        if self.pathway_embedding is not None:
            pw_idx = torch.arange(self._n_pathway, device=device)
            pw_h = self.pathway_embedding(pw_idx)
        else:
            pw_h = torch.zeros(
                self._n_pathway, self.d_hidden, device=device, dtype=dtype,
            )
        features["pathway"] = gamma.unsqueeze(0) * pw_h + beta.unsqueeze(0)

        return features

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------

    def forward(
        self,
        z0: torch.Tensor,
        batch_expression: Optional[torch.Tensor],
        drug_exposure: Optional[torch.Tensor],
        graph_data: HeteroBiologicalGraph,
        gene_columns: Optional[Sequence[str]] = None,
        drug_columns: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        """Produce per-patient graph embeddings.

        Parameters
        ----------
        z0 : Tensor (B, d_latent)
            Patient latent vectors from the foundation encoder.
        batch_expression : Tensor (B, n_genes_rna) or None
            Per-patient RNA expression matrix. Columns aligned with
            ``gene_columns``.
        drug_exposure : Tensor (B, n_drugs) or None
            Per-patient drug exposure vectors. Columns aligned with
            ``drug_columns``.
        graph_data : HeteroBiologicalGraph
            The static biological graph (topology is shared across
            patients; node features are stamped per patient).
        gene_columns : list[str] or None
            HGNC symbols identifying each column of
            ``batch_expression``.
        drug_columns : list[str] or None
            Drug IDs identifying each column of ``drug_exposure``.

        Returns
        -------
        Tensor (B, d_graph)
            Per-patient graph embedding.
        """
        if not self._registered:
            self.register_graph(graph_data)

        B = z0.shape[0]
        device = z0.device
        hetero_template = graph_data.hetero_data

        # Process each patient independently, then stack.
        # (Batching heterogeneous graphs with different node features per
        # sample is non-trivial in PyG; the per-patient loop is the
        # clear-correctness path. For N_patient ~ 30 this is fine.)
        patient_embeddings = []

        for i in range(B):
            z0_i = z0[i]  # (d_latent,)
            expr_i = batch_expression[i] if batch_expression is not None else None
            drug_i = drug_exposure[i] if drug_exposure is not None else None

            # Stamp patient-specific node features.
            x_dict = self._stamp_node_features(
                z0_i, expr_i, drug_i,
                gene_columns, drug_columns,
                hetero_template,
            )

            # Build a per-patient HeteroData with these features.
            h = HeteroData()
            for ntype, feat in x_dict.items():
                h[ntype].x = feat
                h[ntype].num_nodes = feat.shape[0]

            # Copy edge topology from template.
            for etype_key in _EDGE_TYPE_MAP.values():
                if hasattr(hetero_template[etype_key], "edge_index"):
                    h[etype_key].edge_index = hetero_template[etype_key].edge_index.to(device)
                    if hasattr(hetero_template[etype_key], "edge_attr") and hetero_template[etype_key].edge_attr is not None:
                        h[etype_key].edge_attr = hetero_template[etype_key].edge_attr.to(device)

            # Run HeteroConv layers with residual connections.
            for layer_idx, (conv, norm_dict) in enumerate(
                zip(self.convs, self.norms)
            ):
                # Collect current x_dict from h.
                x_in = {ntype: h[ntype].x for ntype in x_dict}
                edge_index_dict = {}
                edge_attr_dict = {}
                for etype_key in _EDGE_TYPE_MAP.values():
                    if hasattr(h[etype_key], "edge_index"):
                        edge_index_dict[etype_key] = h[etype_key].edge_index
                        if hasattr(h[etype_key], "edge_attr") and h[etype_key].edge_attr is not None:
                            edge_attr_dict[etype_key] = h[etype_key].edge_attr

                # HeteroConv forward -- only updates node types that are
                # on the receiving end of at least one edge type.
                x_out = conv(x_in, edge_index_dict, edge_attr_dict=edge_attr_dict)

                # Residual + LayerNorm + activation + dropout for
                # updated node types.
                for ntype in x_dict:
                    if ntype in x_out:
                        residual = h[ntype].x
                        updated = norm_dict[ntype](x_out[ntype])
                        h[ntype].x = self.dropout(F.silu(updated)) + residual
                    # Node types not updated keep their previous features.

            # Pool over protein nodes using global attention.
            protein_h = h["protein"].x  # (n_protein, d_hidden)
            # GlobalAttention expects a batch vector; single-graph ->
            # all zeros.
            batch_vec = torch.zeros(
                protein_h.shape[0], dtype=torch.long, device=device,
            )
            pooled = self.pool(protein_h, batch_vec)  # (1, d_hidden)
            patient_embeddings.append(pooled.squeeze(0))

        # Stack all patients -> (B, d_hidden) then project.
        stacked = torch.stack(patient_embeddings, dim=0)  # (B, d_hidden)
        out = self.out_proj(stacked)  # (B, d_graph)
        out = self.out_norm(out)
        return out


# ===================================================================
# 3. build_patient_graph_emb_fn -- factory for CanonicalMORTFMTrainer
# ===================================================================

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
) -> Tuple[PatientGraphEncoder, Callable]:
    """Factory that builds a :class:`PatientGraphEncoder` and returns it
    together with a ``graph_emb_fn(batch) -> Tensor`` callable compatible
    with :class:`CanonicalMORTFMTrainer`'s ``graph_emb_fn`` parameter.

    Parameters
    ----------
    graph_data_path :
        Directory containing ``biological_edges.parquet`` and
        ``protein_nodes.csv``.
    id_harmonizer :
        Pre-built harmonizer; if ``None`` one is built from the default
        STRING aliases file.
    device :
        Torch device string.
    d_latent, d_graph, d_hidden, n_layers, n_heads, dropout :
        Architecture hyper-parameters forwarded to
        :class:`PatientGraphEncoder`.
    gene_columns :
        HGNC symbols that label the columns of ``MORTBatch.rna``. When
        ``None`` the callable will attempt to align expression anyway
        (zero-padded genes).
    drug_columns :
        Drug IDs that label the columns of ``MORTBatch.drug``. When
        ``None`` drug exposure is not mapped to graph drug nodes.

    Returns
    -------
    (encoder, graph_emb_fn)
        ``encoder`` is the :class:`PatientGraphEncoder` (so the caller
        can register its parameters for optimisation).
        ``graph_emb_fn`` is a ``Callable[[MORTBatch], Tensor]``
        returning ``(B, d_graph)`` on the configured device.
    """
    if not HAS_PYG:
        raise ImportError(
            "torch_geometric is required. Install with: pip install torch_geometric"
        )

    base = Path(graph_data_path)
    edges_pq = str(base / "biological_edges.parquet")
    nodes_csv = str(base / "protein_nodes.csv")

    # Build the static graph (loaded once).
    graph = HeteroBiologicalGraph.from_disk(
        edges_parquet=edges_pq,
        nodes_csv=nodes_csv,
        id_harmonizer=id_harmonizer,
    )
    torch_device = torch.device(device)
    graph.to(torch_device)

    # Build the encoder.
    encoder = PatientGraphEncoder(
        d_latent=d_latent,
        d_graph=d_graph,
        d_hidden=d_hidden,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
    )
    encoder.register_graph(graph)
    encoder.to(torch_device)

    # Capture gene/drug columns in closure.
    _gene_cols = list(gene_columns) if gene_columns is not None else None
    _drug_cols = list(drug_columns) if drug_columns is not None else None

    def graph_emb_fn(batch: Any) -> torch.Tensor:
        """Extract patient features from MORTBatch and run the GNN.

        Expects ``batch`` to expose ``.rna``, ``.drug``, and a method
        or attribute to get z0 (falls back to encoding via the model's
        foundation encoder if z0 is not precomputed).

        For the canonical trainer integration the z0 is available in the
        model outputs dict; here we use ``batch.rna`` as a proxy for
        expression features and ``batch.drug`` for drug exposure,
        delegating the actual z0 computation to the caller (who should
        pass it through the graph embedding function after the foundation
        encoder step).

        This callable signature matches the ``graph_emb_fn`` parameter of
        :class:`CanonicalMORTFMTrainer` (``Callable[[MORTBatch], Tensor]``).
        However, since the trainer calls ``graph_emb_fn(batch)`` before
        the model forward (where z0 is produced), we fall back to a
        zero z0 if the batch has no precomputed latent. The caller
        should prefer using the encoder directly in training loops where
        z0 is available.
        """
        rna = batch.rna  # (B, n_genes) or None
        drug = batch.drug  # (B, n_drugs) or None

        # z0: if the batch carries a precomputed latent, use it.
        # Otherwise, use a zero placeholder -- the caller should prefer
        # calling encoder.forward(z0, ...) directly when z0 is available.
        z0 = getattr(batch, "z0", None)
        if z0 is None:
            z0 = getattr(batch, "latent", None)
        if z0 is None:
            B = batch.batch_size
            z0 = torch.zeros(B, d_latent, device=torch_device)
        z0 = z0.to(torch_device)

        if rna is not None:
            rna = rna.to(torch_device)
        if drug is not None:
            drug = drug.to(torch_device)

        return encoder(
            z0=z0,
            batch_expression=rna,
            drug_exposure=drug,
            graph_data=graph,
            gene_columns=_gene_cols,
            drug_columns=_drug_cols,
        )

    return encoder, graph_emb_fn
