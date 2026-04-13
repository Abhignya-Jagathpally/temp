"""Gap 4.4: Cell-Cell Communication Module for ResistanceMap.

Implements heterogeneous graph neural networks for ligand-receptor (L-R) communication
and spatial decay weighting to predict cell-state-dependent resistance mechanisms.

Architecture:
    1. LRInteractionGraph: Builds heterogeneous graph with protein nodes + cell-type nodes,
       linking them via PPI edges and L-R interactions weighted by CellChat probability
       and spatial distance decay.
    2. DualLayerHeteroGNN: Two-layer architecture separating protein PPI interactions
       (Layer 1: GAT) from cell-type communication (Layer 2: R-GCN) with cross-layer
       message passing.
    3. SpatialDecayWeighting: Implements exponential spatial decay w = exp(-d/σ)
       for ligand-receptor edge weights.
    4. CommunicationAwareResistancePredictor: Combines PPI + communication insights
       for final resistance prediction with uncertainty.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GATConv, RGCNConv, global_mean_pool, GlobalAttention
    from torch_geometric.data import Data as PyGData, Batch as PyGBatch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning("torch_geometric not available, cell communication models will use fallback")


class SpatialDecayWeighting(nn.Module):
    """Exponential spatial decay weighting for ligand-receptor interactions.

    Implements w = exp(-d / σ) where d is Euclidean distance between cell types
    in embedding space and σ is a learned or fixed decay constant.

    Attributes:
        decay_sigma: Decay rate parameter (higher = faster decay with distance).
        learnable: Whether decay_sigma is learnable (default: True).
    """

    def __init__(self, decay_sigma: float = 2.0, learnable: bool = True):
        """Initialize SpatialDecayWeighting.

        Args:
            decay_sigma: Initial decay rate parameter.
            learnable: If True, decay_sigma is learnable; else fixed.
        """
        super().__init__()
        if learnable:
            self.decay_sigma = nn.Parameter(torch.tensor(decay_sigma, dtype=torch.float32))
        else:
            self.register_buffer("decay_sigma", torch.tensor(decay_sigma, dtype=torch.float32))
        self.learnable = learnable

    def forward(
        self,
        cell_type_embeddings: torch.Tensor,
        edge_index_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spatial decay weights for cell-type edges.

        Args:
            cell_type_embeddings: (num_cell_types, embedding_dim) cell type vectors.
            edge_index_cells: (2, num_cell_edges) edge indices between cell types.

        Returns:
            (num_cell_edges,) decay weights in [0, 1].
        """
        src, dst = edge_index_cells
        src_embed = cell_type_embeddings[src]  # (num_edges, dim)
        dst_embed = cell_type_embeddings[dst]  # (num_edges, dim)

        # Compute L2 distance
        distances = torch.norm(src_embed - dst_embed, p=2, dim=-1)  # (num_edges,)

        # Apply exponential decay: exp(-d / sigma)
        weights = torch.exp(-distances / (self.decay_sigma + 1e-8))
        return weights


class LRInteractionGraph(nn.Module):
    """Builds heterogeneous graph of proteins + cell types with L-R interactions.

    Combines:
    - Protein node layer: Connected via PPI edges
    - Cell-type node layer (15-20 types): Connected via spatial decay weighted L-R edges
    - Cross-layer edges: Linking protein targets to cell types

    Attributes:
        num_proteins: Number of proteins in the network.
        num_cell_types: Number of cell types (typically 15-20).
        num_lr_interactions: Number of ligand-receptor interactions.
        ppi_edges: (2, num_ppi_edges) edge index for PPI network.
        lr_pairs: List of (ligand, receptor, cell_type_src, cell_type_dst, probability).
    """

    def __init__(
        self,
        num_proteins: int,
        num_cell_types: int,
        ppi_edges: torch.Tensor,
        lr_pairs: Optional[List[Tuple[int, int, int, int, float]]] = None,
        decay_sigma: float = 2.0,
    ):
        """Initialize LRInteractionGraph.

        Args:
            num_proteins: Number of proteins.
            num_cell_types: Number of cell types.
            ppi_edges: (2, num_ppi_edges) PPI edge indices.
            lr_pairs: List of (ligand_idx, receptor_idx, src_cell, dst_cell, probability).
            decay_sigma: Spatial decay rate for L-R weighting.
        """
        super().__init__()
        self.num_proteins = num_proteins
        self.num_cell_types = num_cell_types
        self.register_buffer("ppi_edges", ppi_edges)

        # Store L-R interactions
        if lr_pairs is None:
            lr_pairs = []
        self.lr_pairs = lr_pairs

        # Spatial decay module
        self.spatial_decay = SpatialDecayWeighting(decay_sigma=decay_sigma, learnable=True)

        logger.info(
            f"LRInteractionGraph: {num_proteins} proteins, {num_cell_types} cell types, "
            f"{len(lr_pairs)} L-R pairs"
        )

    def build_heterogeneous_edges(
        self,
        cell_embeddings: torch.Tensor,
        cellchat_probabilities: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Construct heterogeneous edge index and weights.

        Args:
            cell_embeddings: (num_cell_types, embedding_dim) cell type embeddings.
            cellchat_probabilities: (num_lr_pairs,) interaction probabilities from CellChat.

        Returns:
            Dict with keys:
                - "ppi_edges": Protein-protein interaction edges
                - "lr_edges": Ligand-receptor edges
                - "lr_weights": Spatial decay × CellChat probability weights
                - "cross_layer_edges": Protein-to-cell-type edges
        """
        edges_out = {"ppi_edges": self.ppi_edges.clone()}

        if len(self.lr_pairs) == 0:
            return edges_out

        # Unpack L-R pairs
        ligands = []
        receptors = []
        src_cells = []
        dst_cells = []
        probabilities = []

        for ligand, receptor, src_cell, dst_cell, prob in self.lr_pairs:
            ligands.append(ligand)
            receptors.append(receptor)
            src_cells.append(src_cell)
            dst_cells.append(dst_cell)
            probabilities.append(prob)

        lr_edge_index = torch.stack(
            [
                torch.tensor(ligands, dtype=torch.long),
                torch.tensor(receptors, dtype=torch.long),
            ],
            dim=0,
        )
        cell_edge_index = torch.stack(
            [
                torch.tensor(src_cells, dtype=torch.long),
                torch.tensor(dst_cells, dtype=torch.long),
            ],
            dim=0,
        )

        # Compute spatial decay weights
        decay_weights = self.spatial_decay(cell_embeddings, cell_edge_index)

        # Combine with CellChat probabilities
        cellchat_probs = (
            torch.tensor(probabilities, dtype=torch.float32)
            if cellchat_probabilities is None
            else cellchat_probabilities
        )
        combined_weights = decay_weights * cellchat_probs

        edges_out["lr_edges"] = lr_edge_index
        edges_out["lr_weights"] = combined_weights
        edges_out["cross_layer_edges"] = cell_edge_index

        return edges_out

    def forward(
        self,
        cell_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build the complete heterogeneous graph.

        Args:
            cell_embeddings: (num_cell_types, embedding_dim).

        Returns:
            Dict with edge indices and weights for all interaction types.
        """
        return self.build_heterogeneous_edges(cell_embeddings)


class DualLayerHeteroGNN(nn.Module):
    """Two-layer heterogeneous GNN for protein + cell-type communication.

    Layer 1: Protein PPI aggregation via GAT.
    Layer 2: Cell-type R-GCN with relation types (L-R, spatial proximity).
    Cross-layer: Message passing between protein and cell-type nodes.

    Attributes:
        hidden_dim: Hidden dimension for GNN layers.
        num_proteins: Number of protein nodes.
        num_cell_types: Number of cell-type nodes.
        num_relations: Number of relation types (e.g., 2 for L-R + spatial).
        num_heads: Number of attention heads in GAT.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_proteins: int = 1000,
        num_cell_types: int = 15,
        num_relations: int = 2,
        num_heads: int = 8,
    ):
        """Initialize DualLayerHeteroGNN.

        Args:
            hidden_dim: Hidden dimension.
            num_proteins: Number of proteins.
            num_cell_types: Number of cell types.
            num_relations: Number of relation types in R-GCN.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_proteins = num_proteins
        self.num_cell_types = num_cell_types
        self.num_relations = num_relations

        if not HAS_PYG:
            logger.warning("torch_geometric not available, using fallback GNN")
            self.gat = None
            self.rgcn = None
            return

        # Layer 1: Protein PPI via GAT
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=0.1)
        self.ppi_norm = nn.LayerNorm(hidden_dim)
        self.ppi_dropout = nn.Dropout(0.1)

        # Layer 2: Cell-type R-GCN
        self.rgcn1 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=4)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=4)
        self.cell_norm = nn.LayerNorm(hidden_dim)
        self.cell_dropout = nn.Dropout(0.1)

        # Cross-layer message passing
        self.cross_layer = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.cross_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        protein_features: torch.Tensor,
        cell_features: torch.Tensor,
        ppi_edges: torch.Tensor,
        lr_edges: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through dual-layer hetero-GNN.

        Args:
            protein_features: (num_proteins, hidden_dim) protein embeddings.
            cell_features: (num_cell_types, hidden_dim) cell type embeddings.
            ppi_edges: (2, num_ppi_edges) PPI edge index.
            lr_edges: (2, num_lr_edges) L-R edge index.
            edge_attr: Optional (num_lr_edges,) edge weights.

        Returns:
            Tuple of (updated_protein_features, updated_cell_features).
        """
        if self.gat is None:
            # Fallback: return input
            return protein_features, cell_features

        # Layer 1: Protein PPI aggregation
        x_prot = self.gat1(protein_features, ppi_edges)
        x_prot = F.gelu(x_prot)
        x_prot = self.gat2(x_prot, ppi_edges)
        x_prot = self.ppi_norm(x_prot)
        x_prot = self.ppi_dropout(x_prot)

        # Layer 2: Cell-type R-GCN
        # Use edge_attr to assign relation types
        if edge_attr is not None:
            edge_types = torch.where(edge_attr > 0.5, 0, 1)  # Thresholded into 2 types
        else:
            edge_types = torch.zeros(lr_edges.shape[1], dtype=torch.long)

        x_cell = self.rgcn1(cell_features, lr_edges, edge_types)
        x_cell = F.gelu(x_cell)
        x_cell = self.rgcn2(x_cell, lr_edges, edge_types)
        x_cell = self.cell_norm(x_cell)
        x_cell = self.cell_dropout(x_cell)

        # Cross-layer message passing
        # Attention from cell types to proteins
        x_prot_updated, _ = self.cross_layer(
            x_prot.unsqueeze(0), x_cell.unsqueeze(0), x_cell.unsqueeze(0)
        )
        x_prot_updated = x_prot_updated.squeeze(0)
        x_prot = self.cross_layer_norm(x_prot + x_prot_updated)

        return x_prot, x_cell


class CommunicationAwareResistancePredictor(nn.Module):
    """Predicts resistance by combining PPI + cell communication signals.

    Integrates protein-level features from PPI networks with cell-type communication
    patterns to generate resistance predictions that account for cell state effects.

    Attributes:
        hidden_dim: Hidden dimension for processing.
        num_proteins: Number of proteins.
        num_cell_types: Number of cell types.
        num_drugs: Number of drugs (for drug conditioning).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_proteins: int = 1000,
        num_cell_types: int = 15,
        num_drugs: int = 300,
    ):
        """Initialize CommunicationAwareResistancePredictor.

        Args:
            hidden_dim: Hidden dimension.
            num_proteins: Number of proteins.
            num_cell_types: Number of cell types.
            num_drugs: Number of drugs.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_proteins = num_proteins
        self.num_cell_types = num_cell_types

        # Protein feature processor
        self.protein_encoder = nn.Sequential(
            nn.Linear(num_proteins, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Cell communication encoder
        self.cell_encoder = nn.Sequential(
            nn.Linear(num_cell_types, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Drug conditioning
        self.drug_embedding = nn.Embedding(num_drugs, hidden_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Resistance head (sigmoid for binary classification)
        self.resistance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        logger.info(
            f"CommunicationAwareResistancePredictor: {num_proteins} proteins, "
            f"{num_cell_types} cell types, {num_drugs} drugs"
        )

    def forward(
        self,
        protein_features: torch.Tensor,
        cell_features: torch.Tensor,
        drug_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict resistance and uncertainty.

        Args:
            protein_features: (batch_size, num_proteins) protein expression/features.
            cell_features: (batch_size, num_cell_types) cell communication scores.
            drug_ids: (batch_size,) drug identifiers.

        Returns:
            Dict with keys:
                - "resistance": (batch_size, 1) resistance probability
                - "uncertainty": (batch_size, 1) epistemic uncertainty
        """
        # Encode protein features
        x_prot = self.protein_encoder(protein_features)  # (batch, hidden_dim)

        # Encode cell communication
        x_cell = self.cell_encoder(cell_features)  # (batch, hidden_dim)

        # Embed drugs
        x_drug = self.drug_embedding(drug_ids)  # (batch, hidden_dim)

        # Fuse all signals
        x_fused = torch.cat([x_prot, x_cell, x_drug], dim=-1)  # (batch, 3*hidden_dim)
        x_fused = self.fusion(x_fused)  # (batch, hidden_dim)

        # Predict resistance and uncertainty
        resistance = self.resistance_head(x_fused)  # (batch, 1)
        uncertainty = self.uncertainty_head(x_fused)  # (batch, 1)

        return {"resistance": resistance, "uncertainty": uncertainty}
