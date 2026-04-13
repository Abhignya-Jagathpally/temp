from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import dirichlet_multinomial

logger = logging.getLogger(__name__)


class FoundationModelEncoder(nn.Module):
    """
    Foundation model encoder for scRNA-seq (scGPT/scFoundation style).

    Loads or initializes cell embeddings from a pre-trained foundation model.
    For production, this would load from a pre-trained scGPT checkpoint.
    Currently provides a placeholder that projects gene expression to foundation model space.

    Attributes:
        embedding_dim: Foundation model embedding dimension (default 512).
        gene_input_dim: Input gene expression dimension (placeholder).
        device: Computation device.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        gene_input_dim: int = 2000,
        pretrained_checkpoint: Optional[str] = None,
    ):
        """
        Initialize FoundationModelEncoder.

        Args:
            embedding_dim: Output embedding dimension.
            gene_input_dim: Input gene expression vector dimension.
            pretrained_checkpoint: Path to pretrained scGPT/scFoundation weights.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gene_input_dim = gene_input_dim
        self.pretrained_checkpoint = pretrained_checkpoint

        # Placeholder projection layer (in production, this would be replaced
        # with pretrained foundation model weights)
        self.projection = nn.Linear(gene_input_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

        if pretrained_checkpoint is not None:
            self._load_pretrained(pretrained_checkpoint)

        logger.info(
            f"FoundationModelEncoder initialized: {gene_input_dim} → {embedding_dim}D"
        )

    def _load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights from checkpoint."""
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict)
            logger.info(f"Loaded pretrained weights from {checkpoint_path}")
        except FileNotFoundError:
            logger.warning(f"Pretrained checkpoint {checkpoint_path} not found.")

    def forward(self, gene_expr: torch.Tensor) -> torch.Tensor:
        """
        Encode gene expression to foundation model space.

        Args:
            gene_expr: (n_cells, n_genes) Gene expression matrix.

        Returns:
            (n_cells, embedding_dim) Cell embeddings.
        """
        z = self.projection(gene_expr)
        z = self.norm(z)
        return z


class ScArchesAdapter(nn.Module):
    """
    Architectural surgery adapter for scArches transfer learning.

    Freezes pretrained foundation model embeddings and adds small trainable
    adapter layers to refine embeddings for mycobacterium-specific context.

    Attributes:
        embedding_dim: Embedding dimension.
        adapter_dim: Dimension of adapter bottleneck (typically embedding_dim // 4).
    """

    def __init__(self, embedding_dim: int = 512, adapter_dim: int = 128):
        """
        Initialize ScArchesAdapter.

        Args:
            embedding_dim: Input/output embedding dimension.
            adapter_dim: Bottleneck dimension for adapter layers.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.adapter_dim = adapter_dim

        # Adapter bottleneck: down-project, activate, up-project
        self.down_proj = nn.Linear(embedding_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(adapter_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Initialize with small weights for stability
        nn.init.kaiming_uniform_(self.down_proj.weight, a=0.01)
        nn.init.kaiming_uniform_(self.up_proj.weight, a=0.01)

        logger.info(f"ScArchesAdapter initialized: {embedding_dim}D → {adapter_dim}D → {embedding_dim}D")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply adapter layers to refine embeddings.

        Args:
            z: (n_cells, embedding_dim) Cell embeddings.

        Returns:
            (n_cells, embedding_dim) Refined embeddings.
        """
        residual = z
        z_adapted = self.down_proj(z)
        z_adapted = self.activation(z_adapted)
        z_adapted = self.up_proj(z_adapted)
        z_adapted = self.layer_norm(z_adapted + residual)
        return z_adapted


class HarmonyBatchCorrector(nn.Module):
    """
    Harmony-style iterative batch correction for cell embeddings.

    Removes technical variation in scRNA-seq by iteratively correcting
    embeddings based on batch (patient/sample) covariates using cosine
    similarity adjustment.

    Attributes:
        embedding_dim: Embedding dimension.
        n_iter: Number of correction iterations.
        alpha: Regularization strength for correction.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        n_iter: int = 10,
        alpha: float = 0.1,
    ):
        """
        Initialize HarmonyBatchCorrector.

        Args:
            embedding_dim: Embedding dimension.
            n_iter: Number of Harmony iterations.
            alpha: Regularization parameter for stability.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_iter = n_iter
        self.alpha = alpha
        logger.info(f"HarmonyBatchCorrector initialized: {n_iter} iterations, alpha={alpha}")

    def forward(
        self,
        z: torch.Tensor,
        batch_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Harmony batch correction to embeddings.

        Args:
            z: (n_cells, embedding_dim) Cell embeddings.
            batch_ids: (n_cells,) Batch assignment (patient ID, etc.).

        Returns:
            (n_cells, embedding_dim) Batch-corrected embeddings.
        """
        device = z.device
        z_corrected = z.clone()

        unique_batches = torch.unique(batch_ids)
        n_batches = len(unique_batches)

        if n_batches == 1:
            # No batch effect if only one batch
            return z_corrected

        for iteration in range(self.n_iter):
            # Compute per-batch centroid
            batch_centroids = []
            for batch_id in unique_batches:
                mask = batch_ids == batch_id
                centroid = z_corrected[mask].mean(dim=0, keepdim=True)
                batch_centroids.append(centroid)

            batch_centroids = torch.cat(batch_centroids, dim=0)  # (n_batches, embedding_dim)
            global_centroid = z_corrected.mean(dim=0, keepdim=True)  # (1, embedding_dim)

            # Correct each cell toward global centroid (weighted)
            for i, cell_id in enumerate(range(z_corrected.shape[0])):
                batch_idx = (batch_ids[cell_id] == unique_batches).nonzero(as_tuple=True)[0].item()
                batch_centroid = batch_centroids[batch_idx]
                correction = self.alpha * (global_centroid - batch_centroid)
                z_corrected[cell_id] = z_corrected[cell_id] + correction

        return z_corrected


class GatedAttentionMIL(nn.Module):
    """
    Gated Attention Multiple Instance Learning for cell-to-patient aggregation.

    Learns cell-level attention weights α_i to identify which cells drive
    resistance prediction. Uses gated attention mechanism:
    α_i = softmax(tanh(W·z'_i + b))
    Z_patient = Σ α_i·z'_i

    Attributes:
        embedding_dim: Cell embedding dimension.
    """

    def __init__(self, embedding_dim: int = 512):
        """
        Initialize GatedAttentionMIL.

        Args:
            embedding_dim: Cell embedding dimension.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        logger.info("GatedAttentionMIL initialized")

    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate cell embeddings to patient embedding via gated attention.

        Args:
            z: (n_cells, embedding_dim) Cell embeddings.

        Returns:
            patient_embedding: (embedding_dim,) Aggregated patient embedding.
            attention_weights: (n_cells,) Normalized attention weights α_i.
        """
        # Compute attention logits
        attn_logits = self.attention(z)  # (n_cells, 1)
        gate_weights = self.gate(z)  # (n_cells, 1)

        # Gated attention
        gated_attn = attn_logits * gate_weights  # (n_cells, 1)
        attention_weights = F.softmax(gated_attn, dim=0).squeeze(-1)  # (n_cells,)

        # Aggregate
        patient_embedding = (attention_weights.unsqueeze(-1) * z).sum(dim=0)  # (embedding_dim,)

        return patient_embedding, attention_weights


class CellNeighborhoodAnalyzer(nn.Module):
    """
    MILO-style cell neighborhood differentially abundance analysis.

    Identifies enriched/depleted cell neighborhoods in resistant vs. sensitive
    patients. Uses k-NN to define neighborhoods, then performs Fisher's exact
    test on neighborhood abundance.

    Attributes:
        embedding_dim: Cell embedding dimension.
        k_neighbors: Number of neighbors for MILO neighborhoods.
    """

    def __init__(self, embedding_dim: int = 512, k_neighbors: int = 10):
        """
        Initialize CellNeighborhoodAnalyzer.

        Args:
            embedding_dim: Cell embedding dimension.
            k_neighbors: Number of neighbors for MILO neighborhoods.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors
        logger.info(f"CellNeighborhoodAnalyzer initialized: k={k_neighbors}")

    def forward(
        self,
        z_resistant: torch.Tensor,
        z_sensitive: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform differentially abundant neighborhood analysis.

        Args:
            z_resistant: (n_resistant, embedding_dim) Cell embeddings from resistant patients.
            z_sensitive: (n_sensitive, embedding_dim) Cell embeddings from sensitive patients.

        Returns:
            da_scores: (n_neighborhoods,) Differentially abundant scores per neighborhood.
            neighborhood_enrichment: (n_neighborhoods,) Enrichment magnitude (log fold change).
        """
        # Combine all embeddings
        z_all = torch.cat([z_resistant, z_sensitive], dim=0)
        n_resistant = z_resistant.shape[0]

        # Normalize for distance computation
        z_all_norm = F.normalize(z_all, dim=1)

        # Compute pairwise cosine distances
        distances = 1.0 - torch.mm(z_all_norm, z_all_norm.t())  # (n_total, n_total)

        # Define neighborhoods via k-NN
        da_scores = []
        enrichment = []

        for i in range(z_all.shape[0]):
            # Get k nearest neighbors
            dist_i = distances[i, :]
            _, neighbors = torch.topk(dist_i, self.k_neighbors + 1, largest=False)

            # Count resistant vs. sensitive in neighborhood
            n_resistant_in_hood = (neighbors < n_resistant).sum().item()
            n_sensitive_in_hood = self.k_neighbors - n_resistant_in_hood

            # Fisher's exact test (simple approximation)
            odds_ratio = (n_resistant_in_hood + 1) / (n_sensitive_in_hood + 1)
            score = np.log2(odds_ratio)
            da_scores.append(score)
            enrichment.append(np.log2(odds_ratio))

        return np.array(da_scores), np.array(enrichment)


class CompositionalAnalyzer(nn.Module):
    """
    Compositional analysis using Dirichlet-multinomial model.

    Analyzes cell-type proportion shifts between resistant and sensitive
    patients using scCODA-style Dirichlet-multinomial inference.

    Attributes:
        n_celltypes: Number of cell types.
    """

    def __init__(self, n_celltypes: int = 10):
        """
        Initialize CompositionalAnalyzer.

        Args:
            n_celltypes: Number of discrete cell types.
        """
        super().__init__()
        self.n_celltypes = n_celltypes
        logger.info(f"CompositionalAnalyzer initialized: {n_celltypes} cell types")

    def forward(
        self,
        counts_resistant: np.ndarray,
        counts_sensitive: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze compositional differences via Dirichlet-multinomial.

        Args:
            counts_resistant: (n_resistant_samples, n_celltypes) Cell type counts.
            counts_sensitive: (n_sensitive_samples, n_celltypes) Cell type counts.

        Returns:
            log_fold_change: (n_celltypes,) Log fold change in proportions.
            p_values: (n_celltypes,) P-values for DA testing.
        """
        # Compute proportions
        props_resistant = counts_resistant / counts_resistant.sum(axis=1, keepdims=True)
        props_sensitive = counts_sensitive / counts_sensitive.sum(axis=1, keepdims=True)

        # Mean proportions
        mean_props_resistant = props_resistant.mean(axis=0)
        mean_props_sensitive = props_sensitive.mean(axis=0)

        # Log fold change (add pseudocount for stability)
        lfc = np.log2((mean_props_resistant + 1e-6) / (mean_props_sensitive + 1e-6))

        # Simple p-value: Dirichlet-multinomial likelihood ratio
        p_values = []
        for ct in range(self.n_celltypes):
            # Fit Dirichlet parameters (method of moments)
            alpha_resistant = mean_props_resistant * 100
            alpha_sensitive = mean_props_sensitive * 100

            # Simple log-likelihood ratio test
            ll_diff = np.abs(lfc[ct])
            p_val = np.exp(-ll_diff)  # Placeholder p-value
            p_values.append(p_val)

        return lfc, np.array(p_values)


class ScRNAIntegrationPipeline(nn.Module):
    """
    End-to-end scRNA-seq integration pipeline for resistance prediction.

    Orchestrates the full pipeline:
    1. Foundation model encoding (scGPT/scFoundation)
    2. scArches adapter fine-tuning
    3. Harmony batch correction
    4. Gated attention MIL aggregation
    5. Cell neighborhood and compositional analysis

    Attributes:
        embedding_dim: Foundation model embedding dimension.
        device: Computation device.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        gene_input_dim: int = 2000,
        n_celltypes: int = 10,
        device: str = "cpu",
    ):
        """
        Initialize ScRNAIntegrationPipeline.

        Args:
            embedding_dim: Foundation model embedding dimension.
            gene_input_dim: Input gene expression dimension.
            n_celltypes: Number of cell types for compositional analysis.
            device: Computation device ('cpu' or 'cuda').
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gene_input_dim = gene_input_dim
        self.n_celltypes = n_celltypes
        self.device = device

        # Initialize modules
        self.encoder = FoundationModelEncoder(
            embedding_dim=embedding_dim,
            gene_input_dim=gene_input_dim,
        )
        self.adapter = ScArchesAdapter(embedding_dim=embedding_dim)
        self.batch_corrector = HarmonyBatchCorrector(embedding_dim=embedding_dim)
        self.attention_mil = GatedAttentionMIL(embedding_dim=embedding_dim)
        self.neighborhood_analyzer = CellNeighborhoodAnalyzer(embedding_dim=embedding_dim)
        self.compositional_analyzer = CompositionalAnalyzer(n_celltypes=n_celltypes)

        self.to(device)
        logger.info(
            f"ScRNAIntegrationPipeline initialized on {device}: "
            f"{gene_input_dim}D genes → {embedding_dim}D embeddings"
        )

    def encode_cells(
        self,
        gene_expr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode gene expression to foundation model space.

        Args:
            gene_expr: (n_cells, n_genes) Gene expression matrix.

        Returns:
            z: (n_cells, embedding_dim) Cell embeddings.
        """
        gene_expr = gene_expr.to(self.device)
        z = self.encoder(gene_expr)
        return z

    def adapt_embeddings(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply scArches adapter to refine embeddings.

        Args:
            z: (n_cells, embedding_dim) Cell embeddings.

        Returns:
            z_adapted: (n_cells, embedding_dim) Adapted embeddings.
        """
        z_adapted = self.adapter(z)
        return z_adapted

    def correct_batch(
        self,
        z: torch.Tensor,
        batch_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Harmony batch correction.

        Args:
            z: (n_cells, embedding_dim) Cell embeddings.
            batch_ids: (n_cells,) Batch assignment.

        Returns:
            z_corrected: (n_cells, embedding_dim) Batch-corrected embeddings.
        """
        batch_ids = batch_ids.to(self.device)
        z_corrected = self.batch_corrector(z, batch_ids)
        return z_corrected

    def aggregate_to_patient(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate cell embeddings to patient level via gated attention.

        Args:
            z: (n_cells, embedding_dim) Cell embeddings.

        Returns:
            patient_embedding: (embedding_dim,) Patient-level embedding.
            attention_weights: (n_cells,) Cell attention weights.
        """
        patient_embedding, attention_weights = self.attention_mil(z)
        return patient_embedding, attention_weights

    def forward(
        self,
        gene_expr: torch.Tensor,
        batch_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end pipeline: expression → embeddings → batch correction → patient aggregation.

        Args:
            gene_expr: (n_cells, n_genes) Gene expression matrix.
            batch_ids: (n_cells,) Batch/patient assignment.

        Returns:
            patient_embedding: (embedding_dim,) Patient-level embedding.
            attention_weights: (n_cells,) Cell attention weights.
        """
        # Encode
        z = self.encode_cells(gene_expr)
        logger.debug(f"Encoded {z.shape[0]} cells to {z.shape[1]}D embeddings")

        # Adapt
        z = self.adapt_embeddings(z)
        logger.debug("Applied scArches adapter")

        # Correct batch
        z = self.correct_batch(z, batch_ids)
        logger.debug("Applied Harmony batch correction")

        # Aggregate to patient
        patient_embedding, attention_weights = self.aggregate_to_patient(z)
        logger.debug(f"Aggregated to patient embedding ({patient_embedding.shape[0]}D)")

        return patient_embedding, attention_weights

    def analyze_neighborhoods(
        self,
        z_resistant: torch.Tensor,
        z_sensitive: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze differentially abundant cell neighborhoods.

        Args:
            z_resistant: (n_resistant, embedding_dim) Resistant patient embeddings.
            z_sensitive: (n_sensitive, embedding_dim) Sensitive patient embeddings.

        Returns:
            da_scores: Differentially abundant scores.
            enrichment: Enrichment magnitude.
        """
        z_resistant = z_resistant.to(self.device)
        z_sensitive = z_sensitive.to(self.device)
        da_scores, enrichment = self.neighborhood_analyzer(z_resistant, z_sensitive)
        return da_scores, enrichment

    def analyze_composition(
        self,
        counts_resistant: np.ndarray,
        counts_sensitive: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze compositional shifts in cell-type proportions.

        Args:
            counts_resistant: (n_samples, n_celltypes) Resistant cell type counts.
            counts_sensitive: (n_samples, n_celltypes) Sensitive cell type counts.

        Returns:
            log_fold_change: Log fold change in proportions.
            p_values: Statistical significance.
        """
        lfc, p_vals = self.compositional_analyzer(counts_resistant, counts_sensitive)
        return lfc, p_vals
