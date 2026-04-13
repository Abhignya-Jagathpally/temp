"""Gap 4.7: Pathway Harmonization Module for ResistanceMap.

Implements pathway scoring, harmonization across domains, and dimensionality
reduction to reconcile pathway signals across proteomics, transcriptomics, and
single-cell modalities.

Architecture:
    1. GSVAScorer: Rank-based GSVA (Gene Set Variation Analysis) in pure NumPy/scipy.
       Takes expression matrix + gene sets → pathway-level scores.
    2. PathwaySetManager: Loads GMT gene set files, manages MSigDB + Reactome + MM
       (Molecular Mechanisms) pathway databases.
    3. DANNHarmonizer: Domain adversarial neural network with gradient reversal
       for cross-modality pathway harmonization.
    4. PathwayVAECompressor: Autoencoder reducing 250 pathways → 64-dimensional
       latent representation for efficient integration.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


class GSVAScorer(nn.Module):
    """Rank-based GSVA implementation for pathway scoring.

    Single-sample GSVA computes per-sample pathway scores by comparing
    expression ranks of pathway genes vs. background genes.

    References:
        Hänzelmann et al. (2013). GSVA: Gene set variation analysis for
        microarray and RNA-seq data. BMC Bioinformatics.

    Attributes:
        gene_set_names: Names of gene sets.
        gene_sets: Dict mapping gene set name to list of gene indices.
    """

    def __init__(
        self,
        gene_sets: Dict[str, List[int]],
        min_genes: int = 1,
    ):
        """Initialize GSVAScorer.

        Args:
            gene_sets: Dict mapping gene set name → list of gene indices.
            min_genes: Minimum number of genes required in gene set.
        """
        super().__init__()
        self.gene_sets = {}
        self.gene_set_names = []

        # Filter gene sets by minimum size
        for name, indices in gene_sets.items():
            if len(indices) >= min_genes:
                self.gene_sets[name] = indices
                self.gene_set_names.append(name)

        logger.info(f"GSVAScorer initialized with {len(self.gene_sets)} gene sets")

    def forward(self, expression_matrix: torch.Tensor) -> torch.Tensor:
        """Compute GSVA scores.

        Args:
            expression_matrix: (num_samples, num_genes) expression matrix.
                Can be numpy array or torch tensor; returns tensor.

        Returns:
            (num_samples, num_pathways) GSVA scores.
        """
        if isinstance(expression_matrix, np.ndarray):
            X = expression_matrix
        else:
            X = expression_matrix.cpu().numpy()

        num_samples, num_genes = X.shape
        num_pathways = len(self.gene_sets)

        scores = np.zeros((num_samples, num_pathways), dtype=np.float32)

        for pathway_idx, (name, gene_indices) in enumerate(self.gene_sets.items()):
            # Rank expression across genes for each sample
            # rankdata along axis 1: ranks genes within each sample
            ranks = rankdata(X, axis=1)  # (num_samples, num_genes)

            # Mean rank of pathway genes
            pathway_ranks = ranks[:, gene_indices]  # (num_samples, |pathway|)
            pathway_mean = np.mean(pathway_ranks, axis=1)  # (num_samples,)

            # Mean rank of non-pathway genes (background)
            non_pathway_indices = np.array(
                [i for i in range(num_genes) if i not in gene_indices]
            )
            if len(non_pathway_indices) > 0:
                background_ranks = ranks[:, non_pathway_indices]
                background_mean = np.mean(background_ranks, axis=1)
            else:
                background_mean = num_genes / 2.0

            # GSVA score: standardized difference between pathway and background ranks
            scores[:, pathway_idx] = (pathway_mean - background_mean) / num_genes

        return torch.tensor(scores, dtype=torch.float32)

    def get_pathway_names(self) -> List[str]:
        """Get list of pathway names in order.

        Returns:
            List of pathway names.
        """
        return self.gene_set_names


class PathwaySetManager:
    """Loads and manages gene set databases (GMT format).

    Supports MSigDB, Reactome, and Molecular Mechanisms (MM) pathway collections.

    Attributes:
        gene_sets: Dict mapping gene set name → list of genes.
        gene_to_index: Dict mapping gene name → global index.
        index_to_gene: Reverse mapping.
    """

    def __init__(self):
        """Initialize PathwaySetManager."""
        self.gene_sets = {}
        self.gene_to_index = {}
        self.index_to_gene = {}
        self._gene_counter = 0

    def load_gmt(self, gmt_path: str, collection_name: str = "custom") -> int:
        """Load gene sets from GMT file.

        GMT format: PATHWAY_NAME<TAB>DESCRIPTION<TAB>GENE1<TAB>GENE2<TAB>...

        Args:
            gmt_path: Path to GMT file.
            collection_name: Prefix for pathway names (e.g., "MSigDB", "Reactome").

        Returns:
            Number of gene sets loaded.
        """
        loaded = 0

        try:
            with open(gmt_path, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue

                    pathway_name = f"{collection_name}_{parts[0]}"
                    genes = parts[2:]  # Skip description

                    # Map genes to indices
                    gene_indices = []
                    for gene in genes:
                        if gene not in self.gene_to_index:
                            idx = self._gene_counter
                            self.gene_to_index[gene] = idx
                            self.index_to_gene[idx] = gene
                            self._gene_counter += 1
                        gene_indices.append(self.gene_to_index[gene])

                    self.gene_sets[pathway_name] = gene_indices
                    loaded += 1

            logger.info(f"Loaded {loaded} gene sets from {gmt_path}")
        except FileNotFoundError:
            logger.warning(f"GMT file not found: {gmt_path}")

        return loaded

    def create_from_dict(
        self,
        gene_sets: Dict[str, List[str]],
        collection_name: str = "custom",
    ) -> None:
        """Create gene sets from dictionary.

        Args:
            gene_sets: Dict mapping pathway name → list of genes.
            collection_name: Prefix for pathway names.
        """
        for pathway_name, genes in gene_sets.items():
            full_name = f"{collection_name}_{pathway_name}"
            gene_indices = []

            for gene in genes:
                if gene not in self.gene_to_index:
                    idx = self._gene_counter
                    self.gene_to_index[gene] = idx
                    self.index_to_gene[idx] = gene
                    self._gene_counter += 1
                gene_indices.append(self.gene_to_index[gene])

            self.gene_sets[full_name] = gene_indices

        logger.info(
            f"Created {len(gene_sets)} gene sets from dict. "
            f"Total genes indexed: {self._gene_counter}"
        )

    def get_gene_sets_as_indices(self) -> Dict[str, List[int]]:
        """Get gene sets mapped to indices.

        Returns:
            Dict mapping pathway name → list of gene indices.
        """
        return self.gene_sets.copy()

    def align_with_expression_data(
        self,
        gene_names: List[str],
    ) -> Dict[str, List[int]]:
        """Align gene sets with expression data gene ordering.

        Args:
            gene_names: List of gene names in expression matrix.

        Returns:
            Dict mapping pathway name → indices in expression matrix.
        """
        expr_gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        aligned_sets = {}

        for pathway_name, gene_indices in self.gene_sets.items():
            # Map internal indices back to gene names
            genes = [self.index_to_gene[idx] for idx in gene_indices]

            # Find intersection with expression data
            aligned_indices = [
                expr_gene_to_idx[gene]
                for gene in genes
                if gene in expr_gene_to_idx
            ]

            if len(aligned_indices) > 0:
                aligned_sets[pathway_name] = aligned_indices

        logger.info(
            f"Aligned {len(aligned_sets)}/{len(self.gene_sets)} pathways "
            f"with expression data"
        )
        return aligned_sets


class DANNHarmonizer(nn.Module):
    """Domain adversarial neural network for pathway harmonization.

    Uses gradient reversal to learn pathway representations that are
    invariant across data domains (e.g., proteomics vs transcriptomics).

    Attributes:
        pathway_dim: Input pathway dimension (e.g., 250).
        hidden_dim: Hidden dimension for feature extractor.
        num_domains: Number of domains (e.g., 2 for proteomics + transcriptomics).
    """

    def __init__(
        self,
        pathway_dim: int = 250,
        hidden_dim: int = 128,
        num_domains: int = 2,
        lambda_adv: float = 0.5,
    ):
        """Initialize DANNHarmonizer.

        Args:
            pathway_dim: Input pathway dimension.
            hidden_dim: Hidden dimension for feature extraction.
            num_domains: Number of domains to harmonize.
            lambda_adv: Adversarial loss weight.
        """
        super().__init__()
        self.pathway_dim = pathway_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.lambda_adv = lambda_adv

        # Feature extractor: pathways → shared representation
        self.feature_extractor = nn.Sequential(
            nn.Linear(pathway_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Domain discriminator
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_domains),
        )

        logger.info(
            f"DANNHarmonizer: {pathway_dim}-dim pathways → {hidden_dim}-dim "
            f"shared features, {num_domains} domains"
        )

    def forward(
        self,
        pathways: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with domain adversarial training.

        Args:
            pathways: (batch_size, pathway_dim) pathway scores.
            domain_labels: (batch_size,) domain indices (0, 1, ..., num_domains-1).
                If None, only extraction is done (inference mode).

        Returns:
            Dict with keys:
                - "features": (batch_size, hidden_dim) harmonized features
                - "domain_pred": (batch_size, num_domains) domain predictions (if training)
        """
        # Feature extraction
        features = self.feature_extractor(pathways)

        out = {"features": features}

        # Domain classification with gradient reversal during training
        if self.training and domain_labels is not None:
            # Gradient reversal: forward = identity, backward = multiply by -lambda
            features_reversed = GradientReversalFunction.apply(
                features, self.lambda_adv
            )
            domain_pred = self.domain_classifier(features_reversed)
            out["domain_pred"] = domain_pred

        return out


class GradientReversalFunction(torch.autograd.Function):
    """Custom autograd function for gradient reversal layer.

    Forward: identity mapping.
    Backward: multiply gradients by -lambda.
    """

    @staticmethod
    def forward(ctx, x, lambda_: float = 1.0):
        """Forward pass (identity).

        Args:
            x: Input tensor.
            lambda_: Reversal strength.

        Returns:
            x (unchanged).
        """
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass (gradient reversal).

        Args:
            grad_output: Gradient of loss w.r.t. output.

        Returns:
            Reversed gradient.
        """
        return -ctx.lambda_ * grad_output, None


class PathwayVAECompressor(nn.Module):
    """Variational autoencoder for pathway dimensionality reduction.

    Compresses 250+ pathway scores into 64-dimensional latent space
    while learning a generative model of pathway co-activation patterns.

    Attributes:
        input_dim: Input pathway dimension (e.g., 250).
        latent_dim: Latent space dimension (default 64).
        hidden_dim: Hidden dimensions for encoder/decoder.
    """

    def __init__(
        self,
        input_dim: int = 250,
        latent_dim: int = 64,
        hidden_dim: int = 128,
    ):
        """Initialize PathwayVAECompressor.

        Args:
            input_dim: Input pathway dimension.
            latent_dim: Latent space dimension.
            hidden_dim: Hidden dimension for encoder/decoder.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Latent space
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        logger.info(
            f"PathwayVAECompressor: {input_dim}-dim → {latent_dim}-dim "
            f"(hidden={hidden_dim})"
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode pathways to latent distribution parameters.

        Args:
            x: (batch_size, input_dim) pathway scores.

        Returns:
            Tuple of (mu, logvar) for latent distribution.
        """
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from latent distribution N(mu, exp(logvar)).

        Args:
            mu: (batch_size, latent_dim) mean.
            logvar: (batch_size, latent_dim) log-variance.

        Returns:
            (batch_size, latent_dim) sample from latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent sample to pathway space.

        Args:
            z: (batch_size, latent_dim) latent sample.

        Returns:
            (batch_size, input_dim) reconstructed pathway scores.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full VAE forward pass.

        Args:
            x: (batch_size, input_dim) pathway scores.

        Returns:
            Dict with keys:
                - "recon": Reconstructed pathways
                - "mu": Latent mean
                - "logvar": Latent log-variance
                - "z": Latent sample
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def vae_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss (reconstruction + KL divergence).

        Args:
            x: Original pathways.
            recon: Reconstructed pathways.
            mu: Latent mean.
            logvar: Latent log-variance.
            beta: KL weighting factor (β-VAE).

        Returns:
            Dict with keys:
                - "recon_loss": MSE reconstruction loss
                - "kl_loss": KL divergence
                - "total_loss": recon_loss + β * kl_loss
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction="mean")

        # KL divergence: 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + beta * kl_loss

        return {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
        }

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (for inference).

        Args:
            x: (batch_size, input_dim) pathway scores.

        Returns:
            (batch_size, latent_dim) latent sample (using mean, no noise).
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu
