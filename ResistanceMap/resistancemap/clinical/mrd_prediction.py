from __future__ import annotations

import logging
import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Type & Data Classes
# ============================================================================

class MRDPrediction(NamedTuple):
    """Multi-timepoint, multi-depth MRD predictions with latent state."""
    mrd_prob_10e5: torch.Tensor  # Shape: (batch, 3) for months 2, 6, 12
    mrd_prob_10e6: torch.Tensor  # Shape: (batch, 3)
    latent_state: torch.Tensor   # Shape: (batch, latent_dim)
    stability_score: torch.Tensor  # Shape: (batch,) ∈ [0, 1]
    treatment_modulation: torch.Tensor  # Shape: (batch,) ∈ [0, 1]


# ============================================================================
# 1. GEPEncoder: Gene Expression Profile → Embedding
# ============================================================================

class GEPEncoder(nn.Module):
    """
    Gene Expression Profile encoder. Reduces high-dimensional gene expression
    to a compact latent embedding via a variational bottleneck, enabling
    reconstruction variance quantification for stability estimation.

    Args:
        num_genes: Number of genes in expression profile.
        latent_dim: Latent embedding dimensionality.
        hidden_dim: Hidden layer size in encoder/decoder.
    """

    def __init__(
        self,
        num_genes: int = 2000,
        latent_dim: int = 128,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.latent_dim = latent_dim

        # Encoder: expression → latent
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Latent reparameterization: mean and log-variance
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder: latent → reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_genes),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode gene expression to latent distribution.

        Args:
            x: Gene expression (batch_size, num_genes)

        Returns:
            mu: Latent means (batch_size, latent_dim)
            logvar: Latent log-variances (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick for differentiable sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct gene expression from latent."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode → sample → decode.

        Returns:
            latent_z: Sampled latent vector
            reconstruction: Reconstructed gene expression
            mu: Latent means
            logvar: Latent log-variances
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return z, recon, mu, logvar


# ============================================================================
# 2. ClinicalEncoder: Structured Clinical Features → Embedding
# ============================================================================

class ClinicalEncoder(nn.Module):
    """
    Clinical encoder combining ISS score, cytogenetics, labs, and treatment.

    Args:
        num_features: Total number of clinical input features.
        latent_dim: Output embedding dimensionality.
    """

    def __init__(
        self,
        num_features: int = 64,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(192, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode clinical features.

        Args:
            x: Clinical features (batch_size, num_features)

        Returns:
            Latent embedding (batch_size, latent_dim)
        """
        return self.net(x)


# ============================================================================
# 3. ImmuneEncoder: Immune Markers → Embedding
# ============================================================================

class ImmuneEncoder(nn.Module):
    """
    Immune encoder for T-cell subsets, NK function, and immune landscape.

    Args:
        num_immune_markers: Number of immune markers (T-cell subsets, NK, etc.).
        latent_dim: Output embedding dimensionality.
    """

    def __init__(
        self,
        num_immune_markers: int = 48,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(num_immune_markers, 224),
            nn.BatchNorm1d(224),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(224, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(160, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode immune markers.

        Args:
            x: Immune markers (batch_size, num_immune_markers)

        Returns:
            Latent embedding (batch_size, latent_dim)
        """
        return self.net(x)


# ============================================================================
# 4. LatentStabilityEstimator: VAE Reconstruction Variance → Stability Score
# ============================================================================

class LatentStabilityEstimator(nn.Module):
    """
    Estimates epigenetic stability from VAE latent space properties:
    - Low reconstruction variance → high reversibility
    - Disentangled latent features → malleable disease state
    - High P(MRD-) probability for malleable states

    Args:
        latent_dim: Dimensionality of latent representation.
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Map latent space properties to stability
        self.stability_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output ∈ [0, 1]
        )

    def forward(
        self,
        latent_z: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate stability score from latent state and reconstruction variance.

        High reconstruction variance (high logvar) suggests labile, plastic state
        → high potential for MRD negativity via treatment.

        Args:
            latent_z: Sampled latent vector (batch_size, latent_dim)
            logvar: Log-variance from VAE encoder (batch_size, latent_dim)

        Returns:
            Stability score (batch_size,) ∈ [0, 1]
        """
        # Mean reconstruction variance as indicator of disease plasticity
        mean_var = torch.exp(logvar).mean(dim=1, keepdim=True)

        # Concat latent z with variance signal
        features = torch.cat([latent_z, mean_var], dim=1)

        # Stability prediction (higher var → higher stability → malleable state)
        stability = self.stability_net(features[:, :self.latent_dim])

        return stability


# ============================================================================
# 5. TrajectoryTransformer: 8-Step Temporal Prediction
# ============================================================================

class TrajectoryTransformer(nn.Module):
    """
    Transformer-based trajectory predictor for latent state evolution.
    Predicts disease state evolution from diagnosis through month 12.

    Args:
        latent_dim: Latent embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        num_timesteps: Number of trajectory timesteps (default 8).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_timesteps: int = 8,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps

        # Positional encoding for timesteps
        self.pos_encoding = self._build_positional_encoding(num_timesteps, latent_dim)

        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers,
        )

        # Output: trajectory of latent states
        self.trajectory_head = nn.Linear(latent_dim, latent_dim)

    @staticmethod
    def _build_positional_encoding(
        num_timesteps: int,
        latent_dim: int,
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding for timesteps."""
        pe = torch.zeros(num_timesteps, latent_dim)
        position = torch.arange(0, num_timesteps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, latent_dim, 2, dtype=torch.float)
            * -(math.log(10000.0) / latent_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if latent_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position[..., 0:latent_dim // 2] * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, initial_latent: torch.Tensor) -> torch.Tensor:
        """
        Predict latent state trajectory.

        Args:
            initial_latent: Initial latent state (batch_size, latent_dim)

        Returns:
            Trajectory of latent states (batch_size, num_timesteps, latent_dim)
        """
        batch_size = initial_latent.shape[0]
        device = initial_latent.device

        # Register positional encoding on correct device
        pos_enc = self.pos_encoding.to(device)

        # Expand initial latent across timesteps and apply positional encoding
        trajectory = initial_latent.unsqueeze(1).expand(-1, self.num_timesteps, -1)
        trajectory = trajectory + pos_enc.unsqueeze(0)

        # Apply transformer
        trajectory = self.transformer(trajectory)
        trajectory = self.trajectory_head(trajectory)

        return trajectory


# ============================================================================
# 6. MRDProbabilityRegressor: Multi-Timepoint, Multi-Depth Predictions
# ============================================================================

class MRDProbabilityRegressor(nn.Module):
    """
    Predicts MRD negativity probabilities at multiple timepoints (2, 6, 12 months)
    and thresholds (10^-5 and 10^-6 sensitivity).

    Args:
        latent_dim: Latent embedding dimension.
        num_timesteps: Number of trajectory timesteps.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_timesteps: int = 8,
    ) -> None:
        super().__init__()

        # Shared feature extraction from trajectory
        self.feature_extractor = nn.Sequential(
            nn.Linear(latent_dim * num_timesteps, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Separate heads for each sensitivity threshold
        # Head 1: 10^-5 sensitivity (3 timepoints: 2, 6, 12 months)
        self.head_10e5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),  # Output probabilities ∈ [0, 1]
        )

        # Head 2: 10^-6 sensitivity (3 timepoints)
        self.head_10e6 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )

    def forward(self, trajectory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict MRD probabilities.

        Args:
            trajectory: Latent state trajectory (batch_size, num_timesteps, latent_dim)

        Returns:
            mrd_prob_10e5: P(MRD-) at 10^-5 (batch_size, 3)
            mrd_prob_10e6: P(MRD-) at 10^-6 (batch_size, 3)
        """
        # Flatten trajectory
        batch_size = trajectory.shape[0]
        flattened = trajectory.reshape(batch_size, -1)

        # Extract features
        features = self.feature_extractor(flattened)

        # Predict at both thresholds
        mrd_10e5 = self.head_10e5(features)
        mrd_10e6 = self.head_10e6(features)

        return mrd_10e5, mrd_10e6


# ============================================================================
# 7. CrossAttentionFusion: Multi-Modal Encoder Output Fusion
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism to fuse embeddings from GEP, clinical, and immune encoders.

    Args:
        embed_dim: Dimensionality of each encoder's output.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Query from GEP, keys/values from clinical and immune
        self.gep_to_clinical = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.gep_to_immune = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
            dropout=0.1,
        )

        # Fusion projection
        self.fusion = nn.Linear(embed_dim * 3, embed_dim)

    def forward(
        self,
        gep_embed: torch.Tensor,
        clinical_embed: torch.Tensor,
        immune_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse multi-modal embeddings via cross-attention.

        Args:
            gep_embed: GEP encoder output (batch_size, embed_dim)
            clinical_embed: Clinical encoder output (batch_size, embed_dim)
            immune_embed: Immune encoder output (batch_size, embed_dim)

        Returns:
            Fused embedding (batch_size, embed_dim)
        """
        # Reshape for multi-head attention (seq_len=1)
        gep = gep_embed.unsqueeze(1)
        clinical = clinical_embed.unsqueeze(1)
        immune = immune_embed.unsqueeze(1)

        # Cross-attention: GEP queries clinical and immune knowledge
        attn_clinical, _ = self.gep_to_clinical(gep, clinical, clinical)
        attn_immune, _ = self.gep_to_immune(gep, immune, immune)

        # Concatenate and fuse
        fused = torch.cat([gep, attn_clinical, attn_immune], dim=-1)
        fused = fused.squeeze(1)  # Remove seq dimension
        fused = self.fusion(fused)

        return fused


# ============================================================================
# 8. MRDNegPredictor: Top-Level Orchestrator
# ============================================================================

class MRDNegPredictor(nn.Module):
    """
    Complete MRD Negativity Prediction system. Integrates all components:
    - Multi-modal encoders (GEP, clinical, immune)
    - Cross-attention fusion
    - Latent stability estimation
    - Trajectory prediction
    - Multi-timepoint, multi-depth MRD probability regression

    Args:
        num_genes: Number of genes in expression profile.
        num_clinical_features: Number of clinical input features.
        num_immune_markers: Number of immune markers.
        latent_dim: Latent embedding dimensionality.
        num_heads: Number of attention heads in transformers.
        num_layers: Number of transformer layers.
        num_timesteps: Number of trajectory timesteps.
    """

    def __init__(
        self,
        num_genes: int = 2000,
        num_clinical_features: int = 64,
        num_immune_markers: int = 48,
        latent_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_timesteps: int = 8,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps

        logger.info("Initializing MRDNegPredictor with latent_dim=%d", latent_dim)

        # Encoders
        self.gep_encoder = GEPEncoder(
            num_genes=num_genes,
            latent_dim=latent_dim,
        )
        self.clinical_encoder = ClinicalEncoder(
            num_features=num_clinical_features,
            latent_dim=latent_dim,
        )
        self.immune_encoder = ImmuneEncoder(
            num_immune_markers=num_immune_markers,
            latent_dim=latent_dim,
        )

        # Multi-modal fusion
        self.cross_attention = CrossAttentionFusion(
            embed_dim=latent_dim,
            num_heads=num_heads,
        )

        # Latent stability estimator
        self.stability_estimator = LatentStabilityEstimator(latent_dim)

        # Trajectory prediction
        self.trajectory_transformer = TrajectoryTransformer(
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
        )

        # MRD probability regression
        self.mrd_regressor = MRDProbabilityRegressor(
            latent_dim=latent_dim,
            num_timesteps=num_timesteps,
        )

    def forward(
        self,
        gene_expr: torch.Tensor,
        clinical_feats: torch.Tensor,
        immune_markers: torch.Tensor,
    ) -> MRDPrediction:
        """
        Forward pass: complete MRD negativity prediction pipeline.

        Args:
            gene_expr: Gene expression profile (batch_size, num_genes)
            clinical_feats: Clinical features (batch_size, num_clinical_features)
            immune_markers: Immune markers (batch_size, num_immune_markers)

        Returns:
            MRDPrediction namedtuple with:
                - mrd_prob_10e5: P(MRD-) at 10^-5 threshold (batch_size, 3)
                - mrd_prob_10e6: P(MRD-) at 10^-6 threshold (batch_size, 3)
                - latent_state: Fused latent representation (batch_size, latent_dim)
                - stability_score: Epigenetic stability (batch_size,)
                - treatment_modulation: Treatment intensity modulation (batch_size,)
        """
        # 1. Encode modalities
        gep_z, gep_recon, gep_mu, gep_logvar = self.gep_encoder(gene_expr)
        clinical_embed = self.clinical_encoder(clinical_feats)
        immune_embed = self.immune_encoder(immune_markers)

        # 2. Cross-attention fusion
        fused_latent = self.cross_attention(gep_z, clinical_embed, immune_embed)

        # 3. Estimate latent stability from VAE reconstruction variance
        stability = self.stability_estimator(gep_z, gep_logvar)

        # 4. Predict latent state trajectory
        trajectory = self.trajectory_transformer(fused_latent)

        # 5. Regress MRD probabilities at 10^-5 and 10^-6 thresholds
        mrd_10e5, mrd_10e6 = self.mrd_regressor(trajectory)

        # 6. Treatment modulation: stability score modulates recommended intensity
        # Higher stability → more malleable → can use lower intensity
        # Lower stability → more resistant → may need higher intensity
        treatment_modulation = 1.0 - stability

        return MRDPrediction(
            mrd_prob_10e5=mrd_10e5,
            mrd_prob_10e6=mrd_10e6,
            latent_state=fused_latent,
            stability_score=stability.squeeze(-1),
            treatment_modulation=treatment_modulation.squeeze(-1),
        )

    def get_reconstruction_loss(
        self,
        gene_expr: torch.Tensor,
        gep_recon: torch.Tensor,
        gep_mu: torch.Tensor,
        gep_logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss components for training.

        Args:
            gene_expr: Original gene expression
            gep_recon: Reconstructed gene expression from GEP encoder
            gep_mu: Latent means
            gep_logvar: Latent log-variances

        Returns:
            reconstruction_loss: MSE between original and reconstructed
            kl_divergence: KL(q || p) for latent distribution
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(gep_recon, gene_expr, reduction='mean')

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(
            1 + gep_logvar - gep_mu.pow(2) - gep_logvar.exp()
        )

        return recon_loss, kl_loss

    def get_latent_disentanglement_loss(
        self,
        gep_logvar: torch.Tensor,
        target_variance: float = 0.1,
    ) -> torch.Tensor:
        """
        Encourage disentangled latent features via variance regularization.
        Low, balanced variances across latent dimensions = disentanglement.

        Args:
            gep_logvar: Latent log-variances (batch_size, latent_dim)
            target_variance: Target reconstruction variance

        Returns:
            Disentanglement loss
        """
        var = torch.exp(gep_logvar)
        mean_var = var.mean(dim=1)

        # Penalize high variance and unbalanced variance
        variance_loss = torch.mean(mean_var)
        balance_loss = torch.mean(var.std(dim=1))

        return variance_loss + 0.5 * balance_loss
