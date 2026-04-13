"""Gap 4.6: Cell-Line-to-Patient Transfer Learning via Domain Adaptation.

Implements multi-phase adversarial training to transfer drug response models
from cell lines (CCLE) to patient samples (TCGA, PDX) while maintaining
reconstruction accuracy.

Architecture:
    - GradientReversalFunction: Custom autograd for gradient reversal (λ-scaled)
    - DomainAdversarialVAE: Shared encoder, VAE decoder, domain classifier with GRL
    - WassersteinDomainLoss: Optimal transport-based domain distance
    - MultiPhaseDomainAdaptationTrainer: Phase-wise curriculum training
    - DriftDetector: Monitors domain shift and triggers retraining

Training Pipeline:
    Phase 1: CCLE supervised learning (VAE + drug response)
    Phase 2: Add patient data with domain-adversarial DANN loss
    Phase 3: PDX bridging with triplet loss
    Phase 4: Final validation on held-out patient cohort
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


class GradientReversalFunction(torch.autograd.Function):
    """Custom autograd function for gradient reversal with configurable λ.

    Implements the gradient reversal technique from Ganin et al. (2016).
    During forward pass, acts as identity.
    During backward pass, negates and scales gradients by lambda_.

    This enables domain-adversarial training by making the latent space
    unable to discriminate between domains, thus learning domain-invariant
    representations.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Forward pass: identity function.

        Args:
            ctx: Context for backprop.
            x: Input tensor.
            lambda_: Scaling factor for gradient reversal.

        Returns:
            x unchanged.
        """
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass: negate and scale gradients.

        Args:
            grad_output: Upstream gradients.

        Returns:
            Tuple of (reversed gradients, None for lambda_).
        """
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction in a PyTorch module.

    Args:
        lambda_: Initial scaling factor for gradient reversal (default 1.0).
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal.

        Args:
            x: (B, dim) input tensor.

        Returns:
            (B, dim) tensor with reversed gradients during backprop.
        """
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        """Dynamically adjust gradient reversal strength."""
        self.lambda_ = lambda_


class DomainAdversarialVAE(nn.Module):
    """VAE with adversarial domain classifier for domain-invariant representations.

    Architecture:
        - Shared encoder: 10000 → 256 → 128 → 64 (latent_dim)
        - VAE decoder: 64 → 128 → 256 → 10000 (reconstruction)
        - Domain classifier with GRL: 64 → 32 → 8 → 1 (domain logit)
        - Drug response head: 64 → 32 → 1 (IC50 prediction)

    Combined loss:
        L = α·L_VAE + β·L_domain + γ·L_supervised + δ·L_diversity

    Args:
        input_dim: Input feature dimension (default 10000 for protein features).
        latent_dim: Latent space dimension (default 64).
        domain_lambda: Initial gradient reversal strength (default 1.0).
        use_batch_norm: Whether to use batch normalization (default True).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        input_dim: int = 10000,
        latent_dim: int = 64,
        domain_lambda: float = 1.0,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.domain_lambda = domain_lambda
        self.use_batch_norm = use_batch_norm

        # Shared encoder: 10000 → 256 → 128 → 64
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in [256, 128]:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.GELU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent projections (VAE mu and log_var)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

        # VAE decoder: 64 → 128 → 256 → 10000
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in [128, 256]:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.GELU())
            decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Domain classifier with gradient reversal: 64 → 32 → 8 → 1
        self.grl = GradientReversalLayer(domain_lambda)
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(32, 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(8, 1),
        )

        # Drug response head: 64 → 32 → 1
        self.response_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: (B, input_dim) input features.

        Returns:
            Tuple of (mu, log_var), each (B, latent_dim).
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std.

        Args:
            mu: (B, latent_dim) mean of posterior.
            log_var: (B, latent_dim) log variance.

        Returns:
            (B, latent_dim) sampled latent vector.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            z: (B, latent_dim) latent vector.

        Returns:
            (B, input_dim) reconstructed input.
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass.

        Args:
            x: (B, input_dim) input features.

        Returns:
            Tuple of (reconstruction, mu, log_var).
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def classify_domain(self, z: torch.Tensor) -> torch.Tensor:
        """Classify domain from latent representation with gradient reversal.

        Args:
            z: (B, latent_dim) latent vector.

        Returns:
            (B, 1) domain logit (binary: cell_line vs patient).
        """
        z_reversed = self.grl(z)
        return self.domain_classifier(z_reversed)

    def predict_response(self, z: torch.Tensor) -> torch.Tensor:
        """Predict drug response from latent representation.

        Args:
            z: (B, latent_dim) latent vector.

        Returns:
            (B, 1) predicted drug response (IC50).
        """
        return self.response_head(z)

    def set_domain_lambda(self, lambda_: float) -> None:
        """Dynamically adjust gradient reversal strength."""
        self.grl.set_lambda(lambda_)


class WassersteinDomainLoss(nn.Module):
    """Wasserstein distance with gradient penalty for domain adaptation.

    Implements Wasserstein GAN loss with gradient penalty (Gulrajani et al., 2017).
    More stable than JS divergence for measuring domain discrepancy.

    Args:
        gradient_penalty_weight: Weight for gradient penalty (default 10.0).
    """

    def __init__(self, gradient_penalty_weight: float = 10.0) -> None:
        super().__init__()
        self.gp_weight = gradient_penalty_weight

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        discriminator: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Wasserstein loss with gradient penalty.

        Args:
            source_features: (B, dim) source domain representations.
            target_features: (B, dim) target domain representations.
            discriminator: Function that scores domain membership.

        Returns:
            Tuple of (wasserstein_loss, gradient_penalty).
        """
        batch_size = min(source_features.shape[0], target_features.shape[0])
        source_features = source_features[:batch_size]
        target_features = target_features[:batch_size]

        # Score source and target
        source_score = discriminator(source_features)  # (B, 1)
        target_score = discriminator(target_features)  # (B, 1)

        # Wasserstein loss: E[D(source)] - E[D(target)]
        w_loss = source_score.mean() - target_score.mean()

        # Gradient penalty on interpolates
        alpha = torch.rand(batch_size, 1, device=source_features.device)
        interpolates = (
            alpha * source_features + (1 - alpha) * target_features
        )
        interpolates.requires_grad_(True)

        d_interpolates = discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients_norm = (
            torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-8)
        )
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()

        return w_loss, gradient_penalty


class DriftDetector(nn.Module):
    """Monitors domain classifier confidence to detect distribution shift.

    Tracks the entropy and confidence of domain predictions.
    Triggers retraining when drift exceeds threshold.

    Args:
        confidence_threshold: Alert if domain confidence > this (default 0.8).
        entropy_threshold: Alert if entropy < this (default 0.3).
        window_size: Rolling window for drift detection (default 100).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        entropy_threshold: float = 0.3,
        window_size: int = 100,
    ) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        self.confidence_history: list[float] = []
        self.entropy_history: list[float] = []

    def compute_metrics(
        self, domain_logits: torch.Tensor
    ) -> Tuple[float, float]:
        """Compute domain confidence and entropy.

        Args:
            domain_logits: (B, 1) domain classifier outputs.

        Returns:
            Tuple of (mean_confidence, mean_entropy).
        """
        probs = torch.sigmoid(domain_logits).detach().cpu()  # (B, 1)
        confidence = torch.abs(probs - 0.5).mean().item() * 2  # 0 to 1

        # Entropy: -[p*log(p) + (1-p)*log(1-p)]
        p = probs.clamp(1e-7, 1 - 1e-7)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean().item()

        return confidence, entropy

    def update(self, domain_logits: torch.Tensor) -> None:
        """Update drift detector with new batch of domain logits.

        Args:
            domain_logits: (B, 1) domain predictions.
        """
        confidence, entropy = self.compute_metrics(domain_logits)
        self.confidence_history.append(confidence)
        self.entropy_history.append(entropy)

        # Keep rolling window
        if len(self.confidence_history) > self.window_size:
            self.confidence_history.pop(0)
            self.entropy_history.pop(0)

    def check_drift(self) -> Tuple[bool, Dict[str, float]]:
        """Check if domain shift detected.

        Returns:
            Tuple of (drift_detected, metrics_dict).
        """
        if len(self.confidence_history) < 2:
            return False, {}

        mean_confidence = sum(self.confidence_history) / len(
            self.confidence_history
        )
        mean_entropy = sum(self.entropy_history) / len(self.entropy_history)

        drift_detected = (
            mean_confidence > self.confidence_threshold
            or mean_entropy < self.entropy_threshold
        )

        return drift_detected, {
            "mean_confidence": mean_confidence,
            "mean_entropy": mean_entropy,
            "threshold_confidence": self.confidence_threshold,
            "threshold_entropy": self.entropy_threshold,
        }


class MultiPhaseDomainAdaptationTrainer:
    """Multi-phase domain adaptation training orchestrator.

    Phases:
        1. CCLE supervised: Train VAE + response head on cell lines
        2. Patient DANN: Add patient data with domain-adversarial loss
        3. PDX bridge: Optional intermediate model on PDX samples
        4. Validation: Evaluate on held-out patient cohort

    Args:
        model: DomainAdversarialVAE instance.
        device: Training device.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight.
        gradient_clip_norm: Gradient clipping threshold.
    """

    def __init__(
        self,
        model: DomainAdversarialVAE,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scaler = GradScaler("cuda" if device.type == "cuda" else None)
        self.drift_detector = DriftDetector()
        self.wasserstein_loss = WassersteinDomainLoss()

    def _kl_divergence(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence from N(mu, var) to N(0, I)."""
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    def phase1_ccle_supervised(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        alpha_vae: float = 1.0,
        gamma_supervised: float = 1.0,
    ) -> Dict[str, Any]:
        """Phase 1: CCLE supervised learning (VAE + drug response).

        Args:
            train_loader: DataLoader with CCLE samples.
            val_loader: Validation DataLoader.
            epochs: Number of training epochs.
            alpha_vae: Weight for VAE loss.
            gamma_supervised: Weight for supervised response loss.

        Returns:
            Dict with metrics and checkpoint info.
        """
        logger.info("=== Phase 1: CCLE Supervised Learning ===")
        self.model.train()

        best_val_loss = float("inf")
        metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "response_loss": [],
        }

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            epoch_response = 0.0

            for batch in train_loader:
                x = batch["proteomics"].to(self.device)
                ic50 = batch.get("ic50", None)

                self.optimizer.zero_grad(set_to_none=True)

                with autocast("cuda" if self.device.type == "cuda" else None):
                    # VAE forward
                    recon, mu, log_var = self.model(x)
                    recon_loss = F.mse_loss(recon, x)
                    kl_loss = self._kl_divergence(mu, log_var)

                    loss = alpha_vae * (recon_loss + kl_loss)

                    # Supervised response loss if labels available
                    response_loss = 0.0
                    if ic50 is not None:
                        ic50 = ic50.to(self.device).unsqueeze(-1)
                        z = self.model.reparameterize(mu, log_var)
                        pred_response = self.model.predict_response(z)
                        response_loss = F.mse_loss(pred_response, ic50)
                        loss = loss + gamma_supervised * response_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
                if isinstance(response_loss, torch.Tensor):
                    epoch_response += response_loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["proteomics"].to(self.device)
                    recon, mu, log_var = self.model(x)
                    recon_loss = F.mse_loss(recon, x)
                    kl_loss = self._kl_divergence(mu, log_var)
                    val_loss += (recon_loss + kl_loss).item()

            self.model.train()
            avg_train_loss = epoch_loss / max(len(train_loader), 1)
            avg_val_loss = val_loss / max(len(val_loader), 1)

            metrics_history["train_loss"].append(avg_train_loss)
            metrics_history["val_loss"].append(avg_val_loss)
            metrics_history["recon_loss"].append(
                epoch_recon / max(len(train_loader), 1)
            )
            metrics_history["kl_loss"].append(
                epoch_kl / max(len(train_loader), 1)
            )
            metrics_history["response_loss"].append(
                epoch_response / max(len(train_loader), 1)
            )

            logger.info(
                f"Phase1 Epoch {epoch + 1}/{epochs}: "
                f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

        return {"best_val_loss": best_val_loss, "metrics": metrics_history}

    def phase2_patient_dann(
        self,
        ccle_loader: DataLoader,
        patient_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        alpha_vae: float = 1.0,
        beta_domain: float = 0.1,
        gamma_supervised: float = 1.0,
        delta_diversity: float = 0.01,
    ) -> Dict[str, Any]:
        """Phase 2: Patient data with domain-adversarial loss (DANN).

        Args:
            ccle_loader: CCLE training data.
            patient_loader: Patient training data.
            val_loader: Validation DataLoader.
            epochs: Number of epochs.
            alpha_vae: VAE loss weight.
            beta_domain: Domain adversarial loss weight.
            gamma_supervised: Supervised loss weight.
            delta_diversity: Latent diversity regularization weight.

        Returns:
            Dict with metrics and checkpoint info.
        """
        logger.info("=== Phase 2: Patient DANN ===")
        self.model.train()

        best_val_loss = float("inf")
        metrics_history = {
            "train_loss": [],
            "domain_loss": [],
            "drift_detected": [],
        }

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_domain = 0.0

            # Alternate between CCLE and patient batches
            max_batches = max(len(ccle_loader), len(patient_loader))
            ccle_iter = iter(ccle_loader)
            patient_iter = iter(patient_loader)

            for _ in range(max_batches):
                # Get CCLE batch (source)
                try:
                    ccle_batch = next(ccle_iter)
                except StopIteration:
                    ccle_iter = iter(ccle_loader)
                    ccle_batch = next(ccle_iter)

                # Get patient batch (target)
                try:
                    patient_batch = next(patient_iter)
                except StopIteration:
                    patient_iter = iter(patient_loader)
                    patient_batch = next(patient_iter)

                x_source = ccle_batch["proteomics"].to(self.device)
                x_target = patient_batch["proteomics"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                with autocast("cuda" if self.device.type == "cuda" else None):
                    # Source domain (CCLE)
                    recon_s, mu_s, log_var_s = self.model(x_source)
                    recon_loss_s = F.mse_loss(recon_s, x_source)
                    kl_loss = self._kl_divergence(mu_s, log_var_s)
                    z_s = self.model.reparameterize(mu_s, log_var_s)

                    # Target domain (patient)
                    recon_t, mu_t, log_var_t = self.model(x_target)
                    z_t = self.model.reparameterize(mu_t, log_var_t)

                    # Domain classification (GRL applied inside classify_domain)
                    domain_logits_s = self.model.classify_domain(z_s)
                    domain_logits_t = self.model.classify_domain(z_t)

                    # Domain loss: fool discriminator (wants confusion)
                    domain_loss = (
                        F.binary_cross_entropy_with_logits(
                            domain_logits_s, torch.zeros_like(domain_logits_s)
                        )
                        + F.binary_cross_entropy_with_logits(
                            domain_logits_t, torch.ones_like(domain_logits_t)
                        )
                    ) / 2.0

                    # Latent diversity regularization
                    diversity_loss = -torch.mean(
                        torch.norm(mu_s - mu_t.mean(dim=0), dim=1)
                    )

                    loss = (
                        alpha_vae * (recon_loss_s + kl_loss)
                        + beta_domain * domain_loss
                        + delta_diversity * diversity_loss
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                epoch_domain += domain_loss.item()

                # Drift detection
                self.drift_detector.update(domain_logits_t)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["proteomics"].to(self.device)
                    recon, _, _ = self.model(x)
                    val_loss += F.mse_loss(recon, x).item()

            self.model.train()

            drift_detected, drift_metrics = self.drift_detector.check_drift()
            metrics_history["drift_detected"].append(drift_detected)

            avg_loss = epoch_loss / max_batches
            avg_domain = epoch_domain / max_batches
            avg_val_loss = val_loss / max(len(val_loader), 1)

            metrics_history["train_loss"].append(avg_loss)
            metrics_history["domain_loss"].append(avg_domain)

            logger.info(
                f"Phase2 Epoch {epoch + 1}/{epochs}: "
                f"train_loss={avg_loss:.4f} domain_loss={avg_domain:.4f} "
                f"drift={drift_detected}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

        return {"best_val_loss": best_val_loss, "metrics": metrics_history}

    def phase3_pdx_bridge(
        self,
        pdx_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        alpha_vae: float = 1.0,
        beta_domain: float = 0.1,
    ) -> Dict[str, Any]:
        """Phase 3: Optional PDX intermediate bridging.

        Args:
            pdx_loader: PDX training data.
            val_loader: Validation DataLoader.
            epochs: Number of epochs.
            alpha_vae: VAE loss weight.
            beta_domain: Domain loss weight.

        Returns:
            Dict with metrics and checkpoint info.
        """
        logger.info("=== Phase 3: PDX Bridge ===")
        self.model.train()

        best_val_loss = float("inf")
        metrics_history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch in pdx_loader:
                x = batch["proteomics"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                with autocast("cuda" if self.device.type == "cuda" else None):
                    recon, mu, log_var = self.model(x)
                    recon_loss = F.mse_loss(recon, x)
                    kl_loss = self._kl_divergence(mu, log_var)
                    loss = alpha_vae * (recon_loss + kl_loss)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["proteomics"].to(self.device)
                    recon, _, _ = self.model(x)
                    val_loss += F.mse_loss(recon, x).item()

            self.model.train()

            avg_loss = epoch_loss / max(len(pdx_loader), 1)
            avg_val_loss = val_loss / max(len(val_loader), 1)

            metrics_history["train_loss"].append(avg_loss)
            metrics_history["val_loss"].append(avg_val_loss)

            logger.info(
                f"Phase3 Epoch {epoch + 1}/{epochs}: "
                f"train_loss={avg_loss:.4f} val_loss={avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

        return {"best_val_loss": best_val_loss, "metrics": metrics_history}

    def phase4_validation(
        self, test_loader: DataLoader
    ) -> Dict[str, Any]:
        """Phase 4: Final validation on held-out patient cohort.

        Args:
            test_loader: Test DataLoader.

        Returns:
            Dict with test metrics.
        """
        logger.info("=== Phase 4: Validation ===")
        self.model.eval()

        test_loss = 0.0
        test_response_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                x = batch["proteomics"].to(self.device)
                recon, mu, log_var = self.model(x)

                # Reconstruction loss
                recon_loss = F.mse_loss(recon, x)
                test_loss += recon_loss.item()

                # Response loss if available
                if "ic50" in batch:
                    ic50 = batch["ic50"].to(self.device).unsqueeze(-1)
                    z = self.model.reparameterize(mu, log_var)
                    pred_response = self.model.predict_response(z)
                    response_loss = F.mse_loss(pred_response, ic50)
                    test_response_loss += response_loss.item()

                n_batches += 1

        avg_recon_loss = test_loss / max(n_batches, 1)
        avg_response_loss = test_response_loss / max(n_batches, 1)

        logger.info(
            f"Phase4 Test: recon_loss={avg_recon_loss:.4f} "
            f"response_loss={avg_response_loss:.4f}"
        )

        return {
            "test_recon_loss": avg_recon_loss,
            "test_response_loss": avg_response_loss,
        }
