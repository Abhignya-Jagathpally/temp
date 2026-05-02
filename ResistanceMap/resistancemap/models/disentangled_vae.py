"""Beta-Total Correlation VAE with Anchored Biological Dimensions.

Replaces the standard VAE with a beta-TCVAE (Chen et al., 2018) that
produces disentangled, interpretable latent factors for single-cell
multi-omics data.

Key innovations:
    1. **beta-TCVAE objective**: Decomposes the ELBO into three terms:
           ELBO = E[log p(x|z)] - I(x;z) - beta*TC(z) - sum_j KL(q(z_j)||p(z_j))
       where TC is the total correlation. Penalizing TC with beta > 1
       encourages statistical independence between latent dimensions.
    2. **Anchored dimensions**: First K_anchor dimensions are supervised
       to predict known biological programs (cell cycle phase, drug efflux
       score, stemness signature) via auxiliary classification/regression
       losses. This pins biological meaning to specific latent axes.
    3. **DCI metrics**: Computes Disentanglement, Completeness, and
       Informativeness (Eastwood & Williams, 2018) during training for
       monitoring disentanglement quality.
    4. **Traversal generation**: For each latent dimension, generates
       latent traversals to visualize what biological change it encodes.

Theoretical grounding:
    - Chen et al. (2018). "Isolating Sources of Disentanglement in
      Variational Autoencoders." NeurIPS.
    - Eastwood & Williams (2018). "A Framework for the Quantitative
      Evaluation of Disentangled Representations." ICLR.
    - Hyvarinen & Morioka (2017). "Nonlinear ICA of Temporally
      Dependent Stationary Sources." AISTATS.
    - Khemakhem et al. (2020). "Variational Autoencoders and Nonlinear
      ICA: A Unifying Framework." AISTATS.

Authors: ResistanceMap Team
License: MIT
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

logger = logging.getLogger(__name__)

__all__ = [
    "DisentangledVAEConfig",
    "AnchorSpec",
    "Encoder",
    "Decoder",
    "DisentangledVAE",
    "DCIMetrics",
    "TraversalGenerator",
]


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class AnchorSpec:
    """Specification for an anchored latent dimension.

    Attributes:
        dim_index: Which latent dimension to anchor.
        name: Human-readable name (e.g., "cell_cycle_phase").
        task: "classification" or "regression".
        n_classes: Number of classes (for classification tasks).
        marker_genes: Gene names used to compute the anchor target.
        weight: Loss weight for this anchor.
    """
    dim_index: int
    name: str
    task: str = "regression"  # "classification" or "regression"
    n_classes: int = 2
    marker_genes: List[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class DisentangledVAEConfig:
    """Configuration for the disentangled VAE.

    Attributes:
        input_dim: Dimension of input features (e.g., n_genes).
        latent_dim: Total latent dimension.
        hidden_dims: Encoder/decoder hidden layer sizes.
        beta: Weight on total correlation (beta > 1 for disentanglement).
        alpha: Weight on mutual information term (usually 1).
        gamma: Weight on dimension-wise KL (usually 1).
        n_anchors: Number of anchored dimensions.
        anchor_specs: List of AnchorSpec for supervised dimensions.
        tc_estimation: Method for TC estimation ('minibatch', 'mws').
        dropout: Dropout rate.
        activation: Activation function name ('gelu', 'silu', 'relu').
        batch_norm: Whether to use batch normalization.
    """
    input_dim: int = 2000
    latent_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    beta: float = 4.0
    alpha: float = 1.0
    gamma: float = 1.0
    n_anchors: int = 4
    anchor_specs: List[AnchorSpec] = field(default_factory=lambda: [
        AnchorSpec(0, "drug_efflux", "regression", marker_genes=["ABCB1", "ABCC1"]),
        AnchorSpec(1, "cell_cycle", "classification", n_classes=4),
        AnchorSpec(2, "stemness", "regression", marker_genes=["SOX2", "NANOG"]),
        AnchorSpec(3, "dna_damage", "regression", marker_genes=["ATM", "TP53"]),
    ])
    tc_estimation: str = "minibatch"
    dropout: float = 0.1
    activation: str = "gelu"
    batch_norm: bool = True


# ===================================================================
# Encoder and Decoder
# ===================================================================

def _get_activation(name: str) -> nn.Module:
    """Get activation module by name."""
    activations = {
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "tanh": nn.Tanh(),
    }
    return activations.get(name.lower(), nn.GELU())


class Encoder(nn.Module):
    """VAE encoder: x -> (mu, log_var).

    Architecture: input_dim -> [hidden layers with BN/Dropout] -> 2*latent_dim

    Args:
        input_dim: Input feature dimension.
        latent_dim: Latent space dimension.
        hidden_dims: List of hidden layer sizes.
        dropout: Dropout rate.
        activation: Activation function name.
        batch_norm: Whether to use batch normalization.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(_get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Initialize for stable training
        nn.init.xavier_normal_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_normal_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to posterior parameters.

        Args:
            x: (B, input_dim) input features.

        Returns:
            mu: (B, latent_dim) posterior mean.
            log_var: (B, latent_dim) posterior log-variance.
        """
        h = self.backbone(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mu, log_var


class Decoder(nn.Module):
    """VAE decoder: z -> x_recon.

    Architecture: latent_dim -> [hidden layers with BN/Dropout] -> input_dim

    Args:
        latent_dim: Latent space dimension.
        output_dim: Output (reconstruction) dimension.
        hidden_dims: List of hidden layer sizes (reverse of encoder).
        dropout: Dropout rate.
        activation: Activation function name.
        batch_norm: Whether to use batch normalization.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        # Reverse hidden dims for decoder
        rev_dims = list(reversed(hidden_dims))
        prev_dim = latent_dim
        for h_dim in rev_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(_get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction.

        Args:
            z: (B, latent_dim) latent codes.

        Returns:
            x_recon: (B, output_dim) reconstruction.
        """
        return self.net(z)


# ===================================================================
# Total Correlation Estimation
# ===================================================================

def _log_density_gaussian(z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Compute log N(z; mu, sigma^2) for each sample-distribution pair.

    Args:
        z: (B, D) latent samples.
        mu: (B, D) distribution means.
        log_var: (B, D) distribution log-variances.

    Returns:
        log_density: (B, B, D) where [i, j, d] = log N(z_i; mu_j, sigma_j^2).
            First dim indexes samples, second indexes distributions.
    """
    D = z.shape[1]
    # z_i, mu_j pairwise
    z_expand = z.unsqueeze(1)         # (B, 1, D)
    mu_expand = mu.unsqueeze(0)       # (1, B, D)
    logvar_expand = log_var.unsqueeze(0)  # (1, B, D)

    log_density = -0.5 * (
        math.log(2 * math.pi)
        + logvar_expand
        + (z_expand - mu_expand) ** 2 / (torch.exp(logvar_expand) + 1e-8)
    )
    return log_density  # (B, B, D)


def estimate_tc_minibatch(
    z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate the three ELBO decomposition terms using the minibatch
    weighted sampling estimator from Chen et al. (2018).

    ELBO = E[log p(x|z)] - I(x;z) - TC(z) - sum_j KL(q(z_j)||p(z_j))

    where:
        I(x;z) = mutual information between data and latent
        TC(z) = KL(q(z) || prod_j q(z_j)) = total correlation
        dim_KL = sum_j KL(q(z_j) || p(z_j)) = dimension-wise KL

    The minibatch estimator uses log q(z) approx log (1/M) sum_j q(z|x_j)
    where M is the minibatch size.

    Args:
        z: (B, D) reparameterized latent samples.
        mu: (B, D) posterior means.
        log_var: (B, D) posterior log-variances.

    Returns:
        mi: Scalar mutual information estimate.
        tc: Scalar total correlation estimate.
        dim_kl: Scalar dimension-wise KL estimate.
    """
    B, D = z.shape

    # log q(z_i | x_j) for all i, j pairs and dimensions
    log_qz_xj = _log_density_gaussian(z, mu, log_var)  # (B, B, D)

    # log q(z) = log (1/B) sum_j q(z|x_j) for each sample z_i
    # Using logsumexp for numerical stability
    log_qz = torch.logsumexp(log_qz_xj.sum(dim=2), dim=1) - math.log(B)  # (B,)

    # log q(z_j) for each dimension separately
    log_qz_j = torch.logsumexp(log_qz_xj, dim=1) - math.log(B)  # (B, D)

    # log prod_j q(z_j) = sum_j log q(z_j)
    log_prod_qz_j = log_qz_j.sum(dim=1)  # (B,)

    # log q(z|x) for the matched pairs (diagonal)
    log_qz_x = log_qz_xj.sum(dim=2)  # (B, B)
    log_qz_xi = torch.diag(log_qz_x)  # (B,) -- log q(z_i | x_i)

    # log p(z) = sum_d log N(z_d; 0, 1)
    log_pz = Normal(0, 1).log_prob(z).sum(dim=1)  # (B,)

    # Mutual Information: E[log q(z|x) - log q(z)]
    mi = (log_qz_xi - log_qz).mean()

    # Total Correlation: E[log q(z) - log prod_j q(z_j)]
    tc = (log_qz - log_prod_qz_j).mean()

    # Dimension-wise KL: E[log prod_j q(z_j) - log p(z)]
    dim_kl = (log_prod_qz_j - log_pz).mean()

    return mi, tc, dim_kl


# ===================================================================
# Anchor Heads
# ===================================================================

class AnchorHead(nn.Module):
    """Auxiliary prediction head for an anchored latent dimension.

    Takes a single latent dimension and predicts a biological readout.

    Args:
        spec: AnchorSpec defining the task.
    """

    def __init__(self, spec: AnchorSpec) -> None:
        super().__init__()
        self.spec = spec
        if spec.task == "classification":
            self.head = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, spec.n_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    def forward(self, z_dim: torch.Tensor) -> torch.Tensor:
        """Predict from a single latent dimension.

        Args:
            z_dim: (B,) or (B, 1) values of the anchored dimension.

        Returns:
            pred: (B, n_classes) for classification or (B,) for regression.
        """
        if z_dim.dim() == 1:
            z_dim = z_dim.unsqueeze(-1)
        out = self.head(z_dim)
        if self.spec.task == "regression":
            out = out.squeeze(-1)
        return out

    def loss(
        self, z_dim: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute anchor supervision loss.

        Args:
            z_dim: (B,) latent dimension values.
            target: (B,) targets (class indices or continuous scores).

        Returns:
            Scalar loss.
        """
        pred = self.forward(z_dim)
        if self.spec.task == "classification":
            return F.cross_entropy(pred, target.long())
        else:
            return F.mse_loss(pred, target.float())


# ===================================================================
# Main Disentangled VAE
# ===================================================================

class DisentangledVAE(nn.Module):
    """Beta-Total Correlation VAE with anchored biological dimensions.

    The loss function decomposes as:
        L = recon_loss + alpha * MI + beta * TC + gamma * dim_KL
            + sum_k anchor_weight_k * anchor_loss_k

    where beta > 1 penalizes total correlation to encourage disentanglement,
    and anchor losses pin specific latent dimensions to known biology.

    Args:
        config: DisentangledVAEConfig.
    """

    def __init__(self, config: DisentangledVAEConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = Encoder(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            activation=config.activation,
            batch_norm=config.batch_norm,
        )
        self.decoder = Decoder(
            latent_dim=config.latent_dim,
            output_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            activation=config.activation,
            batch_norm=config.batch_norm,
        )

        # Anchor heads
        self.anchor_heads = nn.ModuleDict()
        for spec in config.anchor_specs:
            if spec.dim_index < config.latent_dim:
                self.anchor_heads[spec.name] = AnchorHead(spec)

        # Training step counter for KL annealing
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon.

        Args:
            mu: (B, D) posterior mean.
            log_var: (B, D) posterior log-variance.

        Returns:
            z: (B, D) sampled latent codes.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and sample.

        Args:
            x: (B, input_dim) input features.

        Returns:
            z, mu, log_var: Sampled codes and posterior parameters.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes.

        Args:
            z: (B, latent_dim) latent codes.

        Returns:
            x_recon: (B, input_dim) reconstruction.
        """
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        anchor_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full forward pass with loss computation.

        Args:
            x: (B, input_dim) input features.
            anchor_targets: Optional dict mapping anchor name -> (B,) targets.

        Returns:
            total_loss: Scalar loss for backpropagation.
            loss_dict: Dict of individual loss components for logging.
        """
        z, mu, log_var = self.encode(x)
        x_recon = self.decode(z)

        losses: Dict[str, torch.Tensor] = {}

        # Reconstruction loss (MSE for continuous data)
        losses["recon"] = F.mse_loss(x_recon, x, reduction="mean")

        # TC decomposition
        mi, tc, dim_kl = estimate_tc_minibatch(z, mu, log_var)
        losses["mi"] = self.config.alpha * mi
        losses["tc"] = self.config.beta * tc
        losses["dim_kl"] = self.config.gamma * dim_kl

        # Anchor losses
        if anchor_targets is not None:
            for spec in self.config.anchor_specs:
                if spec.name in anchor_targets and spec.name in self.anchor_heads:
                    z_dim = z[:, spec.dim_index]
                    target = anchor_targets[spec.name]
                    losses[f"anchor_{spec.name}"] = (
                        spec.weight * self.anchor_heads[spec.name].loss(z_dim, target)
                    )

        total = sum(losses.values())
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict["total"] = total.item()

        self._step += 1
        return total, loss_dict

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get deterministic latent codes (posterior mean).

        Args:
            x: (B, input_dim) input features.

        Returns:
            mu: (B, latent_dim) posterior means.
        """
        mu, _ = self.encoder(x)
        return mu


# ===================================================================
# DCI Metrics
# ===================================================================

class DCIMetrics:
    """Compute Disentanglement, Completeness, Informativeness (DCI)
    metrics for evaluating latent space quality.

    DCI (Eastwood & Williams, 2018) measures:
        - **Disentanglement (D)**: Each latent dim encodes at most one
          generative factor. High D = each z_i predicts only one target.
        - **Completeness (C)**: Each generative factor is captured by at
          most one latent dim. High C = each target predicted by one z_i.
        - **Informativeness (I)**: How well latent dims predict targets
          overall (R^2 or accuracy).

    Uses gradient-based feature importance (faster than the original
    Lasso approach and differentiable).
    """

    @staticmethod
    def compute(
        z: torch.Tensor,
        targets: torch.Tensor,
        method: str = "gradient",
    ) -> Dict[str, float]:
        """Compute DCI metrics.

        Args:
            z: (N, D) latent codes for N samples, D dimensions.
            targets: (N, K) K generative factor values.
            method: "gradient" for gradient-based importance, or
                "correlation" for Pearson correlation.

        Returns:
            Dict with keys "disentanglement", "completeness",
            "informativeness".
        """
        N, D = z.shape
        K = targets.shape[1]

        # Compute importance matrix R[d, k] = importance of z_d for target_k
        if method == "correlation":
            R = DCIMetrics._correlation_importance(z, targets)
        else:
            R = DCIMetrics._gradient_importance(z, targets)

        # Disentanglement: for each latent dim d, how concentrated is
        # its importance across targets?
        # D_d = 1 - H(p_d) / log(K) where p_dk = R[d,k] / sum_k R[d,k]
        R_norm_row = R / (R.sum(dim=1, keepdim=True) + 1e-10)
        entropy_row = -(R_norm_row * (R_norm_row + 1e-10).log()).sum(dim=1)
        max_entropy = math.log(max(K, 2))
        D_per_dim = 1.0 - entropy_row / max_entropy
        # Weight by total importance of each dim
        rho = R.sum(dim=1) / (R.sum() + 1e-10)
        disentanglement = (rho * D_per_dim).sum().item()

        # Completeness: for each target k, how concentrated is it
        # across latent dims?
        R_norm_col = R / (R.sum(dim=0, keepdim=True) + 1e-10)
        entropy_col = -(R_norm_col * (R_norm_col + 1e-10).log()).sum(dim=0)
        max_entropy_d = math.log(max(D, 2))
        C_per_target = 1.0 - entropy_col / max_entropy_d
        completeness = C_per_target.mean().item()

        # Informativeness: average R^2
        informativeness = R.sum().item() / max(K, 1)

        return {
            "disentanglement": disentanglement,
            "completeness": completeness,
            "informativeness": informativeness,
        }

    @staticmethod
    def _correlation_importance(
        z: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute importance via absolute Pearson correlation.

        Args:
            z: (N, D) latent codes.
            targets: (N, K) target values.

        Returns:
            R: (D, K) importance matrix.
        """
        D = z.shape[1]
        K = targets.shape[1]
        R = torch.zeros(D, K)

        z_centered = z - z.mean(dim=0, keepdim=True)
        t_centered = targets - targets.mean(dim=0, keepdim=True)
        z_std = z_centered.std(dim=0, keepdim=True) + 1e-10
        t_std = t_centered.std(dim=0, keepdim=True) + 1e-10

        corr = (z_centered.T @ t_centered) / (z.shape[0] * z_std.T * t_std)
        R = corr.abs()
        return R

    @staticmethod
    def _gradient_importance(
        z: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute importance via gradient of linear regression.

        Fits z -> target_k for each k and uses |weight_d| as importance.

        Args:
            z: (N, D) latent codes.
            targets: (N, K) target values.

        Returns:
            R: (D, K) importance matrix.
        """
        D = z.shape[1]
        K = targets.shape[1]

        # Least squares: W = (Z^T Z)^{-1} Z^T Y
        ZtZ = z.T @ z + 1e-4 * torch.eye(D, device=z.device)
        ZtY = z.T @ targets
        W = torch.linalg.solve(ZtZ, ZtY)  # (D, K)
        R = W.abs()
        return R


# ===================================================================
# Traversal Generator
# ===================================================================

class TraversalGenerator:
    """Generate latent traversals for visualizing what each dimension encodes.

    For each latent dimension d, generates a sequence of latent codes
    where d is varied from -range to +range while all other dimensions
    are held fixed at a reference (e.g., population mean).

    The decoded traversals show the biological change encoded by dimension d.
    """

    @staticmethod
    @torch.no_grad()
    def generate(
        vae: DisentangledVAE,
        reference_z: torch.Tensor,
        n_steps: int = 11,
        traverse_range: float = 3.0,
        dims: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Generate decoded traversals for specified dimensions.

        Args:
            vae: Trained DisentangledVAE.
            reference_z: (D,) reference latent code (e.g., population mean).
            n_steps: Number of steps in traversal.
            traverse_range: Range of traversal in std deviations.
            dims: List of dimension indices to traverse. Default: all.

        Returns:
            Dict mapping dimension index -> (n_steps, output_dim)
            decoded traversals.
        """
        D = reference_z.shape[0]
        if dims is None:
            dims = list(range(D))

        values = torch.linspace(-traverse_range, traverse_range, n_steps)
        traversals: Dict[int, torch.Tensor] = {}

        for d in dims:
            # Create n_steps copies of the reference
            z_batch = reference_z.unsqueeze(0).expand(n_steps, -1).clone()
            z_batch[:, d] = values
            x_decoded = vae.decode(z_batch)
            traversals[d] = x_decoded.cpu()

        return traversals

    @staticmethod
    @torch.no_grad()
    def traversal_sensitivity(
        vae: DisentangledVAE,
        reference_z: torch.Tensor,
        n_steps: int = 21,
        traverse_range: float = 3.0,
    ) -> torch.Tensor:
        """Compute sensitivity of reconstruction to each latent dimension.

        Sensitivity_d = Var(decoder(z_traversal_d)) averaged over output dims.
        Higher sensitivity = that dimension encodes more variation.

        Args:
            vae: Trained DisentangledVAE.
            reference_z: (D,) reference latent code.
            n_steps: Number of traversal steps.
            traverse_range: Traversal range.

        Returns:
            sensitivity: (D,) sensitivity score per dimension.
        """
        D = reference_z.shape[0]
        traversals = TraversalGenerator.generate(
            vae, reference_z, n_steps, traverse_range, list(range(D))
        )
        sensitivity = torch.zeros(D)
        for d, decoded in traversals.items():
            # Variance of decoded output across traversal steps
            sensitivity[d] = decoded.var(dim=0).mean().item()
        return sensitivity


# ===================================================================
# Unit Tests
# ===================================================================

def _test_encoder_decoder() -> None:
    """Test Encoder and Decoder shapes."""
    B, D_in, D_lat = 8, 200, 16
    enc = Encoder(D_in, D_lat, [128, 64], batch_norm=True)
    dec = Decoder(D_lat, D_in, [128, 64], batch_norm=True)

    x = torch.randn(B, D_in)
    mu, logvar = enc(x)
    assert mu.shape == (B, D_lat), f"mu shape: {mu.shape}"
    assert logvar.shape == (B, D_lat), f"logvar shape: {logvar.shape}"
    assert (logvar >= -10).all() and (logvar <= 10).all(), "logvar out of bounds"

    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    x_recon = dec(z)
    assert x_recon.shape == (B, D_in), f"x_recon shape: {x_recon.shape}"
    print("[PASS] Encoder/Decoder")


def _test_tc_estimation() -> None:
    """Test TC decomposition produces reasonable values."""
    B, D = 32, 8
    mu = torch.randn(B, D) * 0.5
    log_var = torch.zeros(B, D) - 1.0
    z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    mi, tc, dim_kl = estimate_tc_minibatch(z, mu, log_var)
    assert torch.isfinite(mi), f"MI not finite: {mi}"
    assert torch.isfinite(tc), f"TC not finite: {tc}"
    assert torch.isfinite(dim_kl), f"dim_kl not finite: {dim_kl}"

    # TC should be roughly non-negative for independent dims
    # (can be slightly negative due to estimation noise)
    print(f"  MI={mi.item():.4f}, TC={tc.item():.4f}, dim_KL={dim_kl.item():.4f}")
    print("[PASS] TC estimation")


def _test_anchor_head() -> None:
    """Test AnchorHead for classification and regression."""
    B = 16
    spec_cls = AnchorSpec(0, "cell_cycle", "classification", n_classes=4)
    head_cls = AnchorHead(spec_cls)
    z_dim = torch.randn(B)
    pred = head_cls(z_dim)
    assert pred.shape == (B, 4), f"cls pred shape: {pred.shape}"
    target_cls = torch.randint(0, 4, (B,))
    loss_cls = head_cls.loss(z_dim, target_cls)
    assert torch.isfinite(loss_cls), "cls loss not finite"

    spec_reg = AnchorSpec(1, "stemness", "regression")
    head_reg = AnchorHead(spec_reg)
    pred_reg = head_reg(z_dim)
    assert pred_reg.shape == (B,), f"reg pred shape: {pred_reg.shape}"
    target_reg = torch.randn(B)
    loss_reg = head_reg.loss(z_dim, target_reg)
    assert torch.isfinite(loss_reg), "reg loss not finite"
    print("[PASS] AnchorHead")


def _test_disentangled_vae() -> None:
    """Test full DisentangledVAE forward pass and loss."""
    cfg = DisentangledVAEConfig(
        input_dim=200, latent_dim=16, hidden_dims=[128, 64],
        beta=4.0, n_anchors=2,
        anchor_specs=[
            AnchorSpec(0, "efflux", "regression"),
            AnchorSpec(1, "cycle", "classification", n_classes=3),
        ],
    )
    vae = DisentangledVAE(cfg)
    B = 16
    x = torch.randn(B, 200)

    # Without anchors
    loss, losses = vae(x)
    assert torch.isfinite(loss), "Loss not finite"
    assert "recon" in losses
    assert "tc" in losses

    # With anchors
    anchors = {"efflux": torch.randn(B), "cycle": torch.randint(0, 3, (B,))}
    loss2, losses2 = vae(x, anchors)
    assert "anchor_efflux" in losses2
    assert "anchor_cycle" in losses2

    # Deterministic encoding
    z = vae.get_latent(x)
    assert z.shape == (B, 16)

    print("[PASS] DisentangledVAE")


def _test_dci_metrics() -> None:
    """Test DCI metric computation."""
    N, D, K = 100, 8, 3
    z = torch.randn(N, D)
    # Create targets that are correlated with specific dims
    targets = torch.zeros(N, K)
    targets[:, 0] = z[:, 0] + 0.1 * torch.randn(N)  # dim 0 -> target 0
    targets[:, 1] = z[:, 3] + 0.1 * torch.randn(N)  # dim 3 -> target 1
    targets[:, 2] = z[:, 5] + 0.1 * torch.randn(N)  # dim 5 -> target 2

    metrics = DCIMetrics.compute(z, targets, method="correlation")
    assert "disentanglement" in metrics
    assert "completeness" in metrics
    assert "informativeness" in metrics
    assert 0 <= metrics["disentanglement"] <= 1.0 + 1e-6
    assert 0 <= metrics["completeness"] <= 1.0 + 1e-6
    print(f"  DCI: D={metrics['disentanglement']:.3f}, "
          f"C={metrics['completeness']:.3f}, I={metrics['informativeness']:.3f}")
    print("[PASS] DCIMetrics")


def _test_traversal_generator() -> None:
    """Test TraversalGenerator."""
    cfg = DisentangledVAEConfig(
        input_dim=100, latent_dim=8, hidden_dims=[64, 32],
        anchor_specs=[],
    )
    vae = DisentangledVAE(cfg)
    vae.eval()

    ref_z = torch.zeros(8)
    traversals = TraversalGenerator.generate(vae, ref_z, n_steps=5, dims=[0, 1, 2])
    assert len(traversals) == 3
    for d, trav in traversals.items():
        assert trav.shape == (5, 100), f"dim {d}: {trav.shape}"

    sens = TraversalGenerator.traversal_sensitivity(vae, ref_z, n_steps=5)
    assert sens.shape == (8,)
    assert (sens >= 0).all()
    print("[PASS] TraversalGenerator")


def run_all_tests() -> None:
    """Run all unit tests."""
    _test_encoder_decoder()
    _test_tc_estimation()
    _test_anchor_head()
    _test_disentangled_vae()
    _test_dci_metrics()
    _test_traversal_generator()
    print("\n=== All disentangled_vae tests passed ===")


if __name__ == "__main__":
    run_all_tests()
