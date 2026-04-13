"""Gap 4.3: MyeloMemory Bistability Integration — Jacobian eigenvalue analysis, dual-head GNN,
reversibility classification, and stability-aware ODE dynamics.

Extends the Sneppen-Ringrose chromatin bistability framework (trajectory.py) with:
    1. JacobianEigenvalueAnalyzer: Computes Jacobian of bistability ODE, extracts eigenvalues
       for basin depth quantification (|λ_max| ∝ basin depth).
    2. DualHeadGNN: Shared GAT backbone with two output heads:
       - Head 1: IC50 regression (drug efficacy prediction)
       - Head 2: Reversibility score S ∈ [0,1] from epigenetic modifiers + eigenvalue features
    3. ReversibilityClassifier: Maps (S, mutation_status) → 4-class output
       (reversible_fast, reversible_slow, partially_reversible, irreversible).
    4. BistabilityIntegratedLoss: Weighted loss combining IC50, reversibility, and consistency.
    5. StabilityAwareODEDynamics: dθ/dt = f(θ) - α(S)·(θ - θ_ref) for stability modulation.

Uses torch.autograd.functional.jacobian for differentiable Jacobian computation.
Targets production deployment on H100 with mixed precision training.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian as torch_jacobian

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GATConv, global_mean_pool, GlobalAttention
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning("torch_geometric not available; using fallback GAT")


class JacobianEigenvalueAnalyzer(nn.Module):
    """Computes Jacobian matrix of bistability ODE and extracts eigenvalues.

    The Jacobian J(x) describes local dynamics near fixed points. The maximum
    eigenvalue λ_max determines stability: |λ_max| quantifies basin depth and
    perturbation sensitivity. Used for reversibility scoring.

    Basin depth ∝ |λ_max|: larger |λ_max| means steeper basin walls,
    harder to perturb away from state.

    Attributes:
        ode_fn: Callable that computes dx/dt = f(x).
        state_dim: Dimension of ODE state (typically 2 for active/repressive).
    """

    def __init__(self, ode_fn: callable = None, state_dim: int = 2) -> None:
        """Initialize JacobianEigenvalueAnalyzer.

        Args:
            ode_fn: ODE function f(t, x) -> dx/dt. If None, must be set later.
            state_dim: Dimension of state vector (default 2).
        """
        super().__init__()
        self.ode_fn = ode_fn
        self.state_dim = state_dim
        self.epsilon = 1e-6  # Regularization for numerical stability

    def set_ode_function(self, ode_fn: callable) -> None:
        """Set ODE function after initialization.

        Args:
            ode_fn: Callable f(t, x) -> dx/dt.
        """
        self.ode_fn = ode_fn

    def _compute_jacobian_numerical(
        self, x: torch.Tensor, t: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute Jacobian via finite differences (fallback).

        Args:
            x: (B, D) state vector.
            t: (B,) time vector or scalar.

        Returns:
            (B, D, D) Jacobian matrices.
        """
        batch_size, state_dim = x.shape
        jacobians = torch.zeros(
            batch_size, state_dim, state_dim, device=x.device, dtype=x.dtype
        )

        f_x = self.ode_fn(t, x)  # (B, D)

        for j in range(state_dim):
            x_perturbed = x.clone()
            x_perturbed[:, j] += self.epsilon
            f_perturbed = self.ode_fn(t, x_perturbed)  # (B, D)
            jacobians[:, :, j] = (f_perturbed - f_x) / self.epsilon

        return jacobians

    def _compute_jacobian_autograd(
        self, x: torch.Tensor, t: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute Jacobian via autograd (more accurate, slower).

        Args:
            x: (B, D) state vector.
            t: (B,) time vector or scalar.

        Returns:
            (B, D, D) Jacobian matrices.
        """
        batch_size, state_dim = x.shape
        jacobians = torch.zeros(
            batch_size, state_dim, state_dim, device=x.device, dtype=x.dtype
        )

        for b in range(batch_size):
            x_b = x[b : b + 1].clone().requires_grad_(True)
            t_b = t[b : b + 1] if t.dim() > 0 else t

            def f_scalar(x_in):
                return self.ode_fn(t_b, x_in)

            try:
                jac_b = torch_jacobian(f_scalar, x_b)  # (1, D, 1, D) or (D, D)
                if jac_b.dim() == 4:
                    jac_b = jac_b.squeeze(0).squeeze(1)
                jacobians[b] = jac_b
            except RuntimeError:
                # Fall back to numerical if autograd fails
                jacobians[b] = self._compute_jacobian_numerical(x_b, t_b)

        return jacobians

    def compute_eigenvalues(
        self, x: torch.Tensor, t: torch.Tensor = None, method: str = "numerical"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues of Jacobian.

        Args:
            x: (B, D) state vector.
            t: (B,) time vector or scalar (optional).
            method: "numerical" or "autograd".

        Returns:
            Tuple of (eigenvalues, eigenvectors):
                - eigenvalues: (B, D) complex eigenvalues
                - eigenvectors: (B, D, D) complex eigenvector matrices
        """
        if self.ode_fn is None:
            raise ValueError("ODE function not set. Call set_ode_function() first.")

        # Compute Jacobian
        if method == "autograd":
            jacobians = self._compute_jacobian_autograd(x, t)
        else:
            jacobians = self._compute_jacobian_numerical(x, t)

        # Compute eigenvalues and eigenvectors
        batch_size = jacobians.shape[0]
        eigenvalues = torch.zeros(
            batch_size, self.state_dim, dtype=torch.complex64, device=x.device
        )
        eigenvectors = torch.zeros(
            batch_size, self.state_dim, self.state_dim, dtype=torch.complex64, device=x.device
        )

        for b in range(batch_size):
            evals, evecs = torch.linalg.eigh(jacobians[b])
            eigenvalues[b] = evals
            eigenvectors[b] = evecs

        return eigenvalues, eigenvectors

    def compute_basin_depth(
        self, x: torch.Tensor, t: torch.Tensor = None, method: str = "numerical"
    ) -> torch.Tensor:
        """Compute basin depth as |λ_max|.

        Basin depth quantifies how stable the current state is. Larger values
        indicate deeper basins, making the state harder to perturb.

        Args:
            x: (B, D) state vector.
            t: (B,) time vector or scalar (optional).
            method: "numerical" or "autograd".

        Returns:
            (B,) basin depth scores.
        """
        eigenvalues, _ = self.compute_eigenvalues(x, t, method)

        # |λ_max| = largest absolute eigenvalue
        abs_evals = torch.abs(eigenvalues)
        basin_depths = torch.max(abs_evals, dim=1).values

        return basin_depths

    def forward(
        self, x: torch.Tensor, t: torch.Tensor = None, return_jacobians: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: compute basin depth scores.

        Args:
            x: (B, D) state vector.
            t: (B,) time vector or scalar (optional).
            return_jacobians: If True, also return computed Jacobian matrices.

        Returns:
            Basin depth scores (B,), or (basin_depths, jacobians) tuple.
        """
        jacobians = self._compute_jacobian_numerical(x, t)
        basin_depths = self.compute_basin_depth(x, t, method="numerical")

        if return_jacobians:
            return basin_depths, jacobians

        return basin_depths


class DualHeadGNN(nn.Module):
    """Dual-head Graph Attention Network for IC50 and reversibility prediction.

    Architecture:
        - Shared GAT backbone (4 layers, 256 dim, 8 attention heads)
        - Head 1: IC50 regression (continuous drug efficacy)
        - Head 2: Reversibility scoring (S ∈ [0,1] from epigenetic modifiers + eigenvalues)

    The shared backbone learns protein interaction patterns. Head 1 predicts
    how much drug kills cells (IC50). Head 2 predicts how reversible the
    resistance is (S=1 fully reversible, S=0 irreversible).

    Attributes:
        hidden_dim: Hidden feature dimension (256).
        num_layers: Number of GAT layers (4).
        num_heads: Number of attention heads per layer (8).
        protein_dim: Input protein feature dimension.
    """

    def __init__(
        self,
        protein_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize DualHeadGNN.

        Args:
            protein_dim: Input protein feature dimension.
            hidden_dim: Hidden layer dimension (default 256).
            num_layers: Number of GAT layers (default 4).
            num_heads: Number of attention heads (default 8).
            dropout: Dropout probability (default 0.1).
        """
        super().__init__()
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        if not HAS_PYG:
            raise ImportError("torch_geometric required for DualHeadGNN")

        # Input projection
        self.input_proj = nn.Linear(protein_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Shared GAT backbone
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            gat_layer = GATConv(
                hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout
            )
            self.gat_layers.append(gat_layer)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Head 1: IC50 regression
        self.ic50_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus(),  # IC50 > 0
        )

        # Head 2: Reversibility score (S ∈ [0, 1])
        # Takes eigenvalue features + protein features
        self.reversibility_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, 128),  # +4 for eigenvalue stats
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # S ∈ [0, 1]
        )

        # Global attention pooling
        self.global_attention = GlobalAttention(nn.Linear(hidden_dim, 1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        eigenvalues: Optional[torch.Tensor] = None,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: predict IC50 and reversibility.

        Args:
            x: (num_nodes, protein_dim) node features.
            edge_index: (2, num_edges) edge indices.
            eigenvalues: (B, D) eigenvalues from JacobianEigenvalueAnalyzer (optional).
            batch_idx: (num_nodes,) batch assignment for global pooling (optional).

        Returns:
            Tuple of (ic50_pred, reversibility_pred):
                - ic50_pred: (B,) IC50 predictions
                - reversibility_pred: (B,) reversibility scores S ∈ [0,1]
        """
        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)

        # Shared GAT backbone with residual connections
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_prev = h
            h = gat_layer(h, edge_index)
            h = layer_norm(h + h_prev) if i > 0 else layer_norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Global pooling
        if batch_idx is None:
            batch_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        h_pool = self.global_attention(h, batch_idx)  # (B, hidden_dim)

        # Head 1: IC50 regression
        ic50_pred = self.ic50_head(h_pool)  # (B, 1)
        ic50_pred = ic50_pred.squeeze(-1)

        # Head 2: Reversibility score
        if eigenvalues is not None:
            # Compute eigenvalue statistics: max, min, trace, determinant approx
            evals_max = torch.max(torch.abs(eigenvalues), dim=1).values  # (B,)
            evals_min = torch.min(torch.abs(eigenvalues), dim=1).values  # (B,)
            evals_trace = torch.sum(eigenvalues, dim=1).real  # (B,)
            evals_norm = torch.norm(eigenvalues, dim=1)  # (B,)
            evals_features = torch.stack(
                [evals_max, evals_min, evals_trace, evals_norm], dim=1
            )  # (B, 4)
        else:
            evals_features = torch.zeros(h_pool.shape[0], 4, device=h_pool.device)

        reversibility_input = torch.cat([h_pool, evals_features], dim=1)  # (B, hidden_dim + 4)
        reversibility_pred = self.reversibility_head(reversibility_input)  # (B, 1)
        reversibility_pred = reversibility_pred.squeeze(-1)

        return ic50_pred, reversibility_pred


class ReversibilityClassifier(nn.Module):
    """Maps (reversibility_score, mutation_status) → 4-class classification.

    Classes:
        0: reversible_fast (S ≥ 0.75, no driver mutations)
        1: reversible_slow (0.5 ≤ S < 0.75)
        2: partially_reversible (0.25 ≤ S < 0.5)
        3: irreversible (S < 0.25 or has TP53 loss)

    Args:
        num_mutation_features: Number of mutation flags (default 5 for common drivers).
    """

    def __init__(self, num_mutation_features: int = 5) -> None:
        """Initialize ReversibilityClassifier.

        Args:
            num_mutation_features: Number of mutation features (default 5).
                Expected: [TP53_mut, KRAS_mut, CDKN2A_mut, MYC_amp, NFE2L2_mut]
        """
        super().__init__()
        self.num_mutation_features = num_mutation_features

        # Neural network to combine score and mutation status
        self.classifier = nn.Sequential(
            nn.Linear(1 + num_mutation_features, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 4),  # 4 classes
        )

    def forward(
        self, reversibility_score: torch.Tensor, mutations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: classify reversibility.

        Args:
            reversibility_score: (B,) reversibility scores from 0 to 1.
            mutations: (B, num_mutation_features) binary mutation status.

        Returns:
            Tuple of (logits, predicted_class):
                - logits: (B, 4) classification logits
                - predicted_class: (B,) class indices {0, 1, 2, 3}
        """
        # Combine score and mutations
        x = torch.cat([reversibility_score.unsqueeze(-1), mutations], dim=1)  # (B, 1 + M)
        logits = self.classifier(x)  # (B, 4)
        pred_class = torch.argmax(logits, dim=1)  # (B,)

        return logits, pred_class

    def get_class_name(self, class_idx: int) -> str:
        """Get human-readable class name.

        Args:
            class_idx: Integer from 0 to 3.

        Returns:
            Class name string.
        """
        names = ["reversible_fast", "reversible_slow", "partially_reversible", "irreversible"]
        return names[class_idx]


class BistabilityIntegratedLoss(nn.Module):
    """Integrated loss combining IC50, reversibility, and consistency terms.

    Loss = 0.6 * L_IC50 + 0.3 * L_reversibility + 0.1 * L_consistency

    L_IC50: MSE between predicted and measured IC50 values.
    L_reversibility: BCE for binary reversibility classification.
    L_consistency: Penalizes inconsistency between IC50 and reversibility
                   (drugs with high IC50 should have high reversibility S).

    Attributes:
        weight_ic50: Weight for IC50 loss (default 0.6).
        weight_reversibility: Weight for reversibility loss (default 0.3).
        weight_consistency: Weight for consistency loss (default 0.1).
    """

    def __init__(
        self,
        weight_ic50: float = 0.6,
        weight_reversibility: float = 0.3,
        weight_consistency: float = 0.1,
    ) -> None:
        """Initialize BistabilityIntegratedLoss.

        Args:
            weight_ic50: Weight for IC50 loss (default 0.6).
            weight_reversibility: Weight for reversibility loss (default 0.3).
            weight_consistency: Weight for consistency loss (default 0.1).
        """
        super().__init__()
        self.weight_ic50 = weight_ic50
        self.weight_reversibility = weight_reversibility
        self.weight_consistency = weight_consistency

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        ic50_pred: torch.Tensor,
        ic50_true: torch.Tensor,
        reversibility_pred: torch.Tensor,
        reversibility_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute integrated loss.

        Args:
            ic50_pred: (B,) predicted IC50 values.
            ic50_true: (B,) ground-truth IC50 values.
            reversibility_pred: (B,) predicted reversibility scores S ∈ [0,1].
            reversibility_true: (B,) ground-truth reversibility labels {0, 1}.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains:
                - "loss_ic50": IC50 MSE
                - "loss_reversibility": Reversibility BCE
                - "loss_consistency": Consistency penalty
                - "loss_total": Weighted sum
        """
        # Normalize IC50 for numerical stability
        ic50_pred_norm = (ic50_pred - ic50_pred.mean()) / (ic50_pred.std() + 1e-8)
        ic50_true_norm = (ic50_true - ic50_true.mean()) / (ic50_true.std() + 1e-8)

        # L_IC50: MSE loss
        loss_ic50 = self.mse_loss(ic50_pred_norm, ic50_true_norm)

        # L_reversibility: BCE loss (convert true labels to float)
        reversibility_true_float = reversibility_true.float()
        loss_reversibility = self.bce_loss(reversibility_pred, reversibility_true_float)

        # L_consistency: Penalize if high IC50 but low reversibility (or vice versa)
        # Normalize both to [0, 1]
        ic50_normalized = (ic50_pred - ic50_pred.min()) / (ic50_pred.max() - ic50_pred.min() + 1e-8)
        consistency_target = ic50_normalized
        loss_consistency = F.mse_loss(reversibility_pred, consistency_target)

        # Weighted sum
        total_loss = (
            self.weight_ic50 * loss_ic50
            + self.weight_reversibility * loss_reversibility
            + self.weight_consistency * loss_consistency
        )

        loss_dict = {
            "loss_ic50": loss_ic50.detach(),
            "loss_reversibility": loss_reversibility.detach(),
            "loss_consistency": loss_consistency.detach(),
            "loss_total": total_loss.detach(),
        }

        return total_loss, loss_dict


class StabilityAwareODEDynamics(nn.Module):
    """Stability-modulated ODE: dθ/dt = f(θ) - α(S)·(θ - θ_ref).

    Adds a stability-dependent damping term to the baseline ODE. When S is high
    (reversible state), the damping term is weak, allowing free dynamics.
    When S is low (irreversible state), the damping term is strong, pulling
    the state back toward a reference point.

    This models the idea that reversible states are more "plastic" whereas
    irreversible states are "locked in."

    Dynamics:
        dθ/dt = f(θ) - α(S) * (θ - θ_ref)

    where:
        f(θ): baseline ODE dynamics
        α(S): stability coefficient as function of reversibility score
        θ_ref: reference state (e.g., initial state)

    Attributes:
        base_ode: Callable for baseline ODE f(t, θ).
        latent_dim: Dimension of state vector.
        alpha_max: Maximum damping coefficient.
    """

    def __init__(
        self, base_ode: callable = None, latent_dim: int = 64, alpha_max: float = 0.5
    ) -> None:
        """Initialize StabilityAwareODEDynamics.

        Args:
            base_ode: Baseline ODE function f(t, θ).
            latent_dim: Dimension of state (default 64).
            alpha_max: Maximum damping coefficient (default 0.5).
        """
        super().__init__()
        self.base_ode = base_ode
        self.latent_dim = latent_dim
        self.alpha_max = alpha_max

        # Neural network to map reversibility score S → damping coefficient α(S)
        self.alpha_network = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Softplus(),  # α ≥ 0
        )

    def set_base_ode(self, base_ode: callable) -> None:
        """Set baseline ODE after initialization.

        Args:
            base_ode: Callable f(t, θ) -> dθ/dt.
        """
        self.base_ode = base_ode

    def compute_alpha(self, reversibility_score: torch.Tensor) -> torch.Tensor:
        """Compute damping coefficient α(S) from reversibility score.

        α(S) ranges from 0 (S=1, reversible) to α_max (S=0, irreversible).

        Args:
            reversibility_score: (B,) scores in [0, 1].

        Returns:
            (B,) damping coefficients.
        """
        # α(S) = α_max * (1 - S) = α_max for irreversible, 0 for reversible
        alpha = self.alpha_max * (1.0 - reversibility_score.unsqueeze(-1))
        alpha = self.alpha_network(alpha).squeeze(-1)
        return alpha

    def forward(
        self,
        t: torch.Tensor,
        theta: torch.Tensor,
        reversibility_score: torch.Tensor,
        theta_ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute dθ/dt with stability modulation.

        Args:
            t: (B,) or scalar, current time.
            theta: (B, D) current state.
            reversibility_score: (B,) reversibility scores S ∈ [0,1].
            theta_ref: (B, D) reference state (default: use theta as reference).

        Returns:
            (B, D) derivatives dθ/dt.
        """
        if self.base_ode is None:
            raise ValueError("Base ODE not set. Call set_base_ode() first.")

        if theta_ref is None:
            theta_ref = theta.clone().detach()

        # Baseline ODE dynamics
        dtheta_dt_base = self.base_ode(t, theta)  # (B, D)

        # Stability-dependent damping term
        alpha = self.compute_alpha(reversibility_score)  # (B,)
        damping_term = alpha.unsqueeze(-1) * (theta - theta_ref)  # (B, D)

        # Combined dynamics
        dtheta_dt = dtheta_dt_base - damping_term

        return dtheta_dt
