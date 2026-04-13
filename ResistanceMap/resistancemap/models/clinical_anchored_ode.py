"""Gap 4.1: Cross-Sectional to Temporal Gap — Clinical Anchored Neural ODE.

Bridges cross-sectional multi-omic snapshots to continuous temporal trajectories
via a clinical-anchored Neural ODE with treatment conditioning.

Key components:
  1. PseudotimeEncoder: Maps cross-sectional features → continuous pseudotime τ ∈ [0,1]
  2. ClinicalAnchoredODEFunc: Neural ODE dynamics dz/dt = f_θ(z, treatment_embedding, τ)
  3. TreatmentEncoder: One-hot/embedding → treatment conditioning vector
  4. VelocityConsistencyLoss: Aligns ODE velocity with RNA velocity estimates
  5. ClinicalAnchorLoss: Soft anchors at landmark clinical timepoints (M-protein, FLC)
  6. TemporalSmoothnessLoss: Penalizes non-smooth trajectories (d²z/dτ²)
  7. ClinicalAnchoredTrajectoryModule: Top-level encoder → ODE → decoder pipeline

Uses torchdiffeq.odeint_adjoint for memory-efficient differentiable ODE solving.
Supports multi-scale loss aggregation with learnable weighting.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    logger.warning("torchdiffeq not available; Neural ODE will not work")
    odeint = None


class PseudotimeEncoder(nn.Module):
    """Encodes cross-sectional multi-omic features into continuous pseudotime.

    Maps high-dimensional multi-omic profiles (proteomics, transcriptomics, etc.)
    into a scalar pseudotime τ ∈ [0,1] representing disease progression or
    epigenetic state advancement. Uses a small MLP followed by sigmoid saturation.

    Architecture:
        input_dim → hidden_dim → GELU → hidden_dim → GELU → 1 → Sigmoid

    Args:
        input_dim: Dimension of concatenated multi-omic features.
        hidden_dim: Width of hidden layers (default 128).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Clamp to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features to pseudotime.

        Args:
            x: (B, input_dim) concatenated multi-omic features.

        Returns:
            τ: (B,) pseudotime values in [0, 1].
        """
        tau = self.net(x).squeeze(-1)
        return tau


class TreatmentEncoder(nn.Module):
    """Encodes treatment regimen into a conditioning vector.

    Maps one-hot treatment indicators (or continuous treatment embeddings)
    to a learned treatment representation used to modulate ODE dynamics via FiLM.

    Architecture:
        treatment_dim → hidden_dim → GELU → conditioning_dim

    Args:
        treatment_dim: Dimension of input treatment representation (one-hot size or embedding).
        conditioning_dim: Output conditioning vector dimension (default 64).
    """

    def __init__(self, treatment_dim: int, conditioning_dim: int = 64) -> None:
        super().__init__()
        self.treatment_dim = treatment_dim
        self.conditioning_dim = conditioning_dim

        self.net = nn.Sequential(
            nn.Linear(treatment_dim, conditioning_dim),
            nn.GELU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )

    def forward(self, treatment: torch.Tensor) -> torch.Tensor:
        """Encode treatment to conditioning vector.

        Args:
            treatment: (B, treatment_dim) one-hot or embedded treatment.

        Returns:
            c_treatment: (B, conditioning_dim) treatment embedding.
        """
        return self.net(treatment)


class ClinicalAnchoredODEFunc(nn.Module):
    """Neural ODE dynamics function: dz/dt = f_θ(z, treatment_embedding, clinical_context).

    Models temporal evolution of latent state z ∈ R^latent_dim as a function
    of treatment (via FiLM conditioning) and pseudotime. Enables learned
    treatment-specific trajectory shapes.

    Architecture:
        concat([z, time_embedding]) → MLP blocks → FiLM(treatment) → output dz/dt

    Args:
        latent_dim: Dimension of latent state (default 64).
        treatment_conditioning_dim: Dimension of treatment embedding (default 64).
        hidden_dim: Width of MLP blocks (default 128).
        n_layers: Number of hidden MLP layers (default 3).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        treatment_conditioning_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.treatment_conditioning_dim = treatment_conditioning_dim
        self.hidden_dim = hidden_dim

        # Time embedding: projects scalar t to a vector for better expressiveness
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main ODE function blocks
        layers = []
        in_dim = latent_dim + hidden_dim  # Concatenate z with time embedding
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim

        self.ode_blocks = nn.Sequential(*layers)

        # FiLM conditioning layer: modulate with treatment
        self.fc_gamma = nn.Linear(treatment_conditioning_dim, hidden_dim)
        self.fc_beta = nn.Linear(treatment_conditioning_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        treatment_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dz/dt at state z and time t, conditioned on treatment.

        Args:
            z: (B, latent_dim) latent state.
            t: (B,) or (1,) time (pseudotime or calendar).
            treatment_embedding: (B, treatment_conditioning_dim) treatment vector.

        Returns:
            dzdt: (B, latent_dim) velocity field dz/dt.
        """
        batch_size = z.shape[0]

        # Ensure t is properly shaped
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size)

        # Embed time
        t_embed = self.time_mlp(t.unsqueeze(-1))  # (B, hidden_dim)

        # Concatenate z with time embedding
        z_t = torch.cat([z, t_embed], dim=-1)  # (B, latent_dim + hidden_dim)

        # ODE blocks
        h = self.ode_blocks(z_t)  # (B, hidden_dim)

        # FiLM conditioning with treatment
        gamma = self.fc_gamma(treatment_embedding)  # (B, hidden_dim)
        beta = self.fc_beta(treatment_embedding)  # (B, hidden_dim)
        h = gamma * h + beta  # (B, hidden_dim)

        # Output
        dzdt = self.out_proj(h)  # (B, latent_dim)

        return dzdt


class VelocityConsistencyLoss(nn.Module):
    """Aligns ODE velocity field with external RNA velocity estimates.

    Computes cosine similarity between dz/dt (from ODE) and v_external (from
    RNA velocity estimates or gradient-based velocity). Higher similarity
    means the learned trajectory is consistent with biological velocity.

    Loss = 1 - mean(cosine_similarity(dz/dt, v_external))

    Args:
        reduction: 'mean' or 'sum' (default 'mean').
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self, dzdt: torch.Tensor, external_velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute velocity consistency loss.

        Args:
            dzdt: (B, latent_dim) ODE velocity.
            external_velocity: (B, latent_dim) external velocity estimate.

        Returns:
            loss: Scalar cosine dissimilarity loss.
        """
        # Compute cosine similarity: cos(θ) = u·v / (||u|| ||v||)
        denom = (
            torch.norm(dzdt, dim=-1, keepdim=True) *
            torch.norm(external_velocity, dim=-1, keepdim=True) + 1e-8
        )
        cosine_sim = torch.sum(dzdt * external_velocity, dim=-1) / denom.squeeze(-1)

        # Loss = 1 - similarity (push similarity toward 1)
        loss = 1.0 - cosine_sim

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"reduction={self.reduction} not supported")


class ClinicalAnchorLoss(nn.Module):
    """Soft anchor loss: ||z(τ_clinical) - z_measured||² at landmark timepoints.

    Encourages ODE trajectories to pass near clinical landmark states, e.g.,
    specific M-protein or FLC levels measured at follow-up timepoints.
    Implements soft constraints (L2 penalty) rather than hard constraints.

    Args:
        reduction: 'mean' or 'sum' (default 'mean').
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        z_trajectory: torch.Tensor,
        pseudotime_trajectory: torch.Tensor,
        clinical_anchor_states: torch.Tensor,
        clinical_anchor_times: torch.Tensor,
    ) -> torch.Tensor:
        """Compute clinical anchor loss.

        Args:
            z_trajectory: (T, B, latent_dim) ODE trajectory.
            pseudotime_trajectory: (T,) pseudotime values along trajectory.
            clinical_anchor_states: (B, latent_dim) target states at landmark times.
            clinical_anchor_times: (B,) landmark pseudotime values.

        Returns:
            loss: Scalar MSE loss penalizing distance from anchors.
        """
        batch_size = z_trajectory.shape[1]
        device = z_trajectory.device

        anchor_loss = 0.0
        valid_anchors = 0

        for b in range(batch_size):
            tau_clinical = clinical_anchor_times[b].item()

            # Skip if anchor time is out of trajectory range
            if tau_clinical < pseudotime_trajectory[0] or tau_clinical > pseudotime_trajectory[-1]:
                continue

            # Linear interpolation: find z at tau_clinical
            idx_left = torch.searchsorted(pseudotime_trajectory, tau_clinical, right=False)
            idx_left = torch.clamp(idx_left, 0, len(pseudotime_trajectory) - 2)
            idx_right = idx_left + 1

            t_left = pseudotime_trajectory[idx_left]
            t_right = pseudotime_trajectory[idx_right]
            alpha = (tau_clinical - t_left) / (t_right - t_left + 1e-8)

            z_at_anchor = (
                (1 - alpha) * z_trajectory[idx_left, b] +
                alpha * z_trajectory[idx_right, b]
            )

            # L2 penalty
            anchor_loss += F.mse_loss(z_at_anchor, clinical_anchor_states[b])
            valid_anchors += 1

        if valid_anchors == 0:
            return torch.tensor(0.0, device=device, dtype=z_trajectory.dtype)

        if self.reduction == "mean":
            return anchor_loss / valid_anchors
        else:
            return anchor_loss


class TemporalSmoothnessLoss(nn.Module):
    """Penalizes non-smooth trajectories: ||d²z/dτ²||².

    Encourages biologically plausible smooth state transitions by penalizing
    second-order acceleration. Computed via finite differences on the trajectory.

    Args:
        reduction: 'mean' or 'sum' (default 'mean').
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, z_trajectory: torch.Tensor) -> torch.Tensor:
        """Compute temporal smoothness loss.

        Args:
            z_trajectory: (T, B, latent_dim) ODE trajectory.

        Returns:
            loss: Scalar L2 loss on second-order finite differences.
        """
        # Compute velocity: dz/dt ≈ z[t+1] - z[t]
        vel = z_trajectory[1:] - z_trajectory[:-1]  # (T-1, B, latent_dim)

        # Compute acceleration: d²z/dt² ≈ vel[t+1] - vel[t]
        accel = vel[1:] - vel[:-1]  # (T-2, B, latent_dim)

        # L2 norm of acceleration
        accel_norm = torch.norm(accel, dim=-1)  # (T-2, B)

        if self.reduction == "mean":
            return accel_norm.mean()
        elif self.reduction == "sum":
            return accel_norm.sum()
        else:
            raise ValueError(f"reduction={self.reduction} not supported")


class ClinicalAnchoredTrajectoryModule(nn.Module):
    """Top-level module: encoder → Neural ODE → decoder with multi-scale losses.

    Integrates all components into an end-to-end trainable pipeline:
      1. Encoder: multi-omic features → latent state z0
      2. Pseudotime encoder: features → pseudotime τ
      3. Treatment encoder: treatment → conditioning
      4. ODE solver: integrate dz/dt over τ ∈ [0, 1]
      5. Decoder: trajectory → reconstructed clinical/molecular predictions
      6. Loss aggregation: reconstruction + velocity + anchor + smoothness

    Args:
        input_dim: Dimension of input multi-omic features.
        latent_dim: Latent state dimension (default 64).
        output_dim: Output dimension (e.g., biomarker predictions).
        treatment_dim: One-hot treatment dimension.
        encoder_hidden_dims: List of encoder hidden layer sizes (default [256, 128]).
        decoder_hidden_dims: List of decoder hidden layer sizes (default [128, 256]).
        ode_hidden_dim: ODE network hidden dimension (default 128).
        ode_n_layers: Number of ODE hidden layers (default 3).
        integration_steps: Number of ODE integration steps (default 50).
        has_velocity: Whether external velocity estimates are provided (default False).
        has_clinical_anchors: Whether clinical anchors are provided (default False).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        output_dim: int | None = None,
        treatment_dim: int = 4,
        encoder_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        ode_hidden_dim: int = 128,
        ode_n_layers: int = 3,
        integration_steps: int = 50,
        has_velocity: bool = False,
        has_clinical_anchors: bool = False,
    ) -> None:
        super().__init__()

        if output_dim is None:
            output_dim = input_dim
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [256, 128]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [128, 256]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.treatment_dim = treatment_dim
        self.integration_steps = integration_steps
        self.has_velocity = has_velocity
        self.has_clinical_anchors = has_clinical_anchors

        # Encoder: multi-omic → latent state z0
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.GELU())
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Pseudotime encoder: features → τ ∈ [0, 1]
        self.pseudotime_encoder = PseudotimeEncoder(input_dim, hidden_dim=128)

        # Treatment encoder: one-hot → treatment embedding
        self.treatment_encoder = TreatmentEncoder(treatment_dim, conditioning_dim=latent_dim)

        # ODE function
        self.ode_func = ClinicalAnchoredODEFunc(
            latent_dim=latent_dim,
            treatment_conditioning_dim=latent_dim,
            hidden_dim=ode_hidden_dim,
            n_layers=ode_n_layers,
        )

        # Decoder: trajectory → predictions
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.GELU())
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Loss modules
        self.recon_loss_fn = nn.MSELoss()
        self.velocity_loss_fn = VelocityConsistencyLoss() if has_velocity else None
        self.anchor_loss_fn = ClinicalAnchorLoss() if has_clinical_anchors else None
        self.smoothness_loss_fn = TemporalSmoothnessLoss()

        # Loss weights (learnable or fixed)
        self.lambda_velocity = nn.Parameter(torch.tensor(0.1))
        self.lambda_anchor = nn.Parameter(torch.tensor(0.1))
        self.lambda_smooth = nn.Parameter(torch.tensor(0.05))

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        treatment: torch.Tensor,
        x_target: torch.Tensor | None = None,
        external_velocity: torch.Tensor | None = None,
        clinical_anchor_states: torch.Tensor | None = None,
        clinical_anchor_times: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: encode → ODE → decode → compute losses.

        Args:
            x: (B, input_dim) multi-omic features.
            treatment: (B, treatment_dim) one-hot treatment.
            x_target: (B, output_dim) target for reconstruction loss.
            external_velocity: (B, latent_dim) external velocity estimates.
            clinical_anchor_states: (B, latent_dim) anchor states.
            clinical_anchor_times: (B,) anchor pseudotime values.

        Returns:
            Dict with keys:
                - 'z_trajectory': (T, B, latent_dim) latent trajectory
                - 'predictions': (B, output_dim) decoded predictions
                - 'pseudotime': (B,) pseudotime values
                - 'loss_recon': Reconstruction MSE
                - 'loss_velocity': Velocity consistency (if enabled)
                - 'loss_anchor': Clinical anchor (if enabled)
                - 'loss_smooth': Temporal smoothness
                - 'loss_total': Weighted sum of all losses
        """
        batch_size = x.shape[0]
        device = x.device

        # Encode to initial latent state
        z0 = self.encoder(x)  # (B, latent_dim)

        # Encode pseudotime
        tau = self.pseudotime_encoder(x)  # (B,)

        # Encode treatment
        c_treatment = self.treatment_encoder(treatment)  # (B, latent_dim)

        # ODE integration: solve from τ=0 to τ=1
        t_eval = torch.linspace(0, 1, self.integration_steps, device=device)

        def ode_wrapper(t, z):
            """Wrapper for odeint: broadcasts treatment conditioning."""
            # Expand t for batch processing
            t_batch = torch.full((batch_size,), t.item(), device=device)
            dzdt = self.ode_func(z, t_batch, c_treatment)
            return dzdt

        # Solve ODE
        if odeint is None:
            raise RuntimeError("torchdiffeq is required for ODE integration")

        z_trajectory = odeint(ode_wrapper, z0, t_eval, method="dopri")  # (T, B, latent_dim)

        # Decode trajectory to predictions
        z_final = z_trajectory[-1]  # (B, latent_dim)
        predictions = self.decoder(z_final)  # (B, output_dim)

        # Compute losses
        loss_dict = {}

        # Reconstruction loss
        if x_target is None:
            x_target = x
        loss_recon = self.recon_loss_fn(predictions, x_target)
        loss_dict["loss_recon"] = loss_recon

        # Velocity consistency loss
        loss_velocity = torch.tensor(0.0, device=device, dtype=z0.dtype)
        if self.has_velocity and external_velocity is not None:
            # Compute dzdt at final state
            t_final = torch.ones(batch_size, device=device)
            dzdt_final = self.ode_func(z_final, t_final, c_treatment)
            loss_velocity = self.velocity_loss_fn(dzdt_final, external_velocity)
        loss_dict["loss_velocity"] = loss_velocity

        # Clinical anchor loss
        loss_anchor = torch.tensor(0.0, device=device, dtype=z0.dtype)
        if self.has_clinical_anchors and clinical_anchor_states is not None:
            loss_anchor = self.anchor_loss_fn(
                z_trajectory, t_eval, clinical_anchor_states, clinical_anchor_times
            )
        loss_dict["loss_anchor"] = loss_anchor

        # Temporal smoothness loss
        loss_smooth = self.smoothness_loss_fn(z_trajectory)
        loss_dict["loss_smooth"] = loss_smooth

        # Aggregate losses
        loss_total = (
            loss_recon +
            F.softplus(self.lambda_velocity) * loss_velocity +
            F.softplus(self.lambda_anchor) * loss_anchor +
            F.softplus(self.lambda_smooth) * loss_smooth
        )
        loss_dict["loss_total"] = loss_total

        return {
            "z_trajectory": z_trajectory,
            "predictions": predictions,
            "pseudotime": tau,
            **loss_dict,
        }

    def extract_trajectory(
        self, x: torch.Tensor, treatment: torch.Tensor
    ) -> torch.Tensor:
        """Extract latent trajectory without computing losses.

        Args:
            x: (B, input_dim) multi-omic features.
            treatment: (B, treatment_dim) one-hot treatment.

        Returns:
            z_trajectory: (T, B, latent_dim) latent trajectory.
        """
        with torch.no_grad():
            result = self.forward(x, treatment)
            return result["z_trajectory"]
