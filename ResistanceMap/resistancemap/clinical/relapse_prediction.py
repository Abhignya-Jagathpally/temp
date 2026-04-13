"""
Gap 5.3: Relapse-Specific Resistance Prediction

Neural ODE-based competing risks model with cause-specific hazard decoders for
multiple myeloma relapse phenotypes. Implements multi-state disease progression
with landmark-based recalibration and DeepHit-style loss optimization.

Multi-state model:
  Post-ASCT → Biochemical relapse ↔ Clinical relapse ↔ EMD/CNS ↔ Death

Author: ResistanceMap Team
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

logger = logging.getLogger(__name__)


class CauseSpecificHazardDecoder(nn.Module):
    """
    Pathway-focused MLP decoder for cause-specific hazard rates.

    Each decoder specializes in a distinct relapse pathway:
    - Early relapse: TP53 pathway mutations, chromosome 17 deletions
    - Late relapse: DNA repair deficiency, DNA-PKcs/ATM pathway
    - EMD: Migration and epithelial-mesenchymal transition genes
    - CNS: Blood-brain barrier penetration markers
    - Non-relapse mortality: Comorbidity, age, organ function

    Args:
        latent_dim: Dimension of ODE hidden state
        output_dim: Always 1 (single hazard rate)
        hidden_dims: List of hidden layer dimensions
        pathway_genes: Optional list of gene names for this pathway
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int = 1,
        hidden_dims: Optional[list[int]] = None,
        pathway_genes: Optional[list[str]] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [latent_dim * 2, latent_dim * 2]

        self.latent_dim = latent_dim
        self.pathway_genes = pathway_genes or []

        layers = []
        in_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Final layer: map to log-hazard (unconstrained), then exp for positivity
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute cause-specific hazard rate.

        Args:
            latent: Shape (batch_size, latent_dim)

        Returns:
            Hazard rate (batch_size, 1), constrained to positive via exp
        """
        log_hazard = self.mlp(latent)
        return torch.exp(log_hazard)


class CompetingRisksODE(nn.Module):
    """
    Neural ODE dynamics for competing risks.

    Shared latent ODE trajectory φ(t) → 5 cause-specific hazard rates via
    specialized decoders. Jointly models progression toward multiple absorbing
    states (relapse types, death).

    Args:
        input_dim: Dimension of input biomarkers/features
        latent_dim: Dimension of ODE hidden state
        num_causes: Number of competing risks (default 5)
        hidden_dims: Hidden dimensions for ODE dynamics
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_causes: int = 5,
        hidden_dims: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [latent_dim * 2, latent_dim]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_causes = num_causes

        # Encoder: biomarkers → latent state
        encoder_layers = [nn.Linear(input_dim, latent_dim), nn.ELU()]
        encoder_layers.append(nn.Linear(latent_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ODE dynamics: shared latent trajectory
        ode_layers = []
        in_dim = latent_dim
        for hidden_dim in hidden_dims:
            ode_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
            ])
            in_dim = hidden_dim
        ode_layers.append(nn.Linear(in_dim, latent_dim))
        self.ode_dynamics = nn.Sequential(*ode_layers)

        # Cause-specific decoders
        pathway_names = [
            "early_relapse_TP53",
            "late_relapse_DNA_repair",
            "emd_migration",
            "cns_bbb_penetration",
            "non_relapse_mortality",
        ]
        self.hazard_decoders = nn.ModuleList([
            CauseSpecificHazardDecoder(
                latent_dim=latent_dim,
                pathway_genes=[pathway_names[k]],
            )
            for k in range(num_causes)
        ])

        logger.info(
            f"Initialized CompetingRisksODE: input_dim={input_dim}, "
            f"latent_dim={latent_dim}, num_causes={num_causes}"
        )

    def forward(
        self,
        biomarkers: torch.Tensor,
        times: torch.Tensor,
        method: str = "dopri",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode biomarkers, solve ODE, decode hazards.

        Args:
            biomarkers: Shape (batch_size, input_dim)
            times: Shape (num_timepoints,), observation times
            method: ODE solver method (default 'dopri' for RK45)

        Returns:
            hazards: (batch_size, num_causes, num_timepoints)
            latent_traj: (batch_size, num_timepoints, latent_dim)
            latent_z0: (batch_size, latent_dim)
        """
        # Encode initial state
        latent_z0 = self.encoder(biomarkers)

        # Solve ODE
        latent_traj = odeint(
            self.ode_dynamics,
            latent_z0,
            times,
            method=method,
        )  # (num_timepoints, batch_size, latent_dim)

        latent_traj = latent_traj.permute(1, 0, 2)  # (batch_size, num_timepoints, latent_dim)

        # Decode hazard rates for each cause
        hazards_list = []
        for decoder in self.hazard_decoders:
            # Reshape for batch processing: (batch_size * num_timepoints, latent_dim)
            batch_size, num_timepoints = latent_traj.shape[0], latent_traj.shape[1]
            latent_flat = latent_traj.reshape(batch_size * num_timepoints, self.latent_dim)
            hazard_flat = decoder(latent_flat)  # (batch_size * num_timepoints, 1)
            hazard = hazard_flat.reshape(batch_size, num_timepoints, 1).squeeze(-1)
            hazards_list.append(hazard)

        hazards = torch.stack(hazards_list, dim=1)  # (batch_size, num_causes, num_timepoints)

        return hazards, latent_traj, latent_z0


class CumulativeIncidenceFunction(nn.Module):
    """
    Cumulative incidence function with competing risks constraint.

    CIF_k(t) = ∫₀ᵗ S(u)·λ_k(u)du, where S(u) = exp(-∑_j ∫₀ᵘ λ_j(s)ds)

    Implements numerical integration via trapezoid rule and enforces
    softmax normalization to guarantee CIF_k ≤ 1 and Σ_k CIF_k ≤ 1.

    Args:
        num_causes: Number of competing risks
    """

    def __init__(self, num_causes: int = 5) -> None:
        super().__init__()
        self.num_causes = num_causes

    def forward(
        self,
        hazards: torch.Tensor,
        times: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute cumulative incidence functions.

        Args:
            hazards: Shape (batch_size, num_causes, num_timepoints)
            times: Shape (num_timepoints,)

        Returns:
            cif: Cumulative incidence, shape (batch_size, num_causes, num_timepoints)
            surv_prob: Survival probability, shape (batch_size, num_timepoints)
            cum_hazards: Cumulative hazards, shape (batch_size, num_causes, num_timepoints)
        """
        batch_size, num_causes, num_timepoints = hazards.shape

        # Total hazard across all causes
        total_hazard = hazards.sum(dim=1)  # (batch_size, num_timepoints)

        # Cumulative hazard via trapezoid rule
        dt = torch.diff(times, prepend=torch.tensor([0.0], device=times.device))
        dt = dt.to(hazards.device)

        cum_hazards = torch.cumsum(total_hazard * dt.unsqueeze(0), dim=1)
        cum_hazards = torch.cat(
            [torch.zeros(batch_size, 1, device=hazards.device), cum_hazards[:, :-1]], dim=1
        )

        # Survival probability: S(t) = exp(-Λ(t))
        surv_prob = torch.exp(-cum_hazards)  # (batch_size, num_timepoints)

        # CIF for each cause: CIF_k(t) = ∫₀ᵗ S(u)·λ_k(u)du
        cif = torch.zeros(batch_size, num_causes, num_timepoints, device=hazards.device)

        for k in range(num_causes):
            integrand = surv_prob * hazards[:, k, :]  # (batch_size, num_timepoints)
            cif_k = torch.cumsum(integrand * dt.unsqueeze(0), dim=1)
            cif[:, k, :] = cif_k

        # Softmax normalization: ensure sum of CIFs ≤ 1 at each timepoint
        cif_sum = cif.sum(dim=1, keepdim=True)  # (batch_size, 1, num_timepoints)
        cif = cif / (cif_sum + 1e-8)

        return cif, surv_prob, cum_hazards


class LandmarkRecalibrator(nn.Module):
    """
    Landmark-based recalibration with updated biomarkers.

    At landmark times (100d, 6mo, 12mo, 24mo), incorporates fresh biomarker
    measurements (M-protein, FLC) to conditionally recalibrate predictions.

    Args:
        latent_dim: Dimension of latent state
        num_biomarkers: Number of updatable biomarkers
        landmark_times: List of landmark times in days
    """

    def __init__(
        self,
        latent_dim: int,
        num_biomarkers: int,
        landmark_times: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        if landmark_times is None:
            landmark_times = [100.0, 180.0, 365.0, 730.0]  # 100d, 6mo, 12mo, 24mo

        self.latent_dim = latent_dim
        self.landmark_times = torch.tensor(landmark_times)
        self.num_biomarkers = num_biomarkers

        # Recalibration network: takes latent state + new biomarkers → updated latent state
        self.recalibrator = nn.Sequential(
            nn.Linear(latent_dim + num_biomarkers, latent_dim * 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        logger.info(
            f"Initialized LandmarkRecalibrator: landmark_times={landmark_times}, "
            f"num_biomarkers={num_biomarkers}"
        )

    def forward(
        self,
        latent_state: torch.Tensor,
        new_biomarkers: torch.Tensor,
        current_time: float,
    ) -> torch.Tensor:
        """
        Recalibrate latent state if at a landmark time.

        Args:
            latent_state: Shape (batch_size, latent_dim)
            new_biomarkers: Shape (batch_size, num_biomarkers)
            current_time: Current time point

        Returns:
            Updated latent state, shape (batch_size, latent_dim)
        """
        # Check if current time matches any landmark
        landmark_times = self.landmark_times.to(latent_state.device)
        at_landmark = torch.any(torch.abs(landmark_times - current_time) < 1e-3)

        if at_landmark:
            combined = torch.cat([latent_state, new_biomarkers], dim=1)
            latent_state = self.recalibrator(combined)
            logger.debug(f"Applied landmark recalibration at t={current_time:.1f}")

        return latent_state


class CompetingRisksLoss(nn.Module):
    """
    DeepHit-style loss for competing risks.

    Combines:
    1. Negative log-likelihood of observed event
    2. Ranking loss for concordance
    3. CIF constraint regularization

    Args:
        alpha: Weight for ranking loss (default 0.5)
        beta: Weight for CIF constraint (default 1.0)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        cif: torch.Tensor,
        observed_times: torch.Tensor,
        observed_causes: torch.Tensor,
        uncensored: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute competing risks loss.

        Args:
            cif: Cumulative incidence, shape (batch_size, num_causes, num_timepoints)
            observed_times: Event times, shape (batch_size,)
            observed_causes: Event types (0-4), shape (batch_size,)
            uncensored: Indicator (1=event, 0=censored), shape (batch_size,)

        Returns:
            loss: Scalar loss tensor
            metrics: Dict with NLL, ranking, and CIF components
        """
        batch_size, num_causes, num_timepoints = cif.shape

        # --- Log-likelihood loss ---
        # Extract predicted CIF at observed time for observed cause
        time_indices = torch.clamp(
            (observed_times).long(), min=0, max=num_timepoints - 1
        )
        cause_indices = observed_causes.long()

        # Gather CIF values: (batch_size,)
        cif_values = cif[torch.arange(batch_size), cause_indices, time_indices]

        # Clamp to avoid log(0)
        cif_values = torch.clamp(cif_values, min=1e-7, max=1.0 - 1e-7)

        # Negative log-likelihood: -log(CIF) for events, -log(1-CIF_sum) for censored
        cif_sum = cif.sum(dim=1)  # (batch_size, num_timepoints)
        cif_sum_at_obs = cif_sum[torch.arange(batch_size), time_indices]
        cif_sum_at_obs = torch.clamp(cif_sum_at_obs, min=1e-7, max=1.0 - 1e-7)

        nll_event = -torch.log(cif_values + 1e-8)
        nll_cens = -torch.log(1.0 - cif_sum_at_obs + 1e-8)

        nll = uncensored * nll_event + (1.0 - uncensored) * nll_cens
        nll_loss = nll.mean()

        # --- Ranking loss (concordance) ---
        # For each uncensored pair (i, j) with t_i < t_j:
        # P(hazard_i at t_i > hazard_j at t_i) should be high
        ranking_loss = self._ranking_loss(cif, observed_times, observed_causes, uncensored)

        # --- CIF constraint: ensure Σ CIF ≤ 1 ---
        cif_sum = cif.sum(dim=1)
        excess = F.relu(cif_sum - 1.0)
        cif_penalty = excess.mean()

        # Total loss
        loss = nll_loss + self.alpha * ranking_loss + self.beta * cif_penalty

        metrics = {
            "nll": nll_loss.item(),
            "ranking_loss": ranking_loss.item(),
            "cif_penalty": cif_penalty.item(),
            "total_loss": loss.item(),
        }

        return loss, metrics

    def _ranking_loss(
        self,
        cif: torch.Tensor,
        observed_times: torch.Tensor,
        observed_causes: torch.Tensor,
        uncensored: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ranking loss for concordance."""
        batch_size = cif.shape[0]
        num_timepoints = cif.shape[2]

        ranking_loss = torch.tensor(0.0, device=cif.device)
        count = 0

        for i in range(batch_size):
            if uncensored[i] == 0:
                continue
            for j in range(batch_size):
                if i == j or uncensored[j] == 0:
                    continue

                if observed_times[i] < observed_times[j]:
                    # Individual i had event before j
                    time_i = min(int(observed_times[i]), num_timepoints - 1)
                    cause_i = int(observed_causes[i])
                    cif_i = cif[i, cause_i, time_i]

                    # Hazard at time of i's event should be high
                    ranking_loss = ranking_loss + F.relu(0.5 - cif_i)
                    count += 1

        if count > 0:
            ranking_loss = ranking_loss / count
        return ranking_loss


class RelapseSpecificPredictor(nn.Module):
    """
    Top-level module for relapse-specific resistance prediction.

    Orchestrates:
    1. Competing risks ODE integration
    2. Cumulative incidence computation
    3. Landmark recalibration
    4. Loss computation

    Args:
        input_dim: Dimension of input biomarkers
        latent_dim: Dimension of ODE latent state (default 64)
        num_causes: Number of competing risks (default 5)
        num_landmark_biomarkers: Number of updatable biomarkers (default 2)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        num_causes: int = 5,
        num_landmark_biomarkers: int = 2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_causes = num_causes

        # Core modules
        self.ode_model = CompetingRisksODE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_causes=num_causes,
        )
        self.cif_computer = CumulativeIncidenceFunction(num_causes=num_causes)
        self.landmark_recalibrator = LandmarkRecalibrator(
            latent_dim=latent_dim,
            num_biomarkers=num_landmark_biomarkers,
        )
        self.loss_fn = CompetingRisksLoss(alpha=0.5, beta=1.0)

        # Observation times (days post-ASCT) - common clinical milestones
        self.register_buffer(
            "default_times",
            torch.tensor([0.0, 50.0, 100.0, 180.0, 365.0, 730.0, 1095.0]),
        )

        logger.info(
            f"Initialized RelapseSpecificPredictor: input_dim={input_dim}, "
            f"latent_dim={latent_dim}, num_causes={num_causes}"
        )

    def forward(
        self,
        biomarkers: torch.Tensor,
        times: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Predict cumulative incidence functions.

        Args:
            biomarkers: Input features, shape (batch_size, input_dim)
            times: Observation times (days), shape (num_timepoints,).
                   If None, uses default milestone times.

        Returns:
            cif: Cumulative incidence, shape (batch_size, num_causes, num_timepoints)
            surv_prob: Survival probability, shape (batch_size, num_timepoints)
            state_dict: Dict with latent states and hazards
        """
        if times is None:
            times = self.default_times

        # ODE integration
        hazards, latent_traj, latent_z0 = self.ode_model(biomarkers, times)

        # CIF computation
        cif, surv_prob, cum_hazards = self.cif_computer(hazards, times)

        # Package outputs
        state_dict = {
            "hazards": hazards,
            "latent_trajectory": latent_traj,
            "cumulative_hazards": cum_hazards,
            "times": times,
        }

        return cif, surv_prob, state_dict

    def compute_loss(
        self,
        biomarkers: torch.Tensor,
        observed_times: torch.Tensor,
        observed_causes: torch.Tensor,
        uncensored: torch.Tensor,
        times: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute training loss.

        Args:
            biomarkers: Shape (batch_size, input_dim)
            observed_times: Event times, shape (batch_size,)
            observed_causes: Event types (0-4), shape (batch_size,)
            uncensored: Event indicator (1=event, 0=censored), shape (batch_size,)
            times: Optional observation times

        Returns:
            loss: Scalar loss
            metrics: Dict with loss components
        """
        if times is None:
            times = self.default_times

        cif, _, _ = self.forward(biomarkers, times)
        loss, metrics = self.loss_fn(cif, observed_times, observed_causes, uncensored)

        return loss, metrics

    def predict_at_landmark(
        self,
        biomarkers: torch.Tensor,
        landmark_updates: Optional[torch.Tensor] = None,
        landmark_time: float = 100.0,
    ) -> torch.Tensor:
        """
        Recalibrated prediction incorporating landmark biomarkers.

        Args:
            biomarkers: Initial biomarkers, shape (batch_size, input_dim)
            landmark_updates: Updated biomarkers at landmark, shape (batch_size, num_landmark_biomarkers)
            landmark_time: Landmark time (days post-ASCT)

        Returns:
            Recalibrated CIF, shape (batch_size, num_causes, num_timepoints)
        """
        if landmark_updates is None:
            landmark_updates = torch.zeros(
                biomarkers.shape[0], 2, device=biomarkers.device
            )

        # Get initial latent state
        _, _, latent_z0 = self.ode_model(biomarkers, torch.tensor([0.0]))

        # Recalibrate with landmark data
        latent_recal = self.landmark_recalibrator(
            latent_z0.squeeze(1), landmark_updates, landmark_time
        )

        # Times from landmark onward
        times_from_landmark = self.default_times[self.default_times >= landmark_time]

        # Integrate ODE from recalibrated state
        latent_traj = odeint(
            self.ode_model.ode_dynamics,
            latent_recal,
            times_from_landmark,
        )
        latent_traj = latent_traj.permute(1, 0, 2)

        # Decode hazards
        hazards_list = []
        for decoder in self.ode_model.hazard_decoders:
            batch_size = latent_recal.shape[0]
            num_timepoints = latent_traj.shape[1]
            latent_flat = latent_traj.reshape(batch_size * num_timepoints, self.latent_dim)
            hazard_flat = decoder(latent_flat)
            hazard = hazard_flat.reshape(batch_size, num_timepoints, 1).squeeze(-1)
            hazards_list.append(hazard)

        hazards = torch.stack(hazards_list, dim=1)

        # Compute CIF from recalibrated trajectories
        cif, _, _ = self.cif_computer(hazards, times_from_landmark)

        return cif


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    batch_size, input_dim = 32, 128
    biomarkers = torch.randn(batch_size, input_dim)

    model = RelapseSpecificPredictor(input_dim=input_dim, latent_dim=64)

    # Prediction
    cif, surv_prob, state = model(biomarkers)
    print(f"CIF shape: {cif.shape}")
    print(f"Survival prob shape: {surv_prob.shape}")

    # Training
    observed_times = torch.randint(0, 730, (batch_size,)).float()
    observed_causes = torch.randint(0, 5, (batch_size,)).long()
    uncensored = torch.bernoulli(0.6 * torch.ones(batch_size))

    loss, metrics = model.compute_loss(
        biomarkers, observed_times, observed_causes, uncensored
    )
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Landmark recalibration
    landmark_updates = torch.randn(batch_size, 2)
    cif_recal = model.predict_at_landmark(biomarkers, landmark_updates, landmark_time=100.0)
    print(f"Recalibrated CIF shape: {cif_recal.shape}")
