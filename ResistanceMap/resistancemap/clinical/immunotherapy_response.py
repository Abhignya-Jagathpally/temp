from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PatientImmuneProfile:
    """Patient immune characterization for immunotherapy response."""
    antigen_bcma: float
    antigen_gprc5d: float
    antigen_fcrl5: float
    exhaustion_score: float
    cd4_cd8_ratio: float
    tcr_clonality: float
    batch: Optional[torch.Tensor] = None


@dataclass
class AgentMechanism:
    """Immunotherapy agent mechanism descriptor."""
    primary_target: str
    mechanism_type: str
    mechanism_idx: int


class PatientImmuneProfiler(nn.Module):
    """
    Characterizes patient immune state for immunotherapy response prediction.

    Encodes:
    - Antigen expression levels (BCMA, GPRC5D, FCRL5)
    - T-cell exhaustion (from PD-1/TIM-3/LAG-3/TOX markers)
    - CD4/CD8 ratio
    - TCR clonality (Shannon entropy)
    """

    def __init__(self, hidden_dim: int = 64):
        """
        Args:
            hidden_dim: Embedding dimension for immune features.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Antigen expression encoder (3 antigens: BCMA, GPRC5D, FCRL5)
        self.antigen_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # T-cell fitness encoder (exhaustion + CD4/CD8 + TCR clonality)
        self.tcell_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, profile: PatientImmuneProfile) -> torch.Tensor:
        """
        Encode patient immune profile.

        Args:
            profile: PatientImmuneProfile with antigen/T-cell measurements.

        Returns:
            Immune characterization embedding of shape (hidden_dim,).
        """
        # Stack antigen levels
        antigen_features = torch.tensor(
            [profile.antigen_bcma, profile.antigen_gprc5d, profile.antigen_fcrl5],
            dtype=torch.float32
        ).unsqueeze(0)

        # Stack T-cell features
        tcell_features = torch.tensor(
            [profile.exhaustion_score, profile.cd4_cd8_ratio, profile.tcr_clonality],
            dtype=torch.float32
        ).unsqueeze(0)

        # Encode
        antigen_emb = self.antigen_encoder(antigen_features)
        tcell_emb = self.tcell_encoder(tcell_features)

        # Fuse
        fused = self.fusion(torch.cat([antigen_emb, tcell_emb], dim=-1))

        return fused.squeeze(0)


class AgentMechanismEncoder(nn.Module):
    """
    Encodes immunotherapy agent mechanism and propagates through target pathway.

    Processes:
    - Primary target antigen embedding
    - Mechanism type (bispecific/CAR-T/CELMoD/ADC)
    - 1-hop STRING pathway neighbor propagation
    """

    MECHANISM_TYPES = ["bispecific", "car_t", "celmod", "adc"]

    def __init__(self, hidden_dim: int = 64, n_pathway_neighbors: int = 3):
        """
        Args:
            hidden_dim: Embedding dimension.
            n_pathway_neighbors: Number of pathway neighbors to encode per target.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_neighbors = n_pathway_neighbors

        # Target antigen embedding
        self.target_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Mechanism type embedding
        self.mechanism_embedding = nn.Embedding(
            len(self.MECHANISM_TYPES), hidden_dim
        )

        # Pathway neighbor encoder (processes multiple neighbor expressions)
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(n_pathway_neighbors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Fusion of target + mechanism + pathway
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        mechanism: AgentMechanism,
        target_expression: float,
        pathway_neighbor_levels: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Encode agent mechanism including target and pathway context.

        Args:
            mechanism: AgentMechanism with target and mechanism type.
            target_expression: Primary target antigen expression level.
            pathway_neighbor_levels: Expression of pathway neighbors (1-hop STRING).

        Returns:
            Mechanism encoding of shape (hidden_dim,).
        """
        # Target embedding
        target_feat = torch.tensor([[target_expression]], dtype=torch.float32)
        target_emb = self.target_encoder(target_feat)

        # Mechanism embedding
        mech_emb = self.mechanism_embedding(
            torch.tensor([mechanism.mechanism_idx], dtype=torch.long)
        )

        # Pathway neighbor embedding
        if pathway_neighbor_levels is None:
            pathway_neighbor_levels = np.zeros(self.n_neighbors)
        neighbor_feat = torch.tensor(
            pathway_neighbor_levels[:self.n_neighbors], dtype=torch.float32
        ).unsqueeze(0)
        if neighbor_feat.shape[1] < self.n_neighbors:
            pad = torch.zeros(1, self.n_neighbors - neighbor_feat.shape[1])
            neighbor_feat = torch.cat([neighbor_feat, pad], dim=1)
        neighbor_emb = self.neighbor_encoder(neighbor_feat)

        # Fuse all components
        fused = self.fusion(torch.cat([target_emb, mech_emb, neighbor_emb], dim=-1))

        return fused.squeeze(0)


class MAMLFewShotAdapter(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) adapter for few-shot learning on novel agents.

    Enables rapid fine-tuning with n=5-20 responder samples through inner/outer loops.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        inner_lr: float = 0.01,
        n_inner_steps: int = 5,
    ):
        """
        Args:
            input_dim: Input embedding dimension.
            hidden_dim: Hidden layer dimension.
            inner_lr: Inner loop learning rate.
            n_inner_steps: Number of inner loop gradient steps.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Response head (binary: responder/non-responder)
        self.response_head = nn.Linear(hidden_dim, 1)

    def inner_loop(
        self,
        support_embeddings: torch.Tensor,
        support_responses: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform inner loop optimization on support set.

        Args:
            support_embeddings: Shape (n_support, input_dim).
            support_responses: Shape (n_support,), binary labels.

        Returns:
            Dictionary of adapted parameters.
        """
        adapted_params = {name: param.clone().detach() for name, param in self.named_parameters()}

        # Create optimizer for inner loop
        optimizer = torch.optim.SGD(self.parameters(), lr=self.inner_lr)

        for _ in range(self.n_inner_steps):
            logits = self.backbone(support_embeddings)
            preds = self.response_head(logits)
            loss = F.binary_cross_entropy_with_logits(
                preds.squeeze(-1), support_responses.float()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return adapted_params

    def outer_loop(
        self,
        query_embeddings: torch.Tensor,
        query_responses: torch.Tensor,
        adapted_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute outer loop loss on query set.

        Args:
            query_embeddings: Shape (n_query, input_dim).
            query_responses: Shape (n_query,), binary labels.
            adapted_params: Parameters adapted during inner loop.

        Returns:
            Outer loop loss (scalar).
        """
        logits = self.backbone(query_embeddings)
        preds = self.response_head(logits)
        loss = F.binary_cross_entropy_with_logits(
            preds.squeeze(-1), query_responses.float()
        )
        return loss

    def forward(
        self,
        support_embeddings: torch.Tensor,
        support_responses: torch.Tensor,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform few-shot adaptation and predict on query set.

        Args:
            support_embeddings: Support set embeddings (n_support, input_dim).
            support_responses: Support set responses (n_support,).
            query_embeddings: Query set embeddings (n_query, input_dim).

        Returns:
            Predictions on query set (n_query,).
        """
        # Inner loop
        self.inner_loop(support_embeddings, support_responses)

        # Query prediction
        logits = self.backbone(query_embeddings)
        preds = torch.sigmoid(self.response_head(logits)).squeeze(-1)

        return preds


class AntigenLossForecaster(nn.Module):
    """
    Predicts trajectory of target antigen expression loss over time.

    Models resistance via BCMA/GPRC5D deletion or downregulation through:
    - Temporal sequence modeling of antigen levels
    - Forecasting escape trajectory
    """

    def __init__(self, hidden_dim: int = 64, n_forecast_steps: int = 4):
        """
        Args:
            hidden_dim: Hidden dimension.
            n_forecast_steps: Number of time steps to forecast (e.g., 4 months).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_forecast_steps = n_forecast_steps

        # LSTM encoder for antigen history
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Decoder for trajectory forecasting
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_forecast_steps * 3),
        )

        # Loss risk head (probability of meaningful loss)
        self.loss_risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        antigen_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forecast antigen expression trajectory.

        Args:
            antigen_history: Shape (batch_size, seq_len, 3) with [BCMA, GPRC5D, FCRL5].

        Returns:
            Tuple of:
            - forecast: Shape (batch_size, n_forecast_steps, 3).
            - loss_risk: Shape (batch_size,), probability of significant loss.
        """
        # Encode history
        _, (hidden, _) = self.lstm(antigen_history)
        final_hidden = hidden[-1]

        # Decode trajectory
        trajectory_logits = self.decoder(final_hidden)
        forecast = trajectory_logits.reshape(-1, self.n_forecast_steps, 3)
        forecast = torch.sigmoid(forecast)  # Bound to [0, 1]

        # Predict loss risk
        loss_risk = torch.sigmoid(self.loss_risk_head(final_hidden)).squeeze(-1)

        return forecast, loss_risk


class SequentialImmunotherapyPlanner(nn.Module):
    """
    Optimizes sequential immunotherapy strategy across multiple agents.

    Plans transitions: BCMA agent -> GPRC5D agent -> CELMoD (alternative MoA).
    """

    AGENTS = ["bcma", "gprc5d", "celmod"]
    N_AGENTS = len(AGENTS)

    def __init__(self, hidden_dim: int = 64, max_sequence_length: int = 3):
        """
        Args:
            hidden_dim: Embedding dimension.
            max_sequence_length: Maximum number of agents in sequence.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_sequence_length

        # Transition encoder (conditions on patient state + current agent)
        self.transition_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Sequence planner (selects next agent)
        self.sequence_planner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.N_AGENTS),
        )

        # Efficacy predictor for sequences
        self.efficacy_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        patient_embedding: torch.Tensor,
        agent_embedding: torch.Tensor,
        n_steps: int = 3,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Plan sequential therapy strategy.

        Args:
            patient_embedding: Patient immune profile (hidden_dim,).
            agent_embedding: Current agent embedding (hidden_dim,).
            n_steps: Number of planning steps.

        Returns:
            Tuple of:
            - sequence: List of agent indices in planned order.
            - efficacy_scores: Per-step efficacy predictions.
        """
        sequence = []
        efficacy_scores = []

        current_agent_emb = agent_embedding.unsqueeze(0)

        for _ in range(min(n_steps, self.max_length)):
            # Encode transition
            combined = torch.cat([patient_embedding.unsqueeze(0), current_agent_emb], dim=-1)
            transition_emb = self.transition_encoder(combined)

            # Select next agent
            agent_logits = self.sequence_planner(transition_emb)
            next_agent_idx = torch.argmax(agent_logits, dim=-1).item()
            sequence.append(next_agent_idx)

            # Predict efficacy
            efficacy = torch.sigmoid(
                self.efficacy_predictor(torch.cat([patient_embedding.unsqueeze(0), transition_emb], dim=-1))
            )
            efficacy_scores.append(efficacy.item())

            # Update current agent for next iteration
            current_agent_emb = transition_emb

        return sequence, torch.tensor(efficacy_scores)


class ImmunotherapyResponsePredictor(nn.Module):
    """
    Top-level orchestrator for novel immunotherapy response prediction.

    Integrates:
    1. Patient characterization (antigen + T-cell fitness)
    2. Agent mechanism encoding (target + pathway)
    3. Few-shot transfer learning (MAML)
    4. Resistance prediction (antigen loss + exhaustion)
    5. Sequential strategy optimization
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_pathway_neighbors: int = 3,
        inner_lr: float = 0.01,
        n_forecast_steps: int = 4,
        max_sequence_length: int = 3,
    ):
        """
        Args:
            hidden_dim: Embedding dimension throughout.
            n_pathway_neighbors: Number of pathway neighbors to encode.
            inner_lr: MAML inner loop learning rate.
            n_forecast_steps: Antigen loss forecast horizon.
            max_sequence_length: Maximum sequential agents.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Component modules
        self.patient_profiler = PatientImmuneProfiler(hidden_dim)
        self.agent_encoder = AgentMechanismEncoder(hidden_dim, n_pathway_neighbors)
        self.maml_adapter = MAMLFewShotAdapter(
            input_dim=2 * hidden_dim,
            hidden_dim=hidden_dim,
            inner_lr=inner_lr,
        )
        self.antigen_forecaster = AntigenLossForecaster(hidden_dim, n_forecast_steps)
        self.sequence_planner = SequentialImmunotherapyPlanner(hidden_dim, max_sequence_length)

        # Risk assessment head
        self.crs_icans_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        logger.info(
            f"Initialized ImmunotherapyResponsePredictor "
            f"(hidden_dim={hidden_dim}, n_forecast_steps={n_forecast_steps})"
        )

    def predict_response(
        self,
        patient_profile: PatientImmuneProfile,
        mechanism: AgentMechanism,
        target_expression: float,
        pathway_neighbors: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Single-point response prediction for a patient-agent pair.

        Args:
            patient_profile: Patient immune characterization.
            mechanism: Immunotherapy agent mechanism.
            target_expression: Primary target antigen level.
            pathway_neighbors: Pathway neighbor expression levels.

        Returns:
            Dictionary with response probability and resistance risk.
        """
        # Encode patient and agent
        patient_emb = self.patient_profiler(patient_profile)
        agent_emb = self.agent_encoder(mechanism, target_expression, pathway_neighbors)

        # Concatenate for response prediction
        combined = torch.cat([patient_emb, agent_emb])

        # Predict with frozen backbone (no few-shot adaptation here)
        logits = self.maml_adapter.backbone(combined.unsqueeze(0))
        response_prob = torch.sigmoid(self.maml_adapter.response_head(logits)).item()

        return {
            "response_probability": response_prob,
            "patient_embedding": patient_emb.detach().numpy(),
            "agent_embedding": agent_emb.detach().numpy(),
        }

    def predict_with_few_shot(
        self,
        patient_profile: PatientImmuneProfile,
        mechanism: AgentMechanism,
        target_expression: float,
        support_profiles: List[PatientImmuneProfile],
        support_responses: List[int],
        pathway_neighbors: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Few-shot adapted response prediction using MAML on novel agent data.

        Args:
            patient_profile: Query patient for prediction.
            mechanism: Query agent mechanism.
            target_expression: Target antigen level.
            support_profiles: Support set of reference patients (n=5-20).
            support_responses: Binary response labels for support set.
            pathway_neighbors: Pathway neighbor levels.

        Returns:
            Dictionary with adapted response prediction and metadata.
        """
        # Encode query patient and agent
        patient_emb = self.patient_profiler(patient_profile)
        agent_emb = self.agent_encoder(mechanism, target_expression, pathway_neighbors)
        query_emb = torch.cat([patient_emb, agent_emb]).unsqueeze(0)

        # Encode support set
        support_embs = []
        for profile in support_profiles:
            profile_emb = self.patient_profiler(profile)
            # Use placeholder for support agent (assume same mechanism)
            support_embs.append(torch.cat([profile_emb, agent_emb]))
        support_embs = torch.stack(support_embs)
        support_responses_tensor = torch.tensor(support_responses, dtype=torch.long)

        # MAML adaptation
        adapted_pred = self.maml_adapter(support_embs, support_responses_tensor, query_emb)

        return {
            "adapted_response_probability": adapted_pred.item(),
            "support_set_size": len(support_profiles),
        }

    def predict_resistance_trajectory(
        self,
        antigen_history: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Forecast antigen loss trajectory and resistance risk.

        Args:
            antigen_history: Temporal sequence of antigen levels
                           (batch_size, seq_len, 3) with [BCMA, GPRC5D, FCRL5].

        Returns:
            Dictionary with forecasted trajectory and loss risk.
        """
        forecast, loss_risk = self.antigen_forecaster(antigen_history)

        return {
            "antigen_forecast": forecast.detach().numpy(),
            "loss_risk": loss_risk.detach().numpy(),
        }

    def plan_sequential_therapy(
        self,
        patient_profile: PatientImmuneProfile,
        initial_mechanism: AgentMechanism,
        initial_target: float,
        n_steps: int = 3,
    ) -> Dict[str, Any]:
        """
        Plan multi-step sequential immunotherapy strategy.

        Args:
            patient_profile: Patient immune characterization.
            initial_mechanism: First agent mechanism.
            initial_target: Initial target expression.
            n_steps: Number of planning steps.

        Returns:
            Dictionary with planned agent sequence and efficacy predictions.
        """
        # Encode patient and initial agent
        patient_emb = self.patient_profiler(patient_profile)
        agent_emb = self.agent_encoder(initial_mechanism, initial_target, None)

        # Plan sequence
        sequence, efficacy_scores = self.sequence_planner(patient_emb, agent_emb, n_steps)

        agent_names = ["BCMA", "GPRC5D", "CELMoD"]
        planned_agents = [agent_names[idx] for idx in sequence]

        return {
            "planned_sequence": planned_agents,
            "agent_indices": sequence,
            "efficacy_predictions": efficacy_scores.tolist(),
        }

    def assess_crs_icans_risk(
        self,
        patient_profile: PatientImmuneProfile,
        mechanism: AgentMechanism,
        target_expression: float,
    ) -> Dict[str, float]:
        """
        Assess cytokine release syndrome (CRS) and immune effector cell
        neurotoxicity syndrome (ICANS) risk.

        Args:
            patient_profile: Patient characterization.
            mechanism: Agent mechanism.
            target_expression: Target antigen level.

        Returns:
            Dictionary with CRS and ICANS risk scores.
        """
        patient_emb = self.patient_profiler(patient_profile)
        agent_emb = self.agent_encoder(mechanism, target_expression, None)
        combined = torch.cat([patient_emb, agent_emb]).unsqueeze(0)

        risk_logits = self.crs_icans_head(combined)
        crs_risk = torch.sigmoid(risk_logits[0, 0]).item()
        icans_risk = torch.sigmoid(risk_logits[0, 1]).item()

        return {
            "crs_risk": crs_risk,
            "icans_risk": icans_risk,
        }


def main():
    """Example usage of ImmunotherapyResponsePredictor."""
    logger.basicConfig(level=logging.INFO)

    # Initialize predictor
    predictor = ImmunotherapyResponsePredictor(hidden_dim=64)

    # Create example patient profile
    patient = PatientImmuneProfile(
        antigen_bcma=0.85,
        antigen_gprc5d=0.72,
        antigen_fcrl5=0.68,
        exhaustion_score=0.55,
        cd4_cd8_ratio=1.2,
        tcr_clonality=0.45,
    )

    # Create example agent mechanism
    agent = AgentMechanism(
        primary_target="BCMA",
        mechanism_type="bispecific",
        mechanism_idx=0,
    )

    # Single-point prediction
    result = predictor.predict_response(patient, agent, target_expression=0.85)
    logger.info(f"Response prediction: {result['response_probability']:.3f}")

    # Sequential therapy planning
    sequence_result = predictor.plan_sequential_therapy(patient, agent, 0.85, n_steps=3)
    logger.info(f"Planned sequence: {sequence_result['planned_sequence']}")

    # Risk assessment
    risk_result = predictor.assess_crs_icans_risk(patient, agent, 0.85)
    logger.info(f"CRS risk: {risk_result['crs_risk']:.3f}, ICANS risk: {risk_result['icans_risk']:.3f}")


if __name__ == "__main__":
    main()
