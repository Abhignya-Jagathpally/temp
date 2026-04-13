from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import odeint

logger = logging.getLogger(__name__)


@dataclass
class PatientObservation:
    """Timestamped clinical observation for a patient."""
    patient_id: str
    time: float
    m_protein: float
    ldh: Optional[float] = None
    egfr: Optional[float] = None
    new_lesions: bool = False
    mrd_status: Optional[str] = None  # "negative", "positive", None


@dataclass
class ClinicalEvent:
    """Triggered clinical event requiring model update."""
    patient_id: str
    event_type: str  # "response", "mrd_conversion", "progression", "trigger_event"
    time: float
    observation: PatientObservation


class JointLongitudinalSurvival(nn.Module):
    """
    Joint longitudinal-survival model with random effects.

    Longitudinal submodel:
        m_i(t) = intercept_i + slope_i * t + ε(t)
        where intercept_i ~ N(μ_intercept, σ²_intercept)
              slope_i ~ N(μ_slope, σ²_slope)

    Survival submodel:
        h_i(t) = h_0(t) * exp(γ * m_i(t))
        Cox partial likelihood with time-varying M-protein trajectory linkage.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_patients: int = 100,
    ):
        """
        Args:
            hidden_dim: Hidden layer dimension.
            latent_dim: Dimension of random effects (intercept + slope).
            n_patients: Number of patients for random effects storage.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_patients = n_patients

        # Longitudinal submodel: encode observations to latent random effects
        self.longitudinal_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Random effects: intercept and slope per patient
        self.log_intercept_std = nn.Parameter(torch.zeros(1))
        self.log_slope_std = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "random_intercepts",
            torch.zeros(n_patients),
            persistent=False
        )
        self.register_buffer(
            "random_slopes",
            torch.zeros(n_patients),
            persistent=False
        )

        # Survival linkage: gamma coefficient (strength of M-protein effect on hazard)
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # Baseline hazard: parametric (Weibull) or non-parametric
        self.log_lambda0 = nn.Parameter(torch.tensor(0.1))  # Weibull scale
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Weibull shape

    def longitudinal_trajectory(
        self,
        patient_idx: int,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute M-protein trajectory for a patient.

        Args:
            patient_idx: Patient index.
            times: Time points of shape (T,).

        Returns:
            Trajectory m_i(t) of shape (T,).
        """
        intercept = self.random_intercepts[patient_idx]
        slope = self.random_slopes[patient_idx]
        trajectory = intercept + slope * times
        return trajectory

    def baseline_hazard(self, times: torch.Tensor) -> torch.Tensor:
        """Weibull baseline hazard h_0(t) = λ₀ * α * t^(α-1)."""
        lambda0 = torch.exp(self.log_lambda0)
        return lambda0 * self.alpha * (times ** (self.alpha - 1))

    def cumulative_hazard(self, times: torch.Tensor) -> torch.Tensor:
        """Cumulative baseline hazard H_0(t) = λ₀ * t^α."""
        lambda0 = torch.exp(self.log_lambda0)
        return lambda0 * (times ** self.alpha)

    def hazard(
        self,
        patient_idx: int,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Patient-specific hazard: h_i(t) = h_0(t) * exp(γ * m_i(t)).

        Args:
            patient_idx: Patient index.
            times: Time points.

        Returns:
            Hazard of shape (T,).
        """
        m_t = self.longitudinal_trajectory(patient_idx, times)
        h0_t = self.baseline_hazard(times)
        return h0_t * torch.exp(self.gamma * m_t)

    def cox_partial_likelihood(
        self,
        patient_idx: int,
        event_time: float,
        times_at_risk: torch.Tensor,
        m_protein_at_risk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cox partial likelihood for one event.

        ℓ(β) = exp(γ * m_i(t_event)) / Σ_{j∈R(t)} exp(γ * m_j(t_event))

        Args:
            patient_idx: Index of event subject.
            event_time: Time of event.
            times_at_risk: Times of subjects at risk (batch).
            m_protein_at_risk: M-protein values at risk (batch).

        Returns:
            Log partial likelihood.
        """
        m_i_event = self.longitudinal_trajectory(
            patient_idx,
            torch.tensor([event_time], device=self.gamma.device)
        )[0]

        eta_event = self.gamma * m_i_event
        eta_risk = self.gamma * m_protein_at_risk

        # Numerator: risk set contribution
        log_likelihood = eta_event - torch.logsumexp(eta_risk, dim=0)
        return log_likelihood

    def forward(
        self,
        patient_indices: torch.Tensor,
        times: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute trajectories and hazards.

        Args:
            patient_indices: Patient indices of shape (B,).
            times: Times of shape (B,).

        Returns:
            (hazards, trajectories) of shapes (B,).
        """
        trajectories = torch.stack([
            self.longitudinal_trajectory(int(idx), times[i:i+1])
            for i, idx in enumerate(patient_indices)
        ]).squeeze()

        hazards = torch.stack([
            self.hazard(int(idx), times[i:i+1])
            for i, idx in enumerate(patient_indices)
        ]).squeeze()

        return hazards, trajectories


class EWCRegularizer(nn.Module):
    """
    Elastic Weight Consolidation (EWC) regularizer.

    Prevents catastrophic forgetting when updating model on new patient data.
    L_EWC = λ_EWC * Σ_i F_i * (w_i - w*_i)²

    where F_i is the Fisher Information Matrix diagonal (parameter importance).
    """

    def __init__(
        self,
        lambda_ewc: float = 0.4,
        num_samples_fisher: int = 32,
    ):
        """
        Args:
            lambda_ewc: EWC regularization strength.
            num_samples_fisher: Number of samples to estimate Fisher.
        """
        super().__init__()
        self.lambda_ewc = lambda_ewc
        self.num_samples_fisher = num_samples_fisher
        self.register_buffer("fisher_diagonal", None)
        self.register_buffer("old_params", None)

    def compute_fisher_diagonal(
        self,
        model: nn.Module,
        data_loader,
        loss_fn: Callable,
    ) -> dict[str, torch.Tensor]:
        """
        Estimate Fisher Information Matrix diagonal via empirical Fisher.

        F_i ≈ E[(∇_i ℓ)²]

        Args:
            model: Neural network module.
            data_loader: DataLoader with (inputs, targets).
            loss_fn: Loss function callable.

        Returns:
            Dictionary mapping param name to Fisher diagonal.
        """
        fisher = {}
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param)

        model.eval()
        with torch.enable_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if batch_idx >= self.num_samples_fisher:
                    break

                model.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward(create_graph=True)

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += (param.grad ** 2) / self.num_samples_fisher

        return fisher

    def consolidate(
        self,
        model: nn.Module,
        fisher_diagonal: dict[str, torch.Tensor],
    ):
        """
        Store current weights and Fisher diagonal for later regularization.

        Args:
            model: Neural network module.
            fisher_diagonal: Fisher diagonal dictionary.
        """
        self.fisher_diagonal = fisher_diagonal
        self.old_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty: λ_EWC * Σ_i F_i * (w_i - w*_i)².

        Args:
            model: Current neural network.

        Returns:
            Scalar penalty loss.
        """
        if self.fisher_diagonal is None or self.old_params is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        penalty = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if name in self.fisher_diagonal and name in self.old_params:
                fisher = self.fisher_diagonal[name]
                old_param = self.old_params[name]
                penalty += (fisher * (param - old_param) ** 2).sum()

        return self.lambda_ewc * penalty


class NeuralODEKalmanFilter(nn.Module):
    """
    Extended Kalman Filter for Neural ODE dynamics.

    Continuous state evolution: dz/dt = f(z, t)
    Discrete measurement updates when new labs arrive.

    State: z ∈ ℝ^d (latent trajectory)
    Measurement: y (M-protein observation + optional labs)
    """

    def __init__(
        self,
        state_dim: int = 8,
        measurement_dim: int = 1,
        hidden_dim: int = 32,
    ):
        """
        Args:
            state_dim: Dimension of latent state.
            measurement_dim: Dimension of measurement (M-protein, etc).
            hidden_dim: Hidden dimension of neural ODE.
        """
        super().__init__()
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # Neural ODE function: z' = f(z, t)
        self.ode_func = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Measurement function: y = h(z)
        self.measurement_func = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, measurement_dim),
        )

        # Jacobian computation (linearization for EKF)
        self.jacobian_eps = 1e-4

        # Covariance matrices
        self.register_buffer("Q", torch.eye(state_dim) * 0.01)  # Process noise
        self.register_buffer("R", torch.eye(measurement_dim) * 0.1)  # Measurement noise
        self.register_buffer("P", torch.eye(state_dim) * 1.0)  # State covariance

    def ode_dynamics(
        self,
        z: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """
        Neural ODE dynamics function.

        Args:
            z: State of shape (batch, state_dim).
            t: Time scalar.

        Returns:
            dz/dt of shape (batch, state_dim).
        """
        t_tensor = torch.full((z.shape[0], 1), t, dtype=z.dtype, device=z.device)
        zt = torch.cat([z, t_tensor], dim=-1)
        return self.ode_func(zt)

    def jacobian_ode_state(
        self,
        z: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """
        Compute Jacobian of ODE w.r.t. state: ∂f/∂z.

        Args:
            z: State of shape (batch, state_dim).
            t: Time scalar.

        Returns:
            Jacobian of shape (batch, state_dim, state_dim).
        """
        batch_size = z.shape[0]
        jacobian = torch.zeros(
            batch_size,
            self.state_dim,
            self.state_dim,
            dtype=z.dtype,
            device=z.device,
        )

        for i in range(self.state_dim):
            z_plus = z.clone()
            z_plus[:, i] += self.jacobian_eps
            z_minus = z.clone()
            z_minus[:, i] -= self.jacobian_eps

            f_plus = self.ode_dynamics(z_plus, t)
            f_minus = self.ode_dynamics(z_minus, t)

            jacobian[:, :, i] = (f_plus - f_minus) / (2 * self.jacobian_eps)

        return jacobian

    def jacobian_measurement(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Jacobian of measurement model: ∂h/∂z.

        Args:
            z: State of shape (batch, state_dim).

        Returns:
            Jacobian of shape (batch, measurement_dim, state_dim).
        """
        batch_size = z.shape[0]
        jacobian = torch.zeros(
            batch_size,
            self.measurement_dim,
            self.state_dim,
            dtype=z.dtype,
            device=z.device,
        )

        for i in range(self.state_dim):
            z_plus = z.clone()
            z_plus[:, i] += self.jacobian_eps
            z_minus = z.clone()
            z_minus[:, i] -= self.jacobian_eps

            h_plus = self.measurement_func(z_plus)
            h_minus = self.measurement_func(z_minus)

            jacobian[:, :, i] = (h_plus - h_minus) / (2 * self.jacobian_eps)

        return jacobian

    def predict(
        self,
        z: torch.Tensor,
        t_current: float,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction step: propagate state using neural ODE.

        Args:
            z: Current state of shape (batch, state_dim).
            t_current: Current time.
            dt: Time step.

        Returns:
            (z_pred, P_pred) - predicted state and covariance.
        """
        # Simple Euler step (production would use RK45)
        dz = self.ode_dynamics(z, t_current)
        z_pred = z + dz * dt

        # Covariance prediction: P = F*P*F^T + Q
        F = self.jacobian_ode_state(z, t_current)
        P_pred = torch.einsum('bpq,qr,bsr->ps', F, self.P, F) + self.Q

        return z_pred, P_pred

    def update(
        self,
        z_pred: torch.Tensor,
        P_pred: torch.Tensor,
        y_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Measurement update step (Kalman update).

        Args:
            z_pred: Predicted state of shape (batch, state_dim).
            P_pred: Predicted covariance of shape (state_dim, state_dim).
            y_obs: Observed measurement of shape (batch, measurement_dim).

        Returns:
            (z_updated, P_updated) - updated state and covariance.
        """
        # Innovation
        y_pred = self.measurement_func(z_pred)
        innovation = y_obs - y_pred

        # Innovation covariance: S = H*P*H^T + R
        H = self.jacobian_measurement(z_pred)
        S = torch.einsum('bmq,qr,bsr->ms', H, P_pred, H) + self.R

        # Kalman gain: K = P*H^T * S^{-1}
        K = torch.einsum('qr,brm,ms->qbs', P_pred, H, torch.linalg.inv(S + 1e-6 * torch.eye(S.shape[0], device=S.device)))

        # State update
        z_updated = z_pred + torch.einsum('qbs,bsm->bqm', K, innovation).squeeze(-1)

        # Covariance update: P = (I - K*H)*P
        I_KH = torch.eye(self.state_dim, device=P_pred.device) - torch.einsum('qbs,bsm->qm', K, H.mean(0))
        P_updated = torch.einsum('qr,rs,su->qu', I_KH, P_pred, I_KH) + 1e-6 * torch.eye(self.state_dim, device=P_pred.device)

        return z_updated, P_updated


class StreamingCoxUpdate(nn.Module):
    """
    Incremental partial likelihood update for Cox model.

    Processes events sequentially, updating model parameters
    as new patient progression events arrive.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size_cox: int = 8,
    ):
        """
        Args:
            learning_rate: Gradient step size.
            batch_size_cox: Risk set batch size.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size_cox = batch_size_cox

    def partial_likelihood_loss(
        self,
        joint_model: JointLongitudinalSurvival,
        event_patient_idx: int,
        event_time: float,
        risk_set_indices: torch.Tensor,
        risk_set_m_protein: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Cox partial likelihood loss for a single event.

        Args:
            joint_model: Joint model instance.
            event_patient_idx: Index of event subject.
            event_time: Time of event.
            risk_set_indices: Indices of at-risk subjects.
            risk_set_m_protein: M-protein values at risk.

        Returns:
            Negative log partial likelihood (loss).
        """
        log_likelihood = joint_model.cox_partial_likelihood(
            event_patient_idx,
            event_time,
            torch.zeros_like(risk_set_m_protein),  # dummy times
            risk_set_m_protein,
        )
        return -log_likelihood

    def update_step(
        self,
        joint_model: JointLongitudinalSurvival,
        optimizer: torch.optim.Optimizer,
        event_patient_idx: int,
        event_time: float,
        risk_set_indices: torch.Tensor,
        risk_set_m_protein: torch.Tensor,
    ):
        """
        Single stochastic gradient update on partial likelihood.

        Args:
            joint_model: Model to update.
            optimizer: PyTorch optimizer.
            event_patient_idx: Event subject index.
            event_time: Time of event.
            risk_set_indices: At-risk subject indices.
            risk_set_m_protein: M-protein values at risk.
        """
        optimizer.zero_grad()

        loss = self.partial_likelihood_loss(
            joint_model,
            event_patient_idx,
            event_time,
            risk_set_indices,
            risk_set_m_protein,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(joint_model.parameters(), 1.0)
        optimizer.step()


class ClinicalTriggerMonitor:
    """
    Rule-based detection of clinical triggers requiring model updates.

    Triggers:
    - Response assessment (>50% M-protein reduction)
    - MRD conversion (negative -> positive or vice versa)
    - ±25% M-protein change
    - >2×ULN LDH
    - New lesions
    - eGFR drop >25%
    - CUSUM/EWMA on log(M-protein) for sustained increase
    """

    def __init__(
        self,
        cusum_threshold: float = 5.0,
        ewma_lambda: float = 0.2,
    ):
        """
        Args:
            cusum_threshold: CUSUM threshold for change detection.
            ewma_lambda: EWMA decay parameter.
        """
        self.cusum_threshold = cusum_threshold
        self.ewma_lambda = ewma_lambda
        self.cusum_sum = {}  # per patient
        self.ewma_state = {}  # per patient
        self.baseline_m_protein = {}  # per patient

    def detect_triggers(
        self,
        observation: PatientObservation,
        previous_observation: Optional[PatientObservation] = None,
    ) -> list[ClinicalEvent]:
        """
        Detect clinical triggers for a patient observation.

        Args:
            observation: Current observation.
            previous_observation: Previous observation for comparison.

        Returns:
            List of triggered clinical events.
        """
        events = []
        patient_id = observation.patient_id

        # Initialize baselines
        if patient_id not in self.baseline_m_protein:
            self.baseline_m_protein[patient_id] = observation.m_protein
            self.cusum_sum[patient_id] = 0.0
            self.ewma_state[patient_id] = np.log(observation.m_protein + 1e-6)

        if previous_observation is not None:
            # M-protein change triggers
            pct_change = (
                (observation.m_protein - previous_observation.m_protein)
                / (previous_observation.m_protein + 1e-6)
            )

            # Response: >50% reduction
            if pct_change < -0.5:
                events.append(ClinicalEvent(
                    patient_id=patient_id,
                    event_type="response",
                    time=observation.time,
                    observation=observation,
                ))

            # ±25% change trigger
            if abs(pct_change) > 0.25:
                events.append(ClinicalEvent(
                    patient_id=patient_id,
                    event_type="trigger_event",
                    time=observation.time,
                    observation=observation,
                ))

        # MRD conversion
        if (previous_observation is not None
            and observation.mrd_status is not None
            and observation.mrd_status != previous_observation.mrd_status):
            events.append(ClinicalEvent(
                patient_id=patient_id,
                event_type="mrd_conversion",
                time=observation.time,
                observation=observation,
            ))

        # LDH trigger: >2×ULN (assume ULN=45)
        if observation.ldh is not None and observation.ldh > 90:
            events.append(ClinicalEvent(
                patient_id=patient_id,
                event_type="trigger_event",
                time=observation.time,
                observation=observation,
            ))

        # New lesions
        if observation.new_lesions:
            events.append(ClinicalEvent(
                patient_id=patient_id,
                event_type="progression",
                time=observation.time,
                observation=observation,
            ))

        # eGFR drop >25%
        if (previous_observation is not None
            and observation.egfr is not None
            and previous_observation.egfr is not None):
            egfr_change = (
                (observation.egfr - previous_observation.egfr)
                / (previous_observation.egfr + 1e-6)
            )
            if egfr_change < -0.25:
                events.append(ClinicalEvent(
                    patient_id=patient_id,
                    event_type="trigger_event",
                    time=observation.time,
                    observation=observation,
                ))

        # CUSUM/EWMA on log M-protein
        log_m = np.log(observation.m_protein + 1e-6)
        mu0 = self.ewma_state[patient_id]
        sigma = 0.1

        # CUSUM update (detect sustained increase)
        self.cusum_sum[patient_id] = max(
            0,
            self.cusum_sum[patient_id] + (log_m - mu0) / sigma
        )

        if self.cusum_sum[patient_id] > self.cusum_threshold:
            events.append(ClinicalEvent(
                patient_id=patient_id,
                event_type="trigger_event",
                time=observation.time,
                observation=observation,
            ))
            self.cusum_sum[patient_id] = 0.0  # Reset after trigger

        # EWMA update
        self.ewma_state[patient_id] = (
            self.ewma_lambda * log_m
            + (1 - self.ewma_lambda) * self.ewma_state[patient_id]
        )

        return events


class ContinualLearningManager:
    """
    Top-level orchestrator for dynamic monitoring and continual learning.

    Coordinates:
    - Clinical trigger detection
    - Model updates via EWC and streaming Cox
    - Kalman filter state tracking
    """

    def __init__(
        self,
        joint_model: JointLongitudinalSurvival,
        state_dim: int = 8,
        lambda_ewc: float = 0.4,
        learning_rate: float = 0.01,
    ):
        """
        Args:
            joint_model: Joint longitudinal-survival model.
            state_dim: Latent state dimension for Kalman filter.
            lambda_ewc: EWC regularization strength.
            learning_rate: Optimizer learning rate.
        """
        self.joint_model = joint_model
        self.state_dim = state_dim
        self.lambda_ewc = lambda_ewc
        self.learning_rate = learning_rate

        # Subcomponents
        self.kalman_filter = NeuralODEKalmanFilter(
            state_dim=state_dim,
            measurement_dim=1,
        )
        self.ewc_regularizer = EWCRegularizer(lambda_ewc=lambda_ewc)
        self.streaming_cox = StreamingCoxUpdate(
            learning_rate=learning_rate,
        )
        self.trigger_monitor = ClinicalTriggerMonitor()

        # Optimizer
        params = list(joint_model.parameters()) + list(self.kalman_filter.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

        # State tracking
        self.patient_states = {}  # patient_id -> z (latent state)
        self.patient_histories = {}  # patient_id -> list of observations
        self.update_count = 0

        logger.info("ContinualLearningManager initialized")

    def add_observation(
        self,
        observation: PatientObservation,
    ) -> Tuple[list[ClinicalEvent], bool]:
        """
        Add new patient observation and detect triggers.

        Args:
            observation: New clinical observation.

        Returns:
            (triggered_events, model_updated) tuple.
        """
        patient_id = observation.patient_id

        # Initialize patient state
        if patient_id not in self.patient_histories:
            self.patient_histories[patient_id] = []
            self.patient_states[patient_id] = torch.randn(1, self.state_dim)

        # Get previous observation if exists
        prev_obs = None
        if self.patient_histories[patient_id]:
            prev_obs = self.patient_histories[patient_id][-1]

        # Detect triggers
        events = self.trigger_monitor.detect_triggers(observation, prev_obs)

        # Add to history
        self.patient_histories[patient_id].append(observation)

        # Update model if events triggered
        model_updated = False
        if events:
            model_updated = self._update_model(patient_id, observation, events)

        return events, model_updated

    def _update_model(
        self,
        patient_id: str,
        observation: PatientObservation,
        events: list[ClinicalEvent],
    ) -> bool:
        """
        Update model based on triggered events.

        Args:
            patient_id: Patient ID.
            observation: Current observation.
            events: Triggered events.

        Returns:
            Whether update was performed.
        """
        try:
            # Kalman filter prediction + update
            z_pred, P_pred = self.kalman_filter.predict(
                self.patient_states[patient_id],
                t_current=observation.time - 0.01,
                dt=0.01,
            )

            y_obs = torch.tensor([[observation.m_protein]], dtype=torch.float32)
            z_updated, P_updated = self.kalman_filter.update(z_pred, P_pred, y_obs)
            self.patient_states[patient_id] = z_updated

            # Streaming Cox update
            if any(e.event_type in ["response", "progression"] for e in events):
                patient_idx = hash(patient_id) % self.joint_model.n_patients

                # Build risk set (simplified)
                risk_indices = torch.tensor([patient_idx])
                risk_m_protein = torch.tensor([[observation.m_protein]])

                self.streaming_cox.update_step(
                    self.joint_model,
                    self.optimizer,
                    patient_idx,
                    observation.time,
                    risk_indices,
                    risk_m_protein,
                )

            # EWC regularization after first update
            if self.update_count == 0:
                # Dummy data loader for Fisher estimation
                self._consolidate_ewc()
            elif self.update_count % 10 == 0:
                # Periodic Fisher updates
                self._consolidate_ewc()

            self.update_count += 1
            logger.info(
                f"Model updated for patient {patient_id} at time {observation.time:.2f}. "
                f"Events: {[e.event_type for e in events]}"
            )
            return True

        except Exception as e:
            logger.warning(f"Model update failed for {patient_id}: {e}")
            return False

    def _consolidate_ewc(self):
        """Consolidate EWC weights after model update."""
        # Simplified: just store current weights
        self.ewc_regularizer.old_params = {
            name: param.data.clone()
            for name, param in self.joint_model.named_parameters()
        }
        # Fisher would be computed on validation set
        logger.debug("EWC consolidation completed")

    def predict_trajectory(
        self,
        patient_id: str,
        times: np.ndarray,
    ) -> np.ndarray:
        """
        Predict M-protein trajectory for a patient.

        Args:
            patient_id: Patient ID.
            times: Time points for prediction.

        Returns:
            Predicted M-protein values.
        """
        if patient_id not in self.patient_histories:
            return np.full_like(times, np.nan)

        patient_idx = hash(patient_id) % self.joint_model.n_patients
        times_tensor = torch.tensor(times, dtype=torch.float32)

        with torch.no_grad():
            trajectory = self.joint_model.longitudinal_trajectory(
                patient_idx,
                times_tensor,
            )

        return trajectory.numpy()

    def get_patient_summary(self, patient_id: str) -> dict:
        """
        Get monitoring summary for a patient.

        Args:
            patient_id: Patient ID.

        Returns:
            Dictionary with patient info and alerts.
        """
        if patient_id not in self.patient_histories:
            return {}

        hist = self.patient_histories[patient_id]
        latest = hist[-1]

        summary = {
            "patient_id": patient_id,
            "last_observation_time": latest.time,
            "last_m_protein": latest.m_protein,
            "num_observations": len(hist),
            "alerts": [],
        }

        if len(hist) > 1:
            prev = hist[-2]
            pct_change = (latest.m_protein - prev.m_protein) / (prev.m_protein + 1e-6)
            summary["m_protein_pct_change"] = pct_change * 100

            if abs(pct_change) > 0.25:
                summary["alerts"].append("M-protein change >25%")

        if latest.ldh is not None and latest.ldh > 90:
            summary["alerts"].append("LDH elevated")

        if latest.new_lesions:
            summary["alerts"].append("New lesions detected")

        return summary
