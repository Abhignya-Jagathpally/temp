"""
Gap 5.1: Treatment-Conditioned Prediction & Causal Inference Module.

This module implements advanced causal inference and treatment-conditioned prediction
for the ResistanceMap pipeline, enabling counterfactual reasoning about personalized
MM treatment strategies.

Key Components:
  - DragonNetHead: Twin propensity/outcome prediction architecture
  - TreatmentConditionedODEFunc: Neural ODE with FiLM-conditioned treatment sequences
  - IPTWEstimator: Inverse Probability of Treatment Weighting with stabilization
  - CounterfactualCrossAttention: Retrieves similar treatment-response pairs for inference
  - TreatmentActorCritic: RL-based decision support (stay/escalate/switch/add)
  - TreatmentConditionedPredictor: Top-level orchestrator

Loss Components:
  L = L_factual + λ_prop * L_propensity + λ_iptw * L_IPTW_weighted + λ_causal * L_ATE_regularization

References:
  - Kennedy et al. (2017): Doubly Robust Off-Policy Evaluation
  - Shi et al. (2019): Adapting Neural Networks for the Estimation of Treatment Effects
  - Chen et al. (2018): Variational E-bounds and the Delta-Method
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Lazy import torchdiffeq for optional ODE support
try:
    from torchdiffeq import odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    odeint_adjoint = None

logger = logging.getLogger(__name__)


class DragonNetHead(nn.Module):
    r"""
    DragonNet architecture: coupled propensity & dual outcome prediction heads.

    Implements the shared representation architecture from Kennedy et al., with:
      - Shared representation layer (s(X))
      - Propensity head: P(T=1|X)
      - Treatment outcome head: E[Y|T=1, X]
      - Control outcome head: E[Y|T=0, X]

    The shared representation is learned jointly with heads to enable confounding
    adjustment through propensity-based weighting.

    Args:
        input_dim (int): Feature dimension after preprocessing.
        hidden_dim (int): Hidden layer width for representation & heads.
        outcome_dim (int): Output dimension (typically 1 for survival/progression risk).
        dropout_rate (float): Dropout for regularization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        outcome_dim: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.outcome_dim = outcome_dim

        # Shared representation: learned feature embedding
        self.representation = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Propensity head: binary treatment assignment probability
        self.propensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Outcome heads (parallel for treatment & control)
        self.outcome_treated = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, outcome_dim),
        )

        self.outcome_control = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, outcome_dim),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass computing propensity & conditional outcomes.

        Args:
            x (Tensor): Input features [batch_size, input_dim]

        Returns:
            Tuple of 4 tensors:
                - representation: shared features [batch_size, hidden_dim]
                - propensity: P(T=1|X) [batch_size, 1]
                - outcome_treated: E[Y|T=1,X] [batch_size, outcome_dim]
                - outcome_control: E[Y|T=0,X] [batch_size, outcome_dim]
        """
        representation = self.representation(x)
        propensity = self.propensity_head(representation)
        outcome_treated = self.outcome_treated(representation)
        outcome_control = self.outcome_control(representation)

        return representation, propensity, outcome_treated, outcome_control


class TreatmentConditionedODEFunc(nn.Module):
    r"""
    Neural ODE dynamics with FiLM conditioning on treatment sequences.

    Implements dz/dt = f_θ(t, z(t), u(t)) where:
      - z(t) = [representation(t); treatment_indicator(t)] is the augmented state
      - u(t) = interpolated treatment sequence (drug class, dose, CD38 status)
      - f_θ uses Feature-wise Linear Modulation (FiLM) to condition on u(t)

    FiLM modulation: h' = γ(u) * h + β(u) where γ, β are treatment-dependent scalars.

    Args:
        state_dim (int): Dimension of augmented state [repr_dim + treatment_dim].
        hidden_dim (int): ODE network hidden layer width.
        treatment_dim (int): Dimension of treatment conditioning signal.
    """

    def __init__(
        self, state_dim: int, hidden_dim: int = 256, treatment_dim: int = 8
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.treatment_dim = treatment_dim

        # FiLM modulation networks
        self.gamma_net = nn.Sequential(
            nn.Linear(treatment_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(treatment_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Core ODE dynamics
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t: Tensor, z: Tensor, u: Tensor) -> Tensor:
        """
        Compute dz/dt = f_θ(t, z, u) with FiLM modulation.

        Args:
            t (Tensor): Time index (scalar or [batch_size], typically not used directly).
            z (Tensor): Augmented state [batch_size, state_dim].
            u (Tensor): Treatment conditioning signal [batch_size, treatment_dim].

        Returns:
            Tensor: Derivative dz/dt [batch_size, state_dim].
        """
        # Compute FiLM modulation parameters
        gamma = self.gamma_net(u)  # [batch_size, state_dim], > 0
        beta = self.beta_net(u)  # [batch_size, state_dim]

        # Base dynamics
        dz_base = self.dynamics(z)  # [batch_size, state_dim]

        # FiLM modulation: dz/dt = γ(u) * dz_base + β(u)
        dz_dt = gamma * dz_base + beta

        return dz_dt


class IPTWEstimator(nn.Module):
    r"""
    Inverse Probability of Treatment Weighting estimator.

    Computes IPTW for confounding adjustment using propensity scores from DragonNet:
      - Weight = 1 / P(T|H) for factual treatment
      - Stabilized weight = P(T) / P(T|H) reduces variance
      - Trimming: clip weights to [q_trim, 1/q_trim] to remove extreme propensities

    Args:
        trim_quantile (float): Quantile for trimming (default 0.05 → [0.05, 20]).
        epsilon (float): Numerical stability for propensity scores.
    """

    def __init__(self, trim_quantile: float = 0.05, epsilon: float = 1e-6):
        super().__init__()
        self.trim_quantile = trim_quantile
        self.epsilon = epsilon

    def forward(
        self,
        propensity: Tensor,
        treatment: Tensor,
        treatment_marginal: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute IPTW weights with optional stabilization.

        Args:
            propensity (Tensor): P(T=1|X) from propensity head [batch_size, 1].
            treatment (Tensor): Observed treatment indicators [batch_size, 1], binary.
            treatment_marginal (Tensor, optional): P(T=1) marginal propensity for
                stabilization. If None, uses simple IPTW (1/P(T|H)).

        Returns:
            Tuple[Tensor, Tensor]:
                - weights: IPTW weights [batch_size, 1], clipped to trim range.
                - propensity_score: Clipped propensity [batch_size, 1] for logging.
        """
        # Clamp propensity for numerical stability
        ps = torch.clamp(propensity, self.epsilon, 1.0 - self.epsilon)

        # Compute factual propensity: P(T_observed | X)
        # For T=1: use P(T=1|X); for T=0: use 1 - P(T=1|X)
        ps_factual = torch.where(
            treatment > 0.5, ps, 1.0 - ps
        )  # [batch_size, 1]

        # Simple IPTW: weight = 1 / P(T_observed | X)
        if treatment_marginal is None:
            weights = 1.0 / (ps_factual + self.epsilon)
        else:
            # Stabilized IPTW: weight = P(T_observed) / P(T_observed | X)
            marginal_factual = torch.where(
                treatment > 0.5, treatment_marginal, 1.0 - treatment_marginal
            )
            weights = marginal_factual / (ps_factual + self.epsilon)

        # Trim extreme weights
        trim_lower = self.trim_quantile
        trim_upper = 1.0 / self.trim_quantile
        weights_trimmed = torch.clamp(weights, trim_lower, trim_upper)

        if (weights > trim_upper).any() or (weights < trim_lower).any():
            logger.debug(
                f"IPTW weights trimmed: {(weights > trim_upper).sum().item()} "
                f"upper, {(weights < trim_lower).sum().item()} lower"
            )

        return weights_trimmed, ps

    def compute_stability_diagnostics(self, weights: Tensor) -> Dict[str, float]:
        """
        Diagnostic statistics for IPTW weight stability.

        Args:
            weights (Tensor): IPTW weights [batch_size, 1].

        Returns:
            Dict with statistics: mean, std, min, max, effective sample size ratio.
        """
        w = weights.detach().squeeze()
        n_eff = (w.sum() ** 2) / (w ** 2).sum()
        return {
            "weight_mean": w.mean().item(),
            "weight_std": w.std().item(),
            "weight_min": w.min().item(),
            "weight_max": w.max().item(),
            "eff_sample_size_ratio": (n_eff / w.shape[0]).item(),
        }


class CounterfactualCrossAttention(nn.Module):
    r"""
    Cross-attention mechanism for retrieving similar treatment-response pairs.

    Enables counterfactual inference by finding historical treatment patterns
    similar to current patient state, providing analogous outcome information.

    Mechanism:
      Query: current patient [representation; recent_labs; treatment_history]
      Key/Value: historical patient database with matched treatment sequences
      Output: attention-weighted counterfactual outcomes

    Args:
        query_dim (int): Dimension of patient state query.
        key_dim (int): Dimension of historical patient states.
        n_heads (int): Multi-head attention heads.
        dropout_rate (float): Dropout in attention.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int = 256,
        n_heads: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        assert key_dim % n_heads == 0, "key_dim must be divisible by n_heads"

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.n_heads = n_heads
        self.head_dim = key_dim // n_heads

        # Query projection
        self.query_proj = nn.Linear(query_dim, key_dim)

        # Key/Value projections for memory (historical data)
        self.key_proj = nn.Linear(key_dim, key_dim)
        self.value_proj = nn.Linear(key_dim, key_dim)

        # Attention scale and dropout
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout_rate)

        # Output projection
        self.output_proj = nn.Linear(key_dim, key_dim)

    def forward(
        self,
        query: Tensor,
        memory_key: Tensor,
        memory_value: Tensor,
        memory_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute multi-head cross-attention over historical treatment data.

        Args:
            query (Tensor): Current patient state [batch_size, query_dim].
            memory_key (Tensor): Historical patient states (keys) [n_memory, key_dim].
            memory_value (Tensor): Historical outcomes (values) [n_memory, key_dim].
            memory_mask (Tensor, optional): Mask invalid historical samples [n_memory].

        Returns:
            Tuple[Tensor, Tensor]:
                - attended: attention-weighted historical outcomes [batch_size, key_dim].
                - weights: attention weights [batch_size, n_heads, n_memory].
        """
        batch_size = query.size(0)
        n_memory = memory_key.size(0)

        # Project query, key, value
        Q = self.query_proj(query)  # [batch_size, key_dim]
        K = self.key_proj(memory_key)  # [n_memory, key_dim]
        V = self.value_proj(memory_value)  # [n_memory, key_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(1, n_memory, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(1, n_memory, self.n_heads, self.head_dim).transpose(1, 2)
        # Q: [batch_size, n_heads, 1, head_dim]
        # K, V: [1, n_heads, n_memory, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [batch_size, n_heads, 1, n_memory]

        # Apply mask if provided
        if memory_mask is not None:
            scores = scores.masked_fill(~memory_mask.unsqueeze(0).unsqueeze(0).unsqueeze(2), float("-inf"))

        # Softmax attention
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, n_heads, 1, n_memory]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [batch_size, n_heads, 1, head_dim]
        attended = attended.transpose(1, 2).contiguous()  # [batch_size, 1, n_heads, head_dim]
        attended = attended.view(batch_size, self.key_dim)  # [batch_size, key_dim]

        # Output projection
        attended = self.output_proj(attended)

        return attended, attn_weights.squeeze(2)  # [batch_size, key_dim], [batch_size, n_heads, n_memory]


class TreatmentActorCritic(nn.Module):
    r"""
    Actor-Critic reinforcement learning head for treatment decision support.

    Action space:
      0: Stay on current treatment
      1: Escalate (increase dose/intensity)
      2: Switch to alternative drug
      3: Add CD38 monoclonal antibody

    State: [z(t); recent_labs; treatment_history; resistance_risk]

    Actor computes: π(a|s) = softmax(actor_head(s))
    Critic computes: V(s) = expected progression-free survival from state

    Args:
        state_dim (int): Dimension of ODE state + labs + history + risk.
        hidden_dim (int): Actor & critic network width.
        n_actions (int): Number of available treatment actions (default 4).
    """

    def __init__(
        self, state_dim: int, hidden_dim: int = 256, n_actions: int = 4
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        # Shared feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head: policy over actions
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        # Critic head: expected value (PFS in months, bounded [0, 60])
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1], scale to [0, 60]
        )

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute policy (actor) and value (critic) outputs.

        Args:
            state (Tensor): Patient state [batch_size, state_dim].

        Returns:
            Tuple[Tensor, Tensor]:
                - policy: action probabilities [batch_size, n_actions].
                - value: expected PFS [batch_size, 1], in months (0-60).
        """
        features = self.feature_encoder(state)
        logits = self.actor(features)
        policy = F.softmax(logits, dim=-1)

        # Scale value to [0, 60] months
        value = self.critic(features) * 60.0

        return policy, value


class TreatmentConditionedPredictor(nn.Module):
    r"""
    Top-level module orchestrating treatment-conditioned prediction & causal inference.

    Integration of:
      1. DragonNetHead: confounding adjustment via propensity scoring
      2. TreatmentConditionedODEFunc: longitudinal dynamics with treatment coupling
      3. IPTWEstimator: IPTW weighting for unbiased outcome estimation
      4. CounterfactualCrossAttention: similarity-based counterfactual inference
      5. TreatmentActorCritic: RL-based treatment decision support

    Loss: L = L_factual + λ_prop * L_propensity + λ_iptw * L_IPTW_weighted
                + λ_causal * L_ATE_regularization

    Args:
        input_dim (int): Dimension of static patient features.
        n_ode_steps (int): Number of time steps for ODE integration (default 10).
        repr_dim (int): Representation dimension (default 256).
        treatment_dim (int): Treatment encoding dimension (default 8).
        use_ode (bool): Enable Neural ODE for longitudinal dynamics (default True).
        lambda_propensity (float): Propensity loss weight.
        lambda_iptw (float): IPTW loss weight.
        lambda_causal (float): Causal regularization weight.
    """

    def __init__(
        self,
        input_dim: int,
        n_ode_steps: int = 10,
        repr_dim: int = 256,
        treatment_dim: int = 8,
        use_ode: bool = True,
        lambda_propensity: float = 0.1,
        lambda_iptw: float = 0.05,
        lambda_causal: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_ode_steps = n_ode_steps
        self.repr_dim = repr_dim
        self.treatment_dim = treatment_dim
        self.use_ode = use_ode and TORCHDIFFEQ_AVAILABLE
        self.lambda_propensity = lambda_propensity
        self.lambda_iptw = lambda_iptw
        self.lambda_causal = lambda_causal

        if use_ode and not TORCHDIFFEQ_AVAILABLE:
            logger.warning(
                "torchdiffeq not installed; Neural ODE disabled. "
                "Install: pip install torchdiffeq"
            )

        # 1. DragonNet for confounding adjustment
        self.dragonnet = DragonNetHead(
            input_dim=input_dim, hidden_dim=repr_dim, outcome_dim=1
        )

        # 2. Neural ODE for longitudinal dynamics (optional)
        if self.use_ode:
            state_dim = repr_dim + treatment_dim
            self.ode_func = TreatmentConditionedODEFunc(
                state_dim=state_dim,
                hidden_dim=repr_dim,
                treatment_dim=treatment_dim,
            )
            self.register_buffer(
                "t_eval",
                torch.linspace(0, 1, n_ode_steps),
            )

        # 3. IPTW estimator
        self.iptw_estimator = IPTWEstimator(trim_quantile=0.05)

        # 4. Counterfactual cross-attention
        self.counterfactual_attention = CounterfactualCrossAttention(
            query_dim=repr_dim + treatment_dim + 32,  # repr + treatment + labs
            key_dim=repr_dim,
            n_heads=4,
        )

        # 5. Actor-Critic RL head
        self.actor_critic = TreatmentActorCritic(
            state_dim=repr_dim + treatment_dim + 32 + 8,  # + labs + history_risk
            hidden_dim=repr_dim,
            n_actions=4,
        )

        # Treatment encoder (map drug/dose to vector)
        self.treatment_encoder = nn.Sequential(
            nn.Linear(self.treatment_dim, repr_dim // 2),
            nn.ReLU(),
            nn.Linear(repr_dim // 2, self.treatment_dim),
        )

    def forward(
        self,
        x: Tensor,
        treatment: Tensor,
        treatment_sequence: Optional[Tensor] = None,
        recent_labs: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass: DragonNet → ODE integration → IPTW weighting → RL decision.

        Args:
            x (Tensor): Static patient features [batch_size, input_dim].
            treatment (Tensor): Current treatment indicator [batch_size, 1], binary.
            treatment_sequence (Tensor, optional): Treatment trajectory over time
                [batch_size, n_ode_steps, treatment_dim]. If None, constant treatment assumed.
            recent_labs (Tensor, optional): Recent lab values [batch_size, 32].

        Returns:
            Dict with keys:
                - "representation": learned representation [batch_size, repr_dim]
                - "propensity": P(T=1|X) [batch_size, 1]
                - "outcome_treated": E[Y|T=1,X] [batch_size, 1]
                - "outcome_control": E[Y|T=0,X] [batch_size, 1]
                - "iptw_weight": IPTW weight [batch_size, 1]
                - "ode_trajectory": ODE state trajectory (if use_ode) [batch_size, n_ode_steps, state_dim]
                - "counterfactual_outcome": counterfactual inference [batch_size, repr_dim]
                - "policy": treatment action policy [batch_size, 4]
                - "value": expected PFS [batch_size, 1]
        """
        output = {}

        # ===== 1. DragonNet: Propensity & Outcome Prediction =====
        repr_x, propensity, outcome_treated, outcome_control = self.dragonnet(x)
        output["representation"] = repr_x
        output["propensity"] = propensity
        output["outcome_treated"] = outcome_treated
        output["outcome_control"] = outcome_control

        # ===== 2. ODE Integration (optional) =====
        if self.use_ode and treatment_sequence is not None:
            z_trajectory = self._integrate_ode(repr_x, treatment_sequence)
            output["ode_trajectory"] = z_trajectory
            z_final = z_trajectory[:, -1, :]  # Use final ODE state
        else:
            # Fallback: concatenate with treatment encoding
            treatment_encoded = self.treatment_encoder(
                torch.zeros(x.size(0), self.treatment_dim, device=x.device)
                if treatment_sequence is None
                else treatment_sequence[:, -1, :]
            )
            z_final = torch.cat([repr_x, treatment_encoded], dim=-1)

        # ===== 3. IPTW Weighting =====
        iptw_weight, _ = self.iptw_estimator(propensity, treatment)
        output["iptw_weight"] = iptw_weight

        # ===== 4. Counterfactual Inference (requires historical memory) =====
        if recent_labs is not None and z_final.size(-1) > self.repr_dim:
            query = torch.cat([repr_x, recent_labs], dim=-1)
            # Memory would come from historical data store
            # Placeholder: use current batch as memory
            memory_key = repr_x
            memory_value = outcome_control
            counterfactual, _ = self.counterfactual_attention(
                query, memory_key, memory_value
            )
            output["counterfactual_outcome"] = counterfactual

        # ===== 5. Actor-Critic RL Head =====
        if recent_labs is not None:
            # Construct full state: [z(t); labs; treatment_history; resistance_risk]
            treatment_history = (
                torch.zeros(x.size(0), 8, device=x.device)
                if treatment_sequence is None
                else treatment_sequence[:, -1, :].repeat(1, 8 // self.treatment_dim + 1)[
                    :, :8
                ]
            )
            resistance_risk = torch.sigmoid(outcome_treated - outcome_control)
            full_state = torch.cat(
                [z_final, recent_labs, treatment_history, resistance_risk], dim=-1
            )
            policy, value = self.actor_critic(full_state)
            output["policy"] = policy
            output["value"] = value

        return output

    def _integrate_ode(self, repr_x: Tensor, treatment_sequence: Tensor) -> Tensor:
        """
        Integrate ODE forward in time with treatment conditioning.

        Args:
            repr_x (Tensor): Initial representation [batch_size, repr_dim].
            treatment_sequence (Tensor): Treatment trajectory [batch_size, n_ode_steps, treatment_dim].

        Returns:
            Tensor: ODE state trajectory [batch_size, n_ode_steps, repr_dim + treatment_dim].
        """
        if not self.use_ode:
            raise RuntimeError("ODE integration not available")

        batch_size = repr_x.size(0)
        device = repr_x.device

        # Initialize augmented state: [representation; treatment_indicator]
        treatment_initial = treatment_sequence[:, 0, :]
        z0 = torch.cat([repr_x, treatment_initial], dim=-1)

        # ODE function wrapper that interpolates treatment sequence
        def ode_func_wrapper(t, z):
            # Linear interpolation of treatment sequence
            t_idx = (t * (self.n_ode_steps - 1)).clamp(0, self.n_ode_steps - 1)
            t_idx_floor = torch.floor(t_idx).long()
            t_idx_ceil = torch.ceil(t_idx).long()
            alpha = t_idx - t_idx_floor.float()

            u_interp = (
                (1 - alpha).unsqueeze(-1) * treatment_sequence[:, t_idx_floor, :]
                + alpha.unsqueeze(-1) * treatment_sequence[:, t_idx_ceil, :]
            )
            return self.ode_func(t, z, u_interp)

        # Integrate ODE
        z_trajectory = odeint_adjoint(
            ode_func_wrapper,
            z0,
            self.t_eval.to(device),
            method="dopri",
            rtol=1e-3,
            atol=1e-4,
        )
        # [n_ode_steps, batch_size, state_dim]

        return z_trajectory.transpose(0, 1)  # [batch_size, n_ode_steps, state_dim]

    def compute_loss(
        self,
        output: Dict[str, Tensor],
        y_true: Tensor,
        treatment: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute combined loss: L_factual + λ_prop L_propensity + λ_iptw L_IPTW + λ_causal L_ATE.

        Args:
            output (Dict): Forward pass outputs.
            y_true (Tensor): True outcomes [batch_size, 1].
            treatment (Tensor): Observed treatments [batch_size, 1], binary.
            **kwargs: Additional arguments (propensity_true for calibration, etc).

        Returns:
            Tuple[Tensor, Dict[str, Tensor]]:
                - total_loss: scalar loss
                - loss_dict: component losses for logging
        """
        loss_dict = {}

        # ===== L_factual: outcome prediction loss =====
        # Weighted by IPTW to remove confounding bias
        outcome_pred = torch.where(
            treatment > 0.5,
            output["outcome_treated"],
            output["outcome_control"],
        )
        iptw_weight = output.get("iptw_weight", torch.ones_like(y_true))
        l_factual = F.mse_loss(outcome_pred, y_true, reduction="none")
        l_factual = (l_factual * iptw_weight).mean()
        loss_dict["l_factual"] = l_factual

        # ===== L_propensity: propensity score calibration =====
        # Propensity should match observed treatment distribution
        propensity_true = kwargs.get("propensity_true", treatment.mean())
        l_propensity = F.binary_cross_entropy(
            output["propensity"], treatment, reduction="mean"
        )
        loss_dict["l_propensity"] = l_propensity

        # ===== L_IPTW_weighted: IPTW-adjusted outcome loss =====
        # Directly encourages low-variance IPTW weights
        l_iptw = (iptw_weight ** 2).mean()
        loss_dict["l_iptw_penalty"] = l_iptw

        # ===== L_ATE: Average Treatment Effect regularization =====
        # Encourage balanced potential outcomes for confounded groups
        ate_regularization = (
            output["outcome_treated"] - output["outcome_control"]
        ).abs().mean()
        loss_dict["l_ate_reg"] = ate_regularization

        # ===== Total loss =====
        total_loss = (
            l_factual
            + self.lambda_propensity * l_propensity
            + self.lambda_iptw * l_iptw
            + self.lambda_causal * ate_regularization
        )

        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict
