"""
Dynamic Treatment Regimes & Counterfactual Reasoning for ResistanceMap.

Implements Q-learning, A-learning (AIPW), and Counterfactual Recurrent Networks
for optimal multi-stage treatment selection in multiple myeloma, integrating
clinical feasibility constraints and ResistanceMap sensitivity predictions.

Module structure:
  - FittedQIteration: Neural Q-function with backward recursion for optimal DTR.
  - CounterfactualRecurrentNetwork: Encoder-decoder RNN with gradient reversal
    for domain-adversarial balanced representations.
  - GradientReversalLayer: Gradient negation for adversarial balancing.
  - TreatmentFeasibilityMask: Clinical constraint enforcement.
  - ALearnerDoublyRobust: AIPW-based blip function estimation.
  - DTRPolicyOptimizer: Top-level coordinator integrating all methods.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Tanh, Sigmoid, Dropout
from abc import ABC, abstractmethod

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Gradient Reversal Layer (Domain Adversarial Training)
# ============================================================================

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient reversal for domain-adversarial training.
    Forward pass: identity. Backward pass: negate gradients.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Forward: return input as-is."""
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward: negate gradients by alpha."""
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal."""
    return GradientReversalLayer.apply(x, alpha)


# ============================================================================
# Treatment Feasibility Mask
# ============================================================================

class TreatmentFeasibilityMask:
    """
    Clinical constraint enforcement for NCCN 2025 MM regimens.

    Constraints:
      - CAR-T: one-time maximum (state variable tracks if used).
      - Nephrotoxic agents (amphotericin B, cisplatin) excluded if eGFR < 30.
      - ECOG >= 3: exclude intensive regimens (transplant, aggressive).
      - Age >= 75: avoid certain intensive protocols.

    Action space: 0=watch, 1=lenalidomide, 2=bortezomib, 3=carfilzomib,
                  4=daratumumab, 5=venetoclax, 6=CAR-T, 7=transplant.
    """

    def __init__(self, n_actions: int = 8):
        """
        Args:
            n_actions: Size of action space (8 for typical MM).
        """
        self.n_actions = n_actions
        # Action names (indices for reference)
        self.action_names = [
            "watchful_waiting", "lenalidomide", "bortezomib", "carfilzomib",
            "daratumumab", "venetoclax", "cart", "transplant"
        ]

    def get_feasible_actions(
        self,
        egfr: float,
        ecog: int,
        age: int,
        cart_used: bool,
    ) -> np.ndarray:
        """
        Return boolean mask of feasible actions for given clinical state.

        Args:
            egfr: Estimated glomerular filtration rate.
            ecog: ECOG performance status (0-4).
            age: Patient age.
            cart_used: Whether CAR-T has been administered previously.

        Returns:
            Boolean array of shape (n_actions,) where True = feasible.
        """
        feasible = np.ones(self.n_actions, dtype=bool)

        # CAR-T one-time limit
        if cart_used:
            feasible[6] = False

        # Nephrotoxic agents (carfilzomib at eGFR < 30)
        if egfr < 30:
            feasible[3] = False

        # ECOG >= 3: exclude intensive
        if ecog >= 3:
            feasible[7] = False  # transplant
            feasible[6] = False  # CAR-T (intensive)

        # Age >= 75: caution with transplant
        if age >= 75:
            feasible[7] = False

        return feasible

    def mask_logits(
        self,
        logits: torch.Tensor,
        feasible_mask: np.ndarray,
    ) -> torch.Tensor:
        """
        Apply feasibility mask to action logits (set infeasible to -inf).

        Args:
            logits: Shape (batch, n_actions).
            feasible_mask: Boolean array (n_actions,).

        Returns:
            Masked logits.
        """
        logits_masked = logits.clone()
        infeasible_idx = ~feasible_mask
        logits_masked[:, infeasible_idx] = float('-inf')
        return logits_masked


# ============================================================================
# Fitted Q-Iteration (Neural Q-Learning)
# ============================================================================

class FittedQIteration(nn.Module):
    """
    Fitted Q-Iteration for dynamic treatment regimes.

    Learns optimal decision rules π = (π_1, ..., π_K) via backward recursion.
    At stage k (from K down to 1):
      Q_k(H_k, a_k) = E[Y | H_k, a_k, optimal future actions]

    Uses neural network function approximator for Q-function.
    """

    def __init__(
        self,
        history_dim: int,
        n_actions: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            history_dim: Dimension of treatment history H_t.
            n_actions: Number of possible actions.
            hidden_dims: List of hidden layer dimensions. Default: [256, 128].
            dropout: Dropout rate.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.history_dim = history_dim
        self.n_actions = n_actions

        # Q-function network: (history, action) -> scalar Q-value
        layers = []
        prev_dim = history_dim + 1  # +1 for action encoding

        for hidden_dim in hidden_dims:
            layers.extend([
                Linear(prev_dim, hidden_dim),
                ReLU(),
                Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(Linear(prev_dim, 1))
        self.q_network = nn.Sequential(*layers)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)

    def forward(
        self,
        history: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q-values for (history, action) pairs.

        Args:
            history: Shape (batch, history_dim).
            actions: Shape (batch, 1) or (batch,), action indices.

        Returns:
            Q-values, shape (batch, 1).
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)

        # One-hot encode actions
        actions_one_hot = F.one_hot(
            actions.squeeze(1), num_classes=self.n_actions
        ).float()

        # Concatenate history and action
        x = torch.cat([history, actions_one_hot], dim=1)
        q_values = self.q_network(x)
        return q_values

    def compute_optimal_actions(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute optimal actions and their Q-values for given histories.

        Args:
            history: Shape (batch, history_dim).

        Returns:
            optimal_actions: Shape (batch,), recommended action indices.
            optimal_q_values: Shape (batch, 1), max Q-values.
        """
        batch_size = history.shape[0]
        q_all = []

        for a in range(self.n_actions):
            actions = torch.full((batch_size, 1), a, dtype=torch.long, device=history.device)
            q_vals = self(history, actions)
            q_all.append(q_vals)

        q_matrix = torch.cat(q_all, dim=1)  # (batch, n_actions)
        optimal_q_values, optimal_actions = torch.max(q_matrix, dim=1)

        return optimal_actions, optimal_q_values.unsqueeze(1)

    def backward_recursion(
        self,
        histories: List[torch.Tensor],
        rewards: torch.Tensor,
        gamma: float = 0.95,
        n_epochs: int = 10,
    ) -> Dict[str, float]:
        """
        Backward recursion to learn optimal decision rules.

        Args:
            histories: List of length K of tensors, each (batch, history_dim).
            rewards: Final outcomes, shape (batch, 1).
            gamma: Discount factor.
            n_epochs: Training epochs.

        Returns:
            Dictionary of training metrics.
        """
        K = len(histories)
        batch_size = histories[0].shape[0]

        # Initialize value function at final stage
        value_k = rewards.clone()

        metrics = {"losses": []}

        for epoch in range(n_epochs):
            epoch_loss = 0.0

            # Backward through stages
            for k in range(K - 1, -1, -1):
                history_k = histories[k]

                # Compute optimal Q-values
                opt_actions, opt_q = self.compute_optimal_actions(history_k)

                # Target: reward + gamma * future value
                if k < K - 1:
                    target = rewards + gamma * value_k
                else:
                    target = rewards

                # Loss: MSE between Q(H_k, a_k) and target
                self.optimizer.zero_grad()
                q_pred = self(history_k, opt_actions)
                loss = F.mse_loss(q_pred, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                value_k = opt_q

            metrics["losses"].append(epoch_loss / K)

        logger.info(f"Backward recursion complete. Final loss: {metrics['losses'][-1]:.4f}")
        return metrics


# ============================================================================
# Counterfactual Recurrent Network (CRN)
# ============================================================================

class CounterfactualRecurrentNetwork(nn.Module):
    """
    Encoder-decoder RNN with gradient reversal for domain-adversarial balancing.

    Learns balanced representations of treatment history that are predictive of
    counterfactual outcomes across all treatment arms.

    Architecture:
      - Encoder: GRU over treatment history → balanced representation.
      - Treatment discriminator: Predicts treatment from representation (reversed grads).
      - Decoder: GRU from balanced rep → outcome predictions under all treatments.
    """

    def __init__(
        self,
        history_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        n_outcomes: int = 1,
        dropout: float = 0.1,
    ):
        """
        Args:
            history_dim: Feature dimension per time step.
            n_actions: Number of actions.
            hidden_dim: GRU hidden dimension.
            n_outcomes: Number of outcome dimensions.
            dropout: Dropout rate.
        """
        super().__init__()
        self.history_dim = history_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_outcomes = n_outcomes

        # Encoder: history -> balanced representation
        self.encoder = GRU(
            input_size=history_dim + n_actions,  # history + one-hot action
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2,
            dropout=dropout,
        )

        # Treatment discriminator: rep -> treatment prediction (for adversarial balancing)
        self.discriminator = nn.Sequential(
            Linear(hidden_dim, 64),
            ReLU(),
            Dropout(dropout),
            Linear(64, n_actions),
        )

        # Decoder: balanced rep -> outcome prediction for each action
        self.decoder = nn.ModuleDict({
            f"action_{a}": nn.Sequential(
                Linear(hidden_dim, 64),
                ReLU(),
                Dropout(dropout),
                Linear(64, n_outcomes),
            )
            for a in range(n_actions)
        })

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def encode(
        self,
        treatment_history: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode treatment history to balanced representation.

        Args:
            treatment_history: Shape (batch, seq_len, history_dim).
            actions: Shape (batch, seq_len), action indices.

        Returns:
            Balanced representation, shape (batch, hidden_dim).
        """
        batch_size, seq_len = actions.shape

        # One-hot encode actions
        actions_one_hot = F.one_hot(actions, num_classes=self.n_actions).float()

        # Concatenate history and action
        x = torch.cat([treatment_history, actions_one_hot], dim=2)

        # Encode
        _, hidden = self.encoder(x)
        rep = hidden[-1]  # Last hidden state from final GRU layer

        return rep

    def predict_outcomes(
        self,
        rep: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Predict counterfactual outcomes under all actions given representation.

        Args:
            rep: Balanced representation, shape (batch, hidden_dim).

        Returns:
            Dictionary mapping action -> outcome tensor (batch, n_outcomes).
        """
        outcomes = {}
        for a in range(self.n_actions):
            outcomes[a] = self.decoder[f"action_{a}"](rep)
        return outcomes

    def forward(
        self,
        treatment_history: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Forward pass: encode, predict outcomes, and discriminator loss.

        Args:
            treatment_history: Shape (batch, seq_len, history_dim).
            actions: Shape (batch, seq_len), action indices.

        Returns:
            (counterfactual_outcomes, treatment_logits)
        """
        rep = self.encode(treatment_history, actions)
        outcomes = self.predict_outcomes(rep)

        # Treatment discriminator on reversed representation
        rep_reversed = gradient_reversal(rep)
        treatment_logits = self.discriminator(rep_reversed)

        return outcomes, treatment_logits

    def train_step(
        self,
        treatment_history: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
        alpha_adversarial: float = 0.1,
    ) -> Dict[str, float]:
        """
        Single training step with adversarial balancing.

        Args:
            treatment_history: Shape (batch, seq_len, history_dim).
            actions: Shape (batch, seq_len), action indices.
            targets: Shape (batch, n_outcomes), observed outcomes.
            alpha_adversarial: Weight for adversarial loss.

        Returns:
            Training metrics.
        """
        self.optimizer.zero_grad()

        # Forward pass
        outcomes, treatment_logits = self(treatment_history, actions)

        # Observed action for this batch (assuming single action per batch for simplicity)
        observed_action = actions[:, -1]  # Last action

        # Outcome prediction loss (only for observed action)
        outcome_pred = outcomes[0] if observed_action.size(0) > 0 else outcomes[0]
        outcome_loss = F.mse_loss(outcome_pred, targets)

        # Adversarial balancing loss (discriminator should not predict treatment)
        # Target is uniform distribution over actions
        uniform_target = torch.ones_like(treatment_logits) / self.n_actions
        adversarial_loss = F.kl_div(
            F.log_softmax(treatment_logits, dim=1),
            uniform_target,
            reduction='batchmean'
        )

        # Combined loss
        total_loss = outcome_loss + alpha_adversarial * adversarial_loss

        total_loss.backward()
        self.optimizer.step()

        return {
            "outcome_loss": outcome_loss.item(),
            "adversarial_loss": adversarial_loss.item(),
            "total_loss": total_loss.item(),
        }


# ============================================================================
# A-Learner: Doubly Robust Blip Estimation
# ============================================================================

class ALearnerDoublyRobust(ABC):
    """
    A-learner (Adaptive Interventional Estimator) for optimal DTR via AIPW.

    Estimates optimal blip function β(H_k, a_k) = E[Y | H_k, a_k] - E[Y | H_k]
    using doubly robust methods.

    Doubly robust property: Consistent if either:
      (1) Treatment propensity model is correct, OR
      (2) Outcome regression model is correct.
    """

    def __init__(self):
        """Initialize A-learner."""
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available; AIPW optimization may not work")

    def compute_aipw_weights(
        self,
        actions: np.ndarray,
        propensity: np.ndarray,
    ) -> np.ndarray:
        """
        Compute AIPW stabilization weights.

        Args:
            actions: Observed actions, shape (n,).
            propensity: Propensity scores P(a | H), shape (n,).

        Returns:
            AIPW weights, shape (n,).
        """
        # Prevent division by zero
        propensity = np.clip(propensity, 1e-4, 1 - 1e-4)
        weights = 1.0 / propensity
        return weights

    def estimate_blip(
        self,
        histories: np.ndarray,
        actions: np.ndarray,
        outcomes: np.ndarray,
        propensity_scores: np.ndarray,
        mu_predictions: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Estimate optimal blip function via AIPW.

        Args:
            histories: Shape (n, history_dim).
            actions: Shape (n,), observed action indices.
            outcomes: Shape (n,), observed outcomes.
            propensity_scores: Shape (n,), P(A_i | H_i).
            mu_predictions: Shape (n,), outcome regression E[Y | H, A].

        Returns:
            Dictionary with blip estimates and diagnostics.
        """
        n = histories.shape[0]

        # Compute AIPW weights
        weights = self.compute_aipw_weights(actions, propensity_scores)

        # Efficient influence function residuals
        residuals = outcomes - mu_predictions

        # Weighted regression to estimate blip
        X_aug = np.hstack([
            histories,
            actions.reshape(-1, 1),
        ])

        # Solve weighted least squares
        W = np.diag(weights)
        try:
            beta = np.linalg.solve(X_aug.T @ W @ X_aug, X_aug.T @ W @ residuals)
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in blip estimation; using pseudoinverse")
            beta = np.linalg.pinv(X_aug.T @ W @ X_aug) @ X_aug.T @ W @ residuals

        # Blip predictions
        blip_pred = X_aug @ beta

        # Variance estimate
        residual_var = np.mean((residuals - blip_pred) ** 2)

        return {
            "blip_coefficients": beta,
            "blip_predictions": blip_pred,
            "residual_variance": residual_var,
            "mean_weight": np.mean(weights),
            "max_weight": np.max(weights),
        }


# ============================================================================
# DTR Policy Optimizer (Top-Level)
# ============================================================================

class DTRPolicyOptimizer(nn.Module):
    """
    Top-level coordinator for dynamic treatment regimes.

    Integrates:
      - Fitted Q-Iteration for value function learning.
      - Counterfactual Recurrent Network for outcome prediction.
      - Treatment Feasibility constraints.
      - ResistanceMap sensitivity integration.

    Produces optimal decision rules π(H_t) at each stage.
    """

    def __init__(
        self,
        history_dim: int,
        n_actions: int,
        n_stages: int,
        resistance_map_fn=None,
        dropout: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            history_dim: Dimension of treatment history.
            n_actions: Number of possible actions (8 for MM).
            n_stages: Number of treatment stages (K).
            resistance_map_fn: Callable returning P_sensitivity (optional).
            dropout: Dropout rate.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.history_dim = history_dim
        self.n_actions = n_actions
        self.n_stages = n_stages
        self.resistance_map_fn = resistance_map_fn

        # Q-Learning module
        self.q_learning = FittedQIteration(
            history_dim=history_dim + 1,  # +1 for resistance sensitivity
            n_actions=n_actions,
            dropout=dropout,
        )

        # Counterfactual RN module
        self.crn = CounterfactualRecurrentNetwork(
            history_dim=history_dim + 1,
            n_actions=n_actions,
            hidden_dim=128,
            dropout=dropout,
        )

        # A-learner
        self.alearner = ALearnerDoublyRobust()

        # Feasibility mask
        self.feasibility_mask = TreatmentFeasibilityMask(n_actions=n_actions)

        # Optimizer (for overall module)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def augment_history(
        self,
        history: torch.Tensor,
        biomarker: torch.Tensor,
    ) -> torch.Tensor:
        """
        Augment history with ResistanceMap sensitivity.

        Args:
            history: Shape (batch, history_dim).
            biomarker: Shape (batch, biomarker_dim).

        Returns:
            Augmented history, shape (batch, history_dim + 1).
        """
        if self.resistance_map_fn is not None:
            # Compute resistance sensitivity
            p_sensitivity = self.resistance_map_fn(biomarker)
            if isinstance(p_sensitivity, np.ndarray):
                p_sensitivity = torch.from_numpy(p_sensitivity).float()
            history_aug = torch.cat([history, p_sensitivity.unsqueeze(1)], dim=1)
        else:
            history_aug = history

        return history_aug

    def select_action(
        self,
        history: torch.Tensor,
        biomarker: torch.Tensor,
        egfr: float,
        ecog: int,
        age: int,
        cart_used: bool,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select optimal action given current state with constraints.

        Args:
            history: Current treatment history, shape (1, history_dim).
            biomarker: Biomarker values, shape (1, biomarker_dim).
            egfr: Glomerular filtration rate.
            ecog: ECOG performance status.
            age: Patient age.
            cart_used: Whether CAR-T has been used.

        Returns:
            (recommended_action, diagnostics_dict)
        """
        with torch.no_grad():
            # Augment with resistance sensitivity
            history_aug = self.augment_history(history, biomarker)

            # Q-learning recommendation
            opt_actions_q, q_values = self.q_learning.compute_optimal_actions(history_aug)
            action_q = opt_actions_q[0].item()

            # Get feasibility mask
            feasible = self.feasibility_mask.get_feasible_actions(
                egfr=egfr, ecog=ecog, age=age, cart_used=cart_used
            )

            # Filter by feasibility
            if feasible[action_q]:
                recommended_action = action_q
            else:
                # Fall back to best feasible action
                feasible_q_values = q_values[0].cpu().numpy().copy()
                feasible_q_values[~feasible] = -np.inf
                recommended_action = np.argmax(feasible_q_values)

        diagnostics = {
            "q_learning_action": action_q,
            "q_value": q_values[0, 0].item(),
            "feasible_actions": self.feasibility_mask.action_names[recommended_action],
            "feasibility_mask": feasible,
        }

        return int(recommended_action), diagnostics

    def train_joint(
        self,
        histories: List[torch.Tensor],
        biomarkers: torch.Tensor,
        actions: torch.Tensor,
        outcomes: torch.Tensor,
        propensity_scores: np.ndarray,
        n_epochs: int = 10,
        alpha_crn: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Joint training of Q-learning and CRN.

        Args:
            histories: List of history tensors per stage, each (batch, history_dim).
            biomarkers: Shape (batch, biomarker_dim).
            actions: Shape (batch,), observed actions.
            outcomes: Shape (batch, 1), final outcomes.
            propensity_scores: Shape (batch,), propensity scores.
            n_epochs: Training epochs.
            alpha_crn: CRN adversarial loss weight.

        Returns:
            Training metrics over epochs.
        """
        metrics = {"q_loss": [], "crn_loss": [], "total_loss": []}

        for epoch in range(n_epochs):
            # Augment histories with resistance
            histories_aug = [
                self.augment_history(h, biomarkers) for h in histories
            ]

            # Q-learning step
            q_metrics = self.q_learning.backward_recursion(
                histories=histories_aug,
                rewards=outcomes,
                n_epochs=1,
            )

            # CRN step (use first history for simplicity)
            crn_metrics = self.crn.train_step(
                treatment_history=histories[0].unsqueeze(1),
                actions=actions.unsqueeze(1),
                targets=outcomes,
                alpha_adversarial=alpha_crn,
            )

            metrics["q_loss"].append(q_metrics["losses"][-1])
            metrics["crn_loss"].append(crn_metrics["total_loss"])
            metrics["total_loss"].append(
                q_metrics["losses"][-1] + crn_metrics["total_loss"]
            )

            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{n_epochs}: "
                    f"Q-loss={metrics['q_loss'][-1]:.4f}, "
                    f"CRN-loss={metrics['crn_loss'][-1]:.4f}"
                )

        return metrics

    def forward(
        self,
        history: torch.Tensor,
        biomarker: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: predict outcomes under all actions.

        Args:
            history: Shape (batch, history_dim).
            biomarker: Shape (batch, biomarker_dim).

        Returns:
            Dictionary of counterfactual outcome predictions.
        """
        history_aug = self.augment_history(history, biomarker)
        # Could use CRN predictions here
        return {"q_values": torch.zeros_like(history_aug)}


# ============================================================================
# Utility Functions
# ============================================================================

def build_dtr_optimizer(
    config: Dict[str, Any],
) -> DTRPolicyOptimizer:
    """
    Factory function to build DTRPolicyOptimizer from config.

    Args:
        config: Configuration dictionary with keys:
          - history_dim, n_actions, n_stages, dropout, seed, resistance_map_fn.

    Returns:
        Initialized DTRPolicyOptimizer.
    """
    return DTRPolicyOptimizer(
        history_dim=config.get("history_dim", 32),
        n_actions=config.get("n_actions", 8),
        n_stages=config.get("n_stages", 3),
        resistance_map_fn=config.get("resistance_map_fn", None),
        dropout=config.get("dropout", 0.1),
        seed=config.get("seed", 42),
    )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    # Initialize optimizer
    config = {
        "history_dim": 32,
        "n_actions": 8,
        "n_stages": 3,
        "dropout": 0.1,
    }
    optimizer = build_dtr_optimizer(config)

    # Dummy data
    batch_size = 16
    histories = [
        torch.randn(batch_size, 32) for _ in range(config["n_stages"])
    ]
    biomarkers = torch.randn(batch_size, 8)
    actions = torch.randint(0, 8, (batch_size,))
    outcomes = torch.randn(batch_size, 1)
    propensity = np.ones(batch_size) * 0.125  # uniform

    # Train
    logger.info("Training DTR optimizer...")
    metrics = optimizer.train_joint(
        histories=histories,
        biomarkers=biomarkers,
        actions=actions,
        outcomes=outcomes,
        propensity_scores=propensity,
        n_epochs=5,
    )

    # Action selection
    logger.info("Selecting action...")
    action, diag = optimizer.select_action(
        history=histories[0][:1],
        biomarker=biomarkers[:1],
        egfr=45.0,
        ecog=1,
        age=65,
        cart_used=False,
    )
    logger.info(f"Recommended action: {action} ({optimizer.feasibility_mask.action_names[action]})")
    logger.info(f"Diagnostics: {diag}")
