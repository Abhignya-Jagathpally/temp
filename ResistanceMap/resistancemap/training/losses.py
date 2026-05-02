#!/usr/bin/env python3
"""Loss functions for ResistanceMap v6.

This module implements all loss components used in the multi-task training
objective. Each loss is a standalone nn.Module that can be composed via
the AutomaticLossWeighting module, which learns uncertainty-based weights
following Kendall et al. (CVPR 2018).

Loss components:
    1. IdentifiableODELoss    — Trajectory MSE + identifiability regularization
    2. LyapunovStabilityLoss  — dV/dt < -epsilon near attractors
    3. DisentanglementLoss    — beta-TCVAE total correlation penalty
    4. PathwayAnchorLoss      — Tie latent dims to known biological pathways
    5. CausalConsistencyLoss  — Enforce DAG structure in causal fusion
    6. AutomaticLossWeighting — Kendall et al. 2018 homoscedastic uncertainty
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Identifiable Neural ODE Loss
# =============================================================================


class IdentifiableODELoss(nn.Module):
    """Combined loss for identifiable Neural ODE training.

    Combines trajectory matching with identifiability-inducing regularizers
    on the interaction matrix A that governs gene-gene dynamics.

    Components:
        - Trajectory MSE: match predicted vs observed states at clinical timepoints.
        - Sparsity (L1): encourage sparse gene interactions in A.
        - Diagonal dominance: penalty when off-diagonal |A_ij| > A_ii.
        - Orthogonality: penalty for non-orthogonal program embeddings.

    Args:
        sparsity_weight: Weight for L1 sparsity penalty on A.
        diagonal_dominance_weight: Weight for diagonal dominance penalty.
        orthogonality_weight: Weight for orthogonality penalty on programs.
        trajectory_weight: Weight for trajectory MSE loss.
    """

    def __init__(
        self,
        sparsity_weight: float = 0.01,
        diagonal_dominance_weight: float = 0.01,
        orthogonality_weight: float = 0.01,
        trajectory_weight: float = 1.0,
    ):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.diagonal_dominance_weight = diagonal_dominance_weight
        self.orthogonality_weight = orthogonality_weight
        self.trajectory_weight = trajectory_weight

    def trajectory_mse(
        self,
        pred_states: torch.Tensor,
        true_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MSE between predicted and observed trajectory states.

        Args:
            pred_states: Predicted states (batch, time, dim).
            true_states: Observed states (batch, time, dim).
            mask: Binary mask (batch, time) for missing observations.

        Returns:
            Scalar MSE loss.
        """
        diff = (pred_states - true_states) ** 2
        if mask is not None:
            # Expand mask to match feature dimension
            mask = mask.unsqueeze(-1).expand_as(diff)
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred_states.device)
            return (diff * mask).sum() / mask.sum()
        return diff.mean()

    def sparsity_l1(self, interaction_matrix: torch.Tensor) -> torch.Tensor:
        """L1 penalty on interaction matrix to encourage sparse gene interactions.

        Args:
            interaction_matrix: The A matrix (dim, dim) governing dx/dt = A*x + f(x).

        Returns:
            Scalar L1 norm of A.
        """
        return interaction_matrix.abs().mean()

    def diagonal_dominance_penalty(
        self, interaction_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Penalty for off-diagonal elements exceeding diagonal magnitude.

        Diagonal dominance ensures that self-regulation is stronger than
        cross-regulation, promoting identifiability of the dynamical system.

        Args:
            interaction_matrix: The A matrix (dim, dim).

        Returns:
            Scalar penalty (0 when diagonally dominant).
        """
        A = interaction_matrix
        diag = A.diag().abs()
        off_diag_sum = A.abs().sum(dim=1) - diag
        # Penalty when off-diagonal sum exceeds diagonal
        violations = F.relu(off_diag_sum - diag)
        return violations.mean()

    def orthogonality_penalty(
        self, program_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Penalty for non-orthogonal program embeddings.

        Encourages each biological program to capture independent variation.

        Args:
            program_embeddings: Matrix (num_programs, embed_dim).

        Returns:
            Scalar penalty measuring deviation from orthogonality.
        """
        # Normalize rows to unit length
        P = F.normalize(program_embeddings, dim=-1)
        # Gram matrix — identity for orthogonal rows
        gram = P @ P.T
        identity = torch.eye(gram.size(0), device=gram.device)
        return (gram - identity).pow(2).mean()

    def forward(
        self,
        pred_states: torch.Tensor,
        true_states: torch.Tensor,
        interaction_matrix: torch.Tensor,
        program_embeddings: Optional[torch.Tensor] = None,
        observation_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all identifiable ODE loss components.

        Args:
            pred_states: Predicted trajectory (batch, time, dim).
            true_states: Observed trajectory (batch, time, dim).
            interaction_matrix: Gene interaction matrix A (dim, dim).
            program_embeddings: Program matrix (num_programs, embed_dim).
            observation_mask: Binary mask (batch, time).

        Returns:
            Dict with keys 'total', 'trajectory_mse', 'sparsity_l1',
            'diagonal_dominance', 'orthogonality'.
        """
        losses = {}

        losses["trajectory_mse"] = self.trajectory_weight * self.trajectory_mse(
            pred_states, true_states, observation_mask
        )
        losses["sparsity_l1"] = self.sparsity_weight * self.sparsity_l1(
            interaction_matrix
        )
        losses["diagonal_dominance"] = (
            self.diagonal_dominance_weight
            * self.diagonal_dominance_penalty(interaction_matrix)
        )

        if program_embeddings is not None:
            losses["orthogonality"] = (
                self.orthogonality_weight
                * self.orthogonality_penalty(program_embeddings)
            )
        else:
            losses["orthogonality"] = torch.tensor(
                0.0, device=pred_states.device
            )

        losses["total"] = sum(losses.values())
        return losses


# =============================================================================
# 2. Lyapunov Stability Loss
# =============================================================================


class LyapunovStabilityLoss(nn.Module):
    """Enforce Lyapunov stability near learned attractors.

    For a learned Lyapunov function V(x) and dynamics f(x), we require:
        1. V(x*) = 0 at attractors x*
        2. V(x) > 0 for x != x*
        3. dV/dt = grad(V) . f(x) < -epsilon in a neighborhood of x*

    This encourages the Neural ODE to learn stable fixed points corresponding
    to drug-resistant and drug-sensitive cell states.

    Args:
        epsilon: Minimum decrease rate for dV/dt.
        neighborhood_radius: Radius around attractor for sampling.
        num_samples: Number of points to sample per attractor.
        attractor_penalty_weight: Weight for V(x*) = 0 constraint.
        decrease_penalty_weight: Weight for dV/dt < -epsilon constraint.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        neighborhood_radius: float = 1.0,
        num_samples: int = 64,
        attractor_penalty_weight: float = 1.0,
        decrease_penalty_weight: float = 1.0,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.neighborhood_radius = neighborhood_radius
        self.num_samples = num_samples
        self.attractor_penalty_weight = attractor_penalty_weight
        self.decrease_penalty_weight = decrease_penalty_weight

    def forward(
        self,
        lyapunov_fn: nn.Module,
        dynamics_fn: nn.Module,
        attractors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute Lyapunov stability loss.

        Args:
            lyapunov_fn: Network V: R^d -> R^1 (the Lyapunov candidate).
            dynamics_fn: Network f: R^d -> R^d (the ODE right-hand side).
            attractors: Tensor (num_attractors, dim) of attractor locations.

        Returns:
            Dict with 'total', 'attractor_value', 'decrease_violation'.
        """
        device = attractors.device
        dim = attractors.shape[-1]

        # 1. V(x*) should be zero at attractors
        V_at_attractors = lyapunov_fn(attractors)
        attractor_loss = V_at_attractors.pow(2).mean()

        # 2. Sample points near attractors and check dV/dt < -epsilon
        decrease_violations = []
        for attractor in attractors:
            # Sample uniformly in ball around attractor
            noise = torch.randn(
                self.num_samples, dim, device=device
            )
            noise = F.normalize(noise, dim=-1)
            radii = (
                torch.rand(self.num_samples, 1, device=device)
                * self.neighborhood_radius
            )
            x_samples = attractor.unsqueeze(0) + noise * radii
            x_samples.requires_grad_(True)

            # Compute V(x) and its gradient
            V_x = lyapunov_fn(x_samples)
            grad_V = torch.autograd.grad(
                V_x.sum(),
                x_samples,
                create_graph=True,
                retain_graph=True,
            )[0]

            # Compute f(x) = dynamics at sampled points
            f_x = dynamics_fn(x_samples)

            # dV/dt = grad(V) . f(x) — should be < -epsilon
            dVdt = (grad_V * f_x).sum(dim=-1)
            violation = F.relu(dVdt + self.epsilon)
            decrease_violations.append(violation.mean())

        decrease_loss = torch.stack(decrease_violations).mean()

        losses = {
            "attractor_value": self.attractor_penalty_weight * attractor_loss,
            "decrease_violation": self.decrease_penalty_weight * decrease_loss,
        }
        losses["total"] = sum(losses.values())
        return losses


# =============================================================================
# 3. Disentanglement Loss (beta-TCVAE)
# =============================================================================


class DisentanglementLoss(nn.Module):
    """beta-TCVAE total correlation penalty for disentangled representations.

    Decomposes the KL divergence of a VAE into three terms:
        KL = Index-Code MI + Total Correlation + Dimension-wise KL

    Only the Total Correlation (TC) term is upweighted by beta, encouraging
    the aggregate posterior to factorize (i.e., latent dimensions are
    statistically independent).

    Reference: Chen et al., "Isolating Sources of Disentanglement in
    Variational Autoencoders", NeurIPS 2018.

    Args:
        beta: Weight on the TC term (>1 for stronger disentanglement).
        anneal_steps: Number of steps to linearly anneal beta from 0 to target.
    """

    def __init__(self, beta: float = 4.0, anneal_steps: int = 10000):
        super().__init__()
        self.beta = beta
        self.anneal_steps = anneal_steps
        self._step = 0

    @property
    def current_beta(self) -> float:
        """Get current beta value accounting for annealing."""
        if self.anneal_steps > 0:
            return self.beta * min(1.0, self._step / self.anneal_steps)
        return self.beta

    def step(self):
        """Advance the annealing schedule by one step."""
        self._step += 1

    def _log_density_gaussian(
        self, sample: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Compute log N(sample | mu, sigma^2).

        Args:
            sample: Samples from the posterior (batch, dim).
            mu: Mean of the Gaussian (batch, dim).
            logvar: Log variance of the Gaussian (batch, dim).

        Returns:
            Log density (batch, batch, dim) — pairwise evaluation.
        """
        norm = -0.5 * (math.log(2.0 * math.pi) + logvar)
        log_density = norm - 0.5 * ((sample.unsqueeze(1) - mu.unsqueeze(0)) ** 2) / logvar.exp()
        return log_density

    def forward(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute beta-TCVAE loss components.

        Uses minibatch-weighted sampling to estimate the TC term.

        Args:
            z: Sampled latent codes (batch, dim).
            mu: Encoder means (batch, dim).
            logvar: Encoder log-variances (batch, dim).

        Returns:
            Dict with 'total', 'reconstruction' (0, handled elsewhere),
            'mi', 'tc', 'dwkl'.
        """
        batch_size = z.size(0)
        dim = z.size(1)

        # log q(z|x) for each sample under its own posterior
        log_qz_condx = self._log_density_gaussian(z, mu, logvar).sum(dim=-1)

        # log q(z) — marginal via minibatch-weighted sampling
        # log q(z_j) for all pairs, then logsumexp over batch dimension
        log_qz_pairwise = self._log_density_gaussian(z, mu, logvar)

        # log q(z) = logsumexp_j log q(z|x_j) - log(N)
        log_qz = (
            torch.logsumexp(log_qz_pairwise.sum(dim=-1), dim=1)
            - math.log(batch_size)
        )

        # log prod_d q(z_d) = sum_d logsumexp_j log q(z_d|x_j) - log(N)
        log_prod_qzd = (
            torch.logsumexp(log_qz_pairwise, dim=1) - math.log(batch_size)
        ).sum(dim=-1)

        # log p(z) — standard normal prior
        log_pz = (
            -0.5 * (math.log(2.0 * math.pi) + z.pow(2))
        ).sum(dim=-1)

        # Three decomposed terms (all per-sample, then averaged)
        # MI: E[log q(z|x) - log q(z)]
        mi = (log_qz_condx.diag() - log_qz).mean()
        # TC: E[log q(z) - log prod_d q(z_d)]
        tc = (log_qz - log_prod_qzd).mean()
        # Dimension-wise KL: E[log prod_d q(z_d) - log p(z)]
        dwkl = (log_prod_qzd - log_pz).mean()

        beta_eff = self.current_beta

        losses = {
            "mi": mi,
            "tc": tc,
            "dwkl": dwkl,
            "total": mi + beta_eff * tc + dwkl,
        }
        return losses


# =============================================================================
# 4. Pathway Anchor Loss
# =============================================================================


class PathwayAnchorLoss(nn.Module):
    """Auxiliary losses tying latent dimensions to known biological pathways.

    Each latent dimension is assigned to a pathway (e.g., NF-kB, PI3K/AKT,
    apoptosis). The anchor loss encourages the latent dimension to correlate
    with the mean expression of pathway member genes.

    This provides biological interpretability and regularization, preventing
    the latent space from learning purely statistical factors.

    Args:
        pathway_gene_sets: Dict mapping pathway name -> list of gene indices.
        num_programs: Number of latent dimensions (programs).
        assignment: Optional explicit assignment of program -> pathway.
            If None, assignment is learned.
        correlation_weight: Weight for the Pearson correlation loss.
        specificity_weight: Weight for cross-pathway specificity penalty.
    """

    def __init__(
        self,
        pathway_gene_sets: dict[str, list[int]],
        num_programs: int,
        assignment: Optional[dict[int, str]] = None,
        correlation_weight: float = 1.0,
        specificity_weight: float = 0.1,
    ):
        super().__init__()
        self.pathway_gene_sets = pathway_gene_sets
        self.num_programs = num_programs
        self.correlation_weight = correlation_weight
        self.specificity_weight = specificity_weight

        # Build assignment: program index -> pathway name
        if assignment is not None:
            self.assignment = assignment
        else:
            # Default: assign in order, cycling if more programs than pathways
            pathway_names = list(pathway_gene_sets.keys())
            self.assignment = {
                i: pathway_names[i % len(pathway_names)]
                for i in range(num_programs)
            }

    def _pearson_correlation(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute Pearson correlation between two tensors.

        Args:
            x: First tensor (batch,).
            y: Second tensor (batch,).

        Returns:
            Scalar Pearson correlation.
        """
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        cov = (x_centered * y_centered).mean()
        std_prod = x_centered.std() * y_centered.std() + 1e-8
        return cov / std_prod

    def forward(
        self,
        latent: torch.Tensor,
        gene_expression: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute pathway anchor loss.

        Args:
            latent: Latent representation (batch, num_programs).
            gene_expression: Gene expression matrix (batch, num_genes).

        Returns:
            Dict with 'total', 'correlation', 'specificity'.
        """
        correlation_losses = []
        specificity_losses = []

        for prog_idx, pathway_name in self.assignment.items():
            if prog_idx >= latent.size(1):
                continue
            if pathway_name not in self.pathway_gene_sets:
                continue

            gene_indices = self.pathway_gene_sets[pathway_name]
            if not gene_indices or max(gene_indices) >= gene_expression.size(1):
                continue

            # Mean expression of pathway genes
            pathway_expr = gene_expression[:, gene_indices].mean(dim=1)
            # Latent dimension value
            latent_dim = latent[:, prog_idx]

            # Maximize absolute correlation (minimize negative correlation)
            corr = self._pearson_correlation(latent_dim, pathway_expr)
            correlation_losses.append(1.0 - corr.abs())

            # Specificity: this latent dim should NOT correlate with other pathways
            for other_name, other_genes in self.pathway_gene_sets.items():
                if other_name == pathway_name or not other_genes:
                    continue
                valid_genes = [g for g in other_genes if g < gene_expression.size(1)]
                if not valid_genes:
                    continue
                other_expr = gene_expression[:, valid_genes].mean(dim=1)
                other_corr = self._pearson_correlation(latent_dim, other_expr)
                specificity_losses.append(other_corr.abs())

        device = latent.device

        if correlation_losses:
            corr_loss = torch.stack(correlation_losses).mean()
        else:
            corr_loss = torch.tensor(0.0, device=device)

        if specificity_losses:
            spec_loss = torch.stack(specificity_losses).mean()
        else:
            spec_loss = torch.tensor(0.0, device=device)

        losses = {
            "correlation": self.correlation_weight * corr_loss,
            "specificity": self.specificity_weight * spec_loss,
        }
        losses["total"] = sum(losses.values())
        return losses


# =============================================================================
# 5. Causal Consistency Loss
# =============================================================================


class CausalConsistencyLoss(nn.Module):
    """Ensure causal fusion respects the learned DAG structure.

    Two components:
        1. Acyclicity penalty: h(W) = tr(e^{W circ W}) - d = 0 for DAGs
           (NOTEARS, Zheng et al. 2018).
        2. Intervention consistency: under do(X_i), downstream effects must
           follow causal ordering.

    Args:
        acyclicity_weight: Weight for the DAG acyclicity penalty.
        intervention_weight: Weight for intervention consistency.
        dag_dim: Dimension of the adjacency matrix.
    """

    def __init__(
        self,
        acyclicity_weight: float = 1.0,
        intervention_weight: float = 0.5,
        dag_dim: int = 4,
    ):
        super().__init__()
        self.acyclicity_weight = acyclicity_weight
        self.intervention_weight = intervention_weight
        self.dag_dim = dag_dim

    def acyclicity_penalty(self, W: torch.Tensor) -> torch.Tensor:
        """NOTEARS acyclicity constraint: h(W) = tr(e^{W*W}) - d.

        For a DAG, this should be exactly zero. We minimize it as a penalty.

        Args:
            W: Weighted adjacency matrix (d, d).

        Returns:
            Scalar acyclicity violation.
        """
        d = W.size(0)
        # W_sq = W element-wise squared
        W_sq = W * W
        # Matrix exponential via eigendecomposition for stability
        # Use truncated power series for efficiency: sum_{k=0}^{d} (W_sq)^k / k!
        eye = torch.eye(d, device=W.device)
        expm = eye.clone()
        Wk = eye.clone()
        for k in range(1, d + 1):
            Wk = Wk @ W_sq / k
            expm = expm + Wk
        h = torch.trace(expm) - d
        return h

    def intervention_consistency(
        self,
        W: torch.Tensor,
        attention_weights: torch.Tensor,
        modality_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Check that attention weights respect causal ordering in W.

        If W[i,j] > 0 (i causes j), then the attention from j to i should
        be higher than from i to j (information flows from cause to effect).

        Args:
            W: Weighted adjacency (d, d).
            attention_weights: Cross-attention weights (batch, d, d).
            modality_outputs: List of modality tensors.

        Returns:
            Scalar consistency violation.
        """
        # For each causal edge i->j, attention[j,i] should dominate attention[i,j]
        d = W.size(0)
        violations = []
        for i in range(d):
            for j in range(d):
                if i != j and W[i, j].abs() > 0.01:
                    # Edge i->j exists; attention from j to i should be high
                    attn_ji = attention_weights[:, j, i].mean()
                    attn_ij = attention_weights[:, i, j].mean()
                    # Violation if attention goes against causal direction
                    violations.append(F.relu(attn_ij - attn_ji))

        if violations:
            return torch.stack(violations).mean()
        return torch.tensor(0.0, device=W.device)

    def forward(
        self,
        adjacency_matrix: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        modality_outputs: Optional[list[torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute causal consistency loss.

        Args:
            adjacency_matrix: Learned weighted adjacency (d, d).
            attention_weights: Cross-attention weights (batch, d, d).
            modality_outputs: List of modality output tensors.

        Returns:
            Dict with 'total', 'acyclicity', 'intervention_consistency'.
        """
        losses = {}

        losses["acyclicity"] = self.acyclicity_weight * self.acyclicity_penalty(
            adjacency_matrix
        )

        if attention_weights is not None and modality_outputs is not None:
            losses["intervention_consistency"] = (
                self.intervention_weight
                * self.intervention_consistency(
                    adjacency_matrix, attention_weights, modality_outputs
                )
            )
        else:
            losses["intervention_consistency"] = torch.tensor(
                0.0, device=adjacency_matrix.device
            )

        losses["total"] = sum(losses.values())
        return losses


# =============================================================================
# 6. Automatic Loss Weighting (Kendall et al. 2018)
# =============================================================================


class AutomaticLossWeighting(nn.Module):
    """Learn task-specific uncertainty weights for multi-task loss.

    Following Kendall, Gal & Cipolla (CVPR 2018), each task loss L_i is
    weighted by a learned homoscedastic uncertainty sigma_i:

        total = sum_i (1 / (2 * sigma_i^2)) * L_i + log(sigma_i)

    We parameterize log(sigma_i^2) to ensure positivity and numerical stability.

    Args:
        num_tasks: Number of loss components to weight.
        task_names: Optional list of names for logging.
        initial_weights: Optional initial log-sigma^2 values. Defaults to 0
            (sigma=1, equal weighting).
    """

    def __init__(
        self,
        num_tasks: int,
        task_names: Optional[list[str]] = None,
        initial_weights: Optional[list[float]] = None,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names or [f"task_{i}" for i in range(num_tasks)]

        # log(sigma^2) parameters — learnable
        if initial_weights is not None:
            init = torch.tensor(initial_weights, dtype=torch.float32)
        else:
            init = torch.zeros(num_tasks)
        self.log_sigma_sq = nn.Parameter(init)

    def forward(
        self, losses: dict[str, torch.Tensor] | list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute uncertainty-weighted total loss.

        Args:
            losses: Either a dict {task_name: loss_tensor} or a list of
                loss tensors in the same order as task_names.

        Returns:
            Dict with 'total' and per-task weighted losses, plus 'weights'
            containing the effective weights (1 / sigma^2).
        """
        if isinstance(losses, dict):
            loss_list = [losses[name] for name in self.task_names]
        else:
            loss_list = losses

        assert len(loss_list) == self.num_tasks, (
            f"Expected {self.num_tasks} losses, got {len(loss_list)}"
        )

        total = torch.tensor(0.0, device=self.log_sigma_sq.device)
        result = {}

        for i, (name, loss) in enumerate(zip(self.task_names, loss_list)):
            # precision = 1 / sigma^2 = exp(-log_sigma_sq)
            precision = torch.exp(-self.log_sigma_sq[i])
            weighted = precision * loss + self.log_sigma_sq[i]
            result[f"weighted_{name}"] = weighted
            total = total + weighted

        result["total"] = total

        # Report effective weights for logging
        with torch.no_grad():
            effective_weights = torch.exp(-self.log_sigma_sq)
            result["effective_weights"] = {
                name: float(effective_weights[i])
                for i, name in enumerate(self.task_names)
            }

        return result

    def get_weights_summary(self) -> dict[str, float]:
        """Get current effective weights for logging.

        Returns:
            Dict mapping task name to effective weight.
        """
        with torch.no_grad():
            weights = torch.exp(-self.log_sigma_sq)
            return {
                name: float(weights[i])
                for i, name in enumerate(self.task_names)
            }


# =============================================================================
# 7. Composite Loss (combines all components)
# =============================================================================


class ResistanceMapLoss(nn.Module):
    """Full composite loss for ResistanceMap v6 training.

    Combines all loss components with either fixed or learned weights.

    Args:
        config: Training configuration dict containing loss_weights and flags.
        pathway_gene_sets: Dict of pathway -> gene indices for anchor loss.
        num_programs: Number of latent programs.
        auto_weight: Whether to use automatic loss weighting.
    """

    TASK_NAMES = [
        "drug_response",
        "trajectory_mse",
        "reconstruction",
        "lyapunov_stability",
        "tc_penalty",
        "sparsity_l1",
        "pathway_anchor",
        "causal_consistency",
    ]

    def __init__(
        self,
        config: dict,
        pathway_gene_sets: Optional[dict[str, list[int]]] = None,
        num_programs: int = 8,
        auto_weight: bool = True,
    ):
        super().__init__()

        loss_cfg = config.get("loss_weights", {})

        # Individual loss modules
        self.ode_loss = IdentifiableODELoss(
            sparsity_weight=loss_cfg.get("sparsity_l1", 0.01),
            diagonal_dominance_weight=loss_cfg.get("diagonal_dominance", 0.01),
            orthogonality_weight=loss_cfg.get("orthogonality", 0.01),
        )
        self.lyapunov_loss = LyapunovStabilityLoss()
        self.disentanglement_loss = DisentanglementLoss(
            beta=config.get("model", {}).get("vae", {}).get("beta", 4.0),
        )
        if pathway_gene_sets:
            self.pathway_loss = PathwayAnchorLoss(
                pathway_gene_sets=pathway_gene_sets,
                num_programs=num_programs,
            )
        else:
            self.pathway_loss = None

        self.causal_loss = CausalConsistencyLoss()

        # Drug response loss
        self.drug_response_bce = nn.BCEWithLogitsLoss()

        # Reconstruction loss
        self.reconstruction_mse = nn.MSELoss()

        # Fixed weights (used as fallback or initial scale)
        self.fixed_weights = {
            name: loss_cfg.get(name, 1.0) for name in self.TASK_NAMES
        }

        # Automatic weighting
        self.auto_weight = auto_weight
        if auto_weight:
            self.auto_weighter = AutomaticLossWeighting(
                num_tasks=len(self.TASK_NAMES),
                task_names=self.TASK_NAMES,
            )
        else:
            self.auto_weighter = None

    def forward(
        self,
        # Drug response
        drug_logits: torch.Tensor,
        drug_labels: torch.Tensor,
        # Trajectory
        pred_states: torch.Tensor,
        true_states: torch.Tensor,
        observation_mask: Optional[torch.Tensor] = None,
        # ODE
        interaction_matrix: Optional[torch.Tensor] = None,
        program_embeddings: Optional[torch.Tensor] = None,
        # VAE
        z: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        recon_x: Optional[torch.Tensor] = None,
        target_x: Optional[torch.Tensor] = None,
        # Lyapunov
        lyapunov_fn: Optional[nn.Module] = None,
        dynamics_fn: Optional[nn.Module] = None,
        attractors: Optional[torch.Tensor] = None,
        # Pathway
        gene_expression: Optional[torch.Tensor] = None,
        # Causal
        adjacency_matrix: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
        modality_outputs: Optional[list[torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the full composite loss.

        Returns:
            Dict with 'total' and all individual loss components.
        """
        device = drug_logits.device
        losses = {}

        # 1. Drug response BCE
        losses["drug_response"] = self.drug_response_bce(
            drug_logits, drug_labels.float()
        )

        # 2. Trajectory + identifiability
        if interaction_matrix is not None:
            ode_losses = self.ode_loss(
                pred_states, true_states, interaction_matrix,
                program_embeddings, observation_mask,
            )
            losses["trajectory_mse"] = ode_losses["trajectory_mse"]
            losses["sparsity_l1"] = ode_losses["sparsity_l1"]
        else:
            losses["trajectory_mse"] = F.mse_loss(pred_states, true_states)
            losses["sparsity_l1"] = torch.tensor(0.0, device=device)

        # 3. Reconstruction
        if recon_x is not None and target_x is not None:
            losses["reconstruction"] = self.reconstruction_mse(recon_x, target_x)
        else:
            losses["reconstruction"] = torch.tensor(0.0, device=device)

        # 4. Lyapunov stability
        if (
            lyapunov_fn is not None
            and dynamics_fn is not None
            and attractors is not None
        ):
            lyap_losses = self.lyapunov_loss(
                lyapunov_fn, dynamics_fn, attractors
            )
            losses["lyapunov_stability"] = lyap_losses["total"]
        else:
            losses["lyapunov_stability"] = torch.tensor(0.0, device=device)

        # 5. Disentanglement (TC penalty)
        if z is not None and mu is not None and logvar is not None:
            tc_losses = self.disentanglement_loss(z, mu, logvar)
            losses["tc_penalty"] = tc_losses["tc"]
        else:
            losses["tc_penalty"] = torch.tensor(0.0, device=device)

        # 6. Pathway anchor
        if (
            self.pathway_loss is not None
            and z is not None
            and gene_expression is not None
        ):
            pw_losses = self.pathway_loss(z, gene_expression)
            losses["pathway_anchor"] = pw_losses["total"]
        else:
            losses["pathway_anchor"] = torch.tensor(0.0, device=device)

        # 7. Causal consistency
        if adjacency_matrix is not None:
            causal_losses = self.causal_loss(
                adjacency_matrix, attention_weights, modality_outputs
            )
            losses["causal_consistency"] = causal_losses["total"]
        else:
            losses["causal_consistency"] = torch.tensor(0.0, device=device)

        # Combine losses
        if self.auto_weighter is not None:
            task_losses = [losses[name] for name in self.TASK_NAMES]
            weighted = self.auto_weighter(task_losses)
            losses["total"] = weighted["total"]
            losses["effective_weights"] = weighted.get("effective_weights", {})
        else:
            total = torch.tensor(0.0, device=device)
            for name in self.TASK_NAMES:
                total = total + self.fixed_weights.get(name, 1.0) * losses[name]
            losses["total"] = total

        return losses
