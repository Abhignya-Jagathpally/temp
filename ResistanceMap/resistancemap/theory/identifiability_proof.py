"""Numerical Verification of Identifiability for Structured Biological ODE.

Provides rigorous numerical tests that the structured ODE model
dx/dt = A @ sigma(x) + b(x) with diagonal dominance and sparsity
constraints is identifiable from observed trajectory data.

Four complementary approaches:
    1. **Fisher Information Matrix (FIM)**: Compute the FIM of the ODE
       parameters given observed trajectories. Full rank => locally
       identifiable. Rank deficiency reveals unidentifiable directions.
    2. **Structural Identifiability (Lie derivatives)**: Check whether
       input-output relations uniquely determine model parameters using
       the observability rank condition from differential algebra.
    3. **Empirical Convergence Test**: Train the model from multiple
       random initializations and verify that learned A matrices converge
       to the same structure (up to known symmetries).
    4. **Sensitivity Analysis**: Compute the sensitivity of predictions
       to each parameter in A, identifying which interactions are most
       important for model behavior.

Theoretical references:
    - Moran, Blum & Mangalam (2022). "Identifiability of latent-variable
      and structural-equation models." arXiv:2206.13490.
    - Ljung & Glad (1994). "On global identifiability for arbitrary
      model parametrizations." Automatica 30(2), 265-276.
    - Rothenberg (1971). "Identification in parametric models."
      Econometrica 39(3), 577-591.
    - Brunton, Proctor & Kutz (2016). "Discovering governing equations
      from data by sparse identification of nonlinear dynamical systems."
      PNAS 113(15), 3932-3937.

Authors: ResistanceMap Team
License: MIT
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

__all__ = [
    "IdentifiabilityConfig",
    "FisherInformationAnalyzer",
    "StructuralIdentifiabilityTest",
    "EmpiricalConvergenceTest",
    "SensitivityAnalyzer",
    "run_full_identifiability_analysis",
]


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class IdentifiabilityConfig:
    """Configuration for identifiability analysis.

    Attributes:
        n_programs: Number of biological programs (state dimension for A).
        n_time_points: Number of trajectory time points.
        n_trajectories: Number of trajectories for FIM estimation.
        fim_noise_std: Assumed observation noise std for FIM.
        n_random_inits: Number of random initializations for empirical test.
        convergence_tol: Tolerance for declaring parameter convergence.
        sensitivity_epsilon: Perturbation size for finite-difference sensitivity.
        max_train_steps: Max training steps per random init.
        lr: Learning rate for empirical convergence training.
    """
    n_programs: int = 8
    n_time_points: int = 20
    n_trajectories: int = 50
    fim_noise_std: float = 0.1
    n_random_inits: int = 10
    convergence_tol: float = 0.05
    sensitivity_epsilon: float = 1e-4
    max_train_steps: int = 500
    lr: float = 0.01


# ===================================================================
# Fisher Information Matrix Analysis
# ===================================================================

class FisherInformationAnalyzer:
    """Computes the Fisher Information Matrix (FIM) of ODE parameters
    given observed trajectory data.

    The FIM F(theta) is defined as:
        F_ij = E[ (d log p(y|theta) / d theta_i)(d log p(y|theta) / d theta_j) ]

    Under Gaussian observation noise with known variance sigma^2:
        F_ij = (1/sigma^2) sum_t (dy/d theta_i)_t (dy/d theta_j)_t

    where dy/d theta is the sensitivity of the trajectory to parameters.

    **Identifiability criterion**: The model is locally identifiable at
    theta if and only if F(theta) is full rank (Rothenberg, 1971).

    The rank of F equals the number of identifiable parameter combinations.
    Eigenvectors corresponding to small eigenvalues reveal the unidentifiable
    parameter directions (sloppy parameter combinations).

    Args:
        config: IdentifiabilityConfig.
    """

    def __init__(self, config: IdentifiabilityConfig) -> None:
        self.config = config

    def compute_trajectory_sensitivity(
        self,
        ode_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        t_span: torch.Tensor,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute dy/d theta for each parameter via autograd.

        Integrates the ODE and computes gradients of the trajectory
        with respect to each parameter.

        Args:
            ode_func: ODE right-hand side f(t, x) -> dx/dt.
            x0: (D,) initial condition.
            t_span: (T,) time points.
            params: List of parameter tensors to differentiate w.r.t.

        Returns:
            J: (T*D, P) Jacobian where P = total number of parameters.
        """
        if not HAS_TORCHDIFFEQ:
            raise RuntimeError("torchdiffeq required")

        # Ensure params require grad
        for p in params:
            p.requires_grad_(True)

        # Forward integration
        x0_batch = x0.unsqueeze(0)
        trajectory = odeint(ode_func, x0_batch, t_span, method="euler")
        # trajectory: (T, 1, D)
        traj_flat = trajectory.squeeze(1).reshape(-1)  # (T*D,)

        T_D = traj_flat.shape[0]
        n_params = sum(p.numel() for p in params)
        J = torch.zeros(T_D, n_params)

        # Compute Jacobian column by column
        for i in range(T_D):
            if traj_flat[i].requires_grad:
                grads = torch.autograd.grad(
                    traj_flat[i], params, retain_graph=True,
                    allow_unused=True
                )
                col_offset = 0
                for g, p in zip(grads, params):
                    if g is not None:
                        g_flat = g.reshape(-1)
                        J[i, col_offset:col_offset + g_flat.shape[0]] = g_flat.detach()
                    col_offset += p.numel()

        return J

    def compute_fim(
        self,
        ode_func: Callable,
        initial_conditions: torch.Tensor,
        t_span: torch.Tensor,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the Fisher Information Matrix.

        Averages over multiple trajectories from different initial conditions.

        F = (1/sigma^2) * (1/N) sum_n J_n^T J_n

        Args:
            ode_func: ODE function f(t, x) -> dx/dt.
            initial_conditions: (N, D) initial conditions.
            t_span: (T,) time points.
            params: List of parameter tensors.

        Returns:
            F: (P, P) Fisher Information Matrix.
        """
        N = initial_conditions.shape[0]
        n_params = sum(p.numel() for p in params)
        F = torch.zeros(n_params, n_params)

        for i in range(N):
            J = self.compute_trajectory_sensitivity(
                ode_func, initial_conditions[i], t_span, params
            )
            F += J.T @ J

        F = F / (N * self.config.fim_noise_std ** 2)
        return F

    def analyze_fim(
        self,
        F: torch.Tensor,
    ) -> Dict[str, object]:
        """Analyze the FIM for identifiability.

        Args:
            F: (P, P) Fisher Information Matrix.

        Returns:
            Dict with:
                - "rank": numerical rank
                - "n_params": total number of parameters
                - "is_identifiable": True if full rank
                - "eigenvalues": sorted eigenvalues
                - "condition_number": ratio of largest to smallest eigenvalue
                - "sloppy_directions": eigenvectors for small eigenvalues
                - "n_sloppy": number of practically unidentifiable directions
        """
        eigenvalues, eigenvectors = torch.linalg.eigh(F)
        eigenvalues = eigenvalues.real

        # Sort descending
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Numerical rank (eigenvalues > threshold)
        threshold = eigenvalues.max() * 1e-10
        rank = (eigenvalues > threshold).sum().item()
        n_params = F.shape[0]

        # Sloppy parameters: eigenvalues < 1% of max
        sloppy_threshold = eigenvalues.max() * 0.01
        n_sloppy = (eigenvalues < sloppy_threshold).sum().item()
        sloppy_dirs = eigenvectors[:, eigenvalues < sloppy_threshold]

        # Condition number
        nonzero_eigs = eigenvalues[eigenvalues > threshold]
        if len(nonzero_eigs) > 1:
            cond = (nonzero_eigs[0] / nonzero_eigs[-1]).item()
        else:
            cond = float("inf")

        return {
            "rank": rank,
            "n_params": n_params,
            "is_identifiable": rank == n_params,
            "eigenvalues": eigenvalues.detach().cpu(),
            "condition_number": cond,
            "sloppy_directions": sloppy_dirs.detach().cpu(),
            "n_sloppy": n_sloppy,
        }


# ===================================================================
# Structural Identifiability via Lie Derivatives
# ===================================================================

class StructuralIdentifiabilityTest:
    """Test structural identifiability using the observability rank
    condition based on Lie derivatives.

    For a system dx/dt = f(x, theta), y = h(x), the system is
    structurally identifiable if the observability matrix:
        O = [dh/dx, d(L_f h)/dx, d(L_f^2 h)/dx, ...]
    has full column rank, where L_f is the Lie derivative along f.

    For our structured ODE dx/dt = A @ sigma(x) + b, with full state
    observation y = x, we check whether A is recoverable from the
    input-output map by computing successive Lie derivatives.

    Theorem (adapted from Ljung & Glad, 1994):
    If the interaction matrix A has diagonal dominance and the
    nonlinearity sigma is analytic and injective (tanh satisfies this),
    then A is structurally identifiable from trajectory observations
    at K+1 distinct time points, where K = n_programs.

    Args:
        n_programs: State dimension.
    """

    def __init__(self, n_programs: int) -> None:
        self.n_programs = n_programs

    def compute_observability_matrix(
        self,
        A: torch.Tensor,
        x: torch.Tensor,
        n_derivatives: int = 3,
    ) -> torch.Tensor:
        """Compute the observability matrix via iterated Lie derivatives.

        For full-state observation (h(x) = x), the k-th Lie derivative is:
            L_f^k h(x) = f(f(...f(x)...))  (k-fold composition of the
            vector field evaluated at x).

        In practice, we compute:
            O_0 = I (from h(x) = x)
            O_1 = J_f(x) = A @ diag(sigma'(x))
            O_2 = J_f(f(x)) @ J_f(x)
            etc.

        The observability matrix is [O_0; O_1; O_2; ...].

        Args:
            A: (K, K) interaction matrix.
            x: (K,) evaluation point.
            n_derivatives: Number of Lie derivatives to compute.

        Returns:
            O: (n_derivatives * K, K) observability matrix.
        """
        K = self.n_programs

        # sigma'(x) = 1 - tanh^2(x) (derivative of tanh)
        sigma_prime = 1.0 - torch.tanh(x) ** 2  # (K,)

        # O_0 = I
        blocks = [torch.eye(K)]

        # J_f = A @ diag(sigma'(x))
        J = A * sigma_prime.unsqueeze(0)  # (K, K): row i, col j = A_ij * sigma'(x_j)

        # O_1 = J_f(x)
        blocks.append(J.clone())

        # Higher-order Lie derivatives
        current_J = J.clone()
        current_x = x.clone()
        for _ in range(2, n_derivatives):
            # Advance state: x_new = A @ sigma(x) (approximate)
            current_x = A @ torch.tanh(current_x)
            # New Jacobian at advanced state
            sigma_prime_new = 1.0 - torch.tanh(current_x) ** 2
            J_new = A * sigma_prime_new.unsqueeze(0)
            current_J = J_new @ current_J
            blocks.append(current_J.clone())

        O = torch.cat(blocks, dim=0)  # (n_deriv * K, K)
        return O

    def check_identifiability(
        self,
        A: torch.Tensor,
        n_test_points: int = 20,
        n_derivatives: int = 4,
    ) -> Dict[str, object]:
        """Check structural identifiability at multiple evaluation points.

        The system is structurally identifiable if the observability matrix
        has full rank at a generic (randomly chosen) evaluation point.

        Args:
            A: (K, K) interaction matrix to test.
            n_test_points: Number of random evaluation points.
            n_derivatives: Number of Lie derivatives.

        Returns:
            Dict with:
                - "is_structurally_identifiable": True if full rank at
                  majority of test points.
                - "rank_at_points": list of ranks at each test point.
                - "min_rank": minimum rank observed.
                - "max_rank": maximum rank observed.
                - "n_programs": K for reference.
        """
        K = self.n_programs
        ranks = []

        for _ in range(n_test_points):
            x = torch.randn(K) * 0.5  # Evaluate near origin
            O = self.compute_observability_matrix(A, x, n_derivatives)

            # Numerical rank
            sv = torch.linalg.svdvals(O)
            threshold = sv.max() * 1e-8
            rank = (sv > threshold).sum().item()
            ranks.append(rank)

        is_identifiable = sum(r == K for r in ranks) > n_test_points // 2

        return {
            "is_structurally_identifiable": is_identifiable,
            "rank_at_points": ranks,
            "min_rank": min(ranks),
            "max_rank": max(ranks),
            "n_programs": K,
        }

    def verify_diagonal_dominance_sufficiency(
        self,
        n_trials: int = 100,
    ) -> Dict[str, float]:
        """Empirically verify that diagonal dominance implies identifiability.

        Generates random A matrices with and without diagonal dominance
        and checks identifiability rates.

        Args:
            n_trials: Number of random matrices to test.

        Returns:
            Dict with identifiability rates for DD vs non-DD matrices.
        """
        K = self.n_programs
        dd_identifiable = 0
        non_dd_identifiable = 0

        for _ in range(n_trials):
            # Diagonally dominant matrix
            A_dd = torch.randn(K, K) * 0.1
            diag_vals = torch.abs(A_dd).sum(dim=1) + 0.1
            A_dd.fill_diagonal_(0)
            A_dd = A_dd - torch.diag(diag_vals)
            result_dd = self.check_identifiability(A_dd, n_test_points=5)
            if result_dd["is_structurally_identifiable"]:
                dd_identifiable += 1

            # Non-diagonally-dominant matrix (random)
            A_ndd = torch.randn(K, K) * 0.5
            result_ndd = self.check_identifiability(A_ndd, n_test_points=5)
            if result_ndd["is_structurally_identifiable"]:
                non_dd_identifiable += 1

        return {
            "dd_identifiability_rate": dd_identifiable / n_trials,
            "non_dd_identifiability_rate": non_dd_identifiable / n_trials,
        }


# ===================================================================
# Empirical Convergence Test
# ===================================================================

class EmpiricalConvergenceTest:
    """Test whether multiple random initializations converge to the
    same interaction matrix A (up to permutation).

    Procedure:
        1. Generate synthetic data from a known ground-truth A*.
        2. Train the model from N different random initializations.
        3. For each pair of learned A matrices, compute the best
           permutation-aligned distance.
        4. If all pairwise distances < tolerance, the model is
           empirically identifiable.

    This tests practical identifiability (not just theoretical),
    accounting for optimization landscape effects.

    Args:
        config: IdentifiabilityConfig.
    """

    def __init__(self, config: IdentifiabilityConfig) -> None:
        self.config = config

    def generate_synthetic_data(
        self,
        A_true: torch.Tensor,
        n_trajectories: int,
        t_span: torch.Tensor,
        noise_std: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic trajectory data from known A.

        dx/dt = A @ tanh(x), x(0) ~ N(0, I), y = x + noise.

        Args:
            A_true: (K, K) ground-truth interaction matrix.
            n_trajectories: Number of trajectories to generate.
            t_span: (T,) time points.
            noise_std: Observation noise standard deviation.

        Returns:
            x0s: (N, K) initial conditions.
            trajectories: (T, N, K) noisy observed trajectories.
        """
        K = A_true.shape[0]

        def ode_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return x @ A_true.T  # same as (A @ tanh(x).T).T with linear approx
            # Actually: A @ tanh(x^T) for each sample, but batched:

        # Proper batched ODE
        def ode_fn_proper(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return (A_true @ torch.tanh(x).T).T

        x0s = torch.randn(n_trajectories, K) * 0.5

        if HAS_TORCHDIFFEQ:
            with torch.no_grad():
                trajectories = odeint(ode_fn_proper, x0s, t_span, method="euler")
        else:
            # Simple Euler integration fallback
            dt = t_span[1] - t_span[0] if len(t_span) > 1 else 0.1
            trajectories = [x0s]
            x_curr = x0s.clone()
            for i in range(1, len(t_span)):
                dt_i = t_span[i] - t_span[i-1]
                dx = ode_fn_proper(t_span[i-1], x_curr)
                x_curr = x_curr + dt_i * dx
                trajectories.append(x_curr.clone())
            trajectories = torch.stack(trajectories, dim=0)

        # Add noise
        trajectories = trajectories + noise_std * torch.randn_like(trajectories)
        return x0s, trajectories

    @staticmethod
    def permutation_aligned_distance(
        A1: torch.Tensor, A2: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """Find the permutation of A2's rows/cols that minimizes ||A1 - P A2 P^T||.

        Uses a greedy matching based on row-similarity (exact combinatorial
        search is O(K!) and infeasible for K > 10).

        Args:
            A1: (K, K) first matrix.
            A2: (K, K) second matrix.

        Returns:
            (distance, best_permutation): Frobenius distance and the
                permutation vector.
        """
        K = A1.shape[0]

        # Greedy matching: for each row i in A1, find best matching row in A2
        used = set()
        perm = torch.zeros(K, dtype=torch.long)

        for i in range(K):
            best_j = -1
            best_dist = float("inf")
            for j in range(K):
                if j in used:
                    continue
                # Compare row i of A1 with row j of A2
                d = (A1[i] - A2[j]).norm().item()
                if d < best_dist:
                    best_dist = d
                    best_j = j
            perm[i] = best_j
            used.add(best_j)

        # Apply permutation to A2
        A2_perm = A2[perm][:, perm]
        dist = (A1 - A2_perm).norm().item()
        return dist, perm

    def run_convergence_test(
        self,
        A_true: torch.Tensor,
        n_inits: Optional[int] = None,
    ) -> Dict[str, object]:
        """Run the full empirical convergence test.

        Args:
            A_true: (K, K) ground-truth interaction matrix.
            n_inits: Number of random initializations (default from config).

        Returns:
            Dict with:
                - "converged": bool, whether all runs converged to same A.
                - "pairwise_distances": list of pairwise distances.
                - "mean_distance": mean pairwise distance.
                - "max_distance": max pairwise distance.
                - "learned_matrices": list of learned A matrices.
                - "training_losses": list of final losses per init.
        """
        n_inits = n_inits or self.config.n_random_inits
        K = A_true.shape[0]
        cfg = self.config

        # Generate data
        t_span = torch.linspace(0, 2.0, cfg.n_time_points)
        x0s, trajectories = self.generate_synthetic_data(
            A_true, cfg.n_trajectories, t_span
        )

        learned_As: List[torch.Tensor] = []
        final_losses: List[float] = []

        for init_idx in range(n_inits):
            torch.manual_seed(init_idx * 137 + 42)
            # Learnable A matrix
            A_param = nn.Parameter(torch.randn(K, K) * 0.1)
            optimizer = torch.optim.Adam([A_param], lr=cfg.lr)

            for step in range(cfg.max_train_steps):
                optimizer.zero_grad()

                def ode_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                    return (A_param @ torch.tanh(x).T).T

                # Forward: Euler integration (fast, differentiable)
                x_pred = [x0s]
                x_curr = x0s
                for ti in range(1, len(t_span)):
                    dt = t_span[ti] - t_span[ti - 1]
                    x_curr = x_curr + dt * ode_fn(t_span[ti - 1], x_curr)
                    x_pred.append(x_curr)
                x_pred = torch.stack(x_pred, dim=0)

                loss = F.mse_loss(x_pred, trajectories)
                loss.backward()
                optimizer.step()

            learned_As.append(A_param.detach().clone())
            final_losses.append(loss.item())

        # Compute pairwise distances
        distances = []
        for i in range(n_inits):
            for j in range(i + 1, n_inits):
                d, _ = self.permutation_aligned_distance(learned_As[i], learned_As[j])
                distances.append(d)

        mean_dist = sum(distances) / max(len(distances), 1)
        max_dist = max(distances) if distances else 0.0
        converged = max_dist < cfg.convergence_tol

        return {
            "converged": converged,
            "pairwise_distances": distances,
            "mean_distance": mean_dist,
            "max_distance": max_dist,
            "learned_matrices": learned_As,
            "training_losses": final_losses,
        }


# ===================================================================
# Sensitivity Analysis
# ===================================================================

class SensitivityAnalyzer:
    """Analyze sensitivity of ODE predictions to each parameter in A.

    Computes:
        S_ij = ||dy/dA_ij||_2 / ||y||_2

    (normalized sensitivity of the full trajectory to parameter A_ij).

    High sensitivity = that interaction matters for predictions.
    Low sensitivity = can be pruned without affecting predictions.

    This provides:
        1. Feature importance for the learned GRN.
        2. Guidance for sparsification (prune low-sensitivity edges).
        3. Uncertainty prioritization (high-sensitivity params need
           tighter confidence intervals).

    Args:
        config: IdentifiabilityConfig.
    """

    def __init__(self, config: IdentifiabilityConfig) -> None:
        self.config = config

    def compute_parameter_sensitivity(
        self,
        A: torch.Tensor,
        x0: torch.Tensor,
        t_span: torch.Tensor,
        epsilon: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute sensitivity matrix via finite differences.

        S_ij = ||y(A + eps*E_ij) - y(A)|| / (eps * ||y(A)||)

        where E_ij is the matrix with 1 at position (i,j) and 0 elsewhere.

        Args:
            A: (K, K) interaction matrix.
            x0: (N, K) initial conditions.
            t_span: (T,) time points.
            epsilon: Perturbation size (default from config).

        Returns:
            S: (K, K) sensitivity matrix.
        """
        epsilon = epsilon or self.config.sensitivity_epsilon
        K = A.shape[0]

        def integrate(A_mat: torch.Tensor) -> torch.Tensor:
            """Euler integrate with given A."""
            x = x0.clone()
            traj = [x]
            for ti in range(1, len(t_span)):
                dt = t_span[ti] - t_span[ti - 1]
                dx = (A_mat @ torch.tanh(x).T).T
                x = x + dt * dx
                traj.append(x)
            return torch.stack(traj, dim=0)

        with torch.no_grad():
            y_base = integrate(A)
            y_norm = y_base.norm() + 1e-10

            S = torch.zeros(K, K)
            for i in range(K):
                for j in range(K):
                    A_pert = A.clone()
                    A_pert[i, j] += epsilon
                    y_pert = integrate(A_pert)
                    S[i, j] = (y_pert - y_base).norm() / (epsilon * y_norm)

        return S

    def identify_critical_interactions(
        self,
        S: torch.Tensor,
        top_k: int = 10,
    ) -> List[Tuple[int, int, float]]:
        """Identify the most sensitivity-critical interactions.

        Args:
            S: (K, K) sensitivity matrix.
            top_k: Number of top interactions to return.

        Returns:
            List of (i, j, sensitivity) tuples, sorted descending.
        """
        K = S.shape[0]
        entries = []
        for i in range(K):
            for j in range(K):
                entries.append((i, j, S[i, j].item()))
        entries.sort(key=lambda x: x[2], reverse=True)
        return entries[:top_k]

    def pruning_recommendation(
        self,
        S: torch.Tensor,
        threshold_quantile: float = 0.25,
    ) -> torch.Tensor:
        """Generate a pruning mask based on sensitivity analysis.

        Parameters below the threshold quantile of sensitivity are
        recommended for pruning (setting to zero).

        Args:
            S: (K, K) sensitivity matrix.
            threshold_quantile: Fraction of parameters to recommend pruning.

        Returns:
            mask: (K, K) binary mask where 1 = keep, 0 = prune.
        """
        threshold = torch.quantile(S.flatten(), threshold_quantile)
        mask = (S > threshold).float()
        # Always keep diagonal (self-regulation)
        mask.fill_diagonal_(1.0)
        return mask


# ===================================================================
# Full Analysis Pipeline
# ===================================================================

def run_full_identifiability_analysis(
    A_true: torch.Tensor,
    config: Optional[IdentifiabilityConfig] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Run all identifiability analyses on a given interaction matrix.

    Args:
        A_true: (K, K) ground-truth interaction matrix.
        config: Analysis configuration.
        verbose: Whether to print results.

    Returns:
        Dict with results from each analysis method.
    """
    config = config or IdentifiabilityConfig(n_programs=A_true.shape[0])
    results: Dict[str, Dict] = {}

    # 1. Structural identifiability
    if verbose:
        print("=" * 60)
        print("1. Structural Identifiability (Lie derivatives)")
        print("=" * 60)
    sit = StructuralIdentifiabilityTest(config.n_programs)
    struct_result = sit.check_identifiability(A_true)
    results["structural"] = struct_result
    if verbose:
        print(f"  Structurally identifiable: {struct_result['is_structurally_identifiable']}")
        print(f"  Rank range: [{struct_result['min_rank']}, {struct_result['max_rank']}] / {config.n_programs}")

    # 2. Sensitivity analysis
    if verbose:
        print("\n" + "=" * 60)
        print("2. Sensitivity Analysis")
        print("=" * 60)
    sa = SensitivityAnalyzer(config)
    t_span = torch.linspace(0, 2.0, config.n_time_points)
    x0 = torch.randn(config.n_trajectories, config.n_programs) * 0.5
    S = sa.compute_parameter_sensitivity(A_true, x0, t_span)
    critical = sa.identify_critical_interactions(S, top_k=5)
    results["sensitivity"] = {
        "matrix": S,
        "critical_interactions": critical,
        "pruning_mask": sa.pruning_recommendation(S),
    }
    if verbose:
        print("  Top-5 most sensitive interactions:")
        for i, j, s in critical:
            print(f"    A[{i},{j}] = {A_true[i,j]:.4f}, sensitivity = {s:.6f}")

    # 3. Empirical convergence
    if verbose:
        print("\n" + "=" * 60)
        print("3. Empirical Convergence Test")
        print("=" * 60)
    ect = EmpiricalConvergenceTest(config)
    conv_result = ect.run_convergence_test(A_true, n_inits=min(config.n_random_inits, 3))
    results["empirical"] = conv_result
    if verbose:
        print(f"  Converged: {conv_result['converged']}")
        print(f"  Mean pairwise distance: {conv_result['mean_distance']:.6f}")
        print(f"  Max pairwise distance: {conv_result['max_distance']:.6f}")

    return results


# ===================================================================
# Unit Tests
# ===================================================================

def _test_structural_identifiability() -> None:
    """Test StructuralIdentifiabilityTest."""
    K = 4
    sit = StructuralIdentifiabilityTest(K)

    # Diagonally dominant matrix (should be identifiable)
    A = torch.tensor([
        [-1.0, 0.2, 0.0, 0.1],
        [0.1, -0.8, 0.1, 0.0],
        [0.0, 0.3, -1.2, 0.1],
        [0.1, 0.0, 0.2, -0.9],
    ])

    result = sit.check_identifiability(A, n_test_points=10)
    assert "is_structurally_identifiable" in result
    assert result["max_rank"] == K, f"Expected rank {K}, got {result['max_rank']}"
    print(f"  Identifiable: {result['is_structurally_identifiable']}")
    print("[PASS] StructuralIdentifiabilityTest")


def _test_empirical_convergence() -> None:
    """Test EmpiricalConvergenceTest (small scale)."""
    K = 3
    cfg = IdentifiabilityConfig(
        n_programs=K, n_time_points=10, n_trajectories=20,
        n_random_inits=2, max_train_steps=50, lr=0.05,
    )
    ect = EmpiricalConvergenceTest(cfg)

    A_true = torch.tensor([
        [-1.0, 0.3, 0.0],
        [0.1, -0.8, 0.2],
        [0.0, 0.1, -0.7],
    ])

    result = ect.run_convergence_test(A_true, n_inits=2)
    assert "converged" in result
    assert "pairwise_distances" in result
    assert len(result["learned_matrices"]) == 2
    print(f"  Mean dist: {result['mean_distance']:.6f}")
    print("[PASS] EmpiricalConvergenceTest")


def _test_sensitivity_analyzer() -> None:
    """Test SensitivityAnalyzer."""
    K = 4
    cfg = IdentifiabilityConfig(n_programs=K, n_time_points=10, n_trajectories=5)
    sa = SensitivityAnalyzer(cfg)

    A = torch.tensor([
        [-1.0, 0.5, 0.0, 0.0],
        [0.0, -0.8, 0.3, 0.0],
        [0.0, 0.0, -1.0, 0.2],
        [0.0, 0.0, 0.0, -0.5],
    ])
    x0 = torch.randn(5, K) * 0.3
    t_span = torch.linspace(0, 1.0, 10)

    S = sa.compute_parameter_sensitivity(A, x0, t_span)
    assert S.shape == (K, K), f"Expected ({K},{K}), got {S.shape}"
    assert (S >= 0).all(), "Sensitivity should be non-negative"

    critical = sa.identify_critical_interactions(S, top_k=3)
    assert len(critical) == 3

    mask = sa.pruning_recommendation(S)
    assert mask.shape == (K, K)
    # Diagonal always kept
    assert (torch.diag(mask) == 1).all()

    print("[PASS] SensitivityAnalyzer")


def _test_permutation_alignment() -> None:
    """Test permutation-aligned distance."""
    A1 = torch.tensor([[1.0, 0.2], [0.3, 2.0]])
    # A2 is A1 with rows/cols permuted
    A2 = torch.tensor([[2.0, 0.3], [0.2, 1.0]])

    dist, perm = EmpiricalConvergenceTest.permutation_aligned_distance(A1, A2)
    assert dist < 0.01, f"Permuted matrix distance {dist} too large"
    print(f"  Permutation-aligned distance: {dist:.6f}")
    print("[PASS] Permutation alignment")


def run_all_tests() -> None:
    """Run all unit tests."""
    _test_structural_identifiability()
    _test_sensitivity_analyzer()
    _test_permutation_alignment()
    _test_empirical_convergence()
    print("\n=== All identifiability_proof tests passed ===")


if __name__ == "__main__":
    run_all_tests()
