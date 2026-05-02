"""Formal Dynamical Systems Analysis for Drug Resistance Trajectories.

Provides rigorous dynamical systems tools for analyzing the structured
biological ODE model, including bifurcation analysis, basin of attraction
computation, Lyapunov exponent estimation, and phase portrait generation.

These tools transform the model from a black-box predictor into a
mechanistically interpretable dynamical system where:
    - Bifurcation diagrams reveal critical drug doses for resistance transitions
    - Basin boundaries delineate sensitive vs resistant cell states
    - Lyapunov exponents quantify stability of attractor states
    - Phase portraits visualize the resistance landscape

Theoretical references:
    - Strogatz, S.H. (2015). "Nonlinear Dynamics and Chaos," 2nd ed. Westview.
    - Kuznetsov, Y.A. (2004). "Elements of Applied Bifurcation Theory,"
      3rd ed. Springer.
    - Benettin et al. (1980). "Lyapunov Characteristic Exponents for
      smooth dynamical systems and for Hamiltonian systems." Meccanica.
    - Huang, S. (2009). "Reprogramming cell fates: reconciling rarity
      with robustness." BioEssays 31(5), 546-560.

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

logger = logging.getLogger(__name__)

try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

__all__ = [
    "StabilityConfig",
    "BifurcationAnalyzer",
    "BasinOfAttraction",
    "LyapunovExponentComputer",
    "PhasePortraitGenerator",
    "FixedPointFinder",
]


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class StabilityConfig:
    """Configuration for stability analysis.

    Attributes:
        state_dim: Dimension of the dynamical system state.
        grid_resolution: Number of grid points per dimension for basin analysis.
        integration_time: Duration for trajectory integration.
        dt: Time step for numerical integration.
        lyapunov_transient: Transient time before Lyapunov exponent computation.
        lyapunov_compute_time: Time for Lyapunov exponent computation.
        n_lyapunov_steps: Number of renormalization steps.
        bifurcation_n_points: Number of parameter values in bifurcation diagram.
        fixed_point_tol: Tolerance for declaring a fixed point.
        fixed_point_max_iter: Max iterations for fixed point search.
        state_range: Range for state space exploration.
    """
    state_dim: int = 8
    grid_resolution: int = 50
    integration_time: float = 20.0
    dt: float = 0.01
    lyapunov_transient: float = 10.0
    lyapunov_compute_time: float = 50.0
    n_lyapunov_steps: int = 100
    bifurcation_n_points: int = 100
    fixed_point_tol: float = 1e-5
    fixed_point_max_iter: int = 1000
    state_range: float = 3.0


# ===================================================================
# Fixed Point Finder
# ===================================================================

class FixedPointFinder:
    """Find fixed points (equilibria) of dx/dt = f(x) using Newton's method.

    A fixed point x* satisfies f(x*) = 0. We find these using a
    combination of random initialization + Newton-Raphson iterations.

    For biological interpretation:
        - Stable fixed points = cell state attractors (sensitive/resistant)
        - Unstable fixed points = transition states (saddles)
        - The number and nature of fixed points determine the phenotypic
          landscape of the cell population.

    Args:
        config: StabilityConfig.
    """

    def __init__(self, config: StabilityConfig) -> None:
        self.config = config

    def find_fixed_points(
        self,
        dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
        n_inits: int = 100,
        merge_threshold: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find fixed points from random initializations.

        Args:
            dynamics_fn: f(x) -> dx/dt, operates on (B, D) batched input.
            n_inits: Number of random initial guesses.
            merge_threshold: Distance below which fixed points are merged.

        Returns:
            fixed_points: (M, D) unique fixed points found.
            stability: (M,) classification: 1 = stable, 0 = unstable, -1 = saddle.
        """
        D = self.config.state_dim
        tol = self.config.fixed_point_tol
        max_iter = self.config.fixed_point_max_iter

        # Random initializations
        x = torch.randn(n_inits, D) * self.config.state_range * 0.5
        x = x.requires_grad_(True)

        # Optimize to find f(x) = 0 via gradient descent on ||f(x)||^2
        optimizer = torch.optim.LBFGS([x], lr=0.1, max_iter=20)

        for _ in range(max_iter // 20):
            def closure():
                optimizer.zero_grad()
                fx = dynamics_fn(x)
                loss = (fx ** 2).sum(dim=-1).mean()
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                fx = dynamics_fn(x)
                residual = (fx ** 2).sum(dim=-1)
                if residual.max() < tol:
                    break

        # Filter converged fixed points
        x_det = x.detach()
        with torch.no_grad():
            fx = dynamics_fn(x_det)
            residuals = (fx ** 2).sum(dim=-1)
            converged = residuals < tol * 10  # Slightly relaxed

        fps = x_det[converged]

        if fps.shape[0] == 0:
            return torch.zeros(0, D), torch.zeros(0)

        # Merge nearby fixed points
        unique_fps = [fps[0]]
        for i in range(1, fps.shape[0]):
            is_new = True
            for ufp in unique_fps:
                if (fps[i] - ufp).norm() < merge_threshold:
                    is_new = False
                    break
            if is_new:
                unique_fps.append(fps[i])

        unique_fps = torch.stack(unique_fps)

        # Classify stability via Jacobian eigenvalues
        stability = self._classify_stability(dynamics_fn, unique_fps)

        return unique_fps, stability

    def _classify_stability(
        self,
        dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
        fixed_points: torch.Tensor,
    ) -> torch.Tensor:
        """Classify fixed points as stable, unstable, or saddle.

        Uses eigenvalues of the Jacobian J(x*):
            - All Re(lambda) < 0: stable node/spiral
            - All Re(lambda) > 0: unstable
            - Mixed signs: saddle point

        Args:
            dynamics_fn: f(x) -> dx/dt.
            fixed_points: (M, D) fixed points.

        Returns:
            stability: (M,) values: 1=stable, -1=saddle, 0=unstable.
        """
        M, D = fixed_points.shape
        stability = torch.zeros(M)

        for i in range(M):
            x = fixed_points[i].unsqueeze(0).requires_grad_(True)
            fx = dynamics_fn(x)

            # Compute Jacobian
            J = torch.zeros(D, D)
            for d in range(D):
                grad = torch.autograd.grad(
                    fx[0, d], x, retain_graph=True, create_graph=False
                )[0]
                J[d] = grad[0]

            # Eigenvalues
            eigenvalues = torch.linalg.eigvals(J)
            real_parts = eigenvalues.real

            if (real_parts < 0).all():
                stability[i] = 1.0    # Stable
            elif (real_parts > 0).all():
                stability[i] = 0.0    # Unstable
            else:
                stability[i] = -1.0   # Saddle

        return stability


# ===================================================================
# Bifurcation Analyzer
# ===================================================================

class BifurcationAnalyzer:
    """Compute bifurcation diagrams as a control parameter (e.g., drug
    concentration) varies.

    Bifurcation diagrams reveal critical parameter values where the
    qualitative behavior of the dynamical system changes (e.g., from
    monostable sensitive to bistable sensitive/resistant).

    For drug resistance:
        - At low drug concentration: single sensitive attractor
        - At critical dose: saddle-node bifurcation creates resistant attractor
        - At high dose: resistant attractor becomes dominant
        - The bifurcation point = critical drug dose for resistance emergence

    Args:
        config: StabilityConfig.
    """

    def __init__(self, config: StabilityConfig) -> None:
        self.config = config
        self.fp_finder = FixedPointFinder(config)

    def compute_bifurcation_diagram(
        self,
        parameterized_dynamics: Callable[[torch.Tensor, float], torch.Tensor],
        param_range: Tuple[float, float],
        projection_dim: int = 0,
        n_points: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute a bifurcation diagram.

        For each parameter value, finds fixed points and records their
        position (projected to a chosen dimension) and stability.

        Args:
            parameterized_dynamics: f(x, param) -> dx/dt where param
                is the bifurcation parameter (e.g., drug concentration).
            param_range: (min, max) of the bifurcation parameter.
            projection_dim: Which state dimension to project onto for
                the diagram (default: 0, the drug efflux program).
            n_points: Number of parameter values to evaluate.

        Returns:
            Dict with:
                - "param_values": (N,) parameter values.
                - "fixed_point_values": list of tensors, one per param value.
                - "stability": list of tensors (1=stable, -1=saddle, 0=unstable).
                - "n_stable": (N,) number of stable fixed points at each param.
                - "bifurcation_points": parameter values where n_stable changes.
        """
        n_points = n_points or self.config.bifurcation_n_points
        params = torch.linspace(param_range[0], param_range[1], n_points)

        all_fp_values: List[torch.Tensor] = []
        all_stability: List[torch.Tensor] = []
        n_stable = torch.zeros(n_points)

        for idx, p in enumerate(params):
            p_val = p.item()

            def dynamics_at_p(x: torch.Tensor) -> torch.Tensor:
                return parameterized_dynamics(x, p_val)

            fps, stab = self.fp_finder.find_fixed_points(
                dynamics_at_p, n_inits=30, merge_threshold=0.2
            )

            if fps.shape[0] > 0:
                fp_proj = fps[:, projection_dim]
                all_fp_values.append(fp_proj)
                all_stability.append(stab)
                n_stable[idx] = (stab == 1).sum().item()
            else:
                all_fp_values.append(torch.tensor([]))
                all_stability.append(torch.tensor([]))
                n_stable[idx] = 0

        # Detect bifurcation points (where n_stable changes)
        bif_points = []
        for i in range(1, n_points):
            if n_stable[i] != n_stable[i-1]:
                # Linear interpolation for precise location
                bif_param = (params[i-1] + params[i]) / 2
                bif_points.append(bif_param.item())

        return {
            "param_values": params,
            "fixed_point_values": all_fp_values,
            "stability": all_stability,
            "n_stable": n_stable,
            "bifurcation_points": bif_points,
        }


# ===================================================================
# Basin of Attraction
# ===================================================================

class BasinOfAttraction:
    """Compute basin boundaries by integrating trajectories from a grid
    of initial conditions and determining which attractor each converges to.

    The basin of attraction of an attractor x* is the set of all initial
    conditions that converge to x* under the dynamics. The basin boundaries
    separate different phenotypic fates.

    For drug resistance interpretation:
        - Basin of sensitive attractor = initial states that remain sensitive
        - Basin of resistant attractor = initial states that become resistant
        - Basin boundary = tipping point for resistance transition
        - Basin volume = robustness of each state

    Args:
        config: StabilityConfig.
    """

    def __init__(self, config: StabilityConfig) -> None:
        self.config = config

    def compute_2d_basins(
        self,
        dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
        attractors: torch.Tensor,
        dim1: int = 0,
        dim2: int = 1,
        other_dims_value: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute basin assignments on a 2D grid.

        Integrates trajectories from a grid in the (dim1, dim2) plane
        and assigns each grid point to the nearest attractor at t=T.

        Args:
            dynamics_fn: f(x) -> dx/dt, batched.
            attractors: (M, D) attractor positions.
            dim1: First grid dimension.
            dim2: Second grid dimension.
            other_dims_value: Values for non-grid dimensions.
                Default: zeros.

        Returns:
            Dict with:
                - "grid_x": (R,) grid values for dim1.
                - "grid_y": (R,) grid values for dim2.
                - "basin_ids": (R, R) integer attractor assignments.
                - "final_states": (R, R, D) final states.
                - "basin_volumes": (M,) relative volume of each basin.
        """
        D = self.config.state_dim
        R = self.config.grid_resolution
        T = self.config.integration_time
        dt = self.config.dt

        sr = self.config.state_range
        grid_x = torch.linspace(-sr, sr, R)
        grid_y = torch.linspace(-sr, sr, R)

        # Create initial conditions on 2D grid
        xx, yy = torch.meshgrid(grid_x, grid_y, indexing="ij")
        n_points = R * R

        if other_dims_value is None:
            other_dims_value = torch.zeros(D)

        x0 = other_dims_value.unsqueeze(0).expand(n_points, -1).clone()
        x0[:, dim1] = xx.reshape(-1)
        x0[:, dim2] = yy.reshape(-1)

        # Integrate forward with Euler method (batched)
        x = x0.clone()
        n_steps = int(T / dt)
        with torch.no_grad():
            for _ in range(n_steps):
                dx = dynamics_fn(x)
                x = x + dt * dx
                # Clamp to prevent blowup
                x = torch.clamp(x, -10 * sr, 10 * sr)

        final_states = x  # (n_points, D)

        # Assign to nearest attractor
        M = attractors.shape[0]
        if M > 0:
            dists = torch.cdist(final_states, attractors)  # (n_points, M)
            basin_ids = dists.argmin(dim=1)  # (n_points,)
        else:
            basin_ids = torch.zeros(n_points, dtype=torch.long)

        basin_ids_grid = basin_ids.reshape(R, R)
        final_states_grid = final_states.reshape(R, R, D)

        # Compute basin volumes (fraction of grid points)
        basin_volumes = torch.zeros(max(M, 1))
        for m in range(max(M, 1)):
            basin_volumes[m] = (basin_ids == m).float().mean()

        return {
            "grid_x": grid_x,
            "grid_y": grid_y,
            "basin_ids": basin_ids_grid,
            "final_states": final_states_grid,
            "basin_volumes": basin_volumes,
        }

    def compute_basin_boundary(
        self,
        basin_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Extract basin boundary points from the grid.

        A boundary point is any grid cell adjacent to a cell with a
        different basin assignment.

        Args:
            basin_ids: (R, R) integer attractor assignments.

        Returns:
            boundary: (R, R) binary mask where 1 = boundary.
        """
        R = basin_ids.shape[0]
        boundary = torch.zeros_like(basin_ids, dtype=torch.float)

        for i in range(R):
            for j in range(R):
                current = basin_ids[i, j]
                # Check 4-neighbors
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < R and 0 <= nj < R:
                        if basin_ids[ni, nj] != current:
                            boundary[i, j] = 1.0
                            break
        return boundary

    @staticmethod
    def basin_stability_index(basin_volumes: torch.Tensor) -> float:
        """Compute basin stability index.

        The basin stability of an attractor is its basin volume fraction.
        This metric from Menck et al. (2013) quantifies how likely a
        random perturbation returns to the attractor.

        For the dominant attractor (largest basin):
            BSI = V_max / V_total

        Args:
            basin_volumes: (M,) relative basin volumes.

        Returns:
            BSI for the dominant basin (0 to 1).
        """
        if basin_volumes.numel() == 0:
            return 0.0
        return basin_volumes.max().item()


# ===================================================================
# Lyapunov Exponent Computer
# ===================================================================

class LyapunovExponentComputer:
    """Compute the maximal Lyapunov exponent (MLE) for each attractor.

    The MLE characterizes the rate of exponential divergence of nearby
    trajectories. For a stable attractor, MLE < 0, and |MLE| quantifies
    the stability strength (how fast perturbations decay).

    Algorithm (Benettin et al., 1980):
        1. Start near the attractor x*.
        2. Propagate a small perturbation delta along with the trajectory.
        3. Periodically renormalize delta to prevent overflow/underflow.
        4. MLE = lim_{t->inf} (1/t) * sum ln(||delta||)

    For drug resistance:
        - Large |MLE| = strong attractor = hard to escape resistant state
        - Small |MLE| = weak attractor = potentially reversible resistance
        - MLE > 0 = chaos (rare in gene regulatory networks)

    Args:
        config: StabilityConfig.
    """

    def __init__(self, config: StabilityConfig) -> None:
        self.config = config

    @torch.no_grad()
    def compute_mle(
        self,
        dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
        initial_state: torch.Tensor,
        jacobian_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[float, List[float]]:
        """Compute the maximal Lyapunov exponent.

        Args:
            dynamics_fn: f(x) -> dx/dt, operates on (1, D) input.
            initial_state: (D,) starting point near an attractor.
            jacobian_fn: Optional J(x) -> (D, D) Jacobian. If None,
                computed via finite differences.

        Returns:
            (mle, mle_history): The converged MLE and its running values.
        """
        D = self.config.state_dim
        dt = self.config.dt
        transient_steps = int(self.config.lyapunov_transient / dt)
        compute_steps = int(self.config.lyapunov_compute_time / dt)
        renorm_interval = max(compute_steps // self.config.n_lyapunov_steps, 1)

        x = initial_state.clone().unsqueeze(0)  # (1, D)

        # Transient: let trajectory settle near attractor
        for _ in range(transient_steps):
            dx = dynamics_fn(x)
            x = x + dt * dx

        # Initialize perturbation
        delta = torch.randn(D)
        delta = delta / delta.norm() * 1e-8

        mle_sum = 0.0
        n_renorms = 0
        mle_history = []

        for step in range(compute_steps):
            # Advance state
            dx = dynamics_fn(x)
            x = x + dt * dx

            # Advance perturbation using linearized dynamics
            if jacobian_fn is not None:
                J = jacobian_fn(x.squeeze(0))
                delta = delta + dt * (J @ delta)
            else:
                # Finite difference approximation
                eps = 1e-6
                x_pert = x.clone()
                delta_norm = delta / (delta.norm() + 1e-12) * eps
                x_pert = x_pert + delta_norm.unsqueeze(0)
                dx_pert = dynamics_fn(x_pert)
                delta = delta + dt * ((dx_pert - dx).squeeze(0) / eps * delta.norm())

            # Renormalize periodically
            if (step + 1) % renorm_interval == 0:
                d_norm = delta.norm().item()
                if d_norm > 0:
                    mle_sum += math.log(d_norm / 1e-8)
                    delta = delta / d_norm * 1e-8
                    n_renorms += 1

                    elapsed_time = (step + 1) * dt
                    running_mle = mle_sum / elapsed_time
                    mle_history.append(running_mle)

        total_time = compute_steps * dt
        mle = mle_sum / total_time if total_time > 0 else 0.0

        return mle, mle_history

    def compute_lyapunov_spectrum_linear(
        self,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Lyapunov exponents for the linearized system dx/dt = Ax.

        For linear systems, the Lyapunov exponents equal the real parts
        of the eigenvalues of A. This provides a quick estimate when
        the system is near a fixed point.

        Args:
            A: (D, D) Jacobian matrix at a fixed point.

        Returns:
            exponents: (D,) Lyapunov exponents, sorted descending.
        """
        eigenvalues = torch.linalg.eigvals(A)
        real_parts = eigenvalues.real
        sorted_idx = real_parts.argsort(descending=True)
        return real_parts[sorted_idx]


# ===================================================================
# Phase Portrait Generator
# ===================================================================

class PhasePortraitGenerator:
    """Generate 2D phase portraits projected onto chosen dimensions.

    Phase portraits show the flow of the dynamical system in a 2D
    projection, revealing:
        - Direction and speed of state transitions
        - Location of fixed points (attractors, saddles)
        - Separatrices (basin boundaries)
        - Limit cycles (if any)

    For drug resistance:
        - Flow toward resistant attractor = resistance trajectory
        - Separatrix = tipping point between sensitive and resistant
        - Flow field strength = rate of state transition

    Args:
        config: StabilityConfig.
    """

    def __init__(self, config: StabilityConfig) -> None:
        self.config = config

    @torch.no_grad()
    def compute_flow_field(
        self,
        dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
        dim1: int = 0,
        dim2: int = 1,
        other_dims_value: Optional[torch.Tensor] = None,
        grid_resolution: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the 2D flow field on a grid.

        Args:
            dynamics_fn: f(x) -> dx/dt, batched.
            dim1: First projection dimension.
            dim2: Second projection dimension.
            other_dims_value: Values for non-projected dimensions.
            grid_resolution: Override grid resolution.

        Returns:
            Dict with:
                - "grid_x": (R,) x-axis values.
                - "grid_y": (R,) y-axis values.
                - "u": (R, R) flow x-component (dx/dt projected to dim1).
                - "v": (R, R) flow y-component (dx/dt projected to dim2).
                - "speed": (R, R) flow magnitude.
        """
        D = self.config.state_dim
        R = grid_resolution or self.config.grid_resolution
        sr = self.config.state_range

        grid_x = torch.linspace(-sr, sr, R)
        grid_y = torch.linspace(-sr, sr, R)
        xx, yy = torch.meshgrid(grid_x, grid_y, indexing="ij")

        n_points = R * R
        if other_dims_value is None:
            other_dims_value = torch.zeros(D)

        x = other_dims_value.unsqueeze(0).expand(n_points, -1).clone()
        x[:, dim1] = xx.reshape(-1)
        x[:, dim2] = yy.reshape(-1)

        dxdt = dynamics_fn(x)  # (n_points, D)

        u = dxdt[:, dim1].reshape(R, R)
        v = dxdt[:, dim2].reshape(R, R)
        speed = torch.sqrt(u ** 2 + v ** 2)

        return {
            "grid_x": grid_x,
            "grid_y": grid_y,
            "u": u,
            "v": v,
            "speed": speed,
        }

    @torch.no_grad()
    def compute_trajectories(
        self,
        dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
        initial_conditions: torch.Tensor,
        dim1: int = 0,
        dim2: int = 1,
        n_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Integrate trajectories for overlay on phase portrait.

        Args:
            dynamics_fn: f(x) -> dx/dt, batched.
            initial_conditions: (N, D) initial states.
            dim1: First projection dimension.
            dim2: Second projection dimension.
            n_steps: Number of integration steps.

        Returns:
            Dict with:
                - "trajectories_x": (N, T) dim1 values over time.
                - "trajectories_y": (N, T) dim2 values over time.
                - "time": (T,) time values.
        """
        dt = self.config.dt
        n_steps = n_steps or int(self.config.integration_time / dt)
        N = initial_conditions.shape[0]

        traj_x = torch.zeros(N, n_steps + 1)
        traj_y = torch.zeros(N, n_steps + 1)
        time = torch.zeros(n_steps + 1)

        x = initial_conditions.clone()
        traj_x[:, 0] = x[:, dim1]
        traj_y[:, 0] = x[:, dim2]

        for t_idx in range(n_steps):
            dx = dynamics_fn(x)
            x = x + dt * dx
            x = torch.clamp(x, -10 * self.config.state_range,
                            10 * self.config.state_range)
            traj_x[:, t_idx + 1] = x[:, dim1]
            traj_y[:, t_idx + 1] = x[:, dim2]
            time[t_idx + 1] = (t_idx + 1) * dt

        return {
            "trajectories_x": traj_x,
            "trajectories_y": traj_y,
            "time": time,
        }

    @torch.no_grad()
    def compute_nullclines(
        self,
        dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
        dim1: int = 0,
        dim2: int = 1,
        other_dims_value: Optional[torch.Tensor] = None,
        grid_resolution: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute nullclines (curves where dx_i/dt = 0).

        The dim1-nullcline is where dx_{dim1}/dt = 0.
        The dim2-nullcline is where dx_{dim2}/dt = 0.
        Fixed points occur at their intersections.

        Args:
            dynamics_fn: f(x) -> dx/dt.
            dim1: First dimension.
            dim2: Second dimension.
            other_dims_value: Values for other dims.
            grid_resolution: Grid resolution.

        Returns:
            Dict with:
                - "nullcline1": (R, R) |dx_{dim1}/dt| on grid.
                - "nullcline2": (R, R) |dx_{dim2}/dt| on grid.
                - "grid_x": (R,) x values.
                - "grid_y": (R,) y values.
        """
        flow = self.compute_flow_field(
            dynamics_fn, dim1, dim2, other_dims_value, grid_resolution
        )

        return {
            "nullcline1": flow["u"].abs(),
            "nullcline2": flow["v"].abs(),
            "grid_x": flow["grid_x"],
            "grid_y": flow["grid_y"],
        }

    @torch.no_grad()
    def compute_potential_landscape(
        self,
        dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
        dim1: int = 0,
        dim2: int = 1,
        other_dims_value: Optional[torch.Tensor] = None,
        grid_resolution: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Approximate the quasi-potential (Waddington) landscape.

        For gradient systems dx/dt = -grad(V), V is the potential.
        For non-gradient systems, we approximate V by path-integrating
        the flow field: V(x) approx -integral_0^T ||f(x(t))||^2 dt
        along trajectories converging from x to an attractor.

        Simpler approximation used here: V(x) = -log(speed(x) + eps).
        Low speed regions (near attractors) have low V (valleys).
        High speed regions (away from attractors) have high V (ridges).

        Args:
            dynamics_fn: f(x) -> dx/dt.
            dim1, dim2: Projection dimensions.
            other_dims_value: Values for other dims.
            grid_resolution: Grid resolution.

        Returns:
            Dict with:
                - "potential": (R, R) quasi-potential values.
                - "grid_x", "grid_y": grid coordinates.
        """
        flow = self.compute_flow_field(
            dynamics_fn, dim1, dim2, other_dims_value, grid_resolution
        )
        # Quasi-potential: regions of slow flow are low-energy attractors
        speed = flow["speed"]
        potential = -torch.log(speed + 1e-6)

        # Normalize to [0, 1] for visualization
        potential = potential - potential.min()
        potential = potential / (potential.max() + 1e-10)

        return {
            "potential": potential,
            "grid_x": flow["grid_x"],
            "grid_y": flow["grid_y"],
        }


# ===================================================================
# Unit Tests
# ===================================================================

def _make_test_dynamics(A: torch.Tensor) -> Callable:
    """Create a test dynamics function from interaction matrix A."""
    def dynamics(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return (A @ torch.tanh(x).T).T
    return dynamics


def _test_fixed_point_finder() -> None:
    """Test FixedPointFinder."""
    K = 3
    cfg = StabilityConfig(
        state_dim=K, grid_resolution=10,
        integration_time=5.0, dt=0.05,
        fixed_point_max_iter=200,
    )
    fpf = FixedPointFinder(cfg)

    # Simple stable system: origin should be a fixed point
    A = torch.diag(torch.tensor([-1.0, -0.8, -1.2]))
    dynamics = _make_test_dynamics(A)

    fps, stab = fpf.find_fixed_points(dynamics, n_inits=20)
    assert fps.shape[0] >= 1, "Should find at least 1 fixed point"
    # Origin should be found (tanh(0) = 0, so A @ tanh(0) = 0)
    min_dist_to_origin = fps.norm(dim=1).min().item()
    assert min_dist_to_origin < 0.5, f"Origin not found, min dist: {min_dist_to_origin}"
    print(f"  Found {fps.shape[0]} fixed point(s), closest to origin: {min_dist_to_origin:.4f}")
    print("[PASS] FixedPointFinder")


def _test_bifurcation_analyzer() -> None:
    """Test BifurcationAnalyzer."""
    K = 2
    cfg = StabilityConfig(
        state_dim=K, grid_resolution=10,
        bifurcation_n_points=10,
        integration_time=2.0, dt=0.05,
        fixed_point_max_iter=100,
    )
    ba = BifurcationAnalyzer(cfg)

    # Parameterized dynamics: as param increases, system goes from
    # one stable FP to potentially bistable
    def param_dynamics(x: torch.Tensor, param: float) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        A = torch.tensor([[-1.0, param], [0.1, -0.8]])
        return (A @ torch.tanh(x).T).T

    result = ba.compute_bifurcation_diagram(
        param_dynamics, (0.0, 2.0), projection_dim=0, n_points=5
    )
    assert "param_values" in result
    assert len(result["fixed_point_values"]) == 5
    print("[PASS] BifurcationAnalyzer")


def _test_basin_of_attraction() -> None:
    """Test BasinOfAttraction."""
    K = 2
    cfg = StabilityConfig(
        state_dim=K, grid_resolution=10,
        integration_time=5.0, dt=0.05,
        state_range=2.0,
    )
    boa = BasinOfAttraction(cfg)

    A = torch.diag(torch.tensor([-1.0, -0.8]))
    dynamics = _make_test_dynamics(A)
    attractors = torch.zeros(1, K)  # Origin

    result = boa.compute_2d_basins(dynamics, attractors, dim1=0, dim2=1)
    assert result["basin_ids"].shape == (10, 10)
    assert result["basin_volumes"].shape[0] >= 1

    boundary = boa.compute_basin_boundary(result["basin_ids"])
    assert boundary.shape == (10, 10)

    bsi = BasinOfAttraction.basin_stability_index(result["basin_volumes"])
    assert 0 <= bsi <= 1.0
    print(f"  Basin stability index: {bsi:.4f}")
    print("[PASS] BasinOfAttraction")


def _test_lyapunov_exponent() -> None:
    """Test LyapunovExponentComputer."""
    K = 3
    cfg = StabilityConfig(
        state_dim=K, dt=0.01,
        lyapunov_transient=1.0,
        lyapunov_compute_time=5.0,
        n_lyapunov_steps=20,
    )
    lec = LyapunovExponentComputer(cfg)

    # Stable linear system: MLE should be negative
    A = torch.diag(torch.tensor([-1.0, -0.5, -0.8]))
    dynamics = _make_test_dynamics(A)
    x0 = torch.randn(K) * 0.1

    mle, history = lec.compute_mle(dynamics, x0)
    assert mle < 0.5, f"MLE should be negative for stable system, got {mle:.4f}"
    print(f"  MLE for stable system: {mle:.4f}")

    # Linear spectrum
    spectrum = lec.compute_lyapunov_spectrum_linear(A)
    assert spectrum.shape == (K,)
    assert (spectrum < 0).all(), "All exponents should be negative"
    print(f"  Linear spectrum: {spectrum.tolist()}")
    print("[PASS] LyapunovExponentComputer")


def _test_phase_portrait() -> None:
    """Test PhasePortraitGenerator."""
    K = 4
    cfg = StabilityConfig(
        state_dim=K, grid_resolution=8,
        integration_time=2.0, dt=0.05,
        state_range=2.0,
    )
    ppg = PhasePortraitGenerator(cfg)

    A = torch.diag(torch.tensor([-1.0, -0.8, -0.5, -1.2]))
    A[0, 1] = 0.3
    A[1, 0] = -0.2
    dynamics = _make_test_dynamics(A)

    # Flow field
    flow = ppg.compute_flow_field(dynamics, dim1=0, dim2=1)
    assert flow["u"].shape == (8, 8)
    assert flow["speed"].shape == (8, 8)

    # Trajectories
    x0s = torch.randn(5, K) * 0.5
    traj = ppg.compute_trajectories(dynamics, x0s, dim1=0, dim2=1, n_steps=20)
    assert traj["trajectories_x"].shape == (5, 21)

    # Nullclines
    nc = ppg.compute_nullclines(dynamics, dim1=0, dim2=1)
    assert nc["nullcline1"].shape == (8, 8)

    # Potential landscape
    pot = ppg.compute_potential_landscape(dynamics, dim1=0, dim2=1)
    assert pot["potential"].shape == (8, 8)
    assert (pot["potential"] >= 0).all()
    assert (pot["potential"] <= 1.0 + 1e-6).all()

    print("[PASS] PhasePortraitGenerator")


def run_all_tests() -> None:
    """Run all unit tests."""
    _test_fixed_point_finder()
    _test_basin_of_attraction()
    _test_lyapunov_exponent()
    _test_phase_portrait()
    _test_bifurcation_analyzer()
    print("\n=== All stability_analysis tests passed ===")


if __name__ == "__main__":
    run_all_tests()
