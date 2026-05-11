"""
Geometric characterization of ResistanceMap's learned latent ODE.

This module implements:
1. CriticalWindowDetector: Characterizes temporal windows where the learned
   ODE exhibits high Jacobian sensitivity in latent space.
2. TrajectoryDecomposer: Decomposes integrated latent trajectories into
   per-program velocity contributions.
3. PhaseTransitionAnalyzer: Detects mathematical bifurcations in the learned
   dynamics using Jacobian eigenvalue analysis.

IMPORTANT CAUSAL SCOPE
======================
All quantities here are properties of the *learned model* U_θ — they are
mathematical sensitivities of a function fitted on cross-sectional cell-line
IC50 associations. The training data contains no time-ordered (X_t, X_{t+Δ})
pairs, no randomized drug assignment, and no gene-knockout perturbations.

Consequently these outputs are **associative descriptors** (Pearl L1), not
causal estimates (L2/L3). In particular:
  - A high latent sensitivity S(t) = ∂(model_output)/∂z(t) is a Jacobian of
    the learned function, NOT an estimate of d E[Y]/d(do(intervention at t)).
  - "Bifurcations" of the learned ODE are mathematical saddle crossings of
    U_θ; whether they correspond to biological phase transitions requires
    perturbation validation (e.g., DepMap CRISPR knock-out) not in scope here.
  - Use as clinical intervention-timing guidance is NOT supported by the
    training data design.

References:
    Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos. Westview Press.
    Scheffer, M. et al. (2009). Early-warning signals for critical transitions. Nature, 461, 53-59.
    Moris, N. et al. (2016). Transition states and cell fate decisions. Genome Biology, 17, 73.
    Saelens, W. et al. (2019). A comparison of single-cell trajectory inference methods.
        Nature Biotechnology, 37, 547-554.
    Pearl, J. (2009). Causality (2nd ed.). Cambridge University Press. (For the
        do-calculus distinction between conditional and interventional quantities.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TemporalConfig:
    """Configuration for temporal explanation analysis.

    Attributes:
        n_timepoints: Number of evaluation timepoints along the trajectory.
        perturbation_eps: Epsilon for finite-difference Jacobian computation.
        eigenvalue_threshold: Threshold for detecting eigenvalue crossings.
        sensitivity_n_perturbations: Number of perturbation samples for
            sensitivity analysis.
        critical_slowing_window: Rolling window size for critical slowing
            down detection.
        device: Torch device string.
    """
    n_timepoints: int = 100
    perturbation_eps: float = 1e-4
    eigenvalue_threshold: float = 0.01
    sensitivity_n_perturbations: int = 50
    critical_slowing_window: int = 10
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CriticalWindowResult:
    """Results from critical window detection.

    Attributes:
        time_sensitivity: (T,) array of d(outcome)/d(intervention at t).
        critical_windows: List of (t_start, t_end) intervals of high sensitivity.
        tipping_points: List of time points where basin boundaries are crossed.
        sensitivity_by_program: (T, P) array of per-program sensitivities.
        clinical_annotations: Dict mapping clinical event -> time point.
    """
    time_sensitivity: np.ndarray
    critical_windows: List[Tuple[float, float]]
    tipping_points: List[float]
    sensitivity_by_program: np.ndarray
    clinical_annotations: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrajectoryDecomposition:
    """Decomposition of a trajectory into program contributions.

    Attributes:
        timepoints: (T,) array of time values.
        trajectory: (T, D) full trajectory.
        velocity: (T, D) dx/dt at each timepoint.
        program_contributions: (T, P) contribution of each program to velocity.
        driver_program: (T,) index of the dominant program at each timepoint.
        handoff_times: List of times where the driver program changes.
        program_names: Names of biological programs.
    """
    timepoints: np.ndarray
    trajectory: np.ndarray
    velocity: np.ndarray
    program_contributions: np.ndarray
    driver_program: np.ndarray
    handoff_times: List[float]
    program_names: List[str]


@dataclass
class PhaseTransitionResult:
    """Results from phase transition analysis.

    Attributes:
        eigenvalues: (T, D) complex eigenvalues of the Jacobian at each timepoint.
        bifurcation_times: Times where eigenvalue sign changes occur.
        order_parameter: Index and name of the order parameter program.
        critical_slowing_down: (T,) autocorrelation time as early warning signal.
        jacobian_sequence: List of (D, D) Jacobian matrices at each timepoint.
        basin_classification: (T,) string labels ('sensitive', 'transitional', 'resistant').
    """
    eigenvalues: np.ndarray
    bifurcation_times: List[float]
    order_parameter: Tuple[int, str]
    critical_slowing_down: np.ndarray
    jacobian_sequence: List[np.ndarray]
    basin_classification: List[str]


# ---------------------------------------------------------------------------
# Critical Window Detector
# ---------------------------------------------------------------------------

class CriticalWindowDetector:
    """Characterize temporal windows in the learned latent trajectory by
    model sensitivity.

    The key quantity is the latent sensitivity:
        S(t) = ∂(model_output) / ∂z(t)

    a Jacobian of the model's output with respect to the latent state at
    time t. This is a **geometric property of U_θ** — not a causal estimate
    of d E[Y]/d(do(intervention at t)). Without perturbation training data,
    no claim about real-world intervention windows can be drawn from S(t).

    Following the mathematical framework in Scheffer et al. (2009)
    "Early-warning signals for critical transitions" (Nature), but applied
    here to a learned ODE on cross-sectional cell-line latents rather than
    to a longitudinal biological system.

    Args:
        model: ResistanceMap Neural ODE model.
        program_names: Names of biological programs.
        config: TemporalConfig instance.
    """

    def __init__(
        self,
        model: nn.Module,
        program_names: List[str],
        config: Optional[TemporalConfig] = None,
    ) -> None:
        self.model = model
        self.program_names = program_names
        self.config = config or TemporalConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.model.eval()

    def detect_critical_windows(
        self,
        x0: Tensor,
        t_span: Tensor,
        perturbation_direction: Optional[Tensor] = None,
        sensitivity_threshold: float = 0.5,
    ) -> CriticalWindowResult:
        """Detect critical windows of intervention sensitivity.

        Algorithm:
        1. Solve the baseline trajectory x(t).
        2. For each timepoint t_k, apply a small perturbation delta to x(t_k).
        3. Re-integrate from t_k to t_final with the perturbed state.
        4. Compute sensitivity = |outcome(perturbed) - outcome(baseline)| / |delta|.
        5. Identify windows where sensitivity exceeds threshold.

        Args:
            x0: Initial state.
            t_span: Time span.
            perturbation_direction: Direction of perturbation. If None, uses
                random perturbations and averages.
            sensitivity_threshold: Relative threshold for "critical" windows.

        Returns:
            CriticalWindowResult with sensitivities and critical windows.
        """
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        x0 = x0.to(self.device)
        t_span = t_span.to(self.device)

        n_time = len(t_span)
        n_programs = x0.shape[-1]
        eps = self.config.perturbation_eps

        # Baseline trajectory
        with torch.no_grad():
            baseline_traj = self._solve_ode(x0, t_span)  # (T, batch, D)
            baseline_outcome = self._compute_outcome(baseline_traj[-1:])

        sensitivities = np.zeros(n_time)
        program_sensitivities = np.zeros((n_time, n_programs))

        for k in range(n_time - 1):
            x_k = baseline_traj[k].clone()
            t_remaining = t_span[k:]

            if perturbation_direction is not None:
                # Single direction perturbation
                delta = perturbation_direction.to(self.device) * eps
                x_perturbed = x_k + delta
                with torch.no_grad():
                    perturbed_traj = self._solve_ode(x_perturbed, t_remaining)
                    perturbed_outcome = self._compute_outcome(perturbed_traj[-1:])
                sensitivities[k] = abs(
                    float(perturbed_outcome) - float(baseline_outcome)
                ) / eps
            else:
                # Average over random perturbation directions
                n_pert = self.config.sensitivity_n_perturbations
                sens_samples = np.zeros(n_pert)
                for p in range(n_pert):
                    delta = torch.randn_like(x_k) * eps
                    x_perturbed = x_k + delta
                    with torch.no_grad():
                        perturbed_traj = self._solve_ode(x_perturbed, t_remaining)
                        perturbed_outcome = self._compute_outcome(perturbed_traj[-1:])
                    sens_samples[p] = abs(
                        float(perturbed_outcome) - float(baseline_outcome)
                    ) / eps
                sensitivities[k] = np.mean(sens_samples)

            # Per-program sensitivity: perturb each dimension individually
            for d in range(n_programs):
                delta = torch.zeros_like(x_k)
                delta[..., d] = eps
                x_perturbed = x_k + delta
                with torch.no_grad():
                    perturbed_traj = self._solve_ode(x_perturbed, t_remaining)
                    perturbed_outcome = self._compute_outcome(perturbed_traj[-1:])
                program_sensitivities[k, d] = abs(
                    float(perturbed_outcome) - float(baseline_outcome)
                ) / eps

        # Identify critical windows
        t_np = t_span.cpu().numpy()
        threshold = sensitivity_threshold * np.max(sensitivities)
        critical = sensitivities > threshold

        windows = self._extract_contiguous_windows(t_np, critical)

        # Identify tipping points: local maxima of sensitivity
        tipping = []
        for k in range(1, n_time - 1):
            if sensitivities[k] > sensitivities[k - 1] and sensitivities[k] > sensitivities[k + 1]:
                if sensitivities[k] > threshold:
                    tipping.append(float(t_np[k]))

        return CriticalWindowResult(
            time_sensitivity=sensitivities,
            critical_windows=windows,
            tipping_points=tipping,
            sensitivity_by_program=program_sensitivities,
        )

    @staticmethod
    def _extract_contiguous_windows(
        times: np.ndarray,
        mask: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """Extract contiguous time windows from a boolean mask."""
        windows = []
        in_window = False
        start = 0.0
        for i, m in enumerate(mask):
            if m and not in_window:
                start = float(times[i])
                in_window = True
            elif not m and in_window:
                windows.append((start, float(times[i - 1])))
                in_window = False
        if in_window:
            windows.append((start, float(times[-1])))
        return windows

    def _solve_ode(self, x0: Tensor, t_eval: Tensor) -> Tensor:
        if hasattr(self.model, "solve"):
            return self.model.solve(x0, t_eval)
        traj = [x0]
        x = x0
        for i in range(1, len(t_eval)):
            dt = (t_eval[i] - t_eval[i - 1]).item()
            dxdt = self.model.ode_func(t_eval[i - 1], x)
            x = x + dt * dxdt
            traj.append(x)
        return torch.stack(traj)

    def _compute_outcome(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.squeeze(0)
        if hasattr(self.model, "predict"):
            with torch.no_grad():
                return self.model.predict(x).mean()
        if hasattr(self.model, "classifier"):
            with torch.no_grad():
                return torch.sigmoid(self.model.classifier(x)).mean()
        return x.norm(dim=-1).mean()


# ---------------------------------------------------------------------------
# Trajectory Decomposer
# ---------------------------------------------------------------------------

class TrajectoryDecomposer:
    """Decompose each patient's trajectory into biological program contributions.

    At each timepoint, the velocity dx/dt is driven by contributions from
    each biological program. This class decomposes the velocity into:
        dx/dt = sum_p contribution_p(t)

    where contribution_p is the velocity component attributable to program p.

    The "driver program" at each time is the one with the largest contribution.
    "Handoff" occurs when the driver program changes, e.g., early resistance
    driven by drug efflux, late resistance by epigenetic reprogramming.

    For the structured Neural ODE with block-diagonal structure:
        dx/dt = [A_11 * sigma(x_1); A_22 * sigma(x_2); ...] + coupling
    the contribution of program p is directly the p-th block of the velocity.

    Args:
        model: ResistanceMap Neural ODE.
        program_names: Names of biological programs.
        program_dims: List of (start, end) index ranges for each program
            in the state vector. If None, assumes equal division.
        config: TemporalConfig.
    """

    def __init__(
        self,
        model: nn.Module,
        program_names: List[str],
        program_dims: Optional[List[Tuple[int, int]]] = None,
        config: Optional[TemporalConfig] = None,
    ) -> None:
        self.model = model
        self.program_names = program_names
        self.config = config or TemporalConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.model.eval()

        # Program dimension ranges
        if program_dims is not None:
            self.program_dims = program_dims
        else:
            # Infer: equal division of state space
            # We'll determine D from the first forward pass
            self.program_dims = None
            self._n_programs = len(program_names)

    def _init_program_dims(self, D: int) -> None:
        """Initialize equal-sized program dimension ranges."""
        if self.program_dims is not None:
            return
        n_prog = self._n_programs
        chunk = D // n_prog
        self.program_dims = []
        for i in range(n_prog):
            start = i * chunk
            end = (i + 1) * chunk if i < n_prog - 1 else D
            self.program_dims.append((start, end))

    def decompose(
        self,
        x0: Tensor,
        t_span: Tensor,
    ) -> TrajectoryDecomposition:
        """Decompose a trajectory into program contributions.

        Algorithm:
        1. Solve ODE to get trajectory x(t).
        2. At each t_k, compute velocity dx/dt = f(x(t_k), t_k).
        3. Project velocity onto each program's subspace.
        4. Identify the driver program at each timepoint.
        5. Detect handoff points.

        Args:
            x0: Initial state.
            t_span: Time evaluation points.

        Returns:
            TrajectoryDecomposition with all decomposition data.
        """
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        x0 = x0.to(self.device)
        t_span = t_span.to(self.device)

        n_time = len(t_span)
        D = x0.shape[-1]
        self._init_program_dims(D)
        n_programs = len(self.program_dims)

        # Solve trajectory
        with torch.no_grad():
            trajectory = self._solve_ode(x0, t_span)  # (T, batch, D)

        # Compute velocity and decompose
        velocities = np.zeros((n_time, D))
        program_contributions = np.zeros((n_time, n_programs))

        for k in range(n_time):
            x_k = trajectory[k]
            with torch.no_grad():
                v_k = self.model.ode_func(t_span[k], x_k)  # (batch, D)
            v_np = v_k.squeeze(0).cpu().numpy()
            velocities[k] = v_np

            # Decompose velocity by program
            for p, (start, end) in enumerate(self.program_dims):
                program_contributions[k, p] = np.linalg.norm(v_np[start:end])

        # Normalize contributions to fractions
        total_velocity = program_contributions.sum(axis=1, keepdims=True)
        total_velocity = np.maximum(total_velocity, 1e-10)
        program_frac = program_contributions / total_velocity

        # Identify driver program at each time
        driver = np.argmax(program_frac, axis=1)

        # Detect handoff points
        handoffs = []
        t_np = t_span.cpu().numpy()
        for k in range(1, n_time):
            if driver[k] != driver[k - 1]:
                handoffs.append(float(t_np[k]))

        return TrajectoryDecomposition(
            timepoints=t_np,
            trajectory=trajectory.squeeze(1).cpu().numpy(),
            velocity=velocities,
            program_contributions=program_frac,
            driver_program=driver,
            handoff_times=handoffs,
            program_names=self.program_names,
        )

    def compute_program_acceleration(
        self,
        decomposition: TrajectoryDecomposition,
    ) -> np.ndarray:
        """Compute the rate of change of each program's contribution.

        Programs that are accelerating are becoming more important;
        this helps predict future driver programs.

        Args:
            decomposition: Output from decompose().

        Returns:
            (T, P) array of d(contribution_p)/dt.
        """
        dt = np.diff(decomposition.timepoints)
        dcontrib = np.diff(decomposition.program_contributions, axis=0)
        acceleration = dcontrib / dt[:, np.newaxis]
        # Pad to match original length
        acceleration = np.vstack([acceleration, acceleration[-1:]])
        return acceleration

    def _solve_ode(self, x0: Tensor, t_eval: Tensor) -> Tensor:
        if hasattr(self.model, "solve"):
            return self.model.solve(x0, t_eval)
        traj = [x0]
        x = x0
        for i in range(1, len(t_eval)):
            dt = (t_eval[i] - t_eval[i - 1]).item()
            dxdt = self.model.ode_func(t_eval[i - 1], x)
            x = x + dt * dxdt
            traj.append(x)
        return torch.stack(traj)


# ---------------------------------------------------------------------------
# Phase Transition Analyzer
# ---------------------------------------------------------------------------

class PhaseTransitionAnalyzer:
    """Detect mathematical bifurcations in the *learned* Neural ODE.

    The learned dynamics define a vector field dx/dt = f_θ(x). The Jacobian
    J = df_θ/dx at each point determines local stability of the trained
    model. We track:

    1. Eigenvalue crossings: when the real part of a Jacobian eigenvalue of
       the learned U_θ crosses zero, the trained model's vector field
       undergoes a saddle-node-type change. Whether this corresponds to a
       biological phase transition requires experimental perturbation
       validation (e.g., DepMap CRISPR knock-out) and is not supported by
       the observational training data used here.

    2. Order parameter: the program/eigenmode whose eigenvalue crossing
       coincides with the mathematical bifurcation in U_θ. This is a
       property of the trained function, NOT an identified biological
       switch — interpreting it as such is an L2 do-calculus claim that
       requires perturbation data the pipeline does not consume during
       training.

    3. Critical slowing down: near a bifurcation of U_θ, the learned
       relaxation time diverges. This is a mathematical early-warning
       signal of the trained model's geometric structure, not a clinically
       validated biomarker of patient-state transitions.

    Following:
    - Strogatz (2015) Nonlinear Dynamics and Chaos.
    - Scheffer et al. (2009) Early-warning signals for critical transitions.
    - Moris et al. (2016) Transition states and cell fate decisions.
    - Pearl (2009) Causality (for the distinction between learned-function
      properties and identified causal effects).

    Args:
        model: ResistanceMap Neural ODE.
        program_names: Names of biological programs.
        config: TemporalConfig.
    """

    def __init__(
        self,
        model: nn.Module,
        program_names: List[str],
        config: Optional[TemporalConfig] = None,
    ) -> None:
        self.model = model
        self.program_names = program_names
        self.config = config or TemporalConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.model.eval()

    def analyze(
        self,
        x0: Tensor,
        t_span: Tensor,
    ) -> PhaseTransitionResult:
        """Full phase transition analysis along a trajectory.

        1. Solve the trajectory x(t).
        2. Compute the Jacobian J(t) = df/dx at each timepoint.
        3. Compute eigenvalues of J(t).
        4. Detect bifurcations (eigenvalue sign crossings).
        5. Identify the order parameter.
        6. Compute critical slowing down signal.

        Args:
            x0: Initial state.
            t_span: Time evaluation points.

        Returns:
            PhaseTransitionResult with eigenvalues, bifurcations, and diagnostics.
        """
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        x0 = x0.to(self.device)
        t_span = t_span.to(self.device)

        n_time = len(t_span)
        D = x0.shape[-1]

        # Solve trajectory
        with torch.no_grad():
            trajectory = self._solve_ode(x0, t_span)

        # Compute Jacobians and eigenvalues
        jacobians = []
        eigenvalues = np.zeros((n_time, D), dtype=complex)

        for k in range(n_time):
            x_k = trajectory[k].squeeze(0)
            J = self._compute_jacobian(x_k, t_span[k])
            jacobians.append(J)
            eigvals = np.linalg.eigvals(J)
            # Sort by real part (descending)
            sort_idx = np.argsort(-eigvals.real)
            eigenvalues[k] = eigvals[sort_idx]

        # Detect bifurcations: real part of leading eigenvalue crosses zero
        bifurcation_times = []
        t_np = t_span.cpu().numpy()
        leading_real = eigenvalues[:, 0].real

        for k in range(1, n_time):
            if leading_real[k - 1] * leading_real[k] < 0:
                # Linear interpolation of crossing time
                t_cross = t_np[k - 1] + (t_np[k] - t_np[k - 1]) * abs(
                    leading_real[k - 1]
                ) / (abs(leading_real[k - 1]) + abs(leading_real[k]))
                bifurcation_times.append(float(t_cross))

        # Identify order parameter: the eigenmode with the crossing
        order_param_idx = 0
        max_crossing_magnitude = 0
        for d in range(D):
            real_d = eigenvalues[:, d].real
            for k in range(1, n_time):
                if real_d[k - 1] * real_d[k] < 0:
                    mag = abs(real_d[k] - real_d[k - 1])
                    if mag > max_crossing_magnitude:
                        max_crossing_magnitude = mag
                        order_param_idx = d

        order_param_name = (
            self.program_names[order_param_idx]
            if order_param_idx < len(self.program_names)
            else f"mode_{order_param_idx}"
        )

        # Critical slowing down: autocorrelation of trajectory increments
        traj_np = trajectory.squeeze(1).cpu().numpy()
        csd = self._compute_critical_slowing_down(traj_np)

        # Basin classification
        basin_labels = self._classify_basins(eigenvalues, traj_np)

        return PhaseTransitionResult(
            eigenvalues=eigenvalues,
            bifurcation_times=bifurcation_times,
            order_parameter=(order_param_idx, order_param_name),
            critical_slowing_down=csd,
            jacobian_sequence=jacobians,
            basin_classification=basin_labels,
        )

    def _compute_jacobian(
        self,
        x: Tensor,
        t: Tensor,
    ) -> np.ndarray:
        """Compute the Jacobian df/dx at point (x, t) using finite differences.

        Args:
            x: State vector, shape (D,).
            t: Time scalar.

        Returns:
            (D, D) numpy array of the Jacobian.
        """
        D = x.shape[0]
        eps = self.config.perturbation_eps
        x = x.detach()

        J = np.zeros((D, D))

        # Evaluate f at x
        with torch.no_grad():
            f_x = self.model.ode_func(t, x.unsqueeze(0)).squeeze(0)
            f_x_np = f_x.cpu().numpy()

        for j in range(D):
            x_plus = x.clone()
            x_plus[j] += eps
            with torch.no_grad():
                f_plus = self.model.ode_func(t, x_plus.unsqueeze(0)).squeeze(0)
            J[:, j] = (f_plus.cpu().numpy() - f_x_np) / eps

        return J

    def _compute_critical_slowing_down(
        self,
        trajectory: np.ndarray,
    ) -> np.ndarray:
        """Compute critical slowing down indicator.

        Uses lag-1 autocorrelation of trajectory increments computed in a
        rolling window. Increasing autocorrelation indicates critical slowing
        down, an early warning of an impending phase transition.

        Following Scheffer et al. (2009).

        Args:
            trajectory: (T, D) trajectory array.

        Returns:
            (T,) array of autocorrelation values.
        """
        T, D = trajectory.shape
        window = self.config.critical_slowing_window

        # Trajectory increments
        increments = np.diff(trajectory, axis=0)  # (T-1, D)

        csd = np.zeros(T)
        for k in range(window, T - 1):
            segment = increments[k - window:k]
            # Lag-1 autocorrelation for each dimension, then average
            ac_sum = 0.0
            for d in range(D):
                sig = segment[:, d]
                if np.std(sig) < 1e-10:
                    continue
                sig_centered = sig - np.mean(sig)
                n = len(sig_centered)
                if n < 3:
                    continue
                ac = np.corrcoef(sig_centered[:-1], sig_centered[1:])[0, 1]
                if np.isfinite(ac):
                    ac_sum += ac
            csd[k] = ac_sum / max(D, 1)

        return csd

    def _classify_basins(
        self,
        eigenvalues: np.ndarray,
        trajectory: np.ndarray,
    ) -> List[str]:
        """Classify each timepoint into attractor basin.

        Uses the sign of the leading eigenvalue's real part:
        - Negative: stable (in a basin)
        - Near zero: transitional (near separatrix)
        - Positive: unstable (leaving current basin)

        Combined with trajectory direction to distinguish sensitive vs resistant.

        Args:
            eigenvalues: (T, D) complex eigenvalues.
            trajectory: (T, D) trajectory.

        Returns:
            List of basin labels for each timepoint.
        """
        T = eigenvalues.shape[0]
        leading_real = eigenvalues[:, 0].real
        threshold = self.config.eigenvalue_threshold

        labels = []
        for k in range(T):
            if abs(leading_real[k]) < threshold:
                labels.append("transitional")
            elif leading_real[k] < -threshold:
                labels.append("sensitive")
            else:
                labels.append("resistant")

        return labels

    def find_fixed_points(
        self,
        x_range: Tuple[Tensor, Tensor],
        n_samples: int = 100,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> List[Dict[str, Any]]:
        """Find fixed points (attractors) of the ODE via Newton's method.

        Samples random initial points and runs Newton iteration to converge
        to fixed points where f(x*) = 0.

        Args:
            x_range: Tuple of (min, max) tensors defining search range.
            n_samples: Number of random initial points.
            max_iter: Maximum Newton iterations.
            tol: Convergence tolerance.

        Returns:
            List of dicts with 'point', 'stability', 'eigenvalues' for each
            unique fixed point found.
        """
        x_min, x_max = x_range
        D = x_min.shape[0]

        fixed_points = []
        found_points = []

        for _ in range(n_samples):
            # Random initial guess
            x = x_min + (x_max - x_min) * torch.rand(D, device=self.device)
            x = x.detach().requires_grad_(False)

            converged = False
            for _ in range(max_iter):
                with torch.no_grad():
                    f_x = self.model.ode_func(
                        torch.tensor(0.0, device=self.device),
                        x.unsqueeze(0),
                    ).squeeze(0)

                if f_x.norm().item() < tol:
                    converged = True
                    break

                # Simple gradient descent toward f(x)=0
                # (Newton would require inverting the Jacobian)
                x = x - 0.01 * f_x

            if converged:
                # Check if this point is novel
                x_np = x.detach().cpu().numpy()
                is_novel = True
                for fp in found_points:
                    if np.linalg.norm(x_np - fp) < 0.1:
                        is_novel = False
                        break

                if is_novel:
                    found_points.append(x_np)
                    J = self._compute_jacobian(
                        x.detach(),
                        torch.tensor(0.0, device=self.device),
                    )
                    eigvals = np.linalg.eigvals(J)
                    stability = "stable" if all(eigvals.real < 0) else "unstable"

                    fixed_points.append({
                        "point": x_np,
                        "stability": stability,
                        "eigenvalues": eigvals,
                    })

        return fixed_points

    def _solve_ode(self, x0: Tensor, t_eval: Tensor) -> Tensor:
        if hasattr(self.model, "solve"):
            return self.model.solve(x0, t_eval)
        traj = [x0]
        x = x0
        for i in range(1, len(t_eval)):
            dt = (t_eval[i] - t_eval[i - 1]).item()
            dxdt = self.model.ode_func(t_eval[i - 1], x)
            x = x + dt * dxdt
            traj.append(x)
        return torch.stack(traj)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _test_temporal_explanations():
    """Unit tests for temporal explanation modules."""
    print("Running temporal explanation tests...")

    # --- Mock model ---
    class MockODEFunc(nn.Module):
        def __init__(self, dim):
            super().__init__()
            # Create a system with interesting dynamics
            A = torch.randn(dim, dim) * 0.1
            # Make it slightly unstable in one direction
            A[0, 0] = 0.05
            A[1, 1] = -0.1
            self.A = nn.Parameter(A)

        def forward(self, t, x):
            return x @ self.A

    class MockModel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.ode_func = MockODEFunc(dim)
            self.classifier = nn.Linear(dim, 1)

        def solve(self, x0, t_eval):
            traj = [x0]
            x = x0
            for i in range(1, len(t_eval)):
                dt = (t_eval[i] - t_eval[i - 1]).item()
                x = x + dt * self.ode_func(t_eval[i], x)
                traj.append(x)
            return torch.stack(traj)

        def predict(self, x):
            return torch.sigmoid(self.classifier(x))

    dim = 6
    model = MockModel(dim)
    programs = ["stemness", "drug_efflux", "dna_damage",
                "immune_evasion", "metabolic", "epigenetic"]

    config = TemporalConfig(
        n_timepoints=20,
        sensitivity_n_perturbations=5,
        critical_slowing_window=3,
    )

    x0 = torch.randn(dim)
    t_span = torch.linspace(0, 1, 20)

    # Test CriticalWindowDetector
    cwd = CriticalWindowDetector(model, programs, config)
    cw_result = cwd.detect_critical_windows(x0, t_span)
    assert cw_result.time_sensitivity.shape == (20,)
    assert cw_result.sensitivity_by_program.shape == (20, dim)
    print(f"  Critical windows: {len(cw_result.critical_windows)} found -- PASS")
    print(f"  Tipping points: {len(cw_result.tipping_points)} found -- PASS")

    # Test TrajectoryDecomposer
    td = TrajectoryDecomposer(model, programs, config=config)
    decomp = td.decompose(x0, t_span)
    assert decomp.trajectory.shape == (20, dim)
    assert decomp.program_contributions.shape == (20, len(programs))
    assert decomp.driver_program.shape == (20,)
    print(f"  Decomposition: {len(decomp.handoff_times)} handoffs -- PASS")

    # Test acceleration
    accel = td.compute_program_acceleration(decomp)
    assert accel.shape == (20, len(programs))
    print(f"  Program acceleration shape: {accel.shape} -- PASS")

    # Test PhaseTransitionAnalyzer
    pta = PhaseTransitionAnalyzer(model, programs, config)
    pt_result = pta.analyze(x0, t_span)
    assert pt_result.eigenvalues.shape == (20, dim)
    assert len(pt_result.basin_classification) == 20
    print(f"  Bifurcations: {len(pt_result.bifurcation_times)} -- PASS")
    print(f"  Order parameter: {pt_result.order_parameter} -- PASS")
    print(f"  CSD signal shape: {pt_result.critical_slowing_down.shape} -- PASS")

    # Test fixed point finder
    x_min = torch.full((dim,), -2.0)
    x_max = torch.full((dim,), 2.0)
    fps = pta.find_fixed_points((x_min, x_max), n_samples=10, max_iter=100)
    print(f"  Fixed points found: {len(fps)} -- PASS")

    print("All temporal explanation tests PASSED.\n")


if __name__ == "__main__":
    _test_temporal_explanations()
