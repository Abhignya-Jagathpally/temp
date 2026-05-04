"""Neural-ODE forward operator over a learned scalar Waddington potential U_θ.

Replaces the v10 single-Euler-step approximation:

    z_pred = z0 - η ∇U(z0)                                  [v10, O(η²) error]

with the actual gradient-flow ODE solved by adaptive Runge-Kutta:

    ż(t) = -∇U(z(t)),    z(0) = z0,    z_pred = z(T)        [v11, O(η^p) p=5]

Mathematical guarantees the v10 operator does NOT have:

(M1) Energy decay (Lyapunov): for any C¹ scalar U_θ,
        dU/dt = ∇U(z) · ż = ∇U · (-∇U) = -|∇U|² ≤ 0
     so U(z(T)) ≤ U(z(0)) for all T ≥ 0, with equality iff ż=0
     everywhere on the trajectory (ż=0 ⟺ ∇U=0 ⟺ z is a critical point).
     The single-Euler-step operator does NOT satisfy this for finite η — it can
     overshoot critical points and increase U.

(M2) Curl-free drift by mathematical identity: ∇ × (-∇U) = 0 for any twice-
     differentiable U (vector calculus identity). The current architecture
     parameterizes U as a scalar field, so the drift vector field is curl-free
     by construction — no projection layer is needed. The v10 F4 gate measures
     curl of the *empirical-minus-model residual*, which is exactly the
     rotational structure in the data that gradient flow cannot capture; not
     the curl of the model itself.

(M3) Convergence to critical points: as T → ∞, z(T) → some z* with ∇U(z*) = 0,
     for any sufficiently regular U_θ with bounded sublevel sets.

Single-Euler vs adaptive RK truncation error per step:
    Euler:   z_{n+1} = z_n + h f(z_n);                local error O(h²)
    Dopri5:  embedded RK 4(5);                        local error O(h^6),
                                                      step size adapted to keep
                                                      |error| ≤ rtol·|z| + atol.

For η = O(1) on a non-trivial U_θ, the Euler approximation can be off by 10-50 %
relative to the true ODE flow; this matters for the F8-DPS gate where the
*direction* of the prior-induced drift is the testable quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from .scalar_potential import ScalarPotential


@dataclass
class NeuralODEFlowConfig:
    method: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-7
    use_adjoint: bool = False
    max_norm: float = 1e3


class _GradFieldRHS(nn.Module):
    """Right-hand side of ż = -∇U(z), wrapped as nn.Module for odeint."""

    def __init__(self, potential: ScalarPotential, max_norm: float):
        super().__init__()
        self.potential = potential
        self.max_norm = max_norm

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            z_in = z.detach().requires_grad_(True)
            u = self.potential.forward(z_in).sum()
            (g,) = torch.autograd.grad(u, z_in, create_graph=False)
        drift = -g
        norm = drift.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(self.max_norm / norm, max=1.0)
        return drift * scale


class NeuralODEFlow(nn.Module):
    """Adaptive Neural-ODE flow on a learned scalar Waddington potential.

    Wraps a frozen :class:`ScalarPotential` and integrates ż = -∇U via
    `torchdiffeq.odeint`. Provides
        * `integrate(z0, t_span)`        -> trajectory at requested timepoints
        * `forward_to_T(z0, T)`          -> z(T) only
        * `energy_decay(z0, T)`          -> (U(z0), U(z(T))) for the Lyapunov check
    """

    def __init__(self, potential: ScalarPotential, cfg: Optional[NeuralODEFlowConfig] = None):
        super().__init__()
        self.cfg = cfg or NeuralODEFlowConfig()
        self.potential = potential
        for p in self.potential.parameters():
            p.requires_grad_(False)
        self.rhs = _GradFieldRHS(potential, max_norm=self.cfg.max_norm)

    def integrate(self, z0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        solver = odeint_adjoint if self.cfg.use_adjoint else odeint
        traj = solver(
            self.rhs,
            z0,
            t_span.to(z0.device),
            method=self.cfg.method,
            rtol=self.cfg.rtol,
            atol=self.cfg.atol,
        )
        return traj

    def forward_to_T(self, z0: torch.Tensor, T: float) -> torch.Tensor:
        t_span = torch.tensor([0.0, float(T)], dtype=z0.dtype, device=z0.device)
        return self.integrate(z0, t_span)[-1]

    def energy_decay(self, z0: torch.Tensor, T: float) -> tuple[float, float]:
        zT = self.forward_to_T(z0, T)
        u0 = float(self.potential.forward(z0).mean().item())
        uT = float(self.potential.forward(zT).mean().item())
        return u0, uT

    def euler_step(self, z0: torch.Tensor, T: float) -> torch.Tensor:
        """v10 single-Euler approximation, kept for direct comparison."""
        with torch.enable_grad():
            z_in = z0.detach().requires_grad_(True)
            u = self.potential.forward(z_in).sum()
            (g,) = torch.autograd.grad(u, z_in)
        return (z0 - T * g).detach()


def truncation_error(
    flow: NeuralODEFlow,
    z0: torch.Tensor,
    T: float,
    n_steps_grid: Sequence[int] = (1, 2, 4, 8, 16, 32),
) -> dict:
    """Compare single-Euler, fixed-step Euler chain, and adaptive Dopri5 at the
    same final time T. Returns ‖z_method − z_dopri5‖ for each method.

    A well-resolved adaptive integration is treated as ground truth; methods that
    converge to it as h→0 confirm the implementation is correct.
    """
    with torch.no_grad():
        z_ref = flow.forward_to_T(z0, T)
        results = {"reference_method": flow.cfg.method, "T": float(T)}
        z_euler1 = flow.euler_step(z0, T)
        results["euler_1step_l2"] = float((z_euler1 - z_ref).norm(dim=-1).mean().item())
        for n in n_steps_grid:
            h = T / n
            z = z0.clone()
            for _ in range(n):
                z = flow.euler_step(z, h)
            err = float((z - z_ref).norm(dim=-1).mean().item())
            results[f"euler_chain_{n}step_l2"] = err
        return results


__all__ = [
    "NeuralODEFlow",
    "NeuralODEFlowConfig",
    "truncation_error",
]
