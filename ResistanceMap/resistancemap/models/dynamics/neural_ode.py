"""
resistancemap/models/dynamics/neural_ode.py
===========================================
Graph-conditioned Neural ODE: ``dz/dt = f_theta(z, t, drug, graph)``.

Uses ``torchdiffeq.odeint`` if available, else falls back to an explicit RK4
integrator written in pure PyTorch.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from resistancemap.models.dynamics.graph_conditioned_drift import GraphConditionedDrift

try:
    from torchdiffeq import odeint, odeint_adjoint
    _HAS_TORCHDIFFEQ = True
except ImportError:
    _HAS_TORCHDIFFEQ = False


def _rk4_step(f, z, t0, dt):
    k1 = f(t0, z)
    k2 = f(t0 + dt / 2, z + dt / 2 * k1)
    k3 = f(t0 + dt / 2, z + dt / 2 * k2)
    k4 = f(t0 + dt, z + dt * k3)
    return z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _manual_rk4_integrate(f, z0, time_grid):
    states = [z0]
    z = z0
    for i in range(len(time_grid) - 1):
        t0 = time_grid[i]
        t1 = time_grid[i + 1]
        dt = t1 - t0
        z = _rk4_step(f, z, t0, dt)
        states.append(z)
    return torch.stack(states, dim=0)  # (T, N, d)


class GraphConditionedNeuralODE(nn.Module):
    """Wraps :class:`GraphConditionedDrift` as a Neural ODE.

    Parameters
    ----------
    drift :
        The drift module; if None, instantiated from ``d_latent`` defaults.
    solver :
        "rk4" (manual fallback) or any solver string accepted by torchdiffeq
        ("dopri5", "euler", "rk4"). Auto-uses torchdiffeq if installed.
    use_adjoint :
        If True and torchdiffeq is available, use ``odeint_adjoint`` for
        O(1) memory backward. Default False (the latent is small so direct
        autograd is usually fine).
    """

    def __init__(
        self,
        d_latent: int,
        *,
        drift: Optional[GraphConditionedDrift] = None,
        solver: str = "rk4",
        rtol: float = 1e-5,
        atol: float = 1e-7,
        use_adjoint: bool = False,
        drug_dim: int = 0,
        n_graph_nodes: int = 0,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint
        self.drift = drift or GraphConditionedDrift(
            d_latent=d_latent, drug_dim=drug_dim, n_graph_nodes=n_graph_nodes,
        )

    def forward(
        self,
        z0: torch.Tensor,
        time_grid: torch.Tensor,
        *,
        drug: Optional[torch.Tensor] = None,
        graph: Any = None,
    ) -> torch.Tensor:
        """Integrate ``z(t)`` along ``time_grid``.

        Returns
        -------
        torch.Tensor
            Shape ``(T, N, d_latent)`` where ``T = len(time_grid)``.
        """

        def func(t, z):
            return self.drift(z, t, drug=drug, graph=graph)

        if _HAS_TORCHDIFFEQ:
            integrate = odeint_adjoint if self.use_adjoint else odeint
            z_path = integrate(
                func, z0, time_grid.to(z0.dtype),
                method=self.solver, rtol=self.rtol, atol=self.atol,
            )
        else:
            z_path = _manual_rk4_integrate(func, z0, time_grid.to(z0.dtype))
        return z_path
