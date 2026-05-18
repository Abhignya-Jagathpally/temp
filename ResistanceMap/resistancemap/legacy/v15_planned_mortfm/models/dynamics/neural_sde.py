"""
resistancemap/models/dynamics/neural_sde.py
===========================================
Graph-conditioned Neural SDE on a learned Waddington landscape:

    dz_t = [ -grad U_theta(z_t, drug) + g_theta(z_t, G, drug) ] dt
           + sigma_theta(z_t, drug) dW_t

where U_theta is the learned potential (see
:mod:`resistancemap.landscape.potential`), g_theta is a graph-conditioned
drift correction, and sigma_theta is a state-dependent diffusion.

The "one genotype -> many phenotypes" empirical observation from epigenetic
plasticity motivates the stochastic term: a deterministic ODE cannot produce
multiple terminal states from one initial condition.

Uses ``torchsde.sdeint`` if available, else falls back to Euler-Maruyama.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from resistancemap.legacy.v15_planned_mortfm.models.dynamics.graph_conditioned_drift import GraphConditionedDrift

try:
    import torchsde
    _HAS_TORCHSDE = True
except ImportError:
    _HAS_TORCHSDE = False


class _DiffusionNet(nn.Module):
    """State- and drug-dependent diffusion coefficient ``sigma_theta``."""

    def __init__(self, d_latent: int, drug_dim: int = 0, hidden: int = 64) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.drug_dim = drug_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_latent + drug_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_latent),
        )

    def forward(self, z: torch.Tensor, drug: Optional[torch.Tensor]) -> torch.Tensor:
        if drug is None:
            drug = z.new_zeros(z.shape[0], self.drug_dim)
        x = torch.cat([z, drug], dim=-1) if self.drug_dim > 0 else z
        return torch.nn.functional.softplus(self.mlp(x)) + 1e-3  # ensure positive


def _euler_maruyama_integrate(drift, diffusion, z0, time_grid, n_samples=1):
    """Euler-Maruyama integrator returning ``(S, T, N, d)`` paths."""
    paths = []
    for _ in range(n_samples):
        z = z0
        states = [z]
        for i in range(len(time_grid) - 1):
            t0 = time_grid[i]
            t1 = time_grid[i + 1]
            dt = (t1 - t0).to(z.dtype)
            mu = drift(t0, z)
            sigma = diffusion(z)
            noise = torch.randn_like(z)
            z = z + mu * dt + sigma * noise * dt.abs().sqrt()
            states.append(z)
        paths.append(torch.stack(states, dim=0))
    return torch.stack(paths, dim=0)


class GraphConditionedNeuralSDE(nn.Module):
    """Stochastic latent dynamics over a learned Waddington landscape.

    Parameters
    ----------
    d_latent :
        Latent dimension.
    potential :
        A module implementing ``U(z, drug) -> (N,)`` energy. The drift is then
        ``-grad U + g_theta``. Pass ``None`` to disable the potential term;
        the drift is then purely the graph-conditioned correction.
    graph_drift :
        :class:`GraphConditionedDrift` providing the data-driven correction.
    diffusion :
        Diffusion coefficient module. If None, instantiates a default.
    """

    def __init__(
        self,
        d_latent: int,
        *,
        potential: Optional[nn.Module] = None,
        graph_drift: Optional[GraphConditionedDrift] = None,
        diffusion: Optional[_DiffusionNet] = None,
        drug_dim: int = 0,
        n_graph_nodes: int = 0,
        noise_type: str = "diagonal",  # for torchsde
        sde_type: str = "ito",
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.potential = potential
        self.graph_drift = graph_drift or GraphConditionedDrift(
            d_latent=d_latent, drug_dim=drug_dim, n_graph_nodes=n_graph_nodes,
        )
        self.diffusion = diffusion or _DiffusionNet(d_latent, drug_dim=drug_dim)
        self.noise_type = noise_type
        self.sde_type = sde_type

    def drift_fn(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        *,
        drug: Optional[torch.Tensor] = None,
        graph: Any = None,
    ) -> torch.Tensor:
        g = self.graph_drift(z, t, drug=drug, graph=graph)
        if self.potential is not None:
            # The potential gradient is computed under a *local* enable_grad
            # so that the SDE drift remains well-defined inside no_grad blocks
            # (e.g. during validation / inference). ``create_graph`` is only
            # set when the *outer* context is gradient-enabled — otherwise we
            # would try to build a graph through an inert tensor.
            with torch.enable_grad():
                z_req = z.detach().requires_grad_(True)
                U = self.potential(z_req, drug=drug).sum()
                create_graph = self.training and torch.is_grad_enabled()
                grad_U, = torch.autograd.grad(U, z_req, create_graph=create_graph)
            return g - grad_U.detach() if not create_graph else g - grad_U
        return g

    def diffusion_fn(self, z: torch.Tensor, drug: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.diffusion(z, drug)

    def forward(
        self,
        z0: torch.Tensor,
        time_grid: torch.Tensor,
        *,
        n_samples: int = 1,
        drug: Optional[torch.Tensor] = None,
        graph: Any = None,
    ) -> torch.Tensor:
        """Sample ``S = n_samples`` paths.

        Returns
        -------
        torch.Tensor
            Shape ``(S, T, N, d_latent)``.
        """
        drift = lambda t, z: self.drift_fn(z, t, drug=drug, graph=graph)
        diffusion = lambda z: self.diffusion_fn(z, drug=drug)

        if _HAS_TORCHSDE:
            paths = []
            for _ in range(n_samples):
                class _SDEModule(nn.Module):
                    noise_type = self.noise_type
                    sde_type = self.sde_type

                    def __init__(self_inner):
                        super().__init__()

                    def f(self_inner, t, z):
                        return drift(t, z)

                    def g(self_inner, t, z):
                        sig = diffusion(z)
                        if self.noise_type == "diagonal":
                            return sig
                        return torch.diag_embed(sig)

                sde = _SDEModule()
                z_path = torchsde.sdeint(sde, z0, time_grid.to(z0.dtype))
                paths.append(z_path)
            return torch.stack(paths, dim=0)

        return _euler_maruyama_integrate(drift, diffusion, z0, time_grid, n_samples=n_samples)
