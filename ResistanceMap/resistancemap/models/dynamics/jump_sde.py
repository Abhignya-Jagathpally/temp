"""
resistancemap/models/dynamics/jump_sde.py
=========================================
Neural Jump-SDE for discontinuous resistance events.

Resistance can emerge through *jumps* — clonal expansion, sudden copy-number
event, therapy switch, epigenetic locking. A pure SDE cannot represent these
because its sample paths are continuous a.s.

This module adds a compound Poisson jump term:

  dz_t = drift(z_t) dt + diffusion(z_t) dW_t + jump(z_t) dN_t

where N_t is a Poisson process whose intensity ``lambda_theta(z_t, drug)`` is
learned. The jump size ``jump_theta(z_t, drug)`` is a learned vector field.

The implementation is an Euler-Maruyama integrator with thinning for the
Poisson process — explicit and torch-only, no extra dependencies.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from resistancemap.models.dynamics.neural_sde import GraphConditionedNeuralSDE


class _JumpHead(nn.Module):
    def __init__(self, d_latent: int, drug_dim: int = 0, hidden: int = 64) -> None:
        super().__init__()
        self.drug_dim = drug_dim
        in_dim = d_latent + drug_dim
        self.intensity_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        self.jump_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, d_latent),
        )

    def forward(self, z: torch.Tensor, drug: Optional[torch.Tensor]):
        if drug is None:
            drug = z.new_zeros(z.shape[0], self.drug_dim)
        x = torch.cat([z, drug], dim=-1) if self.drug_dim > 0 else z
        # Intensity must be positive.
        intensity = nn.functional.softplus(self.intensity_net(x)).squeeze(-1)
        jump = self.jump_net(x)
        return intensity, jump


class NeuralJumpSDE(nn.Module):
    """SDE + compound Poisson jumps.

    Wraps a :class:`GraphConditionedNeuralSDE` and adds a per-step jump test.
    Jumps are sampled by thinning: draw ``u ~ Uniform(0, 1)``; if
    ``u < 1 - exp(-lambda * dt)``, apply the jump.
    """

    def __init__(
        self,
        d_latent: int,
        *,
        sde: Optional[GraphConditionedNeuralSDE] = None,
        drug_dim: int = 0,
        n_graph_nodes: int = 0,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.sde = sde or GraphConditionedNeuralSDE(
            d_latent=d_latent, drug_dim=drug_dim, n_graph_nodes=n_graph_nodes,
        )
        self.jump_head = _JumpHead(d_latent, drug_dim=drug_dim)

    def forward(
        self,
        z0: torch.Tensor,
        time_grid: torch.Tensor,
        *,
        n_samples: int = 1,
        drug: Optional[torch.Tensor] = None,
        graph: Any = None,
    ) -> torch.Tensor:
        """Integrate the jump-SDE with explicit Euler-Maruyama + thinning."""
        paths = []
        for _ in range(n_samples):
            z = z0
            states = [z]
            for i in range(len(time_grid) - 1):
                t0 = time_grid[i]
                t1 = time_grid[i + 1]
                dt = (t1 - t0).to(z.dtype)
                drift = self.sde.drift_fn(z, t0, drug=drug, graph=graph)
                sigma = self.sde.diffusion_fn(z, drug=drug)
                noise = torch.randn_like(z)
                z = z + drift * dt + sigma * noise * dt.abs().sqrt()

                # Jump test.
                lam, jump_vec = self.jump_head(z, drug)
                u = torch.rand_like(lam)
                p_jump = 1.0 - torch.exp(-lam * dt.abs())
                mask = (u < p_jump).float().unsqueeze(-1)
                z = z + mask * jump_vec

                states.append(z)
            paths.append(torch.stack(states, dim=0))
        return torch.stack(paths, dim=0)
