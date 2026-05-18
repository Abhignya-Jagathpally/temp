"""
resistancemap/models/dynamics/trajectory_sampler.py
===================================================
High-level wrapper around the dynamics models.

Exposes a unified ``sample_paths(z0, drug, graph, n_samples, time_grid)`` API
that returns:

* deterministic ``mean_path`` of shape ``(T, N, d)``,
* sample ensemble ``z_samples`` of shape ``(S, T, N, d)`` (None for ODE),
* per-time-bin standard deviation ``(T, N, d)`` (None for ODE),
* attractor / basin probabilities — computed lazily by callers using
  :mod:`resistancemap.landscape.attractor_basin`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from resistancemap.models.dynamics.neural_ode import GraphConditionedNeuralODE
from resistancemap.models.dynamics.neural_sde import GraphConditionedNeuralSDE
from resistancemap.models.dynamics.jump_sde import NeuralJumpSDE


@dataclass
class TrajectorySamples:
    mean_path: torch.Tensor                     # (T, N, d)
    samples: Optional[torch.Tensor]             # (S, T, N, d)
    std_path: Optional[torch.Tensor]            # (T, N, d)
    time_grid: torch.Tensor                     # (T,)


class TrajectorySampler(nn.Module):
    """Dispatches to ODE / SDE / Jump-SDE based on ``kind``."""

    def __init__(
        self,
        d_latent: int,
        *,
        kind: str = "neural_sde",
        drug_dim: int = 0,
        n_graph_nodes: int = 0,
        potential: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.kind = kind
        if kind == "neural_ode":
            self.dynamics: nn.Module = GraphConditionedNeuralODE(
                d_latent=d_latent, drug_dim=drug_dim, n_graph_nodes=n_graph_nodes,
            )
        elif kind == "neural_sde":
            self.dynamics = GraphConditionedNeuralSDE(
                d_latent=d_latent, drug_dim=drug_dim,
                n_graph_nodes=n_graph_nodes, potential=potential,
            )
        elif kind == "jump_sde":
            sde = GraphConditionedNeuralSDE(
                d_latent=d_latent, drug_dim=drug_dim,
                n_graph_nodes=n_graph_nodes, potential=potential,
            )
            self.dynamics = NeuralJumpSDE(d_latent=d_latent, sde=sde, drug_dim=drug_dim,
                                          n_graph_nodes=n_graph_nodes)
        else:
            raise ValueError(f"Unknown dynamics kind: {kind!r}; expected neural_ode|neural_sde|jump_sde")

    def sample_paths(
        self,
        z0: torch.Tensor,
        time_grid: torch.Tensor,
        *,
        drug: Optional[torch.Tensor] = None,
        graph: Any = None,
        n_samples: int = 16,
    ) -> TrajectorySamples:
        if self.kind == "neural_ode":
            path = self.dynamics(z0, time_grid, drug=drug, graph=graph)
            return TrajectorySamples(mean_path=path, samples=None, std_path=None, time_grid=time_grid)

        # SDE / Jump-SDE produce S samples
        paths = self.dynamics(z0, time_grid, n_samples=n_samples, drug=drug, graph=graph)
        mean_path = paths.mean(dim=0)
        std_path = paths.std(dim=0)
        return TrajectorySamples(mean_path=mean_path, samples=paths, std_path=std_path, time_grid=time_grid)
