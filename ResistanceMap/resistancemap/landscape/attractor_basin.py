"""
resistancemap/landscape/attractor_basin.py
==========================================
Assign cells to attractor basins in a learned Waddington potential.

Two assignment modes:
1. **Nearest-centre** — argmin distance to ``potential.attractor_centres``.
   Fast, deterministic, but cannot resolve cells sitting on barriers.
2. **Gradient-descent rollout** — for each cell, run a short steepest-descent
   trajectory on ``U_theta`` and report which centre it terminates closest to.
   More faithful to "what basin does this cell *belong to*?", but expensive.
"""

from __future__ import annotations

from typing import Optional

import torch

from resistancemap.landscape.potential import WaddingtonPotential


@torch.no_grad()
def assign_basin_nearest(
    z: torch.Tensor,
    potential: WaddingtonPotential,
) -> torch.Tensor:
    """Return ``(N,)`` long tensor of attractor indices nearest to each z."""
    centres = potential.attractor_positions()                  # (K, d)
    diff = z.unsqueeze(1) - centres.unsqueeze(0)               # (N, K, d)
    dist_sq = diff.pow(2).sum(dim=-1)
    return dist_sq.argmin(dim=-1)


def assign_basin_descent(
    z: torch.Tensor,
    potential: WaddingtonPotential,
    *,
    n_steps: int = 50,
    lr: float = 0.05,
    drug: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Steepest-descent rollout assignment.

    Runs ``z_{i+1} = z_i - lr * grad U(z_i)`` for ``n_steps`` then returns
    the nearest-centre index of the final point.
    """
    z = z.clone()
    for _ in range(n_steps):
        z_ = z.detach().requires_grad_(True)
        U = potential(z_, drug=drug).sum()
        g, = torch.autograd.grad(U, z_)
        z = z_ - lr * g
        z = z.detach()
    return assign_basin_nearest(z, potential)
