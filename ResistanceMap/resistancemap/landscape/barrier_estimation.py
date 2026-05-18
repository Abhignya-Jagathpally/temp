"""
resistancemap/landscape/barrier_estimation.py
=============================================
Estimate transition barriers between attractor basins in a Waddington
landscape.

For each ordered pair of attractors ``(k1, k2)``, finds the minimum-energy
point along a straight-line path connecting them and reports
``barrier(k1->k2) = max_t U(mu_k1 + t(mu_k2 - mu_k1))``.

This is a crude approximation (true minimum-energy paths are non-linear), but
it is fast and good enough to rank "which drug raises the barrier from
sensitive -> resistant?".

For a more accurate estimate, use the Nudged Elastic Band (NEB) method —
left as a future-work hook.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from resistancemap.landscape.potential import WaddingtonPotential


@torch.no_grad()
def estimate_pairwise_barriers(
    potential: WaddingtonPotential,
    *,
    n_path_points: int = 50,
    drug: Optional[torch.Tensor] = None,
) -> Dict[Tuple[int, int], float]:
    """For each ordered (k1, k2) pair, return the linear-path barrier height."""
    centres = potential.attractor_positions()                  # (K, d)
    K = centres.shape[0]
    out: Dict[Tuple[int, int], float] = {}
    for k1 in range(K):
        for k2 in range(K):
            if k1 == k2:
                continue
            ts = torch.linspace(0.0, 1.0, n_path_points, device=centres.device).unsqueeze(-1)
            path = (1 - ts) * centres[k1] + ts * centres[k2]
            d = drug.expand(n_path_points, -1) if drug is not None else None
            U = potential(path, drug=d)
            barrier = float(U.max() - U[0])
            out[(k1, k2)] = barrier
    return out


@torch.no_grad()
def estimate_transition_probability(
    barriers: Dict[Tuple[int, int], float],
    temperature: float = 1.0,
) -> Dict[Tuple[int, int], float]:
    """Map barriers to Arrhenius-style hopping probabilities.

    ``P(k1 -> k2) ~ exp(-barrier / temperature)`` (unnormalised).
    """
    return {pair: float(torch.tensor(-b / temperature).exp()) for pair, b in barriers.items()}
