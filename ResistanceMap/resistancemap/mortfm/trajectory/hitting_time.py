"""
resistancemap/mortfm/trajectory/hitting_time.py
================================================
Estimator for the resistance hitting time

    tau^R = inf{t : z_t in B_R}

using Monte-Carlo samples from :class:`GraphEnergyResistanceSDE`.

Given S sample trajectories, basin probability ``p(z_t in B_R)`` at every
timepoint, and a threshold ``prob_threshold``, hitting_time returns the
empirical CDF ``P(tau^R <= t)`` for each batch element.

This is the survival-curve-style output the resistance_emergence and
longitudinal_trajectory claim gates evaluate against PFS / TT2L /
relapse-time endpoints.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from resistancemap.mortfm.trajectory.resistance_basin import ResistanceBasin


class HittingTime(nn.Module):
    """Compute empirical P(tau^R <= t) for a designated resistance basin."""

    def __init__(
        self,
        basin_module: ResistanceBasin,
        resistant_basin_index: int = 4,
        prob_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.basin = basin_module
        self.resistant_basin_index = resistant_basin_index
        self.prob_threshold = prob_threshold

    def forward(
        self,
        z_samples: torch.Tensor,        # (S, B, T, d_latent)
        t_grid: torch.Tensor,           # (T,)
    ) -> dict:
        """Return per-batch hitting-time statistics.

        Returns
        -------
        dict with keys:
            "cdf"        — (B, T) empirical P(tau^R <= t_k) averaged over S
            "mean_tau"   — (B,)   E[tau^R | tau^R <= T_max] (NaN if never hit)
            "frac_hit"   — (B,)   fraction of MC samples that reach B_R
        """
        S, B, T, D = z_samples.shape
        # Compute basin probability per (S, B, T).
        flat = z_samples.reshape(S * B * T, D)
        probs = self.basin(flat).reshape(S, B, T, -1)         # (S, B, T, K)
        p_resist = probs[..., self.resistant_basin_index]     # (S, B, T)
        # First-hit time per (S, B): earliest t_k where p_resist >= threshold.
        hit_mask = p_resist >= self.prob_threshold            # (S, B, T)
        # cumulative hit mask
        cum_hit = torch.cummax(hit_mask.int(), dim=-1).values.bool()
        cdf = cum_hit.float().mean(dim=0)                     # (B, T)
        # Per-sample first-hit index (or -1 if never)
        first_idx = torch.where(
            hit_mask.any(dim=-1),
            hit_mask.float().argmax(dim=-1),                  # (S, B)
            torch.full((S, B), -1, dtype=torch.long, device=z_samples.device),
        )
        # Mean tau over samples that hit.
        hit_taus_list = []
        frac_hit_list = []
        for b in range(B):
            mask_b = first_idx[:, b] >= 0
            n_hit = int(mask_b.sum().item())
            if n_hit > 0:
                taus = t_grid[first_idx[mask_b, b]]
                hit_taus_list.append(taus.float().mean())
            else:
                hit_taus_list.append(torch.full((), float("nan"), device=z_samples.device))
            frac_hit_list.append(torch.tensor(n_hit / max(S, 1), device=z_samples.device))
        mean_tau = torch.stack(hit_taus_list)
        frac_hit = torch.stack(frac_hit_list)
        return {"cdf": cdf, "mean_tau": mean_tau, "frac_hit": frac_hit, "t_grid": t_grid}
