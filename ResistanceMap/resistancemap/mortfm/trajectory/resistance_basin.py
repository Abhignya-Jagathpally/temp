"""
resistancemap/mortfm/trajectory/resistance_basin.py
====================================================
Resistance-basin assignment + soft probabilities.

A basin B_k is parameterised by a centroid c_k and a (learned) width
gamma_k. The basin assignment at time t is computed as a softmax over
negative squared distances from each centroid:

    p_k(z_t) = softmax_k(-|| z_t - c_k ||^2 / gamma_k^2)

The basin centroids are tied to the WaddingtonPotential's centroids when
the same instance is used for both — see GraphEnergyResistanceSDE.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ResistanceBasin(nn.Module):
    """Soft assignment of latent states to K resistance basins."""

    def __init__(
        self,
        d_latent: int,
        n_basins: int = 5,
        basin_names: Optional[list[str]] = None,
        share_centroids_with: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.n_basins = n_basins
        self.basin_names = basin_names or [
            "sensitive", "persister", "MRD_like", "relapse_like", "drug_specific_resistant",
        ][:n_basins]
        if share_centroids_with is not None and hasattr(share_centroids_with, "basin_centroids"):
            self.basin_centroids = share_centroids_with.basin_centroids
            self.basin_width = share_centroids_with.basin_width
            self._shared = True
        else:
            self.basin_centroids = nn.Parameter(torch.randn(n_basins, d_latent) * 0.3)
            self.basin_width = nn.Parameter(torch.ones(n_basins))
            self._shared = False

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return basin probabilities ``(B, K)`` for latents ``z (B, d_latent)``."""
        diff = z.unsqueeze(1) - self.basin_centroids.unsqueeze(0)
        sqd = (diff ** 2).sum(dim=-1)                       # (B, K)
        widths = torch.clamp(self.basin_width, min=0.1) ** 2
        logits = -sqd / widths.unsqueeze(0)
        return torch.softmax(logits, dim=-1)

    def trajectory_probs(self, z_traj: torch.Tensor) -> torch.Tensor:
        """Apply :meth:`forward` along the time axis.

        ``z_traj`` shape ``(B, T, d_latent)`` -> ``(B, T, K)``.
        """
        B, T, D = z_traj.shape
        p = self.forward(z_traj.reshape(B * T, D))
        return p.reshape(B, T, self.n_basins)
