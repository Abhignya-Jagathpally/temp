"""
resistancemap/models/heads/trajectory_head.py
=============================================
Trajectory distribution head.

Predicts the *distribution* of future molecular states, not a point estimate.
Output is a per-time-bin diagonal Gaussian: ``(mean, log_var)`` for each
``z(t)`` in the rollout. The :class:`~resistancemap.training.mortfm_losses`
module pairs this with Wasserstein-2 or MMD against the observed future
distribution (when ``MORTBatch.future_state`` is present).

The head can also sample from its predictive distribution at inference time
via :meth:`sample`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryDistributionHead(nn.Module):
    def __init__(self, d_latent: int, d_output: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_output),
        )
        self.log_var_net = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_output),
        )

    def forward(self, z_path: torch.Tensor):
        """Returns ``(mean, log_var)`` matched to ``z_path`` shape (preserves leading dims)."""
        mean = self.mean_net(z_path)
        log_var = self.log_var_net(z_path).clamp(-10, 10)
        return mean, log_var

    def sample(self, z_path: torch.Tensor, *, n_samples: int = 16) -> torch.Tensor:
        mean, log_var = self.forward(z_path)
        std = (0.5 * log_var).exp()
        eps = torch.randn(n_samples, *mean.shape, device=mean.device, dtype=mean.dtype)
        return mean.unsqueeze(0) + std.unsqueeze(0) * eps
