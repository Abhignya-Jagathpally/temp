"""
resistancemap/models/dynamics/treatment_conditioned_dynamics.py
===============================================================
Treatment-conditioned drift term.

Given a current latent ``z(t)`` and a drug token ``d`` (or a multi-drug list
for combination therapy), produces a drug-specific drift contribution. This
is the term that lets the same baseline latent evolve toward *different*
resistance basins under different therapies.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TreatmentConditionedDynamics(nn.Module):
    def __init__(self, d_latent: int, drug_dim: int, hidden: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.drug_dim = drug_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_latent + drug_dim + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_latent),
        )

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        drug: Optional[torch.Tensor],
    ) -> torch.Tensor:
        N = z.shape[0]
        if drug is None:
            drug = z.new_zeros(N, self.drug_dim)
        t_in = t.expand(N, 1) if t.ndim == 0 else (t.unsqueeze(-1) if t.ndim == 1 else t)
        return self.mlp(torch.cat([z, drug, t_in.to(z.dtype)], dim=-1))
