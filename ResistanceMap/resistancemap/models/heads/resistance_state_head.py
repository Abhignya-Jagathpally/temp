"""
resistancemap/models/heads/resistance_state_head.py
===================================================
Classifies the predicted future resistance phenotype.

Standard class set (configurable):
    0 = sensitive
    1 = drug-tolerant persister
    2 = MRD-like (minimal residual disease)
    3 = relapsed-resistant
    4..K = drug-specific resistant subtypes

The labels come from clinical adjudication or unsupervised basin assignment
in the Waddington landscape (see :mod:`resistancemap.landscape.attractor_basin`).
The head itself is a small MLP — the heavy lifting is upstream in the
foundation encoder and dynamics.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResistanceStateHead(nn.Module):
    def __init__(
        self,
        d_latent: int,
        n_states: int = 4,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_states = n_states
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_states),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns class logits of shape ``(N, n_states)``."""
        return self.net(z)

    @staticmethod
    def loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        ignore_index: int = -100,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits, targets, weight=class_weights, ignore_index=ignore_index
        )
