"""
resistancemap/mortfm/trajectory/graph_projector.py
==================================================
Project a patient latent z0 into the graph-embedding space the SDE drift
consumes.

Why: at v15-defd02d, the LENS resistance trainer feeds ``graph_emb = 0``
because Block A's biological-graph encoder is on the cell-line side and
is NOT directly applicable to MMRF patient-level latents (different
feature space). This module bridges that gap with a small learned
projection:

    graph_emb = LatentToGraphProjector(z0)

The projector lets the SDE drift condition on patient-specific graph
context *learned from the latent*, instead of being identical-for-all-
patients. It is NOT a graph neural network; it is a routing layer that
makes the latent's graph-relevant component explicit.

A future revision can replace this with a real GNN that takes the
biological-graph node-features + the patient's RNA. For n=29 with a
64-dim latent that already encodes some of that information, this
projector is the right scope.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentToGraphProjector(nn.Module):
    def __init__(
        self,
        d_latent: int,
        d_graph: int,
        d_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_graph),
        )
        # Layer-norm output so the SDE drift doesn't see exploding scales.
        self.out_norm = nn.LayerNorm(d_graph)

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.net(z0))
