"""
resistancemap/models/heads/pathway_route_head.py
================================================
Pathway / protein attribution head.

Given the foundation latent ``z`` and a list of pathway / protein scores
emitted by the graph branch of the dynamics, this head produces:

* ``protein_scores`` — ``(N, |V|)`` probability that each protein lies on
  the resistance route.
* ``edge_scores`` — ``(N, |E|)`` probability per PPI edge.
* ``pathway_scores`` — ``(N, |P|)`` per-pathway activity score.

Implementation is a learned linear projection + softmax over each axis. The
real semantic content lives in the graph attention upstream — this head is a
*calibration* layer that turns raw attention weights into a normalised
attribution.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathwayRouteHead(nn.Module):
    def __init__(
        self,
        d_latent: int,
        n_proteins: int,
        n_edges: int = 0,
        n_pathways: int = 0,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.n_proteins = n_proteins
        self.n_edges = n_edges
        self.n_pathways = n_pathways

        self.protein_net = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.GELU(), nn.Linear(hidden, n_proteins),
        )
        if n_edges > 0:
            self.edge_net = nn.Sequential(
                nn.Linear(d_latent, hidden), nn.GELU(), nn.Linear(hidden, n_edges),
            )
        else:
            self.edge_net = None
        if n_pathways > 0:
            self.pathway_net = nn.Sequential(
                nn.Linear(d_latent, hidden), nn.GELU(), nn.Linear(hidden, n_pathways),
            )
        else:
            self.pathway_net = None

    def forward(self, z: torch.Tensor):
        protein_scores = torch.sigmoid(self.protein_net(z))
        edge_scores = torch.sigmoid(self.edge_net(z)) if self.edge_net is not None else None
        pathway_scores = (
            torch.softmax(self.pathway_net(z), dim=-1) if self.pathway_net is not None else None
        )
        return protein_scores, edge_scores, pathway_scores

    @staticmethod
    def attribution_loss(
        protein_scores: torch.Tensor,
        positive_targets: Optional[torch.Tensor],
        *,
        sparsity_weight: float = 1e-3,
    ) -> torch.Tensor:
        """Weak-supervision BCE against known target proteins + L1 sparsity.

        ``positive_targets`` is a ``(N, n_proteins)`` 0/1 mask. ``None`` means
        no weak supervision; in that case only the sparsity penalty applies.
        """
        sparsity = sparsity_weight * protein_scores.abs().sum(dim=-1).mean()
        if positive_targets is None:
            return sparsity
        bce = F.binary_cross_entropy(
            protein_scores.clamp(1e-6, 1 - 1e-6), positive_targets.float()
        )
        return bce + sparsity
