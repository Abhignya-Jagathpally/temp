"""
resistancemap/mortfm/single_cell/pseudotime_regularizer.py
=============================================================
Pseudotime-ordinal regularisation.

For pseudobulks ordered by disease stage HD < MGUS < SMM < MM, we ask
that the latent z preserves the order along a single (learned)
projection direction. Concretely:

    loss = mean( (proj(z_i) - proj(z_j))_+  ) for i, j with stage_i > stage_j

Where ``proj`` is the model's learnable 1-D projection of z. This is
the rank-loss formulation used for ordinal regression — it does NOT
mistakenly treat pseudotime as calendar time.
"""

from __future__ import annotations

import torch


def pseudotime_ordinal_loss(
    proj_scores: torch.Tensor,    # (B,) scalar projections of latents
    stage_labels: torch.Tensor,   # (B,) integer ordinal stage labels
    *,
    margin: float = 0.5,
) -> torch.Tensor:
    """Ranking loss: stage_i > stage_j should imply proj_i > proj_j + margin."""
    B = proj_scores.shape[0]
    if B < 2:
        return proj_scores.new_zeros(())
    score_diff = proj_scores.unsqueeze(1) - proj_scores.unsqueeze(0)  # (B,B)
    label_diff = stage_labels.unsqueeze(1) - stage_labels.unsqueeze(0)  # (B,B)
    # Wherever label_diff > 0, we want score_diff > margin.
    pos = (label_diff > 0).float()
    violation = torch.clamp(margin - score_diff, min=0.0)
    return (pos * violation).sum() / pos.sum().clamp(min=1.0)
