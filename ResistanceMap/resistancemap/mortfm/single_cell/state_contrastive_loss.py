"""
resistancemap/mortfm/single_cell/state_contrastive_loss.py
============================================================
Supervised-contrastive loss (Khosla et al. 2020) over disease stage labels.

For each anchor latent z_i with stage s_i, positives are all other
latents in the batch with the same stage, negatives are all others.
The loss pulls same-stage embeddings together and pushes different-
stage embeddings apart in the latent geometry.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.07,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Khosla et al. 2020 SupCon loss.

    Parameters
    ----------
    z : (B, D) latent embeddings (will be L2-normalised internally)
    labels : (B,) integer/categorical stage labels
    temperature : softmax temperature

    Returns
    -------
    scalar loss
    """
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / max(temperature, eps)               # (B, B)
    B = z.shape[0]
    # Mask out the diagonal (self-similarity).
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e4)
    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
    labels = labels.view(-1, 1)
    pos_mask = labels.eq(labels.t()) & ~self_mask
    n_pos = pos_mask.sum(dim=-1).clamp(min=1)
    loss = -(log_prob * pos_mask).sum(dim=-1) / n_pos
    return loss.mean()
