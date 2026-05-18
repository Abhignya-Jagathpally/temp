"""
resistancemap/models/encoders/atac_encoder.py
=============================================
scATAC encoder with Bernoulli reconstruction.

The standard convention for scATAC is to binarise peaks (1 = open, 0 = closed)
and model them as independent Bernoulli outputs. Encoder is an MLP that
outputs a token; decoder predicts per-peak open probability.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ATACEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_token: int = 64,
        hidden_dims: Tuple[int, ...] = (1024, 256),
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_token = d_token

        in_dim = input_dim
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, d_token))
        self.encoder = nn.Sequential(*layers)

        self.decoder = nn.Linear(d_token, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        presence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.encoder(x)
        if presence_mask is not None:
            z = z * presence_mask.float().unsqueeze(-1)
        return z

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """Return logits over peaks; pass through sigmoid for probabilities."""
        return self.decoder(z)


def atac_bernoulli_nll(
    peak_targets: torch.Tensor,
    peak_logits: torch.Tensor,
    *,
    presence_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean Bernoulli NLL over peaks. ``peak_targets`` should be {0, 1}."""
    loss = F.binary_cross_entropy_with_logits(peak_logits, peak_targets, reduction="none")
    if presence_mask is not None:
        loss = loss * presence_mask.float().unsqueeze(-1)
    return loss.sum(dim=-1).mean()
