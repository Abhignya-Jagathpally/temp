"""
resistancemap/models/encoders/phosphoproteomic_encoder.py
=========================================================
Phosphosite encoder.

Architecturally similar to :class:`ProteomicEncoder` but with a different
reconstruction prior: phosphosite intensities are sign-preserving (an
increase or decrease in phosphorylation is biologically meaningful), so the
Gaussian reconstruction operates in log-magnitude * sign space which matches
:func:`resistancemap.data.phosphoproteomics_loader.load_phosphoproteomics`'s
output convention.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PhosphoproteomicEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_token: int = 64,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_token = d_token

        in_dim = input_dim
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, d_token))
        self.encoder = nn.Sequential(*layers)

        self.decoder_mean = nn.Linear(d_token, input_dim)
        self.log_var = nn.Parameter(torch.zeros(input_dim))

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

    def reconstruct(self, z: torch.Tensor):
        return self.decoder_mean(z), self.log_var.expand_as(self.decoder_mean(z)).clamp(-10, 10)
