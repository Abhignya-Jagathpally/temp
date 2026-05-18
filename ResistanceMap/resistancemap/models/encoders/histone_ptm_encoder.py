"""
resistancemap/models/encoders/histone_ptm_encoder.py
====================================================
Histone PTM encoder.

Treats each (channel, gene/region) column as one feature. The encoder is a
small MLP with a Gaussian reconstruction loss. The interpretability layer
(:mod:`resistancemap.interpretability.pathway_route_extractor`) groups
columns back by channel via the ``<channel>__<gene>`` naming convention
documented in :mod:`resistancemap.data.histone_ptm_loader`.

Why this modality matters
-------------------------
The v14 ResistanceMap README identifies HDAC inhibitors (Panobinostat,
Vorinostat, Romidepsin) as the consistent failure case. The proposed
mechanism is that the current encoder has no direct view of histone-mark
state. This encoder gives MORT-FM that view.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class HistonePTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_token: int = 64,
        hidden_dims: Tuple[int, ...] = (256, 128),
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
        return self.decoder(z)
