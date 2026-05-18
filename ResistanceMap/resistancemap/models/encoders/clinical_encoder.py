"""
resistancemap/models/encoders/clinical_encoder.py
=================================================
Clinical-covariate encoder.

Handles a mix of continuous (age, beta-2 microglobulin, albumin, LDH),
categorical (ISS stage, cytogenetic risk, prior lines), and binary (high-risk
flags) features. Missing values are encoded as zeros with an explicit missing
mask appended as an extra feature channel.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ClinicalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_token: int = 64,
        hidden_dims: Tuple[int, ...] = (64, 64),
        dropout: float = 0.1,
        append_missing_mask: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_token = d_token
        self.append_missing_mask = append_missing_mask

        in_dim = input_dim * (2 if append_missing_mask else 1)
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, d_token))
        self.encoder = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        *,
        presence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.append_missing_mask:
            miss = torch.isnan(x).float()
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x = torch.cat([x, 1.0 - miss], dim=-1)
        z = self.encoder(x)
        if presence_mask is not None:
            z = z * presence_mask.float().unsqueeze(-1)
        return z
