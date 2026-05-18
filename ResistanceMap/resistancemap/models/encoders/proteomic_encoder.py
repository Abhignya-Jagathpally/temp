"""
resistancemap/models/encoders/proteomic_encoder.py
==================================================
Bulk / RPPA / DIA-MS proteomics encoder.

Uses a Gaussian reconstruction likelihood with learned per-protein variance.
Supports an optional protein-aware bottleneck: when ``use_protein_attention=True``
the encoder routes features through a multi-head self-attention layer over
proteins, which gives the protein-network module a more structured starting
point.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteomicEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_token: int = 64,
        hidden_dims: Tuple[int, ...] = (1024, 256),
        dropout: float = 0.1,
        use_protein_attention: bool = False,
        attention_chunks: int = 16,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_token = d_token
        self.use_protein_attention = use_protein_attention

        if use_protein_attention:
            assert input_dim % attention_chunks == 0, (
                f"input_dim ({input_dim}) must be divisible by attention_chunks "
                f"({attention_chunks}) when use_protein_attention=True"
            )
            self.chunk_size = input_dim // attention_chunks
            self.chunk_proj = nn.Linear(self.chunk_size, 128)
            self.attn = nn.MultiheadAttention(
                embed_dim=128, num_heads=attention_heads, dropout=dropout, batch_first=True
            )
            in_dim = 128 * attention_chunks
        else:
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
        if self.use_protein_attention:
            n, d = x.shape
            chunks = x.view(n, -1, self.chunk_size)         # (N, n_chunks, chunk_size)
            chunks = self.chunk_proj(chunks)                # (N, n_chunks, 128)
            attended, _ = self.attn(chunks, chunks, chunks)
            x = attended.reshape(n, -1)
        z = self.encoder(x)
        if presence_mask is not None:
            z = z * presence_mask.float().unsqueeze(-1)
        return z

    def reconstruct(self, z: torch.Tensor):
        mean = self.decoder_mean(z)
        log_var = self.log_var.expand_as(mean).clamp(-10, 10)
        return mean, log_var


def gaussian_nll(
    target: torch.Tensor,
    params: Tuple[torch.Tensor, torch.Tensor],
    *,
    presence_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mean, log_var = params
    inv_var = (-log_var).exp()
    nll = 0.5 * (log_var + (target - mean).pow(2) * inv_var)
    if presence_mask is not None:
        nll = nll * presence_mask.float().unsqueeze(-1)
    return nll.sum(dim=-1).mean()
