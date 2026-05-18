"""
resistancemap/models/encoders/rna_encoder.py
============================================
RNA encoder with negative-binomial likelihood + library-size normalisation.

Models scRNA-seq counts as draws from NB(mu_i, theta) where mu_i is the
gene-specific mean and theta is a learned per-gene dispersion. This matches
the assumptions of scVI / MultiVI and is more appropriate than Gaussian for
sparse count data.

The encoder produces an ``(N, d_token)`` latent token plus a presence flag.
The decoder produces (mu, theta, dropout_logits) for the NB reconstruction
loss.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNAEncoder(nn.Module):
    """Negative-binomial encoder for single-cell or pseudobulk RNA counts.

    Parameters
    ----------
    input_dim
        Number of genes.
    d_token
        Output token dimension.
    hidden_dims
        MLP hidden layer sizes.
    dropout
        Dropout probability between hidden layers.
    n_batches
        If > 0, prepend a batch-embedding to the encoder input
        (for technical batch correction).
    use_layernorm
        Apply LayerNorm between hidden layers (more stable for scRNA than
        BatchNorm because batch size varies per cell-line / patient).
    """

    def __init__(
        self,
        input_dim: int,
        d_token: int = 64,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        n_batches: int = 0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_token = d_token
        self.n_batches = n_batches

        in_dim = input_dim + (16 if n_batches > 0 else 0)
        if n_batches > 0:
            self.batch_embed = nn.Embedding(n_batches, 16)
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

        # NB decoder: rate (mu), dispersion (theta), dropout (pi)
        self.decoder_mu = nn.Linear(d_token, input_dim)
        self.log_theta = nn.Parameter(torch.zeros(input_dim))
        self.decoder_pi = nn.Linear(d_token, input_dim)

    def encode(
        self,
        x: torch.Tensor,
        *,
        batch_id: Optional[torch.Tensor] = None,
        presence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``x`` is ``(N, input_dim)`` raw or log-normalised counts."""
        if batch_id is not None and self.n_batches > 0:
            be = self.batch_embed(batch_id)
            x = torch.cat([x, be], dim=-1)
        z = self.encoder(x)
        if presence_mask is not None:
            z = z * presence_mask.float().unsqueeze(-1)
        return z

    def decode(self, z: torch.Tensor, library_size: Optional[torch.Tensor] = None):
        """Return (mu, theta, pi_logits) for NB-ZI reconstruction."""
        rate = F.softplus(self.decoder_mu(z)) + 1e-8
        if library_size is not None:
            rate = rate * library_size.unsqueeze(-1)
        theta = F.softplus(self.log_theta) + 1e-4
        pi_logits = self.decoder_pi(z)
        return rate, theta, pi_logits

    def forward(
        self,
        x: torch.Tensor,
        *,
        batch_id: Optional[torch.Tensor] = None,
        presence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encode(x, batch_id=batch_id, presence_mask=presence_mask)


def nb_negative_log_likelihood(
    counts: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    *,
    presence_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative log-likelihood under NB(mu, theta).

    Parameterised so that ``theta -> inf`` recovers the Poisson limit.
    """
    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + counts * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(counts + 1)
    )
    if presence_mask is not None:
        res = res * presence_mask.float().unsqueeze(-1)
    return -res.sum(dim=-1).mean()
