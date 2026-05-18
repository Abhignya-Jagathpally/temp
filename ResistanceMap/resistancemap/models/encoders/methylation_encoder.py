"""
resistancemap/models/encoders/methylation_encoder.py
====================================================
DNA methylation encoder (beta-value or M-value).

For beta values (in [0, 1]), the encoder uses a Beta-distributed
reconstruction likelihood (parameterised by alpha + beta). For M-values,
falls back to a Gaussian likelihood.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MethylationEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_token: int = 64,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        value_kind: str = "beta",  # "beta" | "m_value"
    ) -> None:
        super().__init__()
        assert value_kind in {"beta", "m_value"}
        self.input_dim = input_dim
        self.d_token = d_token
        self.value_kind = value_kind

        in_dim = input_dim
        layers = []
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, d_token))
        self.encoder = nn.Sequential(*layers)

        # For beta: predict (alpha, beta) > 0
        # For m_value: predict (mean, log_var)
        self.decoder_param_a = nn.Linear(d_token, input_dim)
        self.decoder_param_b = nn.Linear(d_token, input_dim)

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
        if self.value_kind == "beta":
            alpha = F.softplus(self.decoder_param_a(z)) + 1e-3
            beta = F.softplus(self.decoder_param_b(z)) + 1e-3
            return alpha, beta
        else:
            mean = self.decoder_param_a(z)
            log_var = self.decoder_param_b(z).clamp(-10, 10)
            return mean, log_var


def methylation_nll(
    target: torch.Tensor,
    params: Tuple[torch.Tensor, torch.Tensor],
    *,
    value_kind: str = "beta",
    presence_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-4,
) -> torch.Tensor:
    if value_kind == "beta":
        alpha, beta = params
        target = target.clamp(eps, 1.0 - eps)
        dist = torch.distributions.Beta(alpha, beta)
        nll = -dist.log_prob(target)
    else:
        mean, log_var = params
        nll = 0.5 * (log_var + (target - mean).pow(2) / log_var.exp())
    if presence_mask is not None:
        nll = nll * presence_mask.float().unsqueeze(-1)
    return nll.sum(dim=-1).mean()
