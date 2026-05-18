"""
resistancemap/models/heads/uncertainty_head.py
==============================================
Evidential uncertainty head.

For classification: outputs Dirichlet parameters ``alpha_k > 0`` and
decomposes uncertainty into aleatoric + epistemic following Sensoy 2018.

For regression: outputs Normal-Inverse-Gamma parameters ``(gamma, v, alpha, beta)``
(Amini 2020).

The head is intended to replace MC-dropout for cheap calibrated uncertainty.
It is *not* a replacement for the SDE ensemble — those two capture different
flavours of uncertainty (epistemic vs trajectory stochasticity).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialUncertaintyHead(nn.Module):
    def __init__(
        self,
        d_latent: int,
        n_classes: int = 4,
        mode: str = "classification",  # "classification" | "regression"
        hidden: int = 64,
    ) -> None:
        super().__init__()
        assert mode in {"classification", "regression"}
        self.mode = mode
        self.n_classes = n_classes
        out_dim = n_classes if mode == "classification" else 4
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.GELU(), nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor):
        if self.mode == "classification":
            evidence = F.softplus(self.net(z))
            alpha = evidence + 1.0
            return alpha
        else:
            raw = self.net(z)
            gamma = raw[..., 0]
            v = F.softplus(raw[..., 1]) + 1e-6
            alpha = F.softplus(raw[..., 2]) + 1.0 + 1e-6
            beta = F.softplus(raw[..., 3]) + 1e-6
            return gamma, v, alpha, beta

    @staticmethod
    def evidential_classification_loss(
        alpha: torch.Tensor,
        targets: torch.Tensor,
        *,
        kl_weight: float = 0.1,
    ) -> torch.Tensor:
        """Sensoy 2018 evidential CE + KL prior."""
        S = alpha.sum(dim=-1, keepdim=True)
        y = F.one_hot(targets, num_classes=alpha.shape[-1]).float()
        loglik = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1).mean()
        # KL to uniform Dirichlet (alpha = 1).
        kl = (
            torch.lgamma(alpha.sum(dim=-1))
            - torch.lgamma(alpha).sum(dim=-1)
            + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(alpha.sum(dim=-1, keepdim=True)))).sum(dim=-1)
        ).mean()
        return loglik + kl_weight * kl
