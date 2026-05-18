"""
resistancemap/models/heads/time_to_resistance_head.py
=====================================================
Survival / time-to-resistance heads.

Three flavours, configurable via :class:`MORTFMConfig.survival_kind`:

1. :class:`CoxSurvivalHead` — outputs a scalar log-hazard; loss is Cox partial
   likelihood. Cheap, calibration-friendly, no time discretisation.
2. :class:`DiscreteTimeSurvivalHead` — outputs hazards on a fixed time grid;
   loss is the cross-entropy variant from Gensheimer & Narasimhan 2019. Good
   when you want a survival *curve* not just a ranking.
3. :class:`TimeToResistanceHead` — facade that dispatches to one of the above
   based on a string kind.

All three handle right-censoring correctly via the ``event_observed`` mask.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cox proportional hazards
# ---------------------------------------------------------------------------


class CoxSurvivalHead(nn.Module):
    def __init__(self, d_latent: int, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns log-hazard of shape ``(N,)``."""
        return self.net(z).squeeze(-1)

    @staticmethod
    def loss(
        log_hazard: torch.Tensor,
        event_time: torch.Tensor,
        event_observed: torch.Tensor,
        *,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Cox partial likelihood (Breslow approximation) over the batch.

        Implementation is a straightforward dense version; for large batches
        prefer the sorting-trick implementation in ``pycox``.
        """
        # Filter out NaN event_time rows (unlabelled).
        finite = torch.isfinite(event_time)
        log_hazard = log_hazard[finite]
        event_time = event_time[finite]
        event_observed = event_observed[finite]
        if log_hazard.numel() == 0 or event_observed.sum() == 0:
            return torch.zeros((), device=log_hazard.device, dtype=log_hazard.dtype)
        order = torch.argsort(event_time, descending=True)
        log_h = log_hazard[order]
        e = event_observed[order]
        hazard_ratio = log_h.exp()
        log_risk = (hazard_ratio.cumsum(dim=0) + eps).log()
        uncensored_likelihood = (log_h - log_risk) * e
        return -uncensored_likelihood.sum() / (e.sum() + eps)


# ---------------------------------------------------------------------------
# Discrete-time survival (Gensheimer & Narasimhan)
# ---------------------------------------------------------------------------


class DiscreteTimeSurvivalHead(nn.Module):
    """Predicts per-bin hazards on a discrete time grid.

    Output convention: ``forward(z)`` returns a ``(N, n_bins)`` tensor of
    *hazard logits*; pass through sigmoid to get per-bin hazards in [0, 1].
    The survival curve is then ``S(t_k) = prod_{j<=k} (1 - h_j)``.
    """

    def __init__(self, d_latent: int, n_bins: int = 12, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_bins),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def survival_curve(self, hazard_logits: torch.Tensor) -> torch.Tensor:
        hazards = hazard_logits.sigmoid()
        return torch.cumprod(1.0 - hazards, dim=-1)

    @staticmethod
    def loss(
        hazard_logits: torch.Tensor,
        event_time: torch.Tensor,
        event_observed: torch.Tensor,
        *,
        time_bins: torch.Tensor,
    ) -> torch.Tensor:
        """Gensheimer & Narasimhan 2019 discrete-time NLL.

        ``time_bins`` is a ``(n_bins+1,)`` monotone tensor of bin edges in
        the same units as ``event_time``. Rows with NaN event_time are skipped.
        """
        finite = torch.isfinite(event_time)
        hazard_logits = hazard_logits[finite]
        event_time = event_time[finite]
        event_observed = event_observed[finite]
        if hazard_logits.numel() == 0:
            return torch.zeros((), device=hazard_logits.device, dtype=hazard_logits.dtype)
        N, K = hazard_logits.shape
        # Find bin index for each event_time.
        bin_idx = torch.bucketize(event_time, time_bins[1:], right=False).clamp(max=K - 1)
        # Build a per-row mask of observed bins.
        ar = torch.arange(K, device=hazard_logits.device).unsqueeze(0).expand(N, -1)
        survived_mask = ar < bin_idx.unsqueeze(-1)             # bins before the event
        event_mask = ar == bin_idx.unsqueeze(-1)               # bin of event
        hazards = hazard_logits.sigmoid().clamp(1e-6, 1 - 1e-6)
        log_survive = (1.0 - hazards).log()
        log_event = hazards.log()
        # For event_observed=1: contribute log_event in the event bin + sum log_survive before.
        # For event_observed=0: contribute sum log_survive up to *and including* the bin.
        e = event_observed.unsqueeze(-1)
        ll = survived_mask.float() * log_survive
        ll = ll + event_mask.float() * (e * log_event + (1 - e) * log_survive)
        return -ll.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class TimeToResistanceHead(nn.Module):
    """Survival facade that picks Cox vs discrete-time based on ``kind``."""

    def __init__(
        self,
        d_latent: int,
        *,
        kind: str = "discrete",
        n_bins: int = 12,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kind = kind
        if kind == "cox":
            self.head: nn.Module = CoxSurvivalHead(d_latent, hidden=hidden, dropout=dropout)
        elif kind == "discrete":
            self.head = DiscreteTimeSurvivalHead(d_latent, n_bins=n_bins, hidden=hidden, dropout=dropout)
        else:
            raise ValueError(f"Unknown survival kind {kind!r}; expected cox|discrete.")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)
