"""
resistancemap/mortfm/survival/competing_risk_head.py
=====================================================
Discrete-time hazard head supporting competing risks.

For a latent ``z`` and a time grid ``[t_1, ..., t_K]`` the head emits, per
event type ``r`` and per time bin ``k``, the conditional hazard

    h_{r,k}(z) = sigmoid(w_r^T z + b_{r,k})

The survival function (event-free probability up to bin k) is then

    S_k(z) = prod_{j <= k} prod_r (1 - h_{r,j}(z))

and the event-r-specific CDF is

    F_{r,k}(z) = sum_{j <= k} h_{r,j}(z) * S_{j-1}(z) * prod_{r' != r}(1 - h_{r',j}(z))

This is the discrete competing-risk likelihood and integrates trivially
with right-censoring.

The head is deliberately lightweight: one linear per event type. The
trajectory module supplies ``z_t`` over time, so the temporal dependence
of the hazard is captured by re-evaluating the head along the SDE
trajectory, not by stacking extra per-bin parameters.
"""

from __future__ import annotations

import warnings
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from resistancemap.mortfm.trajectory.grid import (
    CanonicalTimeGridConfig,
    canonical_time_grid,
)


def discrete_time_grid(t_max_days: float, n_bins: int) -> torch.Tensor:
    """Equal-width bins from 0 to t_max_days inclusive."""
    return torch.linspace(0.0, float(t_max_days), n_bins + 1)


class CompetingRiskHead(nn.Module):
    """Discrete-time hazard head with K competing events.

    Parameters
    ----------
    d_latent
        Latent dimension consumed by the head.
    n_bins
        Number of time bins K.
    event_names
        Optional names for the competing events; default is
        ``["progression"]``. For MM resistance: ``["progression", "death"]``
        gives a competing-risk PFS analysis.
    """

    def __init__(
        self,
        d_latent: int,
        n_bins: int = 8,
        event_names: Optional[List[str]] = None,
        time_grid_config: Optional[CanonicalTimeGridConfig] = None,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.event_names = list(event_names or ["progression"])
        self.n_events = len(self.event_names)

        # v19 Phase 7: when a CanonicalTimeGridConfig is supplied, the
        # number of hazard bins is forced to match the canonical grid so
        # the survival curve and the SDE hitting CDF are computed on the
        # SAME time axis. Without a config we keep the legacy n_bins value
        # but warn — the directional-consistency invariant cannot be
        # evaluated in dual-grid mode.
        if time_grid_config is not None:
            self.n_bins = int(time_grid_config.n_steps)
            grid = time_grid_config.build()
            self.register_buffer("t_grid", grid, persistent=False)
            self._uses_canonical_grid = True
        else:
            warnings.warn(
                "CompetingRiskHead constructed without time_grid_config; "
                f"defaulting to n_bins={n_bins} legacy bins with no shared "
                "time axis. survival_hitting_consistency_loss will be "
                "syntactically invalid because grids will not align.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.n_bins = n_bins
            # No grid buffer to register — downstream code that asks for
            # t_grid will trip a clear AttributeError instead of silently
            # using an arbitrary grid.
            self._uses_canonical_grid = False

        # One linear per event: scores per time bin.
        self.event_heads = nn.ModuleList([
            nn.Linear(d_latent, self.n_bins) for _ in self.event_names
        ])

    def forward(self, z: torch.Tensor) -> dict:
        """Compute conditional hazards + survival curves.

        Parameters
        ----------
        z : (B, d_latent)

        Returns
        -------
        dict with:
          ``hazard``           — (B, K, n_events) sigmoid hazards per bin per event
          ``survival_curve``   — (B, K) overall event-free probability after bin k
          ``cif_per_event``    — (B, K, n_events) cumulative incidence per event
        """
        hs = [torch.sigmoid(head(z)) for head in self.event_heads]
        hazard = torch.stack(hs, dim=-1)                                # (B, K, n_events)
        survival = survival_curve_from_hazards(hazard)                  # (B, K)
        # Cumulative incidence per event: F_{r,k} = sum_{j<=k} h_{r,j} * S_{j-1} * prod_{r'!=r}(1-h_{r',j})
        # Compute incremental hazard contribution per event at each bin.
        per_event_step_haz = hazard.clone()                              # (B,K,R)
        not_other = torch.ones_like(hazard)
        for r in range(hazard.shape[-1]):
            mask = torch.ones(hazard.shape[-1], device=hazard.device)
            mask[r] = 0.0
            not_other[..., r] = torch.prod(1.0 - hazard * mask, dim=-1)
        S_prev = torch.cat([
            torch.ones_like(survival[:, :1]),
            survival[:, :-1],
        ], dim=1)                                                        # (B, K)
        # Broadcast over event dim
        step_cif = per_event_step_haz * S_prev.unsqueeze(-1) * not_other
        cif = torch.cumsum(step_cif, dim=1)                              # (B, K, n_events)
        return {"hazard": hazard, "survival_curve": survival, "cif_per_event": cif}


def survival_curve_from_hazards(hazard: torch.Tensor) -> torch.Tensor:
    """Convert per-bin per-event hazards to overall event-free survival.

    ``hazard``: (B, K, R) sigmoid-conditional hazards.
    Returns ``S_k = prod_{j<=k} prod_r (1 - h_{r,j})``, shape (B, K).
    """
    overall_not_event = torch.prod(1.0 - hazard, dim=-1)                # (B, K)
    return torch.cumprod(overall_not_event, dim=1)


def nll_competing_risk(
    hazard: torch.Tensor,            # (B, K, R)
    survival: torch.Tensor,          # (B, K)
    event_bin: torch.Tensor,         # (B,) int — bin index of event (clipped)
    event_observed: torch.Tensor,    # (B,) float in {0,1}
    event_type: Optional[torch.Tensor] = None,  # (B,) int in [0, R)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Discrete-time competing-risk negative log-likelihood.

    For uncensored patient (i, k_i, r_i):
        L_i = log h_{r_i, k_i} + log S_{k_i - 1}
    For censored patient (i, k_i):
        L_i = log S_{k_i}
    """
    B, K, R = hazard.shape
    device = hazard.device
    if event_type is None:
        event_type = torch.zeros_like(event_bin)
    bins = event_bin.clamp(0, K - 1)
    # log S_{k - 1}
    S_prev = torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)
    logS_at_event_minus_1 = torch.log(S_prev[torch.arange(B, device=device), bins].clamp(min=eps))
    logS_at_event = torch.log(survival[torch.arange(B, device=device), bins].clamp(min=eps))
    h_at_event = hazard[torch.arange(B, device=device), bins, event_type].clamp(min=eps)
    log_h = torch.log(h_at_event)
    # Per-patient NLL
    nll_event = -(log_h + logS_at_event_minus_1)
    nll_censored = -logS_at_event
    obs = event_observed.float()
    nll = obs * nll_event + (1 - obs) * nll_censored
    return nll.mean()
