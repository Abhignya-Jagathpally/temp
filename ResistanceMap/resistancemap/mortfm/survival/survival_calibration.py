"""
resistancemap/mortfm/survival/survival_calibration.py
======================================================
Temperature-scaling calibration on the predicted survival curves.

Given out-of-fold predicted survivals ``S_k(z)`` and observed events,
fits a single scalar ``T > 0`` such that the rescaled hazards
``h_T = sigmoid((logit h - log T))`` improve a held-out NLL. This is the
direct generalisation of Guo et al. 2017 temperature scaling to the
discrete-time hazard parameterisation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from resistancemap.mortfm.survival.competing_risk_head import (
    nll_competing_risk, survival_curve_from_hazards,
)


class SurvivalCalibration(nn.Module):
    """Single-parameter temperature scaling on competing-risk hazards.

    Usage::

        calib = SurvivalCalibration().fit(
            hazard=val_hazard, event_bin=val_bin, event_observed=val_obs,
        )
        calibrated_hazard = calib(hazard)
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def fit(
        self,
        hazard: torch.Tensor,
        event_bin: torch.Tensor,
        event_observed: torch.Tensor,
        event_type: Optional[torch.Tensor] = None,
        max_iter: int = 200,
        lr: float = 1e-2,
    ) -> "SurvivalCalibration":
        opt = torch.optim.LBFGS([self.log_temperature], max_iter=max_iter, lr=lr,
                                line_search_fn="strong_wolfe")
        logits = torch.log(hazard.clamp(min=1e-6) / (1 - hazard.clamp(max=1 - 1e-6)))

        def _closure():
            opt.zero_grad()
            adj = torch.sigmoid(logits - self.log_temperature)
            S = survival_curve_from_hazards(adj)
            loss = nll_competing_risk(adj, S, event_bin, event_observed, event_type)
            loss.backward()
            return loss

        opt.step(_closure)
        return self

    def forward(self, hazard: torch.Tensor) -> torch.Tensor:
        logits = torch.log(hazard.clamp(min=1e-6) / (1 - hazard.clamp(max=1 - 1e-6)))
        return torch.sigmoid(logits - self.log_temperature)
