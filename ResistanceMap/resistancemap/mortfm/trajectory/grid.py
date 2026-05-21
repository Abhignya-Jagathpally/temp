"""
resistancemap/mortfm/trajectory/grid.py
========================================
Single source of truth for the canonical MORT-FM time grid (v19 Phase 7).

The v17 codebase carried two divergent grids:
  * ``GraphEnergyResistanceSDE`` defaulted to ``linspace(0, 1.0, 25)``
    (unitless, internal rollout time).
  * ``CompetingRiskHead`` was driven by the trainer with
    ``linspace(0, 12.0, 13)`` (months).
They were never reconciled. The directional-consistency invariant
"high hitting CDF by month 12 implies high hazard by month 12" was
syntactically unevaluable because index ``T-1`` of the SDE referred to
a different absolute time than index ``K-1`` of the survival head.

Phase 7 fixes this by constructing the time grid in ONE place and passing
the same tensor (or a copy thereof) to both subsystems. Everything else
downstream (hitting_time_nll, survival_hitting_consistency_loss,
integrated_brier_score, time-dependent AUC) reads from this grid.

The grid is in MONTHS by convention. Conversion to days happens at the
data-loader boundary (delta_t_days) and is handled by the loss functions
that mix the two units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

__all__ = [
    "canonical_time_grid",
    "time_grid_meta",
    "CanonicalTimeGridConfig",
]


def canonical_time_grid(
    t_max_months: float = 12.0,
    n_steps: int = 25,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return the canonical 1-D time grid covering ``[0, t_max_months]``.

    Parameters
    ----------
    t_max_months :
        Upper edge of the grid in months. Default 12.0, matching
        ``MORTFMConfig.integration_time`` for MMRF-style frontline PFS
        horizons.
    n_steps :
        Number of grid points (inclusive of both endpoints). Default 25,
        matching ``MORTFMConfig.n_time_grid``.
    device, dtype :
        Optional torch placement; default is CPU float32.

    Returns
    -------
    torch.Tensor
        Shape ``(n_steps,)`` covering ``[0, t_max_months]``.

    Raises
    ------
    ValueError
        If ``t_max_months <= 0`` or ``n_steps < 2``.
    """
    if t_max_months <= 0.0:
        raise ValueError(
            f"canonical_time_grid: t_max_months must be positive; got "
            f"{t_max_months}"
        )
    if n_steps < 2:
        raise ValueError(
            f"canonical_time_grid: n_steps must be >= 2; got {n_steps}"
        )
    if dtype is None:
        dtype = torch.float32
    return torch.linspace(0.0, float(t_max_months), int(n_steps),
                          device=device, dtype=dtype)


def time_grid_meta(grid: torch.Tensor) -> Dict[str, Any]:
    """Return ``{"t_max_months", "n_steps", "dt_months"}`` for a 1-D grid."""
    if grid.ndim != 1:
        raise ValueError(
            f"time_grid_meta: grid must be 1-D; got shape {tuple(grid.shape)}"
        )
    n = int(grid.numel())
    if n < 2:
        raise ValueError(
            f"time_grid_meta: grid must have >= 2 points; got {n}"
        )
    t_max = float(grid[-1].item())
    dt = t_max / max(n - 1, 1)
    return {"t_max_months": t_max, "n_steps": n, "dt_months": dt}


@dataclass
class CanonicalTimeGridConfig:
    """Lightweight config object passable to SDE/survival constructors.

    A Pydantic-free dataclass keeps the module's import surface narrow —
    Pydantic isn't a hard dep of the canonical losses package. The fields
    mirror the ``canonical_time_grid`` signature, plus a ``.to_dict``
    helper so calling code can splat it as ``**cfg.to_dict()``.
    """

    t_max_months: float = 12.0
    n_steps: int = 25

    def to_dict(self) -> Dict[str, Any]:
        return {"t_max_months": float(self.t_max_months),
                "n_steps": int(self.n_steps)}

    def build(self, device: Optional[torch.device] = None,
              dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convenience: build the actual grid tensor in one call."""
        return canonical_time_grid(
            t_max_months=self.t_max_months, n_steps=self.n_steps,
            device=device, dtype=dtype,
        )
