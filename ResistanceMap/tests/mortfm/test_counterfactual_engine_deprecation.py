"""tests/mortfm/test_counterfactual_engine_deprecation.py

Regression test for Bug B4: the v6-era ``WhatIfAnalyzer`` uses *latent
clamping* which is associational, not interventional (Pearl's do-operator
requires severing incoming edges, which clamping does not do).

To stop downstream users from silently relying on it for causal claims,
both the ``__init__`` and ``inhibit_pathway`` must emit a
``DeprecationWarning`` pointing at the planned Phase-8 simulator.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from resistancemap.interpretability.counterfactual_engine import (
    CounterfactualConfig,
    WhatIfAnalyzer,
)


class _TinyODEModel(nn.Module):
    """Minimal stand-in: provides the ``ode_func`` + ``predict`` surface the
    generator pokes at, with no actual learned dynamics."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim)

    def ode_func(self, t, x):  # noqa: D401 — minimal stub
        return self.linear(x) * 0.0

    def forward(self, t, x):
        return self.ode_func(t, x)

    def predict(self, x):
        return x.mean(dim=-1, keepdim=True)


def test_whatif_init_emits_deprecation_warning():
    model = _TinyODEModel(dim=4)
    cfg = CounterfactualConfig(device="cpu", n_trajectory_samples=2, ode_solver_steps=2)
    with pytest.warns(DeprecationWarning, match="associational, not interventional"):
        WhatIfAnalyzer(model=model, program_names=["a", "b", "c", "d"], config=cfg)


def test_whatif_inhibit_pathway_emits_deprecation_warning():
    model = _TinyODEModel(dim=4)
    cfg = CounterfactualConfig(device="cpu", n_trajectory_samples=2, ode_solver_steps=2)
    # Swallow the __init__-time warning so the test only asserts on the
    # method-call warning below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        analyzer = WhatIfAnalyzer(model=model, program_names=["a", "b", "c", "d"], config=cfg)

    x0 = torch.zeros(1, 4)
    t_span = torch.linspace(0.0, 1.0, 2)
    with pytest.warns(DeprecationWarning, match="associational, not interventional"):
        try:
            analyzer.inhibit_pathway(x0, t_span, "a", inhibition_strength=1.0)
        except Exception:
            # The model stub may not produce a fully usable trajectory; we
            # only care that the warning fires *before* any failure inside.
            pass
