"""Tests for off-policy eval gate and executable guardrails."""
from __future__ import annotations

import numpy as np
import pytest

from resistancemap.models.rl.ope import (
    weighted_importance_sampling, doubly_robust, positivity_check, evaluate,
)
from resistancemap.infrastructure.guardrails import adjudicate, GUARDRAILS


# ---------- OPE ------------------------------------------------------------
def test_wis_matches_on_policy():
    n = 500
    lp = np.log(np.full(n, 0.5))
    r = np.random.default_rng(0).normal(1.0, 0.1, n)
    est = weighted_importance_sampling(lp, lp, r)
    assert abs(est - r.mean()) < 0.05


def test_positivity_check_flags_low_prob():
    lp = np.log(np.array([0.001, 0.5, 0.5]))
    assert positivity_check(lp, min_prob=1e-2) > 0


def test_evaluate_fails_on_positivity_violation():
    lp_b = np.log(np.full(100, 1e-4))          # behaviour never takes these
    lp_t = np.log(np.full(100, 0.5))
    r = np.zeros(100); q = np.zeros(100); v = np.zeros(100)
    res = evaluate(lp_b, lp_t, r, q, v)
    assert not res.passed
    assert "positivity" in res.notes


def test_evaluate_passes_when_agreed():
    n = 200
    lp = np.log(np.full(n, 0.5))
    r = np.ones(n) * 0.7
    q = np.ones(n) * 0.7
    v = np.ones(n) * 0.7
    res = evaluate(lp, lp, r, q, v)
    assert res.passed


# ---------- Guardrails -----------------------------------------------------
def test_probability_range_rule():
    assert adjudicate({"resistance_score": [0.1, 0.9]}) == []
    assert "probability_range" in adjudicate({"resistance_score": [-0.1, 0.5]})


def test_stochastic_ci_rule_blocks_degenerate_interval():
    fails = adjudicate({
        "resistance_score": [0.5],
        "ci_lo": [0.5], "ci_hi": [0.5],    # zero-width interval
    })
    assert "stochastic_ci" in fails


def test_cohort_access_tag_rule():
    fails = adjudicate({
        "resistance_score": [0.5],
        "used_controlled_access": True,
        "public_safe": False,
    })
    assert "cohort_access_tag" in fails


def test_monotone_dose_rule():
    ok = {"resistance_score": [0.1], "dose_response_curves":
          [{"viability": [1.0, 0.8, 0.5, 0.2]}]}
    bad = {"resistance_score": [0.1], "dose_response_curves":
           [{"viability": [0.2, 0.5, 0.8, 1.0]}]}   # rises with dose
    assert "monotone_dose" not in adjudicate(ok)
    assert "monotone_dose" in adjudicate(bad)
