"""Unit tests for per-drug metrics, baselines, and calibration."""
from __future__ import annotations

import numpy as np
import pytest

from resistancemap.evaluation.metrics.per_drug import (
    per_drug_metrics, pooled_with_ci, require_per_drug_with_pooled,
)
from resistancemap.evaluation.calibration import (
    empirical_coverage, interval_score, conformal_quantile,
    conformal_trajectory_intervals,
)


def test_per_drug_returns_row_per_drug():
    rng = np.random.default_rng(0)
    n = 200
    drug = rng.choice(["a", "b", "c"], n)
    y = rng.integers(0, 2, n)
    p = rng.random(n)
    rows = per_drug_metrics(y, p, drug, n_boot=50)
    assert {r.drug for r in rows} == {"a", "b", "c"}


def test_per_drug_flags_insufficient_samples():
    y = np.array([0, 0, 0, 1, 1])     # only 2 positives
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    d = np.array(["x"] * 5)
    rows = per_drug_metrics(y, p, d, n_boot=10, min_each_class=3)
    assert "insufficient" in rows[0].note


def test_pooled_ci_narrows_with_n():
    rng = np.random.default_rng(0)
    y_small = rng.integers(0, 2, 50); p_small = rng.random(50)
    y_big   = rng.integers(0, 2, 5000); p_big   = rng.random(5000)
    w_small = pooled_with_ci(y_small, p_small, n_boot=200)
    w_big   = pooled_with_ci(y_big,   p_big,   n_boot=200)
    assert (w_small["auroc_hi"] - w_small["auroc_lo"]) > \
           (w_big["auroc_hi"] - w_big["auroc_lo"])


def test_require_per_drug_with_pooled_contract():
    rng = np.random.default_rng(0)
    n = 120
    out = require_per_drug_with_pooled(
        rng.integers(0, 2, n), rng.random(n), rng.choice(["a", "b"], n), n_boot=50
    )
    assert "pooled" in out and "per_drug" in out


# ---------- Calibration -----------------------------------------------------
def test_empirical_coverage_matches_nominal_when_well_calibrated():
    rng = np.random.default_rng(0)
    n = 10_000
    y = rng.normal(size=n)
    # 90% interval around the truth
    lo, hi = y - 1.645, y + 1.645
    cov = empirical_coverage(y, lo, hi, nominal=0.9)
    assert cov["empirical"] == 1.0   # these intervals contain y by construction


def test_interval_score_penalises_miscoverage():
    y = np.array([0.0, 0.0, 0.0])
    tight_miss = interval_score(y, np.array([2, 2, 2.]), np.array([3, 3, 3.]), alpha=0.1)
    tight_cov  = interval_score(y, np.array([-1, -1, -1.]), np.array([1, 1, 1.]), alpha=0.1)
    assert tight_miss > tight_cov


def test_conformal_quantile_monotone_in_alpha():
    res = np.linspace(0, 1, 200)
    q_tight  = conformal_quantile(res, alpha=0.1)   # 90% coverage
    q_loose  = conformal_quantile(res, alpha=0.3)   # 70% coverage
    assert q_tight > q_loose


def test_conformal_intervals_shape():
    pred = np.arange(10, dtype=float)
    cal = np.abs(np.random.default_rng(0).normal(size=100))
    lo, hi = conformal_trajectory_intervals(pred, cal, alpha=0.1)
    assert lo.shape == pred.shape == hi.shape
    assert np.all(hi >= lo)
