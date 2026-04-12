"""Known-answer unit tests for resistancemap.evaluation.metrics.

All inputs are tiny handcrafted arrays with analytically known answers.
No synthetic pipeline data is constructed: each test verifies a metric
against an identity, an extremum, or a closed-form value.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# Skip the whole module if the evaluation package is not yet present.
pytest.importorskip(
    "resistancemap.evaluation",
    reason="resistancemap.evaluation package not yet merged into worktree",
)


# ---------------------------------------------------------------------------
# Calibration: expected_calibration_error
# ---------------------------------------------------------------------------


def test_ece_perfectly_calibrated_is_zero():
    """A perfectly calibrated predictor has ECE ~ 0."""
    from resistancemap.evaluation.metrics.calibration import (
        expected_calibration_error,
    )

    # 10 predictions in two bins; the empirical frequency in each bin equals
    # the predicted probability exactly.
    probs = np.array([0.1] * 10 + [0.9] * 10, dtype=float)
    labels = np.array([1] + [0] * 9 + [1] * 9 + [0], dtype=int)
    # bin 0.0-0.5: mean prob 0.1, empirical pos rate = 1/10 = 0.1
    # bin 0.5-1.0: mean prob 0.9, empirical pos rate = 9/10 = 0.9

    ece = expected_calibration_error(probs, labels, n_bins=2)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_ece_worst_case_is_one():
    """Predicting 1.0 when the label is 0 yields ECE = 1.0."""
    from resistancemap.evaluation.metrics.calibration import (
        expected_calibration_error,
    )

    probs = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    labels = np.array([0, 0, 0, 0], dtype=int)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Survival: integrated_brier_score
# ---------------------------------------------------------------------------


def test_integrated_brier_score_known_values():
    """Closed-form Brier score on a 5-sample fixture.

    The integrated Brier score reduces to the average squared error between
    the predicted survival probability and the observed (binary) survival
    indicator at each evaluation time, averaged over the time grid.

    For a single time horizon and squared error this collapses to mean
    squared error.
    """
    pytest.importorskip("scipy")
    from resistancemap.evaluation.metrics.survival import integrated_brier_score

    # 5 samples; ground truth survival indicator at the single horizon t=1:
    times = np.array([0.5, 2.0, 0.5, 2.0, 2.0], dtype=float)
    events = np.array([1, 0, 1, 0, 0], dtype=int)
    # Survival function predictions evaluated at horizons:
    # rows = samples, columns = horizons
    pred_surv = np.array(
        [
            [0.2],
            [0.8],
            [0.2],
            [0.8],
            [0.8],
        ],
        dtype=float,
    )
    horizons = np.array([1.0])

    ibs = integrated_brier_score(times, events, pred_surv, horizons)
    # samples 0,2 dead by t=1 -> indicator 0, prediction 0.2 -> err 0.04
    # samples 1,3,4 alive at t=1 -> indicator 1, prediction 0.8 -> err 0.04
    # mean = 0.04
    assert ibs == pytest.approx(0.04, abs=1e-6)


# ---------------------------------------------------------------------------
# Forecasting: pinball_loss
# ---------------------------------------------------------------------------


def test_pinball_loss_at_median_is_half_mean_abs_error():
    """At tau=0.5 the pinball loss equals 0.5 * mean(|y - q|)."""
    from resistancemap.evaluation.metrics.forecasting import pinball_loss

    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    q = np.array([1.5, 1.5, 3.5, 3.5], dtype=float)
    expected = 0.5 * np.mean(np.abs(y - q))

    actual = pinball_loss(y, q, tau=0.5)
    assert actual == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Pathway stability: attribution_overlap
# ---------------------------------------------------------------------------


def test_attribution_overlap_identical_inputs_is_one():
    from resistancemap.evaluation.metrics.pathway_stability import (
        attribution_overlap,
    )

    a = ["TP53", "KRAS", "NRAS", "MYC"]
    assert attribution_overlap(a, a) == pytest.approx(1.0, abs=1e-9)


def test_attribution_overlap_disjoint_inputs_is_zero():
    from resistancemap.evaluation.metrics.pathway_stability import (
        attribution_overlap,
    )

    a = ["TP53", "KRAS"]
    b = ["BCMA", "CD38"]
    assert attribution_overlap(a, b) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Distribution shift: maximum_mean_discrepancy
# ---------------------------------------------------------------------------


def test_mmd_same_distribution_is_near_zero():
    """MMD between identical samples must be near zero."""
    pytest.importorskip("scipy")
    from resistancemap.evaluation.metrics.shift import maximum_mean_discrepancy

    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 4))
    # Use the same samples for both arguments — MMD is exactly 0 by identity
    # for identical sample sets, regardless of kernel bandwidth.
    mmd = maximum_mean_discrepancy(x, x.copy())
    assert mmd == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Concordance index
# ---------------------------------------------------------------------------


def test_concordance_index_perfect_ranking_is_one():
    pytest.importorskip("scipy")
    from resistancemap.evaluation.metrics.survival import concordance_index

    # Two uncensored samples; longer survival time gets the higher score
    # (lower hazard / higher survival prediction).
    times = np.array([1.0, 2.0])
    events = np.array([1, 1])
    scores = np.array([2.0, 1.0])  # higher score == higher hazard
    c = concordance_index(times, events, scores)
    assert c == pytest.approx(1.0, abs=1e-9)


def test_concordance_index_anti_ranked_is_zero():
    pytest.importorskip("scipy")
    from resistancemap.evaluation.metrics.survival import concordance_index

    times = np.array([1.0, 2.0])
    events = np.array([1, 1])
    scores = np.array([1.0, 2.0])  # inverted hazard ranking
    c = concordance_index(times, events, scores)
    assert c == pytest.approx(0.0, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
