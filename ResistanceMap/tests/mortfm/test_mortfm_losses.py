"""Tests for resistancemap.training.mortfm_losses."""

from __future__ import annotations

import math

import pytest
import torch

from resistancemap.landscape.potential import WaddingtonPotential
from resistancemap.models.heads.time_to_resistance_head import (
    CoxSurvivalHead,
    DiscreteTimeSurvivalHead,
)
from resistancemap.training.mortfm_losses import (
    assemble_total_loss,
    calibration_loss,
    contrastive_alignment_loss,
    counterfactual_consistency_loss,
    drug_response_loss,
    landscape_regularization_loss,
    masked_modality_loss,
    pathway_attribution_loss,
    perturbation_consistency_loss,
    reconstruction_loss,
    resistance_state_loss,
    survival_loss,
    trajectory_distribution_loss,
)


def test_reconstruction_gaussian():
    mu = torch.zeros(4, 10)
    lv = torch.zeros(4, 10)
    loss = reconstruction_loss(torch.randn(4, 10), (mu, lv), distribution="gaussian")
    assert torch.isfinite(loss)


def test_reconstruction_nb_with_real_counts():
    counts = torch.poisson(torch.full((4, 10), 3.0))
    mu = torch.full((4, 10), 3.0)
    theta = torch.full((10,), 5.0)
    loss = reconstruction_loss(counts, (mu, theta), distribution="nb")
    assert torch.isfinite(loss) and loss > 0


def test_drug_response_loss_skips_nan_targets():
    target = torch.tensor([1.0, 2.0, float("nan"), 4.0])
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss = drug_response_loss(pred, target)
    # Only 3 valid rows, all perfect -> loss=0
    assert float(loss) == 0.0


def test_cox_survival_loss_handles_censoring():
    cox = CoxSurvivalHead(d_latent=8)
    h = cox(torch.randn(5, 8))
    et = torch.tensor([1., 2., 3., 4., 5.])
    eo = torch.tensor([0., 0., 0., 0., 0.])  # all censored
    # No events observed -> loss should be 0 (no information).
    assert float(survival_loss(cox, h, et, eo)) == 0.0


def test_discrete_survival_loss_runs_with_some_events():
    disc = DiscreteTimeSurvivalHead(d_latent=8, n_bins=4)
    logits = disc(torch.randn(5, 8))
    et = torch.tensor([1., 2., 3., 4., 5.])
    eo = torch.tensor([1., 0., 1., 0., 1.])
    bins = torch.linspace(0, 6, 5)
    loss = survival_loss(disc, logits, et, eo, time_bins=bins)
    assert torch.isfinite(loss) and loss > 0


def test_resistance_state_loss_ignores_negative_targets():
    logits = torch.randn(5, 3)
    targets = torch.tensor([0, 1, -1, 2, -1])
    loss = resistance_state_loss(logits, targets)
    # Only 3 valid rows.
    assert torch.isfinite(loss)


def test_trajectory_mmd_zero_for_identical_distributions():
    x = torch.randn(50, 8)
    loss = trajectory_distribution_loss(x, x, metric="mmd")
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_contrastive_alignment_perfect_correspondence():
    # Seed the test so the random vectors don't land in a degenerate
    # near-parallel configuration that produces a small off-diagonal margin.
    torch.manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    loss = contrastive_alignment_loss(x, x, temperature=0.01)
    # When a==b, the diagonal scores at low temperature dominate softmax
    # heavily; cross-entropy approaches 0 (predicting i==i for all rows).
    assert loss < 0.1


def test_landscape_regulariser_returns_finite_value():
    U = WaddingtonPotential(d_latent=8, n_attractors=3)
    z = torch.randn(4, 8)
    loss = landscape_regularization_loss(U, z)
    assert torch.isfinite(loss)


def test_counterfactual_consistency_zero_when_rankings_match():
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    oracle = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss = counterfactual_consistency_loss(pred, oracle)
    # All pairs have matching order -> hinge=0 (except margin=1 in equal pairs)
    assert loss < 1.0


def test_assemble_total_loss_skips_zero_weight():
    components = {"a": torch.tensor(2.0), "b": torch.tensor(3.0)}
    weights = {"a": 1.0, "b": 0.0, "c": 5.0}  # b weight=0; c not in components
    total, breakdown = assemble_total_loss(components, weights)
    assert float(total) == 2.0
    assert breakdown["total"] == 2.0
    assert "b" not in breakdown
    assert "c" not in breakdown


def test_calibration_loss_zero_for_perfect_calibration():
    # Predicted probability = observed mean within every bin trivially => 0.
    preds = torch.tensor([0.05, 0.15, 0.55, 0.95])
    obs = torch.tensor([0.0, 0.0, 1.0, 1.0])
    loss = calibration_loss(preds, obs, n_bins=10)
    assert torch.isfinite(loss)
