"""tests/mortfm/test_phase7_canonical_grid.py
=================================================
v19 Phase 7 — shared canonical_time_grid + survival↔hitting consistency.

Tests:
  * canonical_time_grid: shape + endpoints + dt.
  * SDE and survival head wired with the same config produce a t_grid of
    identical shape (and identical values when both buffers are read).
  * hitting_time_nll: closed-form check on a 4-row, 3-bin fixture and a
    monotonicity check when the predicted mass is shifted away from the
    true bin.
  * survival_hitting_consistency_loss: zero when consistent, positive when
    inconsistent.
  * Bug B19: MORTFM(config).lens_sde and MORTFM(config).lens_graph_projector
    agree on d_graph.
"""

from __future__ import annotations

import pytest
import torch

from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import MORTFMConfig
from resistancemap.mortfm.survival.competing_risk_head import CompetingRiskHead
from resistancemap.mortfm.trajectory.graph_energy_sde import GraphEnergyResistanceSDE
from resistancemap.mortfm.trajectory.grid import (
    CanonicalTimeGridConfig,
    canonical_time_grid,
    time_grid_meta,
)
from resistancemap.training.canonical_mortfm_losses import (
    hitting_time_nll,
    survival_hitting_consistency_loss,
)


# --------------------------------------------------------------------------
# canonical_time_grid
# --------------------------------------------------------------------------


def test_canonical_time_grid_default_shape_and_endpoints():
    g = canonical_time_grid()
    assert g.shape == (25,)
    assert g[0].item() == 0.0
    assert g[-1].item() == 12.0


def test_canonical_time_grid_custom_shape_and_endpoints():
    g = canonical_time_grid(t_max_months=6.0, n_steps=13)
    assert g.shape == (13,)
    assert g[0].item() == 0.0
    assert g[-1].item() == 6.0


def test_canonical_time_grid_rejects_bad_args():
    with pytest.raises(ValueError):
        canonical_time_grid(t_max_months=-1.0, n_steps=5)
    with pytest.raises(ValueError):
        canonical_time_grid(t_max_months=12.0, n_steps=1)


def test_time_grid_meta_extracts_dt():
    g = canonical_time_grid(t_max_months=10.0, n_steps=6)
    meta = time_grid_meta(g)
    assert meta["t_max_months"] == 10.0
    assert meta["n_steps"] == 6
    assert pytest.approx(meta["dt_months"]) == 10.0 / 5


# --------------------------------------------------------------------------
# SDE + survival head sharing the grid
# --------------------------------------------------------------------------


def test_sde_and_survival_head_share_grid_shape():
    cfg = CanonicalTimeGridConfig(t_max_months=12.0, n_steps=25)
    sde = GraphEnergyResistanceSDE(
        d_latent=16, d_graph=8, d_drug=4, d_clinical=2,
        n_basins=3, time_grid_config=cfg,
    )
    head = CompetingRiskHead(d_latent=16, time_grid_config=cfg)
    assert sde.t_grid.shape == head.t_grid.shape == (25,)
    assert torch.equal(sde.t_grid, head.t_grid)


def test_sde_without_config_warns_and_uses_legacy_grid():
    with pytest.warns(DeprecationWarning):
        sde = GraphEnergyResistanceSDE(
            d_latent=16, d_graph=8, d_drug=4, d_clinical=2,
            n_basins=3, integration_time=1.0, n_time_grid=8,
        )
    assert sde.t_grid.shape == (8,)
    assert sde.t_grid[0].item() == 0.0
    assert sde.t_grid[-1].item() == 1.0


def test_survival_head_without_config_warns_and_uses_legacy_n_bins():
    with pytest.warns(DeprecationWarning):
        head = CompetingRiskHead(d_latent=16, n_bins=8)
    assert head.n_bins == 8


# --------------------------------------------------------------------------
# hitting_time_nll
# --------------------------------------------------------------------------


def test_hitting_time_nll_low_when_mass_concentrated_at_event_bin():
    """If the predicted CDF jumps from 0 to ~1 exactly at the event bin,
    the NLL is small."""
    t_grid = torch.tensor([0.0, 1.0, 2.0, 3.0])
    # Two observed-event rows, event at bin index 1 and 2 respectively.
    cdf = torch.tensor([
        [0.0, 0.95, 0.99, 1.00],   # row 0: jump at bin 1
        [0.0, 0.05, 0.95, 1.00],   # row 1: jump at bin 2
    ])
    event_time = torch.tensor([1.0, 2.0])
    event_observed = torch.tensor([1.0, 1.0])
    sub = hitting_time_nll(cdf, t_grid, event_time, event_observed)
    assert sub["n_supervised"] == 2
    # log(0.95) ≈ -0.051; log(0.90) ≈ -0.105; mean small.
    assert sub["loss_hit_nll"].item() < 0.5


def test_hitting_time_nll_grows_when_mass_shifted_away_from_event_bin():
    t_grid = torch.tensor([0.0, 1.0, 2.0, 3.0])
    event_time = torch.tensor([2.0, 2.0])
    event_observed = torch.tensor([1.0, 1.0])
    # Correct: large pmf at bin 2.
    cdf_good = torch.tensor([
        [0.0, 0.05, 0.95, 1.00],
        [0.0, 0.05, 0.95, 1.00],
    ])
    # Wrong: mass concentrated at bin 0 — pmf at bin 2 is near zero.
    cdf_bad = torch.tensor([
        [0.9, 0.95, 0.96, 0.97],
        [0.9, 0.95, 0.96, 0.97],
    ])
    nll_good = hitting_time_nll(cdf_good, t_grid, event_time, event_observed)["loss_hit_nll"]
    nll_bad = hitting_time_nll(cdf_bad, t_grid, event_time, event_observed)["loss_hit_nll"]
    assert nll_bad.item() > nll_good.item()


def test_hitting_time_nll_censored_uses_survival_at_t():
    """A censored row should contribute -log(1 - F(t_censor))."""
    t_grid = torch.tensor([0.0, 1.0, 2.0])
    cdf = torch.tensor([[0.0, 0.10, 0.20]])     # P(no event by t_2) = 0.8
    event_time = torch.tensor([2.0])
    event_observed = torch.tensor([0.0])
    sub = hitting_time_nll(cdf, t_grid, event_time, event_observed)
    # -log(0.8) ≈ 0.2231
    assert pytest.approx(sub["loss_hit_nll"].item(), abs=1e-4) == -torch.log(torch.tensor(0.8)).item()


def test_hitting_time_nll_grid_length_mismatch_raises():
    cdf = torch.zeros(2, 3)
    t_grid = torch.tensor([0.0, 1.0, 2.0, 3.0])    # wrong length
    with pytest.raises(ValueError, match="canonical_time_grid"):
        hitting_time_nll(cdf, t_grid, torch.tensor([1.0, 2.0]),
                         torch.tensor([1.0, 1.0]))


# --------------------------------------------------------------------------
# survival_hitting_consistency_loss
# --------------------------------------------------------------------------


def test_consistency_loss_zero_when_consistent():
    # S(t) is small, 1 - F_hit(t) is large => S(t) <= 1 - F_hit(t) holds.
    s = torch.tensor([[0.3, 0.2, 0.1]])
    f_hit = torch.tensor([[0.5, 0.7, 0.85]])     # 1 - f_hit = [0.5, 0.3, 0.15]
    sub = survival_hitting_consistency_loss(s, f_hit)
    assert sub["loss_surv_hit_consistency"].item() == 0.0
    assert sub["n_supervised"] == 1


def test_consistency_loss_positive_when_inconsistent():
    # S(t) = 0.9 while 1 - F_hit(t) = 0.2 — violates the invariant.
    s = torch.tensor([[0.9, 0.9, 0.9]])
    f_hit = torch.tensor([[0.8, 0.8, 0.8]])      # 1 - f_hit = 0.2
    sub = survival_hitting_consistency_loss(s, f_hit)
    assert sub["loss_surv_hit_consistency"].item() > 0.0


def test_consistency_loss_with_competing_risk_correction_relaxes_bound():
    s = torch.tensor([[0.9, 0.9, 0.9]])
    f_hit = torch.tensor([[0.8, 0.8, 0.8]])
    # 1 - F_hit + c_other = 0.2 + 0.8 = 1.0 ≥ S=0.9 → consistent.
    cif_other = torch.tensor([[0.8, 0.8, 0.8]])
    sub = survival_hitting_consistency_loss(s, f_hit,
                                             cif_other_events=cif_other)
    assert sub["loss_surv_hit_consistency"].item() == 0.0


def test_consistency_loss_shape_mismatch_raises():
    s = torch.zeros(2, 5)
    f_hit = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="grids unaligned"):
        survival_hitting_consistency_loss(s, f_hit)


# --------------------------------------------------------------------------
# Bug B19 — MORTFM lens SDE / projector d_graph parity
# --------------------------------------------------------------------------


def _b19_cfg():
    return MORTFMConfig(
        rna_input_dim=100, proteomics_input_dim=50, clinical_input_dim=4,
        d_token=16, d_latent=32, fusion_layers=2, fusion_heads=2,
        fusion_dropout=0.0,
        use_atac=False, use_methylation=False, use_histone_ptm=False,
        use_phosphoproteomics=False, use_clinical=False, use_drug=True,
        n_time_grid=8, integration_time=6.0,
        survival_n_bins=8,
        mixed_precision=False, batch_size=2, drug_embed_dim=8,
    )


def test_b19_sde_and_projector_share_d_graph():
    """The SDE drift's first Linear expects d_latent + d_graph + d_drug +
    d_clinical features; subtracting the others recovers d_graph. The
    projector's last Linear emits out_features = d_graph. They must match."""
    cfg = _b19_cfg()
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5,
                   n_resistance_states=4)
    sde_in = model.lens_sde.drift.net[0].in_features
    # SDE constructor pinned d_drug=8 and d_clinical=5 in model.py.
    sde_d_graph = sde_in - cfg.d_latent - 8 - 5
    proj_out = model.lens_graph_projector.net[-1].out_features
    assert sde_d_graph == proj_out, (
        f"Bug B19 regression: SDE d_graph={sde_d_graph} != projector "
        f"out_features={proj_out}"
    )


def test_b19_canonical_forward_runs_end_to_end_with_shared_grid():
    """If the dims were mismatched the canonical forward would raise on the
    concat in GraphConditionedDrift. Running it end-to-end is the strongest
    Bug-B19 regression test."""
    cfg = _b19_cfg()
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5,
                   n_resistance_states=4)
    B = 3
    # Build a minimal batch the encoder can ingest.
    from resistancemap.mortfm.schemas import MORTBatch
    batch = MORTBatch(
        rna=torch.poisson(torch.full((B, 100), 3.0)),
        proteomics=torch.randn(B, 50),
        drug=torch.zeros(B, 3),
        modality_mask={
            "rna": torch.ones(B, dtype=torch.bool),
            "proteomics": torch.ones(B, dtype=torch.bool),
            "drug": torch.ones(B, dtype=torch.bool),
        },
        patient_ids=[f"P{i}" for i in range(B)],
    )
    with torch.no_grad():
        out = model(batch)
    # Phase 7: SDE and survival head share the canonical grid.
    assert out["z_traj"].shape == (B, cfg.n_time_grid, cfg.d_latent)
    assert out["hitting_cdf"].shape == (B, cfg.n_time_grid)
    assert out["survival_curve"].shape == (B, cfg.n_time_grid)
    assert torch.equal(model.lens_sde.t_grid,
                       model.lens_competing_risk.t_grid)
