"""
tests/mortfm/test_canonical_forward.py
========================================
v18.1 — contract test that asserts MORTFM has exactly ONE official
patient-level forward path:

  * ``model(batch)`` returns the LENS dict (z_traj, hazard, basin_probs,
    hitting_cdf, ...) — the canonical v18 surface
  * ``model.forward_legacy(batch, ...)`` is preserved for backward
    compat with v15-v17 checkpoints but is NOT the default
  * ``model.forward_foundation(batch)`` returns just the FoundationState
    for stages A/B/C
"""

from __future__ import annotations

import torch

from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import MORTBatch, MORTFMConfig


def _minimal_batch(d_latent: int = 64, B: int = 4) -> MORTBatch:
    return MORTBatch(
        rna=torch.zeros(B, d_latent, dtype=torch.float32),
        drug=torch.zeros(B, 4),
        modality_mask={
            "rna": torch.ones(B, dtype=torch.bool),
            "drug": torch.ones(B, dtype=torch.bool),
        },
        patient_ids=[f"p{i}" for i in range(B)],
        time=torch.zeros(B),
    )


def test_canonical_forward_returns_lens_dict():
    cfg = MORTFMConfig(
        rna_input_dim=64, d_token=32, d_latent=64,
        n_time_grid=6, integration_time=1.0,
        use_rna=True, use_drug=False,
        survival_n_bins=6,
    )
    model = MORTFM(cfg)
    model.eval()
    batch = _minimal_batch(d_latent=64, B=3)
    with torch.no_grad():
        out = model(batch)
    assert isinstance(out, dict), "canonical forward() must return a dict"
    for key in ("z_traj", "z_samples", "hazard", "survival_curve",
                "basin_probs", "hitting_cdf", "hitting_mean_tau",
                "hitting_frac_hit"):
        assert key in out, f"canonical forward() missing key {key}"
    # Shape sanity
    assert out["z_traj"].shape == (3, cfg.n_time_grid, 64)
    assert out["basin_probs"].shape[:2] == (3, cfg.n_time_grid)


def test_forward_legacy_preserved():
    cfg = MORTFMConfig(
        rna_input_dim=64, d_token=32, d_latent=64,
        n_time_grid=6, integration_time=1.0,
        use_rna=True, use_drug=False,
        survival_n_bins=6,
    )
    model = MORTFM(cfg)
    model.eval()
    batch = _minimal_batch(d_latent=64, B=3)
    with torch.no_grad():
        legacy = model.forward_legacy(batch)
    assert hasattr(legacy, "z_path"), "forward_legacy() must still return TrajectoryPrediction"
    assert legacy.z_path.shape[0] == cfg.n_time_grid


def test_forward_foundation_encode_only():
    cfg = MORTFMConfig(
        rna_input_dim=64, d_token=32, d_latent=64,
        n_time_grid=6, integration_time=1.0,
        use_rna=True, use_drug=False,
        survival_n_bins=6,
    )
    model = MORTFM(cfg)
    model.eval()
    batch = _minimal_batch(d_latent=64, B=3)
    with torch.no_grad():
        state = model.forward_foundation(batch)
    assert hasattr(state, "z0"), "forward_foundation() must return FoundationState"
    assert state.z0.shape == (3, 64)


def test_canonical_and_legacy_are_distinct_methods():
    """forward() and forward_legacy() must NOT be the same function."""
    assert MORTFM.forward is not MORTFM.forward_legacy
    # And forward_lens (the v17 name) should be GONE
    assert not hasattr(MORTFM, "forward_lens"), (
        "v18: forward_lens was renamed to forward(); remove the old name."
    )
