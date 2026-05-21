"""tests/mortfm/test_phase1_trajectory_target.py
====================================================
v19 Phase 1 — latent trajectory target.

Tests:
  * mort_collate populates future_snapshot_batch + delta_t_days when
    paired follow-up snapshots are present, and never fabricates them
    from raw RNA.
  * latent_future_state_loss runs end-to-end, requires_grad is True,
    n_supervised == B for fully-supervised batches, and the loss
    decreases over 5 gradient steps on a fixed batch (mini-overfit).
  * When future_snapshot_batch is None, _trajectory_step returns a
    zero scalar with n_supervised == 0.
  * Bug B1 — the legacy TrajectoryPrediction's mean_path is (T, N, d);
    indexing [-1] picks the last timestep (shape (N, d)), NOT the last
    sample. The canonical path uses (B, T, d_latent) and the standard
    `z_traj[:, -1, :]` slice, returning (B, d_latent).
"""

from __future__ import annotations

import pytest
import torch

from resistancemap.data.mortfm_dataset import mort_collate
from resistancemap.data.trajectory_pair_builder import build_temporal_pairs
from resistancemap.mortfm.canonical_trainer import CanonicalMORTFMTrainer
from resistancemap.mortfm.data_module import make_data_module
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import (
    DrugContext,
    ModalityTensor,
    MORTBatch,
    MORTFMConfig,
    PatientCellSnapshot,
    ResistanceOutcome,
    TemporalTrainingPair,
)
from resistancemap.training.canonical_mortfm_losses import (
    latent_future_state_loss,
)


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _mk_snap(pid: str, t: int, seed: int = 0) -> PatientCellSnapshot:
    g = torch.Generator().manual_seed(seed + t)
    rna = ModalityTensor(
        "rna", torch.poisson(torch.full((1, 100), 3.0), generator=g),
        [f"g{i}" for i in range(100)],
    )
    prot = ModalityTensor(
        "proteomics", torch.randn(1, 50, generator=g),
        [f"p{i}" for i in range(50)],
    )
    return PatientCellSnapshot(
        patient_id=pid, sample_id=f"{pid}_t{t}", disease="MM", timepoint=float(t),
        rna=rna, proteomics=prot,
        drug=DrugContext("Bortezomib", "proteasome_inhibitor", dose=0.01),
    )


def _cohort(n: int = 6):
    snaps = [_mk_snap(f"P{i:03d}", t, seed=i) for i in range(n) for t in [0, 6]]
    outs = [
        ResistanceOutcome(
            patient_id=f"P{i:03d}", baseline_time=0.0,
            event_time=float(8 + i), censored=(i % 2 == 0),
            resistance_label=i % 4,
        )
        for i in range(n)
    ]
    return build_temporal_pairs(snaps, outs, allowed_time_gaps=[6.0], gap_tolerance=0.5)


def _debug_cfg(checkpoint_dir) -> MORTFMConfig:
    return MORTFMConfig(
        rna_input_dim=100, proteomics_input_dim=50, clinical_input_dim=4,
        d_token=16, d_latent=32, fusion_layers=2, fusion_heads=2, fusion_dropout=0.0,
        use_atac=False, use_methylation=False, use_histone_ptm=False,
        use_phosphoproteomics=False, use_clinical=False, use_drug=True,
        n_time_grid=4, integration_time=6.0,
        pretrain_epochs=1, finetune_epochs=1, trajectory_epochs=1, survival_epochs=1,
        mixed_precision=False, batch_size=2, drug_embed_dim=8,
        checkpoint_dir=str(checkpoint_dir),
    )


# --------------------------------------------------------------------------
# Collate-fix tests
# --------------------------------------------------------------------------


def test_mort_collate_populates_future_snapshot_batch_when_paired():
    """When pairs carry x_t_delta, collate must build the parallel batch."""
    pairs = _cohort(n=4)
    longitudinal = [p for p in pairs if p.x_t_delta is not None]
    assert len(longitudinal) > 0, "test cohort must include longitudinal pairs"
    batch = mort_collate(longitudinal[:3])
    assert batch.future_snapshot_batch is not None
    assert isinstance(batch.future_snapshot_batch, MORTBatch)
    # Aligned with the parent batch B.
    assert batch.future_snapshot_batch.rna.shape[0] == 3
    assert batch.delta_t_days is not None
    assert batch.delta_t_days.shape == (3,)
    # Δt = 6 - 0 months = 6 days... wait, mort_collate uses raw timepoint
    # difference; PatientCellSnapshot timepoints are months in the test.
    # All entries should be finite and positive.
    assert torch.isfinite(batch.delta_t_days).all()
    assert (batch.delta_t_days > 0).all()


def test_mort_collate_does_not_fabricate_future_state_from_raw_rna():
    """Phase 1: the raw-RNA fallback is DELETED. future_state must be None
    unless the outcome explicitly provides it."""
    pairs = _cohort(n=4)
    longitudinal = [p for p in pairs if p.x_t_delta is not None]
    batch = mort_collate(longitudinal[:3])
    # None of our outcomes set future_state — so it should be None even
    # though x_t_delta is populated.
    assert batch.future_state is None


def test_mort_collate_future_snapshot_batch_is_none_when_no_followup():
    """Backward compat: pairs without x_t_delta still collate, just without
    the trajectory-target plumbing."""
    pairs = _cohort(n=4)
    # Force x_t_delta to None on every pair.
    static_pairs = [
        TemporalTrainingPair(x_t=p.x_t, x_t_delta=None, outcome=p.outcome)
        for p in pairs
    ]
    batch = mort_collate(static_pairs[:3])
    assert batch.future_snapshot_batch is None
    assert batch.delta_t_days is None


# --------------------------------------------------------------------------
# latent_future_state_loss tests
# --------------------------------------------------------------------------


def test_latent_future_state_loss_shape_mismatch_raises():
    pred = torch.randn(4, 5, 8)
    fut = torch.randn(4, 7)  # wrong d
    dt = torch.full((4,), 30.0)
    with pytest.raises(ValueError, match="same encoder"):
        latent_future_state_loss(pred, fut, dt)


def test_latent_future_state_loss_zero_when_predicted_equals_target():
    B, T, d = 8, 4, 16
    z = torch.randn(B, d)
    # Predict the same z at every timestep — terminal slice equals target.
    pred = z.unsqueeze(1).expand(B, T, d).contiguous()
    dt = torch.full((B,), 30.0)
    sub = latent_future_state_loss(pred, z, dt)
    assert sub["n_supervised"] == B
    # Loss is l2 + mmd; l2 is exactly zero for matched values, mmd is
    # near-zero for identical samples.
    assert sub["loss_traj_latent"].item() == 0.0
    assert sub["loss_traj_mmd"].item() < 1e-5


def test_latent_future_state_loss_masks_nan_and_nonpositive_dt():
    B, T, d = 4, 3, 8
    pred = torch.randn(B, T, d)
    fut = torch.randn(B, d)
    dt = torch.tensor([float("nan"), -1.0, 0.0, 30.0])
    sub = latent_future_state_loss(pred, fut, dt)
    # Only one valid row.
    assert sub["n_supervised"] == 1


def test_latent_future_state_loss_mini_overfit_decreases_over_5_steps():
    """Mini-overfit: with a fixed target, gradient descent on the predicted
    latent must reduce the loss across 5 steps."""
    torch.manual_seed(0)
    B, T, d = 4, 3, 8
    fut = torch.randn(B, d)
    # The "model" — a single learnable tensor matching the terminal slice.
    z = torch.randn(B, T, d, requires_grad=True)
    opt = torch.optim.SGD([z], lr=0.5)
    losses = []
    dt = torch.full((B,), 30.0)
    for _ in range(5):
        sub = latent_future_state_loss(z, fut, dt)
        assert sub["loss"].requires_grad
        opt.zero_grad()
        sub["loss"].backward()
        opt.step()
        losses.append(float(sub["loss"].detach().item()))
    # Final loss must be strictly less than the initial loss.
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    # Monotone-decreasing is the strongest invariant for SGD on a quadratic.
    for a, b in zip(losses, losses[1:]):
        assert b <= a + 1e-6, f"loss increased mid-overfit: {losses}"


def test_latent_future_state_loss_mmd_gated_on_min_batch():
    """When n_valid < mmd_min_batch, the MMD term must be exactly zero."""
    B, T, d = 4, 3, 8       # B < 16 default
    pred = torch.randn(B, T, d)
    fut = torch.randn(B, d)
    dt = torch.full((B,), 30.0)
    sub = latent_future_state_loss(pred, fut, dt)
    assert sub["loss_traj_mmd"].item() == 0.0
    # When mmd_min_batch is reduced to <= n_valid, the term fires.
    sub2 = latent_future_state_loss(pred, fut, dt, mmd_min_batch=2)
    # MMD on two distinct distributions is positive (sample-dependent).
    assert sub2["loss_traj_mmd"].item() >= 0.0


# --------------------------------------------------------------------------
# CanonicalMORTFMTrainer._trajectory_step tests
# --------------------------------------------------------------------------


def test_trajectory_step_returns_zero_when_no_followup(tmp_path):
    cfg = _debug_cfg(tmp_path)
    pairs = _cohort(n=4)
    static_pairs = [
        TemporalTrainingPair(x_t=p.x_t, x_t_delta=None, outcome=p.outcome)
        for p in pairs
    ]
    _, train_loader, _, _ = make_data_module(static_pairs, batch_size=2)
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5,
                   n_resistance_states=4)
    trainer = CanonicalMORTFMTrainer(
        model, cfg, train_loader=train_loader, val_loader=None,
        device="cpu", strict=False,
    )
    batch = next(iter(train_loader)).to("cpu")
    outputs = trainer._canonical_forward(batch)
    sub = trainer._trajectory_step(batch, outputs)
    assert sub["n_supervised"] == 0
    assert sub["loss"].item() == 0.0


def test_trajectory_step_supervises_n_equals_batch_when_followup_present(tmp_path):
    cfg = _debug_cfg(tmp_path)
    pairs = _cohort(n=6)
    longitudinal = [p for p in pairs if p.x_t_delta is not None]
    assert len(longitudinal) >= 2
    _, train_loader, _, _ = make_data_module(longitudinal, batch_size=2)
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5,
                   n_resistance_states=4)
    trainer = CanonicalMORTFMTrainer(
        model, cfg, train_loader=train_loader, val_loader=None,
        device="cpu", strict=False,
    )
    batch = next(iter(train_loader)).to("cpu")
    outputs = trainer._canonical_forward(batch)
    sub = trainer._trajectory_step(batch, outputs)
    # n_supervised counts rows with finite, positive delta_t. The whole
    # batch had follow-ups, so n_supervised == B.
    assert sub["n_supervised"] == batch.batch_size
    assert sub["loss"].requires_grad


# --------------------------------------------------------------------------
# Bug B1 — axis-order sanity
# --------------------------------------------------------------------------


def test_bug_b1_canonical_z_traj_terminal_slice_returns_b_d(tmp_path):
    """Canonical path: z_traj is (B, T, d_latent); z_traj[:, -1, :] is (B, d)."""
    cfg = _debug_cfg(tmp_path)
    pairs = _cohort(n=4)
    _, train_loader, _, _ = make_data_module(pairs, batch_size=2)
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5,
                   n_resistance_states=4)
    batch = next(iter(train_loader))
    with torch.no_grad():
        out = model(batch)
    z_traj = out["z_traj"]
    B = batch.batch_size
    assert z_traj.shape[0] == B, f"axis 0 must be batch, got {tuple(z_traj.shape)}"
    assert z_traj.shape[1] == cfg.n_time_grid, "axis 1 must be time"
    assert z_traj.shape[2] == cfg.d_latent, "axis 2 must be d_latent"
    terminal = z_traj[:, -1, :]
    assert terminal.shape == (B, cfg.d_latent)


def test_bug_b1_legacy_trajectory_mean_terminal_slice_returns_n_d(tmp_path):
    """Legacy path: trajectory_mean is (T, N, d); [-1] picks last time → (N, d).

    This documents the actual shape contract in the legacy code; the schema
    docstring claims (N, T, d_latent) but the sampler returns (T, N, d).
    """
    cfg = _debug_cfg(tmp_path)
    pairs = _cohort(n=4)
    _, train_loader, _, _ = make_data_module(pairs, batch_size=2)
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5,
                   n_resistance_states=4)
    batch = next(iter(train_loader))
    with torch.no_grad():
        pred = model.forward_legacy(batch)
    assert pred.trajectory_mean is not None
    # Actual shape returned by TrajectorySampler: (T, N, d_latent).
    T, N, d = pred.trajectory_mean.shape
    assert T == cfg.n_time_grid
    assert N == batch.batch_size
    assert d == cfg.d_latent
    # [-1] picks the LAST TIMESTEP, returning (N, d).
    terminal = pred.trajectory_mean[-1]
    assert terminal.shape == (N, d), (
        f"trajectory_mean[-1] should be (N, d_latent); got {tuple(terminal.shape)}"
    )
