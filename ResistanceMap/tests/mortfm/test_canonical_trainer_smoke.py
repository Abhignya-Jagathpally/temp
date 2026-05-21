"""tests/mortfm/test_canonical_trainer_smoke.py

Smoke tests for the new v18+ canonical trainer scaffolding. Mirrors the
fixture-building of ``test_mortfm_trainer_smoke.py`` but drives a tiny
``MORTFM`` through ``CanonicalMORTFMTrainer._losses_for_batch`` for each of
the patient-level stages E/F/G/H.

The decisive contract: the canonical trainer must NEVER invoke
``MORTFM.forward_legacy``. A monkeypatch spy enforces that.
"""

from __future__ import annotations

import pytest
import torch

from resistancemap.data.trajectory_pair_builder import build_temporal_pairs
from resistancemap.mortfm.canonical_trainer import (
    CanonicalMORTFMTrainer,
    MissingSupervisionError,
)
from resistancemap.mortfm.data_module import make_data_module
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import (
    DrugContext,
    ModalityTensor,
    MORTFMConfig,
    PatientCellSnapshot,
    ResistanceOutcome,
)


# --------------------------------------------------------------------------
# Fixture builders (re-uses the same shapes as the legacy smoke test).
# --------------------------------------------------------------------------


def _mk_snap(pid: str, t: int) -> PatientCellSnapshot:
    rna = ModalityTensor(
        "rna", torch.poisson(torch.full((1, 100), 3.0)),
        [f"g{i}" for i in range(100)],
    )
    prot = ModalityTensor(
        "proteomics", torch.randn(1, 50),
        [f"p{i}" for i in range(50)],
    )
    return PatientCellSnapshot(
        patient_id=pid, sample_id=f"{pid}_t{t}", disease="MM", timepoint=float(t),
        rna=rna, proteomics=prot,
        drug=DrugContext("Bortezomib", "proteasome_inhibitor", dose=0.01),
    )


def _cohort(n: int = 10):
    snaps = [_mk_snap(f"P{i:03d}", t) for i in range(n) for t in [0, 6]]
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


def _make_model_trainer(tmp_path, strict: bool = False, *, seed: int = 0):
    """Build a tiny MORT-FM + canonical trainer pair with a deterministic seed.

    Seeding before both cohort construction (so random RNA / proteomics are
    reproducible) AND model construction (so weight init is reproducible)
    keeps these tests order-independent.
    """
    torch.manual_seed(seed)
    cfg = _debug_cfg(tmp_path)
    pairs = _cohort()
    _, train_loader, val_loader, _ = make_data_module(pairs, batch_size=2)
    torch.manual_seed(seed)
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5, n_resistance_states=4)
    trainer = CanonicalMORTFMTrainer(
        model, cfg,
        train_loader=train_loader, val_loader=val_loader,
        device="cpu", strict=strict,
    )
    return model, trainer, train_loader


# --------------------------------------------------------------------------
# Per-stage smoke tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["E", "F", "G", "H"])
def test_losses_for_batch_returns_canonical_bundle(tmp_path, stage):
    """For each patient-level stage, _losses_for_batch should return a
    dict with the canonical keys and a finite scalar loss tensor."""
    _, trainer, train_loader = _make_model_trainer(tmp_path, strict=False)
    batch = next(iter(train_loader))
    bundle = trainer._losses_for_batch(batch, stage)
    # Canonical-bundle contract.
    assert "loss_total" in bundle
    assert "components" in bundle
    assert "per_term_n_supervised" in bundle
    assert "outputs" in bundle
    assert "required_term" in bundle
    # Loss must be a scalar Tensor (finite or zero, never NaN).
    loss = bundle["loss_total"]
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item(), f"stage {stage}: non-finite loss {loss.item()}"
    # Outputs must contain the canonical LENS dict keys.
    out = bundle["outputs"]
    for key in ("z_traj", "hazard", "survival_curve", "basin_probs",
                "hitting_cdf", "hitting_mean_tau", "t_grid"):
        assert key in out, f"stage {stage}: canonical forward missing {key}"


def test_canonical_trainer_never_calls_forward_legacy(tmp_path, monkeypatch):
    """The canonical trainer must drive the model exclusively through
    ``MORTFM.forward`` — ``forward_legacy`` is OFF-LIMITS."""
    model, trainer, train_loader = _make_model_trainer(tmp_path, strict=False)

    called = {"n": 0}

    def _spy(*args, **kwargs):
        called["n"] += 1
        raise AssertionError(
            "CanonicalMORTFMTrainer must not invoke MORTFM.forward_legacy"
        )

    monkeypatch.setattr(model, "forward_legacy", _spy)

    # Drive _losses_for_batch through every canonical stage.
    batch = next(iter(train_loader))
    for stage in ("E", "F", "G", "H"):
        trainer._losses_for_batch(batch, stage)

    assert called["n"] == 0, "forward_legacy was called %d time(s)" % called["n"]


def test_fit_stage_F_runs_one_epoch_and_records_supervision(tmp_path):
    """fit_stage('F', 1) should produce one per-epoch dict whose
    per_term_n_supervised records that supervision was present."""
    _, trainer, _ = _make_model_trainer(tmp_path, strict=False)
    metrics = trainer.fit_stage("F", n_epochs=1)
    assert len(metrics) == 1
    m = metrics[0]
    assert m["stage"] == "F"
    assert m["epoch"] == 0
    assert m["n_batches"] > 0
    assert "survival" in m["per_term_n_supervised"]
    # Cohort fixture supplies event_time on every row, so survival should
    # have at least one supervised row.
    assert m["per_term_n_supervised"]["survival"] > 0


def test_strict_mode_raises_on_missing_supervision(tmp_path):
    """If strict=True and the required term has no supervision in a batch,
    _losses_for_batch raises MissingSupervisionError."""
    _, trainer, train_loader = _make_model_trainer(tmp_path, strict=True)
    batch = next(iter(train_loader))
    # Wipe the supervision the survival term requires for this one batch.
    batch.event_time = None
    batch.event_observed = None
    with pytest.raises(MissingSupervisionError):
        trainer._losses_for_batch(batch, "F")
