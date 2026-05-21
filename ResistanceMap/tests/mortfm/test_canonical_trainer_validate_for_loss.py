"""tests/mortfm/test_canonical_trainer_validate_for_loss.py

Regression test for Bug B14: ``MORTBatch.validate_for_loss`` was a
first-class fail-fast helper that the canonical trainer never invoked,
so a batch missing ``event_time`` / ``event_observed`` would silently
flow into the survival NLL and either NaN-out or produce a misleading
zero loss.

The fix wires ``validate_for_loss`` into ``_losses_for_batch`` BEFORE the
canonical forward + loss assembly. Behaviour:
 * ``strict_supervision=True``  → raise ``MissingSupervisionError``
 * ``strict_supervision=False`` → log + let downstream skip the term
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


# Re-use the shapes that the existing canonical_trainer_smoke test relies on
# so we know the fixture flows through MORTFM.forward cleanly.


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


def _cfg(tmp_path) -> MORTFMConfig:
    return MORTFMConfig(
        rna_input_dim=100, proteomics_input_dim=50, clinical_input_dim=4,
        d_token=16, d_latent=32, fusion_layers=2, fusion_heads=2, fusion_dropout=0.0,
        use_atac=False, use_methylation=False, use_histone_ptm=False,
        use_phosphoproteomics=False, use_clinical=False, use_drug=True,
        n_time_grid=4, integration_time=6.0,
        pretrain_epochs=1, finetune_epochs=1, trajectory_epochs=1, survival_epochs=1,
        mixed_precision=False, batch_size=2, drug_embed_dim=8,
        checkpoint_dir=str(tmp_path),
    )


def _make_trainer(tmp_path, *, strict_supervision: bool):
    torch.manual_seed(0)
    cfg = _cfg(tmp_path)
    pairs = _cohort()
    _, train_loader, val_loader, _ = make_data_module(pairs, batch_size=2)
    torch.manual_seed(0)
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5, n_resistance_states=4)
    trainer = CanonicalMORTFMTrainer(
        model, cfg,
        train_loader=train_loader, val_loader=val_loader,
        device="cpu", strict_supervision=strict_supervision,
    )
    return trainer, train_loader


def test_strict_supervision_raises_when_event_time_missing(tmp_path):
    """Stage F + strict_supervision=True + missing event_time / event_observed
    → ``MissingSupervisionError`` (NOT a silent NaN loss)."""
    trainer, train_loader = _make_trainer(tmp_path, strict_supervision=True)
    batch = next(iter(train_loader))
    # Wipe both survival labels (validate_for_loss('survival') requires both).
    batch.event_time = None
    batch.event_observed = None
    with pytest.raises(MissingSupervisionError, match="validate_for_loss"):
        trainer._losses_for_batch(batch, "F")


def test_non_strict_skips_survival_term_without_raising(tmp_path):
    """strict_supervision=False + missing survival labels → no raise; the
    bundle is returned and the survival term reports n_supervised == 0
    (the downstream training loop will not back-prop a non-existent loss)."""
    trainer, train_loader = _make_trainer(tmp_path, strict_supervision=False)
    batch = next(iter(train_loader))
    batch.event_time = None
    batch.event_observed = None
    # Must NOT raise.
    bundle = trainer._losses_for_batch(batch, "F")
    # Survival term should be present but with n_supervised == 0 (the term
    # had nothing to consume) — i.e. the loss is effectively excluded.
    assert "survival" in bundle["per_term_n_supervised"]
    assert bundle["per_term_n_supervised"]["survival"] == 0


def test_strict_supervision_alias_back_to_strict(tmp_path):
    """The v19 ``strict_supervision`` kw must map onto the v18 ``strict`` flag
    so existing call sites keep working and the alias is observable."""
    trainer, _ = _make_trainer(tmp_path, strict_supervision=True)
    assert trainer.strict is True
    assert trainer.strict_supervision is True

    trainer2, _ = _make_trainer(tmp_path, strict_supervision=False)
    assert trainer2.strict is False
    assert trainer2.strict_supervision is False
