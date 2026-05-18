"""
tests/mortfm/test_stage_supervision_strict.py
==============================================
v18.2 — strict supervision policy contract test.

Asserts:
  * Stage F (survival) raises MissingSupervisionError when no batches
    have event_time + event_observed.
  * Stage A (foundation) does NOT raise even when supervision is missing
    (it's not a supervised stage).
  * StageSupervisionReport.coverage is computed correctly.
"""

from __future__ import annotations

import pytest
import torch

from resistancemap.data.trajectory_pair_builder import build_temporal_pairs
from resistancemap.mortfm.data_module import make_data_module
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import (
    DrugContext,
    ModalityTensor,
    MORTFMConfig,
    PatientCellSnapshot,
    ResistanceOutcome,
)
from resistancemap.mortfm.trainer import (
    MORTFMTrainer,
    MissingSupervisionError,
    STRICT_SUPERVISION_COVERAGE,
    StageSupervisionReport,
)


def _mk_snap(pid: str, t: int) -> PatientCellSnapshot:
    rna = ModalityTensor("rna", torch.poisson(torch.full((1, 100), 3.0)),
                          [f"g{i}" for i in range(100)])
    return PatientCellSnapshot(
        patient_id=pid, sample_id=f"{pid}_t{t}", disease="MM", timepoint=float(t),
        rna=rna,
        drug=DrugContext("Bortezomib", "proteasome_inhibitor", dose=0.01),
    )


def _cfg(checkpoint_dir) -> MORTFMConfig:
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


def _unsupervised_pairs(n: int = 6):
    """Cohort with NO event-time labels (event_time=None for everyone)."""
    snaps = [_mk_snap(f"P{i:03d}", t) for i in range(n) for t in [0, 6]]
    outs = [
        ResistanceOutcome(
            patient_id=f"P{i:03d}", baseline_time=0.0,
            event_time=None, censored=True, resistance_label=None,
            drug_response=torch.tensor([0.5]),
        )
        for i in range(n)
    ]
    return build_temporal_pairs(snaps, outs, allowed_time_gaps=[6.0], gap_tolerance=0.5,
                                 include_drug_response_only=True)


def test_stage_F_raises_when_no_event_labels(tmp_path):
    cfg = _cfg(tmp_path)
    pairs = _unsupervised_pairs(n=4)
    splits, train_loader, _, _ = make_data_module(pairs, batch_size=2)
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5, n_resistance_states=4)
    trainer = MORTFMTrainer(model, cfg, train_loader=train_loader, val_loader=None,
                             device=torch.device("cpu"))
    with pytest.raises(MissingSupervisionError):
        trainer.train_epoch(stage="F", epoch=0)


def test_stage_A_does_not_raise_without_supervision(tmp_path):
    """Foundation stage A is unsupervised; missing labels are fine."""
    cfg = _cfg(tmp_path)
    pairs = _unsupervised_pairs(n=4)
    _, train_loader, _, _ = make_data_module(pairs, batch_size=2)
    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5, n_resistance_states=4)
    trainer = MORTFMTrainer(model, cfg, train_loader=train_loader, val_loader=None,
                             device=torch.device("cpu"))
    # Should NOT raise.
    metric = trainer.train_epoch(stage="A", epoch=0)
    assert metric.stage == "A"


def test_strict_thresholds_are_documented():
    assert "E" in STRICT_SUPERVISION_COVERAGE
    assert "F" in STRICT_SUPERVISION_COVERAGE
    assert "G" in STRICT_SUPERVISION_COVERAGE
    # Survival is the strictest.
    assert STRICT_SUPERVISION_COVERAGE["F"] >= STRICT_SUPERVISION_COVERAGE["E"]
    assert STRICT_SUPERVISION_COVERAGE["F"] >= STRICT_SUPERVISION_COVERAGE["G"]


def test_supervision_report_coverage_math():
    r = StageSupervisionReport(stage="F", n_batches=10, n_supervised_batches=7)
    assert r.coverage == 0.7
    r = StageSupervisionReport(stage="F", n_batches=0, n_supervised_batches=0)
    assert r.coverage == 0.0
