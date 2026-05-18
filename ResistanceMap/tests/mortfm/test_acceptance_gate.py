"""Tests for resistancemap.mortfm.acceptance_gate."""

from __future__ import annotations

import pytest
import torch

from resistancemap.data.trajectory_pair_builder import build_temporal_pairs
from resistancemap.mortfm.acceptance_gate import (
    AcceptanceError,
    AcceptanceReport,
    VALID_LEVELS,
    evaluate_cohort,
    require_level,
)
from resistancemap.mortfm.schemas import (
    DrugContext,
    ModalityTensor,
    MORTFMConfig,
    PatientCellSnapshot,
    ResistanceOutcome,
)


def _snap(pid, t):
    rna = ModalityTensor("rna", torch.zeros(1, 4), ["g1", "g2", "g3", "g4"])
    prot = ModalityTensor("proteomics", torch.zeros(1, 3), ["p1", "p2", "p3"])
    return PatientCellSnapshot(
        patient_id=pid, sample_id=f"{pid}_{t}", disease="MM", timepoint=float(t),
        rna=rna, proteomics=prot,
        drug=DrugContext("Bortezomib", "proteasome_inhibitor", dose=0.01),
    )


def _cohort(n_patients: int, events: bool = True):
    snaps = [_snap(f"P{i:03d}", t) for i in range(n_patients) for t in [0, 6]]
    outs = [
        ResistanceOutcome(patient_id=f"P{i:03d}", baseline_time=0.0,
                          event_time=float(8 + i),
                          censored=(not events) or (i % 2 == 0),
                          resistance_label=i % 4)
        for i in range(n_patients)
    ]
    return build_temporal_pairs(snaps, outs, allowed_time_gaps=[6.0], gap_tolerance=0.5)


def test_unknown_level_raises():
    with pytest.raises(ValueError, match="Unknown acceptance level"):
        evaluate_cohort([], level="paper-ready")


def test_empty_cohort_fails_all_levels():
    for level in VALID_LEVELS:
        report = evaluate_cohort([], level=level)
        assert not report.passed


def test_5_patients_fails_debug():
    pairs = _cohort(5)
    report = evaluate_cohort(pairs, level="debug")
    assert not report.passed
    assert any("debug threshold" in r for r in report.blocking_reasons)


def test_15_patients_passes_debug_but_fails_research():
    pairs = _cohort(15)
    assert evaluate_cohort(pairs, level="debug").passed
    research = evaluate_cohort(pairs, level="research", config=MORTFMConfig())
    assert not research.passed
    assert research.level_achieved in {"debug", "technical"}


def test_require_level_raises_on_block():
    pairs = _cohort(5)
    with pytest.raises(AcceptanceError) as excinfo:
        require_level(pairs, level="debug")
    assert excinfo.value.report.passed is False
    assert "debug" in str(excinfo.value)


def test_strong_level_always_blocks_by_default():
    # Even a "perfect" small cohort must not auto-pass strong; gate requires
    # explicit operational evidence (external cohort, ablations, etc.)
    pairs = _cohort(250, events=True)
    with pytest.raises(AcceptanceError):
        require_level(pairs, level="strong", config=MORTFMConfig())


def test_report_dataclass_carries_metrics():
    pairs = _cohort(20)
    report = evaluate_cohort(pairs, level="debug")
    assert isinstance(report, AcceptanceReport)
    assert report.n_unique_patients == 20
    assert report.n_modalities >= 1
    assert report.has_drug_response_labels is False  # we don't set drug_response, only event_time + label
    assert report.has_survival_labels is True
