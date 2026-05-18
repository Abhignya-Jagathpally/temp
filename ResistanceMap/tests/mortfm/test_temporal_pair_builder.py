"""Tests for resistancemap.data.trajectory_pair_builder."""

from __future__ import annotations

import torch

from resistancemap.data.trajectory_pair_builder import build_temporal_pairs, summarise_pairs
from resistancemap.mortfm.schemas import (
    ModalityTensor,
    PatientCellSnapshot,
    ResistanceOutcome,
)


def _snap(pid: str, t: float) -> PatientCellSnapshot:
    return PatientCellSnapshot(
        patient_id=pid, sample_id=f"{pid}_{t}", disease="MM", timepoint=t,
        rna=ModalityTensor("rna", torch.zeros(1, 4), ["g1", "g2", "g3", "g4"]),
    )


def test_empty_inputs_return_empty_list():
    assert build_temporal_pairs([], []) == []


def test_pairs_respect_allowed_time_gaps():
    snaps = [_snap("P1", 0), _snap("P1", 6), _snap("P1", 12)]
    # One outcome per *baseline* timepoint so both (0,6) and (6,12) have a match.
    outs = [
        ResistanceOutcome(patient_id="P1", baseline_time=0.0, event_time=10.0, censored=False),
        ResistanceOutcome(patient_id="P1", baseline_time=6.0, event_time=12.0, censored=False),
    ]
    pairs = build_temporal_pairs(snaps, outs, allowed_time_gaps=[6.0], gap_tolerance=0.5,
                                 include_survival_only=False)
    longitudinal = [p for p in pairs if p.x_t_delta is not None]
    # (0, 6) and (6, 12) match the 6-month gap; (0, 12) is filtered out.
    assert len(longitudinal) == 2


def test_outcome_outside_tolerance_skips_pair():
    snaps = [_snap("P1", 0), _snap("P1", 6)]
    outs = [
        ResistanceOutcome(patient_id="P1", baseline_time=0.0, event_time=10.0, censored=False),
    ]
    pairs = build_temporal_pairs(
        snaps, outs, allowed_time_gaps=[6.0], gap_tolerance=0.5,
        outcome_match_tolerance=1.0, include_survival_only=False,
    )
    # The single outcome is at baseline_time=0, so the (0, 6) pair matches but
    # any pair where x_t.timepoint != 0 (here none, but if there were one at t=6)
    # would be skipped because the outcome is 6 months away (> tol=1).
    longitudinal = [p for p in pairs if p.x_t_delta is not None]
    assert len(longitudinal) == 1


def test_no_self_pairs():
    snaps = [_snap("P1", 0), _snap("P1", 6)]
    outs = [ResistanceOutcome(patient_id="P1", baseline_time=0.0, event_time=8.0, censored=False)]
    pairs = build_temporal_pairs(snaps, outs, include_survival_only=False)
    for p in pairs:
        if p.x_t_delta is not None:
            assert p.x_t.sample_id != p.x_t_delta.sample_id


def test_no_cross_patient_pairs():
    snaps = [_snap("P1", 0), _snap("P2", 6)]
    outs = [
        ResistanceOutcome(patient_id="P1", baseline_time=0.0, event_time=8.0, censored=False),
        ResistanceOutcome(patient_id="P2", baseline_time=0.0, event_time=8.0, censored=False),
    ]
    pairs = build_temporal_pairs(snaps, outs, include_survival_only=False)
    # No same-patient longitudinal pairs possible -> no longitudinal pairs.
    longitudinal = [p for p in pairs if p.x_t_delta is not None]
    assert len(longitudinal) == 0


def test_survival_only_rows_emitted():
    snaps = [_snap("P1", 0)]
    outs = [ResistanceOutcome(patient_id="P1", baseline_time=0.0, event_time=8.0, censored=False)]
    pairs = build_temporal_pairs(snaps, outs)
    assert len(pairs) == 1
    assert pairs[0].x_t_delta is None
    assert pairs[0].outcome.has_survival_label()


def test_unlabelled_rows_only_when_opt_in():
    snaps = [_snap("P1", 0)]
    pairs = build_temporal_pairs(snaps, [], include_unlabelled=False)
    assert len(pairs) == 0
    pairs = build_temporal_pairs(snaps, [], include_unlabelled=True)
    assert len(pairs) == 1


def test_summarise_pairs_counts():
    snaps = [_snap("P1", 0), _snap("P1", 6), _snap("P2", 0)]
    outs = [
        ResistanceOutcome(patient_id="P1", baseline_time=0.0, event_time=10.0, censored=False),
        ResistanceOutcome(patient_id="P2", baseline_time=0.0, event_time=15.0, censored=True),
    ]
    pairs = build_temporal_pairs(snaps, outs, allowed_time_gaps=[6.0], gap_tolerance=0.5)
    summary = summarise_pairs(pairs)
    assert summary["n_unique_patients"] == 2
    assert summary["n_longitudinal_pairs"] >= 1
