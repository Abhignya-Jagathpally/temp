"""Patient-disjoint split tests — leakage prevention is critical for honest MORT-FM eval."""

from __future__ import annotations

import torch

from resistancemap.data.trajectory_pair_builder import build_temporal_pairs
from resistancemap.mortfm.data_module import split_patients, make_data_module
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


def _cohort(n_patients: int = 20):
    snaps = [_snap(f"P{i:03d}", t) for i in range(n_patients) for t in [0, 6]]
    outs = [
        ResistanceOutcome(patient_id=f"P{i:03d}", baseline_time=0.0,
                          event_time=float(8 + i), censored=(i % 2 == 0))
        for i in range(n_patients)
    ]
    return build_temporal_pairs(snaps, outs, allowed_time_gaps=[6.0], gap_tolerance=0.5)


def test_split_is_patient_disjoint():
    pairs = _cohort(20)
    splits = split_patients(pairs, val_fraction=0.2, test_fraction=0.2, seed=0)
    train_p = {p.x_t.patient_id for p in splits.train}
    val_p = {p.x_t.patient_id for p in splits.val}
    test_p = {p.x_t.patient_id for p in splits.test}
    assert not train_p & val_p
    assert not train_p & test_p
    assert not val_p & test_p


def test_split_is_seed_deterministic():
    pairs = _cohort(20)
    s1 = split_patients(pairs, seed=42)
    s2 = split_patients(pairs, seed=42)
    s3 = split_patients(pairs, seed=99)
    p1 = sorted(p.x_t.patient_id for p in s1.train)
    p2 = sorted(p.x_t.patient_id for p in s2.train)
    p3 = sorted(p.x_t.patient_id for p in s3.train)
    assert p1 == p2
    assert p1 != p3  # different seed -> different split


def test_followup_snapshots_go_with_baselines():
    """A patient's t=6 snapshot must not leak into a different split from their t=0 baseline."""
    pairs = _cohort(20)
    splits = split_patients(pairs, val_fraction=0.2, test_fraction=0.2, seed=0)
    for split_name, split in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        for p in split:
            if p.x_t_delta is not None:
                assert p.x_t_delta.patient_id == p.x_t.patient_id, \
                    f"Pair leaks across patients in {split_name}: {p.x_t.patient_id} vs {p.x_t_delta.patient_id}"


def test_dataloader_creation_does_not_crash_on_minimum_cohort():
    pairs = _cohort(8)
    splits, train_loader, val_loader, test_loader = make_data_module(
        pairs, batch_size=2, val_fraction=0.25, test_fraction=0.25, seed=0,
    )
    assert sum([splits.summary()["n_train"], splits.summary()["n_val"], splits.summary()["n_test"]]) == len(pairs)
    # Loaders iterate without exception.
    assert next(iter(train_loader)) is not None
