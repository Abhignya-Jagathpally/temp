"""Tests for resistancemap.evaluation.leakage.

Two leakage modes are checked against handcrafted minimal splits:
1. Patient ID overlap across train / val / test partitions.
2. Temporal leakage where a train sample for a patient is timestamped at
   or after a test sample for the same patient.

A clean split must produce LeakageReport(clean=True) with empty violation
lists.
"""

from __future__ import annotations

import pytest

evaluation = pytest.importorskip(
    "resistancemap.evaluation",
    reason="resistancemap.evaluation package not yet merged into worktree",
)

from resistancemap.evaluation.leakage import (  # noqa: E402
    audit_patient_split,
    detect_temporal_leakage,
)


# ---------------------------------------------------------------------------
# Patient-level split audit
# ---------------------------------------------------------------------------


def test_audit_patient_split_detects_overlap():
    train = ["P001", "P002", "P003"]
    val = ["P004", "P002"]   # P002 leaks from train -> val
    test = ["P005", "P003"]  # P003 leaks from train -> test

    report = audit_patient_split(train=train, val=val, test=test)

    assert report.clean is False
    overlapping = set()
    # Tolerate either a single combined list or per-pair sets.
    for attr in ("overlaps", "violations", "duplicates"):
        if hasattr(report, attr):
            data = getattr(report, attr)
            if isinstance(data, dict):
                for v in data.values():
                    overlapping.update(v)
            elif isinstance(data, (list, set, tuple)):
                overlapping.update(data)
    assert "P002" in overlapping
    assert "P003" in overlapping


def test_audit_patient_split_clean_split_is_clean():
    train = ["P001", "P002", "P003"]
    val = ["P004", "P005"]
    test = ["P006", "P007"]

    report = audit_patient_split(train=train, val=val, test=test)

    assert report.clean is True
    for attr in ("overlaps", "violations", "duplicates"):
        if hasattr(report, attr):
            data = getattr(report, attr)
            if isinstance(data, dict):
                assert all(len(v) == 0 for v in data.values())
            else:
                assert len(list(data)) == 0


# ---------------------------------------------------------------------------
# Temporal leakage
# ---------------------------------------------------------------------------


def test_detect_temporal_leakage_flags_train_after_test():
    """A train sample timestamped at/after a test sample (same patient) leaks."""
    train_samples = [
        {"patient_id": "P001", "timestamp": 100},
        {"patient_id": "P002", "timestamp": 50},
    ]
    test_samples = [
        {"patient_id": "P001", "timestamp": 80},  # train(100) > test(80) -> leak
        {"patient_id": "P002", "timestamp": 200},
    ]

    report = detect_temporal_leakage(train=train_samples, test=test_samples)
    assert report.clean is False

    flagged = set()
    for attr in ("violations", "leaking_patients", "patients"):
        if hasattr(report, attr):
            data = getattr(report, attr)
            if isinstance(data, (list, set, tuple)):
                for entry in data:
                    if isinstance(entry, str):
                        flagged.add(entry)
                    elif isinstance(entry, dict) and "patient_id" in entry:
                        flagged.add(entry["patient_id"])
    assert "P001" in flagged


def test_detect_temporal_leakage_clean_split():
    """A correctly time-ordered split produces a clean LeakageReport."""
    train_samples = [
        {"patient_id": "P001", "timestamp": 10},
        {"patient_id": "P002", "timestamp": 20},
    ]
    test_samples = [
        {"patient_id": "P001", "timestamp": 50},
        {"patient_id": "P002", "timestamp": 60},
    ]

    report = detect_temporal_leakage(train=train_samples, test=test_samples)
    assert report.clean is True
    for attr in ("violations", "leaking_patients", "patients"):
        if hasattr(report, attr):
            data = getattr(report, attr)
            if isinstance(data, (list, set, tuple)):
                assert len(list(data)) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
