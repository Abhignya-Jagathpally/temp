"""
resistancemap/mortfm/longitudinal/temporal_splitter.py
========================================================
Patient-disjoint train/val/test splits for LongitudinalPair records.

Two modes:
  * ``patient_disjoint`` — random shuffle of patients with fixed seed.
  * ``temporal_holdout`` — earliest 70% (by diagnosis_day) go to train,
    next 15% to val, latest 15% to test. Approximates the "would the
    model have generalised forward in time?" question that prospective
    validation answers.

Both modes guarantee no patient appears in more than one split.
"""

from __future__ import annotations

import logging
import random as _random
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from resistancemap.mortfm.longitudinal.schemas import LongitudinalPair

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplit:
    train: List[LongitudinalPair]
    val: List[LongitudinalPair]
    test: List[LongitudinalPair]
    mode: str

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "n_train": len(self.train), "n_val": len(self.val), "n_test": len(self.test),
            "train_patients": len({p.patient_id for p in self.train}),
            "val_patients": len({p.patient_id for p in self.val}),
            "test_patients": len({p.patient_id for p in self.test}),
        }


def split_patient_disjoint(
    pairs: List[LongitudinalPair],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 13,
) -> TemporalSplit:
    patients = sorted({p.patient_id for p in pairs})
    rng = _random.Random(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    test_pats = set(patients[:n_test])
    val_pats = set(patients[n_test:n_test + n_val])
    train_pats = set(patients[n_test + n_val:])
    train = [p for p in pairs if p.patient_id in train_pats]
    val = [p for p in pairs if p.patient_id in val_pats]
    test = [p for p in pairs if p.patient_id in test_pats]
    return TemporalSplit(train=train, val=val, test=test, mode="patient_disjoint")


def split_temporal_holdout(
    pairs: List[LongitudinalPair],
    diagnosis_day_lookup: dict[str, float],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> TemporalSplit:
    """Holdout the latest 15% by diagnosis_day (test) + next 15% (val)."""
    patients = sorted({p.patient_id for p in pairs})
    by_time = sorted(patients, key=lambda p: diagnosis_day_lookup.get(p, 0.0))
    n = len(by_time)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    train_pats = set(by_time[: n - n_test - n_val])
    val_pats = set(by_time[n - n_test - n_val : n - n_test])
    test_pats = set(by_time[n - n_test :])
    train = [p for p in pairs if p.patient_id in train_pats]
    val = [p for p in pairs if p.patient_id in val_pats]
    test = [p for p in pairs if p.patient_id in test_pats]
    return TemporalSplit(train=train, val=val, test=test, mode="temporal_holdout")
