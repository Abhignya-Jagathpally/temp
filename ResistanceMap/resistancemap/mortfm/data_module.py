"""
resistancemap/mortfm/data_module.py
===================================
DataModule for MORT-FM training.

Wraps :class:`~resistancemap.data.mortfm_dataset.MORTFMDataset` + the
patient-grouped split logic into a single object the trainer can construct
once. Performs the patient-disjoint split critical for honest evaluation:
no patient appears in more than one of (train, val, test).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from resistancemap.data.mortfm_dataset import MORTFMDataset, mort_collate
from resistancemap.mortfm.schemas import TemporalTrainingPair

logger = logging.getLogger(__name__)


@dataclass
class MORTFMSplits:
    train: List[TemporalTrainingPair] = field(default_factory=list)
    val: List[TemporalTrainingPair] = field(default_factory=list)
    test: List[TemporalTrainingPair] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "train_patients": len({p.x_t.patient_id for p in self.train}),
            "val_patients": len({p.x_t.patient_id for p in self.val}),
            "test_patients": len({p.x_t.patient_id for p in self.test}),
        }


def split_patients(
    pairs: Sequence[TemporalTrainingPair],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> MORTFMSplits:
    """Patient-disjoint split.

    Uses ``torch.Generator`` (manual seed) and slices a *sorted* patient list
    so the same seed reproduces identical splits — no numpy.random, no time-
    dependent ordering.
    """
    patient_ids = sorted({p.x_t.patient_id for p in pairs})
    n = len(patient_ids)
    if n == 0:
        return MORTFMSplits()

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    patient_ids = [patient_ids[i] for i in perm]

    n_test = max(1, int(round(test_fraction * n))) if n >= 4 else 0
    n_val = max(1, int(round(val_fraction * n))) if n >= 4 else 0
    test_set = set(patient_ids[:n_test])
    val_set = set(patient_ids[n_test : n_test + n_val])
    train_set = set(patient_ids[n_test + n_val :])

    splits = MORTFMSplits()
    for p in pairs:
        pid = p.x_t.patient_id
        if pid in train_set:
            splits.train.append(p)
        elif pid in val_set:
            splits.val.append(p)
        elif pid in test_set:
            splits.test.append(p)

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(
            f"PATIENT-DISJOINT SPLIT VIOLATED: train/val={overlap_tv}, "
            f"train/test={overlap_tt}, val/test={overlap_vt}"
        )
    logger.info("split_patients: %s", splits.summary())
    return splits


def make_loader(
    pairs: Sequence[TemporalTrainingPair],
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        MORTFMDataset(pairs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=mort_collate,
    )


def make_data_module(
    pairs: Sequence[TemporalTrainingPair],
    *,
    batch_size: int = 32,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[MORTFMSplits, DataLoader, DataLoader, DataLoader]:
    splits = split_patients(pairs, val_fraction=val_fraction,
                            test_fraction=test_fraction, seed=seed)
    train_loader = make_loader(splits.train, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    val_loader = make_loader(splits.val, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = make_loader(splits.test, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return splits, train_loader, val_loader, test_loader
