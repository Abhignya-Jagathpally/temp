"""Hierarchical group splits to close patient / lineage / drug leakage.

Replaces the sample-level random split that defects #1/#2/#5 flagged. Any
evaluation in the pipeline must declare which split kind it used; the manifest
is appended to the verification chain.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


SPLIT_KINDS = ("patient", "lineage", "drug_holdout")


class LeakageError(RuntimeError):
    """Raised when group keys overlap across supposedly disjoint cohorts."""


@dataclass
class SplitManifest:
    kind: str
    seed: int
    n_folds: int
    group_hash: str
    fold_sizes: list[tuple[int, int]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _require_columns(meta: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in meta.columns]
    if missing:
        raise KeyError(f"Split requires columns {missing}; got {list(meta.columns)}")


def make_split(
    meta: pd.DataFrame,
    kind: str,
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], SplitManifest]:
    """Produce (train, test) index folds grouped by the chosen leakage key.

    kind:
      * patient       -- no patient appears in both train and test
      * lineage       -- no CCLE lineage subtype appears in both (sibling-line leakage)
      * drug_holdout  -- no drug appears in both (generalisation to new drugs)
    """
    if kind not in SPLIT_KINDS:
        raise ValueError(f"kind must be one of {SPLIT_KINDS}; got {kind!r}")

    if kind == "patient":
        _require_columns(meta, ["patient_id"])
        groups = meta["patient_id"].astype(str)
    elif kind == "lineage":
        _require_columns(meta, ["lineage_subtype"])
        groups = meta["lineage_subtype"].astype(str)
    else:  # drug_holdout
        _require_columns(meta, ["drug"])
        groups = meta["drug"].astype(str)

    if groups.nunique() < n_folds:
        raise ValueError(
            f"Only {groups.nunique()} unique {kind} groups; cannot make {n_folds} folds."
        )

    cv = GroupKFold(n_splits=n_folds)
    folds = [(tr, te) for tr, te in cv.split(meta, groups=groups)]

    manifest = SplitManifest(
        kind=kind,
        seed=seed,
        n_folds=n_folds,
        group_hash=hashlib.sha256(
            json.dumps(sorted(groups.tolist())).encode()
        ).hexdigest(),
        fold_sizes=[(int(len(tr)), int(len(te))) for tr, te in folds],
    )
    return folds, manifest


def assert_no_overlap(
    a_ids, b_ids, key: str = "patient_id", label_a: str = "a", label_b: str = "b"
) -> None:
    """Refuse to proceed if two cohorts share group ids (e.g. pretrain vs eval)."""
    overlap = set(map(str, a_ids)) & set(map(str, b_ids))
    if overlap:
        head = list(overlap)[:5]
        raise LeakageError(
            f"{len(overlap)} {key}s appear in both {label_a} and {label_b}: "
            f"{head}{'...' if len(overlap) > 5 else ''}"
        )


def assert_folds_disjoint(folds, meta: pd.DataFrame, key: str) -> None:
    """Post-hoc check: every (train, test) pair is disjoint on `key`."""
    for i, (tr_idx, te_idx) in enumerate(folds):
        tr = set(meta.iloc[tr_idx][key].astype(str))
        te = set(meta.iloc[te_idx][key].astype(str))
        if tr & te:
            raise LeakageError(
                f"Fold {i}: {len(tr & te)} {key}s overlap between train and test"
            )
