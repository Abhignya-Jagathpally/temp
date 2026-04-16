"""Automated leakage tests. Run on every commit; failing any is a release blocker."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resistancemap.data.splits import (
    make_split, assert_no_overlap, assert_folds_disjoint, LeakageError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_meta():
    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame({
        "sample_id":       [f"s{i}" for i in range(n)],
        "patient_id":      rng.integers(0, 50, n),
        "lineage_subtype": rng.choice(["MM", "AML", "ALL", "CLL"], n),
        "drug":            rng.choice([f"drug_{i}" for i in range(11)], n),
    })


# ---------------------------------------------------------------------------
# Core leakage tests
# ---------------------------------------------------------------------------
def test_patient_split_folds_disjoint(synthetic_meta):
    folds, _ = make_split(synthetic_meta, kind="patient", n_folds=5, seed=0)
    assert_folds_disjoint(folds, synthetic_meta, key="patient_id")


def test_lineage_split_folds_disjoint(synthetic_meta):
    folds, _ = make_split(synthetic_meta, kind="lineage", n_folds=4, seed=0)
    assert_folds_disjoint(folds, synthetic_meta, key="lineage_subtype")


def test_drug_holdout_generalisation(synthetic_meta):
    folds, _ = make_split(synthetic_meta, kind="drug_holdout", n_folds=5, seed=0)
    for i, (tr, te) in enumerate(folds):
        tr_drugs = set(synthetic_meta.iloc[tr]["drug"])
        te_drugs = set(synthetic_meta.iloc[te]["drug"])
        assert not (tr_drugs & te_drugs), f"Fold {i} has shared drugs"


def test_pretrain_eval_cohort_disjoint_happy(synthetic_meta):
    pretrain = synthetic_meta.iloc[:300]["patient_id"]
    evalc = synthetic_meta.iloc[400:]["patient_id"]
    # Force disjoint for the happy path
    evalc = [pid + 1000 for pid in evalc]
    assert_no_overlap(pretrain, evalc, key="patient_id",
                      label_a="pretrain", label_b="eval")


def test_pretrain_eval_cohort_disjoint_raises(synthetic_meta):
    with pytest.raises(LeakageError):
        assert_no_overlap(
            synthetic_meta["patient_id"].iloc[:400],
            synthetic_meta["patient_id"].iloc[200:],  # overlaps 200..399
            key="patient_id", label_a="pretrain", label_b="eval",
        )


def test_bad_kind_rejected(synthetic_meta):
    with pytest.raises(ValueError):
        make_split(synthetic_meta, kind="sample", n_folds=5)


def test_split_manifest_reproducible(synthetic_meta):
    _, m1 = make_split(synthetic_meta, kind="patient", n_folds=5, seed=0)
    _, m2 = make_split(synthetic_meta, kind="patient", n_folds=5, seed=0)
    assert m1.group_hash == m2.group_hash
    assert m1.fold_sizes == m2.fold_sizes


def test_esm2_training_contamination_placeholder():
    """Placeholder: confirm no additional ESM-2 finetuning on test-set proteins.

    Concrete implementation: load the set of protein accessions whose mechanism
    labels appear in the held-out evaluation, and assert they are not present in
    any supplementary ESM-2 finetune corpus.
    """
    # Implement once ESM-2 finetune pipeline is defined.
    pytest.skip("ESM-2 finetune corpus not yet defined; see §4.1 remediation plan.")
