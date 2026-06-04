"""Regression tests for the four v20 lab-first-pivot bug fixes.

Bug #1  clinical encoder had only 5 variables -> lab-aware longitudinal encoder
Bug #2  TT2L (2L-start) conflated with progression -> IMWG-PFS endpoint
Bug #3  survival-events gate counted censored rows -> counts observed events
Bug #4  "random_survival_forest" was a RandomForestRegressor -> sksurv RSF

These tests use only structural fixtures (real schemas, no fabricated results)
and run without any MMRF/CoMMpass data on disk.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_RM_ROOT = Path(__file__).resolve().parent.parent  # ResistanceMap/


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# Bug #1 — lab-aware longitudinal clinical encoder
# ===========================================================================
from resistancemap.mortfm.longitudinal import clinical_features as cf


def test_backward_compat_clinical_feature_names_unchanged():
    # The canonical model consumes clinical as width-5; the default must not move.
    assert cf.CLINICAL_FEATURE_NAMES == cf.BASELINE_FEATURE_NAMES
    assert len(cf.CLINICAL_FEATURE_NAMES) == 5


def test_full_clinical_vocabulary_has_at_least_21_features():
    assert len(cf.FULL_CLINICAL_FEATURE_NAMES) >= 21
    # 16 longitudinal labs (15 direct + kl_ratio) + 5 baseline.
    assert len(cf.LONGITUDINAL_LAB_FEATURES) == 16
    assert "kl_ratio" in cf.LONGITUDINAL_LAB_FEATURES


def test_longitudinal_lab_encoder_shapes_masking_and_kl_ratio(tmp_path):
    visit = tmp_path / "visits.csv"
    # Two patients, P1 has two visits (bins 0 and 1), P2 has one.
    pd.DataFrame(
        {
            "PUBLIC_ID": ["P1", "P1", "P2"],
            "VISITDY": [0, 60, 0],
            "D_LAB_serum_m_protein": [3.0, 1.0, np.nan],
            "D_LAB_serum_kappa": [200.0, 50.0, 100.0],
            "D_LAB_serum_lambda": [10.0, 10.0, 0.0],  # P2 lambda=0 -> no kl_ratio
            "D_LAB_cbc_hemoglobin": [10.0, 11.0, 9.0],
            # Intentionally omit other D_LAB_* columns -> they must be masked.
        }
    ).to_csv(visit, index=False)

    enc = cf.encode_longitudinal_labs(
        str(visit), patient_ids=["P1", "P2"], max_timepoints=4, interval_days=60
    )
    assert enc.features.shape == (2, 4, 16)
    assert enc.mask.shape == (2, 4, 16)
    fi = enc.feature_names.index

    # P1 m_protein observed at bins 0 and 1.
    assert enc.mask[0, 0, fi("serum_m_protein")] and enc.mask[0, 1, fi("serum_m_protein")]
    assert enc.features[0, 0, fi("serum_m_protein")] == pytest.approx(3.0)
    # kl_ratio derived for P1 (200/10 = 20), observed.
    assert enc.mask[0, 0, fi("kl_ratio")]
    assert enc.features[0, 0, fi("kl_ratio")] == pytest.approx(20.0)
    # P2 lambda == 0 -> kl_ratio must NOT be fabricated.
    assert not enc.mask[1, 0, fi("kl_ratio")]
    # P2 m_protein was NaN -> masked, not zero-imputed as real.
    assert not enc.mask[1, 0, fi("serum_m_protein")]
    # A column absent from the file is masked everywhere (never fabricated).
    assert not enc.mask[..., fi("chem_creatinine")].any()
    # time_mask True where any obs exists.
    assert enc.time_mask[0, 0] and enc.time_mask[0, 1] and enc.time_mask[1, 0]


def test_longitudinal_encoder_missing_patient_col_raises(tmp_path):
    visit = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(visit, index=False)
    with pytest.raises(ValueError, match="patient column"):
        cf.encode_longitudinal_labs(str(visit), patient_ids=["P1"])


# ===========================================================================
# Bug #4 — proper censoring-aware Random Survival Forest
# ===========================================================================
@pytest.mark.filterwarnings("ignore")
def test_random_survival_forest_uses_sksurv_and_respects_censoring():
    pytest.importorskip("sksurv")
    mod = _load_module(_RM_ROOT / "scripts" / "mortfm" / "10_run_baselines.py", "rb10")

    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(size=(n, 4))
    # Higher x[:,0] -> shorter time (higher risk).
    base = 100.0 - 30.0 * x[:, 0] + rng.normal(scale=5.0, size=n)
    event_time = np.clip(base, 1.0, None)
    event_observed = (rng.uniform(size=n) < 0.7)  # 30% censored

    model = mod._PatientLongitudinalBaseline("random_survival_forest", seed=0)
    model.fit(x, event_time, event_observed, None)

    # Structural: the fixed model is a sksurv RandomSurvivalForest, NOT a regressor.
    from sksurv.ensemble import RandomSurvivalForest
    assert "rsf" in model._params and "rf" not in model._params
    assert isinstance(model._params["rsf"], RandomSurvivalForest)

    # Functional: risk score is concordant with the censored outcome.
    risk = model.predict_risk(x, None)
    # Harrell C-index of risk vs (time, event) should beat chance.
    from sksurv.metrics import concordance_index_censored
    c = concordance_index_censored(event_observed.astype(bool), event_time, risk)[0]
    assert c > 0.6, f"RSF risk should be concordant with survival, got C={c:.3f}"


# ===========================================================================
# Bugs #2 and #3 — Spark lakehouse: IMWG-PFS endpoint + observed-event gate
# ===========================================================================
pyspark = pytest.importorskip("pyspark")


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    s = (
        SparkSession.builder.master("local[1]")
        .appName("v20-bugfix-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield s
    s.stop()


@pytest.fixture(scope="module")
def lakehouse_mod():
    return _load_module(
        _RM_ROOT / "spark_jobs" / "mortfm_lakehouse.py", "mortfm_lakehouse"
    )


def test_imwg_pfs_endpoint_semantics(spark, lakehouse_mod, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    # Clinical: P1 progresses, P2 dies w/o PD, P3 censored.
    clinical = spark.createDataFrame(
        [
            ("P1", 500.0, "Alive", None),
            ("P2", 400.0, "Dead", 400.0),
            ("P3", 600.0, "Alive", None),
        ],
        ["submitter_id", "days_to_last_follow_up", "vital_status", "days_to_death"],
    )
    clinical.write.mode("overwrite").parquet(str(raw / "clinical"))
    # Response: only P1 has a PD assessment.
    resp = spark.createDataFrame(
        [
            ("P1", "VGPR", 100.0),
            ("P1", "PD", 300.0),
            ("P2", "VGPR", 100.0),
            ("P3", "PR", 200.0),
        ],
        ["submitter_id", "bestrespid", "days_to_assessment"],
    )
    resp.write.mode("overwrite").parquet(str(raw / "response"))

    cfg = {
        "lakehouse": {
            "lake_root": str(tmp_path),
            "cleansed": {
                "clinical_outcomes_source": "clinical",
                "patient_col": "submitter_id",
                "endpoint": "imwg_pfs",
                "imwg_response_source": "response",
                "time_col": "days_to_last_follow_up",
            },
        }
    }
    out = lakehouse_mod._cleanse_clinical_outcomes(spark, cfg)
    rows = {r["submitter_id"]: r for r in out.collect()}

    assert all(r["endpoint_type"] == "imwg_pfs" for r in rows.values())
    assert all(r["endpoint_family"] == "progression_free_survival" for r in rows.values())
    # P1: progression at 300 -> event, time 300.
    assert rows["P1"]["event_observed"] is True
    assert rows["P1"]["event_time_days"] == pytest.approx(300.0)
    # P2: death at 400, no PD -> event, time 400.
    assert rows["P2"]["event_observed"] is True
    assert rows["P2"]["event_time_days"] == pytest.approx(400.0)
    # P3: no PD, alive -> censored at last follow-up 600.
    assert rows["P3"]["event_observed"] is False
    assert rows["P3"]["event_time_days"] == pytest.approx(600.0)
    # Crucially: not everyone is an "event" (the TT2L proxy bug made 2L-start->event).
    assert sum(int(r["event_observed"]) for r in rows.values()) == 2


def test_imwg_pfs_without_response_source_refuses_silent_fallback(spark, lakehouse_mod, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    spark.createDataFrame(
        [("P1", 500.0, "Alive")],
        ["submitter_id", "days_to_last_follow_up", "vital_status"],
    ).write.mode("overwrite").parquet(str(raw / "clinical"))
    cfg = {
        "lakehouse": {
            "lake_root": str(tmp_path),
            "cleansed": {
                "clinical_outcomes_source": "clinical",
                "patient_col": "submitter_id",
                "endpoint": "imwg_pfs",
                # imwg_response_source intentionally omitted
            },
        }
    }
    with pytest.raises(ValueError, match="imwg_response_source"):
        lakehouse_mod._cleanse_clinical_outcomes(spark, cfg)


def test_event_gate_counts_observed_events_not_rows(spark, lakehouse_mod, tmp_path):
    confirmed = tmp_path / "confirmed" / "mortfm_training_dataset"
    confirmed.parent.mkdir(parents=True, exist_ok=True)
    # 5 supervised rows, only 2 observed events.
    df = spark.createDataFrame(
        [
            ("h1", False, True, True),
            ("h2", False, True, True),
            ("h3", False, True, False),
            ("h4", False, True, False),
            ("h5", False, True, False),
        ],
        [
            "patient_id_hash",
            "has_valid_trajectory_supervision",
            "has_valid_survival_supervision",
            "event_observed",
        ],
    )
    df.write.mode("overwrite").parquet(str(confirmed))

    cfg = {
        "lakehouse": {
            "lake_root": str(tmp_path),
            "gates": {
                "min_survival_patients": 1,
                "min_survival_events": 2,
                "min_trajectory_pairs": 0,
            },
        }
    }
    # Gate passes (2 events >= 2); report is written regardless.
    lakehouse_mod.stage_audit(spark, cfg)
    report = json.loads(
        (tmp_path / "confirmed" / "data_quality_report.json").read_text()
    )
    assert report["n_survival_events"] == 2          # observed events
    assert report["n_survival_rows"] == 5            # supervised rows
    gate = report["gates"]["min_survival_events"]
    assert gate["actual"] == 2 and gate["actual_supervised_rows"] == 5
    assert gate["passed"] is True
