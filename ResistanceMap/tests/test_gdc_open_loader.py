"""Unit tests for the GDC open-tier loader — pure parsing only, no network.

Exercises the clinical-hit flattening, OS-label extraction, and STAR-Counts
matrix assembly with tiny structural fixtures. The network functions
(``iter_mmrf_cases``, ``download_*``, ``query_rnaseq_manifest``) are not called.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resistancemap.data import gdc_open_loader as gdc


def test_parse_clinical_hits_flattens_expected_fields():
    hits = [
        {
            "submitter_id": "MMRF_0001",
            "case_id": "c1",
            "demographic": {"gender": "male", "vital_status": "Dead", "days_to_death": 400},
            "diagnoses": [{"iss_stage": "II", "age_at_diagnosis": 22000,
                           "days_to_last_follow_up": 380}],
        },
        {
            "submitter_id": "MMRF_0002",
            "case_id": "c2",
            "demographic": {"gender": "female", "vital_status": "Alive", "days_to_death": None},
            "diagnoses": [{"iss_stage": "III", "age_at_diagnosis": 25000,
                           "days_to_last_follow_up": 600}],
        },
    ]
    df = gdc.parse_gdc_clinical_hits(hits)
    assert list(df["submitter_id"]) == ["MMRF_0001", "MMRF_0002"]
    assert df.loc[0, "vital_status"] == "Dead"
    assert df.loc[0, "iss_stage"] == "II"
    assert df.loc[1, "days_to_last_follow_up"] == 600


def test_parse_clinical_hits_empty_raises():
    with pytest.raises(ValueError, match="empty cohort"):
        gdc.parse_gdc_clinical_hits([])


def test_extract_survival_labels_uses_death_or_followup_and_drops_missing():
    df = pd.DataFrame(
        {
            "submitter_id": ["A", "B", "C"],
            "vital_status": ["Dead", "Alive", "Alive"],
            "days_to_death": [400.0, np.nan, np.nan],
            "days_to_last_follow_up": [np.nan, 600.0, np.nan],  # C has no usable time
        }
    )
    out = gdc.extract_survival_labels(df)
    # A: dead -> event True, time 400. B: alive -> event False, time 600. C dropped.
    assert list(out["submitter_id"]) == ["A", "B"]
    assert out["event"].tolist() == [True, False]
    assert out["time"].tolist() == [400.0, 600.0]
    assert out["structured"].dtype.names == ("event", "time")


def test_extract_survival_labels_missing_columns_raises():
    with pytest.raises(ValueError, match="survival columns"):
        gdc.extract_survival_labels(pd.DataFrame({"submitter_id": ["A"]}))


def test_extract_survival_labels_no_usable_time_raises():
    df = pd.DataFrame(
        {
            "submitter_id": ["A"],
            "vital_status": ["Alive"],
            "days_to_death": [np.nan],
            "days_to_last_follow_up": [np.nan],
        }
    )
    with pytest.raises(ValueError, match="usable OS time"):
        gdc.extract_survival_labels(df)


def _write_star(path, rows):
    cols = ["gene_id", "gene_name", "gene_type", "unstranded",
            "stranded_first", "stranded_second", "tpm_unstranded",
            "fpkm_unstranded", "fpkm_uq_unstranded"]
    pd.DataFrame(rows, columns=cols).to_csv(path, sep="\t", index=False)


def test_build_expression_matrix_excludes_summary_rows_and_strips_version(tmp_path):
    # STAR-Counts: 1 summary N_ row + 2 gene rows.
    _write_star(
        tmp_path / "SAMPLE1.tsv",
        [
            ["N_unmapped", "", "", 10, 5, 5, 0.0, 0.0, 0.0],
            ["ENSG00000000003.15", "TSPAN6", "protein_coding", 100, 50, 50, 12.5, 1.0, 1.0],
            ["ENSG00000000005.6", "TNMD", "protein_coding", 0, 0, 0, 0.0, 0.0, 0.0],
        ],
    )
    mat = gdc.build_expression_matrix(tmp_path)
    assert list(mat.columns) == ["SAMPLE1"]
    # N_unmapped excluded; versions stripped.
    assert "ENSG00000000003" in mat.index
    assert "N_unmapped" not in mat.index
    assert mat.loc["ENSG00000000003", "SAMPLE1"] == pytest.approx(12.5)


def test_build_expression_matrix_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No STAR-Counts"):
        gdc.build_expression_matrix(tmp_path)


def test_merge_expression_clinical_inner_joins_on_case(tmp_path):
    expr = pd.DataFrame(
        {"S1": [1.0, 2.0], "S2": [3.0, 4.0]}, index=["g1", "g2"]
    )
    clinical = pd.DataFrame(
        {"submitter_id": ["MMRF_A", "MMRF_Z"], "vital_status": ["Dead", "Alive"]}
    )
    merged = gdc.merge_expression_clinical(
        expr, clinical, sample_to_case={"S1": "MMRF_A", "S2": "MMRF_B"}
    )
    # Only S1 -> MMRF_A matches a clinical row.
    assert len(merged) == 1
    assert merged.iloc[0]["submitter_id"] == "MMRF_A"
    assert merged.iloc[0]["vital_status"] == "Dead"
