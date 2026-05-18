"""Tests for resistancemap.data.manifest_schemas."""

from __future__ import annotations

import pandas as pd
import pytest

from resistancemap.data.manifest_schemas import (
    load_dataset_manifest,
    load_feature_manifest,
    load_public_data_manifest,
    load_samples_table,
)


def _write_samples(tmp_path, df):
    p = tmp_path / "samples.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_samples_table(str(tmp_path / "nope.csv"))
    with pytest.raises(FileNotFoundError):
        load_feature_manifest(str(tmp_path / "nope.csv"))
    with pytest.raises(FileNotFoundError):
        load_dataset_manifest(str(tmp_path / "nope.csv"))


def test_samples_missing_required_column_raises(tmp_path):
    df = pd.DataFrame({"sample_id": ["s1"], "patient_id_or_model_id": ["P1"]})
    p = _write_samples(tmp_path, df)
    with pytest.raises(ValueError, match="missing required columns"):
        load_samples_table(p)


def test_samples_duplicate_sample_id_raises(tmp_path):
    df = pd.DataFrame({
        "sample_id": ["s1", "s1"],
        "patient_id_or_model_id": ["P1", "P2"],
        "source_dataset": ["x", "x"], "disease": ["MM", "MM"],
        "sample_type": ["x"]*2, "modality_available": ["rna"]*2,
        "split_group": ["train", "train"],
    })
    p = _write_samples(tmp_path, df)
    with pytest.raises(ValueError, match="sample_id must be unique"):
        load_samples_table(p)


def test_samples_invalid_split_raises(tmp_path):
    df = pd.DataFrame({
        "sample_id": ["s1"], "patient_id_or_model_id": ["P1"],
        "source_dataset": ["x"], "disease": ["MM"], "sample_type": ["x"],
        "modality_available": ["rna"], "split_group": ["INVALID"],
    })
    p = _write_samples(tmp_path, df)
    with pytest.raises(ValueError, match="invalid values"):
        load_samples_table(p)


def test_samples_patient_disjoint_violation_raises(tmp_path):
    df = pd.DataFrame({
        "sample_id": ["s1", "s2"],
        "patient_id_or_model_id": ["P1", "P1"],
        "source_dataset": ["x"]*2, "disease": ["MM"]*2,
        "sample_type": ["x"]*2, "modality_available": ["rna"]*2,
        "split_group": ["train", "test"],
    })
    p = _write_samples(tmp_path, df)
    with pytest.raises(ValueError, match="Patient-disjoint"):
        load_samples_table(p)


def test_samples_valid_loads(tmp_path):
    df = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(6)],
        "patient_id_or_model_id": [f"P{i}" for i in range(6)],
        "source_dataset": ["DepMap"]*6, "disease": ["MM"]*6,
        "sample_type": ["cl"]*6,
        "modality_available": ["rna,proteomics", "rna", "rna,atac", "rna", "rna", "rna,proteomics"],
        "split_group": ["train"]*4 + ["val", "test"],
    })
    p = _write_samples(tmp_path, df)
    table = load_samples_table(p)
    assert len(table.patients()) == 6
    cov = table.modality_coverage()
    assert cov["rna"] == 1.0
    assert cov["proteomics"] == 2/6
    assert "atac" in cov


def test_feature_manifest_load(tmp_path):
    df = pd.DataFrame({
        "feature_id": ["ENSG1", "ENSG2"],
        "modality": ["rna", "rna"],
        "primary_identifier_type": ["ensembl_gene", "ensembl_gene"],
        "primary_identifier": ["ENSG1", "ENSG2"],
        "gene_symbol": ["CD38", "TNFRSF17"],
    })
    p = tmp_path / "fm.csv"
    df.to_csv(p, index=False)
    fm = load_feature_manifest(str(p))
    assert len(fm.features_for_modality("rna")) == 2
    mapping = fm.map_to_hgnc(["ENSG1", "ENSG_unknown"])
    assert mapping["ENSG1"] == "CD38"
    assert mapping["ENSG_unknown"] is None


def test_dataset_manifest_load(tmp_path):
    df = pd.DataFrame({
        "dataset_name": ["DepMap", "GDSC"],
        "source_url_or_accession": ["url1", "url2"],
        "download_date": ["2026-01-01"]*2,
        "license_or_access_terms": ["CC-BY", "free academic"],
        "raw_file_path": ["/a", "/b"], "processed_file_path": ["/c", "/d"],
        "organism": ["Homo sapiens"]*2,
        "modality": ["rna,crispr", "drug_response"],
        "controlled_access": [False, False],
    })
    p = tmp_path / "dm.csv"
    df.to_csv(p, index=False)
    dm = load_dataset_manifest(str(p))
    assert len(dm.datasets_by_modality("drug_response")) == 1
