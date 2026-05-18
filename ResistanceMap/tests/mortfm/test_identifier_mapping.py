"""Tests for resistancemap.data.identifier_mapping."""

from __future__ import annotations

import pandas as pd
import pytest

from resistancemap.data.identifier_mapping import load_identifier_maps, unify_identifiers


def _write_maps(tmp_path):
    pd.DataFrame({"hgnc_symbol": ["CD38", "TNFRSF17"],
                  "ensembl_gene_id": ["ENSG0001", "ENSG0002"]}).to_csv(
        tmp_path / "hgnc.tsv", sep="\t", index=False)
    pd.DataFrame({"hgnc_symbol": ["CD38", "TNFRSF17"],
                  "uniprot_id": ["P28907", "Q02223"]}).to_csv(
        tmp_path / "up.tsv", sep="\t", index=False)
    pd.DataFrame({"string_id": ["9606.E1", "9606.E2"],
                  "uniprot_id": ["P28907", "Q02223"]}).to_csv(
        tmp_path / "str.tsv", sep="\t", index=False)
    pd.DataFrame({"hgnc_symbol": ["CD38", "CD38", "TNFRSF17"],
                  "reactome_id": ["R1", "R2", "R3"]}).to_csv(
        tmp_path / "rx.tsv", sep="\t", index=False)
    pd.DataFrame({"chembl_id": ["CHEMBL_D1", "CHEMBL_D1"],
                  "uniprot_id": ["P28907", "Q02223"]}).to_csv(
        tmp_path / "ch.tsv", sep="\t", index=False)
    return {
        "hgnc_ensembl_path": str(tmp_path / "hgnc.tsv"),
        "uniprot_xref_path": str(tmp_path / "up.tsv"),
        "string_alias_path": str(tmp_path / "str.tsv"),
        "reactome_membership_path": str(tmp_path / "rx.tsv"),
        "chembl_target_path": str(tmp_path / "ch.tsv"),
    }


def test_missing_file_raises(tmp_path):
    paths = _write_maps(tmp_path)
    paths["hgnc_ensembl_path"] = str(tmp_path / "missing.tsv")
    with pytest.raises(FileNotFoundError):
        load_identifier_maps(**paths)


def test_round_trip_lookups(tmp_path):
    maps = load_identifier_maps(**_write_maps(tmp_path))
    assert maps.hgnc_to_ensembl["CD38"] == "ENSG0001"
    assert maps.ensembl_to_hgnc["ENSG0001"] == "CD38"
    assert maps.uniprot_to_hgnc["Q02223"] == "TNFRSF17"
    assert maps.string_to_uniprot["9606.E1"] == "P28907"


def test_harmonise_handles_all_namespaces(tmp_path):
    maps = load_identifier_maps(**_write_maps(tmp_path))
    assert maps.harmonise_gene_symbol("CD38") == "CD38"
    assert maps.harmonise_gene_symbol("ENSG0002") == "TNFRSF17"
    assert maps.harmonise_gene_symbol("P28907") == "CD38"
    assert maps.harmonise_gene_symbol("9606.E2") == "TNFRSF17"
    assert maps.harmonise_gene_symbol("nonsense_id") is None


def test_pathways_and_drug_targets(tmp_path):
    maps = load_identifier_maps(**_write_maps(tmp_path))
    pw = maps.pathways_for_gene("CD38")
    assert "R1" in pw and "R2" in pw
    tgs = maps.drug_targets("CHEMBL_D1")
    assert "P28907" in tgs and "Q02223" in tgs


def test_unify_identifiers_returns_none_for_unmapped(tmp_path):
    maps = load_identifier_maps(**_write_maps(tmp_path))
    out = unify_identifiers(["CD38", "P28907", "GIBBERISH"], maps)
    assert out["CD38"] == "CD38"
    assert out["P28907"] == "CD38"
    assert out["GIBBERISH"] is None  # never falls through to the input
