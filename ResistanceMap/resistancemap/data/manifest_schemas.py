"""
resistancemap/data/manifest_schemas.py
======================================
Canonical metadata schemas for the MORT-FM public-data assembly.

Three tables live at the centre of the public-only run:

* ``metadata/samples.csv``           — one row per biological sample (cell,
                                        patient sample, cell-line model).
* ``metadata/feature_manifest.csv``  — one row per measured feature across all
                                        modalities (gene, peak, CpG, protein,
                                        phosphosite).
* ``metadata/dataset_manifest.csv``  — one row per ingested public dataset
                                        (provenance, license, MD5, processing
                                        version).

This module defines the typed dataclasses, the CSV loaders, and the validators.
Everything downstream — encoders, dynamics, heads, evaluation — consumes
these tables.

Honest behaviour
----------------
* Loaders raise :class:`FileNotFoundError` if the CSV is missing.
* :func:`load_samples_table` raises :class:`ValueError` if any required
  column is missing or if patient/model IDs are non-unique within a split.
* No fields are imputed silently. Missing optional fields stay NaN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# samples.csv
# ---------------------------------------------------------------------------


REQUIRED_SAMPLE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "patient_id_or_model_id",
    "source_dataset",
    "disease",
    "sample_type",
    "modality_available",
    "split_group",
)

OPTIONAL_SAMPLE_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "disease_subtype",
    "timepoint",
    "timepoint_unit",
    "treatment_id",
    "treatment_line",
    "batch_id",
    "age",
    "sex",
    "diagnosis_stage",
    "prior_treatment",
    "response_category",
    "progression_status",
    "relapse_status",
    "collection_site",
    "platform",
)


ALLOWED_SPLIT_GROUPS: Set[str] = {"train", "val", "test", "external"}


@dataclass
class SamplesTable:
    """In-memory representation of ``samples.csv``."""

    df: pd.DataFrame

    def patients(self) -> List[str]:
        return sorted(self.df["patient_id_or_model_id"].astype(str).unique().tolist())

    def patients_in_split(self, split: str) -> Set[str]:
        return set(
            self.df.loc[self.df["split_group"] == split, "patient_id_or_model_id"]
            .astype(str)
            .unique()
            .tolist()
        )

    def validate_patient_disjoint_splits(self) -> None:
        """Raises if any patient appears in more than one split."""
        train = self.patients_in_split("train")
        val = self.patients_in_split("val")
        test = self.patients_in_split("test")
        tv = train & val
        tt = train & test
        vt = val & test
        if tv or tt or vt:
            raise ValueError(
                f"Patient-disjoint split violated: "
                f"train∩val={sorted(tv)}, train∩test={sorted(tt)}, val∩test={sorted(vt)}"
            )

    def modality_coverage(self) -> Dict[str, float]:
        """Fraction of samples for which each modality is recorded as available."""
        mods = self.df["modality_available"].astype(str).str.split(",")
        all_mods = sorted({m.strip() for sublist in mods for m in sublist if m.strip()})
        N = len(self.df)
        if N == 0:
            return {m: 0.0 for m in all_mods}
        return {m: float(mods.apply(lambda lst: m in [s.strip() for s in lst]).sum()) / N
                for m in all_mods}


def load_samples_table(path: str) -> SamplesTable:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"samples.csv not found at {p.resolve()}")
    df = pd.read_csv(p)
    missing = [c for c in REQUIRED_SAMPLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"samples.csv {p} missing required columns: {missing}. "
            f"Required: {list(REQUIRED_SAMPLE_COLUMNS)}; present: {list(df.columns)}"
        )
    if df["sample_id"].duplicated().any():
        dups = df.loc[df["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"sample_id must be unique in samples.csv; duplicates: {dups[:10]}")
    bad_splits = set(df["split_group"].astype(str).unique()) - ALLOWED_SPLIT_GROUPS
    if bad_splits:
        raise ValueError(
            f"samples.csv split_group has invalid values: {bad_splits}; "
            f"allowed: {ALLOWED_SPLIT_GROUPS}"
        )
    table = SamplesTable(df=df)
    table.validate_patient_disjoint_splits()
    logger.info(
        "samples.csv: %d samples, %d unique patients/models, splits=%s",
        len(df), len(table.patients()),
        {s: len(table.patients_in_split(s)) for s in ALLOWED_SPLIT_GROUPS},
    )
    return table


# ---------------------------------------------------------------------------
# feature_manifest.csv
# ---------------------------------------------------------------------------


REQUIRED_FEATURE_COLUMNS: tuple[str, ...] = (
    "feature_id",
    "modality",
    "primary_identifier_type",   # one of {hgnc, ensembl_gene, uniprot, chembl, peak, cpg, phosphosite}
    "primary_identifier",
)

OPTIONAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "gene_symbol",
    "ensembl_gene_id",
    "uniprot_id",
    "string_id",
    "chembl_id",
    "reactome_pathway_ids",
    "chrom",
    "start",
    "end",
    "residue",
    "site_position",
    "kinase",
)


@dataclass
class FeatureManifest:
    df: pd.DataFrame

    def features_for_modality(self, modality: str) -> pd.DataFrame:
        return self.df[self.df["modality"] == modality].copy()

    def map_to_hgnc(self, feature_ids: List[str]) -> Dict[str, Optional[str]]:
        """Return ``{feature_id: gene_symbol_or_None}``."""
        sub = self.df.set_index("feature_id")["gene_symbol"] if "gene_symbol" in self.df.columns else None
        if sub is None:
            return {fid: None for fid in feature_ids}
        return {fid: (sub.get(fid) if pd.notna(sub.get(fid)) else None) for fid in feature_ids}


def load_feature_manifest(path: str) -> FeatureManifest:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"feature_manifest.csv not found at {p.resolve()}")
    df = pd.read_csv(p)
    missing = [c for c in REQUIRED_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"feature_manifest.csv missing required columns: {missing}")
    if df["feature_id"].duplicated().any():
        dups = df.loc[df["feature_id"].duplicated(), "feature_id"].tolist()
        raise ValueError(f"feature_id must be unique; duplicates: {dups[:10]}")
    logger.info(
        "feature_manifest.csv: %d features across %d modalities",
        len(df), df["modality"].nunique(),
    )
    return FeatureManifest(df=df)


# ---------------------------------------------------------------------------
# dataset_manifest.csv
# ---------------------------------------------------------------------------


REQUIRED_DATASET_COLUMNS: tuple[str, ...] = (
    "dataset_name",
    "source_url_or_accession",
    "download_date",
    "license_or_access_terms",
    "raw_file_path",
    "processed_file_path",
    "organism",
    "modality",
    "controlled_access",
)

OPTIONAL_DATASET_COLUMNS: tuple[str, ...] = (
    "md5_or_sha256",
    "disease",
    "n_patients",
    "n_samples",
    "n_cells",
    "n_features",
    "has_timepoints",
    "has_outcomes",
    "has_drug_response",
    "notes",
)


@dataclass
class DatasetManifest:
    df: pd.DataFrame

    def datasets_by_modality(self, modality: str) -> pd.DataFrame:
        return self.df[self.df["modality"].astype(str).str.contains(modality, case=False, na=False)]

    def licenses(self) -> Dict[str, str]:
        return dict(zip(self.df["dataset_name"], self.df["license_or_access_terms"]))


def load_dataset_manifest(path: str) -> DatasetManifest:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset_manifest.csv not found at {p.resolve()}")
    df = pd.read_csv(p)
    missing = [c for c in REQUIRED_DATASET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset_manifest.csv missing required columns: {missing}")
    if df["dataset_name"].duplicated().any():
        dups = df.loc[df["dataset_name"].duplicated(), "dataset_name"].tolist()
        raise ValueError(f"dataset_name must be unique; duplicates: {dups[:10]}")
    logger.info(
        "dataset_manifest.csv: %d datasets registered (%d controlled-access)",
        len(df),
        int(df["controlled_access"].astype(str).str.lower().isin({"true", "1", "yes"}).sum()),
    )
    return DatasetManifest(df=df)


# ---------------------------------------------------------------------------
# Convenience: load all three at once
# ---------------------------------------------------------------------------


@dataclass
class PublicDataManifest:
    samples: SamplesTable
    features: FeatureManifest
    datasets: DatasetManifest

    def summary(self) -> dict:
        return {
            "n_samples": len(self.samples.df),
            "n_patients": len(self.samples.patients()),
            "n_features": len(self.features.df),
            "n_datasets": len(self.datasets.df),
            "modality_coverage": self.samples.modality_coverage(),
        }


def load_public_data_manifest(metadata_dir: str) -> PublicDataManifest:
    """Load all three manifest tables from a single ``metadata/`` directory."""
    base = Path(metadata_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"metadata directory not found: {base.resolve()}")
    return PublicDataManifest(
        samples=load_samples_table(str(base / "samples.csv")),
        features=load_feature_manifest(str(base / "feature_manifest.csv")),
        datasets=load_dataset_manifest(str(base / "dataset_manifest.csv")),
    )
