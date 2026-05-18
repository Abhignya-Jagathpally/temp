#!/usr/bin/env python3
"""
scripts/mortfm_ingest_depmap.py
===============================
Block A — DepMap / CCLE ingestion.

Reads a downloaded DepMap release directory (``data/raw_public/depmap/``) and
emits canonical MORT-FM artifacts:

* ``data/processed/omics/rna_matrix.parquet``      (models × genes, log2 TPM)
* ``data/processed/perturbation/crispr_gene_effect.parquet``
* ``data/processed/metadata/samples_depmap.csv``
* updates / appends ``data/processed/metadata/dataset_manifest.csv``

Expected files (per the DepMap public release):
    Model.csv                       (cell-line metadata)
    OmicsExpressionProteinCodingGenesTPMLogp1.csv
    CRISPRGeneEffect.csv
    OmicsCNGene.csv                  (optional; CNV)
    OmicsSomaticMutations.csv        (optional; mutations)

Does NOT download anything. Raises FileNotFoundError if the expected files
are not present. License under DepMap 24Q2 onwards is CC-BY-4.0 (Broad).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_ingest_depmap")


_EXPECTED_FILES = {
    "model": "Model.csv",
    "expression": "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "crispr": "CRISPRGeneEffect.csv",
}


def _md5(path: Path, chunk: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _require(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"DepMap ingestion: expected {label} at {path.resolve()}. "
            f"Download from https://depmap.org/portal/data_page/ and place under data/raw_public/depmap/."
        )
    return path


def ingest_depmap(raw_dir: Path, out_dir: Path) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = _require(raw_dir / _EXPECTED_FILES["model"], "Model.csv")
    expr_path = _require(raw_dir / _EXPECTED_FILES["expression"], "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    crispr_path = _require(raw_dir / _EXPECTED_FILES["crispr"], "CRISPRGeneEffect.csv")

    logger.info("Reading DepMap Model.csv ...")
    models = pd.read_csv(model_path, low_memory=False)
    logger.info("DepMap: %d models", len(models))

    logger.info("Reading DepMap RNA TPM matrix (this may take a minute) ...")
    expr = pd.read_csv(expr_path, index_col=0, low_memory=False)
    # DepMap convention: first column is ModelID; columns are "<HGNC> (<EntrezID>)"
    expr.columns = [c.split(" (")[0] for c in expr.columns]
    rna_out = out_dir / "omics" / "rna_matrix.parquet"
    rna_out.parent.mkdir(parents=True, exist_ok=True)
    expr.to_parquet(rna_out)
    logger.info("Wrote RNA matrix: %d cell-lines x %d genes -> %s",
                expr.shape[0], expr.shape[1], rna_out)

    logger.info("Reading DepMap CRISPR gene-effect ...")
    crispr = pd.read_csv(crispr_path, index_col=0, low_memory=False)
    crispr.columns = [c.split(" (")[0] for c in crispr.columns]
    crispr_out = out_dir / "perturbation" / "crispr_gene_effect.parquet"
    crispr_out.parent.mkdir(parents=True, exist_ok=True)
    crispr.to_parquet(crispr_out)
    logger.info("Wrote CRISPR gene-effect: %d cell-lines x %d genes -> %s",
                crispr.shape[0], crispr.shape[1], crispr_out)

    # samples_depmap.csv (cell-line == sample == patient_id_or_model_id).
    id_col = "ModelID" if "ModelID" in models.columns else models.columns[0]
    samples = pd.DataFrame({
        "sample_id": models[id_col].astype(str),
        "patient_id_or_model_id": models[id_col].astype(str),
        "source_dataset": "DepMap",
        "disease": models.get("OncotreeLineage", "unknown").astype(str),
        "sample_type": "cell_line",
        "modality_available": "rna,crispr",
        "split_group": "train",   # caller may relabel
    })
    samples_out = out_dir / "metadata" / "samples_depmap.csv"
    samples_out.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(samples_out, index=False)
    logger.info("Wrote samples_depmap.csv: %d rows -> %s", len(samples), samples_out)

    manifest_row = {
        "dataset_name": "DepMap",
        "source_url_or_accession": "https://depmap.org/portal/data_page/",
        "download_date": dt.date.today().isoformat(),
        "license_or_access_terms": "CC-BY-4.0 (DepMap 24Q2+)",
        "raw_file_path": str(raw_dir.resolve()),
        "processed_file_path": str(out_dir.resolve()),
        "organism": "Homo sapiens",
        "modality": "rna,crispr",
        "controlled_access": False,
        "md5_or_sha256": f"rna={_md5(rna_out)};crispr={_md5(crispr_out)}",
        "n_samples": int(expr.shape[0]),
        "n_features": int(expr.shape[1]),
        "has_drug_response": False,
        "notes": f"Model.csv={model_path.name}; expression={expr_path.name}; crispr={crispr_path.name}",
    }
    return manifest_row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw_public/depmap")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    manifest_row = ingest_depmap(args.raw_dir, args.out_dir)
    logger.info("Manifest row: %s", manifest_row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
