#!/usr/bin/env python3
"""
scripts/mortfm_ingest_gdsc.py
=============================
Block A — GDSC drug-response ingestion.

Reads a downloaded GDSC release (``data/raw_public/gdsc/``) and emits:

* ``data/processed/drug_response/gdsc_response_long.parquet``
    Long-form table: ``model_id, drug_id, drug_name, response_value, response_metric, dose``
* updates ``data/processed/metadata/dataset_manifest.csv``

Expected files (per the GDSC bulk download):
    GDSC2_fitted_dose_response_*.xlsx  OR  GDSC2_fitted_dose_response.csv
    screened_compounds_rel_*.csv       (drug names + targets)
    Cell_Lines_Details.xlsx            (model metadata)

License: GDSC is freely available to academic/medical use.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_ingest_gdsc")


def _find_one(raw_dir: Path, patterns: list[str]) -> Path:
    for pat in patterns:
        hits = list(raw_dir.rglob(pat))
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(
        f"GDSC ingestion: none of {patterns} found under {raw_dir.resolve()}. "
        f"Download from https://www.cancerrxgene.org/downloads/bulk_download or "
        f"python3 scripts/mortfm_download_public_data.py --only gdsc"
    )


def ingest_gdsc(raw_dir: Path, out_dir: Path) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    response_file = _find_one(raw_dir,
                              ["GDSC2_fitted_dose_response*.csv",
                               "GDSC2_fitted_dose_response*.xlsx",
                               "GDSC1_fitted_dose_response*.csv",
                               "GDSC1_fitted_dose_response*.xlsx"])
    compounds_file = _find_one(raw_dir, ["screened_compounds*.csv", "screened_compounds*.xlsx"])

    logger.info("Reading GDSC dose response from %s", response_file)
    if response_file.suffix.lower() == ".xlsx":
        resp = pd.read_excel(response_file)
    else:
        resp = pd.read_csv(response_file, low_memory=False)

    # Column conventions vary by release; pick the first that matches.
    col_model = next((c for c in ["SANGER_MODEL_ID", "COSMIC_ID", "CELL_LINE_NAME", "cell_line_name"]
                      if c in resp.columns), None)
    col_drug = next((c for c in ["DRUG_ID", "drug_id"] if c in resp.columns), None)
    col_drug_name = next((c for c in ["DRUG_NAME", "drug_name"] if c in resp.columns), None)
    col_ic50 = next((c for c in ["LN_IC50", "IC50_ln"] if c in resp.columns), None)
    col_auc = next((c for c in ["AUC", "auc"] if c in resp.columns), None)
    if col_model is None or col_drug is None or (col_ic50 is None and col_auc is None):
        raise ValueError(
            f"GDSC response file {response_file.name} missing expected columns. "
            f"Available: {list(resp.columns)[:20]}"
        )

    rows = []
    if col_ic50 is not None:
        rows.append(pd.DataFrame({
            "model_id": resp[col_model].astype(str),
            "drug_id": resp[col_drug].astype(str),
            "drug_name": resp[col_drug_name].astype(str) if col_drug_name else "",
            "response_value": resp[col_ic50].astype(float),
            "response_metric": "ln_IC50",
            "dose": float("nan"),
            "source_dataset": "GDSC",
        }))
    if col_auc is not None:
        rows.append(pd.DataFrame({
            "model_id": resp[col_model].astype(str),
            "drug_id": resp[col_drug].astype(str),
            "drug_name": resp[col_drug_name].astype(str) if col_drug_name else "",
            "response_value": resp[col_auc].astype(float),
            "response_metric": "AUC",
            "dose": float("nan"),
            "source_dataset": "GDSC",
        }))
    long_df = pd.concat(rows, ignore_index=True)

    out_path = out_dir / "drug_response" / "gdsc_response_long.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(out_path)
    logger.info("Wrote %d GDSC drug-response rows -> %s", len(long_df), out_path)

    manifest_row = {
        "dataset_name": "GDSC",
        "source_url_or_accession": "https://www.cancerrxgene.org/downloads/bulk_download",
        "download_date": dt.date.today().isoformat(),
        "license_or_access_terms": "free academic/medical use (Wellcome Sanger)",
        "raw_file_path": str(raw_dir.resolve()),
        "processed_file_path": str(out_path.resolve()),
        "organism": "Homo sapiens",
        "modality": "drug_response",
        "controlled_access": False,
        "n_samples": int(long_df["model_id"].nunique()),
        "n_features": int(long_df["drug_id"].nunique()),
        "has_drug_response": True,
        "notes": f"response_file={response_file.name}; compounds_file={compounds_file.name}",
    }
    return manifest_row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw_public/gdsc")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    row = ingest_gdsc(args.raw_dir, args.out_dir)
    logger.info("Manifest row: %s", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
