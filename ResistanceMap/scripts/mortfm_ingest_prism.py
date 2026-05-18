#!/usr/bin/env python3
"""
scripts/mortfm_ingest_prism.py
==============================
Block A — PRISM repurposing-hub drug-response ingestion.

Reads a PRISM secondary-screen export (``secondary-screen-dose-response-curve-parameters.csv``)
and emits:

* ``data/processed/drug_response/prism_response_long.parquet``

License: PRISM data are released under DepMap CC-BY-4.0.
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
logger = logging.getLogger("mortfm_ingest_prism")


def ingest_prism(raw_dir: Path, out_dir: Path) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    files = list(raw_dir.glob("secondary-screen-dose-response-curve-parameters*.csv"))
    if not files:
        raise FileNotFoundError(
            f"PRISM ingestion: expected secondary-screen-dose-response-curve-parameters*.csv "
            f"under {raw_dir.resolve()}. Download from DepMap PRISM Repurposing release."
        )
    src = sorted(files)[0]
    logger.info("Reading PRISM %s", src)
    df = pd.read_csv(src, low_memory=False)
    cols_needed = {"depmap_id", "name", "auc"}
    missing = cols_needed - set(df.columns)
    if missing:
        raise ValueError(
            f"PRISM file missing expected columns: {missing}; available: {list(df.columns)[:20]}"
        )
    long_df = pd.DataFrame({
        "model_id": df["depmap_id"].astype(str),
        "drug_id": df["name"].astype(str),
        "drug_name": df["name"].astype(str),
        "response_value": df["auc"].astype(float),
        "response_metric": "AUC",
        "dose": float("nan"),
        "source_dataset": "PRISM",
    })
    out_path = out_dir / "drug_response" / "prism_response_long.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(out_path)
    logger.info("Wrote %d PRISM rows -> %s", len(long_df), out_path)
    return {
        "dataset_name": "PRISM",
        "source_url_or_accession": "https://depmap.org/portal/data_page/?tab=drug_screens",
        "download_date": dt.date.today().isoformat(),
        "license_or_access_terms": "CC BY 4.0 (DepMap)",
        "raw_file_path": str(raw_dir.resolve()),
        "processed_file_path": str(out_path.resolve()),
        "organism": "Homo sapiens",
        "modality": "drug_response",
        "controlled_access": False,
        "n_samples": int(long_df["model_id"].nunique()),
        "n_features": int(long_df["drug_id"].nunique()),
        "has_drug_response": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw_public/prism")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    row = ingest_prism(args.raw_dir, args.out_dir)
    logger.info("Manifest row: %s", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
