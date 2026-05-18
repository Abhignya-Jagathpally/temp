#!/usr/bin/env python3
"""
scripts/mortfm_ingest_reactome.py
=================================
Block A — Reactome pathway-membership ingestion.

Reads ``NCBI2Reactome_All_Levels.txt`` (or the HGNC variant) and emits:

* ``data/processed/graphs/reactome_membership.parquet`` (hgnc_symbol, reactome_id, pathway_name)

License: Reactome is CC BY 4.0.
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
logger = logging.getLogger("mortfm_ingest_reactome")


def ingest_reactome(raw_dir: Path, out_dir: Path) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    # Prefer HGNC2Reactome if present, else NCBI2Reactome.
    candidates = (
        list(raw_dir.rglob("HGNC2Reactome*All_Levels*.txt"))
        + list(raw_dir.rglob("NCBI2Reactome*All_Levels*.txt"))
        + list(raw_dir.rglob("UniProt2Reactome*All_Levels*.txt"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"Reactome ingestion: expected {{HGNC,NCBI,UniProt}}2Reactome_All_Levels.txt under "
            f"{raw_dir.resolve()}. Download from https://reactome.org/download/current/."
        )
    src = sorted(candidates)[0]
    logger.info("Reading Reactome membership from %s", src)
    df = pd.read_csv(
        src, sep="\t", header=None, low_memory=False,
        names=["source_id", "reactome_id", "url", "pathway_name", "evidence_code", "species"],
    )
    df = df[df["species"].astype(str).str.startswith("Homo sapiens")].copy()
    out_path = out_dir / "graphs" / "reactome_membership.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Caller maps source_id -> HGNC via identifier_mapping; preserve raw column.
    df[["source_id", "reactome_id", "pathway_name"]].to_parquet(out_path)
    logger.info("Wrote %d Reactome membership rows -> %s", len(df), out_path)

    return {
        "dataset_name": "Reactome",
        "source_url_or_accession": "https://reactome.org/download/current/",
        "download_date": dt.date.today().isoformat(),
        "license_or_access_terms": "CC BY 4.0",
        "raw_file_path": str(raw_dir.resolve()),
        "processed_file_path": str(out_path.resolve()),
        "organism": "Homo sapiens",
        "modality": "pathway_membership",
        "controlled_access": False,
        "n_features": int(df["reactome_id"].nunique()),
        "notes": f"source={src.name}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw_public/reactome")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    row = ingest_reactome(args.raw_dir, args.out_dir)
    logger.info("Manifest row: %s", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
