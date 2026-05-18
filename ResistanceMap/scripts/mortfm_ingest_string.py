#!/usr/bin/env python3
"""
scripts/mortfm_ingest_string.py
===============================
Block A — STRING PPI ingestion.

Reads a STRING dump (``data/raw_public/string/``) and emits:

* ``data/processed/graphs/string_edges.parquet`` (source_id, target_id, combined_score)
* ``data/processed/graphs/string_aliases.parquet`` (string_id, uniprot_id) — fed to identifier_mapping

Expected files (STRING v12.0):
    9606.protein.links.full.v12.0.txt.gz
    9606.protein.aliases.v12.0.txt.gz

License: STRING is CC BY 4.0.
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
logger = logging.getLogger("mortfm_ingest_string")


def ingest_string(raw_dir: Path, out_dir: Path, *, score_threshold: int = 700) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    links = list(raw_dir.glob("*protein.links*.txt*"))
    aliases = list(raw_dir.glob("*protein.aliases*.txt*"))
    if not links or not aliases:
        raise FileNotFoundError(
            f"STRING ingestion: expected protein.links and protein.aliases under {raw_dir.resolve()}. "
            f"Download from https://string-db.org/cgi/download (taxon 9606) or "
            f"python3 scripts/mortfm_download_public_data.py --only string"
        )
    links_path = sorted(links)[0]
    aliases_path = sorted(aliases)[0]

    logger.info("Reading STRING links %s (threshold>=%d) ...", links_path, score_threshold)
    edges = pd.read_csv(links_path, sep=" ", low_memory=False)
    if "combined_score" not in edges.columns:
        raise ValueError(f"STRING links file lacks combined_score column; got: {list(edges.columns)}")
    edges = edges[edges["combined_score"] >= score_threshold].copy()
    edges = edges.rename(columns={"protein1": "source_id", "protein2": "target_id"})
    out_edges = out_dir / "graphs" / "string_edges.parquet"
    out_edges.parent.mkdir(parents=True, exist_ok=True)
    edges[["source_id", "target_id", "combined_score"]].to_parquet(out_edges)
    logger.info("Wrote %d STRING edges (>= %d) -> %s", len(edges), score_threshold, out_edges)

    logger.info("Reading STRING aliases %s ...", aliases_path)
    al = pd.read_csv(aliases_path, sep="\t", low_memory=False)
    # Column convention: ['#string_protein_id', 'alias', 'source'] in older releases,
    # or ['string_protein_id', 'alias', 'source'] in newer ones.
    al.columns = [c.lstrip("#") for c in al.columns]
    if "source" not in al.columns or "alias" not in al.columns or "string_protein_id" not in al.columns:
        raise ValueError(f"STRING aliases file lacks expected columns; got: {list(al.columns)}")
    up = al[al["source"].astype(str).str.contains("UniProt", case=False, na=False)].copy()
    out_alias = out_dir / "graphs" / "string_aliases.parquet"
    pd.DataFrame({
        "string_id": up["string_protein_id"].astype(str),
        "uniprot_id": up["alias"].astype(str),
    }).to_parquet(out_alias)
    logger.info("Wrote %d STRING<->UniProt alias rows -> %s", len(up), out_alias)

    return {
        "dataset_name": "STRING",
        "source_url_or_accession": "https://string-db.org/cgi/download (taxon 9606)",
        "download_date": dt.date.today().isoformat(),
        "license_or_access_terms": "CC BY 4.0",
        "raw_file_path": str(raw_dir.resolve()),
        "processed_file_path": str(out_edges.resolve()),
        "organism": "Homo sapiens",
        "modality": "ppi_graph",
        "controlled_access": False,
        "n_features": int(len(edges)),
        "notes": f"links={links_path.name}; aliases={aliases_path.name}; threshold={score_threshold}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw_public/string")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--score-threshold", type=int, default=700)
    args = ap.parse_args()
    row = ingest_string(args.raw_dir, args.out_dir, score_threshold=args.score_threshold)
    logger.info("Manifest row: %s", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
