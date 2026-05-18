#!/usr/bin/env python3
"""
scripts/mortfm_ingest_uniprot.py
================================
Block A — UniProt protein-sequence ingestion.

Reads a UniProt FASTA (``data/raw_public/uniprot/UP000005640_9606.fasta.gz``)
and a UniProt ID-mapping export, and emits:

* ``data/processed/graphs/protein_nodes.csv`` (uniprot_id, gene_symbol, length)
* ``data/processed/graphs/protein_sequences.parquet`` (uniprot_id, sequence)

License: UniProt is CC BY 4.0.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_ingest_uniprot")


def _iter_fasta(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        uid, gene, seq_parts = None, None, []
        for line in f:
            if line.startswith(">"):
                if uid is not None:
                    yield uid, gene, "".join(seq_parts)
                header = line[1:].strip()
                # ">sp|Q02223|TNR17_HUMAN ..." header format
                parts = header.split("|")
                uid = parts[1] if len(parts) >= 2 else parts[0]
                gene = None
                # gene symbol lives in 'GN=GENE'
                for tok in header.split():
                    if tok.startswith("GN="):
                        gene = tok[3:]
                        break
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if uid is not None:
            yield uid, gene, "".join(seq_parts)


def ingest_uniprot(raw_dir: Path, out_dir: Path) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    fastas = list(raw_dir.glob("UP*9606*.fasta*"))
    if not fastas:
        raise FileNotFoundError(
            f"UniProt ingestion: expected UP*9606*.fasta(.gz) under {raw_dir.resolve()}. "
            f"Download from https://www.uniprot.org/proteomes/UP000005640."
        )
    fasta_path = sorted(fastas)[0]
    logger.info("Reading UniProt FASTA %s ...", fasta_path)
    rows = list(_iter_fasta(fasta_path))
    df = pd.DataFrame(rows, columns=["uniprot_id", "gene_symbol", "sequence"])
    df["length"] = df["sequence"].str.len()
    out_seq = out_dir / "graphs" / "protein_sequences.parquet"
    out_nodes = out_dir / "graphs" / "protein_nodes.csv"
    out_seq.parent.mkdir(parents=True, exist_ok=True)
    df[["uniprot_id", "sequence"]].to_parquet(out_seq)
    df[["uniprot_id", "gene_symbol", "length"]].to_csv(out_nodes, index=False)
    logger.info("Wrote %d UniProt protein records -> %s, %s", len(df), out_seq, out_nodes)

    return {
        "dataset_name": "UniProt",
        "source_url_or_accession": "https://www.uniprot.org/proteomes/UP000005640",
        "download_date": dt.date.today().isoformat(),
        "license_or_access_terms": "CC BY 4.0",
        "raw_file_path": str(raw_dir.resolve()),
        "processed_file_path": str(out_seq.resolve()),
        "organism": "Homo sapiens",
        "modality": "protein_sequence",
        "controlled_access": False,
        "n_features": int(len(df)),
        "notes": f"fasta={fasta_path.name}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw_public/uniprot")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    row = ingest_uniprot(args.raw_dir, args.out_dir)
    logger.info("Manifest row: %s", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
