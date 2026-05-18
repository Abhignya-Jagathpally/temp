#!/usr/bin/env python3
"""
scripts/mortfm_ingest_chembl.py
===============================
Block A — ChEMBL drug-target ingestion.

Reads a ChEMBL drug-target export (downloaded via the ChEMBL web interface or
generated from the SQLite/Postgres dump) and emits:

* ``data/processed/drugs/drug_ontology.csv``  (drug_id, drug_name, drug_class, smiles, chembl_id, ...)
* ``data/processed/drugs/drug_targets.csv``   (drug_id, target_gene, target_uniprot, target_type, evidence_source)

Expected file (one of):
    chembl_drug_targets.csv         (custom export with columns drug_chembl_id, target_uniprot, target_gene, ...)
    chembl_drug_indications.csv     (optional; drug + ATC class + indication)

License: ChEMBL is CC BY-SA 3.0.
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
logger = logging.getLogger("mortfm_ingest_chembl")


def ingest_chembl(raw_dir: Path, out_dir: Path) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    target_files = list(raw_dir.glob("*drug_target*.csv")) + list(raw_dir.glob("*chembl*targets*.csv"))
    if not target_files:
        raise FileNotFoundError(
            f"ChEMBL ingestion: expected *drug_target*.csv under {raw_dir.resolve()}. "
            f"Generate via the ChEMBL portal export, live ChEMBL MCP, or "
            f"python3 scripts/mortfm_download_public_data.py --only chembl"
        )
    src = sorted(target_files)[0]
    df = pd.read_csv(src, low_memory=False, dtype=str)
    # Normalise column names.
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if "chembl" in cl and "drug" in cl and "id" in cl: rename[c] = "drug_id"
        elif cl in {"drug_chembl_id", "molecule_chembl_id"}: rename[c] = "drug_id"
        elif cl in {"drug_name", "pref_name"}: rename[c] = "drug_name"
        elif cl in {"target_uniprot", "uniprot", "uniprot_accession"}: rename[c] = "target_uniprot"
        elif cl in {"target_gene", "gene_symbol", "target_name_gene"}: rename[c] = "target_gene"
        elif cl in {"mechanism_of_action", "moa"}: rename[c] = "mechanism_of_action"
    df = df.rename(columns=rename)

    if "drug_id" not in df.columns or "target_uniprot" not in df.columns:
        raise ValueError(
            f"ChEMBL targets file {src.name} missing required columns after normalisation. "
            f"Have: {list(df.columns)[:15]}"
        )

    targets_df = pd.DataFrame({
        "drug_id": df["drug_id"].astype(str),
        "drug_name": df.get("drug_name", pd.Series("", index=df.index)).astype(str),
        "chembl_id": df["drug_id"].astype(str),
        "target_gene": df.get("target_gene", pd.Series("", index=df.index)).astype(str),
        "target_uniprot": df["target_uniprot"].astype(str),
        "target_type": "primary",
        "evidence_source": "ChEMBL",
    })
    out_targets = out_dir / "drugs" / "drug_targets.csv"
    out_targets.parent.mkdir(parents=True, exist_ok=True)
    targets_df.to_csv(out_targets, index=False)
    logger.info("Wrote %d drug-target edges -> %s", len(targets_df), out_targets)

    ontology = (
        df[["drug_id"] + [c for c in ["drug_name", "mechanism_of_action"] if c in df.columns]]
        .drop_duplicates("drug_id")
    )
    if "drug_name" not in ontology.columns:
        ontology["drug_name"] = ontology["drug_id"]
    ontology["canonical_name"] = ontology["drug_name"]
    ontology["drug_class"] = ontology.get("mechanism_of_action", "unknown")
    ontology["chembl_id"] = ontology["drug_id"]
    out_ontology = out_dir / "drugs" / "drug_ontology.csv"
    ontology.to_csv(out_ontology, index=False)
    logger.info("Wrote %d drug ontology rows -> %s", len(ontology), out_ontology)

    return {
        "dataset_name": "ChEMBL",
        "source_url_or_accession": "https://www.ebi.ac.uk/chembl/",
        "download_date": dt.date.today().isoformat(),
        "license_or_access_terms": "CC BY-SA 3.0",
        "raw_file_path": str(raw_dir.resolve()),
        "processed_file_path": str(out_targets.resolve()),
        "organism": "Homo sapiens",
        "modality": "drug_target,drug_ontology",
        "controlled_access": False,
        "n_features": int(ontology["drug_id"].nunique()),
        "notes": f"source={src.name}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw_public/chembl")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    row = ingest_chembl(args.raw_dir, args.out_dir)
    logger.info("Manifest row: %s", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
