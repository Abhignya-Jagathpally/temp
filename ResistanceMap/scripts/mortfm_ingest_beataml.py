#!/usr/bin/env python3
"""
scripts/mortfm_ingest_beataml.py
================================
Block B — Beat AML 1.0 ingestion.

Beat AML provides patient-level genomic + clinical + ex vivo drug-response
data for ~672 specimens / 562 AML patients. License: open clinical
supplement; some molecular files via dbGaP. Source: https://biodev.github.io/BeatAML2/

Expected files (Beat AML 1.0 supplement):
    beataml_waves1to4_norm_exp_dbgap.txt       (RNA-seq normalized expression)
    beataml_wv1to4_clinical.xlsx OR .tsv        (clinical metadata)
    beataml_probit_curve_fits_v4_dbgap.txt     (ex vivo drug response: AUC, IC50, etc.)

Outputs:
    data/processed/omics/beataml_rna_matrix.parquet     (samples x genes)
    data/processed/drug_response/beataml_response_long.parquet
    data/processed/clinical/beataml_outcomes.csv
    data/processed/metadata/samples_beataml.csv
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
logger = logging.getLogger("mortfm_ingest_beataml")


def _find_one(raw_dir: Path, patterns: list[str], label: str) -> Path:
    for pat in patterns:
        hits = list(raw_dir.rglob(pat))
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(
        f"Beat AML ingestion: {label} not found under {raw_dir.resolve()} "
        f"(tried {patterns}). Download from https://biodev.github.io/BeatAML2/."
    )


def ingest_beataml(raw_dir: Path, out_dir: Path) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    expr_path = _find_one(raw_dir, ["beataml*norm_exp*.txt", "beataml*expression*.tsv"], "RNA-seq")
    clin_path = _find_one(raw_dir, ["beataml*clinical*.tsv", "beataml*clinical*.xlsx", "*clinical*.tsv"],
                           "clinical")
    drug_path = _find_one(raw_dir, ["beataml*curve_fits*.txt", "beataml*drug*.tsv"],
                           "drug response")

    logger.info("Reading Beat AML RNA-seq from %s", expr_path)
    expr = pd.read_csv(expr_path, sep="\t", low_memory=False, index_col=0)
    out_rna = out_dir / "omics" / "beataml_rna_matrix.parquet"
    out_rna.parent.mkdir(parents=True, exist_ok=True)
    expr.T.to_parquet(out_rna)   # samples x genes
    logger.info("Wrote %d samples x %d genes -> %s", expr.shape[1], expr.shape[0], out_rna)

    logger.info("Reading Beat AML clinical from %s", clin_path)
    if clin_path.suffix.lower() == ".xlsx":
        clin = pd.read_excel(clin_path)
    else:
        clin = pd.read_csv(clin_path, sep="\t", low_memory=False)
    id_col = next((c for c in ["dbgap_subject_id", "patientId", "Patient_ID", "patient_id"]
                   if c in clin.columns), clin.columns[0])
    clin = clin.rename(columns={id_col: "patient_id"})

    # Outcomes table.
    out_clin = out_dir / "clinical" / "beataml_outcomes.csv"
    out_clin.parent.mkdir(parents=True, exist_ok=True)
    keep = ["patient_id"]
    for c in ["overallSurvival", "vitalStatus", "consensus_response", "FAB_Blast_Morphology",
             "responseToInductionTx", "ageAtDiagnosis", "specimenType"]:
        if c in clin.columns:
            keep.append(c)
    clin[keep].to_csv(out_clin, index=False)
    logger.info("Wrote %d clinical rows -> %s", len(clin), out_clin)

    logger.info("Reading Beat AML drug response from %s", drug_path)
    drug = pd.read_csv(drug_path, sep="\t", low_memory=False)
    sid_col = next((c for c in ["dbgap_subject_id", "lab_id", "sample_id"] if c in drug.columns), drug.columns[0])
    dname_col = next((c for c in ["inhibitor", "drug_name", "compound"] if c in drug.columns), None)
    val_col = next((c for c in ["auc", "ic50", "AUC", "IC50"] if c in drug.columns), None)
    if dname_col is None or val_col is None:
        raise ValueError(
            f"Beat AML drug file missing expected columns; have: {list(drug.columns)[:15]}"
        )
    long_df = pd.DataFrame({
        "sample_id": drug[sid_col].astype(str),
        "model_id": drug[sid_col].astype(str),
        "drug_id": drug[dname_col].astype(str),
        "drug_name": drug[dname_col].astype(str),
        "response_value": drug[val_col].astype(float),
        "response_metric": "AUC" if val_col.lower() == "auc" else "IC50",
        "source_dataset": "BeatAML",
    })
    out_drug = out_dir / "drug_response" / "beataml_response_long.parquet"
    out_drug.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(out_drug)
    logger.info("Wrote %d Beat AML drug-response rows -> %s", len(long_df), out_drug)

    samples = pd.DataFrame({
        "sample_id": expr.columns.astype(str),
        "patient_id_or_model_id": expr.columns.astype(str),
        "source_dataset": "BeatAML",
        "disease": "AML",
        "sample_type": "patient_specimen",
        "modality_available": "rna,drug_response",
        "split_group": "train",
    })
    out_samples = out_dir / "metadata" / "samples_beataml.csv"
    samples.to_csv(out_samples, index=False)

    return {
        "dataset_name": "BeatAML_1.0",
        "source_url_or_accession": "https://biodev.github.io/BeatAML2/",
        "download_date": dt.date.today().isoformat(),
        "license_or_access_terms": "Open clinical supplement; some molecular via dbGaP",
        "raw_file_path": str(raw_dir.resolve()),
        "processed_file_path": str(out_rna.resolve()),
        "organism": "Homo sapiens",
        "disease": "AML",
        "modality": "rna,drug_response,clinical",
        "controlled_access": False,
        "n_samples": int(len(samples)),
        "has_drug_response": True,
        "has_outcomes": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw_public/beataml")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    row = ingest_beataml(args.raw_dir, args.out_dir)
    logger.info("Manifest row: %s", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
