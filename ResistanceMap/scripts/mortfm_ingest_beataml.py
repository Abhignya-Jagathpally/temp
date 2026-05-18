#!/usr/bin/env python3
"""
scripts/mortfm_ingest_beataml.py
================================
Block B — Beat AML 1.0 ingestion with raw/processed/auto modes.

Beat AML provides patient-level genomic + clinical + ex vivo drug-response
data for ~672 specimens / 562 AML patients. License: open clinical
supplement; some molecular files via dbGaP.

Three modes:
    --mode auto       Prefer processed parquet/tsv under data/processed/ if
                       present; else fall back to raw files (default).
    --mode processed  Use only data/processed/beataml_*.{parquet,tsv} files.
    --mode raw        Use only data/raw/beataml/ files.

Always emits the same canonical outputs:
    data/processed/beataml/beataml_expression.parquet  (samples × genes)
    data/processed/beataml/beataml_drug_response.parquet (long form)
    data/processed/beataml/beataml_clinical.csv         (one row per LabId)
    data/processed/beataml/beataml_sample_map.csv       (LabId ↔ PatientId)
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


_PROCESSED_CANDIDATES = {
    "expression": ["beataml_rpkm.parquet", "beataml_expression.parquet"],
    "drug_response": ["beataml_drug_response.tsv", "beataml_drug_response.parquet"],
    "clinical": ["beataml_clinical.tsv", "beataml_clinical.csv"],
}
_RAW_CANDIDATES = {
    "expression": ["beataml*norm_exp*.txt", "beataml*expression*.tsv"],
    "drug_response": ["beataml*curve_fits*.txt", "beataml*drug*.tsv"],
    "clinical": ["beataml*clinical*.tsv", "beataml*clinical*.xlsx", "*clinical*.tsv"],
}


def _resolve_files(raw_dir: Path, processed_dir: Path, mode: str) -> dict:
    """Return {kind: Path} for expression / drug_response / clinical files.

    Falls back through processed -> raw based on mode; raises FileNotFoundError
    if none match.
    """
    resolved: dict = {}
    for kind in ("expression", "drug_response", "clinical"):
        found = None
        if mode in ("auto", "processed"):
            for pat in _PROCESSED_CANDIDATES[kind]:
                candidates = list(processed_dir.glob(pat))
                if candidates:
                    found = sorted(candidates)[0]
                    break
        if found is None and mode in ("auto", "raw"):
            for pat in _RAW_CANDIDATES[kind]:
                candidates = list(raw_dir.rglob(pat))
                if candidates:
                    found = sorted(candidates)[0]
                    break
        if found is None:
            raise FileNotFoundError(
                f"BeatAML {kind} file not found. Tried processed: {_PROCESSED_CANDIDATES[kind]} "
                f"under {processed_dir.resolve()}, raw: {_RAW_CANDIDATES[kind]} under "
                f"{raw_dir.resolve()}. Download from https://biodev.github.io/BeatAML2/."
            )
        resolved[kind] = found
    return resolved


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def ingest_beataml(raw_dir: Path, processed_dir: Path, out_dir: Path, mode: str = "auto") -> dict:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    files = _resolve_files(raw_dir, processed_dir, mode)
    logger.info("BeatAML files (mode=%s):", mode)
    for k, v in files.items():
        logger.info("  %s = %s", k, v)

    # ---- Expression ---------------------------------------------------
    expr_raw = _read_table(files["expression"])
    # Beat AML processed rpkm: rows=genes (with Gene + Symbol columns), columns=specimens
    # Detect orientation:
    if "Symbol" in expr_raw.columns and "Gene" in expr_raw.columns:
        gene_col = "Symbol"
        meta_cols = ["Gene", "Symbol"]
        expr_long = expr_raw.set_index(gene_col).drop(columns=[c for c in meta_cols if c != gene_col])
        expr = expr_long.T  # specimens × genes
    else:
        # Fall back: assume samples are columns, genes are index
        expr = expr_raw.T if expr_raw.shape[1] > expr_raw.shape[0] else expr_raw
    expr.index.name = "lab_id"
    out_expr = out_dir / "beataml" / "beataml_expression.parquet"
    out_expr.parent.mkdir(parents=True, exist_ok=True)
    expr.to_parquet(out_expr)
    logger.info("Wrote %d specimens x %d genes -> %s", expr.shape[0], expr.shape[1], out_expr)

    # ---- Drug response ------------------------------------------------
    drug = _read_table(files["drug_response"])
    drug.columns = [c.lower() for c in drug.columns]
    sid_col = next((c for c in ["lab_id", "labid", "dbgap_subject_id", "sample_id"] if c in drug.columns), None)
    dname_col = next((c for c in ["inhibitor", "drug_name", "compound"] if c in drug.columns), None)
    auc_col = next((c for c in ["auc"] if c in drug.columns), None)
    ic50_col = next((c for c in ["ic50"] if c in drug.columns), None)
    if not sid_col or not dname_col or not (auc_col or ic50_col):
        raise ValueError(
            f"BeatAML drug-response missing required columns. Have: {list(drug.columns)[:15]}"
        )
    long_parts = []
    if auc_col:
        long_parts.append(pd.DataFrame({
            "sample_id": drug[sid_col].astype(str),
            "lab_id": drug[sid_col].astype(str),
            "drug_id": drug[dname_col].astype(str),
            "drug_name": drug[dname_col].astype(str),
            "response_value": pd.to_numeric(drug[auc_col], errors="coerce"),
            "response_metric": "AUC",
            "source_dataset": "BeatAML",
        }))
    if ic50_col:
        long_parts.append(pd.DataFrame({
            "sample_id": drug[sid_col].astype(str),
            "lab_id": drug[sid_col].astype(str),
            "drug_id": drug[dname_col].astype(str),
            "drug_name": drug[dname_col].astype(str),
            "response_value": pd.to_numeric(drug[ic50_col], errors="coerce"),
            "response_metric": "IC50",
            "source_dataset": "BeatAML",
        }))
    long_df = pd.concat(long_parts, ignore_index=True).dropna(subset=["response_value"])
    out_drug = out_dir / "beataml" / "beataml_drug_response.parquet"
    long_df.to_parquet(out_drug)
    logger.info("Wrote %d BeatAML drug-response rows -> %s", len(long_df), out_drug)

    # ---- Clinical ----------------------------------------------------
    clin = _read_table(files["clinical"])
    id_col = next((c for c in ["LabId", "labid", "lab_id", "dbgap_subject_id", "PatientId"]
                   if c in clin.columns), clin.columns[0])
    pid_col = next((c for c in ["PatientId", "patient_id"] if c in clin.columns), None)
    keep = [id_col]
    if pid_col: keep.append(pid_col)
    for c in ["vitalStatus", "overallSurvival", "responseToInductionTx",
              "ageAtDiagnosis", "ageAtSpecimenAcquisition", "specimenType",
              "ELN2017", "dxAtSpecimenAcquisition", "isRelapse", "isDenovo",
              "FLT3-ITD", "NPM1", "TP53", "currentRegimen"]:
        if c in clin.columns:
            keep.append(c)
    clin_out = clin[keep].copy()
    clin_out = clin_out.rename(columns={id_col: "lab_id", pid_col or "PatientId": "patient_id"})
    out_clin = out_dir / "beataml" / "beataml_clinical.csv"
    clin_out.to_csv(out_clin, index=False)
    logger.info("Wrote %d BeatAML clinical rows -> %s", len(clin_out), out_clin)

    # ---- Sample map --------------------------------------------------
    sample_map = clin_out[["lab_id"] + (["patient_id"] if "patient_id" in clin_out.columns else [])]
    sample_map.to_csv(out_dir / "beataml" / "beataml_sample_map.csv", index=False)

    # ---- Canonical samples_beataml.csv -------------------------------
    samples_df = pd.DataFrame({
        "sample_id": expr.index.astype(str),
        "patient_id_or_model_id": expr.index.astype(str),
        "source_dataset": "BeatAML",
        "disease": "AML",
        "sample_type": "patient_specimen",
        "modality_available": "rna,drug_response",
        "split_group": "train",
    })
    # Inject patient_id when available.
    if "patient_id" in clin_out.columns:
        map_df = clin_out[["lab_id", "patient_id"]].drop_duplicates("lab_id").set_index("lab_id")
        samples_df["patient_id_or_model_id"] = samples_df["sample_id"].map(
            map_df["patient_id"].astype(str)
        ).fillna(samples_df["sample_id"])
    samples_path = out_dir / "metadata" / "samples_beataml.csv"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(samples_path, index=False)
    logger.info("Wrote canonical samples_beataml.csv: %d rows", len(samples_df))

    # ---- Acceptance check --------------------------------------------
    n_with_expression = expr.shape[0]
    n_with_drug = long_df["lab_id"].nunique()
    expr_drug_overlap = set(expr.index.astype(str)) & set(long_df["lab_id"].astype(str))
    pass_threshold = (n_with_expression >= 300) and (n_with_drug >= 100)
    logger.info(
        "BeatAML acceptance: %d expression specimens, %d drug-response specimens, "
        "%d overlap -- gate=%s (threshold: 300 expr + 100 drug)",
        n_with_expression, n_with_drug, len(expr_drug_overlap),
        "PASS" if pass_threshold else "FAIL",
    )

    return {
        "dataset_name": "BeatAML_1.0",
        "source_url_or_accession": "https://biodev.github.io/BeatAML2/",
        "download_date": dt.date.today().isoformat(),
        "license_or_access_terms": "Open clinical supplement; some molecular via dbGaP",
        "raw_file_path": str(raw_dir.resolve()),
        "processed_file_path": str(out_expr.resolve()),
        "organism": "Homo sapiens",
        "disease": "AML",
        "modality": "rna,drug_response,clinical",
        "controlled_access": False,
        "n_samples": int(n_with_expression),
        "has_drug_response": True,
        "has_outcomes": True,
        "notes": f"mode={mode}; expression={files['expression'].name}; "
                 f"drug={files['drug_response'].name}; clinical={files['clinical'].name}; "
                 f"acceptance_gate={'PASS' if pass_threshold else 'FAIL'} "
                 f"(n_expr={n_with_expression}, n_drug={n_with_drug}, overlap={len(expr_drug_overlap)})",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/beataml")
    ap.add_argument("--processed-dir", default="data/processed",
                    help="Directory containing the already-processed beataml_*.{parquet,tsv} files")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--mode", default="auto", choices=("auto", "raw", "processed"))
    args = ap.parse_args()
    row = ingest_beataml(Path(args.raw_dir), Path(args.processed_dir),
                          Path(args.out_dir), mode=args.mode)
    logger.info("Manifest row: %s", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
