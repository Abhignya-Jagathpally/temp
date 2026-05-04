"""S1a: Build MMRF baseline (per-patient earliest-visit) expression matrix.

Inputs (already on disk):
    data/raw/mmrf_commpass/gene_expression.tsv  # 859 aliquots × genes (TPM)
    data/raw/mmrf_commpass/file_to_case.tsv     # aliquot_submitter_id -> patient_submitter_id

Output:
    data/processed/mmrf_baseline_expression.parquet
    data/processed/mmrf_baseline_metadata.tsv

Strategy:
    Aliquot IDs encode timepoint as `_T{N}_` (e.g. `_T1_TSMRU_L05318`).
    For each patient, keep the earliest available timepoint as the baseline
    snapshot. If a patient has paired baseline + relapse aliquots, the relapse
    one is dropped here (it will be brought back in Sprint 2 as a paired
    trajectory observation).

Per V10_FOUNDATION_MODEL_PAPER_SPEC.md §2.1, the spec quotes 1313 patient-snapshots
(994 baseline-only + 319 Recurrent-BM). This script reports the actual N
delivered by the RNA-seq files on disk — no fabrication.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "mmrf_commpass"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

ALIQUOT_TIMEPOINT_RE = re.compile(r"_T(\d+)_")


def parse_aliquot(aliquot_id: str) -> tuple[str, int]:
    """Extract (patient_id, timepoint_int) from MMRF aliquot id.

    Example: ``MMRF_1817_1_BM_CD138pos_T2_TSMRU_L05318`` → ("MMRF_1817", 2).
    Returns timepoint=999 if the regex fails (so it loses to any matched one).
    """
    m = ALIQUOT_TIMEPOINT_RE.search(aliquot_id)
    tp = int(m.group(1)) if m else 999
    parts = aliquot_id.split("_")
    patient = "_".join(parts[:2])
    return patient, tp


def main() -> None:
    expr_path = RAW / "gene_expression.tsv"
    f2c_path = RAW / "file_to_case.tsv"
    if not expr_path.exists():
        sys.exit(f"missing {expr_path}")
    if not f2c_path.exists():
        sys.exit(f"missing {f2c_path}")

    print(f"[S1a] reading {expr_path} ...")
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    print(f"[S1a] expression: {expr.shape[0]} genes × {expr.shape[1]} aliquots")

    f2c = pd.read_csv(f2c_path, sep="\t")
    f2c = f2c[f2c["modality"] == "rna"].copy()
    f2c = f2c.drop_duplicates(subset=["aliquot_submitter_id"])
    print(f"[S1a] file_to_case rna rows: {len(f2c)}")

    # Build aliquot → (patient, timepoint) for every aliquot in the matrix
    rows = []
    for aliquot in expr.columns:
        patient, tp = parse_aliquot(aliquot)
        rows.append({"aliquot_submitter_id": aliquot, "patient_id": patient, "timepoint": tp})
    meta = pd.DataFrame(rows)

    # Sanity-check against file_to_case
    cross = meta.merge(
        f2c[["aliquot_submitter_id", "patient_submitter_id"]],
        on="aliquot_submitter_id",
        how="left",
    )
    mismatch = (cross["patient_id"] != cross["patient_submitter_id"]).sum()
    print(f"[S1a] aliquot↔patient parse mismatches vs file_to_case: {mismatch}")

    # Keep earliest timepoint per patient
    meta_sorted = meta.sort_values(["patient_id", "timepoint"])
    baseline = meta_sorted.groupby("patient_id", as_index=False).first()
    print(f"[S1a] baseline (earliest-T per patient): {len(baseline)} patients")

    print("[S1a] timepoint distribution at baseline:")
    print(baseline["timepoint"].value_counts().sort_index().to_string())

    # Build the baseline matrix
    keep_aliquots = baseline["aliquot_submitter_id"].tolist()
    baseline_expr = expr[keep_aliquots].T.copy()  # rows=patients, cols=genes
    baseline_expr.index = baseline["patient_id"].values
    baseline_expr.index.name = "patient_id"

    # Drop genes with zero variance (no signal at all)
    var = baseline_expr.var(axis=0)
    keep_genes = var[var > 0].index
    baseline_expr = baseline_expr[keep_genes]
    print(f"[S1a] after zero-var gene drop: {baseline_expr.shape[0]} patients × {baseline_expr.shape[1]} genes")

    # log1p-TPM transform (deferred to S1c — keep TPM raw here for reuse)
    out_parquet = PROC / "mmrf_baseline_expression.parquet"
    baseline_expr.to_parquet(out_parquet)
    print(f"[S1a] wrote {out_parquet}")

    out_meta = PROC / "mmrf_baseline_metadata.tsv"
    baseline.to_csv(out_meta, sep="\t", index=False)
    print(f"[S1a] wrote {out_meta}")

    summary = {
        "n_aliquots_in_matrix": int(expr.shape[1]),
        "n_patients_baseline": int(baseline_expr.shape[0]),
        "n_genes_after_var_filter": int(baseline_expr.shape[1]),
        "spec_target_1313": "spec assumed 994 baseline-only + 319 Recurrent-BM; actual depends on RNA-seq availability on disk",
        "aliquot_patient_mismatch_count": int(mismatch),
    }
    summary_path = ROOT / "paper" / "v8_artifacts" / "v10_sprint1" / "s1a_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[S1a] wrote {summary_path}")


if __name__ == "__main__":
    main()
