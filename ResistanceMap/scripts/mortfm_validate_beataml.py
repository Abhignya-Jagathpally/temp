#!/usr/bin/env python3
"""
scripts/mortfm_validate_beataml.py
==================================
Block B-1 — validate the processed BeatAML data contract.

Re-uses :mod:`resistancemap.data.manifest_schemas` validators where possible
and adds a focused acceptance check (>=300 expression specimens / >=100 with
drug response / specimen-disjoint split feasible).

No training logic is repeated here — this is a structural validator only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_validate_beataml")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed/beataml")
    ap.add_argument("--out", default="logs/mortfm/beataml_validation_report.json")
    args = ap.parse_args()

    proc = Path(args.processed_dir)
    files = {
        "expression": proc / "beataml_expression.parquet",
        "drug_response": proc / "beataml_drug_response.parquet",
        "clinical": proc / "beataml_clinical.csv",
        "sample_map": proc / "beataml_sample_map.csv",
    }
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"BeatAML validation: missing {missing}. Run scripts/mortfm_ingest_beataml.py --mode auto first."
        )

    expr = pd.read_parquet(files["expression"])
    drug = pd.read_parquet(files["drug_response"])
    clin = pd.read_csv(files["clinical"])
    smap = pd.read_csv(files["sample_map"])

    expr_specs = set(expr.index.astype(str))
    drug_specs = set(drug["lab_id"].astype(str))
    overlap = expr_specs & drug_specs
    drug_per_specimen = drug.groupby("lab_id")["drug_name"].nunique()

    # Patient-disjoint feasibility: every specimen must map to a patient_id.
    pid_map = (
        smap.dropna(subset=["lab_id"]).set_index("lab_id")["patient_id"].astype(str).to_dict()
        if "patient_id" in smap.columns else {}
    )
    specimens_with_patient = sum(1 for s in expr_specs if s in pid_map)

    # Acceptance gate (matches the user-supplied gate_beataml_specimen_drug_response).
    gate = {
        "min_specimens_with_expression": 300,
        "min_specimens_with_drug_response": 100,
        "min_drugs": 20,
        "require_specimen_disjoint_split": True,
    }
    metrics = {
        "n_expression_specimens": len(expr_specs),
        "n_drug_specimens": len(drug_specs),
        "n_overlap_specimens": len(overlap),
        "n_unique_drugs": int(drug["drug_name"].nunique()),
        "median_drugs_per_specimen": float(drug_per_specimen.median()) if len(drug_per_specimen) else 0.0,
        "n_clinical_patients": int(clin["patient_id"].nunique()) if "patient_id" in clin.columns else 0,
        "n_specimens_mapped_to_patient": specimens_with_patient,
        "specimen_disjoint_split_feasible": specimens_with_patient == len(expr_specs),
    }
    pass_ = (
        metrics["n_expression_specimens"] >= gate["min_specimens_with_expression"]
        and metrics["n_drug_specimens"] >= gate["min_specimens_with_drug_response"]
        and metrics["n_unique_drugs"] >= gate["min_drugs"]
        and (metrics["specimen_disjoint_split_feasible"] or not gate["require_specimen_disjoint_split"])
    )
    report = {
        "passed": pass_, "thresholds": gate, "metrics": metrics,
        "drug_response_metric_distribution": drug["response_metric"].value_counts().to_dict(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("BeatAML validation %s -- expr=%d, drug=%d, overlap=%d, drugs=%d -> %s",
                "PASS" if pass_ else "FAIL",
                metrics["n_expression_specimens"], metrics["n_drug_specimens"],
                metrics["n_overlap_specimens"], metrics["n_unique_drugs"], args.out)
    return 0 if pass_ else 1


if __name__ == "__main__":
    sys.exit(main())
