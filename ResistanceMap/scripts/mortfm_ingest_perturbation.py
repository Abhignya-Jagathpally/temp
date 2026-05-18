#!/usr/bin/env python3
"""
scripts/mortfm_ingest_perturbation.py
=====================================
Block D — perturbation / CRISPR / Perturb-seq ingestion.

Wraps :mod:`resistancemap.data.perturbation_loader` into a CLI that produces
the canonical ``data/processed/perturbation/perturbation_response.csv`` long
table consumed by the counterfactual head.

Inputs (any combination, at least one):
    --crispr   path to DepMap CRISPRGeneEffect.csv
    --perturbseq  path to .h5ad with obs[perturbation]
    --drugscreen  path to a long-form drug screen with cell_line + drug + viability
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.perturbation_loader import (
    load_crispr_gene_effect,
    load_perturbseq_deltas,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_ingest_perturbation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crispr", default=None)
    ap.add_argument("--perturbseq", default=None)
    ap.add_argument("--drugscreen", default=None)
    ap.add_argument("--out", default="data/processed/perturbation/perturbation_response.csv")
    args = ap.parse_args()
    if not any([args.crispr, args.perturbseq, args.drugscreen]):
        raise SystemExit("Provide at least one of --crispr, --perturbseq, --drugscreen")

    rows: list[pd.DataFrame] = []

    if args.crispr:
        sample_ids, gene_symbols, values = load_crispr_gene_effect(args.crispr)
        # Long-form: one row per (sample, gene).
        long = []
        for i, sid in enumerate(sample_ids):
            for j, gene in enumerate(gene_symbols):
                v = float(values[i, j].item())
                if v != v:  # NaN
                    continue
                long.append({
                    "sample_id_or_model_id": sid,
                    "perturbation_type": "CRISPR_KO",
                    "perturbed_gene": gene,
                    "perturbed_protein": "",
                    "drug_id": "",
                    "dose": "",
                    "time_hours": "",
                    "response_metric": "gene_effect",
                    "response_value": v,
                    "source_dataset": "DepMap_CRISPR",
                })
        if long:
            rows.append(pd.DataFrame(long))
            logger.info("CRISPR: %d non-NaN gene-effect rows", len(long))

    if args.perturbseq:
        deltas = load_perturbseq_deltas(args.perturbseq)
        # Reduce per perturbation to a scalar magnitude (L2 norm of expression delta).
        rows.append(pd.DataFrame([{
            "sample_id_or_model_id": "perturbseq_pool",
            "perturbation_type": "perturbseq",
            "perturbed_gene": k,
            "perturbed_protein": "",
            "drug_id": "",
            "dose": "",
            "time_hours": "",
            "response_metric": "expression_delta_l2",
            "response_value": float(v.norm()),
            "source_dataset": "Perturb-seq",
        } for k, v in deltas.items()]))

    if args.drugscreen:
        ds = pd.read_csv(args.drugscreen, low_memory=False)
        # Expects columns: cell_line, drug, viability (or response).
        col_cell = next((c for c in ["cell_line", "sample_id", "model_id"] if c in ds.columns), ds.columns[0])
        col_drug = next((c for c in ["drug_id", "drug_name", "compound"] if c in ds.columns), None)
        col_val = next((c for c in ["viability", "response_value", "auc"] if c in ds.columns), None)
        if not (col_drug and col_val):
            raise ValueError("--drugscreen file missing drug + viability columns")
        rows.append(pd.DataFrame({
            "sample_id_or_model_id": ds[col_cell].astype(str),
            "perturbation_type": "drug_treatment",
            "perturbed_gene": "",
            "perturbed_protein": "",
            "drug_id": ds[col_drug].astype(str),
            "dose": "",
            "time_hours": "",
            "response_metric": col_val,
            "response_value": ds[col_val].astype(float),
            "source_dataset": "drugscreen",
        }))

    if not rows:
        logger.error("No perturbation rows ingested -- aborting.")
        return 1

    out = pd.concat(rows, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Wrote %d total perturbation rows -> %s", len(out), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
