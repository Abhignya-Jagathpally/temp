#!/usr/bin/env python3
"""Build a patient x top-K-gene RNA feature table for the lakehouse survival join.

Reads MMRF gene_expression.tsv (genes x aliquots) + samples.tsv (aliquot ->
patient), picks the EARLIEST timepoint aliquot per patient (most baseline),
selects the top-K highest-variance genes (on log1p), and writes a wide
patient(submitter_id) x rna_<gene> parquet that stage_confirmed left-joins.

No fabrication: only real expression; unmapped aliquots are dropped, not invented.

  python scripts/build_mmrf_rna_features.py --top-k 200
  -> data/processed/mmrf_patient_rna_topk.parquet
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_mmrf_rna")

EXPR = Path("data/raw/mmrf_commpass/gene_expression.tsv")
SAMPLES = Path("data/raw/mmrf_commpass/samples.tsv")
OUT = Path("data/processed/mmrf_patient_rna_topk.parquet")
_T = re.compile(r"_T(\d+)_")


def _timepoint(aliquot: str) -> int:
    m = _T.search(aliquot)
    return int(m.group(1)) if m else 99


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=200)
    args = ap.parse_args()

    if not EXPR.exists():
        raise SystemExit(f"missing {EXPR} — run scripts/stage_mmrf_rna_open.py first")
    if not SAMPLES.exists():
        raise SystemExit(f"missing {SAMPLES} — run scripts/pull_mmrf_clinical.py first")

    # aliquot_submitter_id -> patient submitter_id
    smp = pd.read_csv(SAMPLES, sep="\t", dtype=str)
    a2p = dict(zip(smp["aliquot_submitter_id"], smp["submitter_id"]))

    logger.info("Loading expression matrix %s ...", EXPR)
    expr = pd.read_csv(EXPR, sep="\t", index_col=0)
    expr.index = expr.index.astype(str).str.split(".").str[0]  # strip Ensembl version
    expr = expr[~expr.index.duplicated(keep="first")]
    logger.info("  expression: %d genes x %d aliquots", expr.shape[0], expr.shape[1])

    # top-K high-variance genes on log1p
    log = np.log1p(expr.astype("float32"))
    var = log.var(axis=1)
    top = var.sort_values(ascending=False).head(args.top_k).index
    sub = log.loc[top]                      # K genes x aliquots
    wide = sub.T                            # aliquots x K genes
    wide.index.name = "aliquot_submitter_id"
    wide = wide.reset_index()

    # map aliquot -> patient, keep earliest timepoint per patient
    wide["submitter_id"] = wide["aliquot_submitter_id"].map(a2p)
    wide = wide.dropna(subset=["submitter_id"])
    wide["_tp"] = wide["aliquot_submitter_id"].map(_timepoint)
    wide = wide.sort_values("_tp").groupby("submitter_id", as_index=False).first()
    wide = wide.drop(columns=["aliquot_submitter_id", "_tp"])
    wide.columns = ["submitter_id"] + [f"rna_{g}" for g in wide.columns if g != "submitter_id"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUT, index=False)
    logger.info("Wrote %s: %d patients x %d RNA features (baseline timepoint)",
                OUT, len(wide), wide.shape[1] - 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
