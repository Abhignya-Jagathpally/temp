#!/usr/bin/env python3
"""
scripts/mortfm_build_scrna_manifest.py
======================================
Block C-2 — build the canonical single-cell manifest.

Scans ``data/processed/single_cell/scrna/*.h5ad`` and emits a manifest CSV
with the fields the Block-C trainer + the trajectory claim gate need:

    dataset_id, h5ad_path, n_cells, n_genes, n_hvgs, n_samples,
    disease_stages, has_counts_layer, has_real_time_units,
    has_pseudotime_only, qc_report_path

Honest behaviour
----------------
* ``has_pseudotime_only`` is True by default and only flips to False if the
  AnnData has a ``timepoint_months`` obs column with non-NaN entries.
* ``has_real_time_units`` is the inverse: requires real calendar time.
* The manifest is the single source of truth the trajectory gate reads from.
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
logger = logging.getLogger("mortfm_build_scrna_manifest")


def _summarise_h5ad(path: Path) -> dict:
    import anndata as ad
    a = ad.read_h5ad(str(path), backed="r")
    stages = (
        sorted(a.obs["disease_stage"].astype(str).unique().tolist())
        if "disease_stage" in a.obs.columns else []
    )
    has_counts = "counts" in (a.layers.keys() if hasattr(a.layers, "keys") else [])
    has_real_time = (
        "timepoint_months" in a.obs.columns
        and a.obs["timepoint_months"].dropna().shape[0] > 0
    )
    n_hvgs = int(a.var["highly_variable"].sum()) if "highly_variable" in a.var.columns else 0
    sample_col = next((c for c in ["sample", "patient_id", "donor"] if c in a.obs.columns), None)
    n_samples = int(a.obs[sample_col].astype(str).nunique()) if sample_col else 0
    return {
        "h5ad_path": str(path.resolve()),
        "n_cells": int(a.n_obs),
        "n_genes": int(a.n_vars),
        "n_hvgs": n_hvgs,
        "n_samples": n_samples,
        "disease_stages": ",".join(stages),
        "has_counts_layer": has_counts,
        "has_real_time_units": bool(has_real_time),
        "has_pseudotime_only": not bool(has_real_time),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad-dir", default="data/processed/single_cell/scrna")
    ap.add_argument("--out", default="data/processed/single_cell/scrna_manifest.csv")
    args = ap.parse_args()

    h5ad_dir = Path(args.h5ad_dir)
    if not h5ad_dir.is_dir():
        raise FileNotFoundError(f"scRNA dir not found: {h5ad_dir.resolve()}")
    rows = []
    for p in sorted(h5ad_dir.glob("*.h5ad")):
        gse = p.stem
        info = _summarise_h5ad(p)
        info["dataset_id"] = gse
        qc = h5ad_dir / f"{gse}_qc_report.json"
        info["qc_report_path"] = str(qc.resolve()) if qc.exists() else ""
        rows.append(info)

    if not rows:
        logger.error("No .h5ad files found under %s", h5ad_dir)
        return 1

    df = pd.DataFrame(rows)
    cols = ["dataset_id", "h5ad_path", "n_cells", "n_genes", "n_hvgs", "n_samples",
            "disease_stages", "has_counts_layer", "has_real_time_units",
            "has_pseudotime_only", "qc_report_path"]
    df = df[cols]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Wrote scRNA manifest: %d datasets, total %d cells, %d HVGs total -> %s",
                len(df), int(df["n_cells"].sum()), int(df["n_hvgs"].sum()), out)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
