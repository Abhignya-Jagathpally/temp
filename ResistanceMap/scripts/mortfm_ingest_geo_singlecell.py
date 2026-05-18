#!/usr/bin/env python3
"""
scripts/mortfm_ingest_geo_singlecell.py
=======================================
Block C — GEO single-cell ingestion (scRNA / scATAC for MM and AML).

Expected layout (one directory per GSE accession):

    data/raw_public/geo_single_cell/
        GSE161195/          (relapsed/refractory MM scRNA-seq)
            *.h5ad   or   matrix.mtx + barcodes.tsv + features.tsv
        GSE223060/          (single-cell discovery in MM)
        GSE199373/          (scRNA + scATAC + scCNV in primary samples)
        GSE153380/          (paired ATAC + RNA in MM plasma cells)
        GSE167968/          (RNA + ATAC in purified MM plasma cells)

For each subdirectory containing a recognised .h5ad, this script converts it
to a canonical AnnData under ``data/processed/single_cell/scrna/<GSE>.h5ad``
and registers a row in ``data/processed/metadata/samples_<GSE>.csv``.

This script does NOT download GEO data. Use ``geofetch`` or ``sra-tools`` to
retrieve archives first; this script only re-shapes them.
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
logger = logging.getLogger("mortfm_ingest_geo_singlecell")


_GSE_PROFILE = {
    "GSE161195": {"disease": "MM", "modality": "scrna", "notes": "Relapsed/refractory MM (Cohen et al.)"},
    "GSE223060": {"disease": "MM", "modality": "scrna", "notes": "MM therapeutic-target discovery"},
    "GSE106218": {"disease": "MM", "modality": "scrna", "notes": "Early MM scRNA-seq"},
    "GSE110499": {"disease": "MM", "modality": "scrna", "notes": "MM scRNA-seq cohort"},
    "GSE199373": {"disease": "MM", "modality": "scrna,scatac,sc_cnv", "notes": "Multi-omic primary patient samples"},
    "GSE153380": {"disease": "MM", "modality": "scrna,scatac", "notes": "Paired ATAC+RNA in plasma cells"},
    "GSE167968": {"disease": "MM", "modality": "scrna,scatac", "notes": "Purified MM plasma cells"},
}


def _require_anndata():
    try:
        import anndata as ad
        return ad
    except ImportError as exc:
        raise ImportError("Single-cell ingestion requires anndata: pip install anndata scanpy") from exc


def ingest_gse(gse_dir: Path, out_dir: Path) -> dict | None:
    ad = _require_anndata()
    gse_id = gse_dir.name
    profile = _GSE_PROFILE.get(gse_id, {"disease": "unknown", "modality": "scrna",
                                         "notes": "Unregistered GSE"})

    h5ads = list(gse_dir.rglob("*.h5ad"))
    if not h5ads:
        logger.warning("No .h5ad found in %s — skipping", gse_dir)
        return None

    h5ad_path = sorted(h5ads)[0]
    logger.info("Reading %s -> %s", h5ad_path, gse_id)
    adata = ad.read_h5ad(str(h5ad_path))

    out_path = out_dir / "single_cell" / "scrna" / f"{gse_id}.h5ad"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    logger.info("Wrote canonical %s: %d cells x %d genes", out_path, adata.n_obs, adata.n_vars)

    # Per-cell samples table.
    pid_col = next((c for c in ["patient_id", "donor", "sample_id"] if c in adata.obs.columns), None)
    if pid_col is None:
        adata.obs["patient_id"] = gse_id  # fall back: treat the whole dataset as one patient
        pid_col = "patient_id"
    samples = adata.obs[[pid_col]].drop_duplicates().copy()
    samples.columns = ["patient_id_or_model_id"]
    samples["sample_id"] = samples["patient_id_or_model_id"].astype(str) + "__" + gse_id
    samples["source_dataset"] = gse_id
    samples["disease"] = profile["disease"]
    samples["sample_type"] = "single_cell"
    samples["modality_available"] = profile["modality"]
    samples["split_group"] = "train"
    samples_out = out_dir / "metadata" / f"samples_{gse_id}.csv"
    samples.to_csv(samples_out, index=False)
    logger.info("Wrote %d sample rows -> %s", len(samples), samples_out)

    return {
        "dataset_name": gse_id,
        "source_url_or_accession": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}",
        "download_date": dt.date.today().isoformat(),
        "license_or_access_terms": "GEO (public, varies by submitter)",
        "raw_file_path": str(gse_dir.resolve()),
        "processed_file_path": str(out_path.resolve()),
        "organism": "Homo sapiens",
        "disease": profile["disease"],
        "modality": profile["modality"],
        "controlled_access": False,
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "notes": profile["notes"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw_public/geo_single_cell")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"GEO ingestion: {raw_dir.resolve()} not found. "
            f"Run python3 scripts/mortfm_download_public_data.py --only geo, or use "
            f"geofetch / sra-tools for GSE161195, GSE223060, GSE199373 etc."
        )
    rows = []
    for sub in sorted(raw_dir.iterdir()):
        if sub.is_dir():
            row = ingest_gse(sub, Path(args.out_dir))
            if row:
                rows.append(row)
    logger.info("Ingested %d GEO datasets.", len(rows))
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
