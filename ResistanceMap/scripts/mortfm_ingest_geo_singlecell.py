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
    if h5ads:
        h5ad_path = sorted(h5ads)[0]
        logger.info("Reading %s -> %s", h5ad_path, gse_id)
        adata = ad.read_h5ad(str(h5ad_path))
    else:
        # Fallback: 10x mtx triad (matrix.mtx[.gz] + barcodes + features/genes).
        mtx = list(gse_dir.rglob("matrix.mtx*"))
        if mtx:
            import scanpy as sc
            mtx_dir = mtx[0].parent
            logger.info("Reading 10x mtx %s -> %s", mtx_dir, gse_id)
            adata = sc.read_10x_mtx(str(mtx_dir))
            adata.var_names_make_unique()
        else:
            logger.warning("No .h5ad or matrix.mtx in %s — skipping (optional layer)", gse_dir)
            return None

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
    # Also scan the v20 open-access download location for ready .h5ad atlases
    # (e.g. the Zenodo panImmune atlas lives under data/open_access/geo_scrna).
    scan_dirs = [raw_dir]
    extra = Path("data/open_access/geo_scrna")
    if extra.is_dir():
        scan_dirs.append(extra)

    rows = []
    for d in scan_dirs:
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                try:
                    row = ingest_gse(sub, Path(args.out_dir))
                    if row:
                        rows.append(row)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping %s: %s", sub.name, exc)
        # top-level .h5ad files (atlas not in a per-GSE subdir)
        for h5 in sorted(d.glob("*.h5ad")):
            try:
                fake = h5.parent / h5.stem
                # ingest_gse expects a dir; read directly instead.
                ad = _require_anndata()
                adata = ad.read_h5ad(str(h5))
                out_path = Path(args.out_dir) / "single_cell" / "scrna" / f"{h5.stem}.h5ad"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                adata.write_h5ad(out_path)
                rows.append({"dataset_name": h5.stem, "processed_file_path": str(out_path),
                             "n_cells": int(adata.n_obs), "n_features": int(adata.n_vars)})
                logger.info("Ingested atlas %s: %d cells", h5.stem, adata.n_obs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping atlas %s: %s", h5.name, exc)

    logger.info("Ingested %d GEO single-cell datasets.", len(rows))
    # Single-cell is an OPTIONAL augmentation layer: succeed even if nothing was
    # ingestible (the raws may be non-h5ad/non-mtx) rather than failing the DAG.
    if not rows:
        logger.warning("No ingestible single-cell sources found — optional layer skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
