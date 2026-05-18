#!/usr/bin/env python3
"""
scripts/mortfm_preprocess_scrna.py
==================================
Run the MORT-FM single-cell preprocessing pipeline on a raw-counts .h5ad and
write the QC'd / log-normalised / HVG-selected result to
``data/processed/single_cell/scrna/<gse>.h5ad``.

Usage
-----
    # Process the existing consolidated dumps:
    python scripts/mortfm_preprocess_scrna.py \\
        --input data/raw/gse271107.h5ad --gse GSE271107 \\
        --sample-col sample --disease-stage-col disease_stage

    python scripts/mortfm_preprocess_scrna.py \\
        --input data/raw/gse124310.h5ad --gse GSE124310 \\
        --sample-col sample --disease-stage-col condition

After this, ``scripts/mortfm_ingest_geo_singlecell.py`` can pick up the
canonical processed file. Symlinks the result back into
``data/raw_public/geo_single_cell/<gse>/`` for downstream ingest.

Honest behaviour
----------------
* Refuses to run if the input matrix is not raw counts (no silent
  reconstruction).
* Writes the QC report to ``data/processed/single_cell/scrna/<gse>_qc_report.json``.
* Records the original raw counts in ``adata.layers["counts"]`` so any
  downstream NB-likelihood loss still has access to them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_preprocess_scrna")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw-counts .h5ad path")
    ap.add_argument("--gse", required=True, help="GEO accession (e.g. GSE271107)")
    ap.add_argument("--out-dir", default="data/processed/single_cell/scrna")
    ap.add_argument("--public-dir", default="data/raw_public/geo_single_cell",
                    help="Where to symlink the processed file for the ingest script")
    ap.add_argument("--sample-col", default="sample")
    ap.add_argument("--disease-stage-col", default="disease_stage")
    ap.add_argument("--fallback-disease-col", default="condition")
    ap.add_argument("--min-genes", type=int, default=200)
    ap.add_argument("--max-genes", type=int, default=8000)
    ap.add_argument("--max-pct-mt", type=float, default=20.0)
    ap.add_argument("--n-top-genes", type=int, default=2000)
    args = ap.parse_args()

    import anndata as ad
    from resistancemap.data.single_cell_preprocessing import preprocess_scrna

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input .h5ad not found: {input_path.resolve()}")

    logger.info("Reading %s ...", input_path)
    adata = ad.read_h5ad(input_path)
    logger.info("Initial: %d cells x %d genes; obs columns: %s",
                adata.n_obs, adata.n_vars, list(adata.obs.columns))

    adata = preprocess_scrna(
        adata,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        max_pct_mt=args.max_pct_mt,
        n_top_genes=args.n_top_genes,
        sample_obs_col=args.sample_col,
        disease_stage_obs_col=args.disease_stage_col,
        fallback_disease_col=args.fallback_disease_col,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.gse}.h5ad"
    logger.info("Writing %s ...", out_path)
    adata.write_h5ad(out_path)

    qc_path = out_dir / f"{args.gse}_qc_report.json"
    with open(qc_path, "w") as f:
        json.dump(adata.uns["mortfm_qc"], f, indent=2)
    logger.info("QC report -> %s", qc_path)

    # Symlink into raw_public/geo_single_cell/<gse>/ so the ingest script picks it up.
    public = Path(args.public_dir) / args.gse
    public.mkdir(parents=True, exist_ok=True)
    link = public / f"{args.gse}.h5ad"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(out_path.resolve())
    logger.info("Symlinked %s -> %s", link, out_path)

    logger.info(
        "DONE: %s preprocessed: %d cells, %d genes, %d HVGs, pseudotime_disease_stage values=%s",
        args.gse, adata.n_obs, adata.n_vars,
        int(adata.var["highly_variable"].sum()),
        sorted(adata.obs["pseudotime_disease_stage"].unique().tolist()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
