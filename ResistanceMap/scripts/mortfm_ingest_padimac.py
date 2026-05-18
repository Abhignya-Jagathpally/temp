#!/usr/bin/env python3
"""
scripts/mortfm_ingest_padimac.py
================================
Ingest GSE116324 (PADIMAC: bortezomib-Adriamycin-Dexamethasone in newly
diagnosed MM; Chapman et al. 2018, PMID 30181174).

PADIMAC is a **single-baseline-timepoint** RNA-seq cohort (44 patients).
It is NOT paired baseline -> relapse. The Phase-D longitudinal inventory
originally miscategorised it; this script lands it under the correct
"static drug-response with bortezomib treatment context" bucket.

Outputs:
  * data/processed/padimac/padimac_expression.parquet
       (44 patients x n_genes counts)
  * data/processed/padimac/padimac_aligned_to_block_a.parquet
       (44 patients x 2000 Block-A-aligned genes; zero-filled mask)
  * data/processed/padimac/padimac_feature_mask.parquet
  * logs/mortfm/padimac_ingest_report.json
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.feature_alignment import align_to_reference_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_ingest_padimac")


def _download_supplementary(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116324/"
        "suppl/GSE116324_padimacRnaSeq.csv.gz"
    )
    dest = out_dir / "GSE116324_padimacRnaSeq.csv.gz"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        logger.info("Supplementary already cached at %s (%d bytes)",
                    dest, dest.stat().st_size)
        return dest
    logger.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest


def _load_counts(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, index_col=0)
    df.index.name = "ensembl_id"
    # strip the version suffix on ENSG ids (".4", ".15", etc.)
    df.index = df.index.astype(str).str.split(".").str[0]
    return df


def _ensg_to_symbol(ensembl_ids) -> dict[str, str]:
    """Use the MMRF Ensembl<->symbol map already on disk."""
    p = Path("data/processed/mmrf_ensembl_to_symbol.tsv")
    if not p.exists():
        return {}
    df = pd.read_csv(p, sep="\t")
    # Normalise column names.
    col_map = {c.lower(): c for c in df.columns}
    ens_col = (
        col_map.get("ensembl_id") or col_map.get("ensembl_gene_id")
        or col_map.get("ensembl") or col_map.get("ensembl_base")
    )
    sym_col = (
        col_map.get("gene_symbol") or col_map.get("symbol")
        or col_map.get("hgnc_symbol") or col_map.get("gene_name")
    )
    if not (ens_col and sym_col):
        logger.warning("Couldn't find ensembl/symbol columns in %s: %s", p, list(df.columns))
        return {}
    df = df.dropna(subset=[ens_col, sym_col])
    df[ens_col] = df[ens_col].astype(str).str.split(".").str[0]
    return dict(zip(df[ens_col], df[sym_col]))


def main() -> int:
    t0 = time.time()
    raw_dir = Path("data/raw_public/geo/GSE116324")
    raw_csv = _download_supplementary(raw_dir)
    counts = _load_counts(raw_csv)
    logger.info("Loaded PADIMAC counts: %d genes x %d patients",
                counts.shape[0], counts.shape[1])

    # Map ENSG -> HGNC symbol.
    ens2sym = _ensg_to_symbol(counts.index.tolist())
    counts["gene_symbol"] = counts.index.map(ens2sym)
    mapped_frac = counts["gene_symbol"].notna().mean()
    logger.info("Ensembl -> HGNC mapping rate: %.1f%%", 100 * mapped_frac)

    # Aggregate to symbol-level (sum counts where multiple ENSG -> same symbol).
    sym_counts = (
        counts.dropna(subset=["gene_symbol"])
              .groupby("gene_symbol", as_index=True).sum(numeric_only=True)
              .astype("int64")
    )
    # Patient-major layout: rows = patients, cols = genes
    expr = sym_counts.T  # (44, n_symbols)
    expr.index.name = "patient_id"
    out_dir = Path("data/processed/padimac")
    out_dir.mkdir(parents=True, exist_ok=True)
    expr_path = out_dir / "padimac_expression.parquet"
    expr.to_parquet(expr_path)
    logger.info("Wrote PADIMAC expression: %d patients x %d genes -> %s",
                expr.shape[0], expr.shape[1], expr_path)

    # Optionally: align to the Block-A 2000-gene reference if it's recoverable
    # from the Block A checkpoint. We replay the same selection as
    # mortfm_align_beataml_features._features_from_block_a — without
    # duplicating that logic, just import.
    aligned_path = None
    mask_path = None
    aligned_features = None
    block_a_ckpt = Path("checkpoints/mortfm/block_a_cellline_foundation.pt")
    if block_a_ckpt.exists():
        try:
            from scripts.mortfm_align_beataml_features import _features_from_block_a
        except ImportError:
            # Add scripts/ to path
            sys.path.insert(0, str(Path("scripts").resolve()))
            try:
                from mortfm_align_beataml_features import _features_from_block_a
            except ImportError:
                _features_from_block_a = None
        if _features_from_block_a is not None:
            ref_features = _features_from_block_a(block_a_ckpt)
            aligned, mask, rep = align_to_reference_features(expr, ref_features)
            aligned_path = out_dir / "padimac_aligned_to_block_a.parquet"
            mask_path = out_dir / "padimac_feature_mask.parquet"
            aligned.to_parquet(aligned_path)
            mask.to_parquet(mask_path)
            aligned_features = rep
            logger.info("Aligned to Block A: %d/%d features present (%.1f%%)",
                        rep.n_features_present, rep.n_reference_features,
                        100 * rep.coverage_rate)

    report = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-padimac-ingest",
        "geo_accession": "GSE116324",
        "study": "PADIMAC (Chapman et al. 2018, PMID 30181174)",
        "treatment_context": "bortezomib + adriamycin + dexamethasone (PAD)",
        "is_paired_longitudinal": False,
        "is_baseline_only": True,
        "longitudinal_evidence_level": 0,
        "static_drug_response_relevant": True,
        "n_patients": int(expr.shape[0]),
        "n_genes_raw": int(counts.shape[0]),
        "n_genes_after_symbol_agg": int(expr.shape[1]),
        "ensg_to_symbol_mapping_rate": float(mapped_frac),
        "aligned_to_block_a": aligned_path is not None,
        "block_a_coverage": (aligned_features.coverage_rate if aligned_features else None),
        "block_a_features_present": (aligned_features.n_features_present if aligned_features else None),
        "outputs": {
            "expression_parquet": str(expr_path),
            "aligned_expression_parquet": str(aligned_path) if aligned_path else None,
            "feature_mask_parquet": str(mask_path) if mask_path else None,
        },
        "honest_note": (
            "PADIMAC is single-baseline RNA-seq, NOT paired longitudinal. "
            "It expands the static_drug_response training pool for MM with "
            "bortezomib-containing regimen context, but does NOT unlock the "
            "longitudinal_trajectory or resistance_emergence claim gates. "
            "Treatment-response labels (responder/non-responder) live in the "
            "series_matrix sample table; this script writes the expression "
            "matrix only — response labels are a separate join."
        ),
        "wall_time_s": round(time.time() - t0, 1),
    }
    out_report = Path("logs/mortfm/padimac_ingest_report.json")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("PADIMAC ingest report -> %s", out_report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
