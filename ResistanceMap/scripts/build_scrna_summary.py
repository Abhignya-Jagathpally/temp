#!/usr/bin/env python3
"""Build per-(sample, disease_stage) pseudobulk summaries from on-disk scRNA h5ad files.

Inputs (real data, on disk):
  - data/raw/gse124310.h5ad   GSE124310: 27,796 BM plasma cells, NBM/MGUS/MM
  - data/raw/gse271107.h5ad   GSE271107: 143,748 BM cells, healthy/MGUS/SMM/MM

Output:
  - checkpoints/scrna_summary.pt  (torch dict)

Schema of the output dict:
  {
    "GSE124310": {
        "pseudobulk":   np.ndarray (n_groups, n_genes),  log1p(CPM)
        "groups":       list[str]   "<sample>|<stage>"
        "samples":      list[str]
        "stages":       list[str]   in {NBM, MGUS, MM}
        "gene_names":   list[str]
        "n_cells":      np.ndarray (n_groups,)            cells per group
    },
    "GSE271107": {... same schema, stages in {HD, MGUS, SMM, MM}},
    "harmonized": {                  intersection of gene sets across the two
        "pseudobulk":   np.ndarray (n_groups_total, n_genes_shared)
        "groups":       list[str]
        "samples":      list[str]
        "stages":       list[str]   normalized to {HD, MGUS, SMM, MM}
        "source":       list[str]
        "gene_names":   list[str]
        "n_cells":      np.ndarray
    },
    "metadata": {
        "min_cells_per_group": int,
        "stage_normalization": {"NBM":"HD", "healthy":"HD", ...},
        "n_input_cells": dict[str, int],
    },
  }

This is real on-disk data only. No synthesis. No imputation beyond log1p of zeros.
"""
from __future__ import annotations

import logging
from pathlib import Path

import anndata
import numpy as np
import scipy.sparse as sp
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger("build_scrna_summary")

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
OUT = REPO / "checkpoints" / "scrna_summary.pt"

MIN_CELLS_PER_GROUP = 30  # below this, the group's mean is too noisy
STAGE_MAP = {
    # GSE124310 'condition' codes
    "NBM": "HD",
    # GSE271107 'disease_stage' codes
    "healthy": "HD",
    # passthroughs
    "MGUS": "MGUS",
    "SMM": "SMM",
    "MM": "MM",
    "HD": "HD",
}


def _normalize_log_cpm(X: sp.csr_matrix) -> sp.csr_matrix:
    """log1p(CPM) per row. Stays sparse."""
    counts = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    counts[counts == 0] = 1.0
    scale = 1e4 / counts
    # multiply each row by its scale -> diag * X
    D = sp.diags(scale)
    Xn = D @ X
    Xn.data = np.log1p(Xn.data)
    return Xn.tocsr()


def _pseudobulk(adata: anndata.AnnData, group_keys: list[str]) -> dict:
    """Compute mean log1p(CPM) per group. group_keys is a list of obs columns to
    concatenate with '|' to form the group id."""
    # Normalize once on the full sparse matrix
    log.info(f"  normalizing {adata.n_obs} cells × {adata.n_vars} genes (log1p CPM)")
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    Xn = _normalize_log_cpm(X.astype(np.float32))

    # Build group labels
    obs = adata.obs
    labels = obs[group_keys[0]].astype(str)
    for k in group_keys[1:]:
        labels = labels + "|" + obs[k].astype(str)
    labels = labels.values

    uniq, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)

    # Mean per group via group-membership matrix M (n_cells x n_groups)
    n_cells = Xn.shape[0]
    M = sp.csr_matrix(
        (np.ones(n_cells, dtype=np.float32), (np.arange(n_cells), inverse)),
        shape=(n_cells, len(uniq)),
    )
    # Sum: (n_groups x n_genes) = M.T @ Xn
    sums = (M.T @ Xn).toarray()
    means = sums / counts[:, None].astype(np.float32)

    # Filter low-cell groups
    keep = counts >= MIN_CELLS_PER_GROUP
    log.info(
        f"  {len(uniq)} groups; keeping {keep.sum()} (≥ {MIN_CELLS_PER_GROUP} cells/group)"
    )
    return {
        "pseudobulk": means[keep].astype(np.float32),
        "groups": uniq[keep].tolist(),
        "n_cells": counts[keep].astype(np.int32),
    }


def build_one(path: Path, group_cols: list[str], stage_col: str, sample_col: str) -> dict:
    log.info(f"loading {path}")
    a = anndata.read_h5ad(path)
    # Some legacy h5ads have non-unique obs names
    a.obs_names_make_unique()
    pb = _pseudobulk(a, group_cols)
    samples, stages = zip(*[g.split("|", 1) for g in pb["groups"]])
    pb["samples"] = list(samples)
    pb["stages_raw"] = list(stages)
    pb["stages"] = [STAGE_MAP.get(s, s) for s in stages]
    pb["gene_names"] = a.var.index.astype(str).tolist()
    pb["n_input_cells"] = int(a.n_obs)
    return pb


def harmonize(g124: dict, g271: dict) -> dict:
    """Intersect gene sets and stack groups across the two datasets."""
    g1 = list(g124["gene_names"])
    g2 = list(g271["gene_names"])
    shared = sorted(set(g1) & set(g2))
    log.info(f"shared gene set across GSE124310 ∩ GSE271107: {len(shared)} genes")
    idx1 = np.asarray([g1.index(g) for g in shared])
    idx2 = np.asarray([g2.index(g) for g in shared])

    pb1 = g124["pseudobulk"][:, idx1]
    pb2 = g271["pseudobulk"][:, idx2]
    pb = np.concatenate([pb1, pb2], axis=0)

    samples = list(g124["samples"]) + list(g271["samples"])
    stages = list(g124["stages"]) + list(g271["stages"])
    groups = list(g124["groups"]) + list(g271["groups"])
    n_cells = np.concatenate([g124["n_cells"], g271["n_cells"]])
    source = ["GSE124310"] * len(g124["samples"]) + ["GSE271107"] * len(g271["samples"])
    return {
        "pseudobulk": pb.astype(np.float32),
        "groups": groups,
        "samples": samples,
        "stages": stages,
        "source": source,
        "gene_names": shared,
        "n_cells": n_cells.astype(np.int32),
    }


def main() -> None:
    p124 = RAW / "gse124310.h5ad"
    p271 = RAW / "gse271107.h5ad"
    if not p124.exists() or not p271.exists():
        raise SystemExit(f"missing inputs: {p124.exists()=} {p271.exists()=}")

    g124 = build_one(p124, group_cols=["sample", "condition"], stage_col="condition", sample_col="sample")
    g271 = build_one(p271, group_cols=["sample", "disease_stage"], stage_col="disease_stage", sample_col="sample")

    harm = harmonize(g124, g271)

    out = {
        "GSE124310": {
            "pseudobulk": g124["pseudobulk"],
            "groups": g124["groups"],
            "samples": g124["samples"],
            "stages": g124["stages"],
            "stages_raw": g124["stages_raw"],
            "gene_names": g124["gene_names"],
            "n_cells": g124["n_cells"],
        },
        "GSE271107": {
            "pseudobulk": g271["pseudobulk"],
            "groups": g271["groups"],
            "samples": g271["samples"],
            "stages": g271["stages"],
            "stages_raw": g271["stages_raw"],
            "gene_names": g271["gene_names"],
            "n_cells": g271["n_cells"],
        },
        "harmonized": harm,
        "metadata": {
            "min_cells_per_group": MIN_CELLS_PER_GROUP,
            "stage_normalization": STAGE_MAP,
            "n_input_cells": {
                "GSE124310": g124["n_input_cells"],
                "GSE271107": g271["n_input_cells"],
            },
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT)
    log.info(f"wrote {OUT}")
    log.info(
        f"  GSE124310: {g124['pseudobulk'].shape[0]} groups × {g124['pseudobulk'].shape[1]} genes"
    )
    log.info(
        f"  GSE271107: {g271['pseudobulk'].shape[0]} groups × {g271['pseudobulk'].shape[1]} genes"
    )
    log.info(
        f"  harmonized: {harm['pseudobulk'].shape[0]} groups × {harm['pseudobulk'].shape[1]} genes"
    )
    # Stage breakdown
    from collections import Counter
    log.info(f"  GSE124310 stages: {dict(Counter(g124['stages']))}")
    log.info(f"  GSE271107 stages: {dict(Counter(g271['stages']))}")
    log.info(f"  harmonized stages: {dict(Counter(harm['stages']))}")


if __name__ == "__main__":
    main()
