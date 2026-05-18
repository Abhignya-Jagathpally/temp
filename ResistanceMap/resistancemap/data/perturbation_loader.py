"""
resistancemap/data/perturbation_loader.py
=========================================
CRISPR / Perturb-seq / drug-perturbation loader for MORT-FM.

Loads two flavours of perturbation evidence:

1. **Gene-level knockout / knockdown phenotypes** (e.g. DepMap CRISPR
   gene-effect matrices). One scalar per (cell-line, gene) measuring fitness
   delta.
2. **Perturb-seq style expression deltas** (per-cell expression vector after
   perturbation, relative to control). Used as ground truth for the
   ``L_perturbation`` consistency loss.

Honest behaviour
----------------
* Missing files raise ``FileNotFoundError``.
* The loader never imputes missing perturbations — they stay NaN in the mask.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def load_crispr_gene_effect(
    path: str,
    *,
    sample_id_axis: str = "rows",
) -> Tuple[List[str], List[str], torch.Tensor]:
    """Load a DepMap-style CRISPR gene-effect matrix.

    Parameters
    ----------
    path
        CSV with rows=cell lines, columns=genes (or transpose).

    Returns
    -------
    sample_ids, gene_symbols, values :
        ``values`` is a float tensor of shape ``(n_samples, n_genes)``;
        NaN preserved.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CRISPR gene-effect file not found: {p.resolve()}")
    df = pd.read_csv(p, index_col=0)
    if sample_id_axis == "columns":
        df = df.T
    sample_ids = df.index.astype(str).tolist()
    gene_symbols = df.columns.astype(str).tolist()
    values = torch.from_numpy(df.values.astype(np.float32))
    logger.info("CRISPR effect %s: %d samples x %d genes", p.name, len(sample_ids), len(gene_symbols))
    return sample_ids, gene_symbols, values


def load_perturbseq_deltas(
    path: str,
    *,
    perturbation_obs_col: str = "perturbation",
    control_label: str = "control",
) -> Dict[str, torch.Tensor]:
    """Load Perturb-seq expression deltas per perturbation target.

    Returns
    -------
    Dict mapping ``perturbation_name -> mean expression delta tensor (n_genes,)``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Perturb-seq file not found: {p.resolve()}")
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Perturb-seq loading requires anndata: pip install anndata") from exc

    adata = ad.read_h5ad(str(p))
    if perturbation_obs_col not in adata.obs.columns:
        raise KeyError(f"Perturb-seq AnnData has no obs column {perturbation_obs_col!r}")

    obs = adata.obs[perturbation_obs_col].astype(str)
    control_mask = obs == control_label
    if control_mask.sum() == 0:
        raise ValueError(f"No control cells found with label {control_label!r}")

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    control_mean = X[control_mask].mean(axis=0)

    deltas: Dict[str, torch.Tensor] = {}
    for perturbation in obs.unique():
        if perturbation == control_label:
            continue
        mask = (obs == perturbation).values
        if mask.sum() == 0:
            continue
        delta = X[mask].mean(axis=0) - control_mean
        deltas[str(perturbation)] = torch.from_numpy(delta.astype(np.float32))
    logger.info("Loaded %d Perturb-seq deltas from %s", len(deltas), p.name)
    return deltas
