"""
resistancemap/data/proteomics_loader.py
=======================================
Bulk / RPPA / DIA-MS proteomics loader for MORT-FM.

Reads a wide matrix (samples x proteins) and emits a
:class:`~resistancemap.mortfm.schemas.ModalityTensor`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from resistancemap.mortfm.schemas import ModalityName, ModalityTensor

logger = logging.getLogger(__name__)


def load_proteomics_matrix(
    path: str,
    *,
    sample_id_axis: str = "rows",
    log_transform: bool = True,
    min_coverage: float = 0.5,
) -> ModalityTensor:
    """Load a proteomics matrix.

    Parameters
    ----------
    path
        CSV/TSV. Sample IDs on one axis, UniProt/HGNC on the other.
    sample_id_axis
        "rows" (default) or "columns".
    log_transform
        If True, apply ``log2(x + 1)`` after replacing NaN with 0.
    min_coverage
        Drop proteins observed in fewer than this fraction of samples.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Proteomics file not found: {p.resolve()}")
    sep = "\t" if p.suffix in {".tsv", ".txt"} else ","
    df = pd.read_csv(p, sep=sep, index_col=0)

    if sample_id_axis == "columns":
        df = df.T

    # Coverage filter (matches the v14 default of 0.7).
    coverage = df.notna().mean(axis=0)
    keep = coverage[coverage >= min_coverage].index
    df = df[keep]

    X = df.values.astype(np.float32)
    nan_mask = np.isnan(X)
    X = np.where(nan_mask, 0.0, X)
    if log_transform:
        X = np.log2(X + 1.0)

    logger.info("Loaded proteomics %s: %d samples x %d proteins (post coverage>=%.2f)",
                p.name, X.shape[0], X.shape[1], min_coverage)
    return ModalityTensor(
        name=ModalityName.PROTEOMICS.value,
        values=torch.from_numpy(X),
        feature_names=df.columns.astype(str).tolist(),
        mask=torch.from_numpy(~nan_mask),
    )
