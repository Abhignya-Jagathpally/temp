"""
resistancemap/data/methylation_loader.py
========================================
DNA methylation loader (450k / EPIC / WGBS bedGraph) for MORT-FM.

Emits :class:`~resistancemap.mortfm.schemas.ModalityTensor` of beta values
(or M-values when ``as_m_values=True``).
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


def load_methylation_matrix(
    path: str,
    *,
    sample_id_axis: str = "columns",
    as_m_values: bool = False,
    epsilon: float = 1e-3,
) -> ModalityTensor:
    """Load a methylation matrix (CpGs x samples) from CSV/TSV.

    Parameters
    ----------
    path
        Path to the file. Detects TSV vs CSV by suffix.
    sample_id_axis
        Whether samples are along ``"columns"`` (default, GDC convention) or
        ``"rows"``. Output ``ModalityTensor.values`` is always ``(samples, CpGs)``.
    as_m_values
        If True, convert beta -> M-value: ``M = log2(beta / (1 - beta))``.
    epsilon
        Numerical floor / ceiling on beta for the M-value transform.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Methylation file not found: {p.resolve()}")
    sep = "\t" if p.suffix in {".tsv", ".txt"} else ","
    df = pd.read_csv(p, sep=sep, index_col=0)

    if sample_id_axis == "columns":
        cpg_names = df.index.astype(str).tolist()
        sample_ids = df.columns.astype(str).tolist()
        X = df.values.T.astype(np.float32)
    elif sample_id_axis == "rows":
        cpg_names = df.columns.astype(str).tolist()
        sample_ids = df.index.astype(str).tolist()
        X = df.values.astype(np.float32)
    else:
        raise ValueError(f"sample_id_axis must be 'columns' or 'rows'; got {sample_id_axis!r}")

    if as_m_values:
        beta = np.clip(X, epsilon, 1.0 - epsilon)
        X = np.log2(beta / (1.0 - beta))

    logger.info(
        "Loaded methylation %s: %d samples x %d CpGs (as_m=%s)",
        p.name, X.shape[0], X.shape[1], as_m_values,
    )
    mt = ModalityTensor(
        name=ModalityName.METHYLATION.value,
        values=torch.from_numpy(X),
        feature_names=cpg_names,
    )
    # Attach sample IDs as a side-channel so the caller can reconstruct snapshots.
    mt._sample_ids = sample_ids  # type: ignore[attr-defined]
    return mt
