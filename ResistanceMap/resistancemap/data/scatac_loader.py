"""
resistancemap/data/scatac_loader.py
===================================
scATAC-seq loader for MORT-FM.

Produces a :class:`~resistancemap.mortfm.schemas.ModalityTensor` of
``(N_cells, N_peaks)`` accessibility values. Uses ``anndata`` for .h5ad input.

The loader does NOT do peak calling, fragment-file processing, or peak-to-gene
mapping — those belong in :mod:`resistancemap.data.epigenomics_preprocessing`.

Subsampling, when requested, is deterministic (head-of-list) — no RNG.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from resistancemap.mortfm.schemas import ModalityName, ModalityTensor

logger = logging.getLogger(__name__)


def load_scatac_h5ad(
    path: str,
    *,
    binarise: bool = True,
    max_cells: Optional[int] = None,
) -> ModalityTensor:
    """Load a scATAC .h5ad as a ``ModalityTensor``.

    Parameters
    ----------
    path
        Path to a .h5ad file with ``adata.X`` as peak-by-cell counts.
    binarise
        If True, threshold counts to {0, 1} (the convention used by SnapATAC
        and ArchR). Default True.
    max_cells
        Deterministic head-truncation. ``None`` -> use all cells.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"scATAC .h5ad not found: {p.resolve()}")
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("anndata required: pip install anndata") from exc

    adata = ad.read_h5ad(str(p))
    if max_cells is not None and adata.n_obs > max_cells:
        adata = adata[:max_cells].copy()

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if binarise:
        X = (X > 0).astype(np.float32)

    peak_names = list(adata.var_names.astype(str).values)
    logger.info("Loaded scATAC %s: %d cells x %d peaks (binarise=%s)",
                p.name, X.shape[0], X.shape[1], binarise)
    return ModalityTensor(
        name=ModalityName.ATAC.value,
        values=torch.from_numpy(X),
        feature_names=peak_names,
    )
