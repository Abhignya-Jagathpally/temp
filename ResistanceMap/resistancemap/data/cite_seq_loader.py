"""
resistancemap/data/cite_seq_loader.py
=====================================
CITE-seq loader (paired scRNA + surface proteins).

Emits two :class:`~resistancemap.mortfm.schemas.ModalityTensor` objects keyed
to the same cells: an RNA tensor and a proteomics tensor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from resistancemap.mortfm.schemas import ModalityName, ModalityTensor

logger = logging.getLogger(__name__)


def load_citeseq_h5mu(
    path: str,
    *,
    rna_modality_key: str = "rna",
    protein_modality_key: str = "prot",
) -> Tuple[ModalityTensor, ModalityTensor]:
    """Load a .h5mu MuData file with paired RNA + ADT modalities.

    Returns
    -------
    (rna, protein) : Tuple[ModalityTensor, ModalityTensor]
        Both indexed by the same cells (same row order).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CITE-seq .h5mu not found: {p.resolve()}")
    try:
        import muon as mu
    except ImportError as exc:
        raise ImportError("CITE-seq loading requires muon: pip install muon") from exc
    mdata = mu.read_h5mu(str(p))
    if rna_modality_key not in mdata.mod or protein_modality_key not in mdata.mod:
        raise KeyError(
            f"MuData lacks expected modalities {rna_modality_key!r}/{protein_modality_key!r}; "
            f"present: {list(mdata.mod)}"
        )
    rna_adata = mdata.mod[rna_modality_key]
    prot_adata = mdata.mod[protein_modality_key]

    def _as_dense(X) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    rna_X = _as_dense(rna_adata.X)
    prot_X = _as_dense(prot_adata.X)

    rna_mt = ModalityTensor(
        name=ModalityName.RNA.value,
        values=torch.from_numpy(rna_X),
        feature_names=list(rna_adata.var_names.astype(str).values),
    )
    prot_mt = ModalityTensor(
        name=ModalityName.PROTEOMICS.value,
        values=torch.from_numpy(prot_X),
        feature_names=list(prot_adata.var_names.astype(str).values),
    )
    logger.info("CITE-seq %s: %d cells, %d genes, %d ADTs",
                p.name, rna_X.shape[0], rna_X.shape[1], prot_X.shape[1])
    return rna_mt, prot_mt
