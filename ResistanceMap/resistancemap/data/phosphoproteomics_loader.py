"""
resistancemap/data/phosphoproteomics_loader.py
==============================================
Phosphoproteomics loader (phosphosite-level) for MORT-FM.

Phosphoproteomics captures signaling activity, not protein abundance — it is
the modality most directly tied to MAPK / PI3K / NF-kB / JAK-STAT rewiring,
which the MORT-FM design doc names as critical for resistance mechanism.
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


def load_phosphoproteomics(
    path: str,
    *,
    sample_id_axis: str = "rows",
    log_transform: bool = True,
) -> ModalityTensor:
    """Load a phosphosite matrix (samples x phosphosites).

    Phosphosite identifiers should be of the form ``UNIPROT_RESIDUE_POSITION``
    (e.g. ``P38398_S1423``). The encoder uses the UniProt prefix to map sites
    onto the PPI graph.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Phosphoproteomics file not found: {p.resolve()}")
    sep = "\t" if p.suffix in {".tsv", ".txt"} else ","
    df = pd.read_csv(p, sep=sep, index_col=0)
    if sample_id_axis == "columns":
        df = df.T

    X = df.values.astype(np.float32)
    nan_mask = np.isnan(X)
    X = np.where(nan_mask, 0.0, X)
    if log_transform:
        X = np.log2(np.abs(X) + 1.0) * np.sign(X)

    logger.info("Loaded phospho %s: %d samples x %d sites",
                p.name, X.shape[0], X.shape[1])
    return ModalityTensor(
        name=ModalityName.PHOSPHOPROTEOMICS.value,
        values=torch.from_numpy(X),
        feature_names=df.columns.astype(str).tolist(),
        mask=torch.from_numpy(~nan_mask),
    )
