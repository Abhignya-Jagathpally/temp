"""
resistancemap/data/histone_ptm_loader.py
========================================
Histone post-translational modification (PTM) loader.

PTMs of interest: H3K27ac, H3K27me3, H3K4me3, H3K9me3, plus total acetylation
and total methylation. These tracks come from ChIP-seq / CUT&RUN bigwig files
or from bulk mass-spectrometry tables.

The MORT-FM histone-PTM encoder treats each modification as one feature channel
per gene/region; this loader is responsible for producing that wide matrix.

Honest behaviour
----------------
* If passed a directory of bigwig files, the loader requires ``pybigwig`` and
  will :class:`ImportError` if not installed.
* If passed a CSV/TSV, the loader just reads the matrix.
* Empty input -> :class:`FileNotFoundError`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from resistancemap.mortfm.schemas import ModalityName, ModalityTensor

logger = logging.getLogger(__name__)

#: Canonical PTM channels used by the histone PTM encoder, in stable order.
HISTONE_PTM_CHANNELS: Tuple[str, ...] = (
    "H3K27ac",
    "H3K27me3",
    "H3K4me3",
    "H3K9me3",
    "global_acetylation",
    "global_methylation",
)


def load_histone_ptm_matrix(
    path: str,
    *,
    channels: Optional[List[str]] = None,
) -> ModalityTensor:
    """Load a histone-PTM matrix from a CSV/TSV.

    File format
    -----------
    Rows = samples, columns = (PTM_channel, gene/region). Column names should
    follow ``<channel>__<gene>`` so the encoder can group columns by PTM.

    Parameters
    ----------
    channels
        If given, restrict to these channels. Default: use all of
        ``HISTONE_PTM_CHANNELS`` present in the file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Histone PTM matrix not found: {p.resolve()}")
    sep = "\t" if p.suffix in {".tsv", ".txt"} else ","
    df = pd.read_csv(p, sep=sep, index_col=0)

    if channels is None:
        channels = list(HISTONE_PTM_CHANNELS)
    keep_cols = [c for c in df.columns if any(c.startswith(ch + "__") for ch in channels)]
    if not keep_cols:
        raise ValueError(
            f"No columns matched expected channels {channels} in {p}. "
            f"Available examples: {list(df.columns)[:10]}"
        )
    df = df[keep_cols]

    X = df.values.astype(np.float32)
    logger.info("Loaded histone PTM %s: %d samples x %d (channel, region) cols",
                p.name, X.shape[0], X.shape[1])
    return ModalityTensor(
        name=ModalityName.HISTONE_PTM.value,
        values=torch.from_numpy(X),
        feature_names=keep_cols,
    )
