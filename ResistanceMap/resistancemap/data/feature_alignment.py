"""
resistancemap/data/feature_alignment.py
=======================================
Project an arbitrary expression matrix onto a *reference* feature space
(typically the top-K genes used by a pre-trained MORT-FM checkpoint).

Used by Block B to align BeatAML's 22,843 genes onto Block A's 2,000-gene
feature space so the Block-A checkpoint can be loaded with a shape-compatible
RNA encoder.

Honest behaviour
----------------
* Missing genes in the source matrix are **zero-filled** AND flagged in a
  per-feature mask. The mask is returned alongside so downstream code can
  reweight or skip absent features instead of treating zeros as real values.
* This module never imputes expression. It also never changes the reference
  feature order — the output's column order *exactly* matches the reference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AlignmentReport:
    n_reference_features: int
    n_source_features: int
    n_features_present: int
    n_features_missing: int
    coverage_rate: float


def align_to_reference_features(
    source: pd.DataFrame,
    reference_features: Sequence[str],
    *,
    case_normalise: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, AlignmentReport]:
    """Reshape ``source`` to have exactly ``reference_features`` columns.

    Parameters
    ----------
    source :
        Expression DataFrame (samples × genes).
    reference_features :
        Ordered list of feature names from the reference checkpoint.
    case_normalise :
        If True (default), retry the source-column lookup under upper-case if
        the exact name misses. Helps when one source uses Title-case symbols
        and the other UPPER-case.

    Returns
    -------
    aligned : DataFrame
        ``source`` re-indexed to the reference columns. Missing columns are
        zero-filled.
    mask : DataFrame
        Same shape as ``aligned``; True where the column was present in the
        source, False where it was zero-filled.
    report : AlignmentReport
    """
    reference_features = list(reference_features)
    src_cols = {c: c for c in source.columns.astype(str)}
    if case_normalise:
        for c in source.columns.astype(str):
            src_cols.setdefault(c.upper(), c)

    aligned = pd.DataFrame(
        np.zeros((source.shape[0], len(reference_features)), dtype=np.float32),
        index=source.index, columns=list(reference_features),
    )
    mask = pd.DataFrame(
        np.zeros((source.shape[0], len(reference_features)), dtype=bool),
        index=source.index, columns=list(reference_features),
    )
    n_present = 0
    for ref in reference_features:
        hit = src_cols.get(ref) or (src_cols.get(ref.upper()) if case_normalise else None)
        if hit is None:
            continue
        aligned[ref] = source[hit].values.astype(np.float32)
        mask[ref] = True
        n_present += 1
    rep = AlignmentReport(
        n_reference_features=len(reference_features),
        n_source_features=int(source.shape[1]),
        n_features_present=n_present,
        n_features_missing=len(reference_features) - n_present,
        coverage_rate=n_present / max(len(reference_features), 1),
    )
    logger.info(
        "Feature alignment: %d/%d reference features present in source (%.1f%%)",
        n_present, len(reference_features), 100 * rep.coverage_rate,
    )
    return aligned, mask, rep
