"""
resistancemap/data/scrna_loader.py
==================================
Single-cell RNA-seq loader for MORT-FM.

Produces :class:`~resistancemap.mortfm.schemas.ModalityTensor` objects from
``.h5ad`` (AnnData) files, optionally pseudobulked per patient.

This is a *thin wrapper* — most of the heavy QC lifts (doublet removal,
malignant-cell annotation, batch correction) live in
:mod:`resistancemap.data.single_cell_preprocessing` because they are needed
by both v14 and MORT-FM paths. The loader is just responsible for the
``AnnData -> ModalityTensor`` adapter step.

Honest behaviour
----------------
* Raises ``FileNotFoundError`` if the .h5ad does not exist.
* Raises ``ImportError`` (with installation instructions) if ``anndata`` is
  not installed.
* Returns the *raw* matrix; QC/normalisation must be applied separately.
* Subsampling is deterministic (head-of-list), never random — keeps results
  reproducible and satisfies the fabrication-sentinel hook.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from resistancemap.mortfm.schemas import ModalityName, ModalityTensor, PatientCellSnapshot

logger = logging.getLogger(__name__)


def _require_anndata():
    try:
        import anndata as ad
        return ad
    except ImportError as exc:
        raise ImportError(
            "scrna_loader requires anndata. Install with: pip install anndata scanpy"
        ) from exc


def load_h5ad_as_modality(
    path: str,
    *,
    feature_layer: Optional[str] = None,
    feature_names_col: str = "gene_symbols",
    max_cells: Optional[int] = None,
) -> ModalityTensor:
    """Load a single .h5ad as one :class:`ModalityTensor` (cells x genes).

    Parameters
    ----------
    path
        Path to the .h5ad file.
    feature_layer
        Optional layer name (e.g. "counts", "log_norm"). If None, uses ``adata.X``.
    feature_names_col
        Column in ``adata.var`` to use as feature_names. Falls back to ``adata.var_names``.
    max_cells
        If set, deterministically take the first ``max_cells`` rows (no random
        subsampling — keep reproducibility tight). ``None`` -> use all.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"scRNA .h5ad not found: {p.resolve()}")
    ad = _require_anndata()
    adata = ad.read_h5ad(str(p))

    if max_cells is not None and adata.n_obs > max_cells:
        adata = adata[:max_cells].copy()

    X = adata.layers[feature_layer] if feature_layer else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    if feature_names_col in adata.var.columns:
        feature_names = list(adata.var[feature_names_col].astype(str).values)
    else:
        feature_names = list(adata.var_names.astype(str).values)

    logger.info("Loaded %s: %d cells x %d genes", p.name, X.shape[0], X.shape[1])
    return ModalityTensor(
        name=ModalityName.RNA.value,
        values=torch.from_numpy(X),
        feature_names=feature_names,
    )


def pseudobulk_per_patient(
    path: str,
    *,
    patient_obs_col: str = "patient_id",
    aggregation: str = "mean",
) -> Tuple[List[str], ModalityTensor]:
    """Aggregate cells into one row per patient.

    Returns
    -------
    patient_ids :
        Length-``N_patients`` list of patient IDs (sorted).
    modality :
        :class:`ModalityTensor` of shape ``(N_patients, n_genes)``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"scRNA .h5ad not found: {p.resolve()}")
    ad = _require_anndata()
    adata = ad.read_h5ad(str(p))

    if patient_obs_col not in adata.obs.columns:
        raise KeyError(
            f"AnnData.obs has no column {patient_obs_col!r}; "
            f"available: {list(adata.obs.columns)[:20]}"
        )

    patient_ids = sorted(adata.obs[patient_obs_col].astype(str).unique().tolist())
    n_patients = len(patient_ids)
    n_genes = adata.n_vars
    pbulk = np.zeros((n_patients, n_genes), dtype=np.float32)
    for i, pid in enumerate(patient_ids):
        mask = adata.obs[patient_obs_col].astype(str).values == pid
        X = adata[mask].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        if aggregation == "mean":
            pbulk[i] = X.mean(axis=0)
        elif aggregation == "sum":
            pbulk[i] = X.sum(axis=0)
        else:
            raise ValueError(f"Unknown aggregation {aggregation!r}; expected mean|sum.")

    feature_names = list(adata.var_names.astype(str).values)
    logger.info("Pseudobulked to %d patients x %d genes", n_patients, n_genes)
    return patient_ids, ModalityTensor(
        name=ModalityName.RNA.value,
        values=torch.from_numpy(pbulk),
        feature_names=feature_names,
    )


def build_snapshots_from_h5ad(
    path: str,
    *,
    patient_obs_col: str = "patient_id",
    timepoint_obs_col: Optional[str] = "timepoint_months",
    disease: str = "MM",
) -> List[PatientCellSnapshot]:
    """High-level convenience: .h5ad -> list[PatientCellSnapshot].

    One snapshot per (patient_id, timepoint) group, pseudobulked.
    Only RNA is populated; other modalities stay None.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"scRNA .h5ad not found: {p.resolve()}")
    ad = _require_anndata()
    adata = ad.read_h5ad(str(p))
    if patient_obs_col not in adata.obs.columns:
        # Fall back to common patient/donor identifiers (e.g. integrated atlases
        # use donor_id/sample_id rather than patient_id) before giving up.
        for alt in ("patient_id", "donor_id", "donor", "sample_id", "subject_id", "study_id"):
            if alt in adata.obs.columns:
                logger.warning("patient_obs_col %r absent; falling back to %r",
                               patient_obs_col, alt)
                patient_obs_col = alt
                break
        else:
            raise KeyError(
                f"{patient_obs_col!r} not in adata.obs.columns and no known "
                f"patient/donor fallback present (have {list(adata.obs.columns)[:12]})"
            )

    snapshots: List[PatientCellSnapshot] = []
    grouping_cols = [patient_obs_col]
    if timepoint_obs_col and timepoint_obs_col in adata.obs.columns:
        grouping_cols.append(timepoint_obs_col)

    for keys, sub in adata.obs.groupby(grouping_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        pid = str(keys[0])
        t = float(keys[1]) if len(keys) > 1 and keys[1] is not None and not _is_nan(keys[1]) else None
        mask = sub.index
        X = adata[mask].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32).mean(axis=0, keepdims=True)
        mt = ModalityTensor(
            name=ModalityName.RNA.value,
            values=torch.from_numpy(X),
            feature_names=list(adata.var_names.astype(str).values),
        )
        snapshots.append(
            PatientCellSnapshot(
                patient_id=pid,
                sample_id=f"{pid}_t{t}" if t is not None else pid,
                disease=disease,
                timepoint=t,
                rna=mt,
            )
        )
    logger.info("Built %d snapshots from %s", len(snapshots), p.name)
    return snapshots


def _is_nan(v) -> bool:
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False
