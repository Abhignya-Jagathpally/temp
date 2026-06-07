"""GEO bulk-expression + survival loader for external validation (v20 Phase 1.3).

Two MM bulk cohorts with real survival annotations used as held-out external
validation in :mod:`scripts/mortfm/11_validate_open_access`:

* **GSE24080** (N~559) — UAMS/MAQC-II; event-free survival (EFS) + overall
  survival (OS). Typically obtained via refine.bio (gene x sample TSV +
  metadata TSV).
* **GSE136337** (N~426) — carries baseline **B2M, albumin, LDH** plus OS,
  which is exactly what the PK observation decoder
  (:mod:`resistancemap.models.pk_observation_decoder`) is validated against.

Honest behaviour
----------------
* Raises ``FileNotFoundError`` if the expected files are absent.
* Raises ``ValueError`` if required columns (survival time / event) are missing
  or if no patient has a usable survival time.
* NEVER imputes a survival label or fabricates an expression value.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Survival label extraction (shared)
# ---------------------------------------------------------------------------
def extract_survival_labels(
    df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    id_col: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Build a censoring-aware (event, time) structured array for sksurv.

    Rows with a non-finite or non-positive time are dropped (never imputed).
    ``event_col`` is coerced to boolean (truthy / {1,'1','dead','event',
    'yes','true'} -> True). Returns ``{sample_id, event, time, structured}``.
    """
    for c in (time_col, event_col):
        if c not in df.columns:
            raise ValueError(f"survival column {c!r} absent; have {list(df.columns)[:20]}")

    time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype="float64")
    raw_event = df[event_col]
    if raw_event.dtype == bool:
        event = raw_event.to_numpy()
    else:
        truthy = {"1", "dead", "event", "yes", "true", "deceased", "progressed"}
        event = (
            raw_event.astype(str).str.strip().str.lower().isin(truthy)
            | (pd.to_numeric(raw_event, errors="coerce") == 1)
        ).to_numpy()

    keep = np.isfinite(time) & (time > 0)
    if not keep.any():
        raise ValueError(f"No usable survival times in {time_col!r} — refusing empty labels.")

    ids = (
        df.loc[keep, id_col].to_numpy()
        if id_col and id_col in df.columns
        else np.arange(int(keep.sum()))
    )
    ev = event[keep]
    t = time[keep]
    structured = np.array(
        [(bool(e), float(x)) for e, x in zip(ev, t)],
        dtype=[("event", bool), ("time", float)],
    )
    return {"sample_id": ids, "event": ev, "time": t, "structured": structured,
            "keep_mask": keep}


# ---------------------------------------------------------------------------
# GSE24080 (refine.bio layout: expression TSV + metadata TSV)
# ---------------------------------------------------------------------------
def load_gse24080(
    refine_bio_dir: str | Path,
    *,
    expression_name: str = "GSE24080.tsv",
    metadata_name: str = "metadata_GSE24080.tsv",
    efs_time_col: str = "efs_time",
    efs_event_col: str = "efs_event",
) -> Dict[str, object]:
    """Load GSE24080 expression + EFS/OS metadata from a refine.bio export.

    Returns ``{expression: DataFrame(genes x samples), metadata: DataFrame,
    survival: structured-array dict}``. Raises if files / columns are absent.
    """
    d = Path(refine_bio_dir)
    expr_path = d / expression_name
    meta_path = d / metadata_name
    if not expr_path.exists():
        raise FileNotFoundError(f"GSE24080 expression not found: {expr_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"GSE24080 metadata not found: {meta_path}")

    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    meta = pd.read_csv(meta_path, sep="\t")
    if expr.empty:
        raise ValueError(f"{expr_path}: empty expression matrix.")

    survival = extract_survival_labels(
        meta, time_col=efs_time_col, event_col=efs_event_col,
        id_col=meta.columns[0],
    )
    logger.info(
        "GSE24080: %d genes x %d samples, %d with usable EFS (%d events)",
        expr.shape[0], expr.shape[1], len(survival["time"]),
        int(survival["event"].sum()),
    )
    return {"expression": expr, "metadata": meta, "survival": survival}


# ---------------------------------------------------------------------------
# GSE136337 (series-matrix-derived: expression + baseline labs + OS)
# ---------------------------------------------------------------------------
LAB_LABEL_HINTS: Dict[str, Sequence[str]] = {
    "B2M": ("b2m", "beta2", "beta-2", "microglobulin"),
    "albumin": ("albumin", "alb"),
    "LDH": ("ldh", "lactate dehydrogenase"),
}


def _resolve_lab_column(meta: pd.DataFrame, hints: Sequence[str]) -> Optional[str]:
    for c in meta.columns:
        cl = c.lower()
        if any(h in cl for h in hints):
            return c
    return None


def load_gse136337(
    geo_dir: str | Path,
    *,
    expression_name: str = "GSE136337_expression.tsv",
    metadata_name: str = "GSE136337_metadata.tsv",
    os_time_col: str = "os_time",
    os_event_col: str = "os_event",
) -> Dict[str, object]:
    """Load GSE136337 expression + baseline B2M/albumin/LDH + OS.

    The lab columns are resolved by fuzzy header match (``LAB_LABEL_HINTS``);
    a lab that is genuinely absent is reported as ``None`` in the returned
    ``lab_columns`` map — never fabricated. Raises on missing files / OS columns.
    """
    d = Path(geo_dir)
    expr_path = d / expression_name
    meta_path = d / metadata_name
    if not expr_path.exists():
        raise FileNotFoundError(f"GSE136337 expression not found: {expr_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"GSE136337 metadata not found: {meta_path}")

    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    meta = pd.read_csv(meta_path, sep="\t")
    if expr.empty:
        raise ValueError(f"{expr_path}: empty expression matrix.")

    lab_columns = {lab: _resolve_lab_column(meta, hints)
                   for lab, hints in LAB_LABEL_HINTS.items()}
    survival = extract_survival_labels(
        meta, time_col=os_time_col, event_col=os_event_col,
        id_col=meta.columns[0],
    )
    logger.info(
        "GSE136337: %d genes x %d samples; labs resolved: %s",
        expr.shape[0], expr.shape[1],
        {k: v for k, v in lab_columns.items() if v},
    )
    return {
        "expression": expr,
        "metadata": meta,
        "survival": survival,
        "lab_columns": lab_columns,
    }
