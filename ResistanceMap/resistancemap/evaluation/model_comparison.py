"""
resistancemap/evaluation/model_comparison.py
============================================
Unified writer / reader for the baseline-vs-MORT-FM comparison table.

Schema (frozen — see ``phase_10_baselines.md`` §4):

  * ``model_name``    — registry name, e.g. ``"elasticnet"``, ``"mortfm_full"``.
  * ``split``         — ``"train" | "val" | "test"``.
  * ``metric_name``   — ``"mse" | "rmse" | "auroc" | "auprc" | "c_index" | "ibs" | "emd"``.
  * ``metric_value``  — float, real evaluation only.
  * ``ci_low``, ``ci_high`` — bootstrap-CI bounds (NaN if not computed).
  * ``n``             — n_samples in the split.
  * ``seed``          — RNG seed used for fitting.
  * ``hash_data``     — sha256 (first 16 hex chars) of the sorted patient
                         IDs in the split — gates split drift across runs.
  * ``hash_code``     — git SHA at run time (caller-supplied).

Single-writer pattern: every baseline AND MORT-FM appends rows to one
parquet file, so figure builders read from one source of truth.

Honest-behaviour invariants
---------------------------
* No metric is computed inside this module. Callers compute the metric and
  pass the row in; the module only does the I/O + schema validation.
* No np.random / synthetic value is ever written.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union

import pandas as pd
from pydantic import BaseModel, Field


PathLike = Union[str, Path]


class ComparisonRow(BaseModel):
    """A single (model_name, split, metric) tuple in the comparison table."""

    model_name: str = Field(..., description="Registry name of the baseline / MORT-FM variant")
    split: str = Field(..., description="train | val | test")
    metric_name: str = Field(..., description="mse | rmse | auroc | auprc | c_index | ibs | emd")
    metric_value: float = Field(..., description="Computed metric value")
    ci_low: float = Field(float("nan"), description="Bootstrap CI lower bound (NaN if not computed)")
    ci_high: float = Field(float("nan"), description="Bootstrap CI upper bound (NaN if not computed)")
    n: int = Field(..., description="Number of samples in the split")
    seed: int = Field(..., description="Seed used for fitting the model")
    hash_data: str = Field(..., description="sha256[:16] of sorted patient IDs in the split")
    hash_code: str = Field(..., description="Git SHA at run time (caller-supplied)")
    extra: str = Field("{}", description="JSON blob for hyperparameters / provenance")


def hash_patient_ids(patient_ids: Sequence[str]) -> str:
    """Return the canonical split-hash used in ``ComparisonRow.hash_data``.

    Sorting + newline-joining + sha256 + 16-char truncation is the same as
    ``phase_10_baselines.md`` §4. Two runs that produce the same sorted
    patient list produce the same hash.
    """
    joined = "\n".join(sorted(str(p) for p in patient_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def write_comparison_rows(
    rows: Iterable[ComparisonRow],
    path: PathLike,
    *,
    append: bool = True,
) -> Path:
    """Append ``rows`` to the parquet at ``path`` (or overwrite if ``append=False``).

    Parameters
    ----------
    rows : iterable of ComparisonRow
        The new rows to write.
    path : str | Path
        Output parquet file. Parent directories are created as needed.
    append : bool
        If True (default) and ``path`` already exists, the new rows are
        concatenated to the existing table. If False, the existing file
        is overwritten.

    Returns
    -------
    Path
        The path written to.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([r.model_dump() for r in rows])
    if append and p.exists():
        existing = pd.read_parquet(p)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(p, index=False)
    return p


def read_comparison(path: PathLike) -> pd.DataFrame:
    """Read the comparison parquet at ``path`` and return it as a DataFrame.

    Validates that every required ``ComparisonRow`` field is present.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"comparison parquet not found: {p}")
    df = pd.read_parquet(p)
    required = set(ComparisonRow.model_fields.keys())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"comparison parquet at {p} is missing required columns: {sorted(missing)}"
        )
    return df
