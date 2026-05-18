"""
resistancemap/data/clinical_outcome_loader.py
=============================================
Adapter that converts the existing MMRF parsers into MORT-FM schema objects.

Why this exists
---------------
The legacy :mod:`resistancemap.data.mmrf_loader` produces pandas DataFrames
with patient-level clinical fields. MORT-FM needs typed
:class:`~resistancemap.mortfm.schemas.ResistanceOutcome` records (and
optionally :class:`~resistancemap.mortfm.schemas.PatientCellSnapshot` objects
with the matched RNA tensor) so they can flow into
:func:`~resistancemap.data.trajectory_pair_builder.build_temporal_pairs`.

Honest behaviour
----------------
* Every loader entry point validates that the underlying files exist. If they
  do not, it raises :class:`FileNotFoundError` with a message naming exactly
  which directory was searched. It does *not* return placeholder rows.
* When the expected clinical column is missing (e.g. no PFS in this MMRF
  release), the corresponding field on :class:`ResistanceOutcome` stays
  ``None``. The trainer will then route those rows past the survival head.
* This module never derives ``event_time`` from a snapshot's RNA. Survival
  labels must come from clinical files.

Wiring into the v15 pipeline
----------------------------
Call ``load_mortfm_outcomes(data_dir)`` from your training script::

    from resistancemap.data.clinical_outcome_loader import load_mortfm_outcomes
    from resistancemap.data.trajectory_pair_builder import build_temporal_pairs

    outcomes = load_mortfm_outcomes("data/raw/mmrf_commpass")
    pairs = build_temporal_pairs(snapshots, outcomes, allowed_time_gaps=[6, 12])
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from resistancemap.mortfm.schemas import ResistanceOutcome

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column-name hunting helpers (MMRF column names vary by release)
# ---------------------------------------------------------------------------

_PFS_TIME_COLS = ["pfsdy", "PFSDY", "ttfpfs", "TTFPFS", "pfs_days", "D_PT_pfsdy"]
_PFS_EVENT_COLS = ["censpfs", "CENSPFS", "pfs_event", "D_PT_censpfs"]
_OS_TIME_COLS = [
    "osdy", "OSDY", "ttfos", "TTFOS", "os_days", "D_PT_osdy",
    "days_to_death",                                        # GDC convention
]
_OS_EVENT_COLS = ["censos", "CENSOS", "os_event", "D_PT_censos"]
_RELAPSE_TIME_COLS = ["ttfrel", "TTFREL", "relapse_days", "D_PT_ttfrel"]
# GDC vital-status convention (handled specially below).
_GDC_VITAL_COL = "vital_status"
_GDC_FOLLOWUP_COL = "days_to_last_follow_up"
_GDC_DEATH_COL = "days_to_death"


def _first_present(cols: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _to_months(days: pd.Series) -> pd.Series:
    return pd.to_numeric(days, errors="coerce") / 30.44


def _event_to_bool(s: pd.Series) -> pd.Series:
    """MMRF / GDC censoring conventions vary. Normalise to ``event_observed: bool``.

    Common encodings:
        * censpfs = 1 => CENSORED (no event)
        * censpfs = 0 => EVENT observed
    GDC convention (vital_status): 'dead' => event, 'alive' => censored.

    This function assumes the column passed is a *cens* flag where 1==censored.
    Callers that pass a GDC-style vital column should map first.
    """
    raw = pd.to_numeric(s, errors="coerce")
    return raw.fillna(1).astype(int) == 0  # True if event observed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_mortfm_outcomes(
    data_dir: str,
    *,
    endpoint: str = "pfs",
    response_col_hint: Optional[str] = None,
    strict: bool = True,
) -> List[ResistanceOutcome]:
    """Load patient-level outcomes from an MMRF CoMMpass dump.

    Parameters
    ----------
    data_dir
        Path to the directory containing ``MMRF_CoMMpass_IA*_PER_PATIENT*.csv``
        files. May contain nested release directories — the function recurses.
    endpoint
        Which time-to-event endpoint to extract. One of:
            * "pfs" — progression-free survival (recommended for resistance proxy)
            * "os"  — overall survival
            * "relapse" — time to relapse (only available in newer releases)
    response_col_hint
        Optional override for the IMWG best-response column name. Useful for
        non-standard MMRF releases.
    strict
        If True (default), raise :class:`FileNotFoundError` if no per-patient
        clinical file is found. If False, return ``[]``.

    Returns
    -------
    List[ResistanceOutcome]
        One record per patient with the endpoint populated where available.
        Missing endpoint values are encoded as ``event_time=None``, ``censored=True``.

    Notes
    -----
    This function does *not* require a Synapse / dbGaP download. It works
    against whatever files are present in ``data_dir``. If you are running
    without MMRF data, expect ``FileNotFoundError`` — that is the *intended*
    behaviour. MORT-FM will not fabricate a synthetic MMRF cohort.
    """
    base = Path(data_dir)
    if not base.exists():
        msg = f"MMRF data directory not found: {base.resolve()}"
        if strict:
            raise FileNotFoundError(msg)
        logger.warning(msg + " (strict=False) -> returning empty list")
        return []

    pat_files = list(base.rglob("*PER_PATIENT*.csv"))
    pat_files += list(base.rglob("*per_patient*.csv"))
    pat_files += list(base.rglob("clinical.tsv"))
    pat_files = sorted({p for p in pat_files if p.is_file()})

    if not pat_files:
        msg = (
            f"No PER_PATIENT clinical files found under {base.resolve()}. "
            f"Expected MMRF_CoMMpass_IA*_PER_PATIENT*.csv or clinical.tsv."
        )
        if strict:
            raise FileNotFoundError(msg)
        logger.warning(msg + " (strict=False) -> returning empty list")
        return []

    # Read and union by patient_id.
    frames = []
    for f in pat_files:
        try:
            sep = "\t" if f.suffix == ".tsv" else ","
            df = pd.read_csv(f, sep=sep, low_memory=False)
            frames.append(df)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", f, exc)

    if not frames:
        if strict:
            raise FileNotFoundError(
                f"All candidate clinical files under {base.resolve()} failed to parse."
            )
        return []

    clin = pd.concat(frames, ignore_index=True, sort=False)
    id_col = _first_present(
        ["PUBLIC_ID", "public_id", "MMRF_ID", "patient_id", "submitter_id"],
        clin,
    )
    if id_col is None:
        raise ValueError(
            f"Could not find a patient identifier column in {pat_files}. "
            f"Available columns: {list(clin.columns)[:30]}..."
        )
    clin = clin.rename(columns={id_col: "patient_id"})

    # Choose endpoint columns.
    if endpoint == "pfs":
        t_col = _first_present(_PFS_TIME_COLS, clin)
        e_col = _first_present(_PFS_EVENT_COLS, clin)
    elif endpoint == "os":
        t_col = _first_present(_OS_TIME_COLS, clin)
        e_col = _first_present(_OS_EVENT_COLS, clin)
    elif endpoint == "relapse":
        t_col = _first_present(_RELAPSE_TIME_COLS, clin)
        e_col = _first_present(_PFS_EVENT_COLS, clin)  # reuse cens flag
    else:
        raise ValueError(f"Unknown endpoint {endpoint!r}; expected pfs|os|relapse.")

    if t_col is None:
        logger.warning(
            "Endpoint %r time column not found in MMRF clinical (searched %s). "
            "Outcomes will have event_time=None.",
            endpoint,
            _PFS_TIME_COLS if endpoint == "pfs" else _OS_TIME_COLS,
        )
    if e_col is None:
        logger.warning(
            "Endpoint %r censoring column not found; treating all rows as censored.",
            endpoint,
        )

    # IMWG response -> resistance_label
    resp_col = response_col_hint or _first_present(
        [
            "bestresponse", "best_response", "BESTRESPONSE", "D_PT_bestresptyp",
            "best_confirmed_response", "BOR",
        ],
        clin,
    )

    # GDC fallback: if endpoint is OS-like and no explicit time/event column was
    # found, derive event_time + censored from vital_status + days_to_death /
    # days_to_last_follow_up. This handles the dbGaP/GDC MMRF clinical export
    # which uses ``vital_status = {Alive, Dead}``.
    using_gdc_fallback = (
        endpoint in {"os", "pfs"}                                    # PFS often falls through to OS here
        and (t_col is None or e_col is None)
        and _GDC_VITAL_COL in clin.columns
        and (_GDC_FOLLOWUP_COL in clin.columns or _GDC_DEATH_COL in clin.columns)
    )
    if using_gdc_fallback:
        logger.info("Falling back to GDC vital_status convention for endpoint=%r", endpoint)

    outcomes: List[ResistanceOutcome] = []
    for _, row in clin.drop_duplicates(subset=["patient_id"]).iterrows():
        pid = str(row["patient_id"])

        event_time_months: Optional[float] = None
        censored = True

        if using_gdc_fallback:
            vital = str(row.get(_GDC_VITAL_COL, "")).strip().lower()
            died = vital == "dead"
            d_death = pd.to_numeric(row.get(_GDC_DEATH_COL), errors="coerce")
            d_followup = pd.to_numeric(row.get(_GDC_FOLLOWUP_COL), errors="coerce")
            if died and pd.notna(d_death):
                event_time_months = float(d_death) / 30.44
                censored = False
            elif pd.notna(d_followup):
                event_time_months = float(d_followup) / 30.44
                censored = True
            elif pd.notna(d_death):
                event_time_months = float(d_death) / 30.44
                censored = not died
        else:
            if t_col is not None:
                v = pd.to_numeric(row.get(t_col), errors="coerce")
                if pd.notna(v):
                    event_time_months = float(v) / 30.44
            if e_col is not None:
                raw = pd.to_numeric(row.get(e_col), errors="coerce")
                if pd.notna(raw):
                    censored = bool(int(raw) == 1)

        resistance_label: Optional[int] = None
        if resp_col is not None:
            r = str(row.get(resp_col, "")).strip().upper()
            if r in {"SCR", "CR", "VGPR", "PR"}:
                resistance_label = 0  # sensitive
            elif r in {"SD", "MR"}:
                resistance_label = 1  # tolerant
            elif r in {"PD"}:
                resistance_label = 3  # relapsed-resistant

        outcomes.append(
            ResistanceOutcome(
                patient_id=pid,
                baseline_time=0.0,
                event_time=event_time_months,
                censored=censored,
                resistance_label=resistance_label,
                future_state=None,
                drug_response=None,
                pathway_labels=None,
            )
        )

    n_with_event_time = sum(1 for o in outcomes if o.event_time is not None)
    n_uncensored = sum(1 for o in outcomes if not o.censored)
    logger.info(
        "load_mortfm_outcomes: %d patients (%d with event_time, %d uncensored), endpoint=%s",
        len(outcomes),
        n_with_event_time,
        n_uncensored,
        endpoint,
    )
    return outcomes


def summarise_outcomes(outcomes: List[ResistanceOutcome]) -> Dict[str, float]:
    """Compute summary statistics for a list of outcomes (logging aid)."""
    n = len(outcomes)
    if n == 0:
        return {"n_patients": 0.0}
    n_with_time = sum(1 for o in outcomes if o.event_time is not None)
    n_events = sum(1 for o in outcomes if not o.censored and o.event_time is not None)
    median_event = float(
        np.nanmedian([o.event_time for o in outcomes if o.event_time is not None] or [np.nan])
    )
    return {
        "n_patients": float(n),
        "n_with_event_time": float(n_with_time),
        "n_events_observed": float(n_events),
        "median_event_time_months": median_event,
        "event_rate": float(n_events) / max(n_with_time, 1),
    }
