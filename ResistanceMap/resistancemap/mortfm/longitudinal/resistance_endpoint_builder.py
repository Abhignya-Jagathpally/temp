"""
resistancemap/mortfm/longitudinal/resistance_endpoint_builder.py
================================================================
Build :class:`ResistanceEndpoint` records from the MMRF outcomes table.

Available signals in ``mmrf_outcomes_treatment.tsv``:
  * vital_status, days_to_death, days_to_last_follow_up  → OS (NOT resistance)
  * n_treatments                                          → treatment lines used
  * bort_1L                                               → 1L drug exposure
  * tt2L_days, had_2L                                     → time-to-2nd-line (resistance proxy)

Resistance-relevant endpoint built here:
  * ``time_to_next_treatment`` — TT2L; event_observed = had_2L. This is a
    bona-fide resistance-emergence proxy (a 2nd-line switch implies the
    1st-line regimen failed). The endpoint registry accepts it under
    ``progression_free_survival``-equivalent semantics.

OS is NOT emitted as a ResistanceEndpoint. We emit it separately under the
"death" type so the survival head can validate technically without
unlocking resistance claims.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from resistancemap.mortfm.longitudinal.schemas import ResistanceEndpoint

logger = logging.getLogger(__name__)


def build_endpoints_from_mmrf_outcomes(
    outcomes_tsv: str = "data/processed/mmrf_outcomes_treatment.tsv",
) -> dict[str, List[ResistanceEndpoint]]:
    """Return ``{patient_id: [ResistanceEndpoint, ...]}``.

    Always emits:
      * ``time_to_next_treatment`` if tt2L_days and had_2L are present
      * ``overall_survival`` as a ``death`` endpoint (NOT a resistance claim)
    """
    p = Path(outcomes_tsv)
    if not p.exists():
        raise FileNotFoundError(f"MMRF outcomes not found at {p}")
    df = pd.read_csv(p, sep="\t", low_memory=False)
    out: dict[str, List[ResistanceEndpoint]] = {}
    for _, row in df.iterrows():
        pid = str(row["submitter_id"])
        endpoints: List[ResistanceEndpoint] = []
        # time_to_next_treatment (TT2L)
        tt2l = row.get("tt2L_days")
        had_2l = row.get("had_2L")
        if pd.notna(tt2l) and pd.notna(had_2l):
            endpoints.append(ResistanceEndpoint(
                endpoint_name="time_to_next_treatment",
                endpoint_type="progression",
                event_time_days=float(tt2l),
                event_observed=bool(int(had_2l)),
                drug_context=("bortezomib_1L" if int(row.get("bort_1L", 0) or 0) else None),
                evidence_level="clinical",
                notes="TT2L: time from 1L start to 2L treatment switch; "
                      "2L switch is a clinical resistance proxy.",
            ))
        # OS — recorded for survival-head technical validation only
        days_death = row.get("days_to_death")
        days_followup = row.get("days_to_last_follow_up")
        if pd.notna(row.get("vital_status")):
            event_obs = str(row["vital_status"]).lower() == "dead"
            event_time = float(days_death) if event_obs and pd.notna(days_death) else (
                float(days_followup) if pd.notna(days_followup) else 0.0
            )
            endpoints.append(ResistanceEndpoint(
                endpoint_name="overall_survival",
                endpoint_type="death",
                event_time_days=event_time,
                event_observed=event_obs,
                evidence_level="clinical",
                notes="OS is NOT a resistance endpoint. Recorded for "
                      "survival-head technical validation only.",
            ))
        if endpoints:
            out[pid] = endpoints
    logger.info("Built endpoints for %d MMRF patients from %s", len(out), p)
    return out
