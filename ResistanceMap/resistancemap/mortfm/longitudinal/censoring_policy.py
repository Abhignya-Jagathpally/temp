"""
resistancemap/mortfm/longitudinal/censoring_policy.py
======================================================
Single source of truth for how MORT-FM treats right-censoring,
administrative censoring, and missing event-time entries.

Rules (v17):
  * If event_observed == True and event_time is finite > 0 -> uncensored.
  * If event_observed == False:
      - finite event_time > 0  -> administratively censored at that time
      - missing event_time     -> drop (no usable observation)
  * No "best-effort" event-time backfill from non-event metadata.
    Cohorts that need backfill (MMRF stage labels -> TT2L) must do so
    explicitly via :func:`backfill_calendar_time_from_tt2L` and the
    LongitudinalPair record records ``delta_t_days`` provenance.
"""

from __future__ import annotations

import logging
from typing import List

from resistancemap.mortfm.longitudinal.schemas import LongitudinalPair

logger = logging.getLogger(__name__)


def apply_censoring_policy(pairs: List[LongitudinalPair]) -> dict:
    """Filter pairs per the rules above; return a (kept, dropped) report."""
    kept: List[LongitudinalPair] = []
    dropped_no_time: int = 0
    dropped_zero_dt: int = 0
    pure_admin_censoring: int = 0
    pure_event_observed: int = 0
    for p in pairs:
        if p.endpoint is None:
            kept.append(p)
            continue
        et = p.endpoint.event_time_days
        eo = p.endpoint.event_observed
        if et is None or et <= 0:
            if eo:
                dropped_zero_dt += 1
            else:
                dropped_no_time += 1
            continue
        if eo:
            pure_event_observed += 1
        else:
            pure_admin_censoring += 1
        kept.append(p)
    return {
        "n_input": len(pairs),
        "n_kept": len(kept),
        "n_dropped_no_event_time": dropped_no_time,
        "n_dropped_zero_or_negative_event_time": dropped_zero_dt,
        "n_event_observed": pure_event_observed,
        "n_administratively_censored": pure_admin_censoring,
        "kept": kept,
    }
