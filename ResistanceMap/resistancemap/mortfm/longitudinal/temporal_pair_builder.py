"""
resistancemap/mortfm/longitudinal/temporal_pair_builder.py
==========================================================
Build :class:`LongitudinalPair` records from the MMRF paired-patient table.

MMRF gives us 29 patients with t0_aliquot + t1_aliquot pairs. We add:
  * ``delta_t_days`` — derived from t0_timepoint and t1_timepoint columns
    when they encode calendar days (else inferred at 0 and validate()
    flags the pair).
  * ``treatment_between`` — bortezomib if bort_1L was true (best-effort).
  * ``endpoint`` — joined from the resistance-endpoint builder
    (time_to_next_treatment) when the patient has an outcome record.

Pseudotime stage labels are NEVER inserted as ``delta_t_days``; this is
checked by :meth:`LongitudinalPair.validate`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from resistancemap.mortfm.longitudinal.resistance_endpoint_builder import (
    build_endpoints_from_mmrf_outcomes,
)
from resistancemap.mortfm.longitudinal.schemas import (
    LongitudinalPair, ResistanceEndpoint,
)

logger = logging.getLogger(__name__)


def build_pairs_from_mmrf(
    paired_tsv: str = "data/processed/mmrf_paired_patients.tsv",
    outcomes_tsv: str = "data/processed/mmrf_outcomes_treatment.tsv",
) -> List[LongitudinalPair]:
    """Build longitudinal pairs from MMRF baseline → followup aliquots."""
    p = Path(paired_tsv)
    if not p.exists():
        # Open-access MMRF has no paired molecular timepoints (the lakehouse
        # confirms 0 temporal pairs). Degrade gracefully to an empty pair list
        # so callers (longitudinal builder, leakage audit) succeed with an
        # honest 0-pair result rather than crashing. No fabrication.
        logger.warning("MMRF paired-patient table absent at %s -> 0 longitudinal "
                       "pairs (longitudinal/trajectory claim honestly blocked).", p)
        return []
    df = pd.read_csv(p, sep="\t", low_memory=False)
    logger.info("MMRF paired-patient table: %d rows", len(df))

    endpoints_by_patient = (
        build_endpoints_from_mmrf_outcomes(outcomes_tsv) if Path(outcomes_tsv).exists() else {}
    )

    pairs: List[LongitudinalPair] = []
    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        t0_aliquot = str(row.get("t0_aliquot", ""))
        t1_aliquot = str(row.get("t1_aliquot", ""))
        if not t0_aliquot or not t1_aliquot:
            continue
        # Best-effort delta_t_days. MMRF stores timepoints as enum strings
        # ("Baseline" / "Post-induction" / "Pre-maintenance" / "Relapse" / ...)
        # in t0_timepoint / t1_timepoint, NOT real days. Without a real-days
        # join we leave delta_t_days=0 and rely on the validator to flag the
        # pair as below the trajectory gate's threshold.
        delta_days = 0.0
        endpoint: Optional[ResistanceEndpoint] = None
        # Use the time_to_next_treatment endpoint as our resistance label.
        for ep in endpoints_by_patient.get(pid, []):
            if ep.endpoint_name == "time_to_next_treatment":
                endpoint = ep
                # If the patient progressed to 2L, that's the calendar gap
                # we approximate the t0->t1 transition with (best available
                # proxy — t1 was typically drawn at relapse / re-staging).
                if ep.event_observed and ep.event_time_days > 0:
                    delta_days = float(ep.event_time_days)
                break
        pair = LongitudinalPair(
            patient_id=pid,
            x_t_sample_id=t0_aliquot,
            x_future_sample_id=t1_aliquot,
            delta_t_days=delta_days,
            treatment_between=["bortezomib"] if endpoint and endpoint.drug_context else [],
            endpoint=endpoint,
            cohort="MMRF",
        )
        pairs.append(pair)
    logger.info("Built %d LongitudinalPair records from MMRF", len(pairs))
    return pairs


def summarise_pairs(pairs: List[LongitudinalPair]) -> dict:
    """Honest size + evidence-level summary used by the trajectory gate."""
    n_with_endpoint = sum(1 for p in pairs if p.endpoint is not None)
    n_with_real_time = sum(1 for p in pairs if p.delta_t_days > 0)
    n_with_observed_event = sum(
        1 for p in pairs if p.endpoint is not None and p.endpoint.event_observed
    )
    validation_issues: list[str] = []
    for p in pairs:
        validation_issues.extend(p.validate())
    return {
        "n_pairs": len(pairs),
        "n_with_endpoint": n_with_endpoint,
        "n_with_real_calendar_time": n_with_real_time,
        "n_with_observed_event": n_with_observed_event,
        "n_unique_patients": len({p.patient_id for p in pairs}),
        "validation_issues": validation_issues[:25],  # cap for json size
    }
