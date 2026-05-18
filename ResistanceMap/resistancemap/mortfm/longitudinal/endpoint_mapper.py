"""
resistancemap/mortfm/longitudinal/endpoint_mapper.py
=====================================================
Map raw cohort columns to the canonical ResistanceEndpoint schema.

Different cohorts express the same biological event with different
column names. The mapper provides a single bridge to the
:class:`resistancemap.mortfm.longitudinal.schemas.ResistanceEndpoint`
type used by the trainer + gate auditor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from resistancemap.mortfm.longitudinal.schemas import ResistanceEndpoint

logger = logging.getLogger(__name__)


@dataclass
class EndpointMapSpec:
    cohort_name: str
    time_column: str
    event_column: str
    endpoint_type: str
    endpoint_name: str
    drug_column: Optional[str] = None
    notes: str = ""


# Cohort-specific mappings. Single source of truth — never duplicate
# in the trainer scripts.
COHORT_ENDPOINT_MAPS = {
    "MMRF_IA12_paired::TT2L": EndpointMapSpec(
        cohort_name="MMRF_IA12_paired",
        time_column="tt2L_days",
        event_column="had_2L",
        endpoint_type="progression",
        endpoint_name="time_to_next_treatment",
        drug_column="bort_1L",
        notes="2L treatment switch is a clinical resistance proxy.",
    ),
    "MMRF_IA12_paired::OS": EndpointMapSpec(
        cohort_name="MMRF_IA12_paired",
        time_column="days_to_death",
        event_column="vital_status",
        endpoint_type="death",
        endpoint_name="overall_survival",
        notes="OS is NOT a resistance endpoint. Recorded for survival-head "
              "technical validation only.",
    ),
    "GSE39754_Brioli_paired::relapse_pair": EndpointMapSpec(
        cohort_name="GSE39754_Brioli_paired",
        time_column="",
        event_column="",
        endpoint_type="molecular_resistance_state",
        endpoint_name="paired_diagnosis_relapse",
        notes="Paired molecular state at relapse — substrate for "
              "longitudinal_trajectory; no calendar time required.",
    ),
}


def map_outcomes_to_endpoints(
    cohort_name: str,
    outcomes_df: pd.DataFrame,
    *,
    map_key: Optional[str] = None,
    patient_id_col: str = "submitter_id",
) -> dict[str, List[ResistanceEndpoint]]:
    """Return ``{patient_id: [ResistanceEndpoint, ...]}`` for a cohort."""
    keys = [k for k in COHORT_ENDPOINT_MAPS if k.startswith(cohort_name + "::")]
    if map_key:
        keys = [k for k in keys if k == map_key]
    if not keys:
        logger.warning("No endpoint maps for cohort %s", cohort_name)
        return {}
    out: dict[str, List[ResistanceEndpoint]] = {}
    for k in keys:
        spec = COHORT_ENDPOINT_MAPS[k]
        for _, row in outcomes_df.iterrows():
            pid = str(row[patient_id_col])
            if not spec.time_column or not spec.event_column:
                continue
            t = row.get(spec.time_column)
            e = row.get(spec.event_column)
            if pd.isna(t) or pd.isna(e):
                continue
            if isinstance(e, str):
                event_observed = e.lower() == "dead"
            else:
                event_observed = bool(int(e))
            drug_ctx = None
            if spec.drug_column and pd.notna(row.get(spec.drug_column)):
                drug_ctx = f"{spec.drug_column}={row[spec.drug_column]}"
            out.setdefault(pid, []).append(ResistanceEndpoint(
                endpoint_name=spec.endpoint_name,
                endpoint_type=spec.endpoint_type,
                event_time_days=float(t),
                event_observed=event_observed,
                drug_context=drug_ctx,
                evidence_level="clinical",
                notes=spec.notes,
            ))
    return out
