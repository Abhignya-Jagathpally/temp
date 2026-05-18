"""
resistancemap/mortfm/longitudinal/longitudinal_dataset.py
=========================================================
Concrete dataset wrapper over :class:`LongitudinalPair` records.

Thin: the Block-F longitudinal trainer (when it ships in Lane 6) constructs
this dataset, splits patient-disjoint, and hands batches to MORT-FM-LENS.
Here we only carry the data; we don't load expression matrices yet — the
trainer pulls them lazily from disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from resistancemap.mortfm.longitudinal.schemas import LongitudinalPair

logger = logging.getLogger(__name__)


@dataclass
class LongitudinalDataset:
    pairs: List[LongitudinalPair]

    def __len__(self) -> int:
        return len(self.pairs)

    def patient_ids(self) -> List[str]:
        return sorted({p.patient_id for p in self.pairs})

    def to_parquet(self, path: str) -> str:
        rows = []
        for p in self.pairs:
            rows.append({
                "patient_id": p.patient_id,
                "x_t_sample_id": p.x_t_sample_id,
                "x_future_sample_id": p.x_future_sample_id,
                "delta_t_days": p.delta_t_days,
                "treatment_between": ";".join(p.treatment_between),
                "endpoint_name": p.endpoint.endpoint_name if p.endpoint else None,
                "endpoint_event_observed": (
                    p.endpoint.event_observed if p.endpoint else None
                ),
                "endpoint_event_time_days": (
                    p.endpoint.event_time_days if p.endpoint else None
                ),
                "endpoint_drug_context": (
                    p.endpoint.drug_context if p.endpoint else None
                ),
                "endpoint_evidence_level": (
                    p.endpoint.evidence_level if p.endpoint else None
                ),
                "cohort": p.cohort,
            })
        df = pd.DataFrame(rows)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info("Wrote %d LongitudinalPair rows -> %s", len(df), path)
        return path
