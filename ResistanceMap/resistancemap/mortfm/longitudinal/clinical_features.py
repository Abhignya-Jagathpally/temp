"""
resistancemap/mortfm/longitudinal/clinical_features.py
=======================================================
Encode the MMRF clinical covariates needed for the LENS resistance head.

Available signals (from ``mmrf_outcomes_treatment.tsv``):
  * iss_stage          — ordinal {I, II, III}; encoded as 0, 1, 2
  * age_at_diagnosis_days — float days; z-scored cohort-level
  * gender             — {male, female}; binary
  * bort_1L            — int {0, 1}; bortezomib in 1L
  * n_treatments       — int; clipped at 30

The output is a deterministic ``(n_patients, n_features)`` tensor with
**no randomness, no imputation of missing data** — missing values stay
NaN and the caller decides whether to drop or zero-fill (we expose
``encode_for_lens`` which zero-fills + emits a mask so the model can
distinguish "0 = no" from "missing").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


CLINICAL_FEATURE_NAMES: List[str] = [
    "iss_stage_ordinal",
    "age_at_dx_zscore",
    "gender_is_male",
    "bort_1L",
    "n_treatments_norm",
]


@dataclass
class ClinicalEncoding:
    features: np.ndarray   # (n_patients, n_features)
    mask: np.ndarray       # (n_patients, n_features) True where observed
    feature_names: List[str]
    patient_ids: List[str]
    age_mean_days: float
    age_std_days: float


def _iss_to_ordinal(stage: str) -> Optional[int]:
    s = str(stage).strip().upper()
    return {"I": 0, "II": 1, "III": 2}.get(s)


def encode_for_lens(
    outcomes_tsv: str,
    patient_ids: Sequence[str],
) -> ClinicalEncoding:
    """Encode the clinical covariates for a fixed patient_id order.

    ``patient_ids`` MUST be the same order the LENS trainer iterates over;
    the encoder returns rows in that same order.
    """
    df = pd.read_csv(outcomes_tsv, sep="\t", low_memory=False)
    df = df.set_index("submitter_id")
    ages = df["age_at_diagnosis_days"].dropna()
    age_mean = float(ages.mean()) if len(ages) else 0.0
    age_std = float(ages.std()) if len(ages) > 1 else 1.0
    rows = []
    masks = []
    for pid in patient_ids:
        if pid in df.index:
            r = df.loc[pid]
            if isinstance(r, pd.DataFrame):  # duplicates
                r = r.iloc[0]
            iss = _iss_to_ordinal(r.get("iss_stage", ""))
            age = r.get("age_at_diagnosis_days")
            gender = r.get("gender")
            bort = r.get("bort_1L")
            ntx = r.get("n_treatments")
            row = [
                float(iss) if iss is not None else 0.0,
                float((age - age_mean) / max(age_std, 1.0)) if pd.notna(age) else 0.0,
                float(1.0 if str(gender).lower() == "male" else 0.0) if pd.notna(gender) else 0.0,
                float(bort) if pd.notna(bort) else 0.0,
                float(min(ntx, 30) / 30.0) if pd.notna(ntx) else 0.0,
            ]
            mask = [
                iss is not None, pd.notna(age), pd.notna(gender),
                pd.notna(bort), pd.notna(ntx),
            ]
        else:
            row = [0.0] * len(CLINICAL_FEATURE_NAMES)
            mask = [False] * len(CLINICAL_FEATURE_NAMES)
        rows.append(row); masks.append(mask)
    return ClinicalEncoding(
        features=np.asarray(rows, dtype=np.float32),
        mask=np.asarray(masks, dtype=bool),
        feature_names=CLINICAL_FEATURE_NAMES,
        patient_ids=list(patient_ids),
        age_mean_days=age_mean,
        age_std_days=age_std,
    )
