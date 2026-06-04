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


# ---------------------------------------------------------------------------
# Baseline covariates (the original 5). The canonical model consumes clinical
# as width-5; `encode_for_lens` therefore stays backward-compatible and returns
# exactly these unless the caller opts into the lab-aware longitudinal path.
# ---------------------------------------------------------------------------
BASELINE_FEATURE_NAMES: List[str] = [
    "iss_stage_ordinal",
    "age_at_dx_zscore",
    "gender_is_male",
    "bort_1L",
    "n_treatments_norm",
]

# Public name kept stable for existing callers (model.py expects (B, 5)).
CLINICAL_FEATURE_NAMES: List[str] = BASELINE_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Bug #1 workaround — lab-first longitudinal vocabulary.
#
# The handicapped 5-variable clinical encoder is replaced (additively, opt-in)
# by the full routine-lab panel. These names map to ml_mmrf's CoMMpass
# PER_PATIENT_VISIT D_LAB_* columns. The encoder reads whatever columns are
# present and masks the rest — no fabrication, no imputation. It unblocks the
# moment a visit-level lab table is supplied (MMRF Virtual Lab, or a
# GEO/GDC-derived lab proxy table), without changing the default 5-feature path.
# ---------------------------------------------------------------------------
LAB_COLUMN_MAP: dict = {
    "serum_m_protein": "D_LAB_serum_m_protein",
    "serum_kappa": "D_LAB_serum_kappa",
    "serum_lambda": "D_LAB_serum_lambda",
    "serum_beta2_microglobulin": "D_LAB_serum_beta2_microglobulin",
    "serum_igg": "D_LAB_serum_igg",
    "serum_iga": "D_LAB_serum_iga",
    "serum_igm": "D_LAB_serum_igm",
    "chem_albumin": "D_LAB_chem_albumin",
    "chem_creatinine": "D_LAB_chem_creatinine",
    "chem_calcium": "D_LAB_chem_calcium",
    "chem_ldh": "D_LAB_chem_ldh",
    "cbc_hemoglobin": "D_LAB_cbc_hemoglobin",
    "cbc_wbc": "D_LAB_cbc_wbc",
    "cbc_platelet": "D_LAB_cbc_platelet",
    "cbc_abs_neut": "D_LAB_cbc_abs_neut",
}

# Direct labs (order is stable) followed by the derived kappa/lambda ratio.
_DIRECT_LAB_FEATURES: List[str] = list(LAB_COLUMN_MAP.keys())
DERIVED_LAB_FEATURES: List[str] = ["kl_ratio"]
LONGITUDINAL_LAB_FEATURES: List[str] = _DIRECT_LAB_FEATURES + DERIVED_LAB_FEATURES

# Full clinical vocabulary the v20 lab-first model will consume (>= 21 features).
FULL_CLINICAL_FEATURE_NAMES: List[str] = (
    LONGITUDINAL_LAB_FEATURES + BASELINE_FEATURE_NAMES
)


@dataclass
class ClinicalEncoding:
    features: np.ndarray   # (n_patients, n_features)
    mask: np.ndarray       # (n_patients, n_features) True where observed
    feature_names: List[str]
    patient_ids: List[str]
    age_mean_days: float
    age_std_days: float


@dataclass
class LongitudinalClinicalEncoding:
    """Lab trajectories as ``(n_patients, n_timepoints, n_features)``.

    ``features`` is zero-filled where a lab is missing; ``mask`` (same shape)
    is True only where a real value was observed, so the model can distinguish
    "0 = measured zero" from "missing". ``time_mask`` is True where a patient
    has any observation in that time bin.
    """

    features: np.ndarray    # (N, T, F)
    mask: np.ndarray        # (N, T, F) True where observed
    time_mask: np.ndarray   # (N, T) True where the bin has any observation
    feature_names: List[str]
    patient_ids: List[str]
    interval_days: int


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


def encode_longitudinal_labs(
    visit_csv: str,
    patient_ids: Sequence[str],
    *,
    patient_col: str = "PUBLIC_ID",
    day_col: Optional[str] = "VISITDY",
    max_timepoints: int = 33,
    interval_days: int = 60,
    sep: Optional[str] = None,
) -> LongitudinalClinicalEncoding:
    """Encode routine-lab trajectories from a PER_PATIENT_VISIT-style table.

    Bug #1 workaround. Reads the ``D_LAB_*`` columns named in ``LAB_COLUMN_MAP``
    (plus the derived kappa/lambda ratio) into a ``(N, T, F)`` tensor binned on
    ``day_col`` at ``interval_days`` resolution. Columns that are absent from the
    file are masked out — never fabricated. The returned ``feature_names`` is
    exactly ``LONGITUDINAL_LAB_FEATURES``.

    Parameters
    ----------
    visit_csv
        Path to the longitudinal lab table (MMRF PER_PATIENT_VISIT, or any
        GEO/GDC-derived table that exposes the same ``D_LAB_*`` columns).
    patient_ids
        Fixed patient order; rows are returned in this order.
    day_col
        Column holding days-from-baseline for binning. If None (or absent),
        visits are binned by their per-patient observation order instead.
    """
    if sep is None:
        sep = "\t" if str(visit_csv).endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(visit_csv, sep=sep, low_memory=False)

    if patient_col not in df.columns:
        raise ValueError(
            f"Visit table missing patient column {patient_col!r}. "
            f"Available: {sorted(df.columns)[:25]}"
        )

    feature_names = list(LONGITUDINAL_LAB_FEATURES)
    n_pat = len(patient_ids)
    n_t = int(max_timepoints)
    n_f = len(feature_names)

    features = np.zeros((n_pat, n_t, n_f), dtype=np.float32)
    mask = np.zeros((n_pat, n_t, n_f), dtype=bool)
    time_mask = np.zeros((n_pat, n_t), dtype=bool)

    # Which mapped lab columns actually exist on disk.
    present_cols = {k: v for k, v in LAB_COLUMN_MAP.items() if v in df.columns}
    if not present_cols:
        logger.warning(
            "encode_longitudinal_labs: none of the expected D_LAB_* columns are "
            "present in %s — every lab feature will be masked out.",
            visit_csv,
        )

    use_day = bool(day_col) and (day_col in df.columns)
    by_patient = {pid: g for pid, g in df.groupby(patient_col)}

    for i, pid in enumerate(patient_ids):
        g = by_patient.get(pid)
        if g is None:
            continue
        if use_day:
            g = g.sort_values(day_col)
            bins = (g[day_col].astype(float) // interval_days).astype("Int64")
        else:
            # Fall back to per-patient visit order when no day column exists.
            bins = pd.Series(range(len(g)), index=g.index, dtype="Int64")

        for row_idx, (_, r) in enumerate(g.iterrows()):
            b = bins.iloc[row_idx]
            if pd.isna(b):
                continue
            t = int(b)
            if t < 0 or t >= n_t:
                continue
            # Direct labs.
            kappa = lam = None
            for fi, fname in enumerate(_DIRECT_LAB_FEATURES):
                col = present_cols.get(fname)
                if col is None:
                    continue
                val = r.get(col)
                if pd.notna(val):
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        continue
                    features[i, t, fi] = fval
                    mask[i, t, fi] = True
                    time_mask[i, t] = True
                    if fname == "serum_kappa":
                        kappa = fval
                    elif fname == "serum_lambda":
                        lam = fval
            # Derived kappa/lambda ratio (only when both observed and lam != 0).
            kl_idx = feature_names.index("kl_ratio")
            if kappa is not None and lam not in (None, 0.0):
                features[i, t, kl_idx] = kappa / lam
                mask[i, t, kl_idx] = True
                time_mask[i, t] = True

    return LongitudinalClinicalEncoding(
        features=features,
        mask=mask,
        time_mask=time_mask,
        feature_names=feature_names,
        patient_ids=list(patient_ids),
        interval_days=int(interval_days),
    )
