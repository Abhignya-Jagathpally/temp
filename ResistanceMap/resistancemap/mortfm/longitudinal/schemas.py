"""
resistancemap/mortfm/longitudinal/schemas.py
============================================
Typed primitives for longitudinal/resistance modelling.

Each dataclass carries the bare minimum the trainer + claim gates need;
data sources flesh out the rest via free-text fields.

Honest behaviour
----------------
* :attr:`ResistanceEndpoint.evidence_level` is mandatory. A "weak" endpoint
  is allowed in the schema but the claim gate refuses to grant a
  resistance_emergence claim from "weak" evidence alone.
* :class:`LongitudinalPair` requires ``delta_t_days`` in real calendar days.
  Pseudotime stage ORDERS (HD < MGUS < SMM < MM) MUST NOT be cast into
  this field — the trajectory gate explicitly checks for it.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


EvidenceLevel = Literal["clinical", "molecular", "ex_vivo", "weak"]
EndpointType = Literal[
    "progression",
    "relapse",
    "refractory_status",
    "mrd_conversion",
    "ex_vivo_resistance_shift",
    "molecular_resistance_state",
    "death",
    "censoring",
]


@dataclass
class TreatmentExposure:
    drug_name: str
    drug_class: Optional[str] = None
    start_day: Optional[float] = None
    end_day: Optional[float] = None
    dose: Optional[float] = None
    line: Optional[int] = None  # 1L, 2L, 3L, ...
    regimen_id: Optional[str] = None


@dataclass
class MolecularSample:
    sample_id: str
    aliquot_id: Optional[str] = None
    timepoint_label: str = "baseline"   # baseline | followup | relapse | ...
    timepoint_day: Optional[float] = None
    assay: str = "rna_bulk"             # rna_bulk | scrna | proteomics | atac
    parquet_path: Optional[str] = None
    notes: str = ""


@dataclass
class ClinicalEvent:
    event_name: str
    day: Optional[float] = None
    free_text: str = ""


@dataclass
class ResponseEvent:
    response: Literal["CR", "VGPR", "PR", "MR", "SD", "PD", "unknown"]
    day: Optional[float] = None
    drug_context: Optional[str] = None


@dataclass
class ResistanceEndpoint:
    endpoint_name: str
    endpoint_type: EndpointType
    event_time_days: float
    event_observed: bool
    drug_context: Optional[str] = None
    evidence_level: EvidenceLevel = "clinical"
    notes: str = ""


@dataclass
class PatientTimeline:
    patient_id: str
    diagnosis_day: Optional[float] = None
    treatment_lines: List[TreatmentExposure] = field(default_factory=list)
    molecular_samples: List[MolecularSample] = field(default_factory=list)
    clinical_events: List[ClinicalEvent] = field(default_factory=list)
    response_events: List[ResponseEvent] = field(default_factory=list)
    resistance_events: List[ResistanceEndpoint] = field(default_factory=list)
    cohort: str = "MMRF"


@dataclass
class LongitudinalPair:
    patient_id: str
    x_t_sample_id: str            # baseline molecular sample
    x_future_sample_id: str       # followup molecular sample
    delta_t_days: float            # MUST be real calendar days
    treatment_between: List[str] = field(default_factory=list)
    endpoint: Optional[ResistanceEndpoint] = None
    cohort: str = "MMRF"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.delta_t_days <= 0:
            problems.append(f"delta_t_days={self.delta_t_days} must be > 0")
        if not self.x_t_sample_id or not self.x_future_sample_id:
            problems.append("Both x_t_sample_id and x_future_sample_id required")
        return problems


# ---------------------------------------------------------------------------
# Canonical longitudinal panel artifact
# ---------------------------------------------------------------------------

@dataclass
class LongitudinalPanelRow:
    """One row of the canonical longitudinal panel artifact.

    Each row represents a single patient-visit, annotated with modality
    availability, treatment context, and (optionally) a survival/resistance
    endpoint.  The panel is the authoritative flat representation that
    downstream trainers and claim gates consume.
    """
    patient_id_hash: str
    cohort: str
    visit_id: str
    visit_time_days: float
    treatment_line: int
    drug_exposure_start: Optional[float]
    drug_exposure_end: Optional[float]
    modality_available_rna: bool
    modality_available_atac: bool
    modality_available_methylation: bool
    modality_available_proteomics: bool
    modality_available_phospho: bool
    modality_available_clinical: bool
    event_type: Optional[str]
    event_time_days: Optional[float]
    event_observed: Optional[bool]
    resistance_state_label: Optional[int]
    baseline_snapshot_uri: Optional[str]
    followup_snapshot_uri: Optional[str]


# ------ pseudotime sentinel values that MUST NOT appear as calendar days ----
_PSEUDOTIME_SENTINELS = {1.0, 2.0, 3.0, 4.0}  # HD=1, MGUS=2, SMM=3, MM=4


def _hash_patient_id(patient_id: str) -> str:
    """One-way SHA-256 hash of the patient identifier (first 16 hex chars)."""
    return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:16]


def build_panel_from_pairs(
    pairs: List[LongitudinalPair],
    cohort_registry: Any,
) -> List[LongitudinalPanelRow]:
    """Convert a list of :class:`LongitudinalPair` into panel rows.

    Each pair produces **two** rows — one for the baseline visit (t=0) and
    one for the follow-up visit (t=delta_t_days).

    Parameters
    ----------
    pairs:
        Validated :class:`LongitudinalPair` records.
    cohort_registry:
        A ``Dict[str, LongitudinalCohortSpec]`` (or compatible mapping)
        used to look up modality availability for the pair's cohort.
    """
    rows: List[LongitudinalPanelRow] = []
    for pair in pairs:
        pid_hash = _hash_patient_id(pair.patient_id)
        cohort = pair.cohort

        # Resolve modality flags from registry if available
        spec = cohort_registry.get(cohort) if cohort_registry else None
        modality_rna = False
        modality_atac = False
        modality_methylation = False
        modality_proteomics = False
        modality_phospho = False
        modality_clinical = False
        if spec is not None:
            mod = getattr(spec, "molecular_modality", "")
            modality_rna = mod in ("bulk_rna", "scrna")
            modality_clinical = getattr(spec, "treatment_available", False)

        # Treatment context from the pair
        treatment_line = 1  # default; pairs span at least 1L
        drug_start: Optional[float] = None
        drug_end: Optional[float] = None
        if pair.endpoint and pair.endpoint.drug_context:
            drug_start = 0.0
            drug_end = pair.delta_t_days if pair.delta_t_days > 0 else None

        # Endpoint fields
        event_type: Optional[str] = None
        event_time: Optional[float] = None
        event_observed: Optional[bool] = None
        resistance_label: Optional[int] = None
        if pair.endpoint:
            event_type = pair.endpoint.endpoint_type
            event_time = pair.endpoint.event_time_days
            event_observed = pair.endpoint.event_observed
            # Resistance label: 1 if event observed, 0 if censored, None if no endpoint
            resistance_label = int(pair.endpoint.event_observed) if pair.endpoint.event_observed is not None else None

        # Baseline row (visit_time_days = 0)
        rows.append(LongitudinalPanelRow(
            patient_id_hash=pid_hash,
            cohort=cohort,
            visit_id=pair.x_t_sample_id,
            visit_time_days=0.0,
            treatment_line=treatment_line,
            drug_exposure_start=drug_start,
            drug_exposure_end=drug_end,
            modality_available_rna=modality_rna,
            modality_available_atac=modality_atac,
            modality_available_methylation=modality_methylation,
            modality_available_proteomics=modality_proteomics,
            modality_available_phospho=modality_phospho,
            modality_available_clinical=modality_clinical,
            event_type=event_type,
            event_time_days=event_time,
            event_observed=event_observed,
            resistance_state_label=resistance_label,
            baseline_snapshot_uri=pair.x_t_sample_id,
            followup_snapshot_uri=None,
        ))

        # Follow-up row
        rows.append(LongitudinalPanelRow(
            patient_id_hash=pid_hash,
            cohort=cohort,
            visit_id=pair.x_future_sample_id,
            visit_time_days=pair.delta_t_days,
            treatment_line=treatment_line,
            drug_exposure_start=drug_start,
            drug_exposure_end=drug_end,
            modality_available_rna=modality_rna,
            modality_available_atac=modality_atac,
            modality_available_methylation=modality_methylation,
            modality_available_proteomics=modality_proteomics,
            modality_available_phospho=modality_phospho,
            modality_available_clinical=modality_clinical,
            event_type=event_type,
            event_time_days=event_time,
            event_observed=event_observed,
            resistance_state_label=resistance_label,
            baseline_snapshot_uri=None,
            followup_snapshot_uri=pair.x_future_sample_id,
        ))

    logger.info("Built %d LongitudinalPanelRow records from %d pairs", len(rows), len(pairs))
    return rows


def panel_to_parquet(rows: List[LongitudinalPanelRow], out_path: Path) -> Path:
    """Serialise a panel to Parquet via ``pandas``.

    Returns the resolved output path.
    """
    import pandas as pd  # local import — pandas is optional at import-time

    records = [asdict(r) for r in rows]
    df = pd.DataFrame(records)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Wrote panel artifact (%d rows) to %s", len(df), out_path)
    return out_path


def panel_from_parquet(path: Path) -> List[LongitudinalPanelRow]:
    """Deserialise a panel artifact from Parquet."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Panel artifact not found at {path}")
    df = pd.read_parquet(path, engine="pyarrow")

    # Replace NaN with None for Optional fields
    df = df.where(df.notna(), other=None)

    rows: List[LongitudinalPanelRow] = []
    for _, record in df.iterrows():
        row = LongitudinalPanelRow(
            patient_id_hash=str(record["patient_id_hash"]),
            cohort=str(record["cohort"]),
            visit_id=str(record["visit_id"]),
            visit_time_days=float(record["visit_time_days"]),
            treatment_line=int(record["treatment_line"]),
            drug_exposure_start=float(record["drug_exposure_start"]) if record["drug_exposure_start"] is not None else None,
            drug_exposure_end=float(record["drug_exposure_end"]) if record["drug_exposure_end"] is not None else None,
            modality_available_rna=bool(record["modality_available_rna"]),
            modality_available_atac=bool(record["modality_available_atac"]),
            modality_available_methylation=bool(record["modality_available_methylation"]),
            modality_available_proteomics=bool(record["modality_available_proteomics"]),
            modality_available_phospho=bool(record["modality_available_phospho"]),
            modality_available_clinical=bool(record["modality_available_clinical"]),
            event_type=str(record["event_type"]) if record["event_type"] is not None else None,
            event_time_days=float(record["event_time_days"]) if record["event_time_days"] is not None else None,
            event_observed=bool(record["event_observed"]) if record["event_observed"] is not None else None,
            resistance_state_label=int(record["resistance_state_label"]) if record["resistance_state_label"] is not None else None,
            baseline_snapshot_uri=str(record["baseline_snapshot_uri"]) if record["baseline_snapshot_uri"] is not None else None,
            followup_snapshot_uri=str(record["followup_snapshot_uri"]) if record["followup_snapshot_uri"] is not None else None,
        )
        rows.append(row)
    logger.info("Loaded %d LongitudinalPanelRow records from %s", len(rows), path)
    return rows


def validate_panel(rows: List[LongitudinalPanelRow]) -> Dict[str, Any]:
    """Run structural integrity checks on a longitudinal panel.

    Returns a dict with:
      - ``valid``: bool — True if zero errors found.
      - ``n_rows``: int
      - ``errors``: list of human-readable error strings (capped at 50).
      - ``warnings``: list of human-readable warnings (capped at 50).

    Checks performed
    ----------------
    1. **No duplicate visit_ids** — each visit_id must be unique within
       the panel.
    2. **visit_time_days >= 0 for all rows, > 0 for follow-ups** — a row
       whose ``followup_snapshot_uri`` is set must have positive time.
    3. **event_time_days >= visit_time_days** — an event cannot precede the
       visit at which it is recorded.
    4. **No pseudotime-as-calendar** — visit_time_days must not be one of
       the small-integer pseudotime sentinels (1-4) for cohorts known to
       carry pseudotime labels.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Duplicate visit_ids
    seen_visit_ids: Dict[str, int] = {}
    for i, row in enumerate(rows):
        if row.visit_id in seen_visit_ids:
            errors.append(
                f"row {i}: duplicate visit_id '{row.visit_id}' "
                f"(first seen at row {seen_visit_ids[row.visit_id]})"
            )
        else:
            seen_visit_ids[row.visit_id] = i

    # 2. visit_time_days constraints
    for i, row in enumerate(rows):
        if row.visit_time_days < 0:
            errors.append(
                f"row {i}: visit_time_days={row.visit_time_days} is negative"
            )
        if row.followup_snapshot_uri is not None and row.visit_time_days <= 0:
            errors.append(
                f"row {i}: follow-up visit has visit_time_days="
                f"{row.visit_time_days} (must be > 0)"
            )

    # 3. event_time_days >= visit_time_days
    for i, row in enumerate(rows):
        if row.event_time_days is not None and row.visit_time_days > 0:
            if row.event_time_days < row.visit_time_days:
                errors.append(
                    f"row {i}: event_time_days={row.event_time_days} < "
                    f"visit_time_days={row.visit_time_days}"
                )

    # 4. Pseudotime-as-calendar guard
    for i, row in enumerate(rows):
        if (
            row.visit_time_days in _PSEUDOTIME_SENTINELS
            and row.followup_snapshot_uri is not None
        ):
            warnings.append(
                f"row {i}: visit_time_days={row.visit_time_days} matches a "
                f"pseudotime sentinel (1-4) — verify this is real calendar "
                f"time, not a stage ordinal (HD/MGUS/SMM/MM)"
            )

    return {
        "valid": len(errors) == 0,
        "n_rows": len(rows),
        "errors": errors[:50],
        "warnings": warnings[:50],
    }
