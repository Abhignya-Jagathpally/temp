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

from dataclasses import dataclass, field
from typing import List, Literal, Optional


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
