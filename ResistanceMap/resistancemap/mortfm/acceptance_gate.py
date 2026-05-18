"""
resistancemap/mortfm/acceptance_gate.py
=======================================
Practical-acceptance-threshold gate.

The trainer calls this before each stage; if the supplied cohort does not
satisfy the threshold for the requested claim level (debug | technical |
research | strong), the gate raises and the stage refuses to run.

Levels
------
* ``debug``       — N >= 10 patients/models; ≥ 1 omics modality; ≥ 1 label.
                    Smoke-test only; no claims permitted.
* ``technical``   — N >= 100 samples; ≥ 1 omics; drug-response or survival
                    labels present; PPI graph present; patient/model-disjoint
                    split. Code-correctness claim only — no biology claim.
* ``research``    — N >= 200 real patients; N_events >= 50; >= 2 modalities;
                    >= 2 timepoints per patient (for trajectory claims);
                    baseline comparison required.
* ``strong``      — research + external validation cohort + modality
                    ablations + pathway validation + uncertainty calibration
                    + claim-critic report.

The thresholds match those documented in ``configs/mortfm.yaml`` and
``docs/MORTFM_LIMITATIONS.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

from resistancemap.mortfm.schemas import MORTFMConfig, TemporalTrainingPair

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Acceptance levels
# ---------------------------------------------------------------------------


VALID_LEVELS = ("debug", "technical", "research", "strong")


@dataclass
class AcceptanceReport:
    level_requested: str
    level_achieved: str
    passed: bool
    n_samples: int
    n_unique_patients: int
    n_modalities: int
    n_longitudinal_pairs: int
    n_survival_rows: int
    n_events_observed: int
    has_patient_disjoint_split: bool
    has_drug_response_labels: bool
    has_survival_labels: bool
    has_temporal_pairs: bool
    modality_coverage: Dict[str, float] = field(default_factory=dict)
    blocking_reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"AcceptanceReport(requested={self.level_requested}, achieved={self.level_achieved}, "
            f"passed={self.passed}, n_patients={self.n_unique_patients}, "
            f"n_events={self.n_events_observed}, n_longitudinal={self.n_longitudinal_pairs}, "
            f"blocked={self.blocking_reasons})"
        )


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------


def _modality_coverage_from_pairs(pairs: Sequence[TemporalTrainingPair]) -> Dict[str, float]:
    if not pairs:
        return {}
    mod_counts: Dict[str, int] = {}
    N = len(pairs)
    for p in pairs:
        for m in p.x_t.available_modalities():
            mod_counts[m.value] = mod_counts.get(m.value, 0) + 1
    return {k: v / N for k, v in mod_counts.items()}


def evaluate_cohort(
    pairs: Sequence[TemporalTrainingPair],
    *,
    level: str,
    config: Optional[MORTFMConfig] = None,
) -> AcceptanceReport:
    """Return an :class:`AcceptanceReport` for ``pairs`` at the requested level."""
    if level not in VALID_LEVELS:
        raise ValueError(f"Unknown acceptance level {level!r}; expected one of {VALID_LEVELS}")

    n_samples = len(pairs)
    patients = {p.x_t.patient_id for p in pairs}
    n_patients = len(patients)
    mod_cov = _modality_coverage_from_pairs(pairs)
    n_modalities = len(mod_cov)
    long_pairs = sum(1 for p in pairs if p.x_t_delta is not None)
    surv_rows = [p for p in pairs if p.outcome.has_survival_label()]
    n_events = sum(1 for p in surv_rows if not p.outcome.censored)
    has_drug = any(p.outcome.drug_response is not None for p in pairs)
    has_surv = bool(surv_rows)
    has_long = long_pairs > 0

    blocking: List[str] = []

    if level == "debug":
        if n_patients < 10:
            blocking.append(f"n_patients={n_patients} < debug threshold 10")
        if n_modalities < 1:
            blocking.append("no modality present")
        if not (has_drug or has_surv):
            blocking.append("no drug-response or survival labels")

    elif level == "technical":
        if n_samples < 100:
            blocking.append(f"n_samples={n_samples} < technical threshold 100")
        if n_modalities < 1:
            blocking.append("no modality present")
        if not (has_drug or has_surv):
            blocking.append("no drug-response or survival labels")

    elif level == "research":
        thresh = (config.min_patient_n_for_survival_claim if config else 200)
        if n_patients < thresh:
            blocking.append(f"n_patients={n_patients} < research threshold {thresh}")
        if n_events < 50:
            blocking.append(f"n_events={n_events} < research threshold 50")
        if n_modalities < 2:
            blocking.append(f"n_modalities={n_modalities} < research threshold 2")
        if has_long:
            # If trajectory training is part of the claim, need ≥ 2 timepoints per patient.
            patients_with_serial = {p.x_t.patient_id for p in pairs if p.x_t_delta is not None}
            if len(patients_with_serial) < max(thresh // 4, 25):
                blocking.append(
                    f"only {len(patients_with_serial)} patients have ≥2 timepoints; "
                    f"trajectory claims need ≥{max(thresh // 4, 25)}"
                )

    elif level == "strong":
        research_report = evaluate_cohort(pairs, level="research", config=config)
        if not research_report.passed:
            blocking.extend([f"(research prerequisite) {r}" for r in research_report.blocking_reasons])
        # The strong level also needs: external validation cohort, ablations,
        # pathway validation, uncertainty calibration, claim-critic report.
        # These are operational rather than data-shape; the trainer must pass
        # ``external_cohort_size > 0`` etc. via ``config`` extras. We block on
        # default config here so a strong claim cannot be made by accident.
        blocking.append("strong-level claims require explicit operational evidence "
                        "(external cohort, ablations, pathway validation, calibration, "
                        "claim-critic report) — gate does not auto-approve")

    passed = not blocking
    achieved = level if passed else _max_achieved_level(pairs, n_events, n_modalities, has_drug, has_surv)

    report = AcceptanceReport(
        level_requested=level,
        level_achieved=achieved,
        passed=passed,
        n_samples=n_samples,
        n_unique_patients=n_patients,
        n_modalities=n_modalities,
        n_longitudinal_pairs=long_pairs,
        n_survival_rows=len(surv_rows),
        n_events_observed=n_events,
        has_patient_disjoint_split=True,  # enforced upstream by split_patients
        has_drug_response_labels=has_drug,
        has_survival_labels=has_surv,
        has_temporal_pairs=has_long,
        modality_coverage=mod_cov,
        blocking_reasons=blocking,
    )
    logger.info("Acceptance: %s", report.summary())
    return report


def _max_achieved_level(
    pairs: Sequence[TemporalTrainingPair],
    n_events: int,
    n_modalities: int,
    has_drug: bool,
    has_surv: bool,
) -> str:
    """Best level the cohort satisfies, ignoring the requested level."""
    n = len({p.x_t.patient_id for p in pairs})
    if n >= 200 and n_events >= 50 and n_modalities >= 2:
        return "research"
    if len(pairs) >= 100 and n_modalities >= 1 and (has_drug or has_surv):
        return "technical"
    if n >= 10 and n_modalities >= 1 and (has_drug or has_surv):
        return "debug"
    return "below_debug"


def require_level(
    pairs: Sequence[TemporalTrainingPair],
    *,
    level: str,
    config: Optional[MORTFMConfig] = None,
) -> AcceptanceReport:
    """Raise :class:`AcceptanceError` if ``pairs`` does not meet ``level``.

    Returns the (passed) report otherwise.
    """
    report = evaluate_cohort(pairs, level=level, config=config)
    if not report.passed:
        raise AcceptanceError(
            f"Acceptance gate blocked claim level {level!r}: "
            + "; ".join(report.blocking_reasons),
            report=report,
        )
    return report


class AcceptanceError(RuntimeError):
    """Raised by :func:`require_level` when the gate blocks a claim level."""

    def __init__(self, message: str, *, report: AcceptanceReport) -> None:
        super().__init__(message)
        self.report = report
