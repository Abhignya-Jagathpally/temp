"""
resistancemap/data/trajectory_pair_builder.py
=============================================
Construct :class:`~resistancemap.mortfm.schemas.TemporalTrainingPair` objects
from raw snapshot + outcome streams.

A *temporal pair* binds:

* ``x_t`` : baseline molecular snapshot
* ``x_t_delta`` : an optional follow-up snapshot of the same patient/sample
* ``outcome`` : labels associated with ``x_t`` -> ``x_t_delta``

Why this module exists separately
---------------------------------
The current v14 pipeline treats every cell line as i.i.d. — so it never has to
ask "what is the same patient at a later time?". MORT-FM does. Getting the
pairing wrong (mixing patients, mixing arms of an RCT, treating a baseline +
relapse pair from the *same* patient as independent training rows) is the
single most common source of leakage in patient-trajectory ML. Putting this
construction in one place lets us write leakage tests against it.

Honest behaviour
----------------
* If the input snapshot list is empty, ``build_temporal_pairs`` returns ``[]``
  (not a fabricated pair).
* If a patient has only one snapshot and no survival label, that patient is
  skipped — *not* turned into a "self-pair" with ``x_t == x_t_delta``.
* If ``allowed_time_gaps`` is empty, only same-patient pairs at any positive
  ``delta_t`` are produced.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from resistancemap.mortfm.schemas import (
    PatientCellSnapshot,
    ResistanceOutcome,
    TemporalTrainingPair,
)

logger = logging.getLogger(__name__)


def _group_by_patient(
    snapshots: Iterable[PatientCellSnapshot],
) -> Dict[str, List[PatientCellSnapshot]]:
    grouped: Dict[str, List[PatientCellSnapshot]] = defaultdict(list)
    for s in snapshots:
        grouped[s.patient_id].append(s)
    # Sort each patient's snapshots in time order. Snapshots with no timepoint
    # are sorted last; they are treated as "no temporal supervision available".
    for pid, lst in grouped.items():
        lst.sort(key=lambda s: (s.timepoint is None, s.timepoint or 0.0))
    return grouped


def _index_outcomes_by_patient(
    outcomes: Iterable[ResistanceOutcome],
) -> Dict[str, List[ResistanceOutcome]]:
    grouped: Dict[str, List[ResistanceOutcome]] = defaultdict(list)
    for o in outcomes:
        grouped[o.patient_id].append(o)
    for pid, lst in grouped.items():
        lst.sort(key=lambda o: o.baseline_time)
    return grouped


def _match_outcome(
    patient_id: str,
    baseline_time: float,
    outcomes_by_patient: Dict[str, List[ResistanceOutcome]],
    tol: float = 1.0,
) -> Optional[ResistanceOutcome]:
    """Find the outcome record closest to ``baseline_time`` for this patient.

    ``tol`` is in the same units as ``baseline_time`` (typically months).
    Returns the matching outcome or ``None`` if no outcome lies within ``tol``.
    """
    candidates = outcomes_by_patient.get(patient_id, [])
    best: Optional[ResistanceOutcome] = None
    best_gap = float("inf")
    for o in candidates:
        gap = abs(o.baseline_time - baseline_time)
        if gap < best_gap:
            best = o
            best_gap = gap
    if best is not None and best_gap <= tol:
        return best
    return None


def _make_unlabelled_outcome(snap: PatientCellSnapshot) -> ResistanceOutcome:
    """Construct a stub outcome record carrying no labels.

    Used when ``include_unlabelled=True`` so a snapshot can still feed
    masked-modality / contrastive pre-training even when no clinical record
    exists for it. The trainer respects ``ResistanceOutcome.has_survival_label``
    and friends to route these correctly.
    """
    return ResistanceOutcome(
        patient_id=snap.patient_id,
        baseline_time=snap.timepoint if snap.timepoint is not None else 0.0,
        event_time=None,
        censored=True,
        resistance_label=None,
        future_state=None,
        drug_response=None,
        pathway_labels=None,
    )


def build_temporal_pairs(
    snapshots: Sequence[PatientCellSnapshot],
    outcomes: Sequence[ResistanceOutcome],
    *,
    allowed_time_gaps: Optional[Sequence[float]] = None,
    gap_tolerance: float = 1.0,
    outcome_match_tolerance: float = 1.0,
    include_unlabelled: bool = False,
    include_survival_only: bool = True,
) -> List[TemporalTrainingPair]:
    """Build training pairs from a flat snapshot + outcome stream.

    Parameters
    ----------
    snapshots
        All molecular snapshots, possibly from many patients and timepoints.
        The function groups them by ``patient_id`` and orders by ``timepoint``.
    outcomes
        All clinical outcome records, possibly multiple per patient.
    allowed_time_gaps
        If given, only ``(x_t, x_t_delta)`` pairs with
        ``|timepoint_delta - timepoint_t in allowed_time_gaps| <= gap_tolerance``
        are emitted. ``None`` (the default) accepts any positive gap.
    gap_tolerance
        Slack on the time-gap match in the same units as snapshot timepoints.
    outcome_match_tolerance
        How close (in time) an outcome record must be to ``x_t.timepoint`` to be
        considered the matching outcome. Default 1.0 (months).
    include_unlabelled
        If True, also emit pairs with ``x_t_delta=None`` and a stub outcome,
        for masked-modality pretraining. Default False.
    include_survival_only
        If True, emit pairs with ``x_t_delta=None`` whose outcome carries a
        survival label (event_time + censored). Default True.

    Returns
    -------
    List[TemporalTrainingPair]
        Empty list if no pairs could be constructed. Never fabricates.
    """
    if not snapshots:
        logger.info("build_temporal_pairs: no snapshots provided; returning [].")
        return []

    snapshots_by_patient = _group_by_patient(snapshots)
    outcomes_by_patient = _index_outcomes_by_patient(outcomes)

    pairs: List[TemporalTrainingPair] = []
    skipped_no_followup = 0
    skipped_no_outcome = 0
    skipped_bad_gap = 0

    for patient_id, snaps in snapshots_by_patient.items():
        timed = [s for s in snaps if s.timepoint is not None]
        # --- longitudinal pairs ----------------------------------------------
        for i, x_t in enumerate(timed):
            for x_t_delta in timed[i + 1 :]:
                dt = (x_t_delta.timepoint or 0.0) - (x_t.timepoint or 0.0)
                if dt <= 0:
                    continue
                if allowed_time_gaps is not None:
                    if not any(abs(dt - g) <= gap_tolerance for g in allowed_time_gaps):
                        skipped_bad_gap += 1
                        continue
                out = _match_outcome(
                    patient_id,
                    baseline_time=x_t.timepoint or 0.0,
                    outcomes_by_patient=outcomes_by_patient,
                    tol=outcome_match_tolerance,
                )
                if out is None:
                    if include_unlabelled:
                        out = _make_unlabelled_outcome(x_t)
                    else:
                        skipped_no_outcome += 1
                        continue
                pairs.append(TemporalTrainingPair(x_t=x_t, x_t_delta=x_t_delta, outcome=out))

        # --- survival-only rows (single snapshot + outcome) ------------------
        if include_survival_only:
            for x_t in snaps:
                out = _match_outcome(
                    patient_id,
                    baseline_time=x_t.timepoint or 0.0,
                    outcomes_by_patient=outcomes_by_patient,
                    tol=outcome_match_tolerance,
                )
                if out is None or not out.has_survival_label():
                    continue
                # Don't double-count pairs that already exist as longitudinal.
                already_used = any(
                    p.x_t is x_t and p.x_t_delta is not None for p in pairs
                )
                if already_used:
                    continue
                pairs.append(TemporalTrainingPair(x_t=x_t, x_t_delta=None, outcome=out))

        # --- unlabelled rows for pretraining ---------------------------------
        if include_unlabelled:
            for x_t in snaps:
                already_used = any(p.x_t is x_t for p in pairs)
                if already_used:
                    continue
                pairs.append(
                    TemporalTrainingPair(
                        x_t=x_t,
                        x_t_delta=None,
                        outcome=_make_unlabelled_outcome(x_t),
                    )
                )

        if len(timed) < 2:
            skipped_no_followup += 1

    logger.info(
        "build_temporal_pairs: emitted %d pairs from %d snapshots / %d outcomes "
        "(skipped: %d no-followup patients, %d no-outcome rows, %d bad-gap rows)",
        len(pairs),
        len(snapshots),
        len(outcomes),
        skipped_no_followup,
        skipped_no_outcome,
        skipped_bad_gap,
    )
    return pairs


def summarise_pairs(pairs: Sequence[TemporalTrainingPair]) -> Dict[str, float]:
    """Diagnostic counts for a list of pairs. Useful as a logging payload."""
    n_total = len(pairs)
    n_longitudinal = sum(1 for p in pairs if p.x_t_delta is not None)
    n_survival = sum(1 for p in pairs if p.outcome.has_survival_label())
    n_state = sum(1 for p in pairs if p.outcome.resistance_label is not None)
    n_drug = sum(1 for p in pairs if p.outcome.drug_response is not None)
    n_unlabelled = sum(
        1 for p in pairs if not p.has_any_supervision()
    )
    patient_set = {p.x_t.patient_id for p in pairs}
    return {
        "n_pairs_total": float(n_total),
        "n_longitudinal_pairs": float(n_longitudinal),
        "n_survival_rows": float(n_survival),
        "n_state_label_rows": float(n_state),
        "n_drug_response_rows": float(n_drug),
        "n_unlabelled_rows": float(n_unlabelled),
        "n_unique_patients": float(len(patient_set)),
    }
