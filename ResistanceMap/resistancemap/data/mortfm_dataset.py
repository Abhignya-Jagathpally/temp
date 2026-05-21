"""
resistancemap/data/mortfm_dataset.py
====================================
PyTorch ``Dataset`` + collate function for MORT-FM training.

The dataset wraps a list of :class:`~resistancemap.mortfm.schemas.TemporalTrainingPair`
objects produced by :mod:`resistancemap.data.trajectory_pair_builder` (or
loaded from a pickled cache produced by the modality loaders).

The collate function is the only place where per-row, per-modality tensors
are stacked into the canonical :class:`~resistancemap.mortfm.schemas.MORTBatch`.
It is responsible for:

* materialising ``modality_mask`` from each pair's available modalities,
* zero-padding missing rows for present modalities (with the mask set False so
  the model does not learn from the zeros — see
  :class:`~resistancemap.models.foundation_fusion.MultiOmicFoundationFusion`),
* stacking survival labels / future states / drug responses with consistent
  shape,
* validating that all stacked tensors share the same batch dimension.

The collate function NEVER fabricates labels. If a row has no event_time, the
corresponding entry in ``event_time`` is NaN and ``event_observed`` is False;
losses must consume these correctly (the survival losses do).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from resistancemap.mortfm.schemas import (
    MODALITY_ORDER,
    DrugContext,
    MORTBatch,
    ModalityName,
    ModalityTensor,
    PatientCellSnapshot,
    TemporalTrainingPair,
)


class MORTFMDataset(Dataset[TemporalTrainingPair]):
    """A trivially indexable wrapper around a list of training pairs.

    Stays as a list of ``TemporalTrainingPair`` objects (not pre-stacked
    tensors) because (a) different rows have different modality presence
    and (b) early sprints will iterate on the schema, so paying the cost of
    a re-pack every epoch is the right trade for now.
    """

    def __init__(self, pairs: Sequence[TemporalTrainingPair]) -> None:
        self.pairs: List[TemporalTrainingPair] = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> TemporalTrainingPair:
        return self.pairs[idx]


def _stack_modality(
    snapshots: Sequence[PatientCellSnapshot],
    modality: ModalityName,
) -> Optional[tuple[Tensor, Tensor]]:
    """Return (stacked_values, presence_mask) for one modality across a batch.

    ``stacked_values`` has shape ``(N, D)`` where ``D`` is the feature dim of
    the first snapshot that has this modality. Missing rows are zero-filled.
    ``presence_mask`` is a ``(N,)`` bool tensor where True means observed.

    Returns ``None`` if no row in the batch has this modality at all (caller
    should leave the field as None in :class:`MORTBatch`).
    """
    if modality == ModalityName.DRUG:
        return None  # handled by _stack_drug
    first_present: Optional[ModalityTensor] = None
    for s in snapshots:
        t = getattr(s, modality.value, None)
        if t is not None:
            first_present = t
            break
    if first_present is None:
        return None
    D = first_present.feature_dim
    N = len(snapshots)
    values = torch.zeros(N, D, dtype=first_present.values.dtype)
    presence = torch.zeros(N, dtype=torch.bool)
    for i, s in enumerate(snapshots):
        t = getattr(s, modality.value, None)
        if t is None:
            continue
        if t.feature_dim != D:
            raise ValueError(
                f"Inconsistent feature dim for modality {modality.value!r}: "
                f"row 0 has {D}, row {i} (patient={s.patient_id}) has {t.feature_dim}. "
                f"All snapshots in a batch must share a harmonised feature space."
            )
        if t.values.ndim == 3:
            # time-resolved modality — collapse the time axis (mean) for the
            # baseline encoder. The trajectory module re-introduces time.
            values[i] = t.values.mean(dim=1).squeeze(0)
        else:
            values[i] = t.values[0] if t.values.shape[0] == 1 else t.values.mean(dim=0)
        presence[i] = True
    return values, presence


def _stack_drug(snapshots: Sequence[PatientCellSnapshot]) -> Optional[tuple[Tensor, Tensor]]:
    """Pack drug contexts into a tensor of shape ``(N, d_drug)``.

    Convention: ``d_drug = 1 (dose) + 1 (start_time) + 1 (end_time)``. The
    drug *encoder* is responsible for expanding identity / class / target
    into richer embeddings from the raw ``DrugContext``. This collate function
    only carries the scalar fields needed by the encoder's lookup table.
    """
    N = len(snapshots)
    has_any = any(s.drug is not None for s in snapshots)
    if not has_any:
        return None
    out = torch.zeros(N, 3, dtype=torch.float32)
    presence = torch.zeros(N, dtype=torch.bool)
    for i, s in enumerate(snapshots):
        if s.drug is None:
            continue
        out[i, 0] = s.drug.dose if s.drug.dose is not None else float("nan")
        out[i, 1] = s.drug.start_time if s.drug.start_time is not None else float("nan")
        out[i, 2] = s.drug.end_time if s.drug.end_time is not None else float("nan")
        presence[i] = True
    return out, presence


def _stack_optional_tensor(
    tensors: Sequence[Optional[Tensor]],
    fill_shape: Optional[tuple[int, ...]] = None,
) -> Optional[Tensor]:
    """Stack optional tensors with a dtype-appropriate sentinel for missing rows.

    * For float dtypes, fills missing rows with NaN (preserves the convention
      used by survival / drug-response losses, which check ``torch.isfinite``).
    * For integer dtypes, fills with ``-1`` so callers can mask via
      ``targets >= 0`` (matches PyTorch's ``ignore_index=-100`` convention,
      but lets us reserve -100 for class-weight users).
    * For bool, fills with False.

    Returns ``None`` if every entry is ``None``.
    """
    present = [t for t in tensors if t is not None]
    if not present:
        return None
    shape = fill_shape or present[0].shape
    dtype = present[0].dtype
    if dtype.is_floating_point:
        fill_value: float = float("nan")
    elif dtype == torch.bool:
        fill_value = False  # type: ignore[assignment]
    else:
        fill_value = -1  # type: ignore[assignment]
    N = len(tensors)
    out = torch.full((N, *shape), fill_value, dtype=dtype)
    for i, t in enumerate(tensors):
        if t is None:
            continue
        if t.shape != shape:
            raise ValueError(
                f"Inconsistent tensor shape during stacking: expected {shape}, got {tuple(t.shape)}"
            )
        out[i] = t
    return out


def _collate_modalities(
    snapshots: Sequence[PatientCellSnapshot],
    patient_ids: Sequence[str],
    cell_ids: Sequence[str],
) -> MORTBatch:
    """Pack a list of snapshots into a partial MORTBatch carrying ONLY the
    per-modality input tensors + presence mask + provenance.

    v19 Phase 1: extracted from the body of :func:`mort_collate` so the
    follow-up snapshots in ``pair.x_t_delta`` can be collated through the
    same code path — guaranteeing the baseline and follow-up batches have
    identical modality layout and so can both be encoded with
    :meth:`MORTFM.forward_foundation`. No supervision fields are filled in
    here; the caller is responsible for attaching event_time / future_state /
    etc. to the *baseline* batch only.
    """
    out = MORTBatch(
        patient_ids=list(patient_ids),
        cell_ids=list(cell_ids),
        modality_mask={},
    )
    for m in MODALITY_ORDER:
        if m == ModalityName.DRUG:
            packed = _stack_drug(snapshots)
        else:
            packed = _stack_modality(snapshots, m)
        if packed is None:
            continue
        values, presence = packed
        setattr(out, m.value, values)
        out.modality_mask[m.value] = presence
    times = [s.timepoint for s in snapshots]
    if any(t is not None for t in times):
        out.time = torch.tensor(
            [t if t is not None else float("nan") for t in times],
            dtype=torch.float32,
        )
    return out


def mort_collate(batch: List[TemporalTrainingPair]) -> MORTBatch:
    """Collate a list of pairs into a :class:`MORTBatch`.

    The trainer wires this in via ``DataLoader(..., collate_fn=mort_collate)``.
    """
    if not batch:
        raise ValueError("mort_collate received an empty batch.")

    snapshots = [p.x_t for p in batch]
    patient_ids = [s.patient_id for s in snapshots]
    cell_ids = [s.cell_id or "" for s in snapshots]

    # Stack baseline modalities + mask + time.
    out = _collate_modalities(snapshots, patient_ids, cell_ids)

    # Supervision targets.
    outcomes = [p.outcome for p in batch]

    event_times = [
        torch.tensor(o.event_time, dtype=torch.float32) if o.event_time is not None else None
        for o in outcomes
    ]
    out.event_time = _stack_optional_tensor(event_times, fill_shape=())
    if any(t is not None for t in event_times):
        out.event_observed = torch.tensor(
            [0.0 if o.censored else 1.0 for o in outcomes],
            dtype=torch.float32,
        )

    state_labels = [
        torch.tensor(o.resistance_label, dtype=torch.long) if o.resistance_label is not None else None
        for o in outcomes
    ]
    out.resistance_label = _stack_optional_tensor(state_labels, fill_shape=())

    drug_responses = [o.drug_response for o in outcomes]
    out.drug_response = _stack_optional_tensor(drug_responses)

    # Optional precomputed future-state latent (cell-line / oracle path only).
    # v19 Phase 1: the RAW-RNA fallback that used to live here is DELETED.
    # The trainer no longer truncates raw RNA into the latent slot; instead
    # we attach the entire follow-up snapshot as `future_snapshot_batch` and
    # let the canonical trainer encode it through the same foundation
    # encoder. The legacy `future_state` slot remains for explicitly-provided
    # teacher latents but is NEVER constructed from x_t_delta.
    future_states = [o.future_state for o in outcomes]
    out.future_state = _stack_optional_tensor(future_states)

    # v19 Phase 1: build the follow-up MORTBatch when any pair has one.
    # Length matches the parent batch B; rows without an x_t_delta are
    # padded with the baseline snapshot (carried through so feature_dim is
    # consistent). Per-row delta_t_days is NaN for those rows so the
    # canonical loss masks them out — the trainer never trains on a fake
    # follow-up target.
    any_follow = any(p.x_t_delta is not None for p in batch)
    if any_follow:
        follow_snaps = [
            (p.x_t_delta if p.x_t_delta is not None else p.x_t) for p in batch
        ]
        follow_pids = [p.x_t.patient_id for p in batch]
        follow_cells = [
            ((p.x_t_delta.cell_id if p.x_t_delta is not None else p.x_t.cell_id) or "")
            for p in batch
        ]
        out.future_snapshot_batch = _collate_modalities(
            follow_snaps, follow_pids, follow_cells,
        )
        dts: List[float] = []
        for p in batch:
            if p.x_t_delta is None:
                dts.append(float("nan"))
                continue
            t_now = p.x_t.timepoint
            t_later = p.x_t_delta.timepoint
            if t_now is None or t_later is None:
                dts.append(float("nan"))
            else:
                dts.append(float(t_later - t_now))
        out.delta_t_days = torch.tensor(dts, dtype=torch.float32)

    return out
