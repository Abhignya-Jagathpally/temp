# v18 Phase Plan — Phases 1, 6, 7
## Trajectory target, Waddington landscape, survival ↔ hitting-time consistency

Author: trajectory-dynamics PhD review
Branch: `v18-clean-canonical-mortfm`
Scope: canonical MORT-FM only (LENS path in `model.forward`); legacy
`forward_legacy` is out of scope.

This doc replaces three load-bearing shortcuts that the v18.2 strict-supervision
audit revealed in the canonical trajectory pipeline:

1. **Trajectory targets are *raw RNA* projected/truncated into latent space.**
   The MMD loss in stage E is therefore comparing apples (predicted latent
   trajectory) to oranges (mean-pooled, truncated RNA vector). The Wasserstein /
   MMD geometry is meaningless under this mismatch.
2. **Waddington basins are unsupervised quadratic wells with no clinical
   anchor.** `ResistanceBasin` runs a softmax of `-||z-c_k||^2` but no loss
   pulls `c_k` toward any biological state.
3. **Hitting CDF and competing-risk survival curve live on different grids and
   are never jointly constrained.** The SDE rolls out on
   `linspace(0, integration_time=1.0, n_time_grid)` while the survival head
   uses `linspace(0, integration_time, survival_n_bins+1)` (config
   `integration_time=12.0`, `n_time_grid=25`, `survival_n_bins=12`). Different
   units, different counts, no consistency constraint.

The three phases below fix these honestly — no synthetic data, no
hard-coded metrics.

---

## PHASE 1 — Fix MORTBatch / future-state trajectory target

### 1.1 Diagnosis

`resistancemap/data/mortfm_dataset.py:235-246` is the offending block. The
collate function:

```python
future_states = [o.future_state for o in outcomes]
for i, pair in enumerate(batch):
    if future_states[i] is None and pair.x_t_delta is not None:
        tdelta = pair.x_t_delta.rna
        if tdelta is not None:
            future_states[i] = (
                tdelta.values.mean(dim=0) if tdelta.values.ndim == 2 else tdelta.values[0]
            )
out.future_state = _stack_optional_tensor(future_states)
```

So when the outcome lacks an explicit `future_state` tensor, `mort_collate`
fills `MORTBatch.future_state` with the **raw RNA mean** of the follow-up
snapshot. This raw vector has shape `(D_rna,)` (typically 5000), while the SDE
predicts in `R^{d_latent}` (default 128 in `MORTFMConfig.d_latent`, 64 in the
LENS SDE constructor).

The trainer then performs **truncation** at
`resistancemap/mortfm/trainer.py:287-296`:

```python
if "trajectory" in weights and batch.future_state is not None:
    pred_T = prediction.trajectory_mean[-1] if prediction.trajectory_mean is not None else prediction.z_path[-1]
    target = batch.future_state
    if target.shape[-1] != pred_T.shape[-1]:
        # Project target via a learned linear (unsupervised);
        # placeholder: use mean of the modality's first d_latent dims.
        target = target[..., : pred_T.shape[-1]]
    components["trajectory"] = trajectory_distribution_loss(pred_T, target, metric="mmd")
```

`target = target[..., : pred_T.shape[-1]]` is the bug. Taking the first
`d_latent` genes of a 5000-gene RNA vector as the "true future latent" is not
a projection — it is an arbitrary, basis-dependent slice. MMD between the
SDE-evolved latent and an alphabetically-first-128-gene RNA vector is
scientifically vacuous; it can only be minimised by the trainer learning to
output a tensor whose values numerically match the first 128 gene counts,
which is opposite to the geometric inductive bias of a learned latent.
Furthermore, `trainer.py:289` reads `prediction.trajectory_mean[-1]`, indexing
into the time axis with `-1`, but `TrajectoryDistributionHead` returns a tensor
of shape `(B, T, d_out)` — `[-1]` returns the **last sample in the batch**, not
the last timepoint. Both bugs cooperate to silently produce a loss that is
syntactically valid and semantically meaningless.

### 1.2 Proposed schema change

Add a strongly typed `future_snapshot_batch` so the canonical trainer can
re-encode the follow-up snapshot through the *same* foundation encoder used
for the baseline. The canonical loss compares two latents produced by the
same encoder — the only comparison that is meaningful for an SDE in latent
space.

```python
# resistancemap/mortfm/schemas.py — addition to MORTBatch
@dataclass
class MORTBatch:
    ...
    # NEW (Phase 1): a fully-formed follow-up batch with the same modality
    # layout as `self`. Populated by mort_collate when at least one row has
    # x_t_delta != None; otherwise None. NEVER constructed from raw RNA.
    future_snapshot_batch: Optional["MORTBatch"] = None
    # NEW (Phase 1): per-row Δt in months. NaN for rows without a follow-up.
    delta_t: Optional[Tensor] = None
    # NEW (Phase 1): per-row presence mask for the follow-up.
    has_future_snapshot: Optional[Tensor] = None  # (N,) bool
    # NEW (Phase 1): drug exposure window between t and t+Δt.
    treatment_window: Optional[Tensor] = None  # (N, 2) float: [start, end]
```

`future_snapshot_batch` is itself a `MORTBatch` because (a) follow-up
snapshots are partially observed (some modalities drop out at relapse) and we
already have the encoder router to handle modality masks, and (b) it lets us
encode the follow-up through `model.forward_foundation(...)` with the same
`modality_dropout_p` and the same fusion weights.

The existing `future_state: Optional[Tensor]` field stays for backwards
compatibility with the cell-line / oracle path that legitimately *does*
provide a precomputed latent (e.g. a teacher network), but the trainer is
re-wired to prefer `future_snapshot_batch` whenever it is present.

### 1.3 Collate change

```python
# resistancemap/data/mortfm_dataset.py — replace lines 235-246
future_pairs = [p.x_t_delta for p in batch]
has_future = torch.tensor([p is not None for p in future_pairs], dtype=torch.bool)
if has_future.any():
    # Build a parallel MORTBatch using a zero-pad snapshot for rows without
    # x_t_delta. The model never reads those rows (gated by has_future_snapshot).
    placeholder_idx = (~has_future).nonzero(as_tuple=False).flatten().tolist()
    snapshot_for_collate = []
    for i, p in enumerate(batch):
        if p.x_t_delta is not None:
            snapshot_for_collate.append(p.x_t_delta)
        else:
            # Carry the baseline snapshot through so feature_dim is consistent;
            # masked out by has_future_snapshot at loss time.
            snapshot_for_collate.append(p.x_t)
    future_batch = _collate_snapshots_only(snapshot_for_collate)  # no labels
    out.future_snapshot_batch = future_batch
    out.has_future_snapshot = has_future
    # Δt
    dts = []
    for p in batch:
        if p.x_t_delta is not None and p.x_t.timepoint is not None and p.x_t_delta.timepoint is not None:
            dts.append(p.x_t_delta.timepoint - p.x_t.timepoint)
        else:
            dts.append(float("nan"))
    out.delta_t = torch.tensor(dts, dtype=torch.float32)
    # Treatment window (start, end) of the drug between t and t+Δt.
    win = torch.full((len(batch), 2), float("nan"), dtype=torch.float32)
    for i, p in enumerate(batch):
        drug = p.x_t.drug
        if drug is not None and drug.start_time is not None and drug.end_time is not None:
            win[i, 0] = drug.start_time
            win[i, 1] = drug.end_time
    out.treatment_window = win

# The legacy raw-RNA fill-in is DELETED. If `outcome.future_state` is
# explicitly provided, we still expose it via `out.future_state` for backward
# compatibility, but we never synthesise it from x_t_delta.future_state = [o.future_state for o in outcomes]
out.future_state = _stack_optional_tensor([o.future_state for o in outcomes])
```

A new helper `_collate_snapshots_only(snaps) -> MORTBatch` is needed; it is
literally the modality-stacking half of `mort_collate` factored out (the
label-stacking half is skipped for the follow-up).

### 1.4 Canonical trainer step

A new file isolates the trajectory step so that the existing `trainer.py` can
be retired in v19. The skeleton lives in `resistancemap/mortfm/canonical_trainer.py`:

```python
# resistancemap/mortfm/canonical_trainer.py
import torch
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import MORTBatch
from resistancemap.training.canonical_mortfm_losses import (
    latent_future_state_loss,
    sde_path_integral_loss,
)


def _trajectory_step(
    model: MORTFM,
    batch: MORTBatch,
    *,
    integration_time: float,
) -> dict:
    """One canonical stage-E gradient step.

    Returns a dict of named loss tensors. Caller weights and sums.
    """
    if batch.future_snapshot_batch is None or batch.has_future_snapshot is None:
        return {}  # nothing to supervise this batch; strict policy will count

    mask = batch.has_future_snapshot
    if mask.sum() == 0:
        return {}

    # 1. Encode baseline + follow-up through the SAME foundation encoder.
    state0 = model.forward_foundation(batch)
    state1 = model.forward_foundation(batch.future_snapshot_batch)
    # IMPORTANT: detach the follow-up latent so we train the SDE/drift to MATCH
    # the encoder's view of the future, not move the encoder to be predictable.
    # This is the standard target-network trick in latent-dynamics models.
    z0 = state0.z0
    z1_target = state1.z0.detach()

    # 2. Roll out the SDE from z0.
    out = model(batch)              # canonical forward; uses state0 implicitly
    z_traj = out["z_traj"]          # (B, T, d_latent)
    t_grid = out["t_grid"]          # (T,) in MODEL units

    # 3. Map each row's Δt (months) to the nearest index in t_grid.
    delta_t = batch.delta_t.to(z_traj.device)
    # Normalise Δt into the same unit as t_grid (see Phase 7 below).
    delta_t_normalised = delta_t / integration_time
    t_idx = (delta_t_normalised.unsqueeze(-1) - t_grid.unsqueeze(0)).abs().argmin(dim=-1)
    # Gather per-row predicted latent at the matched horizon.
    batch_idx = torch.arange(z_traj.shape[0], device=z_traj.device)
    z_pred_at_dt = z_traj[batch_idx, t_idx]  # (B, d_latent)

    # 4. Per-row latent-target loss (only where has_future_snapshot==True).
    loss_dict = latent_future_state_loss(
        predicted_z_t=z_pred_at_dt,
        encoded_z_future=z1_target,
        delta_t=delta_t,
        row_mask=mask,
    )
    # 5. Composition with SDE path-integral (Phase 6 landscape regularisation).
    loss_dict.update(
        sde_path_integral_loss(
            z_traj=z_traj,
            potential=model.lens_sde.potential,
            drug=out.get("drug_token"),
        )
    )
    return loss_dict
```

The trainer wrapper iterates batches, calls `_trajectory_step`, sums weighted
components, and feeds the strict-supervision audit (the supervised-batch
counter increments only when `loss_dict["latent_future_state"]` is present and
finite).

### 1.5 New loss function

```python
# resistancemap/training/canonical_mortfm_losses.py
from typing import Optional
import torch
import torch.nn.functional as F


def latent_future_state_loss(
    predicted_z_t: torch.Tensor,       # (B, d_latent) SDE output at Δt
    encoded_z_future: torch.Tensor,    # (B, d_latent) encoder(x_{t+Δt}).z0
    delta_t: torch.Tensor,             # (B,) months
    *,
    row_mask: Optional[torch.Tensor] = None,  # (B,) bool, True = supervised
    use_distribution_loss: bool = True,
    mmd_sigma: float = 1.0,
    eps: float = 1e-6,
) -> dict:
    """Latent-vs-latent future-state loss.

    Two components, returned in a dict so the trainer can log both:
      - 'latent_future_state_l2'  : per-row L2 between predicted and target
                                    latent, normalised by Δt to stop short-Δt
                                    rows from dominating the gradient.
      - 'latent_future_state_mmd' : distribution-level MMD across the
                                    *supervised subset of the batch*. This
                                    captures population-level drift even when
                                    individual rows have noisy latents.

    Composition:
      total_latent = lambda_l2 * l2 + lambda_mmd * mmd
    The caller picks weights; both terms operate on the *same encoder's*
    latent space, so they are dimensionally consistent — unlike the v17
    raw-RNA-truncation loss.
    """
    if predicted_z_t.shape != encoded_z_future.shape:
        raise ValueError(
            f"predicted_z_t {tuple(predicted_z_t.shape)} != "
            f"encoded_z_future {tuple(encoded_z_future.shape)} — "
            f"these MUST be the same encoder's output."
        )
    if row_mask is None:
        row_mask = torch.ones(predicted_z_t.shape[0], dtype=torch.bool,
                              device=predicted_z_t.device)
    if row_mask.sum() == 0:
        zero = torch.zeros((), device=predicted_z_t.device, dtype=predicted_z_t.dtype)
        return {"latent_future_state_l2": zero, "latent_future_state_mmd": zero}
    pred_m = predicted_z_t[row_mask]
    targ_m = encoded_z_future[row_mask]
    dt_m = delta_t[row_mask].clamp(min=eps)  # months

    # Per-row L2, normalised by Δt so 1-month and 12-month errors are commensurate.
    l2_per_row = (pred_m - targ_m).pow(2).sum(dim=-1)
    l2 = (l2_per_row / dt_m).mean()

    out = {"latent_future_state_l2": l2}
    if use_distribution_loss and pred_m.shape[0] >= 2:
        out["latent_future_state_mmd"] = _mmd_rbf(pred_m, targ_m, sigma=mmd_sigma)
    else:
        out["latent_future_state_mmd"] = torch.zeros_like(l2)
    return out


def sde_path_integral_loss(
    z_traj: torch.Tensor,                # (B, T, d_latent)
    potential,                           # WaddingtonPotential
    drug: Optional[torch.Tensor] = None,
    *,
    weight_potential: float = 1.0,
) -> dict:
    """Energy along the SDE path — encourages trajectories to descend U.

    Mean potential evaluated along the predicted path. This is NOT the
    physical action (we'd need ||dz/dt + grad U||^2); it is the cheap
    proxy that says "your rollout should pass through low-energy regions."
    """
    B, T, d = z_traj.shape
    flat = z_traj.reshape(B * T, d)
    if drug is not None:
        drug_flat = drug.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        U = potential(flat, drug=drug_flat)
    else:
        U = potential(flat)
    return {"sde_path_potential": weight_potential * U.mean()}


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, *, sigma: float = 1.0) -> torch.Tensor:
    def k(a, b):
        d2 = (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(dim=-1)
        return torch.exp(-d2 / (2 * sigma * sigma))
    return k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()
```

### 1.6 Pair-builder extension

`trajectory_pair_builder.build_temporal_pairs` already records the snapshot
timepoints, but it does not surface the realised Δt, treatment exposure
window, or which modalities survived to follow-up. Two changes:

```python
# resistancemap/data/trajectory_pair_builder.py — additions
@dataclass
class TemporalPairProvenance:
    delta_t_months: float
    treatment_window_months: Tuple[float, float]
    follow_up_available_modalities: Tuple[str, ...]
    matched_outcome_gap_months: float


def build_temporal_pairs(... ) -> List[TemporalTrainingPair]:
    ...
    for x_t_delta in timed[i + 1 :]:
        dt = (x_t_delta.timepoint or 0.0) - (x_t.timepoint or 0.0)
        ...
        prov = TemporalPairProvenance(
            delta_t_months=dt,
            treatment_window_months=(
                x_t.drug.start_time if x_t.drug else float("nan"),
                x_t.drug.end_time if x_t.drug else float("nan"),
            ),
            follow_up_available_modalities=tuple(
                m.value for m in x_t_delta.available_modalities()
            ),
            matched_outcome_gap_months=abs(out.baseline_time - (x_t.timepoint or 0.0)),
        )
        pair = TemporalTrainingPair(x_t=x_t, x_t_delta=x_t_delta, outcome=out)
        pair.provenance = prov   # attach (TemporalTrainingPair must add field)
        pairs.append(pair)
```

`TemporalTrainingPair` gains an optional `provenance: Optional[TemporalPairProvenance] = None`
slot. The collate function reads it to populate `delta_t`, `treatment_window`,
and to log a per-batch follow-up modality histogram.

### 1.7 New tests

```python
# tests/mortfm/test_temporal_pair_builder.py — append
def test_provenance_records_delta_t_and_treatment_window():
    snap0 = _snap("P1", 0.0)
    snap0.drug = DrugContext(drug_name="VRd", start_time=0.0, end_time=4.5)
    snap6 = _snap("P1", 6.0)
    outs = [ResistanceOutcome(patient_id="P1", baseline_time=0.0,
                              event_time=10.0, censored=False)]
    pairs = build_temporal_pairs([snap0, snap6], outs,
                                 allowed_time_gaps=[6.0], gap_tolerance=0.1)
    longitudinal = [p for p in pairs if p.x_t_delta is not None]
    assert len(longitudinal) == 1
    p = longitudinal[0]
    assert p.provenance.delta_t_months == 6.0
    assert p.provenance.treatment_window_months == (0.0, 4.5)
    assert "rna" in p.provenance.follow_up_available_modalities


def test_provenance_nan_when_no_drug():
    pairs = build_temporal_pairs(
        [_snap("P1", 0.0), _snap("P1", 6.0)],
        [ResistanceOutcome(patient_id="P1", baseline_time=0.0,
                           event_time=10.0, censored=False)],
        allowed_time_gaps=[6.0], gap_tolerance=0.1,
    )
    long = [p for p in pairs if p.x_t_delta is not None][0]
    import math
    assert math.isnan(long.provenance.treatment_window_months[0])
```

```python
# tests/mortfm/test_canonical_trajectory_training.py — new file
import torch
import pytest
from resistancemap.mortfm.canonical_trainer import _trajectory_step
from resistancemap.training.canonical_mortfm_losses import latent_future_state_loss


def test_latent_future_state_loss_requires_matching_shape():
    p = torch.randn(4, 64)
    t = torch.randn(4, 32)              # WRONG shape on purpose
    with pytest.raises(ValueError, match="MUST be the same encoder"):
        latent_future_state_loss(p, t, delta_t=torch.ones(4))


def test_latent_future_state_loss_zero_when_predicted_equals_target():
    z = torch.randn(8, 64)
    out = latent_future_state_loss(z, z, delta_t=torch.ones(8))
    assert torch.isclose(out["latent_future_state_l2"], torch.zeros(()))
    assert torch.isclose(out["latent_future_state_mmd"], torch.zeros(()), atol=1e-5)


def test_latent_future_state_loss_respects_row_mask():
    z_pred = torch.randn(4, 16)
    z_targ = torch.randn(4, 16)
    mask = torch.tensor([True, False, False, True])
    out = latent_future_state_loss(z_pred, z_targ, delta_t=torch.ones(4), row_mask=mask)
    # Only rows 0 and 3 contribute. Verify by recomputing on the masked subset.
    expected = ((z_pred[mask] - z_targ[mask]).pow(2).sum(dim=-1)).mean()
    assert torch.isclose(out["latent_future_state_l2"], expected, atol=1e-6)


def test_trajectory_step_returns_empty_when_no_followup(simple_mortfm, batch_no_followup):
    out = _trajectory_step(simple_mortfm, batch_no_followup, integration_time=12.0)
    assert out == {}


def test_trajectory_step_emits_named_components(simple_mortfm, batch_with_followup):
    out = _trajectory_step(simple_mortfm, batch_with_followup, integration_time=12.0)
    assert "latent_future_state_l2" in out
    assert "latent_future_state_mmd" in out
    assert "sde_path_potential" in out
    for v in out.values():
        assert torch.isfinite(v), f"{v}"
```

The fixtures (`simple_mortfm`, `batch_with_followup`, `batch_no_followup`)
must construct from a tiny synthetic MORTBatch — synthetic *test fixtures*
are acceptable because the test asserts shape and finiteness, not any
metric value.

### 1.8 Phase 1 per-file change table

| File | Change | Type |
| --- | --- | --- |
| `resistancemap/mortfm/schemas.py` | Add `future_snapshot_batch`, `delta_t`, `has_future_snapshot`, `treatment_window` to `MORTBatch`; add `provenance` to `TemporalTrainingPair`; add `TemporalPairProvenance` dataclass. | edit |
| `resistancemap/data/mortfm_dataset.py` | Replace lines 235-246 raw-RNA fallback with `future_snapshot_batch` construction; factor out `_collate_snapshots_only`. | edit |
| `resistancemap/data/trajectory_pair_builder.py` | Populate `TemporalPairProvenance` for every pair; never invent Δt. | edit |
| `resistancemap/mortfm/canonical_trainer.py` | NEW file; `_trajectory_step` and a thin wrapper around it. | new |
| `resistancemap/training/canonical_mortfm_losses.py` | NEW file; `latent_future_state_loss`, `sde_path_integral_loss`. | new |
| `resistancemap/mortfm/trainer.py` | Delete the truncation block at lines 287-296; route to `_trajectory_step` when `future_snapshot_batch` is present. | edit |
| `tests/mortfm/test_temporal_pair_builder.py` | Append provenance tests. | edit |
| `tests/mortfm/test_canonical_trajectory_training.py` | NEW; shape and mask invariants. | new |

### 1.9 Phase 1 Definition of Done

1. `mort_collate` produces `future_snapshot_batch` whenever ≥1 row has
   `x_t_delta`, and **never** projects raw RNA into the latent slot.
2. `_trajectory_step` re-encodes the follow-up through the same encoder and
   computes loss in latent space only.
3. The strict-supervision audit in `trainer.py` counts a batch as
   "trajectory-supervised" only when `latent_future_state_l2` is computed and
   finite (i.e. `has_future_snapshot.any()`).
4. New tests pass; the truncation `target[..., : pred_T.shape[-1]]` line is
   gone and `grep -R "target\[..., : pred_T" ResistanceMap/` returns empty.
5. Pair-builder logs include `delta_t_months` and `treatment_window_months`
   distributions per epoch.

---

## PHASE 6 — Waddington landscape strengthening

### 6.1 What exists

- `resistancemap/mortfm/trajectory/resistance_basin.py` — soft basin
  assignment over `n_basins=5` quadratic wells, optionally sharing centroids
  with `WaddingtonPotential` in `graph_energy_sde.py`.
- `resistancemap/landscape/landscape_regularizers.py` — already implements
  `smoothness_penalty`, `basin_separation_penalty`, `drug_barrier_penalty`,
  `basin_entropy_penalty`.
- `resistancemap/landscape/potential.py`, `attractor_basin.py`,
  `waddington_visualizer.py` — older v15 landscape utilities (not on the
  canonical path).

What is **missing**:

- Any **supervised** signal on basin centroids. They drift freely.
- A loss that penalises basin assignment *jumping* discontinuously along a
  trajectory (`p_k(z_t)` should be roughly monotone within a basin's well).
- An **attractor stability** loss: `||grad U(c_k)|| ≈ 0` should hold at the
  learned centres.
- 2D/3D visualisation hooks for the *learned* potential on the canonical
  LENS path (`lens_sde.potential`).

### 6.2 Supervised basin labels — where they come from

MM cohorts give us four clinically defensible basin definitions. None require
single-cell resolution; all are derived from the MMRF CoMMpass +
BeatAML-style endpoints we already load:

| Basin index | Name | Operational definition (real-data rules) |
| --- | --- | --- |
| 0 | sensitive | sCR/CR at end-of-induction AND PFS > 36 mo AND no detectable MRD by NGS at any sampled timepoint |
| 1 | persister | VGPR/PR at end-of-induction AND MRD-positive at any post-induction timepoint AND PFS in [12, 36] mo |
| 2 | MRD-like | MRD-positive at any post-induction timepoint AND no clinical progression yet AND PFS still ongoing OR event |
| 3 | resistant | Progression within 12 months of frontline AND/OR refractory by IMWG criteria |

The label builder lives next to the pair builder. It is rule-based, runs on
the *labels already present* in CoMMpass clinical files, never invents:

```python
# resistancemap/data/basin_label_builder.py — new
from dataclasses import dataclass
from typing import Optional

@dataclass
class BasinLabelRule:
    """All thresholds come from IMWG / MMRF data dictionaries.
    NO defaults are heuristic — every threshold cites a clinical guideline.
    """
    mrd_positive_threshold: float = 1e-5          # IMWG MRD threshold
    persister_pfs_lo_months: float = 12.0
    persister_pfs_hi_months: float = 36.0
    sensitive_min_pfs_months: float = 36.0
    resistant_max_pfs_months: float = 12.0


def assign_basin_label(
    outcome,                       # ResistanceOutcome enriched with MRD field
    rule: BasinLabelRule = BasinLabelRule(),
) -> Optional[int]:
    """Return 0/1/2/3 or None when the data are insufficient. Never guesses."""
    if outcome.event_time is None:
        return None
    ...
```

Rows for which the rule returns `None` are excluded from the basin loss
(again: the strict-supervision audit must count basin-labelled rows
separately from trajectory-labelled rows).

### 6.3 New loss skeletons

```python
# resistancemap/training/canonical_mortfm_losses.py — additions
import torch
import torch.nn.functional as F


def basin_label_loss(
    basin_probs: torch.Tensor,         # (B, K) softmax basin probs at z_T
    basin_labels: torch.Tensor,        # (B,) long, -1 for unlabelled
    *,
    label_smoothing: float = 0.05,
) -> dict:
    valid = basin_labels >= 0
    if valid.sum() == 0:
        return {"basin_label": torch.zeros((), device=basin_probs.device,
                                           dtype=basin_probs.dtype)}
    logp = basin_probs[valid].clamp_min(1e-8).log()
    ce = F.nll_loss(logp, basin_labels[valid], label_smoothing=label_smoothing)
    return {"basin_label": ce}


def barrier_regularization_loss(
    potential,                         # WaddingtonPotential
    z_sensitive: torch.Tensor,         # (B0, d_latent) basin-0 samples
    z_resistant: torch.Tensor,         # (B3, d_latent) basin-3 samples
    drug: torch.Tensor,                # (B, d_drug) treatment context
    *,
    n_interp: int = 5,
    target_min_barrier: float = 1.0,
) -> dict:
    """Penalise low barriers on straight-line interpolations from sensitive to
    resistant centroids — *only* in the under-treatment context.

    Math: for alpha in (0, 1) interior of the line z(alpha) = (1-a) z_s + a z_r,
    sup_a U(z(alpha)) - max(U(z_s), U(z_r)) should be >= target_min_barrier.
    A drug that *lowers* barriers (i.e. makes resistant transitions easier)
    contradicts the standard 'selection pressure' hypothesis; we want the
    learned U to encode that.
    """
    if z_sensitive.shape[0] == 0 or z_resistant.shape[0] == 0:
        return {"barrier_reg": torch.zeros((), device=drug.device, dtype=drug.dtype)}
    # Pair samples to compute per-pair interpolations.
    n = min(z_sensitive.shape[0], z_resistant.shape[0])
    zs, zr = z_sensitive[:n], z_resistant[:n]
    alphas = torch.linspace(0.0, 1.0, n_interp + 2, device=zs.device)[1:-1]
    U_endpoints = torch.maximum(potential(zs, drug=drug[:n]), potential(zr, drug=drug[:n]))
    barrier_max = U_endpoints.clone()
    for a in alphas:
        z_a = (1 - a) * zs + a * zr
        U_a = potential(z_a, drug=drug[:n])
        barrier_max = torch.maximum(barrier_max, U_a)
    actual_barrier = barrier_max - U_endpoints
    return {"barrier_reg": (target_min_barrier - actual_barrier).clamp(min=0.0).mean()}


def attractor_stability_loss(
    potential,
    *,
    drug_at_attractor: Optional[torch.Tensor] = None,
) -> dict:
    """||grad_z U(c_k)||^2 averaged over attractors.

    An attractor by definition has zero gradient; if the learned U has nonzero
    gradient at c_k then c_k is not actually a fixed point of the deterministic
    drift in the absence of f_theta.
    """
    centres = potential.basin_centroids        # (K, d_latent)
    K, d = centres.shape
    if drug_at_attractor is None:
        # Use the zero-drug context as the equilibrium reference.
        drug_at_attractor = centres.new_zeros((K, potential.drug_to_basin_weight.in_features))
    centres_ = centres.detach().requires_grad_(True)
    U = potential(centres_, drug_at_attractor).sum()
    grad, = torch.autograd.grad(U, centres_, create_graph=True)
    return {"attractor_stability": grad.pow(2).sum(dim=-1).mean()}


def monotonicity_loss(
    basin_probs_traj: torch.Tensor,    # (B, T, K) basin probs along the SDE
    basin_labels: torch.Tensor,        # (B,) long, terminal basin (or -1)
) -> dict:
    """Penalise non-monotone increase of P[z_t in B_label] over time for
    labelled rows. Captures the prior that the model should not oscillate
    in and out of the eventual basin.
    """
    valid = basin_labels >= 0
    if valid.sum() == 0:
        return {"monotonicity": torch.zeros((), device=basin_probs_traj.device,
                                            dtype=basin_probs_traj.dtype)}
    B = basin_probs_traj.shape[0]
    idx = basin_labels.clamp(min=0)
    p_label = basin_probs_traj[torch.arange(B), :, idx]   # (B, T)
    diffs = p_label[:, 1:] - p_label[:, :-1]              # (B, T-1)
    # Penalise *decreases* of the predicted-basin probability over time.
    penalty = (-diffs).clamp(min=0.0)
    return {"monotonicity": penalty[valid].mean()}
```

These four losses join the existing `landscape_regularization_loss` block in
`trainer._losses_for_batch` under the `"landscape"` weight group, exposed as
new lambdas in `MORTFMConfig` (`lambda_basin_label`, `lambda_barrier`,
`lambda_attractor`, `lambda_monotonicity`).

### 6.4 Composition into the canonical loss dict

The trainer composes them as:

```python
loss_dict = {}
loss_dict.update(_trajectory_step(...))                     # Phase 1
loss_dict.update(basin_label_loss(out["basin_probs"][:, -1, :], batch.basin_label))
loss_dict.update(monotonicity_loss(out["basin_probs"], batch.basin_label))
loss_dict.update(attractor_stability_loss(model.lens_sde.potential))
loss_dict.update(barrier_regularization_loss(
    model.lens_sde.potential,
    z_sensitive=state.z0[batch.basin_label == 0],
    z_resistant=state.z0[batch.basin_label == 3],
    drug=drug_token,
))
total = sum(weights[k] * v for k, v in loss_dict.items() if k in weights)
```

`MORTBatch.basin_label: Optional[Tensor]` is the new label field, populated
from `assign_basin_label` in the pair builder.

### 6.5 `landscape_regularizers.py` additions

The existing module covers smoothness / separation / drug-barrier / entropy
but not the **canonical-path** variants (those use `lens_sde.potential`, which
is a different class). Add adapters:

```python
# resistancemap/landscape/landscape_regularizers.py — append

def lens_smoothness_penalty(lens_potential, z, drug, **kwargs):
    """Same FD-Hessian-trace estimator, adapted to GraphEnergyResistanceSDE's
    WaddingtonPotential signature U(z, drug)."""
    ...

def lens_basin_separation_penalty(lens_potential, *, min_distance=0.5):
    centres = lens_potential.basin_centroids
    ...
```

This keeps the legacy `WaddingtonPotential` (in `landscape/potential.py`)
untouched and adds dispatcher functions for the LENS variant.

### 6.6 Visualisation

```python
# resistancemap/interpretability/landscape_viz.py — new file
"""2D/3D plotting of the learned potential.

NO synthetic data. The function takes real encoder outputs (e.g. the test-split
z0 vectors) as the dimensionality-reduction anchors. PHATE/UMAP is fit on
those real points; the potential is evaluated on the resulting grid via the
learned encoder/decoder pair.
"""
from typing import Optional
import numpy as np
import torch


def plot_potential_2d(
    potential,
    z_samples: torch.Tensor,             # (N, d_latent) — REAL latents from test split
    drug: torch.Tensor,                  # (d_drug,)     — REAL drug context
    *,
    reducer: str = "phate",              # "phate" | "umap" | "pca"
    n_grid: int = 60,
    ax=None,
):
    """Project z_samples to 2D, evaluate U on a regular grid in 2D, and lift
    back to d_latent via the reducer's inverse_transform (PHATE has one;
    UMAP via inverse_transform; PCA exact).

    Overlays the encoder's z_samples as scattered points, colored by their
    predicted basin index from ResistanceBasin.
    """
    ...


def plot_potential_3d(potential, z_samples, drug, *, **kwargs):
    """3D surface variant. Same contract."""
    ...
```

Key constraint: **the function refuses to run without real `z_samples`** —
no `np.random` defaults, no synthetic seeds. The reducer is fit on the test
split's latents.

### 6.7 New evaluation metrics

```python
# resistancemap/evaluation/landscape_metrics.py — new file
import numpy as np
import torch


def basin_purity(
    basin_probs: torch.Tensor,           # (N, K) at terminal time
    basin_labels: torch.Tensor,          # (N,) long
) -> float:
    """Argmax-basin vs label accuracy on the subset with valid labels."""
    valid = basin_labels >= 0
    if valid.sum() == 0:
        return float("nan")
    pred = basin_probs[valid].argmax(dim=-1)
    return float((pred == basin_labels[valid]).float().mean())


def transition_consistency(
    basin_probs_traj: torch.Tensor,      # (N, T, K)
    basin_labels: torch.Tensor,
) -> float:
    """Fraction of labelled trajectories whose predicted-basin probability is
    monotonically non-decreasing across t. Real, not invented."""
    valid = basin_labels >= 0
    if valid.sum() == 0:
        return float("nan")
    N, T, K = basin_probs_traj.shape
    idx = basin_labels.clamp(min=0)
    p_label = basin_probs_traj[torch.arange(N), :, idx]
    diffs = p_label[:, 1:] - p_label[:, :-1]
    is_monotone = (diffs >= -1e-6).all(dim=-1)
    return float(is_monotone[valid].float().mean())


def barrier_survival_correlation(
    barrier_heights: np.ndarray,         # (N,) max U along the labelled trajectory
    event_times: np.ndarray,             # (N,)
    event_observed: np.ndarray,          # (N,) in {0,1}
) -> dict:
    """Spearman correlation between barrier height and observed survival,
    restricted to uncensored rows. A high barrier on the sensitive→resistant
    path should predict longer time to progression."""
    from scipy.stats import spearmanr
    obs = event_observed.astype(bool)
    if obs.sum() < 5:
        return {"spearman_r": float("nan"), "n_used": int(obs.sum())}
    r, p = spearmanr(barrier_heights[obs], event_times[obs])
    return {"spearman_r": float(r), "p_value": float(p), "n_used": int(obs.sum())}
```

### 6.8 Phase 6 per-file change table

| File | Change | Type |
| --- | --- | --- |
| `resistancemap/data/basin_label_builder.py` | NEW — rule-based, label-only basin labeler. | new |
| `resistancemap/mortfm/schemas.py` | Add `basin_label: Optional[Tensor]` to `MORTBatch`; add `basin_label: Optional[int]` to `ResistanceOutcome`. | edit |
| `resistancemap/data/mortfm_dataset.py` | Stack `basin_label` in collate (with `-1` sentinel). | edit |
| `resistancemap/training/canonical_mortfm_losses.py` | Add `basin_label_loss`, `barrier_regularization_loss`, `attractor_stability_loss`, `monotonicity_loss`. | edit |
| `resistancemap/landscape/landscape_regularizers.py` | Add `lens_*` adapter penalties for `GraphEnergyResistanceSDE.WaddingtonPotential`. | edit |
| `resistancemap/interpretability/landscape_viz.py` | NEW — `plot_potential_2d` / `_3d`. Hard-fail on synthetic input. | new |
| `resistancemap/evaluation/landscape_metrics.py` | NEW — basin purity, transition consistency, barrier↔survival Spearman. | new |
| `resistancemap/mortfm/trainer.py` | Wire the four new losses into the `"E"` stage weight group; add `lambda_basin_label / lambda_barrier / lambda_attractor / lambda_monotonicity` to `MORTFMConfig`. | edit |
| `tests/mortfm/test_basin_label_builder.py` | NEW — rule corner cases (missing MRD, censoring, ambiguous PFS). | new |
| `tests/mortfm/test_landscape_metrics.py` | NEW — basin_purity finite, monotone synthetic trajectory passes consistency. | new |

### 6.9 Phase 6 Definition of Done

1. Every basin label in `MORTBatch.basin_label` traces to an explicit row in
   `assign_basin_label` — no fallback to "majority class".
2. `basin_label_loss` and `monotonicity_loss` are reflected in
   `STRICT_SUPERVISION_COVERAGE["E"]` as additive supervision sources.
3. `landscape_viz.plot_potential_2d` raises `ValueError` if any element of
   `z_samples` is non-finite or if `z_samples.shape[0] < 20` (refuses to plot
   from underpowered evidence).
4. `landscape_metrics.basin_purity` agrees with `sklearn.metrics.accuracy_score`
   on a labelled fixture; `transition_consistency` returns 1.0 on a
   deterministically-monotone fixture and 0.0 on a deterministically-reversing
   fixture.
5. `WADDINGTON_NEURAL_ODE_MATH.md` is updated to cite which inequalities the
   four new losses enforce.

---

## PHASE 7 — Survival ↔ hitting-time consistency

### 7.1 Diagnosis

`hitting_time.HittingTime.forward`
(`resistancemap/mortfm/trajectory/hitting_time.py:43-87`) returns
`{"cdf", "mean_tau", "frac_hit", "t_grid"}`. Its `t_grid` is
`sde_out["t_grid"]` (line 235 of `graph_energy_sde.py`),
`linspace(0, integration_time, n_time_grid)` — in **model time units**,
not months, and `integration_time` is set to `1.0` inside the LENS SDE
constructor (`model.py:151`).

`CompetingRiskHead.forward`
(`resistancemap/mortfm/survival/competing_risk_head.py:72-104`) builds bins
internally via `discrete_time_grid(t_max_days, n_bins)`. The MORT-FM model
calls it with `n_bins=config.survival_n_bins=12` but never passes a `t_grid`
into the head itself — `t_grid` is **only** constructed *outside* in
`trainer.py:307` as
`torch.linspace(0, self.config.integration_time, self.config.survival_n_bins + 1)`
which is `linspace(0, 12.0, 13)` — months.

Therefore today the two systems use:
- SDE / hitting: `linspace(0, 1.0, 25)` (LENS) — unitless
- Survival head: `linspace(0, 12.0, 13)` — months

They are **not commensurable**. The directional-consistency invariant
"high hitting probability by month 12 ⇒ high hazard by month 12" cannot
even be evaluated because index `T-1` in `hitting_cdf` corresponds to a
different absolute time from index `K-1` in `survival_curve`.

### 7.2 Grid alignment

Single canonical grid, constructed once and passed to both subsystems:

```python
# resistancemap/mortfm/trajectory/grid.py — new
import torch

def canonical_time_grid(integration_time_months: float, n_steps: int) -> torch.Tensor:
    """The ONE grid used by SDE, hitting-time, and survival head.

    Units: months. n_steps points inclusive of both endpoints.
    """
    return torch.linspace(0.0, integration_time_months, n_steps)
```

Both `GraphEnergyResistanceSDE.__init__` and `CompetingRiskHead` are extended
to accept an externally-supplied `t_grid` and to expose it via a
`.t_grid` attribute. The `MORTFM.__init__` passes the same grid to both:

```python
# resistancemap/mortfm/model.py — diff
self._canonical_t_grid = canonical_time_grid(
    integration_time_months=config.integration_time,    # = 12.0
    n_steps=config.n_time_grid,                          # = 25
)
self.lens_sde = GraphEnergyResistanceSDE(
    ...,
    integration_time=config.integration_time,            # was 1.0
    n_time_grid=config.n_time_grid,
)
self.lens_competing_risk = CompetingRiskHead(
    d_latent=config.d_latent,
    n_bins=config.n_time_grid - 1,                       # bins between gridpoints
    event_names=["progression"],
    t_grid=self._canonical_t_grid,
)
```

`survival_n_bins` is **removed** from `MORTFMConfig` (or kept and silently
forced to `n_time_grid - 1` with a `warnings.warn` if they disagree). After
this change there is exactly one time axis in the model.

### 7.3 New losses

```python
# resistancemap/training/canonical_mortfm_losses.py — additions
import torch
import torch.nn.functional as F


def hitting_time_nll(
    hitting_cdf: torch.Tensor,           # (B, T) empirical P(tau <= t_k)
    event_time: torch.Tensor,            # (B,) months
    event_observed: torch.Tensor,        # (B,) {0,1}
    t_grid: torch.Tensor,                # (T,) months — MUST match hitting_cdf
    *,
    eps: float = 1e-6,
) -> dict:
    """Discrete NLL treating the empirical hitting CDF as a survival predictor.

    For uncensored row i with event at bin k_i:
        L_i = -log(F(k_i) - F(k_i - 1) + eps)
    For censored row i with last-known-event-free time k_i:
        L_i = -log(1 - F(k_i) + eps)
    """
    B, T = hitting_cdf.shape
    if t_grid.shape[0] != T:
        raise ValueError(f"t_grid length {t_grid.shape[0]} != hitting_cdf T={T}")
    # Bin each row's event_time onto the grid.
    diff = (event_time.unsqueeze(-1) - t_grid.unsqueeze(0)).abs()
    k = diff.argmin(dim=-1)                                          # (B,)
    cdf_at_k = hitting_cdf.gather(1, k.unsqueeze(-1)).squeeze(-1)
    cdf_prev = torch.where(
        k > 0,
        hitting_cdf.gather(1, (k - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1),
        torch.zeros_like(cdf_at_k),
    )
    pmf_at_k = (cdf_at_k - cdf_prev).clamp_min(eps)
    surv_at_k = (1.0 - cdf_at_k).clamp_min(eps)
    obs = event_observed.float()
    nll = -(obs * pmf_at_k.log() + (1 - obs) * surv_at_k.log())
    return {"hitting_nll": nll.mean()}


def survival_hitting_consistency_loss(
    survival_curve: torch.Tensor,        # (B, T) S(t) = P(no event by t)
    hitting_cdf: torch.Tensor,           # (B, T) F_hit(t) = P(tau_R <= t)
    *,
    competing_risk_correction: Optional[torch.Tensor] = None,  # (B, T) in [0,1]
    margin: float = 0.0,
) -> dict:
    """The directional invariant.

    For competing-risk-free analyses:
        F_hit(t) + S(t) <= 1                    (cannot both be event-free
                                                  AND have already hit basin)
    Equivalently:  S(t) <= 1 - F_hit(t).
    For competing risks with correction c(t) in [0,1] capturing
    P(competing event by t):
        S(t) <= 1 - F_hit(t) + c(t)
    """
    if survival_curve.shape != hitting_cdf.shape:
        raise ValueError(
            f"survival_curve {tuple(survival_curve.shape)} != "
            f"hitting_cdf {tuple(hitting_cdf.shape)} — grids unaligned."
        )
    upper = 1.0 - hitting_cdf
    if competing_risk_correction is not None:
        upper = upper + competing_risk_correction
    excess = (survival_curve - upper - margin).clamp(min=0.0)
    return {"survival_hitting_consistency": excess.mean()}
```

### 7.4 Dynamic survival metrics

```python
# resistancemap/evaluation/survival_metrics.py — new file
"""Time-dependent survival metrics. Wraps lifelines / sksurv where possible
and falls back to closed-form implementations otherwise. No metric is
constructed; every value is read from the input arrays."""

from typing import Sequence
import numpy as np


def time_dependent_auc(
    risk_scores: np.ndarray,             # (N,) higher = worse
    event_times: np.ndarray,             # (N,)
    event_observed: np.ndarray,          # (N,) {0,1}
    eval_times: Sequence[float],         # months at which to evaluate
) -> dict:
    """Uno's IPCW time-dependent AUC."""
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv
    y = Surv.from_arrays(event=event_observed.astype(bool), time=event_times)
    auc, mean_auc = cumulative_dynamic_auc(y, y, risk_scores, list(eval_times))
    return {"auc_at_t": dict(zip(eval_times, [float(a) for a in auc])),
            "mean_auc": float(mean_auc), "n": int(len(event_times))}


def integrated_brier_score(
    survival_curves: np.ndarray,         # (N, T) S(t)
    event_times: np.ndarray,
    event_observed: np.ndarray,
    t_grid: np.ndarray,                  # (T,)
) -> float:
    from sksurv.metrics import integrated_brier_score as sks_ibs
    from sksurv.util import Surv
    y = Surv.from_arrays(event=event_observed.astype(bool), time=event_times)
    return float(sks_ibs(y, y, survival_curves, t_grid))


def calibration_slope_intercept(
    predicted_prob: np.ndarray,          # (N,) at a fixed horizon
    observed_event: np.ndarray,          # (N,) {0,1}
) -> dict:
    """Logistic regression of observed on logit(predicted). Standard
    Cox/Royston-Altman calibration with slope=1, intercept=0 ideal."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    p = np.clip(predicted_prob, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    if observed_event.sum() == 0 or observed_event.sum() == len(observed_event):
        return {"slope": float("nan"), "intercept": float("nan"), "n": int(len(p))}
    lr = LogisticRegression().fit(logit, observed_event)
    return {"slope": float(lr.coef_[0, 0]),
            "intercept": float(lr.intercept_[0]),
            "n": int(len(p))}
```

### 7.5 Baselines

```python
# resistancemap/baselines/survival.py — new
"""Canonical survival baselines. ALL are fit on the exact same MMRF/BeatAML
split used by MORT-FM. None of them returns a fabricated metric value;
every output is computed by the underlying library on the given arrays.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SurvivalBaselineRegistry:
    use_clinical_cox: bool = True
    use_cox_omics: bool = True
    use_random_survival_forest: bool = True
    use_deepsurv: bool = True
    use_discrete_hazard_mlp: bool = True


def fit_clinical_cox(clinical_train_df, train_y_struct):
    from lifelines import CoxPHFitter
    cph = CoxPHFitter().fit(clinical_train_df, "time", "event")
    return cph

def fit_cox_with_omics(clinical_train_df, omics_train, train_y_struct):
    import pandas as pd
    df = pd.concat([clinical_train_df, pd.DataFrame(omics_train)], axis=1)
    from lifelines import CoxPHFitter
    return CoxPHFitter(penalizer=0.1).fit(df, "time", "event")

def fit_rsf(X_train, y_struct):
    from sksurv.ensemble import RandomSurvivalForest
    return RandomSurvivalForest(n_estimators=300, n_jobs=-1).fit(X_train, y_struct)

def fit_deepsurv(X_train, y_struct, *, epochs: int = 100):
    # pycox.models.CoxPH
    ...

def fit_discrete_hazard_mlp(X_train, y_struct, *, n_bins: int = 24):
    # pycox.models.LogisticHazard
    ...
```

### 7.6 Directional-consistency invariant — formal statement

For any patient at any time `t_k` on the canonical grid, under
non-competing events:

    P(no progression by t_k) <= 1 - P(tau_resistant <= t_k)

This is just the identity that "still event-free" is a subset of "has not
yet entered the resistant basin". Under competing events with cumulative
incidence `c_other(t_k)`:

    P(no progression by t_k) <= 1 - P(tau_resistant <= t_k) + c_other(t_k)

The audit metric reported alongside C-index/IBS is

    consistency_violation = mean_t mean_i max(0, S_i(t) - (1 - F_hit_i(t) + c_other_i(t)))

which is exactly `survival_hitting_consistency_loss` evaluated on test
predictions. A model that violates this is internally inconsistent —
its hitting head and its hazard head disagree about the same future.

### 7.7 Phase 7 per-file change table

| File | Change | Type |
| --- | --- | --- |
| `resistancemap/mortfm/trajectory/grid.py` | NEW — single source of truth for `t_grid`. | new |
| `resistancemap/mortfm/trajectory/graph_energy_sde.py` | Accept an external `t_grid` arg in `__init__`; expose `.t_grid`. | edit |
| `resistancemap/mortfm/survival/competing_risk_head.py` | Accept `t_grid` arg; store as buffer; remove internal `discrete_time_grid` default when `t_grid` is given. | edit |
| `resistancemap/mortfm/model.py` | Build canonical grid once; pass to LENS SDE and competing-risk head. Remove `survival_n_bins` divergence. | edit |
| `resistancemap/training/canonical_mortfm_losses.py` | Add `hitting_time_nll`, `survival_hitting_consistency_loss`. | edit |
| `resistancemap/evaluation/survival_metrics.py` | NEW — Uno AUC, IBS, calibration slope/intercept. | new |
| `resistancemap/baselines/survival.py` | NEW — Clinical Cox / Cox+omics / RSF / DeepSurv / discrete hazard MLP. | new |
| `tests/mortfm/test_grid_alignment.py` | NEW — assert `model.lens_sde.t_grid` equals `model.lens_competing_risk.t_grid`. | new |
| `tests/mortfm/test_survival_hitting_consistency.py` | NEW — synthetic perfectly-consistent pair returns 0; deliberately-inconsistent pair returns >0. | new |
| `tests/mortfm/test_hitting_time_nll.py` | NEW — closed-form check on a 2-bin example. | new |

### 7.8 Phase 7 Definition of Done

1. `model.lens_sde.t_grid` is literally the same tensor (or
   `torch.equal()`-equivalent) as `model.lens_competing_risk.t_grid`.
2. `survival_hitting_consistency_loss` is logged every epoch and reported in
   the final acceptance report; the report fails the run if mean violation
   exceeds 0.05.
3. `hitting_time_nll` matches a hand-computed value on a 4-row, 3-bin
   fixture to 1e-6.
4. `time_dependent_auc` / `integrated_brier_score` are computed on the test
   split for every baseline in `survival/baselines.py` and for the MORT-FM
   model, all on the **same grid**.
5. `MORTFMConfig.survival_n_bins` is either removed or forced to
   `n_time_grid - 1` with a warn; no two grids may coexist.

---

## Cross-phase invariants

- No phase introduces synthetic / random / hard-coded metric data anywhere
  in production code paths. Test fixtures may use small random tensors only
  to assert shape / mask / finiteness — never to assert a numeric metric
  value.
- `STRICT_SUPERVISION_COVERAGE["E"]` and `["F"]` (and a new `["E_basin"]`)
  are honored: the trainer raises `MissingSupervisionError` instead of
  silently training on under-supervised batches.
- Every new loss returns a `dict`, never a bare tensor, so per-component
  values can be logged and audited separately.
- Every new evaluation metric returns `nan` (and a populated `n` field)
  when the supporting subset is too small, rather than imputing a number.
