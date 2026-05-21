# Phase 0 — Canonical Trainer (v18 -> v19)

Status: design-only. **No code is committed in this phase.** This document is
the contract the v19 implementation work must conform to.

---

## 0. Why this phase exists

The v18 branch landed two important pieces of plumbing:

1. `MORTFM.forward(...)` was redefined as the **canonical, LENS-dict-returning
   patient-level surface** built on the v16/v17 `GraphEnergyResistanceSDE` +
   `CompetingRiskHead` + `ResistanceBasin` + `HittingTime` stack.
2. The v15 task heads (`PathwayRouteHead`,
   `DrugSpecificTrajectoryRiskHead`, `EvidentialUncertaintyHead`,
   `CounterfactualInterventionHead`, etc.) and the
   `TrajectorySampler` were moved under
   `resistancemap.legacy.v15_planned_mortfm.*` and are now only reachable via
   the `forward_legacy(...)` method, which still returns the v15
   `TrajectoryPrediction` dataclass.

The training loop, however, **was not migrated**. It still consumes
`forward_legacy(...)` and the legacy dataclass surface for every patient-level
stage. This means the canonical `forward()` is unreachable from real
training — it is exercised only by a unit test
(`tests/mortfm/test_canonical_forward.py`).

That mismatch is the P0 gap this phase fixes.

---

## 1. Verified evidence (file:line citations)

### 1.1 Trainer routes through `forward_legacy(...)` (P0 gap)

`ResistanceMap/resistancemap/mortfm/trainer.py:227-232`:

```
prediction = self.model.forward_legacy(
    batch,
    time_grid=None,
    n_traj_samples=4 if stage == "E" else 1,
    graph=self.graph,
)
```

The comment block immediately above this call
(`trainer.py:219-226`) explicitly acknowledges the migration is incomplete:

> "v18 — explicitly route through forward_legacy(). The new canonical
> forward() returns the LENS dict surface; the legacy trainer below still
> consumes a TrajectoryPrediction dataclass with state_logits,
> survival_curve, pathway_*, etc. The v18 stage router in
> mortfm/trainer_v18.py will switch to model(batch) (canonical) for
> stages E/F/G/H; this legacy trainer is retained as forward_legacy()
> consumer for backward compat with old checkpoints."

Downstream consumers of the legacy surface (still alive in v18 trainer):

* `trainer.py:267` — `prediction.drug_specific_risk` (stage C)
* `trainer.py:289` — `prediction.trajectory_mean` and `prediction.z_path`
  (stage E)
* `trainer.py:315` — `prediction.resistance_state_logits` (stage F state)
* `trainer.py:319-322` — `prediction.pathway_protein_scores` (stage G)

The stage-F survival branch (`trainer.py:299-312`) bypasses the legacy
`prediction` and re-runs `self.model.encode(batch)` +
`self.model.rollout(...)` + `self.model.survival_head(z_T)` — which is the
**v15 `TimeToResistanceHead` (now a legacy head), not the canonical
`CompetingRiskHead`** allocated at `model.py:165-169`.

### 1.2 Canonical `forward(...)` and its dict surface

`ResistanceMap/resistancemap/mortfm/model.py:256-314` defines the canonical
forward. Returned dict keys (verified from `model.py:302-314`):

```
z0, z_traj, z_samples, t_grid,
hazard, survival_curve, cif_per_event,
basin_probs,
hitting_cdf, hitting_mean_tau, hitting_frac_hit
```

Inputs: `batch: MORTBatch`, `clinical: Optional[Tensor]`,
`drug: Optional[Tensor]`, `graph_emb: Optional[Tensor]`,
`use_graph_projector: bool=True`. Internally:

* `state = self.encode(batch)` (`model.py:284`)
* `sde_out = self.lens_sde(z0, graph_emb, drug, clinical, return_samples=True)`
  (`model.py:297`)
* `head_out = self.lens_competing_risk(z_final)` (`model.py:299`)
* `basin_probs = self.lens_basin.trajectory_probs(sde_out["z_traj"])`
  (`model.py:300`)
* `hit_out = self.lens_hitting(sde_out["z_samples"], sde_out["t_grid"])`
  (`model.py:301`)

### 1.3 Legacy vs canonical split is structurally enforced

`model.py:33-44` — every v15 head and the `TrajectorySampler` are now
imported from `resistancemap.legacy.v15_planned_mortfm.*`.

`tests/mortfm/test_canonical_forward.py:90-96` already asserts that:

* `MORTFM.forward is not MORTFM.forward_legacy`
* `forward_lens` (the v17 name) has been removed.

So the canonical contract is **policed at the model layer** but **not at
the trainer layer** — which is precisely the P0 gap.

### 1.4 Loss-supervision contract surfaces already exist

* `MORTBatch.validate_for_loss(loss_name)` at `mortfm/schemas.py:464-496`
  already raises on missing supervision for `survival`, `trajectory`,
  `resistance_state`, `drug_response`. **It is not called anywhere in
  `trainer.py` today.**
* `training/loss_router.py:58-131` defines a generic `LossRouter` with
  per-loss `required_supervision` and a `LossAuditEntry` audit trail.
  **It is also not called from the v18 trainer.**

Both pieces are infrastructure waiting for the canonical trainer to consume
them.

### 1.5 v18.2 strict-supervision policy

`trainer.py:80-119, 388-406` define `MissingSupervisionError`,
`STRICT_SUPERVISION_COVERAGE = {"E": 0.80, "F": 0.90, "G": 0.70}`, and the
post-epoch check that raises if coverage falls below threshold. This policy
must be **preserved verbatim** by the canonical trainer; it is the
"raise, don't skip" guarantee the v18.2 commit shipped.

### 1.6 Smoke test currently locks in the legacy surface

`tests/mortfm/test_mortfm_trainer_smoke.py:124` explicitly calls
`model.forward_legacy(batch, compute_counterfactuals=True)`. The test asserts
the legacy `TrajectoryPrediction` shape (`z_path`,
`resistance_state_logits`, `pathway_protein_scores`,
`drug_specific_risk`, `counterfactual_rankings`) — none of which exist on
the canonical dict surface. This file must be **split** rather than edited,
so the legacy path stays under test for backward-compat while the canonical
path gets its own dedicated smoke.

---

## 2. Proposed module: `resistancemap/mortfm/canonical_trainer.py`

This is the new public training entry point for v19+. It is a **sibling** of
the existing `trainer.py`, not a replacement — the legacy trainer stays alive
behind `legacy=True` and behind `LegacyMORTFMTrainer` (see §6).

```python
"""
resistancemap/mortfm/canonical_trainer.py
=========================================
Canonical MORT-FM trainer (v19+).

Differences from the legacy v15/v18 trainer:

  * Patient-level stages (E/F/G/H) call ``self.model(batch, clinical=...,
    drug=..., graph_emb=...)`` and consume the LENS dict surface
    (z_traj, hazard, survival_curve, cif_per_event, basin_probs,
    hitting_cdf, hitting_mean_tau, hitting_frac_hit). No v15
    TrajectoryPrediction dataclass is touched.
  * Loss assembly is delegated to LossRouter — supervision presence
    is validated PER loss before the term is added.
  * Per-batch supervision audit drives the v18.2 strict-coverage check
    that already lives in trainer.py.
  * Foundation stages (A/B/C/D) keep their existing semantics —
    they DO NOT need the LENS path — by routing through
    ``model.forward_foundation(batch)`` (encode-only).

NO synthetic / random / hardcoded metric data anywhere. If the supervision
isn't present in the batch, the corresponding loss is *not added*; if the
stage's strict-coverage threshold is missed, the trainer *raises*.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from resistancemap.mortfm.acceptance_gate import (
    AcceptanceError,
    AcceptanceReport,
    evaluate_cohort,
)
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import (
    MORTBatch,
    MORTFMConfig,
    TemporalTrainingPair,
)
from resistancemap.mortfm.trainer import (
    MissingSupervisionError,
    StageSupervisionReport,
    STRICT_SUPERVISION_COVERAGE,
    TrainingMetrics,
    VALID_STAGES,
)
from resistancemap.training.canonical_mortfm_losses import (
    canonical_trajectory_loss,
    lens_survival_loss,
    hitting_time_loss,
    basin_transition_loss,
    sde_path_loss,
)
from resistancemap.training.loss_router import LossRouter
from resistancemap.training.mortfm_losses import (
    assemble_total_loss,
    contrastive_alignment_loss,
    masked_modality_loss,
    reconstruction_loss,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Stage spec — canonical version
# --------------------------------------------------------------------------

#: Which losses are LIVE in each stage, mapped to canonical-dict keys
#: they consume. Stage A/B/C/D are foundation (encoder-only); stages
#: E/F/G/H are patient-level and call the canonical forward().
CANONICAL_STAGE_LOSSES: Dict[str, Tuple[str, ...]] = {
    "A": ("recon",),
    "B": ("recon", "masked", "contrastive"),
    "C": ("recon", "contrastive"),     # drug-response is a separate sub-stage
    "D": ("contrastive",),             # domain-adaptation
    "E": ("trajectory", "sde_path", "basin_transition"),
    "F": ("survival", "hitting"),
    "G": ("basin_transition",),        # pathway/route — driven by basin labels
    "H": ("survival", "calibration"),
}


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


class CanonicalMORTFMTrainer:
    """Canonical v19+ trainer — drives MORT-FM through the LENS surface.

    Parameters
    ----------
    model :
        :class:`MORTFM` instance. The trainer calls ``model(batch, ...)``
        (i.e. the canonical ``forward``) for patient-level stages and
        ``model.forward_foundation(batch)`` for foundation stages.
    config :
        :class:`MORTFMConfig` — controls loss weights, integration grid,
        survival n_bins.
    train_loader, val_loader :
        DataLoaders emitting :class:`MORTBatch`.
    loss_router :
        Optional pre-configured :class:`LossRouter`. If ``None``, the trainer
        constructs the canonical router from ``CANONICAL_STAGE_LOSSES`` and
        ``config.lambda_*`` weights.
    graph_emb_fn :
        Optional callable ``(MORTBatch) -> Tensor[B, d_graph]`` that returns
        a precomputed biological-graph embedding per row. If ``None``, the
        model falls back to its internal ``LatentToGraphProjector``
        (model.py:162-164).
    device :
        Torch device (auto-detected if ``None``).

    Honest-behaviour invariants
    ---------------------------
    1. Every patient-level forward goes through ``self.model(batch, ...)``;
       no call to ``forward_legacy`` is made.
    2. Every loss term first calls ``batch.validate_for_loss(loss_name)``
       or routes through ``LossRouter`` (which encodes the same check).
       Missing supervision -> loss term skipped (audited), not zero-filled.
    3. ``train_epoch`` enforces ``STRICT_SUPERVISION_COVERAGE[stage]``
       at the end of the epoch and raises ``MissingSupervisionError``
       if violated.
    4. No random / synthetic / hardcoded metric data anywhere.
    """

    def __init__(
        self,
        model: MORTFM,
        config: MORTFMConfig,
        *,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_router: Optional[LossRouter] = None,
        graph_emb_fn: Optional[Callable[[MORTBatch], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.graph_emb_fn = graph_emb_fn

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(config.mixed_precision and device.type == "cuda"),
        )
        self.history: List[TrainingMetrics] = []
        self.loss_router = loss_router or self._default_router()
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Default router wiring
    # ------------------------------------------------------------------

    def _default_router(self) -> LossRouter:
        """Build a LossRouter that maps canonical-dict keys to loss_fn(outputs, batch)."""
        cfg = self.config
        loss_fns: Dict[str, Callable] = {
            "trajectory":        lambda o, b: canonical_trajectory_loss(o, b),
            "sde_path":          lambda o, b: sde_path_loss(o, b),
            "basin_transition":  lambda o, b: basin_transition_loss(o, b),
            "survival":          lambda o, b: lens_survival_loss(
                                     o, b, n_bins=cfg.survival_n_bins,
                                     integration_time=cfg.integration_time,
                                 ),
            "hitting":           lambda o, b: hitting_time_loss(o, b),
        }
        weights = {
            "trajectory":       cfg.lambda_trajectory,
            "sde_path":         cfg.lambda_trajectory,
            "basin_transition": cfg.lambda_pathway,
            "survival":         cfg.lambda_survival,
            "hitting":          cfg.lambda_survival,
        }
        required = {
            "trajectory":       ["future_state"],
            "sde_path":         ["future_state"],
            "basin_transition": ["resistance_label"],
            "survival":         ["event_time", "event_observed"],
            "hitting":          ["event_time", "event_observed"],
        }
        return LossRouter(
            enabled=set(weights.keys()),
            loss_fns=loss_fns,
            weights=weights,
            required_supervision=required,
        )

    # ------------------------------------------------------------------
    # Per-batch loss assembly (canonical path)
    # ------------------------------------------------------------------

    def _build_canonical_kwargs(self, batch: MORTBatch) -> Dict[str, Optional[torch.Tensor]]:
        """Materialise the ``clinical`` / ``drug`` / ``graph_emb`` kwargs that
        ``MORTFM.forward(...)`` expects from the batch."""
        # Clinical: take the first d_clinical (=5) cols of batch.clinical, or None.
        clinical: Optional[torch.Tensor] = None
        if batch.clinical is not None:
            clinical = batch.clinical[..., :5]
        # Drug: take the first 8 cols of batch.drug, or None.
        drug: Optional[torch.Tensor] = None
        if batch.drug is not None:
            drug = batch.drug[..., :8]
        graph_emb: Optional[torch.Tensor] = None
        if self.graph_emb_fn is not None:
            graph_emb = self.graph_emb_fn(batch)
        return {"clinical": clinical, "drug": drug, "graph_emb": graph_emb}

    def _losses_for_batch(
        self,
        batch: MORTBatch,
        stage: str,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], List[Any]]:
        """Return (components, weights, audit) for one batch + stage.

        For foundation stages (A/B/C/D) this calls forward_foundation only.
        For patient stages (E/F/G/H) this calls the canonical forward().
        """
        batch = batch.to(self.device)
        active_losses = set(CANONICAL_STAGE_LOSSES[stage])
        components: Dict[str, torch.Tensor] = {}

        # Foundation stages — encode-only path.
        if stage in ("A", "B", "C", "D"):
            state = self.model.forward_foundation(batch)
            # Reconstruction, masked-modality, contrastive — these touch the
            # encoder directly; they do not need the SDE rollout.
            if "recon" in active_losses and batch.rna is not None:
                enc = self.model.fusion.tokenizer.encoders.get("rna")
                if enc is not None:
                    z = enc(batch.rna)
                    mu, theta, _ = enc.decode(
                        z, library_size=batch.rna.sum(dim=-1).clamp(min=1.0)
                    )
                    components["recon"] = reconstruction_loss(
                        batch.rna.clamp(min=0).round(), (mu, theta),
                        distribution="nb",
                    )
            if "masked" in active_losses and batch.proteomics is not None:
                enc = self.model.fusion.tokenizer.encoders.get("proteomics")
                if enc is not None:
                    pred_token = state.modality_embeddings.get("proteomics")
                    target_token = enc(batch.proteomics).detach()
                    if pred_token is not None:
                        components["masked"] = masked_modality_loss(
                            pred_token, target_token
                        )
            if "contrastive" in active_losses:
                embs = list(state.modality_embeddings.values())
                if len(embs) >= 2:
                    a = F.normalize(embs[0], dim=-1)
                    b = F.normalize(embs[1], dim=-1)
                    components["contrastive"] = contrastive_alignment_loss(a, b)
            weights = {
                k: getattr(self.config, f"lambda_{k}", 0.0) for k in components
            }
            return components, weights, []

        # Patient-level stages — canonical forward.
        kwargs = self._build_canonical_kwargs(batch)
        outputs = self.model(batch, **kwargs)

        # The LossRouter enforces required_supervision per loss; we hand it
        # the canonical dict + the batch (as a Mapping).
        batch_as_mapping = self._batch_to_mapping(batch)
        total, comp_floats, audit = self.loss_router.compute(
            outputs=outputs, batch=batch_as_mapping
        )
        # We only keep the active losses for this stage.
        kept = {k: v for k, v in comp_floats.items() if k in active_losses}
        # Convert kept floats back to tensors for assemble_total_loss compat.
        # (Audit trail uses comp_floats as-is.)
        components_tensor: Dict[str, torch.Tensor] = {}
        weights: Dict[str, float] = {}
        for k in kept:
            # We need the raw tensor — re-run just the kept losses via the router's
            # loss_fns to avoid the float<->tensor roundtrip. The router already
            # validated supervision, so this second pass is safe.
            fn = self.loss_router.loss_fns[k]
            components_tensor[k] = fn(outputs, batch_as_mapping)
            weights[k] = self.loss_router.weights[k]
        return components_tensor, weights, audit

    @staticmethod
    def _batch_to_mapping(batch: MORTBatch) -> Dict[str, Any]:
        """LossRouter expects a Mapping[str, Any], not the MORTBatch dataclass."""
        from dataclasses import fields as _f
        return {f.name: getattr(batch, f.name) for f in _f(batch)}

    # ------------------------------------------------------------------
    # Epoch / fit
    # ------------------------------------------------------------------

    def train_epoch(self, stage: str, epoch: int) -> TrainingMetrics:
        """Identical to legacy train_epoch but uses canonical _losses_for_batch
        and the v18.2 strict-coverage check (re-imported, not redefined)."""
        ...  # full implementation mirrors trainer.py:338-422, swapping
        ...  # _losses_for_batch for the canonical version above.

    @torch.no_grad()
    def validate(self, stage: str) -> float:
        ...

    def fit_stage(
        self,
        stage: str,
        n_epochs: int,
        *,
        acceptance_level: Optional[str] = None,
        acceptance_pairs: Optional[Sequence[TemporalTrainingPair]] = None,
    ) -> List[TrainingMetrics]:
        """Same contract as legacy fit_stage; same acceptance-gate semantics."""
        ...

    def fit_all(
        self, stages: Sequence[str] = ("A", "B", "C", "E", "F"),
    ) -> List[TrainingMetrics]:
        ...

    def save_checkpoint(self, name: str) -> str: ...
    def load_checkpoint(self, path: str) -> None: ...
```

---

## 3. Proposed module: `resistancemap/training/canonical_mortfm_losses.py`

These are the **only** loss functions that touch the canonical dict surface.
Each loss function takes `(outputs: Mapping[str, Any], batch: Mapping[str, Any])`
to match `LossRouter`'s call convention (`loss_router.py:110`).

```python
"""
resistancemap/training/canonical_mortfm_losses.py
=================================================
Loss functions that consume the canonical MORTFM.forward() LENS dict.

Every function takes the canonical-dict outputs and a batch-as-Mapping,
and returns a scalar torch.Tensor. NaN-target rows are masked; missing
supervision raises (the LossRouter catches this and skips the term).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F


def lens_survival_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    n_bins: int,
    integration_time: float,
) -> torch.Tensor:
    """Discrete-time NLL on ``outputs['hazard']`` against ``batch['event_time']``
    + ``batch['event_observed']``.

    Consumes canonical-dict keys:
        outputs['hazard']           — (B, K) per-bin hazard
        outputs['survival_curve']   — (B, K) per-bin survival prob
        outputs['cif_per_event']    — (B, n_events, K) CIF
    Consumes batch keys:
        batch['event_time']         — (B,) months
        batch['event_observed']     — (B,) 0/1

    Returns
    -------
    Scalar negative log-likelihood. Censored rows contribute log(S(t)).
    """
    et = batch.get("event_time")
    eo = batch.get("event_observed")
    if et is None or eo is None:
        raise ValueError("lens_survival_loss requires event_time + event_observed")
    hazard = outputs["hazard"]                          # (B, K)
    surv = outputs["survival_curve"]                    # (B, K)
    # Bin assignment (NO synthetic data — pure index math).
    bins = torch.linspace(0.0, integration_time, n_bins + 1, device=hazard.device)
    bin_idx = torch.bucketize(et, bins[1:-1])           # (B,)
    eps = 1e-8
    # event-observed: -log haz at bin - log surv up to that bin
    # censored: -log surv at bin
    log_h = torch.log(hazard.gather(1, bin_idx[:, None]).squeeze(-1) + eps)
    log_s = torch.log(surv.gather(1, bin_idx[:, None]).squeeze(-1) + eps)
    obs = eo.float()
    nll = -(obs * log_h + (1.0 - obs) * log_s + log_s)
    return nll.mean()


def hitting_time_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> torch.Tensor:
    """Penalises mismatch between predicted hitting-time CDF and observed time.

    Consumes:
        outputs['hitting_cdf']      — (B, T) CDF over the SDE grid
        outputs['hitting_mean_tau'] — (B,) predicted mean hitting time
        outputs['t_grid']           — (T,) integration grid
        batch['event_time']         — (B,) observed time-to-event
        batch['event_observed']     — (B,) 0/1

    Loss = MSE(mean_tau, event_time) over observed rows only.
    """
    et = batch.get("event_time")
    eo = batch.get("event_observed")
    if et is None or eo is None:
        raise ValueError("hitting_time_loss requires event_time + event_observed")
    mean_tau = outputs["hitting_mean_tau"]
    obs_mask = eo.bool()
    if obs_mask.sum() == 0:
        return torch.zeros((), device=mean_tau.device, dtype=mean_tau.dtype)
    return F.mse_loss(mean_tau[obs_mask], et[obs_mask])


def basin_transition_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> torch.Tensor:
    """CE on terminal basin assignment vs ``batch['resistance_label']``.

    Consumes:
        outputs['basin_probs']         — (B, T, n_basins)
        batch['resistance_label']      — (B,) integer in [0, n_basins)
    """
    rl = batch.get("resistance_label")
    if rl is None:
        raise ValueError("basin_transition_loss requires resistance_label")
    basin_probs = outputs["basin_probs"]               # (B, T, n_basins)
    terminal = basin_probs[:, -1, :]                   # (B, n_basins)
    log_probs = torch.log(terminal.clamp(min=1e-8))
    targets = rl.long()
    valid = (targets >= 0) & (targets < terminal.shape[-1])
    if valid.sum() == 0:
        return torch.zeros((), device=terminal.device, dtype=terminal.dtype)
    return F.nll_loss(log_probs[valid], targets[valid])


def sde_path_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    metric: str = "mmd",
    sigma: float = 1.0,
) -> torch.Tensor:
    """Distribution-matching between predicted terminal state and
    ``batch['future_state']`` — MMD-RBF by default.

    Consumes:
        outputs['z_traj']           — (B, T, d_latent)
        batch['future_state']       — (B, d_latent) observed future
    """
    fs = batch.get("future_state")
    if fs is None:
        raise ValueError("sde_path_loss requires future_state")
    z_T = outputs["z_traj"][:, -1, :]                  # (B, d_latent)
    target = fs
    if target.shape[-1] != z_T.shape[-1]:
        target = target[..., : z_T.shape[-1]]
    # Re-use _mmd_rbf from mortfm_losses to avoid duplication.
    from resistancemap.training.mortfm_losses import _mmd_rbf
    if metric == "mse":
        return F.mse_loss(z_T, target)
    return _mmd_rbf(z_T, target, sigma=sigma)


def canonical_trajectory_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> torch.Tensor:
    """Composite of ``sde_path_loss`` + ``basin_transition_loss`` for stage E.

    This is the head-line trajectory term reported in the training history.
    The LossRouter still records each sub-term in the audit; this composite
    is the value plotted in the v19 dashboard.
    """
    a = sde_path_loss(outputs, batch)
    try:
        b = basin_transition_loss(outputs, batch)
    except ValueError:
        b = torch.zeros((), device=a.device, dtype=a.dtype)
    return a + b
```

**Returned-dict shape** (when wrapped via `LossRouter.compute`):

```
total:      torch.Tensor (scalar)
components: {"survival": float, "hitting": float, "trajectory": float,
             "sde_path": float, "basin_transition": float}
audit:      [LossAuditEntry(...), ...]
```

---

## 4. Proposed change to `resistancemap/training/loss_router.py`

The current router (`loss_router.py:79-84`) only checks "is this key present
in the batch". It does **not** call `MORTBatch.validate_for_loss(...)`, so
loss-specific dtype/shape errors still bubble up at runtime as opaque tensor
errors. Proposed change:

* Add an optional `validator_fn: Optional[Callable[[Mapping, str], None]] = None`
  kwarg to `LossRouter.__init__`.
* In `_has_required`, after the "missing keys" check, call
  `validator_fn(batch, name)` if provided. The default validator the canonical
  trainer wires in:

```python
def _mortbatch_validator(batch_mapping: Mapping[str, Any], loss_name: str) -> None:
    # Reconstruct a thin MORTBatch view just to call .validate_for_loss.
    # We do this because LossRouter intentionally operates on a Mapping,
    # not a MORTBatch — that keeps the router model-agnostic.
    name_map = {
        "survival":          "survival",
        "hitting":           "survival",
        "trajectory":        "trajectory",
        "sde_path":          "trajectory",
        "basin_transition":  "resistance_state",
        "drug":              "drug_response",
    }
    canonical = name_map.get(loss_name)
    if canonical is None:
        return
    from resistancemap.mortfm.schemas import MORTBatch
    # Build a transient MORTBatch only with the supervision fields the
    # validator inspects — cheap, no tensor allocation.
    mb = MORTBatch(
        event_time=batch_mapping.get("event_time"),
        event_observed=batch_mapping.get("event_observed"),
        future_state=batch_mapping.get("future_state"),
        resistance_label=batch_mapping.get("resistance_label"),
        drug_response=batch_mapping.get("drug_response"),
    )
    mb.validate_for_loss(canonical)
```

`LossAuditEntry.skipped_reason` should record the validator's error message,
so the run-time auditor can distinguish "key missing" from "key present but
malformed".

---

## 5. Test plan: split + new tests

### 5.1 Split `tests/mortfm/test_mortfm_trainer_smoke.py`

* `tests/mortfm/test_legacy_trainer_smoke.py` — **verbatim copy** of the
  existing file, renamed and with the file docstring updated to note
  "exercises forward_legacy(); pinned for backward-compat". The existing
  five tests stay unchanged:
  * `test_smoke_stage_A`
  * `test_smoke_stage_F_survival`
  * `test_smoke_stage_E_trajectory`
  * `test_smoke_checkpoint_roundtrip`
  * `test_smoke_full_forward_emits_all_heads`

* `tests/mortfm/test_canonical_trainer_smoke.py` — new file. Mirrors the
  same fixture-building (`_mk_snap`, `_cohort`, `_debug_cfg`) but uses
  `CanonicalMORTFMTrainer` and asserts canonical-dict component keys:

```python
def test_canonical_smoke_stage_F_survival(tmp_path):
    cfg = _debug_cfg(tmp_path)
    pairs = _cohort()
    _, train_loader, val_loader, _ = make_data_module(pairs, batch_size=2)
    model = MORTFM(cfg, ...)
    trainer = CanonicalMORTFMTrainer(
        model, cfg, train_loader=train_loader, val_loader=val_loader,
        device=torch.device("cpu"),
    )
    m = trainer.fit_stage("F", n_epochs=1)
    assert "survival" in m[0].components  # the LENS survival, not legacy.

def test_canonical_smoke_stage_E_trajectory(tmp_path):
    ...
    m = trainer.fit_stage("E", n_epochs=1)
    assert "sde_path" in m[0].components or "trajectory" in m[0].components

def test_canonical_no_legacy_call(monkeypatch, tmp_path):
    """The canonical trainer must NEVER call model.forward_legacy."""
    called = {"n": 0}
    def _spy(*a, **kw):
        called["n"] += 1
        raise AssertionError("CanonicalMORTFMTrainer must not call forward_legacy")
    cfg = _debug_cfg(tmp_path)
    pairs = _cohort()
    _, train_loader, val_loader, _ = make_data_module(pairs, batch_size=2)
    model = MORTFM(cfg, ...)
    monkeypatch.setattr(model, "forward_legacy", _spy)
    trainer = CanonicalMORTFMTrainer(model, cfg, train_loader=train_loader,
                                     val_loader=val_loader, device=torch.device("cpu"))
    trainer.fit_stage("E", n_epochs=1)
    assert called["n"] == 0
```

### 5.2 New `tests/mortfm/test_canonical_trainer.py`

Unit-level tests for the canonical trainer scaffolding:

* `test_default_router_has_lens_keys` — asserts router enables
  `{survival, hitting, trajectory, sde_path, basin_transition}`.
* `test_build_canonical_kwargs_zero_pad_clinical_drug` — when batch has
  no clinical/drug fields, kwargs are `None` (model's forward fills zeros).
* `test_strict_coverage_raise_on_stage_F_without_event_time` — assembles a
  batch with all `event_time=None`, runs `fit_stage("F", 1)`, asserts
  `MissingSupervisionError` is raised.
* `test_acceptance_gate_blocks_strong_claim_on_small_cohort` — same as legacy.

### 5.3 New `tests/mortfm/test_canonical_loss_router.py`

* `test_router_skips_survival_when_event_observed_missing` — audit entry's
  `skipped_reason` includes `"event_observed"`.
* `test_router_validator_fn_invokes_validate_for_loss` — monkeypatch
  `MORTBatch.validate_for_loss` and assert it is called for `survival`
  but not for `recon`.
* `test_router_records_loss_fn_exception_in_audit` — make a loss_fn raise
  and assert `audit[i].skipped_reason.startswith("loss_fn raised:")`.

---

## 6. File-change list

| Path | Change | New / Modified |
| --- | --- | --- |
| `ResistanceMap/resistancemap/mortfm/canonical_trainer.py` | Canonical v19+ trainer (see §2) | **New** |
| `ResistanceMap/resistancemap/training/canonical_mortfm_losses.py` | LENS-dict-consuming losses (see §3) | **New** |
| `ResistanceMap/resistancemap/training/loss_router.py` | Add `validator_fn` kwarg + `_mortbatch_validator` (see §4) | Modified |
| `ResistanceMap/resistancemap/mortfm/trainer.py` | Add a no-op `legacy: bool = True` flag on `MORTFMTrainer.__init__`; if set to `False`, raise `RuntimeError("use CanonicalMORTFMTrainer for the canonical surface")`. Rename class to `LegacyMORTFMTrainer` and keep `MORTFMTrainer = LegacyMORTFMTrainer` as a deprecation alias. | Modified |
| `ResistanceMap/resistancemap/mortfm/__init__.py` | Export both `LegacyMORTFMTrainer` and `CanonicalMORTFMTrainer`; keep `MORTFMTrainer` as alias | Modified |
| `ResistanceMap/tests/mortfm/test_mortfm_trainer_smoke.py` | Rename to `test_legacy_trainer_smoke.py` (verbatim) | **Renamed** |
| `ResistanceMap/tests/mortfm/test_canonical_trainer_smoke.py` | New smoke tests for canonical trainer (see §5.1) | **New** |
| `ResistanceMap/tests/mortfm/test_canonical_trainer.py` | Unit tests for canonical trainer scaffolding (see §5.2) | **New** |
| `ResistanceMap/tests/mortfm/test_canonical_loss_router.py` | LossRouter validator + audit-trail tests (see §5.3) | **New** |
| `ResistanceMap/scripts/mortfm/05_train_stage_E.py` (and 06/07/08 if present) | Swap `MORTFMTrainer(...)` for `CanonicalMORTFMTrainer(...)`; no other change | Modified |

---

## 7. Backward-compatibility plan for `forward_legacy()`

* `forward_legacy()` stays in `model.py` indefinitely. It is the only way
  old (v15/v16/v17) checkpoints can resume training without surgery.
* The renamed `LegacyMORTFMTrainer` continues to call `forward_legacy`
  unchanged. v18 smoke tests (renamed to `test_legacy_trainer_smoke.py`)
  continue to gate that path.
* `tests/mortfm/test_canonical_forward.py:90-96` already asserts
  `MORTFM.forward is not MORTFM.forward_legacy`; do NOT relax that
  contract.
* Deprecation warning on `LegacyMORTFMTrainer.__init__` (one-shot
  `warnings.warn(..., DeprecationWarning, stacklevel=2)`) — points users to
  `CanonicalMORTFMTrainer`. Suppress in the legacy smoke-test file via
  `warnings.filterwarnings`.
* Three-version sunset: v19 keeps the legacy trainer alive, v20 marks it
  `DeprecationWarning -> FutureWarning`, v21 removes it. The legacy heads
  themselves stay in `resistancemap.legacy.v15_planned_mortfm.*` for as
  long as checkpoints exist on disk.

---

## 8. Definition of done

1. `from resistancemap.mortfm import CanonicalMORTFMTrainer` works.
2. `CanonicalMORTFMTrainer.fit_stage("F", n_epochs=1)` on the smoke fixture
   runs without ever invoking `model.forward_legacy` (verified by the spy
   test in §5.1).
3. `tests/mortfm/test_canonical_trainer_smoke.py`,
   `tests/mortfm/test_canonical_trainer.py`,
   `tests/mortfm/test_canonical_loss_router.py` all pass.
4. `tests/mortfm/test_canonical_forward.py` (existing) and
   `tests/mortfm/test_legacy_trainer_smoke.py` (renamed) both still pass.
5. `MissingSupervisionError` is raised on a stage-F run where
   `event_observed` is None for >10% of batches.
6. No new file contains `np.random`, `torch.randn` for label/metric
   construction, or any hardcoded metric value. (Verified by `grep` in CI:
   `! grep -RnE "np\.random|torch\.randn.*label|c_index *= *0" resistancemap/training/canonical_mortfm_losses.py resistancemap/mortfm/canonical_trainer.py`.)
7. README RUNS table updated with a v19.0 row pointing at the canonical
   trainer.

---

## 9. v19 caller diff — where exactly the swap happens

The line-precise insertion point in the **legacy** `trainer.py` for the v19
caller (if a user wants to opt-in from the existing class):

```
ResistanceMap/resistancemap/mortfm/trainer.py:226 (immediately after the
v18 comment block, before `prediction = self.model.forward_legacy(...)`)

+        if getattr(self, "canonical_trainer", None) is not None and stage in ("E", "F", "G", "H"):
+            # v19 hand-off: route patient-level stages through the
+            # canonical trainer's per-batch path. The legacy code below
+            # this guard runs only when canonical_trainer is None.
+            return self.canonical_trainer._losses_for_batch(batch, stage)
         prediction = self.model.forward_legacy(
             batch,
             ...
         )
```

The corresponding `fit_stage` shim (line `trainer.py:482`) becomes:

```
         for ep in range(n_epochs):
-            out.append(self.train_epoch(stage, ep))
+            if getattr(self, "canonical_trainer", None) is not None and stage in ("E", "F", "G", "H"):
+                out.extend(self.canonical_trainer.fit_stage(stage, 1))
+            else:
+                out.append(self.train_epoch(stage, ep))
```

In **green-field** v19 code the swap is just:

```python
# was:
trainer = MORTFMTrainer(model, cfg, train_loader=..., val_loader=...)
trainer.fit_stage("F", n_epochs=25)

# now:
trainer = CanonicalMORTFMTrainer(model, cfg, train_loader=..., val_loader=...)
trainer.fit_stage("F", n_epochs=25)
```
