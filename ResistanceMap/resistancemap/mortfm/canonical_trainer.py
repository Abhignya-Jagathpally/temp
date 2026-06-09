"""
resistancemap/mortfm/canonical_trainer.py
=========================================
Canonical MORT-FM trainer scaffolding (v19+).

Sibling — not a replacement — of :class:`MORTFMTrainer`. The legacy trainer
stays alive behind ``forward_legacy()`` for backward compatibility with v15/v17
checkpoints; this canonical trainer drives the v18 LENS dict surface
exclusively via :meth:`MORTFM.forward` (no ``forward_legacy`` call ever
escapes from this module).

Key differences from the legacy trainer
---------------------------------------
* Patient-level stages (E/F/G/H) call ``self.model(batch, clinical=...,
  drug=..., graph_emb=None)`` and consume the canonical-dict surface
  defined in :mod:`resistancemap.training.canonical_mortfm_losses`.
* Loss assembly is delegated to
  :func:`~resistancemap.training.canonical_mortfm_losses.assemble_canonical_loss`,
  which returns per-term ``n_supervised`` counts.
* The v18.2 strict-supervision policy (``MissingSupervisionError``) is enforced
  the same way as the legacy trainer: a stage with required supervision must
  have at least one supervised batch in every epoch.

No synthetic or random labels are constructed anywhere in this module. Random
weight initialisation on ``nn.Module`` sub-layers (which is standard for
trainable parameters) is the only stochasticity present.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

# Re-use the legacy MissingSupervisionError so external callers only need
# to catch one exception type regardless of which trainer they invoked.
from resistancemap.mortfm.trainer import (
    MissingSupervisionError,
    STRICT_SUPERVISION_COVERAGE,
)
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import MORTBatch, MORTFMConfig
from resistancemap.training.canonical_mortfm_losses import (
    STAGE_LOSS_SPEC,
    assemble_canonical_loss,
    latent_future_state_loss,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CanonicalMORTFMTrainer",
    "MissingSupervisionError",
    "CanonicalEpochMetrics",
    "STAGE_VALIDATE_FOR_LOSS_KEY",
]


#: Maps each canonical patient-level stage to the ``loss_name`` argument
#: passed to :meth:`MORTBatch.validate_for_loss`. Stages whose required
#: term isn't in the validator's vocabulary (e.g. ``basin_transition``)
#: simply skip the validate call — the per-term ``n_supervised`` audit
#: downstream still catches missing supervision.
STAGE_VALIDATE_FOR_LOSS_KEY: Dict[str, Optional[str]] = {
    "E": "trajectory",  # also "latent_future_state" when future_snapshot_batch is provided
    "F": "survival",
    "G": None,          # basin_transition has no validate_for_loss key (no labelled target)
    "H": "survival",
}


@dataclass
class CanonicalEpochMetrics:
    """Per-epoch metrics emitted by :meth:`CanonicalMORTFMTrainer.fit_stage`."""

    epoch: int = 0
    stage: str = ""
    loss_total: float = float("nan")
    n_batches: int = 0
    per_term_n_supervised: Dict[str, int] = field(default_factory=dict)
    wall_time_s: float = 0.0
    val_loss: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "stage": self.stage,
            "loss_total": self.loss_total,
            "n_batches": self.n_batches,
            "per_term_n_supervised": dict(self.per_term_n_supervised),
            "wall_time_s": self.wall_time_s,
            "val_loss": self.val_loss,
        }


class CanonicalMORTFMTrainer:
    """Drives MORT-FM through the canonical LENS dict surface.

    Parameters
    ----------
    model :
        A :class:`MORTFM` instance.
    cfg :
        Either a :class:`MORTFMConfig` or a ``dict`` with the same fields.
        Only the loss-weight, ``integration_time``, ``survival_n_bins``,
        ``lr``, ``weight_decay``, ``grad_clip``, and ``checkpoint_dir``
        attributes are read.
    train_loader, val_loader :
        DataLoaders emitting :class:`MORTBatch`.
    device :
        Torch device string ("cpu", "cuda", "cuda:0", ...). Defaults to
        "cuda" if available else "cpu".
    strict :
        When ``True`` (default) the v18.2 strict-supervision policy is
        enforced: if no batch in an epoch produced supervision for the
        stage's required term, the trainer raises
        :class:`MissingSupervisionError`.
    weights :
        Optional override map for canonical-loss term weights (keys:
        ``trajectory``, ``sde_path``, ``basin_transition``, ``survival``,
        ``hitting``). When omitted the trainer pulls the appropriate
        ``lambda_*`` values from ``cfg``.
    graph_emb_fn :
        Optional ``Callable[[MORTBatch], Tensor]`` returning the
        precomputed ``(B, d_graph)`` biological-graph embedding. When
        ``None`` the model's internal ``LatentToGraphProjector`` is used.
    """

    def __init__(
        self,
        model: MORTFM,
        cfg: Any,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cpu",
        *,
        strict: bool = True,
        strict_supervision: Optional[bool] = None,
        weights: Optional[Dict[str, float]] = None,
        graph_emb_fn: Optional[Callable[[MORTBatch], torch.Tensor]] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.model.to(self.device)
        # ``strict_supervision`` is the v19 spec name for the v18 ``strict``
        # flag. When both are given, the more explicit one wins.
        if strict_supervision is not None:
            self.strict = bool(strict_supervision)
        else:
            self.strict = bool(strict)
        self.strict_supervision = self.strict  # alias for v19 callers
        self.graph_emb_fn = graph_emb_fn

        # Loss weights (canonical default — fall back to 1.0 if cfg lacks them).
        self.weights = weights or self._default_weights(cfg)

        # Optimiser — AdamW with sensible defaults pulled from cfg.
        lr = self._cfg_value(cfg, "lr", 1e-4)
        wd = self._cfg_value(cfg, "weight_decay", 1e-5)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=wd,
        )
        self.grad_clip = float(self._cfg_value(cfg, "grad_clip", 1.0))

        # Persistent run history.
        self.history: List[CanonicalEpochMetrics] = []
        checkpoint_dir = self._cfg_value(cfg, "checkpoint_dir", None)
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
        if isinstance(cfg, dict):
            return cfg.get(name, default)
        return getattr(cfg, name, default)

    def _default_weights(self, cfg: Any) -> Dict[str, float]:
        """Pull canonical-loss weights from cfg's ``lambda_*`` fields."""
        return {
            "trajectory":       float(self._cfg_value(cfg, "lambda_trajectory", 1.0)),
            "sde_path":         float(self._cfg_value(cfg, "lambda_trajectory", 1.0)),
            "basin_transition": float(self._cfg_value(cfg, "lambda_pathway", 0.3)),
            "survival":         float(self._cfg_value(cfg, "lambda_survival", 1.0)),
            "hitting":          float(self._cfg_value(cfg, "lambda_survival", 1.0)),
        }

    # ------------------------------------------------------------------
    # Per-batch loss assembly (canonical path)
    # ------------------------------------------------------------------

    def _canonical_forward(self, batch: MORTBatch) -> Dict[str, torch.Tensor]:
        """Call the canonical ``MORTFM.forward`` — never ``forward_legacy``.

        The clinical / drug / graph_emb kwargs are pulled directly from
        ``batch``. ``graph_emb=None`` is explicitly forwarded so the model
        decides whether to invoke its internal ``LatentToGraphProjector``.

        NaN handling: the dataloader's drug / clinical tensors may contain
        ``NaN`` placeholders for unfilled slots (see the v15 collate function).
        These would corrupt the LENS SDE if forwarded raw, so we mask NaN
        entries to zero here. This is **structural defensiveness**, not label
        fabrication — the actual supervision (event_time, event_observed,
        resistance_label, future_state) is preserved unchanged.
        """
        clinical = self._clean_clinical(batch.clinical)
        drug = self._drug_padded(batch)

        graph_emb: Optional[torch.Tensor] = None
        if self.graph_emb_fn is not None:
            graph_emb = self.graph_emb_fn(batch)

        # Canonical entry point — DO NOT call forward_legacy from here.
        return self.model(batch, clinical=clinical, drug=drug, graph_emb=graph_emb)

    def _drug_padded(self, batch: MORTBatch) -> Optional[torch.Tensor]:
        """Match the LENS SDE drug width (model.d_drug). Clip or right-pad zeros.

        NaN cells in the raw drug tensor are zeroed so the SDE input stays
        well-conditioned; per-row presence is already encoded via
        ``batch.modality_mask['drug']``.
        """
        return self._fit_width(batch.drug, int(getattr(self.model, "d_drug", 8)))

    def _clean_clinical(self, clinical: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Match the LENS SDE clinical width (model.d_clinical); clip/pad, zero NaN.

        Width is read from the model so it tracks ``config.d_clinical`` (default
        5; 21 for the lab-first panel) instead of the old hardcoded 5.
        """
        return self._fit_width(clinical, int(getattr(self.model, "d_clinical", 5)))

    @staticmethod
    def _fit_width(t: Optional[torch.Tensor], width: int) -> Optional[torch.Tensor]:
        """Clip/right-pad ``t`` to exactly ``width`` along the last dim, zero NaN."""
        if t is None:
            return None
        if t.shape[-1] > width:
            t = t[..., :width]
        elif t.shape[-1] < width:
            pad = t.new_zeros((t.shape[0], width - t.shape[-1]))
            t = torch.cat([t, pad], dim=-1)
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def _trajectory_step(
        self,
        batch: MORTBatch,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Phase 1 latent trajectory step.

        Re-encodes ``batch.future_snapshot_batch`` through the SAME
        foundation encoder used for the baseline (under ``torch.no_grad``
        to act as a target network), then compares the SDE-evolved
        terminal latent against it using
        :func:`latent_future_state_loss`. Per-row Δt comes from
        ``batch.delta_t_days``.

        Returns the loss dict in the canonical
        ``{"loss": ..., "n_supervised": ...}`` shape. If ``future_snapshot_batch``
        or ``delta_t_days`` is missing, returns a zero-loss term with
        ``n_supervised=0`` — the strict-supervision audit will catch persistent
        absence.
        """
        z_traj = outputs["z_traj"]
        device = z_traj.device
        dtype = z_traj.dtype
        zero = torch.zeros((), device=device, dtype=dtype)
        if batch.future_snapshot_batch is None or batch.delta_t_days is None:
            return {
                "loss_traj_latent": zero,
                "loss_traj_mmd": zero,
                "loss": zero,
                "n_supervised": 0,
            }

        future_batch = batch.future_snapshot_batch.to(device)
        # Target-network trick: detach the encoder's view of the future so
        # gradients flow through the SDE/drift, not back into the foundation
        # encoder. Without this, the encoder could collapse both branches
        # into a constant to trivially satisfy the loss.
        with torch.no_grad():
            future_state = self.model.forward_foundation(future_batch)
        encoded_z_future = future_state.z0.detach().to(device=device, dtype=dtype)

        delta_t = batch.delta_t_days.to(device=device, dtype=dtype)
        return latent_future_state_loss(z_traj, encoded_z_future, delta_t)

    def _losses_for_batch(
        self,
        batch: MORTBatch,
        stage: str,
    ) -> Dict[str, Any]:
        """Compute the canonical loss bundle for one batch + stage.

        Returns the dict produced by ``assemble_canonical_loss``, augmented
        with the raw model ``outputs``. Raises :class:`MissingSupervisionError`
        if ``strict`` is on and the stage's required term reported
        ``n_supervised == 0`` for *this* batch — the per-epoch coverage check
        also runs upstream, but the per-batch check makes the failure mode
        immediately legible in tests.
        """
        if stage not in STAGE_LOSS_SPEC:
            raise ValueError(
                f"CanonicalMORTFMTrainer._losses_for_batch: stage {stage!r} "
                f"is not a canonical patient-level stage; expected one of "
                f"{tuple(STAGE_LOSS_SPEC)}. Foundation stages (A/B/C/D) must "
                f"be driven by the legacy trainer."
            )
        batch = batch.to(self.device)

        # v19 Bug B14 wire-up — ``MORTBatch.validate_for_loss`` fails fast
        # when the batch lacks the supervision a loss term will consume.
        # Strict mode propagates the error as ``MissingSupervisionError``;
        # non-strict mode logs at INFO and lets the per-term ``n_supervised``
        # audit downstream decide what to skip.
        validate_key = STAGE_VALIDATE_FOR_LOSS_KEY.get(stage)
        if validate_key is not None:
            try:
                batch.validate_for_loss(validate_key)
            except ValueError as exc:
                if self.strict:
                    raise MissingSupervisionError(
                        f"CanonicalMORTFMTrainer: stage {stage!r} batch failed "
                        f"validate_for_loss({validate_key!r}): {exc}"
                    ) from exc
                logger.info(
                    "CanonicalMORTFMTrainer: stage %r batch missing supervision "
                    "for %r (%s); term will be skipped (strict=False).",
                    stage, validate_key, exc,
                )

        outputs = self._canonical_forward(batch)
        # TODO(v19 Bug B14): replace this assemble_canonical_loss call with
        # `LossRouter.route(stage, batch)` once LossRouter exposes a
        # stage-aware ``route`` API. Today LossRouter only provides
        # ``compute(outputs, batch)`` which takes a Mapping-shaped batch and
        # does not understand the canonical STAGE_LOSS_SPEC — wiring it
        # would require either (a) adding a MORTBatch→Mapping shim plus a
        # `route(stage, batch)` method, or (b) refactoring the per-term
        # dispatch in canonical_mortfm_losses.assemble_canonical_loss into
        # LossRouter directly. Both are deferred to a focused LossRouter PR
        # so this validate_for_loss wireup stays minimal.
        bundle = assemble_canonical_loss(stage, outputs, batch, self.weights)

        # v19 Phase 1: for stage E, override the "trajectory" term with the
        # canonical latent loss if a follow-up snapshot is available. The
        # assembler's default trajectory term still consumes the
        # legacy `batch.future_state` slot for backward compatibility
        # (cell-line teacher latents); Phase 1 prefers the future-snapshot
        # path because the canonical loss is computed entirely in the
        # foundation encoder's latent space — no raw-RNA truncation.
        if (
            stage == "E"
            and "trajectory" in bundle["components"]
            and batch.future_snapshot_batch is not None
            and batch.delta_t_days is not None
        ):
            traj_sub = self._trajectory_step(batch, outputs)
            old_loss = bundle["components"]["trajectory"]["loss"]
            new_loss = traj_sub["loss"]
            weight = float(self.weights.get("trajectory", 1.0))
            bundle["loss_total"] = bundle["loss_total"] - weight * old_loss + weight * new_loss
            bundle["components"]["trajectory"] = traj_sub
            bundle["per_term_n_supervised"]["trajectory"] = int(traj_sub["n_supervised"])

        bundle["outputs"] = outputs
        if self.strict:
            required = bundle["required_term"]
            if required is not None and bundle["per_term_n_supervised"].get(required, 0) == 0:
                raise MissingSupervisionError(
                    f"CanonicalMORTFMTrainer: stage {stage!r} required term "
                    f"{required!r} reported n_supervised=0 for this batch. "
                    f"This is the v18.2 'raise, don't skip' policy."
                )
        return bundle

    # ------------------------------------------------------------------
    # Public training loops
    # ------------------------------------------------------------------

    def fit_stage(self, stage: str, n_epochs: int) -> List[Dict[str, Any]]:
        """Minimal train loop. Returns per-epoch dict metrics.

        Each per-epoch dict carries the keys ``epoch``, ``stage``,
        ``loss_total``, ``n_batches``, ``per_term_n_supervised``,
        ``wall_time_s``, ``val_loss``.
        """
        if stage not in STAGE_LOSS_SPEC:
            raise ValueError(
                f"CanonicalMORTFMTrainer.fit_stage: stage {stage!r} is not a "
                f"canonical patient-level stage; expected one of "
                f"{tuple(STAGE_LOSS_SPEC)}"
            )
        epoch_metrics: List[Dict[str, Any]] = []
        for ep in range(n_epochs):
            metric = self._train_one_epoch(stage, ep)
            if self.val_loader is not None:
                metric.val_loss = self.validate(stage)
            self.history.append(metric)
            epoch_metrics.append(metric.to_dict())
            logger.info(
                "canonical stage=%s epoch=%d loss=%.4f val=%.4f n_batches=%d sup=%s",
                stage, ep, metric.loss_total, metric.val_loss,
                metric.n_batches, metric.per_term_n_supervised,
            )
        return epoch_metrics

    def _train_one_epoch(self, stage: str, epoch: int) -> CanonicalEpochMetrics:
        self.model.train()
        t0 = time.time()
        sum_loss = 0.0
        n_batches = 0
        sup_totals: Dict[str, int] = {}
        sup_at_least_once: Dict[str, int] = {}

        required = STAGE_LOSS_SPEC[stage]
        for term in required:
            sup_totals[term] = 0
            sup_at_least_once[term] = 0

        for batch in self.train_loader:
            try:
                bundle = self._losses_for_batch(batch, stage)
            except MissingSupervisionError:
                # Strict per-batch policy already fired; propagate.
                if self.strict:
                    raise
                continue

            total = bundle["loss_total"]
            if not total.requires_grad:
                # The model produced an all-zero loss (no supervision and
                # nothing differentiable). Skip the optimiser step.
                pass
            else:
                self.optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            sum_loss += float(total.detach().item())
            n_batches += 1
            for term, n_sup in bundle["per_term_n_supervised"].items():
                sup_totals[term] = sup_totals.get(term, 0) + int(n_sup)
                if n_sup > 0:
                    sup_at_least_once[term] = sup_at_least_once.get(term, 0) + 1

        # v18.2 strict-supervision check at the epoch boundary too.
        if self.strict and stage in STRICT_SUPERVISION_COVERAGE:
            required_term = {
                "E": "trajectory", "F": "survival", "G": "basin_transition",
            }.get(stage)
            if required_term is not None and sup_at_least_once.get(required_term, 0) == 0:
                raise MissingSupervisionError(
                    f"CanonicalMORTFMTrainer epoch {epoch} stage {stage!r}: "
                    f"no batch produced supervision for required term "
                    f"{required_term!r}. Refusing to advance an "
                    f"under-supervised stage (v18.2 policy)."
                )

        avg_loss = sum_loss / max(n_batches, 1)
        return CanonicalEpochMetrics(
            epoch=epoch,
            stage=stage,
            loss_total=avg_loss,
            n_batches=n_batches,
            per_term_n_supervised=sup_totals,
            wall_time_s=time.time() - t0,
        )

    @torch.no_grad()
    def validate(self, stage: str) -> float:
        """Run the validation loader (no grads). Returns the mean total loss.

        Returns NaN if ``val_loader`` is None.
        """
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        sum_loss = 0.0
        n_batches = 0
        for batch in self.val_loader:
            batch = batch.to(self.device)
            outputs = self._canonical_forward(batch)
            bundle = assemble_canonical_loss(stage, outputs, batch, self.weights)
            sum_loss += float(bundle["loss_total"].detach().item())
            n_batches += 1
        return sum_loss / max(n_batches, 1)
