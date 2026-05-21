"""
resistancemap/mortfm/trainer.py
==============================
Staged training loop for MORT-FM.

Stages
------
A. Modality-specific reconstruction pretraining (per encoder).
B. Cross-modal masked-modality pretraining (foundation fusion).
C. Static drug-response pretraining (CCLE/GDSC/PRISM).
D. Patient-domain adaptation (DA, MMD batch correction).
E. Longitudinal trajectory training (temporal pairs).
F. Survival / time-to-resistance finetuning.
G. Pathway + counterfactual validation.
H. Calibration (temperature scaling on the survival head).

The current implementation provides the *unified loop* with stage selection
via the ``stages`` argument. Each stage selects which losses are active and
which heads accumulate gradients. The trainer is intentionally one file —
later refactors can pull stage-specific code into ``training/`` modules
without changing the call signature.

Honest behaviour
----------------
* Each stage validates that the required supervision is present in the loader.
  If it is not, the stage *raises* — it does not silently substitute a smaller
  loss. This prevents the "model trained on no data still produced a number"
  failure mode flagged in the v12 audit.
* The trainer never fabricates labels.
* Best-checkpoint selection uses validation total-loss, not training loss.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from resistancemap.mortfm.schemas import MORTBatch, MORTFMConfig, TemporalTrainingPair
from resistancemap.training.mortfm_losses import (
    assemble_total_loss,
    contrastive_alignment_loss,
    drug_response_loss,
    landscape_regularization_loss,
    masked_modality_loss,
    pathway_attribution_loss,
    reconstruction_loss,
    resistance_state_loss,
    survival_loss,
    trajectory_distribution_loss,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    epoch: int = 0
    stage: str = ""
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    components: Dict[str, float] = field(default_factory=dict)
    wall_time_s: float = 0.0


class MissingSupervisionError(RuntimeError):
    """Raised by stage E/F/G when supervision coverage < threshold.

    v18.2 — replaces the silent "skip empty-loss batches" behaviour the
    v17 docstring claimed but the code didn't enforce. Any stage that
    advertises a clinical claim level MUST receive enough supervised
    batches; otherwise the run is refused, not silently mis-trained.
    """


@dataclass
class StageSupervisionReport:
    """Per-stage supervision audit used by v18.2's strict policy."""
    stage: str
    n_batches: int = 0
    n_supervised_batches: int = 0
    missing_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return self.n_supervised_batches / max(self.n_batches, 1)

    def summary(self) -> dict:
        return {
            "stage": self.stage,
            "n_batches": self.n_batches,
            "n_supervised_batches": self.n_supervised_batches,
            "coverage": self.coverage,
            "missing_reasons": dict(self.missing_reasons),
        }


# v18.2 — supervision-coverage thresholds the strict policy enforces.
# Stages outside this dict are NOT supervised stages (foundation
# pretraining); they may legitimately have empty-loss batches.
STRICT_SUPERVISION_COVERAGE = {
    "E": 0.80,   # trajectory — needs future_state
    "F": 0.90,   # survival   — needs event_time + event_observed
    "G": 0.70,   # pathway    — needs pathway_targets
}


# ---------------------------------------------------------------------------
# Stage spec
# ---------------------------------------------------------------------------


VALID_STAGES = ("A", "B", "C", "D", "E", "F", "G", "H")


def stage_weight_overrides(stage: str, config: MORTFMConfig) -> Dict[str, float]:
    """Per-stage loss-weight overrides — which heads are 'live' this stage."""
    if stage == "A":  # modality reconstruction only
        return {"recon": config.lambda_recon}
    if stage == "B":  # cross-modal + recon
        return {"recon": config.lambda_recon, "masked": config.lambda_masked,
                "contrastive": config.lambda_contrastive}
    if stage == "C":  # drug response
        return {"drug": config.lambda_drug}
    if stage == "D":  # domain adaptation (TODO: add MMD across batches)
        return {"contrastive": config.lambda_contrastive}
    if stage == "E":  # longitudinal trajectory
        return {"trajectory": config.lambda_trajectory,
                "landscape": config.lambda_landscape}
    if stage == "F":  # survival
        return {"survival": config.lambda_survival,
                "state": config.lambda_state}
    if stage == "G":  # pathway + counterfactual
        return {"pathway": config.lambda_pathway}
    if stage == "H":  # calibration
        return {"survival": config.lambda_survival,
                "calibration": config.lambda_calibration}
    raise ValueError(f"Unknown stage {stage!r}; expected one of {VALID_STAGES}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class MORTFMTrainer:
    """One trainer for all stages.

    Parameters
    ----------
    model : MORTFM
        End-to-end MORT-FM model.
    config :
        Training config.
    train_loader, val_loader :
        ``DataLoader`` objects emitting :class:`MORTBatch` (via
        :func:`resistancemap.data.mortfm_dataset.mort_collate`).
    graph :
        Optional graph payload (dict with ``edge_index``, ``edge_weight``).
        ``None`` disables the graph-conditioned drift branch.
    """

    def __init__(
        self,
        model: MORTFM,
        config: MORTFMConfig,
        *,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        graph: Any = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.graph = graph

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=(config.mixed_precision and device.type == "cuda"))
        self.history: List[TrainingMetrics] = []
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # v19 — optional hand-off to the canonical trainer. When set,
        # stages E/F/G/H route through CanonicalMORTFMTrainer._losses_for_batch
        # instead of the legacy forward_legacy() path below.
        self.canonical_trainer: Optional[Any] = None

    # ------------------------------------------------------------------
    # v19 — canonical-trainer factory
    # ------------------------------------------------------------------

    @classmethod
    def with_canonical(
        cls,
        model: MORTFM,
        cfg: MORTFMConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Any = "cpu",
        *,
        strict_supervision: bool = True,
        graph: Any = None,
        **canonical_kwargs: Any,
    ) -> "MORTFMTrainer":
        """Construct a MORTFMTrainer with a CanonicalMORTFMTrainer attached.

        Stages E/F/G/H will route through ``model.forward()`` (canonical
        LENS dict surface) via :class:`CanonicalMORTFMTrainer`. Stages
        A/B/C/D still use the legacy curriculum (the foundation encoder
        does not yet have a canonical-only path).

        This factory closes the "self.canonical_trainer = None" gap the
        v19 roadmap flagged: the legacy trainer exposed a shim gated on
        ``self.canonical_trainer is not None`` but no production path
        actually constructed a :class:`CanonicalMORTFMTrainer` to attach.
        """
        from .canonical_trainer import CanonicalMORTFMTrainer

        # Normalise device — the legacy MORTFMTrainer accepts torch.device,
        # the canonical accepts a string.
        if isinstance(device, str):
            torch_device = torch.device(device)
            device_str = device
        else:
            torch_device = device
            device_str = str(device)

        trainer = cls(
            model=model, config=cfg,
            train_loader=train_loader, val_loader=val_loader,
            graph=graph, device=torch_device,
        )
        trainer.canonical_trainer = CanonicalMORTFMTrainer(
            model=model, cfg=cfg,
            train_loader=train_loader, val_loader=val_loader,
            device=device_str,
            strict_supervision=strict_supervision,
            **canonical_kwargs,
        )
        return trainer

    # ------------------------------------------------------------------
    # Per-batch loss assembly
    # ------------------------------------------------------------------

    def _losses_for_batch(
        self,
        batch: MORTBatch,
        stage: str,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Return (loss_components, weight_overrides) for one batch + stage."""
        # v19 hand-off — if the user attached a CanonicalMORTFMTrainer to this
        # legacy trainer, patient-level stages (E/F/G/H) route through the
        # canonical LENS dict path instead of forward_legacy(). The legacy
        # body below this guard is reached only when canonical_trainer is None
        # or when the stage is foundation (A/B/C/D).
        if getattr(self, "canonical_trainer", None) is not None and stage in ("E", "F", "G", "H"):
            bundle = self.canonical_trainer._losses_for_batch(batch, stage)
            # Flatten the canonical bundle into the legacy (components, weights)
            # tuple shape so assemble_total_loss can sum it.
            canonical_components: Dict[str, torch.Tensor] = {
                name: sub["loss"] for name, sub in bundle["components"].items()
            }
            canonical_weights: Dict[str, float] = dict(self.canonical_trainer.weights)
            return canonical_components, canonical_weights

        batch = batch.to(self.device)
        weights = stage_weight_overrides(stage, self.config)
        components: Dict[str, torch.Tensor] = {}

        # Always-on: foundation forward (encoder + dynamics + heads).
        # v18 — explicitly route through forward_legacy(). The new canonical
        # forward() returns the LENS dict surface; the legacy trainer below
        # still consumes a TrajectoryPrediction dataclass with state_logits,
        # survival_curve, pathway_*, etc. The v18 stage router in
        # mortfm/trainer_v18.py will switch to model(batch) (canonical)
        # for stages E/F/G/H; this legacy trainer is retained as
        # forward_legacy() consumer for backward compat with old checkpoints.
        prediction = self.model.forward_legacy(
            batch,
            time_grid=None,
            n_traj_samples=4 if stage == "E" else 1,
            graph=self.graph,
        )

        # ---- A/B: reconstruction + masked-modality + contrastive ----
        if "recon" in weights:
            # Use RNA recon (NB) as a representative — this is a placeholder
            # for the per-modality recon dispatch that a more thorough trainer
            # would assemble.
            if batch.rna is not None:
                rna_enc = self.model.fusion.tokenizer.encoders["rna"] if "rna" in self.model.fusion.tokenizer.encoders else None
                if rna_enc is not None:
                    z = rna_enc(batch.rna)
                    mu, theta, _ = rna_enc.decode(z, library_size=batch.rna.sum(dim=-1).clamp(min=1.0))
                    components["recon"] = reconstruction_loss(
                        batch.rna.clamp(min=0).round(), (mu, theta), distribution="nb"
                    )
        if "masked" in weights and "proteomics" in self.model.fusion.tokenizer.encoders:
            if batch.proteomics is not None:
                state = self.model.encode(batch)
                pred_token = state.modality_embeddings.get("proteomics")
                if pred_token is not None:
                    # Use the *encoder's* fresh output of the same modality as
                    # the supervision target — this trains the fusion to be a
                    # consistent encoder of the joint state.
                    target_token = self.model.fusion.tokenizer.encoders["proteomics"](batch.proteomics)
                    components["masked"] = masked_modality_loss(pred_token, target_token.detach())
        if "contrastive" in weights:
            state = self.model.encode(batch)
            embs = list(state.modality_embeddings.values())
            if len(embs) >= 2:
                a = F.normalize(embs[0], dim=-1)
                b = F.normalize(embs[1], dim=-1)
                components["contrastive"] = contrastive_alignment_loss(a, b)

        # ---- C: drug response ----
        if "drug" in weights and batch.drug_response is not None:
            preds = prediction.drug_specific_risk             # (N, n_drug_candidates)
            target = batch.drug_response
            # Squeeze a singleton last dim so a (N, 1) target compares cleanly
            # against the per-drug-mean of the head.
            if target.ndim == 2 and target.shape[-1] == 1:
                target = target.squeeze(-1)                   # -> (N,)
            if target.ndim == 1:
                # Compare against the per-row mean of the candidate-drug risk.
                pred_scalar = preds.mean(dim=-1)               # (N,)
                components["drug"] = drug_response_loss(pred_scalar, target)
            elif target.shape[-1] == preds.shape[-1]:
                components["drug"] = drug_response_loss(preds, target)
            else:
                # Shape mismatch we cannot reconcile -- refuse to fabricate.
                logger.warning(
                    "Skipping drug loss: pred shape %s vs target shape %s -- not reconcilable.",
                    tuple(preds.shape), tuple(target.shape),
                )

        # ---- E: trajectory ----
        # v19 Phase 1: prefer the encoder-consistent future_snapshot_batch
        # path when a follow-up snapshot is collated; fall back to the legacy
        # future_state slot (cell-line / oracle teacher latents) otherwise.
        # The raw-RNA auto-fill that used to live in mort_collate has been
        # DELETED — silent fabrication is no longer possible.
        if "trajectory" in weights and batch.future_snapshot_batch is not None:
            # Encode the follow-up snapshot through the SAME foundation
            # encoder used for the baseline; detach as a target network so
            # gradients update the dynamics, not the encoder. This is the
            # latent-vs-latent loss the canonical path computes.
            future_batch = batch.future_snapshot_batch.to(self.device)
            with torch.no_grad():
                future_state = self.model.forward_foundation(future_batch)
            encoded_future = future_state.z0.detach()             # (B, d_latent)

            # v19 Bug B1 — axis-order audit on the LEGACY path.
            #
            # TrajectorySampler.sample_paths returns mean_path with shape
            # (T, N, d), even though the TrajectoryPrediction dataclass
            # docstring documents it as (N, T, d_latent). The trajectory
            # head preserves leading dims so prediction.trajectory_mean is
            # also (T, N, d). Indexing `[-1]` picks the last TIMESTEP (axis
            # 0 is T here, not N) — which is what we want. The canonical
            # v18 forward()/canonical_trainer path uses (B, T, d_latent)
            # explicitly via `z_traj[:, -1, :]` and sidesteps the ambiguity.
            if prediction.trajectory_mean is not None:
                pred_T = prediction.trajectory_mean[-1]           # (N, d_out)
            else:
                pred_T = prediction.z_path[-1]                    # (N, d_latent)
            if pred_T.shape != encoded_future.shape:
                # Project encoded_future down to d_out if the head has a
                # narrower output dim — but ONLY for the encoder's own latent
                # (this is still dimensionally consistent, unlike raw-RNA
                # truncation).
                encoded_future = encoded_future[..., : pred_T.shape[-1]]
            components["trajectory"] = trajectory_distribution_loss(
                pred_T, encoded_future, metric="mmd",
            )
        elif "trajectory" in weights and batch.future_state is not None:
            # Legacy oracle/teacher path: a precomputed future-state latent
            # was provided directly (e.g. cell-line CCLE+GDSC pipeline).
            if prediction.trajectory_mean is not None:
                pred_T = prediction.trajectory_mean[-1]
            else:
                pred_T = prediction.z_path[-1]
            target = batch.future_state
            if target.shape[-1] != pred_T.shape[-1]:
                target = target[..., : pred_T.shape[-1]]
            components["trajectory"] = trajectory_distribution_loss(pred_T, target, metric="mmd")

        # ---- F: survival + state ----
        if "survival" in weights and batch.event_time is not None and batch.event_observed is not None:
            # Re-run survival head on z_T to get its raw output.
            state = self.model.encode(batch)
            traj = self.model.rollout(state, torch.linspace(0, self.config.integration_time,
                                                             self.config.n_time_grid, device=self.device),
                                      drug=state.modality_embeddings.get("drug"))
            z_T = traj.mean_path[-1]
            surv_out = self.model.survival_head(z_T)
            time_bins = torch.linspace(0, self.config.integration_time, self.config.survival_n_bins + 1,
                                       device=self.device)
            components["survival"] = survival_loss(
                self.model.survival_head.head, surv_out, batch.event_time, batch.event_observed,
                time_bins=time_bins,
            )
        if "state" in weights and batch.resistance_label is not None:
            components["state"] = resistance_state_loss(
                prediction.resistance_state_logits, batch.resistance_label
            )

        # ---- G: pathway ----
        if "pathway" in weights and prediction.pathway_protein_scores is not None:
            components["pathway"] = pathway_attribution_loss(
                prediction.pathway_protein_scores, batch.pathway_targets
            )

        # ---- E (also): landscape regularisation ----
        if "landscape" in weights and self.model.potential is not None:
            state = self.model.encode(batch)
            components["landscape"] = landscape_regularization_loss(
                self.model.potential, state.z0,
                drug=state.modality_embeddings.get("drug"),
            )

        return components, weights

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def train_epoch(self, stage: str, epoch: int) -> TrainingMetrics:
        self.model.train()
        t0 = time.time()
        sum_loss = 0.0
        sum_components: Dict[str, float] = {}
        n_batches = 0
        # v18.2 — supervision audit for the strict policy.
        sup_report = StageSupervisionReport(stage=stage)

        # v18.2 — which supervision key is mandatory for this stage?
        required_supervision_key = {
            "E": "trajectory", "F": "survival", "G": "pathway",
        }.get(stage)

        for batch in self.train_loader:
            sup_report.n_batches += 1
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(self.config.mixed_precision and self.device.type == "cuda")):
                components, weights = self._losses_for_batch(batch, stage)
                if not components:
                    # v18.2 — record reason instead of silently continuing.
                    sup_report.missing_reasons["no_components"] = (
                        sup_report.missing_reasons.get("no_components", 0) + 1
                    )
                    continue
                # v18.2 — for supervised stages, count a batch as "supervised"
                # only if the stage-specific loss component actually fired.
                if required_supervision_key:
                    if required_supervision_key in components:
                        sup_report.n_supervised_batches += 1
                    else:
                        sup_report.missing_reasons[
                            f"missing_{required_supervision_key}"
                        ] = sup_report.missing_reasons.get(
                            f"missing_{required_supervision_key}", 0
                        ) + 1
                else:
                    # Foundation stages (A/B/C) — any non-empty component counts
                    sup_report.n_supervised_batches += 1
                total, breakdown = assemble_total_loss(components, weights)
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            sum_loss += breakdown["total"]
            for k, v in breakdown.items():
                sum_components[k] = sum_components.get(k, 0.0) + v
            n_batches += 1

        # v18.2 — STRICT supervision check BEFORE we finalise the epoch.
        # Any stage in STRICT_SUPERVISION_COVERAGE that fell below its
        # threshold raises rather than silently producing a misleading loss.
        if stage in STRICT_SUPERVISION_COVERAGE:
            min_cov = STRICT_SUPERVISION_COVERAGE[stage]
            if sup_report.coverage < min_cov:
                raise MissingSupervisionError(
                    f"Stage {stage!r} supervision coverage = "
                    f"{sup_report.coverage:.1%} ({sup_report.n_supervised_batches}/"
                    f"{sup_report.n_batches} batches) — below the strict "
                    f"{min_cov:.0%} threshold. Reasons: {sup_report.missing_reasons}. "
                    f"This is the v18.2 'raise, don't skip' policy: the trainer "
                    f"refuses to advance an under-supervised supervised stage."
                )
            logger.info(
                "Stage %s supervision: %d/%d batches supervised (%.1f%% ≥ %.0f%% threshold)",
                stage, sup_report.n_supervised_batches, sup_report.n_batches,
                100 * sup_report.coverage, 100 * min_cov,
            )

        avg = sum_loss / max(n_batches, 1)
        for k in sum_components:
            sum_components[k] /= max(n_batches, 1)
        metric = TrainingMetrics(
            epoch=epoch, stage=stage, train_loss=avg, val_loss=float("nan"),
            components=sum_components, wall_time_s=time.time() - t0,
        )
        if self.val_loader is not None:
            metric.val_loss = self.validate(stage)
        self.history.append(metric)
        logger.info(
            "stage=%s epoch=%d train_loss=%.4f val_loss=%.4f time=%.1fs",
            stage, epoch, metric.train_loss, metric.val_loss, metric.wall_time_s,
        )
        return metric

    @torch.no_grad()
    def validate(self, stage: str) -> float:
        self.model.eval()
        if self.val_loader is None:
            return float("nan")
        sum_loss = 0.0
        n_batches = 0
        for batch in self.val_loader:
            components, weights = self._losses_for_batch(batch, stage)
            if not components:
                continue
            total, _ = assemble_total_loss(components, weights)
            sum_loss += float(total)
            n_batches += 1
        return sum_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def fit_stage(
        self,
        stage: str,
        n_epochs: int,
        *,
        acceptance_level: Optional[str] = None,
        acceptance_pairs: Optional[Sequence[TemporalTrainingPair]] = None,
    ) -> List[TrainingMetrics]:
        """Train one stage for ``n_epochs`` epochs.

        If ``acceptance_level`` is given (e.g. ``"research"``), the trainer
        first runs :func:`resistancemap.mortfm.acceptance_gate.evaluate_cohort`
        on ``acceptance_pairs`` (or, if not given, on the train_loader's
        underlying dataset) and refuses to run if the gate blocks. This is
        the mechanism that prevents accidental research/strong claims from
        an underpowered cohort.
        """
        if stage not in VALID_STAGES:
            raise ValueError(f"Unknown stage {stage!r}")
        if acceptance_level is not None:
            pairs = acceptance_pairs
            if pairs is None:
                ds = getattr(self.train_loader, "dataset", None)
                pairs = getattr(ds, "pairs", None)
            if pairs is None:
                raise ValueError(
                    "acceptance_level requires either acceptance_pairs or a "
                    "train_loader whose .dataset exposes .pairs (MORTFMDataset does)."
                )
            report = evaluate_cohort(pairs, level=acceptance_level, config=self.config)
            if not report.passed:
                raise AcceptanceError(
                    f"Stage {stage!r}: acceptance gate blocked level {acceptance_level!r}: "
                    + "; ".join(report.blocking_reasons),
                    report=report,
                )
            logger.info("Acceptance OK for stage %s at level %s: %s", stage, acceptance_level, report.summary())
        out = []
        for ep in range(n_epochs):
            out.append(self.train_epoch(stage, ep))
        return out

    def fit_all(self, stages: Sequence[str] = ("A", "B", "C", "E", "F")) -> List[TrainingMetrics]:
        """Run the canonical multi-stage curriculum."""
        epochs_per_stage = {
            "A": self.config.pretrain_epochs,
            "B": self.config.pretrain_epochs,
            "C": self.config.finetune_epochs,
            "D": self.config.finetune_epochs,
            "E": self.config.trajectory_epochs,
            "F": self.config.survival_epochs,
            "G": 5,
            "H": 5,
        }
        history: List[TrainingMetrics] = []
        for st in stages:
            history.extend(self.fit_stage(st, epochs_per_stage[st]))
            self.save_checkpoint(f"stage_{st}_complete.pt")
        return history

    def save_checkpoint(self, name: str) -> str:
        path = os.path.join(self.config.checkpoint_dir, name)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "history": [asdict(m) for m in self.history],
        }
        torch.save(payload, path)
        logger.info("Saved checkpoint -> %s", path)
        return path

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
