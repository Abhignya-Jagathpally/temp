#!/usr/bin/env python3
"""ResistanceMap v6 Trainer.

Multi-task trainer with:
    - Uncertainty-based automatic loss weighting (Kendall et al. 2018)
    - Cosine annealing with warm restarts
    - Gradient clipping (max norm = 1.0)
    - Mixed precision training (torch.cuda.amp)
    - Exponential moving average (EMA) model for evaluation
    - Checkpointing: save every epoch + best model
    - Early stopping on validation AUPRC (patience = 20)
    - Wandb / TensorBoard logging

Usage:
    trainer = ResistanceMapTrainer(model, config)
    trainer.fit(train_loader, val_loader)
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from .losses import ResistanceMapLoss

logger = logging.getLogger(__name__)


# =============================================================================
# Exponential Moving Average
# =============================================================================


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters that is updated as:
        shadow = decay * shadow + (1 - decay) * param

    The EMA model typically generalizes better and is used for evaluation.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay factor (0.999 typical).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model parameters with shadow (EMA) parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original model parameters from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# =============================================================================
# Early Stopping
# =============================================================================


class EarlyStopping:
    """Early stopping monitor.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'max' for metrics like AUPRC, 'min' for metrics like loss.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check whether to stop.

        Args:
            score: Current epoch metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# =============================================================================
# Checkpoint Manager
# =============================================================================


class CheckpointManager:
    """Manages model checkpoints: periodic saves, best model, pruning.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        keep_last_n: Number of recent checkpoints to keep (delete older).
        monitor_metric: Metric name to track for best model.
        monitor_mode: 'max' or 'min'.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 5,
        monitor_metric: str = "val/auprc",
        monitor_mode: str = "max",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.best_score: Optional[float] = None
        self.saved_checkpoints: list[Path] = []

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Optional[GradScaler],
        metrics: dict,
        extra: Optional[dict] = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            epoch: Current epoch number.
            model: The model to save.
            optimizer: Current optimizer state.
            scheduler: Current scheduler state.
            scaler: AMP GradScaler state.
            metrics: Current epoch metrics.
            extra: Any additional state to save.

        Returns:
            Path to the saved checkpoint.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metrics": metrics,
        }
        if extra:
            state.update(extra)

        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(state, path)
        self.saved_checkpoints.append(path)

        # Prune old checkpoints
        while len(self.saved_checkpoints) > self.keep_last_n:
            old = self.saved_checkpoints.pop(0)
            if old.exists():
                old.unlink()

        # Check if this is the best model
        score = metrics.get(self.monitor_metric)
        if score is not None:
            is_best = False
            if self.best_score is None:
                is_best = True
            elif self.monitor_mode == "max" and score > self.best_score:
                is_best = True
            elif self.monitor_mode == "min" and score < self.best_score:
                is_best = True

            if is_best:
                self.best_score = score
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(state, best_path)
                logger.info(
                    f"  New best model: {self.monitor_metric}={score:.6f}"
                )

        return path

    def load_latest(self) -> Optional[dict]:
        """Load the most recent checkpoint.

        Returns:
            Checkpoint state dict, or None if no checkpoint found.
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None
        return torch.load(checkpoints[-1], map_location="cpu", weights_only=False)

    def load_best(self) -> Optional[dict]:
        """Load the best model checkpoint.

        Returns:
            Checkpoint state dict, or None if no best checkpoint found.
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return torch.load(best_path, map_location="cpu", weights_only=False)
        return None


# =============================================================================
# Experiment Tracker
# =============================================================================


class ExperimentTracker:
    """Unified tracking interface for wandb or TensorBoard.

    Args:
        config: Tracking configuration dict.
        run_name: Name for this run.
    """

    def __init__(self, config: dict, run_name: str = "resistancemap"):
        self.backend = config.get("backend", "tensorboard")
        self._writer = None
        self._wandb_run = None

        if self.backend == "wandb":
            try:
                import wandb

                wandb_cfg = config.get("wandb", {})
                self._wandb_run = wandb.init(
                    project=wandb_cfg.get("project", "resistancemap-v6"),
                    entity=wandb_cfg.get("entity"),
                    name=run_name,
                    tags=wandb_cfg.get("tags", []),
                    group=wandb_cfg.get("group", "default"),
                    config=config,
                    reinit=True,
                )
                logger.info(f"Wandb initialized: {self._wandb_run.url}")
            except ImportError:
                logger.warning("wandb not installed, falling back to TensorBoard")
                self.backend = "tensorboard"

        if self.backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_cfg = config.get("tensorboard", {})
                log_dir = tb_cfg.get("log_dir", f"runs/{run_name}")
                self._writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard writer at {log_dir}")
            except ImportError:
                logger.warning("TensorBoard not available, logging to stdout only")

    def log(self, metrics: dict, step: int) -> None:
        """Log metrics at a given step.

        Args:
            metrics: Dict of metric_name -> value.
            step: Global step number.
        """
        if self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

        if self._writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(key, value, step)

    def finish(self) -> None:
        """Close the tracker."""
        if self._wandb_run is not None:
            import wandb

            wandb.finish()
        if self._writer is not None:
            self._writer.close()


# =============================================================================
# Main Trainer
# =============================================================================


class ResistanceMapTrainer:
    """Multi-task trainer for ResistanceMap v6.

    Handles the full training loop including:
        - Mixed precision training
        - Gradient clipping
        - EMA model updates
        - Learning rate scheduling
        - Checkpointing
        - Early stopping
        - Experiment tracking (wandb/TensorBoard)

    Args:
        model: The ResistanceMap model.
        config: Full experiment configuration dict (training section used).
        loss_module: Pre-configured ResistanceMapLoss instance, or None to
            construct from config.
        pathway_gene_sets: Pathway definitions for anchor loss.
        device: Device to train on.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        loss_module: Optional[ResistanceMapLoss] = None,
        pathway_gene_sets: Optional[dict[str, list[int]]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.train_cfg = config.get("training", {})
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Model
        self.model = model.to(self.device)
        logger.info(
            f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

        # Loss
        if loss_module is not None:
            self.loss_fn = loss_module.to(self.device)
        else:
            self.loss_fn = ResistanceMapLoss(
                config=self.train_cfg,
                pathway_gene_sets=pathway_gene_sets,
                num_programs=config.get("model", {}).get("num_programs", 8),
                auto_weight=self.train_cfg.get("auto_loss_weighting", True),
            ).to(self.device)

        # Optimizer
        opt_cfg = self.train_cfg.get("optimizer", {})
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=opt_cfg.get("lr", 1e-3),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        # Scheduler
        sched_cfg = self.train_cfg.get("scheduler", {})
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=sched_cfg.get("T_0", 50),
            T_mult=sched_cfg.get("T_mult", 2),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )

        # Mixed precision
        self.use_amp = self.train_cfg.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient clipping
        self.grad_clip_norm = self.train_cfg.get("gradient_clip_max_norm", 1.0)

        # EMA
        ema_cfg = self.train_cfg.get("ema", {})
        self.use_ema = ema_cfg.get("enabled", True)
        self.ema = EMAModel(
            self.model, decay=ema_cfg.get("decay", 0.999)
        ) if self.use_ema else None

        # Checkpoint manager
        ckpt_cfg = self.train_cfg.get("checkpoint", {})
        output_cfg = config.get("output", {})
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=output_cfg.get(
                "checkpoints_dir", "results/checkpoints"
            ),
            keep_last_n=ckpt_cfg.get("keep_last_n", 5),
            monitor_metric=ckpt_cfg.get("monitor_metric", "val/auprc"),
            monitor_mode=ckpt_cfg.get("monitor_mode", "max"),
        )

        # Early stopping
        es_cfg = self.train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 20),
            min_delta=es_cfg.get("min_delta", 1e-4),
            mode=es_cfg.get("monitor_mode", "max"),
        ) if es_cfg.get("enabled", True) else None

        # Experiment tracker
        tracking_cfg = config.get("tracking", {})
        self.tracker = ExperimentTracker(tracking_cfg, run_name="resistancemap_train")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.train_history: list[dict] = []

    def _train_one_epoch(
        self, train_loader: DataLoader
    ) -> dict[str, float]:
        """Run one training epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Dict of averaged training metrics for this epoch.
        """
        self.model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch in train_loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                # Forward pass through model
                outputs = self.model(batch)

                # Compute composite loss
                loss_dict = self.loss_fn(
                    drug_logits=outputs.get("drug_logits", torch.zeros(1, device=self.device)),
                    drug_labels=batch.get("drug_labels", torch.zeros(1, device=self.device)),
                    pred_states=outputs.get("pred_states", torch.zeros(1, 1, 1, device=self.device)),
                    true_states=batch.get("true_states", torch.zeros(1, 1, 1, device=self.device)),
                    observation_mask=batch.get("observation_mask"),
                    interaction_matrix=outputs.get("interaction_matrix"),
                    program_embeddings=outputs.get("program_embeddings"),
                    z=outputs.get("z"),
                    mu=outputs.get("mu"),
                    logvar=outputs.get("logvar"),
                    recon_x=outputs.get("recon_x"),
                    target_x=batch.get("target_x"),
                    lyapunov_fn=getattr(self.model, "lyapunov_fn", None),
                    dynamics_fn=getattr(self.model, "dynamics_fn", None),
                    attractors=outputs.get("attractors"),
                    gene_expression=batch.get("gene_expression"),
                    adjacency_matrix=outputs.get("adjacency_matrix"),
                    attention_weights=outputs.get("attention_weights"),
                    modality_outputs=outputs.get("modality_outputs"),
                )

            total_loss = loss_dict["total"]

            # Backward pass with mixed precision
            self.scaler.scale(total_loss).backward()

            # Gradient clipping (unscale first for accurate norm computation)
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA update
            if self.ema is not None:
                self.ema.update(self.model)

            # Advance disentanglement annealing
            if hasattr(self.loss_fn, "disentanglement_loss"):
                self.loss_fn.disentanglement_loss.step()

            # Accumulate metrics
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor) and value.dim() == 0:
                    epoch_losses[f"train/{key}"] += value.item()
            epoch_losses["train/grad_norm"] += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            epoch_losses["train/lr"] += self.optimizer.param_groups[0]["lr"]
            n_batches += 1

            # Log per-step metrics
            self.tracker.log(
                {
                    "step/loss": total_loss.item(),
                    "step/lr": self.optimizer.param_groups[0]["lr"],
                    "step/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                },
                step=self.global_step,
            )
            self.global_step += 1

        # Average over batches
        for key in epoch_losses:
            epoch_losses[key] /= max(n_batches, 1)

        # Step scheduler
        self.scheduler.step()

        return dict(epoch_losses)

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Run validation.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dict of averaged validation metrics.
        """
        # Use EMA model if available
        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        self.model.eval()
        val_losses = defaultdict(float)
        all_drug_probs = []
        all_drug_labels = []
        n_batches = 0

        for batch in val_loader:
            batch = self._to_device(batch)

            outputs = self.model(batch)

            loss_dict = self.loss_fn(
                drug_logits=outputs.get("drug_logits", torch.zeros(1, device=self.device)),
                drug_labels=batch.get("drug_labels", torch.zeros(1, device=self.device)),
                pred_states=outputs.get("pred_states", torch.zeros(1, 1, 1, device=self.device)),
                true_states=batch.get("true_states", torch.zeros(1, 1, 1, device=self.device)),
                observation_mask=batch.get("observation_mask"),
                interaction_matrix=outputs.get("interaction_matrix"),
                program_embeddings=outputs.get("program_embeddings"),
                z=outputs.get("z"),
                mu=outputs.get("mu"),
                logvar=outputs.get("logvar"),
                recon_x=outputs.get("recon_x"),
                target_x=batch.get("target_x"),
                gene_expression=batch.get("gene_expression"),
                adjacency_matrix=outputs.get("adjacency_matrix"),
                attention_weights=outputs.get("attention_weights"),
                modality_outputs=outputs.get("modality_outputs"),
            )

            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor) and value.dim() == 0:
                    val_losses[f"val/{key}"] += value.item()

            # Collect predictions for AUROC/AUPRC computation
            if "drug_logits" in outputs:
                probs = torch.sigmoid(outputs["drug_logits"]).cpu()
                all_drug_probs.append(probs)
            if "drug_labels" in batch:
                all_drug_labels.append(batch["drug_labels"].cpu())

            n_batches += 1

        # Average losses
        for key in val_losses:
            val_losses[key] /= max(n_batches, 1)

        # Compute validation AUROC and AUPRC
        if all_drug_probs and all_drug_labels:
            probs_cat = torch.cat(all_drug_probs).numpy()
            labels_cat = torch.cat(all_drug_labels).numpy()

            from sklearn.metrics import roc_auc_score, average_precision_score

            try:
                val_losses["val/auroc"] = roc_auc_score(labels_cat, probs_cat)
            except ValueError:
                val_losses["val/auroc"] = 0.0
            try:
                val_losses["val/auprc"] = average_precision_score(
                    labels_cat, probs_cat
                )
            except ValueError:
                val_losses["val/auprc"] = 0.0

        # Restore original model parameters
        if self.ema is not None:
            self.ema.restore(self.model)

        return dict(val_losses)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        resume: bool = False,
    ) -> dict:
        """Run the full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Override number of epochs from config.
            resume: Whether to resume from the latest checkpoint.

        Returns:
            Dict with final metrics and training history.
        """
        if epochs is None:
            epochs = self.train_cfg.get("epochs", 200)

        # Resume from checkpoint if requested
        if resume:
            ckpt = self.ckpt_manager.load_latest()
            if ckpt is not None:
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if ckpt["scheduler_state_dict"] is not None:
                    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if ckpt["scaler_state_dict"] is not None and self.scaler is not None:
                    self.scaler.load_state_dict(ckpt["scaler_state_dict"])
                self.current_epoch = ckpt["epoch"] + 1
                logger.info(
                    f"Resumed from epoch {ckpt['epoch']} "
                    f"(metrics: {ckpt.get('metrics', {})})"
                )
            else:
                logger.info("No checkpoint found, starting from scratch")

        logger.info(
            f"Starting training: epochs={epochs}, device={self.device}, "
            f"amp={self.use_amp}, ema={self.use_ema}"
        )
        start_time = time.time()

        for epoch in range(self.current_epoch, epochs):
            epoch_start = time.time()
            self.current_epoch = epoch

            # Train
            train_metrics = self._train_one_epoch(train_loader)

            # Validate
            val_metrics = self._validate(val_loader)

            # Merge metrics
            all_metrics = {**train_metrics, **val_metrics, "epoch": epoch}

            # Log effective loss weights
            if hasattr(self.loss_fn, "auto_weighter") and self.loss_fn.auto_weighter is not None:
                weights = self.loss_fn.auto_weighter.get_weights_summary()
                for name, w in weights.items():
                    all_metrics[f"loss_weight/{name}"] = w

            # Log to tracker
            self.tracker.log(all_metrics, step=epoch)

            # Save checkpoint
            self.ckpt_manager.save(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                metrics=all_metrics,
            )

            # Early stopping
            monitor_value = val_metrics.get(
                self.train_cfg.get("early_stopping", {}).get(
                    "monitor_metric", "val/auprc"
                )
            )
            if self.early_stopping is not None and monitor_value is not None:
                if self.early_stopping(monitor_value):
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(best {self.early_stopping.best_score:.6f})"
                    )
                    break

            # Logging
            elapsed = time.time() - epoch_start
            train_loss = train_metrics.get("train/total", 0)
            val_loss = val_metrics.get("val/total", 0)
            val_auprc = val_metrics.get("val/auprc", 0)
            logger.info(
                f"Epoch {epoch:4d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_auprc={val_auprc:.4f} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s"
            )

            self.train_history.append(all_metrics)

        total_time = time.time() - start_time
        logger.info(
            f"Training complete: {self.current_epoch + 1} epochs in {total_time:.1f}s"
        )

        # Close tracker
        self.tracker.finish()

        # Load best model for return
        best_ckpt = self.ckpt_manager.load_best()
        if best_ckpt is not None:
            self.model.load_state_dict(best_ckpt["model_state_dict"])
            logger.info(
                f"Loaded best model from epoch {best_ckpt['epoch']} "
                f"({self.ckpt_manager.monitor_metric}="
                f"{best_ckpt['metrics'].get(self.ckpt_manager.monitor_metric, 'N/A')})"
            )

        return {
            "best_metrics": best_ckpt["metrics"] if best_ckpt else {},
            "history": self.train_history,
            "total_time": total_time,
            "epochs_trained": self.current_epoch + 1,
        }

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict[str, float]:
        """Evaluate the model on a test set.

        Uses the EMA model if available, collecting all predictions
        for comprehensive metric computation.

        Args:
            test_loader: Test data loader.

        Returns:
            Dict of evaluation metrics.
        """
        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        self.model.eval()
        all_probs = []
        all_labels = []
        all_pred_states = []
        all_true_states = []

        for batch in test_loader:
            batch = self._to_device(batch)
            outputs = self.model(batch)

            if "drug_logits" in outputs:
                all_probs.append(torch.sigmoid(outputs["drug_logits"]).cpu().numpy())
            if "drug_labels" in batch:
                all_labels.append(batch["drug_labels"].cpu().numpy())
            if "pred_states" in outputs:
                all_pred_states.append(outputs["pred_states"].cpu().numpy())
            if "true_states" in batch:
                all_true_states.append(batch["true_states"].cpu().numpy())

        if self.ema is not None:
            self.ema.restore(self.model)

        metrics = {}

        if all_probs and all_labels:
            probs = np.concatenate(all_probs)
            labels = np.concatenate(all_labels)

            from sklearn.metrics import (
                roc_auc_score,
                average_precision_score,
                f1_score,
                matthews_corrcoef,
                brier_score_loss,
            )

            preds = (probs > 0.5).astype(int)
            try:
                metrics["test/auroc"] = float(roc_auc_score(labels, probs))
            except ValueError:
                metrics["test/auroc"] = 0.0
            try:
                metrics["test/auprc"] = float(average_precision_score(labels, probs))
            except ValueError:
                metrics["test/auprc"] = 0.0
            metrics["test/f1"] = float(f1_score(labels, preds, zero_division=0))
            metrics["test/mcc"] = float(matthews_corrcoef(labels, preds))
            metrics["test/brier"] = float(brier_score_loss(labels, probs))

        if all_pred_states and all_true_states:
            pred = np.concatenate(all_pred_states)
            true = np.concatenate(all_true_states)
            metrics["test/trajectory_mse"] = float(np.mean((pred - true) ** 2))

        return metrics

    def _to_device(self, batch: dict) -> dict:
        """Move batch tensors to the training device.

        Args:
            batch: Dict of tensors.

        Returns:
            Dict with tensors moved to device.
        """
        return {
            key: val.to(self.device) if isinstance(val, torch.Tensor) else val
            for key, val in batch.items()
        }

    def save_training_summary(self, output_path: str) -> None:
        """Save a JSON summary of the training run.

        Args:
            output_path: Path to write the summary JSON.
        """
        summary = {
            "epochs_trained": self.current_epoch + 1,
            "device": str(self.device),
            "mixed_precision": self.use_amp,
            "ema_enabled": self.use_ema,
            "total_parameters": sum(
                p.numel() for p in self.model.parameters()
            ),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "history": self.train_history,
        }
        # Add loss weight summary
        if hasattr(self.loss_fn, "auto_weighter") and self.loss_fn.auto_weighter is not None:
            summary["final_loss_weights"] = self.loss_fn.auto_weighter.get_weights_summary()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Training summary saved to {output_path}")
