"""End-to-end ResistanceMap inference pipeline (L0 to L5).

Adapted from MyeloMemory's pipeline.py. Orchestrates:
    L0: Data preparation (proteomics normalization)
    L1: VAE pretraining/finetuning
    L2: Stability scoring (resistance memory)
    L3: Protein network analysis (GNN)
    L4: Multi-omics fusion
    L5: Resistance landscape prediction

Given raw proteomics data, produces LandscapeResult with drug resistance
predictions, intervention targets, and basin-of-attraction maps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from resistancemap.landscape.predictor import (
    ResistanceLandscape,
    ResistanceLandscapePredictor,
    LandscapeResult,
    TargetRanker,
)
from resistancemap.models.vae import ProteomeToEpigenomeVAE
from resistancemap.models.fusion import ResistanceMapFusion
from resistancemap.models.trajectory import ChromatinODE, NeuralJumpSDE
from resistancemap.config import VAEConfig, TrajectoryConfig

logger = logging.getLogger(__name__)


@dataclass
class TemporalLandscape:
    """Container for temporal resistance landscape trajectory.

    Attributes:
        timepoints: List of timepoint labels (e.g., ["baseline", "3m", "6m"]).
        landscapes: List of LandscapeResult, one per timepoint.
        trajectory_state_sequence: Observed sequence of resistance states.
        predicted_state_sequence: Model's prediction of resistance state evolution.
        trajectory_accuracy: Fraction of predicted states matching observed (if available).
    """

    timepoints: list[str]
    landscapes: list[LandscapeResult]
    trajectory_state_sequence: list[str]
    predicted_state_sequence: list[str]
    trajectory_accuracy: float | None = None


class ResistanceMapPipeline:
    """Full inference pipeline wiring L0–L5 for ResistanceMap.

    Usage:
        pipeline = ResistanceMapPipeline.from_checkpoints(ckpt_mgr, config)
        result = pipeline.predict_single(proteomics_dict)
        trajectory = pipeline.predict_trajectory(proteomics_timeseries)

    Args:
        landscape_model: Trained ResistanceLandscape module (L5).
        landscape_predictor: ResistanceLandscapePredictor instance.
        drug_names: List of target drug names.
        protein_names: List of protein names in the network.
        state_names: List of resistance state names.
        device: Torch device for inference.
        config: ResistanceMapConfig (optional).
    """

    def __init__(
        self,
        landscape_model: ResistanceLandscape,
        landscape_predictor: ResistanceLandscapePredictor,
        drug_names: list[str],
        protein_names: list[str],
        state_names: list[str] | None = None,
        device: str = "cpu",
        config: Any | None = None,
        vae_model: ProteomeToEpigenomeVAE | None = None,
        stability_scorer: ChromatinODE | NeuralJumpSDE | None = None,
        gnn_model: nn.Module | None = None,
        fusion_model: ResistanceMapFusion | None = None,
    ) -> None:
        """Initialize the ResistanceMapPipeline.

        Args:
            landscape_model: Trained L5 model.
            landscape_predictor: Predictor wrapper for L5.
            drug_names: Drug names.
            protein_names: Protein names.
            state_names: Resistance state names.
            device: Torch device.
            config: Optional configuration dict.
            vae_model: Optional trained ProteomeToEpigenomeVAE (L1).
            stability_scorer: Optional ChromatinODE or NeuralJumpSDE (L2).
            gnn_model: Optional protein network GNN model (L3).
            fusion_model: Optional ResistanceMapFusion model (L4).
        """
        self.landscape_model = landscape_model.to(device).eval()
        self.landscape_predictor = landscape_predictor
        self.drug_names = drug_names
        self.protein_names = protein_names
        self.state_names = state_names or ["sensitive", "intermediate", "resistant"]
        self.device = device
        self.config = config or {}

        # Optional L0-L4 models
        self.vae_model = vae_model.to(device).eval() if vae_model is not None else None
        self.stability_scorer = stability_scorer.to(device).eval() if stability_scorer is not None else None
        self.gnn_model = gnn_model.to(device).eval() if gnn_model is not None else None
        self.fusion_model = fusion_model.to(device).eval() if fusion_model is not None else None

        logger.info(
            f"ResistanceMapPipeline initialized: "
            f"{len(drug_names)} drugs, {len(protein_names)} proteins, "
            f"{len(self.state_names)} resistance states | "
            f"L1 VAE: {self.vae_model is not None}, "
            f"L2 Stability: {self.stability_scorer is not None}, "
            f"L3 GNN: {self.gnn_model is not None}, "
            f"L4 Fusion: {self.fusion_model is not None}"
        )

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_dir: str | None = None,
        config: Any | None = None,
    ) -> "ResistanceMapPipeline":
        """Load all pipeline stages from saved checkpoints.

        Attempts to load L0-L5 models from checkpoint files. Each layer is optional;
        if a checkpoint doesn't exist, that layer is skipped with a warning logged.
        L5 (landscape model) is always created if no checkpoint exists.

        Args:
            checkpoint_dir: Directory containing stage checkpoints.
            config: Pipeline configuration.

        Returns:
            Initialized ResistanceMapPipeline ready for inference.
        """
        config = config or {}
        device = config.get("device", "cpu")
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")

        drug_names = config.get("drug_names", [
            "Bortezomib", "Lenalidomide", "Dexamethasone",
            "Carfilzomib", "Pomalidomide", "Daratumumab"
        ])
        protein_names = config.get("protein_names", [f"PROT_{i}" for i in range(100)])
        state_names = config.get("state_names", ["sensitive", "intermediate", "resistant"])

        # ========================= L1: VAE Model =========================
        vae_model = None
        vae_ckpt = checkpoint_dir / "vae_finetuned.pt"
        if vae_ckpt.exists():
            try:
                vae_config = VAEConfig()  # Use defaults or override from config
                if "vae_config" in config:
                    vae_config = config["vae_config"]
                vae_model = ProteomeToEpigenomeVAE(vae_config)
                checkpoint_data = torch.load(vae_ckpt, map_location=device)
                if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                    vae_model.load_state_dict(checkpoint_data["model_state_dict"])
                else:
                    vae_model.load_state_dict(checkpoint_data)
                vae_model = vae_model.to(device).eval()
                logger.info(f"Loaded L1 VAE from {vae_ckpt}")
            except Exception as e:
                logger.warning(f"Failed to load L1 VAE from {vae_ckpt}: {e}. Skipping L1.")
                vae_model = None
        else:
            logger.warning(f"L1 VAE checkpoint not found at {vae_ckpt}. L1 will be skipped.")

        # ========================= L2: Stability Scorer =========================
        stability_scorer = None
        stability_ckpt = checkpoint_dir / "stability_calibrated.pt"
        if stability_ckpt.exists():
            try:
                # Try to load as ChromatinODE first
                stability_config = TrajectoryConfig()  # Use defaults or override
                if "stability_config" in config:
                    stability_config = config["stability_config"]

                # Attempt ChromatinODE first (default), then NeuralJumpSDE
                try:
                    stability_scorer = ChromatinODE(
                        latent_dim=64,
                        n_feedback_layers=config.get("n_feedback_layers", 2),
                    )
                    checkpoint_data = torch.load(stability_ckpt, map_location=device)
                    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                        stability_scorer.load_state_dict(checkpoint_data["model_state_dict"])
                    else:
                        stability_scorer.load_state_dict(checkpoint_data)
                    stability_scorer = stability_scorer.to(device).eval()
                    logger.info(f"Loaded L2 Stability scorer (ChromatinODE) from {stability_ckpt}")
                except Exception as e_ode:
                    logger.debug(f"ChromatinODE load failed: {e_ode}, trying NeuralJumpSDE...")
                    try:
                        stability_scorer = NeuralJumpSDE(
                            input_dim=64,
                            hidden_dim=config.get("sde_drift_hidden", 128),
                        )
                        checkpoint_data = torch.load(stability_ckpt, map_location=device)
                        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                            stability_scorer.load_state_dict(checkpoint_data["model_state_dict"])
                        else:
                            stability_scorer.load_state_dict(checkpoint_data)
                        stability_scorer = stability_scorer.to(device).eval()
                        logger.info(f"Loaded L2 Stability scorer (NeuralJumpSDE) from {stability_ckpt}")
                    except Exception as e_sde:
                        logger.warning(f"Both ChromatinODE and NeuralJumpSDE failed: {e_sde}. Skipping L2.")
                        stability_scorer = None
            except Exception as e:
                logger.warning(f"Failed to load L2 Stability scorer from {stability_ckpt}: {e}. Skipping L2.")
                stability_scorer = None
        else:
            logger.warning(f"L2 Stability checkpoint not found at {stability_ckpt}. L2 will be skipped.")

        # ========================= L3: GNN Model =========================
        gnn_model = None
        gnn_ckpt = checkpoint_dir / "gnn_trained.pt"
        if gnn_ckpt.exists():
            try:
                # Import GNN model (assumes it's in protein_network module)
                from resistancemap.models.protein_network import PPIGraphNetwork
                gnn_model = PPIGraphNetwork(
                    in_dim=config.get("esm2_dim", 1280),
                    hidden_dim=config.get("gnn_hidden", 256),
                    n_layers=config.get("gnn_layers", 4),
                    n_heads=config.get("gnn_heads", 8),
                    dropout=config.get("gnn_dropout", 0.2),
                )
                checkpoint_data = torch.load(gnn_ckpt, map_location=device)
                if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                    gnn_model.load_state_dict(checkpoint_data["model_state_dict"])
                else:
                    gnn_model.load_state_dict(checkpoint_data)
                gnn_model = gnn_model.to(device).eval()
                logger.info(f"Loaded L3 GNN from {gnn_ckpt}")
            except ImportError:
                logger.warning("PPIGraphNetwork not available in protein_network module. Skipping L3.")
                gnn_model = None
            except Exception as e:
                logger.warning(f"Failed to load L3 GNN from {gnn_ckpt}: {e}. Skipping L3.")
                gnn_model = None
        else:
            logger.warning(f"L3 GNN checkpoint not found at {gnn_ckpt}. L3 will be skipped.")

        # ========================= L4: Fusion Model =========================
        fusion_model = None
        fusion_ckpt = checkpoint_dir / "fusion_trained.pt"
        if fusion_ckpt.exists():
            try:
                fusion_model = ResistanceMapFusion(
                    hidden_dim=config.get("fusion_hidden_dim", 256),
                    output_dim=config.get("fusion_dim", 128),
                    fusion_type=config.get("fusion_type", "cross_attention"),
                    n_heads=config.get("fusion_n_heads", 4),
                    dropout=config.get("fusion_dropout", 0.2),
                )
                checkpoint_data = torch.load(fusion_ckpt, map_location=device)
                if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                    fusion_model.load_state_dict(checkpoint_data["model_state_dict"])
                else:
                    fusion_model.load_state_dict(checkpoint_data)
                fusion_model = fusion_model.to(device).eval()
                logger.info(f"Loaded L4 Fusion from {fusion_ckpt}")
            except Exception as e:
                logger.warning(f"Failed to load L4 Fusion from {fusion_ckpt}: {e}. Skipping L4.")
                fusion_model = None
        else:
            logger.warning(f"L4 Fusion checkpoint not found at {fusion_ckpt}. L4 will be skipped.")

        # ========================= L5: Landscape Model =========================
        fusion_dim = config.get("fusion_dim", 128)
        n_drugs = len(drug_names)
        n_proteins = len(protein_names)
        landscape_model = ResistanceLandscape(
            fusion_dim=fusion_dim,
            n_drugs=n_drugs,
            n_timepoints=3,
            n_states=len(state_names),
            n_proteins=n_proteins,
            hidden_dim=256,
            dropout_rate=0.2,
        )

        landscape_ckpt = checkpoint_dir / "landscape_trained.pt"
        if landscape_ckpt.exists():
            try:
                checkpoint_data = torch.load(landscape_ckpt, map_location=device)
                if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                    landscape_model.load_state_dict(checkpoint_data["model_state_dict"])
                else:
                    landscape_model.load_state_dict(checkpoint_data)
                logger.info(f"Loaded L5 Landscape from {landscape_ckpt}")
            except Exception as e:
                logger.warning(f"Failed to load L5 Landscape from {landscape_ckpt}: {e}. Using untrained model.")
        else:
            logger.warning(f"L5 Landscape checkpoint not found at {landscape_ckpt}. Using untrained model.")

        landscape_model = landscape_model.to(device).eval()

        # ========================= Create Target Ranker & Predictor =========================
        known_drug_targets = config.get("known_drug_targets", {})
        ppi_centrality = config.get("ppi_centrality", {})
        target_ranker = TargetRanker(
            known_drug_targets=known_drug_targets,
            ppi_centrality=ppi_centrality,
            device=device,
        )

        landscape_predictor = ResistanceLandscapePredictor(
            model=landscape_model,
            drug_names=drug_names,
            protein_names=protein_names,
            state_names=state_names,
            target_ranker=target_ranker,
            device=device,
        )

        logger.info(
            f"Pipeline initialized from checkpoints: "
            f"L1={vae_model is not None}, L2={stability_scorer is not None}, "
            f"L3={gnn_model is not None}, L4={fusion_model is not None}, L5=True"
        )

        return cls(
            landscape_model=landscape_model,
            landscape_predictor=landscape_predictor,
            drug_names=drug_names,
            protein_names=protein_names,
            state_names=state_names,
            device=device,
            config=config,
            vae_model=vae_model,
            stability_scorer=stability_scorer,
            gnn_model=gnn_model,
            fusion_model=fusion_model,
        )

    @torch.no_grad()
    def predict_single(
        self,
        proteomics: dict[str, float] | torch.Tensor,
        sample_id: str | None = None,
    ) -> LandscapeResult:
        """Run full L0-L5 inference on a single sample's proteomics profile.

        Pipeline stages with graceful degradation:
        1. L0: Normalize proteomics vector
        2. L1: VAE encode → latent memory state (if vae_model available)
        3. L2: Compute stability score (if stability_scorer available)
        4. L3: GNN protein network propagation (if gnn_model available)
        5. L4: Fusion of all modalities (if fusion_model available)
        6. L5: Landscape prediction (always available)

        Each missing layer is logged with a warning and skipped.

        Args:
            proteomics: Dict mapping protein_name → abundance, or (n_proteins,) tensor.
            sample_id: Optional sample identifier.

        Returns:
            LandscapeResult with predictions.
        """
        # ========================= L0: Normalization =========================
        if isinstance(proteomics, dict):
            prot_vector = torch.zeros(len(self.protein_names), dtype=torch.float32)
            for i, prot_name in enumerate(self.protein_names):
                if prot_name in proteomics:
                    prot_vector[i] = proteomics[prot_name]
            proteomics_tensor = prot_vector
        else:
            proteomics_tensor = torch.as_tensor(proteomics, dtype=torch.float32)

        # Normalize proteomics (L0): min-max normalization to [0, 1]
        prot_min = proteomics_tensor.min()
        prot_max = proteomics_tensor.max()
        if prot_max > prot_min:
            proteomics_normalized = (proteomics_tensor - prot_min) / (prot_max - prot_min)
        else:
            proteomics_normalized = proteomics_tensor

        proteomics_normalized = proteomics_normalized.unsqueeze(0).to(self.device)  # (1, n_proteins)
        logger.debug(f"L0: Proteomics normalized. Shape: {proteomics_normalized.shape}")

        # ========================= L1: VAE Encoding =========================
        epigenetic_state = None
        if self.vae_model is not None:
            try:
                # Get latent memory state from VAE encoder
                mu, log_var = self.vae_model.encode(proteomics_normalized)
                epigenetic_state = mu  # Use mean as deterministic memory state
                logger.debug(f"L1: VAE encoded. Epigenetic state shape: {epigenetic_state.shape}")
            except Exception as e:
                logger.warning(f"L1 VAE encoding failed: {e}. Using zero vector as fallback.")
                epigenetic_state = torch.zeros(1, self.vae_model.latent_dim, device=self.device)
        else:
            logger.warning("L1: VAE model not available. Using zero vector as epigenetic state.")
            epigenetic_state = torch.zeros(1, 64, device=self.device)

        # ========================= L2: Stability Scoring =========================
        stability_score = None
        if self.stability_scorer is not None:
            try:
                # Compute stability score from epigenetic state
                # Stability scorers typically output a single scalar in [0, 1]
                if isinstance(self.stability_scorer, ChromatinODE):
                    # ChromatinODE: compute basin depth
                    stability_score = self.stability_scorer.compute_basin_depth(epigenetic_state)
                elif isinstance(self.stability_scorer, NeuralJumpSDE):
                    # NeuralJumpSDE: forward pass returns stability estimate
                    stability_score = self.stability_scorer(epigenetic_state)
                else:
                    # Generic fallback: assume forward pass returns scalar or (batch, 1)
                    stability_score = self.stability_scorer(epigenetic_state)
                    if stability_score.dim() > 1:
                        stability_score = stability_score.mean(dim=-1, keepdim=True)

                # Ensure shape is (1, 1)
                if stability_score.dim() == 0:
                    stability_score = stability_score.unsqueeze(0).unsqueeze(0)
                elif stability_score.shape[-1] != 1:
                    stability_score = stability_score.unsqueeze(-1)

                logger.debug(f"L2: Stability score computed. Shape: {stability_score.shape}")
            except Exception as e:
                logger.warning(f"L2 Stability scoring failed: {e}. Using zero as fallback.")
                stability_score = torch.zeros(1, 1, device=self.device)
        else:
            logger.warning("L2: Stability scorer not available. Using zero as stability score.")
            stability_score = torch.zeros(1, 1, device=self.device)

        # ========================= L3: GNN Protein Network =========================
        protein_network_output = None
        if self.gnn_model is not None:
            try:
                # GNN expects graph input (node features, edge indices)
                # As fallback, use proteomics as per-protein features
                # In production, would construct actual PPI graph
                protein_network_output = self.gnn_model(proteomics_normalized)
                logger.debug(f"L3: GNN propagation. Output shape: {protein_network_output.shape}")
            except Exception as e:
                logger.warning(f"L3 GNN propagation failed: {e}. Using projection of proteomics.")
                # Fallback: project proteomics to expected GNN output dimension
                gnn_dim = self.config.get("gnn_hidden", 256)
                protein_network_output = torch.zeros(1, gnn_dim, device=self.device)
        else:
            logger.warning("L3: GNN model not available. Using zero vector as network output.")
            gnn_dim = self.config.get("gnn_hidden", 256)
            protein_network_output = torch.zeros(1, gnn_dim, device=self.device)

        # ========================= L4: Multi-Modal Fusion =========================
        fused_repr = None
        if self.fusion_model is not None:
            try:
                # Ensure trajectory_state exists (use epigenetic_state as fallback)
                trajectory_state = epigenetic_state

                # Call fusion model with all modalities
                fusion_output = self.fusion_model(
                    epigenetic_state=epigenetic_state,
                    trajectory_state=trajectory_state,
                    protein_network_output=protein_network_output,
                    stability_score=stability_score,
                )
                fused_repr = fusion_output["fused_representation"]
                logger.debug(f"L4: Fusion complete. Fused representation shape: {fused_repr.shape}")
            except Exception as e:
                logger.warning(f"L4 Fusion failed: {e}. Using concatenation fallback.")
                # Fallback: concatenate all modalities
                fused_repr = torch.cat(
                    [epigenetic_state, trajectory_state, protein_network_output, stability_score],
                    dim=-1
                )
        else:
            logger.warning("L4: Fusion model not available. Using concatenation.")
            trajectory_state = epigenetic_state
            fused_repr = torch.cat(
                [epigenetic_state, trajectory_state, protein_network_output, stability_score],
                dim=-1
            )

        # Ensure fusion output matches expected dimension
        fusion_dim = self.landscape_model.fusion_dim
        if fused_repr.shape[-1] < fusion_dim:
            padding = torch.zeros(
                fused_repr.shape[0],
                fusion_dim - fused_repr.shape[-1],
                device=self.device
            )
            fused_repr = torch.cat([fused_repr, padding], dim=-1)
        elif fused_repr.shape[-1] > fusion_dim:
            logger.warning(f"Truncating fusion output from {fused_repr.shape[-1]} to {fusion_dim} dims")
            fused_repr = fused_repr[:, :fusion_dim]

        logger.debug(f"L4: Fusion output adjusted to shape {fused_repr.shape}")

        # ========================= L5: Landscape Prediction =========================
        result = self.landscape_predictor.predict_single(fused_repr, sample_id=sample_id)
        logger.info(
            f"Prediction complete: sample_id={sample_id}, "
            f"state={result.resistance_state}, confidence={result.confidence_score:.3f}"
        )
        return result

    @torch.no_grad()
    def predict_batch(
        self,
        dataset: Any,
        batch_size: int = 32,
    ) -> list[LandscapeResult]:
        """Run inference on a batch dataset.

        Args:
            dataset: Dataset with proteomics profiles.
            batch_size: Batch size for processing.

        Returns:
            List of LandscapeResult, one per sample.
        """
        results = []
        n_samples = len(dataset)

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_samples = []

            for j in range(i, batch_end):
                sample = dataset[j]
                if isinstance(sample, dict) and "proteomics" in sample:
                    batch_samples.append(sample["proteomics"])
                else:
                    batch_samples.append(sample)

            # Stack into tensor
            if isinstance(batch_samples[0], torch.Tensor):
                batch_tensor = torch.stack(batch_samples).to(self.device)
            else:
                batch_tensor = torch.tensor(batch_samples, dtype=torch.float32).to(self.device)

            # Handle dimension issues
            if batch_tensor.dim() == 1:
                batch_tensor = batch_tensor.unsqueeze(0)  # Add batch dim
            elif batch_tensor.dim() > 2:
                # Flatten extra dimensions
                batch_tensor = batch_tensor.reshape(batch_tensor.shape[0], -1)

            # Normalize proteomics (L0)
            prot_min = batch_tensor.min(dim=1, keepdim=True)[0]
            prot_max = batch_tensor.max(dim=1, keepdim=True)[0]
            batch_normalized = torch.where(
                prot_max > prot_min,
                (batch_tensor - prot_min) / (prot_max - prot_min),
                batch_tensor
            )

            # Route through VAE encoder (L1) if available
            if self.vae_model is not None:
                try:
                    mu, log_var = self.vae_model.encode(batch_normalized)
                    batch_tensor = mu  # Use latent representation
                    logger.debug(f"L1: VAE encoded batch. Shape: {batch_tensor.shape}")
                except Exception as e:
                    logger.warning(f"L1 VAE encoding failed: {e}. Using normalized proteomics.")
                    batch_tensor = batch_normalized
            else:
                # Fallback to normalized proteomics
                batch_tensor = batch_normalized

            # Ensure output matches fusion_dim
            fusion_dim = self.landscape_model.fusion_dim
            if batch_tensor.shape[-1] < fusion_dim:
                padding = torch.zeros(
                    batch_tensor.shape[0],
                    fusion_dim - batch_tensor.shape[-1],
                    device=self.device
                )
                batch_tensor = torch.cat([batch_tensor, padding], dim=-1)
            elif batch_tensor.shape[-1] > fusion_dim:
                batch_tensor = batch_tensor[:, :fusion_dim]

            # Predict batch
            batch_results = self.landscape_predictor.predict_batch(batch_tensor)
            results.extend(batch_results)

        logger.info(f"Batch prediction complete: {len(results)} samples")
        return results

    @torch.no_grad()
    def predict_trajectory(
        self,
        proteomics_timeseries: list[dict[str, float] | torch.Tensor],
        timepoint_labels: list[str] | None = None,
        actual_states: list[str] | None = None,
    ) -> TemporalLandscape:
        """Predict resistance landscape trajectory over time.

        Given multiple proteomics snapshots, predict the full resistance
        state trajectory and basin-of-attraction evolution.

        Args:
            proteomics_timeseries: List of proteomics profiles (one per timepoint).
            timepoint_labels: Optional labels for timepoints (e.g., ["baseline", "3m", "6m"]).
            actual_states: Optional ground truth resistance states for comparison.

        Returns:
            TemporalLandscape with predicted trajectory.
        """
        if timepoint_labels is None:
            timepoint_labels = [f"T{i}" for i in range(len(proteomics_timeseries))]

        assert len(proteomics_timeseries) == len(timepoint_labels), \
            f"Timeseries length {len(proteomics_timeseries)} != labels length {len(timepoint_labels)}"

        landscapes = []
        predicted_states = []

        for t, (prot, label) in enumerate(zip(proteomics_timeseries, timepoint_labels)):
            landscape = self.predict_single(prot, sample_id=f"trajectory_t{t}")
            landscapes.append(landscape)
            predicted_states.append(landscape.resistance_state)

        # Compute trajectory accuracy if ground truth available
        trajectory_accuracy = None
        if actual_states is not None:
            assert len(actual_states) == len(predicted_states), \
                f"Actual states length {len(actual_states)} != predicted {len(predicted_states)}"
            n_correct = sum(1 for a, p in zip(actual_states, predicted_states) if a == p)
            trajectory_accuracy = n_correct / len(actual_states)

        logger.info(
            f"Trajectory prediction: {len(landscapes)} timepoints, "
            f"states={predicted_states}, accuracy={trajectory_accuracy}"
        )

        return TemporalLandscape(
            timepoints=timepoint_labels,
            landscapes=landscapes,
            trajectory_state_sequence=actual_states or [],
            predicted_state_sequence=predicted_states,
            trajectory_accuracy=trajectory_accuracy,
        )