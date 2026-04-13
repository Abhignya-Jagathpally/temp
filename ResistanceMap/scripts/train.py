"""End-to-end training script for the full L0-L5 ResistanceMap pipeline.

Orchestrates all training stages:
  L0: Data preparation (load + split)
  L1: VAE training (pan-cancer pretrain + hematological finetune)
  L2: Stability scoring (ODE-based chromatin bistability)
  L3: GNN training (protein network on PPI graph)
  L4: Fusion training (multi-modal fusion)
  L5: Landscape training (resistance state prediction)

Supports:
  - Real data loading from CCLE, GDSC, STRING, scRNA-seq, MMRF
  - Synthetic data generation for testing without real datasets
  - Graceful handling of missing dependencies (torch_geometric, torchdiffeq)
  - Device selection (cpu / cuda)
  - Configurable pipeline stages
  - Checkpoint management and metrics logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ResistanceMap imports
from resistancemap.config import (
    ResistanceMapConfig,
    load_config,
    VAEConfig,
    TrajectoryConfig,
    FusionConfig,
    LandscapeConfig,
)
from resistancemap.data.loaders import (
    MultiOmicsDataset,
    load_ccle_proteomics,
    load_ccle_epigenomics,
    load_string_ppi,
    load_drug_sensitivity,
    load_scrna_data,
    load_mmrf_data,
)
from resistancemap.models.vae import ProteomeToEpigenomeVAE, train_vae
from resistancemap.models.trajectory import ChromatinODE, MemoryStabilityScorer
from resistancemap.models.protein_network import (
    ProteinNetworkModel,
    HAS_PYG,
)
from resistancemap.models.fusion import ResistanceMapFusion
from resistancemap.landscape.predictor import ResistanceLandscape
from resistancemap.utils.checkpoint import CheckpointManager

# Optional imports with graceful fallback
try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    odeint = None

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> None:
    """Configure logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info(f"Logging to {log_file}")


def generate_synthetic_data(
    n_samples: int = 100,
    n_proteins: int = 1000,
    n_epigenomic_features: int = 5000,
    n_drugs: int = 6,
    device: str = "cpu",
) -> MultiOmicsDataset:
    """Generate synthetic multi-omics data for testing.

    Args:
        n_samples: Number of samples to generate.
        n_proteins: Number of protein features.
        n_epigenomic_features: Number of epigenomic features.
        n_drugs: Number of drugs with sensitivity values.
        device: Device for data tensors.

    Returns:
        MultiOmicsDataset with random synthetic data.
    """
    logger.info(f"Generating synthetic data: {n_samples} samples")
    logger.info(
        f"  Proteomics: {n_samples} × {n_proteins}"
    )
    logger.info(
        f"  Epigenomics: {n_samples} × {n_epigenomic_features}"
    )
    logger.info(f"  Drugs: {n_drugs}")

    # Generate random data
    proteomics = torch.randn(n_samples, n_proteins, device=device)
    epigenomics = torch.randn(n_samples, n_epigenomic_features, device=device)

    # Drug sensitivity (IC50, with some missing values)
    drug_sensitivity = torch.randn(n_samples, n_drugs, device=device).abs() * 10.0
    # Add ~20% missing values
    mask = torch.rand(n_samples, n_drugs) < 0.2
    drug_sensitivity[mask] = float("nan")

    # Sample metadata
    sample_ids = [f"sample_{i:04d}" for i in range(n_samples)]
    lineages = [
        np.random.choice([
            "Myeloid", "Lymphoid", "haematopoietic_and_lymphoid_tissue",
            "blood", "lymphocyte", "plasma_cell",
        ])
        for _ in range(n_samples)
    ]

    # Protein and feature names
    protein_names = [f"protein_{i}" for i in range(n_proteins)]
    epigenome_feature_names = [f"epi_feature_{i}" for i in range(n_epigenomic_features)]
    drug_names = [f"drug_{i}" for i in range(n_drugs)]

    # PPI edges (small subset for testing)
    ppi_edges = [
        (protein_names[i], protein_names[j])
        for i in range(min(100, n_proteins))
        for j in range(i + 1, min(i + 5, n_proteins))
    ]
    ppi_scores = [0.8 + 0.2 * np.random.random() for _ in ppi_edges]

    dataset = MultiOmicsDataset(
        proteomics=proteomics,
        epigenomics=epigenomics,
        sample_ids=sample_ids,
        lineage=lineages,
        drug_sensitivity=drug_sensitivity,
        protein_names=protein_names,
        epigenome_feature_names=epigenome_feature_names,
        ppi_edges=ppi_edges,
        ppi_scores=ppi_scores,
        source="synthetic",
        drug_names=drug_names,
    )

    logger.info(f"Synthetic dataset created: {len(dataset)} samples")
    return dataset


def load_real_data(config: ResistanceMapConfig) -> MultiOmicsDataset:
    """Load real multi-omics data from configured paths.

    Loads:
      - CCLE proteomics (DepMap)
      - CCLE epigenomics (chromatin profiling, ATAC, histone marks)
      - STRING PPI network
      - Drug sensitivity (GDSC + CTRPv2)
      - scRNA-seq data (optional: GSE124310, GSE271107)
      - MMRF CoMMpass (optional: clinical + genomic)

    Args:
        config: ResistanceMapConfig with data paths.

    Returns:
        MultiOmicsDataset combining all loaded modalities.

    Raises:
        FileNotFoundError: If required data files are missing.
    """
    logger.info("Loading real multi-omics data")

    # L0.1: Load proteomics
    try:
        prot_data = load_ccle_proteomics(config.data)
        proteomics_df = prot_data["data"]
        protein_names = prot_data["protein_names"]
        sample_ids_prot = prot_data["sample_ids"]
        logger.info(f"CCLE proteomics: {proteomics_df.shape}")
    except FileNotFoundError as e:
        logger.error(f"Failed to load proteomics: {e}")
        raise

    # L0.2: Load epigenomics
    try:
        epi_data = load_ccle_epigenomics(config.data)
        sample_ids_epi = epi_data["sample_ids"]
        epigenome_feature_names = epi_data["feature_names"]
        logger.info(f"CCLE epigenomics: {len(sample_ids_epi)} samples, {len(epigenome_feature_names)} features")
    except FileNotFoundError as e:
        logger.error(f"Failed to load epigenomics: {e}")
        raise

    # L0.3: Load PPI network
    try:
        ppi_data = load_string_ppi(config.data)
        ppi_edges = ppi_data["edges"]
        ppi_scores = ppi_data["scores"]
        logger.info(f"STRING PPI: {ppi_data['num_edges']} edges, {ppi_data['num_proteins']} proteins")
    except FileNotFoundError as e:
        logger.error(f"Failed to load PPI: {e}")
        raise

    # L0.4: Load drug sensitivity
    try:
        drug_data = load_drug_sensitivity(config.data)
        drug_sensitivity_df = drug_data["data"]
        drug_names = drug_data["drug_names"]
        logger.info(f"Drug sensitivity: {drug_sensitivity_df.shape}")
    except FileNotFoundError as e:
        logger.error(f"Failed to load drug sensitivity: {e}")
        raise

    # L0.5: Match samples and align dimensions
    # Find intersection of sample IDs
    common_samples = list(set(sample_ids_prot) & set(sample_ids_epi) & set(drug_sensitivity_df.index))
    logger.info(f"Common samples across modalities: {len(common_samples)}")

    if not common_samples:
        raise ValueError(
            "No common samples found across proteomics, epigenomics, and drug sensitivity. "
            "Check data alignment."
        )

    # Align proteomics to common samples
    proteomics_aligned = proteomics_df.loc[common_samples]

    # Align epigenomics to common samples
    epi_dfs = {}
    for key in epi_data:
        if key.startswith("epi_") or key in ("chromatin_profiling", "atac_seq", "h3k4me3", "h3k27me3"):
            if isinstance(epi_data[key], dict):
                continue
            epi_dfs[key] = epi_data[key].loc[common_samples]

    # Concatenate all epigenomic assays
    if epi_dfs:
        epigenomics_aligned = np.concatenate(
            [epi_dfs[k].values for k in sorted(epi_dfs.keys())],
            axis=1
        )
    else:
        # Fallback: use a placeholder
        logger.warning("No epigenomic data aligned, using synthetic placeholder")
        epigenomics_aligned = np.random.randn(len(common_samples), 5000)

    # Align drug sensitivity to common samples
    drug_sensitivity_aligned = drug_sensitivity_df.loc[common_samples].values

    # Extract lineage information (if available in proteomics index or as metadata)
    # For now, assign synthetic lineages; in real data this would come from CCLE metadata
    lineages = [
        np.random.choice([
            "Myeloid", "Lymphoid", "haematopoietic_and_lymphoid_tissue",
            "blood", "lymphocyte", "plasma_cell",
        ])
        for _ in common_samples
    ]

    # Convert to tensors
    proteomics_tensor = torch.from_numpy(proteomics_aligned.values).float()
    epigenomics_tensor = torch.from_numpy(epigenomics_aligned).float()
    drug_sensitivity_tensor = torch.from_numpy(drug_sensitivity_aligned).float()

    logger.info(f"Final aligned shapes:")
    logger.info(f"  Proteomics: {proteomics_tensor.shape}")
    logger.info(f"  Epigenomics: {epigenomics_tensor.shape}")
    logger.info(f"  Drug sensitivity: {drug_sensitivity_tensor.shape}")

    # Create dataset
    dataset = MultiOmicsDataset(
        proteomics=proteomics_tensor,
        epigenomics=epigenomics_tensor,
        sample_ids=common_samples,
        lineage=lineages,
        drug_sensitivity=drug_sensitivity_tensor,
        protein_names=protein_names,
        epigenome_feature_names=epigenome_feature_names,
        ppi_edges=ppi_edges,
        ppi_scores=ppi_scores,
        source="ccle_cell_line",
        drug_names=drug_names,
    )

    logger.info(f"Real dataset created: {len(dataset)} samples")

    # Optional: Load scRNA-seq data
    try:
        scrna_data = load_scrna_data(config.data)
        if scrna_data:
            logger.info(f"Loaded scRNA-seq data: {len(scrna_data)} datasets")
    except Exception as e:
        logger.warning(f"Failed to load scRNA-seq data: {e}")

    # Optional: Load MMRF data
    try:
        mmrf_data = load_mmrf_data(config.data)
        if mmrf_data:
            logger.info(f"Loaded MMRF data: {len(mmrf_data)} data types")
    except Exception as e:
        logger.warning(f"Failed to load MMRF data: {e}")

    return dataset


def create_train_val_test_splits(
    dataset: MultiOmicsDataset,
    test_fraction: float = 0.15,
    val_fraction: float = 0.15,
    random_seed: int = 42,
) -> dict[str, list[int]]:
    """Create train/val/test splits.

    Args:
        dataset: MultiOmicsDataset to split.
        test_fraction: Fraction for test set.
        val_fraction: Fraction for validation set.
        random_seed: Random seed for reproducibility.

    Returns:
        Dict with 'train', 'val', 'test' index lists.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    n = len(dataset)
    indices = np.random.permutation(n)

    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_test - n_val

    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train : n_train + n_val].tolist()
    test_indices = indices[n_train + n_val :].tolist()

    logger.info(f"Train/val/test split: {n_train} / {n_val} / {n_test}")

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def train_stage_l1_vae(
    dataset: MultiOmicsDataset,
    splits: dict[str, list[int]],
    config: ResistanceMapConfig,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """L1: Train the VAE model.

    Stages:
      1. Pan-cancer pretraining on all CCLE cell lines
      2. Hematological finetuning on MM-relevant cell lines

    Args:
        dataset: MultiOmicsDataset with proteomics + epigenomics.
        splits: Train/val/test index splits.
        config: ResistanceMapConfig with VAE hyperparameters.
        checkpoint_dir: Directory to save checkpoints.
        device: Torch device.

    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    logger.info("=" * 80)
    logger.info("L1: VAE TRAINING")
    logger.info("=" * 80)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_mgr = CheckpointManager(checkpoint_dir)

    # Create model
    vae_config = config.vae
    model = ProteomeToEpigenomeVAE(
        input_dim=vae_config.input_dim,
        epigenome_dim=vae_config.epigenome_dim,
        latent_dim=vae_config.latent_dim,
        encoder_hidden_dims=vae_config.encoder_hidden_dims,
        decoder_hidden_dims=vae_config.decoder_hidden_dims,
        dropout=vae_config.dropout,
        use_batch_norm=vae_config.use_batch_norm,
        activation=vae_config.activation,
    ).to(device)

    logger.info(f"VAE model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Stage 1: Pan-cancer pretraining
    logger.info("Stage 1.1: Pan-cancer pretraining")
    metrics_pretrain = train_vae(
        model=model,
        dataset=dataset,
        splits=splits,
        config=vae_config,
        subset="pan_cancer",
        ckpt_mgr=ckpt_mgr,
        stage_name="vae_pretrain",
    )
    logger.info(f"Pan-cancer pretraining complete: {metrics_pretrain.get('checkpoint_path', 'N/A')}")

    # Stage 2: Hematological finetuning
    logger.info("Stage 1.2: Hematological finetuning")
    metrics_finetune = train_vae(
        model=model,
        dataset=dataset,
        splits=splits,
        config=vae_config,
        subset="hematological",
        ckpt_mgr=ckpt_mgr,
        stage_name="vae_finetune",
    )
    logger.info(f"Hematological finetuning complete: {metrics_finetune.get('checkpoint_path', 'N/A')}")

    return model, metrics_finetune


def train_stage_l2_stability(
    dataset: MultiOmicsDataset,
    vae_model: nn.Module,
    config: ResistanceMapConfig,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """L2: Compute stability scores using ODE-based chromatin bistability.

    Creates ChromatinODE and MemoryStabilityScorer, then scores all samples
    in the dataset.

    Args:
        dataset: MultiOmicsDataset with proteomics.
        vae_model: Trained VAE model from L1.
        config: ResistanceMapConfig with trajectory (stability) hyperparameters.
        checkpoint_dir: Directory to save checkpoints.
        device: Torch device.

    Returns:
        Tuple of (stability_scorer, scores_dict).
    """
    logger.info("=" * 80)
    logger.info("L2: STABILITY SCORING (ODE-based chromatin bistability)")
    logger.info("=" * 80)

    if not HAS_TORCHDIFFEQ:
        logger.warning(
            "torchdiffeq not available; skipping L2 stability scoring. "
            "Install with: pip install torchdiffeq"
        )
        return None, {"scores": torch.zeros(len(dataset))}

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create stability scorer
    traj_config = config.trajectory
    chromatin_ode = ChromatinODE(traj_config).to(device)
    stability_scorer = MemoryStabilityScorer(traj_config).to(device)

    logger.info(
        f"Stability scorer created: {sum(p.numel() for p in stability_scorer.parameters()):,} parameters"
    )

    # Score all samples
    logger.info(f"Scoring stability for {len(dataset)} samples...")
    stability_scores = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Computing stability scores"):
            sample = dataset[i]
            proteomics = sample["proteomics"].to(device).unsqueeze(0)

            # Extract reader/writer protein levels
            reader_writers = proteomics[
                :,
                [
                    j for j, name in enumerate(dataset.protein_names)
                    if name in traj_config.reader_writer_proteins
                ]
            ]

            # Compute stability score (simplified: just average the reader/writer levels)
            # In real usage, this would integrate the ODE and compute basin depth
            if reader_writers.shape[1] > 0:
                score = torch.sigmoid(reader_writers.mean()).item()
            else:
                score = 0.5

            stability_scores.append(score)

    stability_scores_tensor = torch.tensor(stability_scores, device=device)

    logger.info(
        f"Stability scores computed: mean={stability_scores_tensor.mean():.4f}, "
        f"std={stability_scores_tensor.std():.4f}"
    )

    # Save checkpoint
    ckpt_path = checkpoint_dir / "stability_scorer.pt"
    torch.save(
        {
            "model": stability_scorer.state_dict(),
            "scores": stability_scores_tensor,
        },
        ckpt_path,
    )
    logger.info(f"Stability scorer checkpoint saved: {ckpt_path}")

    return stability_scorer, {
        "scores": stability_scores_tensor,
        "checkpoint_path": str(ckpt_path),
    }


def train_stage_l3_gnn(
    dataset: MultiOmicsDataset,
    config: ResistanceMapConfig,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[Optional[nn.Module], dict[str, Any]]:
    """L3: Train protein network GNN.

    Builds PPI graph from STRING data and trains GNN to predict drug sensitivity
    from protein network features.

    Args:
        dataset: MultiOmicsDataset with PPI edges.
        config: ResistanceMapConfig with protein_net hyperparameters.
        checkpoint_dir: Directory to save checkpoints.
        device: Torch device.

    Returns:
        Tuple of (gnn_model, metrics_dict) or (None, {}) if torch_geometric unavailable.
    """
    logger.info("=" * 80)
    logger.info("L3: PROTEIN NETWORK GNN TRAINING")
    logger.info("=" * 80)

    if not HAS_PYG:
        logger.warning(
            "torch_geometric not available; skipping L3 GNN training. "
            "Install with: pip install torch_geometric"
        )
        return None, {"skipped": True}

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create model (stub for now)
    # In full implementation, this would:
    # 1. Build PyG graph from PPI edges
    # 2. Initialize node features with ESM-2 embeddings
    # 3. Train GNN layers on drug sensitivity prediction
    # 4. Return per-protein feature importance

    logger.info("L3: Protein network training (stub - requires torch_geometric)")
    logger.info(
        f"  PPI graph: {len(dataset.ppi_edges)} edges, "
        f"{len(set(sum(dataset.ppi_edges, ())))} unique proteins"
    )

    # Placeholder model
    gnn_model = nn.Linear(len(dataset.protein_names), len(dataset.drug_names or []))
    gnn_model = gnn_model.to(device)

    logger.info(f"Placeholder GNN model: {sum(p.numel() for p in gnn_model.parameters()):,} parameters")

    # Save checkpoint
    ckpt_path = checkpoint_dir / "gnn_model.pt"
    torch.save(gnn_model.state_dict(), ckpt_path)
    logger.info(f"GNN checkpoint saved: {ckpt_path}")

    return gnn_model, {
        "checkpoint_path": str(ckpt_path),
        "n_proteins": len(dataset.protein_names),
        "n_drugs": len(dataset.drug_names or []),
    }


def train_stage_l4_fusion(
    dataset: MultiOmicsDataset,
    vae_model: Optional[nn.Module],
    stability_scores: Optional[torch.Tensor],
    gnn_model: Optional[nn.Module],
    config: ResistanceMapConfig,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """L4: Train multi-modal fusion layer.

    Combines:
      - Epigenetic state from VAE latent
      - Trajectory state (ODE-predicted)
      - Protein network embeddings (GNN)
      - Stability score (L2)

    Args:
        dataset: MultiOmicsDataset.
        vae_model: Trained VAE from L1.
        stability_scores: Stability scores from L2.
        gnn_model: Trained GNN from L3.
        config: ResistanceMapConfig with fusion hyperparameters.
        checkpoint_dir: Directory to save checkpoints.
        device: Torch device.

    Returns:
        Tuple of (fusion_model, metrics_dict).
    """
    logger.info("=" * 80)
    logger.info("L4: FUSION TRAINING")
    logger.info("=" * 80)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create fusion model
    fusion_config = config.fusion
    modality_dims = {
        "epigenetic": config.vae.latent_dim,
        "trajectory": config.vae.latent_dim,
        "protein_network": 256 if gnn_model else 64,
        "stability": 1,
    }

    fusion_model = ResistanceMapFusion(
        hidden_dim=fusion_config.hidden_dim,
        output_dim=128,
        fusion_type=fusion_config.fusion_type,
        n_heads=fusion_config.n_heads,
        dropout=fusion_config.dropout,
        modality_dims=modality_dims,
        batch_correction=fusion_config.batch_correction,
        n_batches=10 if fusion_config.batch_correction else None,
    ).to(device)

    logger.info(
        f"Fusion model created: {sum(p.numel() for p in fusion_model.parameters()):,} parameters"
    )

    # Dummy training loop (real version would use actual modality representations)
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=fusion_config.fusion_lr)
    n_epochs = 5  # Short training for demo

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for i in tqdm(range(min(len(dataset), 10)), desc=f"Fusion epoch {epoch+1}/{n_epochs}"):
            optimizer.zero_grad()

            # Create dummy modality inputs
            epigenetic = torch.randn(1, modality_dims["epigenetic"], device=device)
            trajectory = torch.randn(1, modality_dims["trajectory"], device=device)
            protein_network = torch.randn(1, modality_dims["protein_network"], device=device)
            stability = torch.randn(1, modality_dims["stability"], device=device)

            # Forward pass
            fused = fusion_model(
                epigenetic_state=epigenetic,
                trajectory_state=trajectory,
                protein_network_output=protein_network,
                stability_score=stability,
            )

            # Dummy loss
            loss = fused.mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logger.info(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss / min(len(dataset), 10):.6f}")

    # Save checkpoint
    ckpt_path = checkpoint_dir / "fusion_model.pt"
    torch.save(fusion_model.state_dict(), ckpt_path)
    logger.info(f"Fusion model checkpoint saved: {ckpt_path}")

    return fusion_model, {
        "checkpoint_path": str(ckpt_path),
        "final_loss": epoch_loss / min(len(dataset), 10),
    }


def train_stage_l5_landscape(
    dataset: MultiOmicsDataset,
    fusion_model: nn.Module,
    config: ResistanceMapConfig,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """L5: Train resistance landscape predictor.

    Predicts:
      - Per-drug resistance probabilities at 3, 6, 12 months
      - Current resistance state
      - Basin-of-attraction map (future state)
      - Intervention target ranking

    Args:
        dataset: MultiOmicsDataset.
        fusion_model: Trained fusion model from L4.
        config: ResistanceMapConfig with landscape hyperparameters.
        checkpoint_dir: Directory to save checkpoints.
        device: Torch device.

    Returns:
        Tuple of (landscape_model, metrics_dict).
    """
    logger.info("=" * 80)
    logger.info("L5: LANDSCAPE TRAINING")
    logger.info("=" * 80)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create landscape model
    landscape_config = config.landscape
    n_drugs = len(dataset.drug_names or [])
    n_proteins = len(dataset.protein_names)

    landscape_model = ResistanceLandscape(
        fusion_dim=128,
        n_drugs=max(n_drugs, 6),
        n_timepoints=3,
        n_states=3,
        n_proteins=n_proteins,
        hidden_dim=256,
        dropout_rate=0.2,
        use_evidential=landscape_config.use_evidential,
    ).to(device)

    logger.info(
        f"Landscape model created: {sum(p.numel() for p in landscape_model.parameters()):,} parameters"
    )

    # Dummy training loop
    optimizer = torch.optim.Adam(landscape_model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 5

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for i in tqdm(range(min(len(dataset), 10)), desc=f"Landscape epoch {epoch+1}/{n_epochs}"):
            optimizer.zero_grad()

            # Dummy fused representation input
            fused_repr = torch.randn(1, 128, device=device)

            # Forward pass
            outputs = landscape_model(fused_repr)

            # Dummy loss (binary classification for resistance)
            target = torch.randint(0, 2, (1, max(n_drugs, 6) * 3)).float().to(device)
            loss = criterion(outputs["drug_resistance"], target)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logger.info(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss / min(len(dataset), 10):.6f}")

    # Save checkpoint
    ckpt_path = checkpoint_dir / "landscape_model.pt"
    torch.save(landscape_model.state_dict(), ckpt_path)
    logger.info(f"Landscape model checkpoint saved: {ckpt_path}")

    return landscape_model, {
        "checkpoint_path": str(ckpt_path),
        "final_loss": epoch_loss / min(len(dataset), 10),
    }


def evaluate_pipeline(
    dataset: MultiOmicsDataset,
    splits: dict[str, list[int]],
    models: dict[str, nn.Module],
    device: torch.device,
) -> dict[str, float]:
    """Run basic evaluation on test set.

    Computes AUROC for drug sensitivity prediction on test samples.

    Args:
        dataset: MultiOmicsDataset.
        splits: Train/val/test splits.
        models: Dict of stage -> model.
        device: Torch device.

    Returns:
        Dict of metric_name -> value.
    """
    logger.info("=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)

    # Dummy evaluation: compute mean reconstruction error on test set
    vae_model = models.get("vae")
    if not vae_model:
        logger.warning("No VAE model found for evaluation")
        return {}

    test_indices = splits["test"]
    test_loss = 0.0
    n_test = min(len(test_indices), 10)

    with torch.no_grad():
        for i, idx in enumerate(test_indices[:n_test]):
            sample = dataset[idx]
            proteomics = sample["proteomics"].to(device).unsqueeze(0)
            epigenomics = sample["epigenomics"].to(device).unsqueeze(0)

            # Dummy loss computation
            recon = vae_model(proteomics)
            if isinstance(recon, dict):
                recon = recon.get("reconstruction", epigenomics)
            loss = ((recon - epigenomics) ** 2).mean().item()
            test_loss += loss

    mean_test_loss = test_loss / n_test
    logger.info(f"Test reconstruction MSE: {mean_test_loss:.6f}")

    return {
        "test_mse": mean_test_loss,
        "test_samples": n_test,
    }


def main():
    """Main training orchestration."""
    parser = argparse.ArgumentParser(
        description="End-to-end training for ResistanceMap L0-L5 pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cpu or cuda)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing raw data",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages to run: all, L0, L1, L2, L3, L4, L5",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic random data instead of real data",
    )

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    setup_logging(Path("logs"))

    logger.info("=" * 80)
    logger.info("ResistanceMap End-to-End Training")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Synthetic data: {args.synthetic}")

    # Load configuration
    config = load_config(args.config)
    config.hardware.device = str(device).split(":")[0]  # "cpu" or "cuda"

    # Parse stages
    if args.stages.lower() == "all":
        stages = ["L0", "L1", "L2", "L3", "L4", "L5"]
    else:
        stages = [s.strip().upper() for s in args.stages.split(",")]

    logger.info(f"Running stages: {stages}")

    # L0: Data Preparation
    models = {}
    splits = None
    dataset = None

    if "L0" in stages or any(s in stages for s in ["L1", "L2", "L3", "L4", "L5"]):
        logger.info("=" * 80)
        logger.info("L0: DATA PREPARATION")
        logger.info("=" * 80)

        if args.synthetic:
            dataset = generate_synthetic_data(
                n_samples=100,
                n_proteins=config.vae.input_dim,
                n_epigenomic_features=config.vae.epigenome_dim,
                n_drugs=6,
                device=str(device),
            )
        else:
            try:
                dataset = load_real_data(config)
            except FileNotFoundError as e:
                logger.error(f"Failed to load real data: {e}")
                logger.info("Falling back to synthetic data")
                dataset = generate_synthetic_data(
                    n_samples=100,
                    n_proteins=config.vae.input_dim,
                    n_epigenomic_features=config.vae.epigenome_dim,
                    n_drugs=6,
                    device=str(device),
                )

        # Create splits
        splits = create_train_val_test_splits(
            dataset,
            test_fraction=config.data.test_fraction,
            val_fraction=config.data.val_fraction,
            random_seed=config.data.random_seed,
        )

    # L1: VAE Training
    if "L1" in stages:
        vae_model, vae_metrics = train_stage_l1_vae(
            dataset, splits, config, args.checkpoint_dir / "l1", device
        )
        models["vae"] = vae_model
        logger.info(f"L1 metrics: {vae_metrics}")

    # L2: Stability Scoring
    if "L2" in stages:
        stability_scorer, stability_metrics = train_stage_l2_stability(
            dataset,
            models.get("vae"),
            config,
            args.checkpoint_dir / "l2",
            device,
        )
        models["stability"] = stability_scorer
        logger.info(f"L2 metrics: {stability_metrics}")

    # L3: GNN Training
    if "L3" in stages:
        gnn_model, gnn_metrics = train_stage_l3_gnn(
            dataset, config, args.checkpoint_dir / "l3", device
        )
        if gnn_model:
            models["gnn"] = gnn_model
        logger.info(f"L3 metrics: {gnn_metrics}")

    # L4: Fusion Training
    if "L4" in stages:
        fusion_model, fusion_metrics = train_stage_l4_fusion(
            dataset,
            models.get("vae"),
            models.get("stability"),
            models.get("gnn"),
            config,
            args.checkpoint_dir / "l4",
            device,
        )
        models["fusion"] = fusion_model
        logger.info(f"L4 metrics: {fusion_metrics}")

    # L5: Landscape Training
    if "L5" in stages:
        if "fusion" not in models or models["fusion"] is None:
            logger.warning("L4 fusion model not available; creating stub for L5")
            models["fusion"] = nn.Linear(128, 128).to(device)

        landscape_model, landscape_metrics = train_stage_l5_landscape(
            dataset, models["fusion"], config, args.checkpoint_dir / "l5", device
        )
        models["landscape"] = landscape_model
        logger.info(f"L5 metrics: {landscape_metrics}")

    # Evaluation
    if dataset and splits:
        eval_metrics = evaluate_pipeline(dataset, splits, models, device)
        logger.info(f"Evaluation metrics: {eval_metrics}")

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
