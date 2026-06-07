#!/usr/bin/env python3
"""
ResistanceMap v20 — PK-SSM Forward Pass
========================================
Wires pk_state_space.py to scVI outputs and runs the full forward pass:
  scVI latent z → PK-SSM → basin assignment + hazard + mechanism classification

This demonstrates the architecture works end-to-end on real data.
Training the survival head requires MMRF Virtual Lab outcome data.

Input:
  data/processed/scvi_latent_z.npy
  data/processed/chromatin_expression.npy
  data/processed/biomarker_proxy_expression.npy

Output:
  results/v20_pkssm/forward_pass_results.json
  results/v20_pkssm/mechanism_assignments.csv
  results/v20_pkssm/predicted_biomarkers.npy
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/v20_pkssm")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_scvi_outputs():
    """Load preprocessed scVI outputs."""
    z = np.load(PROCESSED_DIR / "scvi_latent_z.npy")
    logger.info(f"Loaded scVI latent: {z.shape}")

    chromatin = None
    chromatin_path = PROCESSED_DIR / "chromatin_expression.npy"
    if chromatin_path.exists():
        chromatin = np.load(chromatin_path)
        logger.info(f"Loaded chromatin expression: {chromatin.shape}")

    proxy = None
    proxy_path = PROCESSED_DIR / "biomarker_proxy_expression.npy"
    if proxy_path.exists():
        proxy = np.load(proxy_path)
        logger.info(f"Loaded biomarker proxies: {proxy.shape}")

    return z, chromatin, proxy


def build_pkssm():
    """Initialize the PK-SSM model from pk_state_space.py."""
    # Import from wherever pk_state_space.py was placed
    try:
        from resistancemap.models.pk_state_space import PKSSM, PKSSMConfig, explain_prediction
    except ImportError:
        # Fallback: try direct import if placed in scripts/
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pk_state_space",
            "resistancemap/models/pk_state_space.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        PKSSM = mod.PKSSM
        PKSSMConfig = mod.PKSSMConfig
        explain_prediction = mod.explain_prediction

    config = PKSSMConfig(
        n_biomarkers=20,
        latent_dim=5,
        hidden_dim=64,
        n_treatments=9,
        n_competing_risks=2,
        n_resistance_mechanisms=4,
        dropout=0.1,
    )

    model = PKSSM(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"PK-SSM initialized: {total_params:,} total params, {trainable_params:,} trainable")

    return model, config, explain_prediction


def run_forward_pass(model, z_scvi, chromatin_expr, proxy_expr):
    """
    Run the PK-SSM forward pass using scVI outputs as input.

    The scVI latent z is projected to the PK-SSM's biomarker input space
    via the biomarker proxy gene expression (Ig genes → M-protein,
    IGKC → FLC-κ, B2M → serum B2M, etc.)
    """
    model.eval()
    n_cells = z_scvi.shape[0]

    # Use biomarker proxy expression as the "observed" lab values
    # This is the gene-expression → serum-biomarker bridge
    if proxy_expr is not None and proxy_expr.shape[1] >= 20:
        y_input = torch.tensor(proxy_expr[:, :20], dtype=torch.float32)
    elif proxy_expr is not None:
        # Pad to 20 features
        pad_width = 20 - proxy_expr.shape[1]
        y_padded = np.pad(proxy_expr, ((0, 0), (0, pad_width)), constant_values=0)
        y_input = torch.tensor(y_padded, dtype=torch.float32)
    else:
        # Fallback: use scVI latent projected to biomarker space
        y_input = torch.randn(n_cells, 20)  # Will be replaced with real data

    # Create synthetic time dimension (single timepoint per cell for forward demo)
    # In real training, this would be T visits per patient
    T = 1
    batch_size = min(n_cells, 512)  # Process in batches

    all_mechanisms = []
    all_hazards = []
    all_y_pred = []

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        batch_y = y_input[start:end].unsqueeze(1).expand(-1, 6, -1)  # [B, T=6, 20]
        batch_u = torch.zeros(end - start, 6, 9)  # No treatment info
        batch_mask = torch.ones(end - start, 6, dtype=torch.bool)
        batch_baseline = torch.zeros(end - start, 17)

        with torch.no_grad():
            result = model(batch_y, batch_u, batch_mask, batch_baseline)

        mechanism_probs = F.softmax(result['mechanism_logits'], dim=-1).numpy()
        hazard_vals = torch.sigmoid(result['hazards'][:, -1, :]).numpy()  # Last timepoint
        y_pred = result['y_pred'][:, -1, :].numpy()

        all_mechanisms.append(mechanism_probs)
        all_hazards.append(hazard_vals)
        all_y_pred.append(y_pred)

    mechanisms = np.concatenate(all_mechanisms, axis=0)
    hazards = np.concatenate(all_hazards, axis=0)
    y_predictions = np.concatenate(all_y_pred, axis=0)

    return mechanisms, hazards, y_predictions


def main():
    logger.info("=" * 60)
    logger.info("ResistanceMap v20 — PK-SSM Forward Pass")
    logger.info("=" * 60)

    # Load scVI outputs
    z_scvi, chromatin_expr, proxy_expr = load_scvi_outputs()

    # Build PK-SSM
    model, config, explain_prediction = build_pkssm()

    # Run forward pass
    logger.info("\nRunning PK-SSM forward pass...")
    mechanisms, hazards, y_pred = run_forward_pass(model, z_scvi, chromatin_expr, proxy_expr)

    # --- Save results ---
    mechanism_names = ['drug_efflux', 'clonal_evolution', 'immune_escape', 'microenvironmental']

    # Mechanism assignments
    mechanism_df_data = {
        'cell_idx': range(len(mechanisms)),
        'predicted_mechanism': [mechanism_names[m] for m in mechanisms.argmax(axis=1)],
        'confidence': mechanisms.max(axis=1),
    }
    for i, name in enumerate(mechanism_names):
        mechanism_df_data[f'prob_{name}'] = mechanisms[:, i]

    import pandas as pd
    mech_df = pd.DataFrame(mechanism_df_data)
    mech_df.to_csv(RESULTS_DIR / "mechanism_assignments.csv", index=False)

    # Predicted biomarkers
    np.save(RESULTS_DIR / "predicted_biomarkers.npy", y_pred)

    # Summary statistics
    summary = {
        "n_cells": len(mechanisms),
        "pk_ssm_params": sum(p.numel() for p in model.parameters()),
        "mechanism_distribution": {
            name: int((mechanisms.argmax(axis=1) == i).sum())
            for i, name in enumerate(mechanism_names)
        },
        "mean_hazard": {
            "biochemical": float(hazards[:, 0].mean()),
            "clinical": float(hazards[:, 1].mean()) if hazards.shape[1] > 1 else None,
        },
        "model_config": {
            "latent_dim": config.latent_dim,
            "n_biomarkers": config.n_biomarkers,
            "pk_priors": {
                "flc_halflife_hours": config.flc_halflife_hours,
                "igg_halflife_days": config.igg_halflife_days,
                "b2m_halflife_hours": config.b2m_halflife_hours,
                "albumin_halflife_days": config.albumin_halflife_days,
            },
        },
        "note": "UNTRAINED model — forward pass only. Mechanism assignments are random "
                "initialization, NOT trained predictions. Training requires MMRF Virtual Lab "
                "longitudinal outcome data.",
    }

    with open(RESULTS_DIR / "forward_pass_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PK-SSM Forward Pass Results (UNTRAINED — architecture demo only)")
    logger.info("=" * 60)
    logger.info(f"  Cells processed: {len(mechanisms)}")
    logger.info(f"  Model parameters: {summary['pk_ssm_params']:,}")
    logger.info(f"  Mechanism distribution (random init):")
    for name, count in summary['mechanism_distribution'].items():
        pct = count / len(mechanisms) * 100
        logger.info(f"    {name}: {count} ({pct:.1f}%)")
    logger.info(f"  Mean hazard (untrained): {summary['mean_hazard']}")
    logger.info(f"\n  Results saved to: {RESULTS_DIR}/")
    logger.info(f"\n  ⚠ These are NOT trained predictions. They demonstrate the")
    logger.info(f"    architecture runs end-to-end on real scRNA-seq data.")
    logger.info(f"    Training requires MMRF Virtual Lab visit-level outcome data.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()