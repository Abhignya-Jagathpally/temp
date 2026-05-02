#!/usr/bin/env python3
"""Generate all ablation YAML configs programmatically.

Each ablation modifies the full config to disable one component or modality,
producing a self-contained YAML file that can be passed to the trainer.

Usage:
    python configs/ablation_configs.py --base configs/full.yaml --output configs/ablations/
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# Ablation Definitions
# =============================================================================

ABLATION_REGISTRY: dict[str, dict[str, Any]] = {
    # ── Component ablations ──────────────────────────────────────────────
    "no_identifiable_ode": {
        "description": "Replace Identifiable Neural ODE with vanilla Neural ODE. "
                       "Removes interaction matrix sparsity and diagonal dominance constraints.",
        "modifications": {
            "model.ode.solver": "euler",
            "model.ode.interaction_matrix_dim": 0,
            "training.loss_weights.sparsity_l1": 0.0,
            "training.loss_weights.diagonal_dominance": 0.0,
            "training.loss_weights.orthogonality": 0.0,
        },
        "tag": "component",
    },
    "no_causal_fusion": {
        "description": "Replace causal attention fusion with simple concatenation. "
                       "Removes DAG enforcement and causal consistency loss.",
        "modifications": {
            "model.causal_fusion.num_heads": 0,
            "model.causal_fusion.dag_enforce": False,
            "training.loss_weights.causal_consistency": 0.0,
        },
        "tag": "component",
    },
    "no_beta_tcvae": {
        "description": "Replace beta-TCVAE with standard VAE (beta=1, no TC decomposition).",
        "modifications": {
            "model.vae.beta": 1.0,
            "training.loss_weights.tc_penalty": 0.0,
        },
        "tag": "component",
    },
    "no_lyapunov": {
        "description": "Remove Lyapunov stability constraint from ODE training.",
        "modifications": {
            "training.loss_weights.lyapunov_stability": 0.0,
        },
        "tag": "component",
    },
    "no_sparsity": {
        "description": "Remove L1 sparsity penalty on interaction matrix A.",
        "modifications": {
            "training.loss_weights.sparsity_l1": 0.0,
        },
        "tag": "component",
    },
    "no_pathway_anchoring": {
        "description": "Remove pathway anchor losses that tie latent dimensions to known biology.",
        "modifications": {
            "training.loss_weights.pathway_anchor": 0.0,
        },
        "tag": "component",
    },
    # ── Modality ablations ───────────────────────────────────────────────
    "no_transcriptomics": {
        "description": "Remove RNA-seq modality from fusion input.",
        "modifications": {
            "data.rnaseq_path": None,
            "model.cross_attention.num_modalities_override": 3,
        },
        "tag": "modality",
        "disabled_modality": "rnaseq",
    },
    "no_proteomics": {
        "description": "Remove proteomics modality from fusion input.",
        "modifications": {
            "data.proteomics_path": None,
            "model.cross_attention.num_modalities_override": 3,
        },
        "tag": "modality",
        "disabled_modality": "proteomics",
    },
    "no_epigenomics": {
        "description": "Remove epigenomics (inferred from VAE) from fusion input.",
        "modifications": {
            "model.vae.enabled": False,
            "model.cross_attention.num_modalities_override": 3,
        },
        "tag": "modality",
        "disabled_modality": "epigenomics",
    },
    "no_ppi": {
        "description": "Remove PPI network (GAT) modality from fusion input.",
        "modifications": {
            "data.ppi_path": None,
            "model.gat.enabled": False,
            "model.cross_attention.num_modalities_override": 3,
        },
        "tag": "modality",
        "disabled_modality": "ppi",
    },
    # ── Data scaling ablations ───────────────────────────────────────────
    "data_10pct": {
        "description": "Train on 10% of training data to test scaling behavior.",
        "modifications": {
            "data.train_subsample_frac": 0.10,
        },
        "tag": "scaling",
    },
    "data_25pct": {
        "description": "Train on 25% of training data to test scaling behavior.",
        "modifications": {
            "data.train_subsample_frac": 0.25,
        },
        "tag": "scaling",
    },
    "data_50pct": {
        "description": "Train on 50% of training data to test scaling behavior.",
        "modifications": {
            "data.train_subsample_frac": 0.50,
        },
        "tag": "scaling",
    },
}


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated key path.

    Args:
        d: The dictionary to modify.
        dotted_key: Key path like 'model.ode.solver'.
        value: The value to set.
    """
    keys = dotted_key.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _get_nested(d: dict, dotted_key: str, default: Any = None) -> Any:
    """Get a value from a nested dict using dot-separated key path."""
    keys = dotted_key.split(".")
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def generate_ablation_config(
    base_config: dict,
    ablation_name: str,
    seed: int = 42,
) -> dict:
    """Generate a single ablation config from the base config.

    Args:
        base_config: The full base configuration dictionary.
        ablation_name: Name of the ablation from ABLATION_REGISTRY.
        seed: Random seed for this run.

    Returns:
        Modified configuration dictionary for the ablation.

    Raises:
        KeyError: If ablation_name is not in the registry.
    """
    if ablation_name not in ABLATION_REGISTRY:
        raise KeyError(
            f"Unknown ablation '{ablation_name}'. "
            f"Available: {list(ABLATION_REGISTRY.keys())}"
        )

    ablation_def = ABLATION_REGISTRY[ablation_name]
    config = copy.deepcopy(base_config)

    # Apply all modifications
    for key, value in ablation_def["modifications"].items():
        _set_nested(config, key, value)

    # Override seed
    _set_nested(config, "reproducibility.seed", seed)

    # Add ablation metadata
    config["_ablation"] = {
        "name": ablation_name,
        "description": ablation_def["description"],
        "tag": ablation_def["tag"],
        "seed": seed,
        "base_config": "configs/full.yaml",
    }

    # Update experiment tracking tags
    tags = _get_nested(config, "tracking.wandb.tags", [])
    if isinstance(tags, list):
        tags = tags + [f"ablation:{ablation_name}", f"seed:{seed}"]
        _set_nested(config, "tracking.wandb.tags", tags)

    # Update output directory
    base_output = _get_nested(config, "output.base_dir", "results/icml_submission")
    _set_nested(
        config,
        "output.base_dir",
        f"{base_output}/ablations/{ablation_name}/seed_{seed}",
    )

    return config


def generate_all_ablation_configs(
    base_config: dict,
    output_dir: str,
    seeds: list[int] | None = None,
    conditions: list[str] | None = None,
) -> list[str]:
    """Generate YAML files for all ablation conditions and seeds.

    Args:
        base_config: The full base configuration dictionary.
        output_dir: Directory to write YAML files to.
        seeds: List of random seeds. Defaults to [42, 123, 456, 789, 2024].
        conditions: Ablation conditions to generate. Defaults to all.

    Returns:
        List of paths to generated YAML files.
    """
    if seeds is None:
        seeds = base_config.get("ablation", {}).get(
            "seeds", [42, 123, 456, 789, 2024]
        )
    if conditions is None:
        conditions = base_config.get("ablation", {}).get(
            "conditions", list(ABLATION_REGISTRY.keys())
        )

    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    for condition in conditions:
        if condition not in ABLATION_REGISTRY:
            logger.warning(f"Skipping unknown ablation condition: {condition}")
            continue

        condition_dir = os.path.join(output_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)

        for seed in seeds:
            config = generate_ablation_config(base_config, condition, seed=seed)
            filename = f"{condition}_seed{seed}.yaml"
            filepath = os.path.join(condition_dir, filename)

            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            generated_files.append(filepath)
            logger.debug(f"Generated: {filepath}")

        logger.info(
            f"  {condition}: {len(seeds)} configs "
            f"({ABLATION_REGISTRY[condition]['tag']})"
        )

    logger.info(
        f"Generated {len(generated_files)} ablation configs in {output_dir}/"
    )
    return generated_files


def get_ablation_summary() -> dict[str, list[str]]:
    """Get a summary of ablation conditions grouped by tag.

    Returns:
        Dict mapping tag -> list of ablation names.
    """
    summary: dict[str, list[str]] = {}
    for name, defn in ABLATION_REGISTRY.items():
        tag = defn["tag"]
        if tag not in summary:
            summary[tag] = []
        summary[tag].append(name)
    return summary


def validate_ablation_config(config: dict) -> list[str]:
    """Validate that an ablation config is internally consistent.

    Args:
        config: The ablation configuration dictionary.

    Returns:
        List of warning messages (empty if valid).
    """
    warnings = []

    # Check that loss weights are non-negative
    loss_weights = _get_nested(config, "training.loss_weights", {})
    for key, value in loss_weights.items():
        if isinstance(value, (int, float)) and value < 0:
            warnings.append(f"Negative loss weight: training.loss_weights.{key} = {value}")

    # Check that seed is set
    seed = _get_nested(config, "reproducibility.seed")
    if seed is None:
        warnings.append("No seed set in reproducibility.seed")

    # Check ablation metadata
    ablation_meta = config.get("_ablation", {})
    if not ablation_meta:
        warnings.append("No _ablation metadata found")

    return warnings


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate ablation YAML configs for ResistanceMap v6 experiments."
    )
    parser.add_argument(
        "--base", type=str, default="configs/full.yaml",
        help="Path to base full.yaml configuration file.",
    )
    parser.add_argument(
        "--output", type=str, default="configs/ablations",
        help="Output directory for generated ablation configs.",
    )
    parser.add_argument(
        "--conditions", nargs="*", default=None,
        help="Specific ablation conditions to generate (default: all).",
    )
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=None,
        help="Random seeds to use (default: from base config).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available ablation conditions and exit.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.list:
        summary = get_ablation_summary()
        print("\nAvailable ablation conditions:")
        print("=" * 60)
        for tag, names in summary.items():
            print(f"\n  [{tag.upper()}]")
            for name in names:
                desc = ABLATION_REGISTRY[name]["description"]
                print(f"    {name:30s} — {desc[:60]}...")
        print(f"\nTotal: {len(ABLATION_REGISTRY)} conditions")
        return

    # Load base config
    with open(args.base, "r") as f:
        base_config = yaml.safe_load(f)

    files = generate_all_ablation_configs(
        base_config=base_config,
        output_dir=args.output,
        seeds=args.seeds,
        conditions=args.conditions,
    )

    print(f"\nGenerated {len(files)} ablation config files.")


if __name__ == "__main__":
    main()
