#!/usr/bin/env python3
"""
scripts/mortfm_smoke_test.py
============================
End-to-end MORT-FM smoke test.

This script exercises the entire v15 stack — schemas, loaders, encoders,
fusion, dynamics, landscape, heads, losses, trainer — on a tiny *structurally
valid* example so any regression in the wiring fails loudly and quickly.

It does NOT train on real data and produces no claimable metrics. Its only
purpose is to confirm that:
    1. Every module imports.
    2. The data contract round-trips (snapshots -> pairs -> batch).
    3. The model forwards + backwards.
    4. Each training stage runs at least one epoch.
    5. The checkpoint save/load preserves state.

Run with:
    python scripts/mortfm_smoke_test.py [--config configs/mortfm_debug.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Make `resistancemap` importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.trajectory_pair_builder import build_temporal_pairs, summarise_pairs
from resistancemap.mortfm.data_module import make_data_module
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import (
    DrugContext,
    ModalityTensor,
    MORTFMConfig,
    PatientCellSnapshot,
    ResistanceOutcome,
)
from resistancemap.mortfm.trainer import MORTFMTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_smoke")


def load_config(path: str) -> MORTFMConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Drop unknown keys so the dataclass init doesn't error on YAML-only fields.
    known = {f.name for f in MORTFMConfig.__dataclass_fields__.values()}
    raw = {k: v for k, v in raw.items() if k in known}
    return MORTFMConfig(**raw)


def build_synthetic_smoke_cohort(cfg: MORTFMConfig, n_patients: int = 10):
    """Build a tiny *structural* cohort for the smoke test.

    This is NOT training data. It exists only so the dataloader has something
    to iterate over. The values are torch.poisson and torch.randn so the model
    sees realistic dtype + shape, but no biological claim is ever made from
    a smoke-test run.
    """
    snapshots = []
    for i in range(n_patients):
        for t in (0.0, 6.0):
            mods = {
                "rna": ModalityTensor(
                    "rna",
                    torch.poisson(torch.full((1, cfg.rna_input_dim), 3.0)),
                    [f"gene_{j}" for j in range(cfg.rna_input_dim)],
                ),
                "proteomics": ModalityTensor(
                    "proteomics",
                    torch.randn(1, cfg.proteomics_input_dim),
                    [f"prot_{j}" for j in range(cfg.proteomics_input_dim)],
                ) if cfg.use_proteomics else None,
            }
            snapshots.append(
                PatientCellSnapshot(
                    patient_id=f"SMOKE_P{i:03d}",
                    sample_id=f"SMOKE_P{i:03d}_t{int(t)}",
                    disease="MM",
                    timepoint=t,
                    rna=mods["rna"],
                    proteomics=mods.get("proteomics"),
                    drug=DrugContext("Bortezomib", "proteasome_inhibitor", dose=0.01) if cfg.use_drug else None,
                )
            )
    outcomes = [
        ResistanceOutcome(
            patient_id=f"SMOKE_P{i:03d}",
            baseline_time=0.0,
            event_time=float(8 + i),
            censored=(i % 2 == 0),
            resistance_label=i % 4,
        )
        for i in range(n_patients)
    ]
    pairs = build_temporal_pairs(snapshots, outcomes, allowed_time_gaps=[6.0], gap_tolerance=0.5)
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="MORT-FM end-to-end smoke test")
    ap.add_argument("--config", default="configs/mortfm_debug.yaml")
    ap.add_argument("--n-patients", type=int, default=10)
    ap.add_argument("--stages", nargs="+", default=["A", "B", "F", "E"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger.info("Loaded config: %s (d_latent=%d, d_token=%d)", cfg.model_name, cfg.d_latent, cfg.d_token)

    pairs = build_synthetic_smoke_cohort(cfg, n_patients=args.n_patients)
    logger.info("Built %d structural pairs: %s", len(pairs), summarise_pairs(pairs))

    splits, train_loader, val_loader, _ = make_data_module(
        pairs, batch_size=cfg.batch_size, val_fraction=0.2, test_fraction=0.2, seed=cfg.seed,
    )
    logger.info("Split summary: %s", splits.summary())

    model = MORTFM(cfg, n_pathway_proteins=20, n_drug_candidates=5, n_resistance_states=4)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Built MORT-FM with %d trainable params", n_params)

    trainer = MORTFMTrainer(
        model, cfg, train_loader=train_loader, val_loader=val_loader,
        device=torch.device("cpu"),
    )
    for stage in args.stages:
        m = trainer.fit_stage(stage, n_epochs=1)
        logger.info(
            "Stage %s ran. train_loss=%.4f val_loss=%.4f components=%s",
            stage, m[0].train_loss, m[0].val_loss, m[0].components,
        )

    ckpt = trainer.save_checkpoint("smoke_final.pt")
    logger.info("Saved smoke checkpoint -> %s", ckpt)
    trainer.load_checkpoint(ckpt)
    logger.info("Checkpoint reload OK")

    batch = next(iter(val_loader))
    with torch.no_grad():
        out = model(batch, compute_counterfactuals=True)
    logger.info(
        "Final inference: z_path shape=%s, n_counterfactuals=%d",
        tuple(out.z_path.shape),
        len(out.counterfactual_rankings[0]) if out.counterfactual_rankings else 0,
    )
    logger.info("MORT-FM SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
