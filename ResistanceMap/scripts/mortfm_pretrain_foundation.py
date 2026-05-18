#!/usr/bin/env python3
"""
scripts/mortfm_pretrain_foundation.py
=====================================
Pre-train the MORT-FM foundation encoder (Stages A + B) on a cell-line
multi-omics cohort.

This script:
    1. Loads a config (default ``configs/mortfm_cellline_pretrain.yaml``).
    2. Loads pickled pairs produced by ``mortfm_build_temporal_pairs.py``.
    3. Trains Stage A (per-modality recon) then Stage B (masked-modality +
       contrastive cross-modal).
    4. Saves the resulting checkpoint to ``cfg.checkpoint_dir/stage_B_complete.pt``.

This does NOT train survival or trajectory heads — those are stages D/E/F,
covered by ``mortfm_train_survival.py`` and ``mortfm_train_trajectory.py``.

Refuses to run if the input pairs file is empty.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.mortfm.data_module import make_data_module
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import MORTFMConfig
from resistancemap.mortfm.trainer import MORTFMTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_pretrain")


def load_config(path: str) -> MORTFMConfig:
    raw = yaml.safe_load(Path(path).read_text())
    known = {f.name for f in MORTFMConfig.__dataclass_fields__.values()}
    return MORTFMConfig(**{k: v for k, v in raw.items() if k in known})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mortfm_cellline_pretrain.yaml")
    ap.add_argument("--pairs", required=True, help="Pickled list[TemporalTrainingPair]")
    ap.add_argument("--stages", nargs="+", default=["A", "B"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.pairs, "rb") as f:
        pairs = pickle.load(f)
    if not pairs:
        logger.error("No pairs in %s -- aborting (cannot pretrain on empty data).", args.pairs)
        return 1

    splits, train_loader, val_loader, _ = make_data_module(
        pairs, batch_size=cfg.batch_size, seed=cfg.seed,
    )
    logger.info("Cohort: %s", splits.summary())

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    model = MORTFM(cfg, n_pathway_proteins=500, n_drug_candidates=cfg.drug_n_drugs,
                   n_resistance_states=4)
    trainer = MORTFMTrainer(model, cfg, train_loader=train_loader, val_loader=val_loader,
                             device=device)

    for stage in args.stages:
        logger.info("=== Stage %s ===", stage)
        epochs_map = {"A": cfg.pretrain_epochs, "B": cfg.pretrain_epochs,
                      "C": cfg.finetune_epochs, "D": cfg.finetune_epochs,
                      "E": cfg.trajectory_epochs, "F": cfg.survival_epochs, "G": 5, "H": 5}
        trainer.fit_stage(stage, n_epochs=epochs_map[stage])
        trainer.save_checkpoint(f"stage_{stage}_complete.pt")
    logger.info("Pretraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
