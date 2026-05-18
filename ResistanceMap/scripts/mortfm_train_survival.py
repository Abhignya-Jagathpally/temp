#!/usr/bin/env python3
"""
scripts/mortfm_train_survival.py
================================
Train MORT-FM Stage F (survival / time-to-resistance) + Stage H (calibration).

Refuses to release survival claims if patient count is below
``cfg.min_patient_n_for_survival_claim``.
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
logger = logging.getLogger("mortfm_train_survival")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mortfm_patient_finetune.yaml")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    known = {f.name for f in MORTFMConfig.__dataclass_fields__.values()}
    cfg = MORTFMConfig(**{k: v for k, v in raw.items() if k in known})

    with open(args.pairs, "rb") as f:
        pairs = pickle.load(f)
    survival_rows = [p for p in pairs if p.outcome.has_survival_label()]
    n_patients = len({p.x_t.patient_id for p in survival_rows})
    logger.info("Survival rows: %d (from %d unique patients)", len(survival_rows), n_patients)
    if n_patients < cfg.min_patient_n_for_survival_claim:
        logger.warning(
            "Only %d patients with survival labels; below claim threshold %d. "
            "Training will proceed but Claim Critic will block survival figures.",
            n_patients, cfg.min_patient_n_for_survival_claim,
        )

    splits, train_loader, val_loader, _ = make_data_module(
        pairs, batch_size=cfg.batch_size, seed=cfg.seed,
    )
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    model = MORTFM(cfg, n_pathway_proteins=500, n_drug_candidates=cfg.drug_n_drugs,
                   n_resistance_states=4)
    trainer = MORTFMTrainer(model, cfg, train_loader=train_loader, val_loader=val_loader,
                             device=device)
    if args.init_from:
        trainer.load_checkpoint(args.init_from)
        logger.info("Initialised from %s", args.init_from)

    trainer.fit_stage("F", n_epochs=cfg.survival_epochs)
    trainer.save_checkpoint("stage_F_complete.pt")
    trainer.fit_stage("H", n_epochs=5)
    trainer.save_checkpoint("stage_H_complete.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
