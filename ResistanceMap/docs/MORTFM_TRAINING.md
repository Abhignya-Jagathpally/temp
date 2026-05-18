# MORT-FM — Training Guide

## Entry points

| Task                        | Command                                                          |
|-----------------------------|------------------------------------------------------------------|
| Smoke test (CPU, ~3s)       | `python scripts/mortfm_smoke_test.py`                            |
| Run all MORT-FM unit tests  | `python -m pytest tests/mortfm/ -q --override-ini="addopts="`    |
| Cell-line pretrain          | TODO `scripts/mortfm_pretrain_foundation.py`                     |
| Patient finetune (Stage F)  | TODO `scripts/mortfm_train_survival.py`                          |
| Longitudinal trajectory     | TODO `scripts/mortfm_train_trajectory.py`                        |
| Ablation sweep              | TODO `scripts/mortfm_run_ablations.py`                           |

## Programmatic API

```python
import torch, yaml
from resistancemap.mortfm.schemas import MORTFMConfig
from resistancemap.mortfm.data_module import make_data_module
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.trainer import MORTFMTrainer
from resistancemap.data.trajectory_pair_builder import build_temporal_pairs
from resistancemap.data.clinical_outcome_loader import load_mortfm_outcomes
from resistancemap.data.scrna_loader import build_snapshots_from_h5ad

# 1. Load config.
with open("configs/mortfm.yaml") as f:
    cfg = MORTFMConfig(**{k: v for k, v in yaml.safe_load(f).items()
                          if k in MORTFMConfig.__dataclass_fields__})

# 2. Build data.
snapshots = build_snapshots_from_h5ad("data/raw/gse271107.h5ad")
outcomes = load_mortfm_outcomes("data/raw/mmrf_commpass", endpoint="pfs")
pairs = build_temporal_pairs(snapshots, outcomes, allowed_time_gaps=[3, 6, 12])
splits, train_loader, val_loader, _ = make_data_module(
    pairs, batch_size=cfg.batch_size, val_fraction=0.15, test_fraction=0.15, seed=cfg.seed,
)

# 3. Build model + trainer.
model = MORTFM(cfg, n_pathway_proteins=500, n_drug_candidates=11, n_resistance_states=4)
trainer = MORTFMTrainer(model, cfg, train_loader=train_loader, val_loader=val_loader,
                         device=torch.device("cuda"))

# 4. Run curriculum.
trainer.fit_all(stages=("A", "B", "C", "E", "F"))
trainer.save_checkpoint("final.pt")
```

## Training stages

| Stage | Purpose                                       | Required supervision     |
|------:|-----------------------------------------------|--------------------------|
| A     | Per-modality reconstruction                   | none                     |
| B     | Cross-modal masked-modality + contrastive     | none                     |
| C     | Static drug response                          | `drug_response`          |
| D     | Patient-domain adaptation                     | none (cross-batch)       |
| E     | Longitudinal trajectory                       | `future_state` (or `x_t_delta`) |
| F     | Survival / time-to-resistance                 | `event_time` + `event_observed` |
| G     | Pathway + counterfactual validation           | `pathway_targets`        |
| H     | Calibration                                   | `event_time` + `event_observed` |

A stage with missing supervision raises in `_losses_for_batch` rather than
quietly training on zero. This is intentional — the v12 audit identified
silent supervision substitution as a publication-killing failure mode.

## Hardware

| Config                | GPU mem | Wall-clock per epoch (Stage F) |
|-----------------------|---------|--------------------------------|
| `mortfm_debug.yaml`   | < 1 GB  | < 1s on CPU                    |
| `mortfm_cellline_pretrain.yaml` | ~18 GB (H100) | ~5 min on 1k CCLE cells |
| `mortfm_patient_finetune.yaml`  | ~12 GB | ~10 min on 1k MMRF patients   |
| `mortfm_trajectory.yaml`        | ~24 GB | ~30 min on 200 paired patients |

Numbers above are *projections* from the v14 pipeline budget. Actual numbers
will land in `RUNS.md` after first real training run.
