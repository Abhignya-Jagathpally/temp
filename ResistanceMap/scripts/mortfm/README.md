# `scripts/mortfm/` — canonical MORT-FM workflow (v18)

v18 consolidates the MORT-FM training + evaluation workflow into ten numbered
entry points. Each numbered script is a **thin shim** around the existing
versioned script in `scripts/`. The shim's job is to provide a stable,
discoverable, ordered surface; the heavy lifting still lives in the underlying
script (and stays referenceable by `RUNS.md`).

| # | Stage | Shim | Underlying script |
|---|-------|------|--------------------|
| 00 | Data acquisition | `00_download_public_data.py` | `scripts/mortfm_download_public_data.py` |
| 01 | Identifier harmonization | `01_harmonize_identifiers.py` | `scripts/mortfm_harmonize_identifiers.py` |
| 02 | Longitudinal dataset assembly | `02_build_longitudinal_dataset.py` | `scripts/mortfm_build_longitudinal_dataset.py` |
| 03 | MMRF preparation | `03_prepare_mmrf.py` | `scripts/mortfm_prepare_mmrf.py` |
| 04 | Foundation pretraining | `04_pretrain_foundation.py` | `scripts/mortfm_pretrain_foundation.py` |
| 05 | Block-C contrastive | `05_train_block_c_contrastive.py` | `scripts/mortfm_train_block_c_contrastive.py` |
| 06 | LENS resistance training | `06_train_lens_resistance.py` | `scripts/mortfm_train_lens_resistance.py` |
| 07 | Survival head training | `07_train_survival.py` | `scripts/mortfm_train_survival.py` |
| 08 | Causal evidence v2 | `08_eval_causal_evidence_v2.py` | `scripts/mortfm_causal_evidence_report_v2.py` |
| 09 | Gate revalidation | `09_gate_revalidate.py` | `scripts/mortfm_v17_gate_revalidate.py` |

Other scripts (e.g. ingestors per data source, BeatAML alignment helpers,
visualization utilities) remain under `scripts/mortfm_*.py` and are invoked
by one or more numbered shims as needed. They are not part of the canonical
01–09 surface but are not deprecated either.

Superseded scripts have been moved to `scripts/archive/`. They are kept for
historical reproduction of RUNS.md rows that reference them by path, but
new work should not invoke them.

## Smoke test

The canonical "everything wired" smoke is `scripts/mortfm_v17_unified_smoke.py`
(kept under its original name because v17.x RUNS.md rows reference it).

## Airflow Task Mapping

| Airflow Task | Script | Artifact | Depends On |
|---|---|---|---|
| acquire_public_data | 00_download_public_data.py | data/raw_public/* | — |
| harmonize_identifiers | 01_harmonize_identifiers.py | data/processed/metadata/gene_protein_identifier_map.csv | acquire_public_data |
| build_longitudinal_dataset | 02_build_longitudinal_dataset.py | data/processed/longitudinal/temporal_pairs.parquet | harmonize_identifiers |
| prepare_mmrf | 03_prepare_mmrf.py | data/processed/mortfm/outcomes.pkl | build_longitudinal_dataset |
| audit_leakage | mortfm_longitudinal_pair_audit.py | results/mortfm/leakage_audit.json | prepare_mmrf |
| pretrain_foundation | 04_pretrain_foundation.py | checkpoints/mortfm/stage_B_complete.pt | audit_leakage |
| train_state_encoder | 05_train_block_c_contrastive.py | checkpoints/mortfm/block_c_state_encoder_v17.pt | pretrain_foundation |
| train_lens_resistance | 06_train_lens_resistance.py | logs/mortfm/lens_resistance_summary.json | train_state_encoder |
| train_survival | 07_train_survival.py | checkpoints/mortfm/survival_trained.pt | train_lens_resistance |
| run_baselines | 10_run_baselines.py | results/baselines/baseline_leaderboard.csv | audit_leakage |
| evaluate_causal_evidence | 08_eval_causal_evidence_v2.py | logs/mortfm/causal_evidence_v2_summary.json | train_lens_resistance |
| gate_revalidate | 09_gate_revalidate.py | logs/mortfm/v17_gate_revalidate.json | train_survival, run_baselines, evaluate_causal |
| generate_figures | sota_benchmark_and_visualizations.py | paper/v8_artifacts/sota_benchmark/* | gate_revalidate |
| export_bundle | export_publication_bundle.py | paper/publication_bundle/* | generate_figures |

## Architecture: Airflow-Scripts-Modules

The execution architecture follows a four-layer delegation pattern:

```
Airflow DAG → numbered scripts (thin shims) → existing modules → internal orchestrator
```

**Airflow DAG** defines the task graph (dependencies, retries, sensor gates).
Each task maps to exactly one **numbered script** in `scripts/mortfm/`. These
scripts are thin shims: they parse CLI args, set up logging, and delegate to
the **existing modules** under `ResistanceMap/` (the versioned library code
that stays referenceable by `RUNS.md`). Those modules in turn may call the
**internal orchestrator** (`MORTFMOrchestrator`) for multi-step workflows that
coordinate across model blocks.

This layering keeps Airflow concerns (scheduling, retries, XCom) out of
research code, and keeps research code importable and testable without an
Airflow installation.
