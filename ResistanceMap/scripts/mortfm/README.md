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
