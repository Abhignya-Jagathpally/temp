# MORT-FM — Evaluation Plan

> What metrics will be reported, against what baselines, on what splits.
> Following the v8/v10 evaluation-governance pattern (Tiers A/B/C/D).

## Validation ladder

| Stage | Question                                  | Metrics                                       |
|------:|-------------------------------------------|-----------------------------------------------|
| 1     | Static drug response                      | MSE, MAE, Spearman, AUROC, calibration        |
| 2     | Multi-omic integration quality            | LISI, kBET, ARI, NMI, modality-mixing         |
| 3     | Trajectory prediction                     | Wasserstein-2, MMD, held-out timepoint NLL    |
| 4     | Time-to-resistance                        | C-index, integrated Brier, time-dep AUROC, KM separation |
| 5     | Pathway mechanism                         | Reactome enrichment precision, top-k recovery, CRISPR agreement |
| 6     | Counterfactual                            | Spearman vs CRISPR oracle, predicted ΔT calibration |

## Splits

* **Train / val / test**: patient-disjoint (enforced by
  `data_module.split_patients`).
* **Drug holdout**: hold a whole drug out at training time, evaluate on it.
* **Cell-line lineage holdout**: hold out a cancer subtype (e.g. AML when
  training on MM).
* **External validation**: train on MMRF, evaluate on Beat-AML and CoMMpass
  IA22 follow-up release.

## Baselines

Mandatory head-to-head on the same data + same split:

| Baseline       | Comparable axis                              |
|----------------|----------------------------------------------|
| Elastic net    | Static drug response (Stage 1)               |
| XGBoost        | Static drug response (Stage 1)               |
| MultiVI        | Multi-omic integration (Stage 2)             |
| scGLUE         | Multi-omic integration (Stage 2)             |
| totalVI        | Multi-omic integration (Stage 2, CITE-seq)   |
| PRESCIENT      | Trajectory (Stage 3)                         |
| scNODE         | Trajectory (Stage 3)                         |
| CellRank 2     | Trajectory + fate (Stage 3 + part of 4)      |
| DeepSurv       | Survival (Stage 4)                           |
| mmSYGNAL       | Survival on MM (Stage 4)                     |

Negative controls:
* CNN on gene-order input (expected to lose — included to show *why*
  spatial-local convolutions are biologically weak).
* LSTM on pseudo-time (expected to lose — destructive sc snapshots break
  the same-subject assumption).
* Shuffled-label MORT-FM (expected to lose — sanity check).

## Tiered governance

Inherits the v8/v10 pattern. Each stage advances only if the previous tier
passes.

* **Tier A** (hard stop): no fabricated numbers, no leakage. Tests in
  `tests/mortfm/test_patient_holdout_split.py`.
* **Tier B**: per-drug, per-modality, per-cohort breakdowns published.
* **Tier C**: full baseline comparison table.
* **Tier D**: external validation on a held-out cohort.

Each tier's pass criteria are written to `logs/mortfm/evaluation/`.

## What is **not** in this evaluation plan

* IC50 ranking — that is a v14 ResistanceMap deliverable, not MORT-FM's
  scientific contribution.
* Reproducing v11.5/v12 PFS C-index — MORT-FM is not a direct successor in
  that benchmark.
* Comparison to ResistanceMap v14 on the same task — different output
  spaces; no apples-to-apples comparison.
