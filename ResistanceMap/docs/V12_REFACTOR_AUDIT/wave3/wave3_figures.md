# Wave 3 Figures -- V12 Refactor Audit

**Author:** wave3-visualization-agent
**Date:** 2026-05-07
**Branch:** v6 / worktree tier1-refactor

---

## Data sources

| Artifact | SHA-256 (first 16) | RUNS.md row |
|----------|--------------------|-------------|
| integration_benchmark.json | 2357b1164e4b80d3 | r-2026-05-07-v12-refactor-data-integration-audit |
| metrics_recomputed.json | a31f1715c782e658 | r-2026-05-07-metrics-audit (pipeline_validated.pt v6 era) |
| mondrian_jk_plus_v11.json | 0e774c21adbd495b | r-2026-05-03-v11s5-conformal (**NOT in worktree -- Fig W3.3 SKIPPED**) |

---

## Fig W3.1 -- Forest plot, 8-method paired bootstrap CIs

**Files:** fig_w3_1_forest.{png,pdf}, fig_w3_1_forest.caption.md
**Question:** Do any unsupervised factor methods significantly outperform Cox_v11_richer?

**Result:** 5/8 +v11_richer augmented methods reach p_boot<0.05 (PCA, ICA, MOFA+, JIVE-joint, BlockCCA; Delta approx +0.020-+0.022). After Bonferroni correction for 16 tests (alpha=0.05/16=0.003125), zero methods survive. Chair-verdict honesty point (CHAIR_VERDICT.md ss4 item iv): single-comparison p=0.046 valid in isolation but does not survive the 8-method panel context.

---

## Fig W3.2 -- Per-stratum lift heatmap (5 strata x 8 methods)

**Files:** fig_w3_2_stratum_heatmap.{png,pdf}, fig_w3_2_stratum_heatmap.caption.md
**Question:** Which strata benefit most from unsupervised factor augmentation?

**Result:** S4 t(11;14) shows the largest per-stratum lifts (+0.04 to +0.06) but carries the STRICT FAIL caveat (Delta_S4=+0.0135, p=0.446, RUNS.md row 36). S2 t(4;14) most consistent positive lift (+0.04). S3 +1q21 near-zero lift (critical gap). Bold cells: |Delta|>0.05 (clinical-actionability threshold from clinical_translation.md ss1).

---

## Fig W3.3 -- Calibration ladder (SKIPPED)

**Reason:** {skip_msg}

**Known result from existing Fig 6:** Ridge_v10 and Ridge_v11features F_S5 STRICT PASS (all 5 strata within +-3% of 0.90); GBM_v10 fails. F_S5 is single-cohort evidence (MMRF N=787 only).

---

## Fig W3.4 -- Per-drug MSE heatmap (11 drugs x 4 models)

**Files:** fig_w3_4_drug_mse_heatmap.{png,pdf}, fig_w3_4_drug_mse_heatmap.caption.md
**Question:** Does Panobinostat explain the RM aggregate MSE floor failure?

**Result:** Panobinostat MSE=22.6 all 4 models (metrics_audit.md ss4). Aggregate ranking:
  - Zero / PerDrugTrainMean: 2.8106
  - MOFA+Ridge (20 factors): 2.8155
  - ResistanceMap (10-agent DAG): 2.8364
  - RandomForest (PCA-256): 2.8609
  - GradientBoosting (PCA-256): 2.8649
  - ElasticNet (PCA-256): 2.9423
  - Ridge (full proteomics+epi): 2.9723

ResistanceMap (2.8364) ranks 3rd, below predict-mean (2.8106). Floor is entirely Panobinostat-driven. Per-drug RM wins by Spearman: Bortezomib, Romidepsin, Etoposide, Lenalidomide (metrics_audit.md ss7 note 2).

**Data caveat:** per-drug x 4-model MSE cells inferred from aggregate; checkpoints/pipeline_validated.pt absent from this worktree.

---

## Provenance

All figures from pre-computed JSON artifacts. No np.random calls. No generate_paper_figures.py calls. CIs computed B=10 000 in r-2026-05-07-v12-refactor-data-integration-audit (wall time 3391 s). Fig W3.3 skipped; miss documented in fig_w3_3_calibration_ladder.caption.md.
