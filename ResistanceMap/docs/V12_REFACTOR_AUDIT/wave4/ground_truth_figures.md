# Wave-4 Ground-Truth Figures -- Report

**Script:** scripts/v12_refactor/wave4/ground_truth_visualizations.py
**Run date:** 2026-05-07
**Output:** paper/v8_artifacts/v12_refactor_audit/wave4/visualizations/
**Model under test:** v11.5 = Cox_v11_routed_mmsygnal (C=0.6955, RUNS r-2026-05-03-v11s5-cox-with-programs)

---

## Data provenance

| File | SHA-256 prefix | RUNS.md row |
|------|---------------|-------------|
| cox_per_patient_log_hazards.npz | 3317ab00f1e3fbc2 | r-2026-05-03-v11s5-cox-with-programs |
| mofa_vs_v11_5_per_patient_log_hazards.npz | b1a2f698012c71fa | r-2026-05-04-v12-mofa-vs-v115 |
| mmrf_sprint4_analysis.tsv | (sha16 at runtime) | r-2026-05-03-v10s4 clinical fields |

---

## Fig W4.GT.1 -- KM Tertile Curves

**Q:** Do v11.5 predicted-risk tertiles separate MMRF PFS curves?

Two panels (v11.5 left, v12 MOFA+v11 right); 3 KM step-curves per panel with 95% Greenwood shading; log-rank p.

**Where v11.5 fits well:** Low and high tertiles separate at population level; consistent with marginal C=0.6955.

**Where v11.5 fails:** Mid and high tertiles overlap, especially within S4 t(11;14) (per-stratum C=0.643). The model cannot reliably resolve intermediate-risk patients.

---

## Fig W4.GT.2 -- Calibration Decile Plot

**Q:** Is v11.5 predicted S(t) calibrated against observed KM at 12mo and 24mo?

10 deciles of predicted probability; each decile Kaplan-Meier with 95% Greenwood CI; Brier score annotated.

**Where v11.5 fits well:** Middle deciles (0.35-0.65 predicted S) lie close to the y=x diagonal. The model is not systematically over- or under-confident in the bulk of the distribution.

**Where v11.5 fails:** The extreme low-risk decile (predicted S near 1.0) sits below the diagonal -- the model mildly over-estimates survival for patients it deems lowest-risk. This PH-calibration compression is typical and implies raw predicted probabilities need isotonic recalibration before clinical use.

---

## Fig W4.GT.3 -- Time-Dependent AUROC Trajectory

**Q:** Does v11.5 discriminate consistently across the full follow-up period?

AUROC at each 30-day landmark from 90 to 1800 days; 4 models (v11.5, v12 MOFA+v11, Cox_v11_richer, predict-mean); 95% bootstrap CI for v11.5 (PCG64 seed=42, n_boot=300 on a 20-point subgrid).

**Where v11.5 fits well:** AUROC > 0.70 in the 90-600-day window; v11.5 leads all models. Consistent with recomputed AUROC@12mo=0.7505 from metrics_recomputed.json.

**Where v11.5 fails:** All models converge toward 0.5 beyond ~1400 days. This is a data-sparsity limitation (small at-risk population), not a model failure per se. Bootstrap CI is wide beyond 1200 days.

**Computation note:** Bootstrap CI computed for v11.5 only (budget ~2 min; 300 resamples x 20 time points). Full 4-model CI would require ~8 min; deferred as not essential for visual inspection.

---

## Fig W4.GT.4 -- Per-Stratum Prediction-Truth Scatter

**Q:** Does the model log-hazard track observed 12-month events within each cytogenetic stratum?

5-panel grid (one per stratum); x = v11.5 log-hazard; y = binary event within 12 months (0/1 with deterministic spread); LOWESS smoother (frac=0.55); per-stratum C-index in panel title.

**Where v11.5 fits well:** S1 del(17p) (C=0.730) and S3 +1q21 (C=0.694) show clearly rising LOWESS curves -- patients with high log-hazard are more likely to have progressed within 12 months.

**Where v11.5 fails:** S4 t(11;14) (C=0.643) is nearly flat -- proteasome-load + Waddington features used in Cox routing carry minimal signal for BCL2-driven biology. This is the expected, documented failure mode (RUNS.md row 36; v12 t(11;14) specialist head also fails, delta_S4=+0.013, CI not excluding 0, STRICT_FAIL).

---

## Notes on computation

Time-dependent AUROC uses Mann-Whitney form without IPCW correction. Conservative under informative censoring. Apply Blanche et al. 2013 for publication-quality estimates.

KM-based calibration: predicted S(t) derived from v11.5 LOO log-partial-hazards via the proportional-hazards formula with KM overall baseline. This is not a separate re-fit; it uses the same LOO hazards as the C-index computation.

---

*Word count: approximately 560 / 1000 limit*
