**W4.GT.3 -- Time-dependent AUROC on MMRF TT2L PFS (N=787, n_events=224, 30-day landmark grid).**

Source: `cox_per_patient_log_hazards.npz` SHA `3317ab00f1e3fbc2` (RUNS `r-2026-05-03-v11s5-cox-with-programs`) and `mofa_vs_v11_5_per_patient_log_hazards.npz` SHA `b1a2f698012c71fa` (RUNS `r-2026-05-04-v12-mofa-vs-v115`).

AUROC at each landmark tq: cases = patients who experienced the event before tq; controls = patients who survived past tq.
Mann-Whitney form without IPCW correction (conservative under informative censoring).

v11.5 (red) dominates all models in the 90-600-day window (AUROC>0.70); consistent with AUROC@12mo=0.7505 from `metrics_recomputed.json`.
The 95% bootstrap CI band (PCG64 seed=42, n_boot=300 on a 20-point subgrid) shows where v11.5 is statistically above chance.
Convergence toward 0.5 beyond ~1400 days reflects data sparsity at the tail, not model degradation.
