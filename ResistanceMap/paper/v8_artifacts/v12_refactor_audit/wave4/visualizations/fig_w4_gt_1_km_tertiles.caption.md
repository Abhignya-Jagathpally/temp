**W4.GT.1 -- Kaplan-Meier curves by predicted-risk tertile (MMRF CoMMpass, N=787, n_events=224, TT2L endpoint).**

Source: `cox_per_patient_log_hazards.npz` SHA-256 prefix `3317ab00f1e3fbc2` (RUNS.md `r-2026-05-03-v11s5-cox-with-programs`) and `mofa_vs_v11_5_per_patient_log_hazards.npz` SHA-256 prefix `b1a2f698012c71fa` (RUNS.md `r-2026-05-04-v12-mofa-vs-v115`).

Left panel: v11.5 (Cox_v11_routed_mmsygnal, marginal C=0.6955) tertile stratification; right panel: v12 MOFA+v11 (C=0.6751).
Both panels show 95% Greenwood CI bands (shading) and log-rank p-value.

Substantial separation between low- and high-risk curves confirms population-level discriminability of v11.5.
Overlap of mid/high curves -- particularly in S4 t(11;14) (per-stratum C=0.643) -- is the primary failure mode.
