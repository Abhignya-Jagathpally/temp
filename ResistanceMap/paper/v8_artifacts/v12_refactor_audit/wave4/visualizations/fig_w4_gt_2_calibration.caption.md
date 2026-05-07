**W4.GT.2 -- Calibration decile plot: v11.5 predicted PFS at 12 and 24 months vs observed Kaplan-Meier (MMRF N=787).**

Source: `cox_per_patient_log_hazards.npz` SHA-256 prefix `3317ab00f1e3fbc2` (RUNS.md `r-2026-05-03-v11s5-cox-with-programs`).

Predicted S(t) from v11.5 LOO log-partial-hazards via the PH baseline-hazard formula; 10 deciles; each decile Kaplan-Meier with 95% Greenwood CI; Brier score.

Middle deciles (0.35-0.65 predicted S) lie near the y=x diagonal -- model is well-calibrated in this range.
The extreme low-risk decile sits below the diagonal -- the model over-estimates survival for predicted low-risk patients (PH calibration compression, a known artefact; isotonic recalibration recommended before clinical use).
