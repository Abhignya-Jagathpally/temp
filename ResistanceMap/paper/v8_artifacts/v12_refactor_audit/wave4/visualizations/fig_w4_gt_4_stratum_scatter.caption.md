**W4.GT.4 -- Per-cytogenetic-stratum scatter: v11.5 predicted log-hazard (x) vs observed 12-month event (0/1, y) (MMRF N=787).**

Source: `cox_per_patient_log_hazards.npz` SHA-256 prefix `3317ab00f1e3fbc2` (RUNS.md `r-2026-05-03-v11s5-cox-with-programs`).
Per-stratum C-index values from audited `metrics_recomputed.json` (wave4 audit 2026-05-07).

The LOWESS smoother (statsmodels frac=0.55, or local-constant fallback) shows whether log-hazard monotonically increases toward the event label within each stratum.

S1 del(17p) (C=0.730) and S3 +1q21 (C=0.694) have clearly positive slopes -- high log-hazard patients are more often early progressors.
S4 t(11;14) (C=0.643) is nearly flat -- mmSYGNAL/proteasome-load features used in Cox routing carry little signal for BCL2-driven biology.
This is the expected, documented failure mode (RUNS.md row 36; v12 t(11;14) specialist head also STRICT_FAIL).
