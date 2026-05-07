**Fig W3.3 -- Calibration ladder: per-stratum 90% jackknife+ coverage (SKIPPED -- mondrian_jk_plus_v11.json absent).**

SKIPPED: mondrian_jk_plus_v11.json absent from tier1-refactor worktree (sha256 prefix 0e774c21adbd495b, run r-2026-05-03-v11s5-conformal). Existing fig6_conformal_calibration.{png,pdf} in paper/v8_artifacts/v12_refactor_audit/visualizations/ documents F_S5 STRICT PASS for Ridge models. Wave3 5-panel calibration-ladder form deferred to worktree with mondrian file present.

Known result (existing fig6_conformal_calibration.{png,pdf}, sha256 prefix 0e774c21adbd495b): Ridge_v10 and Ridge_v11features both achieve F_S5 STRICT PASS (all 5 strata within +-3% of nominal 0.90 jackknife+ coverage); GBM_v10 over-covers (0.94-0.97) and fails. F_S5 is single-cohort evidence (MMRF N=787); external replication needed for Nature Methods candidacy (CHAIR_VERDICT.md ss6 gap 2).
