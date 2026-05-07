# V12 Wave-3 PCA Substitution

Run ID: r-2026-05-07-v12-wave3-pca-substitution  
Script: scripts/v12_refactor/wave3/s_v12_pca_vs_v11_5_pfs.py  
JSON: paper/v8_artifacts/v12_refactor_audit/wave3/pca_substitution.json  
Date: 2026-05-07


---

## S1 What changed

The original scripts/v12/s_v12_mofa_vs_v11_5_pfs.py (preserved unmodified) calls mofapy2.entry_point() at lines 217-224 with single view, single group, Gaussian likelihood, ARD-on-weights. The data_integration_validation.md S1 audit proved this reduces to Bayesian PPCA-with-ARD on the same input (no multi-view machinery exercised).

The new file replaces that block with a function get_factors():
- Default (RM_INCLUDE_MOFA_FACTORS unset): sklearn.PCA(n_components=29, random_state=0) on log1p(TPM) HVG-5000. Zero new dependencies, 0.038 s.
- Gate (RM_INCLUDE_MOFA_FACTORS=1): reload pre-computed Z_mofa from SPRINT1_V12 / mofa_vs_v11_5_per_patient_log_hazards.npz. No re-run of mofapy2. Provenance-preserving.

All other code (v11_richer reconstruction, LOO Cox PH penalizer=0.1, B=10000 paired bootstrap seed=20260504, V1/V2/V3 verdict gates) is byte-identical to the original and to data_integration_audit.py.

---

## S2 Run record

| Item | Value |
|------|-------|
| wall time | 471.6 s |
| exit code | 0 |
| N | 787 |
| events | 224 (28.5%) |
| expr_sha16 | 08fc7bf286413112 |
| PCA fit time | 0.038 s |
| factors_npy | paper/v8_artifacts/v12_refactor_audit/wave3/mmrf_pca_K29_factors.npy |
| JSON | paper/v8_artifacts/v12_refactor_audit/wave3/pca_substitution.json |

---

## S3 Side-by-side: mofapy2 original vs new PCA

| Metric | mofapy2 (r-2026-05-04) | PCA K=29 (this run) | diff |
|--------|------------------------|---------------------|------|
| Factor method | mofapy2 K_eff=29 single-view ARD | sklearn.PCA K=29 random_state=0 | -- |
| Factor extraction time | 77.0 s | 0.038 s | -76.96 s |
| C(factors alone) | 0.6600 | 0.6587 | -0.0013 |
| C(Cox_v11_richer) | 0.6538 | 0.6538 | 0.0000 |
| C(combo = factors + v11) | 0.6751 | 0.6754 | +0.0003 |
| delta combo vs richer | +0.0214 | +0.0217 | +0.0003 |
| 95% CI lower | +0.0005 | +0.0005 | 0.0000 |
| 95% CI upper | +0.0425 | +0.0433 | +0.0008 |
| p_boot two-sided | 0.0456 | 0.0450 | -0.0006 |
| verdict | V12_ARCHITECTURAL_OPPORTUNITY | V12_ARCHITECTURAL_OPPORTUNITY | identical |

Per-stratum C(combo):

| Stratum | n | MOFA+v11 | PCA+v11 |
|---------|---|----------|---------|
| S1 del17p | 105 | 0.7123 | 0.7197 |
| S2 t(4;14) | 97 | 0.6260 | 0.6285 |
| S3 1q21 | 165 | 0.6809 | 0.7024 |
| S4 t(11;14) | 95 | 0.5782 | 0.5461 |
| S5 other | 325 | 0.6559 | 0.6468 |

Note on S4: MOFA+v11 shows slightly higher S4 C-index (0.5782 vs 0.5461). The t(11;14) specialist gate has a pre-registered STRICT FAIL at Δ_S4=+0.0135 p=0.446 (RUNS.md row 36) — neither combo model has a validated t(11;14) claim.

---

## S4 Verdict: lift preserved

LIFT PRESERVED: YES

Tolerance checks (from pca_substitution.json lift_preservation block):

- |delta_new - delta_ref| = |0.0217 - 0.0214| = 0.0003 <= 0.001 PASS
- |p_new - p_ref|         = |0.0450 - 0.0456| = 0.0006 <= 0.010 PASS
- CI lower new = +0.0005 > 0, same sign as ref +0.0005 PASS

Verdict gate V2 (combo dominates, CI excludes 0, combo - factors_alone > 0.01) fires identically: V12_ARCHITECTURAL_OPPORTUNITY.

Conclusion: replacing mofapy2 with sklearn.PCA(K=29) preserves the v12 architectural finding within the stated tolerances. The statistical conclusion is identical, the dependency is eliminated, and the false multi-omics framing is removed (replacing it with the accurate claim: 29 unsupervised expression-space PCs concatenated with v11_richer gives delta=+0.0217, 95% CI [+0.0005, +0.0433], p=0.045).

Caveats:
1. p=0.045 is borderline; the 95% CI lower bound of +0.0005 means a single censored event flip could push p above 0.05. No claims of strong significance.
2. The lift is not PCA-specific or MOFA-specific: ICA+v11 (p=0.047), JIVE+v11 (p=0.034), and BlockCCA+v11 (p=0.035) all achieve equivalent lifts on the same N=787 cohort (data_integration_validation.md S3).
3. mofapy2 remains accessible via RM_INCLUDE_MOFA_FACTORS=1 for provenance cross-checks.

---

Absolute paths:
- Script:  /home/aj0486@students.ad.unt.edu/pipeline3/.claude/worktrees/tier1-refactor/ResistanceMap/scripts/v12_refactor/wave3/s_v12_pca_vs_v11_5_pfs.py
- JSON:    /home/aj0486@students.ad.unt.edu/pipeline3/.claude/worktrees/tier1-refactor/ResistanceMap/paper/v8_artifacts/v12_refactor_audit/wave3/pca_substitution.json
- Factors: /home/aj0486@students.ad.unt.edu/pipeline3/.claude/worktrees/tier1-refactor/ResistanceMap/paper/v8_artifacts/v12_refactor_audit/wave3/mmrf_pca_K29_factors.npy
- Original (read-only): /home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/scripts/v12/s_v12_mofa_vs_v11_5_pfs.py
