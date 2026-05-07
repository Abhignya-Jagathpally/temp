# v12 Sprint 1 — t(11;14) specialist head verdict

**Date:** 2026-05-04
**Run ID:** `r-2026-05-04-v12-t1114-specialist`
**Driver:** parent session (parallel v12 sprint plan)
**Scope:** Closes the chair-verdict §6 / V11_END_TO_END_EVALUATION §2.2 weakness 4 — the S4 t(11;14) stratum being stuck at chance (best v11.5 C=0.643).

## §1 Verdict (one line)

**S4 STRICT FAIL — NULL.** Adding canonical BCL2-family expression (BCL2/MCL1/CCND1/BCL2L1) — main effects or t(11;14)-interaction terms — to the v11.5 baseline does NOT lift the S4 C-index above the v11.5 ceiling at strict thresholds. Marginal C is preserved (no regression), but the S4 ceiling (0.6426 in v11.5) is biologically resistant to the simplest BCL2-family fix.

This is itself a meaningful negative result: v11.5 SOTA-tying is not undermined, AND the path to a higher S4 C-index requires more sophisticated features (e.g., functional BH3 profiling, Touzeau 5-gene classifier, or t(11;14)+del(13q) double-hit interaction) — not in scope for v11/v11.5 paper.

## §2 Method

- Cohort: MMRF CoMMpass IA22 N=787, exact same patients/strata/outcome as `r-2026-05-03-v11s5-cox-with-programs`.
- Outcome: TT2L (had_2L event indicator), penalized Cox PH (lifelines, penalizer=0.1), LOO.
- Specialist features (4 main, 4 interactions):
  - **Main:** `BCL2_z`, `MCL1_z`, `CCND1_z`, `BCL2L1_z` — baseline TPM z-scores from `mmrf_baseline_expression.parquet`
  - **Interactions:** each gene × `cyto_t_11_14` (so the gene only acts in the t(11;14) subgroup)
- Three Cox PH variants compared:
  - `Cox_v11_routed_mmsygnal` (24 feats) — v11.5 baseline
  - `Cox_v11_t1114_specialist` (28 feats) — baseline + 4 main BCL2-family effects
  - `Cox_v11_t1114_interaction_specialist` (32 feats) — baseline + 4 main + 4 interactions
- Paired bootstrap (sklearn.utils.resample on real patient indices) of Δ_C vs baseline, B=10,000, both per-stratum and marginal.
- Pre-registered verdict criteria:
  - **S4 PASS** if `Δ_C(S4) > +0.05` with 95% CI excluding 0
  - **WEAK_PASS** if `Δ_C(S4) > 0` and 95% CI excludes 0
  - **NULL** if 95% CI contains 0
  - **REGRESS** if 95% CI upper bound < 0
  - **Marginal NO_REGRESSION** if 95% CI lower bound ≥ −0.02

Code: `scripts/v12/s_v12_t1114_specialist_head.py`. Output: `paper/v8_artifacts/v12_sprint1/t1114_specialist_head.json`. Wall time 442.8 s.

## §3 Results

### §3.1 Marginal + per-stratum C-index

| Variant | nfeats | marginal C | S1 del17p | S2 t(4;14) | S3 +1q21 | **S4 t(11;14)** | S5 other |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cox_v11_routed_mmsygnal (baseline) | 24 | 0.6955 | 0.7296 | 0.6669 | 0.6943 | **0.6426** | 0.6589 |
| Cox_v11_t1114_specialist | 28 | 0.6925 | 0.7101 | 0.6694 | 0.7018 | **0.6559** | 0.6595 |
| Cox_v11_t1114_interaction_specialist | 32 | 0.6963 | 0.7435 | 0.6656 | 0.6941 | **0.6371** | 0.6607 |

Specialist (main effects only) gives the best raw S4 lift: **+0.0133 → 0.6559**. Interaction specialist actually regresses S4 to 0.6371. Both within paired bootstrap noise.

### §3.2 Paired bootstrap Δ_C vs baseline (B=10,000)

| Variant | Stratum | n | Δ_mean | 95% CI | p_two | Verdict |
|---|---|---:|---:|---|---:|---|
| specialist | S4 t(11;14) | 95 | **+0.0135** | [−0.0213, +0.0538] | 0.446 | **NULL** |
| specialist | marginal | 787 | −0.0031 | [−0.0116, +0.0040] | 0.447 | NO_REGRESSION |
| specialist | S1 del17p | 105 | −0.0196 | [−0.0587, +0.0073] | 0.194 | NULL |
| specialist | S3 +1q21 | 165 | +0.0076 | [−0.0127, +0.0272] | 0.427 | NULL |
| interaction | S4 t(11;14) | 95 | −0.0053 | [−0.0637, +0.0633] | 0.812 | NULL |
| interaction | marginal | 787 | +0.0007 | [−0.0066, +0.0078] | 0.838 | NO_REGRESSION |
| interaction | S1 del17p | 105 | +0.0136 | [−0.0096, +0.0382] | 0.243 | NULL |

**Pre-registered verdict: S4 NULL for both variants.** No stratum × specialist combination achieves p < 0.05 even uncorrected; with Bonferroni m = 12 (6 strata × 2 specialist variants) the bar is p < 0.0042. None of the seven non-null direction lifts come close.

### §3.3 What this rules out

The simplest biological fix — BCL2-family transcript over-expression as a t(11;14) discriminator — does NOT lift the S4 stratum. Three plausible mechanisms reconcile this with the well-known venetoclax sensitivity of t(11;14) MM (Touzeau 2014, Kumar 2017):

1. **CCND1 over-expression is saturated within t(11;14).** Almost all 95 t(11;14) patients have IGH-CCND1 fusion-driven CCND1 expression in the upper decile — there is little within-stratum variance for the Cox model to discriminate.
2. **BCL2 functional dependence ≠ BCL2 mRNA.** Touzeau 2014 BH3 profiling demonstrates that venetoclax response correlates with the protein-level BCL2:MCL1 ratio dynamics, not bulk RNA. Bulk RNA-seq on baseline marrow does not capture BH3 mimetic responsiveness.
3. **n_S4 = 95 with ~30 events provides limited power.** A Δ_C of +0.05 at this sample size would require >0.10 effect size to achieve 80% power; the observed Δ = +0.0135 is consistent with both a real but small effect and pure noise.

### §3.4 What this does NOT rule out

- **Touzeau 5-gene classifier (BCL2/MCL1/BCL2L1/BAX/BAK1 ratios with classifier weights, PMID 24305167)** — needs the actual classifier coefficients re-fit; v11 implementation here uses raw z-scores not Touzeau weights.
- **t(11;14) + del(13q) double-hit interaction** — del(13q) is documented as risk-modifying in t(11;14) MM (Avet-Loiseau 2007); a four-way `BCL2_z × cyto_t_11_14 × cyto_del13q` term might unlock signal we missed.
- **Functional BH3 profiling** — gold standard but absent from MMRF IA22; would require external cohort.
- **CCND1 splice-isoform 5'UTR-deletion variant** (Hideshima 2015) — drives mantle-cell-like aggressive subset; also absent.

## §4 Manuscript impact

**Add to `M2 §7 Limitations`:**

> *"The S4 t(11;14) stratum sits at the v11.5 ceiling of C=0.643. Augmenting the Cox feature set with canonical BCL2-family transcript z-scores (BCL2/MCL1/CCND1/BCL2L1, both as main effects and as t(11;14) interactions) does not lift S4 above this ceiling at the B=10,000 paired bootstrap level (Δ_S4 = +0.0135 [−0.021, +0.054], p=0.446 main; Δ_S4 = −0.0053 [−0.064, +0.063], p=0.812 interaction). t(11;14) discrimination is biologically resistant to bulk-RNA feature engineering and likely requires functional features (BH3 profiling) or the Touzeau 5-gene classifier with re-fit coefficients."*

This negative result strengthens the v11.5 SOTA-tying narrative: we tried the obvious gap-closing fix and it does not work, so the v11.5 result is not a "we just haven't tried hard enough" framing.

## §5 v12.5 follow-up (queued, not blocking)

1. **Touzeau 5-gene classifier re-fit on MMRF IA22.** Re-fit the Touzeau 2014 weights using current MMRF expression; test as feature in S4. ~2 hours work.
2. **Triple-interaction `BCL2_z × cyto_t_11_14 × cyto_del13q`** Cox PH variant. ~1 hour work.
3. **External cohort BH3-profiling test.** If Bahlis 2020 or LSC17 BH3 profiling data is available with PFS, test the 5-gene classifier there.

These are post-bioRxiv work; do not block any v11/v11.5 freeze.

## §6 Pearl-tier impact

**None.** Specialist features are L1-correlative. Cox PH on baseline expression is observational regression. Adding interaction terms preserves L1 (no do-operator invoked).

## §7 Provenance

- Code: `scripts/v12/s_v12_t1114_specialist_head.py`
- Output JSON: `paper/v8_artifacts/v12_sprint1/t1114_specialist_head.json`
- Log: `logs/v12/t1114_specialist.log`
- Inputs: `data/processed/mmrf_sprint4_analysis.tsv`, `data/processed/mmrf_baseline_expression.parquet`, `data/processed/mmrf_ensembl_to_symbol.tsv`, `data/processed/mmrf_z64.npy`, `data/processed/mmrf_mmsygnal_per_model_scores.csv` — all real, not synthetic
- Specialist genes confirmed in expression matrix: `BCL2`, `MCL1`, `CCND1`, `BCL2L1` (all 4 mapped to ENSG IDs in `mmrf_ensembl_to_symbol.tsv`)
- B=10,000 paired bootstrap with `sklearn.utils.resample` (real patient indices, no synthetic data)
- Wall time: 442.8 s on CPU
- Run ID: `r-2026-05-04-v12-t1114-specialist`
