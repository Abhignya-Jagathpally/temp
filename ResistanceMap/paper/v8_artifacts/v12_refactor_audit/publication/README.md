# Publication Artifacts — DModel head-to-head

This directory contains publication-renamed copies of the Wave-3 / Wave-4 audit
artifacts. The internal name `Cox_v11_routed_mmsygnal` (also referred to as
`v11.5` in the audit reports) has been replaced with **`DModel`**
throughout these files.

To use a different publication name, edit `PROPOSED_MODEL_NAME` at the top of
`scripts/v12_refactor/publication/build_publication_artifacts.py` and re-run.

## Files

| File | Source | Contents |
|---|---|---|
| `master_comparison.csv` | `wave4/master_head_to_head.csv` | 13-model ranking with paired-bootstrap CIs |
| `master_comparison.md` | `wave4/master_head_to_head.md` | Narrative head-to-head analysis |
| `visualizations/fig_pub_1_forest_DModel_vs_sota.{png,pdf}` | newly rendered | Forest plot of paired Δ C-index for 9 comparators vs `DModel` |
| `visualizations/fig_pub_2_marginal_c_ranking.{png,pdf}` | newly rendered | Sorted bar chart of marginal C across all 13 models, `DModel` highlighted |
| `visualizations/fig_pub_3_per_stratum_heatmap.{png,pdf}` | newly rendered | Per-stratum C-index heatmap across 11 models × 5 cyto strata |
| `visualizations/fig_pub_4_km_tertiles_DModel.{png,pdf}` | newly rendered | Kaplan-Meier curves stratified by `DModel` predicted-risk tertiles |

## Honest framing carried forward (chair §7)

- `DModel` reaches marginal C-index 0.6955 on N=787 MMRF baseline.
- `DModel` ties mmSYGNAL routed (Δ=−0.0017, 95% CI [−0.027, +0.027], p=0.989) on the same cohort. Structural tie because `DModel` IS mmSYGNAL routing wrapped in Cox.
- `DModel` beats RSF (Ishwaran 2008), DeepSurv (Katzman 2018), the SCOPE transformer adapted from Hussain & Sontag 2024 (PMID 39075240), and TrajectoryNet (Tong 2020) by ΔC = +0.06 to +0.08, paired bootstrap p<0.001, all surviving Bonferroni m=8 (α=0.00625).
- `DModel`'s discriminative power is concentrated in ONE engineered feature (`mmsygnal_routed`, β·σ = +0.404, 95% CI [+0.265, +0.542], 2.6× the next feature). Stripped of this feature, `DModel` collapses to v11_richer (C=0.6538), which loses to DeepSurv on some strata.
- Per-stratum: `DModel` wins S1 + S3 vs all alternatives, but mmSYGNAL routed beats `DModel` in S2 t(4;14), S4 t(11;14), and S5 "other" by 2.3 to 5.5 percentage points.
- The contribution is curated-feature engineering + a falsification framework (10 pre-registered F-gates with published failures), NOT a new neural architecture.

## DO NOT say in publications

- `DModel` is "state-of-the-art" — too strong; defensible: `DModel` beats off-the-shelf RSF/DeepSurv/transformer baselines on MMRF.
- "We propose a new architecture" — `DModel` ≈ mmSYGNAL+Cox.
- "Hussain/Sontag is beaten" — we transfer-tested their architecture; we did NOT reproduce TOURMALINE.
- "TrajectoryNet was beaten" — it was structurally mis-applied; the negative result is about cohort fit.
- `DModel` is "uniformly best" — false; mmSYGNAL beats `DModel` in 3/5 strata.
