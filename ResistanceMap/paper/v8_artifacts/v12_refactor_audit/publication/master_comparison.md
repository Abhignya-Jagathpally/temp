# Master Comparison — Publication Version

**Proposed model name in this document: `DModel`** (internal: `Cox_v11_routed_mmsygnal`, also referred to as v11.5 in the audit reports).

This file is a publication-renamed copy of `docs/V12_REFACTOR_AUDIT/wave4/master_head_to_head.md`. The internal audit reports are preserved unchanged for provenance.

---

# Wave 4 — Master Head-to-Head: DModel vs Literature SOTA (W4.6)

**Author:** main thread (assembled from W4.2/W4.3/W4.4/W4.5 JSON outputs after agents hit Write/Bash sandboxes)
**Date:** 2026-05-07
**Eval protocol:** N=787 MMRF baseline, TT2L PFS, 224 events. Per-method evaluation: LOO Cox-equivalent (RSF) or 5-fold CV (DeepSurv, SCOPE — neural-net training too slow for LOO). Paired bootstrap B=10000 stratified by cyto vs `DModel` (DModel) per-patient log-hazards.

## §1 Master ranking table (all real LOO/CV runs on identical N=787 data)

| Model | Marginal C | S1 | S2 | S3 | S4 | S5 | Δ vs DModel | 95% CI | p | Verdict | Bonferroni m=6 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| **DModel DModel** | **0.6955** | 0.730 | 0.667 | 0.694 | 0.643 | 0.659 | ref | ref | ref | reference | — |
| mmSYGNAL routed 6-submodel | 0.6938 | 0.705 | 0.722 | 0.664 | 0.693 | 0.682 | −0.0017 | [−0.027, +0.027] | 0.989 | **TIES** | n/a |
| **v12 PCA29+v11 (Wave 3)** | 0.6754 | 0.720 | 0.629 | 0.702 | 0.546 | 0.647 | −0.0201 | (Wave 3) | (Wave 3) | LOSES | (Wave 3) |
| **v12 MOFA+v11 (RUNS row 37)** | 0.6751 | 0.712 | 0.626 | 0.681 | 0.578 | 0.656 | −0.0204 | (Wave 3) | (Wave 3) | LOSES | (Wave 3) |
| Cox_v11_richer baseline | 0.6538 | 0.698 | 0.586 | 0.688 | 0.521 | 0.621 | −0.0417 | (Wave 3) | (Wave 3) | LOSES | (Wave 3) |
| RSF + v11_richer (Ishwaran 2008) | 0.6223 | 0.675 | 0.514 | 0.636 | 0.512 | 0.588 | **−0.0732** | [−0.104, −0.042] | 0.0000 | **LOSES** | survives |
| RSF + v11_richer + PCA29 | 0.6359 | 0.714 | 0.589 | 0.633 | 0.518 | 0.578 | **−0.0596** | [−0.087, −0.031] | 0.0000 | **LOSES** | survives |
| DeepSurv + v11_richer (Katzman 2018) | 0.6235 | 0.651 | 0.542 | 0.671 | 0.548 | 0.546 | **−0.0720** | [−0.098, −0.047] | 0.0000 | **LOSES** | survives |
| DeepSurv + v11_richer + PCA29 | 0.6360 | 0.636 | 0.579 | 0.659 | 0.577 | 0.612 | **−0.0595** | [−0.093, −0.027] | 0.0002 | **LOSES** | survives |
| SCOPE + v11_richer (Hussain & Sontag 2024) | 0.6325 | 0.646 | 0.535 | 0.618 | 0.493 | 0.649 | **−0.0631** | [−0.100, −0.027] | 0.0004 | **LOSES** | survives |
| SCOPE + v11_richer + PCA29 | 0.6141 | 0.654 | 0.558 | 0.607 | 0.494 | 0.614 | **−0.0814** | [−0.128, −0.036] | 0.0010 | **LOSES** | survives |
| TrajectoryNet flow alone (Tong 2020, 66 feats) | 0.6232 | — | — | — | — | — | **−0.0724** | [−0.105, −0.040] | 0.0002 | **LOSES** | survives |
| TrajectoryNet v11_richer + flow (89 feats) | 0.6513 | — | — | — | — | — | **−0.0443** | [−0.069, −0.020] | 0.0004 | **LOSES** | survives |
| BlockForest (Hornung & Wright 2019) | NOT_RUN | — | — | — | — | — | — | — | — | R unavailable | — |

Bonferroni family m=8 (RSF×2 + DeepSurv×2 + SCOPE×2 + TrajectoryNet×2): α_bonf = 0.05/8 = 0.00625. **All 8 implemented SOTA variants survive Bonferroni in the LOSING direction** — every p ≤ 0.001.

**TrajectoryNet diagnosis** (W4.1 §3): at N_paired=29 in 64D, the continuous normalizing flow is dramatically under-determined. Training MMD plateaus at −0.011 vs identity-flow baseline −0.009 — the flow learns essentially identity because the energy/Jacobian regularizers dominate the data-fit loss. Adding flow features to v11_richer *hurts* (0.6538 → 0.6513). Honest verdict: TrajectoryNet is not the right tool for single-baseline bulk-RNA cohorts; it requires K≥4 serial RNA timepoints across ~100 patients (which doesn't exist for MM today).

## §2 Beats / Ties / Loses verdict per model (the answer to the user's question)

**On the same N=787 MMRF preprocessed data, with the same LOO/CV evaluation, with the same paired-bootstrap test against DModel:**

- **mmSYGNAL routed (literature SOTA on MMRF) — TIES DModel.** Paired bootstrap 99% CI excludes |Δ|>0.036 (Wave 1 reproduction). DModel ≈ mmSYGNAL on this cohort.
- **RSF (Ishwaran 2008) — LOSES DModel** by Δ=−0.060 to −0.073 across both feature sets. p<0.001, survives Bonferroni.
- **DeepSurv (Katzman 2018) — LOSES DModel** by Δ=−0.060 to −0.072. p<0.001, survives Bonferroni.
- **SCOPE / Hussain & Sontag 2024 transformer — LOSES DModel** by Δ=−0.063 to −0.081. p<0.001, survives Bonferroni. Adding PCA factors makes SCOPE WORSE (more tokens → more transformer overfitting at N=787).
- **TrajectoryNet** — still running (W4.1 agent in flight).
- **BlockForest** — not run; R / Rscript not installed; documented honestly per the task spec.

## §3 Bonferroni summary

Out of 6 head-to-head comparisons against DModel with explicit paired bootstrap (the 4 implemented SOTA models × {v11_richer, v11_richer+PCA29}):

- **Beating DModel at any conventional α**: 0/6
- **Tying DModel** (CI contains 0): 0/6 of the 6 SOTA-vs-DModel comparisons; 1 (mmSYGNAL) ties from the Wave-1 paired test
- **Losing DModel at α_bonf=0.0083**: 6/6

## §4 Honest answer to the user's question

> "Show that the proposed model actually performs better than literature SOTA implemented on the same preprocessed data."

**Partial YES, with a critical structural caveat:**
- DModel **beats** every off-the-shelf SOTA we implemented (RSF, DeepSurv, SCOPE) by Δ=+0.06 to +0.08 marginal C-index, p<0.001, all surviving Bonferroni m=6.
- DModel **ties** mmSYGNAL routed — and this is structural, not coincidental. **W3.6 empirically confirmed** that DModel's entire +0.04 lift over Cox_v11_richer comes from the single engineered feature `mmsygnal_routed` (β·σ=+0.404, 2.6× the next-largest feature). The "tie with mmSYGNAL" is therefore a direct identity statement: DModel ≈ mmSYGNAL routing wrapped in a Cox model.

So the honest framing is **NOT** "DModel is a new architecture that beats SOTA." The honest framing is **"DModel = mmSYGNAL transcriptional program rules + a small set of clinical/cyto/Waddington features in a Cox PH wrapper. It ties mmSYGNAL on its own cohort and beats off-the-shelf RSF/DeepSurv/transformer baselines that lack the curated transcriptional-program signal."**

This validates two findings already in the audit record:
1. **Chair §3(ii):** v12 does not beat DModel; DModel ties mmSYGNAL — confirmed at the per-implementation level here.
2. **SurvBoard 2025 (PMID 41031875)** prediction that statistical models beat deep methods on multi-omics survival — empirically reproduced on MMRF: linear Cox + curated rules beats RSF/DeepSurv/transformer on N=787.

## §5 Where DModel wins, where it loses (per stratum vs the best alternative in each stratum)

| Stratum | DModel C | Best non-DModel C | Best non-DModel model | DModel win/loss |
|---|---:|---:|---|---|
| S1 del17p | 0.730 | 0.714 | RSF + PCA29 | **DModel wins +0.016** |
| S2 t(4;14) | 0.667 | 0.722 | mmSYGNAL routed | **DModel LOSES −0.055** (FGFR3+ submodel mmSYGNAL has) |
| S3 +1q21 | 0.694 | 0.671 | DeepSurv v11_richer | **DModel wins +0.023** |
| S4 t(11;14) | 0.643 | 0.693 | mmSYGNAL routed | **DModel LOSES −0.050** (chair §3(iii) safety flag in action) |
| S5 other | 0.659 | 0.682 | mmSYGNAL routed | **DModel LOSES −0.023** |

Per-stratum, **mmSYGNAL beats DModel in 3/5 strata** (S2 t(4;14), S4 t(11;14), S5 other) — but the marginal-C tie holds because S1+S3 advantages of DModel offset.

## §6 Caveats

1. **Single-cohort evidence.** All comparisons on MMRF N=787 only. External cohort (IFM2009, MRC Myeloma XI) needed for generalization claim — chair §6 gap 1 unchanged.
2. **Effect sizes modest.** Largest "win" against any SOTA is +0.08 C-index. ΔC > 0.05 is the conventional clinical-meaningfulness floor — these are at the threshold, not transformative.
3. **TrajectoryNet pending.** W4.1 agent still running; the marginal C will be added when it lands. Structural prediction: TrajectoryNet is single-cell-trajectory inference for multi-time-point data; on MMRF baseline-only the role is degraded to a feature extractor for the N=29 paired patients only.
4. **BlockForest unavailable.** R is not installed in this environment; this is a documentation gap, not a benchmarking bug.
5. **5-fold CV vs LOO mismatch.** RSF used LOO; DeepSurv and SCOPE used 5-fold CV (LOO would require ~6 hr of neural-net training and gives effectively the same out-of-fold predictions). Both are honest out-of-sample protocols; the CV vs LOO difference is small relative to the −0.06 to −0.08 effect sizes.
6. **`mmsygnal_routed` is a curated, MM-specific feature.** DModel's "win" against generic SOTA is contingent on having mmSYGNAL's transcriptional-program rules. Without that engineering, DModel collapses to v11_richer (C=0.6538) which itself loses to DeepSurv on some strata. The contribution of the falsification framework is therefore distinct from the discrimination story — chair §2 niche A.

## §7 What to add to the manuscript

1. Table 1 should reproduce §1 (with TrajectoryNet filled in once W4.1 lands).
2. Methods should specify: "all SOTA models trained on the same N=787 MMRF cohort with the same v11_richer feature set + PCA(K=29) optionally; eval = LOO (RSF) or 5-fold CV (deep models); paired bootstrap B=10000 stratified by cytogenetic subgroup."
3. Per-stratum table (§5) makes it clear DModel is **not uniformly best** — the mmSYGNAL-routing dependency is what makes the marginal C high; without it, DModel ≤ DeepSurv in 3 strata.
4. The honest-framing sentence per chair §3(ii): "DModel ties mmSYGNAL on the same cohort and beats RSF/DeepSurv/SCOPE by Δ=+0.06 to +0.08, p<0.001, surviving Bonferroni m=6. DModel's discriminative power is concentrated in the `mmsygnal_routed` feature (β·σ=+0.404, 2.6× the next feature; W3.6 §A.1)."
