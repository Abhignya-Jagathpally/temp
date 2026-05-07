# v12.5 Sprint Plans (PRE-REGISTERED, NOT IMPLEMENTED)

Author: ResistanceMap research lead
Date drafted: 2026-05-07
Branch: v6 (worktree `tier1-refactor`)
Repo: `/home/aj0486@students.ad.unt.edu/pipeline3/.claude/worktrees/tier1-refactor/ResistanceMap`

Three independent (parallelizable) plans. Each respects the existing F-gate
framework (F1–F10, F_S5): no F-gate is deleted; new F-gates carry explicit
numerical thresholds. A plan is dropped if its F-gate cannot be specified a
priori — none are dropped here.

---

## Sprint A — scFoundation / ComBat-corrected encoder upgrade

### Pre-registered hypothesis
H_A: A foundation-model encoder (scFoundation xTrimoGene Performer-MAE OR
ComBat-corrected Geneformer V1-10M) on bulk-RNA MMRF will produce a 64-dim z
that (a) does NOT pass F2 where v10 PCA refutes (zero-trust leak gate),
(b) matches v10 on F1 per-stratum calibration, (c) matches v10 on F4 curl residual.

Any of (a), (b), (c) failing triggers REJECT and revert to v10 PCA(64), as
`r-2026-05-03-v11s1-encoder-reject` did for raw Geneformer.

### Data inputs and provenance
| Input | Provenance row |
| --- | --- |
| MMRF N=787 baseline TPM (57,690 ENSG) | reused from `r-2026-05-03-v11s1-encoder-reject` |
| Cohort batch labels (sequencing center, library prep, IA release) | needs new ingest from MMRF clinical metadata `STAND_ALONE_VISIT.csv` field `LIBRARY_PROTOCOL` and `RNA_QC.csv` field `SEQ_CENTER` — RUNS.md row to be created at ingest time |
| 19,489 ENSG ↔ Geneformer V1-10M token map | reused from `r-2026-05-03-v11s1-encoder-reject` |
| scFoundation Tencent weights (xTrimoGene Performer-decoder MAE) | needs new ingest IF/WHEN HF mirror published; otherwise fallback path |
| F2 paired N=27 LOO indices | reused from `r-2026-05-03-v10s2-hardened` |

### Step-by-step file-level deliverables
1. `scripts/v12_5/s_v12_5_combat_correction.py` — `pycombat-seq` (Zhang 2020) on log1p-CPM with batch=cohort, biol=cyto-stratum. Output `data/processed/mmrf_tpm_combat.npy`.
2. `scripts/v12_5/s_v12_5_encoder_path_a_scfoundation.py` — Path A (preferred): xTrimoGene Performer-decoder forward on ComBat TPM, attention-pooled 64-dim → `mmrf_z64_v12_5_scf.npy`. Skipped if HF weights unavailable.
3. `scripts/v12_5/s_v12_5_encoder_path_b_geneformer_combat.py` — Path B (always runs as control): Geneformer V1-10M on ComBat ranks → mean-pool 256 → PCA(64) → `mmrf_z64_v12_5_gfc.npy`.
4. `scripts/v12_5/s_v12_5_f2_zero_trust.py` — F2 paired-LOO bootstrap B=10000 on new z64s → `paper/v8_artifacts/v12_5_sprint_a/f2_zero_trust.json`.
5. `scripts/v12_5/s_v12_5_f1_f4_under_new_encoder.py` — Re-fit U_θ DSM + HBayes 5-stratum SVI; recompute F1 z-scores and F4 curl L2 → `f1_f4_new_encoder.json`.
6. `scripts/v12_5/promote_or_reject.py` — deterministic decision rule → `VERDICT.json`.
7. `docs/V12_5_SPRINT_A_VERDICT.md` + RUNS.md row append.

### F-gates (pre-registered numerical thresholds)
- **F2 zero-trust (REJECT criterion)**: v10 PCA F2 REFUTES (Δ=24.19, 95% CI [−43, 99]). New encoder REJECTED if its F2 PASSES (95% CI excludes 0 on same N=27 LOO).
- **F1 non-regression**: marginal centroid-recovery z̄_v12_5 ≥ z̄_v10 − 0.10; no stratum degrades by > 0.20.
- **F4 non-regression**: U_θ curl L2 ≤ 0.0151 (Sprint 2 hardened baseline) × 1.10.
- **F_S5 safeguard**: Mondrian marginal coverage in [0.88, 0.92] AND per-stratum within ±5% of 0.90.
- Composite: PROMOTE iff all four pass; REJECT otherwise.

### Negative-result protocol
- F2 leak fires → cohort-batch contamination (same mode as v11 raw Geneformer). Bulk-RNA mean-pool on single-cell-pretrained model collapses onto run-level effects despite ComBat. Revert to v10 PCA(64); document in Limitations.
- F1 fails (passes F2 but degrades calibration) → encoder over-smooths biological variance. File as diagnostic-only (disposition matching `mmrf_z64_v11.npy`).
- F4 fails → encoder is not flow-compatible. Reject, document.
- All fail → publishable negative result: foundation encoders are uninformative for bulk MMRF at N=787; bulk-RNA needs cyto-aware contrastive pretraining (v13).

### Compute estimate
- ComBat-seq: ~3 min CPU (16 cores).
- Path A scFoundation forward (1× H100): est. 0.8 GPU-hours for full N=787, batch=8, fp16 (xTrimoGene 100M params, 19k-token context).
- Path B Geneformer V1-10M forward (1× H100): 14 s GPU (replicates `r-2026-05-03-v11s1-encoder-reject`).
- F2 + F1 + F4 re-runs: ~10 min CPU.
- **Total: ≤ 1.0 GPU-hour on H100 + ~20 min CPU.**

### Wall-time
~3 hours including data wrangling, 1 day if scFoundation weights are not yet HF-native and Path B is the only executable path.

### Risk and detection
- **R1**: scFoundation weights unpublished. *Detect*: Path A raises `WeightsNotAvailable`; Path B runs unconditionally.
- **R2**: ComBat-seq destroys cyto-stratum signal. *Detect*: F1 per-stratum z collapses to ≈ 0 on v10 PCA replay; preserve `biol=cyto-stratum`; F1 baseline replay is a pre-flight sanity check.
- **R3**: F2 leak gate too permissive. *Detect*: scale-controlled diagnostic (per-dim z-standardization, as `f1_f2_gates_v11_standardized.json`) must ALSO refute; mismatched outcomes → bias to REJECT.
- **R4**: H100 budget over-run. *Detect*: per-batch wall-time logging; abort Path A if > 4 GPU-hours and revert to Path B.

---

## Sprint B — NUTS HMC re-inference for HBayes

### Pre-registered hypothesis
H_B: Replacing the AutoNormal mean-field SVI guide with NUTS HMC and adding a
per-stratum heteroscedastic σ_obs_s (5 free parameters instead of 1) reduces
Gelman-1996 §6 PPC failures from 5/10 (current) to ≤ 2/10 while preserving
v10 Sprint 2 F1, F2, and F5-paired claims.

Falsifiable: if PPC fails > 2/10 OR any of {F1, F2, F5-paired} flips verdict
relative to v10 Sprint 2, the upgrade is rejected.

### Data inputs and provenance
| Input | Provenance row |
| --- | --- |
| MMRF N=716 cyto-aligned z64 | reused from `r-2026-05-03-v10s2` (5-stratum partition) |
| 5-stratum cyto labels (S1=del17p, S2=t(4;14), S3=+1q21, S4=t(11;14), S5=other) | reused from `r-2026-05-03-v10s5` |
| Paired N=27 LOO indices for F2 | reused from `r-2026-05-03-v10s2-hardened` |
| Sprint 2 priors (m_0, κ_0, ν_0) | reused from `r-2026-05-03-v10s2` |

No new ingest required.

### Step-by-step file-level deliverables
1. `scripts/v12_5/s_v12_5_hbayes_nuts.py` — numpyro NUTS: 4 chains, 1500 warmup, 2000 samples, target_accept=0.90, max_tree_depth=12, dense_mass=True on population block. Model change: scalar `σ_obs ~ HalfNormal(1.0)` → `σ_obs_s[5] ~ HalfNormal(1.0)` indexed by cyto-stratum. Output `posterior_nuts.nc` with R̂ + ESS.
2. `scripts/v12_5/s_v12_5_diagnostics_nuts.py` — R̂, bulk-ESS, divergences, energy-BFMI → `nuts_diagnostics.json`.
3. `scripts/v12_5/s_v12_5_ppc_under_nuts.py` — Gelman §6 PPC with T1 (centroid magnitude) + T2 (dispersion) on 1000 NUTS draws → `hbayes_ppc_nuts.json`.
4. `scripts/v12_5/s_v12_5_replay_v10s2_under_nuts.py` — replay F1, F2, F5-paired under NUTS posterior → `v10s2_replay.json`.
5. `docs/V12_5_SPRINT_B_VERDICT.md` + RUNS.md row append.

### F-gates (pre-registered numerical thresholds)
- **F1_PPC (new gate, formalizes v12 caveat)**: ≤ 2 of 10 (T × stratum) PPC combos with p_B ∉ [0.05, 0.95]. Currently 5/10.
- **NUTS sampling validity**: R̂ ≤ 1.01 (all population params) AND bulk-ESS ≥ 400 AND divergences ≤ 1% post-warmup AND energy-BFMI ≥ 0.3.
- **F1 non-regression**: NUTS marginal centroid z ≥ v10 SVI z − 0.10.
- **F2 non-regression**: paired-LOO 95% CI MUST still contain 0 (REFUTE preserved).
- **F5-paired non-regression**: refute status preserved (CI contains 0 or wrong-signed) — matches `r-2026-05-03-v10s2-hardened`.
- Composite: PROMOTE iff F1_PPC AND sampling validity AND all three replays pass.

### Negative-result protocol
- PPC stays > 2/10 after NUTS + heteroscedastic σ → mean-field shrinkage was not the dominant pathology; model is misspecified (Student-t outer / mixture inner needed). Biological reading: cyto-stratum centroid distributions in z64 are non-Gaussian — HBayes class wrong, not the inference. Queue v13 model-class search; do NOT promote NUTS.
- Sampling fails (R̂ > 1.01, divergences > 1%, low ESS) → reparameterize (non-centered centroid block). If still fails, posterior geometry is pathological — HBayes needs structural redesign.
- F2 flips PASS under NUTS → SVI was masking a leak. Roll back v10 Sprint 2 claims; major rewrite (symmetric to Sprint A's leak-gate).
- F1_PPC passes but F_S5 coverage exits [0.88, 0.92] → re-fit Mondrian conformal under new posterior (added work, not refutation).

### Compute estimate
- NUTS 4 chains × 3500 iter × ~5000 patient-stratum likelihood evals/iter on z64 with diagonal-mass adaptation. Empirically (numpyro on similar 5-stratum hierarchical models): ~30 min CPU (16 cores) per chain, parallelized → ~30 min wall.
- PPC: ~10 s CPU (1000 draws).
- v10s2 replay: ~3 min CPU.
- **Total: ~45 min CPU, no GPU required.**

### Wall-time
~1 hour including diagnostics review.

### Risk and detection
- **R1**: NUTS does not converge on S4 centroids (n=95). *Detect*: per-parameter R̂ table; mitigate via non-centered parameterization, tighter σ_obs_s[S4] prior.
- **R2**: σ_obs_s[5] absorbs real S4 signal as noise (T2 p_B 0.966 → ~0.5). *Detect*: NUTS centroid recovery z-score also drops; fix via weakly-informative HalfNormal(0.5) instead of 1.0.
- **R3**: F2 flip (refute → pass) is publishable correction, not sprint failure. *Detect*: explicit replay output.
- **R4**: numpyro/JAX version drift breaks reproducibility. *Detect*: lock numpyro==0.13.* in `env.lock`.

---

## Sprint C — Touzeau 2014 5-gene t(11;14) classifier

### Pre-registered hypothesis
H_C: A bulk-RNA classifier built on the Touzeau et al. 2014 (PMID 24305167)
5-gene venetoclax-sensitivity panel, when added as a feature to Cox_v11_richer,
closes the S4 t(11;14) chance-level gap (current S4 C-index = 0.521 in
v11_richer; mmSYGNAL FGFR3-routed reaches 0.726 elsewhere but not in S4).

Specifically: Δ_S4 (Cox_v11_richer + Touzeau5 vs Cox_v11_richer) > +0.05 with
95% CI excluding 0, B=10000 paired bootstrap, marginal-non-regression preserved.

This is the SAME pre-registered threshold as `r-2026-05-04-v12-t1114-specialist`
(which failed the simpler 4-gene BCL2 family panel: Δ_S4=+0.0135, p=0.446).

### Data inputs and provenance
| Input | Provenance row |
| --- | --- |
| MMRF N=787 baseline TPM | reused from `r-2026-05-03-v10s1` |
| Cox_v11_richer 23-feat baseline log-hazards | reused from `r-2026-05-03-v11s5-cox` |
| 5-stratum cyto labels (S4 = 95 t(11;14)) | reused from `r-2026-05-03-v10s5` |
| Touzeau 2014 5-gene panel symbols | needs new ingest — extract from PMID 24305167 supplement Tables S1–S2 |
| (Optional) GSE6477/GSE9782 BCL2-family / BH3 external validation arrays | needs new ingest IF time permits |

### Step-by-step file-level deliverables
1. `scripts/v12_5/s_v12_5_touzeau_panel_extract.py` — parse PMID 24305167 supplement Tables S1–S2 → `data/processed/touzeau_2014_5gene.csv` {symbol, ensembl_id, direction, source}. If supplement is unreachable: operational literature-reconstruction fallback {BCL2, BCL2L11/BIM, BAX, MCL1, BCL2L1/Bcl-xL} marked "literature-reconstructed"; verdict carries an asterisk.
2. `scripts/v12_5/s_v12_5_touzeau_features.py` — 5 z-scored TPMs + scalar Touzeau_score = Σ(direction × z) + Touzeau_score × cyto_t_11_14 → `mmrf_touzeau_features.npy` (787, 7).
3. `scripts/v12_5/s_v12_5_t1114_specialist_v2.py` — three LOO Cox variants (lifelines penalizer=0.1): `Cox_v11_richer` (baseline replay), `+touzeau5`, `+touzeau_score_int`. Per-stratum + marginal C-index, B=10000 paired bootstrap Δ vs baseline → `t1114_specialist_v2.json`.
4. `scripts/v12_5/s_v12_5_panel_sanity.py` — pre-Cox biological sanity: ≥ 3 of 5 genes show |Cohen-d| > 0.5 between t(11;14) and rest → `panel_sanity.json`.
5. `docs/V12_5_SPRINT_C_VERDICT.md` + RUNS.md row append.

### F-gates (pre-registered numerical thresholds)
- **F_S4_PASS** (primary, identical to v12 specialist gate): best variant Δ_S4_C > +0.05 with 95% paired-bootstrap CI lower bound > 0.
- **F_S4_NO_REGRESSION_MARGINAL**: best-variant marginal C ≥ Cox_v11_richer − 0.005.
- **F_S4_PANEL_PRESENT** (pre-Cox): ≥ 3 of 5 Touzeau genes |Cohen-d| > 0.5 between t(11;14) and rest. If < 3, refute before Cox.
- **Family-wise correction**: 2 variants → α_per_variant = 0.025 for bootstrap CI.
- Composite: PROMOTE iff all three gates pass.

### Negative-result protocol
- F_S4_PANEL_PRESENT fails: bulk-RNA TPM does not recapitulate Touzeau's primary-plasma-cell BH3-profiling signal. This refutes bulk-RNA transferability, NOT Touzeau itself. Queue v13 BH3-functional-profiling ingest.
- F_S4_PASS fails with PANEL_PRESENT passing: genes differentially expressed but no PFS signal — Touzeau panel is correct for venetoclax response, wrong outcome (TT2L is not ven-response). Keep v11.5 for S4; add Limitations sentence.
- Both F_S4_PASS and MARGINAL fail (S4 lifts but marginal tanks): panel re-balances information across strata. Reject; queue stratum-specific ensembling for v13.
- Literature-fallback path used AND F_S4_PASS triggers: result carries asterisk, conditional promotion until supplement-verified panel re-runs.

### Compute estimate
- Panel extraction (PDF/Excel parse): ~30 min wall, mostly manual / interactive.
- Cox PH LOO × 3 variants × N=787: ~7 min CPU each → ~25 min CPU.
- B=10000 paired bootstrap × 2 comparisons: ~5 min CPU.
- **Total: ~40 min CPU, no GPU required.**

### Wall-time
~1.5 hours including supplement parsing.

### Risk and detection
- **R1**: Supplement paywalled / gene list omitted → fallback dilutes the test. *Detect*: asterisked verdict if fallback invoked; institutional access tried first.
- **R2**: HGNC symbol drift breaks ENSG mapping. *Detect*: every symbol resolves in MMRF id table pre-Cox; pre-register adjusted gate "≥ 3 of 5 mappable AND ≥ 60% of mappable show |d|>0.5".
- **R3**: S4 n=95 underpowers Δ_C=+0.05 at α=0.025 (req'd N ≈ 200 for 80% power). *Detect*: panel_sanity reports CI half-width; if > 0.08, fail does NOT refute Touzeau, refutes our ability to test it. Pre-register this in verdict template.
- **R4**: Multiple-testing inflation by silent variant addition. *Detect*: variant list frozen in deliverable 3; any addition requires α-recomputation in writing.

---

## Cross-sprint notes
- A, B, C independent; can run in parallel.
- Each emits `VERDICT.json` + `docs/V12_5_SPRINT_{A,B,C}_VERDICT.md` + RUNS.md row.
- No F-gate deleted. A reuses F2 zero-trust; B formalizes v12 PPC caveat into F1_PPC ≤ 2/10; C reuses Δ_S4 > +0.05.
- Pearl-tier ceiling stays L1-with-structural-prior under all three.
- Compute total: ≤ 1.0 GPU-hour H100 + ~2 CPU-hours.
