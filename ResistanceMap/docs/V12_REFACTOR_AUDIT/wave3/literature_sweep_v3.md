# Wave 3 — Literature Sweep v3 (W3.8)

**Author:** literature-deep-researcher (live PubMed/bioRxiv/HF MCP queries; persisted by main thread)
**Date:** 2026-05-07
**Status:** Open Targets MCP not in active server list — deferred to follow-up.

## §1 Top-10 ranked candidates

| # | DOI / arXiv (PMID) | Title (lead author) | Year | Task | Dataset / N | Best metric | Why compare |
|---|---|---|---|---|---|---|---|
| 1 | [10.1038/s41746-024-01189-3](https://doi.org/10.1038/s41746-024-01189-3) (PMID 39075240) | Joint AI-driven event prediction NDMM/RRMM (Hussain, Sontag) | 2024 | MM PFS+OS+AE+biomarker joint forecast | TOURMALINE NDMM N=703 + RRMM N=720 external | beats ISS p<0.001 Bonferroni; beats SOTA forecasters | **Direct MM-PFS competitor at trial scale; transformer baseline** |
| 2 | [10.1093/bib/bbaf521](https://doi.org/10.1093/bib/bbaf521) (PMID 41031875) | SurvBoard (Wissel) | 2025 | Multi-omics survival benchmarking | Pan-cancer TCGA | Statistical models beat deep methods on calibration | Leaderboard tests whether v12 deep stack is justified vs simple baselines (matches MEMORY floor concern) |
| 3 | [arXiv 2405.08226](https://hf.co/papers/2405.08226) | SeNMo (Waqas) | 2024 | Pan-cancer OS multi-omics | GDC 33 sites | C-index 0.758 held-out | High-width / low-N analogue of MM cohort scale |
| 4 | [10.1093/biomtc/ujaf063](https://doi.org/10.1093/biomtc/ujaf063) (PMID 40417913) | Conformal predictive intervals via resampling (Qin) | 2025 | Conformal prediction for right-censored survival | Simulated + breast cancer | Validity for 2-sided CIs | Direct upgrade for v12 Mondrian JK+ |
| 5 | [10.1101/2024.05.15.593193](https://doi.org/10.1101/2024.05.15.593193) (PMID 38798338) | MMRF Immune Atlas (Pilcher/Bhasin) | 2024 | MM single-cell immune profiling for outcome | CoMMpass N=263 NDMM, 1.15M cells | Immune+genetic improves stratification | Cell-state ground truth for any MM single-cell adapter |
| 6 | [10.1016/j.crmeth.2024.100781](https://doi.org/10.1016/j.crmeth.2024.100781) (PMID 38761803) | Subtype-WGME (Yang) | 2024 | Whole-genome multi-task encoder | 8 TCGA datasets | Beats baselines; 195 biomarkers | Hidden-layer adaptive amalgamation candidate vs MOFA+ |
| 7 | [10.1093/bioinformatics/btae360](https://doi.org/10.1093/bioinformatics/btae360) (PMID 38857453) | Subtype-MGTP (Xie) | 2024 | Multi-omics translation under missing modality | TCGA imbalanced | Beats 9 SOTA, robust to missing protein | Graceful single-view degradation matches v12 spec |
| 8 | [10.1186/s12967-024-04864-x](https://doi.org/10.1186/s12967-024-04864-x) (PMID 38243340) | MOSD (Duan) | 2024 | Weighted affinity + self-diffusion | 10 TCGA cancer types | Significant survival separation; cheap | Lightweight MOFA+ alternative |
| 9 | [10.1016/j.cmpb.2025.108603](https://doi.org/10.1016/j.cmpb.2025.108603) (PMID 39826483) | CRC integrator benchmark (Zhang) | 2025 | Benchmark of 8 integrators | TCGA-CRC + simulated | SNF wins overall | Empirical: classical SNF beats deep methods on bulk subtyping |
| 10 | [10.1016/j.ymeth.2024.09.016](https://doi.org/10.1016/j.ymeth.2024.09.016) (PMID 39423914) | Subtype-MVCC (Kuang) | 2024 | Weakly paired multi-omics contrastive | TCGA | Beats 9 baselines | Matches v12 paired-data limit (N_paired=29) |

## §2 MM-PFS direct competitors (≤5)

1. **Hussain et al. 2024**, *npj Digital Medicine* PMID 39075240 — N=703 NDMM TOURMALINE + N=720 external. Transformer joint PFS+OS+AE+biomarker. Beats ISS p<0.001. **Primary head-to-head target.** Only MM-PFS model with cohort ≥500 surfaced 2024-2026.
2. **Pilcher et al. 2024 MMRF Immune Atlas**, bioRxiv PMID 38798338 — N=263 / 1.15M cells; immune+genetic stratification.
3. **Shan & Susanibar-Adaniya 2026 FHRMM**, *Cancer Medicine* PMID 42057483 ([10.1002/cam4.71900](https://doi.org/10.1002/cam4.71900)) — N=181 functional-HR MM from CoMMpass; baseline cytogenetics + SKY92 lose prognostic value within FHRMM (OS 20.7 vs 18.1mo p=0.059). **Critical falsifier for any baseline-only MM-PFS predictor.**
4. **Morita et al. 2024 3D-CNN MRI**, *J Med Syst* PMID 38456950 — N=142; AUROC 0.804 external for PFS classification.
5. **Faghani et al. 2024 Mayo WBLDCT radiogenomics**, *Skeletal Radiol* PMID 38937291 — N=151; AUROC 0.874 t(4;14) prediction from CT.

Cohort ≥500 filter: only #1 passes.

## §3 Multi-omics integrator candidates (≤5)

1. Subtype-WGME (PMID 38761803)
2. Subtype-MGTP (PMID 38857453)
3. MOSD (PMID 38243340)
4. Subtype-MVCC (PMID 39423914)
5. CRC integrator benchmark (PMID 39826483)

Best graceful-degradation candidates for v12: **Subtype-MGTP** and **Subtype-MVCC**.

## §4 Foundation-model bulk-RNA adapter candidates with F2-leak risk

| Model | DOI / arXiv | Pretrain N (cells) | F2 leak risk | Bulk-RNA / ComBat protocol |
|---|---|---|---|---|
| BMFM-RNA (IBM 2025) | [arXiv 2506.14861](https://hf.co/papers/2506.14861) | CELLxGENE-scale | **MODERATE** — donor-ID filter required | Fine-tuning hooks; no explicit ComBat protocol |
| TEDDY family (2025) | [arXiv 2503.03485](https://hf.co/papers/2503.03485) | 116M cells | **MODERATE-HIGH** — held-out-donor splits documented; pretraining contamination must be audited | No bulk adapter documented |
| GRNFormer (2025) | [arXiv 2503.01682](https://hf.co/papers/2503.01682) | inherited from base RNA-FM | **LOW-MODERATE** — risk inherited | Drug-response evaluations on bulk; documented |
| GeneMamba (2025) | [arXiv 2504.16956](https://hf.co/papers/2504.16956) | ~30M cells | **LOW-MODERATE** — pathway-aware contrastive loss | Rank-based gene encoding (naturally batch-robust); no explicit ComBat |
| Nephrobase Cell+ (2025) | (PMID 41256511) | 39.5M kidney-only | **LOW** — kidney-only methodology template | MoE architecture reusable; no bulk adapter |

**Field-wide gap:** no public bulk-RNA + ComBat-corrected protocol for any of the listed FMs. The MEMORY-recorded Geneformer V1 F2-rejection (RUNS.md row r-2026-05-03-v11s1-encoder-reject) reflects this gap.

## §5 Causal / survival causal subset

1. **Oldham et al. 2026 ILD causal mediation**, Res Square ([10.21203/rs.3.rs-8714555/v1](https://doi.org/10.21203/rs.3.rs-8714555/v1), PMID 41674845). Plasma proteomics → survival via lung-function decline; 7/47 proteins survive validation cohort + sensitivity to unmeasured confounding. n=1963 + n=1172. **Direct methodology template for v10 Sprint 4 NIE workflow.**
2. **Loh & Ananth 2025**, *Epidemiology* PMID 39996615 — interventional indirect effects with multiple mediators; bias formulas for unmeasured confounding.
3. **Wang et al. 2025 acetaminophen MIMIC-IV**, *Front Pharmacol* PMID 40963684 — PSM + IPW + E-value (2.03–2.26) + mediation through body temperature. Worked example of E-value-aware mediation on survival.
4. **SurvHTE-Bench 2026**, [arXiv 2603.05483](https://hf.co/papers/2603.05483) — first comprehensive benchmark for HTE under censoring + assumption violations.
5. **Lambert et al. 2024 weighted CP under covariate shift**, [arXiv 2407.19938](https://hf.co/papers/2407.19938) — density-ratio-weighted conformal prediction in latent space.

## §6 Recommended top-3 head-to-head benchmark targets

| Rank | Target | DOI | Head-to-head design | Cost |
|---|---|---|---|---|
| 1 | Hussain et al. (Sontag MM transformer) | [10.1038/s41746-024-01189-3](https://doi.org/10.1038/s41746-024-01189-3) | Reproduce on MMRF CoMMpass; report C-index, IBS, calibrated PFS at 12/24/36mo. v12 must match-or-beat their ISS-delta. | MEDIUM: ~3-5 person-days reimplement; <100 GPU-hr |
| 2 | SurvBoard (Wissel) | [10.1093/bib/bbaf521](https://doi.org/10.1093/bib/bbaf521) | Submit v12 to leaderboard; standardized preprocessing+scoring; validates v12 deep stack vs RSF/BlockForest. **Direct test of MEMORY floor concern.** | LOW-MEDIUM: ~2-3 person-days API wrapper |
| 3 | Oldham ILD mediation w/ unmeasured-confounder sensitivity | [10.21203/rs.3.rs-8714555/v1](https://doi.org/10.21203/rs.3.rs-8714555/v1) | Adopt their two-cohort NIE + sensitivity workflow as v12 mediation reporting template. Re-run v10 Sprint 4 NIE with E-value alongside. | LOW: ~1-2 person-days |

## §7 Negative results / coverage gaps (real evidence only)

- PubMed query `scGPT bulk RNA fine tuning clinical` 2024-2026: **0 hits.** Confirms gap.
- PubMed query `scFoundation UCE Nicheformer transfer learning`: **0 hits** (these remain arXiv-only).
- PubMed query `scFoundation bulk RNA drug response`: **0 hits.**
- HF Hub search `multiple myeloma`: **0 models, 4 small low-quality datasets.** **CONFIRMED GAP — no MM-specific foundation model exists publicly on HF.**
- Open Targets MCP not available this session — deferred.
- bioRxiv/medRxiv cancer-biology slice 2024-06 to 2026-05 (first 30 results): no MM-specific FM or bulk-adapter preprint surfaced.
- UNVERIFIED — only PMID 39713879 (Wang & Shen 2024) returned for "MM high risk gene expression signature 2024 2025"; small N exploratory, not a serious head-to-head candidate.
