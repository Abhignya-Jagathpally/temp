# SOTA Comparison — ResistanceMap v6 head-to-head against named SOTA + auto-discovered comparators
_Generated: 2026-05-02. PMIDs verified live via PubMed MCP. Companion document: `literature_review.md`._

> **PubMed attribution**: All PMID/DOI references in this document were retrieved from PubMed via the live `mcp__claude_ai_PubMed__*` tools.

## Comparison axes used throughout

For every comparator, I assess:
1. **Task overlap** — does the paper attempt the same prediction (cell-state transition / drug resistance / pathway-level forecasting)?
2. **Direct metric comparability** — different denominators / splits / cohorts make most reported numbers non-comparable. Be explicit when not.
3. **Inductive bias overlap** — VAE+ODE+GAT+CrossAttention vs the SOTA's primary architectural choice.
4. **Where ResistanceMap is uniquely positioned** — concrete niche.

ResistanceMap baseline numbers (verified from this repo):
- test_mse = **2.3731** on 132 samples × 11 drugs (NaN-masked) — `checkpoints/pipeline_validated.pt`
- baseline rank: **4 / 11** in `paper/tables/baseline_comparison.md` — beaten by Zero-predictor (2.3678), Per-drug-train-mean (2.3678), GradientBoosting-PCA-256 (2.3699)
- per-drug performance: best Venetoclax MSE 0.010, worst Panobinostat MSE 22.3, Spearman 0.044 — large variance hidden by aggregate

## User-named comparators

### 1. TRACERx Lung — Genomic-transcriptomic evolution in lung cancer and metastasis (Nature 2023)

According to PubMed, **PMID 37046093** [DOI](https://doi.org/10.1038/s41586-023-05706-4):
- Authors: Martínez-Ruiz, Black, Frankell, …, Swanton, McGranahan
- 354 NSCLC tumors, 947 regions, 96 normal-adjacent — paired WES + RNA-seq
- task: intratumor heterogeneity (ITH), allele-specific expression, metastasis-seeding ML
- metric: not a single headline value — multiple ML approaches that link evolution context to metastasis-seeding probability

**Honest comparability**: NOT a comparable model. This is an *evolutionary genomics study of NSCLC*, not a drug-resistance ML predictor on MM. The framing similarity ("evolution-driven future state") is conceptual, not methodological. Cite as **inspirational framing**, not as a baseline.

### 2. Pan-cancer proteogenomics characterization of tumor immunity (2024)

The exact user-stated title returned no PMID. Closest verified match: **PMID 40957478** [DOI](https://doi.org/10.1016/j.jare.2025.09.014) — Shi et al, J Adv Res 2025, "Integrative proteomic characterization of human lung adenocarcinoma with KRAS G12 mutations." 96 LUAD patients, LC-MS/MS proteomics, three molecular subtypes including immune-modulation subtype. Different disease, different task, but a direct demonstration of clinical proteogenomics-with-immune-context — the methodology family the user likely meant.

**Honest comparability**: Different task (subtype discovery, not drug resistance). Use as a **methodological template** for proteogenomic stratification of MM, NOT as a head-to-head baseline.

### 3. ML immunotherapy-related signature in melanoma (2024)

Closest verified peer-reviewed match: **PMID 41613147** [DOI](https://doi.org/10.3389/fimmu.2025.1742614) — Dong et al, Front Immunol 2026: integrating cuproptosis + ferroptosis gene signatures (CFRGs) to predict prognosis, immunotherapy response, and drug sensitivity in skin cutaneous melanoma. TCGA + GEO + GSE72056 single-cell. ML prognostic model, key genes IFNG/PTPN6/SLC38A1/SOCS1, molecular docking with selumetinib.

**Honest comparability**: Skin melanoma ≠ multiple myeloma. The framing ("ML signature → immunotherapy response + drug sensitivity") is exactly the family ResistanceMap should imitate, but on a different disease. Use as a **template for how to report** prognosis + drug sensitivity in one model.

### 4. iMLGAM (2025)

According to PubMed, **0 hits**. This paper is not in PubMed. Likely venue: arXiv preprint or a conference proceedings (workshop) not indexed by NLM. **Cannot verify** any claim. The user should provide a DOI / arXiv ID.

### 5. TrajectoryNet (Tong, Krishnaswamy 2020)

According to PubMed, **0 hits** for "TrajectoryNet". This is an ICML 2020 paper (Tong, Huang, Wolf, van Dijk, Krishnaswamy) — not indexed by PubMed because it's a pure-ML conference paper. Cite via arXiv:2002.04461 / PMLR v119. The 2025 spiritual successor (Joint Velocity-Growth Flow Matching, [hf.co/papers/2505.13413](https://hf.co/papers/2505.13413)) handles unpaired-unbalanced snapshots more robustly.

**Honest comparability**: TrajectoryNet operates at **single-cell resolution** with **multiple snapshots in time**. ResistanceMap operates at **cell-line resolution** with **a single snapshot per line**. The two cannot be benchmarked head-to-head on the same data. ResistanceMap's "trajectory" is a misnomer relative to the trajectory-inference field — it's better described as an "implicit stability score."

## Auto-discovered direct comparators (PubMed search results, ranked by task overlap)

According to PubMed:

1. **PMID 41814396** [DOI](https://doi.org/10.1186/s12967-026-07946-0) — "Dynamic biomarker-based ML predicts short-term treatment response in MM" — F1=0.75 @ Cycle 4 vs R-ISS F1=0.32 (n=662). **The strongest current MM-clinical comparator.** Different metric (F1, classification) and different cohort (real patients, not cell lines), but it's *the* benchmark to beat for clinical relevance.
2. **PMID 41909977** [DOI](https://doi.org/10.1111/ejh.70177) — Critical review of 13 treatment-specific MM prediction models. Use this as the field-landscape doc.
3. **PMID 41762247** [DOI](https://doi.org/10.1007/s00277-026-06867-8) — Multi-omics + ML platelet-related prognostic signature in MM (2026). 13-gene signature from a 116-algorithm grid search; GSE124310 (which ResistanceMap also uses!) is part of the data.
4. **PMID 41711382** [DOI](https://doi.org/10.2196/75586) — AI assessment of duration-of-treatment in Japanese MM patients (n=2,762, MDV claims). Different outcome (DoT, not resistance), AUC 0.61–0.66 across horizons.

## Auto-discovered methodological comparators

5. **SeNMo** — [hf.co/papers/2405.08226](https://hf.co/papers/2405.08226) — pan-cancer multi-omics, GDC, C-index 0.758 on overall survival. Closest analog architecturally (multi-modal NN on omics).
6. **PaccMann** — [hf.co/papers/1811.06802](https://hf.co/papers/1811.06802) (2018) — multi-modal attention with PPI prior, IC50 prediction. Predates ResistanceMap; same conceptual ingredients minus ODE.
7. **Conditional Monge Gap** — [hf.co/papers/2504.08328](https://hf.co/papers/2504.08328) (2025) — neural OT, conditional on drug+dose, generalizes to unseen drugs. Best methodological comparator for the "transitions through pathways" claim.
8. **Threads slide-level foundation model** — [hf.co/papers/2501.16652](https://hf.co/papers/2501.16652) (2025) — H&E + genomic + transcriptomic, 54 oncology tasks. Different modality (pathology), but same fusion philosophy.
9. **STAGED** ([hf.co/papers/2507.11660](https://hf.co/papers/2507.11660), Krishnaswamy lab) — graph ODE + agent-based modeling on spatial transcriptomics. The Krishnaswamy-lab "successor" if ResistanceMap moved to spatial / single-cell.

## Where ResistanceMap is uniquely positioned

After the literature pass, the **only defensible niche** is:

> _"A reproducible multi-modal pipeline that fuses CCLE proteomics + epigenomics + STRING PPI + GDSC drug-IC50 with an explicit cell-line-level latent (VAE) and a calibrated implicit stability score, runnable end-to-end on a single GPU, with 10-agent DAG provenance, in under 30 minutes from clean checkpoints."_

That's an **engineering/reproducibility niche**, not a **scientific niche**. The scientific contribution claims (cell-state forecasting, "predict before it happens," pathway-level transition) are NOT supported by:
- the architecture (no longitudinal training data → no causal forecasting)
- the metrics (rank 4/11 against trivial baselines on real test data — see `paper/tables/baseline_comparison.md`)
- the resolution (cell-line, not single-cell, so trajectory claim is loose)

## Where ResistanceMap is dominated

| Claim | Dominated by | Evidence |
|---|---|---|
| "Multi-omics integration" | MOFA+ (PMID 32393329) | provides interpretable sparse factors; ResistanceMap's VAE is a black box |
| "MM clinical relevance" | PMID 41814396 (Dynamic biomarker RUSBoost) | F1=0.75 vs R-ISS 0.32 on real MM patients; ResistanceMap is on cell lines |
| "Multi-omics MM prognosis" | PMID 41762247 (Platelet 13-gene signature) | uses overlapping data (GSE124310) and reports per-cohort risk stratification |
| "Cell-state trajectory" | TrajectoryNet, scVelo, dynamo, CellRank, Conditional Monge Gap | all operate at single-cell resolution with proper time series |
| "Drug-response prediction on cell lines" | PaccMann (2018), MCA (2019, R²=0.86) | longer-published, similar attention-based fusion |
| "Pan-omics deep learning" | SeNMo (2024) C-index 0.758 | broader cancer types, similar architecture |

## Recommendation for benchmarking (which 3 to re-implement)

If you want a fair head-to-head:
1. **MOFA+ on the same train/val/test split** (same `data_ready.pt`) — the integration baseline. Already approximated via MiniBatchDictionaryLearning in `scripts/data_integration_audit.py`; replace with `mofapy2` for the real version.
2. **Dynamic-biomarker RUSBoost reformulated for cell-line IC50 prediction** (PMID 41814396 method, applied to GDSC). This forces apples-to-apples ML-on-MM-clinical-features.
3. **Conditional Monge Gap on the cell-line latents** ([hf.co/papers/2504.08328](https://hf.co/papers/2504.08328)) — this is the methodologically closest "predict resistance under unseen drug/dose" model.

## Overall verdict (sota-comparator perspective)

**Conditional pass** on the methodological niche (engineering/reproducibility), **fail** on the scientific-novelty claim. The "predict which resistance state, when, through which pathway, before it happens" framing requires longitudinal multi-snapshot data ResistanceMap does not have, and per-drug-mean already beats it on the available test set.

ResistanceMap's clearest path to genuine novelty is to commit to one of these three reframings:
- (a) become an MM-specific MOFA+ extension with sparse factors → interpretability gap closed
- (b) ingest single-cell scRNA-seq from GSE124310/GSE271107 (already in `data/`) and become a cell-state trajectory model in the Krishnaswamy lineage
- (c) drop the "before it happens" claim and reposition as a **reproducibility-first pan-omics MM cell-line resistance benchmark**, which is itself valuable to the field

---
_End of SOTA comparison._
