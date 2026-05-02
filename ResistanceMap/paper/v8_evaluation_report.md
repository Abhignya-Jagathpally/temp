# v8 Strict Evaluation Report — ResistanceMap

_Generated 2026-05-02 by the v8 multi-agent architecture (`.claude/agents/architecture-chair.md` + 8 specialist subagents). All numerical claims are anchored to on-disk checkpoints; literature claims to PubMed/HF tools verified live. The evidence base is in `paper/v8_artifacts/`._

> **Source attribution**: PubMed PMIDs and DOIs in this document were retrieved from PubMed via `mcp__claude_ai_PubMed__*`; arXiv/HF IDs via `mcp__claude_ai_Hugging_Face__paper_search`.

## Verdict

**CONDITIONAL_PASS for engineering / FAIL for the headline scientific claim.** **TRL: 2** (technology concept formulated; experimental validation pending).

The pipeline runs end-to-end with reproducible provenance and a real test_mse, but on its own benchmark it is dominated by a trivial late-fusion Ridge baseline, and the central scientific claim ("predict which resistance state, when, through which pathway, before it happens") is not supported by the training data.

## Headline finding

> **On its own real test set, ResistanceMap (test_mse=2.3731) is beaten by averaging two per-modality Ridge regressions (test_mse=2.3653) and by predicting "all zeros" (2.3678).** The 4 BLOCKER fixes from the v7 audit landed; the v8 strict-evaluation pass surfaces a different, deeper problem: the architecture is not earning its complexity on the available data.

## What ResistanceMap does that the field's SOTA does NOT

After live PubMed / bioRxiv / HuggingFace searches, the **only defensible niche** is engineering, not science:

> _A reproducible 10-agent DAG that fuses CCLE proteomics + epigenomics + STRING PPI + GDSC drug-IC50 with cryptographic boundary hashing, AgentOps observability, automatic governance gating (Tier A→D), and per-drug + baseline-comparison artifacts — runnable end-to-end on a single GPU in ~30 min from clean checkpoints._

That niche is **real** and **valuable** as infrastructure, but it is not a *scientific* differentiator.

The originally-claimed scientific niches (cell-state forecasting, pathway-level transition prediction, "before it happens") are each occupied by stronger, more directly-tasked methods — see "Where dominated" below.

## What ResistanceMap does that the field's SOTA does BETTER

Top 6 gaps with citations:

1. **Multi-omics integration interpretability** — dominated by **MOFA+** (PMID **32393329** [DOI](https://doi.org/10.1186/s13059-020-02015-1), Argelaguet et al. Genome Biology 2020). MOFA+ produces sparse interpretable factors with per-modality loadings; ResistanceMap's 64-d VAE latent is a black box without sparsity prior or factor decomposition.
2. **MM-clinical relevance** — dominated by **PMID 41814396** [DOI](https://doi.org/10.1186/s12967-026-07946-0) (Xiong et al. J Transl Med 2026, "Dynamic biomarker-based ML predicts short-term treatment response in MM"): F1=0.75 at Cycle 4 vs R-ISS F1=0.32 on n=662 real MM patients. ResistanceMap evaluates on cell lines, not patients.
3. **MM multi-omics prognostic signature** — covered by **PMID 41762247** [DOI](https://doi.org/10.1007/s00277-026-06867-8) (Li et al. Ann Hematol 2026, 13-gene platelet signature) using overlapping data (GSE124310 — same as ResistanceMap) with explicit risk stratification.
4. **Cell-state trajectory forecasting** — dominated by **TrajectoryNet** (Tong & Krishnaswamy, ICML 2020, arXiv:2002.04461), the **Conditional Monge Gap** ([hf.co/papers/2504.08328](https://hf.co/papers/2504.08328), 2025), **STAGED** ([hf.co/papers/2507.11660](https://hf.co/papers/2507.11660), Krishnaswamy lab 2025), **VGFM** ([hf.co/papers/2505.13413](https://hf.co/papers/2505.13413), 2025) — all single-cell-resolution with proper time series. ResistanceMap is cell-line resolution, single snapshot per line.
5. **Pan-omics deep oncology** — covered by **SeNMo** ([hf.co/papers/2405.08226](https://hf.co/papers/2405.08226), 2024, C-index 0.758 across 33 cancers, GDC).
6. **Multi-modal attention drug-sensitivity** — covered by **PaccMann** ([hf.co/papers/1811.06802](https://hf.co/papers/1811.06802), 2018, R²=0.86) and the multimodal CNN attention encoder ([hf.co/papers/1904.11223](https://hf.co/papers/1904.11223), 2019).

## Per-drug performance (real test data)

From `logs/per_drug_metrics.csv` and `checkpoints/pipeline_validated.pt`:

**Best 3 (n_obs ≥ 56, reliable):**
- Venetoclax: MSE 0.010, MAE 0.086, Spearman 0.328
- Bortezomib: MSE 0.032, MAE 0.140, Spearman 0.338
- Dinaciclib: MSE 0.156, MAE 0.248, Spearman 0.373

**Worst 3:**
- Panobinostat: MSE 22.3, Spearman 0.044 (model is essentially noise)
- Doxorubicin: MSE 1.67, MAE 0.634
- Etoposide: MSE 0.531, MAE 0.518

The aggregate `test_mse=2.3731` is dominated by Panobinostat. Without that single drug the model would compete favorably; with it, the model's average is below the per-drug-mean baseline.

## Data-integration finding

From `paper/v8_artifacts/data_integration_audit.md` (head-to-head on `data_ready.pt`, same train/val/test split):

| Rank | Method | test_mse |
|---|---|---|
| 1 | **D. Late fusion (avg of per-modality Ridge)** | **2.3653** |
| 2 | **ResistanceMap CrossModalFusionNet** | **2.3731** |
| 3 | C. MOFA+-like shared factors (64) → Ridge | 2.4440 |
| 4 | A. Naive concat → Ridge | 2.4583 |
| 5 | B. PCA(256/20) per-modality → Ridge | 3.5652 |

Batch-effect proxy on PCA(64) of train proteomics with quartile pseudo-batch:
- kBET mean chi² = 0.180 (acceptable)
- iLISI mean = 3.434 (close to n_batches=4, **good integration**)
- silhouette by pseudo-batch = -0.009 (effectively zero — modalities mix well)

**Verdict on the fusion mechanism**: **CONDITIONAL** — beats naive concat by ~0.09 MSE, but is beaten by simple late fusion. Cross-attention fusion is **not justified** by the current 622-train / 132-test scale.

## Trajectory finding

The "trajectory / cell-state forecasting" claim was assessed against the actual data:

- **Identifiability**: 132 test samples × 11 drugs vs 64-d latent + 8 ODE params per drug ≈ 1.5 data points per param at the optimistic upper bound. Underdetermined.
- **Resolution mismatch**: ResistanceMap's "cell state" is a 64-d cell-line-level VAE embedding. True cell-state trajectory inference (PHATE, scVelo, dynamo, CellRank, TrajectoryNet, VGFM) operates at single-cell resolution. The framing is loose.
- **"Before it happens" claim**: requires longitudinal multi-snapshot training pairs `(X_t, X_{t+Δ})`. The CCLE+GDSC training pairs are contemporaneous `(X, IC50_observed)`. There is no temporal-extrapolation signal in training.

**Verdict on trajectory claim**: **FAIL** — claim outruns training-data structure.

## Causal claim audit

The "predict which resistance state, when, through which pathway, before it happens" claim requires:

| Sub-claim | Required identification assumption | Holds? |
|---|---|---|
| "which state" | overlap (positivity) of cell-state distribution under the drug regimes of interest | **partial** — CCLE has ~886 lines, GDSC IC50 sparsely observed |
| "when" | time-to-event labels in training | **NO** — training pairs are not longitudinal |
| "which pathway" | perturbation data (CRISPR / shRNA screens) for grounding | **NO** — only association, not causal grounding |
| "before it happens" | longitudinal `(X_t, X_{t+Δ})` pairs | **NO** |

**Verdict on causal validity**: **FAIL** for "when", "which pathway" causally, "before it happens"; **CONDITIONAL** for "which state" (associative only).

## Visualizations

5 Krishnaswamy-style figures generated from real test data in `paper/v8_artifacts/visualizations/` (UMAP fallback because PHATE not installed; clearly labeled in captions):
- `fig_latent_manifold.png` — 886 cell-line VAE latents on UMAP, colored by stability score, test points circled
- `fig_pseudotime_per_drug.png` — 11 per-drug pred-vs-truth scatters colored by pseudotime
- `fig_meld_density.png` — drug-class resistant-cell density on the manifold
- `fig_residuals_heatmap.png` — test cell-line × drug residual matrix (Panobinostat column dominates)
- `fig_resistance_landscape.png` — kernel-smoothed resistance score over manifold

Each PNG ships with a `.caption.md` explaining what to interpret and acknowledging the UMAP fallback.

## Falsification criteria

Three concrete tests that would flip this verdict from CONDITIONAL_PASS to PASS:

1. **Beat late fusion** (test_mse < 2.3653) on the same train/val/test split with a paired-Wilcoxon p < 0.05 across 5 random seeds. Currently: ResistanceMap loses by 0.008 MSE.
2. **Provide longitudinal training data** for at least 200 cell lines with 2+ snapshots, AND show forecasting test_mse improves at horizon t=Δ vs at t=0. This makes the "before it happens" claim defensible.
3. **Persist per-drug attribution that recovers ≥6/11 known canonical drug targets** in the top-3 attended proteins (e.g., PSMB5 for Bortezomib, BCL2 for Venetoclax, HDAC1-3 for Panobinostat — verified via ChEMBL `get_mechanism`). Currently no attribution is exposed; the pathway-validator cannot evaluate grounding.

## Required changes (ranked, scientifically prioritized)

1. **Drop the "before it happens" claim until longitudinal data is added.** Reframe paper as a *snapshot* multi-omics MM resistance benchmark. (1-line README change)
2. **Persist per-drug attention weights / attributions** during `train_protein_network` and `train_fusion`, so `proteomics-pathway-validator` can ground them in KEGG/Reactome. (~50 LOC)
3. **Add a sparse-factor head** to the VAE (or replace with `mofapy2`) so the latent is interpretable per-modality. Closes the MOFA+ gap. (~200 LOC or mofapy2 dependency)
4. **Implement statistical significance test** (paired Wilcoxon across 5 seeds) for ResistanceMap vs late-fusion baseline. If non-significant, the architecture's complexity is unjustified. (~50 LOC in a new `scripts/sig_test.py`)
5. **Reframe paper claims around the Panobinostat outlier** — the model is not making one global prediction, it is doing well on 10 drugs and failing catastrophically on one. Either drop Panobinostat from the headline metric or explicitly study why HDAC inhibitors break the model.
6. **Connect `data/mmrf_loader.py` to `external_validation.py`** so MM patient-cohort claims can be tested without the disarmed synthetic path.
7. **Plan a single-cell extension** (use the already-on-disk `gse124310.h5ad` and `gse271107.h5ad`) so a future v9 can claim Krishnaswamy-lineage cell-state trajectory with proper resolution.
8. **Cite TRACERx, CPTAC immunity, the melanoma signature paper, and TrajectoryNet correctly** — as inspirational framings, not as benchmarks. Inserting them as "comparators" in a head-to-head table would be category-confusion (different disease, task, modality).

## Uniqueness brief — what to put in the abstract instead

> _ResistanceMap is a reproducibility-first multi-omics benchmark for hematologic-malignancy drug-resistance prediction. It packages CCLE proteomics, CCLE epigenomics, STRING PPI, GDSC dose-response, and patient single-cell transcriptomics into a 10-agent DAG with cryptographic boundary hashing, AgentOps observability, automatic Tier-A→D evaluation governance, and head-to-head benchmarking against MOFA+, late-fusion, and PCA-Ridge baselines on identical splits. Average test_mse 2.37 (n=132 lines × 11 drugs); per-drug performance is excellent on Venetoclax (MSE 0.010) and Bortezomib (MSE 0.032) but fails catastrophically on Panobinostat (MSE 22.3) — surfacing HDAC-class extrapolation as a concrete future-work item. We release the pipeline, agent definitions, hooks, skills, and 5 Krishnaswamy-style visualizations as a reusable template for MM-specific multi-omics ML, with explicit non-claims around longitudinal forecasting until such data is added._

This positioning is honest, defensible, and useful to the field. The "predicts before it happens" framing is not.

## Where to read the full evidence

- `paper/v8_artifacts/literature_review.md` — 25+ verified PubMed/HF citations
- `paper/v8_artifacts/sota_comparison.md` — head-to-head against the 5 user-named SOTA papers
- `paper/v8_artifacts/data_integration_audit.md` + `.json` — head-to-head MOFA-like / concat / late-fusion / cross-attn
- `paper/v8_artifacts/visualizations/` — 5 PNGs + 5 captions
- `paper/tables/baseline_comparison.md` — original v7 baseline panel
- `logs/per_drug_metrics.csv` — per-drug MSE/MAE/Spearman
- `checkpoints/pipeline_validated.pt` — `metrics` dict with everything

---
_End of v8 evaluation report. The chair stands by all numerical claims; deviations from this report should be supported with the same level of provenance._
