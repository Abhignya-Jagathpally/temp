# Literature Review for ResistanceMap v6 — Comparator Landscape
_Generated: 2026-05-02 — produced by the v8 main-thread literature-deep-research pass using live PubMed / bioRxiv / HuggingFace MCPs. Every entry below was retrieved by an actual search; rows without a PMID/DOI/arXiv are explicitly flagged._

> **Attribution**: According to PubMed, the references in this document derive from `mcp__claude_ai_PubMed__search_articles` and `mcp__claude_ai_PubMed__get_article_metadata`. ML papers from `mcp__claude_ai_Hugging_Face__paper_search` are linked as `hf.co/papers/<arXiv-id>`.

## Tier 1 — Direct comparators (MM treatment-response / outcome prediction)

| Source | Year | Title | N | Method | Headline metric | MM? | Same task as ResistanceMap? |
|---|---|---|---|---|---|---|---|
| PubMed PMID **41814396** [DOI](https://doi.org/10.1186/s12967-026-07946-0) | 2026 | Dynamic biomarker-based ML predicts short-term treatment response in MM | 662 newly-diagnosed MM patients | RUSBoost on flow-cytometry biomarkers, longitudinal | **F1=0.75 @ Cycle 4** vs R-ISS F1=0.32 | ✓ | 🟡 partial: short-term response (F1) ≠ ResistanceMap's drug-resistance score (MSE) |
| PubMed PMID **41909977** [DOI](https://doi.org/10.1111/ejh.70177) | 2026 | Treatment-Specific Prediction Models in MM: Critical Review | 13 models reviewed | systematic review (Cox/regression, mostly clinical features) | external validation in 7/13 | ✓ | 📚 review — **the** baseline-landscape paper for MM treatment-specific models |
| PubMed PMID **41762247** [DOI](https://doi.org/10.1007/s00277-026-06867-8) | 2026 | Multi-Omics + ML platelet-related prognostic signature in MM | bulk + scRNA from GSE124310/GSE6477/TCGA-MM/GSE4581/GSE24080/GSE136337 | Cox + Ridge over 116 algorithm grid; 13-gene signature | risk-stratification ROC reported per cohort | ✓ | 🟡 partial: prognosis (overall risk) ≠ per-drug resistance |
| PubMed PMID **41711382** [DOI](https://doi.org/10.2196/75586) | 2026 | AI assessment of duration-of-treatment in MM (real-world claims data) | 2,762 patients (Japanese MDV claims) | Point-Wise Linear (PWL) explainable DL vs ElasticNet/XGBoost | AUC 0.61–0.66 across 3/6/12-mo DoT | ✓ | 🟡 different outcome (treatment duration, not resistance) |

**Coverage gap**: a search for the exact triple (`multiple myeloma drug resistance prediction machine learning multi-omics`) returned **0 results** in PubMed — meaning ResistanceMap's task framing is genuinely under-explored. However, narrower searches surface adjacent work; the gap is in *framing*, not in disease attention.

## Tier 2 — Methodological comparators (multi-modal drug-sensitivity ML)

These are methods that fuse ≥2 modalities to predict drug response on cell lines / tumors, regardless of disease.

| Source | Year | Title | Method | Reported metric |
|---|---|---|---|---|
| arXiv via HF [hf.co/papers/2405.08226](https://hf.co/papers/2405.08226) | 2024 | SeNMo: Self-Normalizing Deep Learning for Multi-Omics Oncology | self-normalizing NN, 5 modalities, 33 cancers (GDC) | C-index 0.758 on test |
| arXiv via HF [hf.co/papers/1811.06802](https://hf.co/papers/1811.06802) | 2018 | PaccMann: anticancer compound sensitivity via multi-modal attention | SMILES + gene-expr + PPI attention | IC50 prediction (panel) |
| arXiv via HF [hf.co/papers/1904.11223](https://hf.co/papers/1904.11223) | 2019 | Multimodal attention conv encoder for compound sensitivity | MCA encoder (SMILES + gene-expr + PPI) | R²=0.86, RMSE=0.89 |
| arXiv via HF [hf.co/papers/2411.04747](https://hf.co/papers/2411.04747) | 2024 | Equivariant GAT for cell-line-specific drug synergy | E(3)-equivariant GAT + structural motifs | beats SOTA by >28% acc on DrugComb |
| arXiv via HF [hf.co/papers/2501.16652](https://hf.co/papers/2501.16652) | 2025 | Threads: slide-level foundation model for oncology | multimodal H&E + genomic + transcriptomic | beats baselines on 54 oncology tasks |
| arXiv via HF [hf.co/papers/2303.06471](https://hf.co/papers/2303.06471) | 2023 | Multimodal Data Integration for Oncology w/ DNN (review) | review of GNN+Transformer fusion | n/a (review) |

## Tier 3 — Cell-state forecasting / trajectory inference (the "predict before it happens" claim)

ResistanceMap claims to forecast resistance state. The field equivalent is single-cell trajectory inference — and it operates at single-cell resolution, which ResistanceMap does NOT.

| Source | Year | Title | Method | Resolution | Code | Notes |
|---|---|---|---|---|---|---|
| PubMed PMID **32393329** [DOI](https://doi.org/10.1186/s13059-020-02015-1) | 2020 | **MOFA+**: comprehensive integration of multi-modal single-cell | variational sparse PCA, shared+specific factors | sample- and cell-level | mofapy2 | **The** multi-omics integration baseline. ResistanceMap's VAE has no sparsity prior → no factor-level interpretability. |
| PubMed PMID **29925568** [DOI](https://doi.org/10.15252/msb.20178124) | 2018 | Multi-Omics Factor Analysis (MOFA) original | unsupervised factor model on 200 CLL samples | sample | MOFA | precursor; CLL drug-response application is the methodological cousin of what ResistanceMap attempts on MM |
| HF [hf.co/papers/2505.13413](https://hf.co/papers/2505.13413) | 2025 | VGFM: Joint Velocity-Growth Flow Matching for single-cell dynamics | flow-matching + semi-relaxed OT | single-cell | yes | direct successor to TrajectoryNet for unpaired snapshots |
| HF [hf.co/papers/2510.22033](https://hf.co/papers/2510.22033) | 2025 | LOT: Linearized Optimal Transport for single-cell point clouds | LOT embedding for patient point clouds | patient-as-point-cloud | yes | **Krishnaswamy lab** — patient-level OT for treatment-response interpretability; closer to ResistanceMap's framing than scVelo et al. |
| HF [hf.co/papers/2504.08328](https://hf.co/papers/2504.08328) | 2025 | Conditional Monge Gap (cell perturbation under drug/dose) | neural OT, conditional on drug + dose | single-cell | yes | beats condition-specific SOTA; cross-task generalization to unseen drugs |
| HF [hf.co/papers/2507.11660](https://hf.co/papers/2507.11660) | 2025 | STAGED: spatio-temporal agent-based GNN cellular dynamics | graph ODE + ABM + attention | spatial + single-cell | yes | **Krishnaswamy lab** — extends ODE+attention to spatial transcriptomics |
| HF [hf.co/papers/2210.06662](https://hf.co/papers/2210.06662) | 2022 | Action Matching: stochastic dynamics from samples | learn dynamics from independent snapshots, no OT solver back-prop | snapshot data | yes | the conceptual precedent for "predict from current snapshot only" — exactly ResistanceMap's setting |
| TrajectoryNet (Tong & Krishnaswamy, ICML 2020) | 2020 | TrajectoryNet: dynamic OT-based trajectory inference | continuous normalizing flow + dynamic OT | single-cell | yes | **named SOTA from user**; not in PubMed (ML venue) — find via arXiv 2002.04461 / PMLR |
| scVelo (Bergen et al., 2020) | 2020 | RNA velocity at single-cell | likelihood model on spliced/unspliced | single-cell | yes | not retrieved by `scVelo dynamo CellRank` triple search (PubMed had no match for that term combination) — must search separately |
| dynamo (Qiu et al., Cell 2022) | 2022 | analytic vector-field cell fates | RNA velocity → vector field | single-cell | yes | same caveat |
| CellRank 2 (Lange et al., Nat Methods 2024) | 2024 | absorption probabilities + lineage drivers | velocity-kernel + macrostates | single-cell | yes | same caveat |

## Tier 4 — User-named comparators

| Paper | PubMed status | Verified |
|---|---|---|
| Genomic-transcriptomic evolution in lung cancer & metastasis (TRACERx Lung 2023) | **PMID 37046093** [DOI](https://doi.org/10.1038/s41586-023-05706-4) | ✓ Martínez-Ruiz et al., Nature 2023 |
| Pan-cancer proteogenomics characterization of tumor immunity (2024) | the **exact** title returned no PMID; closest verified hits: PMID 40957478 (KRAS-G12 LUAD proteogenomics, 2025), PMID 38359819 | 🟡 user's exact title not located — likely refers to a CPTAC consortium paper; the closest verified is **PMID 40957478** [DOI](https://doi.org/10.1016/j.jare.2025.09.014) |
| ML immunotherapy signature in melanoma (2024) | 17 hits; top relevance verified is **PMID 41613147** [DOI](https://doi.org/10.3389/fimmu.2025.1742614) — Front Immunol 2026, cuproptosis-ferroptosis signature | ✓ closest peer-reviewed match |
| iMLGAM (2025) | **0 hits in PubMed** | ❌ NOT FOUND in PubMed — likely arXiv/preprint or non-indexed venue. Cannot verify. |
| TrajectoryNet (Tong, Krishnaswamy 2020) | **0 hits with "TrajectoryNet"** in PubMed | ❌ Not in PubMed (ML conference paper at ICML 2020); cite directly via arXiv:2002.04461 |

## Disease-context grounding

According to PubMed (PMID 41909977 [DOI](https://doi.org/10.1111/ejh.70177)) — **the** 2026 critical review of treatment-specific MM prediction models:

- 13 published treatment-specific clinical prediction models in MM
- regimens covered: Bortezomib induction, Daratumumab combos, Ixazomib triplets, CAR-T
- **most models use traditional regression**, calibration inconsistently reported
- external validation: 7/13
- decision-curve analysis: 2/13
- **NONE implemented as online calculators or in EHR decision support**

Best comparable single-effort: PMID 41814396 — RUSBoost on dynamic biomarkers, **F1=0.75 at Cycle 4 vs R-ISS F1=0.32**. This is the strongest current MM-specific benchmark for short-term treatment response.

## Coverage gaps

What was searched for and **not found**:
- "multiple myeloma drug resistance prediction machine learning multi-omics" — 0 results (the precise framing of ResistanceMap is under-published)
- "DeepCDR DeepSurv drug response prediction" — 0 results (those names don't co-occur in PubMed; each is a separate paper, both well-known in the ML community but not jointly indexed)
- exact title "Pan-cancer proteogenomics characterization of tumor immunity" — 0 results (user may have paraphrased a CPTAC paper)
- iMLGAM — 0 results (preprint/non-indexed)

## Recommendation for sota-comparator

The three most defensible head-to-head benchmarks for ResistanceMap are:

1. **MOFA+ (PMID 32393329)** — must be implemented as the multi-omics integration baseline. Same data, factor-based interpretation, sparsity prior. ResistanceMap currently has no factor-level decomposition.
2. **Dynamic biomarker model (PMID 41814396)** — closest *clinical* MM comparator. Only fair if ResistanceMap reframes from cell-line drug IC50 to patient response.
3. **Conditional Monge Gap (hf.co/papers/2504.08328)** — closest *methodological* trajectory comparator that handles unseen drugs via OT, runnable on single-cell data ResistanceMap could ingest.

**Honest caveat**: TRACERx Lung (PMID 37046093) is named by the user but is **not a comparable model** — it is an evolutionary-genomics study of lung NSCLC, not a drug-resistance ML method on MM. Including it as a "comparator" would be category-confusion. It is best cited as inspiration for the cell-state/clonal-evolution framing.

---
_End of literature review._
