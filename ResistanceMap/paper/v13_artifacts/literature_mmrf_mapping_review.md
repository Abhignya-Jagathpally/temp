# Literature Review — Mapping scRNA Atlases (GSE124310, GSE271107) onto MMRF CoMMpass for Multiple Myeloma Drug-Resistance / Progression Prediction

**Search date:** 2026-05-12
**Curator:** ResistanceMap v13 lit-scan agent
**Datasets in scope:**
- GSE124310 — Ledergor et al. 2018, ~27,796 BM PCs across HD/MGUS/SMM/MM
- GSE271107 — Yu et al. 2025 / Zhang 2025 used it; ~143,748 cells along HD→MGUS→SMM→MM continuum
- MMRF CoMMpass — 1,143 NDMM patients; bulk RNA-seq + cytogenetics + treatment + PFS/OS

Source attribution: All papers below were retrieved from PubMed (with attribution per their requirements), bioRxiv, and Hugging Face papers. DOIs are linked inline.

---

## Section A — Cross-cohort / cross-modality mapping (scRNA → bulk MMRF)

### A1. The two MM papers that have ALREADY done exactly the GSE124310+GSE271107→MMRF mapping you propose

**Yu et al. 2025 — PDIA4 in MM progression** (PMID 41121130)
- DOI: [10.1186/s12967-025-07098-7](https://doi.org/10.1186/s12967-025-07098-7) — *J Transl Med*
- **What they did:** Integrated scRNA from HD + MGUS + SMM + NDMM retrieved from GSE124310 AND GSE271107 to build a pseudotime trajectory of plasma-cell differentiation; identified PDIA4 as terminal-state-associated; validated prognostic significance in MMRF CoMMpass survival.
- **Method:** Standard Seurat integration → Monocle/Slingshot pseudotime → Cox regression in MMRF.
- **Relevance:** This is the single closest published template for what we are about to do. Their pipeline is GSE124310 ∪ GSE271107 → trajectory → DE-genes terminal vs early → score MMRF bulk → KM/Cox.
- **Caveat for us:** They use a single-gene marker (PDIA4) — *not* a multi-gene resistance signature or a foundation-model embedding. There is methodological headroom here.

**Song et al. 2026 — ELK3 plasma cell subtype C4** (PMID 42088520)
- DOI: [10.3389/fimmu.2026.1793391](https://doi.org/10.3389/fimmu.2026.1793391) — *Front Immunol*
- **What they did:** scRNA-seq of HD/MGUS/SMM/MM from GEO; pseudotime via Monocle; CellChat + SCENIC; in vitro validation of ELK3.
- **Method:** Trajectory-only; did not project onto MMRF bulk.
- **Relevance:** Confirms a "C4 plasma cell subtype" enriched in late stages — useful as a biological landmark for our subclone reference.

### A2. Disease-trajectory / immune-microenvironment atlas papers using the same datasets

**Bergiers, Caers et al. 2025** (PMID 41056512) — *PLoS Genet*
- DOI: [10.1371/journal.pgen.1011848](https://doi.org/10.1371/journal.pgen.1011848)
- **What they did:** scRNA + CITE-seq + BCR-seq of unsorted whole BM from 123 subjects (HD, MGUS, SMM, MM). Identified CD8+ T-cell + macrophage shifts; ligand–receptor (MIF, IL15, CD320, HGF, FAM3C, SERPINA1, BAFF) predicting progression.
- **Relevance:** The largest *paired* HD→MGUS→SMM→MM atlas. Bigger than GSE271107 alone. Cross-references the GSE271107 phenotype space.
- **N (cells):** ~123 subjects, unstated total cells but full BM (so >>plasma cells alone).

**Ledergor et al. 2018** (PMID 30523328) — *Nat Med* — the GSE124310 source paper
- DOI: [10.1038/s41591-018-0269-2](https://doi.org/10.1038/s41591-018-0269-2)
- **What they did:** Original scRNA-seq of 40 individuals across MM spectrum (incl. 11 HD); identified subclonal structure in 10/29 MM; detected MRD-like rare tumor PCs in asymptomatic and post-treatment. The reference paper for GSE124310.
- **N:** 40 individuals, 27,796 cells.
- **Relevance:** Always-cite source. Provides the cytogenetic subgroup metadata (hyperdiploid, t(11;14), t(4;14), t(14;16), del17p, 1q amp) we will need to stratify by.

### A3. The benchmark we should use for the deconvolution step itself

**Chu, Pe'er, Danko 2022 — BayesPrism** (PMID 35469013) — *Nat Cancer*
- DOI: [10.1038/s43018-022-00356-3](https://doi.org/10.1038/s43018-022-00356-3)
- **What they did:** Bayesian cell-proportion reconstruction using scRNA as prior on bulk RNA-seq. Validated in GBM/HNSCC/SKCM.
- **Why it matters:** In the Tran et al. (PMID 37717006, DOI [10.1038/s41467-023-41385-5](https://doi.org/10.1038/s41467-023-41385-5)) head-to-head of 9 methods on simulated breast bulk mixtures, **BayesPrism + DWLS had the lowest combined FP+FN** and best granular-immune-lineage deconvolution. BayesPrism is the current SOTA reference for "use scRNA atlas → deconvolve bulk".

**Ouzounis et al. 2026** (PMID 41028270) — *Methods Mol Biol*
- DOI: [10.1007/978-1-0716-4734-9_16](https://doi.org/10.1007/978-1-0716-4734-9_16)
- **What they did:** Recent (2026) tutorial/practical guide comparing CIBERSORTx, BayesPrism, MuSiC, Scaden, DWLS, hspe, CPM, Bisque, EPIC, scPER. Walks through preprocessing/reference-choice/QC for a deconvolution pipeline. Practical "what could go wrong" reference.
- **Relevance:** Use this as the protocol scaffold for our scRNA→MMRF projection module.

**Li, Zhou, Kalluri 2025 — scPER** (PMID 41309494) — *Adv Sci*
- DOI: [10.1002/advs.202514502](https://doi.org/10.1002/advs.202514502)
- **What they did:** Adversarial autoencoder + XGBoost; "estimating cell Proportions using single-cell RNA-seq Reference"; reported to beat CIBERSORTx, BayesPrism, Scaden, MuSiC, SCDC, DeSide, ReCIDE on melanoma + urothelial.
- **Relevance:** Newest deconvolution kid on the block; we should benchmark against it but BayesPrism remains the most-cited safe baseline.

### A4. Negative result — what we did NOT find
- **No results for query 'scANVI scArches reference mapping single cell transfer learning'** in PubMed.
- **No results for query 'Symphony Azimuth single cell reference mapping'.**
- These methods exist but are not indexed in MeSH; they live primarily in bioRxiv and in the scverse stack. For our purposes, they would be the *embedding-based* alternative to deconvolution-based mapping (and would be the natural sister method to scGPT fine-tuning).

---

## Section B — Disease-continuum modeling (HD→MGUS→SMM→MM)

### B1. Trajectory inference foundations (you must cite these even if generic)

**Bergen et al. 2020 — scVelo** (PMID 32747759) — *Nat Biotechnol*
- DOI: [10.1038/s41587-020-0591-3](https://doi.org/10.1038/s41587-020-0591-3)
- **One-liner:** Likelihood-based dynamical model for RNA velocity; handles transient states; the standard alternative to steady-state velocyto.

**Qiu et al. 2022 — Dynamo** (PMID 35108499) — *Cell*
- DOI: [10.1016/j.cell.2021.12.045](https://doi.org/10.1016/j.cell.2021.12.045)
- **One-liner:** Absolute RNA velocity + continuous vector fields + differential geometry → predicts optimal reprogramming paths and *in silico* perturbation outcomes. The closest published analog to what we are doing with TFM + CellRank in v10.

**Weiler et al. 2024 — CellRank 2** (PMID 38871986) — *Nat Methods*
- DOI: [10.1038/s41592-024-02303-9](https://doi.org/10.1038/s41592-024-02303-9)
- **One-liner:** Multi-view fate mapping at >1M cells; combines pseudotime + velocity + experimental timepoints + stemness in a unified Markov-chain framework. This is exactly the "data view agnostic" framework we should be using over GSE271107.
- **N:** Tested at multi-million-cell scale.

**Weiler & Theis 2026 — CellRank protocol** (PMID 41611959) — *Nat Protoc*
- DOI: [10.1038/s41596-025-01314-w](https://doi.org/10.1038/s41596-025-01314-w)
- **One-liner:** Reference protocol for running CellRank/CellRank2 at scale; this is the implementation tutorial we should follow line-for-line for the GSE271107 trajectory module.

**Shimamura 2025 — RNA velocity review** (PMID 40973591) — *Allergol Int*
- DOI: [10.1016/j.alit.2025.08.005](https://doi.org/10.1016/j.alit.2025.08.005)
- **One-liner:** 2025 review of scVelo / Dynamo / CellRank evolution, with explicit discussion of limitations (steady-state assumption violations, integration with spatial/multimodal). Useful for "what NOT to claim".

### B2. MM-specific trajectory work

**Yu et al. 2025 (PDIA4)** — already cited above (PMID 41121130). They use Monocle for pseudotime on GSE124310+GSE271107.

**Tan et al. 2025 — PTPRG / MalPlasma3 stemness** (PMID 41050668) — *Front Immunol*
- DOI: [10.3389/fimmu.2025.1658028](https://doi.org/10.3389/fimmu.2025.1658028)
- **What they did:** 103,171 scRNA cells from 18 MM patients (10 optimal, 8 suboptimal responders to VMP). Used InferCNV → 5 malignant subclusters. **MalPlasma3** subcluster harbored 93.1% of "poor-survival" cells. Validated in GSE9782, GSE2658, MMRF CoMMpass.
- **N:** 18 patients; bulk validation 760+ patients.
- **Relevance:** Cleanest "single-cell-subcluster → bulk-MMRF Cox" template currently published. The proof-of-concept that scRNA-derived subclone gene programs project meaningfully onto MMRF.

**Ohlstrom et al. 2025 — light-chain escape subclones** (PMID 39699274) — *Cancer Res Commun*
- DOI: [10.1158/2767-9764.CRC-24-0170](https://doi.org/10.1158/2767-9764.CRC-24-0170)
- **What they did:** Serial scRNA-seq of one patient's bone marrow through LCE; identified LAMP5 overexpression as marker of osteolysis + adverse prognosis; validated in MMRF CoMMpass + GSE24080.
- **Relevance:** Demonstrates *intra-patient* longitudinal scRNA-to-MMRF projection for a *resistance-associated* event (LCE). N=1 patient but high methodological rigor.

### B3. The MGUS/SMM transcriptional driver papers

**Fu et al. 2026 — DAP3 / UBE2S** (PMID 41609080) — *Front Biosci*
- DOI: [10.31083/FBL46992](https://doi.org/10.31083/FBL46992)
- **What they did:** WGCNA + LASSO Cox on GSE136337 + TCGA-MM + GSE4581 + GSE57317; identified DAP3 + UBE2S as drivers of MGUS→MM progression. Note: this paper builds a "MGUSscore" — directly analogous to what we want to extract from GSE271107.

**Zhang et al. 2025 — PRKD2** (PMID 41120632) — *Sci Rep*
- DOI: [10.1038/s41598-025-20615-4](https://doi.org/10.1038/s41598-025-20615-4)
- **What they did:** Bulk RNA-seq (24 MM + 6 marrows) + scRNA → WGCNA + pseudotime → identified PRKD2 as a stemness-progression driver; validated in MMRF-style cohort and CCLE drug-screen.
- **Relevance:** Combines bulk + sc via WGCNA + pseudotime — alternative integration strategy.

---

## Section C — Drug-resistance signal extraction from MMRF

### C1. Bortezomib / proteasome-inhibitor resistance signatures derived in MMRF

**Gargano et al. 2025 — MM-5C model (SOX11, METTL11B, C3, RBM10, HOMEZ)** (PMID 41339727) — *Sci Rep*
- DOI: [10.1038/s41598-025-30527-y](https://doi.org/10.1038/s41598-025-30527-y)
- **What they did:** Transcriptomic deconvolution + elastic-net Cox on CD138-negative BM fraction; 5-gene signature stratifies bortezomib-induction-treated patients; prognostically independent from cytogenetics + ISS.
- **Relevance:** This is the most-recent, methodologically-clean "AI-derived MMRF-validated bortezomib resistance signature." Direct comparator for ResistanceMap.

**Ding et al. 2026 — 5-gene BRG signature (IFI16, ARID5B, LTBP1, PNOC, CRIP1)** (PMID 41493673) — *Discov Oncol*
- DOI: [10.1007/s12672-026-04383-9](https://doi.org/10.1007/s12672-026-04383-9)
- **What they did:** Bortezomib-resistant cell-line transcriptomics → DEG → LASSO Cox → 5-gene signature. 1/3/5-year ROC AUC = 0.730/0.734/0.775.
- **Relevance:** Another head-to-head numerical target for our PFS prediction.

**Samur et al. 2023 — HDM mutational burden** (PMID 36603186) — *Blood*
- DOI: [10.1182/blood.2022017094](https://doi.org/10.1182/blood.2022017094)
- **What they did:** Deep WGS of paired diagnosis/relapse in IFM-2009 (N=68); ML model predicts HDM exposure from mutational pattern; clonal evolution analysis.
- **Relevance:** Genomic-evolution comparator for our subclone modeling.

### C2. MM systems-biology / multi-omics network models

**Murie et al. 2025 — mmSYGNAL** (PMID 40169765) — *Br J Cancer*
- DOI: [10.1038/s41416-025-02987-6](https://doi.org/10.1038/s41416-025-02987-6)
- **What they did:** SYGNAL network of transcriptional programs from 881 MM patients → cytogenetic-subtype-specific ML models for individualized risk + drug response. **Tested on 1,367 patients across 5 cohorts.** Significantly outperformed cytogenetics, ISS, and multi-gene panels at primary diagnosis, pre/post-transplant, and after multiple relapses. Drug-response predictions concordant with ex vivo efficacy of 67 drugs.
- **N:** 881 train + 1,367 validation.
- **Relevance:** This is the current published SOTA in cytogenetic-subtype-conditional MM risk and treatment-response prediction. The most important head-to-head comparator for ResistanceMap, given v11.5 already ties mmSYGNAL on PFS C-index.

**Sagar et al. 2022 — GCRS GNN** (PMID 36113255) — *Comput Biol Med*
- DOI: [10.1016/j.compbiomed.2022.106048](https://doi.org/10.1016/j.compbiomed.2022.106048)
- **What they did:** Hybrid GCN integrating clinical + lab connectivity graphs; 3-class risk stratification on 2 NDMM cohorts; outperforms prior methods on C-index + hazard ratio; SHAP interpretability.
- **Relevance:** GNN baseline. Provides the explicit "graph-of-patients" comparator.

### C3. MM joint multi-task modeling — the strongest current benchmark

**Hussain et al. 2024 — joint AI event prediction & longitudinal modeling** (PMID 39075240) — *NPJ Digital Med*
- DOI: [10.1038/s41746-024-01189-3](https://doi.org/10.1038/s41746-024-01189-3)
- **What they did:** Transformer that **jointly** predicts PFS + OS + AE + forecasts biomarkers + estimates treatment effects (IRd vs Rd). Trained on TOURMALINE (N=703 NDMM); externally validated on N=720 RRMM. Beat ISS-based risk model (p<0.001 Bonferroni). Found IgA-kappa subgroup benefits most from IRd.
- **N:** 703 + 720 = 1,423 (similar order to MMRF).
- **Relevance:** This is the published SOTA for "transformer-based joint MM event prediction." It is MIT/Takeda quality, externally validated, multi-task — and it does *not* use scRNA priors. ResistanceMap could differentiate explicitly by adding the scRNA atlas projection that this paper lacks.

### C4. Other recent prognostic signatures (cytogenetic-context aware)

**Fang et al. 2025** (PMID 40325058) — *Sci Rep*
- DOI: [10.1038/s41598-025-00074-7](https://doi.org/10.1038/s41598-025-00074-7)
- **What they did:** xCELL + CIBERSORT + ESTIMATE on N=859 MMRF + N=328 GSE19784; LASSO-Cox 10-gene signature (UBE2T, E2F2, EXO1, SH2D2A, DRP2, WNT9A, SHROOM3, TMC8, CDCA7, GPR132). 1-yr AUC 0.682, 5-yr AUC 0.714.

**De Ramón et al. 2022 — TP53 double-hit / DH-TP53-like** (PMID 35983648) — *Br J Haematol*
- DOI: [10.1111/bjh.18410](https://doi.org/10.1111/bjh.18410)
- **What they did:** N=660 MMRF + 850 microarray cohort; identified transcriptional signature of biallelic TP53 inactivation; "DH-TP53-like" group (N=50) without genetic biallelic TP53 inactivation has median OS<24 mo.
- **Relevance:** Reference for high-risk cytogenetic surrogate signatures.

**Wang et al. 2023 — TTK/GINS1/NCAPG 3-gene model** (PMID 36910651) — *Front Oncol*
- DOI: [10.3389/fonc.2023.1105196](https://doi.org/10.3389/fonc.2023.1105196)
- **What they did:** WGCNA + LASSO on N=860 NDMM MMRF; 3-gene model; OncoPredict drug-sensitivity inference.

**He et al. 2022 — relapsed MM scRNA** (PMID 35297204) — *Clin Transl Med*
- DOI: [10.1002/ctm2.757](https://doi.org/10.1002/ctm2.757)
- **What they did:** scRNA + scVDJ on 18 MM (12 NDMM + 6 RRMM) → 8 meta-programs → SMAD1, STMN1 as biomarkers → validated in MMRF CoMMpass.

**Chong et al. 2024 — t(4;14) REIIBP/eIF3E** (PMID 38124661) — *Haematologica*
- DOI: [10.3324/haematol.2023.283467](https://doi.org/10.3324/haematol.2023.283467)
- **What they did:** Mechanism paper for t(4;14)-specific bortezomib resistance via REIIBP → TLR7 → BTK; demonstrates ibrutinib partial-rescue. Useful biological grounding for our t(4;14) subgroup analysis.

---

## Section D — Multi-omics integration baselines

### D1. The MOFA / MOFA+ canon (you MUST cite these)

**Argelaguet et al. 2018 — MOFA** (PMID 29925568) — *Mol Syst Biol*
- DOI: [10.15252/msb.20178124](https://doi.org/10.15252/msb.20178124)
- **One-liner:** Original Multi-Omics Factor Analysis. Unsupervised latent factors. Applied to CLL (N=200) with somatic muts + RNA + DNA-methylation + drug response.

**Argelaguet et al. 2020 — MOFA+** (PMID 32393329) — *Genome Biol*
- DOI: [10.1186/s13059-020-02015-1](https://doi.org/10.1186/s13059-020-02015-1)
- **One-liner:** MOFA v2 with variational inference; multiple sample groups + modalities at single-cell scale; the *correct* method for "integrate scRNA + bulk + cytogenetics + treatment" in our setting. **Note from v12 audit:** the current ResistanceMap "MOFA wrapper" is NOT actual MOFA+ — switching to the official `mofapy2` would close this gap.

### D2. DeepProg & related deep multi-omics survival baselines

**Poirion et al. 2021 — DeepProg** (PMID 34261540) — *Genome Med*
- DOI: [10.1186/s13073-021-00930-x](https://doi.org/10.1186/s13073-021-00930-x)
- **One-liner:** Ensemble of deep autoencoders + ML for survival subtyping on multi-omics. Liver C-index 0.73–0.80; breast C-index 0.68–0.73. **Not benchmarked on MMRF in original paper**, but the obvious comparator for our cox-style head.
- **Code:** https://github.com/lanagarmire/DeepProg

### D3. Foundation models / scGPT / Geneformer / RegFormer

**Cui et al. 2024 — scGPT** (PMID 38409223) — *Nat Methods*
- DOI: [10.1038/s41592-024-02201-0](https://doi.org/10.1038/s41592-024-02201-0)
- **One-liner:** Generative pretrained transformer over 33M cells; fine-tuneable for cell-type annotation, batch integration, multi-omic integration, perturbation prediction, GRN inference.
- **Relevance:** Already in ResistanceMap v10 stack.

**Theodoris et al. 2023 — Geneformer** (PMID 37258680) — *Nature*
- DOI: [10.1038/s41586-023-06139-9](https://doi.org/10.1038/s41586-023-06139-9)
- **One-liner:** Attention-based context-aware model pretrained on ~30M scRNA transcriptomes; transfer-learning with limited data; identified cardiomyopathy therapeutic targets.
- **Relevance:** The Geneformer-vs-scGPT decision is genuine — both are reasonable; Geneformer is more parameter-efficient and the "rank-based encoding" is simpler than scGPT.

**Hu et al. 2026 — RegFormer** (PMID 42086551) — *Nat Commun*
- DOI: [10.1038/s41467-026-72198-x](https://doi.org/10.1038/s41467-026-72198-x)
- **One-liner:** Mamba SSM-based scFM with GRN priors. Authors claim it **outperforms scGPT, Geneformer, scFoundation, scBERT** on clustering, batch integration, cell-type annotation, GRN inference, *and drug response prediction across cancer cell lines*. Pretrained on 25M human cells / 45 tissues.
- **Relevance:** Newest (2026) foundation model with GRN priors — direct upgrade path from scGPT. Worth benchmarking against if we are choosing the foundation backbone.

### D4. From HuggingFace (preprints, NOT yet PubMed-indexed)

These are listed because the user asked for HF; they are NOT peer-reviewed:

- **TEDDY family (Chevalier et al. 2025)** — [arXiv 2503.03485](https://hf.co/papers/2503.03485) — transformer scFM with biological annotations, claims better disease-state ID than baselines.
- **scMamba (Oh et al. 2025)** — [arXiv 2502.19429](https://hf.co/papers/2502.19429) — snRNA-focused Mamba; relevant if we want a lighter foundation backbone.
- **InstructCell (Fang et al. 2025)** — [arXiv 2501.08187](https://hf.co/papers/2501.08187) — instruction-following multimodal scRNA agent; drug-sensitivity prediction included as a task.
- **CellForge (Tang et al. 2025)** — [arXiv 2508.02276](https://hf.co/papers/2508.02276) — agentic multi-omics model builder; perturbation prediction SOTA claim.
- **EVA (Bandasack et al. 2026)** — [arXiv 2602.10168](https://hf.co/papers/2602.10168) — multimodal foundation model for immunology that harmonizes transcriptomics + histology cross-species — closest "EVA-myeloma" template would be the right structure if we ever combine MMRF with bone-marrow histology.

---

## Section E — Disease-context grounding

### SOTA numbers in published MM drug-resistance / PFS prediction literature

| Metric                          | Reference                                                              | Performance                                  |
| ------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------- |
| PFS prediction across 5 cohorts | **mmSYGNAL** — Murie 2025 (PMID 40169765)                              | Beats cytogenetics + ISS + multi-gene panels |
| Joint PFS+OS+AE on TOURMALINE   | **Hussain 2024** transformer (PMID 39075240)                           | Beats ISS p<0.001 Bonferroni                 |
| Bortezomib-induction C-index    | **MM-5C** — Gargano 2025 (PMID 41339727)                               | Independent of cytogenetics + ISS            |
| 1/3/5-yr OS ROC AUC             | **BRG-5** — Ding 2026 (PMID 41493673)                                  | 0.730 / 0.734 / 0.775                        |
| 1-yr / 5-yr survival AUC        | Fang 2025 immune subtype (PMID 40325058)                               | 0.682 / 0.714                                |
| Risk stratification on 2 NDMM   | **GCRS GNN** — Sagar 2022 (PMID 36113255)                              | Best C-index vs baselines (numbers not extracted in abstract) |
| Liver/breast multi-omics survival | DeepProg 2021 (PMID 34261540)                                        | C-index 0.68–0.80                            |

**Bottom line for our headline number:** ResistanceMap PFS C-index needs to be reported against (a) mmSYGNAL, (b) Hussain 2024 transformer, (c) MM-5C, (d) MMRF-trained DeepProg ensemble. The published bar is C-index ~0.70 for genomic+clinical models on MMRF-style cohorts. v11.5 already ties mmSYGNAL; v13 needs to **beat** mmSYGNAL or establish a clear orthogonal capability (e.g., cell-state forecasting, F4-style falsification).

### SOTA for cell-state-transition forecasting in cancer
No peer-reviewed paper that we found reports a single accepted "AUROC for cell-state transition forecasting" in MM specifically. The closest available numbers:
- CellRank 2 (PMID 38871986): "consistently recovered terminal states + fate probabilities" across hematopoiesis + endoderm. Reports fate probabilities at single-cell resolution, not classification AUROC.
- Dynamo (PMID 35108499): predicts perturbation outcomes; benchmarked on hematopoiesis (PU.1/GATA1). Again, no single AUROC.
- LazyNet (PMID 41514902): r ≈ 0.67 on neuronal Perturb-seq for 1-hour transitions — but this is CRISPR-perturb, not disease-state.

**Implication for ResistanceMap:** there is no published SOTA number we are competing against on the trajectory side. We can SET the bar but must ground-truth carefully (e.g., serial-biopsy MMRF subset, or held-out GSE271107 cells).

---

## Section F — Coverage gaps (negative results — what we EXPECTED but did NOT find)

1. **No paper uses scANVI / scArches / Symphony / Azimuth on MM-specific scRNA-bulk transfer.** They exist as methods, but their MM application is unreported in MeSH-indexed literature.
2. **No paper applies scGPT or Geneformer to MMRF CoMMpass specifically.** The closest is RegFormer (PMID 42086551), which mentions drug-response improvement on CCLE cancer cell lines but not MMRF. This is a *publishable gap* for ResistanceMap v13: "first scGPT / Geneformer / RegFormer fine-tuning for MM PFS on MMRF."
3. **No paper integrates GSE124310 + GSE271107 + MMRF as three datasets with a unified ML model.** Yu et al. 2025 (PMID 41121130) come closest but use them sequentially (sc → pseudotime → marker → MMRF Cox) — not as joint inputs to one model.
4. **No published MOFA+ application on MMRF.** Despite MOFA+ being designed for exactly this kind of multi-modal multi-group problem.
5. **No MM trajectory paper uses CellRank 2 (only Monocle / Slingshot / scVelo-classic).** CellRank 2 was published mid-2024 — most MM scRNA papers pre-date or have not adopted it.
6. **No causal/counterfactual treatment-response model on MMRF.** Hussain 2024 (PMID 39075240) estimates individual treatment effects but uses correlational / non-Pearl-style causal framework.

---

## Section G — Recommended approach for ResistanceMap v13 (the "right answer")

### The hard answer from the literature
**No single paper has done what we want to do. The closest template is Yu et al. 2025 (PMID 41121130), but it is single-gene-marker and pseudotime-only. We have headroom to publish a method paper, not just a benchmark paper.**

### The recommended integration strategy (synthesized from the corpus)

Treat the three datasets as **complementary roles**, not co-equal inputs:

**Step 1 — GSE271107 as the trajectory/state prior (CellRank 2 + scVelo)**
- It is the larger and *more longitudinally informative* dataset (143k cells, HD→MGUS→SMM→MM).
- Use CellRank 2 (PMID 38871986, [DOI](https://doi.org/10.1038/s41592-024-02303-9)) protocol from Weiler 2026 (PMID 41611959, [DOI](https://doi.org/10.1038/s41596-025-01314-w)) to build a Markov-chain over plasma-cell states.
- This gives us a continuous "progression-pseudotime" axis grounded in pre-malignant→malignant biology.
- Output: a per-cell pseudotime + per-state gene-expression profile.

**Step 2 — GSE124310 as the subclone reference (BayesPrism deconvolution)**
- It is heavier in *malignant subclonal structure* (Ledergor 2018 explicitly identified 10/29 subclonal MMs).
- Use BayesPrism (PMID 35469013, [DOI](https://doi.org/10.1038/s43018-022-00356-3)) with GSE124310 plasma-cell subclones as the prior, deconvolve MMRF bulk RNA-seq.
- Output: per-MMRF-patient subclone-proportion vector.
- This is the BayesPrism>DWLS>CIBERSORTx ranking from Tran 2023 (PMID 37717006, [DOI](https://doi.org/10.1038/s41467-023-41385-5)).

**Step 3 — MMRF as the supervised target (multi-task transformer + MOFA+)**
- Use real MOFA+ (Argelaguet 2020, PMID 32393329, [DOI](https://doi.org/10.1186/s13059-020-02015-1)) via `mofapy2` to integrate {bulk RNA-seq, cytogenetics, subclone proportions from Step 2, pseudotime from Step 1} as 4 input modalities, with cytogenetic-risk subgroup as the *group* variable.
- Use Hussain 2024 (PMID 39075240, [DOI](https://doi.org/10.1038/s41746-024-01189-3)) transformer architecture for the joint PFS+OS+AE head — this is the strongest published precedent.
- Optionally fine-tune scGPT (PMID 38409223, [DOI](https://doi.org/10.1038/s41592-024-02201-0)) or RegFormer (PMID 42086551, [DOI](https://doi.org/10.1038/s41467-026-72198-x)) embeddings on MMRF pseudo-bulk per patient as an alternative to Step 3's MOFA-only embedding, and report both.

### Why NOT do the obvious "scGPT-only" approach
- scGPT/Geneformer/RegFormer were trained on healthy + heterogeneous-cancer reference cells. Their plasma-cell representations are **not curated for MM-progression structure**. They will hallucinate transitions.
- The biology of HD→MGUS→SMM→MM is *small relative to the model's prior* — without explicit trajectory grounding (Step 1), the foundation model will absorb the variance as noise.
- Use the foundation model as a **feature extractor**, not the trajectory engine.

### Honest statement about the current ResistanceMap stack
1. **The v12 audit conclusion stands.** The current "MOFA wrapper" is not actual MOFA+. Migrating to `mofapy2` is required for any literature-faithful claim.
2. **scGPT alone won't beat mmSYGNAL.** mmSYGNAL is cytogenetic-subtype-conditional, which our current TFM/HBayes head does not match.
3. **The Bellman-Harris + CellRank 2 combination IS novel** in the MM context — no published MM paper does this. That is publishable independent of beating mmSYGNAL on C-index.
4. **The falsification framework (F1–F10) is our defensible niche.** No paper in this corpus implements a comparable structural falsification protocol. Lead with that, not with C-index.

### Concrete next experiments
- **Experiment 1**: Replace the MOFA wrapper with `mofapy2` on {MMRF bulk, MMRF cytogenetics, BayesPrism-deconvolved GSE124310 subclone proportions, GSE271107-pseudotime} — 4-view, group=cytogenetic subtype. Report cross-validated C-index vs (a) v11.5, (b) mmSYGNAL re-implementation, (c) Hussain 2024 transformer re-implementation.
- **Experiment 2**: Fine-tune RegFormer (or scGPT) on MMRF pseudo-bulk; use as a 5th MOFA+ view; ablate.
- **Experiment 3**: Falsification gate F4 (trajectory consistency between MMRF projected onto GSE271107 pseudotime and observed PFS).

---

## Section H — Code & repository pointers (reusable)

Confirmed code links from peer-reviewed papers:
- **DeepProg** — https://github.com/lanagarmire/DeepProg (Poirion 2021)
- **CellRank** — https://github.com/theislab/cellrank (Weiler 2024)
- **scVelo** — https://github.com/theislab/scvelo (Bergen 2020)
- **Dynamo** — https://github.com/aristoteleo/dynamo-release (Qiu 2022)
- **MOFA / MOFA+** — https://biofam.github.io/MOFA2/ ; `mofapy2` (Argelaguet 2018/2020)
- **BayesPrism** — https://github.com/Danko-Lab/BayesPrism (Chu 2022)
- **scGPT** — https://github.com/bowang-lab/scGPT (Cui 2024)
- **Geneformer** — https://huggingface.co/ctheodoris/Geneformer (Theodoris 2023)
- **MMRFBiolinks** — Bioconductor (Settino & Cannataro 2022, PMID 34902136, [DOI](https://doi.org/10.1007/978-1-0716-1839-4_19))

HuggingFace hub pointers:
- Geneformer: https://huggingface.co/ctheodoris/Geneformer
- RegFormer (per paper, likely under BGI-Research org — not verified): see PMID 42086551
- scGPT checkpoints: distributed via `bowang-lab/scGPT` GitHub; some HF mirrors exist (unverified)

---

## Final ranked picks for "head-to-head" comparators in the v13 paper

1. **mmSYGNAL — Murie 2025 (PMID 40169765, [DOI](https://doi.org/10.1038/s41416-025-02987-6))** — primary head-to-head; cytogenetic-subtype-aware MM risk + drug-response; biggest external validation (N=1,367).
2. **Hussain 2024 transformer (PMID 39075240, [DOI](https://doi.org/10.1038/s41746-024-01189-3))** — secondary head-to-head; joint PFS+OS+AE + biomarker forecasting; best deep-learning MM precedent.
3. **MM-5C — Gargano 2025 (PMID 41339727, [DOI](https://doi.org/10.1038/s41598-025-30527-y))** — recent bortezomib-specific signature; lightweight comparator.
4. **Yu et al. 2025 (PMID 41121130, [DOI](https://doi.org/10.1186/s12967-025-07098-7))** — direct methodological template (GSE124310+GSE271107→MMRF); cite as motivation.
5. **DeepProg — Poirion 2021 (PMID 34261540, [DOI](https://doi.org/10.1186/s13073-021-00930-x))** — generic multi-omics-survival baseline; re-implement on MMRF for an apples-to-apples Cox C-index.

If forced to pick 3 (not 5), drop #3 and #5.
