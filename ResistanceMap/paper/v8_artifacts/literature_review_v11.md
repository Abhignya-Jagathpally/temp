# Literature Review — Comparators for ResistanceMap v11

- **Date of search:** 2026-05-03
- **Reviewer:** automated literature-curator agent (biomedical informatics persona)
- **Project under review:** ResistanceMap v11 — multi-omic foundation model for epigenetic drug-resistance trajectories in hematologic malignancies; Neural ODE on Waddington landscape + PPI propagation; target regime N_paired ≤ 50.

> NOTE on filename: the user requested overwrite of `literature_review.md`. The Write tool refused that path because the existing file had not been Read first, and no Read tool was available in this session. This file is therefore written as `literature_review_v11.md` alongside it. The user should `mv literature_review_v11.md literature_review.md` (or diff first) to complete the intended overwrite.

## CRITICAL CAVEAT — search-tool access status

In this session **every PubMed, bioRxiv, and HuggingFace search/get MCP call was DENIED at the permission gate** (10/10 calls returned "Permission to use … has been denied"). I therefore could not run the live identifier-verification protocol the user mandated.

What I did instead:
- Cited only papers whose DOI/PMID I can recall with high confidence from training data, and **labeled every line with one of three verification tiers**:
  - **[V-K]** — Verified-from-knowledge: high-confidence DOI/PMID/title/year/authors from training, well-known landmark.
  - **[V-K?]** — Verified-from-knowledge but at least one field (year/journal/exact author order) I am NOT certain about.
  - **[UNVERIFIED]** — I believe a paper of roughly this description exists but cannot supply a verifiable identifier; included only because user explicitly named it. Treat as a search target, not a citation.
- Where the user named a candidate I cannot vouch for at all (e.g. "iMLGAM 2025"), I write **"No identifier I can verify"** rather than fabricate.

**Action item for the user:** before any of these are cited in v11, re-run identifier verification through PubMed/Crossref. Several of my recall-only DOIs are likely correct in spirit but may be off by a digit or arXiv-vs-journal version.

---

## Sweep 1 — Neural-ODE-on-Waddington-landscape architectures

| # | Paper | Year | DOI / PMID | Verification | Dataset | Metric (reported) | Code | One-line relevance to v11 |
|---|---|---|---|---|---|---|---|---|
| 1.1 | Tong, Huang, Wolf, van Dijk, Krishnaswamy. *TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics.* ICML 2020 | 2020 | arXiv:2002.04461 | [V-K] | scRNA-seq embryoid body | continuous-NF MSE on held-out timepoints | github.com/KrishnaswamyLab/TrajectoryNet | First continuous normalising flow for population-level cell dynamics; direct conceptual ancestor of v11's drift-on-potential. |
| 1.2 | Bunne, Stark, Gut, del Castillo, Lehmann, Pelkmans, Krause, Rätsch. *Learning single-cell perturbation responses using neural optimal transport (CellOT).* Nat Methods 2023 | 2023 | doi:10.1038/s41592-023-01969-x ; PMID:37592024 | [V-K] | sci-Plex 4-cell-line drug perturbation, lupus IFN-β | r² perturbation-response prediction | github.com/bunnech/cellot | Static neural OT — no continuous time, no potential; v11 must beat its drug-effect prediction at matched N. |
| 1.3 | Lange, Bergen, Klein, Setty, Reuter, Bakhti, Lickert, Ansari, Schniering, Schiller, Pe'er, Theis. *CellRank 2: unified fate mapping in multiview single-cell data.* Nat Methods 2024 | 2024 | doi:10.1038/s41592-024-02303-9 | [V-K] | scRNA-seq atlases; pancreas, hematopoiesis | macrostate identification, fate-probability AUC | github.com/theislab/cellrank | Markov-chain fate maps from kernels (RNA velocity, pseudotime, real time). Baseline v11 must outperform on **paired-trajectory MM** because CellRank 2 was built for atlas-scale, not N≤50. |
| 1.4 | Atanackovic, Tong, Wang, Lee, Bengio, Hartford. *Dyngfn: Bayesian dynamic causal discovery using generative flow networks.* UAI/NeurIPS 2023 | 2023 | arXiv:2302.04178 | [V-K?] | synthetic + scRNA-seq | SHD vs. ground-truth DAG | github.com/lazaratan/dyn-gfn | Dynamic-DAG learning over cell state — comparator for v11's PPI-conditioned drift, but operates on small graphs. |
| 1.5 | Schiebinger et al. *Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming (Waddington-OT).* Cell 2019 | 2019 | doi:10.1016/j.cell.2019.01.006 ; PMID:30712874 | [V-K] | reprogramming time course | OT-coupling validation vs. lineage tracing | github.com/broadinstitute/wot | The original "OT on Waddington" paper. v11's potential-based formulation must position against it. |
| 1.6 | Hashimoto, Gifford, Jaakkola. *Learning Population-Level Diffusions with Generative RNNs.* ICML 2016 | 2016 | proceedings.mlr.press/v48/hashimoto16 | [V-K] | yeast cell cycle | KL of fitted SDE vs. observed populations | (academic code) | Earliest learned-drift-field on cell populations; the Waddington-NeuralODE family literally starts here. |
| 1.7 | Sha, Qiu, Zhou, Nie. *Reconstructing growth and dynamic trajectories from single-cell transcriptomics data (TIGON).* Nat Mach Intell 2024 | 2024 | doi:10.1038/s42256-024-00775-0 | [V-K?] | EB, pancreas, MEF reprogramming | Wasserstein distance between predicted/true distributions | github.com/yutongo/TIGON | 2024 Neural-ODE-on-population with growth term — closest method-of-method to v11 drift+source. |
| 1.8 | Maddu, Sturm, Müller, Sbalzarini. *Stability-preserving learning of Waddington landscapes from single-cell transcriptomics.* | 2024 | bioRxiv:2024.04.09.588498 | [V-K?] | scRNA hematopoiesis | landscape RMSE vs. analytical reference | (bioRxiv supplement) | Direct comparator: explicitly learns the potential U(x). |
| 1.9 | Saelens, Cannoodt, Todorov, Saeys. *A comparison of single-cell trajectory inference methods.* Nat Biotechnol 2019 | 2019 | doi:10.1038/s41587-019-0071-9 ; PMID:30936559 | [V-K] | 110 datasets | accuracy across 45 trajectory methods | dynverse.org | Benchmark anchor; v11 trajectory-quality must be reported on at least one Saelens dataset. |

**Coverage gap note:** I could not run a 2025–2026 sweep, so any drift-field-on-potential paper younger than mid-2024 is missing.

---

## Sweep 2 — Multi-omic foundation models applicable to bulk MM/AML

| # | Paper | Year | DOI / arXiv | Verification | N_pretrain | Downstream metric | HF / GitHub | Relevance |
|---|---|---|---|---|---|---|---|---|
| 2.1 | Cui, Wang, Maan, Pang, Zhang, Wang. *scGPT: toward building a foundation model for single-cell multi-omics using generative AI.* Nat Methods 2024 | 2024 | doi:10.1038/s41592-024-02201-0 | [V-K] | 33M cells | cell-type AUROC, perturbation Δ-r² | github.com/bowang-lab/scGPT ; HF: bowang-lab/scGPT | Top-tier candidate for v11's bulk-RNA encoder warm-start. v10 already chose scGPT. |
| 2.2 | Theodoris et al. *Transfer learning enables predictions in network biology (Geneformer).* Nature 2023 | 2023 | doi:10.1038/s41586-023-06139-9 ; PMID:37258680 | [V-K] | 30M cells (Genecorpus-30M) | dosage-sensitive gene AUROC, in-silico perturbation | huggingface.co/ctheodoris/Geneformer | Rank-encoded gene-token foundation model. Direct competitor encoder. |
| 2.3 | Hao, Gong, Yang, Chen, Yan, Yang, Chen. *Large-scale foundation model on single-cell transcriptomics (scFoundation).* Nat Methods 2024 | 2024 | doi:10.1038/s41592-024-02305-7 | [V-K?] | 50M cells | drug-response Pearson on CCLE; perturbation r² | github.com/biomap-research/scFoundation | Critical comparator: explicitly benchmarks **bulk-RNA drug response on CCLE**. |
| 2.4 | Rosen, Brbic, Roohani, Swanson, Li, Leskovec. *UCE — Universal Cell Embeddings: A foundation model for cell biology.* | 2023 | bioRxiv:2023.11.28.568918 | [V-K?] | 36M cells, 1000+ tissues | zero-shot cell-type kNN F1 | github.com/snap-stanford/UCE | Useful zero-shot baseline encoder. |
| 2.5 | Lin et al. *Evolutionary-scale prediction of atomic-level protein structure (ESM-2 / ESMFold).* Science 2023 | 2023 | doi:10.1126/science.ade2574 ; PMID:36927031 | [V-K] | UniRef ~250M | LDDT ≈ AlphaFold2 on CASP14-class | huggingface.co/facebook/esm2_t33_650M_UR50D | Protein-sequence FM used in v10's PPI-propagation features. |
| 2.6 | Elnaggar et al. *ProtTrans/ProtT5.* IEEE TPAMI 2022 | 2022 | doi:10.1109/TPAMI.2021.3095381 ; PMID:34232869 | [V-K?] | UniRef50/100 | per-residue + per-protein benchmarks | github.com/agemagician/ProtTrans | Alternative to ESM-2 for protein arm. |
| 2.7 | Tabula Sapiens Consortium (Quake et al.). *The Tabula Sapiens.* Science 2022 | 2022 | doi:10.1126/science.abl4896 ; PMID:35549404 | [V-K] | ~500K cells, 24 tissues | reference resource | tabula-sapiens.sf.czbiohub.org | Reference atlas. |
| 2.8 | Megill et al. (CZI). *CELLxGENE Discover.* | 2021 | bioRxiv:2021.04.05.438318 | [V-K?] | infrastructure paper | n/a | cellxgene.cziscience.com | Harmonised data layer most FMs pretrain on. |
| 2.9 | Gong, Hao, Wang et al. *xTrimoGene / xTrimoPGLM* (BGI/BioMap) | 2023 | bioRxiv 2023.03.24.534055 | [V-K?] | tens of millions of cells | scRNA tasks | (paper code) | Comparator FM from same group as scFoundation. |
| 2.10 | **No verified identifier** for a 2024–2026 paper that *explicitly fine-tunes* a single-cell FM for **bulk RNA + drug response on MM specifically**. | — | — | — | — | — | — | Negative result: v11's MM-specific FM-fine-tune is novel as far as my recall goes. |

---

## Sweep 3 — PPI graph-neural-network propagation for drug-target / drug-response

| # | Paper | Year | DOI / PMID | Verification | Graph | Metric | Code | Relevance |
|---|---|---|---|---|---|---|---|---|
| 3.1 | Zitnik, Agrawal, Leskovec. *Decagon — Modeling polypharmacy side effects with GCNs.* Bioinformatics 2018 | 2018 | doi:10.1093/bioinformatics/bty294 ; PMID:29949996 | [V-K] | drug-drug-protein heterograph | AUROC ~0.87 on side-effect link prediction | github.com/marinkaz/decagon | Heterograph baseline. |
| 3.2 | Veličković et al. *Graph Attention Networks (GAT).* ICLR 2018 | 2018 | arXiv:1710.10903 | [V-K] | generic | node-classification AUROC | github.com/PetarV-/GAT | Reference GAT. |
| 3.3 | Brody, Alon, Yahav. *How Attentive are Graph Attention Networks? (GATv2).* ICLR 2022 | 2022 | arXiv:2105.14491 | [V-K] | OGB et al. | improved over GAT | github.com/tech-srl/how_attentive_are_gats | The "GAT-2" referenced in v11 plan. |
| 3.4 | Cowen, Ideker, Raphael, Sharan. *Network propagation: a universal amplifier of genetic associations.* Nat Rev Genet 2017 | 2017 | doi:10.1038/nrg.2017.38 ; PMID:28607512 | [V-K] | review | n/a | n/a | Theoretical anchor for RWR / heat-kernel propagation in v10/v11. |
| 3.5 | Zeng et al. *deepDTnet — Target identification among known drugs by deep learning from heterogeneous networks.* Chem Sci 2020 | 2020 | doi:10.1039/c9sc04336e ; PMID:34122816 | [V-K?] | 15-network heterograph | AUROC drug-target | github.com/ChengF-Lab/deepDTnet | Heterogeneous-GNN drug-target baseline. |
| 3.6 | Huang, Fu, Glass, Zitnik. *Therapeutics Data Commons (TDC) + DeepPurpose.* Nat Chem Biol 2022 | 2022 | doi:10.1038/s41589-022-01131-2 ; PMID:36192599 | [V-K?] | TDC benchmarks | DTI AUROC, ADMET | github.com/mims-harvard/TDC | Standard benchmark suite. |
| 3.7 | Cantini et al. *Benchmarking joint multi-omics dimensionality reduction approaches for the study of cancer.* Nat Commun 2021 | 2021 | doi:10.1038/s41467-020-20430-7 ; PMID:33402734 | [V-K] | TCGA pan-cancer | survival c-index, clustering ARI | github.com/cantinilab/momix-notebook | Multi-omics integration benchmark. |
| 3.8 | Costello, Heiser et al. *NCI-DREAM drug sensitivity prediction.* Nat Biotechnol 2014 | 2014 | doi:10.1038/nbt.2877 ; PMID:24880487 | [V-K] | NCI-DREAM cell-line panel | weighted-rank Pearson | synapse.org/DREAM7 | Historical baseline. |
| 3.9 | Menden, Wang, Mason et al. *NCI-DREAM drug-combination challenge.* Nat Commun 2019 | 2019 | doi:10.1038/s41467-019-09186-x ; PMID:30894541 | [V-K] | AstraZeneca-Sanger 11k combos | Pearson, tie-broken score | (challenge-specific) | Drug-combination DREAM. |
| 3.10 | Yu, Ma, Yu et al. *DCell — Translation of genotype to phenotype by a hierarchy of cell systems.* Nat Methods 2018 | 2018 | doi:10.1038/nmeth.4627 ; PMID:29377497 | [V-K] | yeast → human | growth phenotype prediction | github.com/idekerlab/DCell | Pathway-structured deep nets — direct conceptual cousin of v11's PPI-conditioned propagation. |

---

## Sweep 4 — Conformal prediction in oncology / MM

| # | Paper | Year | DOI / arXiv | Verification | Setting | Metric | Code | Relevance |
|---|---|---|---|---|---|---|---|---|
| 4.1 | Barber, Candès, Ramdas, Tibshirani. *Predictive inference with the jackknife+.* Ann Stat 2021 | 2021 | doi:10.1214/20-AOS1965 ; arXiv:1905.02928 | [V-K] | regression | finite-sample marginal coverage | github.com/ryantibs/conformal | Theoretical anchor for v10 sprint 5 Mondrian-JK+. |
| 4.2 | Vovk, Petej, Nouretdinov, Manokhin, Gammerman. *Mondrian conformal predictive distributions.* COPA 2018 | 2018+ | (book + papers) | [V-K?] | classification & regression | conditional coverage | n/a | Origin of Mondrian conformal — exact source DOI not recalled. |
| 4.3 | Romano, Patterson, Candès. *Conformalized quantile regression (CQR).* NeurIPS 2019 | 2019 | arXiv:1905.03222 | [V-K] | regression | adaptive interval width, coverage | github.com/yromano/cqr | Adaptive-width conformal — v11 should test for time-to-event-like residuals. |
| 4.4 | Candès, Lei, Ren. *Conformalized survival analysis.* J R Stat Soc B 2023 | 2023 | doi:10.1093/jrsssb/qkad043 | [V-K] | right-censored survival | covered survival functions | github.com/zhimeir/cfsurvival | Canonical baseline for any MM survival CI claim. |
| 4.5 | Angelopoulos, Bates. *A gentle introduction to conformal prediction and distribution-free uncertainty quantification.* Found Trends ML 2023 | 2023 | arXiv:2107.07511 | [V-K] | tutorial | n/a | github.com/aangelopoulos/conformal-prediction | Reference tutorial. |
| 4.6 | **No identifier I can verify** for a 2024–2026 conformal-MM-specific peer-reviewed paper. v10 sprint 5 (5/5 strata within ±3%) appears to be ahead of public literature. | — | — | — | — | — | — | Gap = positioning opportunity. |

---

## Sweep 5 — MM clinical models v11 must compete with

| # | Paper | Year | DOI / PMID | Verification | Cohort N | Metric | Code | Relevance |
|---|---|---|---|---|---|---|---|---|
| 5.1 | Palumbo et al. *Revised International Staging System for Multiple Myeloma (R-ISS).* J Clin Oncol 2015 | 2015 | doi:10.1200/JCO.2015.61.2267 ; PMID:26240224 | [V-K] | 4445 NDMM | 5-yr OS, PFS by stage | n/a | Reference clinical comparator. |
| 5.2 | D'Agostino, Cairns, Lahuerta et al. *Second Revision of the International Staging System (R2-ISS) for Overall Survival in Multiple Myeloma — EMN report.* J Clin Oncol 2022 | 2022 | doi:10.1200/JCO.21.02614 ; PMID:35580297 | [V-K] | 10,843 NDMM | OS HR by R2-ISS stage | n/a | Updated MM clinical staging — adds 1q+. v11's primary clinical comparator. |
| 5.3 | Shaughnessy, Zhan, Burington et al. *UAMS GEP70: validated gene-expression model of high-risk MM.* Blood 2007 | 2007 | doi:10.1182/blood-2006-07-038430 ; PMID:17105813 | [V-K] | TT2/TT3 trial cohorts | 3-yr EFS, OS | n/a | First molecular MM risk score. |
| 5.4 | Decaux, Lodé, Magrangeas et al. *IFM-15 — Prediction of survival in MM based on gene expression profiles.* J Clin Oncol 2008 | 2008 | doi:10.1200/JCO.2008.16.4178 ; PMID:18768866 | [V-K?] | IFM-99-02/04 | OS HR | n/a | Companion GEP signature with GEP70. |
| 5.5 | Kuiper, Broyl, de Knegt et al. *EMC92 / SKY92: gene expression signature for high-risk MM.* Leukemia 2012 | 2012 | doi:10.1038/leu.2012.127 ; PMID:22552008 | [V-K] | HOVON-65/GMMG-HD4 | OS HR ~3 | n/a (commercial MMprofiler) | Validated GEP comparator. |
| 5.6 | Kuiper, van Duin, van Vliet et al. *Prediction of high- and low-risk MM based on gene expression and ISS.* Blood 2015 | 2015 | doi:10.1182/blood-2015-05-644039 ; PMID:26240221 | [V-K?] | independent cohorts | combined score | n/a | Combined molecular+clinical comparator. |
| 5.7 | Walker, Mavrommatis, Wardell et al. *A high-risk, double-hit, group of newly diagnosed myeloma identified by genomic analysis.* Leukemia 2019 | 2019 | doi:10.1038/s41375-018-0196-8 ; PMID:30214045 | [V-K] | 1273 NDMM CoMMpass | OS HR | n/a | Genomic "double-hit" risk in MMRF CoMMpass. |
| 5.8 | Maura, Bolli, Angelopoulos et al. *Genomic landscape and chronological reconstruction of driver events in multiple myeloma.* Nat Commun 2019 | 2019 | doi:10.1038/s41467-019-11680-1 ; PMID:31488816 | [V-K] | 804 MM WGS | mutational timing | n/a | Source of MM driver/timing priors. |
| 5.9 | **iMLGAM (2025)** — user named. | 2025? | — | [UNVERIFIED] | — | — | — | I cannot supply DOI/PMID without an MCP search. If user has the citation, please paste. |
| 5.10 | Recent (2023–2025) multi-omic MM relapse-time prognostic models | 2023+ | not recalled with confidence | [UNVERIFIED] | — | — | — | Cannot verifiably cite a *multi-omic* (RNA + proteomic + epigenetic) MM relapse-time model without an MCP search. **Gap.** |
| 5.11 | MMRF CoMMpass IA-series data releases (data resource) | 2018–2024 | research.themmrf.org | [V-K] | 1143 NDMM longitudinal | n/a | research.themmrf.org/rp | The dataset itself. |

---

## Disease-context grounding (best-of-recall, no MCP verification)

- **MM PFS prediction (R2-ISS-class)** — published c-index typically **0.62–0.68** on independent cohorts; SKY92 ~0.65–0.70 in HOVON cohorts. Recall-only; **must be re-verified**.
- **MM OS prediction** — c-index in the **0.66–0.74** range for combined molecular+clinical models on NDMM cohorts.
- **Cell-state-transition forecasting in cancer (drug perturbation)** — CellOT reports r² ≈ 0.4–0.7 across sci-Plex cell lines; CellRank 2 reports macrostate-fate AUROC > 0.85 on hematopoiesis but is *not* a drug-trajectory benchmark. **No public SOTA for "MM patient drug-trajectory at N_paired ≤ 50"** — that regime is essentially v11's claim region.
- **Drug-response prediction on CCLE/GDSC** — DeepCDR/scFoundation/PaccMann report Pearson **0.85–0.93** at the cell-line × drug level (likely leakage-prone splits); under strict drug-blind splits the numbers drop to **0.45–0.6**. v11 must use the strict split.

---

## Coverage gaps (negative results matter)

1. **No verified peer-reviewed paper that fine-tunes a single-cell foundation model on bulk RNA + drug response specifically in multiple myeloma.** scFoundation does pan-cancer; nothing MM-targeted that I can name with confidence.
2. **No verified MM-specific conformal-prediction paper.** v10 sprint 5 (Mondrian-JK+ 5/5 strata ±3%) appears to be ahead.
3. **No verified Neural-ODE-on-Waddington-landscape paper for MM resistance trajectories.** TIGON/TrajectoryNet/Maddu are general; CellOT is per-perturbation snapshot, not patient trajectory.
4. **No verified head-to-head benchmark of CellRank 2, scFoundation, and SKY92 on the SAME MM cohort with the SAME endpoint.** v11 producing one would be a contribution.
5. **iMLGAM (2025)** named by user — unverified; should be confirmed or dropped.

---

## Recommendation — fit-for-purpose comparator shortlist for v11 (≤ 8 papers)

Sorted by directness of overlap with v11's claim region (per-patient MM resistance trajectories at N_paired ≤ 50). Same dataset (MMRF CoMMpass) and same endpoint (paired-trajectory PFS / drug-response) preferred.

| Rank | Paper | Why this one | Same dataset? | Same metric? | Effort to run |
|---|---|---|---|---|---|
| 1 | **R2-ISS — D'Agostino 2022** (5.2) | Clinical SOTA v11 must beat or augment on MMRF. Trivial to compute. | Yes (CoMMpass-compatible) | OS/PFS HR, c-index | 1 day |
| 2 | **SKY92 / EMC92 — Kuiper 2012/2015** (5.5, 5.6) | Molecular SOTA for MM risk; can be re-implemented from publication. | Yes (RNA in CoMMpass) | c-index | 1–2 days |
| 3 | **scFoundation — Hao 2024** (2.3) | Closest FM-fine-tune comparator on bulk-RNA drug-response. Code public. | Cross-domain (CCLE) → re-fit on CoMMpass | drug-response Pearson, MM PFS c-index | 1 week |
| 4 | **scGPT — Cui 2024** (2.1) | Alternative FM encoder; v10 already chose scGPT, so head-to-head with scFoundation and Geneformer is mandatory. | Re-fit | matched | 1 week |
| 5 | **CellOT — Bunne 2023** (1.2) | Closest neural-OT comparator for drug perturbation. Tests v11's "trajectory beats snapshot" claim. | sci-Plex pretrain; re-fit on CoMMpass paired | r² perturbation | 1 week |
| 6 | **CellRank 2 — Lange 2024** (1.3) | Markov-fate baseline. Tests v11's "ODE beats Markov" claim. | Re-fit on CoMMpass paired | fate AUROC, OS c-index | 3–5 days |
| 7 | **MOFA+ — Argelaguet 2018/2020** (RM v6 incumbent) | RM's own incumbent; per memory `project_resistancemap_floor_evidence.md`, v6 ties MOFA+Ridge — v11 must beat this floor decisively. | Yes (CoMMpass) | matched | already implemented |
| 8 | **Walker double-hit — Walker 2019** (5.7) | Genomic risk on the same CoMMpass dataset. v11's gain over genomic-only must be visible here. | Yes (CoMMpass) | OS HR | 1 day |

**Final positioning principle for v11:** the *claim region* is per-patient paired-trajectory prediction at N≤50. None of the 8 comparators above were designed for that regime. Comparators 1–2 are clinical (no trajectory, no per-patient drug); 3–4 are pan-cancer FMs (no MM trajectory); 5–6 are trajectory methods (no MM, no patient-level drug-PFS); 7 is the multi-omics floor; 8 is genomic-only. **v11's contribution is the intersection none of them cover** — provided the head-to-heads are run honestly on matched splits.

---

## Bibliographic-verification TODO (must run before manuscript submission)

- [ ] Re-verify every DOI marked **[V-K?]** via Crossref/PubMed.
- [ ] Confirm or drop: **iMLGAM (2025)**, Maddu Waddington-landscape 2024 bioRxiv id, TIGON Nat Mach Intell 2024 DOI, Atanackovic Dyngfn UAI/NeurIPS venue, Vovk Mondrian conformal source paper, deepDTnet PMID, Cantini momix DOI, ProtT5 PMID.
- [ ] Add 2025–2026 sweep — none of my recall is reliable for that window. Re-run PubMed/bioRxiv with an MCP gate that has search permissions enabled.
- [ ] Verify R2-ISS PMID 35580297 (digit confidence ~90%).
