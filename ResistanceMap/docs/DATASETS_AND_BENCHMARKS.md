# ResistanceMap — Dataset & Benchmark Landscape

_Strategic map of (a) the canonical datasets the field's similar models train and benchmark on, (b) ResistanceMap's current vs. target data inventory, and (c) a priority-ordered roadmap to bring ResistanceMap up to par with the SOTA._

> **Audit anchor:** as of 2026-05-02, ResistanceMap trains on **622 CCLE cell lines × 11 GDSC drugs (3,635 observed IC50 cells)**. Two scRNA datasets (171,544 cells total, with `disease_stage` labels HD→MGUS→SMM→MM) are downloaded but never ingested. CTRPv2, DepMap CRISPR, HMCL panel, PRISM, and MMRF CoMMpass are referenced but not loaded.

## 1 — Dataset cards

### 1.1 Drug-response screens

#### GDSC1 + GDSC2 (Genomics of Drug Sensitivity in Cancer)

- **Provider:** Sanger Institute / Cell Model Passports
- **Scale:** ~1,000 cell lines × 624 drugs (GDSC1+2 combined), IC50 + AUC + Z-score
- **Format:** xlsx + csv from `cog.sanger.ac.uk`
- **Access:** Public, free (no auth)
- **ResistanceMap status:** ✅ ingested (11 of 624 drugs subset to MM-relevant)
- **Used by:** DeepCDR, PaccMann, MOLI, DrugCell, MCA, SeNMo, MOFA+ (drug-response application)
- **Notes:** the de facto baseline screen for cell-line drug response. Always cited.

#### CTRPv2 (Cancer Therapeutics Response Portal v2)

- **Provider:** Broad Institute (CTD2 Network)
- **Scale:** ~860 cell lines × 481 small molecules, AUC + IC50
- **Format:** csv from `ctd2-data.nci.nih.gov`
- **Access:** Public, free
- **ResistanceMap status:** ⚠️ **CONFIGURED but not on disk** — `data/raw/ctrp/` is empty
- **Used by:** DeepCDR, MOLI, TUGDA, PaccMann (transfer learning)
- **Why add:** CTRPv2 has different drug coverage from GDSC; many studies report on the GDSC ∪ CTRPv2 union for full compound coverage. Critically, **most papers cite CTRPv2 as the validation set when GDSC is the training set** — this is the canonical generalization benchmark.

#### PRISM Repurposing Hub

- **Provider:** Broad Institute (Corsello et al.)
- **Scale:** ~4,500 drugs × 500+ cell lines, viability via DNA-barcoded pools
- **Format:** csv via `depmap.org/portal/repurposing`
- **Access:** Public, free
- **ResistanceMap status:** ❌ not configured, not downloaded
- **Used by:** newer drug-repositioning models (post-2020)
- **Why add:** order-of-magnitude more drugs than GDSC/CTRPv2; enables drug-feature pre-training (drug encoders), which is increasingly the SOTA pattern.

#### NCI-60 / CellMiner

- **Provider:** NCI (Reinhold et al.)
- **Scale:** 60 cell lines × ~50,000 compounds, GI50/TGI/LC50
- **Format:** csv via `discover.nci.nih.gov/cellminer`
- **Access:** Public, free
- **ResistanceMap status:** ❌ not configured
- **Used by:** classical pharmacogenomics; still cited but smaller scale than GDSC/CTRPv2
- **Why add:** broadest compound space at the cost of fewest cell lines.

### 1.2 Cell-line multi-omics (features)

#### CCLE / DepMap (Cancer Cell Line Encyclopedia)

- **Provider:** Broad Institute / DepMap
- **Scale:** ~1,400 cell lines, paired RNA-seq + WES + WGS + proteomics + reverse phase protein array
- **Format:** csv from `depmap.org/portal/download`
- **Access:** Public
- **ResistanceMap status:** ✅ proteomics (886 of ~1,400 lines, 19,177 proteins) and epigenomics (886 × 42 features) ingested
- **Used by:** every drug-response model (DeepCDR, DrugCell, PaccMann, MOLI, MCA, SeNMo, etc.)
- **Why expand:** ~500 advertised cell lines are missing — many are MM/leukemia/lymphoma relevant. CCLE 2024 release adds metabolomics + spatial proteomics.

#### DepMap Achilles CRISPR Essentiality

- **Provider:** Broad Institute
- **Scale:** ~1,000 cell lines × ~18,000 genes, CRISPR knockout effects (Chronos / DEMETER scores)
- **Format:** csv via `depmap.org/portal/download`
- **Access:** Public
- **ResistanceMap status:** ❌ not configured (`data/raw/depmap/` only has proteomics dups)
- **Used by:** DrugCell, all causal-grounded drug-response models
- **Why add:** **closes the v8 causal-audit gap.** Without perturbation data, ResistanceMap's pathway predictions are associative only. CRISPR essentiality grounds them.

#### CCLE proteomics (Nusinow et al. 2020)

- **Provider:** Broad Institute / Gygi Lab
- **Scale:** 375 cell lines × ~12,000 proteins, TMT mass spec
- **ResistanceMap status:** ✅ ingested (this is the source of the 19,177 protein vector)

#### HMCL Keats panel

- **Provider:** Mayo Clinic / Keats Lab
- **Scale:** 65+ MM-specific cell lines, paired RNA-seq + WES + WGS, sometimes scRNA
- **Format:** GEO + Keats lab portal
- **Access:** Public
- **ResistanceMap status:** ❌ not configured
- **Used by:** any MM-specific ML paper (e.g., the v8 `proteomics-pathway-validator`'s ground truth would use this)
- **Why add:** **MM specialization.** The CCLE MM panel is small (~10 cell lines) and biased toward end-stage disease. HMCL is the field's canonical MM cell-line reference.

### 1.3 Single-cell transcriptomics

#### GSE124310 — MM patient bone marrow scRNA-seq

- **Provider:** GEO (Ledergor et al. Nat Med 2018)
- **Scale:** 27,796 cells across MM patients (newly diagnosed, relapsed, healthy)
- **Format:** h5ad (190 MB) — already on disk at `data/raw/gse124310.h5ad`
- **Obs columns:** `sample`, `condition` (NDMM / RRMM / HD)
- **ResistanceMap status:** ⚠️ **ON DISK BUT IGNORED** by the pipeline (`harmonize_omics` accepts `scrna_data` arg but doesn't merge into the dataset)
- **Used by:** any MM single-cell trajectory paper
- **Why wire it:** immediate ~30k-cell scope expansion with no download required.

#### GSE271107 — HD→MGUS→SMM→MM progression scRNA-seq

- **Provider:** GEO (recent submission, 2024)
- **Scale:** 143,748 cells across 4 disease stages
- **Format:** h5ad (2.1 GB) — already on disk at `data/raw/gse271107.h5ad`
- **Obs columns:** `sample`, `disease_stage` (HD/MGUS/SMM/MM), `geo_accession`, `batch`
- **ResistanceMap status:** ⚠️ **ON DISK BUT IGNORED** (same as GSE124310)
- **Used by:** any longitudinal MM model (TrajectoryNet, scVelo, dynamo can all consume this)
- **Why wire it:** **this dataset alone provides the longitudinal labels v8 said were missing.** With `disease_stage` as the time axis, ResistanceMap's "predict before it happens" claim becomes verifiable for the first time.

#### GSE193531 (Lohr et al.), GSE161801 (Tirier et al.)

- **Provider:** GEO
- **Scale:** GSE193531: ~50k cells; GSE161801: ~150k cells
- **Access:** Public
- **ResistanceMap status:** ❌ not configured
- **Why consider:** broader coverage of MM single-cell; useful for cross-cohort validation of trajectory inference.

### 1.4 Patient cohorts (clinical)

#### MMRF CoMMpass (IA22+)

- **Provider:** MMRF / dbGaP / GDC Data Portal
- **Scale:** 1,143 newly-diagnosed MM patients, longitudinal: WGS + WES + RNA-seq + clinical (PFS/OS/IMWG response) at multiple time points
- **Format:** VCF + TSV + bam via `gdc-client`
- **Access:** **IRB-controlled** (dbGaP study phs000748)
- **ResistanceMap status:** ❌ not loaded; `resistancemap/data/mmrf_loader.py` exists but is unwired
- **Used by:** every MM clinical-prediction model (PMID 41814396, 41909977, 41762247)
- **Why add:** **the only path to a real patient-level claim.** No published GDSC IC50 → IMWG response calibration exists; MMRF closes the cohort gap.

#### CoMMpass IA22 RNA-seq subset (controlled)

- **Provider:** GDC
- **Scale:** ~700 patients with paired baseline + relapse RNA-seq
- **Why useful:** specifically usable for the "predict resistance state at time t" claim.

### 1.5 Network / pathway

#### STRING PPI v12

- **Provider:** STRING-DB
- **Scale:** 19,566 human proteins, ~6M edges (full); ~930k at confidence ≥ 0.7
- **ResistanceMap status:** ✅ ingested (473,860 edges at confidence 0.7 cutoff)

#### Reactome / KEGG / MSigDB

- **Provider:** Reactome consortium / KEGG / Broad MSigDB
- **Scale:** Reactome: ~2,500 pathways; KEGG: ~340 disease-related; MSigDB: 33,000+ gene sets
- **ResistanceMap status:** ❌ not loaded into the pipeline (the v8 `proteomics-pathway-validator` agent uses live ChEMBL + Open Targets MCP queries but doesn't load these flat files)
- **Why add:** persisted pathway annotations enable offline grounding of attribution scores.

## 2 — How similar SOTA models compose their training data

| Model (year) | Cell lines | Drugs | Multi-omics | Patient cohort | Trajectory | Pathway |
|---|---|---|---|---|---|---|
| **DeepCDR** (Liu, Bioinf 2020) | CCLE 561 | GDSC 238 | RNA + mut + meth | — | — | — |
| **DrugCell** (Kuenzi, Cancer Cell 2020) | CCLE 1,235 | DrugBank 684 | RNA + DepMap CRISPR | — | — | GO/Reactome |
| **MOLI** (Sharifi-Noghabi, Bioinf 2019) | CCLE + GDSC 985 | GDSC 282 | RNA + mut + CNV | TCGA + private | — | — |
| **PaccMann** (Manica/Born, MCB 2018-19) | GDSC | GDSC 208 | RNA-seq + SMILES | — | — | STRING-PPI prior |
| **MCA encoder** (Manica 2019) | GDSC | GDSC 208 | RNA-seq + SMILES + STRING | — | — | STRING-PPI |
| **MOFA+** (Argelaguet, Genome Biol 2020) | CLL 200 | drug-response panel | RNA + meth + mut + drug | CLL cohort | — | — |
| **SeNMo** (Waqas 2024) | TCGA pan-cancer | — | RNA + meth + miRNA + mut + protein + clinical | TCGA 33 cancers | — | — |
| **Threads** (Vaidya 2025) | TCGA + CPTAC | — | H&E + genomic + transcriptomic | 47k tissue sections | — | — |
| **TrajectoryNet** (Tong/Krishnaswamy ICML 2020) | scRNA panels | — | RNA only | EB + iPSC + bone-marrow | ✅ time-series | — |
| **VGFM** (Wang HF 2025) | scRNA panels | — | RNA only | unpaired snapshots | ✅ flow-matching | — |
| **Conditional Monge Gap** (Driessen HF 2025) | scRNA + multiplexed protein | drug + dose | RNA + protein | — | ✅ neural OT | — |
| **STAGED** (Krishnaswamy 2025) | spatial transcriptomics | — | RNA + spatial | — | ✅ graph ODE + ABM | — |
| **Dynamic biomarker MM** (Xiong, J Transl Med 2026, PMID 41814396) | — | — | flow cytometry + cytokine | **MM 662 patients (longitudinal)** | ✅ Cycle-by-Cycle | — |
| **Platelet-related MM signature** (Li, Ann Hematol 2026, PMID 41762247) | — | — | bulk + scRNA + clinical | MM (GSE124310 + others) | — | — |
| **iCluster+ / JIVE** (TCGA pan-cancer) | TCGA tumors | — | RNA + meth + mut + CNV + protein | TCGA | — | — |

| Model (year) | Cell lines | Drugs | Multi-omics | Patient cohort | Trajectory | Pathway |
|---|---|---|---|---|---|---|
| **ResistanceMap (v6/v8)** | **CCLE 622 train / 132 test** | **GDSC 11** | RNA-derived proteomics + 42-d epigenomics + STRING PPI | ❌ none | ⚠️ cell-line only, no time | ❌ no persisted attribution |

Looking at the columns ResistanceMap is missing relative to the field's medians: **patient cohort, trajectory time-axis, persisted pathway/attribution**. All three are addressable with data already on disk (scRNA disease_stage) or with public downloads (DepMap CRISPR, MMRF after IRB).

## 3 — ResistanceMap inventory: current vs. target

| Dataset | Currently used? | Target state | Effort |
|---|---|---|---|
| GDSC drug screen | ✅ 11/624 drugs | Expand to 30–50 MM-relevant + retain CTRPv2 overlap | M |
| CCLE proteomics | ✅ 886/1,400 lines | Bring up to ≥1,200 (CCLE 2024 release) | M |
| CCLE epigenomics | ✅ 42 features | Replace with ATAC-seq full track set (~15k peaks) | M |
| STRING PPI v12 | ✅ 473k edges | OK (confidence 0.7 cutoff is principled) | — |
| **GSE124310 scRNA (27.8k MM cells)** | ❌ **on disk, ignored** | Pseudobulk per-patient → fusion modality | **S (file ready)** |
| **GSE271107 scRNA (143.7k cells, HD→MGUS→SMM→MM)** | ❌ **on disk, ignored** | Pseudobulk per (patient, stage) → trajectory time-axis | **S (file ready)** |
| **CTRPv2** | ❌ configured, not on disk | Download + ingest as cross-validation drug screen | **S (download) + M (loader)** |
| **DepMap CRISPR essentiality** | ❌ not configured | Add as causal-grounding signal for pathway-validator | **M (download + loader)** |
| **HMCL Keats panel** | ❌ not configured | Add 65+ MM-specific cell lines | **L (download + harmonize)** |
| **PRISM Repurposing** | ❌ not configured | Pre-train drug encoders on 4,500-drug viability | **L (download + drug encoder)** |
| **MMRF CoMMpass** | ❌ unwired loader | Wire `data/mmrf_loader.py` into `prepare_data` | **L (IRB + transfer-learning)** |
| Reactome/KEGG flat files | ❌ not loaded | Persist as pathway annotation tables | **S** |

Effort key: **S** ≤ 1 day, **M** 1–3 days, **L** ≥ 1 week.

## 4 — Priority roadmap (ranked by ROI)

### Tier 1 — already on disk, code-only changes (highest ROI)

1. **Wire GSE124310 + GSE271107 scRNA into the pipeline.**
   - Build `checkpoints/scrna_summary.pt` with per-(patient, disease_stage) pseudobulk via `scripts/build_scrna_summary.py`.
   - Update `MultiOmicsDataset` to optionally carry pseudobulk + stage labels.
   - **This single change provides the longitudinal time-axis v8 said was absent**, unlocking the "predict before it happens" claim with proper data backing.
   - Effort: 1–2 days.
   - Models that gain: any trajectory work (TrajectoryNet, VGFM, Monge Gap can ingest the pseudobulk).

### Tier 2 — public downloads, no IRB

2. **Add CTRPv2 as a cross-screen validation dataset** (GDSC train → CTRPv2 generalize is the canonical benchmark for DeepCDR/MOLI/TUGDA).
3. **Add DepMap Achilles CRISPR essentiality** → grounds pathway-validator's predictions in actual perturbation data, closing the v8 causal-audit gap.
4. **Add HMCL Keats panel** for MM-specific cell-line specialization.
5. **Pre-load Reactome + KEGG flat files** as pathway annotation tables for offline grounding.

### Tier 3 — heavier additions

6. **Add PRISM Repurposing** (4,500 drugs) and pre-train a drug encoder; this is the architectural shift that brings ResistanceMap up to current SOTA.
7. **Wire `data/mmrf_loader.py` into `prepare_data`** once an IRB / GDC token is available.
8. **Replace CCLE 42-d epigenomics with ATAC-seq full peak set** (~15k peaks per line).

### Tier 4 — methodological extensions

9. **Add cross-cohort validation** on GSE193531 + GSE161801 (other public MM scRNA datasets).
10. **Add CPTAC pan-cancer proteogenomics** as a transfer-learning source for the protein encoder.

## 5 — Reproducibility-first ingestion checklist

For every new dataset added to `configs/default.yaml`:

- [ ] Loader function in `resistancemap/data/loaders.py`
- [ ] Path entry in `DataConfig`
- [ ] Validation entry in `validate_data_files`
- [ ] Download instructions (or IRB pointer) in `scripts/download_data.sh`
- [ ] Schema doc in `docs/<DATASET>_SCHEMA.md`
- [ ] Smoke-test in `tests/test_data_loaders.py`
- [ ] Update `docs/DATASETS_AND_BENCHMARKS.md` (this doc) ✓
- [ ] Update README data-sources table
- [ ] Add a row to `RUNS.md` for any run that uses the new data

## 6 — Why these specific datasets matter (one-line per benchmark)

- **CTRPv2**: the canonical generalization-screen for any drug-response model trained on GDSC.
- **DepMap CRISPR**: the only public source of causal target essentiality at cell-line scale; without it, "pathway prediction" is association, not causation.
- **GSE124310 + GSE271107**: free 171k-cell longitudinal disease-stage data sitting on disk; the highest-leverage single addition.
- **HMCL panel**: 65+ MM cell lines (vs CCLE's ~10 MM lines); the difference between "MM-relevant" and "MM-specialized."
- **PRISM**: 4,500 drugs vs GDSC's 624 — the difference between a drug-pair-by-drug-pair model and one that learns drug embeddings.
- **MMRF CoMMpass**: the only MM cohort with paired multi-omics + longitudinal clinical outcomes at >1,000-patient scale.

---

_This doc is the strategic anchor for all data-related work going forward. When a dataset is added, update Section 1 with a card and Section 3 with status. When a SOTA paper is added to comparison, update Section 2._
