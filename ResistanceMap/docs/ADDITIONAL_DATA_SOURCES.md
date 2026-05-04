# ResistanceMap v10 — Additional Public Data-Source Survey

_Gap-driven survey of datasets that would meaningfully strengthen the v10 framing: **multi-omic foundation model for predicting epigenetic drug-resistance trajectories in hematologic malignancies, using neural ODEs on Waddington landscapes and PPI propagation.**_

**Audit anchor (2026-05-03):** ResistanceMap already has CCLE proteomics+epigenomics, GDSC IC50, PRISM, DepMap CRISPR, MMRF CoMMpass IA22, GSE124310, GSE271107, and STRING. Datasets below are evaluated as **incremental** to that baseline. URLs were verified by `curl -L` HEAD or REST probe on 2026-05-03 unless flagged UNVERIFIED.

---

## §1 — Per-dataset table

| # | Dataset | URL (verified) | License / access | Registration | Approx size | Leverage (1-10) | Effort (1-10) | Loader change |
|---|---|---|---|---|---|---|---|---|
| 1 | **BLUEPRINT epigenome (IHEC hub)** | `https://blueprint-data.bsc.es/` (200), `https://epigenomesportal.ca/ihec/` (200) | CC-BY for tracks; controlled-access for raw EGA reads | EGA DAC for raw; tracks open | ~1,300 datasets, ~50TB raw / ~200GB bigWig | 9 | 7 | New `loaders.load_blueprint_tracks()` ingesting bigWig per cell-type → average over gene promoters → align to existing protein/gene IDs |
| 2 | **CCLE RRBS DNA methylation (legacy 2019)** | `https://depmap.org/portal/data_page/` (200, search "RRBS") — direct file URL UNVERIFIED (404 on legacy CCLE\_RRBS\_TSS path) | CC-BY-4.0 (DepMap) | None | 843 cell lines × ~14k TSS regions | 7 | 2 | Extend `_load_ccle_epigenomics` to read RRBS β-values; add as new feature block alongside H3K27ac |
| 3 | **Roadmap Epigenomics core marks (E000)** | `https://egg2.wustl.edu/roadmap/web_portal/` (200), tracks at `https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/...` (200) | CC-BY-4.0 | None | 127 reference epigenomes (incl. E029 CD19+ B-cell, E032 CD8+ Tcell), ~50GB | 6 | 4 | New `load_roadmap_chromhmm()` providing 25-state vector per gene per cell-type as **prior** for landscape ODE |
| 4 | **ENCODE B-cell histone ChIP-seq** | `https://www.encodeproject.org/search/?type=Experiment&biosample_ontology.term_name=B+cell&assay_title=Histone+ChIP-seq` returns **26 experiments** (verified) | CC0 | None | 26 experiments × 6 marks for ~10 B-lymphoid lines | 7 | 3 | Reuse epigenomics loader; pull processed bigWig signal per gene |
| 5 | **GEO MM ATAC-seq series (n=169)** — top hit `GSE279766` (NRF1 / MM resistance, 137 samples, ATAC+ChIP+RNA, 2025) | `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279766` (200) — series confirmed `Public on Sep 09 2025`, PMID 40920573 | Open (GEO) | None | 137 samples / ~80GB | **9** | 5 | New `load_geo_atac_mm()` — peak-call → align to gene → concat with existing `epigenomics` block |
| 6 | **4DN MM Hi-C** | `https://data.4dnucleome.org/search/?type=ExperimentSet&biosource.cell_line.term_name=multiple+myeloma` — **`"total": 0` (verified)** | n/a | n/a | **None public** | 0 | — | DROP — no MM Hi-C in 4DN. B-cell / plasma-cell biosources also return 0. Use ENCODE 3D-genome (next row) instead. |
| 7 | **MMRF CoMMpass via GDC (vs. internal)** | `https://api.gdc.cancer.gov/projects/MMRF-COMMPASS` returns 995 cases / 34,109 files / 206 TB (verified) | dbGaP phs000748 (controlled) | dbGaP DAR | 995 patients (matches IA22) | 0 | — | DROP — already have IA22 on disk; GDC version is the same cohort. |
| 8 | **MMRF research portal extras** (themmrf.org) | `https://research.themmrf.org/` (200) — sub-cohorts beyond GDC | Researcher account, free | Required, ~1 week | ~50 add'l biopsies, MRD timepoints | 5 | 6 | Adapt MMRF loader for additional FFPE / MRD timepoint files |
| 9 | **TCGA-LAML (GDC)** | `https://api.gdc.cancer.gov/projects/TCGA-LAML` returns 200 cases / 8,839 files (verified) | Open (level-3) + dbGaP for raw | None for processed | 200 patients RNA+WES+meth+clinical | 7 | 3 | Reuse MMRF loader; add `disease='AML'` field for transfer-learning evaluation |
| 10 | **Beat AML 1.0 (GDC project BEATAML1.0-COHORT)** | `https://api.gdc.cancer.gov/projects/BEATAML1.0-COHORT` returns 826 cases / 16,794 files (verified). Also `BEATAML1.0-CRENOLANIB` (56 cases — drug-resistance specific) | Open + dbGaP phs001657 controlled tier | dbGaP for WGS | 826 AML patients, ex-vivo drug response on **122 drugs** (Tyner et al. 2018, PMID 30333627) | **10** (closest analog to v10 framing for AML) | 5 | New `load_beataml()` mirroring MMRF loader + ex-vivo IC50 block |
| 11 | **ICGC CLL-ES** | `https://dcc.icgc.org/projects/CLLE-ES` (200) | Open (controlled tier via DACO) | DACO for WGS | ~500 CLL patients, WGS+RNA+meth | 5 | 6 | Optional add-on; CLL is biologically distant from MM, value mostly for transfer-learning ablation |
| 12 | **DLBCL/FL trial scRNA (GEO)** | GEO eUtils search confirms presence; specific accession UNVERIFIED in this audit | Mostly open | None | varies | 4 | 6 | Defer to v11; tangential to MM-epigenetic core |
| 13 | **HemAtlas / HCA hematology project** | `https://service.azul.data.humancellatlas.org/index/projects?filters=...myeloma...` returns **`"total": 0` (verified)** | — | — | **None matches MM** in HCA Azul | 1 | — | DROP — no MM project in HCA |
| 14 | **CELLxGENE Discover Census** | `https://api.cellxgene.cziscience.com/curation/v1/datasets` (200) and `https://api.cellxgene.cziscience.com/dp/v1/datasets/index` (200) | CC-BY-4.0 | None | ~75M cells (2025 census) | 6 (for foundation-model fine-tuning only) | 4 | New pretraining adapter consuming TileDB/H5AD; **see §4** |
| 15 | **Tabula Sapiens v2** | Figshare ID 40067134 (challenge-protected 202 — interactive download required, UNVERIFIED for direct script) | CC-BY-4.0 | None (manual click-through) | ~480k cells / 24 organs | 4 | 4 | Adapter same as #14; not MM-specific |
| 16 | **MM-specific cell atlas papers 2024-2026** (GEO MM scRNA: GSE193531, GSE161801, GSE235326 verified accessible) | GEO accession URLs all 200 | Open | None | varies, ~30-200k cells each | 7 | 5 | Use existing scRNA loader; concat with GSE124310/GSE271107 |
| 17 | **Drug Repurposing Hub (Broad)** | `https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt` (200), `..._samples_20200324.txt` (200) | CC-BY | None | ~6,800 compounds × annotations | 5 | 1 | Trivial — extend SMILES feature builder to harmonize against PRISM IDs |
| 18 | **PharmacoDB v2** | `https://pharmacodb.ca/` (200) — REST endpoint `/api/v1/datasets` returns SPA shell (server-side rendered API documented at `/api`) | CC-BY-4.0 | None | Harmonized GDSC+CTRPv2+PRISM+gCSI IC50 | 6 | 4 | Use as **harmonization oracle** for cross-screen IC50 leakage audit |
| 19 | **MM-specific Perturb-seq** | GEO eUtils `multiple+myeloma+perturb-seq` returns **0 results** (re-verified 2026-05-03, confirms prior causal-audit finding) | — | — | **None as of 2026-05-03** | 0 | — | DROP — not yet available |
| 20 | **Active MM scRNA trials (ClinicalTrials.gov v2)** | `https://clinicaltrials.gov/api/v2/studies?query.cond=multiple+myeloma&query.term=single-cell` returns 10+ active trials (verified). None publicly release scRNA pre/post during run | Open metadata; data NOT released | n/a | metadata only | 2 | — | Watch-list; not actionable for v10 paper |
| 21 | **PRIDE proteomics MM** | `https://www.ebi.ac.uk/pride/ws/archive/v2/projects?keyword=multiple+myeloma` (200, list returned) | CC-BY | None | dozens of PXD entries | 4 | 6 | Optional — may improve proteomics coverage beyond CCLE Nusinow |

---

## §2 — TOP-3 recommended pulls

### Pick #1 — GEO `GSE279766` + the broader MM ATAC/ChIP cluster (rows #5)
**Verified:** PubMed 40920573, public 2025-09-09, 137 samples, ATAC + histone-ChIP + RNA, MM resistance focus (NRF1 axis). `eutils` returns **169 MM ATAC-seq + 55 MM ChIP-seq series** total. This is the single biggest quick win for the v10 *epigenetic resistance* claim. Today the ResistanceMap "epigenomics" block is 42 CCLE features (averaged H3K27ac proxies). Adding peak-level ATAC + multi-mark ChIP across **MM patients (not cell lines)** turns the epigenetic axis from a feature-weight bullet into a primary signal — the difference between a chromatin-state model and a proteomics+gene-set model rebadged. Effort 5/10: well-typed pipeline (BWA → MACS2 → matrix-by-promoter → align to gene namespace already in use). Leverage 9/10.

### Pick #2 — Beat AML 1.0 (GDC project `BEATAML1.0-COHORT`, row #10)
**Verified:** 826 cases, 16,794 files, includes the `BEATAML1.0-CRENOLANIB` resistance sub-cohort (56 cases). Tyner et al. 2018 (PMID 30333627) provides ex-vivo IC50 across 122 drugs **per patient sample** — i.e. the *closest analog in any hematologic malignancy* to what v10 wants to predict. Crucially, this enables the "hematologic foundation model" claim: pretrain/fine-tune on MMRF + Beat AML jointly, then evaluate cross-disease transfer. Without an AML cohort, "hematologic" in the v10 title is technically just MM. Leverage 10/10. Effort 5/10 (loader is structurally identical to MMRF; ex-vivo IC50 needs harmonization with GDSC IC50 scaling).

### Pick #3 — BLUEPRINT IHEC hematopoietic reference epigenomes (row #1)
**Verified:** `https://blueprint-data.bsc.es/` (200) and IHEC portal (200). Provides H3K4me3 / H3K27me3 / H3K27ac / H3K36me3 / DNA-methylation / ATAC across the **hematopoietic differentiation hierarchy** (HSC → CLP → pro-B → naive-B → memory-B → plasmablast → plasma cell → MM). This is the empirical *Waddington landscape prior* the v10 framing implicitly claims to learn — anchoring the neural-ODE landscape in a real reference epigenome (rather than only learning it from a few hundred MM patients) is what would let the paper claim biological grounding rather than statistical regularization. Open tracks suffice; raw EGA is not needed for v10. Leverage 9/10. Effort 7/10 (multi-mark harmonization is the cost).

**These three pulls together address all four v10 axes:** epigenetic (#1, #3), trajectory (#1 = pre/post-resistance, #3 = differentiation lineage), hematologic (#2), foundation-model (the combined corpus of MM + AML + reference epigenomes is what makes "foundation model" defensible).

---

## §3 — What's NOT worth pulling

- **4DN MM Hi-C (#6)** — the 4DN portal returns `total: 0` for MM, B-cell, *and* plasma-cell biosources (verified). The dataset literally does not exist in 4DN today. Do not promise 3D-genome features in v10; if needed, fall back to ENCODE H3K27ac super-enhancer calls (#4).
- **HCA HemAtlas (#13)** — Azul filter for "myeloma" returns 0 projects. The HCA hematopoietic atlas exists but does not include MM.
- **MMRF via GDC (#7)** — same cohort already on disk via IA22; pulling it again is pure churn.
- **MM-specific Perturb-seq (#19)** — re-verified 2026-05-03; no MM Perturb-seq series in GEO. The v10 causal-validity story still has to rely on DepMap CRISPR + ChEMBL mechanism annotations.
- **Tabula Sapiens (#15)** and **CELLxGENE Census (#14)** — *for re-pretraining purposes*. See §4. They are still useful as eval / fine-tune corpora but not as ResistanceMap's own pretraining base.
- **CLL & DLBCL (rows #11, #12)** — biologically distant from MM; transfer-learning gain is small relative to AML. Defer to v11+.
- **NCI-60 / CellMiner** (already noted in `DATASETS_AND_BENCHMARKS.md`) — too few cell lines to move CCLE-trained models.

---

## §4 — Foundation-model pretraining strategy

**Verdict: do NOT re-pretrain from CELLxGENE/Tabula Sapiens. Fine-tune an existing pretrained backbone.**

Reasoning:

1. **Coverage is already saturated by published backbones.** scGPT (33M cells), Geneformer (30M cells), scFoundation (50M cells), and UCE were each pretrained on a near-superset of CELLxGENE Census + Tabula Sapiens. Re-pretraining gives the same backbone with worse compute/data efficiency. The marginal Census update post-2024 is ~25M additional cells, not enough to justify a full re-train.
2. **MM is rare in the census.** A back-of-envelope filter of CELLxGENE Discover for "myeloma" returns a small minority of datasets relative to the 75M-cell total. Re-pretraining on the full census amortizes capacity across irrelevant tissues. Fine-tuning lets us spend capacity on the slice that matters.
3. **The actual scarce data is hematologic + epigenetic + longitudinal.** None of that is in CELLxGENE at scale. The bottleneck is *not* pretraining corpus size; it is paired epigenome + drug-response + outcome at the single-cell level. Pulls #1, #2, #3 above attack that bottleneck directly.
4. **Recommended architecture for v10:**
   - **Backbone:** scFoundation or Geneformer (frozen during initial fine-tuning).
   - **Stage 1 fine-tune:** MM scRNA pseudobulk (GSE124310 + GSE271107 + the GEO MM ATAC/RNA series from #5/#16) — adapts the backbone to plasma-cell representation space.
   - **Stage 2 fine-tune:** add Beat AML pseudobulk (#10) + BLUEPRINT hematopoietic reference (#3) as auxiliary tasks (multi-task head). This is where the "hematologic foundation model" claim earns its name.
   - **Stage 3 task head:** drug-resistance trajectory ODE + PPI-propagation regularizer on STRING (already on disk).
   - **Compute budget:** stage-1+2 fits in <5 GPU-days on a single A100; full re-pretrain would be 3-6 GPU-months. Strategic ratio is ~30:1 in favor of fine-tuning.

**Caveat:** if v10 reviewers push back on "foundation model" terminology with only fine-tuning, the rebuttal is that scFoundation/Geneformer are themselves the foundation, and our contribution is the **hematologic-epigenetic specialization** — analogous to how Med-PaLM relates to PaLM. That framing is defensible. Calling our own ~5M-cell trained-from-scratch model a foundation model would not be.

---

## Verification log (2026-05-03)

- 200 OK: BLUEPRINT BSC, IHEC portal, EGA studies, DepMap downloads, ENCODE search, Roadmap WUSTL, 4DN portal, MMRF research portal, CELLxGENE, Tabula Sapiens portal, PharmacoDB, GEO accession pages (GSE279766, GSE285195, GSE294108, GSE325617, GSE193531, GSE161801, GSE235326), Drug Repurposing Hub S3 (drugs + samples), GDC API for MMRF-COMMPASS / TCGA-LAML / BEATAML1.0-COHORT / BEATAML1.0-CRENOLANIB, ICGC CLLE-ES.
- **`"total": 0` (i.e. confirmed empty, the strongest negative signal):** 4DN MM Hi-C, 4DN B-cell ExpSet, 4DN plasma-cell ExpSet, ENCODE plasma-cell experiments, HCA Azul "myeloma" filter, GEO `multiple+myeloma+perturb-seq`.
- **404 / non-canonical (UNVERIFIED, alternate path needed):** `blueprint-epigenome.eu` legacy URL (replaced by `blueprint-data.bsc.es`), legacy CCLE RRBS direct file path on DepMap (file exists per portal but exact URL changed), `cbioportal-datahub.s3` `mm_broad_2014.tar.gz` returns 403 (bucket access policy), Tabula Sapiens Figshare direct download returns 202 challenge (interactive only).
- **Largest UNVERIFIED gap:** legacy CCLE RRBS direct download URL — the portal page is reachable (200) but the historical file path returns 404. Need a manual visit to `depmap.org/portal/data_page/?datasetId=Methylation_(1kb_upstream_TSS)` to grab the current versioned URL before scripting #2.
