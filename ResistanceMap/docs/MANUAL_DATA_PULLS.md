# Manual Data Pulls — ResistanceMap

Pipeline status motivating this doc: **692 labeled samples × 11 drugs, test_mse 2.84** (no signal). Each section is self-contained: paste the command, drop the file, apply the listed loader change. URLs verified by HTTP HEAD/GET on 2026-05-03 unless flagged.

---

## 1. MMRF CoMMpass IA22 — controlled access, NOT a `wget`

The only path to a defensible MM-clinical claim. ~1,150 newly-diagnosed MM patients, paired RNA-seq + WGS/WES + longitudinal clinical (ISS, treatment, PFS, OS).

- **Source**: dbGaP study `phs000748`, distributed via GDC (`portal.gdc.cancer.gov/projects/MMRF-COMMPASS`). MMRF Researcher Gateway (`research.themmrf.org` → redirects to `mmrf.formstack.com/forms/mmrf_virtual_lab_access_request`) is the parallel route + bundles Keats Lab MM lines.
- **License**: NIH Genomic Data User Code of Conduct (IRB-approved use only, no redistribution, must cite phs000748).
- **Registration**: YES, ~2-6 weeks: (1) eRA Commons account; (2) Data Access Request (DAR) via dbGaP with IRB protocol + Research Use Statement, UNT Signing Official countersigns; (3) post-approval, mint GDC token (30-day expiry) at `portal.gdc.cancer.gov` → user menu → Download Token; (4) build manifest at the project page (Repository tab → filter Project=MMRF-COMMPASS, Data Category=Transcriptome Profiling + Clinical → Download Manifest).
- **Size**: full release ~3-5 TB; RNA-seq + clinical subset ~400 GB; clinical TSVs alone <100 MB.
- **Command** (after DAR + token + manifest):
  ```bash
  mkdir -p ~/.local/bin && cd /tmp \
    && curl -L -o gdc-client.zip https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64-py3.8-ubuntu-20.04.zip \
    && unzip gdc-client.zip && mv gdc-client ~/.local/bin/ && chmod +x ~/.local/bin/gdc-client
  mkdir -p data/raw/mmrf_commpass
  ~/.local/bin/gdc-client download -m ~/.gdc/mmrf_manifest.txt -t ~/.gdc/token.txt \
      -d data/raw/mmrf_commpass/ --n-processes 4 --retry-amount 5
  ```
- **Loader change**: `loaders.py::load_mmrf_data` (lines 848-931) expects flat `clinical.txt` / `gene_expression.tsv` at the top of `mmrf_commpass_dir`; `gdc-client` lays files out as `<uuid>/<filename>`. Add a `Path.rglob` walk, group by `MMRF_<id>_BM_*` barcode, match `*.rsem.genes.results` to the per-patient clinical TSV. Also extend `data/clinical_labels.py` to emit a binary "PFS<18mo" or "OS event" label so MMRF rows feed the supervised head.
- **Leverage 10/10** — only dataset producing real MM clinical outcomes; without it the central claim is unsupported. **Effort 8/10** — multi-week IRB + loader rewrite + label engineering. **Risk 4/5** — DAR can bounce; annual renewals; PHI handling.

---

## 2. CCLE/DepMap RNA-seq expression (26Q1)

~1,500 cell lines × ~19k genes. No new labeled samples, but ~3× richer per-sample features and enables proteomics↔mRNA cross-checks. **Filename was renamed** in 24Q4 from the legacy `OmicsExpressionProteinCodingGenesTPMLogp1.csv`.

- **Source**: `depmap.org/portal/data_page/?tab=allData`, release "DepMap Public 26Q1" (2026-04-01).
- **File**: `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`, ~280 MB. License CC-BY 4.0. No registration.
- **Command** (DepMap signed URLs expire ~30 days; regenerate via the API):
  ```bash
  mkdir -p data/raw/depmap
  URL=$(curl -s https://depmap.org/portal/api/download/files \
    | awk -F',' '$1=="DepMap Public 26Q1" && $3=="OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"{print $4; exit}')
  curl -L -o data/raw/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv "$URL"
  ```
- **Loader change**: add `load_depmap_expression(config)` next to `load_depmap_crispr` (loaders.py:795-845 is the template — same `(entrez_id)` regex strip, same return shape). Wire an `expression` tensor through `MultiOmicsDataset.__init__` like `crispr_effect`, into `harmonize_omics`. Add `depmap_expression_path` to `DataConfig`.
- **Leverage 6/10** / **Effort 4/10** / **Risk 1/5**.

---

## 3. CCLE WES driver mutations (26Q1)

MM-relevant drivers (TP53, KRAS, NRAS, BRAF, MYC, MAF, FGFR3, CCND1) are categorical features the model has zero access to today.

- **Source**: same 26Q1 page. **File**: `OmicsSomaticMutationsMatrixDamaging.csv` (binary cell-line × gene matrix, **use this**); long-form `OmicsSomaticMutations.csv` and `OmicsSomaticMutationsMatrixHotspot.csv` are alternatives. ~30 MB. CC-BY 4.0. No registration.
- **Command**:
  ```bash
  URL=$(curl -s https://depmap.org/portal/api/download/files \
    | awk -F',' '$1=="DepMap Public 26Q1" && $3=="OmicsSomaticMutationsMatrixDamaging.csv"{print $4; exit}')
  curl -L -o data/raw/depmap/OmicsSomaticMutationsMatrixDamaging.csv "$URL"
  ```
- **Loader change**: add `load_depmap_mutations(config)`; matrix is already DepMap ModelID-keyed (no remap), so concat into the proteomics feature block in `harmonize_omics`. Add `depmap_mutations_path` to `DataConfig`.
- **Leverage 7/10** — drivers are mechanistically tight to MM resistance (TP53↔Bortezomib, MAF↔IMiD, KRAS/NRAS↔Ras agents). **Effort 3/10** — copy-paste loader. **Risk 1/5**.

---

## 4. CCLE methylation (RRBS, legacy 2019)

The current epigenomics block is **42 features** (chromatin profiling). RRBS adds ~30-50k CpG-cluster values per line — ~1000× expansion. RRBS lives only in the 2019 archive (no methylation in 26Q1).

- **File**: `CCLE_RRBS_TSS1kb_20181022.txt.gz`, ~80-150 MB. Sister files: `CCLE_RRBS_tss_CpG_clusters_20181022.txt.gz`, `..._cgi_...`, `..._enh_...`. CC-BY 4.0. No registration.
- **Command**:
  ```bash
  URL=$(curl -s https://depmap.org/portal/api/download/files \
    | awk -F',' '$1=="CCLE 2019" && $3=="CCLE_RRBS_TSS1kb_20181022.txt.gz"{print $4; exit}')
  mkdir -p data/raw/ccle_epigenomics
  curl -L -o data/raw/ccle_epigenomics/CCLE_RRBS_TSS1kb_20181022.txt.gz "$URL"
  ```
- **Loader change**: extend `load_ccle_epigenomics` (loaders.py:254-320). RRBS file is wide TSV (rows=clusters, cols=cell lines) — `pd.read_csv(sep='\t').T`. Drop clusters with >50% missing, fill column median, log1p. Cell-line column headers are CCLE_Names not DepMap_IDs; route through `_map_sample_ids_to_depmap` (loaders.py:640). Append as a fourth assay slot at line 296, prefix `rrbs_tss1kb:`.
- **Leverage 7/10** — feature multiplier; epigenetic state matters for HDAC drugs (3 of 11 targets) and IMiD response. **Effort 5/10** — wide format + ID join. **Risk 2/5** — 2018 vintage limits cell-line overlap.

---

## 5. GDSC2 (Sanger Cell Model Passports)

**Already on disk** at `data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx`. Pull is for verification.

- **Source**: `https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/` (HTTP 200, content-length 21,330,376 verified). `cancerrxgene.org/downloads/bulk_download` itself returns HTTP 410 — use the COG mirror.
- **Command** (idempotent):
  ```bash
  curl -L -o data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx \
     https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx
  ```
- **11-drug check**: Bortezomib, Lenalidomide, Panobinostat, Vorinostat, Venetoclax, Dinaciclib, Palbociclib, Doxorubicin, Etoposide, Cyclophosphamide are present in GDSC2; **Romidepsin** is borderline (often only in GDSC1).
  ```bash
  python -c "import pandas as pd; print(sorted(set(pd.read_excel('data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx')['DRUG_NAME'])))"
  ```
- **Loader change**: existing `load_drug_sensitivity` already reads GDSC. If not yet converted to CSV:
  ```bash
  python -c "import pandas as pd; pd.read_excel('data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx').to_csv('data/raw/gdsc_drug_sensitivity.csv', index=False)"
  ```
- **Leverage 3/10** / **Effort 1/10** / **Risk 1/5**.

---

## 6. PharmacoDB cross-source IC50/AUC

Pre-harmonized GDSC + CTRPv2 + gCSI + CCLE drug response. Removes much of `_standardize_drug_columns`'s burden and adds gCSI rows.

- **Source**: REST API at `https://api.pharmacodb.ca/v1/` (host live; root requires resource path). Apache-2.0. No registration; rate-limited.
- **Command** (loop client-side; the API doesn't accept multi-name filters):
  ```bash
  mkdir -p data/raw/pharmacodb
  for d in Bortezomib Lenalidomide Panobinostat Vorinostat Romidepsin Venetoclax Dinaciclib Palbociclib Doxorubicin Etoposide Cyclophosphamide; do
    DID=$(curl -s "https://api.pharmacodb.ca/v1/drugs?name=${d}" | python3 -c "import json,sys;print(json.load(sys.stdin)[0]['id'])")
    curl -s "https://api.pharmacodb.ca/v1/drugs/${DID}/experiments?per_page=10000" > "data/raw/pharmacodb/${d}.json"
    sleep 0.3
  done
  ```
  Each record exposes `cell_line.name`, `dataset.name`, `profile.AAC`, `profile.IC50`. Estimated ~50-150k records total.
- **Loader change**: add `load_pharmacodb(config)` flattening JSON → `(sample_id, drug_name, ic50, source)`, then append `("PharmacoDB", config.pharmacodb_dir)` to the `sources` list in `load_drug_sensitivity` (loaders.py:511-521). Cell IDs are CCLE-style names; `_map_sample_ids_to_depmap` already handles them.
- **Leverage 5/10** — adds gCSI + harmonization fixes; most rows overlap GDSC/CTRPv2. **Effort 4/10**. **Risk 2/5** — API rate limits.

---

## 7. HMCL Keats Lab MM cell-line panel

11 target drugs were profiled on **CCLE leukemia/lymphoma lines, not MM**. MM-resistance claims need MM lines: KMS-11, OPM-2, NCI-H929, MM1.S, U266, RPMI-8226, KMS-12-BM, AMO-1.

- **Source**: `keatslab.org` (HTTP 200) for static genomic resources; **drug response** is on the MMRF Researcher Gateway (`research.themmrf.org` redirects to a Formstack request form). MMRF research-use license, non-redistribution.
- **Registration**: YES, single Formstack request, days-to-weeks (lighter than dbGaP).
- **Command**: NONE — manual web download after gateway approval. Place under `data/raw/hmcl_keats/`.
- **Public adjuncts** (no registration): static tables at `keatslab.org/data-repositories/myeloma-cell-line-characterization`; GEO `GSE125580` (RPMI-8226, KMS-11 under PI selection) — fetchable via the existing `fetch_geo_series` helper in `scripts/download_data.sh`.
- **Loader change**: HMCL files arrive as per-line CSVs (drug × dose × replicate). New `load_hmcl_keats(config)` normalizes to `(sample_id, drug_name, ic50)` and feeds the `load_drug_sensitivity` pivot as a fifth source. CCLE_Name-style IDs already exist in `data/raw/sample_info.csv` for the 8 MM lines (verified via `grep KMS11 data/raw/sample_info.csv`).
- **Leverage 8/10** — directly closes the "MM claim, no MM data" gap. **Effort 6/10** — gateway request + custom format. **Risk 3/5** — request can be denied; small N.

---

## 8. PubChem SMILES (drug-feature fallback)

ChEMBL is preferred (already wired via MCP); PubChem REST is the no-auth fallback. Verified Bortezomib lookup → `B(C(CC(C)C)NC(=O)C(CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2)(O)O` (CID 387447).

- **Source**: `pubchem.ncbi.nlm.nih.gov/rest/pug/`. Public domain. No auth; 5 req/s.
- **Command**:
  ```bash
  mkdir -p data/raw/pubchem
  for d in Bortezomib Lenalidomide Panobinostat Vorinostat Romidepsin Venetoclax Dinaciclib Palbociclib Doxorubicin Etoposide Cyclophosphamide; do
    curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${d}/property/CanonicalSMILES,MolecularWeight,XLogP/CSV" > "data/raw/pubchem/${d}.csv"
    sleep 0.3
  done
  cat data/raw/pubchem/*.csv | awk 'NR==1 || !/^"CID"/' > data/raw/drug_smiles.csv
  ```
- **Size** ~3 KB total.
- **Loader change**: trivial — point the existing drug-feature builder at `data/raw/drug_smiles.csv` if ChEMBL fails. SMILES→fingerprint already lives in the drug-encoder branch; no `MultiOmicsDataset` change.
- **Leverage 1/10** / **Effort 1/10** / **Risk 1/5**.

---

## 9. Open Targets disease-gene associations (MM, EFO_0001378)

Verified: GraphQL POST returns 5,639 MM-associated targets; top-5 by score are CRBN (0.706), TNFRSF17 (0.678), TP53 (0.667), LIG4 (0.660), KRAS (0.660). CC0; no registration.

- **Command** (single GraphQL POST, no MCP needed):
  ```bash
  mkdir -p data/raw/opentargets
  curl -s -X POST -H "Content-Type: application/json" \
    -d '{"query":"{ disease(efoId:\"EFO_0001378\"){ associatedTargets(page:{index:0,size:5000}){ count rows{ score datatypeScores{id score} target{ id approvedSymbol } } } } }"}' \
    https://api.platform.opentargets.org/api/v4/graphql > data/raw/opentargets/mm_associations.json
  ```
- **Size** ~2 MB.
- **Loader change**: new `load_opentargets_priors(config)` parses `data.disease.associatedTargets.rows[].{score,target.approvedSymbol}` into `dict[gene_symbol, prior_score]`. Three uses: (a) GNN node-feature weighting, (b) auxiliary signal in `agents/proteomics_pathway_validator.py`, (c) edge upweighting in `data/string_debiasing.py`. No `MultiOmicsDataset` schema change.
- **Leverage 4/10** / **Effort 2/10** / **Risk 1/5**.

---

## TOP-3 RECOMMENDATION

If you only pull three, in this order:

1. **MMRF CoMMpass IA22** — start the DAR today; the 2-6-week wall clock means everything else can stage in parallel. It is the only dataset that converts ResistanceMap from "cell-line drug-response regressor" to "MM clinical resistance model"; without it the central claim is unsupported regardless of test_mse.
2. **CCLE WES driver mutations (`OmicsSomaticMutationsMatrixDamaging.csv`)** — best leverage-per-effort in the public set: 30 MB, ~3-day loader change, mechanistically maps onto every MM resistance pathway in the 11-drug list. Cheapest way to move test_mse off the floor while waiting on MMRF.
3. **HMCL Keats Lab MM panel** — the next biggest credibility gap is "the model has barely any MM cell lines." Even 8-20 KMS/OPM/MM1.S/U266 rows × 11 drugs anchors the cell-line side in real MM biology; gateway request is much faster than dbGaP, submit alongside the MMRF DAR.

CCLE RNA-seq is item 4 — high-value but only after mutations, since mutations encode rarer, more drug-resistance-specific signal per byte. CCLE methylation, GDSC2 verification, PharmacoDB, PubChem, and Open Targets are tier-2 polish — wait until the model has signal worth interpreting.
