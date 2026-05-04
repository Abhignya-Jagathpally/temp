# MMRF CoMMpass Acquisition — Step-by-Step (rewritten 2026-05-03)

The MMRF CoMMpass study is a multi-center longitudinal study of newly-
diagnosed multiple myeloma patients with paired RNA-Seq + WGS/WES + clinical
follow-up, hosted at the NCI Genomic Data Commons (GDC).

> **Earlier versions of this document claimed MMRF was "controlled-access only"
> and pointed at legacy `*.rsem.genes.results` files. Both claims are wrong as
> of 2026-05-03.** The corrected facts and a working open-access path are below.
> All numbers in this doc were verified live against the GDC API on 2026-05-03;
> see `## Verification log` at the end.

---

## TL;DR

You probably do **not** need dbGaP. **2,960 files (~3.7 GB)** are open-access
and cover everything needed for Tier 2 predictive validation per
`docs/CAUSAL_VALIDITY_AUDIT.md`:

| Modality | Files | Size | Format | Access |
|---|---:|---:|---|---|
| RNA-Seq STAR augmented gene counts | 859 | ~3.5 GB | TSV (genes × counts) | open |
| Masked Somatic Mutations | 1,091 | ~50 MB | MAF | open |
| Copy Number Segments | 1,010 | ~150 MB | TXT (segment-mean) | open |
| Clinical / treatment / sample timeline | (API only — see §3) | <1 MB | TSV | open |
| **TOTAL** | **2,960 files + clinical** | **~3.7 GB** | | **no dbGaP, no token** |

dbGaP study `phs000748` is required only if you want raw BAM/FASTQ reads or
unmasked variant calls — those are the bulk of the 206 TB controlled-access
release and we do not need them for Tier 2.

---

## 1. What's actually at GDC for MMRF-COMMPASS

Verified live 2026-05-03 against `https://api.gdc.cancer.gov/projects/MMRF-COMMPASS`:

| Statistic | Value |
|---|---|
| Project ID | MMRF-COMMPASS |
| dbGaP accession | phs000748 |
| Cases (patients) | 995 |
| Total files | 34,109 |
| Total release size | ~206 TB (mostly controlled BAM/FASTQ) |
| **Open-access files** | **2,960** |
| Open-access by data category | Simple Nucleotide Variation 1,091; Copy Number Variation 1,010; Transcriptome Profiling 859 |

There is **no `Clinical` or `Biospecimen` *file* category** for this project —
clinical data lives only as case-level metadata in the `/cases` API. That is
expected; see §3.

---

## 2. Build the manifests via the GDC API (skip the Repository tab)

The Repository tab in the GDC portal has a 10,000-file cart limit and a
notorious failure mode where the Project filter silently resets when you
navigate from the global search. Every cart session we tried that way pulled
in tens of thousands of cross-project files. **Bypass the cart entirely** —
the API supports `return_type=manifest` and emits the exact TSV format
gdc-client (or our pure-Python downloader) consumes.

```bash
mkdir -p ~/.gdc

# Manifest 1: RNA-Seq STAR Counts (859 files, ~3.5 GB)
curl -sS -X POST "https://api.gdc.cancer.gov/files" \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {"op":"and","content":[
      {"op":"in","content":{"field":"cases.project.project_id","value":["MMRF-COMMPASS"]}},
      {"op":"in","content":{"field":"access","value":["open"]}},
      {"op":"in","content":{"field":"data_category","value":["Transcriptome Profiling"]}},
      {"op":"in","content":{"field":"analysis.workflow_type","value":["STAR - Counts"]}}
    ]},
    "return_type":"manifest","size":5000
  }' -o ~/.gdc/mmrf_rna.tsv

# Manifest 2: Masked Somatic Mutations (1,091 files, ~50 MB)
curl -sS -X POST "https://api.gdc.cancer.gov/files" \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {"op":"and","content":[
      {"op":"in","content":{"field":"cases.project.project_id","value":["MMRF-COMMPASS"]}},
      {"op":"in","content":{"field":"access","value":["open"]}},
      {"op":"in","content":{"field":"data_category","value":["Simple Nucleotide Variation"]}},
      {"op":"in","content":{"field":"data_type","value":["Masked Somatic Mutation"]}}
    ]},
    "return_type":"manifest","size":5000
  }' -o ~/.gdc/mmrf_snv.tsv

# Manifest 3: Copy Number Segments (1,010 files, ~150 MB)
curl -sS -X POST "https://api.gdc.cancer.gov/files" \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {"op":"and","content":[
      {"op":"in","content":{"field":"cases.project.project_id","value":["MMRF-COMMPASS"]}},
      {"op":"in","content":{"field":"access","value":["open"]}},
      {"op":"in","content":{"field":"data_category","value":["Copy Number Variation"]}}
    ]},
    "return_type":"manifest","size":5000
  }' -o ~/.gdc/mmrf_cnv.tsv

# Sanity-check counts
for f in ~/.gdc/mmrf_rna.tsv ~/.gdc/mmrf_snv.tsv ~/.gdc/mmrf_cnv.tsv; do
  echo "$f: $(($(wc -l < "$f") - 1)) files"
done
# expected: 859 / 1091 / 1010
```

These manifests are deterministic (same filters → same UUID list). Re-run the
curls to refresh if the GDC release advances.

---

## 3. Pull clinical / treatment / sample data via the `/cases` API

There are no clinical *files* at GDC for MMRF; clinical lives in case-level
metadata. Use the included puller — it paginates through all 995 cases,
expands `demographic + diagnoses + treatments + follow_ups + samples`, and
flattens to three TSVs:

```bash
python scripts/pull_mmrf_clinical.py --output-dir data/raw/mmrf_commpass
```

Outputs (verified on a 2026-05-03 run):

| File | Rows × Cols | What it carries |
|---|---|---|
| `clinical.tsv` | 995 × 16 | One row per patient. ISS stage, demographics, vital status, days_to_death, days_to_last_followup. |
| `treatments.tsv` | 7,184 × 11 | One row per (patient, treatment). `regimen_or_line_of_therapy`, `therapeutic_agents`, `days_to_treatment_start/end`. **This is the time-to-resistance signal.** |
| `samples.tsv` | 4,860 × 7 | One row per (patient, sample, aliquot). `aliquot_submitter_id` is the join key against the RNA/SNV/CNV file names. |

**Cohort highlights** (from the 2026-05-03 pull):

- 988 / 994 patients have ≥2 lines of therapy → resistance event observable
- 191 deaths + 804 censored alive → 995 usable for OS C-index
- ISS: I 348 / II 353 / III 266 / unknown 28
- Drug coverage of our 11 cell-line targets:
  Lenalidomide 1,393 / Bortezomib 1,076 / Cyclophosphamide 694 /
  Carfilzomib 567 / Pomalidomide 127 / Daratumumab 59 /
  Doxorubicin 22 (8 of 11 present)

Open access; no token required for any of this.

---

## 4. Download the open-access files

The historical advice was to use the `gdc-client` binary. As of 2026-05-03 the
GDC's binary distribution links 302-redirect to a not-found page, the recent
GitHub releases (2.1, 2.2, 2.3) ship source-only with no binary asset, and
`pip install gdc-client` is unavailable. We bypass all of that with a pure-
Python downloader (`scripts/gdc_download.py`) that hits the
`/data/<UUID>` endpoint with parallel workers, MD5 verification, and
resume-by-existence.

```bash
mkdir -p data/raw/mmrf_commpass/{rna,snv,cnv} logs/gdc_downloads

# RNA-Seq (~3.5 GB; ~2-5 min on a fast connection at 8 workers)
nohup python scripts/gdc_download.py \
    --manifest ~/.gdc/mmrf_rna.tsv \
    --out-dir data/raw/mmrf_commpass/rna \
    --workers 8 > logs/gdc_downloads/rna.log 2>&1 &

# Mutations (~50 MB, much smaller per-file → bandwidth-limited by per-request overhead)
nohup python scripts/gdc_download.py \
    --manifest ~/.gdc/mmrf_snv.tsv \
    --out-dir data/raw/mmrf_commpass/snv \
    --workers 8 > logs/gdc_downloads/snv.log 2>&1 &

# Copy Number (~150 MB)
nohup python scripts/gdc_download.py \
    --manifest ~/.gdc/mmrf_cnv.tsv \
    --out-dir data/raw/mmrf_commpass/cnv \
    --workers 8 > logs/gdc_downloads/cnv.log 2>&1 &

# Watch progress
tail -f logs/gdc_downloads/rna.log
```

The downloader is **resumable** — re-running it skips files whose existing
size + md5 match the manifest. Failed files retry up to 3× per worker; any
remaining failures are surfaced in the final summary line.

---

## 5. Final disk layout the loader expects

```
data/raw/mmrf_commpass/
├── rna/
│   ├── 1b166f66-….rna_seq.augmented_star_gene_counts.tsv
│   └── … (859 files, gene_id × counts/TPM/FPKM TSVs)
├── snv/
│   └── … (1,091 MAF.gz files, masked somatic mutations)
├── cnv/
│   └── … (1,010 CNV segment TXT files)
├── clinical.tsv             ← from pull_mmrf_clinical.py
├── treatments.tsv           ← from pull_mmrf_clinical.py
└── samples.tsv              ← from pull_mmrf_clinical.py (UUID join key)
```

The current `resistancemap/data/loaders.py:load_mmrf_data` expects
**already-merged** `clinical.txt` + `gene_expression.tsv` + `mutations.tsv`
files. To produce those from the per-sample GDC outputs above, run
`scripts/preprocess_mmrf_gdc.py` (write pending — see §7).

---

## 6. dbGaP path (only if you decide you need raw reads or unmasked variants)

You only need this for the ~31,000 controlled-access files (BAM/FASTQ raw
reads, unmasked VCFs). For Tier 2 validation, we do not.

If you decide to go controlled:

1. Apply for dbGaP study **phs000748** ("Relating Clinical Outcomes in
   Multiple Myeloma to Personal Assessment of Genetic Profile") via the
   eRA Commons — typical 2–6 week approval.
2. Once approved, mint a GDC user token from the GDC portal user menu.
   Tokens last 30 days.
3. Pass `--token PATH/to/gdc-user-token.txt` to `scripts/gdc_download.py`
   (the script already supports this flag).

Storage budget for the full controlled release: ~206 TB. Per-modality
subsets (e.g., WGS BAMs only, or RNA-seq BAMs only) are 5–30 TB each.

---

## 7. After the download — convert to the loader's expected format

The current `loaders.py:load_mmrf_data` reads **single merged files**:
`clinical.txt`, `gene_expression.tsv`, `mutations.tsv`. GDC ships per-sample
files. The adapter `scripts/preprocess_mmrf_gdc.py` (write pending) will:

- Concatenate the 859 STAR TSVs into a single `gene_expression.tsv`
  (rows = Ensembl gene_id, columns = aliquot_submitter_id from
  `samples.tsv`).
- Parse the 1,091 MAFs into a long-format `mutations.tsv`
  (Tumor_Sample_Barcode → patient via `samples.tsv`).
- Concatenate the 1,010 CNV segments and derive per-gene segment-mean +
  binary del17p / chr1q21 flags using GENCODE v36 gene coordinates.

Once the merged files exist, the existing pipeline will pick them up via
`configs/default.yaml: data.mmrf_commpass_dir: data/raw/mmrf_commpass/`.

---

## 8. What I (Claude) can do for you

| Step | Status |
|---|---|
| Build the three API manifests | ✅ done — see `~/.gdc/mmrf_*.tsv` |
| Pull clinical / treatments / samples from `/cases` API | ✅ done — see `data/raw/mmrf_commpass/{clinical,treatments,samples}.tsv` |
| Write `scripts/gdc_download.py` (replaces broken gdc-client) | ✅ done |
| Kick off the three downloads | ✅ running in background; see `logs/gdc_downloads/*.log` |
| Write `scripts/preprocess_mmrf_gdc.py` (merge to loader format) | ⏳ pending |
| Write `scripts/derive_cytogenetics.py` (chr1q21 / del17p flags from CNV) | ⏳ pending |
| Update `loaders.py:load_mmrf_data` schema if needed | ⏳ depends on preprocess output |

You only need to do dbGaP if reviewers later ask for raw-read evidence; for
the Tier 2 escalation in `CAUSAL_VALIDITY_AUDIT.md` and the SOTA bar in
`SOTA_BENCHMARK_PROTOCOL.md`, the open-access path above is sufficient.

---

## Verification log (live fetches, 2026-05-03)

| Endpoint | Filter | Result |
|---|---|---|
| `GET /projects/MMRF-COMMPASS?expand=summary` | — | 995 cases, 34,109 files, 206.5 TB, dbgap=phs000748 |
| `POST /files (return_type=manifest)` | open + Transcriptome + STAR Counts | 859 files |
| `POST /files (return_type=manifest)` | open + SNV + Masked Somatic Mutation | 1,091 files |
| `POST /files (return_type=manifest)` | open + CNV | 1,010 files |
| `POST /files (return_type=manifest)` | Clinical or Biospecimen | **0 files** (expected — clinical is metadata, not files) |
| `GET /cases?expand=...` | project=MMRF-COMMPASS | 995 cases, 7,184 treatments, 4,860 aliquots flattened |
| `GET /data/<uuid>` (one RNA STAR file) | open-access UUID | 4.21 MB TSV with 60,672 genes, GENCODE v36 |
| `gdc-client` binary URLs (1.6.1 / 2.0 / 2.3) | — | All 302→not-found |
| `pip install gdc-client` | — | No matching distribution on PyPI |
| `https://api.github.com/repos/NCI-GDC/gdc-client/releases` | — | Releases 2.1, 2.2, 2.3 ship source-only (no binary assets) |

The pure-Python `scripts/gdc_download.py` we wrote sidesteps the broken
binary distribution and is what this doc now recommends.
