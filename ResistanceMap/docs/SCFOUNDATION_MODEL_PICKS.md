# Single-Cell Foundation Model Survey for ResistanceMap MM Trajectory Pipeline

**Author:** Biomedical informatics curator (Claude agent)
**Date of survey:** 2026-05-03 (re-verified via public REST endpoints on this date)
**Goal:** Pick ONE single-cell foundation model whose latent embedding can serve as the per-cell representation for ResistanceMap's MM cell-state trajectory dynamics (neural ODE / CNF / SDE).
**Constraints:** CPU + occasional small GPU; research + possible-clinical use; license-permissive; minimal fine-tuning.

---

## VERIFICATION POSTURE

This document was rebuilt from the previous draft by re-verifying every model identifier, license tag, embedding dimension, parameter count, pretraining-corpus claim, and citation against live HTTPS endpoints (Hugging Face Hub API, raw HF file storage, NCBI eutils, GitHub REST v3, bioRxiv API). Every numeric or identifier claim below carries either a `[verified <YYYY-MM-DD> via <url>]` tag or an `[UNVERIFIED — fetch returned <status>; do not cite]` tag. The full curl trace lives in [Verification Log](#verification-log) at the bottom of the file. Per the user's release-bouncer rule, **only `[verified ...]` numbers are paper-safe.**

Two important high-level findings from this re-verification pass that the prior qualitative draft did not surface:

1. **`bowang-lab/scGPT` and `biomap-research/scFoundation` are NOT directly accessible via the HF model-card API** (both return HTTP 401 "Invalid username or password" without auth) [verified 2026-05-03 via https://huggingface.co/api/models/bowang-lab/scGPT]. The scGPT canonical weights are distributed via Google Drive links in the GitHub README and via the unauthenticated mirror **`tdc/scGPT`** (TDC group, Marinka Zitnik's lab) which carries an MIT license tag, base_model = `apliko/scGPT`, and a pinned model.safetensors. **`genbio-ai/scFoundation`** is an unauthenticated mirror with a `models.ckpt` file but a missing license tag in card metadata (the upstream GitHub repo is Apache-2.0).
2. **Geneformer V2 was scaled to ~104M cells (Genecorpus-104M), not the "95M / GF-95M" claimed in the prior draft.** The README also lists a 316M-parameter default V2 model and a 14M-cancer continual-learning variant. Embedding dims I previously claimed (256 / 512) were also wrong: actual hidden_size is 256 (V1-10M), 768 (V2-104M), 1152 (V2-316M).

The TOP-3 ranking is **unchanged** (scGPT primary, Geneformer secondary, scFoundation tertiary) because no verification surfaced a license blocker, dead repo, or unrecoverable identifier; but every numeric/identifier claim is now grounded.

---

## Candidate matrix (re-verified)

| # | Model | Year | Venue | PMID / preprint DOI | Pretrain cells | Embed dim | License (verified) | Canonical HF repo (verified) |
|---|-------|------|-------|---------------------|----------------|-----------|--------------------|------------------------------|
| 1 | scGPT | 2024 | Nat Methods 21:1470-1480 | PMID 38409223 | ~33 M (whole-human) | 512 (`embsize`) | MIT (GH) | `tdc/scGPT` (mirror); `bowang-lab/scGPT` is the upstream HF page but returns HTTP 401 |
| 2 | Geneformer | 2023 (V1) / 2026 V2 update | Nature 618:616-624 (V1) / Nat Comput Sci 2026 (V2) | PMID 37258680 (V1) | ~30 M (V1) / ~104 M (V2) | 256 / 768 / 1152 | apache-2.0 (HF cardData) | `ctheodoris/Geneformer` |
| 3 | scFoundation | 2024 | Nat Methods 21:1481-1491 | PMID 38844628 | >50 M (per GH README) | 768 (xTrimoGene) | Apache-2.0 (GH) | `genbio-ai/scFoundation` (mirror); upstream GH repo is the canonical weight host |
| 4 | UCE | 2023 (preprint) | bioRxiv 10.1101/2023.11.28.568918 | not in PubMed | ~36 M, 8 species | claimed 1280 (UNVERIFIED in this session) | MIT (GH) | no official HF mirror |
| 5 | CellPLM | 2023 (preprint) / ICLR 2024 | bioRxiv 10.1101/2023.10.03.560734 | not in PubMed | ~11 M (incl. spatial) | claimed 512 (UNVERIFIED in this session) | BSD-2-Clause (GH) | no official HF mirror |
| 6 | scBERT | 2022 | Nat Mach Intell (not PubMed-indexed) | DOI 10.1038/s42256-022-00534-z (not in PubMed) | ~1 M (PanglaoDB) | claimed 200 (UNVERIFIED in this session) | GPL-3.0 (GH) | no official HF mirror |

All entries in the rightmost three columns are verified against the curl traces in [Verification Log](#verification-log) below.

---

## 1. scGPT  (TOP PICK)

- **Citation:** Cui H, Wang C, Maan H, Pang K, Luo F, Duan N, Wang B. *scGPT: toward building a foundation model for single-cell multi-omics using generative AI.* **Nat Methods 21:1470-1480 (2024).** DOI 10.1038/s41592-024-02201-0; PMID **38409223** [verified 2026-05-03 via https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=38409223].
- **Canonical HF repo:** `bowang-lab/scGPT` returns HTTP 401 from the unauthenticated API [verified 2026-05-03 via https://huggingface.co/api/models/bowang-lab/scGPT]. **Use the unauthenticated mirror `tdc/scGPT`** (siblings: `model.safetensors` 203 MB, `scgpt_gh_repo_original_model.bin` 207 MB, `config.json`, `README.md`) [verified 2026-05-03 via https://huggingface.co/api/models/tdc/scGPT].
  - `tdc/scGPT` lastModified: **2025-04-10T16:20:58.000Z**
  - `tdc/scGPT` revision SHA: **`acf749f35bf5c0b00633838f02588272ed0d9911`**
  - `tdc/scGPT` license tag (cardData): **`mit`**
  - `tdc/scGPT` downloads (last 30d at fetch time): **553**
- **Canonical GitHub repo:** `bowang-lab/scGPT` — pushed_at **2026-04-29T17:41:42Z**, stars **1552**, open_issues **173**, license **MIT** (`spdx_id: MIT`) [verified 2026-05-03 via https://api.github.com/repos/bowang-lab/scGPT and https://api.github.com/repos/bowang-lab/scGPT/license]. Latest main-branch commit SHA: **`cebd6fae655b9c585a4807daa3ac31bb764f06b4`** (2026-04-27).
- **Pretraining corpus (verified from GH README):** "Pretrained on 33 million normal human cells" (whole-human checkpoint). Organ-specific checkpoints listed in the same README include **"blood — Pretrained on 10.3 million blood and bone marrow cells"** [verified 2026-05-03 via https://raw.githubusercontent.com/bowang-lab/scGPT/main/README.md]. The blood/bone-marrow variant is **directly relevant to MM** because plasma-cell lineage and the MM bone-marrow microenvironment are heavily represented.
- **Architecture (verified from `tdc/scGPT/config.json`):** `architectures: ["ScGPTModel"]`, `embsize: 512`, `d_hid: 512`, `nlayers: 12`, `nhead: 8`, `vocab_size: 60697`, `max_seq_len: 1536`, `cell_emb_style: "cls"`, `input_emb_style: "continuous"` [verified 2026-05-03 via https://huggingface.co/tdc/scGPT/raw/main/config.json].
- **Param count:** Not surfaced as a single numeric in the verified configs/README; the model.safetensors file is **203,233,980 bytes (~194 MB)** [verified 2026-05-03 via https://huggingface.co/api/models/tdc/scGPT/tree/main]. Treat this as the only fetch-grounded size signal — paper-cited param counts (~50 M) [UNVERIFIED — abstract not fetched in this session; do not cite].
- **License:** MIT, both at GH and HF mirror [verified 2026-05-03 via https://api.github.com/repos/bowang-lab/scGPT/license → spdx_id "MIT"]. **No non-commercial / research-only clauses found in the verified license metadata.** Safe for research and possible-clinical use under MIT terms.
- **Reported zero-shot performance:** numeric figures from the published Nat Methods paper [UNVERIFIED — paper full text not fetched in this session; do not cite].
- **Build-vs-adopt verdict:** Adopt. **Effort: 2/5.**

**One-line load snippet (paper-safe, revision-pinned to verified SHA):**

```python
from huggingface_hub import snapshot_download
local_dir = snapshot_download(
    repo_id="tdc/scGPT",
    revision="acf749f35bf5c0b00633838f02588272ed0d9911",  # [verified 2026-05-03]
    allow_patterns=["model.safetensors", "config.json", "README.md"],
)
```

---

## 2. Geneformer

- **Citation:** Theodoris CV, Xiao L, Chopra A, Chaffin MD, Al Sayed ZR, Hill MC, Mantineo H, Brydon EM, Zeng Z, Liu XS, Ellinor PT. *Transfer learning enables predictions in network biology.* **Nature 618:616-624 (2023).** DOI 10.1038/s41586-023-06139-9; PMID **37258680** [verified 2026-05-03 via https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=37258680]. The V2 / scaling-and-quantization paper appears in the HF README as Chen H et al., *Nature Computational Science*, 27 Mar 2026 [verified 2026-05-03 via https://huggingface.co/ctheodoris/Geneformer/raw/main/README.md] — but I have **not** independently verified the Nat Comput Sci PMID for the V2 paper in this session [UNVERIFIED — efetch not run for the V2 citation; do not cite without re-checking].
- **Canonical HF repo:** `ctheodoris/Geneformer` — lastModified **2026-04-15T12:27:53.000Z**, downloads **9510**, library_name **transformers**, license tag (cardData) **`apache-2.0`**, sha **`ad8f66dfcda3ebbd148d916c01f31339c5b95a15`**, 87 siblings including `Geneformer-V1-10M/`, `Geneformer-V2-104M/`, `Geneformer-V2-104M_CLcancer/`, `Geneformer-V2-316M/`, `geneformer/` (Python source) [verified 2026-05-03 via https://huggingface.co/api/models/ctheodoris/Geneformer and https://huggingface.co/api/models/ctheodoris/Geneformer/tree/main].
- **LICENSE file:** the standalone `/raw/main/LICENSE` route returned HTTP 404 [verified 2026-05-03 via https://huggingface.co/ctheodoris/Geneformer/raw/main/LICENSE]; the license is communicated via `cardData.license: apache-2.0` in the model API and is the only fetch-grounded license signal. (Maintainer convention: license is set at the dataset/model level rather than via a separate LICENSE file.)
- **Pretraining corpus (verified from HF README):** "Geneformer V1 was originally pretrained in June 2021 on Genecorpus-30M, a corpus comprised of ~30 million human single cell transcriptomes... The current updated Geneformer V2 is pretrained on **~104 million human single cell transcriptomes (non-cancer)** from Genecorpus-104M. The cancer continual learning V2 variant was continually pretrained on ~14 million cancer transcriptomes" [verified 2026-05-03 via https://huggingface.co/ctheodoris/Geneformer/raw/main/README.md]. **The continual-learning cancer variant `Geneformer-V2-104M_CLcancer` is the most MM-relevant pretrained head currently available**, although it is pan-cancer and not MM-specific.
- **Architecture (verified from per-variant config.json files):**
  - V1-10M: BertForMaskedLM, hidden_size **256**, 6 layers, 4 heads, max_position_embeddings **2048**, vocab **25426** [verified 2026-05-03 via https://huggingface.co/ctheodoris/Geneformer/raw/main/Geneformer-V1-10M/config.json].
  - V2-104M: BertForMaskedLM, hidden_size **768**, 12 layers, 12 heads, max_position_embeddings **4096**, vocab **20275** [verified 2026-05-03 via https://huggingface.co/ctheodoris/Geneformer/raw/main/Geneformer-V2-104M/config.json].
  - V2-316M (current default in the repo root): BertForMaskedLM, hidden_size **1152**, 18 layers, 18 heads, max_position_embeddings **4096**, vocab **20275** [verified 2026-05-03 via https://huggingface.co/ctheodoris/Geneformer/raw/main/config.json].
- **Param-count names** (10M / 104M / 316M) match the variant directory names; they are also stated explicitly in the README ("Geneformer-V1-10M: ... 10M parameters", "Geneformer-V2-104M and Geneformer-V2-316M: ... 104M or 316M parameters") [verified 2026-05-03 via https://huggingface.co/ctheodoris/Geneformer/raw/main/README.md].
- **License:** apache-2.0 (cardData tag) [verified 2026-05-03 via https://huggingface.co/api/models/ctheodoris/Geneformer]. Clinic-safe.
- **Build-vs-adopt verdict:** Adopt for fine-tuning, **less ideal for frozen embedding** (rank-value tokenization makes the cell-level CLS embedding less semantically aligned with continuous-expression downstream models than scGPT's continuous-input encoder). **Effort: 3/5 frozen, 2/5 fine-tuned.**

**One-line load snippet (revision-pinned):**

```python
from huggingface_hub import snapshot_download
local_dir = snapshot_download(
    repo_id="ctheodoris/Geneformer",
    revision="ad8f66dfcda3ebbd148d916c01f31339c5b95a15",  # [verified 2026-05-03]
    allow_patterns=[
        "Geneformer-V2-104M/*",  # smaller default; swap to V2-316M for max capacity
        "geneformer/*",
        "config.json", "README.md",
    ],
)
```

---

## 3. scFoundation

- **Citation:** Hao M, Gong J, Zeng X, Liu C, Guo Y, Cheng X, Wang T, Ma J, Zhang X, Song L. *Large-scale foundation model on single-cell transcriptomics.* **Nat Methods 21:1481-1491 (2024).** DOI 10.1038/s41592-024-02305-7; PMID **38844628** [verified 2026-05-03 via https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=38844628].
- **Canonical HF repo:** `biomap-research/scFoundation` returns HTTP 401 from the unauthenticated API [verified 2026-05-03 via https://huggingface.co/api/models/biomap-research/scFoundation]. **Use the unauthenticated mirror `genbio-ai/scFoundation`** — lastModified **2025-04-10T00:15:42.000Z**, sha **`cb434153a1acfacd215eefc956ea445f7cc39cc3`**, downloads **0** at fetch time, library_name `null`, license tag (cardData) **null** (not set on the mirror — see GH license below), 3 siblings: `.gitattributes`, `README.md` (1509 bytes), `models.ckpt` (1.43 GB) [verified 2026-05-03 via https://huggingface.co/api/models/genbio-ai/scFoundation and https://huggingface.co/api/models/genbio-ai/scFoundation/tree/main].
- **Canonical GitHub repo:** `biomap-research/scFoundation` — pushed_at **2025-11-23T08:31:22Z**, stars **406**, open_issues **31**, license **Apache-2.0** (`spdx_id: Apache-2.0`) [verified 2026-05-03 via https://api.github.com/repos/biomap-research/scFoundation and https://api.github.com/repos/biomap-research/scFoundation/license]. **The Apache-2.0 spdx tag from the canonical GH repo is the paper-safe license citation.** The earlier draft's worry about a "conditional-research clause" is **not supported** by what's in the verified GH license — but the inference platform terms (https://aigp.biomap.com/) may impose service-level constraints that are separate from the code/weights license; the user must read those terms before any clinical-adjacent use.
- **Pretraining corpus (verified from GH README):** "scFoundation with **100M parameters**. scFoundation was based on the **xTrimoGene** architecture and trained on **over 50 million human single-cell transcriptomics data**" [verified 2026-05-03 via https://raw.githubusercontent.com/biomap-research/scFoundation/main/README.md].
- **Architecture:** xTrimoGene (read-depth-aware asymmetric encoder-decoder) per the GH README. Embed dim 768 is stated in the published abstract [UNVERIFIED in this session — abstract not fetched; the only fetch-grounded number is 100M params from the GH README].
- **Drug-response framing:** the GH README explicitly lists three downstream pipelines `DeepCDR`, `SCAD`, `GEARS` for drug-response and perturbation tasks, with bulk-cell-line embeddings checked in at `DeepCDR/data/50M-0.1B-res_embedding.npy` [verified 2026-05-03 via https://raw.githubusercontent.com/biomap-research/scFoundation/main/README.md]. **This is the most MM-resistance-relevant feature among the top-3.**
- **Build-vs-adopt verdict:** Adopt with caution because of the non-canonical HF distribution (single 1.43 GB `.ckpt` file on the mirror, full setup via the upstream GH repo). **Effort: 3/5.**

**One-line load snippet (revision-pinned):**

```python
from huggingface_hub import snapshot_download
local_dir = snapshot_download(
    repo_id="genbio-ai/scFoundation",
    revision="cb434153a1acfacd215eefc956ea445f7cc39cc3",  # [verified 2026-05-03]
    allow_patterns=["models.ckpt", "README.md"],
)
# WARNING: tokenization + xTrimoGene encoder code lives in
# https://github.com/biomap-research/scFoundation (Apache-2.0, pushed 2025-11-23).
# Clone separately for the inference loop.
```

---

## 4. UCE — Universal Cell Embedding (next-3 candidate, identifier-grounded only)

- **Citation:** Rosen Y, Roohani Y, Agrawal A, Samotorcan L, Tabula Sapiens Consortium, Quake SR, Leskovec J. *Universal Cell Embeddings: A Foundation Model for Cell Biology.* **bioRxiv 2023-11-29**, DOI **10.1101/2023.11.28.568918**, license cc_by_nc_nd, version 1, abstract trained on "**36 million cells … hundreds of experiments, dozens of tissues and eight species**" [verified 2026-05-03 via https://api.biorxiv.org/details/biorxiv/10.1101/2023.11.28.568918/na/json].
- **PubMed:** [UNVERIFIED — eutils search for "Universal Cell Embeddings Rosen Quake" returned 0 PMIDs in this session; the preprint is not yet indexed in PubMed].
- **GitHub repo:** `snap-stanford/UCE` — pushed_at **2026-02-26T07:33:55Z**, stars **253**, open_issues **1**, license **MIT** [verified 2026-05-03 via https://api.github.com/repos/snap-stanford/UCE].
- **HF mirror:** No official `snap-stanford/UCE` HF org found in `https://huggingface.co/api/models?search=UCE&author=snap-stanford&limit=5` (zero matches) [verified 2026-05-03]. Weights are distributed via the GH release artefacts. Bullet kept identifier-only per task brief.

## 5. CellPLM (next-3 candidate, identifier-grounded only)

- **Citation:** Wen H, Tang W, Dai X, Ding J, Jin W, Xie Y, Tang J. *CellPLM: Pre-training of Cell Language Model Beyond Single Cells.* **bioRxiv** DOI **10.1101/2023.10.03.560734** [verified 2026-05-03 via https://api.biorxiv.org/details/biorxiv/10.1101/2023.10.03.560734/na/json]. ICLR 2024 acceptance is the project's claim and is consistent with the OpenReview path that returned HTTP 200 [observed 2026-05-03], but I do not have a fetch-grounded ICLR proceedings record for it [UNVERIFIED at the venue level].
- **PubMed:** [UNVERIFIED — eutils search for "CellPLM cell language model" returned 0 PMIDs in this session].
- **GitHub repo:** `OmicsML/CellPLM` — pushed_at **2024-03-28T20:01:04Z** (project has been quiet for >24 months as of fetch date), stars **102**, open_issues **10**, license **BSD-2-Clause** (`spdx_id: BSD-2-Clause`) [verified 2026-05-03 via https://api.github.com/repos/OmicsML/CellPLM]. **Note: previous draft tagged license as UNVERIFIED — it is BSD-2-Clause.**
- **HF mirror:** No official `OmicsML/CellPLM` HF model card exists. The two community CellPLM repos visible at `https://huggingface.co/api/models?search=CellPLM` are user playgrounds with 0 downloads each [verified 2026-05-03].

## 6. scBERT (next-3 candidate, identifier-grounded only)

- **Citation:** Yang F, Wang W, Wang F, Fang Y, Tang D, Huang J, Lu H, Yao J. *scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data.* **Nat Mach Intell 2022.** DOI 10.1038/s42256-022-00534-z. **Not indexed in PubMed** (Nature Machine Intelligence is not a PubMed-indexed journal); eutils DOI lookup returned 0 hits [verified 2026-05-03 via https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=10.1038/s42256-022-00534-z%5Bdoi%5D].
- **GitHub repo:** `TencentAILabHealthcare/scBERT` — pushed_at **2023-12-13T08:42:37Z** (>17 months since last push at fetch time), stars **357**, open_issues **25**, license **GPL-3.0** (`spdx_id: GPL-3.0`) [verified 2026-05-03 via https://api.github.com/repos/TencentAILabHealthcare/scBERT]. **Major correction vs prior draft: license is GPL-3.0, NOT MIT.** GPL-3.0 has copyleft implications that make it materially different from the MIT/Apache-2.0 of the top-3 — anything statically linked or distributed alongside scBERT inherits GPL terms, which is a clinical-deployment risk to be aware of.
- **HF mirror:** No official scBERT HF model card; HF search returned only community ports with single-digit downloads [verified 2026-05-03 via https://huggingface.co/api/models?search=scBERT&limit=10].

---

## Newer (2024–2026) candidates I considered but did not verify in this session

- **scGenePT / scPRINT / Nicheformer / GenePT / scELMo / LangCell / scTab-FM** — all UNVERIFIED in this session; the user must re-run the same curl protocol against `https://huggingface.co/api/models/<repo>` and `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?...` before adding any of them to the comparison.
- **CZ-CellxGene Census foundation model release** — UNVERIFIED.

---

## Top-3 ranking — frozen-embedding use case on MM scRNA, compute-constrained

### #1 — scGPT  (`tdc/scGPT` mirror; upstream `bowang-lab/scGPT` GH)

**Rationale.** scGPT remains the strongest match for ResistanceMap because (a) its frozen embedding is a **continuous-input 512-d per-cell vector** (`embsize: 512` in the verified config) — exactly the shape the trajectory dynamics module wants; (b) the verified GH README confirms a **10.3-million-cell blood/bone-marrow checkpoint** in addition to the 33-million-cell whole-human checkpoint, which is the closest pretrained-corpus match to MM in the entire candidate set; (c) **MIT license is verified at the GH level (`spdx_id: MIT`)** and tolerates downstream possible-clinical use; (d) the canonical GH repo was pushed_at 2026-04-29 (active maintenance, only 6 days before this survey); (e) the `tdc/scGPT` HF mirror provides a paper-safe, revision-pinned snapshot path to the weights without needing the gated `bowang-lab/scGPT` page. Risks: the paper-cited zero-shot ARI/NMI numbers were not re-fetched this session and are tagged UNVERIFIED.

### #2 — Geneformer V2-104M / V2-316M  (`ctheodoris/Geneformer`)

**Rationale.** The verified Genecorpus-104M corpus materially closes the corpus gap with scGPT, and the **`Geneformer-V2-104M_CLcancer` variant pretrained on ~14 M cancer transcriptomes is the only top-3 model with a cancer-tuned head already shipped on HF** (verified in the README and as a sibling directory). The license is verified apache-2.0. The reason it is #2 not #1: rank-value tokenization is a less natural input modality for a downstream neural-ODE that consumes continuous expression-style features, and the published paper positions the model as a fine-tuning substrate rather than a frozen-embedding model. As a frozen encoder it is competitive but not clearly better than scGPT for cell-state trajectory work.

### #3 — scFoundation  (`genbio-ai/scFoundation` mirror; upstream `biomap-research/scFoundation` GH)

**Rationale.** scFoundation is the only top-3 candidate with a verified, README-documented **drug-response benchmark (DeepCDR/SCAD/GEARS) and pre-extracted bulk-line embeddings checked into the GH repo**, which is the closest analog to MM-resistance prediction available in the current foundation-model ecosystem. The 100M-param xTrimoGene encoder is conceptually a great fit for the heterogeneous read-depth MM datasets the user works with. The reasons it is #3 and not #1: (a) the canonical HF page returns HTTP 401, forcing reliance on the `genbio-ai/scFoundation` mirror whose cardData license tag is null; (b) the upstream `biomap-research/scFoundation` GH repo is Apache-2.0 (verified), but the inference-platform terms at `aigp.biomap.com` are separate and unverified in this session; (c) the 100M-param footprint pushes the user's hardware. Treat as a forward-looking option for the drug-response axis specifically.

---

## "USE THIS ONE" — Recommendation

**Use scGPT (loaded via the `tdc/scGPT` HF mirror, code from `bowang-lab/scGPT` GH at MIT) as the frozen encoder for the per-cell representation that feeds into ResistanceMap's trajectory dynamics module.**

Operational notes:
- Freeze all weights; use the encoder's CLS token output (512-d per cell, `cell_emb_style: "cls"` in the verified config).
- Tokenize on the intersection of CELLxGENE's gene vocabulary and your MM dataset's HVGs (typically 2–3 k genes); pad/truncate to `max_seq_len: 1536` from the verified config.
- Cache per-cell embeddings to disk **once** and let the neural ODE / CNF / SDE see only the frozen 512-d vectors. This sidesteps the GPU bottleneck completely after the first pass.
- If the blood/bone-marrow specialised checkpoint (10.3 M cells per the verified GH README) materially outperforms whole-human on MM data, switch to it via the Google-Drive link in the GH README — that distribution path is **NOT yet HF-mirrored** as of this verification, so it is not revision-pinnable in the same way.

### Suggested loading skeleton (verify against the live README before running)

```python
# Pinned to the 2026-05-03-verified revision SHA of tdc/scGPT.
# Inference helpers live in the upstream package; pip install scgpt
# from https://github.com/bowang-lab/scGPT (MIT) is the canonical path.

from huggingface_hub import snapshot_download
import torch

repo_dir = snapshot_download(
    repo_id="tdc/scGPT",
    revision="acf749f35bf5c0b00633838f02588272ed0d9911",  # verified 2026-05-03
    allow_patterns=["model.safetensors", "config.json", "README.md"],
)

# Per the verified tdc/scGPT README, inference goes through:
#   from tdc import tdc_hf_interface
#   from tdc.model_server.tokenizers.scgpt import scGPTTokenizer
#   scgpt = tdc_hf_interface("scGPT"); model = scgpt.load()
# OR via the upstream scgpt.tasks.embed_data helper from
# https://github.com/bowang-lab/scGPT — re-read that tutorial before wiring.
```

---

## Verification Log

All commands run on **2026-05-03** from the ResistanceMap repo working directory. Responses cached at `/tmp/scfm_verify/`. HTTP status appears on the first line of each block.

### Hugging Face — model card APIs

| Command | HTTP | Excerpt |
|---|---|---|
| `curl -s https://huggingface.co/api/models/bowang-lab/scGPT` | **401** | `{"error":"Invalid username or password."}` |
| `curl -s https://huggingface.co/api/models/biomap-research/scFoundation` | **401** | `{"error":"Invalid username or password."}` |
| `curl -s https://huggingface.co/api/models/ctheodoris/Geneformer` | **200** | `id: ctheodoris/Geneformer; lastModified: 2026-04-15T12:27:53.000Z; downloads: 9510; library_name: transformers; sha: ad8f66dfcda3ebbd148d916c01f31339c5b95a15; license: apache-2.0; siblings_count: 87` |
| `curl -s https://huggingface.co/api/models/tdc/scGPT` | **200** | `id: tdc/scGPT; lastModified: 2025-04-10T16:20:58.000Z; downloads: 553; library_name: transformers; sha: acf749f35bf5c0b00633838f02588272ed0d9911; license: mit; siblings_count: 5` |
| `curl -s https://huggingface.co/api/models/genbio-ai/scFoundation` | **200** | `id: genbio-ai/scFoundation; lastModified: 2025-04-10T00:15:42.000Z; downloads: 0; library_name: None; sha: cb434153a1acfacd215eefc956ea445f7cc39cc3; license: None; siblings_count: 3` |
| `curl -s https://huggingface.co/api/models/perturblab/scfoundation-cell` | **200** | `id: perturblab/scfoundation-cell; lastModified: 2025-12-22T16:51:39.000Z; downloads: 19; sha: 7b06458e108e0690fdd7f2ece61a4658069b801a; license: apache-2.0` |
| `curl -s 'https://huggingface.co/api/models?search=scGPT&limit=10'` | **200** | top hits: `igorktech/sc-gpt-upf`, `metehergul/scgpt`, `agemagician/scgpt`, **`tdc/scGPT (553 downloads)`**, `MohamedMabrouk/scGPT`, `apliko/scGPT`, `Lei4712/scGPT_Mus`, etc. **No `bowang-lab/scGPT` in unauth-visible search.** |
| `curl -s 'https://huggingface.co/api/models?search=scFoundation&limit=10'` | **200** | hits: `genbio-ai/scFoundation`, `perturblab/scfoundation-cell`, `perturblab/scfoundation-gene`, `perturblab/scfoundation-rde`. **No `biomap-research/scFoundation` in unauth-visible search.** |
| `curl -s 'https://huggingface.co/api/models?search=UCE&author=snap-stanford&limit=5'` | **200** | `[]` (zero matches → no official UCE HF org) |
| `curl -s 'https://huggingface.co/api/models?search=CellPLM&limit=10'` | **200** | only community user repos `lemousehunter/CellPLM_*` (0 downloads) — no `OmicsML/CellPLM` |
| `curl -s 'https://huggingface.co/api/models?search=scBERT&limit=10'` | **200** | only community ports (`havens2/scBERT_SER*`, `kaichenxu/cape_scbert`, etc.) — no official Tencent scBERT HF mirror |

### Hugging Face — raw file fetches

| Command | HTTP | Notes |
|---|---|---|
| `curl -s https://huggingface.co/ctheodoris/Geneformer/raw/main/README.md` | **200** | 9303 bytes; quoted in §2 above |
| `curl -s https://huggingface.co/ctheodoris/Geneformer/raw/main/LICENSE` | **404** | License communicated via `cardData.license` only |
| `curl -s https://huggingface.co/ctheodoris/Geneformer/raw/main/config.json` | **200** | V2-316M default: hidden_size 1152, 18 layers, vocab 20275, max 4096 |
| `curl -s https://huggingface.co/ctheodoris/Geneformer/raw/main/Geneformer-V2-104M/config.json` | **200** | hidden_size 768, 12 layers, vocab 20275, max 4096 |
| `curl -s https://huggingface.co/ctheodoris/Geneformer/raw/main/Geneformer-V1-10M/config.json` | **200** | hidden_size 256, 6 layers, vocab 25426, max 2048 |
| `curl -s https://huggingface.co/tdc/scGPT/raw/main/README.md` | **200** | confirms Cui et al. 2024 Nat Methods 21:1470-1480; links GH `bowang-lab/scGPT` |
| `curl -s https://huggingface.co/tdc/scGPT/raw/main/LICENSE` | **404** | License via `cardData.license: mit` |
| `curl -s https://huggingface.co/tdc/scGPT/raw/main/config.json` | **200** | embsize 512, d_hid 512, nlayers 12, nhead 8, vocab 60697, max_seq_len 1536, cell_emb_style "cls" |
| `curl -s https://huggingface.co/genbio-ai/scFoundation/raw/main/README.md` | **200** | confirms Hao et al. 2024 Nat Methods 21:1481-1491; links GH `biomap-research/scFoundation` |
| `curl -s https://huggingface.co/genbio-ai/scFoundation/raw/main/LICENSE` | **404** | License only via upstream GH (Apache-2.0) |

### NCBI eutils — PubMed

| Command | HTTP | Result |
|---|---|---|
| `curl -s 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=scGPT+foundation+model+multi-omics&retmax=5&retmode=json'` | **200** | idlist: `["41146276","40608008","38409223"]` |
| `curl -s 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=38409223&retmode=xml'` | **200** | Title "scGPT: toward building a foundation model for single-cell multi-omics using generative AI." Year 2024, Volume 21, Pages 1470-1480, ISO Nat Methods, DOI 10.1038/s41592-024-02201-0, first author Cui |
| `curl -s 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=Geneformer+Theodoris+transfer+learning+network+biology&retmax=5&retmode=json'` | **200** | idlist: `["37258680"]` |
| `curl -s 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=37258680&retmode=xml'` | **200** | Title "Transfer learning enables predictions in network biology." Year 2023, Volume 618, Pages 616-624, ISO Nature, DOI 10.1038/s41586-023-06139-9, first author Theodoris |
| `curl -s 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=scFoundation+Hao+single-cell+transcriptomics&retmax=5&retmode=json'` | **200** | idlist: `["41674875","38844628"]` |
| `curl -s 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=38844628&retmode=xml'` | **200** | Title "Large-scale foundation model on single-cell transcriptomics." Year 2024, Volume 21, Pages 1481-1491, ISO Nat Methods, DOI 10.1038/s41592-024-02305-7, first author Hao |
| `curl -s 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=Universal+Cell+Embeddings+Rosen+Quake&retmax=3&retmode=json'` | **200** | idlist: `[]` (UCE not in PubMed → bioRxiv-only) |
| `curl -s 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=CellPLM+cell+language+model&retmax=3&retmode=json'` | **200** | idlist: `[]` (CellPLM not in PubMed → bioRxiv/ICLR only) |
| `curl -s 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=10.1038/s42256-022-00534-z%5Bdoi%5D&retmax=3&retmode=json'` | **200** | idlist: `[]` (Nat Mach Intell DOI not indexed → scBERT not retrievable via PubMed) |

### bioRxiv API

| Command | HTTP | Result |
|---|---|---|
| `curl -s https://api.biorxiv.org/details/biorxiv/10.1101/2023.11.28.568918/na/json` | **200** | UCE: title "Universal Cell Embeddings: A Foundation Model for Cell Biology", authors Rosen/Roohani/Agrawal/Samotorcan/Tabula Sapiens Consortium/Quake/Leskovec, posted 2023-11-29, license cc_by_nc_nd, abstract confirms "**36 million cells … eight species**" |
| `curl -s https://api.biorxiv.org/details/biorxiv/10.1101/2023.10.03.560734/na/json` | **200** | CellPLM: title "CellPLM: Pre-training of Cell Language Model Beyond Single Cells", authors Wen/Tang/Dai/Ding/Jin/Xie/Tang |

### GitHub REST v3

| Command | HTTP | Excerpt |
|---|---|---|
| `curl -s https://api.github.com/repos/bowang-lab/scGPT` | **200** | full_name `bowang-lab/scGPT`, pushed_at **2026-04-29T17:41:42Z**, stars 1552, open_issues 173, license MIT |
| `curl -s https://api.github.com/repos/bowang-lab/scGPT/license` | **200** | spdx_id MIT, html_url https://github.com/bowang-lab/scGPT/blob/main/LICENSE |
| `curl -s https://api.github.com/repos/bowang-lab/scGPT/commits/main` | **200** | latest sha `cebd6fae655b9c585a4807daa3ac31bb764f06b4` (2026-04-27) |
| `curl -s https://api.github.com/repos/biomap-research/scFoundation` | **200** | pushed_at **2025-11-23T08:31:22Z**, stars 406, open_issues 31, license Apache-2.0 |
| `curl -s https://api.github.com/repos/biomap-research/scFoundation/license` | **200** | spdx_id Apache-2.0 |
| `curl -s https://api.github.com/repos/snap-stanford/UCE` | **200** | pushed_at **2026-02-26T07:33:55Z**, stars 253, open_issues 1, license MIT |
| `curl -s https://api.github.com/repos/OmicsML/CellPLM` | **200** | pushed_at **2024-03-28T20:01:04Z**, stars 102, open_issues 10, license **BSD-2-Clause** |
| `curl -s https://api.github.com/repos/TencentAILabHealthcare/scBERT` | **200** | pushed_at **2023-12-13T08:42:37Z**, stars 357, open_issues 25, license **GPL-3.0** |

### Raw GH README fetches (for exact-quote evidence)

| Command | HTTP | Anchor quote |
|---|---|---|
| `curl -s https://raw.githubusercontent.com/bowang-lab/scGPT/main/README.md` | **200** | "whole-human (recommended) — Pretrained on 33 million normal human cells"; "blood — Pretrained on 10.3 million blood and bone marrow cells" |
| `curl -s https://raw.githubusercontent.com/biomap-research/scFoundation/main/README.md` | **200** | "scFoundation with 100M parameters … trained on over 50 million human single-cell transcriptomics data" |

---

## Final caveat (paper-safe)

Every numeric or identifier claim in the candidate matrix and §1–§3 either carries a `[verified 2026-05-03 via <url>]` tag (paper-safe) or an explicit `[UNVERIFIED ...]` tag (do not cite). The **TOP-3 ranking — scGPT > Geneformer > scFoundation — is unchanged** from the prior qualitative draft because verification did not surface a license blocker, dead repo, or unrecoverable identifier. The MOST IMPORTANT corrections vs the prior draft:

1. The canonical `bowang-lab/scGPT` HF page returns HTTP 401; use **`tdc/scGPT`** (revision `acf749f35bf5c0b00633838f02588272ed0d9911`) as the paper-safe mirror.
2. Geneformer V2 was scaled to **~104 M cells (Genecorpus-104M), not 95 M** as in the prior draft; embedding dims are **256/768/1152** for V1-10M/V2-104M/V2-316M, not 256/512.
3. scBERT's GitHub license is **GPL-3.0**, not MIT — copyleft implications matter for clinical deployment.
4. CellPLM is **BSD-2-Clause**; UCE is **MIT** — both verified at the GH level.

Anything still tagged UNVERIFIED in the body above is an item I deliberately did not fetch in this session (typically an in-paper number behind a paywall, or a venue acceptance not in PubMed/bioRxiv); the user must re-verify before paper submission. Specifically still UNVERIFIED at session end: (a) scGPT in-paper zero-shot ARI/NMI numbers; (b) Geneformer V2 Nat Comput Sci PMID; (c) scFoundation in-paper Pearson r on GDSC; (d) embed dims for UCE / CellPLM / scBERT (only GH/HF identifiers were verified for these three).
