# Drug-Response Model Landscape and Porting Ranking

**Date of survey:** 2026-05-03 (re-verified pass)
**Author:** Literature-curator agent
**Context:** ResistanceMap v8 is producing test_mse 2.84 vs zero-floor 2.81. The model is not learning. We need to know which published drug-response models are worth porting or absorbing components from.

---

## Verification policy

Every PMID, DOI, GitHub URL, and license below was re-fetched from a public REST endpoint
on 2026-05-03 (NCBI eutils, api.github.com, raw.githubusercontent.com). Each identifier
is annotated with the URL and the date the value was confirmed. Values that could not
be grounded in a fetched source are flagged `[UNVERIFIED — do not cite]` and must not
be quoted in any paper, slide, or release note without further sourcing.

The previous draft made several factual errors that this pass corrects (see "Errata"
section at the end).

---

## Candidate models — one row per model

| # | Model | Year | Venue | PMID / DOI | GitHub | License | Drug features | Multitask across drugs | Port effort (1-5) | Marginal value over ElasticNet on ResistanceMap |
|---|-------|------|-------|------------|--------|---------|---|---|---|---|
| 1 | DrugCell | 2020 | Cancer Cell 38(5):672-684.e6 | PMID 33096023 / DOI 10.1016/j.ccell.2020.09.014 [verified 2026-05-03 via eutils esearch+efetch] | idekerlab/DrugCell [verified 2026-05-03 via api.github.com/repos/idekerlab/DrugCell] | MIT [verified 2026-05-03 via api.github.com/repos/idekerlab/DrugCell/license] | Morgan fingerprints (radius 2, 2048-bit) [verified via repo README] | Yes (drug-conditioned single model) | 2 | High |
| 2 | MOFA+ | 2020 | Genome Biology 21:111 | PMID 32393329 / DOI 10.1186/s13059-020-02015-1 [verified 2026-05-03 via eutils] | bioFAM/MOFA2 [verified 2026-05-03 via api.github.com] | LGPL-3.0 [verified 2026-05-03 via api.github.com/repos/bioFAM/MOFA2/license] | N/A (factor extractor) | N/A (unsupervised) | 1 | High as feature extractor |
| 3 | PaccMann | 2019 | Mol. Pharm. 16(12):4797-4806 | PMID 31618586 / DOI 10.1021/acs.molpharmaceut.9b00520 [verified 2026-05-03 via eutils] | PaccMann/paccmann_predictor [verified 2026-05-03 via api.github.com] | MIT [verified 2026-05-03 via api.github.com/repos/PaccMann/paccmann_predictor/license] | SMILES (multimodal attention CNN) | Yes | 3 | High |
| 4 | MOLI | 2019 | Bioinformatics 35(14):i501-i509 (ISMB) | PMID 31510700 / DOI 10.1093/bioinformatics/btz318 [verified 2026-05-03 via eutils] | hosseinshn/MOLI [verified 2026-05-03 via api.github.com] | NONE (no LICENSE file) [verified 2026-05-03 via api.github.com/repos/hosseinshn/MOLI] | None (drug-specific) | No (one model per drug) | 3 | Medium |
| 5 | DeepCDR | 2020 | Bioinformatics 36(Suppl_2):i911-i918 | PMID 33381841 / DOI 10.1093/bioinformatics/btaa822 [verified 2026-05-03 via eutils — original draft had wrong PMID 33381848] | kimmo1019/DeepCDR [verified 2026-05-03 via api.github.com] | MIT [verified 2026-05-03 via api.github.com] | GIN over molecular graph (CCLE + multi-omics) | Yes | 3 | High |
| 6 | DeepDRA | 2024 | PLoS ONE | PMID 39058696 / DOI 10.1371/journal.pone.0307649 [verified 2026-05-03 via eutils — original draft cited wrong DOI 10.1371/journal.pone.0298036] | bcb-sut/DeepDRA [verified 2026-05-03 via api.github.com] | NONE (no LICENSE file) [verified 2026-05-03 via api.github.com] | Mol fingerprints + descriptors | Yes (multi-task autoencoder) | 3 | Medium |

The previous draft included rows for SWnet, TCRP, DeepTTA, and a few placeholder
families. None of those were re-verified in this pass and they are out of scope for
the current ranking; if you need them in a paper, run a fresh per-row verification.

---

## Per-candidate detail (TOP-3 — every claim grounded)

### 1. DrugCell — Kuenzi et al. 2020

- **Citation:** Kuenzi BM, Park J, Fong SH, et al. "Predicting Drug Response and Synergy
  Using a Deep Learning Model of Human Cancer Cells." *Cancer Cell* 38(5):672-684.e6,
  2020.
  - PMID **33096023** [verified 2026-05-03 via `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=Kuenzi+DrugCell+2020+drug+response&retmax=3&retmode=json`]
  - DOI **10.1016/j.ccell.2020.09.014** [verified 2026-05-03 via efetch XML for PMID 33096023]
  - Title verbatim from PubMed efetch: "Predicting Drug Response and Synergy Using a
    Deep Learning Model of Human Cancer Cells."
- **Code:** `github.com/idekerlab/DrugCell` (default branch: `public`).
  - License: **MIT** [verified 2026-05-03 via `https://api.github.com/repos/idekerlab/DrugCell/license`]
  - Last push: **2023-09-13** [verified 2026-05-03 via api.github.com `/repos/idekerlab/DrugCell` `pushed_at` field]
- **Architecture:** Visible neural network whose hidden structure mirrors the Gene
  Ontology, concatenated with a Morgan-fingerprint MLP for the drug, then a final MLP
  to predict response. Drug encoding: Morgan radius-2, 2048-bit fingerprints. Cell
  encoding: binary mutation vector over the top 15% most-frequently mutated genes
  (n = 3,008). [All verified 2026-05-03 via raw README at
  `https://raw.githubusercontent.com/idekerlab/DrugCell/public/README.md`]
- **Training data (verified from README):** **509,294 (cell line, drug) pairs across
  1,235 tumor cell lines and 684 drugs**, sourced from GDSC and CTRP v2.
  *Errata: the previous draft said "684 cell lines × 684 drugs" which was wrong by
  about 2x on the cell-line count.*
- **Reported metric:** The PubMed abstract (verified 2026-05-03 via efetch) states
  only that "DrugCell predictions are accurate in cell lines and also stratify
  clinical outcomes" — it does not quote a specific Spearman/Pearson value. **Any
  specific decimal (e.g., the previously cited "approx 0.50") is `[UNVERIFIED — do
  not cite]` until pulled from the paper PDF or a supplementary table.**
- **Port effort:** 2 — clean PyTorch, ontology shipped. Replace the GDSC AUC target
  with ResistanceMap's response and adapt the genotype side (mutation-only in the
  original) to multi-omics.
- **Marginal value:** **High.** It contributes the missing piece — a drug-conditioned
  shared model with structural drug features. With 692 samples × 11 drugs, the
  multitask drug-conditioned formulation is essential to share statistical strength
  across drugs.

### 2. MOFA+ — Argelaguet et al. 2020

- **Citation:** Argelaguet R, Arnol D, Bredikhin D, Deloro Y, Velten B, Marioni JC,
  Stegle O. "MOFA+: a statistical framework for comprehensive integration of
  multi-modal single-cell data." *Genome Biology* 21:111, 2020.
  - PMID **32393329** [verified 2026-05-03 via eutils esearch on Argelaguet+MOFA+2020]
  - DOI **10.1186/s13059-020-02015-1** [verified 2026-05-03 via efetch XML for PMID 32393329]
  - Title verbatim from PubMed efetch: "MOFA+: a statistical framework for
    comprehensive integration of multi-modal single-cell data."
- **Code:** `github.com/bioFAM/MOFA2` (R + Python via `mofapy2`).
  - License: **LGPL-3.0** [verified 2026-05-03 via `https://api.github.com/repos/bioFAM/MOFA2/license`]
  - Last push: **2026-02-23** [verified 2026-05-03 via api.github.com] — actively maintained.
- **Architecture:** Bayesian group factor analysis with sparsity priors and
  computationally efficient variational inference. Decomposes K modalities × N
  samples into a shared factor matrix Z and modality-specific weights. Verified
  description from PubMed abstract (efetch 2026-05-03): "MOFA+ reconstructs a
  low-dimensional representation of the data using computationally efficient
  variational inference and supports flexible sparsity constraints, allowing to
  jointly model variation across multiple sample groups and data modalities."
- **Inputs:** Any number of (samples × features) matrices, possibly with missing
  entries.
- **Reported metric:** **N/A** — MOFA+ is unsupervised; the abstract reports no
  predictive metric. Downstream evaluation in the paper is by variance explained per
  factor; specific values would require the PDF.
- **Port effort:** 1 — `pip install mofapy2`, fit on the 4 modalities, dump Z, feed
  Z to a downstream regressor.
- **Marginal value:** **High** as a feature extractor (not as a predictor). With 692
  samples and 4 modalities, MOFA+ would give a 20-50 dim factor representation that
  ElasticNet/LightGBM can regress on. Cheapest experiment to run; result is
  decision-relevant either way.

### 3. PaccMann — Manica et al. 2019

- **Citation:** Manica M, Oskooei A, Born J, Subramanian V, Sáez-Rodríguez J,
  Rodríguez Martínez M. "Toward Explainable Anticancer Compound Sensitivity
  Prediction via Multimodal Attention-Based Convolutional Encoders." *Mol. Pharm.*
  16(12):4797-4806, 2019.
  - PMID **31618586** [verified 2026-05-03 via efetch XML — title and authors match
    exactly]
  - DOI **10.1021/acs.molpharmaceut.9b00520** [verified 2026-05-03 via efetch
    `<ELocationID EIdType="doi">` field]
  - Title verbatim from PubMed efetch: "Toward Explainable Anticancer Compound
    Sensitivity Prediction via Multimodal Attention-Based Convolutional Encoders."
- **Code:** `github.com/PaccMann/paccmann_predictor` (PyTorch reimplementation of the
  best architecture).
  - License: **MIT** [verified 2026-05-03 via `https://api.github.com/repos/PaccMann/paccmann_predictor/license`]
  - Last push: **2026-02-11** [verified 2026-05-03 via api.github.com] — actively maintained.
  - README confirms: "this is the `pytorch` implementation of the best PaccMann
    architecture (multiscale convolutional encoder)" and links the Mol Pharm 2019
    DOI [verified 2026-05-03 via raw README].
  - Pretrained weights are linked from the README at `https://ibm.biz/paccmann-data`.
- **Architecture:** Multimodal attention-based convolutional encoder. Inputs are
  SMILES + gene expression + protein-protein interaction priors. Verified description
  from the PubMed abstract (efetch 2026-05-03): "Our model is based on the three key
  pillars of drug sensitivity: compounds' structure in the form of a SMILES sequence,
  gene expression profiles of tumors, and prior knowledge on intracellular
  interactions from protein-protein interaction networks."
- **Reported metric:** **The PubMed abstract gives R² = 0.86 and RMSE = 0.89, but
  these are the *baseline / previous state-of-the-art* numbers that PaccMann
  outperforms — NOT PaccMann's own metrics.** Verbatim from the abstract: "We
  demonstrate that our multiscale convolutional attention-based encoder significantly
  outperforms a baseline model trained on Morgan fingerprints and a selection of
  encoders based on SMILES, as well as the previously reported state-of-the-art for
  multimodal drug sensitivity prediction (R² = 0.86 and RMSE = 0.89)." The abstract
  does not quote PaccMann's own R²/RMSE — those would have to be pulled from the PDF
  tables. *Errata: the previous draft attributed "Pearson approx 0.86 / RMSE approx
  0.89" to PaccMann itself, which is the reverse of what the abstract says.* For now,
  PaccMann's own numerical performance is `[UNVERIFIED — do not cite]`.
- **Port effort:** 3 — modular PyTorch; older transformers/torch stack and a custom
  SMILES tokenizer. Pretrained encoders are available, which makes it tractable.
- **Marginal value:** **High.** PaccMann is the closest published match to
  ResistanceMap's intended formulation (drug-conditioned + multi-omics + attention).
  Adapting its expression-centric gene side to proteomics/CRISPR/epigenome is the
  major fork required.

---

## Per-candidate detail (priority 4-6 — citations + GitHub only, no metric deep-dive)

### 4. MOLI — Sharifi-Noghabi et al. 2019

- PMID **31510700** / DOI **10.1093/bioinformatics/btz318** [verified 2026-05-03 via
  eutils — title returned: "MOLI: multi-omics late integration with deep neural
  networks for drug response prediction."]
- *Bioinformatics* 35(14) ISMB proceedings.
- Code: `github.com/hosseinshn/MOLI` [verified 2026-05-03 via api.github.com].
  License: **NONE** (no LICENSE file in repo) [verified 2026-05-03 via api.github.com].
  Last push: **2020-09-22** — unmaintained.
- Architecture (from abstract, verified via efetch): three modality-specific encoder
  sub-networks for somatic mutation, copy number aberration, and gene expression;
  late concatenation; combined triplet + binary cross-entropy loss. Tested on five
  chemotherapy + two targeted agents.
- AUROC values from prior draft (0.74-0.92 range) are `[UNVERIFIED — do not cite]`;
  the abstract does not quote them. They would have to be pulled from the paper's
  Tables 1-2.

### 5. DeepCDR — Liu et al. 2020

- PMID **33381841** / DOI **10.1093/bioinformatics/btaa822** [verified 2026-05-03 via
  eutils — title returned: "DeepCDR: a hybrid graph convolutional network for
  predicting cancer drug response."]
  - **Errata:** the previous draft cited PMID 33381848, which is wrong (off by 7).
    The correct PMID is **33381841**.
- *Bioinformatics* 36(Suppl_2):i911-i918, ECCB 2020 proceedings.
- Code: `github.com/kimmo1019/DeepCDR` [verified 2026-05-03 via api.github.com].
  License: **MIT** [verified via api.github.com]. Last push: **2023-06-26** — partly
  maintained.
- Architecture (verified from README + abstract via efetch): hybrid GCN — uniform
  graph convolutional network over the molecular graph + multiple sub-networks over
  multi-omics (mutation, expression, methylation from CCLE). Includes both
  classification and regression heads.
- Reported Pearson/RMSE numbers from prior draft are `[UNVERIFIED — do not cite]`;
  the abstract only states "DeepCDR outperformed state-of-the-art methods in both
  classification and regression settings" without quoting decimals.

### 6. DeepDRA — Mohammadzadeh-Vardin et al. 2024

- **Errata:** the previous draft claimed DeepDRA had "DOI 10.1371/journal.pone.0298036"
  with the citation flagged as uncertain. The correct identifiers are below; the
  guessed DOI was **wrong**.
- PMID **39058696** / DOI **10.1371/journal.pone.0307649** [verified 2026-05-03 via
  eutils — title returned: "DeepDRA: Drug repurposing using multi-omics data
  integration with autoencoders."]
- *PLoS ONE* 2024. Authors: Mohammadzadeh-Vardin, Ghareyazi, Gharizadeh, Abbasi,
  Rabiee. Affiliation: Bioinformatics and Computational Biology Lab, Sharif
  University of Technology, Tehran.
- Code: `github.com/bcb-sut/DeepDRA` [verified 2026-05-03 via api.github.com].
  License: **NONE** (no LICENSE file) [verified 2026-05-03 via api.github.com].
  Last push: **2024-03-21**.
- Reported metrics from abstract (verified via efetch 2026-05-03): "AUPRC of 0.99"
  on in-dataset (GDSC/CTRP/CCLE) and "AUPRC of 0.72" on cross-dataset (train GDSC,
  test CCLE). These are abstract-level claims, not PDF-table grounded — treat as a
  headline number, not a benchmark.

---

## Top-3 ranked recommendation (after re-verification)

The TOP-3 ranking from the prior draft survives re-verification. All three citations,
DOIs, GitHub URLs, and licenses are now grounded in fetched sources. The recommendation
is unchanged.

### #1 — DrugCell (Kuenzi 2020) [PMID 33096023, MIT]

**Rationale:** ResistanceMap's biggest known gap is the absence of any drug
representation, so it cannot share information across the 11 drugs. DrugCell is the
lowest-effort path to closing that gap because (a) the code is clean PyTorch under
MIT (verified above), (b) Morgan fingerprints are a one-line RDKit call so no SMILES
tokenizer or graph library is needed, (c) it is a single drug-conditioned model —
ideal for 692 × 11, and (d) the GO-ontology genotype side is interpretable, aligning
with ResistanceMap's stated emphasis on biological interpretability. The
mutation-only bottleneck of the original is the weakness, but replacing the mutation
MLP with the existing ResistanceMap proteomics+epigenome+CRISPR encoder is exactly
the kind of surgery a 2-effort port should look like. **Caveat for paper writing:**
do not cite a specific Spearman/Pearson number for DrugCell from this document —
the abstract does not give one, and the previous draft's "approx 0.50" is unverified.
Pull the value from Table 2 of the paper before quoting. Budget: one engineering week.

### #2 — MOFA+ (Argelaguet 2020) [PMID 32393329, LGPL-3.0] — as feature extractor, not as model

Run MOFA+ on the four modalities, dump 20-50 latent factors, regress on
(factor, drug-onehot) with ElasticNet and LightGBM. 1-day experiment. Tells us
whether ResistanceMap's failure is in the encoder (MOFA+ baseline beats current model
→ encoder is the problem) or in the supervised head + lack of drug features (MOFA+
also at zero-floor → confirms the supervised side is where to invest). Either outcome
is decision-relevant. Effort: 1.

**LGPL-3.0 caveat:** unlike the MIT-licensed candidates, MOFA2 is LGPL-3.0. Using
`mofapy2` as a library dependency (the standard usage) is fine; copying source code
into ResistanceMap is not.

### #3 — PaccMann (Manica 2019) [PMID 31618586, MIT]

After DrugCell + MOFA+ are wired in, PaccMann adds a SMILES-attention encoder
(richer than Morgan fingerprints), drug-genotype contextual attention (interpretable),
and ships pretrained weights via `ibm.biz/paccmann-data`. Effort: 3 engineering
weeks. Use this as the v9 upgrade. **Caveat for paper writing:** the R² = 0.86,
RMSE = 0.89 numbers in the PaccMann abstract are *baselines that PaccMann beats*,
not PaccMann itself — do not quote them as PaccMann's performance.

---

## Errata vs. the previous draft

| Item | Previous draft said | Correct (verified 2026-05-03) |
|---|---|---|
| DrugCell training data | "684 cell lines × 684 drugs" | **1,235 cell lines × 684 drugs**, 509,294 pairs (per repo README) |
| DrugCell test Spearman | "approx 0.50" | **Not in the abstract — UNVERIFIED, do not cite without PDF** |
| PaccMann reported metric | "Pearson approx 0.86 / RMSE approx 0.89" attributed to PaccMann | These are the **baseline** numbers PaccMann beats, NOT PaccMann's own |
| DeepCDR PMID | 33381848 | **33381841** (off by 7; original was a typo or hallucination) |
| DeepDRA DOI | 10.1371/journal.pone.0298036 (flagged uncertain) | **10.1371/journal.pone.0307649** (PMID 39058696) |
| DeepDRA GitHub | "ssghost/DeepDRA (or fork)" | **bcb-sut/DeepDRA** (Sharif University BCB lab — matches paper authors) |
| MOLI metric | "AUROC approx 0.74-0.92" | Not in the abstract — UNVERIFIED, do not cite |
| DeepCDR metric | "Pearson approx 0.92 / RMSE approx 1.06" | Not in the abstract — UNVERIFIED, do not cite |
| MOLI license | "no LICENSE last I checked" | Confirmed: **NONE** (no LICENSE file) — using/distributing this code in a publication carries a real legal risk |

---

## Items still UNVERIFIED in this document

1. **All numerical performance metrics** (Spearman, Pearson, RMSE, AUROC, AUPRC) for
   DrugCell, MOLI, DeepCDR. The PubMed abstracts do not quote them. Pull from the
   paper PDFs / supplementary tables before citing.
2. **DeepDRA AUPRC (0.99 in-set, 0.72 cross-set)** — these are from the abstract,
   not from a fetched table — treat as abstract-grade, not benchmark-grade.
3. **The follow-up "PaccMann iScience 2021" reference** mentioned in the previous
   draft — not re-verified in this pass; if needed, re-run an esearch on
   "Born PaccMann iScience 2021" before citing.
4. **SWnet, TCRP, DeepTTA rows** that appeared in the previous draft — dropped from
   this pass. If you need them, run a fresh per-row verification.

## Action items for the next pass

1. Pull the DrugCell, DeepCDR, and PaccMann full-text PDFs and replace every
   `[UNVERIFIED — do not cite]` metric with the exact value plus a table/page
   reference.
2. Confirm the Born et al. 2021 iScience PaccMann follow-up if it ends up in the
   manuscript.
3. Decide whether to keep MOLI in scope given the no-LICENSE status.
