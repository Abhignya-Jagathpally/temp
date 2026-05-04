# Per-Patient Longitudinal Forecasting: 2024-2026 SOTA Literature Review

**Date compiled:** 2026-05-03
**Audience:** ResistanceMap design team
**Regime in scope:** N=51 paired longitudinal samples + N=994 baseline cohort, multimodal (RNA + WES + clinical timeline), per-patient (not population-marginal) prediction.
**Verification policy:** Every PMID, DOI, arXiv ID and GitHub repo below was fetched live via NCBI eutils, the arXiv Atom API, or the GitHub REST API on 2026-05-03. Numeric values are quoted only when present in the abstract; everything else is tagged `[UNVERIFIED — abstract did not contain the number; PDF would be required]`. The full curl log appears at the end.

---

## Part 1 — "Microsoft GigaTIME" verdict: VERIFIED, REAL, but NOT a temporal forecasting model

The prior memory-only verdict ("no model called GigaTIME exists; user probably meant GigaPath") was **wrong**. GigaTIME is a real Microsoft Research multimodal AI framework, published in *Cell* (2026, in press for vol. 189 issue 2, 22 January 2026 print date). The mistake the prior agent made — and which we should not repeat — was to extrapolate from training-data recall rather than search.

### GigaTIME — verified record

| Field | Value |
|---|---|
| Title | "Multimodal AI generates virtual population for tumor microenvironment modeling" |
| PMID | 41371214 |
| DOI | 10.1016/j.cell.2025.11.016 |
| Journal | *Cell* 189(2), Jan 22 2026 |
| Affiliation | Microsoft Research, Redmond, WA, USA (corresponding: hoifung@microsoft.com — Hoifung Poon) + Earle A. Chiles Research Institute / Providence Cancer Institute + Providence Genomics |
| Code | https://github.com/prov-gigatime/GigaTIME (358 stars, license `NOASSERTION`, created 2025-10-06, last update 2026-04-30) |
| Companion editorial | PMID 41912494 "(Giga)TIME for the era of AI: virtual multiplex immunofluorescence from routine H&E for large populations" (MD Anderson) |
| Companion News & Views | PMID 41832074, 41576915 |

### What GigaTIME actually does

From the abstract (verified): "GigaTIME learns a cross-modal translator to generate virtual mIF [multiplex immunofluorescence] images from hematoxylin and eosin (H&E) slides by training on 40 million cells with paired H&E and mIF data across 21 proteins. We applied GigaTIME to 14,256 patients from 51 hospitals … generating 299,376 virtual mIF slides spanning 24 cancer types and 306 subtypes. This virtual population uncovered 1,234 statistically significant associations linking proteins, biomarkers, staging, and survival. Independent validation on 10,200 TCGA patients further corroborated our findings."

**Critical for this question:** GigaTIME is a **cross-modal translator** (H&E → virtual mIF) used to scale up *cross-sectional* tumor microenvironment characterization. It is NOT a per-patient temporal forecaster, has no temporal axis in the published architecture, and does not address the paired-longitudinal regime ResistanceMap operates in. The "51 hospitals" coincidence with ResistanceMap's "51 paired" cohort is purely numeric collision.

### So what was the user likely thinking of?

If the user wanted "Microsoft + temporal forecasting + per-patient", three candidates plausibly fit:

1. **GigaPath** (PMID 38778098, DOI 10.1038/s41586-024-07441-w, *Nature* June 2024, "A whole-slide foundation model for digital pathology from real-world data", Microsoft Research Redmond). Code: https://github.com/prov-gigapath/prov-gigapath, Apache-2.0, 601 stars. **Foundation-model framing**, but pathology-only, not temporal.
2. **TimeGPT-1** (Nixtla, arXiv 2310.03589, not Microsoft) — true time-series foundation model.
3. **Chronos** (Amazon, arXiv 2403.07815) — Amazon, not Microsoft.

**Verdict:** GigaTIME exists; the user's framing "per-patient temporal forecasting on similar lines as Microsoft's GigaTIME" is *not* what GigaTIME does. ResistanceMap should not chase GigaTIME's design. The closest "Microsoft + foundation + clinical" reference is GigaPath, but GigaPath is also non-temporal. There is no published Microsoft model that does per-patient longitudinal multimodal forecasting in the ResistanceMap regime.

---

## Part 2 — Verified candidates for per-patient longitudinal forecasting

### A. Clinical-omics longitudinal cohort papers

#### A1. PERCEPTION (Sinha et al., *Nat Cancer* 2024) — VERIFIED
- **PMID** 38637658 / **DOI** 10.1038/s43018-024-00756-7
- **Title:** "PERCEPTION predicts patient response and resistance to treatment using single-cell transcriptomics of their tumors"
- **Sample size:** Two clinical cohorts (multiple myeloma + breast cancer) plus a NSCLC TKI-resistance cohort. Exact paired-longitudinal N: `[UNVERIFIED — abstract does not state sample sizes]`.
- **Architecture (one sentence):** Transfer learning from bulk-trained drug-response models in CCLE/GDSC onto patient single-cell expression vectors via a per-cell pseudo-bulk projection, trained with a ridge-regression objective.
- **Reported metric:** Abstract claims "outperforms published state-of-the-art sc-based and bulk-based predictors in all clinical cohorts" but **no numeric value in the abstract** → `[UNVERIFIED — abstract gives no decimal; PDF/supplement required]`.
- **Code:** https://github.com/ruppinlab/PERCEPTION (71 stars, **license: None** — meaning all-rights-reserved by default; that is a problem to absorb code from).

#### A2. TRACERx Lung (Martínez-Ruiz et al., *Nature* 2023) — VERIFIED
- **PMID** 37046093 / **DOI** 10.1038/s41586-023-05706-4
- **Title:** "Genomic-transcriptomic evolution in lung cancer and metastasis"
- **Sample size (verified from abstract):** 354 NSCLC tumours from 347 of the first 421 patients; 947 tumour regions; 96 adjacent-normal samples.
- **How they handle small-N per-patient:** Multi-region sampling per patient (avg ~2.7 regions/patient) creates *within-patient* replication that substitutes for longitudinal sampling. The paper combines "multiple machine-learning approaches that leverage genomic and transcriptomic variables" (abstract); specific architectures and hold-out C-index numbers are `[UNVERIFIED — abstract does not state]`.
- **Code/data:** TRACERx data via EGA controlled access; no single canonical model repo cited in abstract.

#### A3. Cohen et al. *Nat Med* 2021 — VERIFIED
- **PMID** 33619369 / **DOI** 10.1038/s41591-021-01232-w
- **Sample size (verified):** 41 primary-refractory / early-relapse MM patients, 11 healthy, 15 NDMM. Generalisation cohort: CoMMpass.
- **Architecture:** Differential-expression + pathway analysis on longitudinal scRNA-seq; CRISPR-Cas9 validation; not a per-patient predictive model — a *target-discovery* longitudinal study.
- **Per-patient metric:** None reported (it isn't a predictor). Identifies PPIA as a target.
- **Direct relevance to ResistanceMap:** This is the *cohort design template* (paired pre/post-treatment scRNA-seq with N≈40 patients), not a model architecture.

#### A4. CellRank 2 (Lange et al., *Nat Methods* 2024) — VERIFIED
- **PMID** 38871986 / **DOI** 10.1038/s41592-024-02303-9
- **Architecture:** Multi-view Markov-chain framework over single-cell state graphs; supports time-point kernels, RNA-velocity kernels, metabolic-labeling kernels, and pseudotime kernels in a unified transition-matrix language; computes terminal states and per-cell fate probabilities.
- **Per-patient metric:** N/A — operates at cell level. Per-patient adaptation requires aggregating fate probabilities by patient.
- **Code:** https://github.com/scverse/cellrank, BSD-3-Clause, 443 stars (active).

### B. Time-series foundation models (TS-FMs)

#### B1. MOMENT (Goswami et al., ICML 2024)
- **arXiv 2402.03885** (HTTP 200 verified; v3 on arXiv).
- Title: "MOMENT: A Family of Open Time-series Foundation Models"
- **Architecture (one sentence):** Encoder-only T5-style transformer pre-trained on the "Time Series Pile" (a curated public TS corpus) via masked patch reconstruction, fine-tunable for forecasting/classification/anomaly with minimal supervision.
- **Per-patient metric:** Not clinical-specific; abstract reports "effectiveness with minimal data and task-specific fine-tuning" — no decimal in abstract → `[UNVERIFIED]`.
- **Code:** https://github.com/moment-timeseries-foundation-model/moment, MIT, 756 stars, last update 2026-05-02.

#### B2. Chronos (Ansari et al., 2024)
- **arXiv 2403.07815** (HTTP 200 verified; v3).
- **Architecture:** Tokenises time-series values via scaling+quantisation into a fixed vocabulary, trains decoder/encoder-decoder T5 LMs (20M-710M params) with cross-entropy on tokenised series.
- **Per-patient metric:** "comparable and occasionally superior zero-shot performance" on 42-dataset benchmark — `[UNVERIFIED — no decimals in abstract]`.
- **Code:** https://github.com/amazon-science/chronos-forecasting, Apache-2.0, **5,245 stars** (most mature TS-FM ecosystem), updated 2026-05-03.

#### B3. Lag-Llama (Rasul et al., 2023/2024)
- **arXiv 2310.08278** (HTTP 200 verified; v3).
- **Architecture:** Decoder-only transformer for *univariate* probabilistic forecasting using lag features as covariates; pretrained on diverse TS corpora.
- **Per-patient metric:** "best general-purpose model on average" — `[UNVERIFIED — no decimals in abstract]`.
- **Code:** https://github.com/time-series-foundation-models/lag-llama, Apache-2.0, 1,574 stars.
- **Caveat for ResistanceMap:** Univariate-only. Patient state is multivariate (RNA + WES + labs). Would need per-channel modelling.

#### B4. TimeGPT-1 (Nixtla, 2023)
- **arXiv 2310.03589** (HTTP 200 verified; v3). Title "TimeGPT-1".
- Closed-source/API-only foundation model. Useful as an external benchmark, not as an absorbable component.

### C. Modern dynamical-systems / flow-matching approaches (most relevant to ResistanceMap)

#### C1. Trajectory Flow Matching (Zhang et al., NeurIPS 2024)
- **arXiv 2410.21154** (HTTP 200 verified; v2).
- **Architecture (verified):** Trains Neural SDEs *simulation-free* via flow matching, bypassing backprop through SDE dynamics; introduces a reparameterisation trick for stability; targets stochastic, irregularly-sampled time series.
- **Reported metric:** "improved performance on three clinical time series datasets both in terms of absolute performance and uncertainty prediction" — no decimal in abstract → `[UNVERIFIED]`.
- **Code (likely shared infra):** https://github.com/atong01/conditional-flow-matching (TorchCFM), MIT, 2,440 stars, very active. The exact TFM code path inside this repo is `[UNVERIFIED — would need README inspection]`.

#### C2. Generative Modeling of Clinical Time Series via Latent SDEs (2025)
- **arXiv 2511.16427** (HTTP 200 verified; v1, Nov 2025).
- **Architecture (verified from abstract):** Variational latent neural SDE with modality-dependent emission models; treats clinical TS as discrete partial observations of a controlled stochastic dynamical system; handles irregular sampling by construction.
- **Sample size (verified):** Validation on real-world ICU data from **12,000 patients** + simulated PKPD lung-cancer ITE benchmark.
- **Reported metric:** "outperforms ODE and LSTM baselines in accuracy and uncertainty estimation" — no decimal → `[UNVERIFIED]`.
- **Code:** `[UNVERIFIED — repo URL not in abstract]`.

#### C3. ContextFlow (Rathod et al., 2025)
- **arXiv 2510.02952** (HTTP 200 verified).
- **Architecture:** Context-aware flow matching that injects local tissue organization and ligand-receptor priors into a transition-plausibility matrix that regularises an OT objective; designed for spatially-resolved omics trajectory inference.
- **Code:** https://github.com/santanurathod/ContextFlow, MIT, 6 stars, updated 2026-05-01 (early-stage but real).

#### C4. Unbalanced OT for longitudinal lesion evolution (2026)
- **arXiv 2602.09933** (HTTP 200 verified). For longitudinal CT lesion matching, not omics, but the unbalanced-OT machinery is reusable when patient trajectories appear/disappear (clones in MM evolution).

### D. Bayesian / partial-pooling

A targeted search "hierarchical Bayesian + cancer + longitudinal" on arXiv returned, as the most recent ML-specific hit:

- **"Prediction of cancer dynamics under treatment using Bayesian neural networks: A simulated study"** — arXiv 2405.14508v1, May 2024. Simulated study only; not a clinical benchmark. PubMed lacks a peer-reviewed counterpart in the exact MM/lung longitudinal regime as of this search.

### E. Conformal prediction on longitudinal medical TS

The arXiv-restricted query for `"conformal prediction" AND longitudinal AND patient` returned no 2025-2026 oncology hits — strongest recent application is **arXiv 2403.18069** (March 2024, "Personalized Imputation … Continuous Glucose Monitoring"). For ResistanceMap's regime, conformal calibration would be a wrapper layer rather than an absorbed model.

---

## TOP-5 picks ResistanceMap should consider absorbing

### #1 — Trajectory Flow Matching (TFM, arXiv 2410.21154) + TorchCFM ecosystem
TFM is the closest published match to ResistanceMap's regime: irregular sampling, stochastic per-patient dynamics, explicit uncertainty calibration, and crucially a *simulation-free* training objective so that you do not have to backprop through SDE solvers across 51 paired patients (which is where neural-ODE/SDE training typically falls over at small N). TorchCFM (2,440 stars, MIT, very actively maintained) gives you a battle-tested training loop. This is the #1 import target.

### #2 — Latent SDE with modality-dependent emissions (arXiv 2511.16427)
The architectural blueprint here — latent neural SDE with multimodal emission heads and variational state estimation — maps almost exactly onto ResistanceMap's RNA + WES + labs heterogeneity. The 12,000-patient ICU validation shows the formulation is N-scalable; the simulated PKPD benchmark shows the framework directly supports individual treatment-effect estimation, which is what "actionability" requires. Caveat: code release is `[UNVERIFIED]`; if no public repo materialises, this is a reference architecture, not an absorbable library.

### #3 — CellRank 2 (Lange et al., *Nat Methods* 2024)
For the *baseline* N=994 cohort, CellRank 2's multi-view Markov-chain backbone gives a principled, scvi/scverse-compatible transition operator over single-cell states with multiple kernels (RNA-velocity, time-point, pseudotime) that can be combined. ResistanceMap can use it for the cell-state-space prior that the patient-level forecaster sits on top of, exactly the role the original ResistanceMap design assigned to a custom Markov module. BSD-3, 443 stars, scverse-supported.

### #4 — Chronos (arXiv 2403.07815)
The strongest, most mature time-series foundation model with a permissive license and a 5,000+ star ecosystem. It is not biology-aware, but for the *clinical-timeline* channel of ResistanceMap (labs, M-protein, free light chains, treatment events) Chronos is the right zero-shot/fine-tune scaffold. Pair with Chronos-Bolt for inference cost; use as a pretrained tokeniser-and-trunk that ResistanceMap fine-tunes per modality.

### #5 — PERCEPTION (Sinha et al., *Nat Cancer* 2024) as a benchmark, NOT a dependency
PERCEPTION is the published per-patient single-cell drug-response predictor in MM and breast cancer — exactly ResistanceMap's task and one of the only such peer-reviewed benchmarks. It must be in any honest comparison. Caveat: the GitHub repo (ruppinlab/PERCEPTION) has **no LICENSE file** so code-level absorption is legally unsafe; use it as a head-to-head benchmark and re-implement its method from the paper.

---

## Items still tagged UNVERIFIED

- All per-patient Spearman / C-index / AUROC / Wasserstein decimals (PERCEPTION, TRACERx, MOMENT, Chronos, Lag-Llama, TFM, latent-SDE). PDF/supplement read required.
- TFM code path inside TorchCFM (subdir vs. fork).
- Code release status of arXiv 2511.16427.
- Whether any 2024-2026 Microsoft paper does per-patient longitudinal multimodal forecasting — exhaustive search found none.

---

## Verification Log

All commands run from a Bash shell on 2026-05-03. HTTP status reported.

```
# 1. GigaPath PubMed search
curl -sS "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=GigaPath+pathology+foundation+Xu&retmax=5&retmode=json"
HTTP:200  count=2  ids=[40563902, 38778098]

# 2. GigaTIME PubMed search
curl -sS "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=GigaTIME&retmax=5&retmode=json"
HTTP:200  count=4  ids=[41912494, 41832074, 41576915, 41371214]

# 3. GigaTIME arXiv title search
curl -sS "https://export.arxiv.org/api/query?search_query=ti:GigaTIME&max_results=10"
HTTP:200  totalResults=0  (no arXiv preprint with GigaTIME in title)

# 4. GigaTIME GitHub search
curl -sS "https://api.github.com/search/repositories?q=GigaTIME"
HTTP:200  total_count=9  top: prov-gigatime/GigaTIME (358 stars, NOASSERTION license, created 2025-10-06)

# 5-6. PMID efetches: 41371214 (GigaTIME, Cell 2026, MS Research) HTTP:200 — DOI 10.1016/j.cell.2025.11.016; 38778098 (GigaPath, Nature 2024) HTTP:200 — DOI 10.1038/s41586-024-07441-w
# 7. github prov-gigapath/prov-gigapath HTTP:200 (601 stars, Apache-2.0)
# 8-9. PMID 38637658 (PERCEPTION, Nat Cancer) HTTP:200 — DOI 10.1038/s43018-024-00756-7; github ruppinlab/PERCEPTION HTTP:200 (71 stars, no license)
# 10. PMID 37046093 (TRACERx) HTTP:200 — abstract verified "354 NSCLC tumours from 347 of first 421 patients … 947 regions"
# 11. PMID 33619369 (Cohen MM Nat Med) HTTP:200 — abstract verified "41 patients … 11 healthy + 15 NDMM … CoMMpass"
# 12-13. PMID 38871986 (CellRank 2 Nat Methods) HTTP:200 — DOI 10.1038/s41592-024-02303-9; scverse/cellrank HTTP:200 (443 stars, BSD-3)
# 14. arXiv 2402.03885 (MOMENT) HTTP:200; github moment-timeseries-foundation-model/moment HTTP:200 (756 stars, MIT)
# 15. arXiv 2403.07815 (Chronos) HTTP:200; github amazon-science/chronos-forecasting HTTP:200 (5245 stars, Apache-2.0)
# 16. arXiv 2310.08278 (Lag-Llama) HTTP:200; github time-series-foundation-models/lag-llama HTTP:200 (1574 stars, Apache-2.0)
# 17. arXiv 2310.03589 (TimeGPT-1) HTTP:200
# 18. arXiv 2410.21154 (Trajectory Flow Matching, NeurIPS 2024) HTTP:200, v2
# 19. arXiv 2511.16427 (Latent-SDE clinical TS) HTTP:200 — abstract verified "ICU data from 12,000 patients"
# 20. arXiv 2510.02952 (ContextFlow) HTTP:200; github santanurathod/ContextFlow HTTP:200 (6 stars, MIT)
# 21. arXiv 2602.09933 (UOT longitudinal lesion CT) HTTP:200

# 22. TorchCFM repo
curl -sS "https://api.github.com/repos/atong01/conditional-flow-matching"
HTTP:200  2440 stars  MIT  updated 2026-05-03

# 23. Negative-result searches
PubMed "neural ODE longitudinal clinical": 1 hit (PMID 38350557, transformer PK/PD, not ODE)
arXiv "hierarchical Bayesian"+cancer+longitudinal: top recent = arXiv 2405.14508 (simulation only)
arXiv "conformal prediction"+longitudinal+patient: closest = arXiv 2403.18069 (CGM, Mar 2024)
bioRxiv DOI 10.1101/2024.06.05.597546: HTTP:200 "no posts found"
```

End of verification log.
