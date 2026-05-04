# SOTA Benchmark Protocol — ResistanceMap (resistance-state trajectory prediction)

Scope: ResistanceMap is being re-targeted from cell-line drug-response (where it does
not beat predict-mean on `paper/tables/baseline_comparison.json`) to **patient-level
resistance-state trajectory prediction** on multiple-myeloma (MM) scRNA-seq cohorts
(GSE124310, GSE271107). This document fixes the published numbers we must match or
beat for any submission to be defensible. Every identifier below was fetched from a
live URL on 2026-05-03; numbers without provenance are flagged **UNVERIFIED**.

---

## 1. Canonical benchmark datasets

| Dataset | Source / URL | Verified | Published winning model on this dataset (and metric) |
|---|---|---|---|
| **Embryoid Body (EB), Moon et al. 2019** — 5 timepoints over 27 days | TrajectoryNet paper, used in MIOFlow Table 2 (PMC10312391) | yes — TrajectoryNet PDF (arXiv 2002.04461) Table 3 reports it; MIOFlow uses identical splits | TrajectoryNet (2020) on EMD; MIOFlow (2022) on W1/MMD |
| **Mouse hematopoiesis, Weinreb et al. 2020** — Cell-Tag lineage tracing, days 2/4/6 | PMID 31974159 (Science 2020), DOI 10.1126/science.aaw3381 | yes | PRESCIENT (Yeo et al. 2021, PMID 34050150) — fate-prediction Pearson r = 0.347 ± 0.029, AUROC = 0.692 ± 0.012 (with proliferation rates) |
| **Pancreas endocrinogenesis, Bastidas-Ponce et al. 2019** | PMID 31160421, DOI 10.1242/dev.173849 (Development) | yes | Used as scVelo / CellRank2 benchmark; CellRank 2 (Nat Methods 2024, PMID 38871986) is current SOTA on terminal-state recovery |
| **MOSCOT mouse-embryo atlas** — 1.7 M cells, 20 time points | Klein et al., Nature 2025 (PMID 39843746, DOI 10.1038/s41586-024-08453-2) | yes | MOSCOT itself (no peer-reviewed competitor published on the full atlas as of fetch date) |
| **MM-specific or hematologic trajectory benchmark** | Tan et al. 2025 (PMID 41050668, DOI 10.3389/fimmu.2025.1658028) — Monocle/pseudotime on 35 944 malignant plasma cells across 18 MM samples | yes (abstract verified) | No standardised benchmark exists. The paper reports a bifurcating pseudotime to a resistant phenotype (MalPlasma3, 93.1 % poor-survival cells) but does **not** publish a held-out-timepoint W1/W2/MMD score. **There is no canonical MM trajectory benchmark with reported numbers.** This is a real gap, and ResistanceMap can claim to define it. |

GEO datasets the ResistanceMap-trajectory pipeline targets, both verified live on
2026-05-03 via `ncbi.nlm.nih.gov/geo/query/acc.cgi`:

- **GSE124310** — "Single-cell RNA sequencing reveals compromised immune
  microenvironment in precursor stages of multiple myeloma." Bone-marrow scRNA-seq
  across MGUS / SMM / MM and healthy donors. **Cross-sectional (stages), not
  longitudinal per patient.**
- **GSE271107** — sequential bone-marrow aspirate samples from 7 MM patients at
  different disease states + 5 healthy donors. **This is the only longitudinal
  patient-level cohort; it is what makes a true trajectory task possible.**

---

## 2. Canonical metrics (with definitions and citing paper)

| Metric | Formal definition | Original use in this domain |
|---|---|---|
| **Wasserstein-2 (W₂)** between predicted and observed cell distributions at held-out timepoints | W₂(μ,ν)² = inf_{π∈Π(μ,ν)} ∫ ‖x−y‖² dπ(x,y) | TrajectoryNet, arXiv 2002.04461 (Tong et al. 2020). Their Table 3 reports EB EMD on left-out timepoints t=1,2,3 (best mean = 0.784 with Base+D regularisation). |
| **Wasserstein-1 (W₁)** | as W₂ with Euclidean ground cost, p=1 | MIOFlow (PMC10312391, arXiv 2206.14928) Table 2 — EB t=2 W₁ = 25.744 (GAE α-decay); t=3 W₁ = 32.227. Baseline (avg of prev/next) = 33.415, 35.319. |
| **Maximum Mean Discrepancy (MMD)** with RBF (Gaussian) kernel | MMD²(μ,ν) = E[k(x,x′)] + E[k(y,y′)] − 2E[k(x,y)] | MIOFlow uses Gaussian (G) and Mean (M) kernels: EB t=2 MMD(G) = 0.061, MMD(M) = 99.747; t=3 MMD(G) = 0.135, MMD(M) = 213.465. (Source: PMC10312391.) |
| **Earth Mover's Distance (EMD)** | discrete W₁ between empirical distributions on a fixed cost matrix | TrajectoryNet (arXiv 2002.04461) Table 3 — EB best mean EMD = 0.784. |
| **Pearson r + AUROC for fate bias** | r between predicted and clonal fate-bias scalar; AUROC for classifying clonal fate-bias > 0.5 | PRESCIENT (PMC8163769). With proliferation rates: r = 0.347 ± 0.029, AUROC = 0.692 ± 0.012; without: r = 0.196 ± 0.020, AUROC = 0.601 ± 0.006; held-out clonal upper bound: r = 0.487, AUROC = 0.771. |
| **Concordance index (C-index)** for time-to-event | Pr(predicted-risk(i) > predicted-risk(j) ∣ t_i < t_j, observed) | Standard in survival analysis (Harrell 1982 — outside PubMed scope). For MM bortezomib-resistance time-to-event, no model on GSE271107 has published a C-index. **UNVERIFIED for this exact dataset.** |
| **Pathway accuracy@k** | fraction of held-out patients for whom the true driver pathway appears in the model's top-k | Not standardised; closest relative is iMLGAM's CRISPR-screen-derived top-gene validation (PMID 40236779). |
| **Brier score** | (1/N) Σ (p_i − y_i)² for binary state outcomes | Standard calibration metric; no trajectory paper reports Brier on cell-state predictions. **Use with caveat.** |

---

## 3. Top-3 trajectory-model reference numbers (verified)

| Model | Dataset / split | Metric | Value | Source URL |
|---|---|---|---|---|
| **TrajectoryNet** (Tong et al. 2020) | EB Moon-2019, leave-one-out on each of 3 intermediate timepoints | EMD (≈ W₁), mean over t=1,2,3 | **0.784** (Base + density reg.) | arXiv:2002.04461 PDF, Table 3 (fetched 2026-05-03) |
| **TrajectoryNet** | EB | EMD baseline = "previous timepoint" | 1.309 mean | same Table 3 |
| **MIOFlow** (Huguet et al. 2022) | EB Moon-2019, leave-one-out t=2 | W₁ | **25.744** (GAE α-decay) | arXiv:2206.14928 PDF, Table 2; PMC10312391 |
| **MIOFlow** | EB t=2 | MMD(Gaussian) | **0.059** (GAE Gaussian) | same |
| **MIOFlow** | EB t=2 | MMD(Mean) | **99.747** | same |
| **MIOFlow** | EB t=2 | W₁ baseline (avg prev/next) | 33.415 | same |
| **MIOFlow** | EB t=3 | W₁ | **32.227** | same |
| **PRESCIENT** (Yeo et al. 2021) | Weinreb hematopoiesis, fate-prediction over 5 seeds | Pearson r | **0.347 ± 0.029** (with proliferation) | Nat Commun, PMC8163769; verified text on 2026-05-03 |
| **PRESCIENT** | same | AUROC | **0.692 ± 0.012** | same |
| **PRESCIENT** | same | Wasserstein on held-out day 4 | reported as figure-only in Fig 2a; absolute number not in main text → **UNVERIFIED numerical value** | PMC8163769 |
| **Waddington-OT** (cited in PRESCIENT) | Weinreb fate prediction | Pearson r / AUROC | 0.150 / 0.599 | reported in PRESCIENT Fig 2 text (PMC8163769) |
| **FateID** (cited in PRESCIENT) | Weinreb fate prediction | Pearson r / AUROC | 0.225 / 0.602 | same |
| **MOSCOT** (Klein et al. 2025, Nature) | Mouse-embryo atlas, 1.7 M cells, 20 timepoints | not a single headline scalar; uses cell-type-fraction recovery and developmental-trajectory accuracy | numbers are figure-bound in the published Nature article body | DOI 10.1038/s41586-024-08453-2; PMID 39843746 — **specific scalar UNVERIFIED at this fetch level** |
| **CellRank 2** (Weiler et al. 2024) | human hematopoiesis, endodermal development | terminal-state recovery, fate probabilities | reported as figures, not a single benchmark number | Nat Methods, PMID 38871986, DOI 10.1038/s41592-024-02303-9 — **specific scalar UNVERIFIED** |

Code: `KrishnaswamyLab/TrajectoryNet`, `gifford-lab/prescient`,
`KrishnaswamyLab/MIOFlow`, `theislab/moscot` — all four GitHub URLs returned HTTP 200
on 2026-05-03.

---

## 4. ResistanceMap evaluation protocol (resistance-state task)

### 4.1 Cohort and split strategy

- **Primary cohort**: GSE271107 (sequential MM patient samples). 7 MM patients with
  longitudinal sampling — only this cohort supports a real **train-on-prior-timepoint,
  predict-future-timepoint** task per patient.
- **Auxiliary cohort**: GSE124310 (cross-sectional, MGUS → SMM → MM stages, 22
  patients). Treated as a *pseudo-temporal* validation only; never as a substitute for
  longitudinal split.
- **Split unit = patient**. Cells from a single patient must not appear in both train
  and test. With n=7 longitudinal MM patients we adopt **leave-one-patient-out cross
  validation (LOPO-CV)** with an inner 5-fold for hyper-parameters on the remaining 6
  patients. The split is fixed, hashed, and committed before any evaluation; this is
  enforced by the existing `EVALUATION_GOVERNANCE.md` protocol.
- **Held-out timepoints**: for each held-out patient, the *latest* available
  timepoint is masked (not random); the model sees only pre-treatment (or earliest)
  scRNA snapshot and must predict cell-state distribution at the masked later time.
- **Cell-state vocabulary**: the k cell-state classes are fixed up-front via a
  reference clustering on the *training* patients only (no information from held-out
  patient touches the clustering). MalPlasma1–5 from Tan et al. 2025 (PMID 41050668)
  is a defensible starting taxonomy; final clusters must be re-derived under the
  governance protocol.

### 4.2 What "predicted resistance state" means

The model output is a vector p ∈ Δ^(k−1) per patient at horizon Δ — a probability
distribution over the k pre-defined cell states at time t+Δ (where Δ is in the
patient's observed time grid; we do not interpolate to unobserved Δ). Three
quantitative outputs are scored:

1. **Cell-state distribution prediction** — score with W₁ between predicted p and
   empirical cell-state frequencies in the held-out timepoint. Secondary: MMD(G)
   on the latent space embedding of the cells (matches MIOFlow protocol).
2. **Time-to-resistance** — for the subset of patients with a recorded resistance
   event (bortezomib-melphalan-prednisone failure label exists in GSE271107
   metadata), report **C-index** with 95 % bootstrap CI.
3. **Driver pathway** — top-k pathway-accuracy: of the k pathways the model ranks
   highest, what fraction includes the literature-validated driver (E2F/MYC,
   G2M-checkpoint stemness, per Tan et al. 2025). Report at k = 3 and k = 5.

### 4.3 Minimum-publishable bar

- Beats the **previous-timepoint baseline** (i.e. assume no change) by ≥ 15 % on W₁,
  with non-overlapping 95 % CIs, computed under LOPO-CV.
- Beats the **predict-mean** baseline (constant cell-state distribution = training-set
  mean) by ≥ 10 % on W₁.
- C-index ≥ 0.65 with lower-CI > 0.55 (so that the lower bound clears chance).
- Pathway accuracy@5 ≥ 0.50 (matches the stringency of iMLGAM's CRISPR-validation
  story without its scale).
- Calibration: Brier on cell-state predictions ≤ 0.20 (no published reference; this
  is a self-imposed floor).

If any of the four bars is missed under the governance-locked split, the work is not
defensible as a methods paper.

---

## 5. Recent (2024–2026) SOTA papers ResistanceMap-trajectory will be judged against

| Paper | Year | PMID | DOI | Same task? | Why it matters for our review |
|---|---|---|---|---|---|
| TRACERx Lung "Genomic-transcriptomic evolution in lung cancer and metastasis" — Martínez-Ruiz, Black, Puttick et al. | 2023 | 37046093 | 10.1038/s41586-023-05706-4 | **No** — bulk WES/RNA on lung adenocarcinoma evolution; not single-cell trajectory inference. | Different organ, different modality. Will be cited only as "evolutionary-genomics context", not as a benchmark. |
| Pan-cancer proteogenomics characterization of tumor immunity — Petralia, Ma, Yaron et al. (Cell) | 2024 | 38359819 | 10.1016/j.cell.2024.01.027 | **No** — bulk proteogenomics, not trajectory. | Cited as "proteogenomic substrate that informs the pathway taxonomy." Not a comparator. |
| ML-based identification of an immunotherapy-related signature in melanoma (2024) | 2024 | search returned 27 candidates; **no single canonical PMID without an exact title** | n/a | **No** | Cannot select a canonical melanoma-2024 paper without the user's exact title. **Flagged UNVERIFIED.** Skip from the reference set unless title is provided. |
| iMLGAM — Ye, Fan, Xue et al. (iMeta) | 2025 | 40236779 | 10.1002/imt2.70011 | **No** — bulk multi-omics ICB-response prediction; not single-cell, not trajectory. | Will be cited as "multi-omics ICB benchmark." Not a comparator on the trajectory task. |
| TrajectoryNet — Tong et al. | 2020 | 34337419 | n/a (PMLR 119:9526) | **Yes — direct comparator** on EB; partial on MM (no MM cohort published) | Re-implement; report W₁ and EMD on EB and on GSE271107 LOPO. |
| PRESCIENT — Yeo et al. | 2021 | 34050150 | 10.1038/s41467-021-23518-w | **Yes — direct comparator** for fate-bias on hematopoiesis; relevant for MM by analogy | Re-implement; report Pearson-r + AUROC on Weinreb; transfer the protocol to the MM resistant-vs-non-resistant fate. |
| MIOFlow — Huguet et al. | 2022 | 37397786 | n/a (NeurIPS 2022) | **Yes — direct comparator** on EB and AML; closest architectural cousin of ResistanceMap (Neural-ODE + autoencoder). | Re-implement; report W₁/MMD on EB. |
| CellRank 2 — Weiler et al. | 2024 | 38871986 | 10.1038/s41592-024-02303-9 | **Partial** — fate mapping in multi-view scRNA, includes hematopoiesis. | Use as a reference baseline for terminal-state recovery on MM cells. |
| MOSCOT — Klein et al. | 2025 | 39843746 | 10.1038/s41586-024-08453-2 | **Yes — direct comparator** for cross-time-point optimal transport; multimodal. | Use as the strongest current OT baseline. |

---

## 6. Decision table — what we must hit to publish

| Metric | Comparator | ResistanceMap target | Source of comparator number |
|---|---|---|---|
| **W₁** on held-out final-timepoint cell-state distribution, LOPO-CV on GSE271107 | "previous-timepoint" baseline (assume-no-change) **and** TrajectoryNet re-implemented on the same split | **≥ 15 % relative reduction vs previous-timepoint, and ≤ TrajectoryNet's W₁ within overlapping 95 % CIs** | Baseline = MIOFlow Table 2 protocol (PMC10312391); TrajectoryNet number must be re-derived locally because no MM number exists |
| **MMD (Gaussian kernel)** on latent embedding at held-out timepoint | MIOFlow re-implemented on GSE271107 | **≤ MIOFlow's MMD(G) within overlapping 95 % CIs** | MIOFlow Table 2 EB t=2: MMD(G)=0.059 (PMC10312391) — reference scale only; absolute number on MM will differ |
| **C-index** for time-to-bortezomib-melphalan-prednisone-failure | random / clinical-only Cox-PH baseline | **C-index ≥ 0.65, lower 95 %-CI > 0.55** | No published MM C-index on GSE271107 → defines our bar |
| **Pathway accuracy @ 5** for the driver claim | iMLGAM CRISPR-validation logic (categorical "did it find the right gene?") | **≥ 0.50** | iMLGAM (PMID 40236779) validates a single gene, CEP55, via CRISPR; we adopt the spirit, not the number |
| **Brier score** on cell-state predictions | none published — self-imposed | **≤ 0.20** | n/a |

**The bottom line**: to publish, ResistanceMap-trajectory must (a) reduce W₁ on the
GSE271107 LOPO held-out timepoint by ≥ 15 % relative to the previous-timepoint
baseline *and* be statistically indistinguishable from a re-implemented TrajectoryNet
and MIOFlow under the governance-locked split; (b) deliver C-index ≥ 0.65 (lower-CI
> 0.55) on time-to-resistance; (c) reach pathway accuracy@5 ≥ 0.50 against the Tan
et al. 2025 driver set. Any subset of these that fails forces a scope reduction
(method paper → applied case study) or an alternative cohort.

**Honesty caveats** that must travel with the manuscript:
1. There is no canonical MM trajectory benchmark; we are *defining* one. We must
   release the split, the cell-state taxonomy, and the evaluator code.
2. The MIOFlow / TrajectoryNet absolute numbers above were collected on EB and on
   AML, **not on MM**. They cannot be quoted as "ResistanceMap beats X = 25.744";
   the only fair comparison is local re-implementation on the same MM split.
3. n = 7 longitudinal patients is small. LOPO-CV with bootstrap CIs is the upper
   bound of statistical rigour available; we cannot claim "generalisation across
   cohorts" without a second longitudinal MM cohort.
4. The "ML-melanoma immunotherapy 2024" reference cannot be canonically resolved
   from the user's description alone (27 candidate PMIDs); flagged UNVERIFIED until
   the exact title is supplied. Do not cite it.

---
*Compiled 2026-05-03. All PMIDs / DOIs / arXiv IDs / GEO accessions verified by live
HTTP fetch on the same day. Numbers extracted directly from the published PDFs and
PMC HTML referenced inline; nothing in this document is reproduced from memory.*
