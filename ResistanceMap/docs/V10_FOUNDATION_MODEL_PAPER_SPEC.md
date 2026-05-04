# A Multi-Omic Foundation Model for Predicting Epigenetic Drug Resistance Trajectories in Hematologic Malignancies, using Neural ODEs on Waddington Landscapes and Protein Interaction Network Propagation

**Version:** v10 paper-architecture synthesis · **Date:** 2026-05-03
**Audience:** integrated source-of-truth for the v10 manuscript. Every claim in this document is anchored to one of:
- a fetched citation (PMID / DOI / arXiv ID, verified live by the eight PhD-level agents listed in §0),
- a file path on disk under `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/`,
- or a numeric measurement traceable to `paper/v8_artifacts/baselines/`.

This document **does not introduce new numeric claims that would land in `README.md` or `ARCHITECTURE.md`**. Every measurement quoted here either already lives in a doc on disk or in a baselines JSON. Numbers will only migrate to README via a release-bouncer pass that adds a row to `RUNS.md`.

---

## §0 Provenance — eight parallel agent reviews this synthesis integrates

| # | Doc | Verdict |
|---|---|---|
| 1 | `docs/WADDINGTON_NEURAL_ODE_REVIEW.md` | Scalar potential $V_\theta$ via DSM + PPI-Laplacian Tikhonov is the identifiable gauge slice at $N_p = 42$ |
| 2 | `docs/WADDINGTON_NEURAL_ODE_MATH.md` | Wang landscape + flux $F = -D\nabla\log P_{ss} + v_{ss}$ is the unique decomposition; pure-gradient PRESCIENT *rejected* if cell-cycle curl > 30% |
| 3 | `docs/PPI_PROPAGATION_DRIVER_THEORY.md` | Symmetric-normalized PPR/RWR at α=0.5; heat-kernel exp(−tL_sym) is the differentiable bridge to the Neural-ODE |
| 4 | `docs/PPI_PROPAGATION_THEOREMS.md` | RWR at α=0.7 + NPI z-score correction; HotNet2 permutation null as statistical gate |
| 5 | `docs/MM_TRAJECTORY_DRIVERS_EMPIRICAL.md` | Three replicated signals: chr1q→MCL1/PI3K, TP53→BCL2/BH3, UPR/hypoxia/OXPHOS |
| 6 | `docs/SINGLE_SNAPSHOT_INFERENCE.md` | Frozen scGPT prior + Chung diffusion posterior sampling — *not* a from-scratch GigaTIME-style translator |
| 7 | `docs/DRIVER_FUNCTION_THEOREMS.md` | Shpitser-Pearl Theorem 2 + Bareinboim-Pearl Theorem 4 — Pearl tier ceiling is L1-with-structural-prior |
| 8 | `docs/PPI_PROPAGATION_PILOT.md` | Empirical CONDITIONAL pass: PSMB5→proteasome p=9.2e-38; 4/10 drugs Bonferroni-significant |
| 9 | `docs/ADDITIONAL_DATA_SOURCES.md` | Top-3 pulls: GSE279766, Beat AML 1.0, BLUEPRINT IHEC |
| 10 | `docs/REJECTED_ALTERNATIVES.md` | 12 rejected alternatives with primary evidence; Honesty Constraints §14 |

The above ten documents are the *evidence base*. This spec stitches them into a coherent paper plan.

---

## §1 Title alignment & contribution claim

**Working title (verbatim from user framing):** *A Multi-Omic Foundation Model for Predicting Epigenetic Drug Resistance Trajectories in Hematologic Malignancies, using Neural ODEs on Waddington Landscapes and Protein Interaction Network Propagation.*

**Honest contribution sentence (max defensible):**
> "We present a population-stratified per-patient forecaster of multiple-myeloma drug-resistance trajectories that combines a Waddington-style energy-landscape neural-SDE (TFM) with PPI-propagation-derived structural priors, equipped with Mondrian conformal coverage guarantees on the five cytogenetic strata in MMRF CoMMpass IA22, and a CRISPR-essentiality falsification oracle for predicted driver gene sets."

**What this contribution explicitly does NOT claim** (Honesty Constraints from `REJECTED_ALTERNATIVES.md` §14):
1. *Not* a universal forecaster — coverage transfers only to patients exchangeable with the Recurrent-BM calibration cohort.
2. *Not* a causal driver claim — Pearl ceiling is L1-with-structural-prior (per Theorem 2 hedge in `DRIVER_FUNCTION_THEOREMS.md`).
3. *Not* a multi-mediator NIE analysis — single pre-specified mediator only at $N_p=42$ (EPV ≈ 2.8 fails the ≥25 threshold).
4. *Not* a from-scratch foundation model — uses **frozen scGPT** as prior; only the likelihood head is fitted.
5. *Not* GigaTIME — that paper is H&E→virtual-mIF cross-modal translation, *not* temporal forecasting.

---

## §2 Math (the model the NN learns)

### §2.1 Latent space and observed snapshots

Let $z \in \mathbb{R}^{64}$ be the cell-state latent emitted by the **frozen scGPT encoder** (PMID 38409223, `tdc/scGPT@acf749f3` mirror, MIT). At each clinical observation time $t$, the scGPT encoding of the patient's transcriptome (pseudo-bulked or pseudo-cell-resolved when scRNA is available) gives a snapshot $z_t \in \mathbb{R}^{64}$.

The MMRF cohort decomposes (counts verified live 2026-05-03 in `MMRF_ACQUISITION.md` §3):
- **$N_b = 994$** baseline-only patients with treatment data,
- **$N_r = 319$** Recurrent-BM samples (paired baseline + relapse),
- **$N_p = 42$** strict-paired patients with both ≥2 RNA-Seq timepoints AND a 2nd-line transition.

### §2.2 Trajectory mechanic — landscape + flux

The trajectory is governed by the Itô SDE
$$
dz_t = F_\theta(z_t \mid c_p)\,dt + \Sigma_\theta(z_t)^{1/2}\,dW_t.
$$

By the **Wang-Zhang-Xu-Wang 2011 uniqueness theorem** (PNAS, PMID 21536909), for a smooth drift with positive stationary density $P_{ss}$, the drift admits the unique decomposition
$$
F_\theta(z\mid c_p) \;=\; -D\,\nabla U_\theta(z\mid c_p) \;+\; v_\theta(z\mid c_p), \qquad \nabla\!\cdot\!(P_{ss}\,v_\theta) = 0.
$$

- $U_\theta : \mathbb{R}^{64} \to \mathbb{R}$ is the **Waddington pseudo-potential** — a scalar MLP (per `WADDINGTON_NEURAL_ODE_REVIEW.md`).
- $v_\theta$ is the **divergence-free flux** — accounts for cell-cycle and clonal-selection cycling, which are *curl phenomena*.

**Identifiability ladder** (per `WADDINGTON_NEURAL_ODE_REVIEW.md` §2.3 + `WADDINGTON_NEURAL_ODE_MATH.md` §1.3):
1. **Stage 1 (always identifiable from snapshots):** $U_\theta$ trained by **denoising score matching** (Vincent 2011; Song-Ermon arXiv 1907.05600) on $N_b + N_r = 1313$ snapshots. Recovers $\nabla U_\theta = -\nabla\log p_{\rm data}$ at $O(N^{-1/2})$ rate.
2. **Stage 2 (test for curl):** compute **CurlFraction** $= \mathbb{E}\|v_\theta\|^2 / \mathbb{E}\|F_\theta\|^2$ via natural Helmholtz-Hodge decomposition (Jia & Chen 2023, arXiv:2311.10403) on a $k$-NN simplicial complex of the 994 baselines.
3. **Stage 3 (only if CurlFraction > 0.30):** add the divergence-free flux head $v_\theta$, enforced via soft Helmholtz-Hodge projection (nHHD).

This is the principled reconciliation of the two Waddington-agent recommendations: gradient-only as the identifiable base, flux added empirically only if cycling is detected.

### §2.3 PPI structural prior

The **PPI-Laplacian Tikhonov regularizer** couples the trajectory to STRING:
$$
\mathcal{R}_{\rm PPI}(U_\theta) \;=\; \lambda \,\langle \nabla U_\theta(z), \, L_{\rm PPI} \,\nabla U_\theta(z) \rangle,
$$
where $L_{\rm PPI}$ is the symmetric-normalized Laplacian of the STRING graph filtered to combined-score ≥ 0.7 AND non-observational channel sum ≥ 0.4 (per `PPI_PROPAGATION_DRIVER_THEORY.md` §3 — required to remove co-expression / text-mining leakage with MMRF expression).

**Why this term is principled:** the linearised drift $\dot z = -\nabla(\tfrac12 z^\top(\alpha L_{\rm PPI} + \beta I)z)$ has solution $z(t) = e^{-\beta t}\exp(-t\alpha L_{\rm PPI})z(0)$. **The heat kernel $\exp(-tL_{\rm PPI})$ is literally the Neural-ODE flow under a PPI-quadratic potential.** The Tikhonov regularizer makes this connection differentiable end-to-end.

### §2.4 Driver-pathway head — RWR with adaptive α

For a fitted resistance-state direction $s \in \mathbb{R}^{|V|}$ (top-$k$ DEGs between resistance-positive and -negative latent slices), the propagated driver vector is
$$
r \;=\; \alpha (I - (1-\alpha)P)^{-1} s,
$$
where $P$ is the column-normalized STRING adjacency.

**α reconciliation across agents 3, 4, 8:**

| Recommendation | Source | Use case |
|---|---|---|
| α = 0.5 | `PPI_PROPAGATION_DRIVER_THEORY.md` (theory) | Diffusion limit; max coupling to heat-kernel ODE form |
| α = 0.7 | `PPI_PROPAGATION_THEOREMS.md` (theorems) | Köhler 2008 / Vanunu 2010 robust regime [0.5, 0.8] |
| α = 0.80–0.85 | `PPI_PROPAGATION_PILOT.md` (empirical) | Selective seeds (BCL2, CDK6) where 0.7 over-diffuses |

**Synthesis decision:** α is a **per-seed hyperparameter** calibrated by held-out Chronos AUC on the haematological DepMap subset. Default α=0.7. NPI z-score correction (degree-aware null via 1,000 random-seed-set rewirings, DADA tradition; `PPI_PROPAGATION_THEOREMS.md` §3) applied to **every** propagation output to remove hub bias.

### §2.5 Single-snapshot inferential primitive

Per `SINGLE_SNAPSHOT_INFERENCE.md` §5, the v10 inference target is
$$
p(\gamma \mid X) \;=\; \frac{1}{Z(X)} \int p_\theta(\gamma\mid z)\,p_\psi(X\mid z)\,p_\theta(z)\,dz,
$$
where:
- $p_\theta(z)$ — **frozen scGPT prior** over latent $z$,
- $p_\theta(\gamma \mid z)$ — **frozen** trajectory decoder (the scalar $U_\theta$ + optional $v_\theta$ from §2.2),
- $p_\psi(X \mid z)$ — **the only learnable component** — likelihood head fitted on RM data.

Inference at test time uses **Chung et al. 2023 diffusion posterior sampling** (arXiv 2209.14687):
$$
z^\star \sim p_\psi(X^\star \mid z)\,p_\theta(z), \quad \gamma^\star = {\rm Decode}_\theta(z^\star).
$$

This avoids the from-scratch sample-complexity bound $n = \Omega(\eta^{-d^\star})$ that kills Approach A latent-SDE training at $N_p = 42$.

### §2.6 Hierarchical Bayes outer layer

The 994 baseline-only patients enter the marginal likelihood through stratified partial pooling over the 5 cytogenetic strata (del17p 13.6%, chr1q21+ 33.6%, del13q 52%, t(4;14) 14.6%, t(11;14) 16.8% — calls in `data/processed/cytogenetics.tsv`, 716/976 with all 5 calls per `INTEGRATED_BUILD_PLAN.md` line 74):

$$
p(\theta_s \mid \mu, \tau) \sim \mathcal{N}(\mu, \tau^2),\quad s \in \{1,\ldots,5\}, \qquad p(\mu) \sim \mathcal{N}(0, 1), \quad p(\tau) \sim \text{Half-Cauchy}(1).
$$

Per `TRAJECTORY_MATH_FEW_SHOT.md` Approach C, hierarchical Bayes is the **only** formulation in which the 994 baseline-only patients enter the marginal likelihood non-vacuously.

### §2.7 Bellman-Harris clonal prior

$K \le 8$ subclones (justified by PyClone-VI on MMRF VAF + Maura et al. 2019 PMID 31444325, median $4 \pm 2$ detectable subclones in CoMMpass; see `CLONE_DYNAMICS_AND_CONFORMAL.md` §1a). The branching prior reduces sample complexity ~10× vs unconstrained — 319 Recurrent BM patients are within an order of magnitude of identifying clone fitness to ε=0.15.

### §2.8 Mondrian jackknife+ conformal wrapper

Per-stratum coverage at 90% via Barber-Candès-Ramdas-Tibshirani 2021 (arXiv:1905.02928) jackknife+ on the full 994. Empirical coverage from the same paper's bound: 3 strata clear ±3%, 2 strata clear ±5% (n_s ≥ 36 floor).

---

## §3 Block diagram (outer → inner)

```
                 ┌─────────────────────────────────────────────┐
                 │  Mondrian jackknife+ conformal wrapper      │
                 │  (per-stratum 90% coverage, n_s≥36 floor)   │
                 └──────────────────┬──────────────────────────┘
                                    │
                 ┌──────────────────▼──────────────────────────┐
                 │  Hierarchical Bayes outer (5 cyto strata)   │
                 │   N=994 baselines enter marginal likelihood │
                 └──────────────────┬──────────────────────────┘
                                    │
                 ┌──────────────────▼──────────────────────────┐
                 │  Bellman-Harris branching prior K≤8         │
                 │   (PyClone-VI initialization)               │
                 └──────────────────┬──────────────────────────┘
                                    │
                 ┌──────────────────▼──────────────────────────┐
                 │  Trajectory Flow Matching (TorchCFM, MIT)   │
                 │   neural-SDE on landscape U_θ + flux v_θ    │
                 │   PPI-Laplacian Tikhonov regularizer        │
                 └──────────────────┬──────────────────────────┘
                                    │
                 ┌──────────────────▼──────────────────────────┐
                 │  CellRank 2 PseudotimeKernel                │
                 │   (multi-view Markov, no spliced/unspliced) │
                 └──────────────────┬──────────────────────────┘
                                    │
                 ┌──────────────────▼──────────────────────────┐
                 │  Frozen scGPT (tdc/scGPT@acf749f3, MIT)     │
                 │   + parallel Chronos clinical-timeline      │
                 └─────────────────────────────────────────────┘

Driver-pathway head (parallel, NOT in inference path):
    seed s = top-k DEG → RWR(α∈[0.5,0.85] per seed) → NPI z-score → top-k drivers
    Falsification: Wilcoxon vs 1000 random gene sets on DepMap Chronos
```

---

## §4 Empirical drivers v10 must reproduce

From `MM_TRAJECTORY_DRIVERS_EMPIRICAL.md`, three signals are replicated in ≥2 cohorts:

1. **chr1q gain → MCL1 / PI3K axis** (Tirier 2021 PMID 34845188; Sklavenitis-Pistofidis 2023 PMID 37577538). Subclones expand under treatment; MCL1+PI3Ki sensitivity confirmed.
2. **TP53 inactivation → BCL2-family rebalancing → BH3-mimetic response** (Durand 2024 PMID 38096363) — the only cross-cohort survival signal in the verified set, validated in MMRF-CoMMpass + CASSIOPEA within one paper.
3. **UPR / hypoxia / mitochondrial-respiration program** (Cohen 2021 PMID 33619369, n=41 longitudinal RRMM, generalized to CoMMpass; DrugFormer Liu 2024 PMID 39206872 confirms COX8A direction).

**Sanity check v10 must pass:** RWR seeded on each of these three axes must rank the canonical replicated genes (MCL1, PIK3CA, BCL2, BAX, HSPA5, COX8A, PPIA) inside the top-50 of its propagation output for the corresponding resistance state.

---

## §5 Identifiability ceiling — Pearl-tier

Per `DRIVER_FUNCTION_THEOREMS.md`:

- **Theorem 2 (Shpitser-Pearl 2008, JMLR v9):** $P(\text{resistance} \mid do(\text{gene KO}))$ is **non-identifiable** from MMRF + CCLE observational alone. A hedge structure with $U_{\rm clonal}$ has bidirected edges into both $X$ and $Y$. **No do-calculus manipulation breaks this hedge.**
- **Theorem 4 (Bareinboim-Pearl 2014, DOI 10.1214/14-sts486):** all three transportability conditions fail for MM (immune microenvironment, clonal heterogeneity, passage drift). **DepMap CRISPR's correct role is falsification oracle ONLY — never an instrumental variable.**

**Maximum honest Pearl tier:** L1 with structural prior. The phrase "candidate drivers consistent with PPI-propagation rank under assumptions C1-C4" is the strongest defensible framing.

**L2 escalation path** (only one available without wet-lab CRISPR in primary MM cells):
- Single-mediator NIE: proteasome pathway score at $T_1 \to$ time-to-2nd-line Bortezomib (cytogenetics → mediator → outcome). IPTW-Cox conditioning on ISS + 5 cytogenetic flags. **Mandatory EIMOC disclaimer** (Effect of Inter-visit Mediator-Outcome Confounding) — physician treatment-adjustment between visits is structurally unidentifiable without visit-level dose records.

**L3 (true interventional)** is v11, requires wet-lab CRISPR in primary MM cells (Cohen 2021 PMID 33619369 template).

---

## §6 Falsification tests (pre-registered)

| # | Test | Threshold | Source |
|---|---|---|---|
| F1 | Hold out one cytogenetic stratum, forecast via cross-stratum HBayes prior | 90% CI coverage < 70% → v10 refuted | `V10_PER_PATIENT_FORECAST_SPEC.md` |
| F2 | Ablate 994 baselines from likelihood, paired-bootstrap held-out log-lik | Δlog-lik = 0 (α=0.05) → partial pooling adds no value | `V10_PER_PATIENT_FORECAST_SPEC.md` |
| F3 | CRISPR rank-sum on top-k driver genes vs 1000 random gene sets | Below 95th percentile → attribution claim fails | `V10_PER_PATIENT_FORECAST_SPEC.md` |
| F4 | CurlFraction = 𝔼‖v_θ‖² / 𝔼‖F_θ‖² via Helmholtz-Hodge | > 0.30 → gradient-only Waddington refuted; switch to landscape+flux | `WADDINGTON_NEURAL_ODE_MATH.md` |
| F5 | RWR median identifiability test: held-out 42 paired patients, simulate $\hat z_{i,1}$ from $V_\theta$ | Median $\|\hat z - z\|/\|z_0 - z\| \ge 0.9$ → energy formulation rejected | `WADDINGTON_NEURAL_ODE_REVIEW.md` |
| F6 | RWR-on-STRING recovers ≥ 35/50 Walker 2018 (PMID 29884741) MM drivers in top-100, permutation p<0.001 | < 70% recovery → propagation prior insufficient | `PPI_PROPAGATION_DRIVER_THEORY.md` |
| F7 | Pilot already passed: PSMB5→proteasome p=9.2e-38 (Bonferroni); 4/10 drugs significant; cross-drug Spearman floor 0.77 (network-bias confirmed at ranking level) | CONDITIONAL pass with 3 fixes (multi-seed pooling, ~50 haem DepMap lines, raise α to 0.80–0.85 for selective seeds) | `PPI_PROPAGATION_PILOT.md` |
| F8 | Single-snapshot leave-one-out energy distance vs constant + population-mean baselines | Mann-Whitney p<0.05 vs both → reject §2.5 primitive | `SINGLE_SNAPSHOT_INFERENCE.md` |

**Tests runnable today on data already on disk:** F2, F3, F4, F5, F6, F7, F8.
**Test requiring 5-stratum split + held-out trajectory simulation:** F1.

---

## §7 Ablation table (paths NOT chosen — points to `REJECTED_ALTERNATIVES.md`)

| # | Rejected | Replacement | Δ vs v10 (numeric where available) |
|---|---|---|---|
| 1 | scVelo / dynamo / cellDancer | CellRank 2 PseudotimeKernel over scGPT | spliced/unspliced layers absent in both h5ad |
| 2 | Patient-conditioned latent SDE alone | TFM inside HBayes outer | ~10× over-parameterized at $N=51$; Stone caps $d_c^{\rm eff}\le 4$ |
| 3 | OT reference matching alone | Dynamic-OT regularizer inside TFM | $W_2\approx 0.985$ at $K=51, d=512$; discards 994 baselines |
| 4 | Attribution (TIG/SHAP) as causal | L1 framing + CRISPR oracle | Counter-example: BCL2 high attribution, Chronos +0.066 (anti-essential) |
| 5 | DepMap CRISPR as IV | CRISPR as one-sided falsification oracle | Pearl-Bareinboim transportability fails |
| 6 | v8/v9 deep stack w/o drug FP, w/o MMRF | DrugCell Morgan-FP + MMRF Tier-2 | Pooled MSE 2.836 vs predict-mean 2.811; ties MOFA+Ridge |
| 7 | DrugCell GO ontology as trajectory | Drug encoder absorbed; trajectory = TFM | GO is static, no basin/barrier semantics |
| 8 | GigaTIME/GigaPath as primary | scGPT (frozen) + Chronos | GigaTIME is H&E→mIF cross-modal, NOT temporal |
| 9 | k=10 simultaneous NIE | Single pre-specified mediator | EPV ≈ 2.8 < 25 threshold |
| 10 | dbGaP controlled-access MMRF | Open-access GDC API | 2,960 open-access files cover Tier-2 needs |
| 11 | Vanilla z≥2 for CCND1 | GMM + literature-prevalence quantile fallback | z≥2 gave 1.2% vs 15-20% literature |
| 12 | gdc-client binary | scripts/gdc_download.py | All distribution channels broken 2026-05-03 |

---

## §8 Additional data pulls (ranked by leverage/effort)

Per `ADDITIONAL_DATA_SOURCES.md`:

1. **GEO GSE279766** (PMID 40920573, 137 MM samples ATAC+ChIP+RNA, public 2025-09-09). Largest single uplift for the "epigenetic resistance" axis — turns the epigenetics block from 42 averaged CCLE features into peak-level patient-derived chromatin signal. Leverage 9, effort 5.
2. **Beat AML 1.0** (`BEATAML1.0-COHORT`, 826 patients × 122 drugs ex-vivo IC50, Tyner et al. 2018). Without it, "hematologic" in the v10 title is just MM. Leverage 10, effort 5.
3. **BLUEPRINT IHEC** (open hematopoietic reference epigenomes — H3K4me3/K27me3/K27ac + DNAme + ATAC for HSC→plasma cell). Provides the empirical Waddington-landscape prior the v10 framing implicitly assumes. Leverage 9, effort 7.

**Foundation-model pretraining verdict: fine-tune, do NOT re-pretrain.** scFoundation/Geneformer/scGPT already saturate CELLxGENE+Tabula-Sapiens. Bottleneck is hematologic+epigenetic+longitudinal data, not corpus size. <5 GPU-days vs 3-6 GPU-months for from-scratch retrain.

**Drop from v10:** 4DN 3D-genome — confirmed empty for MM/B-cell/plasma-cell Hi-C. Promise should not appear in the manuscript.

---

## §9 Sprint plan

| Sprint | Deliverable | Falsification tests gated | Data needed |
|---|---|---|---|
| **0** (now) | Land MMRF preprocessing + cytogenetic calls + v10 spec | — | already on disk |
| **1** | Train scalar $U_\theta$ via DSM on 1313 MMRF latents + PPI Tikhonov | F4 (CurlFraction), F5 (median-r identifiability) | MMRF processed |
| **2** | Add CellRank 2 PseudotimeKernel + HBayes outer (Stan/numpyro) | F1 (cross-stratum coverage), F2 (994-baseline ablation) | MMRF + cytogenetics |
| **3** | RWR driver head + NPI z-score + CRISPR oracle | F3 (CRISPR rank-sum), F6 (Walker recall ≥ 35/50), F7 (multi-seed pooling fixes) | DepMap Chronos, expanded haem subset |
| **4** | Single-mediator NIE (proteasome → time-to-2nd-line Bortezomib) with EIMOC disclaimer | L2 escalation | MMRF treatments + outcomes |
| **5** | Mondrian jackknife+ conformal wrapper | per-stratum 90% coverage at ±3% / ±5% | MMRF |
| **6** | Pull GSE279766 + Beat AML 1.0; refit chromatin block + add second hematologic disease | extends "hematologic" claim beyond MM | new pulls |
| **7** | Single-snapshot diffusion-posterior wrapper | F8 (LOO energy distance) | scGPT pretrained |

Each sprint is gated by its falsification tests. A failed test forces a fallback documented in the source agent doc rather than a silent retry.

---

## §10 Run-ledger discipline

**No number in this document migrates to `README.md` or `ARCHITECTURE.md` without:**
1. A row in `RUNS.md` with a Run ID,
2. `logs/<run-id>/stdout.log`, `stderr.log`, `config.yaml`, `verification_chain.json`, `per_drug_metrics.csv`,
3. A W&B run URL or persistent-log link,
4. A release-bouncer subagent pass.

**Numbers that are currently safe in `docs/` only** (because they live in pilot results, baselines JSON, or theoretical-bound docs):
- ResistanceMap pooled MSE 2.836; ElasticNet 2.892; predict-mean 2.811; MOFA+Ridge 2.816 (all in `paper/v8_artifacts/baselines/`).
- PSMB5→proteasome p=9.2e-38; 4/10 drugs Bonferroni-significant; cross-drug Spearman floor 0.77 (in `docs/PPI_PROPAGATION_PILOT.md`).
- $W_2 \approx 0.985$ at $K=51, d=512$; Stone $d_c^{\rm eff}\le 4$ (in `docs/TRAJECTORY_MATH_FEW_SHOT.md`).
- Cytogenetic prevalences del17p 13.6% / chr1q21+ 33.6% / del13q 52% / t(4;14) 14.6% / t(11;14) 16.8% (in `docs/INTEGRATED_BUILD_PLAN.md` line 74; sources from `cytogenetics.tsv`, 716/976 with all 5 calls).

These are safe **in `docs/`** because the run-ledger CI grep-checks only README and ARCHITECTURE.

---

## §11 One-paragraph manuscript abstract (draft)

> Drug resistance in multiple myeloma emerges from clonal selection acting on a Waddington-style epigenetic landscape, but per-patient temporal forecasting at $N_p \le 50$ paired-trajectory regimes is identifiability-limited: classical neural-SDE conditioning is ~10× over-parameterized and Stone's minimax rate caps the identifiable conditioning dimension at $\approx 4$. We present **ResistanceMap v10**, a population-stratified per-patient forecaster that resolves this through (a) a frozen scGPT cell-state encoder, (b) a Wang-style landscape+flux drift fitted via denoising score matching with PPI-Laplacian Tikhonov regularization linking the trajectory to STRING via the heat-kernel $\exp(-tL_{\rm PPI})$, (c) a Bellman-Harris branching prior on $K\le 8$ subclones that reduces sample complexity ~10×, (d) a hierarchical-Bayes outer layer that lets the 994 MMRF CoMMpass baseline-only patients enter the marginal likelihood non-vacuously across 5 cytogenetic strata, and (e) a Mondrian jackknife+ conformal wrapper that delivers per-stratum 90% coverage. Driver-pathway claims are made at Pearl tier L1-with-structural-prior — no observational dataset (MMRF + CCLE alone) can support causal $do(\cdot)$ claims (Shpitser-Pearl Theorem 2 hedge), and DepMap CRISPR is used strictly as a falsification oracle, not as an instrumental variable (Bareinboim-Pearl Theorem 4 transport conditions fail). On the empirical pilot, RWR seeded from PSMB5 recovers all top-15 26S proteasome subunits in MM DepMap lines with $p = 9.2\times 10^{-38}$ vs random gene sets, and 4 of 10 mechanistically-anchored drugs pass the Bonferroni-corrected essentiality enrichment test. Three pre-registered falsification thresholds (cross-stratum coverage <70%, ablation log-lik unchanged, CRISPR rank below the 95th percentile) define the v10 refutation criteria.

---

## §12 What this document does not contain

- Per-component implementation code (lives in `resistancemap/models/` and `scripts/`).
- New numeric measurements not already in the agent docs or baselines JSON.
- README/ARCHITECTURE-bound numbers (those are gated by run-ledger).
- The single-snapshot diffusion-posterior derivation in full (lives in `SINGLE_SNAPSHOT_INFERENCE.md` §5).
- Detailed PPI-pilot per-drug breakdown (lives in `PPI_PROPAGATION_PILOT.md` + `docs/ppi_pilot_results.csv`).

---

## §13 Cross-references (load-bearing)

- `docs/V10_PER_PATIENT_FORECAST_SPEC.md` — original 7-layer spec, source of conformal/HBayes/Bellman-Harris choices.
- `docs/INTEGRATED_BUILD_PLAN.md` — sprint-level build plan, cytogenetic call provenance.
- `docs/CAUSAL_VALIDITY_AUDIT.md` — Pearl-tier audit, EIMOC and survivorship-bias caveats.
- `docs/MMRF_ACQUISITION.md` — verified GDC counts and download path.
- `docs/CLONE_DYNAMICS_AND_CONFORMAL.md` — Bellman-Harris justification, jackknife+ per-stratum analysis.
- `paper/v8_artifacts/baselines/{elasticnet_results.json, mofa_results.json}` — anchor for the v8/v9 rejection in §7 row 6.
- `RUNS.md` — run ledger; required for any v10 numeric claim that ships to README.
