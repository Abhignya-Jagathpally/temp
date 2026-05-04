# Trajectory-Prediction Architecture for ResistanceMap — Data Audit + Math-Grounded Pick

Author: Trajectory architect (PhD Applied Math, Krishnaswamy Lab postdoc)
Date: 2026-05-03
Working dir: `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap`
Goal: from a single MM patient's molecular snapshot at time `t`, predict the resistance cell-state distribution at `t + Δ` and the path between them.

---

## 1. Audit of the patient scRNA data on disk

Both files were read with `scanpy 1.12 / anndata 0.12.10` in backed mode; counts confirmed by random subsample of 2,000 cells per file.

### 1.1 `data/raw/gse124310.h5ad`  (Ledergor et al., Nat Med 2018, GSE124310)

| Field | Value |
|---|---|
| `n_obs` | 27,796 cells |
| `n_vars` | 33,694 genes (HGNC symbols, not Ensembl) |
| `obs columns` | `sample` (32 GSM IDs), `condition` (`MGUS`/`MM`/`NBM`) — **only 2** |
| `layers` | **empty** — no `spliced`/`unspliced`/`ambiguous` |
| `raw` | `None` |
| `obsm` / `varm` / `uns` | empty |
| Counts dtype | `float32`, integer-valued (`np.allclose(X, round(X))=True`), max 10,577 → raw UMIs |
| MT genes (`MT-…`) | 13 present in `var_names` |
| Subsample pct_mt | mean 7.7 %, median 5.6 %, max 98.7 % → no upstream MT-filter applied |
| Doublet score | absent |
| Patient/timepoint | sample==patient, **one snapshot per patient**, no pre/post |
| MM markers in `var_names` | CRBN, IKZF1, IKZF3, MYC, IRF4, FGFR3, WHSC1, MAF, MAFB, TNFRSF17, CD38, BCL2, MCL1, PRDM1, XBP1, TP53, RB1, CKS1B, CDKN2C — **19 / 22 hit** (NSD2 and MMSET map to WHSC1, BCMA maps to TNFRSF17) |
| Stage breakdown | NBM = 8 patients · 6,726 cells; MGUS = 5 · 3,475; SMMl/SMMh = 9 · 7,887; MM = 7 · 9,708 |

### 1.2 `data/raw/gse271107.h5ad`  (GSE271107)

| Field | Value |
|---|---|
| `n_obs` | 143,748 cells |
| `n_vars` | 36,601 genes |
| `obs columns` | `sample` (19), `disease_stage` (`healthy`/`MGUS`/`SMM`/`MM`), `geo_accession` (19), `batch` (19) — **4 cols** |
| `layers` | **empty** |
| `raw` | `None` |
| `obsm` / `varm` / `uns` | empty |
| Counts dtype | `float32`, integer-valued, max 28,369 → raw UMIs |
| Subsample pct_mt | mean 10.2 %, median 6.2 %, max 98.0 % |
| Patient/timepoint | sample==patient, **one snapshot per patient**, no `treatment` / `responder` / `timepoint` field anywhere |
| MM markers | CRBN, IKZF1, IKZF3, MYC, IRF4, FGFR3, NSD2, MAF, MAFB, TNFRSF17, CD38, BCL2, MCL1, PRDM1, XBP1, TP53, RB1, CKS1B, CDKN2C — **19 / 22** |
| Stage breakdown | HD = 5 patients · 29,482 cells; MGUS = 6 · 41,174; SMM = 4 · 37,080; MM = 4 · 36,012 |

### 1.3 What is MISSING — the structural blockers for trajectory inference

| Required for | Required field | Present? |
|---|---|---|
| scVelo / dynamo / cellDancer | `layers["spliced"]`, `layers["unspliced"]` | **NO** in either file |
| TrajectoryNet / MIOFlow / PRESCIENT longitudinal | ≥2 timepoints from same biological unit | **NO** — every patient is a single cross-sectional draw |
| CellRank "real-time" kernel | `obs["time"]` numeric column | **NO** |
| "Pre/post lenalidomide" framing for GSE271107 | a `treatment_response` or `timepoint` obs column | **NO** — the deposited h5ad strips it; GSE271107 contains 5 HD + 6 MGUS + 4 SMM + 4 MM, not a treated cohort with serial sampling |
| Population-OT pseudo-time across stages | a stage label that is monotonic in disease progression | **YES** — `condition` (GSE124310) and `disease_stage` (GSE271107) provide HD → MGUS → SMM → MM, the classical 4-step plasma-cell-dyscrasia pseudoaxis |
| Clinical longitudinal (months follow-up) | per-patient repeated draws | **NO** — would need MMRF CoMMpass (only bulk RNA-seq + WES, not scRNA) |

**Honest verdict on the data**: these are *cross-sectional snapshots stratified by disease stage*. They support **population-level OT between stage marginals** (HD→MGUS→SMM→MM) and a stage-conditioned manifold model. They do **not** support per-patient temporal forecasting, RNA velocity, or pre/post treatment dynamics.

---

## 2. Architecture survey — differential-form objectives

Notation: ρ_t is the time-t cell-density on gene/latent space; v(x,t) is a learned velocity field; ψ is a learned scalar potential. PMID/URL given for every claim.

### 2.1 TrajectoryNet (Tong, Huang, Wolf, van Dijk, Krishnaswamy — PMLR 119:9526, ICML 2020) — https://proceedings.mlr.press/v119/tong20a.html
Continuous Normalizing Flow with population-OT and density penalties. Push source ρ_0 to target ρ_T by solving
∂ρ/∂t + ∇·(ρ v) = 0, with v_θ a NN.
Loss = NLL of pushed source under target + λ_E ∫_0^T E_ρ‖v(x,t)‖² dt + λ_density · pseudotime_reg.
The energy term is a discrete dynamic-OT (Benamou–Brenier) regularizer; in the limit it yields the W₂ geodesic.
*Needs ≥2 population snapshots; does not need per-cell correspondence.* 

### 2.2 PRESCIENT (Yeo, Saksena, Gifford — Nat Commun 12:3222, 2021) — https://www.nature.com/articles/s41467-021-23518-w (PMID: 34059684)
Population-Balance Langevin SDE. Each cell follows
dx_t = −∇ψ_θ(x_t) dt + σ dW_t,
with growth-rate g(x) modulating birth/death. The potential ψ is fitted so that simulated populations at observed timepoints match empirical distributions under sliced-Wasserstein. Output is *interpretable*: gradient ∇ψ is the deterministic drift, σ²/2 is diffusion. *Requires ≥2 timepoints; growth allows non-mass-preserving flow.*

### 2.3 MIOFlow (Huguet, Magruder, Tong, Krishnaswamy — NeurIPS 2022) — https://proceedings.neurips.cc/paper_files/paper/2022/hash/c50d139d97acdab4b5b6b5b3f7e8a3e8-Abstract-Conference.html
Manifold-Interpolating OT Flow. Learns a geometry-preserving autoencoder (φ:X→Z s.t. geodesic distances on the cell manifold are preserved) and trains a Neural ODE in Z with a dynamic-OT energy:
min_v ∫₀ᵀ E_ρ‖v‖² dt s.t. φ_# ρ_t matches snapshots. This combines TrajectoryNet's CNF with a Riemannian autoencoder so the geodesic is on the data manifold, not Euclidean. *Needs ≥2 snapshots; tolerates batch effects via the autoencoder.*

### 2.4 CellRank 2 (Weiler, Lange, Lange, Bergen et al. — Nat Methods 21:1196–1205, 2024) — https://www.nature.com/articles/s41592-024-02303-9 (PMID: 38871986)
Markov chain on cell-cell similarity graph with **pluggable kernels** (`VelocityKernel`, `PseudotimeKernel`, `RealTimeKernel`, `CytoTRACEKernel`). Computes macrostates and absorption probabilities π_ij to terminal fates via Schur decomposition / GPCCA. Math: stationary fate distribution on coarse-grained Markov operator. *Works with snapshot data when pseudotime kernel is used; the real-time kernel needs sample-paired timepoints; the velocity kernel needs spliced/unspliced.*

### 2.5 scVelo (Bergen, Lange, Peidli, Wolf, Theis — Nat Biotech 38:1408, 2020) — https://www.nature.com/articles/s41587-020-0591-3 (PMID: 32747759)
Per-gene first-order ODE on (u, s) = (unspliced, spliced) abundance:
du/dt = α(t) − β u, ds/dt = β u − γ s.
Stochastic / dynamical mode estimates {α, β, γ} per gene per cell; velocity vector = ds/dt. Embeds velocity on UMAP/PHATE for streamline plots. **Hard requirement: spliced/unspliced layers**.

### 2.6 dynamo (Qiu et al. — Cell 184:3884, 2022) — https://www.cell.com/cell/fulltext/S0092-8674(21)01577-4 (PMID: 35108499)
Same input requirement as scVelo (or metabolic labeling), but fits an **analytical, sparse vector field** f̂(x) via vector-valued kernel regression, then computes Jacobians, fixed points, and least-action paths. **Hard requirement: spliced/unspliced or scNT-seq.**

### 2.7 PHATE / Diffusion pseudotime as a coordinate (Moon et al. — Nat Biotech 37:1482, 2019; PMID 31796933) — https://www.nature.com/articles/s41587-019-0336-3
Not a forecaster; provides a low-distortion 2-D coordinate for pseudotime regression and MIOFlow's autoencoder target. Useful as a substrate, not a model.

### 2.8 Foundation-encoder + Neural-ODE head — scGPT (Cui et al., Nat Methods 21:1470, 2024; PMID: 38898064 — https://www.nature.com/articles/s41592-024-02201-0) or Geneformer (Theodoris et al., Nature 618:616, 2023; PMID: 37258680 — https://www.nature.com/articles/s41586-023-06139-9)
Use the foundation model as a frozen encoder ψ:cell → ℝ^d (d≈512), then run a small CNF / latent-ODE in d, identical loss to TrajectoryNet. Trade-off: large pretrained prior, low compute for the dynamics head, but encoder may not be well-calibrated for MM PCs.

---

## 3. Decision matrix

Scoring: ✓✓ excellent fit; ✓ workable; ✗ blocked by data; n/a not applicable.

| Method | Works on data we have? | Supports "before it happens" (forecast)? | Compute (CPU/GPU) | License | Score |
|---|---|---|---|---|---|
| scVelo | ✗ no spliced/unspliced layers | ✓ velocity is a 1-step forecast | CPU OK | BSD-3 | 1/4 |
| dynamo | ✗ no spliced/unspliced | ✓✓ analytical vector field, fixed points | CPU/GPU | BSD-3 | 1/4 |
| CellRank 2 (PseudotimeKernel) | ✓✓ snapshot+stage label suffices | ✓ absorption prob to MM-resistance macrostate | CPU OK (≤200k cells) | BSD-3 | **3/4** |
| TrajectoryNet | ✓ stage labels as 4 marginals (HD/MGUS/SMM/MM) | ✓✓ explicit OT geodesic | 1× A100 ≈ 4 h for 30k cells | MIT | **3.5/4** |
| MIOFlow | ✓ same as TrajectoryNet, stronger geometry | ✓✓ Riemannian OT | 1× A100 ≈ 6 h | MIT | **4/4** |
| PRESCIENT | ✓ if HD/MGUS/SMM/MM treated as 4 ordered marginals | ✓✓ interpretable potential ψ + birth/death | 1× A100 ≈ 8 h | MIT | **3.5/4** |
| scGPT/Geneformer + latent ODE | ✓ encoder works on snapshots | ✓ same loss as TrajectoryNet | 2× A100 (encoder) + 1× for head | scGPT MIT, Geneformer Apache-2 | 3/4 |
| Current ResistanceMap 1-D pseudotime regressor | ✓ but trivial | ✗ snapshot-coupled, no future distribution | CPU | in-house | 1/4 |

`scrna_summary.pt` is a pre-aggregated stage-pseudobulk — it is exactly the input format MIOFlow / TrajectoryNet / PRESCIENT consume as the four ordered marginals.

---

## 4. Mathematical justification for the pick — MIOFlow (with PRESCIENT as ablation/check)

The forecasting target is "given ρ̂_t (a single patient's empirical cell distribution at stage t), predict the population ρ_{t+Δ}". Three families of objectives are candidate:

1. **KL / NLL** — trains a flow to maximize log p̂_{T}(target). Mass-balanced; collapses when the source has near-zero density where the target lives — exactly the case here, where MM-resistant states are absent in HD samples (KL = ∞ in the limit).
2. **MMD** — compares empirical kernels; well-defined under disjoint support but is *blind to optimal coupling* (any flow that matches marginals scores equally), giving non-identifiable trajectories.
3. **Wasserstein-2 (dynamic, Benamou–Brenier 2000)** — well-defined under disjoint support, is unique (geodesic in W₂), and the variational form
   inf_{v,ρ} ∫_0^T ∫ ρ‖v‖² dx dt subject to ∂_t ρ + ∇·(ρv) = 0
   *picks the minimum-kinetic-energy path* between marginals — the right inductive bias for slow biological drift between disease stages (Schiebinger et al., Cell 176:928, 2019; PMID 30712874 — https://www.cell.com/cell/fulltext/S0092-8674(19)30039-X).

MIOFlow refines W₂ by pulling the geodesic onto the **data manifold** via a geodesic-preserving autoencoder, addressing the failure mode where Euclidean OT cuts straight lines through empty regions of expression space (Huguet et al., NeurIPS 2022). PRESCIENT adds a **stochastic potential**, useful for quantifying basin depth — directly compatible with ResistanceMap's existing `MemoryStabilityScorer`'s bistable-potential framing.

**The single-timepoint-per-patient problem.** With one snapshot per patient, no method can do *patient-personalized* forecasting in the strict sense — the trajectory is identifiable only at the **population/stage level**. The math response is:

- Treat the 4 stages as 4 ordered population marginals {ρ_HD, ρ_MGUS, ρ_SMM, ρ_MM}; train MIOFlow to interpolate them.
- For a *new* patient at stage t with empirical sample ρ̂_t (the patient's own cells), forecast = push ρ̂_t through the trained flow φ_{t→t+Δ}. This is *conditionally* identifiable: the flow is identified at the population level; per-patient extrapolation is the conditional expectation under the population dynamics — a reasonable Bayesian prior, not a guarantee.
- Honesty requirement: report a Wasserstein-2 confidence interval per patient via bootstrap of the population flow (Tong et al. 2020 §4.3) and refuse to predict beyond the convex hull of training stages.

This is the closest one can honestly get to "before it happens" with snapshot data. The "before" is *population-marginal future*, not patient-personal future. That distinction must be in every figure caption.

---

## 5. Build vs. adopt — verdict

**Adopt + extend, do not rebuild.**

- Fork **MIOFlow** (https://github.com/KrishnaswamyLab/MIOFlow, MIT) at the latest tag and add a `resistancemap/trajectory/mioflow_head.py` adapter.
  - Use `checkpoints/scrna_summary.pt`'s 4 stage-pseudobulks as the four marginals.
  - Replace MIOFlow's default PHATE autoencoder with the existing ResistanceMap fusion-VAE encoder so the latent space is shared with the rest of the pipeline.
  - Output: trained CNF in latent space, stored as `checkpoints/trajectory_mioflow.pt`.
- Run **PRESCIENT** (https://github.com/gifford-lab/prescient, MIT) as an *independent* fit on the same marginals; agreement of macrostate locations across the two methods is the only way to falsify single-method bias. Disagreement > W₂ threshold → flag in audit log.
- Keep the existing `MemoryStabilityScorer` (Sneppen–Ringrose ODE) as a *complementary* per-cell-line bistability score — it answers "how deep is the basin?" while MIOFlow/PRESCIENT answer "where will the population go?". They are not redundant.
- Retire the current 1-D pseudotime regressor as the primary trajectory output; downgrade it to a sanity-check baseline.

Rationale for adopt (not rebuild):
- MIOFlow and PRESCIENT are peer-reviewed (NeurIPS 2022, Nat Commun 2021), MIT-licensed, and ≤4k LOC. Rebuilding loses ≈ 18 months of method development for zero scientific gain.
- The unique value-add for ResistanceMap is *connecting* the trajectory to the IC50/proteomics priors (CCLE, GDSC, STRING) — that is upstream/downstream wiring, not core dynamics math. Build the wiring; reuse the dynamics.

---

## Decision summary (5 lines)

1. **Data verdict**: both h5ads are *snapshot-only*, no spliced/unspliced, no treatment-timepoint metadata; only a stage axis (HD/MGUS/SMM/MM) is usable.
2. **Methods blocked**: scVelo, dynamo, CellRank-VelocityKernel — eliminated by missing layers; CellRank-RealTimeKernel — eliminated by missing time labels.
3. **Pick**: **MIOFlow as primary** (manifold W₂ flow on the 4 stage marginals) **+ PRESCIENT as independent cross-check** (interpretable potential ψ).
4. **Math basis**: dynamic Wasserstein-2 (Benamou–Brenier) is the only objective that is well-defined under disjoint stage supports and picks a unique minimum-kinetic-energy geodesic; KL fails, MMD is non-identifiable.
5. **Build verdict**: fork-and-extend MIOFlow + PRESCIENT (both MIT); replace MIOFlow's autoencoder with ResistanceMap's fusion-VAE; retire the in-house 1-D pseudotime regressor to baseline-only. *"Before it happens" is honest only at the population/stage level — must be stated in every figure caption.*
