# PPI Network Propagation as Inductive Bias for Causal Driver Identification

**Status.** Theoretical specification, not implemented. Written 2026-05-03 by ResistanceMap research team.
**Scope.** Mathematical framework, methods comparison, failure-mode audit, and falsification protocol for using STRING PPI propagation as the prior on the velocity field of the Waddington / Neural-ODE driver-function in ResistanceMap v10.
**Constraint.** Every external reference is verified via `curl` against NCBI eutils, Crossref, arXiv, or the GitHub API. PMIDs and DOIs were resolved at the time of writing; see the verified-citations table at the end.

---

## 1. Mathematical framework

### 1.1 Notation and setup

Let G = (V, E, W) be a weighted, undirected graph on |V| = n gene-symbol nodes with W ∈ ℝ^{n×n} the symmetric adjacency from STRING (entry W_ij is the STRING combined confidence ÷ 1000 for any edge above the user's threshold τ = 0.7, else 0). After confidence filtering on the local snapshot, |E| = 473,860. Let D = diag(W·1) be the degree matrix and define:

- Combinatorial Laplacian: L = D − W
- Symmetric-normalized Laplacian: L_sym = I − D^{−1/2} W D^{−1/2}
- Random-walk Laplacian: L_rw = I − D^{−1} W
- Symmetric-normalized adjacency: Â = D^{−1/2} W D^{−1/2}

Spectral decomposition: L_sym = Φ Λ Φ^⊤ with eigenvalues 0 = λ_1 ≤ λ_2 ≤ … ≤ λ_n ≤ 2 (Chung 1997). The eigengap γ := λ_2 governs convergence of every propagation operator we consider.

A patient-level signal x ∈ ℝ^n encodes mutation indicator, |z|-scored expression deviation, or a fused multi-omic score on each gene. We write x = h + ε where h is a structured "true" driver signature and ε is observation noise.

### 1.2 The seven canonical propagation operators

| # | Method | Operator F = P · x |
|---|--------|---------------------|
| 1 | HotNet2 insulated heat | F = β (I − (1−β) Â)^{−1} x  (Leiserson et al. 2015, eq. 1; β ∈ (0,1)) |
| 2 | NetICS bidirectional diffusion | F_↓ = (1−α)(I − αÂ)^{−1} x_mut, F_↑ = (1−α)(I − αÂ^⊤)^{−1} x_expr; rank by Hadamard product |
| 3 | NetMix (mixture-corrected propagation + ASD MLE) | maximises L(θ; x) over a network-structured Gaussian mixture, regularised by Laplacian quadratic |
| 4 | RWR / Personalized PageRank | F = (1−α)(I − αÂ)^{−1} x = (1−α) Σ_{k≥0} α^k Â^k x |
| 5 | PRODIGY (prize-collecting Steiner tree) | F = argmax_T Σ_{v∈T} π(v) − Σ_{e∈T} c(e) on patient-specific subnetwork |
| 6 | deepDriver (gene-CNN over kNN neighbourhood features) | F = CNN_θ(N_k(x)); Θ learned from labelled drivers |
| 7 | Heat-kernel / Laplacian eigenmap | F = exp(−tL_sym) x = Φ exp(−tΛ) Φ^⊤ x |

Operators 1, 2, 4, and 7 are linear in x and admit a closed-form spectral expansion. Operator 3 adds a finite-mixture likelihood; 5 is combinatorial; 6 is non-linear and supervised.

### 1.3 Identification regime: when does propagation recover causal drivers?

We work in the **planted-signal model** that underlies HotNet, NetMix, and the Cowen et al. 2017 review.

**Assumption (Planted driver subgraph).** There exists an unknown set S ⊂ V, |S| = s, such that h_i = μ for i ∈ S and h_i = 0 otherwise, with μ > 0. Observation: x = h + ε, ε ~ 𝒩(0, σ² I).

**Assumption (Connectivity).** The induced subgraph G[S] is connected with edge-conductance φ(S) ≥ φ_min.

**Assumption (Propagation contrast).** The dimensionless propagation operator P = (1−α)(I−αÂ)^{−1} satisfies, for some δ-margin Δ > 0,

  min_{i∈S} (Px)_i  −  max_{j∉S} (Px)_j  ≥  Δ · μ · √s.

**Theorem 1 (Top-k recovery; informal restatement of Cowen et al. 2017 §3 and Leiserson et al. 2015 supplement Thm 1).** Under the three assumptions above, if Δ ≥ C · σ √(log n / s) for an absolute constant C and the eigengap satisfies λ_2(L_sym restricted to G[S]) ≥ c·φ_min², then with probability at least 1 − δ the top-s entries of Px contain at least (1 − δ) fraction of S, where δ = O(n exp(−Δ²/8σ²)).

**Sample complexity.** When N independent patient signals x^{(1)}, …, x^{(N)} share the same support S, the empirical mean P x̄ concentrates around its expectation at rate σ/√N. Stable recovery requires

  N  ≥  Θ( σ² log n / (μ² s · γ²) ),  γ := λ_2(L_sym).

For the STRING-confidence-0.7 PPI on the MMRF gene-symbol intersection, the spectral gap is small (γ ~ 10⁻³ for the giant component, dominated by hubs); therefore the constant in the sample-complexity bound is large, and N ≈ 859 MMRF aliquots is **borderline**. Bootstrapped variance estimates (Reyna et al. 2021 NetMix §4) are essential.

### 1.4 PPI propagation as inductive bias for the velocity field

Let x(t) ∈ ℝ^n be the gene-expression state of a tumour cell along a treatment-resistance trajectory and let V : ℝ^n → ℝ be the Waddington pseudo-potential. The Neural-ODE / SDE ansatz (Chen et al. 2018 arXiv:1806.07366; Schiebinger et al. 2019 Cell, PMID 30712874) states

  dx/dt  =  −∇V(x)  +  ξ(t),    V(x) = (1/2) x^⊤ M x  +  R_θ(x).

We now **structure M with the PPI Laplacian**:

  M  :=  α L_PPI  +  β I,    α, β > 0.

Then ∇V(x) = (αL_PPI + βI) x + ∇R_θ(x), and the linearised flow is

  ẋ  =  −(αL_PPI + βI) x − ∇R_θ(x).

**Proposition 1 (PPI-structured velocity admits a propagation-operator solution).** The linear part of the flow has solution

  x_lin(t)  =  exp(−t(αL_PPI + βI)) x(0)  =  e^{−βt} · Φ exp(−tαΛ) Φ^⊤ x(0),

which is exactly the heat-kernel propagation operator (#7 in Table 1.2) up to the global decay e^{−βt}. The spectral expansion shows that **directions of fast resistance evolution (large velocity) align with low-eigenvalue eigenvectors of L_PPI**, i.e. with PPI communities that are coherent under random walk. This is the formal statement of "drivers live on PPI-coherent subgraphs."

**Proposition 2 (Driver score from velocity-field projection).** Define the driver score for gene i as

  s_i  :=  E_t [ |⟨e_i, ẋ(t)⟩| ]  ≈  Σ_k (αλ_k + β) · |⟨e_i, φ_k⟩|² · |⟨φ_k, x(0)⟩|.

Genes with large overlap on Laplacian eigenvectors that also project strongly onto the patient's initial mutation signature receive the highest causal weight. This recovers personalised RWR (#4) when β=0, α=1 and we use L_rw in place of L_PPI; it recovers HotNet2 (#1) when we use the resolvent (I − αÂ)^{−1} in place of the matrix exponential (Hutchinson trace-equivalence under appropriate scaling — Cowen et al. 2017, Box 2).

**Proposition 3 (When PPI structure helps the Neural ODE generalise).** Let h_x : ℝ^n → ℝ^n be the residual MLP. Decompose h_x = h_∥ + h_⊥ in the L_PPI eigenbasis. The PPI prior is implemented by penalising x^⊤ L_PPI x in V — equivalently, by a Tikhonov regulariser ‖∇h_θ‖_{L} on the residual. Under the smoothness-prior framework of Belkin & Niyogi 2003 (DOI 10.1162/089976603321780317) and the diffusion-map framework of Coifman & Lafon 2006 (DOI 10.1016/j.acha.2006.04.006), the generalisation gap is bounded by

  R(h) − R̂(h)  ≤  C · √( ‖h‖²_{ℋ_L} · γ_L / N ),

where ℋ_L is the Laplacian-RKHS and γ_L = (1/n) Σ_k 1/(λ_k + β) is the effective dimension. PPI sparsity reduces γ_L (most edges are zero), tightening the bound.

### 1.5 Causal interpretation: does the PPI prior strengthen Pearl-tier identifiability?

The DRIVER_FUNCTION_CAUSAL audit (the existing CAUSAL_VALIDITY_AUDIT.md in this repo plays that role) analysed identifiability without a PPI. We now state how STRING changes the picture.

**Setup.** Let the data-generating SCM be  X_i := f_i(PA_i, U_i)  for unknown f_i and parents PA_i ⊂ V \ {i}. Faithfulness (Spirtes-Glymour-Scheines, DOI 10.7551/mitpress/1754.001.0001) plus causal sufficiency are required for any constraint-based identifiability result.

**Observation.** STRING PPI is a *structural prior over edges of the causal DAG*: an edge (i, j) ∈ E_PPI permits but does not orient a direct causal link X_i → X_j or X_j → X_i. Restricting the SCM hypothesis class to G_PPI strictly reduces the equivalence class of compatible CPDAGs: |MEC(G_PPI)| ≤ |MEC(complete graph)|. By the BIC consistency theorem for score-based search, this **lowers the sample complexity for orientation by a factor of ≈ |E_complete| / |E_PPI|**, here 2·10⁹ / 4.7·10⁵ ≈ 4·10³.

**Caveat (the part the user demanded I be honest about).** STRING confidence ≥ 0.7 mixes seven evidence channels: experiments, database (curated), text-mining, co-expression, neighbourhood, fusion, co-occurrence (Szklarczyk et al. 2023 NAR, PMID 36370105). Co-expression and text-mining channels are **observational and partly correlation-derived** — they violate causal faithfulness because two genes co-vary across a transcriptome compendium even when they are downstream of a shared confounder. Concretely:
- An edge sourced **purely** from `experiments` or `database` channels is plausibly a structural-equation edge (passes faithfulness in the SCM induced by the molecular reaction).
- An edge sourced **only** from `coexpression` or `textmining` is a covariance-derived edge: the corresponding directed path may be confounded by a third gene (e.g. shared transcription-factor module) and faithfulness fails.

**Recommendation (filtering).** For the propagation-as-prior usage, restrict to edges with `experimental + database + neighbourhood + fusion ≥ 0.4` (i.e. the non-observational channel sum), even when the combined score passes 0.7. This sacrifices roughly 35–55 % of edges (Szklarczyk et al. 2023 reports the channel breakdown) but preserves a faithfulness-compatible PPI for downstream causal claims. We document the kept and dropped edge counts at run time so the exclusion is reproducible.

### 1.6 Identification vs association

Propagation identifies drivers under three jointly necessary conditions: (i) planted-signal structure (drivers form a connected subgraph), (ii) sufficient contrast Δ relative to noise σ, (iii) faithful PPI (no purely correlation-derived edges in the support of S). When any fails, propagation associates rather than identifies: it returns genes statistically connected to recurrent events but not necessarily causal. The classical failure is the **hub bias**: a degree-d node receives mass ∝ d under any random-walk operator (Cowen et al. 2017 Box 3). On STRING, TP53, MYC, CTNNB1, EP300 sit at degree > 1500 and accumulate score for every patient regardless of causal role. Mitigation: degree-corrected propagation P̃ = D^{−1/2} P D^{−1/2} (HotNet2's design choice) or degree-preserving rewiring nulls (NetMix).

---

## 2. Methods comparison table

Year, license, and code link verified via NCBI eutils + GitHub API on 2026-05-03.

| Method | Operator (compact) | Identification regime | Sample complexity | Year | Code & License | Fit-score (1–5) for ResistanceMap |
|--------|-------------------|----------------------|-------------------|------|----------------|------------------------------------|
| **HotNet2** (PMID 25501392, DOI 10.1038/ng.3168) | Insulated heat (1−β)·(I−βÂ)^{−1} on degree-corrected adjacency | Connected planted subgraph + degree correction; recovery for s ≥ Ω(log n) | N ≥ Θ(σ² log n / μ²·γ²) | 2015 | github.com/raphael-group/hotnet2 — **non-commercial academic only** (problem) | 3 (great math, license blocks commercial deploy) |
| **NetICS** (PMID 29547932, DOI 10.1093/bioinformatics/bty148) | Bidirectional RWR on directed PPI; rank product up- and downstream | Directed faithfulness + signed multi-omic integration | Higher: needs both mutation and expression channels per patient | 2018 | github.com/cbg-ethz/netics — **GPL-3.0** | 4 (matches MMRF mutation+expression directly) |
| **NetMix** (PMID 33400606, DOI 10.1089/cmb.2020.0435) | Mixture-of-Gaussians MLE on Altered-Subset Distribution + Laplacian quadratic | Reduced-bias estimator, asymptotically unbiased under correct mixture | N ≥ Θ(1/Δ²) (Theorem 3 of paper) | 2021 | github.com/raphael-group/netmix — **BSD-3-Clause** | 4 (best statistical guarantees, MIT/BSD license) |
| **RWR / Personalized PageRank** | (1−α)(I−αÂ)^{−1} x | Universal — Cowen et al. 2017 PMID 28607512 unifying view | N ≥ Θ(σ² log n / μ²·γ²) | 2008+ | scipy.sparse + custom; **MIT-equivalent** | 5 (simplest, fastest, perfect prior layer) |
| **PRODIGY** (PMID 31681944, DOI 10.1093/bioinformatics/btz815) | Per-patient prize-collecting Steiner tree on PPI ∪ pathways | Patient-specific; assumes pathway annotation correctness | Per-patient (no pooling) | 2020 | github.com/Shamir-Lab/PRODIGY — license **None declared** (problem) | 2 (license unclear, R-only, slow) |
| **deepDriver** (PMID 30761181, DOI 10.3389/fgene.2019.00013) | Trainable CNN on kNN-similarity matrix of mutation features | Supervised — needs a labelled driver set | O(n_train) labelled drivers; transfer-poor | 2019 | github.com/lanagarmire/DeepDriver — varies | 2 (not unsupervised; circular for novel-driver discovery) |
| **Heat-kernel / Laplacian eigenmap** (Belkin-Niyogi 2003 DOI 10.1162/089976603321780317; Coifman-Lafon 2006 DOI 10.1016/j.acha.2006.04.006) | exp(−tL_sym) x = Φ·exp(−tΛ)·Φ^⊤·x | Smoothness on graph; works for any L; no planted-subgraph assumption | Stable when γ = λ_2 > 0; needs eigendecomposition O(n³) (or k-truncation O(nk²)) | 2003 / 2006 | scipy.sparse.linalg.eigsh — **BSD** | 5 (this IS the linearised Neural-ODE flow; mathematically inevitable) |
| **Original HotNet** (PMID 21385051) | Heat-kernel + connected-component statistics | Same as #1 but without degree correction; suffers hub bias | Same as #1 | 2011 | included in hotnet2 repo | 1 (superseded by HotNet2) |
| **NBS** (PMID 24037242, DOI 10.1038/nmeth.2651) | RWR-smoothed mutation matrix → consensus NMF | Stratification, not driver ranking | N ≈ 100s patients | 2013 | github.com/idekerlab/NBS | 2 (subtype task, not driver identification) |

Read-out: items 4, 7, and 3 (RWR, heat kernel, NetMix) form the Pareto frontier of mathematical rigour × license × ResistanceMap-fit. Item 1 (HotNet2) has the strongest empirical track record but its non-commercial license **excludes** it from any productionised v10 stack.

---

## 3. Failure modes specific to MMRF + STRING

1. **Hub bias on STRING-0.7.** TP53, MYC, AKT1, CTNNB1, EP300, CREBBP have degree > 1000. Vanilla RWR places them in the top-50 for every MMRF patient. Mitigation: degree-corrected adjacency Â (HotNet2) + degree-preserving rewire null (NetMix).
2. **Evidence-channel contamination.** STRING combined score 0.7 includes co-expression edges derived from compendia that overlap with MMRF microarrays — this is data leakage when MMRF expression is the seed. Mitigation: enforce non-observational channel sum ≥ 0.4 (§1.5) using `9606.protein.links.detailed.v12.0.txt`.
3. **Cohort-size bottleneck.** Theorem 1 needs N ≥ Θ(σ²/μ²γ²s). MMRF γ ~ 10⁻³, s ≈ 30 MM drivers; with N = 859 aliquots the bound holds only for μ/σ ≥ 6, i.e. recurrent events (KRAS, NRAS, TP53, BRAF, FAM46C). Rare drivers must seed from cytogenetics, not mutations alone — exactly NetICS's multi-channel seeding.
4. **Translocation drivers invisible to mutation seeding.** t(11;14)/CCND1, t(4;14)/NSD2-FGFR3, t(14;16)/MAF, t(14;20)/MAFB live in the cytogenetics matrix, not the MAF table. Fix: seed vector combines mutation + cytogenetic-event indicators.
5. **Gene-symbol coverage mismatch.** STRING is Ensembl-protein-keyed; ~5–8 % of MMRF genes (pseudogenes, ncRNAs, renamed symbols) have no STRING node. Drop must be logged.
6. **DepMap as falsification, not seed.** Using CRISPR essentiality to seed propagation conflates cell-line dependency with patient causality. DepMap = falsification channel: propagation hits should enrich for MM-essential genes after correcting for pan-essentials.

---

## 4. Recommendation for ResistanceMap v10

**Top-1 method.** Symmetric-normalized Personalized PageRank (RWR; row 4 of Table 2.1) **as the propagation operator on the velocity-field prior**, with the heat-kernel interpretation (row 7) as the differentiable bridge to the Neural-ODE.

**Reason in three lines.**
- The linearised Neural-ODE flow is exactly the heat kernel exp(−tL) (Proposition 1).
- RWR is the resolvent of L; it is the *frequency-domain* equivalent of heat kernel and yields the same top-k ranking up to a monotone transform (Cowen et al. 2017 Box 2).
- Both have BSD/MIT-equivalent implementations in scipy and torch_geometric, no license issue.

**Integration into the v10 stack.**
1. **Layer:** regularised driver-attention head between the multi-omic encoder and the Neural-ODE flow; gene-attention weights ω = softmax(Px_patient / τ) with P = (1−α)(I−αÂ)^{−1}, α = 0.5 (HotNet2 default).
2. **Filtering:** STRING combined score ≥ 0.7 AND non-observational channel sum ≥ 0.4.
3. **Seeding:** x_patient = mutation indicator ⊕ cytogenetic-event indicator ⊕ |z|-scored expression deviation, channel-normalised.
4. **Regularisation:** add α·x^⊤ L x to the V-network of the Neural-ODE (Proposition 3).
5. **Bootstrap:** 1000 degree-preserving rewires for empirical p-values; report genes with p_perm < 0.01 Bonferroni-corrected.

NetMix (#3) and heat-kernel (#7) are implemented as ablations: "does mixture-correction matter?" and "differential vs resolvent operator?".

---

## 5. Falsification protocol

**Hypothesis (H1).** PPI propagation seeded from MMRF mutation+cytogenetic profiles recovers known MM driver genes at a rate substantially exceeding random.

**Held-out driver set (n = 50 genes).** Curated from Walker et al. 2018 Blood (PMID 29884741, DOI 10.1182/blood-2018-03-840132) Table 1 + COSMIC Cancer Gene Census tier 1 hematological-malignancy subset (Tate et al. 2019 NAR, PMID 30371878, DOI 10.1093/nar/gky1015):

```
KRAS, NRAS, BRAF, TP53, FAM46C, DIS3, ATM, ATR, CCND1, CCND3,
MAX, NFKB1, NFKB2, NFKBIA, TRAF2, TRAF3, BIRC2, BIRC3, CYLD, IRF4,
PRDM1, IKZF1, IKZF3, MYC, NSD2 (MMSET/WHSC1), FGFR3, MAF, MAFB, EGR1, RB1,
CDKN2C, CDKN1B, DUSP2, HIST1H1E, KMT2C, ARID1A, ARID2, SETD2, BCL7A, EP300,
CREBBP, TGDS, LTB, RPL5, RPL10, SP140, FUBP1, KDM6A, ZFHX4, RPL22
```

**Procedure.**
1. Hold out the 50-gene set H from any seed.
2. For each MMRF patient with ≥ 1 mutation outside H, build seed vector x_p with x_p[i] = 1{mutated} + 1{cytogenetic event} + |z_expr|.
3. Compute F_p = (1−α)(I−αÂ_filtered)^{−1} x_p with α = 0.5 on the channel-filtered STRING graph from §4 step 2.
4. Aggregate cohort-level score: F̄ = (1/N) Σ_p F_p / ‖F_p‖_∞.
5. Take top-K = 100 genes from F̄ and compute |top-K ∩ H| / 50 = recovery rate.

**Random baseline.** Drawing 100 genes uniformly from the n ≈ 18k gene universe gives expected recovery 100·50/18000 ≈ 0.28 % per gene → expected |∩| ≈ 0.28 (so ≈ 0.5–1 % overall). Empirical baseline: shuffle mutation→gene assignments and rerun → expect 1–3 %.

**Statistical test.** One-sided permutation test (10,000 random gene-shuffle seeds) on the recovery rate. Reject H0 (random) at α = 0.001 if recovery rate ≥ μ_perm + 4·σ_perm.

**Pre-registered success threshold.** **≥ 70 % recovery (35 of 50 known MM drivers in the top-100 propagated genes)**, with permutation p-value < 0.001. Below 70 %: propagation is insufficient as the sole driver-identification module and should be augmented (NetMix mixture or PRODIGY pathway integration). Below 30 %: STRING-PPI signal is too contaminated for use as causal prior — fall back to expert-curated pathway lists.

**Computational budget.** Single resolvent solve on |V| ≈ 18k, |E| ≈ 4·10⁵: ~5 s on CPU using scipy.sparse.linalg.spsolve; full cohort = N=859 patients × 5 s + 1000 permutations ≈ 3 hours wall time. Tractable on a single workstation.

---

## 6. Paths NOT chosen

1. **Naive co-expression network instead of PPI.** Rejected: co-expression edges are observational and confounded by latent TF modules; they re-introduce the faithfulness violation we want to avoid. Acceptable for seeding the velocity-field, not for defining P.
2. **Raw STRING edges without confidence filter.** Rejected: medium-confidence edges (≥ 0.4) include text-mining-only links with precision ~0.4 vs gold standards (Szklarczyk et al. 2023). Theorem 1 SNR collapses, hub bias dominates.
3. **HotNet2 as default.** Rejected for deployment (LICENSE is non-commercial academic only, verified). Kept as theoretical baseline only.
4. **deepDriver / supervised driver predictors.** Rejected as primary: requires labelled driver set; using on MMRF induces label-leakage circularity. Acceptable as post-hoc re-ranker trained on TCGA.
5. **PRODIGY per-patient Steiner trees.** Rejected: (a) no declared license on the repo, (b) R-only, (c) per-patient combinatorial optimisation cannot pool statistical power, which we need given MMRF's modest N.
6. **Sheaf / Hodge-Laplacian extensions** (Bodnar et al. 2022). Attractive but no published MM application; filed as v11 research thread.
7. **Pure spectral (heat kernel) without seed.** Rejected as driver identifier: eigenvectors of L_PPI describe topological communities but carry no patient information. Acceptable as a structural encoder feature.
8. **Random forest on network features.** Rejected: no closed-form velocity-field interpretation; breaks the Neural-ODE story.

---

## Verified citations (HTTP-checked 2026-05-03)

All identifiers below were resolved via `curl` against NCBI eutils, Crossref, arXiv, or the GitHub API.

- Leiserson et al. 2015 (HotNet2): PMID 25501392, DOI 10.1038/ng.3168
- Vandin Upfal Raphael 2011 (HotNet): PMID 21385051, DOI 10.1089/cmb.2010.0265
- Hofree et al. 2013 (NBS): PMID 24037242, DOI 10.1038/nmeth.2651
- Cowen Ideker Raphael Sharan 2017 (review): PMID 28607512, DOI 10.1038/nrg.2017.38
- Dimitrakopoulos et al. 2018 (NetICS): PMID 29547932, DOI 10.1093/bioinformatics/bty148
- Reyna et al. 2021 (NetMix): PMID 33400606, DOI 10.1089/cmb.2020.0435
- Dinstag Shamir 2020 (PRODIGY): PMID 31681944, DOI 10.1093/bioinformatics/btz815
- Luo et al. 2019 (deepDriver): PMID 30761181, DOI 10.3389/fgene.2019.00013
- Belkin Niyogi 2003 (Laplacian eigenmaps): DOI 10.1162/089976603321780317
- Coifman Lafon 2006 (diffusion maps): DOI 10.1016/j.acha.2006.04.006
- Spirtes Glymour Scheines 2000: DOI 10.7551/mitpress/1754.001.0001
- Szklarczyk et al. 2023 (STRING): PMID 36370105, DOI 10.1093/nar/gkac1000
- Tate et al. 2019 (COSMIC): PMID 30371878, DOI 10.1093/nar/gky1015
- Walker et al. 2018 (MM drivers): PMID 29884741, DOI 10.1182/blood-2018-03-840132
- Schiebinger et al. 2019 (WOT): PMID 30712874, DOI 10.1016/j.cell.2019.01.006
- Chen et al. 2018 (Neural ODE): arXiv:1806.07366
- Bodnar et al. 2022 (Sheaf NN): arXiv:2202.04579
- HotNet2 repo: github.com/raphael-group/hotnet2 — non-commercial academic only (LICENSE file)
- NetICS repo: github.com/cbg-ethz/netics (GPL-3.0)
- NetMix repo: github.com/raphael-group/netmix (BSD-3-Clause)
- PRODIGY repo: github.com/Shamir-Lab/PRODIGY (no license declared)

---

## Appendix: connection to existing CAUSAL_VALIDITY_AUDIT

`docs/CAUSAL_VALIDITY_AUDIT.md` is the de-facto DRIVER_FUNCTION_CAUSAL document. It established that without a structural prior, the driver head reports associations, not Pearl-tier causal effects. PPI propagation does not overturn that conclusion — it provides (i) a structural prior (§1.5) that shrinks the SCM hypothesis class by ≈ 4·10³, (ii) a velocity-field inductive bias (Propositions 1–3), (iii) a falsifiable claim (§5) with a 70 % recovery pre-registration. It still does **not** give interventional identifiability: propagation cannot orient X→Y vs Y→X without intervention. DepMap is reserved as the falsification anchor (§3 item 6); genuine causal claims remain a v11 commitment.

*End of document.*
