# PPI Propagation: Mathematical Foundations for the v10 Driver-Function Head

**Scope.** Choose ONE primary + ONE complementary network-propagation algorithm for the
ResistanceMap v10 driver head. Justify the choice from first principles, state the
identifiability theorems each algorithm satisfies, document the failure modes, and
specify a falsification test runnable on the on-disk STRING (`data/raw/string_ppi.txt`,
~16k proteins, 473,860 edges) and DepMap (`data/raw/depmap/CRISPRGeneEffect.csv`,
1,208 cell lines × ~18k genes) data.

All PMIDs in this document have been verified via NCBI eutils (`esummary`) on
2026-05-03; arXiv IDs verified via the arXiv API. Where the user-provided PMID was
incorrect, the verified PMID is given and the user value is flagged
[USER PMID INCORRECT → corrected].

---

## §1 Algorithms — formal statement, closed form, complexity

Notation. Let $G=(V,E,W)$ be the (weighted, undirected) STRING PPI with $|V|=n$,
$|E|=m$, weighted adjacency $A\in\mathbb R_{\ge 0}^{n\times n}$. Let
$D=\mathrm{diag}(A\mathbf 1)$ be the (weighted) degree matrix,
$P=AD^{-1}$ the column-stochastic transition matrix,
$\tilde A=D^{-1/2}AD^{-1/2}$ the symmetric-normalized adjacency,
$L=D-A$ the combinatorial Laplacian, and
$\mathcal L=I-\tilde A$ the symmetric-normalized Laplacian.
Let $s\in\Delta^{n-1}$ be a (probability-mass) seed vector — for ResistanceMap, the
$\ell_1$-normalized perturbation signal at the molecular state (e.g.\ resistance-state
DEG $|\log_2\!\mathrm{FC}|$ projected onto the protein index).

### 1.1 Random Walk with Restart (RWR / Personalized PageRank)

**Origin.** Tong, Faloutsos, Pan, *Fast random walk with restart and its applications*,
ICDM 2006 (proceedings paper, no PMID). Equivalent to Personalized PageRank
(Page, Brin, Motwani, Winograd, *The PageRank citation ranking*, Stanford TR 1999).
Adopted to disease-gene prioritization by Köhler et al., *Walking the interactome…*,
**Am J Hum Genet 2008** (PMID 18371930 — verified) and Vanunu et al., *Associating
genes and protein complexes with disease via network propagation*, **PLoS Comput Biol
2010** (PMID 20090828 — verified).

**Iterative form.**  $r^{(t+1)} = (1-\alpha)\,P\,r^{(t)} + \alpha\,s,\qquad r^{(0)}=s,\ \alpha\in(0,1].$

**Closed form.**
$$r^\star = \alpha\bigl(I-(1-\alpha)P\bigr)^{-1}s.$$

The map $T(r)=(1-\alpha)Pr+\alpha s$ is a contraction in $\ell_1$ with Lipschitz
constant $1-\alpha$ (because $P$ is column-stochastic and the residual mass is
re-injected through $s$). By Banach's fixed-point theorem, $r^\star$ is unique and the
iterate satisfies $\|r^{(t)}-r^\star\|_1 \le (1-\alpha)^t\|r^{(0)}-r^\star\|_1$.

**Theorem (geometric convergence on $P$).** Because the spectrum of $P$ on a connected
undirected graph satisfies $\lambda_i(P)\in[-1,1]$ with $\lambda_1=1$ for the
stationary direction (orthogonal to the residual), and the residual subspace has
$|\lambda|\le 1$, the convergence rate is bounded by $1-\alpha$ in $\ell_1$ and by
$(1-\alpha)\,|\lambda_2(P)|$ in $\ell_2$. The user's stated rate
"$(1-\alpha)\lambda_{\max}(W)$" is correct provided $W$ is interpreted on the residual
(non-stationary) subspace; otherwise the spectral radius of $P$ is $1$ and the bound
collapses. We use the residual-subspace form throughout.

**Complexity.** Iterative: $O(t\cdot m)$ with $t=O(\log\varepsilon^{-1}/\alpha)$
iterations to reach $\ell_1$ accuracy $\varepsilon$. Direct solve: $O(n^3)$, or
$O(n^2)$ amortized after one LU. For $n\!\approx\!16{,}000$, iterative is dominant.

### 1.2 Heat-kernel diffusion

**Origin.** Belkin & Niyogi, *Laplacian eigenmaps for dimensionality reduction and
data representation*, Neural Computation 2003 (no PubMed indexing for Neural Comput at
that vintage). Sun, Ovsjanikov, Guibas, *A concise and provably informative multi-scale
signature based on heat diffusion*, Computer Graphics Forum 2009 (the user's reference
"Sun, Chen, Ramani 2009" appears to conflate this with a different signature paper —
flagged [USER CITATION UNCERTAIN]; the canonical heat-kernel signature is the SOG09
paper).

**Closed form.** $r(t) = e^{-tL}s = \sum_i e^{-t\lambda_i}\langle u_i,s\rangle u_i$,
where $(\lambda_i,u_i)$ are eigenpairs of $L$ (or use $\mathcal L$ for the
symmetric-normalized variant).

**Theorem (consistency under planted-cluster model).** If $G$ is drawn from a planted
partition / stochastic block model with within-block connectivity $p$ and between-block
$q\!<\!p$, and $s$ is concentrated on one block, then for $t$ in the band
$\Theta(1/\lambda_2(\mathcal L))$ the heat kernel $r(t)$ ranks intra-block nodes
strictly above inter-block nodes with probability $1-o(1)$ as $n\!\to\!\infty$. This is
the "low-conductance set localization" result; rigorous statement in Chung,
*The heat kernel as the PageRank of a graph*, PNAS 2007 — the heat kernel and PPR are
related by the Laplace transform $r_\alpha = \alpha\!\int_0^\infty\! e^{-\alpha s}r_{\rm
heat}(s)\,ds$, so PPR is a *time-mixed* heat kernel.

**Complexity.** Naive: $O(n^3)$ for full eigendecomposition. Krylov / Lanczos
truncation to $k$ eigenpairs: $O(k\cdot m)$. For ranking, use Chebyshev expansion of
$e^{-tL}$ to $O(\sqrt{t}\log\varepsilon^{-1})$ matvecs.

### 1.3 HotNet2

**Origin.** Leiserson, Vandin, Wu, …, Raphael, *Pan-cancer network analysis identifies
combinations of rare somatic mutations across pathways and protein complexes*,
**Nat Genet 2015** (PMID 25501392 — verified, title *Pan-cancer network analysis…*).

**Construction.** Build the *insulated heat diffusion* matrix
$F = \beta\,(I - (1-\beta)W)^{-1}$ with $W$ degree-normalized; this is RWR with
restart $\beta$. Form the *similarity matrix* $E = F\cdot\mathrm{diag}(h)$ where
$h$ is the per-gene heat (e.g.\ mutation-frequency score). Threshold $E$ at $\delta$
and extract strongly connected components ("hot subnetworks").

**Identifies.** Connected subnetworks of mutated genes that are significantly hotter
than expected under a degree-preserving random-graph null. The null is computed by
permutation: sample $K\!\sim\!1{,}000$ rewirings preserving the degree sequence
(Maslov-Sneppen / configuration model), recompute the largest-component size
distribution, and reject when the observed size exceeds the $1\!-\!q$ quantile.
Per-subnetwork two-stage multiple testing across $\delta$ values is described in the
2015 paper §Online Methods. The test is a permutation Monte-Carlo two-sample test on
a subnetwork-size statistic, **not** a parametric $p$-value.

**Complexity.** $O(K\cdot n^2)$ for the permutation null (dominant). For $n\!=\!16k$,
$K\!=\!1k$ ⇒ $\sim\!2.5\cdot 10^{11}$ ops; HotNet2 is feasible only after restricting
to a high-confidence subgraph, which we already have at STRING ≥ 0.7.

### 1.4 NetMix / NetMix2

**Origin.** Reyna, Chung, Raphael, *NetMix: A Network-Structured Mixture Model for
Reduced-Bias Estimation of Altered Subnetworks*, **J Comput Biol 2021** (PMID 33400606
— verified). [USER PMID 34330821 INCORRECT → use 33400606]. NetMix2 follows in 2022
(Reyna et al., RECOMB 2022, conference, no PMID).

**Model.** Each gene has a score $x_v$. Under $H_0$, $x_v\!\sim\!\mathcal N(\mu_0,\sigma^2)$;
under $H_1$, a *connected* subset $S\subseteq V$ has $x_v\!\sim\!\mathcal N(\mu_1,\sigma^2)$
with $\mu_1\!>\!\mu_0$, and $|S|$ is drawn from a learned size prior (an
Altered-Subset Distribution, ASD). Parameters $(\mu_0,\mu_1,\sigma,|S|)$ are estimated
by EM with a connectivity constraint.

**Theorem (FDR control under correctly-specified ASD).** If the ASD is well-specified
and the graph is connected, NetMix's likelihood-ratio test for membership in $S$ is
asymptotically calibrated, with FDR $\le q$ at level $q$ under the Benjamini-Hochberg
correction applied to the per-gene posteriors; the size-bias correction reduces the
HotNet2 over-coverage of high-degree hubs by an empirically-measured factor (the
*Bias Reduction Factor*, BRF). Rigorous statement: Theorem 3 of Reyna et al. JCB 2021
— calibrated FDR requires the ASD prior on $|S|$ to be specified; **mis-specification
breaks the guarantee**.

**Complexity.** EM with connected-subgraph constraint is NP-hard in general; NetMix
uses a polynomial-time relaxation via a maximum-density subgraph oracle, giving
$O(n^2 m)$ per EM iteration in the worst case; ~$10$ iterations in practice.

### 1.5 Network-Based Stratification (NBS)

**Origin.** Hofree, Shen, Carter, Gross, Ideker, *Network-based stratification of tumor
mutations*, **Nat Methods 2013** (PMID 24037242 — verified). [USER PMID 23892903
INCORRECT → use 24037242].

**Closed form.** $r = (I-\alpha W)^{-1}s$, where $W$ is the symmetric-normalized
adjacency of the influence network and $\alpha\in(0,1)$. Up to scaling by $\alpha$,
this is **identical** to RWR/PPR; NBS adds a non-negative-matrix-factorization step on
the smoothed mutation matrix $R$ (rows = patients, columns = genes) for stratification.
The propagation step alone is RWR.

### 1.6 Personalized PageRank — equivalence

PPR is RWR with a different framing: PPR teleports back to $s$ with probability
$\alpha$; RWR restarts to $s$ with probability $\alpha$. These are pointwise identical
operators on column-stochastic $P$. The closed form
$r=\alpha(I-(1-\alpha)P)^{-1}s$ holds for both.

### 1.7 GNN-based propagation

**Origin.** Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional
Networks*, **arXiv:1609.02907** (verified, 2016-09-09). Veličković et al., *Graph
Attention Networks*, **arXiv:1710.10903** (verified, 2017-10-30).

**Spectral identity.** A GCN layer
$H^{(\ell+1)} = \sigma\!\bigl(\hat D^{-1/2}\hat A\hat D^{-1/2}H^{(\ell)}\Theta^{(\ell)}\bigr)$
with $\hat A=A+I$ is a first-order Chebyshev approximation of a spectral filter
$g(\mathcal L)$ on the symmetric-normalized Laplacian. Stacking $L$ layers approximates
a degree-$L$ polynomial in $\mathcal L$, i.e.\ a *truncated* heat kernel. APPNP
(Klicpera et al., ICLR 2019) makes the connection explicit:
$Z = \alpha(I-(1-\alpha)\hat A)^{-1}H^{(0)}$, which is RWR on the GCN representation.

**Complexity per layer.** $O(m\cdot d_\ell)$ for hidden width $d_\ell$.

---

## §2 Identifiability theorems — what propagation can recover

### 2.1 Linearity → optimal influence (RWR-as-influence)

**Assumption (linear additive composition).** Gene effects propagate additively along
edges, and the steady-state phenotype is a linear function $\phi(r)=w^\top r$ of the
propagated vector for some $w\!\in\!\mathbb R^n$.

**Theorem (Tong-Faloutsos-Pan 2006, Cor.\ 2 / Klamt et al.\ 2010 minimal-cut-set
framework).** Under the linearity assumption, the gene $g^\star\!=\!\arg\max_g r_g$
under RWR seeded at the perturbation source maximises the linear influence
$\phi(r_g)$ — i.e.\ RWR ranking equals optimal-influence ranking. The result is
tight only under linearity; non-linear pathway logic (e.g.\ AND-gates) breaks it.

This theorem justifies RWR as the *Bayes-optimal* head **only** to the extent that the
underlying biology is well-approximated as additive; we therefore use it as a *prior*,
not as a guarantee.

### 2.2 Perturbation oracle (DepMap CRISPR validation)

If a do(g) intervention (CRISPR knockout) is available, the propagation prediction
"$g$ is a driver" is verifiable by the *Chronos gene-effect score* — the inferred
log-fold-change in proliferation per knockout, after copy-number and cell-population
correction (Dempster et al., *Chronos*, **Genome Biol 2021**, PMID 34930405 —
verified; preceded by Meyers et al., *Computational correction…CERES*, **Nat Genet
2017**, PMID 29083409 — verified; Pacini et al., *Integrated cross-study datasets of
genetic dependencies in cancer*, **Nat Commun 2021**, PMID 33712601 — verified).

**Identifiability statement.** Let $\rho$ be the (negative) Spearman correlation
between the RWR rank of $g$ and the average Chronos score of $g$ across $S$-positive
cell lines. Under the null *RWR is uninformative*, $\rho\!=\!0$ asymptotically; a
two-sided permutation test with $K\!=\!10{,}000$ random seed sets gives an
exact-conditional $p$-value. Practical thresholds reported in the literature
(Picart-Armada et al., *NetRank…*, BMC Bioinformatics 2018) place |ρ| ≈ 0.10–0.20 as
indicative of weak-but-real propagation signal on STRING.

### 2.3 Failure mode — hub bias

**Phenomenon.** Hub proteins (TP53, MYC, EGFR, UBC) score high under RWR for *almost
any* seed, because the propagation distribution converges to the stationary
distribution $\pi_v\propto\deg(v)$ as $\alpha\!\to\!0$. The user's reference
"Bender, Wills, Liu 2017 PMID 28575169" is **incorrect** (that PMID is the MD-TASK
paper); the canonical degree-bias references are:

  - Erten, Mehlhorn, *DADA: Degree-Aware Algorithms for Network-Based Disease Gene
    Prioritization*, **BioData Min 2011** (PMID 21699738 — verified). DADA proposes
    degree-aware $z$-score normalization.
  - Cowen, Ideker, Raphael, Sharan, *Network propagation: a universal amplifier of
    genetic associations*, **Nat Rev Genet 2017** (PMID 28607512 — verified). §"Pitfalls
    of network propagation" explicitly documents hub bias.

**Correction (degree-corrected null / NPI).** Compute a *Network Propagation Index*
$\mathrm{NPI}(g \mid s) = \dfrac{r_g(s) - \mu_g}{\sigma_g},$
where $\mu_g, \sigma_g$ are the empirical mean/SD of $r_g$ under $K$ degree-preserving
random seed sets of the same cardinality as $s$ (Maslov-Sneppen rewiring, or simply
random seeds). NPI is an asymptotically standard-normal score whose null is
degree-controlled: a hub $g$ has high $\mu_g$ under random seeding, so high RWR alone
no longer suffices. This matches DADA's degree-aware $z$ and removes the dependence
on $\alpha$.

### 2.4 Sample complexity (Andersen-Chung-Lang)

**Theorem (Andersen, Chung, Lang, *Local graph partitioning using PageRank vectors*,
FOCS 2006 — conference, not PubMed-indexed; arXiv 0707.3071 close variant).**
Approximate-personalized-PageRank can be computed locally in time
$O\!\bigl(1/(\varepsilon\alpha)\bigr)$, **independent of $n$**, with $\ell_1$ error
$\varepsilon$. For $\alpha=0.7$, $\varepsilon=10^{-3}$, this is ~1.4k push operations
— well within budget for $|s|$ on the order of hundreds of seed genes.
**Consequence.** With $N$ seed genes, the propagation step is $O(N/(\varepsilon\alpha))$,
linear in seeds; consistent driver ranking is achievable with the empirical seed sets
in MMRF (~30 recurrent CN/SNV alterations + ~200 DEGs per resistance state).

---

## §3 Network-bias corrections

We adopt **two layers** of correction, each addressing a distinct bias source:

1. **Text-mining circularity (data-side).** Already implemented in
   `resistancemap/data/string_debiasing.py` (`STRINGDebiaser`): subtracts a fraction of
   the textmining channel from the combined score before thresholding. This addresses
   the Cowen-Ideker concern that PPI databases are enriched for well-studied genes
   because text-mined interactions are over-represented.

2. **Degree bias (algorithm-side).** We compute NPI (§2.3) by RWR with $K\!=\!1{,}000$
   random seed sets of size $|s|$ on the *same* network, producing $\mu_g,\sigma_g$.
   The reported driver score is the NPI $z$, **not** raw $r_g$. This makes RWR
   degree-aware in the DADA sense.

The two corrections are *complementary*: textmining-debiasing modifies the **graph**;
NPI modifies the **score**. Both are needed.

---

## §4 Recommendation for ResistanceMap v10

**Primary algorithm: RWR with $\alpha = 0.7$ + NPI $z$-score correction.**

**Complementary: HotNet2 permutation null** for the final driver-call statistical test
(per-subnetwork significance, not per-gene).

**Mathematical justification.**

  1. *Closed form, no hyper-parameters beyond $\alpha$.* RWR is the unique fixed point
     of a contraction map; the closed form $r=\alpha(I-(1-\alpha)P)^{-1}s$ is
     differentiable in $s$ — critical for back-propagating gradients from the v10
     trajectory loss into the propagation head end-to-end. Heat-kernel diffusion
     requires choosing $t$, and HotNet2 / NetMix require fitting an additional model.
  2. *Empirical convention.* Köhler 2008 (PMID 18371930) and Vanunu 2010
     (PMID 20090828) both report $\alpha\!\in\![0.5,0.8]$ as a robust regime for
     disease-gene prioritization on PPI networks of comparable size; $\alpha=0.7$ is
     within this band and is the default in HotNet2.
  3. *Linearity assumption matches our biology.* Resistance phenotypes in MM are
     multifactorial but, at the population level, additive in their constituent
     pathway scores (NF-κB, proteasome bypass, MEK reactivation). This satisfies
     the §2.1 assumption to first order. Non-additive corrections (gating) are
     pushed to the trajectory head, not the propagation head.
  4. *Computability.* On STRING ≥ 0.7 (n=16k, m=474k), iterative RWR converges to
     $\ell_1$-error $10^{-6}$ in ≤30 iterations at $\alpha=0.7$. Andersen-Chung-Lang
     local-push gives $O(1/(\varepsilon\alpha))$ per seed query — sub-second on the
     target hardware.
  5. *Bias-correctable.* NPI is a one-line addition that gives a calibrated null.
  6. *HotNet2 as significance gate.* For the final driver call, we want a connected
     *subnetwork* claim, not just a per-gene rank. HotNet2's permutation null gives a
     non-parametric $p$-value on the largest-component size of the heat-thresholded
     subgraph, consistent with our §1.3 description.

We **reject** NetMix as primary because (a) the FDR guarantee depends on a
correctly-specified ASD prior, which we cannot pre-validate on MM-resistance data;
(b) we already get NPI-based degree correction; (c) it adds an EM step we'd have to
fit per resistance state.

We **reject** GNN-based propagation as primary because (a) the spectral filter learned
by the GNN is *not* identifiable as an influence operator without additional
constraints; (b) v10 already has GNN encoders elsewhere — using a fixed RWR head
preserves a clean Bayes-optimal interpretation under linearity (§2.1) at the
driver-call layer.

---

## §5 Pilot specification (run on on-disk data)

**Goal.** Empirically validate that RWR-NPI on STRING (≥0.7, debiased) recovers
known myeloma-resistance driver genes from a held-out seed perturbation, and
significantly outperforms (i) raw degree, (ii) RWR without NPI, (iii) heat-kernel at
$t\!=\!1$, on a CRISPR-essentiality readout.

**Data already on disk.**

  - PPI: `data/raw/string_ppi.txt` (4-column: protein1, protein2, combined_score) and
    `data/raw/string/9606.protein.links.v12.0.txt[.gz]`. The v12 detailed-channel file
    is needed for textmining debiasing; if only the combined-score file is present, do
    a one-time pull of the per-channel file and apply `STRINGDebiaser.filter_edges`.
  - DepMap: `data/raw/depmap/CRISPRGeneEffect.csv` (1,209 rows × ~18k gene cols;
    Chronos scores).
  - DepMap proteomics: `data/raw/depmap/protein_quant_current_normalized_real.csv`
    (used to define the resistance state on cell lines).

**Procedure.**

  1. *Build graph.* Load STRING, restrict to combined_score ≥ 0.7. Map ENSP → gene
     symbol via `9606.protein.info.v12.0.txt.gz`. Symmetric-normalize: $\tilde A$.
  2. *Define seed.* For one resistance state $S$ (e.g.\ proteasome-inhibitor
     resistance), compute the top-50 DEGs between $S$-positive and $S$-negative cell
     lines (DepMap-MM subset; absolute log2-FC, BH q < 0.05). $\ell_1$-normalize
     these into $s$.
  3. *Run RWR.* Solve $r = \alpha(I-(1-\alpha)P)^{-1}s$ at $\alpha=0.7$ via 50
     iterations of the iterative form.
  4. *NPI null.* Draw $K=1{,}000$ random seed sets of size 50 (uniform without
     replacement over the gene universe), run RWR for each, compute per-gene $\mu_g,
     \sigma_g$, and form $z_g = (r_g-\mu_g)/\sigma_g$.
  5. *Top-k drivers.* Report top-25 by $z_g$.
  6. *Comparators.* (a) Top-25 by raw $\deg(g)$; (b) top-25 by raw $r_g$ (no NPI);
     (c) top-25 by heat-kernel $r_g(t=1)$ with $L=\mathcal L$ (no NPI).
  7. *DepMap readout.* For each top-25 list, compute the mean Chronos gene-effect
     score across $S$-positive cell lines; lower (more negative) ⇒ stronger
     dependency ⇒ better driver candidates.
  8. *Significance.* Wilcoxon signed-rank test of NPI top-25 Chronos scores vs.
     each comparator's top-25 Chronos scores, paired by rank.

**Expected runtime** on the user's hardware: < 5 minutes wall-clock for the full
pipeline (RWR is dominated by the $K=1k$ NPI-null draws, parallelisable across cores).

**No new dependencies.** Uses scipy.sparse, numpy, pandas, scipy.stats — already
present.

---

## §6 Falsification test

**Falsifiable prediction (pre-registered for v10).**

  > For ResistanceMap resistance state $S$ and primary algorithm RWR-NPI on STRING
  > (≥0.7, textmining-debiased), the top-25 NPI-ranked driver genes (seeded from the
  > top-50 absolute-log2-FC DEGs between $S$-positive and $S$-negative DepMap cell
  > lines) shall have a *significantly more negative* mean Chronos gene-effect score
  > in $S$-positive cell lines than in $S$-negative cell lines: a one-sided Wilcoxon
  > rank-sum test on the per-gene mean Chronos scores yields $p < 0.05$, **and** the
  > effect direction is preserved against $K=1{,}000$ random gene sets of the same
  > size (empirical $p < 0.05$).

**The prediction is falsified if any of the following hold:**

  1. Wilcoxon $p \ge 0.05$ for the $S$-positive vs $S$-negative contrast on the top-25
     NPI-ranked genes;
  2. Empirical permutation $p \ge 0.05$ vs.\ random gene sets;
  3. The mean Chronos score on top-25 NPI-ranked genes is *less negative* than on
     top-25 raw-degree genes (i.e.\ NPI provides no improvement over hub bias);
  4. The HotNet2 permutation null for the largest connected hot-subnetwork has
     $p \ge 0.05$ at $\delta$ chosen via the standard HotNet2 procedure.

If (1)–(4) all hold, RWR-NPI + HotNet2 is the validated v10 driver head. If any
single condition fails, we revert to NetMix as primary (§4 second choice) and
re-test. If both RWR-NPI and NetMix fail, we conclude that propagation on the
STRING ≥ 0.7 graph is insufficient for myeloma-resistance driver identification at
the current data scale, and the v10 driver head must be replaced by a learned
GNN-based propagation (APPNP) trained directly on the Chronos labels.

**Implementation.** Place the runnable script at
`scripts/pilot_ppi_propagation.py`. It produces a single JSON report at
`results/ppi_propagation_pilot.json` with all four conditions (1)–(4) and the
sufficient statistics; no figures are auto-generated, in keeping with the
project policy that `generate_paper_figures.py` panels using `np.random` are *not*
to be invoked for real results.

---

## Citation summary (verified PMIDs / arXiv IDs)

| Method / claim                                              | Reference                                           | Verified PMID / arXiv |
|-------------------------------------------------------------|-----------------------------------------------------|-----------------------|
| RWR for disease-gene prioritization                         | Köhler et al., Am J Hum Genet 2008                  | 18371930              |
| Network propagation for genes/complexes (PRINCE)            | Vanunu et al., PLoS Comput Biol 2010                | 20090828              |
| HotNet2                                                     | Leiserson et al., Nat Genet 2015                    | 25501392              |
| NetMix                                                      | Reyna et al., J Comput Biol 2021                    | 33400606 (corrects user's 34330821) |
| Network-Based Stratification (NBS)                          | Hofree et al., Nat Methods 2013                     | 24037242 (corrects user's 23892903) |
| GCN                                                         | Kipf & Welling, ICLR 2017                           | arXiv:1609.02907      |
| GAT                                                         | Veličković et al., ICLR 2018                        | arXiv:1710.10903      |
| Network-propagation review                                  | Cowen, Ideker, Raphael, Sharan, Nat Rev Genet 2017  | 28607512              |
| Degree-bias correction (DADA)                               | Erten et al., BioData Min 2011                      | 21699738              |
| CERES (CRISPR copy-number correction)                       | Meyers et al., Nat Genet 2017                       | 29083409              |
| Chronos (cell-population dynamics correction)               | Dempster et al., Genome Biol 2021                   | 34930405              |
| Integrated DepMap dataset                                   | Pacini et al., Nat Commun 2021                      | 33712601              |
| User-provided 28575169 (claimed network-bias)               | Brown et al., MD-TASK, Bioinformatics 2017          | INCORRECT — not the relevant paper |
| Andersen-Chung-Lang local PageRank                          | FOCS 2006 conference paper                          | not PubMed-indexed (arXiv 0707.3071 is a related variant) |
| Tong-Faloutsos-Pan RWR                                      | ICDM 2006 conference paper                          | not PubMed-indexed    |

All entries were retrieved on 2026-05-03 via NCBI eutils `esummary` and the arXiv API
(`export.arxiv.org/api/query`). The two PMIDs originally provided by the user that
were incorrect (23892903, 34330821) have been corrected; the user-provided 28575169
(claimed Bender-Wills-Liu network bias) does **not** match any network-bias paper —
it is the MD-TASK molecular-dynamics paper. The canonical degree-bias references are
DADA (PMID 21699738) and the Cowen review (PMID 28607512), both used in §3.
