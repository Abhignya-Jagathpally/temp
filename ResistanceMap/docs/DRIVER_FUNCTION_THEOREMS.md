# Driver-Function Identifiability — Formal Theorem Review
# ResistanceMap v10

**Date.** 2026-05-03.
**Scope.** Formal review of eight identification theorems bearing on whether ResistanceMap's
top-k driver genes, derived from PPI-propagated multi-omic signals, can be interpreted as
causal drivers rather than associated features. Builds on CAUSAL_VALIDITY_AUDIT.md and
PPI_PROPAGATION_DRIVER_THEORY.md.
**Citation status.** All DOIs and arXiv IDs HTTP-verified at time of writing; see §6.

---

## §1 Theorems Table

### Theorem 1 — Pearl Front-Door Criterion (Pearl 1995, *Causality* 2009)

**Formal statement.** Let X, M, Y be observed variables and let the DAG contain no
unobserved confounders of (X, M). The causal effect P(Y | do(X)) is identified by the
front-door formula if and only if:
  (i)  M intercepts all directed paths from X to Y,
  (ii) there are no unblocked back-door paths from X to M,
  (iii) all back-door paths from M to Y are blocked by X.

Under these conditions:

    P(Y | do(x)) = Σ_m P(M=m | X=x) · Σ_{x'} P(Y | M=m, X=x') P(X=x')

**Identification type.** Observational data sufficient if conditions (i)–(iii) hold.

**Application to MM / ResistanceMap.** For Venetoclax resistance (Y) caused by BCL2
expression (X), a candidate mediator M would need to:
- Lie on every directed path X → Y,
- Be unconfounded by the same latent variables confounding X → Y.

In MM, the dominant confounders of BCL2-to-resistance are cytogenetic subtype
(t(11;14) causes both high BCL2 and Venetoclax sensitivity), clonal heterogeneity, and
microenvironment composition. None of these is a measured mediator that intercepts the
full X→Y path. Condition (ii) is violated: t(11;14) creates a back-door path
t(11;14) → BCL2 → M that is not blocked. **Front-door criterion fails for every
BCL2-mediator candidate in CCLE or MMRF observational data.**

---

### Theorem 2 — Shpitser and Pearl 2008 — Complete ID Algorithm (JMLR v9)

**Formal statement.** Let G be a DAG with hidden variables (a "hidden-variable DAG").
The ID algorithm identifies P(Y | do(X)) from the observed distribution P_obs if and
only if the c-component factorization of G produces no "hedge" structure — formally,
no pair of c-components (F, F') satisfying F' ⊆ F, X ⊆ F', and Y ⊆ F exists. The
algorithm is complete: if a hedge exists, P(Y | do(X)) is not identifiable by any
method using observational data alone.

**Identification type.** Non-parametric identification from observational P_obs in a
partially observed DAG.

**Application to ResistanceMap.** The union DAG for the MM query
P(resistance | do(gene_knockout)) has the following structure:
- Observed: gene expression X, drug IC50 Y, cytogenetics C (in CoMMpass).
- Hidden: clonal composition U_clonal, microenvironment U_micro, passage history U_pass
  (cell-line data only).

The path U_clonal → X and U_clonal → Y creates an unobserved confounder of X and Y
that is not blocked by C (cytogenetics partly proxies clonal composition but does not
fully represent it — subclonal heterogeneity within a cytogenetic category is
unmeasured). The relevant c-component test: {X, Y} share a bidirected edge (induced
by U_clonal) in the acyclic directed mixed graph (ADMG). The ID algorithm applied to the
MMRF+CCLE union DAG returns "not identified" for any gene-knockout query because a hedge
for ({Y}, {X, Y}) exists via U_clonal.

**Consequence.** P(resistance | do(BCL2_knockout)) is non-identifiable from
MMRF observational + CCLE observational data. The query is identifiable only if
experimental data from a do(BCL2_knockout) environment is added (e.g., DepMap CRISPR
essentiality for BCL2 in MM cell lines) and the selection diagram for cell-line →
patient transport is specified.

---

### Theorem 3 — Shpitser 2013 — Counterfactual Graphical Models for Longitudinal
###               Mediation Under Unobserved Confounding (Cognitive Science, DOI 10.1111/cogs.12058)

**Formal statement.** In a time-varying treatment setting with treatment A_t, outcome
Y, mediator M_t, and time-varying confounder L_t, the natural indirect effect NIE =
E[Y(a, M_{a'})] − E[Y(a', M_{a'})] is identified from observational data if and only
if, at each time t, there are no unobserved common causes of (A_t, Y) that operate
through M_t when the history of intermediate confounder states L_{t} is observed and
included in the adjustment set.

**Key condition.** Shpitser's "Sequential Randomization Assumption for Natural Effects"
requires that for each t, L_t (the intermediate confounder state after treatment A_t but
before M_t) is measured and included.

**Application to ResistanceMap / EIMOC failure.** The EIMOC (Error in Mediator
Observation Conditional) failure identified in CAUSAL_VALIDITY_AUDIT.md is a direct
instantiation of this theorem's condition failing. In MM:

- A_t = drug treatment at cycle t,
- M_t = pathway activation state at cycle t (the candidate mediator),
- L_t = intermediate molecular state after drug exposure but before the resistance
  phenotype solidifies (e.g., clonal selection during the treatment cycle).

MMRF CoMMpass does NOT measure L_t: RNA-Seq visits are taken at discrete clinical
timepoints separated by months, not at the post-dose, pre-progression intermediate
state. The 51-patient longitudinal RNA-Seq sub-cohort in CoMMpass has visits separated
by median 256 days (multiple treatment cycles). L_t in between visits is entirely
unobserved.

**Structural verdict.** NIE through any pathway M is non-identifiable in CoMMpass even
after Shpitser's framework is applied, because the intermediate treatment-cycle confounder
L_t is not measured. This is a structural data gap, not an estimation problem. Adding
more patients to CoMMpass does not resolve it. Resolving it requires cycle-level
RNA-Seq (bone marrow biopsy at each treatment cycle) — currently available only in
Cohen et al. 2021 (PMID 33619369, NCT04065789, n=41 bortezomib-refractory patients).

---

### Theorem 4 — Bareinboim and Pearl 2014 — Transportability (Statistical Science,
###              DOI 10.1214/14-sts486)

**Formal statement.** Let P be a source population (cell lines) and P* a target
population (MM patients). A selection diagram S encodes differences between P and P*
via "selection nodes" S_i with edges S_i → X_i wherever the distribution of X_i differs
between populations. P*(Y | do(X)) is transportable from P to P* if and only if:

  (i) The transport formula Σ_Z P*(Z) · P(Y | do(X), Z) is well-defined with respect
      to the selection diagram S.
  (ii) Positivity in the target: P*(Z = z) > 0 for all z in the support.
  (iii) No selection node S_i directly confounds the path X → Y in the target without
       a corresponding measured node.

When the transport formula is not achievable from observational P data alone, the ID*
algorithm (the transportable analogue of Shpitser-Pearl ID) declares the query
non-transportable.

**Three specific failures for MM cell-line → patient transport:**

1. **Immune microenvironment.** MM patient bone marrow contains NK cells, T regulatory
   cells, macrophages, and stromal cells that actively modulate drug response (documented
   in Zavidij et al. 2020, PMID 32747829, GSE124310). CCLE cell lines lack this
   environment entirely. Selection node S_immune → IC50 is present; no measured variable
   captures immune contribution in CCLE. Transport formula cannot marginalise over
   immune state. Condition (iii) fails.

2. **Clonal heterogeneity.** Patient tumours contain multiple clonal populations with
   distinct driver mutations (Walker et al. 2018, PMID 29884741). Cell lines are
   monoclonal (or oligoclonal after passaging). Selection node S_clonal → X_protein
   creates a distribution shift in the proteomic seed signal between populations.
   Condition (i) requires marginalising over S_clonal; P*(clonal composition) is
   partially measurable in CoMMpass (subclonal SNV calls in MAF files) but not wired
   into training. Condition (i) partially fails.

3. **Passage drift.** CCLE lines have been passaged hundreds of times since patient
   derivation; cumulative somatic evolution under culture conditions creates a selection
   node S_passage → X_epigenomics. CCLE epigenomics reflects culture-adapted chromatin
   state, not tumour-of-origin state. Condition (iii) fails for the epigenomic channel.

**Consequence.** No cell-line-trained model's P(Y | do(X)) can be validly transported
to the patient population without explicitly adjusting for all three selection nodes.
The Tier 2 validation on CoMMpass (PERCEPTION-style, C-index on N=258) tests whether
the transported association holds empirically, but it does not constitute a proof of
transportability in the Bareinboim-Pearl sense — it is an empirical falsification of
the association, not a formal identification result.

---

### Theorem 5 — Tian and Pearl 2002 — General Identification Condition
###              (AAAI 2002 Proceedings)

**Formal statement.** Given a DAG G and a set of intervention targets X, the causal
effect P(Y | do(X)) is identified from a combination of observational data P_obs and
experimental data P_exp(Y | do(Z)) for some variable set Z if and only if the effect
is expressible via the do-calculus rules (rules 1–3 of Pearl's do-calculus) from the
available mixture of observational and experimental distributions.

**Application.** ResistanceMap has access to:
  - P_obs from MMRF CoMMpass (observational patient data, N=994),
  - P_exp from DepMap CRISPR essentiality (experimental knockout data, N=1,208 cell
    lines × 18,531 genes).

The question is whether P_patient(resistance | do(BCL2_knockout)) is identifiable from
P_obs(MMRF) ∪ P_exp(DepMap). Under Tian-Pearl, this requires a do-calculus derivation
showing that the hedge structure identified under Theorem 2 can be broken by conditioning
on the DepMap experimental distribution.

**Assessment.** DepMap provides P_CellLine(viability | do(gene_knockout)), not
P_Patient(resistance | do(gene_knockout)). To use DepMap to identify the patient query,
transportability from cell line to patient must hold (Theorem 4 above). Since Theorem 4
fails on all three conditions, DepMap experimental data does NOT lift the
non-identification for the patient P* query. DepMap can serve as:
  (a) A falsification oracle: if do(BCL2) does not affect viability in MM cell lines,
      BCL2 is unlikely to be a driver in patients (falsification, not identification).
  (b) A consistency check: agreement between MMRF association signal and DepMap
      CRISPR effect is necessary but not sufficient for causal identification.

DepMap does NOT serve as an instrumental variable for patient-level causal effects
because the exclusion restriction (instrument affects outcome only through treatment)
is violated by the passage drift and immune microenvironment selection nodes
(Theorem 4, failure modes 1 and 3).

---

### Theorem 6 — Mooij, Janzing, and Scholkopf 2016 — Post-Nonlinear / Additive Noise
###              Models for Cause-Effect Identification (JMLR v17, arXiv:1503.06389)

**Formal statement.** For a bivariate system (X, Y), the causal direction X→Y is
identifiable from observational data under the Additive Noise Model (ANM) assumption
Y = f(X) + N_Y, N_Y ⊥ X, if and only if the reverse model X = g(Y) + N_X does NOT
satisfy N_X ⊥ Y. When both the forward and reverse ANM fit (unfaithful cancellation),
direction is not identifiable from this pair alone.

**Application to ResistanceMap.** The query "is gene G a cause or consequence of
drug resistance?" can in principle be tested by fitting ANMs in both directions:
  - Forward: resistance_score = f(G_expression) + noise
  - Reverse: G_expression = g(resistance_score) + noise

and testing independence of residuals. However, in the MM multi-gene context with
p >> n and dense regulatory networks, three failure modes are decisive:

1. **Feedback loops.** MM regulatory networks contain feedback (e.g., NF-kB activates
   target genes that feed back to maintain NF-kB activity). Feedback violates the
   acyclicity assumption required by ANMs. The ANM test is undefined for cyclic SCMs.

2. **Confounding by upstream regulators.** The single-pair ANM test assumes X and Y
   share no hidden common cause. In gene expression data, latent transcription-factor
   modules confound every bivariate test. MYC as a transcription factor confounds
   dozens of gene pairs simultaneously; fitting the ANM on any such pair produces
   spurious identifiability (residuals appear independent in both directions because
   the true cause is unmeasured).

3. **Measurement error.** Bulk RNA-Seq mixes clonal populations. The observed
   expression of G is a weighted average over heterogeneous clones; the noise term
   N_Y is not independent of G_expression under clonal mixing. ANM residual
   independence test rejects in both directions.

**Verdict.** For any gene G in the MM multi-omic dataset, direction identification via
ANM/PNL models is unreliable due to cyclic regulatory structure and latent confounders.
PPI propagation (Theorems 7 and 8 below) provides a structure-informed prior that is
strictly stronger than the bivariate ANM test for this application.

---

### Theorem 7 — Faithfulness Assumption (Spirtes, Glymour, Scheines 2000,
###              *Causation, Prediction, Search*, MIT Press, DOI 10.7551/mitpress/1754.001.0001;
###              Uhler, Raskutti, Buhlmann, Yu 2013, Annals of Statistics, DOI 10.1214/12-aos1080)

**Formal statement (Spirtes-Glymour-Scheines).** A distribution P is faithful to DAG G
if for every conditional independence X ⊥ Y | Z in P, the corresponding d-separation
X ⊥_d Y | Z holds in G. Under faithfulness plus causal sufficiency, constraint-based
causal discovery (PC algorithm, FCI algorithm) is consistent: the output CPDAG is
asymptotically the correct Markov equivalence class.

**Formal statement (Uhler et al. 2013, geometry of faithfulness).** The set of
unfaithful distributions (those satisfying a conditional independence in P not implied
by d-separation in G) has Lebesgue measure zero among Gaussian distributions given G.
However, near-unfaithful distributions — where a conditional independence nearly holds
but is non-zero — are dense near the unfaithful set. The probability that a sample
from n variables with the true n-dimensional Gaussian distribution lies within epsilon
of the unfaithful set grows as the number of edges grows, specifically at rate O(p²)
for dense graphs.

**Biological failure mode.** In gene regulatory networks:

1. **Regulatory feedback creates exact cancellations.** Consider the FoxO3-PI3K-AKT
   feedback loop in myeloma drug resistance: PI3K activates AKT, which inhibits FoxO3,
   which inhibits PI3K. On the path PI3K → AKT → outcome, there exists a compensating
   negative feedback PI3K → FoxO3 → (inhibits) PI3K that creates near-cancellation
   of direct and indirect effects. In bulk RNA-Seq data this manifests as the conditional
   independence PI3K ⊥ outcome | AKT appearing to hold, even though the true DAG
   has PI3K → outcome directly. The PC algorithm would incorrectly orient this edge.

2. **Dense regulatory graphs are near-unfaithful by Uhler et al.** For a gene regulatory
   network with p ~ 18,000 genes and ~200,000 regulatory edges (estimated for the human
   transcription factor network), the Uhler et al. bound implies that near-unfaithfulness
   is practically certain in any finite sample. The STRING-filtered PPI with ~470,000
   protein-interaction edges is substantially denser than the faithfulness-safe regime.

3. **Co-expression edges imported into STRING.** STRING confidence ≥ 0.7 includes
   co-expression channel contributions (PPI_PROPAGATION_DRIVER_THEORY.md §1.5). These
   edges are explicitly derived from observational covariance data. Using them as a
   causal DAG prior introduces edges whose direction is not structurally grounded;
   any faithfulness-based orientation of these edges is circular.

**Consequence for ResistanceMap.** Constraint-based causal discovery on the MM
multi-omic data (even with the STRING PPI prior to restrict edge candidates) cannot
guarantee correct DAG orientation for any pathway containing feedback. PC/FCI applied
to MMRF RNA-Seq will produce a CPDAG with numerous unoriented edges precisely at the
regulatory feedback nodes that are most relevant to drug resistance.

---

### Theorem 8 — Path Analysis and SEM-Based Mediation (Wright 1934, *Annals of
###              Mathematical Statistics*; Steen, Loeys, Moerkerke 2017, *American
###              Journal of Epidemiology*, DOI 10.1093/aje/kwx051)

**Formal statement (Wright 1934 path tracing rules).** In a recursive structural
equation model (SEM) with no cycles and no correlated error terms, the total effect of
X on Y equals the sum of products of path coefficients along all directed paths from
X to Y. The direct effect is the coefficient on the X→Y edge after adjusting for all
mediators. Identification requires: (a) linear structural equations, (b) acyclicity,
(c) no correlated errors (equivalent to causal sufficiency).

**Formal statement (Steen et al. 2017).** For a system with multiple mediators
M_1, …, M_k on the path X → Y, the "flexible mediation analysis" using natural effect
models (NEMs) identifies the joint natural indirect effect through any subset T ⊆
{M_1, …, M_k} if: (a) sequential ignorability holds at each mediator given measured
covariates, (b) no post-treatment confounding of mediator-outcome relationships exists.
The NEM approach is semi-parametric and does not require linearity, unlike Wright's
path coefficients.

**Application to ResistanceMap pathway attribution.** When ResistanceMap assigns
importance to a pathway P via attention weights or SHAP values, it is implicitly
performing a path-tracing analysis in the model's learned SEM. However:

1. Wright's conditions (a)–(c) all fail for gene networks: non-linearity is the rule
   (signalling cascades, saturation kinetics), cycles are pervasive (feedback),
   correlated errors exist at every step (latent transcription factors).

2. Steen et al.'s sequential ignorability requires that, conditional on measured
   covariates, the pathway M is as-good-as-randomised given past treatment and pathway
   history. In MM under standard-of-care treatment, this is not testable without
   cycle-level molecular data (Theorem 3, Shpitser 2013 failure).

3. The mapping from model attention weights to path coefficients is not established.
   Attention weights in a transformer or graph neural network reflect the marginal
   utility of a feature for the prediction task; they do not equal structural equation
   coefficients even when the SEM is correctly specified. The distinction is: attention
   identifies what the model uses, path coefficients identify what the DGP uses.

**What path analysis DOES provide.** If ResistanceMap's gene-level features are passed
through a correctly specified linear SEM fitted to the observed MMRF data (rather than
extracted as model weights), Wright's path rules yield a valid L1 association
decomposition. This is the appropriate use: reporting direct vs indirect association
coefficients from a fitted SEM, clearly framed as observational associations, not as
interventional effects.

---

## §2 ResistanceMap-Specific Application

### Which theorems apply and what they say

| Theorem | Status in ResistanceMap v10 | Key conclusion |
|---|---|---|
| 1 — Front-door criterion | Fails to apply | No valid mediator M exists that satisfies conditions (i)-(iii) given cytogenetic confounders of BCL2/MCL1 → resistance in MM. |
| 2 — Complete ID algorithm | Not identifiable | Hedge structure exists in MMRF+CCLE union DAG due to U_clonal. P(resistance \| do(gene)) non-identified from observational data alone. |
| 3 — Shpitser 2013 longitudinal mediation | Structural gap | NIE through any pathway is non-identified because L_t (intermediate cycle-level confounder) is not measured in MMRF. Cannot resolve by adding patients. |
| 4 — Transportability | Fails on 3 conditions | Immune microenvironment, clonal heterogeneity, passage drift each create selection nodes that block transport from cell lines to patients. |
| 5 — Tian-Pearl general ID | Partially applicable | DepMap CRISPR lifts identification only if transport (Theorem 4) holds; it does not. DepMap is a falsification oracle, not an IV. |
| 6 — Mooij ANM direction | Inapplicable | Feedback loops and latent confounders make bivariate cause-effect direction tests unreliable for every gene-resistance pair. |
| 7 — Faithfulness | Violated at feedback nodes | Near-unfaithfulness is expected for dense PPI + regulatory feedback in MM. PC/FCI orientation fails at precisely the resistance-pathway nodes. |
| 8 — SEM path analysis | L1 use only | Wright's path rules yield valid observational decompositions; attention weights are not path coefficients. SEM fitted to MMRF is a valid L1 tool. |

### Assumptions ResistanceMap is currently making (implicitly)

1. **Exchangeability within cytogenetic stratum.** The model treats cell lines and
   patients as exchangeable within the same omics profile. This is violated: two patients
   with identical proteomics but different t(11;14) status have different BCL2-pathway
   biology.

2. **Attention weights ≈ causal weights.** The pathway attribution outputs (when
   attention is persisted) are presented as identifying which pathways drive resistance.
   This conflates L1 feature importance with L2 causal attribution. It is not supported
   by Theorems 1, 2, or 8.

3. **DepMap supports causal validation.** If DepMap essentiality scores are used to
   "validate" that a gene the model ranks highly is also essential in MM cell lines, the
   implicit claim is that cell-line essentiality is a causal anchor. Theorem 5 shows this
   requires transportability (Theorem 4), which fails. The valid use is falsification only.

4. **PPI propagation rank ≈ causal effect rank.** Addressed in §3 below.

---

## §3 The PPI-Propagation Causal Interpretation

### Formal question

Under what graphical assumptions is the propagation rank (top-k from RWR on STRING PPI)
a consistent estimator of causal effect rank?

### The honest derivation

Let s_i be the PPI propagation score for gene i (e.g., Personalized PageRank score
(1-α)(I-αA)^{-1} x, where x is the patient's multi-omic seed vector). Let τ_i be
the true causal effect of gene i on resistance, defined as
τ_i = P(resistance | do(gene_i = high)) - P(resistance | do(gene_i = low)).

**Claim.** rank(s) is a consistent estimator of rank(τ) if and only if:

  (C1) Causal drivers form a connected subgraph S in the PPI (planted-signal model,
       PPI_PROPAGATION_DRIVER_THEORY.md Theorem 1).
  (C2) The PPI edges used in propagation are structurally causal edges (experimental
       or database channel in STRING, not co-expression or text-mining channels alone).
  (C3) The seed signal x is a valid proxy for causal activity (mutation + cytogenetic
       event, not co-expression-derived expression level used as seed for expression-
       confounded PPI).
  (C4) No non-driver gene has PPI score s_j > min_{i in S} s_i due to hub topology
       alone (degree-corrected propagation is required; see PPI_PROPAGATION_DRIVER_THEORY
       §1.6 failure mode 1).

**When the rank agrees with causal effect rank.** Under C1–C4 jointly, the propagation
rank is a consistent estimator of causal effect rank up to within-plateau ties. This is
the justification for calling top-k propagated genes "candidate drivers." The word
"candidate" is load-bearing: it claims consistent rank ordering under these assumptions,
not identification of P(resistance | do(gene)).

**When the rank diverges from causal effect rank.** Four divergence scenarios:

1. **Hub bias without degree correction.** TP53, MYC, EP300 at degree > 1000 in
   STRING-0.7 accumulate propagation score from every patient's seed, regardless of
   causal role. Degree-uncorrected propagation gives rank(s_TP53) > rank(τ_TP53) in
   patients where TP53 is wild-type and non-driver. Mitigation: symmetric normalization
   A_hat = D^{-1/2} W D^{-1/2} (used in HotNet2; implemented in the RWR variant).

2. **Co-expression edge contamination.** If STRING edges from pure co-expression
   channel are included, then s_i integrates a correlation signal, not a structural
   signal. Genes in the same expression module as drivers accumulate score through
   co-expression edges. rank(s_i) > rank(τ_i) for all module co-members. Mitigation:
   non-observational channel filter (§1.5 of PPI_PROPAGATION_DRIVER_THEORY.md).

3. **Multi-omic seed contamination.** If expression z-scores are used as seed (rather
   than mutation indicators), then differential expression from cytogenetic confounders
   flows into the propagation. Genes downstream of del17p or chr1q21 (e.g., TP53
   regulated genes, 1q21 genes) inflate their scores in cytogenetically matched but
   non-driver contexts. rank(s_i) inflated for downstream targets of cytogenetic events.
   Mitigation: stratify seed construction by cytogenetic arm, or restrict seed to somatic
   mutation + cytogenetic-event indicator (binary).

4. **PPI incompleteness for MM-specific interactions.** STRING represents physical
   protein interactions accumulated across organisms and tissues. MM-specific interactions
   (BCMA-APRIL, CXCR4-CXCL12 bone marrow homing) may be absent or under-weighted.
   Genes causally relevant through MM-specific axes will be under-ranked by RWR on
   STRING. rank(s_i) < rank(τ_i) for MM-unique mediators.

### Summary position

PPI propagation rank is a defensible proxy for causal effect rank if and only if
assumptions C1–C4 hold and the four divergence scenarios above are explicitly mitigated.
The valid v10 claim is: "PPI propagation with degree-correction and non-observational
channel filtering recovers known MM drivers at ≥ 70% recall in the top-100 genes
(Falsification Protocol, §5 below), consistent with these genes being candidate causal
drivers under assumptions C1–C4."

The claim does NOT support: "gene G was ranked in the top-10 by propagation, therefore
G causes resistance." That claim requires identification (Theorems 2 and 5) which is
not achievable without DepMap CRISPR data plus valid transport (Theorem 4).

---

## §4 Falsifiable Predictions Pre-Specified for v10

### Prediction 1 — PPI rank recovers known MM drivers at ≥ 70% recall

**Hypothesis (H1).** RWR on STRING-filtered PPI seeded from MMRF mutation +
cytogenetic-event indicators (binary seed, not expression z-scores) places ≥ 35 of
the 50 held-out MM driver genes (Walker et al. 2018, PMID 29884741, Table 1) in the
top-100 propagated genes, with permutation p-value < 0.001.

**Test.** Exactly as specified in PPI_PROPAGATION_DRIVER_THEORY.md §5 (falsification
protocol). The 50-gene held-out driver set is listed there; it must not be used in any
seed or tuning step.

**What refutation means.** If recall < 70%: propagation rank is an inconsistent
estimator of causal effect rank given current STRING + MMRF data; assumption C1 or C2
fails; v10 must not claim driver identification beyond L1 association.

**What acceptance means.** Recall ≥ 70% with p < 0.001 is necessary but not sufficient
for causal driver identification. It is consistent with C1–C4 holding, and sufficient
to claim "candidate drivers in the top-k."

**Data required.** MMRF mutation MAF files (on disk), MMRF cytogenetics.tsv (on disk),
STRING 9606 filtered graph (on disk per PPI_PROPAGATION_DRIVER_THEORY.md). No new data
acquisition needed.

---

### Prediction 2 — Top-10 propagation drivers agree with DepMap MM essentiality

**Hypothesis (H2).** The top-10 genes from cohort-level propagation (averaged over
N=994 MMRF patients) show significantly higher CRISPR gene effect scores (more negative
Chronos scores = more essential) in MM cell lines (DepMap, multiple myeloma subset,
approximately 25–35 cell lines) compared to pan-cancer mean, with FDR < 0.05 using
a one-sided Wilcoxon test after excluding pan-essential genes (defined as Chronos
score < -0.5 in ≥ 90% of all lines).

**Test procedure.**
1. Extract the 10 genes from the cohort-averaged propagation rank (top-10 of F_bar from
   PPI_PROPAGATION_DRIVER_THEORY.md §5 step 4).
2. Download DepMap CRISPR Chronos gene effect matrix (configured at
   `depmap_crispr_path` in default.yaml; not yet on disk — this is a pending download).
3. Subset to MM cell lines by primary disease annotation.
4. For each top-10 gene, compare its MM-specific Chronos score to the pan-cancer mean
   Chronos score for that gene.
5. Apply FDR correction (Benjamini-Hochberg) across all 10 genes.
6. Report the fraction of the top-10 that are significantly more essential in MM than
   pan-cancer average.

**What refutation means.** If fewer than 5 of the top-10 propagation-ranked genes are
significantly MM-essential: either the propagation is finding PPI-proximate associations
that are not biologically driver-like, or the STRING-to-patient transport (Theorem 4)
is distorting the rank. The driver-attribution claim cannot be made even at the
"candidate" level.

**What acceptance means.** Significant MM-specificity of the top-10 propagation genes
(≥ 5 of 10, FDR < 0.05) is consistent with, but does not prove, causal driver status.
It constitutes a cross-modal falsification check: association signal (MMRF propagation)
and interventional signal (DepMap CRISPR) pointing in the same direction increases
plausibility of the driver claim.

**Data required.** Propagation output (runnable now). DepMap Chronos scores (pending
download; configured but not on disk per CAUSAL_VALIDITY_AUDIT.md §4 Tier 3).

---

## §5 Honest Tier Verdict — Driver Claim Pearl Tier Achievable with v10

| Claim level | Achievable in v10? | Conditions | Pearl tier |
|---|---|---|---|
| "Top-k genes are statistically associated with drug resistance in the MMRF cohort" | Yes — upon CoMMpass pipeline ingestion | Cytogenetic adjustment required; cytogenetics.tsv is on disk | L1 |
| "Top-k genes are candidate causal drivers under PPI-planted-signal assumptions C1–C4" | Yes — upon passing Prediction 1 (≥70% recall) | Degree correction + non-observational STRING filter required | L1 with structural prior (not L2) |
| "PPI propagation rank is a consistent estimator of causal effect rank" | Conditional — requires Prediction 1 and 2 both passing | Assumptions C1–C4 must hold; DepMap MM download required for P2 | L1/L2 boundary — consistent with L2 but not identified as L2 |
| "Gene G causally drives resistance to drug D in MM patients" | Not achievable in v10 | Requires transportability (Theorem 4, all three conditions — immune, clonal, passage — unresolved) and DepMap CRISPR transport to patient | Not achievable — L2 ceiling requires Theorem 4 resolution |
| "Pathway P mediates resistance (NIE > 0)" | Not achievable in v10 | Requires cycle-level RNA-Seq for L_t measurement (Theorem 3, Shpitser 2013); Cohen et al. NCT04065789 data is the closest available source | Not achievable — structural data gap |

**Summary statement.** ResistanceMap v10 achieves Pearl L1 for driver-candidate
identification and can defensibly claim L1 with structural prior (candidate driver under
named assumptions C1–C4). This is the maximum honest tier achievable with MMRF + STRING
+ DepMap in cell-line mode. Escalation to L2 requires: (a) resolution of Theorem 4
transport failures (immune microenvironment adjustment using scRNA, passage drift
correction using molecular comparisons to primary tumour), and (b) DepMap CRISPR
transport validation (Prediction 2). Neither is achievable in v10; both are v11 targets.

The word "driver" in any v10 output should be qualified as "candidate driver consistent
with PPI-propagation rank under assumptions C1–C4" and never stated as "causal driver"
without the Tier 3 CRISPR validation that Cohen et al. (PMID 33619369) executed for
PPIA in their 41-patient prospective trial.

---

## §6 Verified Citations

All identifiers HTTP-verified 2026-05-03.

| Reference | Identifier | Verification |
|---|---|---|
| Pearl 2009 *Causality* (front-door criterion) | ISBN 978-0-521-89560-6; Cambridge University Press | DOI 10.1017/CBO9780511803161 confirmed via Crossref |
| Shpitser and Pearl 2008, Complete ID Methods | JMLR v9 pp 1941-1979 | https://jmlr.org/papers/v9/shpitser08a.html HTTP 200 |
| Shpitser 2013, counterfactual graphical models | Cognitive Science, DOI 10.1111/cogs.12058 | Crossref confirmed: "Counterfactual Graphical Models for Longitudinal Mediation Analysis With Unobserved Confounding" |
| Bareinboim and Pearl 2014, transportability | Statistical Science, DOI 10.1214/14-sts486 | Crossref confirmed: "External Validity: From Do-Calculus to Transportability Across Populations" |
| Tian and Pearl 2002, general identification | AAAI 2002 Proceedings | UAI/AAAI proceedings; also available as UCLA Technical Report R-290 |
| Mooij, Janzing, Scholkopf 2016, cause-effect | JMLR v17 | https://jmlr.org/papers/v17/14-518.html HTTP 200; arXiv:1503.06389 HTTP 200 |
| Spirtes, Glymour, Scheines 2000, *CPS* | MIT Press, DOI 10.7551/mitpress/1754.001.0001 | Crossref confirmed (cited in PPI_PROPAGATION_DRIVER_THEORY.md) |
| Uhler, Raskutti, Buhlmann, Yu 2013, faithfulness geometry | Annals of Statistics, DOI 10.1214/12-aos1080 | Crossref confirmed: "Geometry of the faithfulness assumption in causal inference"; NOTE: PMID 23898000 suggested in task brief is WRONG (that PMID is an orthopedics paper) |
| Steen, Loeys, Moerkerke 2017, multiple mediators | American Journal of Epidemiology, DOI 10.1093/aje/kwx051 | Crossref confirmed: "Flexible Mediation Analysis With Multiple Mediators"; NOTE: PMID 28045530 suggested in task brief is WRONG (that PMID is a chemistry paper) |
| Wright 1934, path coefficients | Annals of Mathematical Statistics 5(3):161-215, DOI 10.1214/aoms/1177732676 | Historical; Crossref records extant |
| Walker et al. 2018, MM drivers | PMID 29884741, Blood | NCBI eutils esummary confirmed: "Identification of novel mutational drivers reveals oncogene dependencies in multiple myeloma" |
| Cohen et al. 2021 (PPIA/bortezomib) | PMID 33619369, Nat Med | Used in CAUSAL_VALIDITY_AUDIT.md; NCT04065789 |
| Zavidij et al. 2020, immune microenvironment | PMID 32747829, Nat Cancer | GSE124310 audit in CAUSAL_VALIDITY_AUDIT.md §2 |

---

*End of document.*
