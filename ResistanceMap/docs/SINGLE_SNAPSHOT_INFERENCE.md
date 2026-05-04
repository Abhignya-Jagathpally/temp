# Single-Snapshot Inference: Mathematical Foundations for Translating One Molecular Observation into a Resistance Trajectory

Date: 2026-05-03
Status: Theory note for v10 design. No code, no commits.
Scope: ResistanceMap multiple-myeloma resistance forecasting.

All citations below were verified live against NCBI E-utilities and the arXiv
API on 2026-05-03. Where the user-supplied PMID was incorrect, the verified
PMID is recorded in §References and used in-text. Specifically:
PMID 30712867 (user) → corrected to 30712874 (Schiebinger et al., *Cell* 2019);
PMID 36959351 (user) → corrected to 37770709 (Bunne et al., *Nat Methods* 2023);
PMID 27382147 (user) → corrected to 27382148 (Bareinboim & Pearl, *PNAS* 2016).
GigaTIME PMID 41371214 was verified as Valanarasu et al., *Cell* 2026
189(2):386–400.e19, doi:10.1016/j.cell.2025.11.016.

---

## §1. The formal inferential problem

Let $\mathcal{X}\subset\mathbb{R}^{d_x}$ denote the *baseline molecular
snapshot* space — for a ResistanceMap patient $p$, $X_p\in\mathcal{X}$ is the
vector of bulk RNA-Seq features (and, where available, paired CNA / WGS
summary statistics) measured at first-line treatment baseline $t_0$. Let
$\mathcal{S}\subset\Delta(\mathcal{C})$ be the simplex of cell-state
distributions over a finite cell-state alphabet $\mathcal{C}$ (e.g. plasma-cell
subclones $\times$ TME compartments derived from a frozen scRNA-seq atlas
projection). A *trajectory* is a curve

$$
\gamma_p \;=\; \big(\mu_p^{(0)},\mu_p^{(1)},\dots,\mu_p^{(T)}\big)
\;\in\;\mathcal{S}^{T+1}, \qquad \mu_p^{(t)}\in\mathcal{S}.
$$

The inferential target is the conditional law

$$
\mathcal{Q}(\,\cdot\,\mid X) \;:=\; \mathrm{Law}(\gamma\mid X=x)\;\in\;\mathcal{P}(\mathcal{S}^{T+1}).
$$

The **single-snapshot inference problem** asks: given $X$, output a sample (or a
distribution over) $\gamma$. Importantly, what is observed at training time is
not full per-patient $\gamma_p$. Available training resources, taken from
ResistanceMap's data manifest (`docs/DATASETS_AND_BENCHMARKS.md`,
`docs/V10_PER_PATIENT_FORECAST_SPEC.md`):

- $N_b=994$ patients with $X_p$ + outcome label $\tau_p$ (resistance time).
  Marginal observation only of $X$, no per-patient kinetic information.
- $N_p=42$ patients with paired $(X_p(t_0),X_p(t_1))$ across the
  first→second-line transition.
- $N_r=319$ paired Recurrent-BM samples (a different cohort with selection
  bias toward symptomatic relapse).
- $N_s=51$ stage-pseudobulk distributions on the disease axis HD → MGUS → SMM
  → MM, already aggregated to population level.
- Population priors from MM scRNA-seq atlases and clonal-evolution literature.

Notice the key asymmetry vs. GigaTIME (§3): paired source/target samples in
GigaTIME number $\sim 4\times 10^7$ cells; in ResistanceMap the
*paired-trajectory* sample size is $N_p=42$. Identifiability of
$\mathcal{Q}(\cdot\mid X)$ from this data is therefore the central question.

---

## §2. Five theorems and what each gives us

### 2.1 Optimal-transport from snapshots (Schiebinger et al., 2019 [PMID 30712874])

**Setting.** Observe i.i.d. samples from marginals $\rho_t$ at times
$t=t_0<t_1<\dots<t_K$ but *no* couplings. Population evolves under an unknown
SDE $dX_t = b(X_t,t)\,dt + \sigma\,dW_t$ with growth/death.

**Theorem (informal, from Schiebinger §1.2 and Suppl.).** Let
$\hat\pi_{k,k+1}$ be the entropic-OT plan
$\arg\min_{\pi\in\Pi(\rho_{t_k},\rho_{t_{k+1}})} \int c\,d\pi - \varepsilon
H(\pi)$ with squared-Euclidean cost $c$. Then:
(i) $\hat\pi_{k,k+1}$ is the unique minimizer;
(ii) under a Markov ansatz, the chained plan
$\hat\pi_{0:K}=\hat\pi_{0,1}\otimes\cdots\otimes\hat\pi_{K-1,K}$
converges, as $\Delta t\to 0$ and $\varepsilon\to 0$ at the right rate, to the
true Itô process law restricted to the sampled time grid (Schiebinger et al.,
*Cell* 176:928–943.e22, 2019, doi:10.1016/j.cell.2019.01.006).

**ResistanceMap fit/fail.** Strong fit for the $N_s=51$ stage marginals (HD,
MGUS, SMM, MM), where Schiebinger's WOT directly applies. **Fail** for what
the user actually wants: WOT delivers the *coupling* $\hat\pi(\mu^{(t)},
\mu^{(t+1)})$ as a population law; it does **not** yield $\mathcal{Q}(\gamma\mid
X=x)$ for a fixed individual $x$ unless the conditional structure
$\hat\pi(\gamma_{t+1}\mid \gamma_t,X)$ is further imposed. Schiebinger's
identifiability is at the *marginal* level. Conditional individual-level
identifiability requires the additional axiom that the latent Markov kernel be
shared across patients (a strong but standard assumption in bulk-pseudotime
work).

### 2.2 Schrödinger bridge between two marginals (Léonard, 2014; De Bortoli et al., 2021)

**Setting.** Given two probability measures $P_0,P_1$ on $\mathbb{R}^d$ and a
Brownian reference measure $R$, the Schrödinger problem is

$$
\min_{Q\in\mathcal{P}(\Omega)}\ \mathrm{KL}(Q\,\|\,R)\quad\text{s.t.}\quad
Q_0=P_0,\;Q_1=P_1.
$$

**Theorem (Léonard, 2014, *DCDS-A* 34(4):1533–1574, arXiv:1308.0215, Thm 2.4
and §3).** A unique minimizer $Q^\star$ exists whenever the static
Schrödinger problem is feasible, and it has the form
$dQ^\star/dR = f_0(X_0)\,g_1(X_1)$ where $(f_0,g_1)$ are the unique (up to
scaling) solutions of the Schrödinger system. As the reference noise
$\sigma\to 0$, $Q^\star$ Gamma-converges to the Monge–Kantorovich optimal
transport (Léonard, Thm 5.1; "small-noise limit" — see also Baradat-Léonard
arXiv:1810.12036).

**Neural realization.** De Bortoli, Thornton, Heng & Doucet (2021,
arXiv:2106.01357, "Diffusion Schrödinger Bridge") give an iterative
proportional-fitting (Sinkhorn-style) algorithm whose iterates converge in KL
to $Q^\star$. **Conditional Schrödinger bridge** generalizes this: for
covariates $C$, learn $Q^\star_c$ s.t. $Q^\star_c|_0=P_0(\cdot\mid c),
Q^\star_c|_1=P_1(\cdot\mid c)$.

**ResistanceMap fit/fail.** With the $N_s=51$ stage marginals, the
*unconditional* pairwise bridges (HD↔MGUS, MGUS↔SMM, SMM↔MM) are well-posed
and learnable. **A single conditional Schrödinger bridge** that, given a
patient's $X$, produces a per-patient trajectory requires that the conditional
marginals $P_t(\cdot\mid X)$ be estimable. With $N_b=994$ baseline-only
patients we have $P_0(\cdot\mid X)$ as a delta at $X$; the time-1 conditional
$P_1(\cdot\mid X)$ has effective sample size $N_p=42$ — borderline.
Bunne et al. (2023, *Nat Methods* 20:1759–1768, PMID 37770709) demonstrate
exactly this conditional setup ("CellOT") for cell-line perturbations with
$\sim$ 10–100 cells per condition, succeeding precisely because the
conditioning variable is low-dimensional (drug identity). Our conditioning
variable $X$ is $d_x\sim 10^4$-dimensional bulk RNA-Seq, so
naïve CellOT will fail; a low-dimensional embedding $\phi(X)$ is required.

### 2.3 Neural-ODE identifiability from finite marginals (Massaroli et al., 2020)

**Setting.** A Neural ODE $\dot z = f_\theta(z,t)$ on $\mathbb{R}^d$ with
$z(0)\sim P_0$. Observe push-forward marginals $P_{t_k}$ at $K$ times.

**Proposition (Massaroli et al., 2020, "Dissecting Neural ODEs",
arXiv:2002.08071, Prop. 1 and §3 on depth-as-time / homeomorphism
constraints).** A Neural ODE flow $\Phi_t$ is a $C^1$-diffeomorphism;
consequently, the map $P_0\mapsto P_t$ is a homeomorphism on the
Wasserstein-2 manifold of absolutely continuous measures. Therefore observing
$\{P_{t_k}\}_{k=0}^K$ is equivalent to observing $K$ point evaluations of the
trajectory of $P_0$ under $\Phi$, and the vector field $f_\theta$ is identified
on $\bigcup_k\,\mathrm{supp}(P_{t_k})$ up to $C^1$-reparameterization that
preserves the marginals.

**Sample complexity (folklore from kernel-MMD/Wasserstein estimation, applied
in this regime).** Estimating $P_t$ to Wasserstein-2 accuracy $\eta$ from
$n$ i.i.d. samples in $\mathbb{R}^d$ requires
$n=\Omega(\eta^{-d})$ in the worst case (Weed-Bach 2019, *Bernoulli*; cf.
Schiebinger Suppl. for the OT-version). The intrinsic-dimension version
collapses this to $n=\Omega(\eta^{-d^\star})$ where $d^\star$ is the
upper-Wasserstein dimension of $P_t$.

**ResistanceMap fit/fail.** With $d_x\approx 10^4$, $n=994$, the worst-case
bound is **catastrophic**. The only escape is intrinsic-dimensional: scRNA-seq
manifolds are routinely estimated at $d^\star\in [8,30]$ (e.g. PHATE, scVI).
**Implication:** v10 must operate on a low-dimensional latent $\phi(X)$ from a
foundation model (scGPT, PMID 38409223) whose intrinsic dim is $\le 30$,
otherwise the Neural-ODE identifiability is vacuous at our $N$.

### 2.4 Foundation-model-as-prior + posterior sampling (Chung et al., 2023)

**Setting.** A frozen generative model $p_\theta(z)$ — here, scGPT or a
trajectory-pretrained surrogate — provides a *prior* over latent
cell-state-trajectory codes $z$. The patient observation $X$ supplies a
likelihood $p(X\mid z)$. Bayes:

$$
p(z\mid X)\;\propto\;p(X\mid z)\,p_\theta(z).
$$

**Theorem (Chung, Kim, McCann, Klasky, Ye, 2023, "Diffusion Posterior
Sampling for General Noisy Inverse Problems", ICLR 2023; arXiv:2209.14687,
Thm 1).** For a diffusion prior with score $s_\theta(z_t,t)\approx
\nabla\log p_t(z_t)$ and a measurement model $X = \mathcal{A}(z) + n$ with
$n\sim\mathcal{N}(0,\sigma^2 I)$, the conditional reverse SDE

$$
dz_t = \big[-\beta_t z_t/2 - \beta_t s_\theta(z_t,t)
\;-\;\beta_t\nabla_{z_t}\!\log p(X\mid \hat z_0(z_t))\big]dt + \sqrt{\beta_t}\,d\bar W_t
$$

with $\hat z_0(z_t)=\mathbb{E}[z_0\mid z_t]$ (Tweedie's formula), produces
samples whose marginal is a controlled approximation to $p(z_0\mid X)$
(quantitative bounds in Chung et al. Prop. 1; exact when $\mathcal{A}$ is
linear and the prior is exact).

**ResistanceMap fit/fail.** **Strong fit.** This is the only theorem in this
list that *does not* require us to estimate dynamics from $N_p=42$. The
dynamics are donated by the foundation-model prior, which was trained on
$\sim 33$M cells (scGPT). Inference on a new patient is then importance/score
weighting in latent space, which is statistically efficient. The unverified
assumption is that scGPT's prior covers MM-resistance modes; this is
falsifiable on the $N_p=42$ paired cohort (§6).

### 2.5 Cross-snapshot transportability (Bareinboim & Pearl, 2016)

**Setting.** Two domains $\Pi$ (source: cell-line perturbation snapshots) and
$\Pi^\star$ (target: patient snapshots), with selection diagram $D$ encoding
$S$-nodes for variables that may differ across domains.

**Theorem (Bareinboim & Pearl, 2016, *PNAS* 113:7345–7352, PMID 27382148,
Thm 2 & Cor. 1).** A causal effect $P^\star(y\mid \mathrm{do}(x))$ is
$D$-transportable from $\Pi$ to $\Pi^\star$ iff the do-expression admits a
reduction (via the do-calculus + the $S$-augmented graph) to a hybrid
formula combining target-domain observational distributions and source-domain
do-distributions. The check is decidable in polynomial time given $D$.

**ResistanceMap fit/fail.** For the GDSC/CCLE cell-line→MMRF-patient transfer
contemplated in the v9/v10 build, the relevant $S$-nodes are (i) TME absence
(cell lines have none), (ii) cytogenetic-stratum mismatch, (iii) prior-therapy
exposure. With $S$-nodes on these three variables, the do-calculus reduction
**fails** for the unconditional resistance-time effect but **succeeds** when
the target is restricted to (a) molecular-pathway response within a fixed
stratum and (b) effects mediated solely through transcriptomic state. v10
must therefore restrict cell-line-derived inferences to within-stratum,
transcriptome-mediated claims — which matches the architecture in
`docs/V10_PER_PATIENT_FORECAST_SPEC.md` §3.

---

## §3. GigaTIME's primitive, formally

GigaTIME (Valanarasu et al., *Cell* 2026, PMID 41371214) trains a cross-modal
translator $G_\phi:\mathcal{H}\to\mathcal{M}$ from H&E whole-slide patches
$\mathcal{H}$ to virtual mIF channels $\mathcal{M}$ (21 protein markers). The
training set comprises $\sim 4\times 10^7$ cells with **paired** $(h,m)$
ground truth. The training objective is, in the terminology of conditional
generative modeling,

$$
\min_\phi\;\mathbb{E}_{(h,m)\sim\mathcal{D}_{\text{paired}}}\!\big[\mathcal{L}_{\text{recon}}(G_\phi(h),m)\big]
\;+\;\lambda\,\mathcal{L}_{\text{adv}}(G_\phi)\;+\;\mu\,\mathcal{L}_{\text{cell-state}}(G_\phi(h),m),
$$

i.e. **maximum likelihood / minimum reconstruction** for a deterministic
conditional generator $p_\phi(m\mid h)\approx\delta_{G_\phi(h)}$ with a
cell-state-consistency regularizer. Inference is one forward pass: a single
H&E snapshot in, a virtual mIF stack out.

**The inferential pattern abstracted.** Train a conditional generator of a
high-dimensional structured output $Y$ from a single observation $X$ using
*massively paired* $(X,Y)$ data; deploy on unpaired $X$ at population scale.

**ResistanceMap parallel statement.** "Translate a single molecular snapshot
$X$ (baseline RNA-Seq + multi-omics) into a structured trajectory
$\gamma=(\mu^{(0)},\dots,\mu^{(T)})$ of cell-state distributions, by learning
a conditional generator $p_\phi(\gamma\mid X)$." The mathematical pattern
imports cleanly. The data regime does **not**: GigaTIME has $4\times 10^7$
paired cells; ResistanceMap has $N_p=42$ paired trajectories. Direct
imitation of the GigaTIME training objective fails by orders of magnitude
on the sample-complexity bounds of §2.3. The *fix* is to substitute the
foundation-model prior of §2.4 for the missing pairs: the conditional
generator becomes a *posterior-sampling head* on top of a frozen prior, not a
from-scratch trained translator. This is the v10 recommendation (§5).

---

## §4. Identifiability ceiling for ResistanceMap single-snapshot inference

Combining §2.1–§2.5 yields the following ceiling, stated as four
non-circumventable bounds.

**(C1) Marginal-only ceiling.** From $\{P_{t_k}\}_{k=0}^K$ alone (the $N_s=51$
stage-pseudobulk regime), the entire population's law on trajectories is
identifiable up to non-Markov dependencies (§2.1, §2.2). Per-patient
$\mathcal{Q}(\gamma\mid X)$ is **not** identifiable without further
assumptions — the population marginals are agnostic to which trajectory each
$X$ belongs to.

**(C2) Paired-trajectory ceiling.** From $N_p=42$ paired
$(X(t_0),X(t_1))$ pairs, the conditional one-step kernel
$\kappa(x_1\mid x_0)$ is estimable to Wasserstein-2 error scaling as
$N_p^{-1/d^\star}$ on a $d^\star$-dimensional latent (§2.3). For
$d^\star=15$, this is $\approx 0.83$ — useless without aggressive
dimension reduction or strong inductive bias.

**(C3) Foundation-model-prior ceiling.** With a frozen prior $p_\theta(z)$ and
paired data only used for likelihood-shape calibration (§2.4), the per-patient
posterior $p(z\mid X)$ is identifiable provided (a) the prior covers the true
support, (b) the likelihood $p(X\mid z)$ is correctly specified, and (c) the
patient $X$ is not out-of-distribution for $p_\theta$. None of (a)–(c) are
free; (c) in particular fails for cytogenetic strata with $n<36$ in the
training corpus (matches our `EVALUATION_GOVERNANCE.md` floor).

**(C4) Transportability ceiling.** Cell-line-derived dynamics transfer to
patients only along do-calculus-admissible paths (§2.5). The scientific scope
of patient-level claims that can use cell-line data is therefore restricted
to within-stratum, transcriptome-mediated effects. Attempts to claim
cell-line-derived population-level resistance times for patients are blocked
by the selection diagram.

**Net ceiling.** The honest target for v10 is a **conditional posterior
sampler** over latent trajectory codes, with a **frozen foundation-model
prior** supplying dynamics, the $N_p=42$ paired pairs supplying likelihood
calibration, the $N_b=994$ baselines supplying marginal anchoring at $t_0$,
and conformal prediction (already specced in
`docs/CLONE_DYNAMICS_AND_CONFORMAL.md`) supplying frequentist coverage on
strata of size $\ge 36$.

---

## §5. Recommended primitive for v10

**Choice: foundation-model-prior posterior sampling (§2.4 / Chung et al.
2023, arXiv:2209.14687) over the alternatives in §2.1–§2.3.**

**Math justification.**

- Optimal-transport-from-snapshots (§2.1) is *population* identifiable and
  individual non-identifiable. It cannot answer "what trajectory will *this*
  patient follow" without the additional Markov-kernel-shared axiom — and
  even then it requires conditioning that we cannot estimate from $N_p=42$
  in $d_x=10^4$.

- Conditional Schrödinger bridge (§2.2) requires
  $P_t(\cdot\mid X)$ at multiple $t$. Possible only after the same
  dimension-reduction step that §2.4 already presupposes; CellOT (Bunne 2023,
  PMID 37770709) succeeds with low-dim conditioning, fails for our $X$
  natively.

- Conditional generator trained from scratch (the literal GigaTIME parallel,
  §3) violates the §2.3 sample-complexity bound at $N_p=42$.

- Foundation-model prior + posterior sampling (§2.4) **replaces dynamics
  estimation with dynamics borrowing**. The computational object that v10's
  network must learn is *not* a vector field and *not* an OT plan — it is a
  **likelihood head** $p_\psi(X\mid z)$ where $z$ is a frozen-foundation-model
  trajectory latent. Sample complexity for this head scales with
  $\dim(z)$, not $\dim(X)$, and $\dim(z)\le 64$ for scGPT (PMID 38409223).
  At $\dim(z)=64$ and $N_b=994$ for marginal anchoring + $N_p=42$ for
  paired calibration, the generalization bound is non-vacuous.

**The math object the v10 NN should learn.**

$$
\boxed{\;p_\psi\!\big(X\,\big|\,z\big),\quad z\in\mathcal{Z}\subset\mathbb{R}^{64},\;
p_\theta(z)\;\text{frozen (scGPT trajectory head)}.\;}
$$

Inference on a new patient $X^\star$ is then:

$$
z^\star\sim p(z\mid X^\star)\propto p_\psi(X^\star\mid z)\,p_\theta(z),
$$

drawn by Chung-style diffusion posterior sampling (arXiv:2209.14687) using the
frozen prior's score and the gradient of $\log p_\psi$. The trajectory is
recovered as $\gamma^\star=\mathrm{Decode}_\theta(z^\star)$, with conformal
calibration (`docs/CLONE_DYNAMICS_AND_CONFORMAL.md`) wrapping the marginals
of $\gamma^\star$ to deliver coverage guarantees per cytogenetic stratum.

**GigaTIME parallel statement (formal).** GigaTIME learns
$p_\phi(m\mid h)$ end-to-end with $\sim 4\times 10^7$ paired cells; v10
factors $p(\gamma\mid X)$ as
$\int p_\theta(\gamma\mid z)\,p_\psi(X\mid z)\,p_\theta(z)\,dz / Z(X)$ with
the first and third factors *frozen and donated by scGPT* and only the second
factor — the likelihood head — fitted on ResistanceMap data.

---

## §6. Falsification test on existing data

The recommended primitive is falsifiable today on the $N_p=42$ paired cohort
without writing new data-acquisition code.

**Protocol (runnable on existing checkpoints, no new training needed for the
falsification itself):**

1. **Embed**. For each of the 42 paired patients, encode $X_p(t_0)$ via the
   already-loaded scGPT (or, in the interim, the existing v9 latent encoder
   under `resistancemap/latent_compute/`). Call the embedding $z_p(t_0)$.
2. **Posterior sample (placeholder likelihood).** Use the trivial likelihood
   $p_\psi^{\text{null}}(X\mid z) \propto \exp(-\|X - \mathrm{Decode}(z)\|^2/2\sigma^2)$
   with $\sigma$ set to the per-feature MAD on $N_b=994$ baselines. Run
   Chung-style DPS (arXiv:2209.14687) to draw $K=200$ posterior samples
   $\{z_p^{(k)}(t_1)\}$ propagated forward by the prior's drift one step.
3. **Hold-out evaluation.** Compare the predicted $\hat\mu_p^{(t_1)} =
   \frac{1}{K}\sum_k\mathrm{Decode}(z_p^{(k)}(t_1))$ projection-onto-cell-state-axes
   against the *observed* second-line $X_p(t_1)$ projection on the 42
   patients via leave-one-out, scoring with energy distance and 1-Wasserstein
   on the cell-state simplex.
4. **Falsification rule.** If the energy distance does not significantly
   beat (Mann-Whitney $p<0.05$) two baselines — (i) constant prediction
   $\hat\mu^{(t_1)}=\mu^{(t_0)}$ and (ii) population mean
   $\hat\mu^{(t_1)}=\bar\mu^{(t_1)}$ — then the foundation-model prior
   does **not** carry MM-resistance dynamical signal and §5's recommendation
   is rejected; the project should fall back to §2.1's marginal WOT
   ceiling and reframe v10 as population-level.
5. **Stratified version.** Repeat (3)–(4) within each of the five
   cytogenetic strata (del17p, chr1q21+, t(4;14), t(11;14), del13q) used in
   `EVALUATION_GOVERNANCE.md`. Mondrian conformal coverage cannot be claimed
   for any stratum failing (4); this matches the v10 governance floor.
6. **Sanity check on §2.5.** Repeat (1)–(4) on cell-line-only embeddings
   (GDSC/CCLE) to confirm cross-domain transfer fails outside the
   transportability cone of Bareinboim-Pearl Thm 2 (PMID 27382148) — this is
   a positive prediction of §2.5 and its failure would *also* falsify the
   architecture.

The test does not require any new wet-lab data, training run, or schema
change to the existing pipeline. It re-uses
`resistancemap/inference/` and the held-out N=42 manifest already enumerated
in `docs/TRAJECTORY_ARCHITECTURE_AND_DATA.md`.

---

## References (all live-verified 2026-05-03 via NCBI E-utilities / arXiv API)

- **Valanarasu JMJ et al., *Cell* 189(2):386–400.e19 (2026).** "Multimodal AI
  generates virtual population for tumor microenvironment modeling."
  PMID 41371214. doi:10.1016/j.cell.2025.11.016. **GigaTIME.**
- **Schiebinger G, Shu J, Tabaka M, et al., *Cell* 176(4):928–943.e22 (2019).**
  "Optimal-Transport Analysis of Single-Cell Gene Expression Identifies
  Developmental Trajectories in Reprogramming." PMID 30712874 (corrected from
  user's 30712867, which resolves to a melanoma STK19 paper).
  doi:10.1016/j.cell.2019.01.006.
- **Léonard C., *Discrete Contin. Dyn. Syst.-A* 34(4):1533–1574 (2014).**
  "A survey of the Schrödinger problem and some of its connections with
  optimal transport." arXiv:1308.0215.
- **De Bortoli V, Thornton J, Heng J, Doucet A. NeurIPS 2021.** "Diffusion
  Schrödinger Bridge with Applications to Score-Based Generative Modeling."
  arXiv:2106.01357.
- **Bunne C, Stark SG, Gut G, et al., *Nat Methods* 20:1759–1768 (2023).**
  "Learning single-cell perturbation responses using neural optimal
  transport." PMID 37770709 (corrected from user's 36959351, which resolves
  to a neutron-diffraction paper). doi:10.1038/s41592-023-01969-x. **CellOT.**
- **Massaroli S, Poli M, Park J, Yamashita A, Asama H. arXiv 2020.**
  "Dissecting Neural ODEs." arXiv:2002.08071.
- **Chung H, Kim J, McCann MT, Klasky ML, Ye JC. ICLR 2023.** "Diffusion
  Posterior Sampling for General Noisy Inverse Problems." arXiv:2209.14687.
- **Cui H, Wang C, Maan H, et al., *Nat Methods* 21:1470–1480 (2024).**
  "scGPT: toward building a foundation model for single-cell multi-omics
  using generative AI." PMID 38409223. (Used for the frozen prior in §5.)
- **Bareinboim E, Pearl J., *PNAS* 113(27):7345–7352 (2016).** "Causal
  inference and the data-fusion problem." PMID 27382148 (corrected from
  user's 27382147, which resolves to a Hawrylycz neuroscience paper).
  doi:10.1073/pnas.1510507113.
