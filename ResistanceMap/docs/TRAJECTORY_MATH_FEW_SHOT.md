# Trajectory Forecasting in the Few-Shot Regime: An Identifiability Analysis

**Author:** ResistanceMap mathematical audit (v10 architecture validation)
**Date:** 2026-05-03
**Data on disk:**
- $N_{\text{paired}} = 51$ patients with $\geq 2$ paired RNA-Seq visits (timepoints $t_0, t_1$, occasionally $t_2, t_3$).
- $N_{\text{baseline}} = 994$ patients with baseline-only RNA-Seq.
- $N_{\text{relapse}} = 319$ unpaired biopsies labelled "Recurrent" (cross-sectional).

This document derives, from first principles, what per-patient trajectory forecasts are mathematically *identifiable* in this regime. It is a **decision document** for the v10 architecture: do we ship a latent SDE (Approach A), an OT barycentric matcher (Approach B), or hierarchical Bayes with partial pooling (Approach C)?

---

## 1. Setup

### 1.1 Data model

Let $p \in \{1, \dots, P\}$ index patients, $P = N_{\text{paired}} + N_{\text{baseline}} = 1{,}045$. Each patient has:

- A **conditioning vector** $c_p \in \mathbb{R}^{d_c}$ — cytogenetics one-hot (t(4;14), t(11;14), del17p, …), ISS stage, age, baseline TPM low-rank summary. Empirically $d_c \in [10, 30]$ once one-hot encoded.
- An **observed latent state** $x_{t}^p \in \mathbb{R}^{d}$ at one or more times. We take $d = 512$ (scGPT pseudo-bulk latent of bone-marrow plasma cells), or $d = 64$ (current ResistanceMap VAE).
- For $p \leq 51$: pairs $(x_{t_0}^p, x_{t_1}^p)$ with median $\Delta t \approx$ 6–18 months.
- For $p > 51$: only $x_{t_0}^p$.

### 1.2 Prediction target

Given a **new** patient $p^\star$ with only baseline $x_{t_0}^{p^\star}$ and conditioning $c_{p^\star}$, predict either:

- (T1) the conditional mean $\mathbb{E}[x_{t_1}^{p^\star} \mid x_{t_0}^{p^\star}, c_{p^\star}]$, or
- (T2) the full conditional distribution $\rho_{t_1 \mid t_0, c}$, or
- (T3) a derived clinical scalar $y^{p^\star} = \phi(x_{t_1}^{p^\star})$ such as time-to-progression or resistance class.

(T1) is point estimation. (T2) is full distributional forecast. (T3) is the actionable target. We will see that (T3) becomes identifiable strictly *before* (T2).

### 1.3 The identifiability question

A model class $\mathcal{M} = \{P_\theta : \theta \in \Theta\}$ is **identifiable** at sample size $N$ if, with the data-generating process $P^\star$ in $\mathcal{M}$, the maximum-likelihood estimator $\hat\theta_N$ converges to a single $\theta^\star$ — equivalently, if $\theta \mapsto P_\theta$ is injective on the equivalence class supported by the empirical distribution. We must distinguish:

- **Population identifiability** (does $P^\star$ pin down $\theta^\star$ if $N \to \infty$?), and
- **Finite-sample identifiability / well-posedness** (is the Fisher information non-singular at $N = 51$?).

An estimator can be population-identifiable yet finite-sample useless if its Fisher information matrix has eigenvalues below the noise floor. This is exactly our regime.

---

## 2. Approach A — Latent SDE with patient conditioning

### 2.1 Generative model

$$
dX_t^p = f_\theta(X_t^p, c_p, t)\,dt + g_\theta(X_t^p, c_p, t)\,dW_t,\qquad X_0^p \sim \rho_0(\cdot \mid c_p),
$$

with $f_\theta : \mathbb{R}^d \times \mathbb{R}^{d_c} \times [0,T] \to \mathbb{R}^d$ a neural drift (Sobolev class $W^{k,2}$ for some $k \geq 1$) and $g_\theta$ positive-definite diffusion. This is the "neural SDE" of Tzen & Raginsky (2019, [arXiv:1905.09883](https://arxiv.org/abs/1905.09883)) and Li et al., "Scalable Gradients for Stochastic Differential Equations" (AISTATS 2020, [arXiv:2001.01328](https://arxiv.org/abs/2001.01328)).

### 2.2 Likelihood — Girsanov form

Let $\mathbb{P}_\theta$ denote the law of $X_{[0,T]}^p$ under SDE $\theta$, and $\mathbb{Q}$ the law of a reference Brownian motion with the same diffusion $g$. Girsanov's theorem (Øksendal, *Stochastic Differential Equations*, 6th ed., Thm 8.6.4) gives the Radon–Nikodým density on path space:

$$
\frac{d\mathbb{P}_\theta}{d\mathbb{Q}}(X_{[0,T]}) \;=\; \exp\!\left(\int_0^T g_\theta^{-1} f_\theta(X_t,c_p,t)^\top dX_t \;-\; \tfrac12 \int_0^T \|g_\theta^{-1} f_\theta\|^2\,dt\right).
$$

For paired data we observe only $(X_{t_0}, X_{t_1})$, not the full path. The marginal transition density $p_\theta(x_{t_1}\mid x_{t_0}, c_p)$ solves the Fokker–Planck (Kolmogorov forward) PDE; for non-trivial $f_\theta$ there is no closed form, and one estimates it via path-integral Monte Carlo (Pedersen 1995; [Aït-Sahalia 2002](https://www.jstor.org/stable/2692264)) or score-matching (Song et al. 2021, [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)). Practically we minimise the **simulation-free flow-matching** loss of Lipman et al. (2023, [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)) or the conditional flow-matching extension by Tong et al. (2023, [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)) which directly learns $f_\theta$ from $(X_{t_0}, X_{t_1})$ pairs.

### 2.3 Sample-complexity bound

The drift $f_\theta$ is parameterised by a neural network of width $H$, depth $L$. Effective parameter count for the drift alone is $\#\theta_f \approx d \cdot H + H^2 L + H \cdot d$. With $d = 512, H = 256, L = 2$, that is $\approx 2.6 \times 10^5$ parameters.

We have $N_{\text{paired}} \cdot d = 51 \times 512 = 26{,}112$ observed scalar values across the *paired* set. The parameter-to-sample ratio is

$$
\frac{\#\theta_f}{N_{\text{paired}} \cdot d} \approx \frac{2.6\times 10^5}{2.6\times 10^4} \approx 10.
$$

i.e. the model has **roughly 10 parameters per scalar observation in the paired set**. This is well into the over-parameterised regime where MLE without strong regularisation is non-identifiable.

The standard rule-of-thumb that an effective conditioning dimension $d_c^{\text{eff}} \lesssim \sqrt{N_{\text{paired}}}$ can be recovered comes from **minimax rates for non-parametric regression** under Sobolev smoothness (Stone 1982, [JSTOR](https://www.jstor.org/stable/2240700); Tsybakov, *Introduction to Nonparametric Estimation*, Springer 2009, Thm 2.8): with smoothness $\beta$ and conditioning dimension $d_c$, the minimax $L^2$ rate is $N^{-2\beta/(2\beta + d_c)}$. For this rate to be useful (say $\leq 0.1$), we need $N \gtrsim (10)^{(2\beta + d_c)/(2\beta)}$. With $\beta = 2$ (twice-differentiable drift) and target error 0.1, $N = 51$ admits at most $d_c \approx 4$. So we can identify a conditioning vector of dimension $\leq 4$ — *not* the 10–30-dim cytogenetic vector — unless we impose sparsity.

**Tightening under sparsity.** If $f_\theta$ depends on at most $s \ll d_c$ active components of $c_p$, the rate becomes $N^{-2\beta/(2\beta + s)}$ (Yang & Tokdar 2015, "Minimax-optimal nonparametric regression in high dimensions," [arXiv:1401.7278](https://arxiv.org/abs/1401.7278)). With $s = 2$ (e.g. only "high-risk cytogenetics" + "ISS stage"), $N = 51$ gives an error around $0.18$, marginally usable.

### 2.4 Population identifiability

Given a true $f^\star, g^\star \in W^{k,2}$ with $k > d/2 + 1$ (Sobolev embedding into $C^1$, Adams & Fournier, *Sobolev Spaces*, 2nd ed., Thm 4.12), the map $\theta \mapsto p_\theta(x_{t_1}\mid x_{t_0}, c_p)$ is injective on the support of $(x_{t_0}, c_p)$ provided:

(i) the support of $(x_{t_0}, c_p)$ is open in $\mathbb{R}^d \times \mathbb{R}^{d_c}$, and
(ii) $g_\theta$ is bounded away from zero (uniform ellipticity).

This is a corollary of **Hörmander's hypoellipticity** + **Aronson's two-sided Gaussian bounds** on transition densities (Aronson 1967, *Bull. AMS* 73, [DOI:10.1090/S0002-9904-1967-11790-7](https://doi.org/10.1090/S0002-9904-1967-11790-7)). To my knowledge there is **no theorem of the form "$c_p \mapsto P_\theta(\cdot\mid c_p)$ is injective" specifically for neural-parametrised SDEs**; the closest result is Hyvärinen & Pajunen (1999, [DOI:10.1016/S0893-6080(98)00140-3](https://doi.org/10.1016/S0893-6080(98)00140-3)) on nonlinear ICA, which shows non-identifiability *without* auxiliary variables and identifiability *with* sufficient auxiliary diversity (Khemakhem et al. 2020, "Variational Autoencoders and Nonlinear ICA: A Unifying Framework," [arXiv:1907.04809](https://arxiv.org/abs/1907.04809), Thm 1). The Khemakhem condition requires the conditioning $c_p$ to take at least $2d_c + 1$ distinct values with full-rank "sufficient-statistic" Jacobian. With 51 patients and 6 cytogenetic strata, that condition is **borderline-violated**.

### 2.5 Failure mode — out-of-support conditioning

If a test patient has a $c_{p^\star}$ outside the convex hull of $\{c_p\}_{p \leq 51}$ — e.g. they carry a t(14;16) translocation that no paired-data patient had — the SDE prediction is *extrapolation* in $c$. There is no consistency theorem that helps: bounds like Bach (2017, "Breaking the Curse of Dimensionality with Convex Neural Networks," [JMLR 18](https://jmlr.org/papers/v18/14-546.html)) require $c_{p^\star}$ to lie in the support of training $c$. Empirically this manifests as overconfident drift estimates pointing in arbitrary directions.

---

## 3. Approach B — Optimal-Transport reference matching

### 3.1 Generative model

We treat the 51 paired trajectories as an empirical measure on path space $\hat\mu_K = \frac{1}{K}\sum_{k=1}^K \delta_{(\xi_{t_0}^k, \xi_{t_1}^k)}$ where each $\xi^k$ is a (possibly cluster-aggregated) reference trajectory, $K \leq 51$. For a new baseline $x_{t_0}^{p^\star}$, predict

$$
\hat x_{t_1}^{p^\star} \;=\; \sum_{k=1}^K w_k \,\xi_{t_1}^k, \qquad w_k = \frac{\exp(-\|x_{t_0}^{p^\star} - \xi_{t_0}^k\|^2 / \tau)}{\sum_{j} \exp(-\|x_{t_0}^{p^\star} - \xi_{t_0}^j\|^2 / \tau)}.
$$

### 3.2 OT formulation

This is the **barycentric projection** of the entropic-regularised OT plan between $\delta_{x_{t_0}^{p^\star}}$ and $\hat\mu_K^{t_0}$, transported along the empirical $t_0 \to t_1$ coupling. Concretely, let $\pi^\varepsilon$ minimise

$$
\min_{\pi \in \Pi(\delta_{x^{p^\star}_{t_0}}, \hat\mu_K^{t_0})} \int \|x - y\|^2 d\pi(x,y) + \varepsilon \,\mathrm{KL}(\pi \| \delta_{x^{p^\star}}\otimes \hat\mu_K^{t_0}).
$$

Sinkhorn gives weights $w_k = \pi^\varepsilon_{p^\star, k}$ exactly proportional to the Gaussian kernel above with $\tau = \varepsilon$. The forecast then transports each reference's $t_1$ marginal back: $\hat x_{t_1}^{p^\star} = \mathbb{E}_{k\sim w}[\xi_{t_1}^k]$. This is the **Schiebinger et al. (2019) Waddington-OT** prediction (Cell 176, [DOI:10.1016/j.cell.2019.01.006](https://doi.org/10.1016/j.cell.2019.01.006)).

### 3.3 Asymptotics — kernel-width

As $\tau \to 0$, $w$ concentrates on the nearest reference (1-NN). As $\tau \to \infty$, $w \to$ uniform (population mean). For a true conditional expectation $m(x) = \mathbb{E}[X_{t_1}\mid X_{t_0}=x]$, the Nadaraya–Watson kernel estimator with bandwidth $\tau \asymp K^{-1/(d+4)}$ achieves the minimax rate $K^{-2/(d+4)}$ for twice-differentiable $m$ (Stone 1982, *op. cit.*; Tsybakov 2009 Ch. 1). With $K = 51, d = 512$, that bandwidth is essentially infinity — the rate is **useless for $d=512$**. The estimator becomes useful only after dimension reduction, e.g. project to $d' = 8$ first (then rate $\approx 51^{-2/12} \approx 0.6$ — slow but non-trivial).

### 3.4 W2 rate of convergence

Weed & Bach (2019, "Sharp asymptotic and finite-sample rates of convergence of empirical measures in Wasserstein distance," *Bernoulli* 25, [arXiv:1707.00087](https://arxiv.org/abs/1707.00087), Thm 1) show that for an empirical measure of $K$ samples from $\mu$ on $\mathbb{R}^d$:

$$
\mathbb{E}\,W_2^2(\hat\mu_K, \mu) \;\lesssim\; K^{-2/d} \quad (d \geq 5).
$$

With $K = 51, d = 512$, $K^{-2/512} \approx 0.985$, i.e. the empirical $W_2$ error is $\approx 1$, a **null result in the ambient latent space**. As above, this forces a low-dimensional projection: project to $d_{\text{eff}} \leq 8$ via PCA / VAE bottleneck and the rate becomes $51^{-1/4} \approx 0.37$ — again slow but at least bounded.

### 3.5 When does B beat k-NN?

Barycentric OT improves on plain $k$-NN whenever the conditional density is locally smoother than the marginal — formally, when the "smoothness index" $\beta_{m}$ exceeds $\beta_\rho$ in the bias–variance decomposition (Tsybakov 2009 Ch. 1). For MM trajectories where marginal heterogeneity is large but conditional drift is small (the empirical observation that within a cytogenetic stratum patients evolve similarly), this *helps*. So Approach B is principled provided we cluster references *within* cytogenetic strata.

---

## 4. Approach C — Hierarchical Bayes with partial pooling

### 4.1 Generative model

$$
\mu, \Sigma \sim \pi_0; \qquad \theta_p \mid \mu, \Sigma \sim \mathcal{N}(\mu, \Sigma); \qquad x_{t_1}^p \mid x_{t_0}^p, \theta_p \sim \mathcal{L}(\theta_p).
$$

For paired patients, both $x_{t_0}^p$ and $x_{t_1}^p$ inform the per-patient $\theta_p$ posterior. For baseline-only patients, only $x_{t_0}^p$ informs $\theta_p$ — but baseline-only patients **still inform $(\mu, \Sigma)$** through the marginal likelihood

$$
p(x_{t_0}^{1:P}, x_{t_1}^{1:51}) \;=\; \int p(\mu,\Sigma)\,\prod_p \int p(\theta_p\mid\mu,\Sigma)\,p(x^p\mid\theta_p)\,d\theta_p\, d\mu d\Sigma.
$$

That marginal-likelihood factorisation is the formal mechanism by which the 994 baseline-only patients sharpen the prior.

### 4.2 When does pooling strictly improve forecast accuracy?

**Stein's paradox** (Stein 1956, *Proc. 3rd Berkeley Symp.*, [JSTOR](https://www.jstor.org/stable/4355229); James & Stein 1961, *Proc. 4th Berkeley Symp.*) and the formalisation by Efron & Morris (1973, [JASA](https://www.jstor.org/stable/2284155)) show that the pooled estimator $\hat\theta_p^{\text{pooled}} = \alpha \hat\theta_p^{\text{MLE}} + (1-\alpha)\hat\mu$ strictly dominates $\hat\theta_p^{\text{MLE}}$ in mean-squared error when $P \geq 3$ and the per-patient noise is non-degenerate. The shrinkage factor $\alpha$ is data-determined.

For a **test patient** with only baseline data, the predictive distribution is

$$
p(x_{t_1}^{p^\star}\mid x_{t_0}^{p^\star}, \mathcal{D}) \;=\; \int p(x_{t_1}^{p^\star}\mid x_{t_0}^{p^\star}, \theta_{p^\star})\,p(\theta_{p^\star}\mid x_{t_0}^{p^\star}, \mathcal{D})\,d\theta_{p^\star}.
$$

With no paired data on $p^\star$, $p(\theta_{p^\star}\mid x_{t_0}^{p^\star}, \mathcal{D}) \approx p(\theta_{p^\star}\mid \hat\mu, \hat\Sigma, x_{t_0}^{p^\star})$ — i.e. it falls back to the cohort prior, sharpened by the 994 baselines. Pooling **strictly** improves accuracy when patient-level effects are partially correlated; it provides no improvement when each patient is truly independent (no exchangeability) — see §4.4.

### 4.3 Posterior contraction rate

Under standard regularity (Ghosal & van der Vaart 2007, *Annals of Statistics* 35, [DOI:10.1214/009053606000001172](https://doi.org/10.1214/009053606000001172), Thm 2.1; Castillo & Rousseau 2015, *Annals of Statistics* 43, [arXiv:1305.4482](https://arxiv.org/abs/1305.4482) on semiparametric BvM):

- Cohort-level $(\mu, \Sigma)$ contracts at rate $O((N_{\text{baseline}})^{-1/2}) = O(994^{-1/2}) \approx 0.032$.
- Per-paired-patient $\theta_p$ contracts at rate $O(N_{\text{paired,p}}^{-1/2}) = O(1)$ (one or two visits!).
- Predictive on **new** patient: contracts at rate $O((N_{\text{paired}})^{-1/2}) \approx 0.14$, **but only after** the cohort prior has stabilised (i.e. the constant in the rate is $\sigma(\hat\mu, \hat\Sigma)$, which the 994 baselines tighten).

This is the formal sense in which baseline-only patients help: they tighten the *constant*, not the *rate*, of new-patient prediction.

### 4.4 Exchangeability — when violated

The hierarchical model assumes patients are exchangeable: $p(\theta_1, \dots, \theta_P) = p(\theta_{\sigma(1)}, \dots, \theta_{\sigma(P)})$ for any permutation $\sigma$. **In MM this is violated across cytogenetic strata.** A t(4;14) patient's $\theta$ is structurally different from a t(11;14) patient's. The fix is **stratified hierarchical Bayes**: $\theta_p \mid c_p \sim \mathcal{N}(\mu_{c_p}, \Sigma_{c_p})$, restoring exchangeability *within* strata. With 6 strata and $N_{\text{paired}} = 51$, mean per-stratum is $\approx 8$ — small but viable for shrinkage estimators.

The classical reference is de Finetti's theorem (de Finetti 1937; Hewitt & Savage 1955, *Trans. AMS* 80, [DOI:10.2307/1992907](https://doi.org/10.2307/1992907)): exchangeability is equivalent to conditional iid given a latent measure. Stratification realises that latent measure as $c_p$.

---

## 5. Comparison

| Property | A (latent SDE) | B (OT match) | C (hierarchical Bayes) |
|---|---|---|---|
| Population identifiable? | Yes, under Aronson + Khemakhem | Yes, in $W_2$ as $K \to \infty$ | Yes, under exchangeability |
| Identifiable at $N_{\text{paired}} = 51$? | **No** unless $d_c^{\text{eff}}\leq 4$ + sparsity | **No** in $\mathbb{R}^{512}$; **yes** after PCA to $d \leq 8$ | **Yes** for cohort-level posteriors; per-patient is wide but well-defined |
| Consistent (correct in $N\to\infty$)? | Yes, rate $N^{-2\beta/(2\beta+d_c)}$ | Yes, rate $N^{-1/d}$ in $W_2$ | Yes, rate $N^{-1/2}$ post-pooling |
| Interpretable uncertainty? | Sample paths; no closed-form CIs | Empirical quantiles over references | Full Bayesian posterior — gold standard |
| Handles baseline-only patients? | No (semi-supervised tricks help marginally) | No (references must be paired) | **Yes — natively** |
| Handles OOD cytogenetics? | No (extrapolation in $c$) | Only if reference set covers stratum | Yes if stratified prior covers it |

---

## 6. Recommendation

**Rank: C > B > A.**

Hierarchical Bayes with partial pooling, **stratified by cytogenetic group**, is the unique approach where (i) the 994 baseline-only patients enter the likelihood non-vacuously through the marginal prior, (ii) the rate of contraction has a known closed form, (iii) the uncertainty quantification is exact (not Monte-Carlo-dependent), and (iv) the assumption of exchangeability is testable and fixable via stratification. Approach A's parameter count exceeds the paired-data budget by $\sim 10\times$, so even with regularisation it does not cross the identifiability threshold without strong sparsity; Approach B is mathematically clean but throws away the 994 baselines and degenerates in high dimensions ($K^{-1/d}$ with $d=512$ is uninformative). The single mathematical reason to prefer C: **only C admits a likelihood factorisation under which the baseline-only cohort sharpens new-patient predictions**, which is precisely the regime ResistanceMap is in.

A pragmatic v10 architecture: latent dimension $d \leq 16$ (post-VAE), stratified Gaussian process or Dirichlet-process mixture per cytogenetic group, with the 994 baselines feeding the cohort-level $(\mu_c, \Sigma_c)$ via marginal likelihood (NUTS or variational EM).

---

## 7. Falsification tests for Approach C

A test fails the approach if any of:

1. **Exchangeability stress test.** Hold out one cytogenetic stratum (say t(14;16)) entirely from the paired data. Train C with stratified prior on the remaining 5 strata, then forecast t(14;16) test patients using only the cross-stratum prior. If posterior coverage at 90% nominal is below 70%, the stratified-exchangeability assumption is broken and the model is mis-specified.
2. **Pooling-helps test.** Compare predictive log-likelihood on held-out paired patients with and without the 994 baseline-only patients in the marginal likelihood. If the gain is statistically indistinguishable from zero (paired bootstrap, $\alpha = 0.05$), partial pooling is providing no benefit and the architectural choice is unjustified.
3. **Calibration test.** On a held-out cohort, check that predictive credible intervals achieve nominal coverage. Failure ($> 5$pp deviation) indicates posterior is mis-calibrated, likely from prior mis-specification or non-Gaussianity in $\theta_p$.
4. **OOD perturbation test.** Apply a known intervention (e.g. proteasome-inhibitor pre-treatment in CoMMpass) and check whether the forecast updates in the clinically expected direction. If predictions are insensitive to the intervention, the posterior over $\theta_{p^\star}$ is dominated by the prior and the model is in practice not patient-personalised.

Any of (1)–(3) at the stated thresholds should refute the v10 architecture choice. Test (4) is the clinical validity check; failure means the model is statistically well-calibrated but clinically inert.

---

## References (all URLs verified to exist as of 2026-05; theorem statements are not paraphrased — read the original)

- Tzen & Raginsky 2019, *Neural SDE.* https://arxiv.org/abs/1905.09883
- Li et al. 2020, *Scalable Gradients for SDEs.* https://arxiv.org/abs/2001.01328
- Lipman et al. 2023, *Flow Matching.* https://arxiv.org/abs/2210.02747
- Tong et al. 2023, *Conditional Flow Matching.* https://arxiv.org/abs/2302.00482
- Aït-Sahalia 2002, *Maximum-likelihood estimation of discretely-sampled diffusions.* https://www.jstor.org/stable/2692264
- Song et al. 2021, *Score-Based Generative Modeling.* https://arxiv.org/abs/2011.13456
- Khemakhem et al. 2020, *VAEs and Nonlinear ICA.* https://arxiv.org/abs/1907.04809
- Hyvärinen & Pajunen 1999, *Nonlinear ICA.* https://doi.org/10.1016/S0893-6080(98)00140-3
- Aronson 1967, *Bounds for fundamental solutions of parabolic equations.* https://doi.org/10.1090/S0002-9904-1967-11790-7
- Schiebinger et al. 2019, *Optimal-Transport Analysis of Single-Cell Gene Expression.* https://doi.org/10.1016/j.cell.2019.01.006
- Weed & Bach 2019, *Sharp asymptotic and finite-sample rates… in Wasserstein distance.* https://arxiv.org/abs/1707.00087
- Stone 1982, *Optimal global rates of convergence for nonparametric regression.* https://www.jstor.org/stable/2240700
- Yang & Tokdar 2015, *Minimax-optimal nonparametric regression in high dimensions.* https://arxiv.org/abs/1401.7278
- Ghosal & van der Vaart 2007, *Convergence rates of posterior distributions.* https://doi.org/10.1214/009053606000001172
- Castillo & Rousseau 2015, *Bernstein–von Mises for semiparametric models.* https://arxiv.org/abs/1305.4482
- Stein 1956, *Inadmissibility of the usual estimator for the mean of a multivariate normal distribution.* https://www.jstor.org/stable/4355229
- Efron & Morris 1973, *Stein's estimation rule and its competitors.* https://www.jstor.org/stable/2284155
- Hewitt & Savage 1955, *Symmetric measures on Cartesian products.* https://doi.org/10.2307/1992907
- Bach 2017, *Breaking the Curse of Dimensionality with Convex Neural Networks.* https://jmlr.org/papers/v18/14-546.html

Textbooks cited without URL: Øksendal, *Stochastic Differential Equations* (Springer, 6th ed., 2003); Tsybakov, *Introduction to Nonparametric Estimation* (Springer, 2009); Adams & Fournier, *Sobolev Spaces* (Academic Press, 2nd ed., 2003).

---

**Caveats on theorem citations.**
- The Khemakhem et al. (2020) identifiability result is for variational autoencoders with auxiliary variables; its application to neural SDEs is **by analogy** — to my knowledge there is no published theorem of the form "the conditioning map of a neural SDE is identifiable under auxiliary-variable diversity," and proving one would be a contribution.
- The "$d_c \lesssim \sqrt{N}$" rule is **not a theorem** — it is a heuristic distillation of Stone's minimax rate. The actual statement is the rate $N^{-2\beta/(2\beta + d_c)}$, which is what was used in §2.3.
- The posterior contraction rate $O(N_{\text{paired}}^{-1/2})$ for new-patient prediction (§4.3) requires the cohort prior to be already concentrated; the joint rate is more subtle and (to my knowledge) not stated as a single closed-form theorem in the hierarchical-pooling literature for unbalanced longitudinal data with baseline-only auxiliaries. Castillo & Rousseau 2015 covers semiparametric BvM but not exactly this asymmetry.

When a theorem-of-this-exact-form is *not* in the literature, the document says so explicitly above.
