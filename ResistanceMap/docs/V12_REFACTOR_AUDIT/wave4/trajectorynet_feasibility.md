# V12 Wave-4: TrajectoryNet feasibility and head-to-head

Run id: r-2026-05-07-v12-refactor-trajectorynet-adaptation
Reviewer hat: PhD-level dynamical-systems / single-cell ML.
Source paper: Tong, Huang, van Dijk, Krishnaswamy. ICML 2020 / PMC8320749 (DOI 10.1101/2020.03.04.974899).

## 1. Paper summary

Input data structure: cross-sectional samples of a population at K >= 2 time points. At each timepoint we have a distribution (cloud of cells), not paired trajectories. Cells are anonymous between timepoints. Native validation regime: scRNA-seq in 5D PCA, 4 to 5 timepoints with hundreds to thousands of cells.

Loss: CNF with Neural ODE field f(z,t). (1) KL / MLE on terminal distribution. (2) Energy L_E = lambda * E_t [|f|^2] (Benamou-Brenier W_2). (3) Jacobian Frobenius via Hutchinson. (4) Density k-NN prior. (5) RNA-velocity cosine prior. (6) Growth network g(z,t) for unbalanced OT.

Output: learned vector field f(z,t). Produces phi_T(z_0) for any z_0 and interpolated density at any t in [0,T]. NOT a per-sample risk score - no survival head.

Minimum config: K >= 2 timepoints (paper smallest is K=2 toy; real-data K=4 to 5). Hundreds of cells per timepoint is the practical minimum.

Per-sample feature extraction: phi_T(z_0) and f(z_0,t) are deterministic per-sample functions that can feed downstream regressors. The paper does not do this, but the architecture allows it.

## 2. Fit-for-purpose verdict

(A) Direct survival predictor - STRUCTURAL FAIL. TrajectoryNet outputs a vector field, not patient-level risk scores. No survival likelihood, no Cox partial-likelihood head. Bolting Cox on top reduces to role (B). Reject.

(B) Feature extractor on N_paired = 29 - FEASIBLE BUT WEAK. The only place a true 2-timepoint structure exists in MMRF: the N_paired = 29 patients with both baseline and post-relapse RNA (z0, z1 in 64-dim VAE latent space). Train a CNF that maps the 29-cloud at t=0 to the 29-cloud at t=1, push every one of N=787 baseline z0 vectors through phi_T to get a per-patient embedding for Cox.

Structural mismatch flagged openly: paper regime needs K >= 2 with hundreds-thousands of points per timepoint; we have K=2 with 29 points per timepoint, no replicates, single (z0,z1) pair per patient. The CNF is dramatically under-determined: empirical KL/MMD between two 29-clouds in 64D has high finite-sample variance, and the Jacobian/energy regularizers do more of the lifting than the data fit. Far outside paper validation regime.

Choose (B) anyway and report honestly. A negative result is itself useful evidence about the limits of single-cell-trajectory tooling on bulk-RNA cohorts.

(C) Synthetic time-index from PFS - REJECT. Constructing a pseudotime ordering from the PFS outcome is target leakage. Per hard rule, this path is closed.

Decision: implement (B); document (A) and (C) as structurally / methodologically blocked.

## 3. Implementation

Code: scripts/v12_refactor/wave4/trajectorynet_adaptation.py
Run id: r-2026-05-07-v12-refactor-trajectorynet-adaptation

Architecture: ODEFunc is a 3-layer MLP, 64 hidden units, leaky-ReLU(0.2), input concat(z_64, t_scalar), output dz/dt in R^64. Final layer zero-initialized so the network starts at the identity flow (inductive prior: relapse trajectory is a small perturbation of baseline). Neural ODE: torchdiffeq.odeint_adjoint with dopri5, atol=rtol=1e-5. Adam optimizer lr=1e-3, weight_decay=5e-5 (matches paper section 4.3).

Loss (paper-faithful with one substitution): MMD_unbiased(phi_T(z0_train), z1_train) + 1e-2 * E_t [ ||f(z, t)||^2 ] + 1e-3 * Hutchinson(||J_f||_F^2). We substitute multi-bandwidth Gaussian MMD for the paper KL/MLE loss because at N=29 the empirical density estimate of KL is unstable in 64D. MMD is well-defined for small empirical samples and is a standard distributional discrepancy in OT-flavored work. We omit L_density, L_velocity, L_growth: no RNA-velocity in MMRF bulk RNA, growth not meaningful at the patient level, manifold-density penalty needs more cells than we have.

Training: 10000 iterations on the full 29-point batch. CPU run; wall time reported in JSON. Standardization: z-score using mean/std of paired z0; same transform applied to all 787 baseline z0.

Embedding: for each of N=787 patients we compute z_T (64) + displacement_norm + f_norm_t0 = 66 features.

## 4. Head-to-head vs v11.5

Quantitative results live in paper/v8_artifacts/v12_refactor_audit/wave4/trajectorynet.json (summary_per_set and paired_bootstrap_vs_v115_routed_mmsygnal).

Reference: Cox_v11_routed_mmsygnal (v11.5 production), C = 0.6955 on N=787.

Comparators reported in JSON:
- Cox_v11_routed_mmsygnal (production reference, C = 0.6955)
- Cox_v11_richer (recompute, 23 features)
- Cox_flow_alone (66 flow features)
- Cox_v11_richer + flow_embed (89 features)

Per-stratum C-indices over the 5 cytogenetic strata are in summary_per_set per_stratum.

A model is strict beat iff: 95% CI lower bound > 0 AND empirical p_two_sided < 0.05/3 = 0.0167 (Bonferroni m=3).

## 5. Honest verdict

The numerical verdict against v11.5 is in the JSON; the structural verdict is independent of the numbers and known a priori from sections 1-2.

TrajectoryNet on this cohort cannot beat v11.5 in any defensible way. The paper success regime (K >= 4 timepoints with hundreds-thousands of cells) is a different scientific problem from "embed N=787 single-baseline patients given one 29-patient (z0,z1) pair." Even if the random-seed lottery yielded a marginal Delta-C with a CI excluding 0, the result would not generalize: the CNF is fitting 29 points in 64D with zero replicates, and the embedding for the other 758 patients is pure extrapolation through the learned vector field. v11.5 already exploits the same z64 latent through Waddington features (U_z0, U_zT, grad_norm, displacement) plus first 8 z-PCs - the additional path-integral information from a CNF over a single empirical pair is informationally redundant.

Honest recommendation: TrajectoryNet is not the right method for MMRF-style cohorts (single-baseline bulk RNA, sparse follow-up). It would be the right method for a hypothetical clinical-trial cohort with K >= 4 serial RNA timepoints across 100+ patients - an order of magnitude richer dataset than what exists for MM today. Until such data lands, stay with v11.5 / mmsygnal-routed.

The numerical result simply confirms (or fails to confirm) what the structural analysis already predicts.

---

## Appendix: empirical numbers from the run

Reference: Cox_v11_routed_mmsygnal C = 0.6955 on N=787 events=224.

C-indices (LOO Cox PH penalizer=0.1):

| Model | n_features | Marginal C |
|---|---|---|
| Cox_v11_routed_mmsygnal (v11.5 ref) | production | 0.6955 |
| Cox_v11_richer_production | 23 | 0.6538 |
| Cox_v11_richer_recompute | 23 | 0.6538 (rho=1.000 vs production) |
| Cox_flow_alone | 66 | 0.6232 |
| Cox_v11_richer + flow_embed | 89 | 0.6513 |

Paired bootstrap (B=10000) vs Cox_v11_routed_mmsygnal:

| Model | Delta marginal | 95% CI | p_two_sided | Strict-beat? (Bonferroni m=3, alpha=0.0167) |
|---|---|---|---|---|
| Cox_flow_alone | -0.0724 | [-0.1051, -0.0398] | 0.0002 | NO (strict LOSS) |
| Cox_v11_richer_recompute | -0.0418 | [-0.0583, -0.0250] | 0.0000 | NO (strict LOSS) |
| Cox_v11_richer + flow_embed | -0.0443 | [-0.0691, -0.0201] | 0.0004 | NO (strict LOSS) |

Stratified bootstrap (5 cyto-strata) gives essentially identical numbers. All 3 comparisons strict-LOSS at Bonferroni-corrected alpha = 0.05/3 = 0.0167. The flow embedding even hurts v11_richer when concatenated (C drops 0.6538 to 0.6513).

Training diagnostics: MMD pushed-z0 vs z1 = -0.01075; identity-flow baseline MMD = -0.00879. The trained flow only marginally improves over the identity, confirming the structural prediction in section 2: at N=29 in 64D the energy regularizer dominates and the flow stays near identity. Wall time: training 511.6s + Cox LOO 386s + bootstrap 384s = 774s total CPU.

Final verdict: TrajectoryNet adaptation LOSES to v11.5 with strict statistical significance on all 3 comparisons. The structural analysis correctly anticipated this outcome. TrajectoryNet is not the right tool for single-baseline bulk-RNA cohorts.
