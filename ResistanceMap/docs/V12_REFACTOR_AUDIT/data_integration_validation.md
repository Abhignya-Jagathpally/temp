# V12 refactor — data integration validation

Run id: `r-2026-05-07-v12-refactor-data-integration-audit`
Cohort: MMRF CoMMpass IA22, N=787, events=224 (28.5%), median TT2L 654 d.
Source data: `data/processed/mmrf_baseline_expression.parquet`, expr_sha16 `08fc7bf286413112` (matches the v12 run exactly).
Artifacts:
  - `paper/v8_artifacts/v12_refactor_audit/integration_benchmark.json`
  - `paper/v8_artifacts/v12_refactor_audit/k_convergence_probe.json`
  - `scripts/v12_refactor/data_integration_audit.py`
  - `scripts/v12_refactor/k_convergence_probe.py`
Wall time: factor extraction 103 s; LOO Cox×17 ≈ 23 min; B=10 000 paired bootstrap×17 ≈ 36 min; total 3391 s.

---

## §1 Mechanism audit — real shared-factor inference vs concat

The v12 script `scripts/v12/s_v12_mofa_vs_v11_5_pfs.py` calls `mofapy2.entry_point()` with **a single view, single group** (lines 217-224):

```
ep.set_data_matrix(data=[[X]], likelihoods=["gaussian"],
                   views_names=["expression"], groups_names=["mmrf_ia22"], ...)
```

Model options (lines 229-235): `spikeslab_factors=False`, `spikeslab_weights=True`, `ard_factors=False`, `ard_weights=True`. With **one view × Gaussian likelihood × ARD on weights**, the MOFA+ posterior reduces analytically to **Bayesian probabilistic PCA with ARD-driven factor pruning** (Bishop 1999; Argelaguet et al. 2018 §2 — multi-view loss collapses to single-view ELBO when M=1). There is no cross-modal information to share, so the "shared latent space" claim in `docs/V12_MOFA_VS_V115_VERDICT.md` line 20 is technically vacuous on this slice.

Empirical proof of redundancy with v11 features (computed from `Z_mofa` in `mofa_vs_v11_5_per_patient_log_hazards.npz`):
  - `max_k |corr(Z_mofa[:,k], M_seed3)| = 0.808`
  - `max_k |corr(Z_mofa[:,k], z_pc1)|   = 0.974`

So one MOFA factor essentially **is** `M_seed3` (the proteasome score) and another **is** `z_pc1` (the dominant Sprint-1 latent PC). The combo's lift is therefore not from "shared-factor inference" but from the additional 27 non-redundant expression PCs.

The combo step itself is plain concatenation (`pd.concat([mofa_df, df_v11_richer, ...])`, line 468). No multi-view fusion mechanism beyond Cox-PH coefficient learning.

Verdict on §1: **The v12 result is single-view PPCA-with-ARD followed by feature concatenation, not a multi-omics shared-factor inference.** It is mislabeled "MOFA+" in the colloquial multi-modal sense. As a single-view sparse-factor model it is fine, but it does NOT demonstrate the value of MOFA's multi-view machinery on this cohort.

## §2 K-selection rationale (ARD pruning, ELBO trace)

Probe at iter_cap=20 (`k_convergence_probe.json`):

| K_req | K_eff (ARD) | per-iter (s) | ELBO last-5 % change | iter=1000 projection |
|------:|------------:|-------------:|---------------------:|---------------------:|
| 16 | 15 | 0.89 | +0.069 % | 14.9 min |
| 32 | 29 | 1.53 | +0.27 %  | 25.6 min |
| 48 | 43 | 2.32 | +0.84 %  | 38.6 min |
| 64 | 54 | 3.02 | +1.57 %  | 50.4 min |

Findings:
  1. ARD pruning is real: K=64 collapses to K_eff=54 within 20 iters, K=32 → 29, K=16 → 15. The "K=29 effective" reported by v12 is fully ARD-driven, not arbitrary truncation.
  2. The v12 wall-time claim ("K=64 + iter=1000 cannot finish in 1 hr CPU") is **borderline** — projected 50.4 min — so technically refuted in the ELBO-converged sense, but the v12 author reports two separate runs killed at 18 min and 4 min with "no convergence print," suggesting a watchdog/IO issue rather than a pure compute issue. The convergence-tolerance defense is stronger: at K=32 the ELBO last-5 change of 0.27 % is below the `convergence_mode="fast"` ~0.5 % threshold, while K=64 at iter=20 still moves 1.57 % — K=64 has **not yet converged**, while K=32 has.
  3. K=29 effective is well above v11_richer's 23 features and within the K∈[10, 30] sweet spot Argelaguet et al. 2020 recommend. **Choice is defensible.**

ELBO trace was recorded (n_elbo_recorded=20 for each K) but final_elbo persisted as `None` in the v12 JSON — this is a mofapy2 v0.7.4 stats-extraction bug, not a fit failure (we recover ELBO history in our probe). Recommend the production script add explicit `elbo_history` capture.

## §3 Head-to-head table (LOO Cox PH, penalizer=0.1, real MMRF data)

Numbers from `integration_benchmark.json`. PCA/ICA/NMF/SparsePCA/KernelPCA computed on the same log1p-TPM HVG=5000 matrix that mofapy2 sees. JIVE-joint surrogate = joint SVD of [PCA(expr,29) ‖ v11_richer]; BlockCCA = canonical scores of PCA(expr,29) vs v11_richer.

| Method | n_feat | C-marg | Δ vs v11_richer | 95% CI | p_boot (B=10k) | DeLong z | p_DeLong |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cox_v11_richer (reference)        | 23 | 0.6538 | — | — | — | — | — |
| PCA(K=29) alone                    | 29 | 0.6587 | +0.0051 | [-0.028, +0.038] | 0.758 | +0.30 | 0.760 |
| KernelPCA(K=29) alone              | 29 | 0.6544 | +0.0007 | [-0.033, +0.034] | 0.963 | +0.04 | 0.971 |
| ICA(K=29) alone                    | 29 | 0.6573 | +0.0037 | [-0.029, +0.036] | 0.822 | +0.22 | 0.828 |
| NMF(K=29) alone                    | 29 | 0.6519 | -0.0016 | [-0.035, +0.031] | 0.937 | -0.11 | 0.914 |
| SparsePCA(K=29) alone              | 29 | 0.6503 | -0.0033 | [-0.035, +0.028] | 0.837 | -0.23 | 0.821 |
| **MOFA+(mofapy2, K_eff=29) alone** | 29 | 0.6600 | +0.0064 | [-0.026, +0.038] | 0.689 | +0.39 | 0.695 |
| BlockCCA(expr↔v11) alone           | 22 | 0.6603 | +0.0066 | [-0.023, +0.036] | 0.657 | +0.45 | 0.656 |
| **JIVE-joint(K=29) alone**         | 29 | **0.6763** | **+0.0226** | [-0.001, +0.048] | 0.068 | +1.85 | 0.064 |
| **PCA + v11_richer**               | 52 | **0.6754** | **+0.0217** | [+0.0005, +0.0433] | **0.0450** | **+2.04** | **0.0413** |
| KernelPCA + v11_richer             | 52 | 0.6738 | +0.0200 | [-0.002, +0.043] | 0.077 | +1.82 | 0.068 |
| **ICA + v11_richer**               | 52 | **0.6754** | **+0.0217** | [+0.0003, +0.0434] | **0.0474** | **+2.03** | **0.0419** |
| NMF + v11_richer                   | 52 | 0.6711 | +0.0174 | [-0.004, +0.039] | 0.112 | +1.63 | 0.103 |
| SparsePCA + v11_richer             | 52 | 0.6726 | +0.0188 | [-0.002, +0.040] | 0.074 | +1.87 | 0.061 |
| **MOFA+ + v11_richer (v12)**       | 52 | 0.6751 | +0.0214 | [+0.0005, +0.0425] | **0.0456** | **+2.04** | **0.0417** |
| **JIVE-joint + v11_richer**        | 52 | 0.6751 | +0.0214 | [+0.0017, +0.0418] | **0.0344** | **+2.17** | **0.0298** |
| BlockCCA + v11_richer              | 45 | 0.6737 | +0.0199 | [+0.0017, +0.0388] | **0.0348** | +2.15 | 0.0313 |

Sanity gate: recomputed `Cox_v11_richer` LOO log-hazards have ρ=1.0000 vs the v12 artifact and identical marginal C=0.6538 — no implementation drift.

Cross-method observations:
  - **PCA, ICA, MOFA+, JIVE-joint, BlockCCA all reach the +0.020-+0.022 lift band when concatenated with v11_richer**, all crossing the p_boot<0.05 threshold. The mofapy2 lift is statistically and numerically indistinguishable from PCA's.
  - The MOFA-alone-vs-v11 marginal Δ=+0.0064 (CI overlaps 0) is dominated by simply replacing 23 hand-engineered features with 29 unsupervised expression factors — any reasonable factor model achieves it (PCA: +0.0051, ICA: +0.0037, BlockCCA: +0.0066).
  - **JIVE-joint alone (0.6763, +0.0226) is the only single-model that outperforms the entire MOFA+v11 combo in marginal C-index using only 29 features and no v11_richer concat.** Its CI is [-0.001, +0.048] (just touches 0 from below; p_boot=0.068) — borderline, so not a strict win, but suggests the joint expression-clinical decomposition is what is doing the work.
  - JIVE-joint+v11 has the **tightest CI (+0.0017, +0.0418) and lowest p_boot (0.0344) of all combos**, edging out the v12 MOFA+v11 result.

OmicsNet 2.0 / DIABLO / scAI — literature comparison (no native Python wheel for any):
  - **DIABLO** (Singh et al. 2019, Bioinformatics) is sparse generalised CCA across blocks. Our BlockCCA(expr↔v11) alone gives 0.6603 (Δ=+0.0066, p=0.66) and BlockCCA+v11 gives 0.6737 (Δ=+0.0199, p=0.0348). On a single MMRF-expression view, DIABLO would be expected to land between BlockCCA and SparsePCA — it cannot beat MOFA in this regime because both blocks would need to be true omics blocks, and our second block (v11_richer) is a derived hand-feature set. **Predicted lift: ≤+0.022, statistically indistinguishable from PCA+v11.**
  - **JIVE** (Lock et al. 2013, Ann Appl Stat). Our pure-joint surrogate beats MOFA on the same input. The full JIVE algorithm (joint + per-block individual) on (expression, v11) has no Python implementation we could install (`jive` PyPI package is dead since 2017). Given the joint-only surrogate already matches/beats MOFA, **full JIVE is unlikely to lose; treat the +0.0214 lift as the floor**.
  - **OmicsNet 2.0** (Zhou et al. 2022, Nucleic Acids Res) is fundamentally a network-overlay visualisation/enrichment service, not a survival predictor. There is no published JIVE/MOFA-style C-index benchmark for it. Out of scope for direct head-to-head on PFS.
  - **scAI** (Jin et al. 2020, Genome Biol) jointly factorises scRNA + scATAC. MMRF IA22 has neither at this slice — irrelevant for this cohort. Including for completeness only.

## §4 Statistical validation re-run

The v12 finding survives every test we re-ran:

  - Marginal paired bootstrap (B=10 000): Δ=+0.0214, 95% CI [+0.0005, +0.0425], p=0.0456 ✓ (matches v12 to 4 d.p.).
  - 5-strata cyto-stratified paired bootstrap (B=10 000): Δ=+0.0212, 95% CI [+3.6e-05, +0.0424], p=0.0496 ✓ (CI lower bound essentially at zero — fragile).
  - Pencina-D'Agostino paired-bootstrap SE → z-test: z=+2.04, p_two=0.0417 ✓ (concurs with the bootstrap p).

But two specifics to flag:

  1. **The CI lower bound is +0.0005 (95%) and +3.6×10⁻⁵ (stratified)**. The combo wins, but barely. A single change in the bootstrap seed or one more censored event flipping in or out would push the CI across zero. p=0.046 ↔ p=0.050 is statistically the same conclusion in any frequentist worldview.
  2. **PCA+v11_richer attains an identical p=0.045, ICA+v11_richer p=0.047, JIVE+v11_richer p=0.034** on the same N=787. The "MOFA+ adds something beyond concat" claim does not survive — the lift is from "29 expression factors of any reasonable kind concatenated with v11_richer."

## §5 Live-pipeline wiring proposal (no application)

Production survival-prediction script: `scripts/v11/s5e_cox_with_programs.py`. It currently builds five Cox variants (`Cox_v11_richer`, `Cox_v11_routed_mmsygnal`, `Cox_v11_six_mmsygnal_scores`, `Cox_only_six_mmsygnal_scores`, `Cox_v11_program_activity_pcs`) at lines 261-275. The MOFA factors are **not** in it.

Minimal-diff wiring (recommended — DO NOT apply now, requires v11.5 reviewer approval):

  1. Add to `data/processed/`:
     - `mmrf_mofa_K29_factors.npy`  (787 × 29, written once by `scripts/v12/s_v12_mofa_vs_v11_5_pfs.py`'s npz Z_mofa)
     - `mmrf_mofa_K29_sample_ids.json` (alignment index to `submitter_id`)

  2. In `scripts/v11/s5e_cox_with_programs.py:main()` after line 274, add one variant:
     ```python
     # added by v12 refactor (gated by env var to keep v11 reproducible)
     if os.getenv("RM_INCLUDE_MOFA_FACTORS") == "1":
         Zm = np.load(PROC / "mmrf_mofa_K29_factors.npy")
         m_ids = json.loads((PROC / "mmrf_mofa_K29_sample_ids.json").read_text())
         m_idx = np.array([m_ids.index(s) for s in df["submitter_id"]])
         mofa = pd.DataFrame(
             (Zm[m_idx] - Zm[m_idx].mean(0)) / (Zm[m_idx].std(0, ddof=0) + 1e-9),
             columns=[f"mofa_F{k+1}" for k in range(Zm.shape[1])],
         )
         variants["Cox_v11_with_mofa_K29"] = pd.concat(
             [df_v11_richer, mofa, pd.DataFrame({"t": t_days, "e": event})], axis=1)
     ```

  3. **Important**: gate behind `RM_INCLUDE_MOFA_FACTORS` so v11.5 published artifacts remain bit-reproducible. The Cox_v11_with_mofa_K29 variant becomes the v12-headline learner only after the §6 risks are addressed.

  4. **Equivalent simpler diff**: replace the MOFA factors with `PCA(K=29)` of the same HVG-5000 log1p TPM. This adds 0 new dependencies (no mofapy2 build), gives the same +0.022 lift (95 % CI nearly identical), and avoids the K-convergence and ELBO-extraction issues. **Recommended path of least architectural surface area.**

  5. Downstream: the U_θ neural-ODE checkpoint training (`scripts/v10/sprint1_*.py`, not v11) does not need MOFA factors — it operates on Z and U_θ directly. Wiring scope is confined to the Cox-PH discrimination layer.

## §6 Risks

  1. **Mechanism is mislabeled.** Single-view MOFA = Bayesian PPCA + ARD. The "shared latent space" interpretation is not licensed by the data slice. Any paper claim that this is multi-omics integration must be retracted or re-grounded once paired proteomics/CNV land.
  2. **Equivalence to PCA.** PCA+v11_richer reaches identical Δ=+0.0217 with identical CI and p_boot=0.045 in 0.2 s. ICA+v11_richer and JIVE+v11_richer match or beat it. The +0.021 lift attributed to MOFA is an attribute of "29 unsupervised expression-space factors," not of mofapy2 specifically.
  3. **Borderline p, multiple comparisons.** v12 ran a single (MOFA, MOFA+v11) comparison so its p=0.046 stands at face value. *But this audit has now run 16 paired-bootstrap tests on the same cohort.* Bonferroni at α=0.05/16=0.003 — only **JIVE+v11 (p=0.034)** and **BlockCCA+v11 (p=0.035)** survive after correction... no, neither survives 0.003 either. Holm-step-down places PCA+v11, ICA+v11, MOFA+v11 firmly in the non-significant tail. The v12 finding alone (1 test) is fine; the broader claim "factor models help" is *not* multiple-testing-robust at 0.003.
  4. **K=64 was not tried to convergence.** At K=64 mofapy2 has K_eff_iter20=54 and ELBO still moving 1.57 % per 5 iters — convergence may yield a different factor structure. A 1-hour single run would resolve this; the v12 deviation is documented but not closed.
  5. **CNV / proteomics absent.** The single-modality regime is the *worst case* for MOFA+. The audit cannot rule out that paired modalities would deliver a >+0.05 lift; the current evidence simply does not support claiming MOFA-specific value.
  6. **Borderline CI ↔ "architectural opportunity" labelling.** The v12 verdict tag `V12_ARCHITECTURAL_OPPORTUNITY` is fair only because the +0.01 combo-lift threshold and CI-excludes-zero gates were both pre-registered before the run. If a future paper cites the +0.0214 lift as evidence for MOFA-the-method, this audit must be cited alongside.

---

## Verdict

**CONDITIONAL.** The v12 statistical finding is reproducible, but the +0.0214 C-index lift is **not specific to MOFA+**. PCA, ICA, JIVE-joint, and BlockCCA on the same N=787 input deliver indistinguishable lifts at p≈0.034-0.047. The "shared-factor multi-omics" framing is unjustified on this single-view slice. Recommend (a) replace mofapy2 with PCA(K=29) for the live pipeline (zero dependency cost, identical performance), or (b) defer the multi-omics integration claim until paired CNV/proteomics arrive. Do **not** publish "MOFA+ shared factors close the v11 gap" — publish "29 unsupervised expression factors of any kind close the v11 gap by Δ=+0.021, 95 % CI [0.000, 0.043], p=0.045."
