# V12 Refactor Audit — Visualization Report

**Script:** `scripts/v12_refactor/krishnaswamy_visualizations.py`
**Output directory:** `paper/v8_artifacts/v12_refactor_audit/visualizations/`
**Run date:** 2026-05-07
**Embedding method:** UMAP (PHATE not installed; `phate` package absent from venv)
**Fixed UMAP seed:** `random_state=42`, `n_neighbors=30`, `min_dist=0.3`
**DPI:** 200 (PNG + PDF with embedded fonts)

All figures derive exclusively from on-disk checkpoints and processed data.
`generate_paper_figures.py` was not called. `np.random` is not imported.

---

## Data provenance

| File | SHA-256 prefix | Run ID |
|------|---------------|--------|
| `data/processed/mmrf_z64.npy` | `ef3e96d4df5daeb3` | r-2026-05-03-v10s1 |
| `data/processed/mmrf_paired_z64.npz` | `7b2d274c9ae7bb12` | r-2026-05-03-v10s1 |
| `data/processed/mmrf_outcomes_treatment.tsv` | `49671871ed5d9efa` | r-2026-05-03-v10s4 |
| `paper/v8_artifacts/v11_sprint5/cox_per_patient_log_hazards.npz` | `3317ab00f1e3fbc2` | r-2026-05-03-v11s5-cox-with-programs |
| `paper/v8_artifacts/v12_sprint1/mofa_vs_v11_5_per_patient_log_hazards.npz` | `b1a2f698012c71fa` | r-2026-05-04-v12-mofa-vs-v115 |
| `paper/v8_artifacts/v11_sprint5/mondrian_jk_plus_v11.json` | `0e774c21adbd495b` | r-2026-05-03-v11s5-conformal |
| `checkpoints/u_theta_v10s1.pt` | `e40d8053b20ce25e` | r-2026-05-03-v10s1 |

---

## Figure 1 — `fig1_latent_cyto_embedding.png`

**Question answered:** Are the five cytogenetic strata separable in the v10 latent space?

**Data:** N=787 z64 latents from `mmrf_z64.npy`; stratum labels from `cox_per_patient_log_hazards.npz` (key `strata`).

**Method:** UMAP (fallback from PHATE; labeled in title), `n_neighbors=30`, `min_dist=0.3`, `random_state=42`. Points are 8px, alpha=0.7, rasterized. Muted categorical palette (del17p=crimson, t(4;14)=orange, +1q21=green, t(11;14)=blue, other=grey). No top/right spines, serif font.

**Interpretation:** Partial cluster separation is visible, particularly for del17p and t(4;14). The degree of overlap motivates the Mondrian stratification in Sprint 5 rather than a single global conformal interval. The +1q21 cluster (S3, n=165) is the largest and most diffuse — consistent with its gain being a secondary event less tightly coupled to a single gene-expression signature.

---

## Figure 2 — `fig2_ode_pseudotime.png`

**Question answered:** Do the 29 paired patients show energy-descent pseudotime consistent with the Sprint-1 U_θ gradient flow?

**Data:** `mmrf_paired_z64.npz` (z0, z1, N=29); `u_theta_v10s1.pt` (checkpoint sha256 `e40d8053b20ce25e`).

**Method:** ODE integration via 64-step Euler (`dz/dt = -∇U_θ(z)`, dt=1/64, T=1). This matches the Sprint-1 `euler_chain_64step` reference (L2 deviation from dopri5 = 0.0053 at T=1.0). Predicted z(T) positions projected onto the Fig 1 UMAP by nearest-neighbor lookup in latent space. Left panel: manifold with arrows z0→ẑ(T); color encodes ΔU = U_θ(z0) − U_θ(ẑ(T)). Right panel: U_θ(z0) vs U_θ(z1 observed) scatter.

**Caveats:** (1) The UMAP was fit on all N=787; z1 and ẑ(T) are projected by approximate nearest-neighbor, not exact UMAP transform — positions are indicative only. (2) Sprint-7 verdict remains: F8 STRICT FAIL (ED_prior ≈ ED_const; N_paired=29 < Stone minimax bound for d=64). This figure visualizes geometric coherence, not population-level predictive discrimination.

**Interpretation:** Most paired patients move to lower U_θ (Lyapunov decrease), confirming the gradient flow is numerically stable. Patients with large ΔU (high pseudotime, warm colors) may represent rapid disease progression; this is directional evidence only, not a calibrated prediction.

---

## Figure 3 — `fig3_meld_bort_density.png`

**Question answered:** Is Bortezomib-1L treatment-arm membership localized to specific cell states on the latent manifold?

**Data:** `mmrf_z64.npy` (N=787); `mmrf_outcomes_treatment.tsv` (bort_1L flag: 579 positive, 208 negative); patient IDs linked via `cox_per_patient_log_hazards.npz`.

**Method:** Gaussian KDE (`scipy.stats.gaussian_kde`, bandwidth=0.25) on the Fig 1 UMAP coordinates for each condition independently. Log₂ density ratio = log₂(KDE_Bort / KDE_nonBort) smoothed by Gaussian filter (σ=2 pixels). Colormap: coolwarm (diverging). This is a MELD-style approach (Krishnaswamy 2021) implemented with KDE rather than diffusion-based signal smoothing (MELD library not installed).

**Interpretation:** If resistance to Bortezomib is cell-state-determined, red regions (Bort-enriched) should co-localize with high-risk latent positions; blue regions (non-Bort-enriched) with low-risk regions. The Sprint-4 NIE result (proteasome-mediated effect HR=1.09, F10 STRICT FAIL due to confounding) is consistent with treatment assignment being only weakly cell-state-determined — which should appear as a diffuse rather than sharply localized log-ratio map.

---

## Figure 4 — `fig4_hazard_gradient_map.png`

**Question answered:** What is the spatial structure of predicted risk across the latent manifold under three models?

**Data:** `mofa_vs_v11_5_per_patient_log_hazards.npz` from run r-2026-05-04-v12-mofa-vs-v115 (sha256 `b1a2f698012c71fa`). Keys used: `MOFA_factors`, `Cox_v11_richer`, `MOFA_factors_plus_v11`. All are LOO Cox PH log-partial-hazards on MMRF TT2L PFS (N=787, 224 events).

**Method:** Scatter plot on Fig 1 UMAP coordinates; color = log-partial-hazard (coolwarm; clipped to 2nd–98th percentile to suppress outlier influence). Three panels side-by-side for direct comparison.

**Interpretation:** The combination MOFA+v11 panel (right) should show a more structured risk gradient than either component alone — consistent with the V2 architectural-opportunity verdict (Δ marginal C-index = +0.021, 95% CI [+0.0005, +0.0425], p=0.046). The t(11;14) cluster (S4, blue points in Fig 1) is the region where all three models fail most often (C-index near chance), visible as a noisy / low-contrast region in the risk map.

---

## Figure 5 — `fig5_stratum_residual_heatmap.png`

**Question answered:** Within each cytogenetic stratum, which patients are worst-predicted and how large are the residuals?

**Data:** `cox_per_patient_log_hazards.npz` from run r-2026-05-03-v11s5-cox-with-programs. Keys: `t_days`, `event`, `strata`, `Cox_v11_routed_mmsygnal` (LOO log-partial-hazard; marginal C=0.6955, the best v11.5 model).

**Method:** Rank-based predicted log(TT2L+1): inverse-rank transform (rank of predicted survival ∝ inverse rank of log-hazard; monotone isotonic mapping). Absolute residual = |log(1+t_obs) − log(1+t_pred)|. Each stratum panel displays patients as image rows sorted by descending residual; two columns show observed vs predicted log-days. Violin plot (panel 6) summarizes the residual distribution per stratum.

**Interpretation:** S1 del17p (mean residual 1.12) and S2 t(4;14) (1.12) have larger absolute errors than S5 other (0.75), partly because high-risk strata have shorter, more variable TT2L. S4 t(11;14) (0.96) shows the widest violin — consistent with its near-chance C-index (0.643) and the STRICT FAIL of the v12 BCL2-family specialist head.

---

## Figure 6 — `fig6_conformal_calibration.png`

**Question answered:** Does the jackknife+ conformal coverage hold at the nominal 90% level within each cytogenetic stratum, and for which base learner?

**Data:** `mondrian_jk_plus_v11.json` from run r-2026-05-03-v11s5-conformal (sha256 `0e774c21adbd495b`). Three base learners: Ridge_v10 (11 features), Ridge_v11features (15 features, +4 Waddington), GBM_v10 (HistGBM, 11 features).

**Method:** Horizontal bar chart per stratum, showing empirical 90% coverage for each learner. Dashed vertical line at nominal=0.90; grey band = ±3% tolerance. All coverage values are taken directly from the JSON artifact (no recomputation).

**Key numbers:**
- Ridge_v10: all 5 strata within ±0.005 of 0.90 — F_S5 STRICT PASS
- Ridge_v11features: all 5 strata within ±0.006 of 0.90 — F_S5 STRICT PASS (primary model)
- GBM_v10: all strata between 0.939–0.969 — over-covers by 4–7% — F_S5 FAIL

**Interpretation:** Asymmetric tree residuals break the jackknife+ symmetric-quantile assumption, producing over-wide intervals for GBM. Both Ridge models maintain near-exact coverage across all five cytogenetic strata, satisfying the Barber-Candès-Ramdas-Tibshirani 2021 finite-sample guarantee. The Waddington-feature upgrade (Ridge_v11features) tightens marginal interval width by 8 days (1635d vs 1643d) without sacrificing coverage.

---

## Missing inputs / skipped figures

No required figures were skipped. All 7 input files were present on disk.

The `v11/ode_*.pt` path mentioned in the brief does not contain a standalone `.pt` checkpoint; the v11 Neural-ODE is implemented via the Sprint-1 `u_theta_v10s1.pt` (the U_θ network) combined with the `torchdiffeq` / Euler integrator in-script. The `paper/v8_artifacts/v11_sprint1/ode_forward_diagnostic.json` confirms this is the same checkpoint used for v11 diagnostics.

---

## Reproducibility

```bash
# From repo root, with venv activated:
MPLBACKEND=Agg python scripts/v12_refactor/krishnaswamy_visualizations.py
```

Expected runtime: ~3 min (dominated by UMAP on N=787).
Deterministic: yes (random_state=42 fixed for UMAP; all other steps deterministic).
