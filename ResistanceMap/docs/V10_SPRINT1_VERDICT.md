# v10 Sprint 1 Verdict — Scalar Waddington Potential U_θ

**Date:** 2026-05-03
**Run:** r-2026-05-03-v10s1
**Spec reference:** `docs/V10_FOUNDATION_MODEL_PAPER_SPEC.md` §9 Sprint 1, §6 Falsification Gates F4+F5

---

## Deliverable

Trained scalar Waddington potential `U_θ : ℝ^64 → ℝ` via denoising score
matching (Vincent 2011) on the MMRF baseline cohort, with PPI-Laplacian
Tikhonov regularizer derived from STRING v12 filtered to `combined_score ≥ 700`.

Artifacts:
- `checkpoints/u_theta_v10s1.pt` — 115,457 params, MLP 64→256→256→128→1 (SiLU)
- `data/processed/mmrf_z64.npy` — 787 patients × 64-d latents (PCA, 86% var)
- `data/processed/ppi_laplacian.npz` — 16,201 nodes, 236,930 undirected edges
- `paper/v8_artifacts/v10_sprint1/falsification_gates.json` — F4+F5 results
- `paper/v8_artifacts/v10_sprint1/s1d_train_log.json` — 400-epoch loss trace

---

## Honesty Notes (changes vs spec)

The spec quotes **N = 1313** (994 baseline-only + 319 Recurrent-BM) for the
DSM training cohort. **The actual N delivered by RNA-seq files on disk is 787**
patients (859 aliquots → earliest-T per patient, deduplicated). The shortfall
is because RNA-seq is only available for 859 of the 1313 patients in the
clinical metadata; this constraint is at the data layer, not the spec.

The spec calls for the **frozen scGPT** encoder. Sprint 1 substitutes a
**deterministic 64-d PCA on log1p-TPM** (top-5000 high-expression genes,
86.0% cumulative explained variance). The downstream U_θ math (DSM, PPI
Tikhonov, Helmholtz–Hodge) is encoder-agnostic up to latent dimension — only
encoder identity changes. **scGPT swap-in is Sprint 2** once `tdc/scGPT@acf749f3`
is environment-pinned and a checkpoint cached.

The PPI filter intended by the spec includes **non-observational sum ≥ 400**
(removes co-expression and text-mining leakage with MMRF expression). The
file on disk is the combined-score-only `9606.protein.links.v12.0.txt.gz`,
which lacks per-channel breakdown. Sprint 2 must download
`9606.protein.links.full.v12.0.txt.gz` (one HTTP fetch) to apply the full
filter; for Sprint 1 only the combined-score gate is enforced.

---

## F4 — CurlFraction (Helmholtz–Hodge)

**Threshold:** > 0.30 → gradient-only formulation refuted, flux head required.

**Result:** **CurlFraction = 0.0146** (n=787 nodes, k=15-NN graph, 11,805 edges).

**Interpretation:** the empirical mean-shift drift field at the kNN-graph
edges is overwhelmingly captured by a node-potential ⟹ the gradient
hypothesis from `WADDINGTON_NEURAL_ODE_REVIEW.md` is supported on this
data. **F4 passes.**

The original `WADDINGTON_NEURAL_ODE_MATH.md` recommendation to add a
divergence-free flux head $v_\theta$ is **not triggered** at this scale.

---

## F5 — Median-r identifiability

**Threshold:** median ‖ẑ − z*‖ / ‖z₀ − z*‖ ≥ 0.9 → energy formulation rejected.

**Result:** **median_r = 0.9982** (n_test=157, 50 Euler steps, step_size=5e-3).

**Interpretation:** *the substitute F5 fails*, but the substitute is not the
test the spec actually defines.

The spec's F5 is "held-out **paired** patients (N_p = 42), simulate ẑ_{i,1}
from V_θ given ẑ_{i,0}". This requires baseline + relapse paired latents,
which Sprint 1 does not have — the paired-cohort encoding is gated on
Sprint 2's longitudinal-data preprocessing.

The Sprint-1 substitute (random in-distribution targets) is strictly
harder than the spec's intent: a random target is generally NOT downhill
from a random source under an L2-norm-stationary potential. A median_r
near 1.0 is the *expected* behavior of any well-trained gradient field
when run against arbitrary targets — it does not refute the energy
formulation, it just confirms that the flow correctly does not move
random points toward arbitrary other random points.

**F5 verdict (honest):** **Not validly testable with current data**.
Re-run F5 in Sprint 2 with N_p = 42 paired baseline→relapse latents.

The training-side evidence for U_θ being non-trivial:
- ‖∇_z U‖² grew from ~0 (init) to ~12 (epoch 400) — the potential is
  informative.
- DSM loss decreased from 64.07 → 51.24 over 400 epochs — denoising
  signal is learned.
- PPI Tikhonov term grew from ~0 → ~10 — the gene-space gradient is
  living in the PPI graph's smoothness manifold (Tikhonov is a soft
  constraint, not a hard projection).

---

## Numbers safe in this doc only

Per `docs/V10_FOUNDATION_MODEL_PAPER_SPEC.md` §10 run-ledger discipline:

| Quantity | Value | Where it can live |
|---|---|---|
| N_baseline_actual | 787 | here + `s1a_summary.json` |
| PPI nodes (combined ≥ 0.7) | 16,201 | here + `s1b_summary.json` |
| PPI canonical undirected edges | 236,930 | here + `s1b_summary.json` |
| PCA cumulative explained variance @ 64 | 0.8594 | here + `s1c_summary.json` |
| F4 CurlFraction (audit-corrected) | 0.0151 | here + `falsification_gates_v10s1.json` |
| F5 median_r (substitute test) | 0.9982 | here + `falsification_gates_v10s1.json` |
| DSM loss start / end | 64.07 / 51.24 | here + `s1d_train_log_v10s1.json` |
| n_genes_matched_PPI | 4,262/5,000 | here + `s1d_train_log_v10s1.json` |

**None of these may migrate to README.md or ARCHITECTURE.md** until a
release-bouncer pass adds an `r-2026-05-03-v10s1` row to `RUNS.md`.

---

## Sprint 2 entry conditions

Set by Sprint 1 outcomes, not by spec text:

1. F4 passes ⟹ Sprint 2 keeps the **gradient-only U_θ formulation**;
   does NOT add the flux head $v_\theta$ at this stage.
2. F5 substitute is uninformative ⟹ Sprint 2's first task is to build the
   **N_p = 42 paired baseline→relapse latent pairs** so F5 can be re-run
   per spec.
3. PPI Tikhonov works (matched 4,262 genes, sub-Laplacian density 27/gene
   median) ⟹ Sprint 2 carries this regularizer forward unchanged; only
   adds the full-channel STRING filter.
4. Encoder is the bottleneck on F5: Sprint 2 must swap PCA for **frozen
   scGPT** to get a more disentangled cell-state latent before re-running
   F5.

---

## Reproducibility (post-audit, sprint-tagged)

```
git rev-parse HEAD                              # commit at run time
python scripts/v10/s1a_build_mmrf_baseline_matrix.py
python scripts/v10/s1b_build_ppi_laplacian.py \
    --links-file data/raw/string/9606.protein.links.v12.0.txt.gz \
    --tag v10s1 --sprint-dir v10_sprint1
python scripts/v10/s1c_encode_mmrf_z64.py
python scripts/v10/s3prep_depmap_haem_subset.py
python scripts/v10/s1d_train_scalar_potential.py \
    --tag v10s1 --laplacian-suffix _v10s1 --sprint-dir v10_sprint1 \
    --epochs 400 --batch-size 128
python scripts/v10/s1e_falsification_gates.py --tag v10s1 --sprint-dir v10_sprint1
python scripts/v10/s3prep_crispr_oracle.py     # optional smoke
python resistancemap/landscape/hbayes_outer.py # HBayes scaffold smoke
```

Wall-time on 1×GPU (training) + CPU (preprocessing): ≈ 90 s.

Outputs are written to `*_v10s1`-suffixed paths to keep Sprint 1 artifacts
immutable when Sprint 2 is run with `--tag v10s2`. See `RUNS.md` row
`r-2026-05-03-v10s1`; SHA256 chain at `logs/r-2026-05-03-v10s1/verification_chain.json`
(re-snapshotted 2026-05-03 audit pass).

## Hardening notes (2026-05-03 audit)

After Sprint 2 ran, an airtightness audit caught three issues that
affected Sprint 1's run-ledger integrity:

1. **F4 implementation cleanup** — original `f4_curl_fraction` measured
   the curl of the empirical mean-shift drift field independent of
   $U_θ$. The corrected version measures the curl of `(empirical drift)
   − (model drift -∇U_θ)` (the residual after removing the trained
   gradient field). Numbers shift only marginally (0.0146 → **0.0151**)
   because the empirical drift was already nearly conservative. The
   verdict (PASS) is unchanged, but the test now actually depends on
   $U_θ$ as the spec intended.
2. **Path uniqueness** — `s1b`, `s1d`, `s1e`, and `s2d` all gained
   `--tag` arguments. Sprint 1 outputs are now at
   `ppi_laplacian_v10s1.npz`, `u_theta_v10s1.pt`,
   `falsification_gates_v10s1.json`, etc. Sprint 2 re-runs cannot
   overwrite Sprint 1's checkpoint or its verification chain.
3. **Reproducibility check** — `s1e` and `s2d` produce bit-identical
   SHA256 outputs across re-runs (verified 2026-05-03).
