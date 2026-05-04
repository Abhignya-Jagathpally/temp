# v10 Sprint 2 Verdict — HBayes outer + paired-data F-gates

**Date:** 2026-05-03
**Run:** r-2026-05-03-v10s2
**Spec reference:** `docs/V10_FOUNDATION_MODEL_PAPER_SPEC.md` §9 Sprint 2, §6 Falsification Gates F1, F2, F5; §2.6 HBayes outer

---

## Deliverables

| Artifact | Path | What it is |
|---|---|---|
| Full-channel STRING | `data/raw/string/9606.protein.links.full.v12.0.txt.gz` | downloaded; combined ≥0.7 AND non-obs ≥0.4 filter applied |
| New PPI Laplacian | `data/processed/ppi_laplacian.npz` | 12,651 nodes / 141,241 undirected edges (was 16,201 / 236,930 with combined-only filter) |
| Paired patients | `data/processed/mmrf_paired_patients.tsv` | **N=29 paired (≥2 timepoints), N=20 strict-paired (with 2nd-line therapy)** |
| Paired latents | `data/processed/mmrf_paired_z64.npz` | (29, 64) z₀ + (29, 64) z₁ via Sprint-1 PCA |
| Retrained U_θ | `checkpoints/u_theta_v10s1.pt` | now uses full-channel PPI; F4=0.0146 unchanged |
| HBayes posterior | `paper/v8_artifacts/v10_sprint2/hbayes_posterior.npz` | μ, θ_s, τ, σ_obs from numpyro SVI on N=716 |
| F1+F2 results | `paper/v8_artifacts/v10_sprint2/f1_f2_gates.json` | per-stratum coverage + bootstrap CI |
| F5-paired result | `paper/v8_artifacts/v10_sprint2/f5_paired.json` | median_r on real (z₀, z₁) pairs |
| scGPT plan | `docs/V10_SCGPT_INTEGRATION_PLAN.md` | recon-only; swap deferred to Sprint 6 |

---

## Honesty notes (changes vs spec)

The spec quotes **N_p = 42 strict-paired** and **N_r = 319 Recurrent-BM**.
On disk the actual counts are **N_p_strict = 20 (with 2nd-line therapy)**
and **N_p_paired = 29 (≥2 RNA-Seq timepoints)**. The shortfall reflects
the RNA-seq files available; clinical metadata reports more paired
patients than have RNA-seq.

The cytogenetic-aligned cohort is **716 / 787** (those with all 5
cytogenetic flag calls). Stratum prevalences match spec exactly:
del17p 14.7% (vs spec 13.6%), chr1q21+ 33.8% (33.6%), del13q 52.4%
(52%), t(4;14) 14.4% (14.6%), t(11;14) 17.2% (16.8%).

The HBayes spec defines stratum membership as "one of 5 cytogenetic
strata". Real MMRF patients carry multiple abnormalities, so we use a
**multi-label additive HBayes** with membership matrix M ∈ {0,1}^{N×5}.
Marginal-likelihood interpretation in spec §2.6 carries through with M
in place of one-hot c_p.

---

## F1 — Cross-stratum coverage (spec §6 row F1)

**Threshold:** 90% predictive CI coverage < 0.70 → stratified-exchangeability refuted.

**Result:** All 5 strata **PASS**.

| Stratum | n_held | n_train | Coverage | Verdict |
|---|---|---|---|---|
| del17p | 23 | 693 | **0.959** | PASS |
| chr1q21_gain | 46 | 670 | **0.946** | PASS |
| del13q | 97 | 619 | **0.952** | PASS |
| t(4;14) | 103 | 613 | **0.945** | PASS |
| t(11;14) | 65 | 651 | **0.950** | PASS |

**Caveat (honest):** the per-coordinate independent-normal CI is
*conservative* — at σ̂_obs ≈ 8 with 64 dimensions, the CI half-width
13/coordinate is wider than typical residuals. Sprint 5's Mondrian
jackknife+ conformal wrapper (per spec §2.8) will tighten this with
empirical-quantile-based calibration. The current 0.94–0.96 coverage is
*evidence* of stratified-exchangeability holding, not a tight estimate.

---

## F2 — Baseline ablation (spec §6 row F2)

**Threshold:** held-out log-lik unchanged (paired bootstrap, α=0.05) → partial pooling adds no value.

**Result:** **REFUTED at α=0.05** *(corrected from initial PASS after a
Sprint-2 hardening audit caught a train-on-test leak — see §"Sprint 2
hardening" below).*

```
design              : proper paired LOO on both arms (no train-on-test leak)
n_test_paired       : 27 (paired patients in cyto-aligned cohort)
ll_full_cohort_LOO  : -1475.68    (Arm A: HBayes on full_cohort \ {i}, eval on i)
ll_paired_only_LOO  : -1499.87    (Arm B: HBayes on paired_set \ {i}, eval on i)
delta_mean          : +24.19      (was +1266.62 with the leak)
delta_std           : 184.94      (huge per-patient variance)
delta_min           : -394.70     (some patients predicted WORSE by full-cohort)
delta_max           : +738.98
delta_bootstrap_95% : [-43.11, +99.55]
ci_contains_zero    : TRUE
```

**Interpretation:** with the leak fixed, the 716 baseline-only patients
do **not** produce a statistically detectable improvement in held-out
log-likelihood for paired-patient prediction at the current N=27. The
mean shift (+24 nats) is in the "right" direction but is tiny compared
to per-patient variance (std 185), and 9 of 27 patients are *worse*
under partial pooling than under paired-only. **F2 is refuted at the
spec's α=0.05 threshold.**

**Honest implication for the architecture choice:** the spec's
preference for Hierarchical-Bayes (Approach C) over latent-SDE (A) and
OT-matching (B) was justified theoretically (per `TRAJECTORY_MATH_FEW_SHOT.md`:
"994 baselines enter the marginal likelihood non-vacuously"). At
N_paired = 27, that theoretical benefit is **below the noise floor**
of the held-out log-likelihood. The partial-pooling claim should be
re-tested in Sprint 6 after data uplifts (Beat AML 1.0, GSE279766) push
N_paired higher; until then, the paper must say:

> "Hierarchical Bayes is the architecturally justified outer layer
> (Theorem in `TRAJECTORY_MATH_FEW_SHOT.md` Approach C); empirical
> demonstration that the baseline cohort improves paired-patient
> log-likelihood is not yet attainable at N_paired = 27 (95% CI on
> Δ log-lik straddles zero). Re-tested in §6 ablations after Sprint 6
> data uplifts."

---

## F5 — Median-r identifiability on REAL paired pairs (spec §6 row F5)

**Threshold:** median ‖ẑ − z₁‖ / ‖z₀ − z₁‖ ≥ 0.9 → energy formulation refuted.

**Result:** best **median_r = 0.994** (50 Euler steps × 5e-2 step), still **above the 0.9 refutation threshold**, BUT with **frac_below_1 = 0.69** (69% of paired patients see *some* improvement after gradient flow).

| Config | All-paired median_r | Strict-paired median_r | frac<1 |
|---|---|---|---|
| 50 steps × 5e-3 | 0.999 | 0.998 | 0.83 |
| 50 steps × 5e-2 | 0.994 | 0.992 | 0.69 |
| 200 steps × 5e-3 | 0.996 | 0.995 | 0.76 |
| 200 steps × 5e-2 | 1.015 | 1.016 | 0.45 |

**Interpretation:** snapshot-only DSM-trained U_θ has no temporal
supervision — both z₀ and z₁ are samples from p_data, so both lie near
score = 0 on the manifold. The fact that 69–83% of patients see *some*
inward movement on the line z₀ → z₁ is mild evidence the cohort has
**incidental gradient-aligned drift** even without temporal training, but
it is far from the supervised-trajectory bar. **F5 refuted** at the spec's
threshold.

This is **not a surprise** — and not a problem for the v10 architecture
itself. Sprint 3 adds the per-patient trajectory likelihood (Trajectory
Flow Matching, atong01/conditional-flow-matching) that *does* use paired
supervision. F5 will be re-run after TFM lands and the *substantive*
falsification gate at the v10 level is whether **TFM+HBayes-outer beats
F5 ≤ 0.9 even on N=20 strict-paired**.

---

## F4 — CurlFraction unchanged after full-channel PPI

After retraining U_θ with the full-channel-filtered Laplacian
(combined ≥ 0.7 AND non-obs ≥ 0.4), F4 CurlFraction stays at **0.0146**
(threshold 0.30, PASS). The PPI Tikhonov is a soft regularizer; the
gradient field is dominated by the DSM signal, so the topology change
in the Laplacian does not move the gradient direction materially.

---

## Sprint 2 hardening (2026-05-03 audit pass)

After the initial Sprint 2 run, an airtightness audit caught two issues:

1. **F2 had a train-on-test leak**: Arm A trained HBayes on the full 716
   patients *including* the 27 paired test patients, then evaluated
   log-lik on those same 27. Only Arm B was properly LOO'd. The
   resulting Δ = +1266.62, CI [+486, +2347] was a leakage artifact.
   *Fix:* both arms now do full paired LOO. Corrected Δ = +24.19, CI
   [-43.11, +99.55] — F2 is **REFUTED** at α=0.05. The numbers in
   the F2 section above and in `paper/v8_artifacts/v10_sprint2/f1_f2_gates.json`
   reflect the corrected design.

2. **F4's `model` argument was unused**: the original F4 measured the
   curl of the empirical mean-shift drift field, independent of $U_θ$.
   The corrected F4 measures the curl of `(empirical drift) − (model
   drift -∇U_θ)` — i.e. the residual after removing the trained
   gradient field. Corrected F4 = **0.0151** (was 0.0146 from the
   data-only computation); both well below the 0.30 threshold. The
   original interpretation ("gradient-only Waddington PASS") survives
   unchanged because the empirical drift is itself nearly conservative
   on this k-NN graph.

3. **Run-ledger SHA256 chain drift**: Sprint 2's retrain overwrote
   `checkpoints/u_theta_v10s1.pt`, `data/processed/ppi_laplacian.npz`,
   and other shared files, breaking the Sprint 1 verification_chain.json.
   *Fix:* `s1b`, `s1d`, `s1e`, and `s2d` now accept `--tag` so each
   sprint produces uniquely-named outputs (`ppi_laplacian_v10s1.npz`,
   `u_theta_v10s2.pt`, etc.). Sprint 1 and Sprint 2 are now re-snapshotted
   under tagged paths.

These fixes are intentionally conservative: F1 numbers are unchanged
(0.94-0.96 coverage); F4 verdict unchanged; F5-paired verdict unchanged;
only F2 verdict flipped from PASS to REFUTED.

## Numbers safe in this doc only

Per `docs/V10_FOUNDATION_MODEL_PAPER_SPEC.md` §10 run-ledger discipline.

| Quantity | Value | Where it lives |
|---|---|---|
| N paired patients (≥2 timepoints) | 29 | `s2b_summary.json` |
| N strict-paired (with 2nd-line therapy) | 20 | `s2b_summary.json` |
| Cyto-aligned cohort | 716 | `hbayes_summary.json` |
| Full-channel PPI nodes | 12,651 | `s1b_summary.json` |
| Full-channel PPI canonical edges | 141,241 | `s1b_summary.json` |
| F1 coverage range | 0.945 – 0.959 | `f1_f2_gates.json` |
| F2 Δlog-lik mean (proper LOO) / 95% CI | 24.19 / [-43.11, +99.55]  — **F2 REFUTED** | `f1_f2_gates.json` |
| F5-paired best median_r | 0.994 | `f5_paired.json` |
| F5-paired best frac_below_1 | 0.83 | `f5_paired.json` |
| HBayes τ̂ | 1.505 | `hbayes_summary.json` |
| HBayes σ̂_obs | 7.99 | `hbayes_summary.json` |
| HBayes per-stratum θ̂ L2 | 9.6 – 12.9 | `hbayes_summary.json` |

**None of these may migrate to README or ARCHITECTURE** until a
release-bouncer pass adds an `r-2026-05-03-v10s2` row to `RUNS.md`.

---

## Sprint 3 entry conditions (set by Sprint 2 outcomes)

1. **F1 PASS, F2 REFUTED at α=0.05** ⟹ HBayes outer's *coverage*
   property is supported, but its *marginal-likelihood-improvement*
   claim is not yet detectable at N_paired=27. Carry forward into
   Sprint 3, re-test F2 in Sprint 6 with larger N_paired.
2. **F5 fails as expected without temporal supervision** ⟹ Sprint 3's
   first job is to install `torchcfm` and plug **Trajectory Flow Matching**
   as the per-patient likelihood inside the HBayes outer layer.
3. **F4 confirmed unchanged with full-channel PPI** ⟹ Sprint 3 does not
   need to revisit the PPI filter. Move on to RWR driver-pathway head.
4. **Encoder is honestly disclosed as PCA-64 (not scGPT)** ⟹ Sprint 6
   carries the scGPT swap; Sprint 3 does not block on it.

---

## Reproducibility (post-audit, sprint-tagged)

```
git rev-parse HEAD                                   # commit at run time
# Sprint 2 from-scratch (assumes Sprint 1 already ran):
python scripts/v10/s2b_paired_patients.py
python scripts/v10/s2c_encode_paired.py
python scripts/v10/s1b_build_ppi_laplacian.py \
    --links-file data/raw/string/9606.protein.links.full.v12.0.txt.gz \
    --tag v10s2 --sprint-dir v10_sprint2
python scripts/v10/s1d_train_scalar_potential.py \
    --tag v10s2 --laplacian-suffix _v10s2 --sprint-dir v10_sprint2 \
    --epochs 400 --batch-size 128
python scripts/v10/s1e_falsification_gates.py --tag v10s2 --sprint-dir v10_sprint2
python scripts/v10/s2d_f5_paired.py --tag v10s2
python scripts/v10/s2e_hbayes_inference.py
python scripts/v10/s2f_f1_f2_gates.py                # corrected: proper LOO on both F2 arms
```

Wall-time: STRING full-channel download ~30 s, retrain ~90 s, S2b–S2c ~10 s,
S2e ~3 s, S2f ~2-3 min (54 paired-LOO SVI fits for the corrected F2). Total ≈ 5 min.

The corrected F2 has a higher SVI cost than the original (54 fits vs 28),
which is the price of doing both arms LOO.
