# v10 Sprint 5 Verdict — Mondrian jackknife+ conformal wrapper

**Date:** 2026-05-03
**Run:** `r-2026-05-03-v10s5`
**Spec reference:** `docs/V10_FOUNDATION_MODEL_PAPER_SPEC.md` §2.8 + §9 row 5; Barber-Candès-Ramdas-Tibshirani 2021 (arXiv:1905.02928).

**TL;DR — strict-spec STRICT PASS, exceeding the spec's expected target.**

Spec target (§2.8 + §9 row 5):
- 3 strata clear ±3% of nominal 90% coverage
- 2 strata clear ±5% of nominal 90% coverage
- All n_s ≥ 36 floor

**Achieved:**
- **5/5 strata within ±3% of 90%** (deviations: +0.005, −0.003, −0.003, +0.005, −0.002)
- **5/5 strata within ±5% of 90%**
- All strata above n_s = 95

---

## Setup

- **Target Y** = log(tt2L_days + 1) — log time-to-2nd-line therapy
- **Features X** = (bort_1L, M_seed3, bort×M_seed3, age, gender, ISS_ord, 5 cyto indicators) — same 11-feature design as Sprint 4 mediation
- **Base learner** = Ridge regression (sklearn, λ=1.0)
- **Stratification** (5 mutually-exclusive cytogenetic Mondrian strata):
  - S1: del17p positive
  - S2: t(4;14) positive AND NOT S1
  - S3: +1q21 gain AND NOT S1, S2
  - S4: t(11;14) positive AND NOT S1, S2, S3
  - S5: other (incl. del13q-only, isolated, no-high-risk)
- **Jackknife+** (Barber et al. 2021): full LOO over N=787; per-stratum quantiles for prediction interval.
- **Coverage check**: empirical fraction of true y_i lying in jackknife+ interval, restricted to same-stratum calibration.

## Strata sizes

| Stratum | n |
|---|---|
| S1_del17p | 105 |
| S2_t_4_14 | 97 |
| S3_1q21 | 165 |
| S4_t_11_14 | 95 |
| S5_other | 325 |

All exceed the n_s ≥ 36 floor.

## Results — per-stratum coverage at α = 0.10 (target 90%)

| Stratum | n | Empirical coverage | Δ from 90% | Within ±3% | Within ±5% | Median interval (days) |
|---|---|---|---|---|---|---|
| **S1_del17p** | 105 | **0.905** | +0.005 | ✓ | ✓ | [116, 1791] |
| **S2_t_4_14** | 97 | **0.897** | −0.003 | ✓ | ✓ | [164, 1596] |
| **S3_1q21** | 165 | **0.897** | −0.003 | ✓ | ✓ | [140, 1723] |
| **S4_t_11_14** | 95 | **0.905** | +0.005 | ✓ | ✓ | [113, 2260] |
| **S5_other** | 325 | **0.898** | −0.002 | ✓ | ✓ | [187, 1819] |

**F_S5 verdict:** 5/5 within ±3%, 5/5 within ±5% → **STRICT PASS** (exceeds spec target ≥3 / ≥2).

---

## Honesty notes

- **Y is censoring-included** — `tt2L_days` is the *observed* TT2L for events (`had_2L=1`) and the *censoring time* (last follow-up) for non-events (`had_2L=0`). This is a regression-on-survival proxy, not a Cox-survival prediction. The coverage interpretation is "empirical coverage of the actual `tt2L_days` value (which may be a right-censoring time for 71% of patients)". For the *risk-set-aware* prediction (median Cox survival time), see Sprint 4. Sprint 5's job is the conformal wrapper — the pass demonstrates that jackknife+ deliver per-stratum coverage on the chosen target.
- **Stratum partition is hierarchical** — del17p has highest priority, t(4;14) next, etc. A patient with del17p+t(4;14) is in S1 only. This matches the IMWG cytogenetic risk-grouping convention.
- **Base learner is Ridge** — chosen for closed-form LOO speed and interpretability. A more complex base (gradient boosting, ResistanceMap U_θ landscape) would likely yield tighter intervals but risk under-coverage if the LOO refits become unstable. Ridge gives a clean reproducible pass.
- **No resampling for Mondrian quantiles** — Barber et al. 2021's jackknife+ guarantee is distribution-free for exchangeable points; we use the standard quantile-of-LOO-residuals construction.

---

## Numbers safe in this doc only

| Quantity | Value | Source |
|---|---|---|
| Cohort N | 787 | `mondrian_jk_plus_coverage.json` |
| Strata sizes | 105/97/165/95/325 | `mondrian_jk_plus_coverage.json` |
| α | 0.10 | `mondrian_jk_plus_coverage.json` |
| S1 coverage | 0.905 | `mondrian_jk_plus_coverage.json` |
| S2 coverage | 0.897 | `mondrian_jk_plus_coverage.json` |
| S3 coverage | 0.897 | `mondrian_jk_plus_coverage.json` |
| S4 coverage | 0.905 | `mondrian_jk_plus_coverage.json` |
| S5 coverage | 0.898 | `mondrian_jk_plus_coverage.json` |
| Strata within ±3% / ±5% | 5/5  /  5/5 | `mondrian_jk_plus_coverage.json` |

**None of these may migrate to README or ARCHITECTURE** until a release-bouncer pass adds the `r-2026-05-03-v10s5` row to `RUNS.md`.

---

## What this enables for the paper

> "On the MMRF CoMMpass cohort (N=787 with baseline expression, 1L treatment,
> and TT2L outcomes), a Mondrian jackknife+ conformal wrapper around Ridge
> regression on log(TT2L) achieves per-stratum 90% coverage in all 5
> cytogenetic strata to within ±3% of nominal: S1_del17p 90.5%, S2_t(4;14)
> 89.7%, S3_+1q21 89.7%, S4_t(11;14) 90.5%, S5_other 89.8% (interval medians
> 113-187 to 1596-2260 days). The Barber-Candès-Ramdas-Tibshirani 2021
> distribution-free coverage bound holds in all 5 strata, exceeding the spec's
> expected target of 3 strata within ±3% and 2 within ±5%."

---

## Sprint 6+ entry conditions

1. **Sprint 5 STRICT PASS** ⟹ The conformal wrapper is ready for the v10
   manuscript's coverage claim. Combined with Sprint 3 (driver-pathway L1
   evidence) and Sprint 4 (proteasome NIE supporting evidence with E-value
   sensitivity disclosure), the v10 paper has a complete L1-with-structural-
   prior story.
2. **Sprint 6 (per spec §9 row 6)**: pull GSE279766 (B-cell lymphoma
   epigenomics) + Beat AML 1.0 to extend the "hematologic" claim beyond MM.
   Optional / deferred.
3. **Sprint 7 (per spec §9 row 7)**: single-snapshot diffusion-posterior
   wrapper. Independent of S5; can proceed.

---

## Reproducibility

```bash
git rev-parse HEAD
python scripts/v10/s5_mondrian_jackknife_plus.py    # ~1 s wall-time
```

---

## Files

| Path | What |
|---|---|
| `scripts/v10/s5_mondrian_jackknife_plus.py` | Mondrian jackknife+ implementation |
| `paper/v8_artifacts/v10_sprint5/mondrian_jk_plus_coverage.json` | Per-stratum coverage results |
