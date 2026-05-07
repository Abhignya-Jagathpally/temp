# Wave 3 — Parallelism Patches (W3.2)

**Author:** general-purpose agent (worktree-isolated; persisted by main thread)
**Date:** 2026-05-07
**Source files inspected (read-only):** `scripts/v10/{s5_mondrian_jackknife_plus.py, s4d_nie_estimation.py, s3fix_v3.py}` and `resistancemap/landscape/rwr_propagation.py`.

## §1 Per-target diff summary

**Target 1 — `s5_mondrian_jackknife_plus.py` (LOO Ridge, N≈787)**
LOO loop in `jackknife_plus_intervals()` (lines 105–115) → `joblib.Parallel(n_jobs=N_JOBS, backend="loky")`. Worker `_fit_loo_one` computes `(r_i, pred_test_i)` from a fresh Ridge fit. **No RNG involved** (Ridge is closed-form), so bit-identicality is automatic regardless of order. `N_JOBS = int(os.environ.get("RM_N_JOBS", "-1"))`; `RM_N_JOBS=1` → byte-identical to original.

**Target 2 — `s4d_nie_estimation.py` (B=1000 paired Cox bootstrap)**
In `bootstrap_decomposition()` (lines 124–148): pre-generate `idx_list = [torch.randint(0, n, (n,), generator=g).numpy() for _ in range(B)]` BEFORE dispatching. Same single `torch.Generator(seed=rng_seed)` consumed in same order as serial → bootstrap index sets are bit-identical. Workers receive `idx_list[b]` and compute `(NDE, NIE, TE, beta_AM, beta_M, ok)` via `fit_cox_with_interaction` + `decompose_effects`. Failures (Cox non-convergence) signaled by `ok=False`.

**Target 3 — `s3fix_v3.py` (B=10000 perm test)**
In `f6_v3()` (lines 173–194): pre-generate the FULL random index matrix shape `(B, n_drv, n_drv-1)` upfront with the same single `rng` in the same nested order `for b: for j:`, then dispatch each `b` to a worker that just looks up rank-means. **No RNG inside workers.** Memory cost: `B*n_drv*(n_drv-1)*8 = 10000*45*44*8 ≈ 158 MB` extra int64 indices — modest on a 32 GB host.

## §2 Bit-identicality (analytical)

| Target | Why np.allclose passes at rtol=1e-10 |
|---|---|
| s5 | Each `Ridge.fit` is a closed-form solve `(XᵀX + αI)β = Xᵀy`. No RNG, no thread non-determinism on a single matmul. Outputs written by index → bit-identical. |
| s4d | Pre-generating `idx_list` with same `torch.Generator(seed=rng_seed)` consumes RNG in same order as serial loop's `torch.randint(...)`. `idx_list[b]` byte-equal to serial `idx` at iteration b. Inner functions deterministic in `idx`. |
| s3fix_v3 | Same pre-gen invariant: `sample_degree_matched(... rng ...)` called in identical (b, j) nesting order with same `rng`, so `indices_per_b[b][j]` equals serial `random_idx`. `_perm_one` deterministic in indices. |

The verifier's Target-2 check empirically demonstrates the RNG-consumption invariant on (n=200, B=5, seed=20260503). Targets 1 and 3 are checked end-to-end on a 5-iteration subset with `np.allclose(rtol=1e-10, atol=1e-12)`.

## §3 Wall-time speedup (expected on 16-core / 32 GB host)

| Target | Serial baseline | Parallel target | Expected speedup |
|---|---|---|---|
| s5 LOO (N=787) | ~30–60 s (787 Ridge fits at p≈11) | n_jobs=16 | **12–24×** |
| s4d Cox bootstrap (B=1000) | ~600–1200 s (lifelines Cox PH at N≈775) | n_jobs=16 | **8–16×** |
| s3fix_v3 perm (B=10000, n_drv=45) | ~400–800 s | n_jobs=16 | **10–20×** |

Verifier's "full-N(80)" and "B=200" probes give a quick on-host indicator (typically 4–8× on a small problem due to fixed loky overhead).

## §4 Risks

1. **loky pickling** — all inputs picklable. Workers do not pass CSR sparse matrices.
2. **Daemon-worker constraints** — workers don't themselves spawn `Parallel`. None of the three targets does.
3. **BLAS oversubscription** — set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` before `v10_runner.py`.
4. **Memory** — s3fix_v3 ≈ 158 MB pre-gen; s4d ≈ 6 MB; s5 nothing extra.
5. **Determinism on multi-thread BLAS** — with `OMP_NUM_THREADS=1` set, Ridge / Cox solves are bit-deterministic.
6. **s4d failure ordering** — failed Cox fits filtered as in original. Order of accepted bootstrap samples differs (parallel returns gathered in submission order), but `np.percentile` is order-invariant → all summary stats bit-identical.
7. **`v10_runner.py` does NOT need modification** — each script reads `RM_N_JOBS` from env.

## §5 How to apply

```bash
# From repo root (/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap)

# Verify bit-identicality + speedup BEFORE applying:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python scripts/v12_refactor/wave3/parallelism_patches/verify_parallelism.py

# Dry-run apply (no file changes, exits non-zero on conflict):
git apply --check scripts/v12_refactor/wave3/parallelism_patches/s5_mondrian.patch
git apply --check scripts/v12_refactor/wave3/parallelism_patches/s4d_nie.patch
git apply --check scripts/v12_refactor/wave3/parallelism_patches/s3fix_v3.patch

# Real apply (only if all three --check pass):
git apply scripts/v12_refactor/wave3/parallelism_patches/*.patch

# Run with byte-identical serial behaviour (sanity gate):
RM_N_JOBS=1 OMP_NUM_THREADS=1 python scripts/v10/s5_mondrian_jackknife_plus.py

# Run with full parallelism for production:
RM_N_JOBS=-1 OMP_NUM_THREADS=1 python -m resistancemap.v10_runner

# Roll back if anything regresses:
git checkout -- scripts/v10/s5_mondrian_jackknife_plus.py \
                scripts/v10/s4d_nie_estimation.py \
                scripts/v10/s3fix_v3.py
```

The verifier was not executed in the agent's sandbox — please run it before applying the patches.
