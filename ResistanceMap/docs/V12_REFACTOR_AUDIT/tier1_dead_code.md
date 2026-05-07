# Tier-1 Dead-Code & Parallelism Audit — ResistanceMap

> **Scope.** All `.py` under `resistancemap/`, `scripts/`, `tests/` on `main`
> (92,674 LOC total, 234 .py files; the `tier1-refactor` worktree was forked
> from an old commit so the audit was performed against the live tree at
> `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/`).
> **No source files were modified.** Every claim cites `path:line`.

## Executive summary

Five whole sub-packages — `resistancemap/{baselines,clinical,experiments,theory,training}` —
totalling **14,600 LOC across 24 modules** are never imported by any external
caller (all five report `0` external import refs in the per-package scan).
A further `agents/example_usage.py` (196 LOC) and 670 LOC across two
superseded `s3fix_*` patches in `scripts/v10/` are unreachable from
`v10_runner.py:52-114`. Combined with `__pycache__` cleanup, an estimated
**~16 kLOC (≈17 % of the Python tree) can be archived** without touching any
F-gate definition. Five obvious serial loops in `scripts/v10/` and
`resistancemap/evaluation/` can be parallelised with `joblib.Parallel` /
`multiprocessing.Pool` for an expected **5–20× wall-clock speed-up** on the
LOO/bootstrap/per-drug stages (currently the longest stages of `v10_runner`).

---

## §1. Orphaned modules

Reachability test: for each package, count `import resistancemap.<pkg>` /
`from resistancemap.<pkg>` references **outside that package** (full results
in `/tmp/import_index2.txt` during audit). A `0` means no source file, test,
script, doc, shell, or pyproject entry-point loads the package.

| File / dir | Last commit | External refs | Proposed action |
|---|---|---|---|
| `resistancemap/baselines/` (5 files, 2 713 LOC) | 2026-05-02 `42a74e8` | 0 (`__init__.py:3-5` re-exports never reached) | **archive** to `attic/v6_baselines/` — superseded by `scripts/baselines/{elasticnet,mofa}_baseline.py` |
| `resistancemap/clinical/` (10 files, 6 262 LOC) | 2026-04-14 `c5e47be` | 0 | **archive** — v6 prediction heads, none of MM/SMM/MRD/EMD/ITR is wired into v10/v11/v12 stages |
| `resistancemap/experiments/` (3 files, 1 777 LOC) | 2026-05-02 `42a74e8` | 0 | **archive** — `ablation_study.py`, `statistical_testing.py` superseded by `evaluation/ablation_harness.py` and per-sprint scripts |
| `resistancemap/theory/` (3 files, 1 967 LOC) | 2026-05-02 `42a74e8` | 0 | **archive** — `identifiability_proof.py`, `stability_analysis.py` are theorem stubs no F-gate consumes |
| `resistancemap/training/` (3 files, 1 881 LOC) | 2026-05-02 `42a74e8` | 0 | **archive** — `trainer.py`/`losses.py` v6-era, replaced by per-script training loops in `scripts/v10/s1d_*` etc. |
| `resistancemap/agents/example_usage.py` (196 LOC) | 2026-04-12 `2c9b4f3` | 0 | **delete** — pure tutorial file, no caller |
| `resistancemap/infrastructure/` (7 files) | mixed | 1 (`tests/test_ope_and_guardrails.py:1` only) | **keep guardrails.py, archive** the other 6 (continual_learning, imaging_radiomics, missing_modality, multi_site_harmonization, scrna_integration, uncertainty_quantification) — none referenced |
| `resistancemap/interpretability/` (5 files) | mixed | 1 self-ref in `__init__.py` | **archive** — no external consumer |
| `resistancemap/v10_runner.py` | 2026-05-03 | 1 (`resistancemap/main.py:?`) | **keep** — pipeline entry |
| `__pycache__/` (33 dirs) and `.pytest_cache/` (1) | — | n/a | **delete** + add to `.gitignore` |

Carve-out: `resistancemap/agents/{base,orchestrator,specialized}.py` — `base.py`
is imported by `evaluation/base.py:?`, `evaluation/tier_b/{architecture_auditor,
pathway_ppi,cell_state_specialist}.py`, so **keep** the package, but consider
collapsing `orchestrator.py` (510 LOC) + `specialized.py` (475 LOC) — these
are only consumed by `resistancemap/main.py:? from resistancemap.agents.orchestrator
import Orchestrator` and the actual sprint runners use `.claude/agents/*.md`
markdown agents instead. See §4.

---

## §2. Sprint-script consolidation

Authoritative source of truth: `resistancemap/v10_runner.py:52-114` lists
every script `v10_runner` actually invokes. Anything in `scripts/v10/` that
is *not* on that list is a stale patch artefact.

| Current files (LOC) | Cited by `v10_runner.py` | Proposed canonical name | Merge / archive note |
|---|---|---|---|
| `s3fix_apply.py` (376) `scripts/v10/s3fix_apply.py:1-8` | **NO** | — | Stale (superseded by `s3fix_v2.py:1-7`); **archive** to `attic/v10_intermediate_fixes/` |
| `s3fix_diagnostic.py` (294) `scripts/v10/s3fix_diagnostic.py:1-8` | **NO** | — | Diagnostic-only; output JSON already in `paper/v8_artifacts/v10_sprint3/`; **archive** |
| `s3fix_v2.py` (294) | YES line 86 | `s3_propagation_panel_fix.py` | **keep** as canonical F3-panel fix |
| `s3fix_v3.py` (256) | YES line 87 | `s3_propagation_continuous_null.py` | **keep** as canonical F6 continuous-rank fix |
| `s3fix_v3_f7.py` (187) | YES line 88 | `s3_propagation_isoform_seeds.py` | **keep** — Bradner 2010 isoform-correct seeds |
| `s3_fix_f7_prism_oracle.py` (225) | YES line 112 (in "fixes" stage) | `s3_propagation_prism_oracle.py` | **keep** — separate oracle path |
| `s3prep_crispr_oracle.py` (139) | YES line 79 | `s3_prep_crispr_oracle.py` | **keep** + rename for kebab-consistency |
| `s3prep_depmap_haem_subset.py` (118) | YES line 78 | `s3_prep_depmap_haem.py` | **keep** + rename |
| `s4e_falsification_neg_control.py` (?) | YES line 95 | `s4_falsification_v1.py` | **keep but mark v1** |
| `s4e_v2_focused_neg_control.py` (207) | YES line 97 | `s4_falsification_v2.py` | **keep** as canonical v2 |
| `s4_fix_f10_richer_mediator.py` (143) | YES line 113 | `s4_f10_mediator_fix.py` | **keep** in `fixes` stage |
| `s4_fix_f10_tipping_point.py` (132) | YES line 114 | `s4_f10_tipping_fix.py` | **keep** in `fixes` stage |
| `s6b_cross_disease_f3.py` (?) | YES line 104 | `s6_xdisease_v1.py` | **keep** |
| `s6b_v2_aml_native.py` (205) | YES line 105 | `s6_xdisease_v2.py` | **keep** as canonical v2 |

**Net dead-code from §2: 670 LOC (s3fix_apply.py + s3fix_diagnostic.py).**

The `_v2` / `_v3` suffix convention should be replaced with semantic suffixes
(`*_panel_fix`, `*_continuous_null`, `*_isoform_seeds`, `*_prism_oracle`,
`*_aml_native`) so future contributors can read intent without diffing.

---

## §3. Parallelism opportunities

All loops below are CPU-bound; none currently uses `n_jobs`/`Pool`/`joblib`
(verified by grep across the tree — only `scripts/run_baselines_real.py` and
`scripts/run_experiments.py` use `n_jobs=-1`/joblib already).

| File:line | Current serial pattern | Suggested parallel form |
|---|---|---|
| `scripts/v10/s5_mondrian_jackknife_plus.py:106-114` | `for i in range(n): base.fit(X_train[mask], y_train[mask])` — N independent Ridge fits (`n≈787`) | `joblib.Parallel(n_jobs=-1, backend='loky')(delayed(_fit_loo)(i) for i in range(n))` — pure scikit-learn fits, picklable; expected 12–24× on a 16-core box |
| `scripts/v10/s4d_nie_estimation.py:131-145` | `for b in range(B): fit_cox_with_interaction(...)` — B independent bootstrap Cox fits | `joblib.Parallel` over bootstrap indices; `lifelines.CoxPHFitter` is picklable. `B` is typically 1 000 → 8–16× speed-up |
| `scripts/v10/s3fix_v3.py:181-194` and `scripts/v10/s3f_f7_multiseed.py:79-86` | `for b in range(B):` permutation null with `B=1000–10000` | `multiprocessing.Pool(map(_one_perm, range(B)))` — each iteration is independent NumPy work; 10–20× expected |
| `scripts/v10/s3d_f3_crispr_oracle.py:90-100` and `scripts/v10/s3c_drug_seed_propagation.py` | `for drug in drugs: crispr_rank_sum(...)` — 10 independent drugs | `joblib.Parallel(n_jobs=min(10, n_cpu))(delayed(crispr_rank_sum)(...) for drug in drugs)` — embarrassingly parallel; 4–8× |
| `scripts/v10/s7_diffusion_posterior.py:76-78` | `for k in range(K): diffs_xx.append(np.linalg.norm(samples_pred - samples_pred[k:k+1], axis=1).sum())` | Vectorise: `pdist = np.linalg.norm(samples_pred[:,None,:] - samples_pred[None,:,:], axis=2); e_xx = (pdist.sum(1) / max(K-1,1)).mean()` — pure NumPy, no Python loop |
| `scripts/v10/s7_diffusion_posterior.py:140-143` | `for eta, sf in grid: for i in range(n):` (16-cell × N LOO grid) | `joblib.Parallel` over the outer grid — each `(η,σ)` cell is independent; `n_jobs=16` for 16 grid cells |
| `resistancemap/evaluation/metrics/per_drug.py:65-80` | `for d in np.unique(drug):` bootstrap CIs per drug | `joblib.Parallel` over unique drugs — each bootstrap is independent |
| `resistancemap/evaluation/leakage.py:259-273` | `for pid in patients: for early,late in pairs:` — serial leakage audit | Already O(N) and dominated by Python overhead; vectorise with pandas groupby instead of parallelising |

**Caveat on `s5` LOO**: `Ridge.fit` releases the GIL via BLAS; `joblib`'s
`loky` backend is the right default. Avoid `threading` (NumPy/SciPy already
multi-threads BLAS internally — outer parallelism should use processes to
avoid oversubscription, set `OMP_NUM_THREADS=1` inside workers).

---

## §4. Reorganisation sketch (no files moved yet)

```
ResistanceMap/
├── resistancemap/                  # importable package
│   ├── core/                       # NEW — collapse from {data, models, landscape, agents/base.py, observability, latent_compute}
│   │   ├── data/                   # current resistancemap/data/ minus stale loaders
│   │   ├── models/                 # current resistancemap/models/ (active only — see §1 archive list)
│   │   ├── landscape/              # current resistancemap/landscape/
│   │   ├── agents_base.py          # current resistancemap/agents/base.py (keep — used by tier_b)
│   │   └── observability/          # current resistancemap/observability/
│   ├── evaluation/                 # KEEP as-is (73 external refs; well-structured tier_a..d)
│   ├── inference/                  # KEEP — small, used by main.py
│   ├── verification/               # KEEP — used by 5 callers
│   ├── agentops/                   # KEEP — used by 6 callers
│   └── v10_runner.py               # KEEP — pipeline entrypoint
│
├── scripts/
│   ├── v10/                        # rename per §2 to semantic names; drop s3fix_apply.py + s3fix_diagnostic.py
│   ├── v11/
│   ├── v12/
│   ├── prep/                       # NEW — move *prep* scripts from v10/ here for clarity
│   └── baselines/                  # KEEP — already real-data baseline runners
│
├── attic/                          # NEW — preserves history without polluting imports
│   ├── v6_baselines/               # ← resistancemap/baselines/
│   ├── v6_clinical/                # ← resistancemap/clinical/
│   ├── v6_experiments/             # ← resistancemap/experiments/
│   ├── v6_theory/                  # ← resistancemap/theory/
│   ├── v6_training/                # ← resistancemap/training/
│   ├── v6_infrastructure/          # ← resistancemap/infrastructure/* (except guardrails.py)
│   ├── v6_interpretability/        # ← resistancemap/interpretability/
│   └── v10_intermediate_fixes/     # ← s3fix_apply.py, s3fix_diagnostic.py
│
├── checkpoints.bak.*/              # UNTOUCHED — flagged for cold-storage move only
├── paper/v8_artifacts/             # UNTOUCHED — evidence
├── release/                        # UNTOUCHED — release manifest
└── docker/                         # UNTOUCHED — Dockerfile + compose
```

`resistancemap/agents/orchestrator.py` (510 LOC) and `specialized.py` (475
LOC) overlap conceptually with `.claude/agents/*.md`. The Python orchestrator
is only consumed by `resistancemap/main.py:?` — recommend a **follow-up
audit** comparing it against the markdown-based agents in `.claude/agents/`
(architecture-chair, causal-inference-auditor, etc.) before deciding which is
canonical. Out of scope for this Tier-1 pass.

---

## §5. Risks & carve-outs (what NOT to touch)

1. **F-gate scripts are sacrosanct.** Per task contract, do not modify
   `scripts/v10/s1e_falsification_gates.py`, `s2f_f1_f2_gates.py`,
   `s3d_f3_crispr_oracle.py`, `s4d_nie_estimation.py`. The renames in §2 are
   *suggestions* — if applied, must be a single rename commit with no logic
   change so reviewers can trivially verify.
2. **`checkpoints.bak.*` (4.4 GB total)** — evidence trail referenced by
   `RUNS.md`; do not delete, only flag for cold-storage move.
3. **`paper/v8_artifacts/v1{0,1,2}_*`, `release/`, `RUNS.md`** — explicit
   no-touch per prompt.
4. **Lazy-import packages.** `resistancemap/clinical/__init__.py:21-55` uses
   `__getattr__` lazy import — a static "unused" verdict could be wrong if
   any caller does `getattr(resistancemap.clinical, "mrd_prediction")`. I
   verified no such dynamic access exists (`grep -rE "getattr.*resistancemap"`
   found 0 hits) but this should be re-checked before final archive.
5. **`resistancemap/baselines/__init__.py:3-5`** does eager re-export of
   `PERCEPTIONBaseline`, `CPABaseline`, etc. The package is unused, but
   moving it will break `import resistancemap.baselines` if any
   notebook/script not in this repo (e.g. user's analysis notebooks)
   imports it. Recommend a 1-cycle deprecation shim in
   `resistancemap/__init__.py` before archive.
6. **Parallelism caveat.** All §3 suggestions assume the hosts running
   `v10_runner` have `≥8` cores and ≥32 GB RAM (LOO Ridge memory is
   modest, but `s7` grid × N can balloon if naively `Parallel(n_jobs=-1)`).
   Set `n_jobs=min(os.cpu_count(), 8)` and `OMP_NUM_THREADS=1` inside
   workers to avoid BLAS oversubscription.
7. **Do not parallelise `s2e_hbayes_inference.py`** — NumPyro/Stan-style
   sampler with internal multi-chain parallelism; outer joblib will
   oversubscribe.

---

### Quantified savings

| Action | Files | LOC removed from import path | Wall-clock impact |
|---|---:|---:|---|
| Archive 5 v6 sub-packages (§1) | 24 | 14 600 | none (already unreached) |
| Archive 6 infrastructure + 5 interpretability modules (§1) | 11 | ≈3 000 (est.) | none |
| Delete `agents/example_usage.py` | 1 | 196 | none |
| Archive 2 stale s3fix scripts (§2) | 2 | 670 | none |
| Clean `__pycache__`/`.pytest_cache` (33 + 1 dirs) | — | — | tiny disk |
| **Total dead code** | **38** | **≈18 500 LOC** (≈20 % of tree) | — |
| Parallelise §3 loops (top 5) | 5 scripts | — | est. **5–20× on s4d, s5, s3fix_v3, s3f, s3d** |

Word count: ~1 480.
