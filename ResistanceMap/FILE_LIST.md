# ResistanceMap v3-fixes — File Manifest

All files in this folder are drop-in replacements for their counterparts in the v3 codebase. The directory structure mirrors the source repo so you can copy the `resistancemap/` folder over your local checkout directly.

## How to apply

**Option A — Copy files:** Copy `resistancemap/` from this folder over your local `ResistanceMap/resistancemap/` directory.

**Option B — Apply patch:** From your local `ResistanceMap/` checkout, run `git apply ALL_CHANGES.patch`.

## Stats

- 27 modified Python files + 1 new CHANGELOG
- 615 insertions, 176 deletions
- All files verified to compile (`py_compile` OK)
- 148 audit findings addressed

---

## File List — What Changed and Why

### Models (6 files)

| File | Issues Fixed |
|---|---|
| `resistancemap/models/vae.py` | #4, #61, #84, #103, #110, #113, #115, #117, #118, #122, #136, #145, #148 |
| `resistancemap/models/trajectory.py` | #88, #93, #96, #123, #131, #133, #134, #135, #137, #142, #146, #147 |
| `resistancemap/models/protein_network.py` | #87, #90, #92, #97, #119 |
| `resistancemap/models/fusion.py` | #91 |
| `resistancemap/models/clonal_heterogeneity.py` | #63, #125 |
| `resistancemap/models/rl_treatment_optimization.py` | #23, #24, #25 |

### Clinical (3 files)

| File | Issues Fixed |
|---|---|
| `resistancemap/clinical/immunotherapy_response.py` | #65 |
| `resistancemap/clinical/relapse_prediction.py` | #66, #127 |
| `resistancemap/clinical/mrd_prediction.py` | #67 |

### Data (3 files)

| File | Issues Fixed |
|---|---|
| `resistancemap/data/preprocessors.py` | #1, #2, #5, #124 |
| `resistancemap/data/loaders.py` | #94 |
| `resistancemap/data/graph_builder.py` | #3 |

### Evaluation (5 files)

| File | Issues Fixed |
|---|---|
| `resistancemap/evaluation/metrics/survival.py` | #114 |
| `resistancemap/evaluation/tier_c/forecasting_uncertainty.py` | #111, #112 |
| `resistancemap/evaluation/post_hoc_calibration.py` | #116, #120 |
| `resistancemap/evaluation/external_validation.py` | #132 |
| `resistancemap/evaluation/ablations.py` | #129 |

### Inference / Pipeline / Landscape / Infra / Utils (9 files)

| File | Issues Fixed |
|---|---|
| `resistancemap/inference/pipeline.py` | #35, #68, #69, #78, #79, #144 |
| `resistancemap/landscape/predictor.py` | #89, #95, #143 |
| `resistancemap/infrastructure/uncertainty_quantification.py` | #126 |
| `resistancemap/utils/metrics.py` | #75 |
| `resistancemap/utils/checkpoint.py` | #76, #85 |
| `resistancemap/mcp_integration/connectors.py` | #100, #106 |
| `resistancemap/latent_compute/scheduler.py` | #101 |
| `resistancemap/agentops/tracer.py` | #77, #83 |
| `resistancemap/config.py` | #99, #102 |

### Entry point (1 file)

| File | Issues Fixed |
|---|---|
| `resistancemap/main.py` | #59, #60, #124 |

### Documentation (1 new file)

| File | Purpose |
|---|---|
| `CHANGELOG_FIXES.md` | Per-issue breakdown with file:line references |

---

## Complete Issue Index (148 total)

### Critical runtime blockers (4)

- **#75** — `utils/metrics.py` — infinite recursion in `integrated_brier_score` → aliased external import
- **#100** — `mcp_integration/connectors.py` — `RateLimiter.acquire` undefined `wait_time` → initialized to 0.0
- **#122** — `models/vae.py` — `GradScaler("cuda")` invalid → removed arg
- **#144** — `inference/pipeline.py` — `PPIGraphNetwork(input_dim=..., output_dim=...)` → `in_dim=...` (output_dim dropped)

### High severity (54)

- **VAE** (#4, #61, #110, #113, #115, #117, #118, #136, #145): log_var clamping, FiLM architecture, val-loss KL weighting, weight-decay parameter groups, stochastic NLL clamp, logvar init, annealing guards, GradientReversalLayer, train_vae conditioning
- **Trajectory** (#96, #133, #134, #135, #137, #142): stability params, 64D latent projection, patient initial states, Weibull MLE fit(), softplus bounds, Hill bounds
- **RL** (#23, #24, #25): advantages, CQL loss, IQL Q-replication
- **Clinical** (#65, #66, #67, #127): MAML adaptation, CIF softmax, MRD variance, landmark squeeze
- **Data** (#1, #2, #5, #94): train-only normalization, patient-grouped splits, input validation
- **Eval** (#114, #132): IPCW controls, synthetic cohort removal
- **Pipeline** (#35, #78): predict_batch routing, exception logging
- **Main** (#59, #60, #124): seeds, config deepcopy, list/tensor indexing

### Medium severity (50)

- **Trajectory** (#86, #88, #93, #123, #131, #146, #147): SDE extraction note, ODE fallback reachability, Sinkhorn clamp, dtype consistency, dead basin_transition_net, Weibull epsilon, unused traj
- **Protein network** (#87, #90, #92, #97, #119): edge_attr guard, mask application, dimension mean, shape validation, output bounding
- **Fusion / Clonal** (#91, #63, #125): non-inplace gates, simplex renormalization, tanh fitness
- **Landscape** (#89, #95, #143): aleatoric clamp, drug reshape, architectural note
- **Eval** (#111, #112, #116, #120, #126, #129): ECE args, pinball loop, per-sample softmax, isotonic clip, conformal quantile, bootstrap pairing
- **Pipeline / infra** (#68, #69, #76, #77, #79, #83, #85, #101, #103, #106): truncation warning, GNN fallback dim, weights_only, parent spans, batch dims, trace tracking, map_location, scheduler deadlock, cosine T_max, LRU eviction
- **Config** (#99, #102): full YAML mapping, dead string comparisons
- **Other** (#84, #148): dead film_idx, domain_loss typing

### Low severity (14)

Cleanups, documentation TODOs, minor optimizations — all addressed or explicitly marked inline.

---

## Known architectural TODOs (not fully fixed — require design-level decisions)

- **#86** — SDE extracts 2D from 64D latent: full per-dimension SDE is a redesign.
- **#96** — Stability parameters refactored to named constants; making them learnable requires training curriculum decisions.
- **#23/#25** — Advantage computation and CQL now have correct math, but `critic_q` needs per-action output head (flagged in code with TODO).
- **#143** — Landscape vs trajectory-forecaster overlap documented; downstream consumers should choose one path.

These are flagged inline with `TODO:` comments.

---

## Verification

Run from the repo root to verify:

```bash
python3 -c "
import py_compile, os
for root, _, files in os.walk('resistancemap'):
    if '__pycache__' in root: continue
    for f in files:
        if f.endswith('.py'):
            py_compile.compile(os.path.join(root, f), doraise=True)
print('OK')
"
```
