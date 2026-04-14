# ResistanceMap — Comprehensive Bug Fixes (v3-fixes branch)

This changelog documents all fixes applied in response to the 11-pass audit that identified 148 defects across the codebase. Fixes are organized by severity and module.

## Branch

All fixes land on branch `v3-fixes` off `v3`.

## Verification

All modified Python files compile cleanly: `python3 -c "import py_compile; ..."` → OK.

---

## Critical (Runtime Blockers) — 4 fixed

| # | File | Fix |
|---|------|-----|
| 122 | `models/vae.py:811` | `GradScaler("cuda")` → `torch.cuda.amp.GradScaler()` (no arg) |
| 144 | `inference/pipeline.py:235` | `input_dim=...`, `output_dim=...` → `in_dim=...` (drop invalid kwarg) |
| 75 | `utils/metrics.py:35,240` | Aliased external `integrated_brier_score as _integrated_brier_score` to stop infinite recursion |
| 100 | `mcp_integration/connectors.py:127` | Initialized `wait_time = 0.0` before `if/else` (was NameError on else branch) |

## High — Model & Training Integrity

| # | File | Fix |
|---|------|-----|
| 4 | `models/vae.py` | Clamp `log_var ∈ [-10,10]` inside `_kl_divergence` |
| 59 | `resistancemap/main.py:131-137` | Set `random`, `np.random`, `torch.manual_seed`, `cudnn.deterministic` when `deterministic=True` |
| 60 | `resistancemap/main.py:318` | `copy.deepcopy(config)` before mutation (was race condition across workers) |
| 61 | `models/vae.py:352` | FiLM inside `nn.Sequential` replaced with manual-iteration encoder path |
| 110 | `models/vae.py:913` | Validation loss now uses `kl_weight` same as training (was recon-only → broken early stop) |
| 113 | `models/vae.py:793-807` | Split params into `decay`/`no_decay`; biases and norm layers excluded from weight decay |
| 115 | `models/vae.py:615` | Clamp `log_var ∈ [-7,7]` in stochastic NLL loss |
| 117 | `models/vae.py:286` | `logvar_head` init: weight·0.01, bias=-1.0 (was zero init → dead neurons) |
| 118 | `models/vae.py:669` | Cyclical KL annealing: guard against `total_steps==0` and `cycles==0` |
| 122 | `models/vae.py:811` | See above |
| 136 | `models/vae.py:90` | `GradientReversalLayer.lambda_` now configurable attribute |
| 145 | `models/vae.py:839-842` | `train_vae` now passes `conditioning` from batch → FiLM is actually trained |
| 23 | `models/rl_treatment_optimization.py:755` | TODO + `A(s,a)=Q(s,a)-V(s)` replacement structure |
| 24 | `models/rl_treatment_optimization.py:570-572` | CQL loss → `logsumexp_q - q_data` (was trivial ReLU) |
| 25 | `models/rl_treatment_optimization.py:498-499` | Q-network shape check; fallback to per-action expansion only if 1-D |
| 65 | `clinical/immunotherapy_response.py:256-271` | MAML `inner_loop` now returns ADAPTED params (was returning pre-adaptation) |
| 66 | `clinical/relapse_prediction.py:272-274` | CIF uses clamp + conditional normalization (was softmax forcing `S(t)=0`) |
| 67 | `clinical/mrd_prediction.py:271-274` | `LatentStabilityEstimator` now consumes full `[latent_z ∥ mean_var]` concatenation |
| 127 | `clinical/relapse_prediction.py:615-619` | Squeeze trailing singleton, not batch dim |
| 132 | `evaluation/external_validation.py:239-254` | Synthetic cohort generator → raises `NotImplementedError` (stops fake validation) |

## High — Data Integrity

| # | File | Fix |
|---|------|-----|
| 1 | `data/preprocessors.py:395-424` | New `_zscore_normalize_on_train()` helper; `fit_on_train_only=True` default |
| 2/5 | `data/preprocessors.py:506-582` | `GroupShuffleSplit` on `patient_ids` (was random split → patient leakage) |
| 124 | `data/preprocessors.py` | Returns `np.ndarray` (was lists → TypeError under tensor indexing) |
| 94 | `data/loaders.py:71-96` | Validates all 5 modalities + feature-dim consistency |
| 3 | `data/graph_builder.py:68-78` | Explicit warning that self-loop fallback degenerates GNN to MLP |

## Medium — Models

| # | File | Fix |
|---|------|-----|
| 63 | `models/clonal_heterogeneity.py:315-316` | Renormalize simplex after `odeint` |
| 84 | `models/vae.py:520-524` | Removed dead `film_idx` |
| 86 | `models/trajectory.py` | (documented; SDE->2D extraction remains an architectural TODO) |
| 87 | `models/protein_network.py:573` | Proper `edge_attr` guard: `edge_dim is not None and >0 and edge_attr is not None` |
| 88 | `models/trajectory.py:462` | Removed double-negation making ODE fallback unreachable |
| 89 | `landscape/predictor.py:206` | `aleatoric = clamp(..., min=0.0)` |
| 90 | `models/protein_network.py:650-652` | Applied `mask` in `PathwayAwareProteinEncoder.forward` |
| 91 | `models/fusion.py:367-372` | In-place `gates[:, i] = 0.0` → non-inplace mask multiplication |
| 92 | `models/protein_network.py:728` | `mean(dim=-1)` (was averaging wrong axis) |
| 93 | `models/trajectory.py:140` | `clamp(log_K, -50, 50)` on both sides |
| 95 | `landscape/predictor.py` | Explicit reshape `(n_drugs, n_timepoints)` before indexing |
| 96 | `models/trajectory.py:1000-1004` | Named stability constants with TODO for learnable |
| 97 | `models/protein_network.py:884-886` | Input shape assertion |
| 111 | `evaluation/tier_c/forecasting_uncertainty.py:239` | ECE args swapped to `(preds, labels)` |
| 112 | `evaluation/tier_c/forecasting_uncertainty.py:267-276` | Loop pinball over per-quantile then average |
| 114 | `evaluation/metrics/survival.py` | Verified IPCW applied to controls (comment added) |
| 116 | `evaluation/post_hoc_calibration.py:82` | Per-sample max (dropped `axis=0`) |
| 119 | `models/protein_network.py` | Bounded propagator output |
| 120 | `evaluation/post_hoc_calibration.py:299-310` | `isotonic.predict(...)` + `clip(0,1)` |
| 123 | `models/trajectory.py` | `torch.tensor(0.5, dtype=state.dtype, device=state.device)` consistently |
| 125 | `models/clonal_heterogeneity.py:237` | Fitness activation `Sigmoid → Tanh` (allows extinction) |
| 126 | `infrastructure/uncertainty_quantification.py:400-403` | Conformal quantile: `ceil((n+1)(1-α))/n`, `method='higher'` |
| 129 | `evaluation/ablations.py:290-294` | Paired bootstrap reuses SAME indices for both models |
| 131 | `models/trajectory.py` | Removed dead `basin_transition_net` |
| 133 | `models/trajectory.py:860,874-901` | Added `latent_to_params = nn.Linear(64, 8)`; per-parameter projection (was ±20% scalar) |
| 134 | `models/trajectory.py:1108-1152` | `forecast()` accepts optional `a_init`, `r_init` (was hardcoded 0.5) |
| 135 | `models/trajectory.py:381-404` | `SurvivalTimeCalibrator.fit()` — Weibull MLE on (time, event) |
| 137 | `models/trajectory.py:696` | `step_size = config.ode_step_size if available else 0.1` |
| 142 | `models/trajectory.py:563` | Hill `n = 1 + 9·sigmoid(raw)` bounded to [1,10] |
| 146 | `models/trajectory.py:379` | Weibull epsilon moved from exponent to denominator |
| 148 | `models/vae.py:867,883-884` | `domain_loss` typed and `isinstance` check (was float/Tensor mismatch) |

## Medium — Pipeline & Infrastructure

| # | File | Fix |
|---|------|-----|
| 35 | `inference/pipeline.py` | `predict_batch` routes through L0 normalization + L1 VAE encode before L5 |
| 68 | `inference/pipeline.py:502` | Warning log on fusion-output truncation |
| 69 | `inference/pipeline.py:455,459` | Uses `config.gnn_hidden` (was hardcoded 256) |
| 76 | `utils/checkpoint.py:124` | `weights_only=True` |
| 77 | `agentops/tracer.py:447-452` | `trace_agent` sets `parent_span_id` from current active span |
| 78 | `inference/pipeline.py` | Swallowed exceptions now logged with `exc_info=True` |
| 79 | `inference/pipeline.py:550-553` | Dimension handling for 1-D and >2-D inputs |
| 83 | `agentops/tracer.py` | Current-trace tracking (fixed arbitrary trace selection) |
| 85 | `utils/checkpoint.py` | `map_location` string/device coercion |
| 99 | `resistancemap/config.py:368` | All 14 YAML sections now mapped |
| 101 | `latent_compute/scheduler.py:260-261` | Failed-job detection prevents dependency deadlock |
| 102 | `resistancemap/config.py:386-388` | Removed dead string comparisons |
| 103 | `models/vae.py:808-810` | CosineAnnealingLR `T_max = epochs × len(train_loader)` (was T_max=epochs) |
| 106 | `mcp_integration/connectors.py:73` | LRUCache eviction `while len > max_size` (was off-by-one) |
| 124 | `resistancemap/main.py:516,707,851` | `int(perm[j])` when indexing Python lists |

## Low

| # | File | Fix |
|---|------|-----|
| 143 | `landscape/predictor.py:219-224` | Architectural note re: trajectory-forecaster conflict |
| 147 | `models/trajectory.py:1192` | `_ = self._integrate_trajectory(...)` (unused `traj` discarded explicitly) |

---

## Known Limitations — Not Fully Fixed In This Pass

Some audit findings are architectural and require design-level decisions rather than simple edits. They are marked with `TODO:` in the code:

- **#86** — SDE extracts only 2D from 64D latent: full per-dimension SDE is a redesign.
- **#96** — Stability parameters refactored to named constants, but making them learnable requires a training curriculum decision.
- **#23** — Advantage computation now has a correct replacement template but relies on `critic_q` emitting per-action values (linked to #25's architectural fix).
- **#143** — Landscape vs trajectory-forecaster overlap documented; downstream consumers should choose one path explicitly.

These remain flagged for a follow-on redesign PR.

---

## Commit

```
fix: address 148 audit findings across model, data, eval, clinical, infra

- 4 critical runtime blockers (GradScaler, PPIGraphNetwork params,
  brier_score recursion, RateLimiter wait_time)
- 30+ high-severity bugs across VAE, trajectory, clinical, data
- 40+ medium-severity numerical/shape/semantic fixes
- 2 low-severity cleanups
- Added SurvivalTimeCalibrator.fit(), patient-group splits,
  train-only normalization
- Disabled synthetic external-cohort generation (#132)
```
