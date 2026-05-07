# Wave 5 — Category C Eval Math

**Source agent:** `a8b8010aba86e6748`
**Date:** 2026-05-07
**Persisted by:** main thread (subagent Write was blocked by sandbox)

---

Confirmed forecasting_uncertainty IS used in production eval pipeline (Tier C agent). The bare except was masking the kwarg failure — the eval logs show "criteria_results: {}" with empty evidence, which is consistent with both IBS and time-dependent AUC silently failing then producing no metrics. Downstream eval reports/findings are affected.

I have all I need. Now write the full audit. Per hard rules: no source modification.

---

# Audit: 5 BLOCKER math bugs (Category C — eval-mathematics correctness)

**Repo:** `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap`
**Branch audited:** `v6` (the 4 missing-from-worktree files were read via `git show v6:...`)
**Note on paths:** the blocker doc says `evaluation/...`; the actual file paths are `ResistanceMap/resistancemap/evaluation/...`. All line numbers verified against v6 HEAD.

Per "do not modify source": no patches were applied — patches are presented as unified diffs only.

---

## §C.1 Verification — `_softmax` is a sigmoid (and worse on 1-D)

`ResistanceMap/resistancemap/evaluation/post_hoc_calibration.py:79-89`

```python
def _softmax(logits: np.ndarray) -> np.ndarray:
    ...
    if logits.ndim == 1:
        # Binary: apply sigmoid with per-sample max normalization
        x = logits - np.max(logits)        # << over the whole vector
        ex = np.exp(x)
        return ex / (1.0 + ex)             # << sigmoid, not softmax
    else:
        # Multiclass: apply softmax with per-sample max normalization
        x = logits - np.max(logits, axis=1, keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=1, keepdims=True)
```

The 2-D branch is correct softmax. The 1-D branch is genuinely broken:
- Subtracts the **global** max from a 1-D vector → every element ≤ 0 → every "probability" ≤ 0.5.
- Worse than the report describes: for a single-sample call, `x = z - max(z) = 0` always, so the function returns **0.5 for every input** — temperature scaling on length-1 batches is a constant.
- `_validate_logits_labels` doctstring says 1-D = "binary classification", so this 1-D path is the binary `sigmoid(z)` path. It should be `sigmoid(z)`, not `sigmoid(z − max(z))`.

### §C.1 Patch (unified diff, not applied)

```diff
--- a/ResistanceMap/resistancemap/evaluation/post_hoc_calibration.py
+++ b/ResistanceMap/resistancemap/evaluation/post_hoc_calibration.py
@@ -79,12 +79,15 @@ def _softmax(logits: np.ndarray) -> np.ndarray:
-    if logits.ndim == 1:
-        # Binary: apply sigmoid with per-sample max normalization
-        x = logits - np.max(logits)
-        ex = np.exp(x)
-        return ex / (1.0 + ex)
-    else:
-        # Multiclass: apply softmax with per-sample max normalization
-        x = logits - np.max(logits, axis=1, keepdims=True)
-        ex = np.exp(x)
-        return ex / ex.sum(axis=1, keepdims=True)
+    if logits.ndim == 1:
+        # Binary single-logit: numerically stable sigmoid sigmoid(z)=1/(1+e^-z)
+        # Use np.where for stability on both signs of z (avoids overflow).
+        pos = logits >= 0
+        out = np.empty_like(logits)
+        out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
+        ez = np.exp(logits[~pos])
+        out[~pos] = ez / (1.0 + ez)
+        return out
+    # Multiclass: row-wise softmax (correct; unchanged)
+    x = logits - np.max(logits, axis=1, keepdims=True)
+    ex = np.exp(x)
+    return ex / ex.sum(axis=1, keepdims=True)
```

### §C.1 Test (one-liner)

```python
import numpy as np
z = np.array([0.1, 2.0, -1.0]);  p = _softmax(z)
# BROKEN: max(p)<=0.5 and p does NOT equal 1/(1+exp(-z))
# FIXED:  np.allclose(p, 1/(1+np.exp(-z)))  -> True
```

For the 2-D path: `_softmax(np.array([[1,2,3]])).sum() == 1.0` — already True; no regression.

### §C.1 Downstream

- `TemperatureScaling.fit` (line 145) minimises NLL over T. With every prob clipped ≤ 0.5 in binary mode, the optimiser converges to whatever T best matches the **empirical positive rate p̄ ≈ 0.5**, not the true calibration optimum. **Any binary calibration result currently in `paper/` from `TemperatureScaling` is wrong.** Re-run `calibrate_model_outputs(method="temperature")` after the fix.
- Multiclass calibration (≥ 2 classes, 2-D logits) is unaffected — no re-run needed for those.

---

## §C.2 Verification — `z_alpha` formula returns 1.43, not 1.96

`ResistanceMap/resistancemap/evaluation/power_analysis.py:206-207`

```python
        # Critical z-value for two-tailed test
        z_alpha = sqrt(2) * erfc(self.significance_level / 2) ** (-0.5)
```

Numerical check at α=0.05:
`erfc(0.025) ≈ 0.97181` → `0.97181**(-0.5) ≈ 1.01441` → `sqrt(2)*1.01441 ≈ 1.4346`.
True value: `Φ⁻¹(1 − 0.025) ≈ 1.95996`. The code is off by ~27 %.

The correct two-tailed critical value uses the inverse-CDF: `Φ⁻¹(1 − α/2)`, which can be expressed via `erfcinv` (no scipy needed) as `sqrt(2) * erfcinv(α)`.

**Worse than reported:** because `z_alpha` is too small, the rejection region is too lax, so `power = 0.5*erfc((z_alpha − λ)/√2) + …` is **systematically inflated**. Every power number in the codebase that flows through `_compute_power` over-states power. This affects the "is_adequately_powered" boolean and `minimum_recommended_n` (which under-counts the n needed by ~30–50 % depending on effect size).

### §C.2 Patch (unified diff, not applied)

```diff
--- a/ResistanceMap/resistancemap/evaluation/power_analysis.py
+++ b/ResistanceMap/resistancemap/evaluation/power_analysis.py
@@ -25,7 +25,7 @@ import logging
 from dataclasses import dataclass, field
-from math import sqrt, log10, erfc
+from math import sqrt, log10, erfc
+from scipy.special import erfcinv
 from typing import Optional
@@ -204,7 +204,8 @@
         """
-        # Critical z-value for two-tailed test
-        z_alpha = sqrt(2) * erfc(self.significance_level / 2) ** (-0.5)
+        # Critical z-value for two-tailed test: Phi^{-1}(1 - alpha/2)
+        # Equivalent to scipy.stats.norm.ppf(1 - alpha/2).
+        z_alpha = sqrt(2.0) * erfcinv(self.significance_level)
```

### §C.2 Test

```python
from scipy.special import erfcinv;  from math import sqrt
assert abs(sqrt(2)*erfcinv(0.05) - 1.959963984540054) < 1e-9
# BROKEN: sqrt(2)*erfc(0.025)**(-0.5) ~ 1.4346
```

### §C.2 Downstream

- All `PowerAnalysisResult.power_at_current_n` and `minimum_recommended_n` numbers in the codebase need re-computing after fix.
- I checked `git grep` over `ResistanceMap/paper/**` — no JSON/Markdown artefact under `paper/` currently embeds a `power_at_current_n` value, so paper figures are not affected today. However, any future run of `power_analysis.PowerAnalysis.analyze(...)` (and any agent invoking it through `epistemic_boundaries.py`) must be re-run. Recommend tagging the previous outputs as "computed with biased z_α — invalid".

---

## §C.3 Verification — non-identifiable hazard mixture

`ResistanceMap/resistancemap/evaluation/competing_risks.py:111-143`

```python
            # Overall survival: S(t) = exp(-(p*h_int + (1-p)*h_acq)*t)
            overall_hazard = p_int * h_int + (1.0 - p_int) * h_acq
            surv = np.exp(-overall_hazard * t)
            ...
            elif r == 0:
                # Event: intrinsic risk
                # log(P(intrinsic) * h_int * S(t))
                nll -= np.log(np.clip(p_int * h_int * surv, 1e-15, 1.0))
            else:
                # Event: acquired risk
                nll -= np.log(np.clip((1.0 - p_int) * h_acq * surv, 1e-15, 1.0))
```

**Internally inconsistent:** the survival term uses a *mixture-weighted average hazard* (each subject experiences `p·h_int + (1−p)·h_acq`), while the event term uses the *cause-conditional density* `p·h_int · S` (subject is type-int with prob p, has only h_int, fails with density h_int). The two come from different generative models — they cannot both be correct.

**Worse than reported:** beyond non-identifiability, the optimisation has a flat ridge: any rescaling `(p, h_int, h_acq) → (p·c, h_int/c, …)` leaves the type-0 likelihood term unchanged, and similarly for type-1, so the Nelder-Mead estimate of `p_int` is essentially noise. The logged `p_int` in any report is uninterpretable.

Recommended fix (cause-specific hazards, Putter et al. 2007): **drop the mixture probability**, parameterise two cause-specific hazards directly. Likelihood becomes the standard CR form (here Exp baseline):

`L = ∏_i [h_int^{δ_i·I(r=0)} · h_acq^{δ_i·I(r=1)}] · exp(-(h_int + h_acq)·t_i)`

### §C.3 Patch (unified diff, not applied)

```diff
--- a/ResistanceMap/resistancemap/evaluation/competing_risks.py
+++ b/ResistanceMap/resistancemap/evaluation/competing_risks.py
@@
-        def neg_log_likelihood(params: np.ndarray) -> float:
-            h_int, h_acq, p_int = params
-            if h_int <= 0 or h_acq <= 0 or not (0 < p_int < 1):
-                return 1e10
-            nll = 0.0
-            for i in range(len(times)):
-                t = times[i]; e = events[i]; r = risk_types[i]
-                overall_hazard = p_int * h_int + (1.0 - p_int) * h_acq
-                surv = np.exp(-overall_hazard * t)
-                if e == 0:
-                    nll -= np.log(np.clip(surv, 1e-15, 1.0))
-                elif r == 0:
-                    nll -= np.log(np.clip(p_int * h_int * surv, 1e-15, 1.0))
-                else:
-                    nll -= np.log(np.clip((1.0 - p_int) * h_acq * surv, 1e-15, 1.0))
-            return nll
+        # Cause-specific hazards (Putter et al. 2007, Stat Med 26:2389).
+        # No mixture probability: the cause label r identifies which hazard
+        # contributed the event. Overall S(t) = exp(-(h_int + h_acq) t).
+        def neg_log_likelihood(params: np.ndarray) -> float:
+            h_int, h_acq = params
+            if h_int <= 0 or h_acq <= 0:
+                return 1e10
+            H_total = (h_int + h_acq) * times  # cumulative hazard at t_i
+            log_h = np.where(events == 0, 0.0,
+                             np.where(risk_types == 0,
+                                      np.log(max(h_int, 1e-15)),
+                                      np.log(max(h_acq, 1e-15))))
+            nll = float(np.sum(H_total - events * log_h))
+            return nll
@@
-        x0 = [init_hazard_intrinsic, init_hazard_acquired, init_mixture_prob]
+        x0 = [init_hazard_intrinsic, init_hazard_acquired]
@@
-        self.hazard_intrinsic, self.hazard_acquired, self.mixture_prob = result.x
+        self.hazard_intrinsic, self.hazard_acquired = result.x
+        # mixture_prob is non-identifiable in cause-specific framing; expose
+        # the empirical event-type fraction instead for backwards-compat.
+        n_events = int(np.sum(events))
+        self.mixture_prob = (
+            float(np.sum((events == 1) & (risk_types == 0)) / n_events)
+            if n_events > 0 else float("nan")
+        )
```

(Class attribute `mixture_prob` is preserved as the empirical type-0 fraction so the public API does not break; this should be documented.)

### §C.3 Test

```python
# Cause-specific MLE has closed form for Exp:  h_k_hat = N_events_of_type_k / sum(t)
times = np.array([1., 2., 3., 4., 5.]); events = np.array([1,1,0,1,1]); rt = np.array([0,1,0,0,1])
m = CompetingRisksModel().fit(times, events, rt)
T = times.sum();  h0 = ((events==1)&(rt==0)).sum()/T;  h1 = ((events==1)&(rt==1)).sum()/T
assert abs(m.hazard_intrinsic - h0) < 1e-3 and abs(m.hazard_acquired - h1) < 1e-3
```

### §C.3 Downstream

- No artefact under `paper/` currently embeds CompetingRisks numbers (verified via `git grep`). The module is imported only inside its own file.
- Any future report invoking `fit_competing_risks` / `CompetingRisksModel` must re-fit and re-interpret; **previous `p_int` values should be retracted** as non-identifiable.

---

## §C.4 Verification — kwarg mismatch silently swallowed

Caller (`forecasting_uncertainty.py:172-194`):

```python
ibs = float(integrated_brier_score(
    y_true_time, y_true_event, y_pred_risk, times=horizons))   # times=
...
td_auc = time_dependent_auc(
    y_true_time, y_true_event, y_pred_risk, times=horizons)    # times=
```

Callee (`metrics/survival.py:103-107` and `:182-187`):

```python
def integrated_brier_score(times, events, surv_pred, eval_times) -> float:
def time_dependent_auc   (times, events, risk_scores, eval_times) -> np.ndarray:
```

The callee already binds positional `times` to the **observed event times** (1st arg). The caller's `times=horizons` then collides with the positional `times` argument → `TypeError: ... got multiple values for argument 'times'`. The bare `except Exception` swallows it. Field result: `criteria_results == {}` and IBS / td-AUC are silently absent. `logs/evaluation/.../findings.jsonl` confirms `"criteria_results": {}` for the forecasting_uncertainty agent.

**Worse than reported (two extra defects):**

1. Even after renaming the kwarg, `integrated_brier_score` expects a **survival-probability matrix `surv_pred`**, but the caller passes `y_pred_risk` (risk score, ≈ 1 − surv). Passing risk where survival is expected makes the Brier score numerically meaningless (it's `BS(1−surv)`, which equals `BS(surv)` only at `surv = 0.5`).
2. `surv_pred` is required to be `(N, T)` shaped; the caller's `y_pred_risk` may be `(N,)` or `(N, T)` depending on agent intake. If 1-D, the reshape will misalign.

### §C.4 Patch (unified diff, not applied)

```diff
--- a/ResistanceMap/resistancemap/evaluation/tier_c/forecasting_uncertainty.py
+++ b/ResistanceMap/resistancemap/evaluation/tier_c/forecasting_uncertainty.py
@@
-            try:
-                ibs = float(
-                    integrated_brier_score(
-                        y_true_time, y_true_event, y_pred_risk, times=horizons
-                    )
-                )
-                criteria_results["integrated_brier_score"] = ibs
-                ...
-            except Exception as exc:  # pragma: no cover - defensive
-                logger.warning("IBS computation failed: %s", exc)
-                criteria_results["integrated_brier_score_error"] = str(exc)
+            # IBS expects survival probabilities S(t|x); the agent intake
+            # holds *risk* scores (cumulative incidence proxy in [0,1]).
+            # Convert and reshape to (N, T) to match the metric signature.
+            surv_pred = np.atleast_2d(1.0 - np.asarray(y_pred_risk, dtype=float))
+            if surv_pred.shape[0] == 1 and surv_pred.shape[1] == y_true_time.shape[0]:
+                surv_pred = surv_pred.T  # was (1,N); want (N,1) when single horizon
+            try:
+                ibs = float(integrated_brier_score(
+                    y_true_time, y_true_event, surv_pred, eval_times=horizons))
+                criteria_results["integrated_brier_score"] = ibs
+                ...
+            except (TypeError, ValueError) as exc:
+                logger.warning("IBS computation failed: %s", exc)
+                criteria_results["integrated_brier_score_error"] = str(exc)
@@
-            try:
-                td_auc = time_dependent_auc(
-                    y_true_time, y_true_event, y_pred_risk, times=horizons
-                )
-                ...
-            except Exception as exc:  # pragma: no cover
-                logger.warning("Time-dependent AUC failed: %s", exc)
+            try:
+                td_auc = time_dependent_auc(
+                    y_true_time, y_true_event, y_pred_risk, eval_times=horizons)
+                ...
+            except (TypeError, ValueError) as exc:
+                logger.warning("Time-dependent AUC failed: %s", exc)
```

(Narrow `except` to TypeError/ValueError so future bugs propagate. Time-dependent AUC keeps `y_pred_risk` since it expects `risk_scores`.)

### §C.4 Test

```python
# After fix: dummy data must yield a finite IBS, not a swallowed exception.
N = 50; horizons = np.array([1.0, 2.0, 3.0])
t = np.random.uniform(0.5, 5, N); e = (np.random.rand(N) > 0.3).astype(int)
risk = np.random.rand(N, 3)
out = ForecastUncertainty().assess({"y_true_time": t, "y_true_event": e,
                                    "y_pred_risk": risk, "horizons": horizons})
assert np.isfinite(out.criteria_results["integrated_brier_score"])
```

### §C.4 Downstream

- All Tier-C `forecasting_uncertainty_agent` runs in `ResistanceMap/logs/evaluation/**` show `criteria_results: {}` — IBS / td-AUC have **never** been computed in any logged run. Anything quoting these from those logs is reporting nulls.
- Re-run the entire Tier-C eval pipeline (`AGENT_ORCHESTRATOR.md` Tier C path) once patched — at least the forecasting_uncertainty agent and any aggregator that consumes its findings (`principal_integrator_agent`).

---

## §C.5 Verification — MAML inner loop is a no-op

`ResistanceMap/resistancemap/evaluation/cytogenetic_stratification.py:418-461`

```python
        adapted_params = {name: param.clone().detach().requires_grad_(True)
                         for name, param in self.task_model.named_parameters()}
        adapted_optimizer = optim.SGD(adapted_params.values(), lr=self.inner_lr)
        for _ in range(self.inner_steps):
            adapted_optimizer.zero_grad()
            logits = self._forward_with_params(support_x, adapted_params)
            loss = loss_fn(logits, support_y)
            loss.backward()
            adapted_optimizer.step()
        return adapted_params, loss.item()
    ...
    def _forward_with_params(self, x, params):
        return self.task_model(x)             # << params is IGNORED
    ...
    for support_x, support_y in tasks:
        adapted_params, _ = self.inner_loop(support_x, support_y, loss_fn)
        meta_loss += _                         # << "_" is the float loss; no graph
```

Confirmed and **far worse than reported**:

1. `_forward_with_params` ignores `params` entirely → the inner loop trains the cloned tensors on a **constant graph that doesn't include them**. `loss.backward()` populates `.grad` on the original `task_model` parameters via the leaf path, while the cloned tensors get no grad (they aren't in the graph) → `adapted_optimizer.step()` is a no-op.
2. The optimiser steps may even raise on `None` grads; if it doesn't, it just leaves clones unchanged.
3. Outer loop accumulates `meta_loss += _` where `_` is `loss.item()` — a Python float. No `.backward()`, no meta-update, no optimiser. The "outer loop" returns a number but updates **nothing**.
4. No `create_graph=True` on `loss.backward()` → even if (1)–(3) were fixed, this would be FOMAML, not MAML.
5. The class is imported by `cytogenetic_stratification.py` itself — need to confirm whether any production agent calls it.

Per the report's recommendation, option (b) is the right choice for v0.1: gate the class behind an explicit experimental flag and remove it from any production code path.

### §C.5 Patch (unified diff, not applied — option (b) per report)

```diff
--- a/ResistanceMap/resistancemap/evaluation/cytogenetic_stratification.py
+++ b/ResistanceMap/resistancemap/evaluation/cytogenetic_stratification.py
@@
-class MAMLLearner:
+EXPERIMENTAL_NOT_VALIDATED = True
+
+
+class MAMLLearner:
     """
-    Model-Agnostic Meta-Learning for cytogenetic subgroup adaptation.
+    EXPERIMENTAL — NOT VALIDATED.
+
+    Current implementation is a no-op:
+      * `_forward_with_params` ignores its `params` argument, so the inner
+        loop's adapted parameters are never used in any forward pass.
+      * The outer loop accumulates a Python float (`loss.item()`) instead of
+        a differentiable tensor, so no meta-gradient flows.
+      * `loss.backward()` is called without `create_graph=True`, so even
+        first-order MAML semantics are not achieved.
+
+    DO NOT use in production until reimplemented (see Finn 2017 + the
+    `torch.func.functional_call` pattern). Audit 2026-05-07.
     """
+
+    def __init__(self, *args, **kwargs):
+        raise NotImplementedError(
+            "MAMLLearner is EXPERIMENTAL_NOT_VALIDATED and disabled. "
+            "Set resistancemap.evaluation.cytogenetic_stratification."
+            "EXPERIMENTAL_NOT_VALIDATED = False to opt in for development."
+        )
```

Plus: `git grep MAMLLearner` to find callers and remove from production paths.

### §C.5 Test

```python
# Before fix: instantiating "succeeds" but does nothing — adapted weights == originals.
m = MAMLLearner(task_model, ...);  ap, _ = m.inner_loop(x, y, F.cross_entropy)
orig = dict(m.task_model.named_parameters())
assert all(torch.equal(ap[k], orig[k].detach()) for k in ap)   # PROVES no-op
# After fix (option b): MAMLLearner(...) raises NotImplementedError.
```

### §C.5 Downstream

- `git grep MAMLLearner` outside this file → no production callers found in v6, but the class is in `__all__`-style export of the module. Any external script importing it has been silently doing nothing — its outputs (if any are persisted) should be tagged invalid.
- No paper artefact references MAML on inspection. After option (b), confirm that no notebook or `scripts/` entry instantiates it, and remove tests that exercise MAML semantics.

---

## §Summary table

| ID  | File:line | Bug | Numerical sanity check | Reviewer-claim accurate? | Worse-than-reported? | Downstream re-run |
|-----|-----------|-----|------------------------|--------------------------|----------------------|---|
| C.1 | `evaluation/post_hoc_calibration.py:79-89` | 1-D branch is `sigmoid(z − max(z))`, all probs ≤ 0.5; degenerate to 0.5 on N=1 | `_softmax([0.1,2.,-1.]) → max ≤ 0.5` (broken) vs `==1/(1+e⁻ᶻ)` (fixed) | Yes | Yes — N=1 always returns 0.5 | All binary `TemperatureScaling` calibration outputs |
| C.2 | `evaluation/power_analysis.py:206-207` | `z_α = √2 · erfc(α/2)^(-0.5)` ≈ 1.43 not 1.96 | `√2 · erfcinv(0.05) ≈ 1.95996` | Yes | Yes — all reported powers systematically inflated | Every `PowerAnalysisResult` (none in `paper/` today; future runs only) |
| C.3 | `evaluation/competing_risks.py:111-143` | Mixes mixture-survival with cause-conditional density; non-identifiable | Cause-specific MLE: `ĥ_k = N_k / Σt_i` (closed form) | Yes | Yes — `p_int` lies on a flat ridge, value is noise | Any reported `mixture_prob` should be retracted (none in `paper/`) |
| C.4 | `evaluation/tier_c/forecasting_uncertainty.py:173-194` vs `metrics/survival.py:103,107,182,186` | `times=` collides with positional `times` (event times); bare `except` hides it; **also** passes risk where survival expected | After fix, `out.criteria_results["integrated_brier_score"]` is finite | Yes | Yes — IBS argument is risk, not survival; `surv_pred` shape may also mismatch | Re-run **all** Tier-C `forecasting_uncertainty_agent` evals (`logs/evaluation/**` show `criteria_results: {}` historically) |
| C.5 | `evaluation/cytogenetic_stratification.py:418-461` | Inner loop's adapted params never enter the graph; outer loop accumulates a float; no `create_graph` | Cloned `adapted_params` ≡ original `task_model` params after `inner_loop` | Yes | Yes — outer-loop "meta_loss" is a float; nothing propagates anywhere | No production callers found; flag `EXPERIMENTAL_NOT_VALIDATED` and disable |

**Cross-cutting:**
- C.1 + C.4 corrupt actively-reported metrics today.
- C.2 + C.3 corrupt metrics that are *available to be reported* but not currently in `paper/` artefacts — the bigger risk is that they're invoked silently by future agents.
- C.5 has been a no-op since introduction.
- All five bugs share an absence of unit tests that exercise the math (e.g., a single assertion `np.isclose(_softmax(z).sum(), 1)` would have caught C.1; a 2-line analytic-MLE check would have caught C.3). Adding such "math-axiom" tests should be a Tier-1 follow-up.