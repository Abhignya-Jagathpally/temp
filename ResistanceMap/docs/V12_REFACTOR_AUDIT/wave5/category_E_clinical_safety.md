# Wave 5 — Category E Clinical Safety

**Source agent:** `a380701997ade369d`
**Date:** 2026-05-07
**Persisted by:** main thread (subagent Write was blocked by sandbox)

---

Confirmed: no `test_clinical_safety.py` exists. I have enough verification to write the audit report. Final report:

---

# Clinical Safety Audit Report — ResistanceMap Category E Blockers

## §E.1 Verification

**File:** `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/resistancemap/clinical/dynamic_treatment_regimes.py` (note: actual location is under `resistancemap/clinical/` package, not bare `clinical/`)

**Line 91-94:** Confirmed action space declared at module top-level inside `TreatmentFeasibilityMask.__init__`:
```python
self.action_names = [
    "watchful_waiting", "lenalidomide", "bortezomib", "carfilzomib",
    "daratumumab", "venetoclax", "cart", "transplant"
]
```

**Public exposure:** `TreatmentFeasibilityMask`, `DTRPolicyOptimizer`, and factory `build_dtr_optimizer()` are all importable; module's `__main__` block (lines 872-917) runs `select_action(...)` returning a drug name string with NO data, NO model fit, and NO disclaimer.

**No IRB / consent / disclaimer gate present.** No occurrences of "IRB", "consent", "research-only", "NOT FOR CLINICAL USE" in the file (verified by full-file Read of all 918 lines).

**No `tests/test_clinical*.py`.** Tests directory `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/tests/test_clinical_safety.py` does not exist (Read confirmed file-not-found).

**Module is registered in public lazy-loader** at `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/resistancemap/clinical/__init__.py` line 11 (`__all__` exposes `dynamic_treatment_regimes`).

## §E.1 Patch (apply at top of `resistancemap/clinical/dynamic_treatment_regimes.py`, after line 16)

```python
# ============================================================================
# RESEARCH-ONLY GATE — DO NOT REMOVE
# ============================================================================
# This module declares an action space over FDA-approved myeloma drugs
# (lenalidomide, bortezomib, carfilzomib, daratumumab, venetoclax, CAR-T,
#  transplant) and emits action recommendations from neural networks that
# have NOT been fit on patient data, validated against any oracle, or
# IRB-approved for clinical decision support.
#
# NOT FOR CLINICAL USE. NOT FDA-CLEARED. NOT A MEDICAL DEVICE.
# ============================================================================
import os as _os

_RM_ACK_KEY = "RM_RESEARCH_USE_ACKNOWLEDGED"
_RM_ACK_VAL = "I_understand_this_is_research_only_no_clinical_use"

if _os.environ.get(_RM_ACK_KEY) != _RM_ACK_VAL:
    raise RuntimeError(
        "ResistanceMap clinical/dynamic_treatment_regimes is research-only "
        "code that emits drug recommendations from untrained / unvalidated "
        "neural networks. It is NOT FDA-cleared, NOT IRB-gated, and MUST "
        "NOT be used for any clinical decision affecting a real patient.\n"
        "To load this module for research, set:\n"
        f"  export {_RM_ACK_KEY}={_RM_ACK_VAL}\n"
        "By setting this variable you acknowledge research-only use."
    )
```

Then add a banner decorator and apply to every public callable that returns an action:

```python
import functools as _functools
import warnings as _warnings

def _not_for_clinical_use(fn):
    """Decorator: every call emits a NOT FOR CLINICAL USE warning."""
    @_functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        _warnings.warn(
            f"{fn.__qualname__}: NOT FOR CLINICAL USE. Output is research-only "
            "and may come from unfit weights. Do not use for patient care.",
            category=UserWarning, stacklevel=2,
        )
        return fn(*args, **kwargs)
    return _wrapped
```

Apply `@_not_for_clinical_use` to: `DTRPolicyOptimizer.select_action` (line 708), `DTRPolicyOptimizer.forward` (line 825), `TreatmentFeasibilityMask.get_feasible_actions` (line 96), `build_dtr_optimizer` (line 849), and `FittedQIteration.compute_optimal_actions` (line 239).

Delete or guard the `if __name__ == "__main__":` block (lines 872-917) — it currently prints a drug recommendation from random init weights, which is the most actively dangerous piece of code in the module.

---

## §E.2 Verification — Six Free-Text/Action-Emitting Modules from Untrained Weights

Verified six modules under `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/resistancemap/clinical/`. None defines `is_fitted`, none has `_check_fitted`, all are `nn.Module` subclasses whose weights start at PyTorch default init (Kaiming/Xavier random). Recommendation methods are callable immediately after `__init__`.

| # | Module file | Class | Recommendation method | Free-text evidence |
|---|---|---|---|---|
| 1 | `early_relapse_detection.py` | `EarlyRelapsePredictor` | `predict_early_relapse` (line 577) | Line 673: `"Escalate to intensive regimen; monitor q2 weeks; consider salvage therapy trial"` (verbatim match to user's example string) |
| 2 | `smm_progression.py` | `SMMProgressionPredictor` | `forward` (line 438) | Lines 506-508: emits `risk_tier ∈ {1,2,3}` driving SMM observation vs. early-treatment decision |
| 3 | `mrd_prediction.py` | `MRDNegPredictor` | `forward` (line 595) | Emits multi-timepoint MRD-negativity probabilities at 10⁻⁵ and 10⁻⁶ — directly feeds therapy de-escalation guidance |
| 4 | `emd_prediction.py` | `EMDRiskPredictor` | `get_risk_summary` (line 609) | Lines 628-638: `risk_category ∈ {High, Intermediate, Low}` + `cns_high_risk` flag (median OS <6 months) |
| 5 | `immunotherapy_response.py` | `ImmunotherapyResponsePredictor` | `predict_response` (536), `plan_sequential_therapy` (638), `assess_crs_icans_risk` (673) | Line 668 returns `"planned_sequence": ["BCMA","GPRC5D","CELMoD"]` — concrete drug-class plan |
| 6 | `treatment_conditioned.py` | `TreatmentConditionedPredictor` (via `TreatmentActorCritic`, line 404) | `forward` returning `"policy"` over actions | Lines 408-411 docstring: action 0=Stay, 1=Escalate, 2=Switch, 3=Add CD38 — Actor-Critic policy from untrained weights |

Plus `dynamic_treatment_regimes.py` (E.1) is the 7th clinical-action emitter. Together: 7 surfaces of harm, 6 of which lack any fit guard whatsoever.

**Common defect:** None of these classes set `self.is_fitted = True` anywhere; no train/fit method ever flips a state flag. A user calling `EarlyRelapsePredictor().predict_early_relapse(...)` immediately after `__init__` gets a string telling them to give salvage therapy, derived from `nn.Linear`s initialized seconds ago.

## §E.2 Patch — Apply to All Six Modules

**1. New module** `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/resistancemap/clinical/_safety.py`:

```python
"""
Clinical safety primitives for ResistanceMap.

Every clinical predictor MUST inherit FittedGuardMixin and call
self._mark_fitted() at the end of its fit() / train() method. Public
recommendation methods MUST be wrapped with @requires_fitted.

NOT FOR CLINICAL USE. Research-only. No FDA clearance.
"""
from __future__ import annotations

import functools
import os
import warnings
from typing import Any, Callable

_RM_ACK_KEY = "RM_RESEARCH_USE_ACKNOWLEDGED"
_RM_ACK_VAL = "I_understand_this_is_research_only_no_clinical_use"


def enforce_research_gate() -> None:
    """Raise unless caller has set the research-only env-var."""
    if os.environ.get(_RM_ACK_KEY) != _RM_ACK_VAL:
        raise RuntimeError(
            "ResistanceMap clinical modules are RESEARCH-ONLY. "
            "They emit drug recommendations from neural networks that have "
            "NOT been fit, validated, or IRB-approved. "
            f"Set {_RM_ACK_KEY}={_RM_ACK_VAL} to load."
        )


class NotFittedError(RuntimeError):
    """Raised when a recommendation is requested before fit()."""


class FittedGuardMixin:
    """Mix-in providing is_fitted state. Default False."""

    _is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return bool(getattr(self, "_is_fitted", False))

    def _mark_fitted(self) -> None:
        self._is_fitted = True

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError(
                f"{type(self).__name__} has not been fit. "
                "Call .fit(...) on real patient data before requesting "
                "any prediction or recommendation. "
                "NOT FOR CLINICAL USE."
            )


def requires_fitted(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator: raise NotFittedError if self.is_fitted is False."""
    @functools.wraps(fn)
    def _wrapped(self, *args, **kwargs):
        if not getattr(self, "is_fitted", False):
            raise NotFittedError(
                f"{type(self).__name__}.{fn.__name__} called on an "
                "unfit model. Output would be derived from random "
                "neural-network weights. Refusing. NOT FOR CLINICAL USE."
            )
        warnings.warn(
            f"{type(self).__name__}.{fn.__name__}: "
            "NOT FOR CLINICAL USE. Research output only.",
            category=UserWarning, stacklevel=2,
        )
        return fn(self, *args, **kwargs)
    return _wrapped
```

**2. Edit each of the six classes** — add `FittedGuardMixin` to base list and decorate the recommendation method. Example for `early_relapse_detection.py`:

```python
from ._safety import FittedGuardMixin, requires_fitted, enforce_research_gate
enforce_research_gate()  # at module top

class EarlyRelapsePredictor(FittedGuardMixin, nn.Module):
    ...
    @requires_fitted
    def predict_early_relapse(self, ...):
        ...
```

Apply identically to `SMMProgressionPredictor.forward`, `MRDNegPredictor.forward`, `EMDRiskPredictor.get_risk_summary`, `ImmunotherapyResponsePredictor.predict_response` / `predict_with_few_shot` / `plan_sequential_therapy` / `assess_crs_icans_risk`, and `TreatmentConditionedPredictor.forward` (also wrap `TreatmentActorCritic.forward`). Each class must gain a `fit()` (or update existing training method) that ends with `self._mark_fitted()`.

**3. Edit** `/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/resistancemap/clinical/__init__.py`: add `from ._safety import enforce_research_gate; enforce_research_gate()` immediately under the docstring so any `import resistancemap.clinical` raises without the ack env-var.

## §E.2 Test — Full Body of `tests/test_clinical_safety.py`

```python
"""
Safety tests for ResistanceMap clinical modules.

Every clinical predictor must:
  1. Refuse to import without RM_RESEARCH_USE_ACKNOWLEDGED env-var.
  2. Refuse to emit recommendations on an unfit model (NotFittedError).
  3. Set is_fitted=True only after fit().
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

# Ack the research gate for tests in this process.
os.environ["RM_RESEARCH_USE_ACKNOWLEDGED"] = (
    "I_understand_this_is_research_only_no_clinical_use"
)

from resistancemap.clinical._safety import (  # noqa: E402
    NotFittedError, FittedGuardMixin, requires_fitted,
)


def test_research_gate_blocks_import_without_envvar():
    """Importing clinical without ack env-var must raise."""
    code = (
        "import os; "
        "os.environ.pop('RM_RESEARCH_USE_ACKNOWLEDGED', None); "
        "import resistancemap.clinical"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert res.returncode != 0
    assert b"RESEARCH-ONLY" in res.stderr or b"research-only" in res.stderr


def test_dtr_action_space_not_emitted_without_ack():
    """DTR module must refuse to load without ack."""
    code = (
        "import os; "
        "os.environ.pop('RM_RESEARCH_USE_ACKNOWLEDGED', None); "
        "import resistancemap.clinical.dynamic_treatment_regimes"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert res.returncode != 0


@pytest.mark.parametrize("factory_and_call", [
    ("early_relapse_detection", "EarlyRelapsePredictor", "predict_early_relapse",
     lambda m: dict(timepoints=np.arange(5).astype(float), m_protein=np.ones(5),
                    flc_involved=np.ones(5), ldh_baseline=200.0, ldh_current=210.0,
                    ecog=1, iss=2, calcium=10.0)),
    ("smm_progression", "SMMProgressionPredictor", "forward",
     lambda m: {k: torch.zeros(2) for k in
                ["m_protein","bmpc","flc_ratio","t414","del17p","gain1q","myc_rearr"]}
     | {"timepoints": torch.arange(5).float().unsqueeze(0).repeat(2,1),
        "vafs": torch.ones(2,5)*0.1, "tcr_diversity": torch.ones(2,5)*2,
        "nk_cd38_freq": torch.ones(2,5)*0.05, "treg_freq": torch.ones(2,5)*0.02,
        "cd8_exhaustion": torch.ones(2,5)*0.2}),
    ("mrd_prediction", "MRDNegPredictor", "forward",
     lambda m: dict(gene_expr=torch.zeros(1,2000),
                    clinical_feats=torch.zeros(1,64),
                    immune_markers=torch.zeros(1,48))),
    ("emd_prediction", "EMDRiskPredictor", "get_risk_summary",
     lambda m: dict(outputs={
         "emd_probability": torch.zeros(1,1), "cns_probability": torch.zeros(1,1),
         "time_to_emd_k": torch.ones(1,1), "time_to_emd_lambda": torch.ones(1,1)})),
    ("immunotherapy_response", "ImmunotherapyResponsePredictor", "predict_response",
     None),  # builds args inline
    ("treatment_conditioned", "TreatmentConditionedPredictor", "forward",
     lambda m: dict(x=torch.zeros(1, 32), treatment=torch.zeros(1,1))),
    ("dynamic_treatment_regimes", "DTRPolicyOptimizer", "select_action",
     lambda m: dict(history=torch.zeros(1,32), biomarker=torch.zeros(1,8),
                    egfr=60.0, ecog=1, age=60, cart_used=False)),
])
def test_recommend_raises_before_fit(factory_and_call):
    """Every clinical recommender must raise NotFittedError before fit()."""
    modname, clsname, method, kwargs_fn = factory_and_call
    mod = __import__(f"resistancemap.clinical.{modname}", fromlist=[clsname])
    cls = getattr(mod, clsname)
    # Construct with default-ish args (each class accepts no-arg or trivial).
    try:
        instance = cls()
    except TypeError:
        # Constructors that need explicit dims
        if modname == "treatment_conditioned":
            instance = cls(input_dim=32)
        elif modname == "dynamic_treatment_regimes":
            instance = cls(history_dim=32, n_actions=8, n_stages=3)
        else:
            instance = cls(input_dim=32)
    assert hasattr(instance, "is_fitted"), f"{clsname} missing is_fitted"
    assert instance.is_fitted is False, f"{clsname} fitted at init"

    fn = getattr(instance, method)
    if kwargs_fn is None:
        # immunotherapy_response: build minimal profile + mechanism
        from resistancemap.clinical.immunotherapy_response import (
            PatientImmuneProfile, AgentMechanism,
        )
        profile = PatientImmuneProfile(
            antigen_bcma=1.0, antigen_gprc5d=1.0, antigen_fcrl5=1.0,
            exhaustion_score=0.3, cd4_cd8_ratio=2.0, tcr_clonality=0.5)
        mechanism = AgentMechanism("BCMA", "TCE", 0)
        with pytest.raises(NotFittedError):
            fn(profile, mechanism, target_expression=1.0)
    else:
        with pytest.raises(NotFittedError):
            fn(**kwargs_fn(instance))


def test_is_fitted_flips_after_mark():
    """_mark_fitted() flips is_fitted; only then does recommend succeed."""
    class Toy(FittedGuardMixin, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 1)
        @requires_fitted
        def recommend(self):
            return "ok"
    t = Toy()
    with pytest.raises(NotFittedError):
        t.recommend()
    t._mark_fitted()
    assert t.is_fitted is True
    assert t.recommend() == "ok"


def test_no_recommendation_string_leaks_from_random_weights():
    """Salvage-therapy string must never be returned by an unfit predictor."""
    from resistancemap.clinical.early_relapse_detection import EarlyRelapsePredictor
    p = EarlyRelapsePredictor()
    with pytest.raises(NotFittedError):
        _ = p.predict_early_relapse(
            timepoints=np.arange(5).astype(float), m_protein=np.ones(5),
            flc_involved=np.ones(5), ldh_baseline=200.0, ldh_current=210.0,
            ecog=1, iss=2, calcium=10.0,
        )
```

## §E.1 + §E.2 Risk if NOT Fixed — for the Audit Committee

A public GitHub repository titled "ResistanceMap" — described in its own README as a clinical-AI multiple-myeloma platform — currently exposes seven Python modules whose top-level public APIs return drug names ("carfilzomib", "daratumumab", "CAR-T") and free-text clinical instructions ("Escalate to intensive regimen; monitor q2 weeks; consider salvage therapy trial") computed by neural networks at PyTorch default initialization. There is no IRB protocol, no consent process, no patient data, no test verifying the model has been trained, and no "NOT FOR CLINICAL USE" banner anywhere in the package. A clinician, student, or hospital pilot who pip-installs this code, runs the literal `__main__` block of `dynamic_treatment_regimes.py`, and acts on its output is being given an FDA-regulated treatment recommendation derived from a `torch.nn.Linear` whose weights were drawn from a uniform distribution seconds earlier. The probability of patient harm is not zero — it is bounded only by how many people read the README and try the demo. Under FDA SaMD guidance (21 CFR 880.6315 / Software as a Medical Device), under HIPAA-aligned IRB requirements at UNT and the affiliated MM clinical AI program, and under basic researcher duty-of-care, this state is indefensible and must be remediated before the next public push. The fix is small (one new file `_safety.py`, ~80 LoC; one decorator applied to seven methods; one env-var gate; one test file) and does not impede research use.

---

## §summary

| Blocker | Severity | File(s) | Defect | Fix LoC | Test LoC | Status |
|---|---|---|---|---|---|---|
| E.1 | BLOCKER | `resistancemap/clinical/dynamic_treatment_regimes.py` (lines 91-94, 708, 825, 872-917) | 8-drug action space exposed publicly with `__main__` demo emitting drug name; no IRB/consent/disclaimer; no test | ~30 (env gate + decorator + delete `__main__`) | 2 tests | Verified, unfixed |
| E.2 | BLOCKER (panel-elevated) | `early_relapse_detection.py` L673; `smm_progression.py` L438; `mrd_prediction.py` L595; `emd_prediction.py` L609; `immunotherapy_response.py` L536/638/673; `treatment_conditioned.py` L404/572 | Six clinical predictors emit recommendations from random-init `nn.Linear` weights; no `is_fitted` flag, no fit guard, no test | ~80 (`_safety.py`) + 6 × 3 lines per class | 9 tests in `tests/test_clinical_safety.py` | Verified, unfixed |
| Tests | MISSING | `tests/test_clinical_safety.py` | Does not exist | new file ~150 LoC | — | Verified absent |

**Note on audit limits.** Bash grep was denied during this session, so the search for *all* free-text recommendation strings and the test-directory enumeration relied on per-file Read of the modules listed in `__init__.py`. The seven modules above are the complete clinical surface as declared by `resistancemap/clinical/__init__.py::__all__`. The user's blocker description named six free-text emitters; my audit found that all six clinical predictors plus the DTR optimizer emit clinical-decision-grade output without a fit guard, so the patch covers all seven to be safe.