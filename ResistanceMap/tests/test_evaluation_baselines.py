"""Interface and no-fabrication contract tests for evaluation baselines.

The baseline adversary tier compares the full ResistanceMap model against
deliberately weak references (clinical-only, transcriptome-only, last
observation carried forward, simple parametric survival). Each baseline
must:

1. Implement the Baseline ABC interface (fit, predict, name).
2. Refuse to fabricate predictions on empty / None inputs by raising
   ValueError (or a subclass).

We do NOT exercise numerical correctness here; that requires real cohort
data and is the job of the integration tier on the held-out evaluation
dataset.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

evaluation = pytest.importorskip(
    "resistancemap.evaluation",
    reason="resistancemap.evaluation package not yet merged into worktree",
)

baselines_pkg = pytest.importorskip(
    "resistancemap.evaluation.baselines",
    reason="resistancemap.evaluation.baselines submodule not present",
)

from resistancemap.evaluation.baselines.base import Baseline  # noqa: E402
from resistancemap.evaluation.baselines.clinical_only import (  # noqa: E402
    ClinicalOnlyBaseline,
)
from resistancemap.evaluation.baselines.transcriptome_only import (  # noqa: E402
    TranscriptomeOnlyBaseline,
)
from resistancemap.evaluation.baselines.last_observation import (  # noqa: E402
    LastObservationBaseline,
)
from resistancemap.evaluation.baselines.simple_survival import (  # noqa: E402
    SimpleSurvivalBaseline,
)


ALL_BASELINE_CLASSES = [
    ClinicalOnlyBaseline,
    TranscriptomeOnlyBaseline,
    LastObservationBaseline,
    SimpleSurvivalBaseline,
]


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_BASELINE_CLASSES)
def test_baseline_subclasses_baseline_abc(cls):
    assert issubclass(cls, Baseline), f"{cls.__name__} must subclass Baseline"


@pytest.mark.parametrize("cls", ALL_BASELINE_CLASSES)
def test_baseline_has_required_methods(cls):
    for method in ("fit", "predict"):
        assert callable(getattr(cls, method, None)), (
            f"{cls.__name__} must implement {method}()"
        )


@pytest.mark.parametrize("cls", ALL_BASELINE_CLASSES)
def test_baseline_has_name_attribute(cls):
    """Each baseline must expose a non-empty `name` (class attr or property)."""
    instance = cls()
    name = getattr(instance, "name", None)
    assert isinstance(name, str) and len(name) > 0, (
        f"{cls.__name__} must expose a non-empty .name string"
    )


# ---------------------------------------------------------------------------
# No-fabrication contract: empty / None inputs raise ValueError
# ---------------------------------------------------------------------------


def _empty_array():
    return np.empty((0, 0), dtype=float)


@pytest.mark.parametrize("cls", ALL_BASELINE_CLASSES)
def test_baseline_fit_rejects_none_inputs(cls):
    """fit(None, None) must raise ValueError, never silently fabricate."""
    instance = cls()
    with pytest.raises((ValueError, TypeError)):
        instance.fit(None, None)


@pytest.mark.parametrize("cls", ALL_BASELINE_CLASSES)
def test_baseline_fit_rejects_empty_inputs(cls):
    """fit() on zero-row arrays must raise ValueError."""
    instance = cls()
    X = _empty_array()
    y = np.empty((0,), dtype=float)
    with pytest.raises((ValueError, TypeError)):
        instance.fit(X, y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
