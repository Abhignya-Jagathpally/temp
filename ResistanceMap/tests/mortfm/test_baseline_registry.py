"""tests/mortfm/test_baseline_registry.py

Phase 10 baseline registry — round-trip + determinism + import-skip tests.

The decisive contracts:
  1. ``REGISTRY.list_available()`` returns the names registered by the
     ``static_ml``, ``deep_static`` and ``sequence`` modules at import time.
  2. Every name resolves to a concrete class implementing ``fit / predict``.
  3. Two baselines fitted on identical (seed, data) reproduce identical
     predictions bit-for-bit (determinism gate).
  4. Optional-dep baselines (xgboost, lightgbm) skip cleanly when the
     backend is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Side-effect: each import registers its baselines against REGISTRY.
import resistancemap.baselines.deep_static  # noqa: F401
import resistancemap.baselines.sequence     # noqa: F401
import resistancemap.baselines.static_ml    # noqa: F401
from resistancemap.baselines.registry import REGISTRY, BaselineModel


# --------------------------------------------------------------------------
# Fixture data (NOT metrics — just shapes for the contract tests).
# --------------------------------------------------------------------------


@pytest.fixture()
def synthetic_static_xy():
    """A tiny (N=24, F=10) Gaussian-RNG dataset used purely as input for
    contract tests. The y vector is a deterministic linear function of X so
    "did the model learn something" is not the question — "did the wrapper
    fit / predict cleanly" is."""
    rng = np.random.default_rng(seed=12345)
    n, f = 24, 10
    X = rng.standard_normal((n, f)).astype(np.float32)
    w = np.linspace(-1.0, 1.0, f, dtype=np.float32)
    y = (X @ w).astype(np.float32)
    return {"rna": X}, y


@pytest.fixture()
def synthetic_sequence_xy():
    """3-D ``visits`` array, T=3 visits per patient, F=6 features."""
    rng = np.random.default_rng(seed=67890)
    n, t, f = 20, 3, 6
    X = rng.standard_normal((n, t, f)).astype(np.float32)
    y = X.sum(axis=(1, 2)).astype(np.float32)
    return {"visits": X}, y


# --------------------------------------------------------------------------
# Registry-level contract.
# --------------------------------------------------------------------------


def test_registry_lists_static_ml_baselines():
    names = REGISTRY.list_available()
    for expected in ("elasticnet", "ridge", "random_forest"):
        assert expected in names, f"static_ml baseline {expected!r} not registered"


def test_registry_lists_deep_static_baselines():
    names = REGISTRY.list_available()
    for expected in ("mlp", "cnn_feature_order", "ft_transformer"):
        assert expected in names, f"deep_static baseline {expected!r} not registered"


def test_registry_lists_sequence_baselines():
    names = REGISTRY.list_available()
    for expected in ("lstm_visits", "gru_visits"):
        assert expected in names, f"sequence baseline {expected!r} not registered"


def test_registry_resolves_to_baseline_contract():
    """Every registered name (other than optional-dep ones we may not have
    installed) must resolve to an object implementing the BaselineModel
    contract — fit, predict, predict_survival, save, load, name."""
    skip_if_missing = {"xgboost": "xgboost", "lightgbm": "lightgbm"}
    for name in REGISTRY.list_available():
        if name in skip_if_missing:
            pytest.importorskip(skip_if_missing[name])
        model = REGISTRY.instantiate(name, {"seed": 0})
        for required in ("fit", "predict", "predict_survival", "save", "load", "name"):
            assert hasattr(model, required), (
                f"baseline {name!r} ({type(model).__name__}) lacks attribute "
                f"{required!r} — BaselineModel contract violation"
            )
        assert model.name == name, (
            f"baseline {name!r} has .name = {model.name!r} (must match registry key)"
        )


# --------------------------------------------------------------------------
# Determinism — same seed + same data => identical predictions.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["elasticnet", "ridge", "random_forest"])
def test_static_ml_determinism(name, synthetic_static_xy):
    X, y = synthetic_static_xy
    a = REGISTRY.instantiate(name, {"seed": 42})
    b = REGISTRY.instantiate(name, {"seed": 42})
    a.fit(X, y)
    b.fit(X, y)
    pa = a.predict(X)
    pb = b.predict(X)
    assert pa.shape == (X["rna"].shape[0],)
    np.testing.assert_array_almost_equal(pa, pb, decimal=10)


@pytest.mark.parametrize("name", ["mlp", "cnn_feature_order", "ft_transformer"])
def test_deep_static_determinism(name, synthetic_static_xy):
    X, y = synthetic_static_xy
    a = REGISTRY.instantiate(name, {"seed": 7, "epochs": 2, "batch_size": 8})
    b = REGISTRY.instantiate(name, {"seed": 7, "epochs": 2, "batch_size": 8})
    a.fit(X, y)
    b.fit(X, y)
    pa = a.predict(X)
    pb = b.predict(X)
    assert pa.shape == (X["rna"].shape[0],)
    # Float32 tolerance: torch determinism on CPU is bit-exact in eager mode
    # for these tiny models, but allow a small slack for cuDNN-free CPU paths.
    np.testing.assert_allclose(pa, pb, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", ["lstm_visits", "gru_visits"])
def test_sequence_determinism(name, synthetic_sequence_xy):
    X, y = synthetic_sequence_xy
    a = REGISTRY.instantiate(name, {"seed": 3, "epochs": 2, "batch_size": 8})
    b = REGISTRY.instantiate(name, {"seed": 3, "epochs": 2, "batch_size": 8})
    a.fit(X, y)
    b.fit(X, y)
    pa = a.predict(X)
    pb = b.predict(X)
    assert pa.shape == (X["visits"].shape[0],)
    np.testing.assert_allclose(pa, pb, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# Optional-dep gating.
# --------------------------------------------------------------------------


def test_xgboost_baseline_or_skip():
    pytest.importorskip("xgboost")
    model = REGISTRY.instantiate("xgboost", {"seed": 0, "n_estimators": 10})
    assert model.name == "xgboost"


def test_lightgbm_baseline_or_skip():
    pytest.importorskip("lightgbm")
    model = REGISTRY.instantiate("lightgbm", {"seed": 0, "n_estimators": 10})
    assert model.name == "lightgbm"


# --------------------------------------------------------------------------
# Round-trip save / load.
# --------------------------------------------------------------------------


def test_static_ml_save_load_roundtrip(tmp_path, synthetic_static_xy):
    X, y = synthetic_static_xy
    a = REGISTRY.instantiate("ridge", {"seed": 0})
    a.fit(X, y)
    pa = a.predict(X)
    ckpt = tmp_path / "ridge.joblib"
    a.save(ckpt)

    b = REGISTRY.instantiate("ridge", {"seed": 0})
    b.load(ckpt)
    pb = b.predict(X)
    np.testing.assert_array_almost_equal(pa, pb, decimal=10)


def test_deep_static_save_load_roundtrip(tmp_path, synthetic_static_xy):
    X, y = synthetic_static_xy
    a = REGISTRY.instantiate("mlp", {"seed": 0, "epochs": 1})
    a.fit(X, y)
    pa = a.predict(X)
    ckpt = tmp_path / "mlp.pt"
    a.save(ckpt)

    b = REGISTRY.instantiate("mlp", {"seed": 0, "epochs": 1})
    b.load(ckpt)
    pb = b.predict(X)
    np.testing.assert_allclose(pa, pb, rtol=1e-5, atol=1e-6)
