"""
resistancemap/baselines/static_ml.py
====================================
Static-ML baseline panel (sklearn / xgboost / lightgbm wrappers) implementing
the :class:`BaselineModel` contract.

Each wrapper:
  * Lazily imports its backend inside ``__init__`` so the module can be
    imported (and the registry populated) even when an optional dependency
    like ``xgboost`` is missing — the import error is surfaced at
    *instantiation* time instead, with a clear actionable message.
  * Concatenates the modality dict ``X`` into a 2-D feature matrix in a
    stable, sorted-by-modality-name order, so the same X always produces
    the same feature ordering for the same seed.
  * Uses ``joblib`` for round-trip ``save`` / ``load``.

Auto-registration: at the bottom of this module each baseline is added to
``REGISTRY`` so simply importing the module makes them all reachable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from resistancemap.baselines.registry import REGISTRY


def _concat_modalities(X: Mapping[str, np.ndarray]) -> np.ndarray:
    """Concatenate modality arrays along axis=1 in sorted-key order.

    Stable ordering matters: two runs with the same X must produce the same
    feature matrix and thus (with the same seed) the same fitted weights.
    """
    if not X:
        raise ValueError("static_ml baselines require at least one modality")
    keys = sorted(X)
    arrays = [np.asarray(X[k]) for k in keys]
    # All modalities must agree on the row count.
    n_rows = arrays[0].shape[0]
    for k, a in zip(keys, arrays):
        if a.shape[0] != n_rows:
            raise ValueError(
                f"modality {k!r} has {a.shape[0]} rows, expected {n_rows}"
            )
        if a.ndim != 2:
            raise ValueError(f"modality {k!r} must be 2-D, got shape {a.shape}")
    return np.concatenate(arrays, axis=1)


class _SklearnBaseline:
    """Shared scaffolding for the sklearn-backed static baselines."""

    name: str = "_sklearn_base"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config: Dict[str, Any] = dict(config)
        self.seed: int = int(self.config.get("seed", 0))
        self._model = None  # set by subclass __init__

    def fit(
        self,
        X_train: Mapping[str, np.ndarray],
        y_train: np.ndarray,
        **kwargs: Any,
    ) -> None:
        X = _concat_modalities(X_train)
        self._model.fit(X, np.asarray(y_train))

    def predict(self, X_test: Mapping[str, np.ndarray]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.name}: model not fitted")
        return np.asarray(self._model.predict(_concat_modalities(X_test)))

    def predict_survival(
        self, X_test: Mapping[str, np.ndarray]
    ) -> Optional[Dict[str, np.ndarray]]:
        return None  # static-ML baselines are not survival-capable.

    def save(self, path) -> None:
        import joblib
        joblib.dump({"model": self._model, "config": self.config}, str(path))

    def load(self, path) -> None:
        import joblib
        payload = joblib.load(str(path))
        self._model = payload["model"]
        self.config = payload.get("config", self.config)


class ElasticNetBaseline(_SklearnBaseline):
    """sklearn ElasticNet regressor."""

    name = "elasticnet"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        from sklearn.linear_model import ElasticNet  # lazy import
        self._model = ElasticNet(
            alpha=float(self.config.get("alpha", 1.0)),
            l1_ratio=float(self.config.get("l1_ratio", 0.5)),
            max_iter=int(self.config.get("max_iter", 1000)),
            random_state=self.seed,
        )


class RidgeBaseline(_SklearnBaseline):
    """sklearn Ridge regressor."""

    name = "ridge"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        from sklearn.linear_model import Ridge  # lazy import
        self._model = Ridge(
            alpha=float(self.config.get("alpha", 1.0)),
            random_state=self.seed,
        )


class RandomForestBaseline(_SklearnBaseline):
    """sklearn RandomForestRegressor."""

    name = "random_forest"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        from sklearn.ensemble import RandomForestRegressor  # lazy import
        self._model = RandomForestRegressor(
            n_estimators=int(self.config.get("n_estimators", 200)),
            max_depth=self.config.get("max_depth"),
            n_jobs=int(self.config.get("n_jobs", 1)),
            random_state=self.seed,
        )


class XGBoostBaseline(_SklearnBaseline):
    """xgboost.XGBRegressor wrapper. Raises ImportError if xgboost is absent.

    Optional dependency: ``pip install xgboost``.
    """

    name = "xgboost"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        try:
            import xgboost as xgb  # lazy import
        except ImportError as exc:
            raise ImportError(
                "XGBoostBaseline requires xgboost; install with `pip install xgboost`"
            ) from exc
        self._model = xgb.XGBRegressor(
            n_estimators=int(self.config.get("n_estimators", 200)),
            max_depth=int(self.config.get("max_depth", 6)),
            learning_rate=float(self.config.get("learning_rate", 0.1)),
            n_jobs=int(self.config.get("n_jobs", 1)),
            random_state=self.seed,
            tree_method="hist",
        )


class LightGBMBaseline(_SklearnBaseline):
    """lightgbm.LGBMRegressor wrapper. Raises ImportError if lightgbm is absent.

    Optional dependency: ``pip install lightgbm``.
    """

    name = "lightgbm"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        try:
            import lightgbm as lgb  # lazy import
        except ImportError as exc:
            raise ImportError(
                "LightGBMBaseline requires lightgbm; install with `pip install lightgbm`"
            ) from exc
        self._model = lgb.LGBMRegressor(
            n_estimators=int(self.config.get("n_estimators", 200)),
            max_depth=int(self.config.get("max_depth", -1)),
            learning_rate=float(self.config.get("learning_rate", 0.1)),
            n_jobs=int(self.config.get("n_jobs", 1)),
            random_state=self.seed,
            verbose=-1,
        )


# ---------------------------------------------------------------------------
# Auto-register at import.
# ---------------------------------------------------------------------------

for _cls in (ElasticNetBaseline, RidgeBaseline, RandomForestBaseline):
    # Use default-argument trick to capture _cls by value, not by reference.
    REGISTRY.register(_cls.name, lambda cfg, _c=_cls: _c(cfg))


def _register_optional(name: str, cls) -> None:
    """Register an optional-dep baseline; instantiation will raise if dep absent.

    The registry entry still exists so ``REGISTRY.list_available()`` shows the
    baseline; calling ``instantiate(name, ...)`` will surface the ImportError.
    """
    REGISTRY.register(name, lambda cfg, _c=cls: _c(cfg))


_register_optional(XGBoostBaseline.name, XGBoostBaseline)
_register_optional(LightGBMBaseline.name, LightGBMBaseline)
