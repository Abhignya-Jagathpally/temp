"""
resistancemap/baselines/registry.py
===================================
Central baseline registry for MORT-FM Phase 10.

Today the v6 baselines (``classical_baselines.py``,
``integration_baselines.py``, ``perception_baseline.py``, ``cpa_baseline.py``)
live as ad-hoc classes with bespoke ``fit / predict`` signatures. The Phase
10 plan calls for a *uniform contract* so the same evaluator can drive
elasticnet, RSF, an LSTM-on-visits, or a graph baseline against the same
splits and write rows into the same parquet table.

This module provides that contract.

Contract
--------
Every registered baseline implements :class:`BaselineModel`:

* ``fit(X_train, y_train, **kwargs)`` — train on a dict of modality arrays
  + a 1-D target array.
* ``predict(X_test) -> np.ndarray`` — return point predictions on the same
  modality dict shape used at fit time.
* ``predict_survival(X_test) -> dict | None`` — return ``{"risk": np.ndarray,
  "time_grid": np.ndarray, "survival": np.ndarray}`` if the model supports
  time-to-event, else ``None``.
* ``save(path: Path)`` / ``load(path: Path)`` — round-trip the fitted
  state.
* ``name -> str`` — a stable string identifier (also the registry key).

Registration is module-level: each ``static_ml.py`` / ``deep_static.py`` /
``sequence.py`` ends with ``REGISTRY.register("name", factory)`` so the
registry is populated by import side-effect. Hard-import the baseline module
once and every baseline it defines becomes available.

This module makes NO calls to ``np.random`` / ``torch.randn`` / hardcoded
metrics. Random initialization of ``nn.Module`` weights happens inside the
individual baseline ``__init__`` only (and is gated by an explicit seed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Union, runtime_checkable

import numpy as np


PathLike = Union[str, Path]


@runtime_checkable
class BaselineModel(Protocol):
    """The contract every registered baseline must satisfy."""

    #: Stable string identifier; also the registry key.
    name: str

    def fit(
        self,
        X_train: Mapping[str, np.ndarray],
        y_train: np.ndarray,
        **kwargs: Any,
    ) -> None:
        ...

    def predict(self, X_test: Mapping[str, np.ndarray]) -> np.ndarray:
        ...

    def predict_survival(
        self, X_test: Mapping[str, np.ndarray]
    ) -> Optional[Dict[str, np.ndarray]]:
        ...

    def save(self, path: PathLike) -> None:
        ...

    def load(self, path: PathLike) -> None:
        ...


Factory = Callable[[Mapping[str, Any]], BaselineModel]


class BaselineRegistry:
    """Name -> factory map for MORT-FM baseline panel.

    Designed as an *instance* (singleton ``REGISTRY``) rather than a classmethod
    bucket so test isolation is easy: tests can construct a fresh
    :class:`BaselineRegistry` and exercise it without polluting global state.
    """

    def __init__(self) -> None:
        self._factories: Dict[str, Factory] = {}

    def register(
        self,
        name: str,
        factory: Factory,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register ``factory`` under ``name``.

        Parameters
        ----------
        name : str
            Stable identifier (must match the baseline's ``name`` attribute).
        factory : callable
            ``factory(config: dict) -> BaselineModel`` constructor wrapper.
        overwrite : bool
            If ``False`` (default) a ``KeyError`` is raised when ``name`` is
            already registered. Set ``True`` only for tests / hot-reload.
        """
        if not overwrite and name in self._factories:
            raise KeyError(
                f"BaselineRegistry: {name!r} is already registered "
                f"(use overwrite=True to replace)"
            )
        self._factories[name] = factory

    def list_available(self) -> List[str]:
        """Return a sorted list of every registered baseline name."""
        return sorted(self._factories)

    def instantiate(self, name: str, config: Mapping[str, Any]) -> BaselineModel:
        """Build the baseline named ``name`` from ``config``.

        Raises
        ------
        KeyError
            ``name`` is not registered.
        TypeError
            The factory returned an object that does not satisfy
            :class:`BaselineModel` (missing ``fit``/``predict``/etc.).
        """
        if name not in self._factories:
            raise KeyError(
                f"BaselineRegistry: unknown baseline {name!r}; "
                f"available = {self.list_available()}"
            )
        model = self._factories[name](dict(config))
        for required in ("fit", "predict", "predict_survival", "save", "load", "name"):
            if not hasattr(model, required):
                raise TypeError(
                    f"BaselineRegistry: factory for {name!r} returned "
                    f"{type(model).__name__} which lacks required attribute "
                    f"{required!r} (BaselineModel contract)"
                )
        return model

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def clear(self) -> None:
        """Drop every registered baseline (test-only helper)."""
        self._factories.clear()


#: Module-level singleton — every baseline file registers against this.
REGISTRY = BaselineRegistry()
