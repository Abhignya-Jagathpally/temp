"""
resistancemap/baselines/sequence.py
===================================
Longitudinal-aware baselines: LSTM and GRU over the per-patient visit
sequence.

Input contract (see ``phase_10_baselines.md`` §3.3):
The modality dict X may carry a 3-D ``visits`` array of shape
``(N, T_max, F)`` plus an optional ``visit_mask`` ``(N, T_max)`` bool array.
If only 2-D modalities are present (no temporal axis) the baseline treats
each row as a sequence of length 1 — this keeps the contract symmetric with
the static-ML baselines.

Honest-behaviour invariants
---------------------------
* Random nn.Module weight init is the only stochasticity.
* The visit_mask field is **not** invented if absent; we assume every visit
  in the input is real.
* No np.random.* and no fabricated metric values anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from resistancemap.baselines.registry import REGISTRY


def _extract_sequence(X: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return ``(visits, mask)`` from the modality dict.

    Priority order:
      1. If ``visits`` is present, use it (and ``visit_mask`` if also present).
      2. Else concatenate every modality (sorted-key order) and add a
         time axis of length 1.
    """
    if "visits" in X:
        visits = np.asarray(X["visits"], dtype=np.float32)
        if visits.ndim != 3:
            raise ValueError(
                f"'visits' modality must be 3-D (N, T, F), got shape {visits.shape}"
            )
        mask = X.get("visit_mask")
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != visits.shape[:2]:
                raise ValueError(
                    f"visit_mask shape {mask.shape} != visits shape "
                    f"{visits.shape[:2]}"
                )
        return visits, mask
    keys = sorted(k for k in X if k not in ("visit_mask",))
    if not keys:
        raise ValueError("sequence baselines require at least one modality")
    arrs = [np.asarray(X[k], dtype=np.float32) for k in keys]
    n = arrs[0].shape[0]
    for k, a in zip(keys, arrs):
        if a.shape[0] != n:
            raise ValueError(f"modality {k!r} has {a.shape[0]} rows, expected {n}")
        if a.ndim != 2:
            raise ValueError(f"modality {k!r} must be 2-D, got shape {a.shape}")
    flat = np.concatenate(arrs, axis=1)
    return flat[:, None, :], None  # add a length-1 time axis


def _seeded_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


class _RecurrentBaseline:
    """Shared scaffolding for LSTM/GRU visit baselines."""

    name: str = "_rnn_base"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config: Dict[str, Any] = dict(config)
        self.seed: int = int(self.config.get("seed", 0))
        self.lr: float = float(self.config.get("lr", 1e-3))
        self.epochs: int = int(self.config.get("epochs", 30))
        self.batch_size: int = int(self.config.get("batch_size", 32))
        self.hidden: int = int(self.config.get("hidden", 64))
        self.n_layers: int = int(self.config.get("n_layers", 1))
        self.dropout: float = float(self.config.get("dropout", 0.0))
        self.device: torch.device = torch.device(self.config.get("device", "cpu"))
        torch.manual_seed(self.seed)
        self.module: Optional[nn.Module] = None
        self._in_dim: Optional[int] = None

    def _make_rnn(self, in_dim: int) -> nn.Module:
        raise NotImplementedError

    def _ensure_module(self, in_dim: int) -> None:
        if self.module is None or self._in_dim != in_dim:
            torch.manual_seed(self.seed)
            self.module = self._make_rnn(in_dim).to(self.device)
            self._in_dim = in_dim

    def fit(
        self,
        X_train: Mapping[str, np.ndarray],
        y_train: np.ndarray,
        **kwargs: Any,
    ) -> None:
        visits, mask = _extract_sequence(X_train)
        y = np.asarray(y_train, dtype=np.float32).reshape(-1)
        self._ensure_module(visits.shape[-1])
        assert self.module is not None
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        x_t = torch.from_numpy(visits).to(self.device)
        y_t = torch.from_numpy(y).to(self.device)
        mask_t = (
            torch.from_numpy(mask).to(self.device)
            if mask is not None else None
        )
        n = x_t.shape[0]
        gen = _seeded_generator(self.seed)
        self.module.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                batch_x = x_t[idx]
                batch_m = mask_t[idx] if mask_t is not None else None
                pred = self.module((batch_x, batch_m)).reshape(-1)
                loss = F.mse_loss(pred, y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict(self, X_test: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.module is None:
            raise RuntimeError(f"{self.name}: model not fitted")
        visits, mask = _extract_sequence(X_test)
        self.module.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(visits).to(self.device)
            m_t = torch.from_numpy(mask).to(self.device) if mask is not None else None
            pred = self.module((x_t, m_t)).reshape(-1).cpu().numpy()
        return pred

    def predict_survival(
        self, X_test: Mapping[str, np.ndarray]
    ) -> Optional[Dict[str, np.ndarray]]:
        return None

    def save(self, path) -> None:
        if self.module is None:
            raise RuntimeError(f"{self.name}: model not fitted")
        torch.save(
            {
                "state_dict": self.module.state_dict(),
                "config": self.config,
                "in_dim": self._in_dim,
            },
            str(path),
        )

    def load(self, path) -> None:
        payload = torch.load(str(path), map_location=self.device, weights_only=False)
        self.config = payload.get("config", self.config)
        self._in_dim = int(payload["in_dim"])
        self._ensure_module(self._in_dim)
        assert self.module is not None
        self.module.load_state_dict(payload["state_dict"])


class _LSTMVisitModule(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, payload):
        x, mask = payload
        out, _ = self.rnn(x)
        if mask is not None:
            # Use the last valid (mask True) timestep per row.
            lengths = mask.sum(dim=1).clamp(min=1).long() - 1
            idx = lengths.view(-1, 1, 1).expand(-1, 1, out.shape[-1])
            last = out.gather(1, idx).squeeze(1)
        else:
            last = out[:, -1, :]
        return self.head(last)


class LSTMVisitBaseline(_RecurrentBaseline):
    """LSTM over per-patient visit sequence."""

    name = "lstm_visits"

    def _make_rnn(self, in_dim: int) -> nn.Module:
        return _LSTMVisitModule(in_dim, self.hidden, self.n_layers, self.dropout)


class _GRUVisitModule(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=in_dim, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, payload):
        x, mask = payload
        out, _ = self.rnn(x)
        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1).long() - 1
            idx = lengths.view(-1, 1, 1).expand(-1, 1, out.shape[-1])
            last = out.gather(1, idx).squeeze(1)
        else:
            last = out[:, -1, :]
        return self.head(last)


class GRUVisitBaseline(_RecurrentBaseline):
    """GRU over per-patient visit sequence."""

    name = "gru_visits"

    def _make_rnn(self, in_dim: int) -> nn.Module:
        return _GRUVisitModule(in_dim, self.hidden, self.n_layers, self.dropout)


# ---------------------------------------------------------------------------
# Auto-register.
# ---------------------------------------------------------------------------


for _cls in (LSTMVisitBaseline, GRUVisitBaseline):
    REGISTRY.register(_cls.name, lambda cfg, _c=_cls: _c(cfg))
