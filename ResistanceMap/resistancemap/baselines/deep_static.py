"""
resistancemap/baselines/deep_static.py
======================================
Deep-tabular baselines that operate on the static feature concatenation
``X = np.concat([X['rna'], X['clinical'], ...], axis=1)``.

Three wrappers — all implementing the ``BaselineModel`` contract:

* :class:`MLPBaseline` — 3-layer fully-connected regressor.
* :class:`CNNFeatureOrderBaseline` — 1-D CNN over the feature axis.
* :class:`FTTransformerBaseline` — minimal FT-Transformer-lite (token
  embeddings + a single self-attention block).

These three constitute the "depth-matched non-ablated tabular DL" comparators
listed in `phase_10_baselines.md` §3.2 and the ablation list in §6.

Honest-behaviour invariants
---------------------------
* Standard nn.Module random weight initialization is the only stochasticity.
* No ``np.random`` / fake metrics ever.
* Training uses Adam + MSE; the train loop is fixed-epoch (no hidden
  validation-based early stop that could leak the test split).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from resistancemap.baselines.registry import REGISTRY


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _concat_modalities(X: Mapping[str, np.ndarray]) -> np.ndarray:
    """Stable-order modality concatenation. See ``static_ml._concat_modalities``."""
    if not X:
        raise ValueError("deep_static baselines require at least one modality")
    keys = sorted(X)
    arrays = [np.asarray(X[k]) for k in keys]
    n_rows = arrays[0].shape[0]
    for k, a in zip(keys, arrays):
        if a.shape[0] != n_rows:
            raise ValueError(f"modality {k!r} has {a.shape[0]} rows, expected {n_rows}")
        if a.ndim != 2:
            raise ValueError(f"modality {k!r} must be 2-D, got shape {a.shape}")
    return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)


def _seeded_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


class _DeepBaseline:
    """Shared scaffolding: deterministic init, ``fit / predict / save / load``."""

    name: str = "_deep_base"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config: Dict[str, Any] = dict(config)
        self.seed: int = int(self.config.get("seed", 0))
        self.lr: float = float(self.config.get("lr", 1e-3))
        self.epochs: int = int(self.config.get("epochs", 50))
        self.batch_size: int = int(self.config.get("batch_size", 64))
        self.device: torch.device = torch.device(self.config.get("device", "cpu"))
        # Seed BEFORE building the module so weight init is reproducible.
        torch.manual_seed(self.seed)
        self.module: Optional[nn.Module] = None
        self._in_dim: Optional[int] = None

    def _build_module(self, in_dim: int) -> nn.Module:
        raise NotImplementedError

    def _ensure_module(self, in_dim: int) -> None:
        if self.module is None or self._in_dim != in_dim:
            torch.manual_seed(self.seed)
            self.module = self._build_module(in_dim).to(self.device)
            self._in_dim = in_dim

    def fit(
        self,
        X_train: Mapping[str, np.ndarray],
        y_train: np.ndarray,
        **kwargs: Any,
    ) -> None:
        X = _concat_modalities(X_train)
        y = np.asarray(y_train, dtype=np.float32).reshape(-1)
        self._ensure_module(X.shape[1])
        assert self.module is not None
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        x_t = torch.from_numpy(X).to(self.device)
        y_t = torch.from_numpy(y).to(self.device)
        n = x_t.shape[0]
        gen = _seeded_generator(self.seed)
        self.module.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                pred = self.module(x_t[idx]).reshape(-1)
                loss = F.mse_loss(pred, y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict(self, X_test: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.module is None:
            raise RuntimeError(f"{self.name}: model not fitted")
        X = _concat_modalities(X_test)
        self.module.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X).to(self.device)
            pred = self.module(x_t).reshape(-1).cpu().numpy()
        return pred

    def predict_survival(
        self, X_test: Mapping[str, np.ndarray]
    ) -> Optional[Dict[str, np.ndarray]]:
        return None  # not survival-capable.

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


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class _MLPModule(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPBaseline(_DeepBaseline):
    """3-layer fully-connected regressor."""

    name = "mlp"

    def _build_module(self, in_dim: int) -> nn.Module:
        hidden = int(self.config.get("hidden", 128))
        dropout = float(self.config.get("dropout", 0.1))
        return _MLPModule(in_dim, hidden, dropout)


# ---------------------------------------------------------------------------
# CNN over feature order
# ---------------------------------------------------------------------------


class _CNNFeatureOrderModule(nn.Module):
    def __init__(self, in_dim: int, channels: int, kernel: int) -> None:
        super().__init__()
        # Treat the feature dim as a 1-D sequence with one input channel.
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1))         # (B, C, 1)
        h = h.squeeze(-1)                      # (B, C)
        return self.head(h)


class CNNFeatureOrderBaseline(_DeepBaseline):
    """1-D CNN that treats the feature axis as a sequence.

    Default kernel=5, channels=16 — same depth+width as the MLP for fair
    comparison.
    """

    name = "cnn_feature_order"

    def _build_module(self, in_dim: int) -> nn.Module:
        channels = int(self.config.get("channels", 16))
        kernel = int(self.config.get("kernel", 5))
        if kernel % 2 == 0:
            kernel += 1  # force odd kernel so padding is symmetric.
        return _CNNFeatureOrderModule(in_dim, channels, kernel)


# ---------------------------------------------------------------------------
# FT-Transformer-lite
# ---------------------------------------------------------------------------


class _FTTransformerLiteModule(nn.Module):
    """Minimal FT-Transformer flavour: one learnable token per feature + a
    single self-attention block + a CLS-style aggregator.

    Citation: Gorishniy et al. NeurIPS 2021 (arXiv:2106.11959). The original
    paper is NOT in PubMed (NeurIPS is unindexed); cite arXiv only.
    """

    def __init__(self, in_dim: int, d_token: int, n_heads: int) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            # Round d_token up to the nearest multiple of n_heads.
            d_token = ((d_token + n_heads - 1) // n_heads) * n_heads
        self.feature_embed = nn.Linear(1, d_token)
        self.feature_bias = nn.Parameter(torch.zeros(in_dim, d_token))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        self.attn = nn.MultiheadAttention(d_token, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) -> per-feature token (B, F, d_token)
        tokens = self.feature_embed(x.unsqueeze(-1)) + self.feature_bias.unsqueeze(0)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        h = self.norm(attn_out[:, 0, :])  # CLS slot
        return self.head(h)


class FTTransformerBaseline(_DeepBaseline):
    """FT-Transformer-lite tabular regressor."""

    name = "ft_transformer"

    def _build_module(self, in_dim: int) -> nn.Module:
        d_token = int(self.config.get("d_token", 32))
        n_heads = int(self.config.get("n_heads", 4))
        return _FTTransformerLiteModule(in_dim, d_token, n_heads)


# ---------------------------------------------------------------------------
# Auto-register.
# ---------------------------------------------------------------------------


for _cls in (MLPBaseline, CNNFeatureOrderBaseline, FTTransformerBaseline):
    REGISTRY.register(_cls.name, lambda cfg, _c=_cls: _c(cfg))
