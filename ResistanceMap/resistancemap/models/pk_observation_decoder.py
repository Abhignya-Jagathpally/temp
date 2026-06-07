"""PK-constrained observation decoder: latent state -> serum biomarkers (v20 Phase 5).

Maps a latent resistance state ``z`` to the *expected* values of routine serum
biomarkers (M-protein, FLC kappa/lambda, B2M, albumin, LDH, sBCMA), then
propagates them forward in time with first-order pharmacokinetic decay using
literature half-lives. This is the interpretability bridge that lets the
latent-space model speak in clinically legible biomarker units, and the module
validated against GSE136337's real baseline B2M / albumin / LDH.

Honest scope
------------
* Half-lives are fixed literature priors (constants below); they are NOT fit
  to fabricated data.
* When trained on scRNA-seq proxy expression, the target is the
  :data:`~resistancemap.models.scvi_encoder.BIOMARKER_PROXY_GENES` aggregate;
  ``compute_loss`` masks any biomarker the data does not provide.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

# Elimination half-lives (days). Sources: IgG ~21 d; serum albumin ~19 d;
# free light chains ~2-6 h (~0.15 d); B2M ~2.5 h (~0.1 d); LDH ~1-7 d (~3 d);
# soluble BCMA ~24-36 h (~1.25 d). Used to propagate predicted levels in time.
BIOMARKER_HALF_LIFE_DAYS: Dict[str, float] = {
    "M_protein": 21.0,
    "FLC_kappa": 0.15,
    "FLC_lambda": 0.15,
    "B2M": 0.10,
    "Albumin": 19.0,
    "LDH": 3.0,
    "sBCMA": 1.25,
}

# Gene-expression proxies for each serum biomarker (from the v20 plan).
GENE_TO_BIOMARKER: Dict[str, List[str]] = {
    "M_protein": ["IGHG1", "IGHG2", "IGHG3", "IGHA1", "IGHA2"],
    "FLC_kappa": ["IGKC"],
    "FLC_lambda": ["IGLC1", "IGLC2", "IGLC3"],
    "B2M": ["B2M"],
    "Albumin": ["ALB"],
    "LDH": ["LDHA", "LDHB"],
    "sBCMA": ["TNFRSF17"],
}

DEFAULT_BIOMARKERS: List[str] = list(BIOMARKER_HALF_LIFE_DAYS.keys())


class PKObservationDecoder(nn.Module):
    """Decode latent ``z`` -> non-negative expected biomarker levels + PK decay."""

    def __init__(self, d_latent: int, biomarkers: Sequence[str] = DEFAULT_BIOMARKERS,
                 hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.biomarkers = list(biomarkers)
        n = len(self.biomarkers)
        self.trunk = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.level_head = nn.Linear(hidden, n)  # softplus -> >= 0
        # Per-biomarker clearance rate k = ln2 / half_life, stored as log k so it
        # stays positive and is initialised from the literature prior.
        ln2 = math.log(2.0)
        k0 = torch.tensor(
            [ln2 / max(BIOMARKER_HALF_LIFE_DAYS.get(b, 7.0), 1e-3) for b in self.biomarkers],
            dtype=torch.float32,
        )
        self.log_clearance = nn.Parameter(torch.log(k0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Expected current biomarker levels, shape (B, n_biomarkers), >= 0."""
        h = self.trunk(z)
        return nn.functional.softplus(self.level_head(h))

    def predict_after(self, z: torch.Tensor, delta_t_days: torch.Tensor) -> torch.Tensor:
        """First-order PK decay of predicted levels over ``delta_t_days``.

        ``level(t+Δ) = level(t) * exp(-k Δ)``. ``delta_t_days`` broadcasts over
        the biomarker axis (shape (B,) or (B,1)).
        """
        level = self.forward(z)
        k = torch.exp(self.log_clearance)  # (n,)
        dt = delta_t_days.reshape(-1, 1).to(level.dtype)
        return level * torch.exp(-k.unsqueeze(0) * dt)

    def compute_loss(self, z: torch.Tensor, target: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Masked MSE in log1p space (robust to the wide dynamic range of labs).

        ``target`` is (B, n_biomarkers); ``mask`` (same shape) is True where a
        real measurement exists. Biomarkers with no measurement contribute zero.
        """
        pred = self.forward(z)
        lp = torch.log1p(torch.clamp(pred, min=0.0))
        lt = torch.log1p(torch.clamp(target, min=0.0))
        sq = (lp - lt) ** 2
        if mask is not None:
            m = mask.to(sq.dtype)
            denom = m.sum().clamp(min=1.0)
            return (sq * m).sum() / denom
        return sq.mean()
