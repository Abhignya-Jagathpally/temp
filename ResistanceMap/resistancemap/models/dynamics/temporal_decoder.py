"""
resistancemap/models/dynamics/temporal_decoder.py
=================================================
Decode latent trajectory ``z(t)`` back to observable molecular space.

Used by the trajectory-distribution loss to compare predicted future RNA /
protein abundance against the follow-up snapshot encoded through the
foundation tokenizer. Without this decoder, the trajectory loss is constrained
to operate purely on the latent state — which is sufficient for some metrics
(Wasserstein in latent space) but does not let us visualise predicted gene
expression at the held-out timepoint.

The decoder is intentionally small — the heavy lifting is done by the per-
modality reconstruction heads inside each encoder. The temporal decoder just
maps the latent at time ``t`` to a per-modality token, then re-uses each
modality's ``reconstruct`` method.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class TemporalDecoder(nn.Module):
    def __init__(
        self,
        d_latent: int,
        d_token: int,
        modality_names: List[str],
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.modality_names = list(modality_names)
        # One small projection per modality from latent to that modality's token.
        self.projections = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(d_latent, hidden), nn.GELU(), nn.Linear(hidden, d_token),
                )
                for m in self.modality_names
            }
        )

    def forward(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Project ``z(t)`` into one token per modality.

        Parameters
        ----------
        z_t :
            ``(N, d_latent)`` or ``(T, N, d_latent)``. The leading time
            dimension is preserved if present.
        """
        return {m: self.projections[m](z_t) for m in self.modality_names}
