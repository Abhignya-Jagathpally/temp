"""
resistancemap/mortfm/drug_response/drug_conditioned_head.py
=============================================================
Per-drug response head: (z_patient, drug_token, drug_family_token) -> scalar.

Replaces the v16 ``DrugSpecificTrajectoryRiskHead`` for the BeatAML
specimen track. The new head consumes the actual drug identity as an
embedding and emits ONE scalar response per (patient, drug) pair so
the per-drug supervision actually lands on per-drug parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DrugConditionedSpecimenResponseHead(nn.Module):
    def __init__(
        self,
        d_latent: int,
        n_drugs: int,
        n_drug_families: int,
        d_drug_embed: int = 32,
        d_family_embed: int = 8,
        d_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.drug_embed = nn.Embedding(n_drugs + 1, d_drug_embed)        # +1 for "unknown"
        self.family_embed = nn.Embedding(n_drug_families + 1, d_family_embed)
        self.fuse = nn.Sequential(
            nn.Linear(d_latent + d_drug_embed + d_family_embed, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        z_patient: torch.Tensor,        # (B, d_latent)
        drug_idx: torch.Tensor,         # (B,) long, drug identity index
        family_idx: torch.Tensor,       # (B,) long, drug family index
    ) -> torch.Tensor:
        d = self.drug_embed(drug_idx)
        f = self.family_embed(family_idx)
        x = torch.cat([z_patient, d, f], dim=-1)
        return self.fuse(x).squeeze(-1)
