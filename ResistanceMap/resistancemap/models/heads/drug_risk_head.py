"""
resistancemap/models/heads/drug_risk_head.py
============================================
Drug-conditioned trajectory risk head.

For a panel of candidate drugs, predicts a *trajectory-derived* risk score:
"how likely is this patient to evolve toward a resistant basin under each
drug, over a fixed horizon?".

This is the head most directly useful for therapy selection. It takes the
foundation latent ``z0`` and a one-hot (or embedded) drug identity, and
returns a per-drug scalar risk.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DrugSpecificTrajectoryRiskHead(nn.Module):
    def __init__(
        self,
        d_latent: int,
        n_drugs: int,
        drug_embed_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_drugs = n_drugs
        self.drug_embed = nn.Embedding(n_drugs, drug_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(d_latent + drug_embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        drug_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns risk scores.

        If ``drug_idx`` is None, returns ``(N, n_drugs)`` — one risk per row
        per candidate drug. If ``drug_idx`` is ``(N,)``, returns ``(N,)``.
        """
        if drug_idx is None:
            embeds = self.drug_embed.weight                       # (n_drugs, d_drug)
            z_exp = z.unsqueeze(1).expand(-1, self.n_drugs, -1)   # (N, n_drugs, d_latent)
            d_exp = embeds.unsqueeze(0).expand(z.shape[0], -1, -1)
            x = torch.cat([z_exp, d_exp], dim=-1)
            return self.net(x).squeeze(-1)
        else:
            d_emb = self.drug_embed(drug_idx)
            x = torch.cat([z, d_emb], dim=-1)
            return self.net(x).squeeze(-1)
