"""
resistancemap/models/encoders/drug_encoder.py
=============================================
Drug context encoder.

Expands raw scalar features (dose, start_time, end_time, identity, class) plus
optional pre-computed chemical embeddings into a fixed-width drug token.

The encoder accepts either:
1. A pre-encoded ``(N, d_drug_raw)`` tensor from
   :func:`resistancemap.data.mortfm_dataset.mort_collate` (current MORTBatch.drug),
   *and* per-row integer indices into the drug-identity and drug-class
   lookups (passed via ``drug_idx`` and ``class_idx``), OR
2. A list of :class:`~resistancemap.mortfm.schemas.DrugContext` objects via
   ``encode_contexts`` (slower path, used during inference).

The output is ``(N, d_token)``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from resistancemap.data.drug_ontology import (
    class_to_index,
    drug_to_index,
    materialise_drug_context,
)
from resistancemap.mortfm.schemas import DrugContext


class DrugEncoder(nn.Module):
    def __init__(
        self,
        n_drugs: int = 32,
        n_classes: int = 16,
        d_drug_raw: int = 3,
        d_token: int = 64,
        d_chem: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_drugs = n_drugs
        self.n_classes = n_classes
        self.d_drug_raw = d_drug_raw
        self.d_token = d_token
        self.d_chem = d_chem

        self.drug_embed = nn.Embedding(n_drugs, 16)
        self.class_embed = nn.Embedding(n_classes, 8)

        in_dim = d_drug_raw + 16 + 8 + d_chem
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_token),
        )

    def forward(
        self,
        raw: torch.Tensor,
        *,
        drug_idx: torch.Tensor,
        class_idx: torch.Tensor,
        chem_embedding: Optional[torch.Tensor] = None,
        presence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Replace NaNs in scalar fields with 0 — the presence mask carries
        # the "missing dose" signal, so the encoder doesn't need to learn NaN.
        raw = torch.where(torch.isnan(raw), torch.zeros_like(raw), raw)
        de = self.drug_embed(drug_idx)
        ce = self.class_embed(class_idx)
        parts = [raw, de, ce]
        if self.d_chem > 0:
            if chem_embedding is None:
                chem_embedding = raw.new_zeros(raw.shape[0], self.d_chem)
            parts.append(chem_embedding)
        x = torch.cat(parts, dim=-1)
        z = self.encoder(x)
        if presence_mask is not None:
            z = z * presence_mask.float().unsqueeze(-1)
        return z

    @torch.no_grad()
    def encode_contexts(
        self,
        contexts: Sequence[Optional[DrugContext]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Inference-path: encode a list of DrugContext objects.

        Returns ``(N, d_token)``. Absent contexts become zero rows.
        """
        device = device or next(self.parameters()).device
        N = len(contexts)
        raw = torch.zeros(N, self.d_drug_raw, device=device)
        drug_idx = torch.zeros(N, dtype=torch.long, device=device)
        class_idx = torch.zeros(N, dtype=torch.long, device=device)
        presence = torch.zeros(N, dtype=torch.bool, device=device)
        d2i = drug_to_index()
        c2i = class_to_index()
        for i, ctx in enumerate(contexts):
            if ctx is None:
                continue
            presence[i] = True
            raw[i, 0] = ctx.dose if ctx.dose is not None else 0.0
            raw[i, 1] = ctx.start_time if ctx.start_time is not None else 0.0
            raw[i, 2] = ctx.end_time if ctx.end_time is not None else 0.0
            drug_idx[i] = min(d2i.get(ctx.drug_name, 0), self.n_drugs - 1)
            class_idx[i] = min(c2i.get(ctx.drug_class or "unknown", 0), self.n_classes - 1)
        return self.forward(
            raw, drug_idx=drug_idx, class_idx=class_idx, presence_mask=presence
        )
