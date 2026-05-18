"""
resistancemap/models/foundation_fusion.py
=========================================
Tokenised cross-modal transformer fusion for MORT-FM.

Generalises the v14 :mod:`resistancemap.models.fusion` (a 4-input cell-line
attention block) into a token-based foundation-model fusion that:

1. Accepts a variable number of modality tokens per row (RNA / ATAC /
   methylation / histone PTM / proteomics / phosphoproteomics / clinical /
   drug).
2. Carries a *per-row, per-modality presence mask* so missing modalities do
   not poison the attention. Masked positions are excluded from softmax
   (key-padding-mask = ~presence).
3. Prepends a learnable ``[CLS]`` token whose output becomes the foundation
   latent ``z0``.
4. Optionally appends patient / time / drug auxiliary tokens.
5. Returns soft modality gates (column-wise mean attention to the CLS token)
   for the modality-importance reporting.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from resistancemap.mortfm.schemas import FoundationState, MORTBatch, MORTFMConfig
from resistancemap.models.encoders.modality_tokenizer import ModalityTokenizer


class MultiOmicFoundationFusion(nn.Module):
    """Cross-modal transformer over modality tokens.

    Parameters
    ----------
    config :
        :class:`~resistancemap.mortfm.schemas.MORTFMConfig` driving encoder
        instantiation and fusion dimensions.
    return_attention :
        If True, collect attention weights from each layer into
        :attr:`FoundationState.attention_maps`. Default False (saves memory).
    """

    def __init__(self, config: MORTFMConfig, *, return_attention: bool = False) -> None:
        super().__init__()
        self.config = config
        self.return_attention = return_attention

        self.tokenizer = ModalityTokenizer(config)

        d = config.d_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Auxiliary tokens: time + patient. Patient embedding is *not* a
        # nn.Embedding (cohort size is unknown at construction time and we
        # don't want to leak patient identity into the model); we use a small
        # projection of a hash of the patient ID instead, or pass a
        # zero token. Time is projected from a scalar.
        self.time_proj = nn.Linear(1, d)
        self.patient_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.patient_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.fusion_heads,
            dim_feedforward=4 * d,
            dropout=config.fusion_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.fusion_layers)
        self.layer_norm = nn.LayerNorm(d)

        # Project the CLS-token output to the model's latent dimension.
        self.latent_proj = nn.Linear(d, config.d_latent)

        # Per-modality reconstruction not done here — owned by the tokenizer's
        # encoders. The fusion is responsible only for joint latent + gates.

    def _build_token_sequence(
        self,
        modality_tokens: torch.Tensor,
        presence: torch.Tensor,
        time: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N, K, d = modality_tokens.shape
        cls = self.cls_token.expand(N, -1, -1)
        patient = self.patient_token.expand(N, -1, -1)
        tokens = [cls, patient, modality_tokens]
        # CLS and patient tokens are always present.
        cls_pres = torch.ones(N, 2, dtype=torch.bool, device=presence.device)
        if time is not None:
            t = torch.nan_to_num(time, nan=0.0).unsqueeze(-1).to(modality_tokens.device)
            time_tok = self.time_proj(t).unsqueeze(1)
            tokens.insert(2, time_tok)
            cls_pres = torch.cat(
                [cls_pres, torch.ones(N, 1, dtype=torch.bool, device=presence.device)], dim=1
            )
        seq = torch.cat(tokens, dim=1)
        mask = torch.cat([cls_pres, presence], dim=1)
        return seq, mask

    def forward(self, batch: MORTBatch) -> FoundationState:
        modality_tokens, presence, mod_names = self.tokenizer(batch)
        if modality_tokens.shape[1] == 0:
            raise ValueError(
                "MultiOmicFoundationFusion received zero instantiated modalities. "
                "Configure at least one use_* flag in MORTFMConfig."
            )
        seq, mask = self._build_token_sequence(modality_tokens, presence, batch.time)
        # Transformer expects key_padding_mask where True = ignore.
        kpm = ~mask

        attn_maps: Dict[str, torch.Tensor] = {}
        if self.return_attention:
            x = seq
            for i, layer in enumerate(self.transformer.layers):
                # Manually invoke MHA to grab attention weights.
                normed = layer.norm1(x)
                attn_out, attn_w = layer.self_attn(
                    normed, normed, normed,
                    key_padding_mask=kpm,
                    need_weights=True,
                    average_attn_weights=False,
                )
                x = x + layer.dropout1(attn_out)
                x = x + layer.dropout2(layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))
                ))
                attn_maps[f"layer_{i}"] = attn_w.detach()
            x = self.layer_norm(x)
        else:
            x = self.transformer(seq, src_key_padding_mask=kpm)
            x = self.layer_norm(x)

        cls_out = x[:, 0]  # (N, d)
        z0 = self.latent_proj(cls_out)

        # Gates: cosine similarity between cls_out and each modality token,
        # softmax-normalised across present modalities only.
        n_mod = modality_tokens.shape[1]
        # x[:, 2 (or 3 with time):] are the modality tokens after the aux tokens.
        offset = 2 + (1 if batch.time is not None else 0)
        mod_out = x[:, offset : offset + n_mod]
        sim = F.cosine_similarity(
            cls_out.unsqueeze(1), mod_out, dim=-1
        )  # (N, n_mod)
        # mask out absent modalities by setting their logits to -inf.
        sim = sim.masked_fill(~presence, float("-inf"))
        gates = F.softmax(sim, dim=-1)
        gates = torch.where(torch.isnan(gates), torch.zeros_like(gates), gates)

        return FoundationState(
            z0=z0,
            modality_embeddings={
                name: modality_tokens[:, i] for i, name in enumerate(mod_names)
            },
            attention_maps=attn_maps,
            modality_gates=gates,
            uncertainty=None,
        )
