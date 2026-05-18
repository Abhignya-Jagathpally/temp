"""
resistancemap/models/encoders/modality_tokenizer.py
===================================================
Aggregator that routes a :class:`~resistancemap.mortfm.schemas.MORTBatch` through
the per-modality encoders and produces a ``(N, K, d_token)`` token tensor + a
``(N, K)`` token-presence mask consumed by the foundation fusion.

Only the modalities that are *configured to be used* (per
:class:`~resistancemap.mortfm.schemas.MORTFMConfig`) are instantiated. Missing
modalities at *runtime* (per ``MORTBatch.modality_mask``) yield zero tokens
with mask=False; the fusion layer is responsible for ignoring those rows.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from resistancemap.models.encoders.atac_encoder import ATACEncoder
from resistancemap.models.encoders.clinical_encoder import ClinicalEncoder
from resistancemap.models.encoders.drug_encoder import DrugEncoder
from resistancemap.models.encoders.histone_ptm_encoder import HistonePTMEncoder
from resistancemap.models.encoders.methylation_encoder import MethylationEncoder
from resistancemap.models.encoders.phosphoproteomic_encoder import PhosphoproteomicEncoder
from resistancemap.models.encoders.proteomic_encoder import ProteomicEncoder
from resistancemap.models.encoders.rna_encoder import RNAEncoder
from resistancemap.mortfm.schemas import MODALITY_ORDER, MORTBatch, MORTFMConfig, ModalityName


class ModalityTokenizer(nn.Module):
    """Routes a :class:`MORTBatch` through per-modality encoders.

    Returns
    -------
    tokens : ``(N, K, d_token)``
        Per-modality token tensor. Modalities that were not instantiated are
        omitted from ``K``.
    presence : ``(N, K)``
        Per-row, per-modality presence mask (True = real, False = missing).
    modality_names : ``List[str]``
        Names of the K modalities in the order they appear along dim=1.
    """

    def __init__(self, config: MORTFMConfig) -> None:
        super().__init__()
        self.config = config
        self.encoders: nn.ModuleDict = nn.ModuleDict()

        if config.use_rna:
            self.encoders[ModalityName.RNA.value] = RNAEncoder(
                input_dim=config.rna_input_dim,
                d_token=config.d_token,
                dropout=config.fusion_dropout,
            )
        if config.use_atac:
            self.encoders[ModalityName.ATAC.value] = ATACEncoder(
                input_dim=config.atac_input_dim,
                d_token=config.d_token,
                dropout=config.fusion_dropout,
            )
        if config.use_methylation:
            self.encoders[ModalityName.METHYLATION.value] = MethylationEncoder(
                input_dim=config.methylation_input_dim,
                d_token=config.d_token,
                dropout=config.fusion_dropout,
            )
        if config.use_histone_ptm:
            self.encoders[ModalityName.HISTONE_PTM.value] = HistonePTMEncoder(
                input_dim=config.histone_ptm_input_dim,
                d_token=config.d_token,
                dropout=config.fusion_dropout,
            )
        if config.use_proteomics:
            self.encoders[ModalityName.PROTEOMICS.value] = ProteomicEncoder(
                input_dim=config.proteomics_input_dim,
                d_token=config.d_token,
                dropout=config.fusion_dropout,
            )
        if config.use_phosphoproteomics:
            self.encoders[ModalityName.PHOSPHOPROTEOMICS.value] = PhosphoproteomicEncoder(
                input_dim=config.phosphoproteomics_input_dim,
                d_token=config.d_token,
                dropout=config.fusion_dropout,
            )
        if config.use_clinical:
            self.encoders[ModalityName.CLINICAL.value] = ClinicalEncoder(
                input_dim=config.clinical_input_dim,
                d_token=config.d_token,
                dropout=config.fusion_dropout,
            )
        if config.use_drug:
            self.encoders[ModalityName.DRUG.value] = DrugEncoder(
                n_drugs=config.drug_n_drugs,
                n_classes=config.drug_n_classes,
                d_token=config.d_token,
                dropout=config.fusion_dropout,
            )

        # Modality embedding so the fusion layer can distinguish tokens.
        self.modality_pos_embed = nn.Embedding(len(MODALITY_ORDER), config.d_token)
        self._mod_idx = {m.value: i for i, m in enumerate(MODALITY_ORDER)}

    def forward(
        self,
        batch: MORTBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Tokenise a batch.

        Empty batch (no instantiated encoders or all-missing modalities)
        returns shape ``(N, 0, d_token)``.
        """
        N = batch.batch_size
        device = next(self.parameters()).device
        tokens: List[torch.Tensor] = []
        presence_cols: List[torch.Tensor] = []
        names: List[str] = []

        for name, enc in self.encoders.items():
            x = getattr(batch, name)
            if x is None:
                token = torch.zeros(N, self.config.d_token, device=device)
                present = torch.zeros(N, dtype=torch.bool, device=device)
            else:
                x = x.to(device)
                presence_mask = batch.modality_mask.get(name)
                if presence_mask is not None:
                    presence_mask = presence_mask.to(device)
                if name == ModalityName.DRUG.value:
                    # Drug encoder needs drug_idx / class_idx. Without precomputed
                    # indices we fall back to zeros (the dose/time scalars still
                    # flow through). Callers that want full identity should call
                    # ``DrugEncoder.encode_contexts`` and replace this token.
                    drug_idx = torch.zeros(N, dtype=torch.long, device=device)
                    class_idx = torch.zeros(N, dtype=torch.long, device=device)
                    token = enc(x, drug_idx=drug_idx, class_idx=class_idx,
                                presence_mask=presence_mask)
                else:
                    token = enc(x, presence_mask=presence_mask)
                if presence_mask is None:
                    present = torch.ones(N, dtype=torch.bool, device=device)
                else:
                    present = presence_mask
            # Add modality positional embedding so the fusion layer can
            # distinguish "RNA token" vs "proteomics token".
            mod_pe = self.modality_pos_embed.weight[self._mod_idx[name]]
            token = token + mod_pe.unsqueeze(0)
            tokens.append(token)
            presence_cols.append(present)
            names.append(name)

        if not tokens:
            return (
                torch.zeros(N, 0, self.config.d_token, device=device),
                torch.zeros(N, 0, dtype=torch.bool, device=device),
                [],
            )

        token_tensor = torch.stack(tokens, dim=1)
        presence_tensor = torch.stack(presence_cols, dim=1)
        return token_tensor, presence_tensor, names
