from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TileEncoder(nn.Module):
    """
    Placeholder wrapper for UNI/GigaPath foundation models.

    In production, this would load a pretrained foundation model (UNI or GigaPath)
    and extract 2048-dimensional embeddings from 256×256 pathology tiles.

    Currently returns randomly initialized embeddings for demonstration.
    In practice, replace with actual model loading and inference.
    """

    def __init__(self, embedding_dim: int = 2048, freeze_backbone: bool = True):
        """
        Initialize the tile encoder.

        Args:
            embedding_dim: Dimension of output embeddings (UNI/GigaPath: 2048).
            freeze_backbone: If True, freeze pretrained model parameters.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone

        # Placeholder: In production, load actual foundation model
        self.fc = nn.Linear(2048, embedding_dim)

        logger.info(f"TileEncoder initialized with embedding_dim={embedding_dim}")

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """
        Encode tiles to embeddings.

        Args:
            tiles: Batch of tiles, shape (batch_size, 3, 256, 256).

        Returns:
            Embeddings of shape (batch_size, embedding_dim).
        """
        # Placeholder: In production, pass through actual foundation model
        # For now, return random embeddings with the right shape
        batch_size = tiles.shape[0]
        embeddings = torch.randn(batch_size, self.embedding_dim, device=tiles.device)
        return embeddings


class TransMILPooling(nn.Module):
    """
    Transformer-based Multiple Instance Learning pooling with Nyström attention.

    Aggregates sequence of tile embeddings to a single slide-level (patient-level)
    embedding using efficient Nyström approximation of self-attention.
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        num_heads: int = 8,
        num_landmarks: int = 128,
        dropout: float = 0.1,
    ):
        """
        Initialize TransMIL pooling layer.

        Args:
            embedding_dim: Dimension of input embeddings.
            num_heads: Number of attention heads.
            num_landmarks: Number of Nyström landmarks for efficient attention.
            dropout: Dropout rate.
        """
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 1000, embedding_dim) * 0.02
        )

        # Query, key, value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        # Landmark projection for Nyström approximation
        self.landmark_proj = nn.Linear(embedding_dim, num_landmarks)

        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Layer norm and dropout
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"TransMILPooling initialized: embedding_dim={embedding_dim}, "
            f"num_heads={num_heads}, num_landmarks={num_landmarks}"
        )

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate tile embeddings to patient-level embedding using Nyström attention.

        Args:
            tile_embeddings: Shape (num_tiles, embedding_dim) or (batch_size, num_tiles, embedding_dim).

        Returns:
            Patient-level embedding of shape (embedding_dim,) or (batch_size, embedding_dim).
        """
        # Handle 2D input (single sample)
        squeeze_batch = False
        if tile_embeddings.dim() == 2:
            tile_embeddings = tile_embeddings.unsqueeze(0)
            squeeze_batch = True

        batch_size, num_tiles, embed_dim = tile_embeddings.shape

        # Add positional encoding
        pos_enc = self.pos_encoding[:, :num_tiles, :]
        x = tile_embeddings + pos_enc

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, num_tiles, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, num_tiles, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, num_tiles, self.num_heads, self.head_dim)

        # Nyström approximation: sample landmarks from K
        if num_tiles > self.num_landmarks:
            # Select random landmarks
            landmark_indices = torch.randperm(num_tiles, device=x.device)[:self.num_landmarks]
            K_landmarks = K[:, landmark_indices, :, :]
        else:
            K_landmarks = K

        # Compute attention with landmarks
        Q = Q.transpose(1, 2)  # (batch, num_heads, num_tiles, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        K_landmarks = K_landmarks.transpose(1, 2)

        # A = softmax(Q @ K^T / sqrt(d))
        attn = torch.matmul(Q, K_landmarks.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output = attn @ V
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, num_tiles, embed_dim)

        # Global average pooling to get single embedding per sample
        patient_embedding = out.mean(dim=1)  # (batch_size, embedding_dim)

        # Output projection and residual
        patient_embedding = self.out_proj(patient_embedding)
        patient_embedding = self.norm(patient_embedding + tile_embeddings.mean(dim=1))

        if squeeze_batch:
            patient_embedding = patient_embedding.squeeze(0)

        return patient_embedding


class MorphologyFeatureExtractor(nn.Module):
    """
    Extract morphological features from pathology images.

    Computes:
    - Bartl infiltration pattern (categorical: focal/diffuse/mixed)
    - Microvessel density (MVD, vessels per high-power field)
    - Fibrosis grade (0-4 scale)

    These are clinical prognostic markers:
    - Diffuse infiltration: HR 4.16 for worse outcome
    - High MVD (>50/field): 2.6yr vs 5.1yr median OS
    """

    def __init__(self, feature_dim: int = 128):
        """
        Initialize morphology feature extractor.

        Args:
            feature_dim: Output embedding dimension for morphological features.
        """
        super().__init__()
        self.feature_dim = feature_dim

        # Embeddings for categorical features
        self.pattern_embedding = nn.Embedding(3, feature_dim // 2)  # focal, diffuse, mixed
        self.fibrosis_embedding = nn.Embedding(5, feature_dim // 2)  # grades 0-4

        # Linear layers to process continuous features
        self.mvd_fc = nn.Linear(1, feature_dim // 2)
        self.combined_fc = nn.Linear(feature_dim, feature_dim)

        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(feature_dim)

        logger.info(f"MorphologyFeatureExtractor initialized with feature_dim={feature_dim}")

    def forward(
        self,
        pattern_class: torch.Tensor,  # 0=focal, 1=diffuse, 2=mixed
        mvd_count: torch.Tensor,      # microvessel density (vessels per field)
        fibrosis_grade: torch.Tensor, # 0-4
    ) -> torch.Tensor:
        """
        Extract morphological feature embedding.

        Args:
            pattern_class: Shape (batch_size,), values in {0, 1, 2}.
            mvd_count: Shape (batch_size, 1), continuous MVD values.
            fibrosis_grade: Shape (batch_size,), values in {0, 1, 2, 3, 4}.

        Returns:
            Morphological feature embedding of shape (batch_size, feature_dim).
        """
        # Pattern embedding
        pattern_emb = self.pattern_embedding(pattern_class)  # (batch, feature_dim//2)

        # Fibrosis embedding
        fibrosis_emb = self.fibrosis_embedding(fibrosis_grade)  # (batch, feature_dim//2)

        # MVD processing
        mvd_emb = self.mvd_fc(mvd_count)  # (batch, feature_dim//2)

        # Combine all features
        combined = torch.cat([pattern_emb, fibrosis_emb, mvd_emb, mvd_emb], dim=-1)
        combined = self.relu(combined)
        combined = self.combined_fc(combined)
        combined = self.norm(combined)

        return combined


class RadiomicsFeatureEncoder(nn.Module):
    """
    Encode PET/CT and MRI radiomics features.

    PET/CT features:
    - Metabolic Tumor Volume (MTV)
    - Total Lesion Glycolysis (TLG)
    - SUVmax, SUVmean
    - GLCM (Gray-Level Co-occurrence Matrix) texture features
    - GLRLM (Gray-Level Run-Length Matrix) features

    MRI features:
    - Pattern classification: focal/diffuse/mixed
    - DCE-MRI angiogenesis metrics (wash-in rate, wash-out rate, AUC)

    Clinical thresholds:
    - MTV ≥ 56.4 cm³ associated with poor outcome
    - High angiogenesis (fast wash-out) associated with aggressive disease
    """

    def __init__(self, embedding_dim: int = 512):
        """
        Initialize radiomics feature encoder.

        Args:
            embedding_dim: Output embedding dimension.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # PET/CT encoder (7 continuous features)
        self.pet_ct_fc = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim // 2),
        )

        # MRI pattern embedding
        self.mri_pattern_embedding = nn.Embedding(3, embedding_dim // 4)

        # DCE-MRI encoder (3 continuous features: wash-in, wash-out, AUC)
        self.dce_mri_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, embedding_dim // 4),
        )

        # Combined encoder
        self.combined_fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.norm = nn.LayerNorm(embedding_dim)

        logger.info(f"RadiomicsFeatureEncoder initialized with embedding_dim={embedding_dim}")

    def forward(
        self,
        mtv: torch.Tensor,                    # (batch, 1)
        tlg: torch.Tensor,                    # (batch, 1)
        suv_max: torch.Tensor,                # (batch, 1)
        suv_mean: torch.Tensor,               # (batch, 1)
        glcm_features: torch.Tensor,          # (batch, 3) - energy, contrast, homogeneity
        glrlm_features: torch.Tensor,         # (batch, 1) - run percentage
        mri_pattern: torch.Tensor,            # (batch,) - 0/1/2
        dce_wash_in: torch.Tensor,            # (batch, 1)
        dce_wash_out: torch.Tensor,           # (batch, 1)
        dce_auc: torch.Tensor,                # (batch, 1)
    ) -> torch.Tensor:
        """
        Encode radiomics features to embedding.

        Args:
            mtv: Metabolic Tumor Volume (cm³).
            tlg: Total Lesion Glycolysis.
            suv_max, suv_mean: PET SUV measurements.
            glcm_features: GLCM texture features.
            glrlm_features: GLRLM features.
            mri_pattern: MRI infiltration pattern (0=focal, 1=diffuse, 2=mixed).
            dce_wash_in, dce_wash_out, dce_auc: DCE-MRI angiogenesis metrics.

        Returns:
            Radiomics embedding of shape (batch_size, embedding_dim).
        """
        # PET/CT encoding
        pet_ct = torch.cat([mtv, tlg, suv_max, suv_mean, glcm_features, glrlm_features], dim=-1)
        pet_ct_emb = self.pet_ct_fc(pet_ct)  # (batch, embedding_dim//2)

        # MRI pattern embedding
        mri_pattern_emb = self.mri_pattern_embedding(mri_pattern)  # (batch, embedding_dim//4)

        # DCE-MRI encoding
        dce_features = torch.cat([dce_wash_in, dce_wash_out, dce_auc], dim=-1)
        dce_emb = self.dce_mri_fc(dce_features)  # (batch, embedding_dim//4)

        # Combine all radiomics features
        radiomics_emb = torch.cat([pet_ct_emb, mri_pattern_emb, dce_emb], dim=-1)
        radiomics_emb = self.combined_fc(radiomics_emb)
        radiomics_emb = self.norm(radiomics_emb)

        return radiomics_emb


class HierarchicalCrossAttentionFusion(nn.Module):
    """
    4-layer hierarchical cross-attention fusion module.

    Sequentially fuses modalities in order of information hierarchy:
    Layer 1: Genomics ↔ Proteomics (direct molecular interactions)
    Layer 2: (Genomics+Proteomics) ↔ Metabolomics (downstream effects)
    Layer 3: (Genomics+Proteomics+Metabolomics) ↔ Pathomics (cellular morphology)
    Layer 4: (All) ↔ Radiomics (system-level imaging)

    Uses asymmetric learned attention weights since imaging may be less
    informative than genomics in some contexts.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Initialize hierarchical cross-attention fusion.

        Args:
            embedding_dim: Dimension of all input embeddings.
            num_heads: Number of attention heads per layer.
            hidden_dim: Hidden dimension in feedforward networks.
            dropout: Dropout rate.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Layer 1: Genomics ↔ Proteomics
        self.layer1_self_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer1_cross_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer 2: (Genomics+Proteomics) ↔ Metabolomics
        self.layer2_cross_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer 3: (All+Metabolomics) ↔ Pathomics
        self.layer3_cross_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer 4: (All+Pathomics) ↔ Radiomics (asymmetric)
        self.layer4_cross_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Learned asymmetry weights for Layer 4 (imaging may be less informative)
        self.layer4_asymmetry_weight = nn.Parameter(torch.tensor(0.7))

        # Feedforward networks
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Layer norms
        self.norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(4)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(4)])

        self.dropout_layer = nn.Dropout(dropout)

        logger.info(
            f"HierarchicalCrossAttentionFusion initialized: "
            f"embedding_dim={embedding_dim}, num_heads={num_heads}"
        )

    def forward(
        self,
        genomics_emb: torch.Tensor,      # (batch, embedding_dim)
        proteomics_emb: torch.Tensor,    # (batch, embedding_dim)
        metabolomics_emb: torch.Tensor,  # (batch, embedding_dim)
        pathomics_emb: torch.Tensor,     # (batch, embedding_dim)
        radiomics_emb: torch.Tensor,     # (batch, embedding_dim)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Fuse multimodal embeddings through hierarchical cross-attention.

        Args:
            genomics_emb, proteomics_emb, metabolomics_emb, pathomics_emb, radiomics_emb:
                All shape (batch_size, embedding_dim).

        Returns:
            - fused_embedding: Final integrated embedding (batch_size, embedding_dim).
            - layer_outputs: Dict with outputs from each fusion layer.
        """
        layer_outputs = {}

        # Layer 1: Genomics ↔ Proteomics
        genomics_and_proteomics = torch.stack(
            [genomics_emb, proteomics_emb], dim=1
        )  # (batch, 2, embedding_dim)
        attn1_out, _ = self.layer1_cross_attn(
            genomics_and_proteomics, genomics_and_proteomics, genomics_and_proteomics
        )
        attn1_out = self.norms[0](attn1_out + genomics_and_proteomics)
        ffn1_out = self.ffn(attn1_out)
        layer1_out = self.ffn_norms[0](ffn1_out + attn1_out)
        layer_outputs["layer1"] = layer1_out.mean(dim=1)  # (batch, embedding_dim)

        # Aggregate Layer 1 outputs
        gp_fused = layer1_out.mean(dim=1)

        # Layer 2: (Genomics+Proteomics) ↔ Metabolomics
        gpm_query = torch.stack([gp_fused, metabolomics_emb], dim=1)
        attn2_out, _ = self.layer2_cross_attn(
            gpm_query, gpm_query, gpm_query
        )
        attn2_out = self.norms[1](attn2_out + gpm_query)
        ffn2_out = self.ffn(attn2_out)
        layer2_out = self.ffn_norms[1](ffn2_out + attn2_out)
        layer_outputs["layer2"] = layer2_out.mean(dim=1)

        gpm_fused = layer2_out.mean(dim=1)

        # Layer 3: (All+Metabolomics) ↔ Pathomics
        gpmp_query = torch.stack([gpm_fused, pathomics_emb], dim=1)
        attn3_out, _ = self.layer3_cross_attn(
            gpmp_query, gpmp_query, gpmp_query
        )
        attn3_out = self.norms[2](attn3_out + gpmp_query)
        ffn3_out = self.ffn(attn3_out)
        layer3_out = self.ffn_norms[2](ffn3_out + attn3_out)
        layer_outputs["layer3"] = layer3_out.mean(dim=1)

        gpmp_fused = layer3_out.mean(dim=1)

        # Layer 4: (All+Pathomics) ↔ Radiomics (asymmetric)
        gpmpr_query = torch.stack([gpmp_fused, radiomics_emb], dim=1)
        attn4_out, attn4_weights = self.layer4_cross_attn(
            gpmpr_query, gpmpr_query, gpmpr_query
        )

        # Apply asymmetry: reduce radiomics contribution
        attn4_out[:, 1, :] = attn4_out[:, 1, :] * self.layer4_asymmetry_weight

        attn4_out = self.norms[3](attn4_out + gpmpr_query)
        ffn4_out = self.ffn(attn4_out)
        layer4_out = self.ffn_norms[3](ffn4_out + attn4_out)
        layer_outputs["layer4"] = layer4_out.mean(dim=1)

        fused_embedding = layer4_out.mean(dim=1)

        logger.debug(f"Hierarchical fusion complete. Asymmetry weight: {self.layer4_asymmetry_weight.item():.3f}")

        return fused_embedding, layer_outputs


class ImagingRadiomicsIntegrator(nn.Module):
    """
    Top-level integration module for imaging and radiomics.

    Orchestrates:
    1. Pathomics encoding: WSI → tiles → embeddings → TransMIL aggregation
    2. Morphology feature extraction: Pattern, MVD, fibrosis
    3. Radiomics encoding: PET/CT + MRI features
    4. Cross-attention fusion with genomic/proteomic/metabolomic modalities
    """

    def __init__(
        self,
        tile_embedding_dim: int = 2048,
        output_dim: int = 512,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Initialize the imaging/radiomics integrator.

        Args:
            tile_embedding_dim: Dimension of tile embeddings from foundation model.
            output_dim: Final output embedding dimension.
            num_heads: Number of attention heads.
            hidden_dim: Hidden dimension in feedforward networks.
            dropout: Dropout rate.
        """
        super().__init__()
        self.output_dim = output_dim

        # Pathomics components
        self.tile_encoder = TileEncoder(embedding_dim=tile_embedding_dim)
        self.transmil_pooling = TransMILPooling(
            embedding_dim=tile_embedding_dim,
            num_heads=num_heads,
        )
        self.pathomics_proj = nn.Linear(tile_embedding_dim, output_dim)

        # Morphology feature extractor
        self.morphology_extractor = MorphologyFeatureExtractor(feature_dim=output_dim)

        # Radiomics feature encoder
        self.radiomics_encoder = RadiomicsFeatureEncoder(embedding_dim=output_dim)

        # Combine pathomics and morphology
        self.pathomics_morphology_fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

        # Cross-attention fusion (with genomics/proteomics/metabolomics)
        self.hierarchical_fusion = HierarchicalCrossAttentionFusion(
            embedding_dim=output_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Final projection
        self.final_proj = nn.Linear(output_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)

        logger.info(
            f"ImagingRadiomicsIntegrator initialized: "
            f"tile_embedding_dim={tile_embedding_dim}, output_dim={output_dim}"
        )

    def forward(
        self,
        tiles: torch.Tensor,
        pattern_class: torch.Tensor,
        mvd_count: torch.Tensor,
        fibrosis_grade: torch.Tensor,
        mtv: torch.Tensor,
        tlg: torch.Tensor,
        suv_max: torch.Tensor,
        suv_mean: torch.Tensor,
        glcm_features: torch.Tensor,
        glrlm_features: torch.Tensor,
        mri_pattern: torch.Tensor,
        dce_wash_in: torch.Tensor,
        dce_wash_out: torch.Tensor,
        dce_auc: torch.Tensor,
        genomics_emb: Optional[torch.Tensor] = None,
        proteomics_emb: Optional[torch.Tensor] = None,
        metabolomics_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Integrate pathomics, morphology, and radiomics features.

        Args:
            tiles: WSI tiles, shape (batch_size, num_tiles, 3, 256, 256) or
                   (num_tiles, 3, 256, 256) for single patient.
            pattern_class, mvd_count, fibrosis_grade: Morphology labels.
            mtv, tlg, suv_max, suv_mean, glcm_features, glrlm_features: PET/CT features.
            mri_pattern, dce_wash_in, dce_wash_out, dce_auc: MRI features.
            genomics_emb, proteomics_emb, metabolomics_emb: Optional genomic modalities
                (batch_size, output_dim). If not provided, initialized as zeros.

        Returns:
            - integrated_embedding: Final integrated embedding (batch_size, output_dim).
            - metadata: Dict with intermediate outputs and diagnostic info.
        """
        metadata = {}

        # Handle batch dimension
        squeeze_batch = False
        if tiles.dim() == 4:
            tiles = tiles.unsqueeze(0)
            squeeze_batch = True

        batch_size = tiles.shape[0]

        # Encode tiles
        num_tiles = tiles.shape[1]
        tiles_flat = tiles.view(batch_size * num_tiles, *tiles.shape[2:])
        tile_embeddings = self.tile_encoder(tiles_flat)
        tile_embeddings = tile_embeddings.view(batch_size, num_tiles, -1)
        metadata["tile_embeddings_shape"] = tile_embeddings.shape

        # TransMIL pooling
        pathomics_emb = self.transmil_pooling(tile_embeddings)
        pathomics_emb = self.pathomics_proj(pathomics_emb)
        metadata["pathomics_embedding_shape"] = pathomics_emb.shape

        # Extract morphology features
        morphology_emb = self.morphology_extractor(
            pattern_class, mvd_count, fibrosis_grade
        )
        metadata["morphology_embedding_shape"] = morphology_emb.shape

        # Fuse pathomics and morphology
        pathomics_morphology = torch.cat([pathomics_emb, morphology_emb], dim=-1)
        pathomics_fused = self.pathomics_morphology_fusion(pathomics_morphology)
        metadata["pathomics_morphology_fused"] = pathomics_fused.shape

        # Encode radiomics features
        radiomics_emb = self.radiomics_encoder(
            mtv, tlg, suv_max, suv_mean, glcm_features, glrlm_features,
            mri_pattern, dce_wash_in, dce_wash_out, dce_auc
        )
        metadata["radiomics_embedding_shape"] = radiomics_emb.shape

        # Use provided genomic embeddings or initialize as zeros
        if genomics_emb is None:
            genomics_emb = torch.zeros(batch_size, self.output_dim, device=tiles.device)
        if proteomics_emb is None:
            proteomics_emb = torch.zeros(batch_size, self.output_dim, device=tiles.device)
        if metabolomics_emb is None:
            metabolomics_emb = torch.zeros(batch_size, self.output_dim, device=tiles.device)

        # Hierarchical cross-attention fusion
        fused_embedding, layer_outputs = self.hierarchical_fusion(
            genomics_emb, proteomics_emb, metabolomics_emb,
            pathomics_fused, radiomics_emb
        )
        metadata["layer_outputs"] = {k: v.shape for k, v in layer_outputs.items()}

        # Final projection
        integrated_embedding = self.final_proj(fused_embedding)
        integrated_embedding = self.final_norm(integrated_embedding)

        if squeeze_batch:
            integrated_embedding = integrated_embedding.squeeze(0)

        metadata["output_shape"] = integrated_embedding.shape

        logger.info(f"Integration complete. Output shape: {integrated_embedding.shape}")

        return integrated_embedding, metadata
