from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TranscriptomicEMDEncoder(nn.Module):
    """
    Encodes adhesion loss, migration propensity, and immune escape gene signatures.

    Integrates:
    - Adhesion molecules: CD56/NCAM1, VLA-4/ITGA4 (loss → EMD risk)
    - Migration axis: CXCR4/CXCL12 activity
    - Immune escape: PD-L1/CD274, VISTA/VSIR upregulation
    """

    def __init__(self, input_dim: int = 6, embedding_dim: int = 64):
        """
        Args:
            input_dim: Number of gene expression features (adhesion + migration + escape genes)
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Adhesion module: CD56/NCAM1, VLA-4/ITGA4
        self.adhesion_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        # Migration module: CXCR4/CXCL12 axis
        self.migration_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        # Immune escape module: PD-L1/CD274, VISTA/VSIR
        self.escape_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(48, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 6] tensor with gene expression values
               Expected order: [CD56, ITGA4, CXCR4, CXCL12, PD-L1, VISTA]

        Returns:
            [batch_size, embedding_dim] transcriptomics embedding
        """
        adhesion = self.adhesion_net(x[:, :2])
        migration = self.migration_net(x[:, 2:4])
        escape = self.escape_net(x[:, 4:6])

        combined = torch.cat([adhesion, migration, escape], dim=1)
        embedding = self.fusion(combined)
        return embedding


class RadiomicsEncoder(nn.Module):
    """
    Extracts and encodes PET/MRI radiomics features for EMD risk.

    Features:
    - PET SUV heterogeneity (entropy, uniformity)
    - MRI diffuse pattern (HR 4.16 for EMD)
    - DCE-MRI angiogenesis metrics (Ktrans, peak enhancement)
    """

    def __init__(self, n_features: int = 8, embedding_dim: int = 64):
        """
        Args:
            n_features: Number of radiomics features (PET entropy, PET uniformity,
                       MRI diffusion metrics, DCE metrics)
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        # PET radiomics: heterogeneity features
        self.pet_radiomics = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        # MRI diffusion: ADC, MD metrics
        self.mri_diffusion = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        # DCE-MRI perfusion: Ktrans, vp, ve
        self.dce_perfusion = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        # Texture features (GLCM, GLRLM)
        self.texture = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        self.fusion = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 8] radiomics feature vector
               Expected order: [PET_entropy, PET_uniformity, ADC, MD,
                               Ktrans, vp, GLCM_contrast, GLRLM_RLNU]

        Returns:
            [batch_size, embedding_dim] radiomics embedding
        """
        pet = self.pet_radiomics(x[:, :2])
        mri_diff = self.mri_diffusion(x[:, 2:4])
        dce = self.dce_perfusion(x[:, 4:6])
        texture = self.texture(x[:, 6:8])

        combined = torch.cat([pet, mri_diff, dce, texture], dim=1)
        embedding = self.fusion(combined)
        return embedding


class MicroenvironmentEncoder(nn.Module):
    """
    Encodes tumor microenvironment cytokine profiling for EMD prediction.

    Cytokine signatures:
    - High-risk: THBS1 (thrombospondin), NAP-2/PPBP (neutrophil chemotaxis)
    - Protective: EGF, BDNF (growth factors - low in EMD)
    """

    def __init__(self, n_cytokines: int = 4, embedding_dim: int = 64):
        """
        Args:
            n_cytokines: Number of cytokines in panel
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        self.n_cytokines = n_cytokines
        self.embedding_dim = embedding_dim

        # High-risk cytokines: THBS1, PPBP (inverted scoring)
        self.pro_migration = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        # Protective cytokines: EGF, BDNF
        self.anti_migration = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        self.fusion = nn.Sequential(
            nn.Linear(32, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 4] cytokine concentration vector
               Expected order: [THBS1, PPBP, EGF, BDNF]

        Returns:
            [batch_size, embedding_dim] microenvironment embedding
        """
        pro_mig = self.pro_migration(x[:, :2])
        anti_mig = self.anti_migration(x[:, 2:4])

        combined = torch.cat([pro_mig, anti_mig], dim=1)
        embedding = self.fusion(combined)
        return embedding


class ProteinPathwayEncoder(nn.Module):
    """
    Encodes protein signaling pathway activity scores driving EMD.

    Pathways:
    - Focal adhesion disruption (FAK, paxillin, talin loss)
    - PI3K/AKT activation
    - MAPK/ERK signaling
    - Wnt/beta-catenin pathway
    """

    def __init__(self, embedding_dim: int = 64):
        """
        Args:
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Focal adhesion: FAK, paxillin activity
        self.focal_adhesion = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        # PI3K/AKT axis
        self.pi3k_akt = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        # MAPK/ERK cascade
        self.mapk_erk = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        # Wnt/beta-catenin
        self.wnt_signaling = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        self.fusion = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 8] pathway activity scores (typically phosphorylation levels)
               Expected order: [FAK_pY397, paxillin, pAKT, PTEN, pERK, p38, beta_catenin, Tcf4]

        Returns:
            [batch_size, embedding_dim] pathway embedding
        """
        focal = self.focal_adhesion(x[:, :2])
        pi3k = self.pi3k_akt(x[:, 2:4])
        mapk = self.mapk_erk(x[:, 4:6])
        wnt = self.wnt_signaling(x[:, 6:8])

        combined = torch.cat([focal, pi3k, mapk, wnt], dim=1)
        embedding = self.fusion(combined)
        return embedding


class ClinicalRiskEncoder(nn.Module):
    """
    Encodes structured clinical features for EMD risk.

    Features:
    - LDH (elevated → EMD risk)
    - Circulating clonal cells (CCC) burden
    - High-risk cytogenetics (del(17p), t(4;14), amp(1q))
    - Prior treatment lines (chemotherapy burden)
    """

    def __init__(self, embedding_dim: int = 64):
        """
        Args:
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Lab values: LDH, calcium
        self.lab_values = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        # Cell burden: CCC, bone marrow infiltration
        self.cell_burden = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        # Cytogenetics (binary/ordinal encoding)
        self.cytogenetics = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        # Treatment history
        self.treatment_history = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        self.fusion = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 9] clinical feature vector
               Expected order: [LDH_norm, Calcium, CCC_burden, BM_infiltration,
                               del17p_present, t414_present, amp1q_present,
                               prior_treatment_lines, response_status]

        Returns:
            [batch_size, embedding_dim] clinical embedding
        """
        labs = self.lab_values(x[:, :2])
        burden = self.cell_burden(x[:, 2:4])
        cyto = self.cytogenetics(x[:, 4:7])
        tx = self.treatment_history(x[:, 7:9])

        combined = torch.cat([labs, burden, cyto, tx], dim=1)
        embedding = self.fusion(combined)
        return embedding


class EMDSurvivalHead(nn.Module):
    """
    Predicts time-to-EMD using Weibull survival model + CNS involvement flag.

    Outputs:
    - P(EMD development): binary classification head
    - Time-to-EMD: Weibull shape (k) and scale (lambda) parameters
    - CNS-MM flag: separate high-risk indicator (OS <6 months)
    """

    def __init__(self, input_dim: int = 320):
        """
        Args:
            input_dim: Dimension of fused embedding from 5 layers
        """
        super().__init__()
        self.input_dim = input_dim

        # EMD risk prediction head
        self.emd_risk_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Weibull survival parameters (log-transformed for stability)
        self.survival_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [log_k, log_lambda]
        )

        # CNS involvement (high-risk extramedullary site)
        self.cns_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, input_dim] fused embedding

        Returns:
            emd_prob: [batch_size, 1] P(EMD development)
            survival_params: [batch_size, 2] [log_k, log_lambda] for Weibull
            cns_prob: [batch_size, 1] P(CNS-MM)
        """
        emd_prob = self.emd_risk_head(x)
        survival_params = self.survival_head(x)
        cns_prob = self.cns_head(x)

        return emd_prob, survival_params, cns_prob


class EMDRiskPredictor(nn.Module):
    """
    5-layer EMD risk prediction module: fusion of transcriptomics, radiomics,
    microenvironment, protein networks, and clinical data.

    Outputs:
    - P(EMD development)
    - Time-to-EMD distribution (Weibull)
    - Risk component attribution (layer-wise logits)
    - CNS-MM flagging (median OS <6 months)
    """

    def __init__(
        self,
        transcriptomics_dim: int = 6,
        radiomics_dim: int = 8,
        cytokine_dim: int = 4,
        pathway_dim: int = 8,
        clinical_dim: int = 9,
        embedding_dim: int = 64,
    ):
        """
        Args:
            transcriptomics_dim: Number of gene expression features
            radiomics_dim: Number of radiomics features
            cytokine_dim: Number of cytokine measurements
            pathway_dim: Number of pathway activity scores
            clinical_dim: Number of clinical features
            embedding_dim: Dimension of intermediate embeddings
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Layer 1: Transcriptomics
        self.transcriptomics = TranscriptomicEMDEncoder(
            input_dim=transcriptomics_dim,
            embedding_dim=embedding_dim
        )

        # Layer 2: Radiomics
        self.radiomics = RadiomicsEncoder(
            n_features=radiomics_dim,
            embedding_dim=embedding_dim
        )

        # Layer 3: Microenvironment
        self.microenvironment = MicroenvironmentEncoder(
            n_cytokines=cytokine_dim,
            embedding_dim=embedding_dim
        )

        # Layer 4: Protein pathways
        self.pathways = ProteinPathwayEncoder(
            embedding_dim=embedding_dim
        )

        # Layer 5: Clinical risk
        self.clinical = ClinicalRiskEncoder(
            embedding_dim=embedding_dim
        )

        # Fusion: attention-based combination of 5 layers
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )

        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(embedding_dim * 5, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim * 5),
            nn.LayerNorm(embedding_dim * 5)
        )

        # Survival and risk heads
        self.survival_head = EMDSurvivalHead(input_dim=embedding_dim * 5)

    def forward(
        self,
        transcriptomics: torch.Tensor,
        radiomics: torch.Tensor,
        cytokines: torch.Tensor,
        pathways: torch.Tensor,
        clinical: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            transcriptomics: [batch_size, 6] gene expression
            radiomics: [batch_size, 8] imaging features
            cytokines: [batch_size, 4] cytokine panel
            pathways: [batch_size, 8] pathway activities
            clinical: [batch_size, 9] clinical features

        Returns:
            Dictionary with keys:
            - 'emd_probability': [batch_size, 1] P(EMD)
            - 'time_to_emd_k': [batch_size, 1] Weibull shape (k)
            - 'time_to_emd_lambda': [batch_size, 1] Weibull scale (lambda)
            - 'cns_probability': [batch_size, 1] P(CNS-MM)
            - 'layer_embeddings': dict of 5 layer embeddings for attribution
            - 'fused_embedding': [batch_size, embedding_dim*5] final representation
        """
        # Layer-wise encoding
        emb_tx = self.transcriptomics(transcriptomics)
        emb_rad = self.radiomics(radiomics)
        emb_cyto = self.microenvironment(cytokines)
        emb_path = self.pathways(pathways)
        emb_clin = self.clinical(clinical)

        # Store layer embeddings for SHAP-style attribution
        layer_embeddings = {
            'transcriptomics': emb_tx,
            'radiomics': emb_rad,
            'microenvironment': emb_cyto,
            'pathways': emb_path,
            'clinical': emb_clin,
        }

        # Attention-based fusion of layers
        stacked = torch.stack([emb_tx, emb_rad, emb_cyto, emb_path, emb_clin], dim=1)
        attn_out, _ = self.fusion_attention(stacked, stacked, stacked)

        # Flatten and project
        batch_size = attn_out.shape[0]
        attn_flat = attn_out.reshape(batch_size, -1)
        fused = self.fusion_proj(attn_flat)

        # Predict outcomes
        emd_prob, survival_params, cns_prob = self.survival_head(fused)

        # Weibull parameterization (exp for positivity)
        k = torch.exp(survival_params[:, 0:1])
        lambda_param = torch.exp(survival_params[:, 1:2])

        return {
            'emd_probability': emd_prob,
            'time_to_emd_k': k,
            'time_to_emd_lambda': lambda_param,
            'cns_probability': cns_prob,
            'layer_embeddings': layer_embeddings,
            'fused_embedding': fused,
        }

    def compute_layer_attribution(
        self,
        layer_embeddings: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SHAP-style attribution scores for each layer.

        Args:
            layer_embeddings: Dictionary of layer embeddings from forward pass

        Returns:
            Dictionary mapping layer names to attribution scores
        """
        attribution = {}
        for layer_name, embedding in layer_embeddings.items():
            # L2 norm of embedding as simple attribution metric
            score = torch.norm(embedding, p=2, dim=1, keepdim=True)
            attribution[layer_name] = score

        return attribution

    def get_risk_summary(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Generate interpretable risk summary from model outputs.

        Args:
            outputs: Dictionary from forward pass

        Returns:
            Dictionary with risk stratification and clinical flags
        """
        emd_prob = outputs['emd_probability'].detach().cpu().numpy()
        cns_prob = outputs['cns_probability'].detach().cpu().numpy()
        k = outputs['time_to_emd_k'].detach().cpu().numpy()
        lambda_param = outputs['time_to_emd_lambda'].detach().cpu().numpy()

        # Risk stratification: EMD development probability
        risk_category = np.where(
            emd_prob > 0.7,
            'High',
            np.where(emd_prob > 0.4, 'Intermediate', 'Low')
        )

        # Time-to-EMD median from Weibull: median = lambda * ln(2)^(1/k)
        median_time_to_emd = lambda_param * np.power(np.log(2), 1.0 / k)

        # CNS flagging: high priority if P(CNS) > 0.5 (OS <6 months)
        cns_flag = cns_prob > 0.5

        return {
            'emd_probability': emd_prob,
            'emd_risk_category': risk_category,
            'cns_probability': cns_prob,
            'cns_high_risk': cns_flag,
            'median_time_to_emd_months': median_time_to_emd,
            'weibull_k': k,
            'weibull_lambda': lambda_param,
        }


def create_emd_predictor(
    embedding_dim: int = 64,
) -> EMDRiskPredictor:
    """
    Factory function to create initialized EMD risk predictor.

    Args:
        embedding_dim: Intermediate embedding dimension

    Returns:
        EMDRiskPredictor module
    """
    logger.info(f"Creating EMD predictor with embedding_dim={embedding_dim}")

    model = EMDRiskPredictor(
        transcriptomics_dim=6,
        radiomics_dim=8,
        cytokine_dim=4,
        pathway_dim=8,
        clinical_dim=9,
        embedding_dim=embedding_dim,
    )

    # Log model size
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"EMD predictor initialized with {n_params:,} parameters")

    return model
