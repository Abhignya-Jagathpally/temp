"""
Gap 5.5: SMM-to-MM Progression Prediction Module

Implements PANGEA-style static risk scoring, clonal/immune trajectory analysis,
and Weibull mixture cure survival modeling for predicting SMM progression to MM.

Classes:
  - PANGEARiskScorer: Static features → 2-year progression probability
  - ClonalExpansionTracker: VAF trajectories → expansion rates
  - ImmuneDecayEstimator: Immune marker trends → surveillance score
  - MixtureCureSurvival: Weibull survival with cure fraction
  - SMMProgressionPredictor: Top-level predictor with risk tier assignment
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import special, stats

logger = logging.getLogger(__name__)


class PANGEARiskScorer(nn.Module):
    """
    PANGEA-style static risk model using M-protein, BMPC, FLC ratio, and genomics.

    Logistic regression head combining:
      - M-protein level (g/dL)
      - Bone marrow plasma cell (BMPC) percentage
      - Free light chain (FLC) ratio
      - Genomic drivers: t(4;14), del(17p), gain(1q), MYC rearrangement

    Outputs 2-year progression probability.
    """

    def __init__(self, hidden_dim: int = 32):
        """
        Initialize static risk scorer.

        Args:
            hidden_dim: Hidden layer dimension for feature processing.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Feature layers
        self.feature_embedding = nn.Sequential(
            nn.Linear(7, hidden_dim),  # 3 lab + 4 genomic binary features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Logistic head for 2-year progression probability
        self.progression_head = nn.Linear(hidden_dim // 2, 1)

        # Feature scaling parameters (learnable)
        self.register_parameter('m_protein_scale', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('bmpc_scale', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('flc_scale', nn.Parameter(torch.tensor(1.0)))

        logger.info("PANGEARiskScorer initialized")

    def forward(
        self,
        m_protein: torch.Tensor,
        bmpc: torch.Tensor,
        flc_ratio: torch.Tensor,
        t414: torch.Tensor,
        del17p: torch.Tensor,
        gain1q: torch.Tensor,
        myc_rearr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 2-year progression probability.

        Args:
            m_protein: M-protein level (g/dL), shape (batch,)
            bmpc: Bone marrow PC percentage, shape (batch,)
            flc_ratio: Free light chain ratio, shape (batch,)
            t414: t(4;14) presence (binary), shape (batch,)
            del17p: del(17p) presence (binary), shape (batch,)
            gain1q: gain(1q) presence (binary), shape (batch,)
            myc_rearr: MYC rearrangement (binary), shape (batch,)

        Returns:
            Progression probability (0-1), shape (batch,)
        """
        batch_size = m_protein.shape[0]

        # Normalize features
        m_norm = (m_protein / 5.0) * self.m_protein_scale
        bmpc_norm = (bmpc / 60.0) * self.bmpc_scale
        flc_norm = torch.clamp(flc_ratio / 25.0, 0, 2) * self.flc_scale

        # Stack features
        features = torch.stack([
            m_norm, bmpc_norm, flc_norm,
            t414.float(), del17p.float(), gain1q.float(), myc_rearr.float()
        ], dim=1)  # (batch, 7)

        # Embed and score
        embedded = self.feature_embedding(features)  # (batch, hidden_dim // 2)
        logits = self.progression_head(embedded)  # (batch, 1)
        prob = torch.sigmoid(logits).squeeze(-1)  # (batch,)

        return prob


class ClonalExpansionTracker(nn.Module):
    """
    Tracks clonal expansion from serial VAF (variant allele frequency) measurements.

    Estimates clonal expansion rate from longitudinal WGS VAF dynamics.
    Assumes exponential VAF growth: VAF(t) = VAF_0 * exp(growth_rate * t)
    """

    def __init__(self):
        """Initialize clonal expansion tracker."""
        super().__init__()
        logger.info("ClonalExpansionTracker initialized")

    @staticmethod
    def estimate_growth_rate(
        timepoints: np.ndarray,
        vafs: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Estimate clonal expansion rate from VAF trajectory.

        Args:
            timepoints: Sample timepoints in years, shape (n_timepoints,)
            vafs: VAF measurements at timepoints, shape (n_timepoints,)

        Returns:
            (growth_rate, confidence): Annual growth rate (per year) and R^2
        """
        # Filter out very low VAFs (detection limit ~1%)
        mask = vafs > 0.01
        if mask.sum() < 2:
            return 0.0, 0.0

        t_valid = timepoints[mask]
        vaf_valid = vafs[mask]

        # Log-linear regression: ln(VAF) = ln(VAF_0) + growth_rate * t
        log_vafs = np.log(vaf_valid)
        coeffs = np.polyfit(t_valid, log_vafs, 1)
        growth_rate = coeffs[0]  # per year

        # Compute R^2
        preds = np.polyval(coeffs, t_valid)
        ss_res = np.sum((log_vafs - preds) ** 2)
        ss_tot = np.sum((log_vafs - np.mean(log_vafs)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return float(growth_rate), float(r2)

    def forward(
        self,
        timepoints: torch.Tensor,
        vafs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute clonal expansion rates for batch of samples.

        Args:
            timepoints: Timepoints, shape (batch, max_timepoints) with padding
            vafs: VAF measurements, shape (batch, max_timepoints) with padding

        Returns:
            (growth_rates, confidences): shape (batch,) each
        """
        batch_size = timepoints.shape[0]
        growth_rates = []
        confidences = []

        for i in range(batch_size):
            t = timepoints[i].cpu().numpy()
            v = vafs[i].cpu().numpy()

            # Remove padding (zero entries)
            valid = (v > 0) & (t > 0)
            if valid.sum() >= 2:
                rate, conf = self.estimate_growth_rate(t[valid], v[valid])
            else:
                rate, conf = 0.0, 0.0

            growth_rates.append(rate)
            confidences.append(conf)

        return (
            torch.tensor(growth_rates, dtype=torch.float32, device=timepoints.device),
            torch.tensor(confidences, dtype=torch.float32, device=timepoints.device),
        )


class ImmuneDecayEstimator(nn.Module):
    """
    Estimates immune surveillance decay from longitudinal immune marker trajectories.

    Tracks:
      - TCR diversity decline (T-cell clonal exhaustion)
      - NK CD38+ frequency decline (NK cell activation loss)
      - Treg frequency increase (immune suppression rise)
      - CD8+ exhaustion markers (PD-1, TIM-3, LAG-3 co-expression)
    """

    def __init__(self):
        """Initialize immune decay estimator."""
        super().__init__()
        logger.info("ImmuneDecayEstimator initialized")

    @staticmethod
    def estimate_immune_score(
        tcr_diversity: np.ndarray,
        nk_cd38_freq: np.ndarray,
        treg_freq: np.ndarray,
        cd8_exhaustion: np.ndarray,
    ) -> float:
        """
        Compute immune surveillance score (0-1, higher = better surveillance).

        Args:
            tcr_diversity: Shannon entropy of TCR clonotypes, shape (n_timepoints,)
            nk_cd38_freq: NK CD38+ frequency, shape (n_timepoints,)
            treg_freq: Regulatory T-cell frequency, shape (n_timepoints,)
            cd8_exhaustion: CD8+ exhaustion index (0-1), shape (n_timepoints,)

        Returns:
            Immune surveillance score (0-1)
        """
        # Normalize to (0-1) ranges
        tcr_norm = (tcr_diversity - np.min(tcr_diversity)) / (
            np.max(tcr_diversity) - np.min(tcr_diversity) + 1e-8
        )
        nk_norm = (nk_cd38_freq - np.min(nk_cd38_freq)) / (
            np.max(nk_cd38_freq) - np.min(nk_cd38_freq) + 1e-8
        )
        treg_norm = (treg_freq - np.min(treg_freq)) / (
            np.max(treg_freq) - np.min(treg_freq) + 1e-8
        )
        exhaust_norm = (cd8_exhaustion - np.min(cd8_exhaustion)) / (
            np.max(cd8_exhaustion) - np.min(cd8_exhaustion) + 1e-8
        )

        # Combine with weights: TCR and NK are protective, Treg and exhaustion are detrimental
        score = (0.3 * tcr_norm[-1] +
                 0.3 * nk_norm[-1] +
                 0.2 * (1 - treg_norm[-1]) +
                 0.2 * (1 - exhaust_norm[-1]))

        return float(np.clip(score, 0, 1))

    def forward(
        self,
        tcr_diversity: torch.Tensor,
        nk_cd38_freq: torch.Tensor,
        treg_freq: torch.Tensor,
        cd8_exhaustion: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute immune surveillance scores for batch.

        Args:
            tcr_diversity: shape (batch, max_timepoints)
            nk_cd38_freq: shape (batch, max_timepoints)
            treg_freq: shape (batch, max_timepoints)
            cd8_exhaustion: shape (batch, max_timepoints)

        Returns:
            (immune_scores, decay_rates): shape (batch,) each
        """
        batch_size = tcr_diversity.shape[0]
        immune_scores = []
        decay_rates = []

        for i in range(batch_size):
            tcr = tcr_diversity[i].cpu().numpy()
            nk = nk_cd38_freq[i].cpu().numpy()
            treg = treg_freq[i].cpu().numpy()
            exhaust = cd8_exhaustion[i].cpu().numpy()

            # Remove zero padding
            valid = (tcr > 0) & (nk > 0) & (treg > 0) & (exhaust >= 0)
            if valid.sum() >= 2:
                score = self.estimate_immune_score(
                    tcr[valid], nk[valid], treg[valid], exhaust[valid]
                )
                # Estimate decay rate as slope of immune score over time
                rate = -np.mean(np.diff(score)) if len(score) > 1 else 0.0
            else:
                score = 0.5
                rate = 0.0

            immune_scores.append(score)
            decay_rates.append(rate)

        return (
            torch.tensor(immune_scores, dtype=torch.float32, device=tcr_diversity.device),
            torch.tensor(decay_rates, dtype=torch.float32, device=tcr_diversity.device),
        )


class MixtureCureSurvival(nn.Module):
    """
    Weibull survival model with cure fraction (mixture cure model).

    Models time-to-progression as:
      S(t) = π + (1 - π) * exp(-(λt)^k)

    where:
      - π: cure fraction (proportion never progressing)
      - λ: scale parameter
      - k: shape parameter (Weibull)
    """

    def __init__(self):
        """Initialize mixture cure survival model."""
        super().__init__()
        # Learnable parameters
        self.register_parameter('log_lambda', nn.Parameter(torch.tensor(-1.0)))
        self.register_parameter('log_k', nn.Parameter(torch.tensor(0.0)))
        logger.info("MixtureCureSurvival initialized")

    def forward(
        self,
        times: torch.Tensor,
        cure_fraction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute survival probability S(t) = P(T > t).

        Args:
            times: Event times, shape (batch,) in years
            cure_fraction: Cure fraction π for each sample, shape (batch,)

        Returns:
            Survival probability, shape (batch,)
        """
        lambda_param = torch.exp(self.log_lambda)
        k_param = torch.exp(self.log_k)

        # Weibull: S(t) = exp(-(λt)^k)
        weibull_surv = torch.exp(-((lambda_param * times) ** k_param))

        # Mixture: S(t) = π + (1-π)*S_Weibull(t)
        survival = cure_fraction + (1 - cure_fraction) * weibull_surv

        return torch.clamp(survival, 0, 1)

    def predict_time_to_progression(
        self,
        risk_score: torch.Tensor,
        cure_fraction: torch.Tensor,
        quantile: float = 0.5,
    ) -> torch.Tensor:
        """
        Predict time-to-progression at given quantile.

        Args:
            risk_score: Risk score (0-1), shape (batch,)
            cure_fraction: Cure fraction, shape (batch,)
            quantile: Quantile (default 0.5 for median)

        Returns:
            Time-to-progression in years, shape (batch,)
        """
        lambda_param = torch.exp(self.log_lambda)
        k_param = torch.exp(self.log_k)

        # For mixture cure model: quantile t satisfies
        # S(t) = 1 - quantile
        # Solve: π + (1-π)*exp(-(λt)^k) = 1 - quantile
        # => (λt)^k = -ln((1 - quantile - π) / (1 - π))

        numerator = 1 - quantile - cure_fraction
        denominator = 1 - cure_fraction

        # Handle edge case where cure fraction > 1 - quantile
        safe_ratio = torch.clamp(numerator / (denominator + 1e-8), 1e-8, 1.0)

        log_term = -torch.log(safe_ratio)
        t_quantile = (log_term ** (1 / k_param)) / lambda_param

        return t_quantile


class SMMProgressionPredictor(nn.Module):
    """
    Top-level SMM-to-MM progression predictor.

    Combines:
      1. Static PANGEA risk scoring
      2. Clonal expansion trajectory analysis
      3. Immune decay estimation
      4. Weibull mixture cure survival modeling

    Outputs:
      - 2-year progression probability
      - Risk tier (1=low, 2=intermediate, 3=high)
      - Predicted time-to-progression (years)
    """

    def __init__(self, hidden_dim: int = 32):
        """
        Initialize SMM progression predictor.

        Args:
            hidden_dim: Hidden dimension for static risk scorer.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Component modules
        self.static_scorer = PANGEARiskScorer(hidden_dim=hidden_dim)
        self.clonal_tracker = ClonalExpansionTracker()
        self.immune_estimator = ImmuneDecayEstimator()
        self.survival_model = MixtureCureSurvival()

        # Fusion network: combine static + trajectory signals
        self.fusion_net = nn.Sequential(
            nn.Linear(5, 16),  # static_prob, growth_rate, immune_score, decay_rate, +1 spare
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

        logger.info("SMMProgressionPredictor initialized")

    def forward(
        self,
        # Static features
        m_protein: torch.Tensor,
        bmpc: torch.Tensor,
        flc_ratio: torch.Tensor,
        t414: torch.Tensor,
        del17p: torch.Tensor,
        gain1q: torch.Tensor,
        myc_rearr: torch.Tensor,
        # Trajectory features
        timepoints: torch.Tensor,
        vafs: torch.Tensor,
        tcr_diversity: torch.Tensor,
        nk_cd38_freq: torch.Tensor,
        treg_freq: torch.Tensor,
        cd8_exhaustion: torch.Tensor,
    ) -> dict:
        """
        Predict SMM progression risk and time-to-progression.

        Args:
            m_protein, bmpc, flc_ratio, t414, del17p, gain1q, myc_rearr: Static features
            timepoints, vafs: VAF trajectory (batch, max_timepoints)
            tcr_diversity, nk_cd38_freq, treg_freq, cd8_exhaustion: Immune trajectories

        Returns:
            Dictionary with:
              - 'progression_prob': 2-year progression probability (batch,)
              - 'risk_tier': Risk tier 1/2/3 (batch,)
              - 'time_to_progression': Predicted years to MM (batch,)
              - 'cure_fraction': Estimated cure fraction (batch,)
        """
        batch_size = m_protein.shape[0]

        # 1. Static risk scoring
        static_prob = self.static_scorer(
            m_protein, bmpc, flc_ratio, t414, del17p, gain1q, myc_rearr
        )  # (batch,)

        # 2. Clonal expansion tracking
        growth_rates, growth_conf = self.clonal_tracker(timepoints, vafs)  # (batch,)
        # Normalize growth rate to (0-1) via sigmoid
        norm_growth = torch.sigmoid(torch.tensor(growth_rates))

        # 3. Immune decay estimation
        immune_scores, decay_rates = self.immune_estimator(
            tcr_diversity, nk_cd38_freq, treg_freq, cd8_exhaustion
        )  # (batch,)
        immune_scores = torch.tensor(immune_scores)

        # 4. Fusion: combine static + trajectory signals
        fusion_input = torch.stack([
            static_prob,
            norm_growth,
            immune_scores,
            torch.abs(torch.tensor(decay_rates)),
            static_prob * norm_growth,  # interaction term
        ], dim=1)  # (batch, 5)

        fusion_logits = self.fusion_net(fusion_input)  # (batch, 1)
        fusion_prob = torch.sigmoid(fusion_logits).squeeze(-1)  # (batch,)

        # Ensemble: average static and fusion
        prog_prob = 0.6 * static_prob + 0.4 * fusion_prob

        # 5. Assign risk tiers based on 2-year progression probability
        #    Tier 1: <6%, Tier 2: 6-18%, Tier 3: >18% (conservative) or >44% (PANGEA)
        risk_tier = torch.ones(batch_size, dtype=torch.long)
        risk_tier = torch.where(prog_prob >= 0.18, 2 * torch.ones_like(risk_tier), risk_tier)
        risk_tier = torch.where(prog_prob >= 0.44, 3 * torch.ones_like(risk_tier), risk_tier)

        # 6. Estimate cure fraction (inverse of progression risk)
        cure_fraction = torch.clamp(1.0 - prog_prob, 0.1, 0.9)

        # 7. Predict time-to-progression (median)
        time_to_prog = self.survival_model.predict_time_to_progression(
            prog_prob, cure_fraction, quantile=0.5
        )

        return {
            'progression_prob': prog_prob,
            'risk_tier': risk_tier,
            'time_to_progression': time_to_prog,
            'cure_fraction': cure_fraction,
            'static_risk': static_prob,
            'fusion_risk': fusion_prob,
            'clonal_growth_rate': growth_rates,
            'immune_score': immune_scores,
        }


def create_example_batch(batch_size: int = 4) -> dict:
    """
    Create example batch for testing/demonstration.

    Args:
        batch_size: Number of samples in batch.

    Returns:
        Dictionary with example inputs.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    return {
        # Static features
        'm_protein': torch.randn(batch_size).abs() * 3,  # 0-3 g/dL
        'bmpc': torch.randn(batch_size).abs() * 30 + 10,  # 10-40%
        'flc_ratio': torch.randn(batch_size).abs() * 20 + 1,  # 1-20
        't414': torch.randint(0, 2, (batch_size,)),
        'del17p': torch.randint(0, 2, (batch_size,)),
        'gain1q': torch.randint(0, 2, (batch_size,)),
        'myc_rearr': torch.randint(0, 2, (batch_size,)),
        # Trajectory features (5 timepoints)
        'timepoints': torch.arange(5, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1),
        'vafs': torch.randn(batch_size, 5).abs() * 0.3 + 0.1,
        'tcr_diversity': torch.randn(batch_size, 5).abs() + 2,
        'nk_cd38_freq': torch.randn(batch_size, 5).abs() * 0.1 + 0.05,
        'treg_freq': torch.randn(batch_size, 5).abs() * 0.05 + 0.02,
        'cd8_exhaustion': torch.randn(batch_size, 5).abs() * 0.3 + 0.2,
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example usage
    predictor = SMMProgressionPredictor(hidden_dim=32)
    batch = create_example_batch(batch_size=4)

    results = predictor(**batch)

    print("\n" + "="*60)
    print("SMM-to-MM Progression Prediction Results")
    print("="*60)
    for i in range(4):
        print(f"\nSample {i+1}:")
        print(f"  2-year progression probability: {results['progression_prob'][i]:.3f}")
        print(f"  Risk tier: {results['risk_tier'][i].item()}")
        print(f"  Time-to-progression (years): {results['time_to_progression'][i]:.2f}")
        print(f"  Cure fraction: {results['cure_fraction'][i]:.3f}")
