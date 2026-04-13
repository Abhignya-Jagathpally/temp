"""
Shared utilities for clinical modules in ResistanceMap.

Provides treatment registries, feature normalization, genetic encoding,
and survival analysis utilities for multiple myeloma applications.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


# ============================================================================
# NCCN 2025 Multiple Myeloma Treatment Registry
# ============================================================================

@dataclass
class MMRegimen:
    """Metadata for an NCCN-approved MM treatment regimen."""
    name: str
    mechanism: List[str]  # e.g., ['proteasome_inhibitor', 'immunomodulator']
    line_of_therapy: int  # 1 = induction, 2 = consolidation, 3+ = relapsed
    is_maintenance: bool = False
    contraindications: List[str] = field(default_factory=list)
    typical_duration_months: Optional[int] = None
    iss_optimal_range: Tuple[int, int] = (1, 3)  # ISS stage range
    requires_monitoring: List[str] = field(default_factory=list)  # e.g., ['neuropathy', 'renal_function']


class MMTreatmentRegistry(Enum):
    """NCCN 2025-approved multiple myeloma regimens."""

    # Induction (1st-line)
    VRd = MMRegimen(
        name='Bortezomib-Lenalidomide-Dexamethasone',
        mechanism=['proteasome_inhibitor', 'immunomodulator', 'corticosteroid'],
        line_of_therapy=1,
        typical_duration_months=4,
        requires_monitoring=['neuropathy', 'thromboembolism']
    )

    KRd = MMRegimen(
        name='Carfilzomib-Lenalidomide-Dexamethasone',
        mechanism=['proteasome_inhibitor', 'immunomodulator', 'corticosteroid'],
        line_of_therapy=1,
        typical_duration_months=4,
        requires_monitoring=['cardiac_toxicity', 'renal_function']
    )

    DRd = MMRegimen(
        name='Daratumumab-Lenalidomide-Dexamethasone',
        mechanism=['monoclonal_antibody', 'immunomodulator', 'corticosteroid'],
        line_of_therapy=1,
        typical_duration_months=4,
        requires_monitoring=['infusion_reaction', 'cytopenias']
    )

    DKRd = MMRegimen(
        name='Daratumumab-Carfilzomib-Lenalidomide-Dexamethasone',
        mechanism=['monoclonal_antibody', 'proteasome_inhibitor', 'immunomodulator', 'corticosteroid'],
        line_of_therapy=1,
        typical_duration_months=4,
        requires_monitoring=['infusion_reaction', 'cardiac_toxicity', 'neuropathy']
    )

    # Maintenance
    VRd_maintenance = MMRegimen(
        name='Bortezomib-Lenalidomide-Dexamethasone Maintenance',
        mechanism=['proteasome_inhibitor', 'immunomodulator', 'corticosteroid'],
        line_of_therapy=1,
        is_maintenance=True,
        typical_duration_months=24,
        requires_monitoring=['neuropathy']
    )

    Rd_maintenance = MMRegimen(
        name='Lenalidomide-Dexamethasone Maintenance',
        mechanism=['immunomodulator', 'corticosteroid'],
        line_of_therapy=1,
        is_maintenance=True,
        typical_duration_months=24,
        requires_monitoring=['thromboembolism']
    )

    # Relapsed/Refractory
    IRd = MMRegimen(
        name='Ixazomib-Lenalidomide-Dexamethasone',
        mechanism=['proteasome_inhibitor', 'immunomodulator', 'corticosteroid'],
        line_of_therapy=2,
        typical_duration_months=6,
        requires_monitoring=['neuropathy', 'renal_function']
    )

    CYAD = MMRegimen(
        name='Cyclophosphamide-Bortezomib-Dexamethasone',
        mechanism=['alkylating_agent', 'proteasome_inhibitor', 'corticosteroid'],
        line_of_therapy=2,
        typical_duration_months=3,
        contraindications=['prior_stem_cell_transplant_within_6mo'],
        requires_monitoring=['neuropathy', 'cytopenias']
    )

    PomDex = MMRegimen(
        name='Pomalidomide-Dexamethasone',
        mechanism=['immunomodulator', 'corticosteroid'],
        line_of_therapy=2,
        typical_duration_months=6,
        contraindications=['del_17p'],
        requires_monitoring=['thromboembolism', 'neuropathy']
    )

    # CAR-T eligible
    CART_preparation = MMRegimen(
        name='CAR-T Preparation (Rd or VRd)',
        mechanism=['immunomodulator', 'proteasome_inhibitor', 'corticosteroid'],
        line_of_therapy=2,
        typical_duration_months=2,
        requires_monitoring=['tumor_burden', 'immune_status']
    )

    # Novel agents (2025)
    BsDex = MMRegimen(
        name='Bispecific Antibody-Dexamethasone',
        mechanism=['bispecific_antibody', 'corticosteroid'],
        line_of_therapy=2,
        typical_duration_months=6,
        requires_monitoring=['cytokine_release', 'immune_activation']
    )


# ============================================================================
# Clinical Feature Normalizer
# ============================================================================

class ClinicalFeatureNormalizer:
    """Standard normalization for clinical features in MM."""

    # Reference ranges for normalization
    ISS_STAGES = {
        'I': 1,
        'II': 2,
        'III': 3,
    }

    ECOG_SCORES = list(range(0, 5))  # 0-4

    EGFR_NORMAL = 90  # mL/min/1.73m2
    EGFR_SEVERE = 15

    LDH_NORMAL = 245  # U/L (upper limit)
    LDH_HIGH = 800

    M_PROTEIN_THRESHOLD = 3.0  # g/dL
    FLC_RATIO_NORMAL = (0.26, 1.65)

    @staticmethod
    def normalize_iss_stage(iss_value: Union[str, int]) -> float:
        """Normalize ISS stage to [0, 1]."""
        if isinstance(iss_value, str):
            stage = ClinicalFeatureNormalizer.ISS_STAGES.get(iss_value, 2)
        else:
            stage = int(iss_value)
        return (stage - 1) / 2.0  # Maps 1->0, 2->0.5, 3->1

    @staticmethod
    def normalize_ecog(ecog_value: int) -> float:
        """Normalize ECOG performance status to [0, 1]."""
        return float(ecog_value) / 4.0

    @staticmethod
    def normalize_egfr(egfr_value: float) -> float:
        """Normalize eGFR to [0, 1]. Lower is worse."""
        # Clamp to [SEVERE, NORMAL] and normalize
        clamped = np.clip(egfr_value, ClinicalFeatureNormalizer.EGFR_SEVERE,
                          ClinicalFeatureNormalizer.EGFR_NORMAL)
        return (clamped - ClinicalFeatureNormalizer.EGFR_SEVERE) / \
               (ClinicalFeatureNormalizer.EGFR_NORMAL - ClinicalFeatureNormalizer.EGFR_SEVERE)

    @staticmethod
    def normalize_ldh(ldh_value: float) -> float:
        """Normalize LDH to [0, 1]. Lower is better."""
        # Values below normal = 0, at high = 1
        if ldh_value <= ClinicalFeatureNormalizer.LDH_NORMAL:
            return 0.0
        normalized = (ldh_value - ClinicalFeatureNormalizer.LDH_NORMAL) / \
                     (ClinicalFeatureNormalizer.LDH_HIGH - ClinicalFeatureNormalizer.LDH_NORMAL)
        return np.clip(normalized, 0.0, 1.0)

    @staticmethod
    def normalize_m_protein(m_protein_value: float) -> float:
        """Normalize M-protein level to [0, 1]. Higher is worse."""
        normalized = m_protein_value / ClinicalFeatureNormalizer.M_PROTEIN_THRESHOLD
        return np.clip(normalized, 0.0, 1.0)

    @staticmethod
    def normalize_flc_ratio(flc_ratio: float) -> float:
        """Normalize free light chain ratio to [0, 1]."""
        lower, upper = ClinicalFeatureNormalizer.FLC_RATIO_NORMAL
        if lower <= flc_ratio <= upper:
            return 0.0  # Normal range
        elif flc_ratio < lower:
            return np.clip((lower - flc_ratio) / lower, 0.0, 1.0)
        else:
            return np.clip((flc_ratio - upper) / upper, 0.0, 1.0)

    @staticmethod
    def normalize_features(features: Dict[str, float]) -> Dict[str, float]:
        """Normalize all clinical features at once."""
        normalized = {}

        if 'iss_stage' in features:
            normalized['iss_stage'] = ClinicalFeatureNormalizer.normalize_iss_stage(features['iss_stage'])
        if 'ecog' in features:
            normalized['ecog'] = ClinicalFeatureNormalizer.normalize_ecog(features['ecog'])
        if 'egfr' in features:
            normalized['egfr'] = ClinicalFeatureNormalizer.normalize_egfr(features['egfr'])
        if 'ldh' in features:
            normalized['ldh'] = ClinicalFeatureNormalizer.normalize_ldh(features['ldh'])
        if 'm_protein' in features:
            normalized['m_protein'] = ClinicalFeatureNormalizer.normalize_m_protein(features['m_protein'])
        if 'flc_ratio' in features:
            normalized['flc_ratio'] = ClinicalFeatureNormalizer.normalize_flc_ratio(features['flc_ratio'])

        return normalized


# ============================================================================
# Cytogenetic Encoder
# ============================================================================

class CytogeneticEncoder:
    """Binary encoding of MM cytogenetic markers."""

    MARKERS = [
        'del_17p',      # TP53 deletion (adverse)
        't_4_14',       # FGFR3/MMSET (intermediate-adverse)
        't_14_16',      # MAF (adverse)
        't_14_20',      # MAFB (adverse)
        'gain_1q',      # Extra 1q (adverse)
        'del_1p',       # 1p deletion (adverse)
    ]

    @staticmethod
    def encode(cytogenetics: Dict[str, bool]) -> np.ndarray:
        """
        Encode cytogenetic markers to binary vector.

        Args:
            cytogenetics: Dict with marker names (e.g., 'del_17p') as keys, bool as values

        Returns:
            Binary vector of shape (6,)
        """
        encoding = np.zeros(len(CytogeneticEncoder.MARKERS), dtype=int)
        for i, marker in enumerate(CytogeneticEncoder.MARKERS):
            if cytogenetics.get(marker, False):
                encoding[i] = 1
        return encoding

    @staticmethod
    def decode(encoding: np.ndarray) -> Dict[str, bool]:
        """Decode binary vector back to cytogenetics dict."""
        return {marker: bool(encoding[i])
                for i, marker in enumerate(CytogeneticEncoder.MARKERS)}

    @staticmethod
    def get_high_risk_status(cytogenetics: Dict[str, bool]) -> bool:
        """Determine if patient has high-risk cytogenetics."""
        high_risk_markers = ['del_17p', 't_4_14', 't_14_16', 't_14_20']
        return any(cytogenetics.get(marker, False) for marker in high_risk_markers)


# ============================================================================
# Treatment History Encoder
# ============================================================================

class TreatmentHistoryEncoder:
    """One-hot and duration encoding of treatment history."""

    COMMON_AGENTS = [
        'bortezomib', 'carfilzomib', 'ixazomib',  # PIs
        'lenalidomide', 'pomalidomide', 'thalidomide',  # IMiDs
        'daratumumab', 'isatuximab',  # Monoclonal antibodies
        'dexamethasone', 'prednisone',  # Corticosteroids
        'cyclophosphamide', 'melphalan',  # Alkylating agents
    ]

    @staticmethod
    def encode_agents(agents_used: List[str], durations_months: Optional[List[int]] = None) -> Dict:
        """
        Encode treatment agents as one-hot vectors with optional duration.

        Args:
            agents_used: List of agent names (lowercase)
            durations_months: Optional list of durations in months (must match length)

        Returns:
            Dict with 'agents' (one-hot) and 'durations' (normalized) keys
        """
        if durations_months is None:
            durations_months = [1] * len(agents_used)

        one_hot = np.zeros(len(TreatmentHistoryEncoder.COMMON_AGENTS), dtype=int)
        duration_vector = np.zeros(len(TreatmentHistoryEncoder.COMMON_AGENTS), dtype=float)

        for agent, duration in zip(agents_used, durations_months):
            agent_lower = agent.lower()
            if agent_lower in TreatmentHistoryEncoder.COMMON_AGENTS:
                idx = TreatmentHistoryEncoder.COMMON_AGENTS.index(agent_lower)
                one_hot[idx] = 1
                # Normalize duration: log scale, capped at 36 months
                duration_vector[idx] = np.log1p(min(duration, 36)) / np.log1p(36)

        return {
            'agents': one_hot,
            'durations': duration_vector,
            'num_unique_agents': int(np.sum(one_hot)),
            'total_treatment_months': float(np.sum(durations_months)),
        }


# ============================================================================
# Survival Utilities
# ============================================================================

class SurvivalUtils:
    """Kaplan-Meier and log-rank test utilities using pure NumPy."""

    @staticmethod
    def kaplan_meier(time: np.ndarray, event: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Kaplan-Meier survival curve.

        Args:
            time: Array of follow-up times
            event: Binary array indicating event occurrence (1=event, 0=censored)

        Returns:
            (sorted_times, survival_prob, stderr): Sorted event times, survival probabilities, standard errors
        """
        # Sort by time
        sort_idx = np.argsort(time)
        time_sorted = time[sort_idx]
        event_sorted = event[sort_idx]

        # Get unique event times
        unique_times, unique_indices = np.unique(time_sorted, return_index=True)

        n_risk = np.zeros(len(unique_times))
        n_event = np.zeros(len(unique_times))

        for i, t_unique in enumerate(unique_times):
            n_risk[i] = np.sum(time_sorted >= t_unique)
            n_event[i] = np.sum((time_sorted == t_unique) & (event_sorted == 1))

        # Kaplan-Meier product-limit estimator
        survival_prob = np.ones(len(unique_times))
        variance = np.zeros(len(unique_times))

        for i in range(len(unique_times)):
            if n_risk[i] > 0:
                survival_prob[i] = np.prod(1.0 - (n_event[:i+1] / n_risk[:i+1]))
                # Greenwood's formula for variance
                variance[i] = np.sum(n_event[:i+1] / (n_risk[:i+1] * (n_risk[:i+1] - n_event[:i+1])))

        stderr = np.sqrt(variance) * survival_prob

        return unique_times, survival_prob, stderr

    @staticmethod
    def log_rank_test(time1: np.ndarray, event1: np.ndarray,
                      time2: np.ndarray, event2: np.ndarray) -> Tuple[float, float]:
        """
        Perform log-rank test comparing two survival curves.

        Args:
            time1, event1: Time and event arrays for group 1
            time2, event2: Time and event arrays for group 2

        Returns:
            (test_statistic, p_value): Log-rank test statistic and p-value
        """
        # Combine data
        time_all = np.concatenate([time1, time2])
        event_all = np.concatenate([event1, event2])
        group = np.concatenate([np.zeros(len(time1)), np.ones(len(time2))])

        # Sort by time
        sort_idx = np.argsort(time_all)
        time_all = time_all[sort_idx]
        event_all = event_all[sort_idx]
        group = group[sort_idx]

        unique_times = np.unique(time_all[event_all == 1])

        o1 = 0  # Observed events in group 1
        e1 = 0  # Expected events in group 1
        var_logrank = 0  # Variance

        for t in unique_times:
            mask_at_t = (time_all == t)
            n_event_at_t = np.sum(event_all[mask_at_t] == 1)
            n_risk_at_t = np.sum(time_all >= t)
            n_risk1_at_t = np.sum((time_all >= t) & (group == 0))

            o1 += np.sum(event_all[mask_at_t & (group == 0)] == 1)
            if n_risk_at_t > 0:
                e1 += (n_risk1_at_t / n_risk_at_t) * n_event_at_t

                # Hypergeometric variance
                if n_risk_at_t > 1:
                    var_logrank += (n_risk1_at_t * (n_risk_at_t - n_risk1_at_t) * n_event_at_t *
                                   (n_risk_at_t - n_event_at_t)) / (n_risk_at_t ** 2 * (n_risk_at_t - 1))

        # Log-rank test statistic
        if var_logrank > 0:
            z_stat = (o1 - e1) / np.sqrt(var_logrank)
            chi2_stat = z_stat ** 2
        else:
            chi2_stat = 0.0

        # Two-tailed p-value from chi-squared distribution with df=1
        p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=1)

        return float(chi2_stat), float(p_value)
