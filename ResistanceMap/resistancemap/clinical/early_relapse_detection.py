from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from scipy.stats import linregress

logger = logging.getLogger(__name__)


@dataclass
class MProteinKineticsParams:
    """Parameters from M-protein kinetics fitting."""
    model_type: str  # "mono_exponential" or "bi_exponential"
    alpha_1: float
    alpha_2: Optional[float] = None
    amplitude_a: Optional[float] = None
    amplitude_b: Optional[float] = None
    m_0: float = 1.0
    r_squared: float = 0.0
    fit_error: Optional[str] = None


@dataclass
class EarlyWarningSignal:
    """Early warning signal indicators."""
    signal_name: str
    signal_value: float
    threshold: float
    is_triggered: bool
    clinical_urgency: str  # "low", "moderate", "high"


@dataclass
class EarlyRelapseRiskAssessment:
    """Comprehensive early relapse risk assessment."""
    er18_risk_probability: float  # Probability of relapse within 18 months
    risk_category: str  # "low", "intermediate", "high"
    contributing_factors: list[str]
    kinetic_trajectory: str  # "favorable", "stable", "unfavorable"
    recommended_action: str


def _mono_exponential(t: np.ndarray, m0: float, alpha: float) -> np.ndarray:
    """M(t) = M_0 * exp(-alpha * t)."""
    return m0 * np.exp(-alpha * t)


def _bi_exponential(
    t: np.ndarray, m0: float, alpha1: float, a: float, alpha2: float, b: float
) -> np.ndarray:
    """M(t) = A*exp(-alpha1*t) + B*exp(-alpha2*t), normalized to M_0."""
    return (a * np.exp(-alpha1 * t) + b * np.exp(-alpha2 * t)) / m0


class MProteinKineticsModel(nn.Module):
    """
    M-protein kinetics model with mono and bi-exponential fitting.

    Supports standard single-phase decay and biphasic response patterns
    common in NDMM treatment response.
    """

    def __init__(self, regimen_type: str = "standard"):
        """
        Initialize kinetics model.

        Args:
            regimen_type: Treatment regimen ("standard", "intensive", "maintenance")
        """
        super().__init__()
        self.regimen_type = regimen_type
        # Regimen-specific baseline decay constants (week^-1)
        self.alpha_baselines = {
            "standard": 0.08,
            "intensive": 0.12,
            "maintenance": 0.04,
        }
        self.alpha_baseline = self.alpha_baselines.get(regimen_type, 0.08)

    def fit_kinetics(
        self,
        timepoints: np.ndarray,
        m_protein_values: np.ndarray,
        m_0: Optional[float] = None,
        try_biexponential: bool = True,
    ) -> MProteinKineticsParams:
        """
        Fit M-protein kinetics to mono or bi-exponential model.

        Args:
            timepoints: Time points in weeks (shape: (n,))
            m_protein_values: M-protein concentration (shape: (n,))
            m_0: Baseline M-protein (inferred from data if None)
            try_biexponential: If True, attempt bi-exponential fit

        Returns:
            MProteinKineticsParams with fitted parameters
        """
        if len(timepoints) < 2:
            logger.warning("Insufficient data points for kinetics fitting")
            return MProteinKineticsParams(
                model_type="mono_exponential",
                alpha_1=self.alpha_baseline,
                m_0=m_0 or m_protein_values[0],
                fit_error="Insufficient data"
            )

        if m_0 is None:
            m_0 = m_protein_values[0]

        # Sort by timepoint
        idx = np.argsort(timepoints)
        t_sorted = timepoints[idx]
        m_sorted = m_protein_values[idx]

        # Try mono-exponential fit first
        try:
            popt_mono, _ = curve_fit(
                _mono_exponential,
                t_sorted,
                m_sorted,
                p0=[m_0, self.alpha_baseline],
                bounds=([0.1, 0.001], [10 * m_0, 1.0]),
                maxfev=5000,
            )
            m_pred_mono = _mono_exponential(t_sorted, *popt_mono)
            ss_res_mono = np.sum((m_sorted - m_pred_mono) ** 2)
            ss_tot = np.sum((m_sorted - np.mean(m_sorted)) ** 2)
            r2_mono = 1 - (ss_res_mono / ss_tot) if ss_tot > 0 else 0

            logger.debug(f"Mono-exponential fit: alpha={popt_mono[1]:.4f}, R²={r2_mono:.4f}")

        except RuntimeError as e:
            logger.warning(f"Mono-exponential fit failed: {e}")
            return MProteinKineticsParams(
                model_type="mono_exponential",
                alpha_1=self.alpha_baseline,
                m_0=m_0,
                fit_error=str(e)
            )

        # Try bi-exponential fit if requested
        if try_biexponential and len(t_sorted) >= 4:
            try:
                popt_bi, _ = curve_fit(
                    _bi_exponential,
                    t_sorted,
                    m_sorted,
                    p0=[m_0, self.alpha_baseline * 2, 0.5 * m_0, self.alpha_baseline / 2, 0.5 * m_0],
                    bounds=([0.1, 0.001, 0, 0.001, 0], [10 * m_0, 1.0, 10 * m_0, 1.0, 10 * m_0]),
                    maxfev=5000,
                )
                m_pred_bi = _bi_exponential(t_sorted, *popt_bi)
                ss_res_bi = np.sum((m_sorted - m_pred_bi) ** 2)
                r2_bi = 1 - (ss_res_bi / ss_tot) if ss_tot > 0 else 0

                logger.debug(f"Bi-exponential fit: α₁={popt_bi[1]:.4f}, α₂={popt_bi[3]:.4f}, R²={r2_bi:.4f}")

                # Use bi-exponential if significantly better
                if r2_bi > r2_mono + 0.05:
                    return MProteinKineticsParams(
                        model_type="bi_exponential",
                        alpha_1=popt_bi[1],
                        alpha_2=popt_bi[3],
                        amplitude_a=popt_bi[2],
                        amplitude_b=popt_bi[4],
                        m_0=m_0,
                        r_squared=r2_bi,
                    )

            except RuntimeError as e:
                logger.debug(f"Bi-exponential fit failed: {e}, using mono-exponential")

        return MProteinKineticsParams(
            model_type="mono_exponential",
            alpha_1=popt_mono[1],
            m_0=m_0,
            r_squared=r2_mono,
        )

    def forward(
        self,
        kinetics_params: MProteinKineticsParams,
        timepoints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict M-protein concentration given kinetics parameters.

        Args:
            kinetics_params: Fitted kinetics parameters
            timepoints: Time points in weeks (shape: (n,))

        Returns:
            Predicted M-protein values (shape: (n,))
        """
        t = timepoints.cpu().numpy() if isinstance(timepoints, torch.Tensor) else timepoints

        if kinetics_params.model_type == "bi_exponential":
            m_pred = (
                kinetics_params.amplitude_a * np.exp(-kinetics_params.alpha_1 * t)
                + kinetics_params.amplitude_b * np.exp(-kinetics_params.alpha_2 * t)
            ) / kinetics_params.m_0
        else:
            m_pred = _mono_exponential(t, kinetics_params.m_0, kinetics_params.alpha_1)

        return torch.from_numpy(m_pred).float()


class EarlyWarningSignalDetector:
    """
    Rule-based detector for early warning signals of relapse.

    Integrates mHR-IMS criteria and kinetic thresholds.
    """

    def __init__(self):
        """Initialize warning signal thresholds."""
        self.thresholds = {
            "m_protein_reduction_week4": 0.25,  # <25% reduction
            "plateau_duration": 4,  # weeks 8-12
            "ldh_rise_threshold": 1.1,  # 10% rise
        }

    def detect_warnings(
        self,
        timepoints: np.ndarray,
        m_protein: np.ndarray,
        ldh_baseline: float,
        ldh_current: float,
        ecog: int,
        iss: int,
        calcium: float,
    ) -> list[EarlyWarningSignal]:
        """
        Detect early warning signals.

        Args:
            timepoints: Time points in weeks
            m_protein: M-protein concentrations
            ldh_baseline: Baseline LDH
            ldh_current: Current LDH
            ecog: ECOG performance status (0-3)
            iss: ISS stage (1-3)
            calcium: Serum calcium

        Returns:
            List of detected warning signals
        """
        warnings_list = []

        # M-protein reduction at week 4
        idx_week4 = np.argmin(np.abs(timepoints - 4))
        if idx_week4 > 0:
            m_reduction = 1 - (m_protein[idx_week4] / m_protein[0])
            is_triggered = m_reduction < self.thresholds["m_protein_reduction_week4"]
            warnings_list.append(
                EarlyWarningSignal(
                    signal_name="M-protein_reduction_week4",
                    signal_value=m_reduction,
                    threshold=self.thresholds["m_protein_reduction_week4"],
                    is_triggered=is_triggered,
                    clinical_urgency="high" if is_triggered else "low",
                )
            )

        # Plateau detection (weeks 8-12)
        idx_plateau = (timepoints >= 8) & (timepoints <= 12)
        if np.sum(idx_plateau) >= 2:
            plateau_values = m_protein[idx_plateau]
            plateau_change = np.std(plateau_values) / (np.mean(plateau_values) + 1e-6)
            is_triggered = plateau_change < 0.05  # Small variation = plateau
            warnings_list.append(
                EarlyWarningSignal(
                    signal_name="M_protein_plateau",
                    signal_value=plateau_change,
                    threshold=0.05,
                    is_triggered=is_triggered,
                    clinical_urgency="moderate" if is_triggered else "low",
                )
            )

        # Rising LDH despite M-protein decline
        ldh_ratio = ldh_current / ldh_baseline
        is_triggered = ldh_ratio > self.thresholds["ldh_rise_threshold"]
        warnings_list.append(
            EarlyWarningSignal(
                signal_name="rising_LDH",
                signal_value=ldh_ratio,
                threshold=self.thresholds["ldh_rise_threshold"],
                is_triggered=is_triggered,
                clinical_urgency="high" if is_triggered else "low",
            )
        )

        # mHR-IMS criteria integration
        mhr_ims_score = self._compute_mhr_ims_score(ecog, iss, calcium)
        is_triggered = mhr_ims_score >= 2  # 2+ risk factors
        warnings_list.append(
            EarlyWarningSignal(
                signal_name="mHR_IMS_criteria",
                signal_value=float(mhr_ims_score),
                threshold=2.0,
                is_triggered=is_triggered,
                clinical_urgency="high" if mhr_ims_score == 3 else ("moderate" if is_triggered else "low"),
            )
        )

        return warnings_list

    @staticmethod
    def _compute_mhr_ims_score(ecog: int, iss: int, calcium: float) -> int:
        """
        Compute mHR-IMS risk score.

        Returns number of risk factors (0-3).
        """
        score = 0
        if ecog == 2:
            score += 1
        if iss == 3:
            score += 1
        if calcium >= 11:
            score += 1
        return score


class KineticTrajectoryClassifier:
    """
    Classifies early relapse risk based on kinetic trajectory analysis.

    Uses velocity and acceleration of M-protein and FLC curves.
    """

    def __init__(self):
        """Initialize trajectory thresholds."""
        self.velocity_threshold = 0.02  # mg/dL per week (unfavorable)
        self.acceleration_threshold = 0.005  # Curvature threshold

    def classify_trajectory(
        self,
        timepoints: np.ndarray,
        m_protein: np.ndarray,
        flc_involved: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Classify kinetic trajectory and estimate ER18 risk.

        Args:
            timepoints: Time points in weeks
            m_protein: M-protein concentrations
            flc_involved: Involved FLC concentrations

        Returns:
            (trajectory_category, er18_risk_estimate)
        """
        # Calculate velocity (first derivative)
        if len(timepoints) >= 2:
            velocity_m = np.gradient(m_protein, timepoints)
            mean_velocity = np.mean(velocity_m[-4:]) if len(velocity_m) >= 4 else np.mean(velocity_m)
        else:
            mean_velocity = 0.0

        # Calculate acceleration (second derivative)
        if len(timepoints) >= 3:
            acceleration = np.gradient(velocity_m, timepoints)
            mean_accel = np.mean(acceleration[-3:]) if len(acceleration) >= 3 else np.mean(acceleration)
        else:
            mean_accel = 0.0

        # Classify trajectory
        if mean_velocity > self.velocity_threshold or mean_accel > self.acceleration_threshold:
            trajectory = "unfavorable"
            er18_risk = 0.60
        elif mean_velocity < -self.velocity_threshold / 2:
            trajectory = "favorable"
            er18_risk = 0.15
        else:
            trajectory = "stable"
            er18_risk = 0.35

        logger.debug(
            f"Kinetic trajectory: {trajectory}, velocity={mean_velocity:.4f}, "
            f"acceleration={mean_accel:.4f}, ER18_risk={er18_risk:.2f}"
        )

        return trajectory, er18_risk


class CUSUMChangeDetector(nn.Module):
    """
    Cumulative Sum Control Chart for detecting sustained changes in M-protein.

    Detects departure from expected trajectory (upward drift).
    """

    def __init__(self, threshold: float = 5.0, slack_param: float = 0.5):
        """
        Initialize CUSUM detector.

        Args:
            threshold: CUSUM threshold for signal
            slack_param: Slack parameter for two-sided CUSUM
        """
        super().__init__()
        self.threshold = threshold
        self.slack_param = slack_param
        self.register_buffer("cusum_pos", torch.tensor(0.0))
        self.register_buffer("cusum_neg", torch.tensor(0.0))

    def forward(
        self,
        log_m_protein: torch.Tensor,
        expected_log_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CUSUM statistics.

        Args:
            log_m_protein: Log-transformed M-protein series (shape: (n,))
            expected_log_m: Expected log M-protein from kinetics model (shape: (n,))

        Returns:
            (cusum_pos, cusum_neg) tensors tracking positive/negative deviations
        """
        deviations = log_m_protein - expected_log_m

        cusum_pos_list = []
        cusum_neg_list = []
        c_pos = 0.0
        c_neg = 0.0

        for dev in deviations:
            c_pos = max(0, c_pos + dev.item() - self.slack_param)
            c_neg = max(0, c_neg - dev.item() - self.slack_param)
            cusum_pos_list.append(c_pos)
            cusum_neg_list.append(c_neg)

        return torch.tensor(cusum_pos_list), torch.tensor(cusum_neg_list)

    def detect_change(self, cusum_values: torch.Tensor) -> bool:
        """
        Detect if CUSUM exceeds threshold (signal of change).

        Args:
            cusum_values: CUSUM values

        Returns:
            True if change detected
        """
        return bool((cusum_values > self.threshold).any())


class EWMAMonitor(nn.Module):
    """
    Exponentially Weighted Moving Average monitor.

    Tracks short-term trends in M-protein while smoothing noise.
    """

    def __init__(self, alpha: float = 0.3):
        """
        Initialize EWMA monitor.

        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
        """
        super().__init__()
        self.alpha = alpha
        self.register_buffer("ewma_value", torch.tensor(float('nan')))

    def forward(self, new_observation: torch.Tensor) -> torch.Tensor:
        """
        Update EWMA with new observation.

        Args:
            new_observation: New M-protein measurement

        Returns:
            Updated EWMA value
        """
        if torch.isnan(self.ewma_value):
            self.ewma_value = new_observation.clone()
        else:
            self.ewma_value = (self.alpha * new_observation +
                              (1 - self.alpha) * self.ewma_value)

        return self.ewma_value.clone()

    def reset(self):
        """Reset EWMA state."""
        self.ewma_value = torch.tensor(float('nan'))


class CytokineActivityFingerprint:
    """
    Models microenvironment-driven resistance via cytokine profiles.

    Integrates IL-6, TNF-α, VEGF, HGF for resistance phenotype prediction.
    """

    def __init__(self):
        """Initialize cytokine thresholds (clinical reference values)."""
        self.cytokine_thresholds = {
            "IL6": 5.0,  # pg/mL (elevated in MM)
            "TNFa": 15.0,  # pg/mL
            "VEGF": 300.0,  # pg/mL
            "HGF": 500.0,  # pg/mL
        }

    def compute_resistance_phenotype(
        self,
        il6: float,
        tnfa: float,
        vegf: float,
        hgf: float,
    ) -> Tuple[float, str]:
        """
        Compute microenvironment-driven resistance phenotype.

        Args:
            il6: IL-6 level (pg/mL)
            tnfa: TNF-α level (pg/mL)
            vegf: VEGF level (pg/mL)
            hgf: HGF level (pg/mL)

        Returns:
            (resistance_score, phenotype_label)
        """
        elevated_count = 0
        elevated_count += 1 if il6 > self.cytokine_thresholds["IL6"] else 0
        elevated_count += 1 if tnfa > self.cytokine_thresholds["TNFa"] else 0
        elevated_count += 1 if vegf > self.cytokine_thresholds["VEGF"] else 0
        elevated_count += 1 if hgf > self.cytokine_thresholds["HGF"] else 0

        resistance_score = elevated_count / 4.0

        if resistance_score >= 0.75:
            phenotype = "high_microenvironment_resistance"
        elif resistance_score >= 0.5:
            phenotype = "moderate_microenvironment_resistance"
        else:
            phenotype = "low_microenvironment_resistance"

        return resistance_score, phenotype


class EarlyRelapsePredictor(nn.Module):
    """
    Comprehensive early relapse prediction system.

    Integrates kinetics, warning signals, trajectory classification,
    and cytokine profiles into unified ER18 risk assessment.
    """

    def __init__(self, regimen_type: str = "standard"):
        """
        Initialize early relapse predictor.

        Args:
            regimen_type: Treatment regimen type
        """
        super().__init__()
        self.kinetics_model = MProteinKineticsModel(regimen_type=regimen_type)
        self.warning_detector = EarlyWarningSignalDetector()
        self.trajectory_classifier = KineticTrajectoryClassifier()
        self.cusum_detector = CUSUMChangeDetector(threshold=5.0)
        self.ewma_monitor = EWMAMonitor(alpha=0.3)
        self.cytokine_fingerprint = CytokineActivityFingerprint()

    def predict_early_relapse(
        self,
        timepoints: np.ndarray,
        m_protein: np.ndarray,
        flc_involved: np.ndarray,
        ldh_baseline: float,
        ldh_current: float,
        ecog: int,
        iss: int,
        calcium: float,
        il6: Optional[float] = None,
        tnfa: Optional[float] = None,
        vegf: Optional[float] = None,
        hgf: Optional[float] = None,
    ) -> EarlyRelapseRiskAssessment:
        """
        Predict early relapse risk within 18 months.

        Args:
            timepoints: Time points in weeks
            m_protein: M-protein concentrations
            flc_involved: Involved FLC concentrations
            ldh_baseline: Baseline LDH
            ldh_current: Current LDH
            ecog: ECOG performance status
            iss: ISS stage
            calcium: Serum calcium
            il6: IL-6 level (optional)
            tnfa: TNF-α level (optional)
            vegf: VEGF level (optional)
            hgf: HGF level (optional)

        Returns:
            EarlyRelapseRiskAssessment with comprehensive risk evaluation
        """
        contributing_factors = []

        # 1. Kinetics fitting
        kinetics_params = self.kinetics_model.fit_kinetics(
            timepoints, m_protein, try_biexponential=True
        )

        # 2. Early warning signals
        warnings = self.warning_detector.detect_warnings(
            timepoints, m_protein, ldh_baseline, ldh_current, ecog, iss, calcium
        )
        triggered_warnings = [w for w in warnings if w.is_triggered]
        for w in triggered_warnings:
            contributing_factors.append(f"{w.signal_name} (urgency: {w.clinical_urgency})")

        # 3. Trajectory classification
        trajectory, trajectory_er18_risk = self.trajectory_classifier.classify_trajectory(
            timepoints, m_protein, flc_involved
        )
        if trajectory == "unfavorable":
            contributing_factors.append("Unfavorable kinetic trajectory")

        # 4. CUSUM change detection
        log_m = np.log(m_protein + 1e-6)
        log_m_tensor = torch.from_numpy(log_m).float()
        m_pred = self.kinetics_model.forward(kinetics_params, torch.from_numpy(timepoints).float())
        log_m_pred = torch.log(m_pred + 1e-6)
        cusum_pos, cusum_neg = self.cusum_detector(log_m_tensor, log_m_pred)
        if self.cusum_detector.detect_change(cusum_pos):
            contributing_factors.append("CUSUM positive change detected")

        # 5. Cytokine fingerprint (if available)
        cytokine_risk = 0.0
        if all(v is not None for v in [il6, tnfa, vegf, hgf]):
            cytokine_risk, phenotype = self.cytokine_fingerprint.compute_resistance_phenotype(
                il6, tnfa, vegf, hgf
            )
            if cytokine_risk > 0.5:
                contributing_factors.append(f"Microenvironment resistance: {phenotype}")

        # 6. Aggregate ER18 risk
        base_risk = 0.25  # Population baseline
        risk_from_trajectory = trajectory_er18_risk - 0.25
        risk_from_warnings = 0.15 * len(triggered_warnings) / len(warnings)
        risk_from_cytokines = 0.10 * cytokine_risk

        er18_risk = np.clip(
            base_risk + risk_from_trajectory + risk_from_warnings + risk_from_cytokines,
            0.0, 0.95
        )

        # Categorize risk
        if er18_risk < 0.30:
            risk_category = "low"
        elif er18_risk < 0.55:
            risk_category = "intermediate"
        else:
            risk_category = "high"

        # Recommend action
        if risk_category == "high":
            action = "Escalate to intensive regimen; monitor q2 weeks; consider salvage therapy trial"
        elif risk_category == "intermediate":
            action = "Monitor every 3-4 weeks; reassess at week 12; prepare intensive backup plan"
        else:
            action = "Continue current regimen; routine monitoring q4-6 weeks"

        return EarlyRelapseRiskAssessment(
            er18_risk_probability=er18_risk,
            risk_category=risk_category,
            contributing_factors=contributing_factors,
            kinetic_trajectory=trajectory,
            recommended_action=action,
        )


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for early relapse detection module."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
