from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.distributions import Normal

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyOutput:
    """Container for multi-layer UQ results."""

    point_estimate: torch.Tensor
    aleatoric_variance: torch.Tensor
    epistemic_variance: torch.Tensor
    prediction_interval_lower: torch.Tensor
    prediction_interval_upper: torch.Tensor
    risk_category: list[str]
    confidence_band: list[Tuple[float, float]]
    abstain_flag: torch.Tensor
    raw_ensemble: Optional[torch.Tensor] = None
    evidential_params: Optional[dict[str, torch.Tensor]] = None


class DeepEnsembleWrapper(nn.Module):
    """Layer 1: Deep Ensemble with K independent models.

    Trains K models with different random seeds. Ensemble predictions provide
    aleatoric (within-seed variance) and epistemic (cross-seed variance) uncertainty.

    Args:
        model_fn: Callable that returns a fresh model instance.
        n_models: Number of independent models (5-10 recommended).
        device: torch device for computation.
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        n_models: int = 5,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__()
        self.n_models = n_models
        self.device = device
        self.models = nn.ModuleList([model_fn() for _ in range(n_models)])
        for model in self.models:
            model.to(device)
        logger.info(f"DeepEnsembleWrapper initialized with {n_models} models")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, *features)

        Returns:
            Tuple of (ensemble_mean, ensemble_variance)
        """
        predictions = []
        for model in self.models:
            with torch.no_grad():
                predictions.append(model(x))

        predictions = torch.stack(predictions, dim=0)  # (n_models, batch, ...)
        ensemble_mean = predictions.mean(dim=0)
        ensemble_variance = predictions.var(dim=0)

        return ensemble_mean, ensemble_variance


class EvidentialRegressionHead(nn.Module):
    """Layer 2: Evidential Deep Learning with Normal Inverse-Gamma (NIG) distribution.

    Outputs parameters (γ, ν, α, β) for NIG distribution:
    - Aleatoric variance = β/(α-1)
    - Epistemic variance = β/(ν(α-1))

    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension (number of predictive variables).
        lambda_coef: KL regularization coefficient.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        lambda_coef: float = 1e-2,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.lambda_coef = lambda_coef

        # NIG parameterization network
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4 * output_dim),
        )

        logger.info(f"EvidentialRegressionHead initialized (output_dim={output_dim})")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Tuple of (predicted_mean, params_dict) where params_dict has keys
            'gamma', 'nu', 'alpha', 'beta' (all shape batch, output_dim)
        """
        params = self.fc(x)  # (batch, 4 * output_dim)

        # Reshape to separate NIG parameters
        batch_size = x.shape[0]
        params = params.view(batch_size, self.output_dim, 4)

        gamma = params[..., 0]  # location
        nu = F.softplus(params[..., 1]) + 1e-6  # degrees of freedom
        alpha = F.softplus(params[..., 2]) + 1.0  # shape
        beta = F.softplus(params[..., 3]) + 1e-6  # rate

        # Predicted mean
        mu = gamma

        return mu, {
            'gamma': gamma,
            'nu': nu,
            'alpha': alpha,
            'beta': beta,
        }

    def nig_negative_log_likelihood(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute NIG NLL loss.

        Args:
            y: Target values (batch, output_dim)
            mu: Predicted mean (batch, output_dim)
            params: Dict with 'gamma', 'nu', 'alpha', 'beta'

        Returns:
            Scalar loss
        """
        gamma = params['gamma']
        nu = params['nu']
        alpha = params['alpha']
        beta = params['beta']

        # Two-parameter gamma distribution mode
        alpha = torch.clamp(alpha, min=1.0 + 1e-6)

        # NIG NLL: -log p(y|γ,ν,α,β)
        diff = y - gamma
        term1 = 0.5 * torch.log(beta * (nu + 1) / (nu * np.pi * alpha))
        term2 = -(2 * alpha + 1) / 2 * torch.log(1 + (nu * diff ** 2) / (2 * beta * (alpha + 1)))
        nll = -(term1 + term2).mean()

        return nll

    def kl_regularization(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """KL divergence regularization for Bayesian inference.

        Args:
            params: Dict with NIG parameters

        Returns:
            Scalar KL loss
        """
        alpha = params['alpha']
        beta = params['beta']
        nu = params['nu']

        alpha = torch.clamp(alpha, min=1.0 + 1e-6)

        # KL(NIG posterior || NIG prior) with weak prior
        prior_alpha = 1.0
        prior_beta = 1.0
        prior_nu = 1.0

        kl = (
            (alpha - prior_alpha) * (torch.digamma(alpha) - np.log(beta))
            - torch.lgamma(alpha) + torch.lgamma(torch.tensor(prior_alpha))
            + prior_alpha * torch.log(beta / prior_beta)
            - (nu + 2) / 2 / (alpha - 1)
        )

        return self.lambda_coef * kl.mean()

    def compute_uncertainties(
        self,
        params: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute aleatoric and epistemic variances from NIG parameters.

        Args:
            params: Dict with NIG parameters

        Returns:
            Tuple of (aleatoric_var, epistemic_var)
        """
        alpha = torch.clamp(params['alpha'], min=1.0 + 1e-6)
        beta = params['beta']
        nu = params['nu']

        aleatoric_var = beta / (alpha - 1)
        epistemic_var = beta / (nu * (alpha - 1))

        return aleatoric_var, epistemic_var


class BayesianODEWrapper(nn.Module):
    """Layer 3: Bayesian ODE with MC sampling of parameters.

    Wrapper for uncertainty quantification via Monte Carlo sampling of ODE
    system parameters. Generates trajectory credible bands.

    Args:
        ode_solver: Callable(params, t) -> trajectory
        param_mean: Mean of parameter prior (shape: n_params)
        param_std: Std of parameter prior (shape: n_params)
        n_samples: Number of MC samples for credible bands.
    """

    def __init__(
        self,
        ode_solver: Callable,
        param_mean: torch.Tensor,
        param_std: torch.Tensor,
        n_samples: int = 50,
    ):
        super().__init__()
        self.ode_solver = ode_solver
        self.register_buffer('param_mean', param_mean)
        self.register_buffer('param_std', param_std)
        self.n_samples = n_samples
        logger.info(f"BayesianODEWrapper initialized with {n_samples} MC samples")

    def forward(
        self,
        t: torch.Tensor,
        observed_trajectory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample trajectories and compute credible bands.

        Args:
            t: Time points for ODE solution (shape: n_timepoints)
            observed_trajectory: Optional observations for likelihood weighting

        Returns:
            Tuple of (mean_trajectory, lower_band, upper_band)
                Each shape (n_timepoints,)
        """
        trajectories = []

        for _ in range(self.n_samples):
            # Sample parameters from prior
            params = torch.randn_like(self.param_mean) * self.param_std + self.param_mean

            # Solve ODE with sampled parameters
            traj = self.ode_solver(params, t)
            trajectories.append(traj)

        trajectories = torch.stack(trajectories)  # (n_samples, n_timepoints)

        mean_traj = trajectories.mean(dim=0)
        lower_band = trajectories.quantile(0.025, dim=0).values
        upper_band = trajectories.quantile(0.975, dim=0).values

        return mean_traj, lower_band, upper_band


class ConformalCalibrator(nn.Module):
    """Layer 4: Conformalized Quantile Regression (CQR).

    Trains quantile regression at α/2 and 1-α/2 with conformal calibration
    on holdout set to achieve distribution-free coverage guarantee.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension.
        alpha: Miscoverage level (1-alpha = target coverage).
        quantile_loss_weight: Weight for quantile loss.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        alpha: float = 0.1,
        quantile_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.quantile_loss_weight = quantile_loss_weight

        # Quantile regression networks
        self.lower_quantile_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )
        self.upper_quantile_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

        # Conformal score (initialized in calibrate)
        self.register_buffer('q_hat', torch.tensor(0.0))
        self.calibrated = False

        logger.info(f"ConformalCalibrator initialized (alpha={alpha})")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict conformal prediction intervals.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        lower = self.lower_quantile_net(x)
        upper = self.upper_quantile_net(x)
        return lower, upper

    def quantile_loss(
        self,
        y: torch.Tensor,
        y_lower: torch.Tensor,
        y_upper: torch.Tensor,
    ) -> torch.Tensor:
        """Quantile loss for quantile regression.

        Args:
            y: Target values (batch, output_dim)
            y_lower: Lower quantile predictions
            y_upper: Upper quantile predictions

        Returns:
            Scalar loss
        """
        q_low = self.alpha / 2
        q_high = 1 - self.alpha / 2

        # Quantile loss: ρ_q(u) = u * (q - 1{u < 0})
        loss_lower = torch.where(
            y < y_lower,
            q_low * (y - y_lower),
            (1 - q_low) * (y_lower - y),
        ).mean()

        loss_upper = torch.where(
            y > y_upper,
            (1 - q_high) * (y - y_upper),
            q_high * (y_upper - y),
        ).mean()

        return self.quantile_loss_weight * (loss_lower + loss_upper)

    def calibrate(
        self,
        x_calib: torch.Tensor,
        y_calib: torch.Tensor,
    ) -> None:
        """Calibrate conformal score on holdout set.

        Args:
            x_calib: Calibration input (batch, input_dim)
            y_calib: Calibration targets (batch, output_dim)
        """
        with torch.no_grad():
            y_lower, y_upper = self.forward(x_calib)

            # Conformity scores: non-conformity measure
            nonconformity = torch.max(
                y_lower - y_calib,
                y_calib - y_upper,
            )

            # Quantile of nonconformity scores: for (1-alpha) coverage, use ceil((n+1)*(1-alpha))/n
            # This gives the correct tail quantile without double normalization
            n = len(y_calib)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(q_level, 1.0)  # Ensure quantile is in [0, 1]
            self.q_hat = torch.quantile(nonconformity, q_level, method='higher')
            self.calibrated = True

        logger.info(f"ConformalCalibrator calibrated with q_hat={self.q_hat.item():.4f}")

    def predict_with_conformal(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with conformal adjustment.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Tuple of (lower_adjusted, upper_adjusted)
        """
        if not self.calibrated:
            warnings.warn("Calibrator not calibrated; returning uncalibrated intervals")

        y_lower, y_upper = self.forward(x)

        if self.calibrated:
            y_lower = y_lower - self.q_hat
            y_upper = y_upper + self.q_hat

        return y_lower, y_upper


class SelectivePredictionGate(nn.Module):
    """Layer 5: Selective Prediction with abstention.

    Abstains on high-uncertainty predictions based on threshold τ.
    Provides coverage-accuracy trade-off.

    Args:
        threshold: Uncertainty threshold for abstention.
        use_epistemic: If True, use epistemic uncertainty; else use aleatoric.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        use_epistemic: bool = True,
    ):
        super().__init__()
        self.threshold = threshold
        self.use_epistemic = use_epistemic

    def forward(
        self,
        prediction: torch.Tensor,
        aleatoric_var: torch.Tensor,
        epistemic_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decide whether to predict or abstain.

        Args:
            prediction: Point estimate (batch,)
            aleatoric_var: Aleatoric variance (batch,)
            epistemic_var: Epistemic variance (batch,)

        Returns:
            Tuple of (prediction, abstain_flag)
                abstain_flag: Boolean tensor, True = abstain
        """
        uncertainty = epistemic_var if self.use_epistemic else aleatoric_var
        uncertainty_std = torch.sqrt(uncertainty)

        abstain_flag = uncertainty_std > self.threshold

        return prediction, abstain_flag

    def get_coverage_accuracy_curve(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[list[float], list[float]]:
        """Compute coverage-accuracy trade-off curve.

        Args:
            predictions: Shape (n,)
            uncertainties: Shape (n,)
            targets: Shape (n,)

        Returns:
            Tuple of (coverage_rates, accuracies)
        """
        thresholds = torch.linspace(0, uncertainties.max(), 10)
        coverages = []
        accuracies = []

        for tau in thresholds:
            abstain = uncertainties > tau
            coverage = (~abstain).float().mean().item()

            if coverage > 0:
                acc = (predictions[~abstain] == targets[~abstain]).float().mean().item()
            else:
                acc = 0.0

            coverages.append(coverage)
            accuracies.append(acc)

        return coverages, accuracies


class ClinicalUncertaintyCommunicator(nn.Module):
    """Layer 6: Convert prediction intervals to clinical risk categories.

    Maps predictions and uncertainties to actionable risk levels with
    confidence bands for clinical decision-making.

    Args:
        bounds: Dict mapping risk categories to threshold tuples
            e.g., {'Low': (0, 0.3), 'Moderate': (0.3, 0.6), ...}
        confidence_levels: Confidence levels for confidence bands
    """

    def __init__(
        self,
        bounds: Optional[dict[str, Tuple[float, float]]] = None,
        confidence_levels: list[float] = [0.68, 0.95],
    ):
        super().__init__()
        if bounds is None:
            bounds = {
                'Low': (0.0, 0.3),
                'Moderate': (0.3, 0.6),
                'High': (0.6, 0.85),
                'Very High': (0.85, 1.0),
            }
        self.bounds = bounds
        self.confidence_levels = confidence_levels
        logger.info(f"ClinicalUncertaintyCommunicator initialized with {len(bounds)} risk categories")

    def forward(
        self,
        prediction: torch.Tensor,
        prediction_std: torch.Tensor,
    ) -> Tuple[list[str], list[Tuple[float, float]]]:
        """Assign risk category and confidence bands.

        Args:
            prediction: Point estimate (batch,)
            prediction_std: Standard deviation (batch,)

        Returns:
            Tuple of (risk_categories, confidence_bands)
                risk_categories: List of category names
                confidence_bands: List of (lower, upper) confidence interval tuples
        """
        prediction_np = prediction.detach().cpu().numpy()
        prediction_std_np = prediction_std.detach().cpu().numpy()

        risk_categories = []
        confidence_bands = []

        for pred, std in zip(prediction_np.flatten(), prediction_std_np.flatten()):
            # Find risk category
            category = 'Unknown'
            for cat_name, (lower, upper) in self.bounds.items():
                if lower <= pred <= upper:
                    category = cat_name
                    break
            risk_categories.append(category)

            # Compute confidence band (95%)
            z_crit = 1.96
            ci_lower = max(0.0, pred - z_crit * std)
            ci_upper = min(1.0, pred + z_crit * std)
            confidence_bands.append((ci_lower, ci_upper))

        return risk_categories, confidence_bands


class UncertaintyQuantifier(nn.Module):
    """Top-level orchestrator for 6-layer UQ architecture.

    Integrates:
    1. Deep Ensemble
    2. Evidential Regression
    3. Bayesian ODE (optional)
    4. Conformal Quantile Regression
    5. Selective Prediction
    6. Clinical Communication

    Args:
        base_model_fn: Callable that returns base model
        input_dim: Input feature dimension
        output_dim: Output dimension
        n_ensemble: Number of ensemble models
        alpha: Conformal coverage miscoverage level
        selective_threshold: Abstention threshold
        ode_solver: Optional ODE solver function
    """

    def __init__(
        self,
        base_model_fn: Callable[[], nn.Module],
        input_dim: int,
        output_dim: int = 1,
        n_ensemble: int = 5,
        alpha: float = 0.1,
        selective_threshold: float = 0.5,
        ode_solver: Optional[Callable] = None,
    ):
        super().__init__()

        # Layer 1: Deep Ensemble
        self.ensemble = DeepEnsembleWrapper(base_model_fn, n_models=n_ensemble)

        # Layer 2: Evidential Regression
        self.evidential_head = EvidentialRegressionHead(input_dim, output_dim)

        # Layer 3: Bayesian ODE (optional)
        self.bayesian_ode = None
        if ode_solver is not None:
            param_mean = torch.zeros(10)  # Placeholder dimension
            param_std = torch.ones(10)
            self.bayesian_ode = BayesianODEWrapper(
                ode_solver, param_mean, param_std, n_samples=50
            )

        # Layer 4: Conformal Quantile Regression
        self.conformal_calibrator = ConformalCalibrator(input_dim, output_dim, alpha)

        # Layer 5: Selective Prediction
        self.selective_gate = SelectivePredictionGate(
            threshold=selective_threshold,
            use_epistemic=True,
        )

        # Layer 6: Clinical Communication
        self.clinical_comm = ClinicalUncertaintyCommunicator()

        logger.info("UncertaintyQuantifier (6-layer) initialized successfully")

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> UncertaintyOutput:
        """Execute full 6-layer UQ pipeline.

        Args:
            x: Input tensor (batch, input_dim)
            y: Optional target for loss computation

        Returns:
            UncertaintyOutput with all UQ components
        """
        device = x.device

        # Layer 1: Deep Ensemble
        ensemble_mean, ensemble_var = self.ensemble(x)

        # Layer 2: Evidential Regression
        evid_mean, evid_params = self.evidential_head(x)
        aleatoric_var, epistemic_var = self.evidential_head.compute_uncertainties(evid_params)

        # Layer 4: Conformal Quantile Regression
        cqr_lower, cqr_upper = self.conformal_calibrator(x)

        # Use ensemble + evidential for final point estimate
        point_estimate = ensemble_mean
        total_variance = ensemble_var + epistemic_var
        std_dev = torch.sqrt(total_variance)

        # Layer 5: Selective Prediction
        _, abstain_flag = self.selective_gate(point_estimate, aleatoric_var, epistemic_var)

        # Layer 6: Clinical Communication
        risk_categories, confidence_bands = self.clinical_comm(point_estimate, std_dev)

        # Build output
        output = UncertaintyOutput(
            point_estimate=point_estimate,
            aleatoric_variance=aleatoric_var.mean(dim=-1) if aleatoric_var.dim() > 1 else aleatoric_var,
            epistemic_variance=epistemic_var.mean(dim=-1) if epistemic_var.dim() > 1 else epistemic_var,
            prediction_interval_lower=cqr_lower.squeeze(-1),
            prediction_interval_upper=cqr_upper.squeeze(-1),
            risk_category=risk_categories,
            confidence_band=confidence_bands,
            abstain_flag=abstain_flag,
            raw_ensemble=ensemble_mean,
            evidential_params=evid_params,
        )

        return output

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lambda_ensemble: float = 0.3,
        lambda_evidential: float = 0.5,
        lambda_cqr: float = 0.2,
    ) -> torch.Tensor:
        """Compute combined loss across all layers.

        Args:
            x: Input tensor (batch, input_dim)
            y: Target tensor (batch, output_dim or batch)
            lambda_*: Loss weighting coefficients

        Returns:
            Scalar loss
        """
        # Expand y if needed
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # Ensemble loss (MSE)
        ensemble_mean, _ = self.ensemble(x)
        loss_ensemble = F.mse_loss(ensemble_mean, y)

        # Evidential loss
        evid_mean, evid_params = self.evidential_head(x)
        loss_nll = self.evidential_head.nig_negative_log_likelihood(y, evid_mean, evid_params)
        loss_kl = self.evidential_head.kl_regularization(evid_params)
        loss_evidential = loss_nll + loss_kl

        # CQR loss
        cqr_lower, cqr_upper = self.conformal_calibrator(x)
        loss_cqr = self.conformal_calibrator.quantile_loss(y, cqr_lower, cqr_upper)

        # Combined loss
        total_loss = (
            lambda_ensemble * loss_ensemble
            + lambda_evidential * loss_evidential
            + lambda_cqr * loss_cqr
        )

        return total_loss

    def calibrate_conformal(
        self,
        x_calib: torch.Tensor,
        y_calib: torch.Tensor,
    ) -> None:
        """Calibrate conformal intervals on holdout set.

        Args:
            x_calib: Calibration input (batch, input_dim)
            y_calib: Calibration targets (batch,) or (batch, 1)
        """
        if y_calib.dim() == 1:
            y_calib = y_calib.unsqueeze(-1)

        self.conformal_calibrator.calibrate(x_calib, y_calib)
        logger.info("Conformal calibration complete")