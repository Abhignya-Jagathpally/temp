from __future__ import annotations

import logging
from typing import Optional, Literal, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from scipy.special import gammaln, digamma, polygamma
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class HarmonizationConfig:
    """Configuration for multi-site harmonization."""
    method: Literal['combat_seq', 'harmony', 'fedbatch', 'gsva'] = 'harmony'
    combat_shrinkage_strength: float = 0.95
    harmony_num_clusters: int = 30
    harmony_convergence_tol: float = 1e-3
    harmony_max_iter: int = 50
    fedbatch_momentum: float = 0.9
    lora_rank: int = 8
    monitor_auc_threshold: float = 0.05
    warning_threshold: float = 0.10


@dataclass
class CorrectionsApplied:
    """Track corrections applied to data."""
    method: str
    sites_corrected: list[str] = field(default_factory=list)
    auc_variance_before: float = 0.0
    auc_variance_after: float = 0.0
    iterations: int = 0


class ComBatSeqCorrector:
    """ComBat-seq: empirical Bayes location-scale batch correction for count data.

    Corrects for technical batch effects in count-based sequencing data while
    preserving biological signal through empirical Bayes shrinkage of location
    and scale parameters.
    """

    def __init__(self, shrinkage_strength: float = 0.95):
        """Initialize ComBat-seq corrector.

        Args:
            shrinkage_strength: Strength of empirical Bayes shrinkage (0-1).
                Higher values increase shrinkage toward global mean.
        """
        self.shrinkage_strength = shrinkage_strength
        self.location_mean = None
        self.location_var = None
        self.scale_mean = None
        self.scale_var = None

    def fit(self, data: np.ndarray, batch_labels: np.ndarray) -> ComBatSeqCorrector:
        """Fit empirical Bayes parameters per batch.

        Args:
            data: Count matrix (n_samples, n_features). Non-negative integers.
            batch_labels: Batch assignment per sample (n_samples,).

        Returns:
            Self for chaining.
        """
        n_samples, n_features = data.shape
        unique_batches = np.unique(batch_labels)
        n_batches = len(unique_batches)

        # Estimate location (mean) and scale (variance) per batch and feature
        self.location_mean = np.zeros((n_batches, n_features))
        self.location_var = np.zeros((n_batches, n_features))
        self.scale_mean = np.zeros((n_batches, n_features))
        self.scale_var = np.zeros((n_batches, n_features))

        for batch_idx, batch in enumerate(unique_batches):
            mask = batch_labels == batch
            batch_data = data[mask, :]

            # Location: empirical mean
            self.location_mean[batch_idx] = np.mean(batch_data, axis=0)
            batch_data_centered = batch_data - self.location_mean[batch_idx]

            # Scale: empirical variance (via method-of-moments for count data)
            raw_var = np.var(batch_data, axis=0)
            self.location_var[batch_idx] = raw_var

            # Dispersion parameter (log-scale)
            dispersions = np.log(np.maximum(raw_var / (self.location_mean[batch_idx] + 1), 1e-8))
            self.scale_mean[batch_idx] = dispersions
            self.scale_var[batch_idx] = np.var(dispersions)

        # Apply empirical Bayes shrinkage
        global_location_mean = np.mean(self.location_mean, axis=0)
        global_scale_mean = np.mean(self.scale_mean, axis=0)

        self.location_mean = (self.shrinkage_strength * global_location_mean +
                              (1 - self.shrinkage_strength) * self.location_mean)
        self.scale_mean = (self.shrinkage_strength * global_scale_mean +
                          (1 - self.shrinkage_strength) * self.scale_mean)

        logger.info(f"ComBat-seq fitted on {n_batches} batches, {n_features} features")
        return self

    def transform(self, data: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
        """Apply batch correction.

        Args:
            data: Count matrix to correct (n_samples, n_features).
            batch_labels: Batch assignment per sample (n_samples,).

        Returns:
            Corrected count matrix (n_samples, n_features).
        """
        if self.location_mean is None:
            raise ValueError("Corrector not fitted. Call fit() first.")

        data_corrected = data.copy().astype(np.float64)
        unique_batches = np.unique(batch_labels)
        batch_to_idx = {b: i for i, b in enumerate(unique_batches)}

        for batch in unique_batches:
            mask = batch_labels == batch
            batch_idx = batch_to_idx[batch]

            # Remove location effect: subtract batch-specific mean, add global mean
            data_corrected[mask] = (data_corrected[mask] - self.location_mean[batch_idx])

            # Remove scale effect via log-space adjustment
            scale_adj = np.exp(self.scale_mean[batch_idx] - np.mean(self.scale_mean, axis=0))
            data_corrected[mask] = data_corrected[mask] / np.maximum(scale_adj, 1e-8)

        return np.maximum(np.round(data_corrected), 0).astype(data.dtype)


class HarmonyIntegrator:
    """Harmony integration via iterative soft-clustering and ridge regression.

    Achieves multi-site harmony by alternating between:
    1. Soft-clustering with maximum diversity objective
    2. Ridge regression-based correction from cluster centroids
    """

    def __init__(self, num_clusters: int = 30, convergence_tol: float = 1e-3,
                 max_iter: int = 50):
        """Initialize Harmony integrator.

        Args:
            num_clusters: Number of soft clusters (K).
            convergence_tol: Convergence criterion for clustering objective.
            max_iter: Maximum iterations.
        """
        self.num_clusters = num_clusters
        self.convergence_tol = convergence_tol
        self.max_iter = max_iter
        self.cluster_centroids = None
        self.cluster_assignments = None
        self.correction_weights = None

    def fit(self, embeddings: np.ndarray, batch_labels: np.ndarray) -> HarmonyIntegrator:
        """Fit Harmony clustering and correction.

        Args:
            embeddings: Sample embeddings (n_samples, n_dims). Pre-normalized recommended.
            batch_labels: Batch assignment per sample (n_samples,).

        Returns:
            Self for chaining.
        """
        n_samples, n_dims = embeddings.shape
        unique_batches = np.unique(batch_labels)

        # Initialize clusters via k-means++ on subset
        self.cluster_centroids = self._kmeans_plusplus_init(embeddings, self.num_clusters)

        for iteration in range(self.max_iter):
            # E-step: soft assignment
            distances = np.linalg.norm(
                embeddings[:, np.newaxis, :] - self.cluster_centroids[np.newaxis, :, :],
                axis=2
            )  # (n_samples, num_clusters)

            # Soft assignment with diversity constraint (max diversity = min within-batch variance)
            self.cluster_assignments = np.exp(-distances / np.mean(distances))
            self.cluster_assignments /= self.cluster_assignments.sum(axis=1, keepdims=True)

            # M-step: ridge regression correction
            self.correction_weights = {}
            for batch in unique_batches:
                mask = batch_labels == batch
                batch_embeddings = embeddings[mask]
                batch_assignments = self.cluster_assignments[mask]

                # Ridge regression: predict batch effect from cluster assignments
                alpha = 1.0  # regularization
                XtX = batch_assignments.T @ batch_assignments + alpha * np.eye(self.num_clusters)
                Xty = batch_assignments.T @ batch_embeddings
                try:
                    weights = np.linalg.solve(XtX, Xty)
                except np.linalg.LinAlgError:
                    weights = np.linalg.pinv(XtX) @ Xty
                self.correction_weights[batch] = weights

            # Check convergence via cluster centroid movement
            old_centroids = self.cluster_centroids.copy()
            self.cluster_centroids = self.cluster_assignments.T @ embeddings
            self.cluster_centroids /= self.cluster_assignments.sum(axis=0, keepdims=True).T

            centroid_shift = np.linalg.norm(self.cluster_centroids - old_centroids)
            if centroid_shift < self.convergence_tol:
                logger.info(f"Harmony converged at iteration {iteration}")
                break

        logger.info(f"Harmony fitted with {self.num_clusters} clusters in {iteration + 1} iterations")
        return self

    def transform(self, embeddings: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
        """Apply Harmony correction.

        Args:
            embeddings: Sample embeddings (n_samples, n_dims).
            batch_labels: Batch assignment per sample (n_samples,).

        Returns:
            Harmony-corrected embeddings (n_samples, n_dims).
        """
        if self.cluster_centroids is None:
            raise ValueError("Integrator not fitted. Call fit() first.")

        embeddings_corrected = embeddings.copy()
        unique_batches = np.unique(batch_labels)

        for batch in unique_batches:
            if batch not in self.correction_weights:
                continue

            mask = batch_labels == batch
            batch_embeddings = embeddings[mask]

            # Compute soft assignment for new samples
            distances = np.linalg.norm(
                batch_embeddings[:, np.newaxis, :] - self.cluster_centroids[np.newaxis, :, :],
                axis=2
            )
            assignments = np.exp(-distances / np.mean(distances))
            assignments /= assignments.sum(axis=1, keepdims=True)

            # Remove batch effect
            batch_effect = assignments @ self.correction_weights[batch]
            embeddings_corrected[mask] = batch_embeddings - batch_effect

        return embeddings_corrected

    @staticmethod
    def _kmeans_plusplus_init(data: np.ndarray, k: int) -> np.ndarray:
        """K-means++ initialization."""
        n = len(data)
        centers = [data[np.random.randint(n)]]

        for _ in range(k - 1):
            distances = np.min([np.linalg.norm(data - c, axis=1) for c in centers], axis=0)
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            centers.append(data[np.random.choice(n, p=probabilities)])

        return np.array(centers)


class FederatedBatchNorm(nn.Module):
    """Federated Batch Normalization with site-specific statistics and global weight sharing.

    Each site maintains its own running mean/variance while sharing global BN parameters.
    Enables site-specific adaptation without per-site models.
    """

    def __init__(self, num_features: int, num_sites: int, momentum: float = 0.9,
                 eps: float = 1e-5):
        """Initialize Federated Batch Norm.

        Args:
            num_features: Number of features (channels).
            num_sites: Number of data collection sites.
            momentum: Momentum for running statistics updates.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.num_features = num_features
        self.num_sites = num_sites
        self.momentum = momentum
        self.eps = eps

        # Global parameters (shared across sites)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Site-specific running statistics
        self.register_buffer('running_mean', torch.zeros(num_sites, num_features))
        self.register_buffer('running_var', torch.ones(num_sites, num_features))

    def forward(self, x: torch.Tensor, site_idx: int, training: bool = True) -> torch.Tensor:
        """Forward pass with site-specific normalization.

        Args:
            x: Input tensor (batch_size, num_features) or (batch_size, num_features, ...).
            site_idx: Site index (0 to num_sites - 1).
            training: Whether in training mode.

        Returns:
            Normalized output using site-specific running statistics.
        """
        if site_idx < 0 or site_idx >= self.num_sites:
            raise ValueError(f"site_idx {site_idx} out of range [0, {self.num_sites})")

        if training:
            # Compute batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=True)

            # Update site-specific running statistics
            self.running_mean[site_idx] = (
                (1 - self.momentum) * self.running_mean[site_idx] +
                self.momentum * batch_mean.detach()
            )
            self.running_var[site_idx] = (
                (1 - self.momentum) * self.running_var[site_idx] +
                self.momentum * batch_var.detach()
            )
        else:
            batch_mean = self.running_mean[site_idx]
            batch_var = self.running_var[site_idx]

        # Normalize using site-specific statistics
        x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

        # Scale and shift with global parameters
        return self.weight * x_normalized + self.bias


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation for efficient fine-tuning on new sites.

    Adds low-rank matrices to existing parameters without modifying base model weights.
    Enables rapid adaptation with minimal parameters.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8):
        """Initialize LoRA adapter.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            rank: Low-rank dimension (r << min(in, out)).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Low-rank decomposition: A (out x r) and B (r x in)
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)

        # Initialize with small values
        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        self.scaling = 1.0 / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute low-rank adaptation.

        Args:
            x: Input tensor (batch_size, in_features).

        Returns:
            LoRA output (batch_size, out_features).
        """
        return self.scaling * self.lora_b(self.lora_a(x))


class CorrectionRouter:
    """Decision tree for selecting appropriate correction method based on data characteristics.

    Routes data to ComBat-seq, Harmony, scVI, or GSVA based on:
    - Data type (counts vs. embeddings)
    - Batch effect severity
    - Data dimensionality and size
    """

    def __init__(self):
        """Initialize router with decision thresholds."""
        self.severity_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 1.0
        }

    def assess_batch_severity(self, data: np.ndarray, batch_labels: np.ndarray) -> float:
        """Assess severity of batch effects via silhouette-like metric.

        Args:
            data: Data matrix (n_samples, n_features).
            batch_labels: Batch assignment.

        Returns:
            Severity score in [0, 1]. Higher = more severe batch effect.
        """
        unique_batches = np.unique(batch_labels)

        if len(unique_batches) < 2:
            return 0.0

        # Compute within-batch vs. between-batch variance
        within_batch_var = 0.0
        total_var = np.var(data)

        for batch in unique_batches:
            mask = batch_labels == batch
            within_batch_var += np.var(data[mask], axis=0).mean()

        within_batch_var /= len(unique_batches)
        severity = 1.0 - (within_batch_var / (total_var + 1e-8))

        return np.clip(severity, 0.0, 1.0)

    def route(self, data: np.ndarray, batch_labels: np.ndarray,
              data_type: Literal['counts', 'embeddings'] = 'embeddings') -> Literal['combat_seq', 'harmony', 'lora', 'gsva']:
        """Route data to appropriate correction method.

        Args:
            data: Data matrix (n_samples, n_features).
            batch_labels: Batch assignment.
            data_type: Type of data ('counts' or 'embeddings').

        Returns:
            Recommended correction method.
        """
        severity = self.assess_batch_severity(data, batch_labels)
        n_samples, n_features = data.shape

        # Decision tree
        if data_type == 'counts':
            return 'combat_seq'

        if severity < self.severity_thresholds['low']:
            return 'gsva'  # Mild effect, use pathway abstraction

        if severity < self.severity_thresholds['medium']:
            if n_samples > 10000 or n_features > 2000:
                return 'lora'  # LoRA for large datasets
            return 'harmony'

        # High severity
        if n_samples < 1000:
            return 'harmony'
        return 'lora'


class CrossSiteMonitor:
    """Monitor cross-site consistency via AUC variance tracking and drift detection.

    Tracks model performance variance across sites. Acceptable: <5%, Warning: 5-10%,
    Alert: >10%. Enables early detection of harmonization failures.
    """

    def __init__(self, acceptable_variance: float = 0.05,
                 warning_variance: float = 0.10):
        """Initialize cross-site monitor.

        Args:
            acceptable_variance: AUC variance threshold (acceptable).
            warning_variance: AUC variance threshold (warning level).
        """
        self.acceptable_variance = acceptable_variance
        self.warning_variance = warning_variance
        self.site_aucs = {}
        self.history = []

    def record_site_performance(self, site_id: str, auc: float) -> None:
        """Record AUC for a site.

        Args:
            site_id: Unique site identifier.
            auc: Area under ROC curve (0-1).
        """
        if site_id not in self.site_aucs:
            self.site_aucs[site_id] = []
        self.site_aucs[site_id].append(auc)

    def compute_variance(self) -> float:
        """Compute AUC variance across all sites.

        Returns:
            Relative AUC variance (std / mean).
        """
        aucs = [np.mean(v) for v in self.site_aucs.values() if len(v) > 0]

        if len(aucs) < 2:
            return 0.0

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        return (std_auc / (mean_auc + 1e-8))

    def check_drift(self) -> Tuple[str, dict]:
        """Check for AUC drift across sites.

        Returns:
            Tuple of (status, details) where status is 'ok', 'warning', or 'alert'.
        """
        variance = self.compute_variance()
        details = {
            'variance': variance,
            'num_sites': len(self.site_aucs),
            'mean_auc': np.mean([np.mean(v) for v in self.site_aucs.values()]),
        }

        if variance <= self.acceptable_variance:
            status = 'ok'
        elif variance <= self.warning_variance:
            status = 'warning'
            logger.warning(f"Cross-site AUC variance {variance:.4f} exceeds acceptable threshold")
        else:
            status = 'alert'
            logger.error(f"Cross-site AUC variance {variance:.4f} exceeds warning threshold")

        self.history.append({'variance': variance, 'status': status})
        return status, details


class MultiSiteHarmonizer:
    """Top-level orchestrator for multi-site data harmonization.

    Coordinates method selection, correction application, and monitoring across
    multiple data collection sites.
    """

    def __init__(self, config: Optional[HarmonizationConfig] = None):
        """Initialize multi-site harmonizer.

        Args:
            config: Harmonization configuration. Uses defaults if not provided.
        """
        self.config = config or HarmonizationConfig()
        self.router = CorrectionRouter()
        self.monitor = CrossSiteMonitor(
            acceptable_variance=self.config.monitor_auc_threshold,
            warning_variance=self.config.warning_threshold
        )
        self.correctors = {}
        self.corrections_log = []

    def fit(self, data: np.ndarray, batch_labels: np.ndarray,
            data_type: Literal['counts', 'embeddings'] = 'embeddings',
            site_ids: Optional[list[str]] = None) -> CorrectionsApplied:
        """Fit harmonization model.

        Args:
            data: Data matrix (n_samples, n_features).
            batch_labels: Batch/site assignment per sample (n_samples,).
            data_type: Type of data ('counts' or 'embeddings').
            site_ids: Optional list of site identifiers.

        Returns:
            CorrectionsApplied tracking object.
        """
        unique_sites = np.unique(batch_labels)
        if site_ids is None:
            site_ids = [str(s) for s in unique_sites]

        # Route to appropriate method
        method = self.router.route(data, batch_labels, data_type)
        logger.info(f"Routing to {method} for {data_type} harmonization")

        corrections = CorrectionsApplied(method=method, sites_corrected=site_ids)

        # Fit corrector
        if method == 'combat_seq':
            corrector = ComBatSeqCorrector(
                shrinkage_strength=self.config.combat_shrinkage_strength
            )
            corrector.fit(data, batch_labels)
            self.correctors['primary'] = corrector

        elif method == 'harmony':
            corrector = HarmonyIntegrator(
                num_clusters=self.config.harmony_num_clusters,
                convergence_tol=self.config.harmony_convergence_tol,
                max_iter=self.config.harmony_max_iter
            )
            corrector.fit(data, batch_labels)
            self.correctors['primary'] = corrector
            corrections.iterations = self.config.harmony_max_iter

        elif method == 'lora':
            # Initialize federated batch norm and LoRA
            n_sites = len(unique_sites)
            self.correctors['fedbatch'] = FederatedBatchNorm(
                num_features=data.shape[1],
                num_sites=n_sites,
                momentum=self.config.fedbatch_momentum
            )
            self.correctors['lora'] = LoRAAdapter(
                in_features=data.shape[1],
                out_features=data.shape[1],
                rank=self.config.lora_rank
            )

        self.corrections_log.append(corrections)
        return corrections

    def transform(self, data: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
        """Apply fitted harmonization.

        Args:
            data: Data matrix (n_samples, n_features).
            batch_labels: Batch/site assignment per sample (n_samples,).

        Returns:
            Harmonized data matrix (n_samples, n_features).
        """
        if not self.correctors:
            raise ValueError("No corrector fitted. Call fit() first.")

        if 'primary' in self.correctors:
            return self.correctors['primary'].transform(data, batch_labels)

        # For LoRA/FedBN: apply in PyTorch
        if 'fedbatch' in self.correctors:
            data_tensor = torch.from_numpy(data).float()
            unique_sites = np.unique(batch_labels)
            data_corrected = torch.zeros_like(data_tensor)

            for site_idx, site in enumerate(unique_sites):
                mask = batch_labels == site
                batch_data = data_tensor[mask]
                batch_data_norm = self.correctors['fedbatch'](
                    batch_data, site_idx=site_idx, training=False
                )
                batch_data_corrected = batch_data_norm + \
                    self.correctors['lora'](batch_data_norm)
                data_corrected[mask] = batch_data_corrected

            return data_corrected.numpy()

        return data

    def fit_transform(self, data: np.ndarray, batch_labels: np.ndarray,
                      data_type: Literal['counts', 'embeddings'] = 'embeddings',
                      site_ids: Optional[list[str]] = None) -> Tuple[np.ndarray, CorrectionsApplied]:
        """Fit and apply harmonization in one call.

        Args:
            data: Data matrix (n_samples, n_features).
            batch_labels: Batch/site assignment per sample (n_samples,).
            data_type: Type of data ('counts' or 'embeddings').
            site_ids: Optional list of site identifiers.

        Returns:
            Tuple of (harmonized_data, corrections_applied).
        """
        corrections = self.fit(data, batch_labels, data_type, site_ids)
        harmonized = self.transform(data, batch_labels)
        return harmonized, corrections

    def get_monitor_status(self) -> Tuple[str, dict]:
        """Get current harmonization monitoring status.

        Returns:
            Tuple of (status, details) from CrossSiteMonitor.
        """
        return self.monitor.check_drift()
