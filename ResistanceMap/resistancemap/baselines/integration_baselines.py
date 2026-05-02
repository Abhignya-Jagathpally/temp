"""
resistancemap/baselines/integration_baselines.py
==================================================
Multi-omics integration baselines for comparison with ResistanceMap v6.

Includes:
  - MOFA+ (Multi-Omics Factor Analysis): Bayesian factor model
  - CellRank: Markov-chain cell fate prediction
  - scVI: Deep generative model for single-cell integration

Each baseline wraps the original library when available, with fallback
numpy implementations that mirror the core approach.

Requirements:
    pip install numpy pandas scikit-learn scipy
    Optional: pip install muon mofapy2 cellrank scvi-tools scanpy
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, roc_auc_score, average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from mofapy2.run.entry_point import entry_point as mofa_entry
    HAS_MOFA = True
except ImportError:
    HAS_MOFA = False

try:
    import cellrank as cr
    import scvelo as scv
    HAS_CELLRANK = True
except ImportError:
    HAS_CELLRANK = False

try:
    import scvi as scvi_lib
    HAS_SCVI = True
except ImportError:
    HAS_SCVI = False

try:
    import anndata as ad
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False


# ---------------------------------------------------------------------------
# Integration Quality Metrics
# ---------------------------------------------------------------------------

def silhouette_by_label(latent: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score for integration quality."""
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    try:
        return float(silhouette_score(latent, labels, metric="euclidean"))
    except ValueError:
        return float("nan")


def compute_kbet(
    latent: np.ndarray,
    batch_labels: np.ndarray,
    n_neighbors: int = 25,
) -> float:
    """Simplified kBET (k-nearest-neighbor Batch Effect Test).

    Returns rejection rate (lower is better integration).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(latent) - 1))
    nn.fit(latent)
    _, indices = nn.kneighbors(latent)

    batch_set = np.unique(batch_labels)
    global_freq = {b: (batch_labels == b).mean() for b in batch_set}

    rejections = 0
    for i in range(len(latent)):
        neighbor_batches = batch_labels[indices[i]]
        local_freq = {b: (neighbor_batches == b).mean() for b in batch_set}
        # Chi-squared test against global frequencies
        chi2 = sum(
            (local_freq.get(b, 0) - global_freq[b]) ** 2 / max(global_freq[b], 1e-10)
            for b in batch_set
        )
        chi2 *= n_neighbors
        p_value = 1 - stats.chi2.cdf(chi2, df=len(batch_set) - 1)
        if p_value < 0.05:
            rejections += 1

    return rejections / len(latent)


def compute_lisi(
    latent: np.ndarray,
    labels: np.ndarray,
    perplexity: int = 30,
) -> float:
    """Simplified LISI (Local Inverse Simpson Index).

    Higher LISI = better mixing of labels in local neighborhoods.
    """
    from sklearn.neighbors import NearestNeighbors

    k = min(perplexity * 3, len(latent) - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(latent)
    distances, indices = nn.kneighbors(latent)

    lisi_values = []
    for i in range(len(latent)):
        neighbor_labels = labels[indices[i]]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        freqs = counts / counts.sum()
        simpson = np.sum(freqs ** 2)
        lisi_values.append(1.0 / max(simpson, 1e-10))

    return float(np.median(lisi_values))


def evaluate_integration(
    latent: np.ndarray,
    cell_type_labels: np.ndarray,
    batch_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Comprehensive integration quality evaluation."""
    metrics = {}

    # Bio conservation
    metrics["silhouette_celltype"] = silhouette_by_label(latent, cell_type_labels)
    metrics["lisi_celltype"] = compute_lisi(latent, cell_type_labels)

    # Batch mixing
    if batch_labels is not None:
        metrics["kbet_rejection"] = compute_kbet(latent, batch_labels)
        metrics["silhouette_batch"] = silhouette_by_label(latent, batch_labels)
        metrics["lisi_batch"] = compute_lisi(latent, batch_labels)

    return metrics


# ---------------------------------------------------------------------------
# MOFA+ Baseline
# ---------------------------------------------------------------------------

class MOFAPlusBaseline:
    """MOFA+ multi-omics factor analysis baseline.

    When mofapy2 is available, runs the full Bayesian factor model.
    Otherwise, falls back to a joint PCA-based factorization that
    approximates the MOFA factor decomposition.

    Parameters
    ----------
    n_factors : int
        Number of latent factors to learn.
    seed : int
        Random seed.
    """

    def __init__(self, n_factors: int = 15, seed: int = 42):
        self.n_factors = n_factors
        self.seed = seed
        self._factors: Optional[np.ndarray] = None
        self._weights: Dict[str, np.ndarray] = {}
        self._is_fitted = False
        self._scaler = StandardScaler()

    def fit(
        self,
        modalities: Dict[str, np.ndarray],
        sample_ids: Optional[np.ndarray] = None,
    ) -> "MOFAPlusBaseline":
        """Fit MOFA+ factor model on multi-omics data.

        Parameters
        ----------
        modalities : dict of modality_name -> np.ndarray (n_samples, n_features)
        sample_ids : np.ndarray, optional
        """
        n_samples = next(iter(modalities.values())).shape[0]
        logger.info(f"MOFA+ fitting: {n_samples} samples, {len(modalities)} modalities")

        if HAS_MOFA:
            self._fit_mofa(modalities)
        else:
            self._fit_joint_pca(modalities)

        self._is_fitted = True
        return self

    def _fit_mofa(self, modalities: Dict[str, np.ndarray]) -> None:
        """Fit using mofapy2."""
        try:
            ent = mofa_entry()
            data = [[None]] * len(modalities)
            for i, (name, mat) in enumerate(modalities.items()):
                data[i] = [mat.astype(np.float64)]
            ent.set_data_options(scale_groups=False, scale_views=True)
            ent.set_data_matrix(data, views_names=list(modalities.keys()))
            ent.set_model_options(factors=self.n_factors)
            ent.set_training_options(seed=self.seed, convergence_mode="fast")
            ent.build()
            ent.run()
            self._factors = ent.model.getExpectations()["Z"]["E"][0]
            for i, name in enumerate(modalities.keys()):
                self._weights[name] = ent.model.getExpectations()["W"]["E"][i]
            logger.info("MOFA+ (mofapy2) fit complete")
        except Exception as e:
            logger.warning(f"mofapy2 failed ({e}), falling back to joint PCA")
            self._fit_joint_pca(modalities)

    def _fit_joint_pca(self, modalities: Dict[str, np.ndarray]) -> None:
        """Fallback: approximate MOFA with joint PCA on concatenated modalities."""
        # Scale each modality
        scaled = {}
        for name, mat in modalities.items():
            s = StandardScaler()
            scaled[name] = s.fit_transform(np.nan_to_num(mat, nan=0.0))

        # Concatenate
        concat = np.hstack(list(scaled.values()))

        # PCA
        n_comp = min(self.n_factors, concat.shape[0] - 1, concat.shape[1])
        pca = PCA(n_components=n_comp, random_state=self.seed)
        self._factors = pca.fit_transform(concat)

        # Extract per-modality weights (loadings)
        offset = 0
        for name, mat in modalities.items():
            n_feat = mat.shape[1]
            self._weights[name] = pca.components_[:, offset:offset + n_feat].T
            offset += n_feat

        variance_explained = pca.explained_variance_ratio_.sum()
        logger.info(f"Joint PCA: {n_comp} factors, {variance_explained:.1%} variance explained")

    def get_factors(self) -> np.ndarray:
        """Return learned latent factors (n_samples, n_factors)."""
        if self._factors is None:
            raise RuntimeError("Must call fit first")
        return self._factors

    def get_weights(self, modality: str) -> np.ndarray:
        """Return factor weights for a specific modality."""
        if modality not in self._weights:
            raise ValueError(f"Unknown modality: {modality}")
        return self._weights[modality]

    def predict_downstream(
        self,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Use learned factors for downstream drug response prediction."""
        Z = self.get_factors()
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(Z[train_idx])
        Z_test = scaler.transform(Z[test_idx])

        clf = LogisticRegression(
            C=1.0, penalty="l2", max_iter=2000,
            class_weight="balanced", random_state=self.seed,
        )
        clf.fit(Z_train, y[train_idx].astype(int))
        y_prob = clf.predict_proba(Z_test)[:, 1]

        metrics = {}
        try:
            metrics["auroc"] = float(roc_auc_score(y[test_idx], y_prob))
        except ValueError:
            metrics["auroc"] = float("nan")
        try:
            metrics["auprc"] = float(average_precision_score(y[test_idx], y_prob))
        except ValueError:
            metrics["auprc"] = float("nan")

        return y_prob, metrics

    def evaluate_integration(
        self,
        cell_type_labels: np.ndarray,
        batch_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate integration quality of learned factors."""
        return evaluate_integration(self.get_factors(), cell_type_labels, batch_labels)


# ---------------------------------------------------------------------------
# CellRank Baseline
# ---------------------------------------------------------------------------

class CellRankBaseline:
    """CellRank cell fate prediction baseline.

    CellRank combines RNA velocity with transcriptomic similarity to
    compute cell fate probabilities via a Markov chain. We adapt it
    to predict drug resistance trajectory.

    When cellrank is not available, falls back to a diffusion-based
    pseudotime approach.
    """

    def __init__(self, n_states: int = 3, seed: int = 42):
        self.n_states = n_states
        self.seed = seed
        self._fate_probs: Optional[np.ndarray] = None
        self._pseudotime: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(
        self,
        expression: np.ndarray,
        gene_names: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
    ) -> "CellRankBaseline":
        """Fit CellRank trajectory model.

        Parameters
        ----------
        expression : np.ndarray, shape (n_cells, n_genes)
        gene_names : np.ndarray, optional
        velocity : np.ndarray, optional
            RNA velocity estimates.
        """
        if HAS_CELLRANK and HAS_SCANPY:
            self._fit_cellrank(expression, gene_names, velocity)
        else:
            self._fit_diffusion(expression)

        self._is_fitted = True
        return self

    def _fit_cellrank(self, expression, gene_names, velocity):
        """Fit using cellrank."""
        try:
            adata = ad.AnnData(X=expression.astype(np.float32))
            if gene_names is not None:
                adata.var_names = pd.Index(gene_names)

            sc.pp.pca(adata)
            sc.pp.neighbors(adata)

            if velocity is not None:
                adata.layers["velocity"] = velocity
                scv.pp.moments(adata)
                vk = cr.kernels.VelocityKernel(adata)
                vk.compute_transition_matrix()
                kernel = vk
            else:
                ck = cr.kernels.ConnectivityKernel(adata)
                ck.compute_transition_matrix()
                kernel = ck

            estimator = cr.estimators.GPCCA(kernel)
            estimator.fit(n_states=self.n_states)
            estimator.predict()

            self._fate_probs = estimator.fate_probabilities.values
            logger.info(f"CellRank fit: {self._fate_probs.shape[1]} terminal states")
        except Exception as e:
            logger.warning(f"CellRank failed ({e}), falling back to diffusion")
            self._fit_diffusion(expression)

    def _fit_diffusion(self, expression: np.ndarray) -> None:
        """Fallback: diffusion pseudotime + spectral clustering for fate."""
        n_cells = expression.shape[0]

        # PCA
        pca = PCA(n_components=min(50, n_cells - 1, expression.shape[1]),
                  random_state=self.seed)
        Z = pca.fit_transform(expression)

        # k-NN graph
        from sklearn.neighbors import NearestNeighbors
        k = min(30, n_cells - 1)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(Z)
        distances, indices = nn.kneighbors(Z)

        # Transition matrix (Gaussian kernel)
        sigma = np.median(distances[:, -1])
        T = np.zeros((n_cells, n_cells))
        for i in range(n_cells):
            for j_idx in range(k):
                j = indices[i, j_idx]
                w = np.exp(-(distances[i, j_idx] ** 2) / (2 * sigma ** 2))
                T[i, j] = w
        # Row normalize
        row_sums = T.sum(axis=1, keepdims=True)
        T = T / np.maximum(row_sums, 1e-10)

        # Diffusion pseudotime via dominant eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(T + T.T)
        # Sort descending
        idx_sort = np.argsort(eigenvalues)[::-1]
        self._pseudotime = eigenvectors[:, idx_sort[1]]  # 2nd eigenvector = pseudotime

        # Fate probabilities: cluster terminal states, compute absorption probs
        kmeans = KMeans(n_clusters=self.n_states, random_state=self.seed, n_init=10)
        clusters = kmeans.fit_predict(Z)

        # Approximate fate probs via softmax of distance to cluster centers
        dists = cdist(Z, kmeans.cluster_centers_)
        inv_dists = 1.0 / (dists + 1e-8)
        self._fate_probs = inv_dists / inv_dists.sum(axis=1, keepdims=True)

        logger.info(f"Diffusion pseudotime computed: {self.n_states} states")

    def get_fate_probabilities(self) -> np.ndarray:
        """Return fate probabilities (n_cells, n_states)."""
        if self._fate_probs is None:
            raise RuntimeError("Must call fit first")
        return self._fate_probs

    def get_pseudotime(self) -> Optional[np.ndarray]:
        """Return pseudotime ordering."""
        return self._pseudotime

    def predict_downstream(
        self,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Use fate probs for drug response prediction."""
        Z = self.get_fate_probabilities()
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                random_state=self.seed)
        clf.fit(Z[train_idx], y[train_idx].astype(int))
        y_prob = clf.predict_proba(Z[test_idx])[:, 1]

        metrics = {}
        try:
            metrics["auroc"] = float(roc_auc_score(y[test_idx], y_prob))
        except ValueError:
            metrics["auroc"] = float("nan")
        try:
            metrics["auprc"] = float(average_precision_score(y[test_idx], y_prob))
        except ValueError:
            metrics["auprc"] = float("nan")

        return y_prob, metrics


# ---------------------------------------------------------------------------
# scVI Baseline
# ---------------------------------------------------------------------------

class ScVIBaseline:
    """scVI deep generative model baseline.

    scVI fits a variational autoencoder to single-cell RNA-seq,
    producing a latent representation useful for downstream tasks.
    Falls back to a simple VAE when scvi-tools is unavailable.
    """

    def __init__(self, n_latent: int = 30, n_epochs: int = 100, seed: int = 42):
        self.n_latent = n_latent
        self.n_epochs = n_epochs
        self.seed = seed
        self._latent: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(
        self,
        expression: np.ndarray,
        batch_labels: Optional[np.ndarray] = None,
        gene_names: Optional[np.ndarray] = None,
    ) -> "ScVIBaseline":
        """Fit scVI model.

        Parameters
        ----------
        expression : np.ndarray, shape (n_cells, n_genes)
            Raw or normalized count matrix.
        batch_labels : np.ndarray, optional
            Batch assignments for integration.
        """
        if HAS_SCVI and HAS_SCANPY:
            self._fit_scvi(expression, batch_labels, gene_names)
        elif HAS_TORCH:
            self._fit_simple_vae(expression)
        else:
            self._fit_pca_fallback(expression)

        self._is_fitted = True
        return self

    def _fit_scvi(self, expression, batch_labels, gene_names):
        """Fit using scvi-tools."""
        try:
            adata = ad.AnnData(X=expression.astype(np.float32))
            if gene_names is not None:
                adata.var_names = pd.Index(gene_names)
            if batch_labels is not None:
                adata.obs["batch"] = batch_labels.astype(str)

            scvi_lib.model.SCVI.setup_anndata(
                adata, batch_key="batch" if batch_labels is not None else None
            )
            model = scvi_lib.model.SCVI(adata, n_latent=self.n_latent)
            model.train(max_epochs=self.n_epochs)
            self._latent = model.get_latent_representation()
            self._model = model
            logger.info(f"scVI fit: latent shape {self._latent.shape}")
        except Exception as e:
            logger.warning(f"scvi-tools failed ({e}), falling back to simple VAE")
            if HAS_TORCH:
                self._fit_simple_vae(expression)
            else:
                self._fit_pca_fallback(expression)

    def _fit_simple_vae(self, expression: np.ndarray) -> None:
        """Fallback: simple VAE in PyTorch."""
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_genes = expression.shape[1]

        # Scale
        scaler = StandardScaler()
        X = scaler.fit_transform(expression).astype(np.float32)

        # Simple VAE
        hidden = 256
        encoder = nn.Sequential(
            nn.Linear(n_genes, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        mu_layer = nn.Linear(hidden, self.n_latent)
        logvar_layer = nn.Linear(hidden, self.n_latent)
        decoder = nn.Sequential(
            nn.Linear(self.n_latent, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_genes),
        )

        all_params = (
            list(encoder.parameters()) + list(mu_layer.parameters()) +
            list(logvar_layer.parameters()) + list(decoder.parameters())
        )
        encoder.to(device); mu_layer.to(device)
        logvar_layer.to(device); decoder.to(device)
        optimizer = torch.optim.Adam(all_params, lr=1e-3)

        dataset = TensorDataset(torch.tensor(X))
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in range(min(self.n_epochs, 50)):
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                h = encoder(batch_x)
                mu = mu_layer(h)
                logvar = logvar_layer(h)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                x_recon = decoder(z)
                recon_loss = nn.functional.mse_loss(x_recon, batch_x)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.5 * kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Extract latent
        encoder.eval(); mu_layer.eval()
        with torch.no_grad():
            X_t = torch.tensor(X).to(device)
            h = encoder(X_t)
            self._latent = mu_layer(h).cpu().numpy()

        logger.info(f"Simple VAE fit: latent shape {self._latent.shape}")

    def _fit_pca_fallback(self, expression: np.ndarray) -> None:
        """Ultimate fallback: PCA as latent space."""
        scaler = StandardScaler()
        X = scaler.fit_transform(np.nan_to_num(expression, nan=0.0))
        n_comp = min(self.n_latent, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=self.seed)
        self._latent = pca.fit_transform(X)
        logger.info(f"PCA fallback: latent shape {self._latent.shape}")

    def get_latent(self) -> np.ndarray:
        """Return latent representation."""
        if self._latent is None:
            raise RuntimeError("Must call fit first")
        return self._latent

    def predict_downstream(
        self,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Use latent space for drug response prediction."""
        Z = self.get_latent()
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(Z[train_idx])
        Z_test = scaler.transform(Z[test_idx])

        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                random_state=self.seed)
        clf.fit(Z_train, y[train_idx].astype(int))
        y_prob = clf.predict_proba(Z_test)[:, 1]

        metrics = {}
        try:
            metrics["auroc"] = float(roc_auc_score(y[test_idx], y_prob))
        except ValueError:
            metrics["auroc"] = float("nan")
        try:
            metrics["auprc"] = float(average_precision_score(y[test_idx], y_prob))
        except ValueError:
            metrics["auprc"] = float("nan")

        return y_prob, metrics

    def evaluate_integration(
        self,
        cell_type_labels: np.ndarray,
        batch_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate latent space quality."""
        return evaluate_integration(self.get_latent(), cell_type_labels, batch_labels)


# ---------------------------------------------------------------------------
# Combined Integration Benchmark Runner
# ---------------------------------------------------------------------------

class IntegrationBenchmark:
    """Run all integration baselines on the same data.

    Evaluates both integration quality (silhouette, kBET, LISI) and
    downstream prediction (AUROC, AUPRC).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.results: Dict[str, Dict[str, float]] = {}

    def run(
        self,
        modalities: Dict[str, np.ndarray],
        expression: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        cell_type_labels: Optional[np.ndarray] = None,
        batch_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Run all integration baselines."""
        # MOFA+
        logger.info("Running MOFA+ baseline...")
        try:
            mofa = MOFAPlusBaseline(seed=self.seed)
            mofa.fit(modalities)
            _, mofa_metrics = mofa.predict_downstream(y, train_idx, test_idx)
            if cell_type_labels is not None:
                integ_metrics = mofa.evaluate_integration(cell_type_labels, batch_labels)
                mofa_metrics.update(integ_metrics)
            self.results["MOFA+"] = mofa_metrics
        except Exception as e:
            logger.warning(f"MOFA+ failed: {e}")
            self.results["MOFA+"] = {"error": str(e)}

        # CellRank
        logger.info("Running CellRank baseline...")
        try:
            cellrank = CellRankBaseline(seed=self.seed)
            cellrank.fit(expression)
            _, cr_metrics = cellrank.predict_downstream(y, train_idx, test_idx)
            self.results["CellRank"] = cr_metrics
        except Exception as e:
            logger.warning(f"CellRank failed: {e}")
            self.results["CellRank"] = {"error": str(e)}

        # scVI
        logger.info("Running scVI baseline...")
        try:
            scvi_bl = ScVIBaseline(seed=self.seed)
            scvi_bl.fit(expression, batch_labels)
            _, scvi_metrics = scvi_bl.predict_downstream(y, train_idx, test_idx)
            if cell_type_labels is not None:
                integ_metrics = scvi_bl.evaluate_integration(cell_type_labels, batch_labels)
                scvi_metrics.update(integ_metrics)
            self.results["scVI"] = scvi_metrics
        except Exception as e:
            logger.warning(f"scVI failed: {e}")
            self.results["scVI"] = {"error": str(e)}

        return self.results

    def summary_table(self) -> pd.DataFrame:
        """Return results as a DataFrame."""
        rows = []
        for name, metrics in self.results.items():
            row = {"model": name}
            row.update({k: v for k, v in metrics.items() if k != "error"})
            rows.append(row)
        return pd.DataFrame(rows).set_index("model").round(4)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_mofa_baseline():
    """Test MOFA+ baseline."""
    np.random.seed(42)
    n = 100
    modalities = {
        "expression": np.random.randn(n, 200),
        "proteomics": np.random.randn(n, 50),
    }
    y = np.random.choice([0, 1], n)
    train_idx = np.arange(70)
    test_idx = np.arange(70, 100)

    mofa = MOFAPlusBaseline(n_factors=10, seed=42)
    mofa.fit(modalities)
    factors = mofa.get_factors()
    assert factors.shape == (n, 10)

    y_prob, metrics = mofa.predict_downstream(y, train_idx, test_idx)
    assert len(y_prob) == 30
    assert "auroc" in metrics
    print("  [PASS] test_mofa_baseline")


def test_cellrank_baseline():
    """Test CellRank baseline."""
    np.random.seed(42)
    n = 100
    expr = np.random.randn(n, 200)
    y = np.random.choice([0, 1], n)

    cr_bl = CellRankBaseline(n_states=3, seed=42)
    cr_bl.fit(expr)
    fate = cr_bl.get_fate_probabilities()
    assert fate.shape == (n, 3)
    assert np.allclose(fate.sum(axis=1), 1.0, atol=1e-5)

    y_prob, metrics = cr_bl.predict_downstream(y, np.arange(70), np.arange(70, 100))
    assert "auroc" in metrics
    print("  [PASS] test_cellrank_baseline")


def test_scvi_baseline():
    """Test scVI baseline."""
    np.random.seed(42)
    n = 100
    expr = np.random.randn(n, 200)
    y = np.random.choice([0, 1], n)

    scvi_bl = ScVIBaseline(n_latent=10, n_epochs=5, seed=42)
    scvi_bl.fit(expr)
    latent = scvi_bl.get_latent()
    assert latent.shape[0] == n

    y_prob, metrics = scvi_bl.predict_downstream(y, np.arange(70), np.arange(70, 100))
    assert "auroc" in metrics
    print("  [PASS] test_scvi_baseline")


def test_integration_metrics():
    """Test integration quality metrics."""
    np.random.seed(42)
    latent = np.random.randn(100, 10)
    labels = np.random.choice(["A", "B", "C"], 100)
    batch = np.random.choice(["batch1", "batch2"], 100)

    metrics = evaluate_integration(latent, labels, batch)
    assert "silhouette_celltype" in metrics
    assert "kbet_rejection" in metrics
    assert "lisi_batch" in metrics
    print("  [PASS] test_integration_metrics")


def test_benchmark_runner():
    """Test full benchmark runner."""
    np.random.seed(42)
    n = 100
    modalities = {
        "expression": np.random.randn(n, 200),
        "proteomics": np.random.randn(n, 50),
    }
    y = np.random.choice([0, 1], n)
    train_idx = np.arange(70)
    test_idx = np.arange(70, 100)

    bench = IntegrationBenchmark(seed=42)
    results = bench.run(
        modalities, modalities["expression"], y, train_idx, test_idx,
    )
    assert len(results) == 3
    table = bench.summary_table()
    assert len(table) > 0
    print("  [PASS] test_benchmark_runner")


def run_tests():
    """Run all integration baseline tests."""
    print("Running integration baseline tests...")
    test_integration_metrics()
    test_mofa_baseline()
    test_cellrank_baseline()
    test_scvi_baseline()
    test_benchmark_runner()
    print("All integration baseline tests passed!")


if __name__ == "__main__":
    run_tests()
