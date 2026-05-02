"""
resistancemap/baselines/cpa_baseline.py
========================================
Wrapper around the Compositional Perturbation Autoencoder (CPA / chemCPA)
for benchmarking against ResistanceMap v6.

CPA predicts gene expression changes under chemical/genetic perturbations
using a disentangled latent space. It can predict expression responses to
unseen drugs at unseen doses.

This baseline:
  - Wraps the scvi-tools CPA implementation (if available)
  - Falls back to a simplified VAE+perturbation model
  - Evaluates on expression prediction (Pearson r) and drug response
  - Bridges the evaluation paradigm gap: CPA predicts expression changes,
    ResistanceMap predicts binary response

Reference: Lotfollahi et al., Molecular Systems Biology, 2023.

Requirements:
    pip install numpy pandas scikit-learn scipy torch
    Optional: pip install scvi-tools
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import scvi
    from scvi.external import CPA as ScviCPA
    HAS_SCVI_CPA = True
except (ImportError, AttributeError):
    HAS_SCVI_CPA = False


# ---------------------------------------------------------------------------
# Simplified CPA Model (fallback when scvi-tools unavailable)
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class SimpleCPAEncoder(nn.Module):
        """Encoder: gene expression -> latent space (basal + perturbation)."""

        def __init__(self, n_genes: int, latent_dim: int = 128, hidden_dim: int = 256):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_genes, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.mu = nn.Linear(hidden_dim, latent_dim)
            self.logvar = nn.Linear(hidden_dim, latent_dim)

        def forward(self, x):
            h = self.encoder(x)
            return self.mu(h), self.logvar(h)

    class SimpleCPADecoder(nn.Module):
        """Decoder: latent + drug embedding -> reconstructed expression."""

        def __init__(self, n_genes: int, latent_dim: int = 128, hidden_dim: int = 256):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_genes),
            )

        def forward(self, z):
            return self.decoder(z)

    class DrugEncoder(nn.Module):
        """Encode drug identity into a perturbation embedding."""

        def __init__(self, n_drugs: int, embedding_dim: int = 64):
            super().__init__()
            self.embedding = nn.Embedding(n_drugs, embedding_dim)
            self.dose_encoder = nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.ReLU(),
            )

        def forward(self, drug_ids: torch.Tensor, doses: torch.Tensor) -> torch.Tensor:
            drug_emb = self.embedding(drug_ids)
            dose_emb = self.dose_encoder(doses.unsqueeze(-1))
            return drug_emb * dose_emb  # multiplicative interaction

    class SimpleCPA(nn.Module):
        """Simplified Compositional Perturbation Autoencoder.

        Disentangles basal cell state from perturbation effect:
          z = z_basal + f(drug, dose)
          x_reconstructed = decoder(z)
        """

        def __init__(
            self,
            n_genes: int,
            n_drugs: int,
            latent_dim: int = 128,
            drug_emb_dim: int = 64,
            hidden_dim: int = 256,
        ):
            super().__init__()
            self.encoder = SimpleCPAEncoder(n_genes, latent_dim, hidden_dim)
            self.decoder = SimpleCPADecoder(n_genes, latent_dim, hidden_dim)
            self.drug_encoder = DrugEncoder(n_drugs, drug_emb_dim)
            self.perturbation_projector = nn.Linear(drug_emb_dim, latent_dim)
            self.adversary = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_drugs),
            )
            self.latent_dim = latent_dim

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x, drug_ids, doses):
            mu, logvar = self.encoder(x)
            z_basal = self.reparameterize(mu, logvar)

            # Perturbation effect
            drug_emb = self.drug_encoder(drug_ids, doses)
            perturbation = self.perturbation_projector(drug_emb)

            # Compose: basal + perturbation
            z = z_basal + perturbation
            x_recon = self.decoder(z)

            # Adversarial: basal state should NOT predict drug
            adv_pred = self.adversary(z_basal)

            return x_recon, mu, logvar, z_basal, adv_pred

        def predict_perturbation(self, x_control, drug_ids, doses):
            """Predict expression after perturbation from control expression."""
            with torch.no_grad():
                mu, _ = self.encoder(x_control)
                drug_emb = self.drug_encoder(drug_ids, doses)
                perturbation = self.perturbation_projector(drug_emb)
                z = mu + perturbation
                return self.decoder(z)


# ---------------------------------------------------------------------------
# CPA Baseline Wrapper
# ---------------------------------------------------------------------------

class CPABaseline:
    """CPA/chemCPA baseline for benchmarking.

    Supports two backends:
      1. scvi-tools CPA (if installed) -- full implementation
      2. SimpleCPA -- lightweight PyTorch reimplementation

    Evaluation modes:
      - Expression prediction: Pearson r between predicted and observed
      - Drug response prediction: convert expression predictions to binary
        response via a downstream classifier

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space.
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    use_scvi : bool
        Try to use scvi-tools CPA (falls back if unavailable).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 128,
        use_scvi: bool = True,
        seed: int = 42,
    ):
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.use_scvi = use_scvi and HAS_SCVI_CPA
        self.seed = seed

        self._model = None
        self._scaler = None
        self._drug_map: Dict[str, int] = {}
        self._response_classifier = None
        self._is_fitted = False

    def fit(
        self,
        expression: np.ndarray,
        drug_labels: np.ndarray,
        doses: np.ndarray,
        control_mask: Optional[np.ndarray] = None,
    ) -> "CPABaseline":
        """Train CPA on perturbation data.

        Parameters
        ----------
        expression : np.ndarray, shape (n_samples, n_genes)
            Gene expression (perturbed and control).
        drug_labels : np.ndarray of str, shape (n_samples,)
            Drug name for each sample ('control' for untreated).
        doses : np.ndarray, shape (n_samples,)
            Drug dose (0 for control).
        control_mask : np.ndarray of bool, optional
            True for control samples.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for CPA: pip install torch")

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Build drug mapping
        unique_drugs = sorted(set(drug_labels))
        self._drug_map = {d: i for i, d in enumerate(unique_drugs)}
        n_drugs = len(unique_drugs)
        n_genes = expression.shape[1]

        # Scale expression
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(expression).astype(np.float32)

        drug_ids = np.array([self._drug_map[d] for d in drug_labels])
        doses_arr = doses.astype(np.float32)

        if control_mask is None:
            control_mask = np.array([d == "control" for d in drug_labels])

        logger.info(f"CPA training: {X.shape[0]} samples, {n_genes} genes, {n_drugs} drugs")

        if self.use_scvi:
            self._fit_scvi(X, drug_labels, doses)
        else:
            self._fit_simple(X, drug_ids, doses_arr, n_genes, n_drugs)

        self._is_fitted = True
        return self

    def _fit_simple(
        self, X: np.ndarray, drug_ids: np.ndarray, doses: np.ndarray,
        n_genes: int, n_drugs: int
    ) -> None:
        """Train the simplified CPA model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SimpleCPA(n_genes, n_drugs, self.latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        X_t = torch.tensor(X, dtype=torch.float32)
        drug_t = torch.tensor(drug_ids, dtype=torch.long)
        dose_t = torch.tensor(doses, dtype=torch.float32)

        dataset = TensorDataset(X_t, drug_t, dose_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                           drop_last=len(dataset) > self.batch_size)

        model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for batch_x, batch_drug, batch_dose in loader:
                batch_x = batch_x.to(device)
                batch_drug = batch_drug.to(device)
                batch_dose = batch_dose.to(device)

                x_recon, mu, logvar, z_basal, adv_pred = model(batch_x, batch_drug, batch_dose)

                # Reconstruction loss
                recon_loss = F.mse_loss(x_recon, batch_x)
                # KL divergence
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                # Adversarial loss (maximize entropy of drug prediction from basal state)
                adv_loss = F.cross_entropy(adv_pred, batch_drug)

                loss = recon_loss + 0.1 * kl_loss - 0.05 * adv_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / max(len(loader), 1)
                logger.debug(f"  Epoch {epoch+1}/{self.n_epochs}, loss={avg_loss:.4f}")

        model.eval()
        self._model = model
        self._device = device

    def _fit_scvi(self, X, drug_labels, doses):
        """Train using scvi-tools CPA."""
        try:
            import anndata as ad
            adata = ad.AnnData(X=X.astype(np.float32))
            adata.obs["drug"] = drug_labels
            adata.obs["dose"] = doses
            adata.obs["control"] = [1 if d == "control" else 0 for d in drug_labels]

            ScviCPA.setup_anndata(adata, drug_key="drug", dose_key="dose",
                                  control_key="control")
            self._model = ScviCPA(adata, n_latent=self.latent_dim)
            self._model.train(max_epochs=self.n_epochs, batch_size=self.batch_size)
            logger.info("scvi-tools CPA training complete")
        except Exception as e:
            logger.warning(f"scvi-tools CPA failed ({e}), falling back to SimpleCPA")
            self.use_scvi = False
            drug_ids = np.array([self._drug_map[d] for d in drug_labels])
            self._fit_simple(X, drug_ids, doses.astype(np.float32),
                           X.shape[1], len(self._drug_map))

    def predict_expression(
        self,
        control_expression: np.ndarray,
        drug_name: str,
        dose: float = 1.0,
    ) -> np.ndarray:
        """Predict expression after drug perturbation.

        Parameters
        ----------
        control_expression : np.ndarray, shape (n_samples, n_genes)
        drug_name : str
        dose : float

        Returns
        -------
        predicted_expression : np.ndarray, shape (n_samples, n_genes)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit first")
        if drug_name not in self._drug_map:
            raise ValueError(f"Unknown drug: {drug_name}. Known: {list(self._drug_map.keys())}")

        X = self._scaler.transform(control_expression).astype(np.float32)

        if self.use_scvi:
            # scvi-tools prediction path would go here
            raise NotImplementedError("scvi-tools prediction not yet implemented")

        device = self._device
        model = self._model
        drug_id = self._drug_map[drug_name]

        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        drug_t = torch.full((X_t.shape[0],), drug_id, dtype=torch.long).to(device)
        dose_t = torch.full((X_t.shape[0],), dose, dtype=torch.float32).to(device)

        with torch.no_grad():
            pred = model.predict_perturbation(X_t, drug_t, dose_t)

        pred_np = pred.cpu().numpy()
        return self._scaler.inverse_transform(pred_np)

    def fit_response_classifier(
        self,
        expression: np.ndarray,
        drug_labels: np.ndarray,
        doses: np.ndarray,
        response_labels: np.ndarray,
    ) -> None:
        """Fit a downstream classifier to convert expression predictions to binary response.

        Bridges the gap between CPA's expression prediction and binary drug response.
        """
        # Get predicted expression for each sample
        predictions = []
        for i in range(len(expression)):
            drug = drug_labels[i]
            dose = float(doses[i])
            if drug in self._drug_map and drug != "control":
                pred = self.predict_expression(expression[i:i+1], drug, dose)
                # Use difference from control as features
                diff = pred - self._scaler.transform(expression[i:i+1])
                predictions.append(diff.flatten())
            else:
                predictions.append(np.zeros(expression.shape[1]))

        X_diff = np.array(predictions)
        valid = ~np.isnan(response_labels)
        X_valid = X_diff[valid]
        y_valid = response_labels[valid].astype(int)

        self._response_classifier = LogisticRegression(
            C=1.0, penalty="l2", max_iter=1000, random_state=self.seed
        )
        self._response_classifier.fit(X_valid, y_valid)
        logger.info(f"Response classifier fitted on {valid.sum()} samples")

    def predict_response(
        self,
        control_expression: np.ndarray,
        drug_name: str,
        dose: float = 1.0,
    ) -> np.ndarray:
        """Predict binary drug response from expression prediction."""
        if self._response_classifier is None:
            raise RuntimeError("Must call fit_response_classifier first")

        pred_expr = self.predict_expression(control_expression, drug_name, dose)
        control_scaled = self._scaler.transform(control_expression)
        diff = pred_expr - self._scaler.inverse_transform(control_scaled)
        # Actually use the scaled diff
        diff_scaled = self._scaler.transform(control_expression + diff) - control_scaled
        return self._response_classifier.predict_proba(diff_scaled)[:, 1]

    def evaluate_expression(
        self,
        control_expression: np.ndarray,
        treated_expression: np.ndarray,
        drug_name: str,
        dose: float = 1.0,
    ) -> Dict[str, float]:
        """Evaluate expression prediction accuracy.

        Returns Pearson r, R^2, and MSE per gene.
        """
        pred = self.predict_expression(control_expression, drug_name, dose)
        actual = treated_expression

        # Per-gene correlations
        gene_correlations = []
        for g in range(actual.shape[1]):
            if actual[:, g].std() > 1e-6:
                r, _ = stats.pearsonr(actual[:, g], pred[:, g])
                gene_correlations.append(r)

        # Per-sample correlations
        sample_correlations = []
        for s in range(actual.shape[0]):
            if actual[s, :].std() > 1e-6:
                r, _ = stats.pearsonr(actual[s, :], pred[s, :])
                sample_correlations.append(r)

        metrics = {
            "mean_gene_pearson_r": float(np.mean(gene_correlations)) if gene_correlations else float("nan"),
            "median_gene_pearson_r": float(np.median(gene_correlations)) if gene_correlations else float("nan"),
            "mean_sample_pearson_r": float(np.mean(sample_correlations)) if sample_correlations else float("nan"),
            "r2_score": float(r2_score(actual.flatten(), pred.flatten())),
            "mse": float(np.mean((actual - pred) ** 2)),
            "n_genes_evaluated": len(gene_correlations),
        }
        return metrics

    def evaluate_response(
        self,
        control_expression: np.ndarray,
        response_labels: np.ndarray,
        drug_name: str,
        dose: float = 1.0,
    ) -> Dict[str, float]:
        """Evaluate binary drug response prediction."""
        y_prob = self.predict_response(control_expression, drug_name, dose)
        valid = ~np.isnan(response_labels)
        yt = response_labels[valid].astype(int)
        yp = y_prob[valid]

        metrics = {}
        try:
            metrics["auroc"] = float(roc_auc_score(yt, yp))
        except ValueError:
            metrics["auroc"] = float("nan")
        try:
            metrics["auprc"] = float(average_precision_score(yt, yp))
        except ValueError:
            metrics["auprc"] = float("nan")
        metrics["n_samples"] = int(valid.sum())
        return metrics


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_simple_cpa():
    """Test SimpleCPA model forward pass."""
    if not HAS_TORCH:
        print("  [SKIP] test_simple_cpa (no PyTorch)")
        return

    n_genes, n_drugs = 200, 5
    model = SimpleCPA(n_genes, n_drugs, latent_dim=32)
    x = torch.randn(16, n_genes)
    drug_ids = torch.randint(0, n_drugs, (16,))
    doses = torch.rand(16)

    x_recon, mu, logvar, z_basal, adv_pred = model(x, drug_ids, doses)
    assert x_recon.shape == (16, n_genes)
    assert mu.shape == (16, 32)
    assert adv_pred.shape == (16, n_drugs)

    # Test perturbation prediction
    pred = model.predict_perturbation(x, drug_ids, doses)
    assert pred.shape == (16, n_genes)
    print("  [PASS] test_simple_cpa")


def test_cpa_baseline():
    """Test CPA baseline end-to-end."""
    if not HAS_TORCH:
        print("  [SKIP] test_cpa_baseline (no PyTorch)")
        return

    np.random.seed(42)
    n_samples, n_genes = 200, 100
    drugs = ["DrugA", "DrugB", "control"]
    drug_labels = np.random.choice(drugs, n_samples)
    doses = np.where(drug_labels == "control", 0.0, np.random.uniform(0.1, 10.0, n_samples))
    expression = np.random.randn(n_samples, n_genes)

    cpa = CPABaseline(latent_dim=32, n_epochs=10, use_scvi=False, seed=42)
    cpa.fit(expression, drug_labels, doses)
    assert cpa._is_fitted

    # Predict expression
    control_expr = expression[drug_labels == "control"][:10]
    if len(control_expr) > 0:
        pred = cpa.predict_expression(control_expr, "DrugA", dose=1.0)
        assert pred.shape == control_expr.shape

    print("  [PASS] test_cpa_baseline")


def test_cpa_evaluation():
    """Test CPA evaluation metrics."""
    if not HAS_TORCH:
        print("  [SKIP] test_cpa_evaluation (no PyTorch)")
        return

    np.random.seed(42)
    n, g = 100, 50
    drugs = ["DrugA", "control"]
    drug_labels = np.random.choice(drugs, n)
    doses = np.where(drug_labels == "control", 0.0, 1.0)
    expression = np.random.randn(n, g)

    cpa = CPABaseline(latent_dim=16, n_epochs=5, use_scvi=False, seed=42)
    cpa.fit(expression, drug_labels, doses)

    control = expression[drug_labels == "control"][:10]
    treated = expression[drug_labels == "DrugA"][:10]
    if len(control) > 0 and len(treated) > 0:
        min_n = min(len(control), len(treated))
        metrics = cpa.evaluate_expression(control[:min_n], treated[:min_n], "DrugA")
        assert "mean_gene_pearson_r" in metrics
        assert "mse" in metrics
    print("  [PASS] test_cpa_evaluation")


def run_tests():
    """Run all CPA tests."""
    print("Running CPA baseline tests...")
    test_simple_cpa()
    test_cpa_baseline()
    test_cpa_evaluation()
    print("All CPA tests passed!")


if __name__ == "__main__":
    run_tests()
