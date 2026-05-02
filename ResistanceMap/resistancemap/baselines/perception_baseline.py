"""
resistancemap/baselines/perception_baseline.py
================================================
Reimplementation of the PERCEPTION framework (Nature Cancer 2024) for
benchmarking against ResistanceMap v6.

PERCEPTION transfers drug response predictions from cancer cell lines to
patients via domain adaptation. Core idea: train ElasticNet models on cell
line gene expression + drug features, then apply to bulk tumor RNA-seq
from patients.

This implementation:
  - Pre-trains on GDSC/CCLE cell line drug response data
  - Computes pathway-level and gene-module scores as features
  - Applies domain adaptation (mean-shift + variance scaling)
  - Evaluates on MMRF patient cohort

Reference: Sinha et al., Nature Cancer, 2024.

Requirements:
    pip install numpy pandas scikit-learn scipy
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pathway / Gene Module Scoring
# ---------------------------------------------------------------------------

# Representative gene modules from MSigDB Hallmark collection
HALLMARK_MODULES = {
    "PI3K_AKT_MTOR": ["PIK3CA", "AKT1", "MTOR", "PTEN", "TSC1", "TSC2",
                       "RPTOR", "RICTOR", "EIF4EBP1", "RPS6KB1"],
    "MTORC1": ["MTOR", "RPTOR", "MLST8", "AKT1S1", "DEPTOR", "EIF4EBP1",
               "RPS6KB1", "ULK1", "ATG13"],
    "MYC_TARGETS": ["MYC", "MAX", "MYCN", "CDK4", "CCND1", "NPM1",
                     "NCL", "DDX18", "NOP56", "BYSL"],
    "P53_PATHWAY": ["TP53", "MDM2", "CDKN1A", "BAX", "BBC3", "PMAIP1",
                     "GADD45A", "SESN1", "TIGAR", "DDB2"],
    "NFKB_SIGNALING": ["NFKB1", "NFKB2", "RELA", "RELB", "REL", "IKBKB",
                        "CHUK", "IKBKG", "TNFAIP3", "BCL3"],
    "WNT_BETA_CATENIN": ["CTNNB1", "APC", "AXIN1", "GSK3B", "DVL1",
                          "FZD1", "LRP5", "LRP6", "WNT3A", "TCF7L2"],
    "APOPTOSIS": ["BCL2", "BAX", "BAK1", "MCL1", "BCL2L1", "BID",
                   "CASP3", "CASP8", "CASP9", "CYCS"],
    "DNA_REPAIR": ["BRCA1", "BRCA2", "RAD51", "ATM", "ATR", "CHEK1",
                    "CHEK2", "PARP1", "MSH2", "MLH1"],
    "PROTEASOME": ["PSMA1", "PSMB5", "PSMB8", "PSMD1", "PSMD4",
                    "PSME1", "PSME2", "USP14", "UCHL5"],
    "UNFOLDED_PROTEIN_RESPONSE": ["XBP1", "ATF6", "ERN1", "EIF2AK3",
                                    "DDIT3", "HSPA5", "CALR", "CANX"],
}


def compute_pathway_scores(
    expression: np.ndarray,
    gene_names: np.ndarray,
    modules: Optional[Dict[str, List[str]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute pathway-level scores via mean z-scored expression.

    For each gene module, computes the mean expression of its member genes
    (after z-scoring) as a single score per sample.

    Parameters
    ----------
    expression : np.ndarray, shape (n_samples, n_genes)
    gene_names : np.ndarray
    modules : dict of module_name -> gene_list

    Returns
    -------
    scores : np.ndarray, shape (n_samples, n_modules)
    module_names : list of str
    """
    if modules is None:
        modules = HALLMARK_MODULES

    gene_set = set(gene_names)
    gene_idx = {g: i for i, g in enumerate(gene_names)}

    # Z-score genes
    scaler = StandardScaler()
    expr_z = scaler.fit_transform(expression)

    scores = []
    module_names = []
    for name, genes in modules.items():
        present = [g for g in genes if g in gene_set]
        if len(present) < 3:
            continue
        indices = [gene_idx[g] for g in present]
        module_score = expr_z[:, indices].mean(axis=1)
        scores.append(module_score)
        module_names.append(name)

    if not scores:
        logger.warning("No pathway modules with sufficient genes")
        return np.zeros((expression.shape[0], 0)), []

    return np.column_stack(scores), module_names


def compute_gene_program_scores(
    expression: np.ndarray,
    n_programs: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Compute data-driven gene program scores via PCA.

    Each PC represents a gene expression program.
    """
    pca = PCA(n_components=min(n_programs, expression.shape[1], expression.shape[0]),
              random_state=seed)
    return pca.fit_transform(expression)


# ---------------------------------------------------------------------------
# Domain Adaptation
# ---------------------------------------------------------------------------

class DomainAdapter:
    """Simple domain adaptation: align cell line distribution to patient.

    Methods:
      - mean_shift: Match mean and variance per gene
      - quantile: Quantile normalization to shared reference
      - coral: CORrelation ALignment (align covariance structures)
    """

    def __init__(self, method: str = "mean_shift"):
        self.method = method
        self._source_mean: Optional[np.ndarray] = None
        self._source_std: Optional[np.ndarray] = None
        self._target_mean: Optional[np.ndarray] = None
        self._target_std: Optional[np.ndarray] = None
        self._transform_matrix: Optional[np.ndarray] = None

    def fit(self, source: np.ndarray, target: np.ndarray) -> "DomainAdapter":
        """Fit domain adaptation parameters.

        Parameters
        ----------
        source : np.ndarray
            Cell line expression (training domain).
        target : np.ndarray
            Patient expression (target domain).
        """
        self._source_mean = source.mean(axis=0)
        self._source_std = source.std(axis=0) + 1e-8
        self._target_mean = target.mean(axis=0)
        self._target_std = target.std(axis=0) + 1e-8

        if self.method == "coral":
            # CORAL: align second-order statistics
            Cs = np.cov(source.T) + np.eye(source.shape[1]) * 1e-4
            Ct = np.cov(target.T) + np.eye(target.shape[1]) * 1e-4
            # Whitening source, then coloring with target covariance
            Us, Ss, _ = np.linalg.svd(Cs)
            Us_inv_sqrt = Us @ np.diag(1.0 / np.sqrt(Ss + 1e-8)) @ Us.T
            Ut, St, _ = np.linalg.svd(Ct)
            Ut_sqrt = Ut @ np.diag(np.sqrt(St + 1e-8)) @ Ut.T
            self._transform_matrix = Us_inv_sqrt @ Ut_sqrt

        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transform source domain data to align with target."""
        if self.method == "mean_shift":
            # Standardize source, then rescale to target distribution
            z = (source - self._source_mean) / self._source_std
            return z * self._target_std + self._target_mean

        elif self.method == "quantile":
            qt = QuantileTransformer(output_distribution="normal", random_state=42)
            return qt.fit_transform(source)

        elif self.method == "coral":
            centered = source - self._source_mean
            aligned = centered @ self._transform_matrix
            return aligned + self._target_mean

        return source


# ---------------------------------------------------------------------------
# PERCEPTION Baseline
# ---------------------------------------------------------------------------

class PERCEPTIONBaseline:
    """PERCEPTION-style transfer learning baseline.

    Training pipeline:
      1. Pre-train on cell line drug response (GDSC/CCLE)
      2. Compute pathway scores + gene program features
      3. Fit ElasticNet per drug on cell line data
      4. Apply domain adaptation to patient features
      5. Predict patient drug response using adapted model

    Parameters
    ----------
    n_pathway_scores : int
        Number of PCA-based gene programs.
    use_pathway_modules : bool
        Whether to include curated pathway scores.
    domain_adaptation : str
        Domain adaptation method: 'mean_shift', 'coral', 'quantile', 'none'.
    alpha_range : tuple
        Range for ElasticNet alpha (regularization).
    l1_ratio_range : tuple
        Range for ElasticNet l1_ratio.
    n_cv_folds : int
        Cross-validation folds for ElasticNet tuning.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_pathway_scores: int = 50,
        use_pathway_modules: bool = True,
        domain_adaptation: str = "mean_shift",
        alpha_range: Tuple[float, ...] = (0.001, 0.01, 0.1, 0.5, 1.0, 5.0),
        l1_ratio_range: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
        n_cv_folds: int = 5,
        seed: int = 42,
    ):
        self.n_pathway_scores = n_pathway_scores
        self.use_pathway_modules = use_pathway_modules
        self.domain_adaptation = domain_adaptation
        self.alpha_range = alpha_range
        self.l1_ratio_range = l1_ratio_range
        self.n_cv_folds = n_cv_folds
        self.seed = seed

        self._models: Dict[str, Any] = {}  # per-drug models
        self._scalers: Dict[str, StandardScaler] = {}
        self._domain_adapters: Dict[str, DomainAdapter] = {}
        self._gene_names: Optional[np.ndarray] = None
        self._is_fitted = False

    def _build_features(
        self, expression: np.ndarray, gene_names: np.ndarray
    ) -> np.ndarray:
        """Build feature matrix from expression data."""
        features = [expression]

        if self.use_pathway_modules:
            pw_scores, _ = compute_pathway_scores(expression, gene_names)
            if pw_scores.shape[1] > 0:
                features.append(pw_scores)

        gp_scores = compute_gene_program_scores(
            expression, n_programs=self.n_pathway_scores, seed=self.seed
        )
        features.append(gp_scores)

        return np.hstack(features)

    def fit(
        self,
        cell_line_expression: np.ndarray,
        cell_line_response: Dict[str, np.ndarray],
        gene_names: np.ndarray,
        patient_expression: Optional[np.ndarray] = None,
    ) -> "PERCEPTIONBaseline":
        """Train per-drug models on cell line data.

        Parameters
        ----------
        cell_line_expression : np.ndarray, shape (n_cell_lines, n_genes)
        cell_line_response : dict of drug_name -> np.ndarray of shape (n_cell_lines,)
            Binary labels: 1=sensitive, 0=resistant.
        gene_names : np.ndarray
        patient_expression : np.ndarray, optional
            Needed for domain adaptation fitting.
        """
        self._gene_names = gene_names
        logger.info(f"PERCEPTION: Training on {cell_line_expression.shape[0]} cell lines, "
                    f"{len(cell_line_response)} drugs")

        # Build features for cell lines
        cl_features = self._build_features(cell_line_expression, gene_names)

        # Fit domain adapter if patient data is provided
        if patient_expression is not None and self.domain_adaptation != "none":
            patient_features = self._build_features(patient_expression, gene_names)

        for drug_name, labels in cell_line_response.items():
            valid = ~np.isnan(labels)
            if valid.sum() < 20:
                logger.debug(f"Skipping {drug_name}: only {valid.sum()} valid samples")
                continue

            X_drug = cl_features[valid]
            y_drug = labels[valid]

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_drug)
            self._scalers[drug_name] = scaler

            # Fit ElasticNet with CV
            try:
                model = ElasticNetCV(
                    l1_ratio=list(self.l1_ratio_range),
                    alphas=list(self.alpha_range),
                    cv=min(self.n_cv_folds, int(valid.sum()) // 2),
                    random_state=self.seed,
                    max_iter=5000,
                    n_jobs=-1,
                )
                model.fit(X_scaled, y_drug)
                self._models[drug_name] = model
                logger.debug(f"  {drug_name}: alpha={model.alpha_:.4f}, "
                            f"l1_ratio={model.l1_ratio_:.2f}, "
                            f"n_nonzero={np.sum(model.coef_ != 0)}")
            except Exception as e:
                logger.warning(f"Failed to fit {drug_name}: {e}")
                continue

            # Domain adapter per drug
            if patient_expression is not None and self.domain_adaptation != "none":
                da = DomainAdapter(method=self.domain_adaptation)
                da.fit(X_scaled, scaler.transform(patient_features))
                self._domain_adapters[drug_name] = da

        self._is_fitted = True
        logger.info(f"PERCEPTION: Fitted models for {len(self._models)} drugs")
        return self

    def predict(
        self,
        patient_expression: np.ndarray,
        drug_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Predict drug response for patients.

        Parameters
        ----------
        patient_expression : np.ndarray, shape (n_patients, n_genes)
        drug_names : list of str, optional
            Drugs to predict. Defaults to all fitted drugs.

        Returns
        -------
        predictions : dict of drug_name -> np.ndarray of predicted probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit first")

        if drug_names is None:
            drug_names = list(self._models.keys())

        features = self._build_features(patient_expression, self._gene_names)

        predictions = {}
        for drug_name in drug_names:
            if drug_name not in self._models:
                continue

            scaler = self._scalers[drug_name]
            X_scaled = scaler.transform(features)

            # Apply domain adaptation
            if drug_name in self._domain_adapters:
                X_scaled = self._domain_adapters[drug_name].transform(X_scaled)

            model = self._models[drug_name]
            y_pred = model.predict(X_scaled)
            # Clip to [0, 1] for probability interpretation
            y_pred = np.clip(y_pred, 0, 1)
            predictions[drug_name] = y_pred

        return predictions

    def evaluate(
        self,
        patient_expression: np.ndarray,
        patient_response: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate predictions against ground truth.

        Returns per-drug metrics: AUROC, AUPRC, Pearson r.
        """
        predictions = self.predict(patient_expression)
        results = {}

        for drug_name, y_true in patient_response.items():
            if drug_name not in predictions:
                continue

            y_pred = predictions[drug_name]
            valid = ~np.isnan(y_true)
            if valid.sum() < 5:
                continue

            yt = y_true[valid]
            yp = y_pred[valid]

            metrics = {}
            try:
                metrics["auroc"] = float(roc_auc_score(yt, yp))
            except ValueError:
                metrics["auroc"] = float("nan")
            try:
                metrics["auprc"] = float(average_precision_score(yt, yp))
            except ValueError:
                metrics["auprc"] = float("nan")
            if len(np.unique(yt)) > 1:
                r, p = stats.pearsonr(yt, yp)
                metrics["pearson_r"] = float(r)
                metrics["pearson_p"] = float(p)
            metrics["n_samples"] = int(valid.sum())
            metrics["prevalence"] = float(yt.mean())

            results[drug_name] = metrics

        # Aggregate across drugs
        if results:
            all_auroc = [m["auroc"] for m in results.values() if not np.isnan(m.get("auroc", np.nan))]
            all_auprc = [m["auprc"] for m in results.values() if not np.isnan(m.get("auprc", np.nan))]
            results["_aggregate"] = {
                "mean_auroc": float(np.mean(all_auroc)) if all_auroc else float("nan"),
                "mean_auprc": float(np.mean(all_auprc)) if all_auprc else float("nan"),
                "n_drugs": len(results) - 1,
            }

        return results

    def get_feature_importance(self, drug_name: str) -> Optional[pd.DataFrame]:
        """Get feature importance for a specific drug model."""
        if drug_name not in self._models:
            return None
        model = self._models[drug_name]
        coefs = model.coef_
        # Map back to feature names
        n_genes = len(self._gene_names)
        feature_names = list(self._gene_names)
        feature_names += [f"pathway_{i}" for i in range(len(coefs) - n_genes - self.n_pathway_scores)]
        feature_names += [f"gp_{i}" for i in range(self.n_pathway_scores)]
        # Truncate or pad
        feature_names = feature_names[:len(coefs)]

        df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs[:len(feature_names)],
            "abs_coefficient": np.abs(coefs[:len(feature_names)]),
        })
        return df.sort_values("abs_coefficient", ascending=False)


# ---------------------------------------------------------------------------
# Convenience: Run full PERCEPTION benchmark
# ---------------------------------------------------------------------------

def run_perception_benchmark(
    cell_line_expr: np.ndarray,
    cell_line_response: Dict[str, np.ndarray],
    patient_expr: np.ndarray,
    patient_response: Dict[str, np.ndarray],
    gene_names: np.ndarray,
    n_seeds: int = 5,
) -> Dict[str, Any]:
    """Run PERCEPTION benchmark across multiple seeds.

    Returns aggregated results with mean and std.
    """
    all_results = []
    for seed in range(n_seeds):
        model = PERCEPTIONBaseline(seed=seed)
        model.fit(cell_line_expr, cell_line_response, gene_names, patient_expr)
        results = model.evaluate(patient_expr, patient_response)
        all_results.append(results)

    # Aggregate
    aggregated = {}
    drug_names = set()
    for r in all_results:
        drug_names.update(r.keys())
    drug_names.discard("_aggregate")

    for drug in drug_names:
        aurocs = [r[drug]["auroc"] for r in all_results if drug in r and not np.isnan(r[drug].get("auroc", np.nan))]
        auprcs = [r[drug]["auprc"] for r in all_results if drug in r and not np.isnan(r[drug].get("auprc", np.nan))]
        aggregated[drug] = {
            "auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
            "auroc_std": float(np.std(aurocs)) if aurocs else float("nan"),
            "auprc_mean": float(np.mean(auprcs)) if auprcs else float("nan"),
            "auprc_std": float(np.std(auprcs)) if auprcs else float("nan"),
        }

    return aggregated


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_pathway_scores():
    """Test pathway score computation."""
    np.random.seed(42)
    expr = np.random.randn(50, 100)
    genes = np.array(["TP53", "KRAS", "AKT1", "MTOR", "PTEN", "BCL2",
                       "MYC", "BRCA1", "PIK3CA", "BAX"] +
                      [f"G{i}" for i in range(90)])
    scores, names = compute_pathway_scores(expr, genes)
    assert scores.shape[0] == 50
    assert len(names) > 0
    print("  [PASS] test_pathway_scores")


def test_domain_adapter():
    """Test domain adaptation methods."""
    np.random.seed(42)
    source = np.random.randn(100, 20) * 2 + 5
    target = np.random.randn(50, 20) * 1 + 0

    for method in ["mean_shift", "coral", "quantile"]:
        da = DomainAdapter(method=method)
        da.fit(source, target)
        adapted = da.transform(source)
        assert adapted.shape == source.shape
        # Adapted source should be closer to target distribution
        if method == "mean_shift":
            assert np.abs(adapted.mean() - target.mean()) < np.abs(source.mean() - target.mean())
    print("  [PASS] test_domain_adapter")


def test_perception_baseline():
    """Test PERCEPTION end-to-end."""
    np.random.seed(42)
    n_cl, n_patients, n_genes = 200, 50, 500
    gene_names = np.array([f"GENE_{i}" for i in range(n_genes)])
    # Override some with real gene names for pathway scoring
    for i, g in enumerate(["TP53", "KRAS", "AKT1", "MTOR", "PTEN", "BCL2", "MYC"]):
        gene_names[i] = g

    cl_expr = np.random.randn(n_cl, n_genes)
    cl_response = {
        "Bortezomib": np.random.choice([0.0, 1.0], n_cl),
        "Lenalidomide": np.random.choice([0.0, 1.0], n_cl),
    }

    pt_expr = np.random.randn(n_patients, n_genes) * 0.5 + 1  # shifted domain
    pt_response = {
        "Bortezomib": np.random.choice([0.0, 1.0], n_patients),
    }

    model = PERCEPTIONBaseline(domain_adaptation="mean_shift", seed=42)
    model.fit(cl_expr, cl_response, gene_names, pt_expr)
    assert model._is_fitted

    predictions = model.predict(pt_expr)
    assert "Bortezomib" in predictions
    assert predictions["Bortezomib"].shape == (n_patients,)
    assert np.all((predictions["Bortezomib"] >= 0) & (predictions["Bortezomib"] <= 1))

    results = model.evaluate(pt_expr, pt_response)
    assert "Bortezomib" in results
    assert "auroc" in results["Bortezomib"]

    importance = model.get_feature_importance("Bortezomib")
    assert importance is not None
    assert len(importance) > 0

    print("  [PASS] test_perception_baseline")


def run_tests():
    """Run all PERCEPTION tests."""
    print("Running PERCEPTION baseline tests...")
    test_pathway_scores()
    test_domain_adapter()
    test_perception_baseline()
    print("All PERCEPTION tests passed!")


if __name__ == "__main__":
    run_tests()
