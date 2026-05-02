"""
resistancemap/baselines/classical_baselines.py
================================================
Comprehensive classical baseline panel for fair comparison with
ResistanceMap v6.

Includes:
  - Ridge regression
  - Random Forest with feature importance
  - ElasticNet with built-in feature selection
  - Logistic regression (L1 and L2)
  - XGBoost with SHAP explanations
  - SVM (RBF kernel)
  - k-Nearest Neighbors
  - Trivial baselines (per-drug mean, global mean)

All baselines share the same preprocessing, features, and CV splits.

Requirements:
    pip install numpy pandas scikit-learn scipy
    Optional: pip install xgboost shap
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import (
    Ridge, RidgeClassifier, LogisticRegression, ElasticNet, ElasticNetCV,
    SGDClassifier,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score, brier_score_loss, log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# Optional imports
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ---------------------------------------------------------------------------
# Evaluation Utilities
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {}
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auroc"] = float("nan")
    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auprc"] = float("nan")
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    return metrics


# ---------------------------------------------------------------------------
# Individual Baseline Implementations
# ---------------------------------------------------------------------------

class BaseBaseline:
    """Base class for all classical baselines."""

    name: str = "base"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._model = None
        self._scaler = StandardScaler()
        self._imputer = SimpleImputer(strategy="median")
        self._is_fitted = False
        self._fit_time: float = 0.0

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Standardize and impute missing values."""
        X = np.asarray(X, dtype=np.float64)
        if fit:
            X = self._imputer.fit_transform(X)
            X = self._scaler.fit_transform(X)
        else:
            X = self._imputer.transform(X)
            X = self._scaler.transform(X)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseBaseline":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_prob = self.predict_proba(X)
        metrics = compute_metrics(y, y_prob)
        metrics["fit_time_sec"] = self._fit_time
        return metrics

    def get_feature_importance(self) -> Optional[np.ndarray]:
        return None


class RidgeBaseline(BaseBaseline):
    """Ridge regression (L2 regularized linear model) for drug response."""

    name = "Ridge"

    def __init__(self, alpha: float = 1.0, seed: int = 42):
        super().__init__(seed)
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        t0 = time.time()
        Xp = self._preprocess(X, fit=True)
        self._model = LogisticRegression(
            C=1.0 / max(self.alpha, 1e-8), penalty="l2", solver="lbfgs",
            max_iter=2000, random_state=self.seed
        )
        self._model.fit(Xp, y.astype(int))
        self._fit_time = time.time() - t0
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xp = self._preprocess(X)
        return self._model.predict_proba(Xp)[:, 1]

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if self._model is not None:
            return np.abs(self._model.coef_[0])
        return None


class RandomForestBaseline(BaseBaseline):
    """Random Forest with feature importance extraction."""

    name = "RandomForest"

    def __init__(self, n_estimators: int = 500, max_depth: Optional[int] = None, seed: int = 42):
        super().__init__(seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        t0 = time.time()
        Xp = self._preprocess(X, fit=True)
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=5, class_weight="balanced",
            random_state=self.seed, n_jobs=-1,
        )
        self._model.fit(Xp, y.astype(int))
        self._fit_time = time.time() - t0
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xp = self._preprocess(X)
        return self._model.predict_proba(Xp)[:, 1]

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if self._model is not None:
            return self._model.feature_importances_
        return None


class ElasticNetBaseline(BaseBaseline):
    """ElasticNet with automatic alpha/l1_ratio selection via CV."""

    name = "ElasticNet"

    def __init__(self, seed: int = 42):
        super().__init__(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetBaseline":
        t0 = time.time()
        Xp = self._preprocess(X, fit=True)
        self._model = SGDClassifier(
            loss="log_loss", penalty="elasticnet",
            l1_ratio=0.5, alpha=1e-4, max_iter=2000,
            class_weight="balanced", random_state=self.seed,
        )
        self._model.fit(Xp, y.astype(int))
        self._fit_time = time.time() - t0
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xp = self._preprocess(X)
        # SGDClassifier with log_loss has decision_function
        decision = self._model.decision_function(Xp)
        # Convert to probability via sigmoid
        prob = 1.0 / (1.0 + np.exp(-decision))
        return prob

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if self._model is not None:
            return np.abs(self._model.coef_[0])
        return None


class LogisticL1Baseline(BaseBaseline):
    """Logistic regression with L1 (Lasso) penalty -- sparse feature selection."""

    name = "LogisticL1"

    def __init__(self, C: float = 1.0, seed: int = 42):
        super().__init__(seed)
        self.C = C

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticL1Baseline":
        t0 = time.time()
        Xp = self._preprocess(X, fit=True)
        self._model = LogisticRegression(
            C=self.C, penalty="l1", solver="saga",
            max_iter=3000, class_weight="balanced", random_state=self.seed,
        )
        self._model.fit(Xp, y.astype(int))
        self._fit_time = time.time() - t0
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xp = self._preprocess(X)
        return self._model.predict_proba(Xp)[:, 1]

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if self._model is not None:
            return np.abs(self._model.coef_[0])
        return None


class LogisticL2Baseline(BaseBaseline):
    """Logistic regression with L2 (Ridge) penalty."""

    name = "LogisticL2"

    def __init__(self, C: float = 1.0, seed: int = 42):
        super().__init__(seed)
        self.C = C

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticL2Baseline":
        t0 = time.time()
        Xp = self._preprocess(X, fit=True)
        self._model = LogisticRegression(
            C=self.C, penalty="l2", solver="lbfgs",
            max_iter=2000, class_weight="balanced", random_state=self.seed,
        )
        self._model.fit(Xp, y.astype(int))
        self._fit_time = time.time() - t0
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xp = self._preprocess(X)
        return self._model.predict_proba(Xp)[:, 1]


class XGBoostBaseline(BaseBaseline):
    """XGBoost gradient boosted trees with SHAP explanations."""

    name = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        seed: int = 42,
    ):
        super().__init__(seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        if not HAS_XGB:
            raise ImportError("xgboost required: pip install xgboost")
        t0 = time.time()
        Xp = self._preprocess(X, fit=True)
        scale_pos = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=self.seed,
            n_jobs=-1,
        )
        self._model.fit(Xp, y.astype(int))
        self._fit_time = time.time() - t0
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xp = self._preprocess(X)
        return self._model.predict_proba(Xp)[:, 1]

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if self._model is not None:
            return self._model.feature_importances_
        return None

    def get_shap_values(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Compute SHAP values for explainability."""
        if not HAS_SHAP or self._model is None:
            return None
        Xp = self._preprocess(X)
        explainer = shap.TreeExplainer(self._model)
        return explainer.shap_values(Xp)


class SVMBaseline(BaseBaseline):
    """SVM with RBF kernel."""

    name = "SVM_RBF"

    def __init__(self, C: float = 1.0, gamma: str = "scale", seed: int = 42):
        super().__init__(seed)
        self.C = C
        self.gamma = gamma

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMBaseline":
        t0 = time.time()
        Xp = self._preprocess(X, fit=True)
        self._model = SVC(
            C=self.C, kernel="rbf", gamma=self.gamma,
            probability=True, class_weight="balanced",
            random_state=self.seed,
        )
        self._model.fit(Xp, y.astype(int))
        self._fit_time = time.time() - t0
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xp = self._preprocess(X)
        return self._model.predict_proba(Xp)[:, 1]


class KNNBaseline(BaseBaseline):
    """k-Nearest Neighbors classifier."""

    name = "KNN"

    def __init__(self, n_neighbors: int = 15, seed: int = 42):
        super().__init__(seed)
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNBaseline":
        t0 = time.time()
        Xp = self._preprocess(X, fit=True)
        self._model = KNeighborsClassifier(
            n_neighbors=min(self.n_neighbors, len(y) - 1),
            weights="distance", n_jobs=-1,
        )
        self._model.fit(Xp, y.astype(int))
        self._fit_time = time.time() - t0
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xp = self._preprocess(X)
        return self._model.predict_proba(Xp)[:, 1]


class TrivialMeanBaseline(BaseBaseline):
    """Trivial baseline: predict the training set prevalence for all samples."""

    name = "GlobalMean"

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._mean_prob = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrivialMeanBaseline":
        t0 = time.time()
        self._mean_prob = float(y.mean())
        self._fit_time = time.time() - t0
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._mean_prob)


class PerDrugMeanBaseline(BaseBaseline):
    """Trivial baseline: predict per-drug mean response rate.

    For single-drug evaluation, this collapses to TrivialMeanBaseline.
    For multi-drug, each drug gets its own prevalence estimate.
    """

    name = "PerDrugMean"

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._drug_means: Dict[str, float] = {}
        self._global_mean = 0.5

    def fit_multi_drug(
        self, drug_labels: np.ndarray, response_labels: np.ndarray
    ) -> "PerDrugMeanBaseline":
        """Fit per-drug prevalence estimates."""
        self._global_mean = float(response_labels.mean())
        for drug in np.unique(drug_labels):
            mask = drug_labels == drug
            self._drug_means[drug] = float(response_labels[mask].mean())
        self._is_fitted = True
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerDrugMeanBaseline":
        self._global_mean = float(y.mean())
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._global_mean)

    def predict_drug(self, drug_name: str, n_samples: int) -> np.ndarray:
        prob = self._drug_means.get(drug_name, self._global_mean)
        return np.full(n_samples, prob)


# ---------------------------------------------------------------------------
# Suite Runner
# ---------------------------------------------------------------------------

class SimpleBaselinesSuite:
    """Run all classical baselines with the same data and splits.

    Ensures fair comparison by using identical features, preprocessing,
    and evaluation protocol for every baseline.
    """

    def __init__(self, seed: int = 42, include_xgb: bool = True):
        self.seed = seed
        self.baselines: Dict[str, BaseBaseline] = {
            "Ridge": RidgeBaseline(seed=seed),
            "RandomForest": RandomForestBaseline(seed=seed),
            "ElasticNet": ElasticNetBaseline(seed=seed),
            "LogisticL1": LogisticL1Baseline(seed=seed),
            "LogisticL2": LogisticL2Baseline(seed=seed),
            "SVM_RBF": SVMBaseline(seed=seed),
            "KNN": KNNBaseline(seed=seed),
            "GlobalMean": TrivialMeanBaseline(seed=seed),
        }
        if include_xgb and HAS_XGB:
            self.baselines["XGBoost"] = XGBoostBaseline(seed=seed)
        elif include_xgb:
            logger.info("XGBoost not available, skipping")

    def run_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Fit and evaluate all baselines.

        Returns dict of baseline_name -> metrics dict.
        """
        results = {}
        for name, baseline in self.baselines.items():
            logger.info(f"Running {name}...")
            try:
                baseline.fit(X_train, y_train)
                metrics = baseline.evaluate(X_test, y_test)
                results[name] = metrics
                logger.info(f"  {name}: AUROC={metrics.get('auroc', 'nan'):.4f}, "
                           f"AUPRC={metrics.get('auprc', 'nan'):.4f}")
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")
                results[name] = {"error": str(e)}
        return results

    def run_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all baselines with cross-validation.

        Returns dict of baseline_name -> {metric: [fold_values]}.
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        results = {name: defaultdict(list) for name in self.baselines}

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for name, baseline in self.baselines.items():
                try:
                    # Create fresh instance for each fold
                    bl = type(baseline)(seed=self.seed + fold)
                    bl.fit(X_train, y_train)
                    metrics = bl.evaluate(X_test, y_test)
                    for k, v in metrics.items():
                        results[name][k].append(v)
                except Exception as e:
                    logger.warning(f"  {name} fold {fold} failed: {e}")

        # Summarize
        summary = {}
        for name, fold_results in results.items():
            summary[name] = {}
            for metric, values in fold_results.items():
                if values and isinstance(values[0], (int, float)):
                    values_clean = [v for v in values if not np.isnan(v)]
                    if values_clean:
                        summary[name][f"{metric}_mean"] = float(np.mean(values_clean))
                        summary[name][f"{metric}_std"] = float(np.std(values_clean))
        return summary

    def get_feature_importances(
        self, feature_names: Optional[np.ndarray] = None
    ) -> Dict[str, pd.DataFrame]:
        """Get feature importances from all baselines that support it."""
        importances = {}
        for name, baseline in self.baselines.items():
            fi = baseline.get_feature_importance()
            if fi is not None:
                if feature_names is not None and len(feature_names) == len(fi):
                    df = pd.DataFrame({"feature": feature_names, "importance": fi})
                else:
                    df = pd.DataFrame({"feature": range(len(fi)), "importance": fi})
                importances[name] = df.sort_values("importance", ascending=False)
        return importances

    def results_table(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Convert results dict to a formatted DataFrame."""
        rows = []
        for name, metrics in results.items():
            row = {"model": name}
            row.update(metrics)
            rows.append(row)
        df = pd.DataFrame(rows).set_index("model")
        return df.round(4)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_individual_baselines():
    """Test each baseline individually."""
    np.random.seed(42)
    n_train, n_test, p = 200, 50, 100
    X_train = np.random.randn(n_train, p)
    y_train = np.random.choice([0, 1], n_train, p=[0.4, 0.6])
    X_test = np.random.randn(n_test, p)
    y_test = np.random.choice([0, 1], n_test, p=[0.4, 0.6])

    baselines = [
        RidgeBaseline(), RandomForestBaseline(n_estimators=50),
        ElasticNetBaseline(), LogisticL1Baseline(), LogisticL2Baseline(),
        SVMBaseline(), KNNBaseline(), TrivialMeanBaseline(),
    ]
    if HAS_XGB:
        baselines.append(XGBoostBaseline(n_estimators=50))

    for bl in baselines:
        bl.fit(X_train, y_train)
        probs = bl.predict_proba(X_test)
        assert probs.shape == (n_test,), f"{bl.name}: wrong shape {probs.shape}"
        assert np.all((probs >= 0) & (probs <= 1)), f"{bl.name}: probs out of range"
        metrics = bl.evaluate(X_test, y_test)
        assert "auroc" in metrics, f"{bl.name}: missing auroc"
        print(f"  [PASS] {bl.name}: AUROC={metrics['auroc']:.4f}")


def test_suite_runner():
    """Test the full baseline suite."""
    np.random.seed(42)
    n, p = 200, 50
    X = np.random.randn(n, p)
    y = np.random.choice([0, 1], n, p=[0.4, 0.6])

    suite = SimpleBaselinesSuite(seed=42, include_xgb=HAS_XGB)
    results = suite.run_cv(X, y, n_folds=3)
    assert len(results) > 0
    for name, metrics in results.items():
        if "auroc_mean" in metrics:
            assert 0 <= metrics["auroc_mean"] <= 1
    print("  [PASS] test_suite_runner")


def test_feature_importances():
    """Test feature importance extraction."""
    np.random.seed(42)
    n, p = 100, 20
    X = np.random.randn(n, p)
    y = np.random.choice([0, 1], n)
    names = np.array([f"gene_{i}" for i in range(p)])

    suite = SimpleBaselinesSuite(seed=42)
    suite.run_all(X, y, X, y)
    importances = suite.get_feature_importances(names)
    assert len(importances) > 0
    for name, df in importances.items():
        assert "feature" in df.columns
        assert "importance" in df.columns
    print("  [PASS] test_feature_importances")


def run_tests():
    """Run all classical baseline tests."""
    print("Running classical baseline tests...")
    test_individual_baselines()
    test_suite_runner()
    test_feature_importances()
    print("All classical baseline tests passed!")


if __name__ == "__main__":
    from collections import defaultdict
    run_tests()
