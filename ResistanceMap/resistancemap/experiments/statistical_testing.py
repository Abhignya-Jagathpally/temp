"""
resistancemap/experiments/statistical_testing.py
==================================================
Rigorous statistical testing for ResistanceMap v6 ICML/ICLR submission.

Provides:
  - DeLong test for AUROC comparison
  - Bootstrap confidence intervals (1000 resamples)
  - McNemar's test for classification agreement
  - Multiple comparison correction (Bonferroni, Benjamini-Hochberg)
  - Effect size (Cohen's d)
  - Power analysis
  - Friedman + Nemenyi for multi-model multi-dataset comparison
  - Results table generation

Requirements:
    pip install numpy pandas scipy scikit-learn
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeLong Test for AUROC Comparison
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks for DeLong test."""
    n = len(x)
    sorted_idx = np.argsort(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and x[sorted_idx[j]] == x[sorted_idx[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based midrank
        for k in range(i, j):
            ranks[sorted_idx[k]] = avg_rank
        i = j
    return ranks


def _auc_variance_components(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute AUC and its variance components for DeLong test."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return float("nan"), np.array([]), np.array([])

    # Compute placement values
    scores = y_score.copy()
    order = np.argsort(scores)
    ranks = np.zeros_like(scores)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)

    # Handle ties
    ranks = _compute_midrank(scores)

    # V statistics
    V_pos = (ranks[pos_idx] - np.arange(1, n_pos + 1).astype(float)) / n_neg
    V_neg = 1 - (ranks[neg_idx] - np.arange(1, n_neg + 1).astype(float)) / n_pos

    auc = np.mean(V_pos)
    return auc, V_pos, V_neg


def delong_test(
    y_true: np.ndarray,
    y_score_1: np.ndarray,
    y_score_2: np.ndarray,
) -> Dict[str, float]:
    """DeLong's test for comparing two AUROC values.

    Tests H0: AUC1 = AUC2 vs H1: AUC1 != AUC2.

    Parameters
    ----------
    y_true : np.ndarray, binary labels
    y_score_1 : np.ndarray, predicted probabilities from model 1
    y_score_2 : np.ndarray, predicted probabilities from model 2

    Returns
    -------
    dict with: auc1, auc2, z_statistic, p_value
    """
    auc1, V10, V01 = _auc_variance_components(y_true, y_score_1)
    auc2, V20, V02 = _auc_variance_components(y_true, y_score_2)

    if np.isnan(auc1) or np.isnan(auc2):
        return {"auc1": auc1, "auc2": auc2, "z_statistic": float("nan"), "p_value": float("nan")}

    n_pos = len(V10)
    n_neg = len(V01)

    # Covariance matrix of (AUC1, AUC2)
    S10 = np.cov(np.column_stack([V10, V20]).T) if n_pos > 1 else np.zeros((2, 2))
    S01 = np.cov(np.column_stack([V01, V02]).T) if n_neg > 1 else np.zeros((2, 2))

    S = S10 / max(n_pos, 1) + S01 / max(n_neg, 1)

    # Variance of AUC1 - AUC2
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var_diff <= 0:
        return {"auc1": auc1, "auc2": auc2, "z_statistic": 0.0, "p_value": 1.0}

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * stats.norm.sf(abs(z))

    return {
        "auc1": float(auc1),
        "auc2": float(auc2),
        "z_statistic": float(z),
        "p_value": float(p_value),
        "auc_diff": float(auc1 - auc2),
    }


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: str = "auroc",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence intervals for a metric.

    Parameters
    ----------
    metric_fn : str
        'auroc', 'auprc', 'f1', 'mcc', or callable.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI).

    Returns
    -------
    dict with: point_estimate, ci_lower, ci_upper, se
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Metric function
    if metric_fn == "auroc":
        fn = lambda yt, yp: roc_auc_score(yt, yp)
    elif metric_fn == "auprc":
        fn = lambda yt, yp: average_precision_score(yt, yp)
    elif callable(metric_fn):
        fn = metric_fn
    else:
        raise ValueError(f"Unknown metric: {metric_fn}")

    # Point estimate
    try:
        point = fn(y_true, y_score)
    except ValueError:
        return {
            "point_estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "se": float("nan"),
        }

    # Bootstrap
    boot_values = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt_boot = y_true[idx]
        yp_boot = y_score[idx]
        # Ensure both classes present
        if len(np.unique(yt_boot)) < 2:
            continue
        try:
            boot_values.append(fn(yt_boot, yp_boot))
        except ValueError:
            continue

    if not boot_values:
        return {
            "point_estimate": float(point),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "se": float("nan"),
        }

    boot_arr = np.array(boot_values)
    alpha = 1 - ci_level
    ci_lower = np.percentile(boot_arr, 100 * alpha / 2)
    ci_upper = np.percentile(boot_arr, 100 * (1 - alpha / 2))
    se = np.std(boot_arr)

    return {
        "point_estimate": float(point),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "se": float(se),
        "n_successful_bootstraps": len(boot_values),
    }


def bootstrap_paired_test(
    y_true: np.ndarray,
    y_score_1: np.ndarray,
    y_score_2: np.ndarray,
    metric_fn: str = "auroc",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap test for paired comparison of two models.

    Returns p-value for H0: metric1 <= metric2.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    if metric_fn == "auroc":
        fn = lambda yt, yp: roc_auc_score(yt, yp)
    elif metric_fn == "auprc":
        fn = lambda yt, yp: average_precision_score(yt, yp)
    else:
        fn = metric_fn

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            m1 = fn(yt, y_score_1[idx])
            m2 = fn(yt, y_score_2[idx])
            diffs.append(m1 - m2)
        except ValueError:
            continue

    if not diffs:
        return {"p_value": float("nan"), "mean_diff": float("nan")}

    diffs = np.array(diffs)
    p_value = (diffs <= 0).mean()

    return {
        "p_value": float(p_value),
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std()),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
    }


# ---------------------------------------------------------------------------
# McNemar's Test
# ---------------------------------------------------------------------------

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
) -> Dict[str, float]:
    """McNemar's test for comparing two classifiers.

    Tests whether the two classifiers have the same error rate.
    """
    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)

    # Contingency table entries
    b = np.sum(correct_1 & ~correct_2)  # model 1 correct, model 2 wrong
    c = np.sum(~correct_1 & correct_2)  # model 1 wrong, model 2 correct

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": int(b), "c": int(c)}

    # McNemar with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "b": int(b),
        "c": int(c),
        "n_discordant": int(b + c),
    }


# ---------------------------------------------------------------------------
# Multiple Comparison Correction
# ---------------------------------------------------------------------------

def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Bonferroni correction for multiple comparisons."""
    m = len(p_values)
    adjusted = np.minimum(p_values * m, 1.0)
    significant = adjusted < alpha

    return {
        "adjusted_p_values": adjusted,
        "significant": significant,
        "n_significant": int(significant.sum()),
        "corrected_alpha": alpha / m,
    }


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Benjamini-Hochberg procedure for FDR control."""
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Adjusted p-values
    adjusted = np.zeros(m)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            adjusted[sorted_idx[i + 1]],
            sorted_p[i] * m / (i + 1)
        )
    adjusted = np.minimum(adjusted, 1.0)
    significant = adjusted < alpha

    return {
        "adjusted_p_values": adjusted,
        "significant": significant,
        "n_significant": int(significant.sum()),
        "fdr_threshold": alpha,
    }


# ---------------------------------------------------------------------------
# Effect Size
# ---------------------------------------------------------------------------

def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Cohen's d effect size.

    Positive d means x1 > x2.
    """
    n1, n2 = len(x1), len(x2)
    s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
    # Pooled standard deviation
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if sp < 1e-10:
        return 0.0
    return float((x1.mean() - x2.mean()) / sp)


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Power Analysis
# ---------------------------------------------------------------------------

def power_analysis(
    n: int,
    effect_size: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Compute statistical power for a two-sample t-test.

    Parameters
    ----------
    n : int
        Sample size per group.
    effect_size : float
        Expected Cohen's d.
    alpha : float
        Significance level.
    alternative : str
        'two-sided' or 'greater'.

    Returns
    -------
    power : float
    """
    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    noncentrality = effect_size * np.sqrt(n / 2)
    power = 1 - stats.norm.cdf(z_alpha - noncentrality)

    return float(power)


def required_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """Compute required sample size per group to achieve target power."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) / max(effect_size, 1e-10)) ** 2
    return int(np.ceil(n))


def detectable_effect_size(
    n: int,
    power: float = 0.8,
    alpha: float = 0.05,
) -> float:
    """Compute minimum detectable effect size given sample size and power."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return float((z_alpha + z_beta) / np.sqrt(n / 2))


# ---------------------------------------------------------------------------
# Friedman + Nemenyi
# ---------------------------------------------------------------------------

def friedman_test(
    metric_matrix: np.ndarray,
) -> Dict[str, float]:
    """Friedman test for comparing >2 models across >2 datasets.

    Parameters
    ----------
    metric_matrix : np.ndarray, shape (n_datasets, n_models)
        Performance metric for each dataset-model pair.

    Returns
    -------
    dict with: statistic, p_value, average_ranks
    """
    n_datasets, n_models = metric_matrix.shape

    # Rank within each dataset (higher is better, so negate for rankdata)
    ranks = np.zeros_like(metric_matrix)
    for i in range(n_datasets):
        ranks[i] = stats.rankdata(-metric_matrix[i])  # rank 1 = best

    avg_ranks = ranks.mean(axis=0)

    # Friedman chi-squared statistic
    chi2 = (12 * n_datasets / (n_models * (n_models + 1))) * \
           (np.sum(avg_ranks ** 2) - n_models * (n_models + 1) ** 2 / 4)
    p_value = 1 - stats.chi2.cdf(chi2, df=n_models - 1)

    return {
        "statistic": float(chi2),
        "p_value": float(p_value),
        "average_ranks": avg_ranks.tolist(),
        "n_datasets": n_datasets,
        "n_models": n_models,
    }


def nemenyi_post_hoc(
    metric_matrix: np.ndarray,
    model_names: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Nemenyi post-hoc test after Friedman.

    Returns pairwise comparison DataFrame with critical differences.
    """
    n_datasets, n_models = metric_matrix.shape

    if model_names is None:
        model_names = [f"Model_{i}" for i in range(n_models)]

    # Rank within each dataset
    ranks = np.zeros_like(metric_matrix)
    for i in range(n_datasets):
        ranks[i] = stats.rankdata(-metric_matrix[i])

    avg_ranks = ranks.mean(axis=0)

    # Critical difference (Nemenyi)
    # q_alpha values for alpha=0.05 (from table)
    # Approximation using studentized range distribution
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha = q_alpha_table.get(n_models, 2.0 + 0.1 * n_models)  # rough approx
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))

    # Pairwise comparisons
    rows = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            rank_diff = abs(avg_ranks[i] - avg_ranks[j])
            significant = rank_diff > cd
            rows.append({
                "model_1": model_names[i],
                "model_2": model_names[j],
                "rank_1": float(avg_ranks[i]),
                "rank_2": float(avg_ranks[j]),
                "rank_diff": float(rank_diff),
                "critical_diff": float(cd),
                "significant": significant,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical Comparison Suite
# ---------------------------------------------------------------------------

class StatisticalComparison:
    """Comprehensive statistical comparison of models.

    Orchestrates all tests and produces publication-ready tables.
    """

    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 1000, seed: int = 42):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.results: Dict[str, Any] = {}

    def compare_two_models(
        self,
        y_true: np.ndarray,
        y_score_1: np.ndarray,
        y_score_2: np.ndarray,
        model_1_name: str = "ResistanceMap",
        model_2_name: str = "Baseline",
    ) -> Dict[str, Any]:
        """Full pairwise comparison of two models."""
        result = {
            "model_1": model_1_name,
            "model_2": model_2_name,
        }

        # DeLong test
        result["delong"] = delong_test(y_true, y_score_1, y_score_2)

        # Bootstrap CIs
        for name, scores in [(model_1_name, y_score_1), (model_2_name, y_score_2)]:
            for metric in ["auroc", "auprc"]:
                ci = bootstrap_ci(y_true, scores, metric, self.n_bootstrap, seed=self.seed)
                result[f"{name}_{metric}"] = ci

        # Bootstrap paired test
        result["bootstrap_paired"] = bootstrap_paired_test(
            y_true, y_score_1, y_score_2, "auroc", self.n_bootstrap, self.seed
        )

        # McNemar's
        y_pred_1 = (y_score_1 >= 0.5).astype(int)
        y_pred_2 = (y_score_2 >= 0.5).astype(int)
        result["mcnemar"] = mcnemar_test(y_true, y_pred_1, y_pred_2)

        self.results[f"{model_1_name}_vs_{model_2_name}"] = result
        return result

    def compare_multiple_models(
        self,
        y_true: np.ndarray,
        model_scores: Dict[str, np.ndarray],
        reference_model: str = "ResistanceMap",
    ) -> pd.DataFrame:
        """Compare multiple models against a reference.

        Returns DataFrame with all pairwise comparisons.
        """
        if reference_model not in model_scores:
            raise ValueError(f"Reference model {reference_model} not in model_scores")

        ref_scores = model_scores[reference_model]
        rows = []
        p_values = []

        for name, scores in model_scores.items():
            if name == reference_model:
                continue

            comparison = self.compare_two_models(
                y_true, ref_scores, scores, reference_model, name
            )

            row = {
                "model": name,
                "auroc": comparison[f"{name}_auroc"]["point_estimate"],
                "auroc_ci": f"[{comparison[f'{name}_auroc']['ci_lower']:.4f}, {comparison[f'{name}_auroc']['ci_upper']:.4f}]",
                "ref_auroc": comparison[f"{reference_model}_auroc"]["point_estimate"],
                "delong_p": comparison["delong"]["p_value"],
                "bootstrap_p": comparison["bootstrap_paired"]["p_value"],
                "mcnemar_p": comparison["mcnemar"]["p_value"],
                "auc_diff": comparison["delong"]["auc_diff"],
            }
            rows.append(row)
            p_values.append(comparison["delong"]["p_value"])

        df = pd.DataFrame(rows)

        # Multiple comparison correction
        if len(p_values) > 0:
            p_arr = np.array(p_values)
            bonf = bonferroni_correction(p_arr, self.alpha)
            bh = benjamini_hochberg(p_arr, self.alpha)
            df["bonferroni_p"] = bonf["adjusted_p_values"]
            df["bh_p"] = bh["adjusted_p_values"]
            df["significant_bonf"] = bonf["significant"]
            df["significant_bh"] = bh["significant"]

        return df.set_index("model")

    def compare_across_datasets(
        self,
        metric_matrix: np.ndarray,
        model_names: List[str],
        dataset_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Friedman + Nemenyi across datasets."""
        friedman = friedman_test(metric_matrix)
        nemenyi = nemenyi_post_hoc(metric_matrix, model_names, self.alpha)

        return {
            "friedman": friedman,
            "nemenyi": nemenyi,
            "model_names": model_names,
            "average_ranks": dict(zip(model_names, friedman["average_ranks"])),
        }

    def power_analysis_report(
        self, n_samples: int, effect_sizes: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """Generate power analysis table."""
        if effect_sizes is None:
            effect_sizes = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

        rows = []
        for d in effect_sizes:
            pwr = power_analysis(n_samples, d, self.alpha)
            n_req = required_sample_size(d, power=0.8, alpha=self.alpha)
            rows.append({
                "effect_size_d": d,
                "interpretation": interpret_cohens_d(d),
                f"power_n={n_samples}": pwr,
                "n_required_80pct_power": n_req,
            })

        mdes = detectable_effect_size(n_samples)
        rows.append({
            "effect_size_d": mdes,
            "interpretation": f"MDES at n={n_samples}",
            f"power_n={n_samples}": 0.8,
            "n_required_80pct_power": n_samples,
        })

        return pd.DataFrame(rows)

    def generate_results_table(self) -> pd.DataFrame:
        """Generate comprehensive results table for the paper."""
        rows = []
        for comparison_name, result in self.results.items():
            m1 = result["model_1"]
            m2 = result["model_2"]
            row = {
                "Comparison": f"{m1} vs {m2}",
                f"{m1} AUROC": f"{result[f'{m1}_auroc']['point_estimate']:.4f} "
                               f"[{result[f'{m1}_auroc']['ci_lower']:.4f}-{result[f'{m1}_auroc']['ci_upper']:.4f}]",
                f"{m2} AUROC": f"{result[f'{m2}_auroc']['point_estimate']:.4f} "
                               f"[{result[f'{m2}_auroc']['ci_lower']:.4f}-{result[f'{m2}_auroc']['ci_upper']:.4f}]",
                "DeLong p": f"{result['delong']['p_value']:.4e}",
                "McNemar p": f"{result['mcnemar']['p_value']:.4e}",
                "AUC Diff": f"{result['delong']['auc_diff']:+.4f}",
            }
            rows.append(row)

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_delong():
    """Test DeLong test."""
    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1], n)
    y_score_1 = y_true * 0.7 + np.random.randn(n) * 0.2
    y_score_2 = y_true * 0.5 + np.random.randn(n) * 0.3

    result = delong_test(y_true, y_score_1, y_score_2)
    assert "auc1" in result
    assert "p_value" in result
    assert result["auc1"] > result["auc2"]  # model 1 should be better
    print(f"  [PASS] test_delong: AUC1={result['auc1']:.4f}, AUC2={result['auc2']:.4f}, p={result['p_value']:.4e}")


def test_bootstrap_ci():
    """Test bootstrap confidence intervals."""
    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1], n)
    y_score = y_true * 0.6 + np.random.randn(n) * 0.3

    ci = bootstrap_ci(y_true, y_score, "auroc", n_bootstrap=500)
    assert ci["ci_lower"] < ci["point_estimate"] < ci["ci_upper"]
    assert ci["se"] > 0
    print(f"  [PASS] test_bootstrap_ci: {ci['point_estimate']:.4f} [{ci['ci_lower']:.4f}-{ci['ci_upper']:.4f}]")


def test_mcnemar():
    """Test McNemar's test."""
    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1], n)
    y_pred_1 = (np.random.randn(n) > 0).astype(int)
    y_pred_2 = (np.random.randn(n) > 0.5).astype(int)

    result = mcnemar_test(y_true, y_pred_1, y_pred_2)
    assert "p_value" in result
    assert result["b"] >= 0 and result["c"] >= 0
    print(f"  [PASS] test_mcnemar: stat={result['statistic']:.4f}, p={result['p_value']:.4e}")


def test_multiple_correction():
    """Test multiple comparison corrections."""
    p_values = np.array([0.001, 0.01, 0.04, 0.06, 0.5])

    bonf = bonferroni_correction(p_values)
    assert bonf["n_significant"] <= len(p_values)
    assert np.all(bonf["adjusted_p_values"] >= p_values)

    bh = benjamini_hochberg(p_values)
    assert bh["n_significant"] >= bonf["n_significant"]  # BH is less conservative
    print(f"  [PASS] test_multiple_correction: Bonf sig={bonf['n_significant']}, BH sig={bh['n_significant']}")


def test_effect_size():
    """Test Cohen's d computation."""
    x1 = np.array([1, 2, 3, 4, 5], dtype=float)
    x2 = np.array([2, 3, 4, 5, 6], dtype=float)
    d = cohens_d(x1, x2)
    # Known: difference of 1, pooled std ~1.58
    assert -1.0 < d < 0
    assert interpret_cohens_d(d) in ("small", "medium")
    print(f"  [PASS] test_effect_size: d={d:.4f} ({interpret_cohens_d(d)})")


def test_power_analysis():
    """Test power analysis."""
    power = power_analysis(n=100, effect_size=0.5)
    assert 0 < power < 1

    n_req = required_sample_size(effect_size=0.5, power=0.8)
    assert n_req > 0

    mdes = detectable_effect_size(n=100)
    assert mdes > 0
    print(f"  [PASS] test_power_analysis: power={power:.4f}, n_req={n_req}, MDES={mdes:.4f}")


def test_friedman_nemenyi():
    """Test Friedman + Nemenyi."""
    np.random.seed(42)
    # 5 datasets x 4 models
    metric_matrix = np.array([
        [0.85, 0.78, 0.72, 0.65],
        [0.88, 0.80, 0.75, 0.68],
        [0.82, 0.76, 0.70, 0.62],
        [0.90, 0.82, 0.78, 0.70],
        [0.86, 0.79, 0.73, 0.66],
    ])
    names = ["ResistanceMap", "PERCEPTION", "MOFA+", "Ridge"]

    friedman = friedman_test(metric_matrix)
    assert friedman["p_value"] < 0.05  # should be significant
    assert len(friedman["average_ranks"]) == 4

    nemenyi = nemenyi_post_hoc(metric_matrix, names)
    assert len(nemenyi) == 6  # C(4,2) pairs
    print(f"  [PASS] test_friedman_nemenyi: Friedman p={friedman['p_value']:.4e}")


def test_full_comparison():
    """Test the full StatisticalComparison suite."""
    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1], n, p=[0.4, 0.6])
    scores = {
        "ResistanceMap": y_true * 0.7 + np.random.randn(n) * 0.2,
        "PERCEPTION": y_true * 0.5 + np.random.randn(n) * 0.3,
        "Ridge": y_true * 0.4 + np.random.randn(n) * 0.35,
    }

    comp = StatisticalComparison(n_bootstrap=200, seed=42)
    table = comp.compare_multiple_models(y_true, scores, "ResistanceMap")
    assert len(table) == 2
    assert "bonferroni_p" in table.columns

    power_df = comp.power_analysis_report(n_samples=200)
    assert len(power_df) > 0

    results_table = comp.generate_results_table()
    assert len(results_table) > 0
    print("  [PASS] test_full_comparison")


def run_tests():
    """Run all statistical testing tests."""
    print("Running statistical testing tests...")
    test_delong()
    test_bootstrap_ci()
    test_mcnemar()
    test_multiple_correction()
    test_effect_size()
    test_power_analysis()
    test_friedman_nemenyi()
    test_full_comparison()
    print("All statistical testing tests passed!")


if __name__ == "__main__":
    run_tests()
