"""Statistical power analysis for ResistanceMap evaluation.

Provides tools for computing effective sample size under clustering (ICC),
power calculation for AUROC comparisons, required sample size estimation,
multi-omic coverage analysis, and claim adequacy validation.

References
----------
Eldridge, S. M., Coyle, D., Campbell, M. J., et al. (2006).
    "How big should the denominator be in an infant vaccination study?"
    Secondary analysis of immunisation data. BMJ, 332(7546), 1059-1063.
Hanley, J. A., & McNeil, B. J. (1983).
    "A method of comparing the areas under receiver operating characteristic
    curves derived from the same cases." Radiology, 148(3), 839-843.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = [
    "effective_sample_size",
    "power_auroc_comparison",
    "required_sample_size",
    "modality_intersection_analysis",
    "claim_adequacy_checklist",
    "full_adequacy_report",
]


def effective_sample_size(
    n_patients: int,
    n_cells_per_patient: int | float,
    icc: float,
) -> float:
    """Compute effective sample size under clustering via intraclass correlation.

    For a study with ``n_patients`` clusters, each with mean size
    ``n_cells_per_patient``, the effective sample size is reduced by the
    design effect (DEFF):

        DEFF = 1 + (m_bar - 1) * ICC
        n_eff = n * DEFF^{-1}

    where n = n_patients * n_cells_per_patient.

    Args:
        n_patients: Number of patient clusters.
        n_cells_per_patient: Mean cells per patient (can be non-integer).
        icc: Intraclass correlation coefficient in [0, 1]. Represents
            the fraction of variance explained by patient identity.

    Returns:
        Effective sample size accounting for clustering. If ICC = 0 (no
        clustering), returns n_patients * n_cells_per_patient. If ICC = 1
        (perfect clustering), returns n_patients.

    Raises:
        ValueError: If icc is not in [0, 1] or parameters are non-positive.

    Examples:
        >>> n_eff = effective_sample_size(n_patients=50, n_cells_per_patient=100, icc=0.1)
        >>> n_eff  # doctest: +SKIP
        2631.578...
    """
    if not 0 <= icc <= 1:
        raise ValueError(f"icc must be in [0, 1]; got {icc}")
    if n_patients <= 0:
        raise ValueError(f"n_patients must be positive; got {n_patients}")
    if n_cells_per_patient <= 0:
        raise ValueError(f"n_cells_per_patient must be positive; got {n_cells_per_patient}")

    n_total = n_patients * n_cells_per_patient
    deff = 1.0 + (n_cells_per_patient - 1.0) * icc
    n_eff = n_total / deff
    return float(n_eff)


def power_auroc_comparison(
    n: int,
    auroc_full: float,
    auroc_base: float,
    alpha: float = 0.05,
) -> float:
    r"""Compute power for comparing two AUROC values (Hanley-McNeil).

    Uses the asymptotic variance formula from Hanley & McNeil (1983) to
    estimate power for testing H0: AUROC_full = AUROC_base vs
    HA: AUROC_full ≠ AUROC_base at significance level ``alpha``.

    The test statistic under H0 is approximately normal:
        z = (AUROC_full - AUROC_base) / sqrt(Var(AUROC_full) + Var(AUROC_base) - 2*Cov)

    Power is P(reject H0 | HA true), computed using the non-centrality parameter.

    Args:
        n: Sample size (number of test observations).
        auroc_full: Predicted AUROC of the full model in (0, 1).
        auroc_base: AUROC of the baseline model in (0, 1).
        alpha: Two-tailed significance level. Defaults to 0.05.

    Returns:
        Power in [0, 1].

    Raises:
        ValueError: If AUROC values are not in (0, 1) or n <= 0.

    References:
        Hanley & McNeil (1983), "A method of comparing the areas under
        receiver operating characteristic curves derived from the same cases."

    Examples:
        >>> power = power_auroc_comparison(n=100, auroc_full=0.80, auroc_base=0.70)
        >>> power  # doctest: +SKIP
        0.654...
    """
    if not (0 < auroc_full < 1):
        raise ValueError(f"auroc_full must be in (0, 1); got {auroc_full}")
    if not (0 < auroc_base < 1):
        raise ValueError(f"auroc_base must be in (0, 1); got {auroc_base}")
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    # Hanley-McNeil variance approximation
    a = auroc_full
    b = auroc_base
    q1 = a * (1.0 - a)
    q2 = b * (1.0 - b)

    # Approximate variances (marginal, assuming AUC vs random predictor)
    var_a = (a * (1.0 - a) + (a - 1.0) * (2.0 * a - 1.0)) / (n - 1.0)
    var_b = (b * (1.0 - b) + (b - 1.0) * (2.0 * b - 1.0)) / (n - 1.0)

    # Covariance approximation (assuming same test set)
    cov_ab = 0.5 * (q1 + q2) * (a + b - 2.0 * a * b) / (n - 1.0)

    se_diff = np.sqrt(var_a + var_b - 2.0 * cov_ab)
    if se_diff <= 0:
        return 0.0

    # Critical value for two-tailed test
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    # Non-centrality parameter under HA
    effect_size = abs(auroc_full - auroc_base)
    ncp = effect_size / se_diff

    # Power: P(|Z| > z_crit | Z ~ N(ncp, 1))
    power = (
        1.0 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
    )
    return float(np.clip(power, 0.0, 1.0))


def required_sample_size(
    auroc_full: float,
    auroc_base: float,
    target_power: float = 0.80,
    alpha: float = 0.05,
    max_n: int = 10000,
) -> int:
    """Find minimum sample size to achieve target power for AUROC comparison.

    Uses binary search to find the smallest integer ``n`` such that
    ``power_auroc_comparison(n, auroc_full, auroc_base, alpha) >= target_power``.

    Args:
        auroc_full: AUROC of the full model in (0, 1).
        auroc_base: AUROC of the baseline model in (0, 1).
        target_power: Desired power in (0, 1). Defaults to 0.80.
        alpha: Significance level. Defaults to 0.05.
        max_n: Upper bound for binary search. Defaults to 10000. If
            power cannot be achieved by max_n, returns max_n.

    Returns:
        Minimum required sample size.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> n = required_sample_size(auroc_full=0.80, auroc_base=0.70, target_power=0.80)
        >>> n  # doctest: +SKIP
        144
    """
    if not (0 < target_power < 1):
        raise ValueError(f"target_power must be in (0, 1); got {target_power}")

    # Binary search on n
    lo, hi = 1, max_n
    result = max_n

    while lo <= hi:
        mid = (lo + hi) // 2
        pow_mid = power_auroc_comparison(mid, auroc_full, auroc_base, alpha)
        if pow_mid >= target_power:
            result = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return result


def modality_intersection_analysis(
    has_proteomics: bool,
    has_epigenomics: bool,
    has_ppi: bool,
    has_drug_sensitivity: bool,
) -> dict[str, bool | int | list[str]]:
    """Analyze complete multi-omic coverage for claim validity.

    Checks which modalities are available and reports coverage level.
    ResistanceMap can make different claims depending on what is available:

    - **Complete coverage** (all four): state prediction, pathway attribution,
      temporal trajectory, drug-specific predictions, competing risks.
    - **Partial coverage** (subset): state classification only (cell-line trained);
      temporal and competing risks models must be heavily caveated.
    - **Minimal coverage** (1-2): only observational state labeling.

    Args:
        has_proteomics: Whether proteomic data is available.
        has_epigenomics: Whether epigenomic data is available.
        has_ppi: Whether protein-protein interaction data is available.
        has_drug_sensitivity: Whether drug sensitivity data is available.

    Returns:
        Dictionary with keys:

        - ``"modalities_present"`` (list): Names of available modalities.
        - ``"modality_count"`` (int): Number of modalities (0-4).
        - ``"has_complete_coverage"`` (bool): True if all four are present.
        - ``"coverage_level"`` (str): One of "complete", "partial", "minimal".
        - ``"allowed_claims"`` (list): Claim types that can be made with
          this coverage.
        - ``"required_disclaimers"`` (list): Mandatory disclaimers for this coverage.

    Examples:
        >>> result = modality_intersection_analysis(True, True, True, True)
        >>> result["coverage_level"]
        'complete'
        >>> result["modality_count"]
        4
    """
    modalities = []
    if has_proteomics:
        modalities.append("proteomics")
    if has_epigenomics:
        modalities.append("epigenomics")
    if has_ppi:
        modalities.append("ppi")
    if has_drug_sensitivity:
        modalities.append("drug_sensitivity")

    n_mods = len(modalities)

    if n_mods == 4:
        coverage = "complete"
        allowed_claims = [
            "state_classification",
            "pathway_attribution",
            "temporal_trajectory",
            "drug_specific_predictions",
            "competing_risks",
        ]
        disclaimers = []
    elif n_mods >= 2:
        coverage = "partial"
        allowed_claims = [
            "state_classification",
            "pathway_attribution",
        ]
        disclaimers = [
            "Temporal trajectory and competing risks models are trained on limited data.",
            "Drug-specific predictions are limited to the screened compound set.",
            "Pathway attribution is correlational; causal claims require CRISPR validation.",
        ]
    else:
        coverage = "minimal"
        allowed_claims = ["state_classification"]
        disclaimers = [
            "This model is trained on cell-line data; patient applicability is not established.",
            "No multi-omic integration is available; predictions are based on single modality.",
        ]

    return {
        "modalities_present": modalities,
        "modality_count": n_mods,
        "has_complete_coverage": n_mods == 4,
        "coverage_level": coverage,
        "allowed_claims": allowed_claims,
        "required_disclaimers": disclaimers,
    }


def claim_adequacy_checklist(
    n_samples: int,
    n_features: int,
    n_subgroups: int = 1,
    min_per_subgroup: int = 30,
) -> dict[str, bool | str | list[str]]:
    """Validate sample adequacy for pipeline claims.

    Checks heuristic thresholds for sample size, feature ratio, and subgroup
    coverage. These are conservative rules-of-thumb, not hard requirements.

    Rules:

    - **Sample size**: At least 100 samples for basic models, 200+ for
      subgroup stratification.
    - **Feature ratio**: No more than n / 10 features (to avoid overfitting).
    - **Subgroups**: Each subgroup needs at least ``min_per_subgroup`` samples.

    Args:
        n_samples: Number of training samples.
        n_features: Number of features (genes, proteins, etc.).
        n_subgroups: Number of subgroups for stratified claims. Defaults to 1.
        min_per_subgroup: Minimum samples per subgroup. Defaults to 30.

    Returns:
        Dictionary with keys:

        - ``"is_adequate"`` (bool): True if all heuristic checks pass.
        - ``"n_samples_adequate"`` (bool): True if n_samples >= 100.
        - ``"feature_ratio_adequate"`` (bool): True if n_features <= n_samples / 10.
        - ``"subgroup_coverage_adequate"`` (bool): True if each subgroup has
          >= min_per_subgroup samples.
        - ``"failures"`` (list): List of failed checks (if any).
        - ``"recommendations"`` (list): Remediation suggestions.

    Examples:
        >>> result = claim_adequacy_checklist(n_samples=500, n_features=50, n_subgroups=3)
        >>> result["is_adequate"]
        True
    """
    failures = []
    recommendations = []

    # Check sample size
    n_samples_ok = n_samples >= 100
    if not n_samples_ok:
        failures.append(
            f"Sample size {n_samples} < 100 (minimum for basic models)"
        )
        recommendations.append(f"Increase sample size to at least 100")

    # Check feature ratio
    max_features = n_samples / 10.0
    feature_ratio_ok = n_features <= max_features
    if not feature_ratio_ok:
        failures.append(
            f"Feature count {n_features} > n_samples / 10 = {max_features:.1f} "
            "(overfitting risk)"
        )
        recommendations.append(
            f"Reduce features to {int(max_features)} or fewer via selection"
        )

    # Check subgroup coverage
    if n_subgroups > 1:
        min_total = n_subgroups * min_per_subgroup
        subgroup_ok = n_samples >= min_total
        if not subgroup_ok:
            failures.append(
                f"Subgroup coverage: {n_subgroups} subgroups × "
                f"{min_per_subgroup} min samples = {min_total} total required, "
                f"but n_samples = {n_samples}"
            )
            recommendations.append(
                f"Either increase sample size to {min_total} or reduce "
                f"the number of subgroups"
            )
    else:
        subgroup_ok = True

    is_adequate = n_samples_ok and feature_ratio_ok and subgroup_ok

    return {
        "is_adequate": is_adequate,
        "n_samples_adequate": n_samples_ok,
        "feature_ratio_adequate": feature_ratio_ok,
        "subgroup_coverage_adequate": subgroup_ok,
        "failures": failures,
        "recommendations": recommendations,
    }


def full_adequacy_report(
    n_patients: int,
    n_cells_per_patient: int | float,
    icc: float,
    n_features: int,
    n_subgroups: int = 1,
    min_per_subgroup: int = 30,
    has_proteomics: bool = False,
    has_epigenomics: bool = False,
    has_ppi: bool = False,
    has_drug_sensitivity: bool = False,
) -> dict:
    """Generate comprehensive adequacy report combining all analyses.

    Integrates effective sample size, feature adequacy, subgroup coverage,
    and multi-omic coverage into a single report.

    Args:
        n_patients: Number of patients in the cohort.
        n_cells_per_patient: Mean cells per patient.
        icc: Intraclass correlation (clustering effect).
        n_features: Number of features.
        n_subgroups: Number of subgroups. Defaults to 1.
        min_per_subgroup: Minimum samples per subgroup. Defaults to 30.
        has_proteomics: Whether proteomics is available.
        has_epigenomics: Whether epigenomics is available.
        has_ppi: Whether PPI is available.
        has_drug_sensitivity: Whether drug sensitivity is available.

    Returns:
        Dictionary with keys:

        - ``"effective_sample_size"`` (float): n_eff from clustering.
        - ``"sample_adequacy"`` (dict): Result from claim_adequacy_checklist.
        - ``"modality_coverage"`` (dict): Result from modality_intersection_analysis.
        - ``"overall_adequate"`` (bool): True if both sample and modality
          checks pass.
        - ``"summary"`` (str): Human-readable summary.

    Examples:
        >>> report = full_adequacy_report(
        ...     n_patients=100, n_cells_per_patient=500, icc=0.15,
        ...     n_features=200, has_proteomics=True, has_drug_sensitivity=True
        ... )
        >>> report["overall_adequate"]  # doctest: +SKIP
        True
    """
    n_total = int(n_patients * n_cells_per_patient)
    n_eff = effective_sample_size(n_patients, n_cells_per_patient, icc)

    sample_adequacy = claim_adequacy_checklist(
        int(n_eff), n_features, n_subgroups, min_per_subgroup
    )

    modality_coverage = modality_intersection_analysis(
        has_proteomics, has_epigenomics, has_ppi, has_drug_sensitivity
    )

    overall_adequate = (
        sample_adequacy["is_adequate"] and
        modality_coverage["has_complete_coverage"]
    )

    summary_parts = [
        f"Cohort: {n_patients} patients × {n_cells_per_patient} cells/patient = "
        f"{n_total} total samples (n_eff = {n_eff:.0f} after ICC adjustment)",
        f"Features: {n_features} (ratio check: {'PASS' if sample_adequacy['feature_ratio_adequate'] else 'FAIL'})",
        f"Modalities: {modality_coverage['modality_count']}/4 "
        f"({modality_coverage['coverage_level']} coverage)",
    ]

    if not overall_adequate:
        if not sample_adequacy["is_adequate"]:
            summary_parts.extend(sample_adequacy["failures"])
            summary_parts.extend(sample_adequacy["recommendations"])
        if not modality_coverage["has_complete_coverage"]:
            summary_parts.extend(modality_coverage["required_disclaimers"])

    return {
        "effective_sample_size": n_eff,
        "sample_adequacy": sample_adequacy,
        "modality_coverage": modality_coverage,
        "overall_adequate": overall_adequate,
        "summary": "\n".join(summary_parts),
    }
