"""
resistancemap/training/mortfm_losses.py
=======================================
Composable loss functions for MORT-FM.

The trainer assembles a weighted sum of these per training stage. Every loss
is a pure function (no module state) so they can be unit-tested independently
of any model. Where a loss requires labels, it explicitly validates them up
front; rows with missing labels are masked out before the loss is taken
(never silently zero-filled).

Inventory
---------
* :func:`reconstruction_loss` — Gaussian recon for any continuous modality.
* :func:`masked_modality_loss` — predicts a held-out modality from the other
  observed modalities.
* :func:`contrastive_alignment_loss` — InfoNCE across modality token pairs.
* :func:`drug_response_loss` — MSE on IC50 / AUC.
* :func:`trajectory_distribution_loss` — Wasserstein-2 + MMD between predicted
  and observed future-state distributions.
* :func:`optimal_transport_trajectory_loss` — Sinkhorn approximation, used
  when ``geomloss`` is available; falls back to MMD otherwise.
* :func:`survival_loss` — Cox or discrete-time (dispatcher).
* :func:`resistance_state_loss` — CE for the state head.
* :func:`pathway_attribution_loss` — BCE + L1 sparsity for the pathway head.
* :func:`perturbation_consistency_loss` — match predicted vs observed
  expression delta under a known CRISPR / drug perturbation.
* :func:`landscape_regularization_loss` — weighted sum of the
  Waddington regularisers (smoothness, basin sep, drug barrier).
* :func:`counterfactual_consistency_loss` — predicted counterfactual rankings
  must agree with held-out perturbation-screen rankings.
* :func:`calibration_loss` — temperature-scaling penalty on the survival head.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from resistancemap.landscape.landscape_regularizers import (
    basin_entropy_penalty,
    basin_separation_penalty,
    drug_barrier_penalty,
    smoothness_penalty,
)
from resistancemap.landscape.potential import WaddingtonPotential
from resistancemap.models.encoders.atac_encoder import atac_bernoulli_nll
from resistancemap.models.encoders.proteomic_encoder import gaussian_nll
from resistancemap.models.encoders.rna_encoder import nb_negative_log_likelihood
from resistancemap.models.heads.time_to_resistance_head import (
    CoxSurvivalHead,
    DiscreteTimeSurvivalHead,
)


# ---------------------------------------------------------------------------
# Reconstruction / pretraining
# ---------------------------------------------------------------------------


def reconstruction_loss(
    targets: torch.Tensor,
    params: Tuple[torch.Tensor, torch.Tensor],
    *,
    distribution: str = "gaussian",
    presence_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Modality-agnostic reconstruction dispatcher.

    ``distribution``: ``"gaussian"`` | ``"nb"`` | ``"bernoulli"``.
    For ``"nb"``, ``params`` is ``(mu, theta)``.
    For ``"bernoulli"``, ``params`` is a single tensor of logits (pass as a
    tuple of length 1 or directly via ``atac_bernoulli_nll``).
    """
    if distribution == "gaussian":
        return gaussian_nll(targets, params, presence_mask=presence_mask)
    if distribution == "nb":
        mu, theta = params
        return nb_negative_log_likelihood(targets, mu, theta, presence_mask=presence_mask)
    if distribution == "bernoulli":
        # params can be (logits,) or just logits — accept both.
        logits = params[0] if isinstance(params, tuple) else params
        return atac_bernoulli_nll(targets, logits, presence_mask=presence_mask)
    raise ValueError(f"Unknown reconstruction distribution {distribution!r}")


def masked_modality_loss(
    predicted_token: torch.Tensor,
    target_token: torch.Tensor,
    *,
    presence_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L2 between fusion's prediction of a masked modality and its true token."""
    diff = (predicted_token - target_token).pow(2)
    if presence_mask is not None:
        diff = diff * presence_mask.float().unsqueeze(-1)
    return diff.sum(dim=-1).mean()


def contrastive_alignment_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric InfoNCE on a pair of modality embeddings.

    ``embeddings_a``, ``embeddings_b`` are ``(N, d)`` and are assumed
    L2-normalised by the caller (apply ``F.normalize`` upstream).
    """
    if embeddings_a.shape != embeddings_b.shape:
        raise ValueError(
            f"Contrastive embeddings must have the same shape; "
            f"got {tuple(embeddings_a.shape)} vs {tuple(embeddings_b.shape)}"
        )
    logits = embeddings_a @ embeddings_b.t() / temperature
    labels = torch.arange(embeddings_a.shape[0], device=embeddings_a.device)
    l_ab = F.cross_entropy(logits, labels)
    l_ba = F.cross_entropy(logits.t(), labels)
    return 0.5 * (l_ab + l_ba)


# ---------------------------------------------------------------------------
# Drug response
# ---------------------------------------------------------------------------


def drug_response_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error, NaN-target rows ignored."""
    finite = torch.isfinite(target)
    if finite.sum() == 0:
        return torch.zeros((), device=predicted.device, dtype=predicted.dtype)
    return F.mse_loss(predicted[finite], target[finite])


# ---------------------------------------------------------------------------
# Trajectory distribution
# ---------------------------------------------------------------------------


def trajectory_distribution_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    metric: str = "mmd",
    sigma: float = 1.0,
) -> torch.Tensor:
    """Distribution-matching loss between predicted and observed states.

    ``metric``:
        ``"mse"`` — point-wise MSE (no distribution aspect, fastest).
        ``"mmd"`` — gaussian-kernel MMD^2.
        ``"sinkhorn"`` — uses ``geomloss`` if available else falls back to MMD.
    """
    if metric == "mse":
        return F.mse_loss(predicted, target)
    if metric == "mmd":
        return _mmd_rbf(predicted, target, sigma=sigma)
    if metric == "sinkhorn":
        try:
            from geomloss import SamplesLoss
            loss = SamplesLoss("sinkhorn", p=2, blur=0.05)
            return loss(predicted, target)
        except ImportError:
            return _mmd_rbf(predicted, target, sigma=sigma)
    raise ValueError(f"Unknown trajectory metric {metric!r}")


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian-kernel maximum mean discrepancy^2."""
    def k(a, b):
        d2 = (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(dim=-1)
        return torch.exp(-d2 / (2 * sigma * sigma))
    return k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()


def optimal_transport_trajectory_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    blur: float = 0.05,
) -> torch.Tensor:
    """Sinkhorn OT — alias for ``trajectory_distribution_loss(metric='sinkhorn')``."""
    return trajectory_distribution_loss(predicted, target, metric="sinkhorn", sigma=blur)


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------


def survival_loss(
    head: nn.Module,
    head_output: torch.Tensor,
    event_time: torch.Tensor,
    event_observed: torch.Tensor,
    *,
    time_bins: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatcher: routes to Cox or discrete-time NLL based on ``head`` type."""
    if isinstance(head, CoxSurvivalHead):
        return head.loss(head_output, event_time, event_observed)
    if isinstance(head, DiscreteTimeSurvivalHead):
        if time_bins is None:
            raise ValueError("Discrete-time survival loss requires time_bins.")
        return head.loss(head_output, event_time, event_observed, time_bins=time_bins)
    # Fallback: assume head_output is a scalar Cox risk.
    return CoxSurvivalHead.loss(head_output, event_time, event_observed)


# ---------------------------------------------------------------------------
# Resistance state
# ---------------------------------------------------------------------------


def resistance_state_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy, NaN targets ignored."""
    if targets.dtype != torch.long:
        targets = targets.long()
    valid = (targets >= 0) & torch.isfinite(targets.float())
    if valid.sum() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits[valid], targets[valid], weight=class_weights)


# ---------------------------------------------------------------------------
# Pathway attribution
# ---------------------------------------------------------------------------


def pathway_attribution_loss(
    protein_scores: torch.Tensor,
    positive_targets: Optional[torch.Tensor],
    *,
    sparsity_weight: float = 1e-3,
) -> torch.Tensor:
    sparsity = sparsity_weight * protein_scores.abs().sum(dim=-1).mean()
    if positive_targets is None:
        return sparsity
    bce = F.binary_cross_entropy(
        protein_scores.clamp(1e-6, 1 - 1e-6), positive_targets.float()
    )
    return bce + sparsity


# ---------------------------------------------------------------------------
# Perturbation consistency
# ---------------------------------------------------------------------------


def perturbation_consistency_loss(
    predicted_delta: torch.Tensor,
    observed_delta: torch.Tensor,
) -> torch.Tensor:
    """Cosine-similarity-style penalty on predicted vs observed delta vectors."""
    pred = F.normalize(predicted_delta, dim=-1)
    obs = F.normalize(observed_delta, dim=-1)
    return 1.0 - (pred * obs).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Landscape regularisation
# ---------------------------------------------------------------------------


def landscape_regularization_loss(
    potential: WaddingtonPotential,
    z_batch: torch.Tensor,
    *,
    drug: Optional[torch.Tensor] = None,
    smoothness_weight: float = 1e-3,
    basin_sep_weight: float = 1.0,
    basin_min_distance: float = 0.5,
    drug_barrier_weight: float = 0.0,
    z_sensitive: Optional[torch.Tensor] = None,
    z_resistant: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = torch.zeros((), device=z_batch.device, dtype=z_batch.dtype)
    if smoothness_weight > 0:
        out = out + smoothness_weight * smoothness_penalty(potential, z_batch, drug=drug)
    if basin_sep_weight > 0:
        out = out + basin_sep_weight * basin_separation_penalty(potential, min_distance=basin_min_distance)
    if drug_barrier_weight > 0 and z_sensitive is not None and z_resistant is not None and drug is not None:
        out = out + drug_barrier_weight * drug_barrier_penalty(potential, z_sensitive, z_resistant, drug)
    return out


# ---------------------------------------------------------------------------
# Counterfactual consistency
# ---------------------------------------------------------------------------


def counterfactual_consistency_loss(
    predicted_rank_scores: torch.Tensor,
    oracle_rank_scores: torch.Tensor,
) -> torch.Tensor:
    """Spearman-rank-style proxy: pairwise margin loss on ranking.

    For each pair (i, j), if oracle says i > j then predicted score(i) should
    > predicted score(j). The loss is the hinge margin violated by the model.
    """
    N = predicted_rank_scores.shape[0]
    if N < 2:
        return torch.zeros((), device=predicted_rank_scores.device, dtype=predicted_rank_scores.dtype)
    diff_pred = predicted_rank_scores.unsqueeze(0) - predicted_rank_scores.unsqueeze(1)
    diff_oracle = oracle_rank_scores.unsqueeze(0) - oracle_rank_scores.unsqueeze(1)
    sign_oracle = diff_oracle.sign()
    margin = (1.0 - sign_oracle * diff_pred).clamp(min=0.0)
    return margin.mean()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_loss(
    predicted_probability: torch.Tensor,
    observed: torch.Tensor,
    *,
    n_bins: int = 10,
) -> torch.Tensor:
    """Expected Calibration Error as a differentiable surrogate (mean abs gap)."""
    bins = torch.linspace(0, 1, n_bins + 1, device=predicted_probability.device)
    out = torch.zeros((), device=predicted_probability.device, dtype=predicted_probability.dtype)
    for i in range(n_bins):
        in_bin = (predicted_probability >= bins[i]) & (predicted_probability < bins[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_acc = observed[in_bin].float().mean()
        bin_conf = predicted_probability[in_bin].mean()
        out = out + (bin_acc - bin_conf).abs() * in_bin.float().mean()
    return out


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def assemble_total_loss(
    components: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Multiply components by weights, sum, and return (total, per-component float dict).

    Missing keys in ``weights`` default to 0 (component disabled).
    Missing keys in ``components`` are skipped silently.
    """
    total = torch.zeros((), device=next(iter(components.values())).device)
    breakdown: Dict[str, float] = {}
    for name, val in components.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        total = total + w * val
        breakdown[name] = float(val.detach())
    breakdown["total"] = float(total.detach())
    return total, breakdown
