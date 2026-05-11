"""Per-drug specialized regression head + inverse-variance loss weighting.

Replaces the shared ``Linear(64, n_drugs)`` tail used in v11.5/v12 drug-head
modules with one independent ``Linear(64, 1)`` per drug, plus an inverse-
variance loss weight to prevent HDAC-class gradient dominance.

Motivation (v12 audit + R1/R2 synthesis)
========================================
- Late-fusion (avg of per-modality Ridge) beat the joint cross-attention
  drug-head by 0.008 pooled MSE on test data; the joint head therefore
  does not earn its complexity.
- Per-drug train variance Var(IC50_d) ranges ~0.02 (Venetoclax) to ~10
  (Panobinostat). Under unweighted shared MSE, Panobinostat's per-sample
  gradient is ~500× larger than Venetoclax's, dominating the shared
  trunk via Adam's normalization and dragging well-performing drugs
  toward the per-drug mean.

Design
======
- Shared encoder up through fusion stays unchanged (modality fusion still
  learns pan-drug signal).
- The final ``Linear(64, n_drugs)`` is replaced by ``n_drugs`` independent
  ``Sequential(Dropout, Linear(64, 1))`` blocks.
- Per-drug loss weight ``w_d ∝ 1 / Var(y_d_train)`` clipped to
  ``[0.1, 10.0]`` and normalized to mean 1.0, so HDAC outliers don't
  dominate gradient magnitudes at the shared trunk.

Honest performance expectation
==============================
Predicted aggregate Δ(test MSE) from per-drug heads alone: roughly
``-0.005 to -0.006`` (i.e., closing 60-80% of the 0.008 late-fusion
oracle gap). Per-drug effects are asymmetric: Bortezomib/Venetoclax/
Dinaciclib expected to improve modestly; Panobinostat/Romidepsin
unchanged or marginal (HDAC mechanism gap is a feature problem, not a
head-architecture problem). DO NOT expect this patch alone to break the
predict-mean floor on aggregate MSE.

LOC budget: ~95 (this file) + ~10-15 caller patches in main.py /
landscape/predictor.py.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def variance_weights(
    drug_sensitivity_train: torch.Tensor,
    eps: float = 1e-3,
    clip: tuple[float, float] = (0.1, 10.0),
) -> torch.Tensor:
    """Compute inverse-variance loss weights per drug from the TRAIN split.

    Returns a 1-D tensor of length ``n_drugs`` whose entries are
    ``1 / Var(y_d_train)`` (NaN-aware), clipped to ``clip`` and
    normalized so the weights have mean 1.0.

    Args:
        drug_sensitivity_train: ``(N_train, n_drugs)`` tensor of IC50
            z-scores. NaNs (per-drug missing labels) are ignored.
        eps: Numerical floor on per-drug variance to avoid division by
            zero on near-constant drugs.
        clip: ``(min, max)`` cap on raw inverse-variance weights before
            normalization. Default ``(0.1, 10.0)`` keeps a drug from
            being silently downweighted to near-zero or from dominating.

    Returns:
        ``(n_drugs,)`` float32 tensor of weights, normalized to mean 1.
    """
    n_drugs = drug_sensitivity_train.shape[1]
    inv_vars: list[float] = []
    for d in range(n_drugs):
        col = drug_sensitivity_train[:, d]
        valid = col[~torch.isnan(col)]
        if valid.numel() < 2:
            inv_vars.append(1.0)
            continue
        v = float(valid.var(unbiased=True))
        inv_vars.append(1.0 / max(v, eps))
    w = torch.tensor(inv_vars, dtype=torch.float32)
    w = w.clamp(min=clip[0], max=clip[1])
    # Normalize to mean=1 so per-drug-head reproduces unweighted-MSE scale
    # when variances are equal.
    return w / w.mean()


class PerDrugHead(nn.Module):
    """One independent ``Linear(fusion_dim → 1)`` per drug + variance-weighted loss.

    Drop-in replacement for the legacy
    ``Sequential(Linear(fusion_dim, 64), ReLU, Linear(64, n_drugs))``
    tail. Each drug owns its final projection so gradients from drug ``d``
    cannot update drug ``d'``'s column. The shared encoder remains
    upstream and continues to receive the *sum* of per-drug gradients,
    but each is rescaled by the inverse-variance ``loss_weights``.

    Args:
        fusion_dim: Dimensionality of the incoming fused representation
            (typically ``config.fusion.hidden_dim``, e.g. 128).
        n_drugs: Number of drugs (e.g., 11 for the v12 panel).
        hidden: Shared bottleneck width before the per-drug heads.
        dropout: Per-drug-head dropout probability.
        loss_weights: Optional ``(n_drugs,)`` tensor of inverse-variance
            weights. If ``None``, uniform weights of 1.0 are used (no
            HDAC equalization). Compute via :func:`variance_weights`.
    """

    def __init__(
        self,
        fusion_dim: int,
        n_drugs: int,
        hidden: int = 64,
        dropout: float = 0.1,
        loss_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.n_drugs = n_drugs
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            for _ in range(n_drugs)
        ])
        if loss_weights is None:
            loss_weights = torch.ones(n_drugs)
        # Buffer so it travels with ``.to(device)`` and ``state_dict`` but is
        # not optimized.
        self.register_buffer("loss_weights", loss_weights.float())

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """``(B, fusion_dim) → (B, n_drugs)``."""
        z = self.trunk(fused)
        per_drug = [h(z) for h in self.heads]
        return torch.cat(per_drug, dim=-1)

    def masked_weighted_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Per-drug masked MSE, weighted by ``self.loss_weights``.

        Drugs with high label variance (e.g., Panobinostat) get *down*-
        weighted so their gradient magnitude cannot dominate the shared
        encoder. Drug ``d`` contributes to the loss only on samples where
        ``~isnan(target[:, d])``.

        Args:
            pred: ``(B, n_drugs)`` model predictions.
            target: ``(B, n_drugs)`` IC50 z-scores; NaN where unobserved.

        Returns:
            Scalar weighted-mean per-drug MSE.
        """
        mask = ~torch.isnan(target)
        if not mask.any():
            # Safe zero loss with grad flow preserved.
            return pred.sum() * 0.0
        # Squared error with NaN target masked out so the .detach() trick
        # avoids ``nan * 0 = nan`` issues during autograd.
        safe_target = torch.where(mask, target, pred.detach())
        sq = (pred - safe_target) ** 2
        per_drug_count = mask.sum(dim=0).clamp_min(1)
        per_drug_sum = (sq * mask).sum(dim=0)
        per_drug_mse = per_drug_sum / per_drug_count
        weights = self.loss_weights / self.loss_weights.sum()
        return (per_drug_mse * weights).sum()
