"""
resistancemap/landscape/landscape_regularizers.py
=================================================
Auxiliary losses that shape the learned Waddington potential.

Without these regularisers, the potential MLP residual can absorb arbitrary
non-physical shapes (cusps, ridges) that fit the training trajectories but do
not generalise.

Regularisers
------------
1. **smoothness** — penalise the Hessian Frobenius norm at sample points so
   the energy surface stays C^2.
2. **basin_separation** — penalise overlap between attractor Gaussians (keep
   them resolvable).
3. **drug_barrier** — encourage the drug-conditioned residual to *raise*
   barriers under treatment, not lower them (modelling assumption).
4. **entropy** — encourage diversity of basin assignments in a batch so the
   model does not collapse to "everything is resistant".
"""

from __future__ import annotations

from typing import Optional

import torch

from resistancemap.landscape.potential import WaddingtonPotential


def smoothness_penalty(
    potential: WaddingtonPotential,
    z: torch.Tensor,
    *,
    drug: Optional[torch.Tensor] = None,
    n_samples: int = 1,
    eps: float = 1e-2,
) -> torch.Tensor:
    """Finite-difference Hessian-trace approximation.

    Penalises sum_i |U(z + eps e_i) - 2U(z) + U(z - eps e_i)| / eps^2.
    Cheap and torch-only.
    """
    N, d = z.shape
    z = z.detach()
    U0 = potential(z, drug=drug)
    out = torch.zeros((), device=z.device, dtype=z.dtype)
    for i in range(d):
        e = torch.zeros_like(z)
        e[:, i] = eps
        U_plus = potential(z + e, drug=drug)
        U_minus = potential(z - e, drug=drug)
        out = out + ((U_plus - 2 * U0 + U_minus).abs()).mean() / (eps * eps)
    return out / d


def basin_separation_penalty(
    potential: WaddingtonPotential,
    *,
    min_distance: float = 0.5,
) -> torch.Tensor:
    """Hinge-penalty for attractor centres closer than ``min_distance``."""
    centres = potential.attractor_centres
    K = centres.shape[0]
    if K < 2:
        return torch.zeros((), device=centres.device, dtype=centres.dtype)
    diff = centres.unsqueeze(0) - centres.unsqueeze(1)              # (K, K, d)
    dist = diff.norm(dim=-1)
    mask = torch.ones_like(dist) - torch.eye(K, device=dist.device)
    excess = (min_distance - dist).clamp(min=0.0) * mask
    return excess.sum() / max(K * (K - 1), 1)


def drug_barrier_penalty(
    potential: WaddingtonPotential,
    z_sensitive: torch.Tensor,
    z_resistant: torch.Tensor,
    drug: torch.Tensor,
) -> torch.Tensor:
    """Encourage drug treatment to *raise* the energy of resistant cells.

    Reasoning: the drug pressure should make sensitive cells dip into the
    landscape (drug-killing) while making resistant cells climb out (drug
    selection pressure). A negative value here means the drug is lowering the
    resistant-cell energy, which we penalise as biologically backwards.
    """
    U_sens = potential(z_sensitive, drug=drug)
    U_res = potential(z_resistant, drug=drug)
    return (U_sens - U_res).clamp(min=0.0).mean()


def basin_entropy_penalty(
    basin_logits: torch.Tensor,
    *,
    target_uniform: bool = True,
) -> torch.Tensor:
    """Encourage non-degenerate basin usage across a batch."""
    probs = basin_logits.softmax(dim=-1).mean(dim=0)
    K = probs.shape[-1]
    if target_uniform:
        uniform = torch.full_like(probs, 1.0 / K)
        return torch.nn.functional.kl_div(probs.log(), uniform, reduction="sum")
    # Negative entropy => maximise entropy => minimise -H(p).
    return -(-(probs * probs.clamp_min(1e-8).log()).sum())
