"""
resistancemap/landscape/potential.py
====================================
Learned Waddington potential ``U_theta(z, drug)``.

The potential is a scalar function of the latent state ``z`` (and optionally
of a drug-conditioning vector). The dynamics module evolves cells under the
gradient field ``-grad U`` plus a graph-conditioned correction (see
:mod:`resistancemap.models.dynamics.neural_sde`).

Why a learned potential
-----------------------
A constant or pre-specified potential can encode the right *attractor count*
but not their *positions*. Learning U end-to-end against trajectory + survival
losses lets the basins emerge from data — which matches the empirical
observation that distinct hematologic malignancy patients converge on a small
number of resistance phenotypes (sensitive, drug-tolerant persister, MRD-like,
relapsed-resistant).

Design choices
--------------
* The potential is parameterised as a *sum of mixture-of-Gaussians plus an MLP
  residual*. The mixture gives well-defined basins; the MLP residual lets the
  network represent non-isotropic, anharmonic landscapes.
* The mixture means are learnable parameters initialised on a grid in latent
  space (configured via ``n_attractors``).
* The drug conditioning enters the MLP residual only — the basin geometry
  itself does not depend on drug (otherwise every drug induces a different
  landscape, which is statistically inefficient and not biologically
  motivated). Drug pressure changes the *barriers* between basins (via the
  graph drift), not the basins themselves.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaddingtonPotential(nn.Module):
    """Sum of Gaussian wells + MLP residual.

    Parameters
    ----------
    d_latent : int
        Latent state dimension.
    n_attractors : int
        Number of explicit Gaussian basins.
    drug_dim : int
        Dimension of drug conditioning. 0 disables drug input.
    init_well_spread : float
        Standard deviation of the random init of attractor centres.
    """

    def __init__(
        self,
        d_latent: int,
        *,
        n_attractors: int = 4,
        drug_dim: int = 0,
        hidden: int = 256,
        init_well_spread: float = 1.0,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.n_attractors = n_attractors
        self.drug_dim = drug_dim
        self.residual_scale = residual_scale

        # Attractor centres mu_k in R^d.
        centres = torch.randn(n_attractors, d_latent) * init_well_spread
        self.attractor_centres = nn.Parameter(centres)
        # Log-precision per basin (diagonal Gaussian).
        self.log_precision = nn.Parameter(torch.zeros(n_attractors, d_latent))
        # Depth (negative log-density) per basin.
        self.depth = nn.Parameter(torch.ones(n_attractors))

        # MLP residual for anharmonic terms + drug conditioning.
        in_dim = d_latent + drug_dim
        self.residual = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def _mixture_energy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute -log sum_k depth_k * N(z | mu_k, diag(prec_k^-1))."""
        # z: (..., d_latent)
        prec = F.softplus(self.log_precision) + 1e-3            # (K, d)
        diff = z.unsqueeze(-2) - self.attractor_centres          # (..., K, d)
        log_kernel = -0.5 * (prec * diff.pow(2)).sum(dim=-1)     # (..., K)
        # Add depth as a logit and normalising constant.
        normaliser = 0.5 * prec.log().sum(dim=-1) - 0.5 * self.d_latent * math.log(2 * math.pi)
        log_kernel = log_kernel + normaliser - F.softplus(self.depth)
        # Energy is negative log-density, so we *negate* the LSE.
        return -torch.logsumexp(log_kernel, dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        *,
        drug: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scalar potential per row in ``z``. Shape ``(N,)``."""
        U_mix = self._mixture_energy(z)
        if self.drug_dim > 0:
            if drug is None:
                drug = z.new_zeros(z.shape[0], self.drug_dim)
            inp = torch.cat([z, drug], dim=-1)
        else:
            inp = z
        U_resid = self.residual(inp).squeeze(-1)
        return U_mix + self.residual_scale * U_resid

    @torch.no_grad()
    def attractor_positions(self) -> torch.Tensor:
        """Returns ``(n_attractors, d_latent)`` learned basin centres."""
        return self.attractor_centres.detach()


def potential_gradient(
    potential: WaddingtonPotential,
    z: torch.Tensor,
    drug: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``grad_z U_theta(z, drug)``."""
    z = z.detach().requires_grad_(True)
    U = potential(z, drug=drug).sum()
    grad, = torch.autograd.grad(U, z, create_graph=True)
    return grad
