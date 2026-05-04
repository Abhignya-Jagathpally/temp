"""Scalar Waddington potential U_θ : ℝ^d → ℝ.

Trained by denoising score matching (Vincent 2011, Neural Computation 23(7))
with a PPI-Laplacian Tikhonov regularizer that ties the spatial gradient of
U to the STRING graph topology (per V10_FOUNDATION_MODEL_PAPER_SPEC.md §2.2 +
§2.3, and docs/WADDINGTON_NEURAL_ODE_REVIEW.md §2.3).

The PPI graph is in gene space (16,201 dim); the latent z lives in ℝ^64.
We bridge the two via a learnable projection P : ℝ^d → ℝ^|V| that maps
∇_z U → gene-gradient-space, where the L_PPI quadratic form is evaluated.
This avoids the impossible bigger-than-RAM gradient-on-genes computation
of an unrolled gene-space ODE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ScalarPotentialConfig:
    latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 256, 128)
    gene_proj_dim: int = 0  # 0 = skip PPI Tikhonov; otherwise = |V|
    activation: str = "silu"  # smooth → smooth gradients


class ScalarPotential(nn.Module):
    """Scalar potential U_θ : ℝ^d → ℝ as smooth MLP.

    Forward returns the scalar value U(z). The gradient ∇_z U(z) is recovered
    via torch.autograd.grad, which is the quantity all downstream losses (DSM,
    Tikhonov, identifiability test F5) actually depend on.
    """

    def __init__(self, cfg: Optional[ScalarPotentialConfig] = None):
        super().__init__()
        cfg = cfg or ScalarPotentialConfig()
        self.cfg = cfg

        act_cls = {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}[cfg.activation]
        layers: list[nn.Module] = []
        prev = cfg.latent_dim
        for h in cfg.hidden_dims:
            layers += [nn.Linear(prev, h), act_cls()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

        if cfg.gene_proj_dim > 0:
            # Linear gene-space projector for PPI Tikhonov regularizer
            self.gene_proj = nn.Linear(cfg.latent_dim, cfg.gene_proj_dim, bias=False)
        else:
            self.register_module("gene_proj", None)

        # Conservative init: U(z=0) ~ 0; small weights for stable DSM
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        if self.gene_proj is not None:
            nn.init.xavier_normal_(self.gene_proj.weight, gain=0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

    def grad_z(self, z: torch.Tensor) -> torch.Tensor:
        """∇_z U_θ(z), shape (B, d)."""
        z_in = z.detach().clone().requires_grad_(True)
        u = self.forward(z_in).sum()
        (g,) = torch.autograd.grad(u, z_in, create_graph=True)
        return g

    def gene_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """Project ∇_z U(z) into gene space for the PPI Tikhonov form.

        Returns shape (B, |V|).
        """
        if self.gene_proj is None:
            raise RuntimeError("gene_proj is disabled; pass gene_proj_dim>0 to config.")
        return self.gene_proj(self.grad_z(z))


def dsm_loss(model: ScalarPotential, z: torch.Tensor, sigma: float) -> torch.Tensor:
    """Vincent 2011 denoising score matching with fixed σ.

    L_DSM = E_{ε ~ N(0,I)} ‖σ ∇_x U(x + σε) + ε‖² where ∇U = -score (we
    parametrize U so that -∇U = score, i.e. data flows downhill on U).
    """
    eps = torch.randn_like(z)
    z_noisy = z + sigma * eps
    grad_u = model.grad_z(z_noisy)  # ≈ -score → score = -grad_u
    # Target for score: -eps/σ. Loss matches σ*score to -eps.
    # Since score ≈ -grad_u, we want σ*(-grad_u) ≈ -eps → σ*grad_u ≈ eps.
    target = eps
    pred = sigma * grad_u
    return ((pred - target) ** 2).sum(dim=-1).mean()


def ppi_tikhonov_loss(
    model: ScalarPotential,
    z: torch.Tensor,
    L_ppi_indices: torch.Tensor,
    L_ppi_values: torch.Tensor,
    n_genes: int,
) -> torch.Tensor:
    """λ ⟨∇U, L_PPI ∇U⟩ where the gradient is gene-projected.

    L_ppi is sparse (COO). We compute  g^T L g  by  (L g)·g.
    """
    g_gene = model.gene_gradient(z)  # (B, |V|)
    # Sparse L matvec on the batch: L is symmetric, so L @ g.T  = (L g)
    L_sparse = torch.sparse_coo_tensor(L_ppi_indices, L_ppi_values, size=(n_genes, n_genes))
    Lg = torch.sparse.mm(L_sparse, g_gene.T).T  # (B, |V|)
    quad = (g_gene * Lg).sum(dim=-1)  # (B,)
    return quad.mean()


__all__ = [
    "ScalarPotential",
    "ScalarPotentialConfig",
    "dsm_loss",
    "ppi_tikhonov_loss",
]
