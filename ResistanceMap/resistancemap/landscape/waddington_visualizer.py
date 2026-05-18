"""
resistancemap/landscape/waddington_visualizer.py
================================================
Visualisation helpers for the learned Waddington landscape.

The 2-D plot is produced by:
1. Projecting latents to 2-D using either PCA, UMAP, or the model's first two
   latent dimensions.
2. Evaluating ``U_theta`` on a grid in that projection.
3. Overlaying cells (coloured by drug response / basin assignment).
4. Optionally drawing predicted trajectories.

To keep dependencies light, plotting is only imported lazily (matplotlib).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from resistancemap.landscape.potential import WaddingtonPotential


def landscape_grid(
    potential: WaddingtonPotential,
    *,
    extent: Tuple[float, float] = (-3.0, 3.0),
    n: int = 64,
    proj: Optional[torch.Tensor] = None,
    drug: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(X, Y, U)`` grids suitable for ``plt.contourf``.

    Parameters
    ----------
    proj :
        Optional ``(2, d_latent)`` projection matrix so the grid lives in a
        meaningful 2-D plane. Defaults to the first two latent dims.
    """
    d = potential.d_latent
    if proj is None:
        proj = torch.zeros(2, d)
        proj[0, 0] = 1.0
        proj[1, 1] = 1.0 if d > 1 else 0.0

    lo, hi = extent
    xs = torch.linspace(lo, hi, n)
    ys = torch.linspace(lo, hi, n)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    coords_2d = torch.stack([X.flatten(), Y.flatten()], dim=-1)    # (n*n, 2)
    z = coords_2d @ proj                                            # (n*n, d)
    if drug is not None:
        drug = drug.expand(z.shape[0], -1)
    with torch.no_grad():
        U = potential(z, drug=drug).view(n, n)
    return X, Y, U


def plot_landscape(
    potential: WaddingtonPotential,
    *,
    out_path: str,
    title: str = "Learned Waddington potential",
    cells: Optional[torch.Tensor] = None,
    cell_labels: Optional[Sequence[int]] = None,
    proj: Optional[torch.Tensor] = None,
    extent: Tuple[float, float] = (-3.0, 3.0),
    n: int = 64,
) -> str:
    """Save a 2-D contour plot. Returns the saved path."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("waddington_visualizer.plot_landscape needs matplotlib") from exc

    X, Y, U = landscape_grid(potential, extent=extent, n=n, proj=proj)
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(X.numpy(), Y.numpy(), U.numpy(), levels=20)
    fig.colorbar(cf, ax=ax, label="U_theta")
    if cells is not None:
        c = cell_labels if cell_labels is not None else "k"
        ax.scatter(cells[:, 0].cpu().numpy(), cells[:, 1].cpu().numpy(),
                   c=c, s=8, alpha=0.7)
    centres = potential.attractor_positions()
    if proj is None and potential.d_latent >= 2:
        ax.scatter(centres[:, 0].cpu().numpy(), centres[:, 1].cpu().numpy(),
                   marker="x", c="red", s=80, label="attractors")
    ax.set_xlabel("z_0")
    ax.set_ylabel("z_1")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
