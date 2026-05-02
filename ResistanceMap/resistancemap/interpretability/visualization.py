"""
Publication-quality visualization suite for ResistanceMap v6.

Generates figures suitable for ICML/ICLR and Nature/Science journals.

Plot types:
1. WaddingtonLandscapePlot: 3D potential landscape with patient trajectories.
2. PathwayCircosPlot: Circos-style pathway interaction diagram.
3. CounterfactualTrajectoryPlot: Factual vs counterfactual trajectory comparison.
4. TemporalAttributionHeatmap: Time x Pathway attribution heatmap.
5. PatientJourneyPlot: Comprehensive single-patient interpretability summary.

All figures use matplotlib with Nature/Science style defaults:
- Font: Arial/Helvetica, 7pt body, 8pt titles
- Single column: 3.5 inches (89 mm)
- Double column: 7.0 inches (178 mm)
- Line width: 0.5pt axes, 1pt data
- DPI: 300

References:
    Waddington, C. H. (1957). The Strategy of the Genes. Allen & Unwin.
    Krzywinski, M. et al. (2009). Circos: An information aesthetic for
        comparative genomics. Genome Research, 19, 1639-1645.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# ---------------------------------------------------------------------------
# Publication defaults
# ---------------------------------------------------------------------------

NATURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.5,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
}

# Color palette: colorblind-friendly (Wong, 2011 Nature Methods)
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "cyan": "#56B4E9",
    "black": "#000000",
    "gray": "#999999",
}

PROGRAM_COLORS = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["green"],
    PALETTE["red"],
    PALETTE["purple"],
    PALETTE["cyan"],
    PALETTE["yellow"],
    PALETTE["gray"],
]


def apply_nature_style():
    """Apply Nature/Science publication style to matplotlib."""
    matplotlib.rcParams.update(NATURE_STYLE)


# ---------------------------------------------------------------------------
# 1. Waddington Landscape Plot
# ---------------------------------------------------------------------------

class WaddingtonLandscapePlot:
    """3D surface plot of the learned potential landscape.

    The Waddington epigenetic landscape is a metaphor for cell fate decisions.
    In ResistanceMap, the landscape is computed from the learned ODE potential:
        V(x) = -integral f(x) dx
    where f is the ODE vector field. Basins correspond to sensitive/resistant
    attractors; saddle points are decision boundaries.

    Following Waddington (1957) and computational reconstructions by
    Wang et al. (2011) "Quantifying the Waddington landscape" (PNAS).
    """

    @staticmethod
    def plot(
        potential_grid: np.ndarray,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        trajectories: Optional[List[np.ndarray]] = None,
        trajectory_labels: Optional[List[str]] = None,
        attractors: Optional[List[Tuple[float, float, str]]] = None,
        saddle_points: Optional[List[Tuple[float, float]]] = None,
        title: str = "Waddington Landscape",
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (3.5, 3.0),
        view_angle: Tuple[float, float] = (25, -60),
    ) -> plt.Figure:
        """Render the 3D Waddington landscape.

        Args:
            potential_grid: (nx, ny) potential values on a 2D grid.
            x_range: (xmin, xmax) for first latent dimension.
            y_range: (ymin, ymax) for second latent dimension.
            trajectories: List of (T, 2) patient trajectories in latent space.
            trajectory_labels: Labels for each trajectory.
            attractors: List of (x, y, label) for attractor positions.
            saddle_points: List of (x, y) for saddle point positions.
            title: Figure title.
            save_path: If set, saves figure.
            figsize: Figure size in inches.
            view_angle: (elevation, azimuth) for 3D view.

        Returns:
            Matplotlib Figure object.
        """
        apply_nature_style()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        nx, ny = potential_grid.shape
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Surface plot with basin coloring
        norm = plt.Normalize(potential_grid.min(), potential_grid.max())
        cmap = plt.cm.RdYlBu_r

        ax.plot_surface(
            X, Y, potential_grid,
            facecolors=cmap(norm(potential_grid)),
            alpha=0.6,
            rstride=2, cstride=2,
            edgecolor="none",
            antialiased=True,
        )

        # Plot trajectories rolling down the landscape
        if trajectories is not None:
            for i, traj in enumerate(trajectories):
                label = trajectory_labels[i] if trajectory_labels else f"Patient {i + 1}"
                color = PROGRAM_COLORS[i % len(PROGRAM_COLORS)]

                # Interpolate potential values for trajectory z-coordinate
                from scipy.interpolate import RegularGridInterpolator
                try:
                    interp = RegularGridInterpolator((x, y), potential_grid)
                    z_traj = interp(traj[:, :2])
                    ax.plot(
                        traj[:, 0], traj[:, 1], z_traj + 0.05,
                        color=color, linewidth=1.2, label=label, zorder=5,
                    )
                    # Start marker
                    ax.scatter(
                        [traj[0, 0]], [traj[0, 1]], [z_traj[0] + 0.05],
                        color=color, s=20, marker="o", zorder=6,
                    )
                    # End marker
                    ax.scatter(
                        [traj[-1, 0]], [traj[-1, 1]], [z_traj[-1] + 0.05],
                        color=color, s=20, marker="^", zorder=6,
                    )
                except (ImportError, ValueError):
                    # If scipy not available, plot without z interpolation
                    z_flat = np.full(len(traj), potential_grid.min())
                    ax.plot(
                        traj[:, 0], traj[:, 1], z_flat,
                        color=color, linewidth=1.2, label=label,
                    )

        # Mark attractors
        if attractors is not None:
            for ax_x, ax_y, alabel in attractors:
                ax.scatter(
                    [ax_x], [ax_y], [potential_grid.min() - 0.1],
                    color=PALETTE["black"], s=40, marker="*", zorder=10,
                )
                ax.text(
                    ax_x, ax_y, potential_grid.min() - 0.2,
                    alabel, fontsize=5, ha="center",
                )

        # Mark saddle points
        if saddle_points is not None:
            for sx, sy in saddle_points:
                ax.scatter(
                    [sx], [sy], [potential_grid.max() * 0.5],
                    color=PALETTE["red"], s=30, marker="x", zorder=10,
                )

        ax.set_xlabel("Program 1", labelpad=1)
        ax.set_ylabel("Program 2", labelpad=1)
        ax.set_zlabel("Potential V(x)", labelpad=1)
        ax.set_title(title, pad=5)
        ax.view_init(*view_angle)

        # Reduce clutter
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")

        if trajectories is not None and len(trajectories) <= 6:
            ax.legend(loc="upper left", fontsize=5)

        if save_path:
            fig.savefig(save_path)
        return fig


# ---------------------------------------------------------------------------
# 2. Pathway Circos Plot
# ---------------------------------------------------------------------------

class PathwayCircosPlot:
    """Circos-style plot of pathway interactions from the learned matrix A.

    Displays pathways as arcs around a circle with chord connections showing
    interactions. Edge width encodes interaction strength; color encodes
    activating (red) vs inhibiting (blue).

    Following Krzywinski et al. (2009) "Circos" (Genome Research).
    """

    @staticmethod
    def plot(
        pathway_names: List[str],
        interaction_matrix: np.ndarray,
        known_mask: Optional[np.ndarray] = None,
        threshold: float = 0.01,
        title: str = "Pathway Interactions",
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (3.5, 3.5),
    ) -> plt.Figure:
        """Render the Circos-style pathway interaction plot.

        Args:
            pathway_names: Names of pathways.
            interaction_matrix: (P, P) interaction weight matrix.
            known_mask: (P, P) boolean mask of known interactions.
                True = known (solid lines), False = novel (dashed).
            threshold: Minimum absolute weight to display.
            title: Figure title.
            save_path: Save path.
            figsize: Figure size.

        Returns:
            Figure object.
        """
        apply_nature_style()

        n = len(pathway_names)
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": False})
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Place pathway nodes around a circle
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = 1.0
        node_x = radius * np.cos(angles)
        node_y = radius * np.sin(angles)

        # Draw pathway arcs (colored segments)
        for i in range(n):
            color = PROGRAM_COLORS[i % len(PROGRAM_COLORS)]
            ax.scatter(node_x[i], node_y[i], s=60, color=color, zorder=5)

            # Label
            angle_deg = np.degrees(angles[i])
            ha = "left" if -90 < angle_deg < 90 or angle_deg > 270 else "right"
            rotation = angle_deg if -90 < angle_deg < 90 else angle_deg - 180
            if angle_deg > 180:
                rotation = angle_deg - 360 if angle_deg > 270 else angle_deg - 180

            label_r = 1.15
            ax.text(
                label_r * np.cos(angles[i]),
                label_r * np.sin(angles[i]),
                pathway_names[i],
                fontsize=5,
                ha=ha,
                va="center",
                rotation=0,
            )

        # Draw interaction chords
        max_weight = np.abs(interaction_matrix).max()
        if max_weight < 1e-10:
            max_weight = 1.0

        for i in range(n):
            for j in range(i + 1, n):
                w = interaction_matrix[i, j]
                if abs(w) < threshold:
                    continue

                # Color: red for activating, blue for inhibiting
                color = PALETTE["red"] if w > 0 else PALETTE["blue"]
                width = 0.5 + 2.0 * abs(w) / max_weight

                # Line style: solid if known, dashed if novel
                ls = "-"
                if known_mask is not None and not known_mask[i, j]:
                    ls = "--"

                # Draw as a Bezier-like curve through the center
                mid_x = 0.3 * (node_x[i] + node_x[j])
                mid_y = 0.3 * (node_y[i] + node_y[j])

                from matplotlib.path import Path
                import matplotlib.patches as patches

                verts = [
                    (node_x[i], node_y[i]),
                    (mid_x, mid_y),
                    (node_x[j], node_y[j]),
                ]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                path = Path(verts, codes)
                patch = patches.PathPatch(
                    path,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=width,
                    linestyle=ls,
                    alpha=0.6,
                )
                ax.add_patch(patch)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=PALETTE["red"], linewidth=1, label="Activating"),
            Line2D([0], [0], color=PALETTE["blue"], linewidth=1, label="Inhibiting"),
        ]
        if known_mask is not None:
            legend_elements.extend([
                Line2D([0], [0], color="gray", linewidth=1, linestyle="-", label="Known"),
                Line2D([0], [0], color="gray", linewidth=1, linestyle="--", label="Novel"),
            ])
        ax.legend(handles=legend_elements, loc="lower center",
                  ncol=2, fontsize=5, bbox_to_anchor=(0.5, -0.05))

        ax.set_title(title, fontsize=8, fontweight="bold", pad=10)

        if save_path:
            fig.savefig(save_path)
        return fig


# ---------------------------------------------------------------------------
# 3. Counterfactual Trajectory Plot
# ---------------------------------------------------------------------------

class CounterfactualTrajectoryPlot:
    """Side-by-side observed vs counterfactual trajectory visualization.

    Highlights the intervention window, marks the critical decision point,
    and shades the region where the intervention changes the outcome.
    """

    @staticmethod
    def plot(
        timepoints: np.ndarray,
        factual: np.ndarray,
        counterfactuals: Dict[str, np.ndarray],
        intervention_window: Optional[Tuple[float, float]] = None,
        outcome_threshold: float = 0.5,
        dim: int = 0,
        title: str = "Counterfactual Trajectories",
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (7.0, 2.5),
    ) -> plt.Figure:
        """Plot factual and counterfactual trajectories.

        Args:
            timepoints: (T,) time values.
            factual: (T,) or (T, D) factual trajectory.
            counterfactuals: Dict mapping label -> (T,) or (T, D) trajectory.
            intervention_window: (t_start, t_end) to shade.
            outcome_threshold: Resistance threshold line.
            dim: Dimension to plot if multi-dimensional.
            title: Title.
            save_path: Save path.
            figsize: Figure size.

        Returns:
            Figure.
        """
        apply_nature_style()

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        # Extract the plotting dimension
        if factual.ndim == 2:
            factual_1d = factual[:, dim]
        else:
            factual_1d = factual

        # Left panel: factual trajectory
        ax = axes[0]
        ax.plot(timepoints, factual_1d, color=PALETTE["black"],
                linewidth=1.2, label="Observed")
        ax.axhline(outcome_threshold, color=PALETTE["gray"],
                    linestyle=":", linewidth=0.5, label="Resistance threshold")

        if intervention_window is not None:
            ax.axvspan(
                intervention_window[0], intervention_window[1],
                alpha=0.1, color=PALETTE["orange"], label="Intervention window",
            )

        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Resistance Score")
        ax.set_title("Observed", fontsize=7, fontweight="bold")
        ax.legend(fontsize=5, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Right panel: counterfactuals
        ax = axes[1]
        ax.plot(timepoints, factual_1d, color=PALETTE["gray"],
                linewidth=0.8, linestyle=":", alpha=0.5, label="Observed")

        for i, (label, cf_traj) in enumerate(counterfactuals.items()):
            if cf_traj.ndim == 2:
                cf_1d = cf_traj[:, dim]
            else:
                cf_1d = cf_traj
            color = PROGRAM_COLORS[i % len(PROGRAM_COLORS)]
            ax.plot(timepoints, cf_1d, color=color, linewidth=1.0, label=label)

            # Shade divergence region
            diff = np.abs(cf_1d - factual_1d)
            significant = diff > 0.05 * np.abs(factual_1d).max()
            ax.fill_between(
                timepoints, factual_1d, cf_1d,
                where=significant, alpha=0.1, color=color,
            )

        ax.axhline(outcome_threshold, color=PALETTE["gray"],
                    linestyle=":", linewidth=0.5)

        if intervention_window is not None:
            ax.axvspan(
                intervention_window[0], intervention_window[1],
                alpha=0.1, color=PALETTE["orange"],
            )

        ax.set_xlabel("Time (months)")
        ax.set_title("Counterfactual", fontsize=7, fontweight="bold")
        ax.legend(fontsize=5, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.suptitle(title, fontsize=8, fontweight="bold", y=1.02)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)
        return fig


# ---------------------------------------------------------------------------
# 4. Temporal Attribution Heatmap
# ---------------------------------------------------------------------------

class TemporalAttributionHeatmap:
    """Time x Pathway heatmap of attribution scores.

    Rows = pathways, Columns = timepoints, Color = attribution magnitude.
    Annotated with clinical events.
    """

    @staticmethod
    def plot(
        timepoints: np.ndarray,
        pathway_names: List[str],
        attributions: np.ndarray,
        clinical_events: Optional[Dict[str, float]] = None,
        cmap: str = "RdBu_r",
        title: str = "Temporal Pathway Attribution",
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (7.0, 3.0),
    ) -> plt.Figure:
        """Render the temporal attribution heatmap.

        Args:
            timepoints: (T,) time values.
            pathway_names: (P,) pathway names for y-axis labels.
            attributions: (P, T) or (T, P) attribution matrix. Will be
                transposed to (P, T) if needed (rows=pathways, cols=time).
            clinical_events: Dict mapping event name -> time.
            cmap: Colormap name.
            title: Title.
            save_path: Save path.
            figsize: Figure size.

        Returns:
            Figure.
        """
        apply_nature_style()

        # Ensure (P, T) orientation
        if attributions.shape[0] == len(timepoints) and attributions.shape[1] == len(pathway_names):
            attributions = attributions.T

        fig, ax = plt.subplots(figsize=figsize)

        vmax = np.abs(attributions).max()
        im = ax.imshow(
            attributions,
            aspect="auto",
            cmap=cmap,
            vmin=-vmax, vmax=vmax,
            interpolation="nearest",
        )

        # Axis labels
        n_time_ticks = min(10, len(timepoints))
        tick_idx = np.linspace(0, len(timepoints) - 1, n_time_ticks, dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([f"{timepoints[i]:.1f}" for i in tick_idx], rotation=45)
        ax.set_xlabel("Time (months)")

        ax.set_yticks(range(len(pathway_names)))
        ax.set_yticklabels(pathway_names, fontsize=5)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Attribution Score", fontsize=6)
        cbar.ax.tick_params(labelsize=5)

        # Clinical event annotations
        if clinical_events is not None:
            for event_name, event_time in clinical_events.items():
                # Find closest timepoint index
                idx = np.argmin(np.abs(timepoints - event_time))
                ax.axvline(idx, color=PALETTE["black"], linewidth=0.5, linestyle="--")
                ax.text(
                    idx, -0.8, event_name, fontsize=4, ha="center",
                    rotation=45, color=PALETTE["black"],
                )

        ax.set_title(title, fontsize=8, fontweight="bold")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)
        return fig


# ---------------------------------------------------------------------------
# 5. Patient Journey Plot
# ---------------------------------------------------------------------------

class PatientJourneyPlot:
    """Comprehensive single-patient interpretability summary.

    Layout:
    - Top: trajectory through latent space (2D projection)
    - Middle: pathway activations over time (stacked area)
    - Bottom: drug intervention timeline
    - Right: counterfactual alternatives (small multiples)
    """

    @staticmethod
    def plot(
        timepoints: np.ndarray,
        trajectory: np.ndarray,
        program_contributions: np.ndarray,
        program_names: List[str],
        drug_history: Optional[List[Dict[str, Any]]] = None,
        counterfactual_outcomes: Optional[Dict[str, float]] = None,
        patient_id: str = "Patient",
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (7.0, 6.0),
    ) -> plt.Figure:
        """Render the patient journey summary.

        Args:
            timepoints: (T,) time values.
            trajectory: (T, D) trajectory through latent space.
            program_contributions: (T, P) fractional program contributions.
            program_names: Names of biological programs.
            drug_history: List of dicts with 'name', 'start', 'end', 'dose'.
            counterfactual_outcomes: Dict mapping scenario -> outcome.
            patient_id: Patient identifier for title.
            save_path: Save path.
            figsize: Figure size.

        Returns:
            Figure.
        """
        apply_nature_style()

        # Layout: 3 rows, 2 columns (right column for counterfactuals)
        has_cf = counterfactual_outcomes is not None and len(counterfactual_outcomes) > 0
        if has_cf:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[3, 1],
                                  hspace=0.35, wspace=0.3)
        else:
            fig = plt.figure(figsize=(figsize[0] * 0.75, figsize[1]))
            gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.35)

        # --- Top: Latent space trajectory ---
        ax_traj = fig.add_subplot(gs[0, 0])

        # Use first two dimensions or PCA if D > 2
        if trajectory.shape[1] >= 2:
            x_plot = trajectory[:, 0]
            y_plot = trajectory[:, 1]
        else:
            x_plot = trajectory[:, 0]
            y_plot = np.zeros_like(x_plot)

        # Color by time
        scatter = ax_traj.scatter(
            x_plot, y_plot, c=timepoints, cmap="viridis",
            s=8, zorder=3, edgecolors="none",
        )
        ax_traj.plot(x_plot, y_plot, color=PALETTE["gray"],
                     linewidth=0.5, alpha=0.5, zorder=2)

        # Mark start and end
        ax_traj.scatter([x_plot[0]], [y_plot[0]], color=PALETTE["green"],
                        s=30, marker="o", zorder=5, label="Start")
        ax_traj.scatter([x_plot[-1]], [y_plot[-1]], color=PALETTE["red"],
                        s=30, marker="^", zorder=5, label="End")

        cbar = fig.colorbar(scatter, ax=ax_traj, fraction=0.03, pad=0.02)
        cbar.set_label("Time", fontsize=5)
        cbar.ax.tick_params(labelsize=4)

        ax_traj.set_xlabel("Latent Dim 1")
        ax_traj.set_ylabel("Latent Dim 2")
        ax_traj.set_title("Trajectory in Latent Space", fontsize=7, fontweight="bold")
        ax_traj.legend(fontsize=5, loc="upper right")
        ax_traj.spines["top"].set_visible(False)
        ax_traj.spines["right"].set_visible(False)

        # --- Middle: Stacked area of program contributions ---
        ax_prog = fig.add_subplot(gs[1, 0])

        n_programs = program_contributions.shape[1]
        colors = [PROGRAM_COLORS[i % len(PROGRAM_COLORS)] for i in range(n_programs)]

        ax_prog.stackplot(
            timepoints, program_contributions.T,
            labels=program_names, colors=colors, alpha=0.8,
        )
        ax_prog.set_xlabel("Time (months)")
        ax_prog.set_ylabel("Contribution")
        ax_prog.set_title("Biological Program Contributions", fontsize=7, fontweight="bold")
        ax_prog.set_ylim(0, 1)
        ax_prog.legend(fontsize=4, loc="upper left", ncol=min(n_programs, 4))
        ax_prog.spines["top"].set_visible(False)
        ax_prog.spines["right"].set_visible(False)

        # --- Bottom: Drug intervention timeline ---
        ax_drug = fig.add_subplot(gs[2, 0])

        if drug_history is not None and len(drug_history) > 0:
            for i, drug in enumerate(drug_history):
                name = drug.get("name", f"Drug {i + 1}")
                t_start = drug.get("start", 0)
                t_end = drug.get("end", timepoints[-1])
                dose = drug.get("dose", 1.0)
                color = PROGRAM_COLORS[i % len(PROGRAM_COLORS)]

                ax_drug.barh(
                    i, t_end - t_start, left=t_start,
                    height=0.6, color=color, alpha=0.7,
                    label=f"{name} ({dose}x)",
                )
                ax_drug.text(
                    (t_start + t_end) / 2, i, name,
                    ha="center", va="center", fontsize=5, color="white",
                    fontweight="bold",
                )

            ax_drug.set_yticks(range(len(drug_history)))
            ax_drug.set_yticklabels([d.get("name", "") for d in drug_history], fontsize=5)
        else:
            ax_drug.text(
                0.5, 0.5, "No treatment recorded",
                transform=ax_drug.transAxes, ha="center", va="center",
                fontsize=6, color=PALETTE["gray"],
            )

        ax_drug.set_xlabel("Time (months)")
        ax_drug.set_title("Drug Intervention History", fontsize=7, fontweight="bold")
        ax_drug.spines["top"].set_visible(False)
        ax_drug.spines["right"].set_visible(False)

        # --- Right column: Counterfactual outcomes ---
        if has_cf:
            ax_cf = fig.add_subplot(gs[:, 1])

            names = list(counterfactual_outcomes.keys())
            outcomes = [counterfactual_outcomes[n] for n in names]
            y_pos = range(len(names))
            colors_cf = [
                PALETTE["green"] if o < 0.5 else PALETTE["red"]
                for o in outcomes
            ]

            ax_cf.barh(y_pos, outcomes, color=colors_cf, height=0.6, alpha=0.8)
            ax_cf.set_yticks(y_pos)
            ax_cf.set_yticklabels(names, fontsize=5)
            ax_cf.set_xlabel("Resistance Score")
            ax_cf.set_title("Counterfactual\nOutcomes", fontsize=7, fontweight="bold")
            ax_cf.axvline(0.5, color=PALETTE["gray"], linestyle=":", linewidth=0.5)
            ax_cf.spines["top"].set_visible(False)
            ax_cf.spines["right"].set_visible(False)

        fig.suptitle(
            f"{patient_id} Interpretability Summary",
            fontsize=9, fontweight="bold", y=1.01,
        )

        if save_path:
            fig.savefig(save_path)
        return fig


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _test_visualizations():
    """Unit tests for all visualization classes."""
    print("Running visualization tests...")

    np.random.seed(42)

    # --- Test WaddingtonLandscapePlot ---
    nx, ny = 50, 50
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    # Double well potential
    V = (X**2 - 1)**2 + 0.5 * Y**2

    trajectories = [
        np.column_stack([
            np.linspace(-1.5, 1.0, 30) + np.random.randn(30) * 0.05,
            np.linspace(0.5, -0.5, 30) + np.random.randn(30) * 0.05,
        ]),
        np.column_stack([
            np.linspace(-1.5, -1.0, 30) + np.random.randn(30) * 0.05,
            np.linspace(-0.5, 0.0, 30) + np.random.randn(30) * 0.05,
        ]),
    ]

    fig = WaddingtonLandscapePlot.plot(
        V, (-2, 2), (-2, 2),
        trajectories=trajectories,
        trajectory_labels=["Resistant", "Sensitive"],
        attractors=[(-1, 0, "Sensitive"), (1, 0, "Resistant")],
        saddle_points=[(0, 0)],
    )
    assert fig is not None
    plt.close(fig)
    print("  Waddington landscape: rendered -- PASS")

    # --- Test PathwayCircosPlot ---
    pw_names = ["Stemness", "Drug Efflux", "DNA Repair",
                "Immune Evasion", "Metabolism", "Epigenetic"]
    A = np.random.randn(6, 6) * 0.1
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)

    known = np.random.rand(6, 6) > 0.5
    known = known | known.T

    fig = PathwayCircosPlot.plot(pw_names, A, known_mask=known)
    assert fig is not None
    plt.close(fig)
    print("  Circos plot: rendered -- PASS")

    # --- Test CounterfactualTrajectoryPlot ---
    t = np.linspace(0, 12, 50)
    factual = 0.3 + 0.5 * (1 / (1 + np.exp(-0.5 * (t - 6))))
    cf1 = 0.3 + 0.2 * (1 / (1 + np.exp(-0.3 * (t - 8))))
    cf2 = 0.3 - 0.1 * t / 12

    fig = CounterfactualTrajectoryPlot.plot(
        t, factual,
        {"Venetoclax": cf1, "Venetoclax+Dex": cf2},
        intervention_window=(3, 6),
    )
    assert fig is not None
    plt.close(fig)
    print("  Counterfactual plot: rendered -- PASS")

    # --- Test TemporalAttributionHeatmap ---
    attr = np.random.randn(6, 50) * 0.1
    attr[1, 20:30] = 0.5  # Drug efflux spike
    attr[5, 30:40] = -0.4  # Epigenetic shift

    fig = TemporalAttributionHeatmap.plot(
        t, pw_names, attr,
        clinical_events={"Cycle 1": 0, "Response": 3, "Relapse": 9},
    )
    assert fig is not None
    plt.close(fig)
    print("  Temporal heatmap: rendered -- PASS")

    # --- Test PatientJourneyPlot ---
    traj = np.cumsum(np.random.randn(50, 8) * 0.1, axis=0)
    contrib = np.abs(np.random.randn(50, 5))
    contrib /= contrib.sum(axis=1, keepdims=True)
    programs = ["Stemness", "Drug Efflux", "DNA Repair",
                "Immune Evasion", "Metabolism"]
    drugs = [
        {"name": "Bortezomib", "start": 0, "end": 4, "dose": 1.3},
        {"name": "Dexamethasone", "start": 0, "end": 6, "dose": 40},
        {"name": "Venetoclax", "start": 4, "end": 12, "dose": 400},
    ]
    cf_out = {"No treatment": 0.9, "Venetoclax only": 0.4,
              "VenDex": 0.2, "Bort+Ven+Dex": 0.15}

    fig = PatientJourneyPlot.plot(
        t, traj, contrib, programs,
        drug_history=drugs,
        counterfactual_outcomes=cf_out,
        patient_id="MM-0042",
    )
    assert fig is not None
    plt.close(fig)
    print("  Patient journey: rendered -- PASS")

    print("All visualization tests PASSED.\n")


if __name__ == "__main__":
    _test_visualizations()
