# Phase 11 — Scientific Visualizations

**Branch:** `v18-clean-canonical-mortfm`
**Date:** 2026-05-20
**Status:** Design spec; implementation target `scripts/mortfm/10_generate_figures.py`

---

## 0. Principles

Every plotting function in this phase must satisfy three invariants:

1. **Zero fabrication.** If the required artifact file is absent or its
   provenance metadata does not match the expected `run_id`, the function raises
   `MissingArtifactError` — it never falls back to synthetic data, random noise,
   or placeholder arrays.
2. **Provenance firewall.** Each artifact bundle (`*.pt`, `*.parquet`) is
   expected to carry a `metadata` dict containing at minimum `run_id`,
   `git_sha`, and `data_hash`. Every plotting function reads those fields and
   embeds them in the figure's `fig.text` footer (bottom-right, 6 pt) and in the
   PNG `iTXt` metadata chunk written by `savefig`. A figure whose provenance
   cannot be verified is saved to `<name>_UNVERIFIED.png` and a warning is
   logged at `ERROR` level.
3. **Single responsibility.** Each function answers exactly one scientific
   question, stated in the figure title.

**Matplotlib global style** applied once in `10_generate_figures.py` before any
figure call:

```python
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
})
```

**Palette convention:**
- Continuous scalar fields (potential energy, risk score): `viridis` (muted
  variant clipped to [5th, 95th] percentile to suppress outlier saturation).
- Diverging residuals: `coolwarm` centred at zero.
- Categorical (cohort / modality): `tab10` restricted to ≤ 10 classes; beyond
  that, `Set2`.
- No rainbow / `jet` anywhere.

---

## 1. Shared Exception and Provenance Utilities

**File:** `resistancemap/visualization/__init__.py`

```python
class MissingArtifactError(FileNotFoundError):
    """Raised when a required artifact file is absent or unverifiable."""

class ProvenanceMismatchError(RuntimeError):
    """Raised when artifact run_id / git_sha / data_hash does not match
    the run being plotted."""
```

```python
def load_artifact(path: str | Path, *, run_id: str) -> dict:
    """Load a torch checkpoint and verify provenance.

    Raises
    ------
    MissingArtifactError
        If ``path`` does not exist.
    ProvenanceMismatchError
        If the stored ``metadata.run_id`` differs from ``run_id``.

    Returns
    -------
    dict
        The full payload dict as saved by ``MORTFMTrainer.save_checkpoint``
        (keys: ``model_state_dict``, ``config``, ``history``, ``metadata``).
    """

def stamp_figure(
    fig: matplotlib.figure.Figure,
    *,
    run_id: str,
    git_sha: str,
    data_hash: str,
) -> None:
    """Write a 6-pt provenance footer into the figure and embed the same
    fields into the PNG iTXt metadata via ``fig.savefig`` metadata kwarg."""
```

---

## 2. Module Specifications

### 2.1 `resistancemap/visualization/latent.py`

**Scientific question answered:** Where do patients / cell lines sit on the
multi-omic manifold and is resistance clustered?

**Required artifact:**
```
checkpoints/mortfm/stage_B_complete.pt   # contains foundation_state.z0
                                          # shape (N, d_latent)
```
The payload must contain `metadata.run_id`, `metadata.patient_ids` (list of
length N), `metadata.modality_coverage` (dict patient_id → frozenset of present
modalities), and `metadata.split_labels` (list, values in
`{"train","val","test"}`).

**Public functions:**

```python
def plot_latent_manifold(
    artifact_path: str | Path,
    *,
    run_id: str,
    color_by: Literal["cohort", "drug_response", "modality_coverage", "split"],
    out_path: str | Path,
    reducer: Literal["phate", "umap", "pca"] = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    point_size: float = 8.0,
    alpha: float = 0.7,
    highlight_test: bool = True,
) -> matplotlib.figure.Figure:
    """Produce a 2-D embedding of all N z0 latents.

    Parameters
    ----------
    artifact_path:
        Path to ``stage_B_complete.pt``. Function raises ``MissingArtifactError``
        if absent.
    run_id:
        Expected run identifier; checked against artifact metadata.
    color_by:
        Which scalar / categorical column to map to colour:
        - ``"cohort"``: MMRF / BeatAML / CCLE / GDSC cohort label.
        - ``"drug_response"``: aggregate AUC / IC50 from drug-response head.
        - ``"modality_coverage"``: number of modalities present per patient
          (0-8 scale, viridis).
        - ``"split"``: train / val / test membership.
    reducer:
        Dimensionality-reduction backend.  Prefers ``phate`` if installed;
        falls back in order to ``umap`` then ``pca``.  The figure title and
        axis labels reflect whichever reducer was actually used.
    highlight_test:
        If True, test-set points are outlined with a black ring (zorder high).

    Returns
    -------
    matplotlib.figure.Figure
        Single-panel scatter.  Also saves to ``out_path``.

    Raises
    ------
    MissingArtifactError
        ``artifact_path`` does not exist.
    ProvenanceMismatchError
        Stored ``run_id`` differs from argument.
    """
```

**Guard-rail implementation note:**
```python
if not Path(artifact_path).exists():
    raise MissingArtifactError(
        f"latent.plot_latent_manifold: artifact not found at {artifact_path}. "
        "Run scripts/mortfm/04_pretrain_foundation.py first."
    )
payload = load_artifact(artifact_path, run_id=run_id)  # raises ProvenanceMismatchError
```

**Paper figure:** Figure 2 — Multi-omic latent UMAP / PHATE.

---

### 2.2 `resistancemap/visualization/trajectory.py`

**Scientific question answered:** Do predicted latent trajectories match the
direction and magnitude of observed longitudinal state change?

**Required artifacts:**
```
checkpoints/mortfm/stage_E_complete.pt
    payload["test_predictions"]["z_traj"]   # (N_test, T, d_latent)
    payload["test_predictions"]["z_target"] # (N_test, d_latent)  observed t1
    payload["test_predictions"]["patient_ids"]
    payload["test_predictions"]["t_grid"]   # (T,)
    payload["test_predictions"]["z_samples"] # (N_test, K, T, d_latent) optional
```

**Public functions:**

```python
def plot_trajectory_ribbons(
    artifact_path: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    n_patients: int = 20,
    quantile_lo: float = 0.1,
    quantile_hi: float = 0.9,
    reducer: Literal["pca"] = "pca",
    palette: str = "viridis",
) -> matplotlib.figure.Figure:
    """Per-patient ribbon plot in 2-D PCA of latent space.

    Each ribbon spans the ``z_samples`` quantile envelope (shaded) around the
    mean predicted trajectory ``z_traj``; the observed ``z_target`` is plotted
    as a star at t=T.  Patients are sampled deterministically from the test set
    (top ``n_patients`` by hit-time uncertainty, so the figure shows the
    hardest cases first).

    Raises
    ------
    MissingArtifactError
        ``artifact_path`` absent.
    KeyError
        Required key missing from payload (implies incomplete training run).
    """

def plot_pred_vs_obs_scatter(
    artifact_path: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    metric: Literal["cosine", "l2"] = "cosine",
) -> matplotlib.figure.Figure:
    """Scatter of per-patient predicted-vs-observed latent similarity.

    x-axis: predicted trajectory endpoint cosine similarity to z_target.
    y-axis: baseline (z0 to z_target similarity, i.e. "no-change" model).
    Diagonal = break-even line.  Points above diagonal = model beats baseline.
    Coloured by pseudotime = distance z0→z_target in latent space.

    Raises
    ------
    MissingArtifactError
        ``artifact_path`` absent.
    """
```

**Paper figure:** Figure 4 — Predicted vs observed future state.

---

### 2.3 `resistancemap/visualization/waddington.py`

**Scientific question answered:** Does the learned energy landscape have
distinct basins that correspond to clinically meaningful resistance states?

**Required artifacts:**
```
checkpoints/mortfm/stage_E_complete.pt
    payload["potential_grid"]["X"]         # (G, G) meshgrid x
    payload["potential_grid"]["Y"]         # (G, G) meshgrid y
    payload["potential_grid"]["U"]         # (G, G) potential values
    payload["potential_grid"]["proj"]      # (2, d_latent) projection matrix
    payload["basin_centroids"]             # (n_basins, 2) in projected space
    payload["basin_labels"]               # list[str] length n_basins
    payload["test_predictions"]["z_traj"] # (N_test, T, d_latent)
    payload["test_predictions"]["basin_probs"] # (N_test, T, n_basins)
```

If `potential_grid` is absent from the payload (stage E did not run the
Waddington regulariser), the function raises `MissingArtifactError` with a
specific message directing the user to re-run with `--waddington-grid`.

**Public functions:**

```python
def plot_landscape_2d(
    artifact_path: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    n_trajectory_samples: int = 30,
    contour_levels: int = 20,
    cmap: str = "viridis",
    alpha_surface: float = 0.85,
) -> matplotlib.figure.Figure:
    """Filled-contour Waddington landscape with overlaid trajectories.

    The contour shows ``U(x, y)`` on the stored PCA-projected grid.
    Each predicted trajectory is drawn as a thin line (alpha=0.4) from
    ``z_traj[:, 0, :]`` to ``z_traj[:, -1, :]`` after applying the stored
    projection matrix.  Basin centroids are annotated with their clinical label.

    Raises
    ------
    MissingArtifactError
        ``artifact_path`` absent or ``potential_grid`` key missing.
    """

def plot_landscape_3d(
    artifact_path: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    elev: float = 30.0,
    azim: float = -60.0,
    cmap: str = "viridis",
    alpha_surface: float = 0.6,
    n_trajectory_samples: int = 15,
) -> matplotlib.figure.Figure:
    """3-D surface plot (``Axes3D.plot_surface``) with trajectory ribbons.

    Uses ``matplotlib`` 3-D projection; does not require mayavi or plotly.
    Surface colour encodes ``U``; trajectories are drawn as 3-D lines at
    z = U(x(t), y(t)) + 0.05 offset so they sit above the surface.

    Raises
    ------
    MissingArtifactError
        ``artifact_path`` absent or ``potential_grid`` key missing.
    """
```

**Paper figure:** Figure 3 — Waddington landscape with predicted trajectory.

---

### 2.4 `resistancemap/visualization/survival.py`

**Scientific question answered:** Does MORT-FM discriminate high- vs low-risk
patients and is it calibrated?

**Required artifacts:**
```
checkpoints/mortfm/stage_F_complete.pt
    payload["test_predictions"]["survival_curve"]   # (N_test, n_time_bins)
    payload["test_predictions"]["hazard"]           # (N_test, n_time_bins)
    payload["test_predictions"]["cif_per_event"]    # (N_test, n_events, n_bins)
    payload["test_predictions"]["event_observed"]   # (N_test,) bool
    payload["test_predictions"]["time_to_event"]    # (N_test,) float days
    payload["test_predictions"]["patient_ids"]
    payload["time_bins"]                            # (n_time_bins,) float days
```

**Public functions:**

```python
def plot_kaplan_meier(
    artifact_path: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    risk_threshold: float = 0.5,
    time_unit: str = "months",
    log_rank: bool = True,
    confidence_interval: bool = True,
    at_risk_table: bool = True,
) -> matplotlib.figure.Figure:
    """KM curves for high-risk vs low-risk strata.

    Risk is defined by ``survival_curve[:, t_landmark]`` where
    ``t_landmark`` is the median follow-up time in the test set.
    Stratification uses ``risk_threshold`` on the predicted 1 - S(t_landmark).
    Log-rank p-value is computed via ``lifelines.statistics.logrank_test``
    if installed; otherwise via the Mantel-Haenszel formula from scipy.

    Raises
    ------
    MissingArtifactError
        ``artifact_path`` absent.
    ValueError
        Test set has fewer than 10 events; too small for reliable KM.
    """

def plot_calibration(
    artifact_path: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    n_groups: int = 10,
    time_points: tuple[float, ...] = (12.0, 24.0, 36.0),
    style: Literal["d_agostino", "loess"] = "d_agostino",
) -> matplotlib.figure.Figure:
    """D'Agostino-style calibration plot at multiple landmark times.

    For each time in ``time_points``, patients are grouped into ``n_groups``
    deciles of predicted risk; observed Kaplan-Meier event probability at that
    time is plotted against mean predicted probability.  The identity line is
    the perfect-calibration reference.

    Raises
    ------
    MissingArtifactError
        ``artifact_path`` absent.
    """

def plot_time_dependent_auc(
    artifact_path: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    time_grid_months: tuple[float, ...] = (6.0, 12.0, 18.0, 24.0, 36.0, 48.0),
) -> matplotlib.figure.Figure:
    """Time-dependent AUC (Uno's C at each landmark).

    Plots AUC(t) as a line with 95 % bootstrap CI (n_boot=500, seed=42).
    Shaded region between AUC(t) curve and the 0.5 reference line is coloured
    by whether the model is above or below chance (green / red fill).

    Raises
    ------
    MissingArtifactError
        ``artifact_path`` absent.
    """
```

**Paper figure:** Figure 5 — KM high-risk vs low-risk + calibration (two
sub-panels on one figure, the calibration and AUC panels as insets).

---

### 2.5 `resistancemap/visualization/pathway.py`

**Scientific question answered:** Which proteins and pathways are driving the
predicted resistance trajectory for each drug?

**Required artifacts:**
```
checkpoints/mortfm/stage_G_complete.pt
    payload["test_predictions"]["protein_scores"]   # (N_test, n_proteins) float
    payload["test_predictions"]["edge_scores"]      # (N_test, n_edges) float
    payload["test_predictions"]["pathway_scores"]   # (N_test, n_pathways) float
    payload["graph_meta"]["protein_names"]          # list[str] length n_proteins
    payload["graph_meta"]["edge_index"]             # (2, n_edges) int
    payload["graph_meta"]["pathway_names"]          # list[str] length n_pathways
```

Stage G is the pathway + counterfactual validation stage. If `stage_G_complete.pt`
is absent the function must raise with a message that Stage G has not been run.

**Public functions:**

```python
def plot_ppi_subgraph(
    artifact_path: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    top_k_proteins: int = 40,
    top_k_edges: int = 80,
    node_cmap: str = "viridis",
    edge_cmap: str = "viridis",
    layout: Literal["spring", "kamada_kawai"] = "kamada_kawai",
    aggregate: Literal["mean", "median"] = "mean",
) -> matplotlib.figure.Figure:
    """PPI subgraph heat-overlay of mean test-set protein and edge scores.

    Selects the ``top_k_proteins`` by mean ``protein_scores`` across the test
    set, then retains the ``top_k_edges`` among them ranked by mean
    ``edge_scores``.  Node colour = protein score (viridis); edge colour and
    width = edge score.  Rendered via ``networkx`` + ``matplotlib``.

    Raises
    ------
    MissingArtifactError
        ``artifact_path`` absent or ``protein_scores`` key missing.
    """

def plot_pathway_bars(
    artifact_path: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    top_k: int = 20,
    aggregate: Literal["mean", "median"] = "mean",
    sort_by: Literal["score", "name"] = "score",
    error_bars: Literal["sem", "std", "iqr"] = "sem",
) -> matplotlib.figure.Figure:
    """Horizontal bar chart of top pathway scores across the test cohort.

    Bars show ``aggregate`` of ``pathway_scores`` per pathway; error bars show
    ``error_bars`` across test patients.  Coloured by absolute score magnitude
    (viridis).

    Raises
    ------
    MissingArtifactError
        ``artifact_path`` absent or ``pathway_scores`` key missing.
    """
```

**Paper figure:** Figure 6 — Pathway route on PPI graph.

---

### 2.6 `resistancemap/visualization/benchmarking.py`

**Scientific question answered:** Does MORT-FM outperform published baselines
on the MMRF and CCLE/GDSC test sets?

**Required artifact:**
```
evaluation/model_comparison.parquet
    columns: model_name, cohort, metric_name, value, ci_lo, ci_hi, n_test
```

This parquet is produced by `resistancemap/evaluation/benchmarking.py`
→ `SOTABenchmarkRunner.run_comparison()`. If absent, raise `MissingArtifactError`;
do not fall back to hardcoded numbers from memory.

**Public functions:**

```python
def plot_baseline_comparison(
    comparison_parquet: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    metric: str = "c_index",
    cohort: str = "MMRF",
    models_order: list[str] | None = None,
    palette: str = "tab10",
    ci_style: Literal["bar", "errorbar"] = "errorbar",
) -> matplotlib.figure.Figure:
    """Grouped bar chart comparing MORT-FM against baselines.

    Each bar = one model; height = ``metric`` value; error bars = [ci_lo,
    ci_hi].  MORT-FM bar is highlighted with a distinct edge colour.  A
    horizontal dashed line marks the performance of the predict-mean baseline.

    Raises
    ------
    MissingArtifactError
        ``comparison_parquet`` absent.
    KeyError
        ``metric`` or ``cohort`` not found in parquet columns / values.
    """

def plot_metric_grid(
    comparison_parquet: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    metrics: list[str] = ("c_index", "ibs", "auc_2yr", "auc_3yr"),
    cohorts: list[str] = ("MMRF", "BeatAML"),
) -> matplotlib.figure.Figure:
    """Grid of bar charts: rows = cohorts, cols = metrics.

    Each cell reproduces the logic of ``plot_baseline_comparison`` for that
    (metric, cohort) pair.  Shared legend across panels.

    Raises
    ------
    MissingArtifactError
        ``comparison_parquet`` absent.
    """
```

**Paper figure:** Figure 7 — Baseline comparison.

---

### 2.7 `resistancemap/visualization/counterfactual.py`

**Scientific question answered:** Which drug interventions most change the
predicted resistance trajectory for each patient?

**Required artifact:**
```
checkpoints/mortfm/stage_G_complete.pt
    payload["test_predictions"]["counterfactual_rankings"]
        # list[dict] length N_test, each dict:
        #   "patient_id": str
        #   "drug_names": list[str] length K_cf
        #   "delta_hitting_cdf": (K_cf, n_bins) float
        #   "delta_risk_score": (K_cf,) float
        #   "baseline_hitting_cdf": (n_bins,) float
    payload["time_bins"]   # (n_bins,)
```

**Public functions:**

```python
def plot_counterfactual_ranking(
    artifact_path: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    top_k_drugs: int = 5,
    n_patients: int = 6,
    cmap_positive: str = "Blues",
    cmap_negative: str = "Reds",
) -> matplotlib.figure.Figure:
    """Top-K counterfactual drug ranking grid for N patients.

    Layout: rows = patients (sampled as the N with highest baseline risk),
    cols = top-K drugs ranked by |delta_risk_score|.  Cell colour encodes
    delta_risk_score (blue = risk reduction, red = risk increase).

    Raises
    ------
    MissingArtifactError
        ``artifact_path`` absent or ``counterfactual_rankings`` key missing.
    """

def plot_hitting_cdf_before_after(
    artifact_path: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    patient_ids: list[str] | None = None,
    top_k_drugs: int = 3,
) -> matplotlib.figure.Figure:
    """Before/after hitting-time CDF panels for selected patients.

    Each patient gets a sub-panel: baseline CDF (grey) overlaid with per-drug
    counterfactual CDFs (coloured lines).  Shaded area between baseline and
    best-drug CDF = risk reduction benefit.

    Raises
    ------
    MissingArtifactError
        ``artifact_path`` absent or ``counterfactual_rankings`` key missing.
    ValueError
        ``patient_ids`` contains an ID not in the test set.
    """
```

**Paper figure:** Figure 9 — Counterfactual intervention ranking.

---

### 2.8 `resistancemap/visualization/ablation.py`

**Scientific question answered:** Which components of MORT-FM contribute to
performance gains over ablated variants?

**Required artifact:**
```
evaluation/ablation_results.parquet
    columns: variant_name, metric_name, value, ci_lo, ci_hi, n_test, stage_removed
```

Produced by `resistancemap/evaluation/ablation_harness.py`
→ `run_ablations()`. If absent, raise `MissingArtifactError`.

**Public functions:**

```python
def plot_ablation_bars(
    ablation_parquet: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    metric: str = "c_index",
    full_model_name: str = "MORT-FM (full)",
    palette: str = "Set2",
    annotate_delta: bool = True,
) -> matplotlib.figure.Figure:
    """Horizontal bar chart: MORT-FM full vs ablated variants.

    Each bar = one variant; the full-model bar is highlighted.  If
    ``annotate_delta`` is True, the delta vs full model is annotated on each
    bar (negative deltas in red).  Variants are sorted by performance descending.

    Raises
    ------
    MissingArtifactError
        ``ablation_parquet`` absent.
    """

def plot_ablation_heatmap(
    ablation_parquet: str | Path,
    *,
    run_id: str,
    out_path: str | Path,
    metrics: list[str] = ("c_index", "ibs", "auc_2yr", "trajectory_mse"),
) -> matplotlib.figure.Figure:
    """Heatmap: rows = ablation variants, cols = metrics.

    Cell = z-score of metric relative to the full-model value.  Colour map
    = coolwarm centred at 0.  Enables simultaneous inspection of which
    ablation hurts which metric most.

    Raises
    ------
    MissingArtifactError
        ``ablation_parquet`` absent.
    """
```

**Paper figure:** Figure 8 — Ablation study.

---

## 3. Orchestrator Script

**File:** `scripts/mortfm/10_generate_figures.py`

```python
#!/usr/bin/env python
"""
scripts/mortfm/10_generate_figures.py
======================================
Run-ID-scoped figure generation for MORT-FM paper figures.

Usage
-----
    python scripts/mortfm/10_generate_figures.py \\
        --run-id  <run_id>  \\
        --ckpt-dir checkpoints/mortfm \\
        --eval-dir evaluation \\
        --out-dir  paper/v18_artifacts/visualizations/<run_id>

All PNGs and PDFs are written to ``out_dir``.  The script exits non-zero if
ANY required artifact is absent (fail-fast; no partial figure sets).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib as mpl

# Must be called before any figure import.
mpl.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
})

from resistancemap.visualization import MissingArtifactError, ProvenanceMismatchError
from resistancemap.visualization.latent import plot_latent_manifold
from resistancemap.visualization.trajectory import (
    plot_trajectory_ribbons,
    plot_pred_vs_obs_scatter,
)
from resistancemap.visualization.waddington import plot_landscape_2d, plot_landscape_3d
from resistancemap.visualization.survival import (
    plot_kaplan_meier,
    plot_calibration,
    plot_time_dependent_auc,
)
from resistancemap.visualization.pathway import plot_ppi_subgraph, plot_pathway_bars
from resistancemap.visualization.benchmarking import (
    plot_baseline_comparison,
    plot_metric_grid,
)
from resistancemap.visualization.counterfactual import (
    plot_counterfactual_ranking,
    plot_hitting_cdf_before_after,
)
from resistancemap.visualization.ablation import plot_ablation_bars, plot_ablation_heatmap

log = logging.getLogger(__name__)


def _save(fig, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        log.info("Saved %s", p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ckpt-dir", default="checkpoints/mortfm")
    parser.add_argument("--eval-dir", default="evaluation")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    run_id: str = args.run_id
    ckpt = Path(args.ckpt_dir).resolve()
    evl = Path(args.eval_dir).resolve()
    out = Path(args.out_dir or f"paper/v18_artifacts/visualizations/{run_id}").resolve()
    out.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []

    def _run(fn, *a, **kw):
        try:
            fig = fn(*a, **kw)
            return fig
        except (MissingArtifactError, ProvenanceMismatchError, KeyError, ValueError) as exc:
            errors.append(str(exc))
            log.error("FIGURE FAILED: %s", exc)
            return None

    # Figure 2 — Multi-omic latent UMAP / PHATE
    _save(
        _run(plot_latent_manifold,
             ckpt / "stage_B_complete.pt",
             run_id=run_id, color_by="cohort",
             out_path=out / "fig2_latent_manifold.png"),
        out, "fig2_latent_manifold",
    )

    # Figure 3 — Waddington landscape
    _save(
        _run(plot_landscape_2d,
             ckpt / "stage_E_complete.pt",
             run_id=run_id, out_path=out / "fig3_landscape_2d.png"),
        out, "fig3_landscape_2d",
    )
    _run(plot_landscape_3d,
         ckpt / "stage_E_complete.pt",
         run_id=run_id, out_path=out / "fig3_landscape_3d.png")

    # Figure 4 — Predicted vs observed future state
    _save(
        _run(plot_pred_vs_obs_scatter,
             ckpt / "stage_E_complete.pt",
             run_id=run_id, out_path=out / "fig4_pred_vs_obs.png"),
        out, "fig4_pred_vs_obs",
    )
    _run(plot_trajectory_ribbons,
         ckpt / "stage_E_complete.pt",
         run_id=run_id, out_path=out / "fig4_traj_ribbons.png")

    # Figure 5 — KM + calibration
    _save(
        _run(plot_kaplan_meier,
             ckpt / "stage_F_complete.pt",
             run_id=run_id, out_path=out / "fig5_km.png"),
        out, "fig5_km",
    )
    _run(plot_calibration,
         ckpt / "stage_F_complete.pt",
         run_id=run_id, out_path=out / "fig5_calibration.png")
    _run(plot_time_dependent_auc,
         ckpt / "stage_F_complete.pt",
         run_id=run_id, out_path=out / "fig5_td_auc.png")

    # Figure 6 — Pathway PPI
    _save(
        _run(plot_ppi_subgraph,
             ckpt / "stage_G_complete.pt",
             run_id=run_id, out_path=out / "fig6_ppi_subgraph.png"),
        out, "fig6_ppi_subgraph",
    )
    _run(plot_pathway_bars,
         ckpt / "stage_G_complete.pt",
         run_id=run_id, out_path=out / "fig6_pathway_bars.png")

    # Figure 7 — Baseline comparison
    _save(
        _run(plot_metric_grid,
             evl / "model_comparison.parquet",
             run_id=run_id, out_path=out / "fig7_baseline_grid.png"),
        out, "fig7_baseline_grid",
    )

    # Figure 8 — Ablation
    _save(
        _run(plot_ablation_heatmap,
             evl / "ablation_results.parquet",
             run_id=run_id, out_path=out / "fig8_ablation_heatmap.png"),
        out, "fig8_ablation_heatmap",
    )

    # Figure 9 — Counterfactual
    _save(
        _run(plot_counterfactual_ranking,
             ckpt / "stage_G_complete.pt",
             run_id=run_id, out_path=out / "fig9_cf_ranking.png"),
        out, "fig9_cf_ranking",
    )
    _run(plot_hitting_cdf_before_after,
         ckpt / "stage_G_complete.pt",
         run_id=run_id, out_path=out / "fig9_hitting_cdf.png")

    if errors:
        log.error("%d figure(s) failed:", len(errors))
        for e in errors:
            log.error("  %s", e)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
```

---

## 4. Paper Figure Mapping

| Figure | Title | Module / function | Primary artifact |
|--------|-------|-------------------|-----------------|
| 1 | Architecture diagram | TODO — BioRender via MCP (leave as placeholder SVG committed to `paper/v18_artifacts/visualizations/fig1_architecture_PLACEHOLDER.svg`) | None (schematic) |
| 2 | Multi-omic latent UMAP / PHATE | `latent.plot_latent_manifold` | `stage_B_complete.pt` → `foundation_state.z0` (N, d_latent) |
| 3 | Waddington landscape with predicted trajectory | `waddington.plot_landscape_2d` + `plot_landscape_3d` | `stage_E_complete.pt` → `potential_grid.U`, `basin_centroids`, `z_traj` |
| 4 | Predicted vs observed future state | `trajectory.plot_pred_vs_obs_scatter` + `plot_trajectory_ribbons` | `stage_E_complete.pt` → `z_traj`, `z_samples`, `z_target`, `t_grid` |
| 5 | KM high-risk vs low-risk + calibration | `survival.plot_kaplan_meier` + `plot_calibration` + `plot_time_dependent_auc` | `stage_F_complete.pt` → `survival_curve`, `hazard`, `cif_per_event`, `event_observed`, `time_to_event` |
| 6 | Pathway route on PPI graph | `pathway.plot_ppi_subgraph` + `plot_pathway_bars` | `stage_G_complete.pt` → `protein_scores`, `edge_scores`, `pathway_scores`, `graph_meta` |
| 7 | Baseline comparison | `benchmarking.plot_metric_grid` | `evaluation/model_comparison.parquet` → columns `model_name, metric_name, value, ci_lo, ci_hi` |
| 8 | Ablation study | `ablation.plot_ablation_heatmap` + `plot_ablation_bars` | `evaluation/ablation_results.parquet` → columns `variant_name, metric_name, value, ci_lo, ci_hi` |
| 9 | Counterfactual intervention ranking | `counterfactual.plot_counterfactual_ranking` + `plot_hitting_cdf_before_after` | `stage_G_complete.pt` → `counterfactual_rankings[].delta_hitting_cdf`, `delta_risk_score`, `baseline_hitting_cdf` |

---

## 5. Smoke Tests

**File:** `tests/mortfm/test_visualization_smoke.py`

```python
"""
tests/mortfm/test_visualization_smoke.py
=========================================
Smoke tests for all visualization modules.

Two invariants per module:
  1. Raises MissingArtifactError when the artifact path does not exist.
  2. Renders a non-empty PNG when given a minimal but structurally valid
     toy payload saved from a deterministic fixture.

The fixture payload is produced by _make_toy_stage_E_payload() and written to
a tmp_path before each test.  It uses only torch / numpy — no model forward
pass — and is seeded at 0 for reproducibility.  The payload deliberately
contains all required keys so that missing-key failures surface as distinct
KeyError, not MissingArtifactError.

IMPORTANT: The fixture must NOT call any np.random or torch.rand without a
fixed seed.  All arrays are deterministically constructed from arange / linspace.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from resistancemap.visualization import MissingArtifactError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _provenance(run_id: str = "smoke-test-000") -> dict:
    return {"run_id": run_id, "git_sha": "deadbeef", "data_hash": "abc123"}


def _make_toy_stage_B(tmp_path: Path, run_id: str = "smoke-test-000") -> Path:
    """Minimal stage_B_complete.pt with z0 for 12 patients."""
    N, D = 12, 64
    g = torch.Generator().manual_seed(0)
    z0 = torch.randn(N, D, generator=g)
    payload = {
        "foundation_state": {"z0": z0},
        "patient_ids": [f"p{i}" for i in range(N)],
        "split_labels": ["train"] * 8 + ["val"] * 2 + ["test"] * 2,
        "modality_coverage": {f"p{i}": {"rna", "drug"} for i in range(N)},
        "metadata": _provenance(run_id),
    }
    p = tmp_path / "stage_B_complete.pt"
    torch.save(payload, p)
    return p


def _make_toy_stage_E(tmp_path: Path, run_id: str = "smoke-test-000") -> Path:
    """Minimal stage_E_complete.pt."""
    N, T, D, K = 12, 6, 64, 4
    g = torch.Generator().manual_seed(1)
    G = 32
    xs = torch.linspace(-3, 3, G)
    ys = torch.linspace(-3, 3, G)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    U = X ** 2 + Y ** 2  # simple bowl — no random
    payload = {
        "test_predictions": {
            "z_traj":   torch.randn(N, T, D, generator=g),
            "z_target": torch.randn(N, D, generator=g),
            "z_samples": torch.randn(N, K, T, D, generator=g),
            "t_grid":   torch.linspace(0.0, 1.0, T),
            "basin_probs": torch.softmax(torch.randn(N, T, 3, generator=g), dim=-1),
            "patient_ids": [f"p{i}" for i in range(N)],
        },
        "potential_grid": {
            "X": X, "Y": Y, "U": U,
            "proj": torch.eye(2, D),
        },
        "basin_centroids": torch.tensor([[-2.0, -2.0], [0.0, 2.0], [2.0, -1.0]]),
        "basin_labels": ["sensitive", "intermediate", "resistant"],
        "metadata": _provenance(run_id),
    }
    p = tmp_path / "stage_E_complete.pt"
    torch.save(payload, p)
    return p


def _make_toy_stage_F(tmp_path: Path, run_id: str = "smoke-test-000") -> Path:
    """Minimal stage_F_complete.pt."""
    N, B = 12, 8
    g = torch.Generator().manual_seed(2)
    raw = torch.randn(N, B, generator=g)
    surv = torch.cumprod(torch.sigmoid(-raw.abs()), dim=1)
    payload = {
        "test_predictions": {
            "survival_curve": surv,
            "hazard": torch.sigmoid(raw),
            "cif_per_event": torch.rand(N, 2, B, generator=g),
            "event_observed": torch.tensor([True] * 8 + [False] * 4),
            "time_to_event": torch.arange(N, dtype=torch.float32) * 30.0,
            "patient_ids": [f"p{i}" for i in range(N)],
        },
        "time_bins": torch.linspace(0.0, 48.0, B),
        "metadata": _provenance(run_id),
    }
    p = tmp_path / "stage_F_complete.pt"
    torch.save(payload, p)
    return p


def _make_toy_stage_G(tmp_path: Path, run_id: str = "smoke-test-000") -> Path:
    """Minimal stage_G_complete.pt."""
    N, P, E, PW, K_cf, Bins = 12, 50, 80, 10, 5, 8
    g = torch.Generator().manual_seed(3)
    edge_index = torch.stack([
        torch.randint(0, P, (E,), generator=g),
        torch.randint(0, P, (E,), generator=g),
    ])
    cf = []
    for i in range(N):
        b_cdf = torch.linspace(0.0, 0.8, Bins)
        cf.append({
            "patient_id": f"p{i}",
            "drug_names": [f"drug_{j}" for j in range(K_cf)],
            "delta_hitting_cdf": torch.randn(K_cf, Bins, generator=g) * 0.1,
            "delta_risk_score": torch.randn(K_cf, generator=g) * 0.2,
            "baseline_hitting_cdf": b_cdf,
        })
    payload = {
        "test_predictions": {
            "protein_scores": torch.rand(N, P, generator=g),
            "edge_scores":    torch.rand(N, E, generator=g),
            "pathway_scores": torch.rand(N, PW, generator=g),
            "counterfactual_rankings": cf,
        },
        "graph_meta": {
            "protein_names": [f"PROT{i}" for i in range(P)],
            "edge_index": edge_index,
            "pathway_names": [f"PW{i}" for i in range(PW)],
        },
        "time_bins": torch.linspace(0.0, 48.0, Bins),
        "metadata": _provenance(run_id),
    }
    p = tmp_path / "stage_G_complete.pt"
    torch.save(payload, p)
    return p


# ---------------------------------------------------------------------------
# Absence tests — every module must raise MissingArtifactError on bad path
# ---------------------------------------------------------------------------

ABSENT = Path("/tmp/does_not_exist_mortfm_viz_smoke.pt")
ABSENT_PQ = Path("/tmp/does_not_exist_mortfm_viz_smoke.parquet")

RUN_ID = "smoke-test-000"


def test_latent_raises_on_missing():
    from resistancemap.visualization.latent import plot_latent_manifold
    with pytest.raises(MissingArtifactError):
        plot_latent_manifold(ABSENT, run_id=RUN_ID,
                             color_by="cohort", out_path="/dev/null")


def test_trajectory_ribbons_raises_on_missing():
    from resistancemap.visualization.trajectory import plot_trajectory_ribbons
    with pytest.raises(MissingArtifactError):
        plot_trajectory_ribbons(ABSENT, run_id=RUN_ID, out_path="/dev/null")


def test_trajectory_scatter_raises_on_missing():
    from resistancemap.visualization.trajectory import plot_pred_vs_obs_scatter
    with pytest.raises(MissingArtifactError):
        plot_pred_vs_obs_scatter(ABSENT, run_id=RUN_ID, out_path="/dev/null")


def test_waddington_2d_raises_on_missing():
    from resistancemap.visualization.waddington import plot_landscape_2d
    with pytest.raises(MissingArtifactError):
        plot_landscape_2d(ABSENT, run_id=RUN_ID, out_path="/dev/null")


def test_survival_km_raises_on_missing():
    from resistancemap.visualization.survival import plot_kaplan_meier
    with pytest.raises(MissingArtifactError):
        plot_kaplan_meier(ABSENT, run_id=RUN_ID, out_path="/dev/null")


def test_pathway_ppi_raises_on_missing():
    from resistancemap.visualization.pathway import plot_ppi_subgraph
    with pytest.raises(MissingArtifactError):
        plot_ppi_subgraph(ABSENT, run_id=RUN_ID, out_path="/dev/null")


def test_benchmarking_raises_on_missing():
    from resistancemap.visualization.benchmarking import plot_baseline_comparison
    with pytest.raises(MissingArtifactError):
        plot_baseline_comparison(ABSENT_PQ, run_id=RUN_ID,
                                 out_path="/dev/null")


def test_counterfactual_raises_on_missing():
    from resistancemap.visualization.counterfactual import plot_counterfactual_ranking
    with pytest.raises(MissingArtifactError):
        plot_counterfactual_ranking(ABSENT, run_id=RUN_ID, out_path="/dev/null")


def test_ablation_raises_on_missing():
    from resistancemap.visualization.ablation import plot_ablation_bars
    with pytest.raises(MissingArtifactError):
        plot_ablation_bars(ABSENT_PQ, run_id=RUN_ID, out_path="/dev/null")


# ---------------------------------------------------------------------------
# Render tests — each must produce a non-empty PNG from the toy fixture
# ---------------------------------------------------------------------------

def _assert_nonempty_png(path: Path) -> None:
    assert path.exists(), f"Figure not written to {path}"
    assert path.stat().st_size > 1024, f"PNG suspiciously small: {path.stat().st_size} bytes"


def test_latent_renders(tmp_path):
    from resistancemap.visualization.latent import plot_latent_manifold
    art = _make_toy_stage_B(tmp_path)
    out = tmp_path / "fig2.png"
    fig = plot_latent_manifold(art, run_id=RUN_ID, color_by="split",
                               out_path=out, reducer="pca")
    assert fig is not None
    _assert_nonempty_png(out)


def test_trajectory_scatter_renders(tmp_path):
    from resistancemap.visualization.trajectory import plot_pred_vs_obs_scatter
    art = _make_toy_stage_E(tmp_path)
    out = tmp_path / "fig4.png"
    fig = plot_pred_vs_obs_scatter(art, run_id=RUN_ID, out_path=out)
    assert fig is not None
    _assert_nonempty_png(out)


def test_waddington_2d_renders(tmp_path):
    from resistancemap.visualization.waddington import plot_landscape_2d
    art = _make_toy_stage_E(tmp_path)
    out = tmp_path / "fig3.png"
    fig = plot_landscape_2d(art, run_id=RUN_ID, out_path=out)
    assert fig is not None
    _assert_nonempty_png(out)


def test_survival_km_renders(tmp_path):
    from resistancemap.visualization.survival import plot_kaplan_meier
    art = _make_toy_stage_F(tmp_path)
    out = tmp_path / "fig5.png"
    fig = plot_kaplan_meier(art, run_id=RUN_ID, out_path=out)
    assert fig is not None
    _assert_nonempty_png(out)


def test_counterfactual_renders(tmp_path):
    from resistancemap.visualization.counterfactual import plot_counterfactual_ranking
    art = _make_toy_stage_G(tmp_path)
    out = tmp_path / "fig9.png"
    fig = plot_counterfactual_ranking(art, run_id=RUN_ID, out_path=out)
    assert fig is not None
    _assert_nonempty_png(out)
```

---

## 6. Fabrication Firewall Pattern

Every public plotting function must execute the following guard block as its
**first three executable lines** after resolving the path:

```python
# --- Fabrication firewall ---------------------------------------------------
_p = Path(artifact_path)
if not _p.exists():
    raise MissingArtifactError(
        f"{__name__}: artifact not found at {_p!s}. "
        "Generate it by running the corresponding scripts/mortfm/0X_*.py stage."
    )
payload = load_artifact(_p, run_id=run_id)   # raises ProvenanceMismatchError if mismatch
meta = payload.get("metadata", {})
stamp_figure(fig, run_id=meta.get("run_id", "UNKNOWN"),
             git_sha=meta.get("git_sha", "UNKNOWN"),
             data_hash=meta.get("data_hash", "UNKNOWN"))
# ---------------------------------------------------------------------------
```

The `stamp_figure` call writes the provenance footer **before** `savefig` is
called. If any of `run_id`, `git_sha`, `data_hash` is `"UNKNOWN"` (metadata
block absent from the checkpoint), the figure stem is renamed to
`<name>_UNVERIFIED` and a `logging.ERROR` message is emitted. The script does
not abort — an unverified figure is still better than a fabricated one — but
the CI gate in `scripts/mortfm/09_gate_revalidate.py` will fail on any
`_UNVERIFIED` file in the output directory.

The metadata block is written by the trainer's `save_checkpoint` method. As
of v18.6 this block is **not yet present** in `MORTFMTrainer.save_checkpoint`.
The implementation of Phase 11 therefore also requires a one-line patch to
`resistancemap/mortfm/trainer.py`:

```python
# In save_checkpoint(), add to payload before torch.save:
import hashlib, subprocess, time as _time
payload["metadata"] = {
    "run_id":    self.config.run_id,          # add run_id to MORTFMTrainerConfig
    "git_sha":   subprocess.check_output(
                     ["git", "rev-parse", "--short", "HEAD"],
                     text=True).strip(),
    "data_hash": self.config.data_hash,       # add data_hash to MORTFMTrainerConfig
    "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
}
```

This one-line patch (three new config fields, four metadata values) is a hard
prerequisite for the firewall to function. It should be landed in v18.7 before
Phase 11 plotting functions are implemented.

---

## 7. Figure 1 Placeholder

Figure 1 is an architecture schematic. It cannot be generated from checkpoint
data. The placeholder path committed to the repo is:

```
paper/v18_artifacts/visualizations/fig1_architecture_PLACEHOLDER.svg
```

Recommended workflow: export from BioRender (accessible via the Figma MCP or
directly at biorender.com), save as SVG, commit. The orchestrator script
`10_generate_figures.py` checks for this file's existence but does not
regenerate it:

```python
fig1 = out.parent / "fig1_architecture_PLACEHOLDER.svg"
if not fig1.exists():
    log.warning("Figure 1 placeholder SVG not found at %s — skipping", fig1)
```

---

## 8. Dependencies

| Library | Required by | Graceful fallback |
|---------|------------|-------------------|
| `phate` | `latent.py` reducer | `umap-learn` then `sklearn.decomposition.PCA`; title updated |
| `umap-learn` | `latent.py` reducer fallback | `sklearn` PCA; title updated |
| `networkx` | `pathway.py` graph layout | Raise `ImportError` with install hint |
| `lifelines` | `survival.py` log-rank, KM | Scipy Mantel-Haenszel and manual KM; log at WARNING |
| `pandas` / `pyarrow` | `benchmarking.py`, `ablation.py` | Hard requirement; raise on import failure |
| `matplotlib` | All | Hard requirement |

---

*End of Phase 11 specification.*
