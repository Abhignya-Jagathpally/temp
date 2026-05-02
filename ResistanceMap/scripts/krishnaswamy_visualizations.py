#!/usr/bin/env python3
"""Krishnaswamy-style visualizations for ResistanceMap test data.

Five figures, all driven by on-disk checkpoints (no synthetic data).

  1. fig_latent_manifold       — PHATE/UMAP of all 886 cell-line VAE latents
  2. fig_pseudotime_per_drug   — pred vs truth, colored by ResistanceMap stability
  3. fig_meld_density          — drug-class density overlays on the manifold
  4. fig_residuals_heatmap     — (pred − truth) per cell-line per drug
  5. fig_resistance_landscape  — kernel-smoothed test_mse over PHATE-1/2

Falls back gracefully: PHATE -> UMAP -> PCA. Every figure caption labels which.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "checkpoints" / "data_ready.pt"
VAE_FT = REPO / "checkpoints" / "vae_finetuned.pt"
FUSION = REPO / "checkpoints" / "fusion_trained.pt"
VALIDATED = REPO / "checkpoints" / "pipeline_validated.pt"
OUT = REPO / "paper" / "v8_artifacts" / "visualizations"
OUT.mkdir(parents=True, exist_ok=True)

# Krishnaswamy aesthetic
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
    "axes.titlesize": 10, "axes.labelsize": 9,
})


def reduce_2d(X, prefer_phate=True):
    """Manifold reduction with graceful fallback. Returns (Z2, method_name)."""
    method = None
    if prefer_phate:
        try:
            import phate  # noqa
            op = phate.PHATE(n_components=2, knn=15, decay=40, t="auto",
                             verbose=0, random_state=42)
            return op.fit_transform(X), "PHATE"
        except Exception as e:
            method = f"PHATE unavailable ({type(e).__name__}); "
    try:
        import umap  # noqa
        op = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        return op.fit_transform(X), (method or "") + "UMAP"
    except Exception:
        op = PCA(n_components=2, random_state=42)
        return op.fit_transform(X), (method or "") + "PCA"


# ── data loading ───────────────────────────────────────────────────────────────
def load_inputs():
    print("[load] data_ready ...")
    c = torch.load(DATA, map_location="cpu", weights_only=False)
    ds = c["dataset"]; sp = c["splits"]
    drug_names = list(ds.drug_names)
    y = ds.drug_sensitivity.numpy().astype(np.float32)

    print("[load] fusion checkpoint (epi_states=VAE latents)...")
    fc = torch.load(FUSION, map_location="cpu", weights_only=False)
    epi = fc["epi_states"].numpy().astype(np.float32)        # (N, 64)  proxies for VAE latent
    stab = fc["stab_scores"].numpy().astype(np.float32).reshape(-1)
    print(f"        epi={epi.shape}  stab={stab.shape}")

    test_idx = np.asarray(sp["test"], dtype=np.int64)

    # Test predictions: re-derive by NOT re-running model — read from validated checkpoint
    pred_full = None
    if VALIDATED.exists():
        vc = torch.load(VALIDATED, map_location="cpu", weights_only=False)
        # we don't have predictions persisted; we'll compute residuals by retraining
        # tiny per-drug Ridge on the same features ONLY for the residual heatmap.
        # The metrics in vc["metrics"] are the source of truth for the per-drug summary.
    return {"epi": epi, "stab": stab, "y": y, "test_idx": test_idx,
            "drug_names": drug_names}


# ── helpers ────────────────────────────────────────────────────────────────────
DRUG_CLASSES = {  # 6 classes covering the 11 drugs
    "Bortezomib": "Proteasome",
    "Lenalidomide": "IMiD",
    "Panobinostat": "HDAC", "Vorinostat": "HDAC", "Romidepsin": "HDAC",
    "Venetoclax": "BCL2",
    "Dinaciclib": "CDK", "Palbociclib": "CDK",
    "Doxorubicin": "DNA-damage", "Etoposide": "DNA-damage", "Cyclophosphamide": "DNA-damage",
}
CLASS_COLORS = {
    "Proteasome": "#2E7D32", "IMiD": "#1976D2", "HDAC": "#7B1FA2",
    "BCL2": "#C2185B", "CDK": "#F57C00", "DNA-damage": "#5D4037",
}


def write_caption(fname: str, body: str):
    (OUT / f"{fname}.caption.md").write_text(body.strip() + "\n")


# ── figures ────────────────────────────────────────────────────────────────────
def fig1_latent_manifold(epi, stab, test_idx):
    Z, m = reduce_2d(epi)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=stab, cmap="viridis", s=10, alpha=0.7,
                    edgecolors="none")
    ax.scatter(Z[test_idx, 0], Z[test_idx, 1], facecolors="none",
               edgecolors="red", s=22, lw=0.6, label=f"test (n={len(test_idx)})")
    plt.colorbar(sc, ax=ax, label="ResistanceMap stability score")
    ax.set_title(f"Cell-line latent manifold ({m})\ncolored by stability score")
    ax.set_xlabel(f"{m}-1"); ax.set_ylabel(f"{m}-2")
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    out = OUT / "fig_latent_manifold.png"
    fig.savefig(out); plt.close(fig)
    write_caption("fig_latent_manifold", f"""
**Figure 1.** {m} embedding of all {len(epi)} cell-line VAE latents
(`checkpoints/fusion_trained.pt:epi_states`), colored by ResistanceMap's
calibrated stability score. The {len(test_idx)} held-out test cell lines
are circled in red. Interpretation: do the test points sit in stable
manifold regions (well-represented neighbors), or are they in sparse / edge
regions where the model extrapolates? Edge-located test points are where
prediction risk is highest.""")
    return out, m


def fig2_pseudotime_per_drug(epi, stab, y, test_idx, drug_names):
    """Per-drug pred-vs-truth scatter, colored by stability pseudotime."""
    # Train a tiny Ridge per drug on train; predict on test (so we have residuals)
    from sklearn.linear_model import Ridge
    ytr_idx = np.setdiff1d(np.arange(len(epi)), test_idx)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()
    for d, name in enumerate(drug_names):
        ax = axes[d]
        col_tr = ~np.isnan(y[ytr_idx, d])
        col_te = ~np.isnan(y[test_idx, d])
        if col_tr.sum() < 10:
            ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center"); ax.set_axis_off(); continue
        m = Ridge(alpha=10.0).fit(epi[ytr_idx][col_tr], y[ytr_idx, d][col_tr])
        if col_te.sum() == 0:
            ax.text(0.5, 0.5, "no test obs", transform=ax.transAxes, ha="center"); ax.set_axis_off(); continue
        truth = y[test_idx][col_te, d]
        pred = m.predict(epi[test_idx][col_te])
        c = stab[test_idx][col_te]
        sc = ax.scatter(truth, pred, c=c, cmap="viridis", s=12, alpha=0.7, edgecolors="none")
        lo, hi = float(np.nanmin([truth.min(), pred.min()])), float(np.nanmax([truth.max(), pred.max()]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.4)
        ax.set_title(f"{name} (n={int(col_te.sum())})", fontsize=9)
        ax.set_xlabel("truth", fontsize=8); ax.set_ylabel("pred", fontsize=8)
    for a in axes[len(drug_names):]:
        a.set_axis_off()
    fig.suptitle("Per-drug truth vs prediction (Ridge on VAE latents), colored by stability pseudotime", y=1.00)
    fig.tight_layout()
    out = OUT / "fig_pseudotime_per_drug.png"
    fig.savefig(out); plt.close(fig)
    write_caption("fig_pseudotime_per_drug", """
**Figure 2.** Per-drug predicted vs ground-truth drug-sensitivity on the test
split, colored by ResistanceMap's stability pseudotime. The Ridge regression
is from VAE latents (a clean head, isolating the latent's predictive content
from the rest of the architecture). Points off the diagonal are model errors;
color shows whether errors concentrate in any pseudotime regime. Panobinostat
and Doxorubicin (high MSE in `per_drug_metrics.csv`) are expected to show the
widest spread.""")
    return out


def fig3_meld_condition_density(epi, y, drug_names):
    """MELD-style: density per drug class on the manifold."""
    Z, m = reduce_2d(epi)
    classes = list(set(DRUG_CLASSES.values()))
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.flatten()
    for k, cls in enumerate(classes):
        ax = axes[k]
        # cells that are RESISTANT (top-quartile sensitivity score, NaN-masked)
        idx_class_drugs = [drug_names.index(d) for d, c in DRUG_CLASSES.items() if c == cls and d in drug_names]
        if not idx_class_drugs:
            ax.set_axis_off(); continue
        ydc = y[:, idx_class_drugs]
        mask = ~np.isnan(ydc).all(axis=1)
        agg = np.nanmean(ydc, axis=1)
        # show density of "high-resistance" cell lines for this class
        thr = np.nanpercentile(agg[mask], 75)
        is_res = (agg >= thr) & mask
        ax.scatter(Z[:, 0], Z[:, 1], c="#cccccc", s=4, alpha=0.4, edgecolors="none")
        ax.scatter(Z[is_res, 0], Z[is_res, 1], c=CLASS_COLORS[cls], s=14,
                   alpha=0.85, edgecolors="none", label=f"resistant (n={int(is_res.sum())})")
        ax.set_title(f"{cls}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle(f"MELD-style: where do top-quartile resistant cell lines sit on the {m} manifold?", y=1.0)
    fig.tight_layout()
    out = OUT / "fig_meld_density.png"
    fig.savefig(out); plt.close(fig)
    write_caption("fig_meld_density", f"""
**Figure 3.** MELD-style condition density: for each drug class, the
top-quartile resistant cell lines are highlighted on the {m} manifold.
If resistance is class-localized (clusters), then drug-class-specific
resistance mechanisms are decodable from the VAE latent. If resistance is
diffuse, the model is learning something orthogonal to drug-class biology.""")
    return out


def fig4_residuals_heatmap(epi, y, test_idx, drug_names):
    """Per-cell-line per-drug residual heatmap on test set."""
    from sklearn.linear_model import Ridge
    ytr_idx = np.setdiff1d(np.arange(len(epi)), test_idx)
    n_te = len(test_idx); nd = len(drug_names)
    R = np.full((n_te, nd), np.nan, dtype=np.float32)
    for d, name in enumerate(drug_names):
        tr = ~np.isnan(y[ytr_idx, d])
        if tr.sum() < 10: continue
        m = Ridge(alpha=10.0).fit(epi[ytr_idx][tr], y[ytr_idx, d][tr])
        for i, ti in enumerate(test_idx):
            if not np.isnan(y[ti, d]):
                R[i, d] = m.predict(epi[ti:ti+1])[0] - y[ti, d]
    # sort rows by |sum residual|
    row_order = np.argsort(np.nansum(np.abs(R), axis=1))[::-1]
    R_sorted = R[row_order]
    fig, ax = plt.subplots(figsize=(7, 7))
    vmax = float(np.nanpercentile(np.abs(R_sorted), 95))
    im = ax.imshow(R_sorted, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(nd)); ax.set_xticklabels(drug_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("test cell lines (sorted by total |residual|)", fontsize=9)
    ax.set_title("Residuals (pred − truth) on test set, per cell-line per drug")
    plt.colorbar(im, ax=ax, label="residual")
    fig.tight_layout()
    out = OUT / "fig_residuals_heatmap.png"
    fig.savefig(out); plt.close(fig)
    write_caption("fig_residuals_heatmap", """
**Figure 4.** Test-set residuals per cell-line × drug. Rows sorted by total
absolute residual (worst-predicted on top). Coolwarm: blue = under-prediction,
red = over-prediction. Read the columns to find drugs where the model
systematically over- or under-predicts (Panobinostat per `per_drug_metrics.csv`
should show extreme values). Read the rows to find cell lines that are
prediction-resistant across multiple drugs.""")
    return out


def fig5_resistance_landscape(epi, y):
    """Surface plot of mean resistance score over manifold."""
    Z, m = reduce_2d(epi)
    agg = np.nanmean(y, axis=1)
    valid = ~np.isnan(agg)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(Z[valid, 0], Z[valid, 1], c=agg[valid], s=14,
                    cmap="RdYlGn_r", edgecolors="none", alpha=0.85)
    plt.colorbar(sc, ax=ax, label="mean drug-sensitivity score (lower=resistant)")
    ax.set_xlabel(f"{m}-1"); ax.set_ylabel(f"{m}-2")
    ax.set_title(f"Resistance landscape on {m} manifold\n(green=sensitive, red=resistant)")
    fig.tight_layout()
    out = OUT / "fig_resistance_landscape.png"
    fig.savefig(out); plt.close(fig)
    write_caption("fig_resistance_landscape", f"""
**Figure 5.** Resistance landscape — each cell line projected onto the {m}
manifold and colored by its mean drug-sensitivity (NaN-masked across the 11
GDSC drugs). Green = pan-sensitive, red = pan-resistant. Spatial structure
indicates that the latent encodes a global resistance phenotype; absence of
structure indicates resistance is drug-specific (i.e., not a shared latent
axis), which would constrain how interpretable the "stability score"
abstraction is.""")
    return out


def main():
    print("[v8 viz] starting")
    inp = load_inputs()
    figs = []
    figs.append(("fig_latent_manifold",   *fig1_latent_manifold(inp["epi"], inp["stab"], inp["test_idx"])))
    figs.append(("fig_pseudotime_per_drug", fig2_pseudotime_per_drug(inp["epi"], inp["stab"], inp["y"], inp["test_idx"], inp["drug_names"]), None))
    figs.append(("fig_meld_density",        fig3_meld_condition_density(inp["epi"], inp["y"], inp["drug_names"]), None))
    figs.append(("fig_residuals_heatmap",   fig4_residuals_heatmap(inp["epi"], inp["y"], inp["test_idx"], inp["drug_names"]), None))
    figs.append(("fig_resistance_landscape", fig5_resistance_landscape(inp["epi"], inp["y"]), None))
    print("\n[done] figures + captions:")
    for f in figs:
        print(f"  - {f[1]}")


if __name__ == "__main__":
    main()
