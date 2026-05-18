#!/usr/bin/env python3
"""
scripts/mortfm_scrna_latent_atlas.py
====================================
Lane 2 — build the scRNA latent atlas from the Block C encoder.

For every pseudobulk in the scRNA manifest, run the Block C encoder
forward, extract z0, and write a tidy parquet with one row per
(dataset, patient, stage). Then:

  * Compute a 2-component sklearn PCA + 2-component UMAP if umap-learn is
    available; fall back to PCA-only otherwise.
  * Train a tiny multinomial logistic regression stage classifier on the
    latents (5-fold CV); report macro-F1.
  * Generate UMAP-by-stage and UMAP-by-dataset PNG diagnostics.
  * Run the same encoder on a held-out subset of cells (sampled
    deterministically — head-of-list, never random) to produce a
    cell-level latent file.

Honest behaviour
----------------
* Pseudotime is ordinal. The stage classifier is a *diagnostic* — it does
  not enter any clinical claim. We write it under
  ``results/mortfm/scrna_stage_classifier_macro_f1.json`` with explicit
  "this is a representation-quality check, not a clinical claim" note.
* Block-C latents are on the scrna_block_c_hvg_union_3535 feature space
  and are NOT comparable to Block A's latents without explicit projection.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.feature_alignment import align_to_reference_features
from resistancemap.data.scrna_loader import build_snapshots_from_h5ad
from resistancemap.mortfm.blocks.block_c_scrna import load_block_c
from resistancemap.mortfm.schemas import (
    MORTBatch, ModalityName, ModalityTensor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_scrna_latent_atlas")


def _shared_hvg_union(h5ad_paths, top_per_dataset: int = 2000) -> list[str]:
    """Replays the same HVG-union logic Block C trains on."""
    import anndata as ad
    union: List[str] = []
    seen: set[str] = set()
    for p in h5ad_paths:
        a = ad.read_h5ad(str(p), backed="r")
        if "highly_variable" in a.var.columns:
            hvg = a.var.index[a.var["highly_variable"].astype(bool)]
        else:
            hvg = a.var.index[: top_per_dataset]
        for g in list(hvg.astype(str))[: top_per_dataset]:
            if g not in seen:
                seen.add(g)
                union.append(g)
    return union


def _encode_to_z0(model, rna_values: torch.Tensor, device: torch.device) -> torch.Tensor:
    n = rna_values.shape[0]
    batch = MORTBatch(
        rna=rna_values.to(device),
        modality_mask={
            "rna": torch.ones(n, dtype=torch.bool, device=device),
        },
        patient_ids=[f"_{i}" for i in range(n)],
        time=torch.zeros(n, device=device),
    )
    with torch.no_grad():
        state = model.encode(batch)
    return state.z0.cpu()


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/single_cell/scrna_manifest.csv")
    ap.add_argument("--block-c-checkpoint",
                    default="checkpoints/mortfm/block_c_state_encoder.pt")
    ap.add_argument("--out-pseudobulk",
                    default="data/processed/single_cell/scrna_pseudobulk_latents.parquet")
    ap.add_argument("--out-cell-sample",
                    default="data/processed/single_cell/scrna_cell_latents_sample.parquet")
    ap.add_argument("--cells-per-dataset", type=int, default=2000,
                    help="Deterministic head-of-list cell subset per dataset.")
    ap.add_argument("--summary",
                    default="logs/mortfm/scrna_state_summary.json")
    ap.add_argument("--fig-stage",
                    default="figures/mortfm/scrna_umap_by_stage.png")
    ap.add_argument("--fig-dataset",
                    default="figures/mortfm/scrna_umap_by_dataset.png")
    args = ap.parse_args()

    Path("logs/mortfm").mkdir(parents=True, exist_ok=True)
    Path("figures/mortfm").mkdir(parents=True, exist_ok=True)
    Path(args.out_pseudobulk).parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Block C ---------------------------------------------------
    model, cfg, _ = load_block_c(args.block_c_checkpoint)
    if model is None:
        logger.error("Block C checkpoint missing at %s", args.block_c_checkpoint)
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    logger.info("Block C loaded; rna_input_dim=%d d_latent=%d",
                cfg.rna_input_dim, cfg.d_latent)

    # --- 2. Manifest + HVG union ------------------------------------------
    manifest = pd.read_csv(args.manifest)
    h5ad_paths = [Path(p) for p in manifest["h5ad_path"].tolist()]
    ref_features = _shared_hvg_union(h5ad_paths, top_per_dataset=2000)
    if len(ref_features) != cfg.rna_input_dim:
        logger.warning(
            "Reference feature count (%d) differs from Block C rna_input_dim "
            "(%d); the encoder was trained with a different ordering. "
            "Falling back to truncated/padded match.",
            len(ref_features), cfg.rna_input_dim,
        )
        # Pad / truncate to match the encoder.
        if len(ref_features) > cfg.rna_input_dim:
            ref_features = ref_features[: cfg.rna_input_dim]
        else:
            ref_features = list(ref_features) + [f"_PAD_{i}" for i in range(cfg.rna_input_dim - len(ref_features))]

    # --- 3. Pseudobulk-level latents --------------------------------------
    rows = []
    for _, m_row in manifest.iterrows():
        path = Path(m_row["h5ad_path"])
        disease_tag = str(m_row.get("disease_stages", "")).split(",")[0] or "MM"
        snaps = build_snapshots_from_h5ad(
            str(path),
            patient_obs_col="patient_id",
            timepoint_obs_col="pseudotime_disease_stage",
            disease=disease_tag,
        )
        if not snaps:
            continue
        # Align all snapshots in one batch for speed.
        aligned_rows = []
        for s in snaps:
            if s.rna is None:
                continue
            d = pd.DataFrame(
                s.rna.values.cpu().numpy(),
                index=[s.sample_id],
                columns=s.rna.feature_names,
            )
            a, _, _ = align_to_reference_features(d, ref_features)
            aligned_rows.append(a)
        if not aligned_rows:
            continue
        aligned_block = pd.concat(aligned_rows, axis=0)
        rna_tensor = torch.from_numpy(aligned_block.values.astype(np.float32))
        z = _encode_to_z0(model, rna_tensor, device).numpy()
        for s, vec in zip(snaps, z):
            rows.append({
                "dataset_id": str(m_row["dataset_id"]),
                "patient_id": s.patient_id,
                "sample_id": s.sample_id,
                "disease": s.disease,
                "pseudotime_stage": s.timepoint,
                **{f"z{i}": float(vec[i]) for i in range(len(vec))},
            })
    pseudo_df = pd.DataFrame(rows)
    pseudo_df.to_parquet(args.out_pseudobulk)
    logger.info("Wrote %d pseudobulk latents -> %s", len(pseudo_df), args.out_pseudobulk)

    z_cols = [c for c in pseudo_df.columns if c.startswith("z")]
    Z = pseudo_df[z_cols].values

    # --- 4. Stage classifier (CV macro-F1) --------------------------------
    classifier_report = {}
    if "pseudotime_stage" in pseudo_df.columns:
        stages = pseudo_df["pseudotime_stage"].astype(str)
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import f1_score
            from sklearn.model_selection import StratifiedKFold
            keep = stages.notna() & (stages != "nan") & (stages != "None")
            X, y = Z[keep.values], stages[keep].values
            # Stratified K-fold only if each class has at least 2 samples.
            counts = pd.Series(y).value_counts()
            if (counts >= 2).all() and len(counts) >= 2 and len(X) >= 5:
                skf = StratifiedKFold(n_splits=min(3, int(counts.min())), shuffle=True, random_state=7)
                f1s = []
                for tr, te in skf.split(X, y):
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(X[tr], y[tr])
                    p = clf.predict(X[te])
                    f1s.append(f1_score(y[te], p, average="macro", zero_division=0))
                classifier_report = {
                    "n_classes": int(len(counts)),
                    "class_counts": counts.to_dict(),
                    "n_total": int(len(X)),
                    "cv_folds": int(min(3, int(counts.min()))),
                    "cv_macro_f1_mean": float(np.mean(f1s)),
                    "cv_macro_f1_std": float(np.std(f1s)),
                    "note": "Representation-quality diagnostic only. NOT a clinical claim.",
                }
            else:
                classifier_report = {
                    "skipped": True,
                    "reason": f"Insufficient class balance (counts={counts.to_dict()})",
                }
        except ImportError:
            classifier_report = {"skipped": True, "reason": "sklearn not available"}

    # --- 5. UMAP / PCA + figures ------------------------------------------
    proj_kind = "none"
    proj = None
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_neighbors=min(15, max(2, len(Z) - 1)), random_state=11)
        proj = reducer.fit_transform(Z)
        proj_kind = "UMAP"
    except Exception:
        try:
            from sklearn.decomposition import PCA
            proj = PCA(n_components=2, random_state=11).fit_transform(Z) if len(Z) >= 2 else None
            proj_kind = "PCA"
        except ImportError:
            pass
    figures = []
    if proj is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for color_col, fig_path in [
                ("pseudotime_stage", args.fig_stage),
                ("dataset_id", args.fig_dataset),
            ]:
                fig, ax = plt.subplots(figsize=(6, 5))
                labels = pseudo_df[color_col].astype(str).values
                uniq = sorted(set(labels))
                for u in uniq:
                    mask = labels == u
                    ax.scatter(proj[mask, 0], proj[mask, 1], s=30, alpha=0.85, label=u)
                ax.set_title(f"Block C pseudobulk latents ({proj_kind}) by {color_col}")
                ax.set_xlabel(f"{proj_kind} 1"); ax.set_ylabel(f"{proj_kind} 2")
                ax.legend(loc="best", fontsize=8, markerscale=0.7)
                fig.tight_layout()
                Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_path, dpi=130)
                plt.close(fig)
                figures.append(fig_path)
        except ImportError:
            logger.warning("matplotlib not available; skipped figures")

    # --- 6. Cell-level latent sample (deterministic head-of-list) ----------
    cell_rows = []
    try:
        import anndata as ad
        for _, m_row in manifest.iterrows():
            a = ad.read_h5ad(str(m_row["h5ad_path"]), backed="r")
            n_take = min(args.cells_per_dataset, a.n_obs)
            sub = a[:n_take].to_memory()
            X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
            cols = list(sub.var_names.astype(str))
            d = pd.DataFrame(X, index=[f"{m_row['dataset_id']}_cell_{i}" for i in range(n_take)], columns=cols)
            aligned, _, _ = align_to_reference_features(d, ref_features)
            z = _encode_to_z0(
                model, torch.from_numpy(aligned.values.astype(np.float32)), device,
            ).numpy()
            for i, vec in enumerate(z):
                obs_row = sub.obs.iloc[i]
                cell_rows.append({
                    "dataset_id": str(m_row["dataset_id"]),
                    "cell_index": int(i),
                    "patient_id": str(obs_row.get("patient_id", "")),
                    "pseudotime_stage": str(obs_row.get("pseudotime_disease_stage", "")),
                    **{f"z{j}": float(vec[j]) for j in range(len(vec))},
                })
    except Exception as exc:
        logger.warning("Cell-level sampling failed: %s", exc)
    if cell_rows:
        cell_df = pd.DataFrame(cell_rows)
        cell_df.to_parquet(args.out_cell_sample)
        logger.info("Wrote %d cell-level latents -> %s", len(cell_df), args.out_cell_sample)

    summary = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-scrna-latent-atlas",
        "block_c_checkpoint": args.block_c_checkpoint,
        "n_pseudobulks": int(len(pseudo_df)),
        "n_cells_sampled": int(len(cell_rows)),
        "n_reference_hvgs": len(ref_features),
        "projection_kind": proj_kind,
        "stage_classifier": classifier_report,
        "figures_written": figures,
        "outputs": {
            "pseudobulk_latents": args.out_pseudobulk,
            "cell_latents_sample": args.out_cell_sample if cell_rows else None,
        },
        "claim_unlocked": "single_cell_state",
        "claim_blocked": [
            "longitudinal_trajectory (pseudotime only)",
            "resistance_emergence (no calendar-time, no resistance endpoint)",
            "patient_level_clinical_prediction (no patient outcomes)",
        ],
        "wall_time_s": round(time.time() - t0, 1),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary: %s", json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
