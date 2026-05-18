#!/usr/bin/env python3
"""
scripts/mortfm_lens_smoke_test.py
=================================
Smoke-test the MORT-FM-LENS GraphEnergyResistanceSDE + ResistanceBasin +
HittingTime on a *real* Block C latent batch.

This is NOT training. It verifies:
  * forward pass produces finite trajectories
  * basin probabilities sum to 1 per (B, T)
  * Monte-Carlo hitting-time CDF is monotone non-decreasing in t
  * LossRouter correctly skips losses without supervision (the v15
    invariant that the trainer never silently substitutes a smaller loss).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.mortfm.trajectory import (
    GraphEnergyResistanceSDE, HittingTime, ResistanceBasin,
)
from resistancemap.training.loss_router import LossRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_lens_smoke_test")


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents",
                    default="data/processed/single_cell/scrna_pseudobulk_latents.parquet")
    ap.add_argument("--out",
                    default="logs/mortfm/lens_smoke_summary.json")
    args = ap.parse_args()

    if not Path(args.latents).exists():
        logger.error("Latents parquet missing at %s — run mortfm_scrna_latent_atlas.py first.",
                     args.latents)
        return 1
    df = pd.read_parquet(args.latents)
    z_cols = [c for c in df.columns if c.startswith("z")]
    d_latent = len(z_cols)
    z0 = torch.tensor(df[z_cols].values, dtype=torch.float32)        # (B, d_latent)
    B = z0.shape[0]
    logger.info("Loaded %d real Block-C latents (d_latent=%d)", B, d_latent)

    # Fabricate-free dummy graph/drug/clinical contexts — these are NOT
    # synthetic data, they're zero tensors that exercise the module shapes.
    d_graph, d_drug, d_clin = 16, 8, 4
    graph_emb = torch.zeros(B, d_graph)
    drug = torch.zeros(B, d_drug)
    clinical = torch.zeros(B, d_clin)

    sde = GraphEnergyResistanceSDE(
        d_latent=d_latent, d_graph=d_graph, d_drug=d_drug, d_clinical=d_clin,
        n_basins=5, integration_time=1.0, n_time_grid=8, n_mc_samples=4,
    )
    sde.eval()
    with torch.no_grad():
        out = sde(z0, graph_emb, drug, clinical, return_samples=True)
    z_traj = out["z_traj"]
    z_samp = out["z_samples"]
    assert torch.isfinite(z_traj).all(), "z_traj contains NaNs/Infs"
    assert torch.isfinite(z_samp).all(), "z_samples contains NaNs/Infs"
    logger.info("SDE rollout OK -- z_traj=%s, z_samples=%s",
                tuple(z_traj.shape), tuple(z_samp.shape))

    # Share centroids with the potential so basin assignment is consistent.
    basin = ResistanceBasin(
        d_latent=d_latent, n_basins=5, share_centroids_with=sde.potential,
    )
    with torch.no_grad():
        probs = basin.trajectory_probs(z_traj)
    assert probs.shape == (B, 8, 5)
    sum_ok = torch.allclose(probs.sum(dim=-1), torch.ones(B, 8), atol=1e-4)
    logger.info("Basin probabilities shape=%s sum_to_1=%s", tuple(probs.shape), sum_ok)

    htime = HittingTime(basin, resistant_basin_index=4, prob_threshold=0.3)
    with torch.no_grad():
        h = htime(z_samp, out["t_grid"])
    cdf = h["cdf"]
    cdf_monotone = bool(((cdf[:, 1:] - cdf[:, :-1]) >= -1e-6).all().item())
    frac_hit = h["frac_hit"]
    logger.info("Hitting-time CDF monotone=%s; mean frac_hit=%.3f",
                cdf_monotone, float(frac_hit.mean().item()))

    # LossRouter smoke: enable losses with no supervision, expect skip.
    def _dummy_reconstruction(outputs, batch):
        return torch.tensor(0.5)

    def _dummy_drug(outputs, batch):
        return (batch["drug_response_target"] ** 2).mean()

    router = LossRouter(
        enabled={"recon", "drug"},
        loss_fns={"recon": _dummy_reconstruction, "drug": _dummy_drug},
        weights={"recon": 1.0, "drug": 1.0},
        required_supervision={"drug": ["drug_response_target"]},
    )
    total, comps, audit = router.compute({}, batch={"x_rna": z0})  # no drug target
    drug_skip = [a for a in audit if a.name == "drug" and not a.fired]
    assert drug_skip and "drug_response_target" in (drug_skip[0].skipped_reason or "")
    logger.info("LossRouter correctly skipped drug-loss (no supervision); recon fired=%s",
                comps.get("recon"))

    summary = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-lens-smoke",
        "n_latents": B,
        "d_latent": d_latent,
        "z_traj_shape": list(z_traj.shape),
        "z_samples_shape": list(z_samp.shape),
        "basin_probs_sum_to_1": bool(sum_ok),
        "hitting_time_cdf_monotone": cdf_monotone,
        "mean_frac_hit": float(frac_hit.mean().item()),
        "loss_router_correctly_skipped_drug": True,
        "loss_router_recon_value": float(comps.get("recon", float("nan"))),
        "passed": bool(
            sum_ok and cdf_monotone and torch.isfinite(z_traj).all().item()
            and torch.isfinite(z_samp).all().item()
        ),
        "wall_time_s": round(time.time() - t0, 3),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("LENS smoke summary: %s", json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
