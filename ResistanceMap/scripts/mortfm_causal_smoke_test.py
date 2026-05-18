#!/usr/bin/env python3
"""
scripts/mortfm_causal_smoke_test.py
====================================
Phase B smoke — exercise the causal counterfactual stack on a small
subset of real biological-graph edges + real MMRF paired-patient z0
latents.

We do NOT train. We do NOT claim causal_mechanism. The smoke verifies:
  * InterventionGraph loads real edges from disk and supports do(...)
  * CounterfactualRunner produces finite Δ per edge
  * PathwayCausalValidator emits an honest "no claim from this run"
    verdict with negative-control statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.mortfm.causal import (
    CounterfactualRunner, InterventionGraph, PathwayCausalValidator,
)
from resistancemap.mortfm.trajectory import (
    GraphEnergyResistanceSDE, HittingTime, ResistanceBasin,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_causal_smoke")


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-edges", type=int, default=200,
                    help="How many edges to score in the smoke run.")
    ap.add_argument("--out", default="logs/mortfm/causal_smoke_summary.json")
    ap.add_argument("--out-csv", default="results/mortfm/causal_edge_effects.csv")
    args = ap.parse_args()

    # --- 1. Load real MMRF z0 latents ---------------------------------------
    z = np.load("data/processed/mmrf_paired_z64.npz", allow_pickle=True)
    z0 = torch.tensor(z["z0"][:8], dtype=torch.float32)            # 8 real patients
    B = z0.shape[0]
    logger.info("Loaded %d real MMRF z0 latents (d_latent=%d)", B, z0.shape[1])

    d_latent = z0.shape[1]
    d_graph, d_drug, d_clin = 16, 8, 4

    # --- 2. Build LENS SDE + basin + hitting --------------------------------
    sde = GraphEnergyResistanceSDE(
        d_latent=d_latent, d_graph=d_graph, d_drug=d_drug, d_clinical=d_clin,
        n_basins=5, integration_time=1.0, n_time_grid=8, n_mc_samples=4,
    )
    basin = ResistanceBasin(d_latent=d_latent, n_basins=5,
                             share_centroids_with=sde.potential)
    hitting = HittingTime(basin, resistant_basin_index=4, prob_threshold=0.3)

    # --- 3. InterventionGraph ----------------------------------------------
    g = InterventionGraph(d_graph=d_graph)
    if g.edges is None:
        logger.error("Biological edges parquet not on disk — aborting smoke.")
        return 1
    edge_ids = g.list_edges(limit=args.n_edges)
    logger.info("Will score %d edges (of %d total)", len(edge_ids), len(g.edges))

    # --- 4. Run counterfactuals --------------------------------------------
    runner = CounterfactualRunner(sde=sde, basin=basin, hitting=hitting, intervention_graph=g)
    drug = torch.zeros(B, d_drug)
    clinical = torch.zeros(B, d_clin)
    edge_df = runner.run(z0, drug, clinical, edge_ids)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    edge_df.to_csv(args.out_csv, index=False)
    logger.info("Wrote per-edge counterfactual effects -> %s", args.out_csv)

    # --- 5. Validator -------------------------------------------------------
    validator = PathwayCausalValidator()
    rep = validator.validate(edge_df)

    summary = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-causal-smoke",
        "n_real_patients_used": B,
        "n_edges_scored": rep.n_edges_scored,
        "median_delta": rep.median_delta,
        "top_k_drug_target_recall": rep.top_k_drug_target_recall,
        "negative_control_distribution": rep.negative_control_distribution,
        "causal_mechanism_gate_pass": bool(rep.causal_mechanism_gate_pass),
        "reasons_blocked": rep.reasons_blocked,
        "outputs": {"edge_effects_csv": args.out_csv},
        "wall_time_s": round(time.time() - t0, 2),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Causal smoke summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
