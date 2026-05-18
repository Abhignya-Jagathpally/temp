#!/usr/bin/env python3
"""
scripts/mortfm_v17_unified_smoke.py
====================================
v17.4 — verify that resistancemap/mortfm/model.py:MORTFM exposes BOTH the
legacy forward() path AND the new forward_lens() path, that both run on
the same MORTBatch end-to-end on real MMRF z0 latents, and that the
v16 LENS submodules are first-class attributes of the model.

This is a contract test for the v17 unification: after this commit,
new training scripts can call model.forward_lens(batch, clinical=..., drug=...)
and obtain hazards / basin probs / hitting-time CDF in one call.

No new training. No new data. No fabrication.
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

from resistancemap.mortfm.longitudinal.clinical_features import encode_for_lens
from resistancemap.mortfm.model import MORTFM
from resistancemap.mortfm.schemas import MORTBatch, MORTFMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_v17_unified_smoke")


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/mortfm/v17_unified_smoke.json")
    args = ap.parse_args()

    # Real MMRF z0 latents — n=8 patients for the smoke
    z = np.load("data/processed/mmrf_paired_z64.npz", allow_pickle=True)
    pids = [str(p) for p in z["patient_id"][:8]]
    z0 = torch.tensor(z["z0"][:8], dtype=torch.float32)        # (8, 64)
    B, d_latent = z0.shape
    logger.info("Loaded %d real MMRF z0 latents (d=%d)", B, d_latent)

    enc = encode_for_lens(
        "data/processed/mmrf_outcomes_treatment.tsv",
        patient_ids=pids,
    )
    clinical = torch.tensor(enc.features, dtype=torch.float32)
    drug = torch.zeros(B, 8)

    cfg = MORTFMConfig(
        rna_input_dim=d_latent,
        d_token=32, d_latent=d_latent,
        n_time_grid=8, integration_time=1.0,
        use_rna=True, use_proteomics=False, use_atac=False, use_methylation=False,
        use_histone_ptm=False, use_phosphoproteomics=False,
        use_clinical=False, use_drug=False,
        survival_n_bins=8,
    )
    model = MORTFM(cfg)
    model.eval()
    logger.info("MORTFM built — checking v16 LENS submodules ...")
    for name in ["lens_sde", "lens_basin", "lens_hitting",
                  "lens_graph_projector", "lens_competing_risk"]:
        assert hasattr(model, name), f"missing v16 submodule {name}"
        logger.info("  %s: %s", name, type(getattr(model, name)).__name__)

    # Build a minimal MORTBatch with RNA only.
    batch = MORTBatch(
        rna=z0, drug=drug,
        modality_mask={
            "rna": torch.ones(B, dtype=torch.bool),
            "drug": torch.ones(B, dtype=torch.bool),
        },
        patient_ids=pids,
        time=torch.zeros(B),
    )

    # --- 1. Legacy forward_legacy() path (v18: renamed from forward()) ---
    with torch.no_grad():
        legacy = model.forward_legacy(batch, n_traj_samples=4)
    legacy_ok = (
        legacy.z_path is not None and legacy.resistance_state_logits is not None
        and torch.isfinite(legacy.z_path).all()
    )
    logger.info("forward_legacy(): z_path shape=%s, finite=%s",
                list(legacy.z_path.shape), bool(legacy_ok))

    # --- 2. CANONICAL v18 forward() path (was forward_lens() in v17) -----
    with torch.no_grad():
        lens = model(batch, clinical=clinical, drug=drug)
    lens_ok = (
        torch.isfinite(lens["z_traj"]).all().item()
        and torch.isfinite(lens["hazard"]).all().item()
        and torch.allclose(lens["basin_probs"].sum(dim=-1),
                           torch.ones(B, lens["basin_probs"].shape[1]), atol=1e-4)
    )
    logger.info(
        "forward_lens(): z_traj=%s, z_samples=%s, hazard=%s, basin_probs=%s, "
        "hitting_cdf=%s, finite+sumto1=%s",
        list(lens["z_traj"].shape), list(lens["z_samples"].shape),
        list(lens["hazard"].shape), list(lens["basin_probs"].shape),
        list(lens["hitting_cdf"].shape), bool(lens_ok),
    )

    # --- 3. Sanity: clinical-only vs graph-projector ablation -----------
    with torch.no_grad():
        no_graph = model(batch, clinical=clinical, drug=drug,
                          use_graph_projector=False)
    delta_haz = float((lens["hazard"] - no_graph["hazard"]).abs().mean().item())
    logger.info("Mean |hazard delta| projector-on vs off: %.4f", delta_haz)

    out = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-v17-unified-smoke",
        "n_patients": int(B),
        "d_latent": int(d_latent),
        "legacy_forward_pass": bool(legacy_ok),
        "lens_forward_pass": bool(lens_ok),
        "v16_lens_submodules_present": [
            "lens_sde", "lens_basin", "lens_hitting",
            "lens_graph_projector", "lens_competing_risk",
        ],
        "shapes": {
            "z_traj": list(lens["z_traj"].shape),
            "z_samples": list(lens["z_samples"].shape),
            "hazard": list(lens["hazard"].shape),
            "basin_probs": list(lens["basin_probs"].shape),
            "hitting_cdf": list(lens["hitting_cdf"].shape),
        },
        "projector_effect_on_hazard": delta_haz,
        "verdict": "PASS" if (legacy_ok and lens_ok) else "FAIL",
        "wall_time_s": round(time.time() - t0, 2),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Unified smoke verdict: %s", out["verdict"])
    logger.info("Summary: %s", json.dumps(out, indent=2))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
