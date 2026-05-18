#!/usr/bin/env python3
"""
scripts/mortfm_integrate_blocks_bcd.py
======================================
Block E — integrate Blocks A + B + C + D into a single, provenance-tagged
checkpoint.

We do NOT re-train here. The integration step:
  1. Loads the Block A cell-line foundation checkpoint.
  2. Overlays the Block B BeatAML-fine-tuned encoder/heads where they have
     drifted (BeatAML-specific weights win — they are downstream of A).
  3. Records a pointer to the Block C single-cell state-encoder checkpoint
     (kept separate because it operates on a different RNA feature space;
     the integrator records the feature_alignment hash so callers can route
     the right input to the right encoder).
  4. Records a pointer to the Block D ESM-2 embedding cache.
  5. Emits a provenance manifest covering ALL four blocks and the gate
     reports they produced.

The output is a single ``.pt`` payload + a JSON manifest. Downstream Block-F
training / inference scripts load the manifest, not raw checkpoints.

Honest behaviour
----------------
* If a block's checkpoint or gate-report is missing, the integrator records
  ``status: "missing"`` for that block but still emits a manifest, so the
  failure mode is visible.
* It never copies weights between blocks whose shapes differ. Shape
  mismatches are listed under ``shape_mismatches`` in the manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_integrate_blocks")


def _read_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _summarise_block(name: str, ckpt: Path | None, report: dict | None) -> dict:
    info = {"name": name, "status": "missing"}
    if ckpt and ckpt.exists():
        size = ckpt.stat().st_size
        try:
            payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
            n_params = (
                sum(v.numel() for v in payload["model_state_dict"].values())
                if isinstance(payload, dict) and "model_state_dict" in payload
                else None
            )
        except Exception as e:
            cfg, n_params = {}, None
            logger.warning("Failed to read %s: %s", ckpt, e)
        info.update({
            "status": "present", "checkpoint": str(ckpt.resolve()),
            "size_bytes": int(size), "n_params": n_params, "config": cfg,
        })
    if report:
        info["report"] = report
    return info


def _overlay_b_on_a(
    a_state: dict, b_state: dict,
) -> tuple[dict, list[str], list[tuple[str, list, list]]]:
    """Return (merged_state_dict, overlayed_keys, shape_mismatches).

    Block B takes precedence where shapes match; Block A retains responsibility
    for keys present only in A. Keys only in B are kept too.
    """
    merged = dict(a_state)
    overlayed: list[str] = []
    mismatches: list[tuple[str, list, list]] = []
    for k, v in b_state.items():
        if k in merged and merged[k].shape != v.shape:
            mismatches.append((k, list(merged[k].shape), list(v.shape)))
            continue
        merged[k] = v
        overlayed.append(k)
    return merged, overlayed, mismatches


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-a", default="checkpoints/mortfm/block_a_cellline_foundation.pt")
    ap.add_argument("--block-b", default="checkpoints/mortfm/beataml_block_b_finetuned.pt")
    ap.add_argument("--block-c", default="checkpoints/mortfm/block_c_state_encoder.pt")
    ap.add_argument("--block-d-cache", default="data/processed/proteins/esm_embeddings__facebook__esm2_t6_8M_UR50D.pt")
    ap.add_argument("--block-a-report", default=None,
                    help="Optional Block A summary JSON. Auto-discovered from logs/mortfm/ "
                         "if omitted.")
    ap.add_argument("--block-b-report", default="logs/mortfm/beataml_block_b_summary.json")
    ap.add_argument("--block-c-report", default="logs/mortfm/block_c_summary.json")
    ap.add_argument("--block-d-report", default="logs/mortfm/block_d_summary.json")
    ap.add_argument("--out-checkpoint",
                    default="checkpoints/mortfm/mortfm_integrated.pt")
    ap.add_argument("--out-manifest",
                    default="checkpoints/mortfm/mortfm_integrated_manifest.json")
    args = ap.parse_args()

    a_ckpt, b_ckpt = Path(args.block_a), Path(args.block_b)
    c_ckpt, d_cache = Path(args.block_c), Path(args.block_d_cache)

    if not a_ckpt.exists():
        logger.error("Block A checkpoint not found at %s; integration cannot proceed.", a_ckpt)
        return 1

    # ---- Block A weights are the base ----------------------------------------
    a_payload = torch.load(str(a_ckpt), map_location="cpu", weights_only=False)
    merged_state = dict(a_payload["model_state_dict"])
    overlayed_keys, mismatches = [], []

    if b_ckpt.exists():
        b_payload = torch.load(str(b_ckpt), map_location="cpu", weights_only=False)
        merged_state, overlayed_keys, mismatches = _overlay_b_on_a(
            merged_state, b_payload["model_state_dict"],
        )
        logger.info("Overlaid Block B onto A: %d keys overlayed, %d shape mismatches",
                    len(overlayed_keys), len(mismatches))
    else:
        logger.warning("Block B checkpoint missing; integrated model uses Block A only.")

    integrated_payload = {
        "model_state_dict": merged_state,
        "config": a_payload.get("config", {}),
        "integration_metadata": {
            "block_a_checkpoint": str(a_ckpt.resolve()),
            "block_b_checkpoint": str(b_ckpt.resolve()) if b_ckpt.exists() else None,
            "block_b_overlayed_keys_n": len(overlayed_keys),
            "block_b_shape_mismatches": mismatches[:50],
            "block_c_checkpoint": str(c_ckpt.resolve()) if c_ckpt.exists() else None,
            "block_d_embedding_cache": str(d_cache.resolve()) if d_cache.exists() else None,
            "note": (
                "Block C operates on a different RNA feature space (HVG union, "
                "~3,500 genes) than Blocks A/B (variance-top-2000 from DepMap). "
                "It is kept as a SEPARATE pointer in the manifest. Block-F "
                "training scripts should load the appropriate encoder based on "
                "the input feature space."
            ),
        },
    }
    Path(args.out_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(integrated_payload, args.out_checkpoint)
    logger.info("Saved integrated checkpoint -> %s", args.out_checkpoint)

    # ---- Provenance manifest -------------------------------------------------
    a_report = _read_json(Path(args.block_a_report)) if args.block_a_report else None
    if a_report is None:
        for cand in [
            Path("logs/mortfm/block_a_cellline_foundation_summary.json"),
            Path("logs/mortfm/block_a_summary.json"),
        ]:
            r = _read_json(cand)
            if r is not None:
                a_report = r
                break

    manifest = {
        "run_id": f"r-{time.strftime('%Y-%m-%d')}-mortfm-block-e-integrated",
        "integrated_checkpoint": str(Path(args.out_checkpoint).resolve()),
        "blocks": {
            "A_cellline_foundation": _summarise_block("A", a_ckpt, a_report),
            "B_beataml_finetuned":   _summarise_block("B", b_ckpt, _read_json(Path(args.block_b_report))),
            "C_scrna_state_encoder": _summarise_block("C", c_ckpt, _read_json(Path(args.block_c_report))),
            "D_esm2_embeddings": {
                "name": "D",
                "status": "present" if d_cache.exists() else "missing",
                "embedding_cache": str(d_cache.resolve()) if d_cache.exists() else None,
                "size_bytes": int(d_cache.stat().st_size) if d_cache.exists() else None,
                "report": _read_json(Path(args.block_d_report)),
            },
        },
        "overlay": {
            "n_keys_overlayed_b_on_a": len(overlayed_keys),
            "n_shape_mismatches": len(mismatches),
            "first_mismatches": mismatches[:5],
        },
        "wall_time_s": round(time.time() - t0, 1),
    }
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Wrote integration manifest -> %s", args.out_manifest)

    # Summarise to stdout for human eyeballing.
    statuses = {k: v.get("status") for k, v in manifest["blocks"].items()}
    logger.info("Block statuses: %s", statuses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
