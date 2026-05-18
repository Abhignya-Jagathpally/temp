#!/usr/bin/env python3
"""
scripts/mortfm_align_beataml_features.py
========================================
Block B-2 — align the BeatAML expression matrix onto the Block-A checkpoint's
top-2000 RNA feature space.

Re-uses :func:`resistancemap.data.feature_alignment.align_to_reference_features`
— no logic duplication.

Inputs:
    --block-a-checkpoint   Block A .pt (config + state dict; we read config.rna_input_dim
                            and re-derive feature_names from the cell-line summary).
    --feature-names        OR a CSV with one feature_name per row.
    --beataml-expression   data/processed/beataml/beataml_expression.parquet

Outputs:
    --out                  data/processed/beataml/beataml_expression_block_a_features.parquet
    --report               logs/mortfm/beataml_feature_alignment_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.feature_alignment import align_to_reference_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_align_beataml")


def _features_from_block_a(ckpt_path: Path) -> list[str]:
    """Recover the exact top-2000 var-genes Block A trained on.

    Block A's training script (``mortfm_train_cellline_foundation.py``)
    selected features deterministically as ``rna.var().nlargest(2000)`` of the
    DepMap expression matrix. We replay that selection here.
    """
    rna = pd.read_parquet("data/processed/omics/rna_matrix.parquet")
    rna = rna.loc[~rna.index.duplicated(keep="first")]
    # The training script subsetted to the overlap with GDSC first; for
    # alignment we only need the gene order, which is variance-based on the
    # *training overlap*. Reproduce the same overlap selection.
    gdsc = pd.read_parquet("data/processed/drug_response/gdsc_response_long.parquet")
    gdsc = gdsc[gdsc["response_metric"] == "ln_IC50"].dropna(subset=["response_value"])
    from resistancemap.data.model_id_resolver import ModelIDResolver
    meta = pd.read_csv("data/processed/metadata/model_metadata.csv")
    resolver = ModelIDResolver(meta)
    gdsc["resolved_model_id"] = gdsc["model_id"].astype(str).map(resolver.to_model_id)
    overlap_models = sorted(
        set(rna.index.astype(str))
        & set(gdsc["resolved_model_id"].dropna().astype(str))
    )
    rna_overlap = rna.loc[overlap_models]
    cfg = torch.load(str(ckpt_path), map_location="cpu", weights_only=False).get("config", {})
    n_genes = int(cfg.get("rna_input_dim", 2000))
    variances = rna_overlap.var(axis=0).sort_values(ascending=False)
    return variances.index[:n_genes].tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-a-checkpoint", default="checkpoints/mortfm/block_a_cellline_foundation.pt")
    ap.add_argument("--feature-names", default=None)
    ap.add_argument("--beataml-expression",
                    default="data/processed/beataml/beataml_expression.parquet")
    ap.add_argument("--out",
                    default="data/processed/beataml/beataml_expression_block_a_features.parquet")
    ap.add_argument("--mask-out",
                    default="data/processed/beataml/beataml_feature_mask.parquet")
    ap.add_argument("--report",
                    default="logs/mortfm/beataml_feature_alignment_report.json")
    args = ap.parse_args()

    if args.feature_names:
        feats = pd.read_csv(args.feature_names).iloc[:, 0].astype(str).tolist()
    else:
        feats = _features_from_block_a(Path(args.block_a_checkpoint))
    logger.info("Reference feature set: %d genes (recovered from Block A checkpoint)", len(feats))

    source = pd.read_parquet(args.beataml_expression)
    aligned, mask, rep = align_to_reference_features(source, feats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(args.out)
    mask.to_parquet(args.mask_out)
    logger.info("Wrote aligned expression -> %s; mask -> %s", args.out, args.mask_out)

    report = asdict(rep)
    report["beataml_specimens"] = int(aligned.shape[0])
    report["block_a_gate_pass"] = rep.coverage_rate >= 0.80
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Feature alignment %s: %d/%d (%.1f%%); reportwritten to %s",
                "PASS" if report["block_a_gate_pass"] else "FAIL",
                rep.n_features_present, rep.n_reference_features, 100 * rep.coverage_rate,
                args.report)
    return 0 if report["block_a_gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
