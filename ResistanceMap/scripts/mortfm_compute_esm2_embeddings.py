#!/usr/bin/env python3
"""
scripts/mortfm_compute_esm2_embeddings.py
=========================================
Block D — populate the ESM-2 embedding cache for the subset of UniProt
sequences that map to feature genes.

This is a thin orchestrator. All work is delegated to
:mod:`resistancemap.data.protein_sequence_cache`:

  * :func:`load_sequences` reads the UniProt human FASTA into a DataFrame.
  * :class:`ESMEmbeddingCache.compute` runs ESM-2 (via HuggingFace
    transformers) and the cache writes a single ``.pt`` keyed by
    ``uniprot_id``.
  * :func:`write_uniprot_coverage_report` writes the coverage report the
    sequence_aware claim gate reads.

Constraints by design:
  * If ``transformers`` is missing, the cache **raises** rather than silently
    writing zeros — keeps the run honest.
  * Only sequences for feature-gene UniProt IDs in
    ``data/processed/metadata/gene_protein_identifier_map.csv`` are embedded.
  * Default model is the small ESM-2 (``esm2_t6_8M_UR50D``) so the cache can
    be built on commodity GPUs / CPU. Pass ``--model facebook/esm2_t33_650M_UR50D``
    for the full-size variant.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.protein_sequence_cache import (
    ESMEmbeddingCache,
    load_or_compute_embeddings,
    load_sequences,
    write_uniprot_coverage_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_compute_esm2")


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniprot-dir", default="data/raw/uniprot")
    ap.add_argument("--identifier-map",
                    default="data/processed/metadata/gene_protein_identifier_map.csv")
    ap.add_argument("--string-proteins",
                    default="data/processed/graphs/protein_nodes_from_graph.csv",
                    help="STRING protein nodes (UniProt-IDs only) — for the "
                         "string_coverage_rate field of the report.")
    ap.add_argument("--cache-dir", default="data/processed/proteins")
    ap.add_argument("--model", default="facebook/esm2_t6_8M_UR50D",
                    help="HuggingFace model name. Default is the smallest ESM-2 "
                         "so the cache builds in minutes on CPU.")
    ap.add_argument("--max-sequences", type=int, default=None,
                    help="Optional: cap the number of sequences to embed for "
                         "smoke tests. None = all mapped feature-gene UniProt IDs.")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--report",
                    default="logs/mortfm/uniprot_coverage_report.json")
    args = ap.parse_args()

    # ---- 1. Sequences -----------------------------------------------------
    uniprot_df = load_sequences(args.uniprot_dir)
    logger.info("Loaded %d UniProt sequences", len(uniprot_df))

    # ---- 2. Restrict to the mapped feature-gene UniProt subset ------------
    target_uniprot_ids: set[str] = set()
    if Path(args.identifier_map).exists():
        idmap = pd.read_csv(args.identifier_map, low_memory=False)
        target_uniprot_ids = set(
            idmap[idmap.get("tier", 0) > 0]["uniprot_id"].dropna().astype(str)
        )
        logger.info("Feature-gene UniProt subset from identifier map: %d", len(target_uniprot_ids))
    if target_uniprot_ids:
        subset = uniprot_df[uniprot_df["uniprot_id"].isin(target_uniprot_ids)]
    else:
        logger.warning("No identifier map found; falling back to full UniProt set.")
        subset = uniprot_df
    if args.max_sequences:
        subset = subset.head(int(args.max_sequences))
    logger.info("Will embed %d sequences (subset).", len(subset))

    # ---- 3. Compute -------------------------------------------------------
    cache = ESMEmbeddingCache(model_name=args.model, max_len=args.max_len)
    cache.load(args.cache_dir)
    n_before = len(cache)
    cache.compute(subset)
    cache_path = cache.save(args.cache_dir)
    n_after = len(cache)
    logger.info("Cache: %d -> %d embeddings (added %d)", n_before, n_after, n_after - n_before)

    # ---- 4. Coverage report -----------------------------------------------
    string_proteins = None
    if Path(args.string_proteins).exists():
        sp_df = pd.read_csv(args.string_proteins)
        if "uniprot_id" in sp_df.columns:
            string_proteins = sp_df["uniprot_id"].astype(str).tolist()
    feature_genes = None
    if Path(args.identifier_map).exists():
        feature_genes = pd.read_csv(args.identifier_map)["feature_gene"].astype(str).tolist()
    rep = write_uniprot_coverage_report(
        uniprot_df=uniprot_df,
        string_proteins=string_proteins,
        feature_genes=feature_genes,
        embedding_cache_path=str(cache_path),
        embedding_model=args.model,
        out_json=args.report,
    )

    # ---- 5. Gate report ---------------------------------------------------
    # Use the harmonization-map coverage (which considers STRING aliases too)
    # rather than only UniProt's primary gene_symbol. The latter underestimates
    # coverage for genes whose canonical symbol differs from the UniProt entry's
    # GN= header (e.g. older HGNC aliases).
    identifier_map_coverage = None
    if Path(args.identifier_map).exists():
        idmap = pd.read_csv(args.identifier_map)
        identifier_map_coverage = float((idmap["tier"] > 0).sum()) / max(len(idmap), 1)
    sequence_aware_pass = (
        rep.has_embedding_cache
        and n_after >= 100  # bare minimum so claim gate sees a populated cache
        and (
            (identifier_map_coverage is not None and identifier_map_coverage >= 0.70)
            or (feature_genes is None)
        )
    )
    gate_report = {
        "n_uniprot_total": rep.n_uniprot_total,
        "n_string_proteins_mapped": rep.n_string_proteins_mapped,
        "string_coverage_rate": rep.string_coverage_rate,
        "n_feature_genes_mapped_uniprot_direct": rep.n_feature_genes_mapped,
        "feature_gene_coverage_uniprot_direct": rep.feature_gene_coverage_rate,
        "feature_gene_coverage_via_identifier_map": identifier_map_coverage,
        "embedding_model": args.model,
        "embedding_cache_path": str(cache_path),
        "n_embeddings_cached": n_after,
        "n_target_sequences": int(len(subset)),
        "sequence_aware_gate_pass": bool(sequence_aware_pass),
        "honest_note": (
            "feature_gene_coverage_uniprot_direct only counts genes whose "
            "primary HGNC symbol matches the UniProt GN= header; the "
            "identifier-map coverage (used for the gate) also matches via "
            "STRING aliases / previous symbols."
        ),
        "wall_time_s": round(time.time() - t0, 1),
    }
    Path("logs/mortfm").mkdir(parents=True, exist_ok=True)
    with open("logs/mortfm/block_d_summary.json", "w") as f:
        json.dump(gate_report, f, indent=2)
    logger.info("Block D summary -> logs/mortfm/block_d_summary.json")
    logger.info("Done in %.1fs", time.time() - t0)
    return 0 if sequence_aware_pass else 1


if __name__ == "__main__":
    sys.exit(main())
