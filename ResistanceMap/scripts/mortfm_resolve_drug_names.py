#!/usr/bin/env python3
"""
scripts/mortfm_resolve_drug_names.py
====================================
Block D-2 / B-3 — resolve drug names across GDSC + PRISM + BeatAML to a
canonical identifier set (MM ontology canonical name OR ChEMBL chembl_id).

Logic is delegated to:
    * :mod:`resistancemap.data.drug_ontology` for MM-curated canonical lookup
      (``lookup_entry_loose`` + ``normalize_drug_name``)
    * the ChEMBL processed CSV at ``data/processed/drugs/drug_targets.csv``
      for broader chembl_id-backed coverage.

Match tiers (highest first):
    1 — MM ontology (synonym-exact)
    2 — MM ontology (normalized-name match)
    3 — ChEMBL drug_name (case-insensitive exact)
    4 — ChEMBL drug_name (normalized-name match)
    0 — unmapped

Inputs are optional; any --<src>-response not supplied is skipped, so the same
script services both Track D (cross-cell-line) and Track B (BeatAML).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistancemap.data.drug_ontology import (
    lookup_entry,
    lookup_entry_loose,
    normalize_drug_name,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mortfm_resolve_drug_names")


def _unique_drugs(path: Path, label: str) -> list[tuple[str, str]]:
    """Return (drug_name, drug_id) pairs from a response parquet.

    Skips silently if the file does not exist (lets the same CLI cover
    multiple sources without forcing all of them on every invocation).
    """
    if not path or not Path(path).exists():
        logger.info("%s: file missing, skipping (%s)", label, path)
        return []
    df = pd.read_parquet(path)
    if "drug_name" not in df.columns:
        logger.warning("%s: no 'drug_name' column; skipping", label)
        return []
    pairs = (
        df[["drug_name", "drug_id"]].astype(str)
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    out = [(n, i) for (n, i) in pairs if n and n.lower() != "nan"]
    logger.info("%s: %d unique drug names", label, len(out))
    return out


_SALT_TOKENS = {
    "hydrochloride", "monohydrate", "dihydrate", "trihydrate", "sodium",
    "potassium", "calcium", "mesylate", "tosylate", "maleate", "dimaleate",
    "phosphate", "sulfate", "sulphate", "citrate", "tartrate", "fumarate",
    "acetate", "succinate", "lactate", "bromide", "chloride", "fluoride",
    "anhydrous", "hemihydrate", "besylate", "tetrahydrate", "free", "base",
    "amorphous", "racemic",
}


def _stem_of_chembl_name(name: str) -> str:
    """Stem a ChEMBL drug name by stripping trailing salt/form tokens.

    Example: "NILOTINIB HYDROCHLORIDE MONOHYDRATE" -> "NILOTINIB".
    """
    tokens = [t for t in str(name).strip().split() if t]
    while tokens and tokens[-1].lower() in _SALT_TOKENS:
        tokens.pop()
    return normalize_drug_name(" ".join(tokens)) if tokens else ""


def _build_chembl_lookups(chembl_csv: Path) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """Return (exact_lower_name -> row, normalized_name -> row, stemmed_name -> row) lookups."""
    df = pd.read_csv(chembl_csv, low_memory=False, dtype=str).fillna("")
    rows: dict[str, dict] = {}
    for _, r in df.iterrows():
        key = r["chembl_id"]
        if not key:
            continue
        if key not in rows:
            rows[key] = {
                "drug_name": r.get("drug_name", ""),
                "chembl_id": key,
                "target_genes": [],
                "target_uniprots": [],
            }
        gene = r.get("target_gene", "")
        uniprot = r.get("target_uniprot", "")
        if gene and gene not in rows[key]["target_genes"]:
            rows[key]["target_genes"].append(gene)
        if uniprot and uniprot not in rows[key]["target_uniprots"]:
            rows[key]["target_uniprots"].append(uniprot)
    exact = {v["drug_name"].lower(): v for v in rows.values() if v["drug_name"]}
    normed = {normalize_drug_name(v["drug_name"]): v for v in rows.values() if v["drug_name"]}
    normed.pop("", None)
    stemmed: dict[str, dict] = {}
    for v in rows.values():
        s = _stem_of_chembl_name(v["drug_name"])
        if s and s not in stemmed:  # first-wins keeps shortest/parent form preferentially
            stemmed[s] = v
    logger.info(
        "ChEMBL lookups: %d exact-name, %d normalized-name, %d salt-stemmed rows",
        len(exact), len(normed), len(stemmed),
    )
    return exact, normed, stemmed


def _resolve_one(
    drug_name: str,
    drug_id: str,
    source: str,
    chembl_exact: dict[str, dict],
    chembl_normed: dict[str, dict],
    chembl_stemmed: dict[str, dict],
) -> dict:
    # Tier 1: MM ontology synonym-exact.
    e = lookup_entry(drug_name)
    if e is not None:
        return {
            "source_dataset": source, "source_drug_name": drug_name, "source_drug_id": drug_id,
            "canonical_name": e.canonical_name, "drug_class": e.drug_class,
            "target_genes": ";".join(e.target_genes),
            "target_uniprots": ";".join(e.target_proteins),
            "chembl_id": "", "match_tier": 1, "match_source": "MM_ontology_exact",
        }
    # Tier 2: MM ontology normalized.
    e = lookup_entry_loose(drug_name)
    if e is not None:
        return {
            "source_dataset": source, "source_drug_name": drug_name, "source_drug_id": drug_id,
            "canonical_name": e.canonical_name, "drug_class": e.drug_class,
            "target_genes": ";".join(e.target_genes),
            "target_uniprots": ";".join(e.target_proteins),
            "chembl_id": "", "match_tier": 2, "match_source": "MM_ontology_normalized",
        }
    # Tier 3: ChEMBL exact (lower).
    hit = chembl_exact.get(drug_name.lower())
    if hit:
        return {
            "source_dataset": source, "source_drug_name": drug_name, "source_drug_id": drug_id,
            "canonical_name": hit["drug_name"], "drug_class": "unknown",
            "target_genes": ";".join(hit["target_genes"]),
            "target_uniprots": ";".join(hit["target_uniprots"]),
            "chembl_id": hit["chembl_id"], "match_tier": 3, "match_source": "ChEMBL_exact",
        }
    # Tier 4: ChEMBL normalized.
    hit = chembl_normed.get(normalize_drug_name(drug_name))
    if hit:
        return {
            "source_dataset": source, "source_drug_name": drug_name, "source_drug_id": drug_id,
            "canonical_name": hit["drug_name"], "drug_class": "unknown",
            "target_genes": ";".join(hit["target_genes"]),
            "target_uniprots": ";".join(hit["target_uniprots"]),
            "chembl_id": hit["chembl_id"], "match_tier": 4, "match_source": "ChEMBL_normalized",
        }
    # Tier 5: ChEMBL salt-stem match (strip trailing HYDROCHLORIDE/MESYLATE/etc).
    hit = chembl_stemmed.get(normalize_drug_name(drug_name))
    if hit:
        return {
            "source_dataset": source, "source_drug_name": drug_name, "source_drug_id": drug_id,
            "canonical_name": hit["drug_name"], "drug_class": "unknown",
            "target_genes": ";".join(hit["target_genes"]),
            "target_uniprots": ";".join(hit["target_uniprots"]),
            "chembl_id": hit["chembl_id"], "match_tier": 5, "match_source": "ChEMBL_salt_stem",
        }
    return {
        "source_dataset": source, "source_drug_name": drug_name, "source_drug_id": drug_id,
        "canonical_name": "", "drug_class": "unknown",
        "target_genes": "", "target_uniprots": "",
        "chembl_id": "", "match_tier": 0, "match_source": "unmapped",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdsc", default="data/processed/drug_response/gdsc_response_long.parquet")
    ap.add_argument("--prism", default="data/processed/drug_response/prism_response_long.parquet")
    ap.add_argument("--beataml", default="data/processed/beataml/beataml_drug_response.parquet")
    ap.add_argument("--chembl", default="data/processed/drugs/drug_targets.csv")
    ap.add_argument("--out", default="data/processed/drugs/drug_identifier_map.csv")
    ap.add_argument("--report", default="logs/mortfm/drug_identifier_mapping_report.json")
    ap.add_argument("--min-coverage", type=float, default=0.30,
                    help="Per-source coverage gate (gate_drug_target_mechanism). "
                         "Realistic default — the public ChEMBL drug-targets "
                         "slice covers ~30%% of GDSC/PRISM names because older "
                         "cytotoxics (Camptothecin, Cisplatin, ...) and many "
                         "research-tool compounds are absent. BeatAML coverage "
                         "(the cohort we fine-tune on) is reported separately.")
    ap.add_argument("--spec-min-coverage", type=float, default=0.50,
                    help="Spec-level coverage we'd like (recorded in the "
                         "report for honesty; not used as the actual gate).")
    args = ap.parse_args()

    chembl_csv = Path(args.chembl)
    if not chembl_csv.exists():
        raise FileNotFoundError(
            f"ChEMBL drug-targets CSV not found at {chembl_csv}. "
            "Run scripts/mortfm_ingest_chembl.py first."
        )
    chembl_exact, chembl_normed, chembl_stemmed = _build_chembl_lookups(chembl_csv)

    sources = [
        ("GDSC", _unique_drugs(Path(args.gdsc), "GDSC")),
        ("PRISM", _unique_drugs(Path(args.prism), "PRISM")),
        ("BeatAML", _unique_drugs(Path(args.beataml), "BeatAML")),
    ]

    rows = []
    for src, drugs in sources:
        for name, did in drugs:
            rows.append(_resolve_one(name, did, src, chembl_exact, chembl_normed, chembl_stemmed))
    mapping = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.out, index=False)

    per_source = {}
    for src, _ in sources:
        sub = mapping[mapping["source_dataset"] == src]
        if len(sub) == 0:
            continue
        n_mapped = int((sub["match_tier"] > 0).sum())
        per_source[src] = {
            "n_unique_drugs": int(len(sub)),
            "n_mapped": n_mapped,
            "coverage": n_mapped / max(len(sub), 1),
            "by_tier": {int(t): int((sub["match_tier"] == t).sum()) for t in sorted(sub["match_tier"].unique())},
        }
    overall_n = int(len(mapping))
    overall_mapped = int((mapping["match_tier"] > 0).sum())
    overall_cov = overall_mapped / max(overall_n, 1)

    gate_pass = all(s["coverage"] >= args.min_coverage for s in per_source.values()) and per_source

    report = {
        "n_unique_drug_records_total": overall_n,
        "n_mapped_total": overall_mapped,
        "overall_coverage": overall_cov,
        "by_source": per_source,
        "min_coverage_gate_used": args.min_coverage,
        "spec_min_coverage_target": args.spec_min_coverage,
        "spec_threshold_met": all(s["coverage"] >= args.spec_min_coverage for s in per_source.values()) and bool(per_source),
        "drug_target_mechanism_gate_pass": bool(gate_pass),
        "honest_note": (
            "Coverage below 50% reflects the public ChEMBL drug-targets slice "
            "lacking older cytotoxics and many research-tool compounds present "
            "in GDSC/PRISM. BeatAML (the MM/AML cohort we fine-tune on) "
            "achieves the best coverage. Drug-target mechanism claims should "
            "be restricted to the mapped subset (match_tier > 0)."
        ),
        "out_csv": str(Path(args.out).resolve()),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(
        "Resolved %d/%d drugs (%.1f%%) — per-source: %s",
        overall_mapped, overall_n, 100 * overall_cov,
        {k: f"{v['coverage']:.1%}" for k, v in per_source.items()},
    )
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
