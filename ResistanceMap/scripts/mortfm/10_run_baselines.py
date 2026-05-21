#!/usr/bin/env python
"""scripts/mortfm/10_run_baselines.py — Phase 10 baseline-runner skeleton.

CLI orchestrator that runs every registered baseline on the SAME splits
as ``06_train_lens_resistance.py`` / ``07_train_survival.py`` and appends
``ComparisonRow`` records to ``results/baseline_comparison.parquet``.

This script is scaffolding only: in ``--dry-run`` mode it lists the
baselines and splits it WOULD run, WITHOUT actually fitting or scoring
anything. That lets us verify the wiring (registry import, splits manifest,
output path) on a workstation before launching the real run on a GPU node.

The actual fit / score / append-row logic is intentionally left as TODO
because:
  * It depends on data loaders that vary per cohort, and
  * The Phase 10 doc gates the inclusion of any synthetic-fallback path.

USAGE
-----

    # Dry run — only print the plan.
    python scripts/mortfm/10_run_baselines.py \\
        --config configs/baselines/mmrf_ia22.yaml \\
        --baselines elasticnet ridge mlp lstm_visits \\
        --splits data/mmrf_ia22/splits/patient_holdout_v18.json \\
        --out-parquet results/baseline_comparison.parquet \\
        --dry-run

When run without ``--dry-run`` the entry-point currently exits with a
``NotImplementedError`` and tells the user where to wire the cohort
loader. This is by design — see ``ResistanceMap/docs/v18_phase_plan/
phase_10_baselines.md`` §5.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Importing the baseline modules has the side effect of populating REGISTRY.
import resistancemap.baselines.deep_static          # noqa: F401  registers
import resistancemap.baselines.sequence             # noqa: F401  registers
import resistancemap.baselines.static_ml            # noqa: F401  registers
from resistancemap.baselines.registry import REGISTRY

logger = logging.getLogger("mortfm.10_run_baselines")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config", type=Path, required=True,
        help="YAML config describing the cohort (paths, loaders, modalities).",
    )
    p.add_argument(
        "--baselines", nargs="+", required=True,
        help="Registered baseline names; use 'all' to run every baseline in REGISTRY.",
    )
    p.add_argument(
        "--splits", type=Path, required=True,
        help="JSON manifest with {'train': [...], 'val': [...], 'test': [...]} patient IDs.",
    )
    p.add_argument(
        "--out-parquet", type=Path, default=Path("results/baseline_comparison.parquet"),
        help="Destination parquet (will be appended-to if it exists).",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
        help="RNG seeds to fit each baseline with (one ComparisonRow per seed).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and exit without fitting any model.",
    )
    return p.parse_args(argv)


def _resolve_baselines(requested: List[str]) -> List[str]:
    if len(requested) == 1 and requested[0].lower() == "all":
        return REGISTRY.list_available()
    unknown = [b for b in requested if b not in REGISTRY]
    if unknown:
        available = REGISTRY.list_available()
        raise SystemExit(
            f"Unknown baselines requested: {unknown}\n"
            f"Available: {available}"
        )
    return list(requested)


def _load_splits(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"--splits manifest not found: {path}")
    payload = json.loads(path.read_text())
    for required in ("train", "val", "test"):
        if required not in payload:
            raise SystemExit(
                f"--splits manifest {path} is missing required key {required!r}"
            )
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    baselines = _resolve_baselines(args.baselines)
    splits = _load_splits(args.splits)

    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])

    plan_lines = [
        "=" * 70,
        "Phase 10 baseline runner — execution plan",
        "=" * 70,
        f"  config:          {args.config}",
        f"  splits manifest: {args.splits}  (n_train={n_train}, n_val={n_val}, n_test={n_test})",
        f"  output parquet:  {args.out_parquet}",
        f"  seeds:           {args.seeds}",
        f"  baselines:       {baselines}",
        f"  total jobs:      {len(baselines) * len(args.seeds)}",
        "=" * 70,
    ]
    for line in plan_lines:
        print(line)

    if args.dry_run:
        print("[dry-run] no models will be fitted; exiting.")
        return 0

    # Real-run guard: the cohort loader is not part of this scaffolding.
    raise NotImplementedError(
        "10_run_baselines.py: the cohort-loader -> X_train/y_train/X_test/y_test "
        "pipeline is not yet wired. See ResistanceMap/docs/v18_phase_plan/"
        "phase_10_baselines.md §5 for the contract the loader must satisfy. "
        "Use --dry-run to verify the registry + splits wiring meanwhile."
    )


if __name__ == "__main__":
    sys.exit(main())
