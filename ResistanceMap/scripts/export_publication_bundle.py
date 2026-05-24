#!/usr/bin/env python3
"""Export a reproducible publication bundle from pipeline artifacts.

This is the final Airflow task: it collects all evidence into a single
directory that a reviewer or journal submission system can consume.

The bundle includes provenance manifests, audit logs, metric tables,
figures, and SHA-256 checksums for every file.

Usage:
    python scripts/export_publication_bundle.py [--run-id RUN_ID] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# ── paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE_ROOT = REPO / "paper" / "publication_bundle"

# ── artifact sources (relative to REPO) ─────────────────────────────────────
# Each entry: (bundle_name, list_of_candidate_sources)
# The exporter tries each candidate in order and copies the first that exists.
ARTIFACT_SOURCES: list[tuple[str, list[str]]] = [
    (
        "run_manifest.json",
        ["run_manifest.json"],
    ),
    (
        "data_manifest.json",
        [
            "data_manifest.json",
            "paper/v8_artifacts/data_integration_audit.json",
        ],
    ),
    (
        "split_manifest.json",
        ["split_manifest.json"],
    ),
    (
        "leakage_audit.json",
        [
            "results/mortfm/leakage_audit.json",
            "logs/mortfm/leakage_audit.json",
        ],
    ),
    (
        "temporal_validity_audit.json",
        [
            "results/mortfm/temporal_validity_audit.json",
            "logs/mortfm/temporal_validity_audit.json",
        ],
    ),
    (
        "baseline_leaderboard.csv",
        [
            "results/mortfm/baseline_leaderboard.csv",
            "results/mortfm/survival_baseline_suite.csv",
        ],
    ),
    (
        "sota_comparison_table.json",
        [
            "paper/v8_artifacts/sota_benchmark/sota_comparison_table.json",
        ],
    ),
    (
        "per_drug_metrics.csv",
        [
            "logs/per_drug_metrics.csv",
            "results/mortfm/per_drug_metrics.csv",
            "logs/mortfm/beataml_per_drug_metrics.csv",
        ],
    ),
    (
        "survival_metrics.json",
        [
            "results/mortfm/survival_metrics.json",
            "logs/mortfm/survival_metrics.json",
        ],
    ),
    (
        "pathway_evidence.csv",
        [
            "results/mortfm/pathway_evidence.csv",
            "results/mortfm/causal_edge_evidence_v2.csv",
            "results/mortfm/causal_edge_evidence.csv",
        ],
    ),
    (
        "claim_gate_report.json",
        [
            "logs/mortfm/claim_gate_registry.json",
            "logs/mortfm/mmrf_claim_gate_report.json",
            "logs/mortfm/beataml_claim_gate_report.json",
            "logs/mortfm/block_a_claim_gate_report.json",
        ],
    ),
]

# Directories whose PNGs and PDFs should be copied into bundle/figures/
FIGURE_SOURCE_DIRS: list[str] = [
    "paper/v8_artifacts/sota_benchmark",
    "paper/v8_artifacts/visualizations",
]


# ── helpers ──────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file, streaming in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_head_sha(repo: Path) -> str:
    """Return the current HEAD commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_branch(repo: Path) -> str:
    """Return the current branch name, or 'detached'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "detached"
    except Exception:
        return "detached"


def _generate_run_id() -> str:
    """Generate a run ID from the current timestamp."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"pub-{ts}"


def _generate_run_manifest(repo: Path) -> dict[str, Any]:
    """Build a minimal run manifest from the current repo state."""
    return {
        "run_id": _generate_run_id(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_head_sha(repo),
        "git_branch": _git_branch(repo),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "note": "auto-generated by export_publication_bundle.py (no pre-existing run_manifest.json found)",
    }


def _load_or_generate_run_manifest(
    repo: Path, run_id_override: str | None
) -> dict[str, Any]:
    """Load run_manifest.json if present, otherwise generate one."""
    manifest_path = repo / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = _generate_run_manifest(repo)

    # Override run_id if provided via CLI
    if run_id_override:
        manifest["run_id"] = run_id_override

    # Ensure run_id key exists
    if "run_id" not in manifest:
        manifest["run_id"] = _generate_run_id()

    return manifest


def _extract_runs_md_entry(repo: Path, run_id: str) -> str | None:
    """Extract the RUNS.md row matching run_id, if any.

    If no exact match is found, return the last table row (most recent run).
    """
    runs_md = repo / "RUNS.md"
    if not runs_md.exists():
        return None

    text = runs_md.read_text()
    lines = text.splitlines()

    # Look for exact match first
    for line in lines:
        if run_id in line and line.strip().startswith("|"):
            return line

    # Fallback: return the last non-empty table row
    table_rows = [
        ln
        for ln in lines
        if ln.strip().startswith("|")
        and not ln.strip().startswith("| Run ID")
        and not ln.strip().startswith("|---")
    ]
    if table_rows:
        return table_rows[-1]
    return None


def _copy_file(src: Path, dst: Path) -> None:
    """Copy a single file, creating parent dirs as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _collect_figures(repo: Path, figures_dir: Path) -> int:
    """Copy PNGs and PDFs from figure source directories into figures_dir."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for rel_dir in FIGURE_SOURCE_DIRS:
        src_dir = repo / rel_dir
        if not src_dir.is_dir():
            continue
        for fpath in sorted(src_dir.iterdir()):
            if fpath.suffix.lower() in (".png", ".pdf"):
                _copy_file(fpath, figures_dir / fpath.name)
                copied += 1
    return copied


def _collect_claim_gate_reports(repo: Path, bundle_dir: Path) -> Path | None:
    """Collect all claim gate reports into a combined JSON.

    If multiple claim gate report files exist, merge them into a single
    combined document keyed by source filename.
    """
    gate_dir = repo / "logs" / "mortfm"
    if not gate_dir.is_dir():
        return None

    combined: dict[str, Any] = {}
    for fpath in sorted(gate_dir.iterdir()):
        if "claim_gate" in fpath.name and fpath.suffix == ".json":
            try:
                with open(fpath) as f:
                    combined[fpath.stem] = json.load(f)
            except (json.JSONDecodeError, OSError):
                combined[fpath.stem] = {"error": f"could not parse {fpath.name}"}

    if not combined:
        return None

    out = bundle_dir / "claim_gate_report.json"
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)
    return out


def _human_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} TiB"


def _summarise_claim_gates(bundle_dir: Path) -> str:
    """Return a one-line summary of claim gate pass/fail from the bundle."""
    cg = bundle_dir / "claim_gate_report.json"
    if not cg.exists():
        return "claim_gate_report.json: NOT FOUND"

    try:
        with open(cg) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "claim_gate_report.json: PARSE ERROR"

    passed = []
    failed = []

    def _walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            # Look for "pass" keys at this level
            if "pass" in obj:
                label = prefix or "gate"
                if obj["pass"]:
                    passed.append(label)
                else:
                    failed.append(label)
            # Check nested keys
            for k, v in obj.items():
                if k == "pass":
                    continue
                _walk(v, prefix=k if not prefix else f"{prefix}.{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, prefix=f"{prefix}[{i}]")

    _walk(data)
    return (
        f"claim gates: {len(passed)} PASS, {len(failed)} FAIL"
        f"  [pass: {', '.join(passed) or 'none'}]"
        f"  [fail: {', '.join(failed) or 'none'}]"
    )


# ── main export logic ───────────────────────────────────────────────────────


def export_bundle(
    run_id_override: str | None = None,
    out_dir_override: str | None = None,
) -> Path:
    """Collect all pipeline artifacts into a publication bundle directory.

    Returns the path to the bundle directory.
    """
    repo = REPO

    # 1. Load or generate run manifest
    manifest = _load_or_generate_run_manifest(repo, run_id_override)
    run_id = manifest["run_id"]

    # 2. Determine output directory
    if out_dir_override:
        bundle_root = Path(out_dir_override)
    else:
        bundle_root = DEFAULT_BUNDLE_ROOT
    bundle_dir = bundle_root / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    collected: list[str] = []  # relative paths within bundle
    missing: list[str] = []  # artifacts not found

    # 3a. Write run_manifest.json
    run_manifest_dst = bundle_dir / "run_manifest.json"
    with open(run_manifest_dst, "w") as f:
        json.dump(manifest, f, indent=2)
    collected.append("run_manifest.json")

    # 3b. Collect each artifact from candidate sources
    for bundle_name, candidates in ARTIFACT_SOURCES:
        # Skip run_manifest.json (already written above)
        if bundle_name == "run_manifest.json":
            continue

        # Special handling for claim_gate_report: combine all gate reports
        if bundle_name == "claim_gate_report.json":
            result = _collect_claim_gate_reports(repo, bundle_dir)
            if result and result.exists():
                collected.append(bundle_name)
            else:
                # Try individual candidates as fallback
                found = False
                for candidate in candidates:
                    src = repo / candidate
                    if src.exists():
                        _copy_file(src, bundle_dir / bundle_name)
                        collected.append(bundle_name)
                        found = True
                        break
                if not found:
                    missing.append(bundle_name)
            continue

        found = False
        for candidate in candidates:
            src = repo / candidate
            if src.exists():
                _copy_file(src, bundle_dir / bundle_name)
                collected.append(bundle_name)
                found = True
                break
        if not found:
            missing.append(bundle_name)

    # 3c. Collect figures
    figures_dir = bundle_dir / "figures"
    n_figures = _collect_figures(repo, figures_dir)
    if n_figures > 0:
        # Add each figure file to collected list
        for fig_path in sorted(figures_dir.iterdir()):
            collected.append(f"figures/{fig_path.name}")

    # 3d. Extract RUNS.md entry
    runs_entry = _extract_runs_md_entry(repo, run_id)
    if runs_entry:
        runs_dst = bundle_dir / "RUNS_entry.md"
        runs_dst.write_text(
            f"# RUNS.md entry for {run_id}\n\n{runs_entry}\n"
        )
        collected.append("RUNS_entry.md")

    # 4. Compute SHA-256 checksums for every file in the bundle
    checksums: dict[str, str] = {}
    total_size = 0
    for root, _dirs, files in os.walk(bundle_dir):
        for fname in sorted(files):
            fpath = Path(root) / fname
            rel = fpath.relative_to(bundle_dir).as_posix()
            # Skip the manifest itself (we write it last)
            if rel == "bundle_manifest.json":
                continue
            checksums[rel] = sha256_file(fpath)
            total_size += fpath.stat().st_size

    # 5. Write bundle_manifest.json
    bundle_manifest = {
        "run_id": run_id,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_head_sha(repo),
        "git_branch": _git_branch(repo),
        "n_files": len(checksums),
        "total_size_bytes": total_size,
        "total_size_human": _human_size(total_size),
        "collected": sorted(collected),
        "missing": sorted(missing),
        "checksums": dict(sorted(checksums.items())),
    }
    bundle_manifest_path = bundle_dir / "bundle_manifest.json"
    with open(bundle_manifest_path, "w") as f:
        json.dump(bundle_manifest, f, indent=2)

    # Include the manifest's own checksum (self-referential, but useful for
    # downstream integrity checks on the whole directory)
    bundle_manifest["checksums"]["bundle_manifest.json"] = sha256_file(
        bundle_manifest_path
    )

    # 6. Print summary
    print("=" * 72)
    print("PUBLICATION BUNDLE EXPORT SUMMARY")
    print("=" * 72)
    print(f"  Run ID:      {run_id}")
    print(f"  Bundle dir:  {bundle_dir}")
    print(f"  Files:       {len(checksums)}")
    print(f"  Total size:  {_human_size(total_size)}")
    print(f"  Collected:   {len(collected)}")
    print(f"  Missing:     {len(missing)}")
    if missing:
        print(f"  Missing artifacts:")
        for m in sorted(missing):
            print(f"    - {m}")
    print()
    print(f"  {_summarise_claim_gates(bundle_dir)}")
    print("=" * 72)

    return bundle_dir


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a reproducible publication bundle from pipeline artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/export_publication_bundle.py\n"
            "  python scripts/export_publication_bundle.py --run-id r-2026-05-23-final\n"
            "  python scripts/export_publication_bundle.py --out-dir /tmp/bundles\n"
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override the run ID (default: read from run_manifest.json or auto-generate).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Root directory for bundles "
            f"(default: {DEFAULT_BUNDLE_ROOT.relative_to(REPO)})."
        ),
    )
    args = parser.parse_args()
    export_bundle(run_id_override=args.run_id, out_dir_override=args.out_dir)


if __name__ == "__main__":
    main()
