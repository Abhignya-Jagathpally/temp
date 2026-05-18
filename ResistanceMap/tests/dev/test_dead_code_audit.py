"""
tests/dev/test_dead_code_audit.py
==================================
Contract test for the v18.5 dead-code audit.

Asserts:
  * the audit emits a JSON report with the documented top-level keys
  * the by_kind dict is the union of every finding's ``kind``
  * allowlisted paths are NOT reported
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_audit_runs_and_emits_report(tmp_path):
    out_path = tmp_path / "report.json"
    rc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "dev" / "dead_code_audit.py"),
            "--out", str(out_path),
            "--repo-root", str(REPO_ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, f"audit returned {rc.returncode}: {rc.stderr}"
    assert out_path.exists()

    payload = json.loads(out_path.read_text())
    for k in ("run_id", "n_files_scanned", "n_files_allowlisted",
              "by_kind", "findings", "wall_time_s"):
        assert k in payload, f"report missing key {k!r}"

    # by_kind is a histogram of findings.kind values
    counts = {}
    for f in payload["findings"]:
        counts[f["kind"]] = counts.get(f["kind"], 0) + 1
    assert counts == payload["by_kind"]


def test_audit_respects_allowlist():
    # v18.4 shims have unused-looking runpy delegation; they must be allowlisted
    payload_path = REPO_ROOT / "results" / "dev" / "dead_code_report.json"
    if not payload_path.exists():
        pytest.skip("audit has not been run yet")
    payload = json.loads(payload_path.read_text())
    flagged_files = {f["file"] for f in payload["findings"]}
    assert "scripts/mortfm/09_gate_revalidate.py" not in flagged_files
    assert "scripts/mortfm/__init__.py" not in flagged_files
    assert "resistancemap/__init__.py" not in flagged_files
