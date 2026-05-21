"""tests/mortfm/test_eval_causal_script_stub.py

Smoke test for Bug B5: the v18 workflow shim
``scripts/mortfm/08_eval_causal_evidence_v2.py`` previously exited with
code 2 (indistinguishable from an argparse error) when the delegated
target was missing. The fix exits 1 with an actionable stderr message,
OR — when the target exists — successfully shows its --help.

The contract this test enforces is therefore: exit code is in {0, 1}
(NEVER 2) and, when the target is missing, stderr carries a Phase-8
hint pointing to the planning doc.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "mortfm" / "08_eval_causal_evidence_v2.py"
DELEGATED_TARGET = REPO_ROOT / "scripts" / "mortfm_causal_evidence_report_v2.py"


def test_script_exists():
    assert SCRIPT.exists(), f"Workflow shim missing: {SCRIPT}"


def test_script_help_exits_with_code_0_or_1_never_2():
    """`08 --help` must not return exit code 2.

    Exit 0 = delegated target exists and printed its own help.
    Exit 1 = target missing and shim printed the actionable hint.
    Exit 2 = the regression we're guarding against.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode in {0, 1}, (
        f"08_eval_causal_evidence_v2.py --help returned exit {result.returncode}; "
        f"expected 0 (delegated target works) or 1 (clean missing-target hint). "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    if result.returncode == 1:
        # Missing-target branch: stderr MUST carry an actionable hint.
        assert "Phase 8" in result.stderr, (
            "Exit 1 should be paired with a Phase-8 actionable hint in stderr; "
            f"got: {result.stderr!r}"
        )


def test_actionable_message_when_target_missing(tmp_path, monkeypatch):
    """If the delegated target file does not exist, the shim must print an
    actionable hint to stderr (Phase-8 reference) and exit 1.

    We simulate the missing-target condition by copying the shim into a
    sandbox where the delegated target sibling doesn't exist.
    """
    sandbox = tmp_path / "scripts" / "mortfm"
    sandbox.mkdir(parents=True)
    sandbox_shim = sandbox / "08_eval_causal_evidence_v2.py"
    sandbox_shim.write_text(SCRIPT.read_text())

    result = subprocess.run(
        [sys.executable, str(sandbox_shim), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        f"With delegated target missing, expected exit 1; got {result.returncode}. "
        f"stderr={result.stderr!r}"
    )
    assert "Phase 8" in result.stderr
    assert "not yet wired" in result.stderr or "phase_8_counterfactual.md" in result.stderr
