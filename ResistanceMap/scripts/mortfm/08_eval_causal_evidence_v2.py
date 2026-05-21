#!/usr/bin/env python3
"""scripts/mortfm/08_eval_causal_evidence_v2.py — v18 canonical workflow shim.

Stage: causal evidence v2 report
Delegates to: scripts/mortfm_causal_evidence_report_v2.py (CLI args + behavior preserved).

When the delegated script is missing, this shim exits with code 1 and a
clear, actionable hint (not the bare exit-2 it used to do, which was
indistinguishable from an argparse failure in the underlying script).
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parents[1] / "mortfm_causal_evidence_report_v2.py"
_LEGACY_V17 = Path(__file__).resolve().parents[1] / "mortfm_v17_causal_evidence.py"
_PHASE_8_DOC = "docs/v18_phase_plan/phase_8_counterfactual.md"

if __name__ == "__main__":
    if not _TARGET.exists():
        sys.stderr.write(
            "08_eval_causal_evidence_v2.py is not yet wired. "
            "Causal evidence aggregation depends on Phase 8 (see "
            f"{_PHASE_8_DOC}). "
            f"Expected delegated target: {_TARGET}\n"
        )
        if _LEGACY_V17.exists():
            sys.stderr.write(
                f"The legacy v17 evaluation lives at {_LEGACY_V17}; "
                "use that until Phase 8 lands.\n"
            )
        else:
            sys.stderr.write(
                "No legacy v17 evaluation found under scripts/. "
                "Run `git log --diff-filter=D --name-only -- scripts/` to locate "
                "the most recent removed implementation, or wait for Phase 8.\n"
            )
        sys.exit(1)
    sys.argv[0] = str(_TARGET)
    runpy.run_path(str(_TARGET), run_name="__main__")
