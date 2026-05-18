#!/usr/bin/env python3
"""scripts/mortfm/02_build_longitudinal_dataset.py — v18 canonical workflow shim.

Stage: build longitudinal patient pair dataset
Delegates to: scripts/mortfm_build_longitudinal_dataset.py (CLI args + behavior preserved).
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parents[1] / "mortfm_build_longitudinal_dataset.py"

if __name__ == "__main__":
    if not _TARGET.exists():
        sys.stderr.write(f"[v18-shim] underlying script missing: {_TARGET}\n")
        sys.exit(2)
    sys.argv[0] = str(_TARGET)
    runpy.run_path(str(_TARGET), run_name="__main__")
