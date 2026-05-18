#!/usr/bin/env python3
"""scripts/mortfm/01_harmonize_identifiers.py — v18 canonical workflow shim.

Stage: identifier harmonization (STRING/HGNC bridge, drug aliases)
Delegates to: scripts/mortfm_harmonize_identifiers.py (CLI args + behavior preserved).
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parents[1] / "mortfm_harmonize_identifiers.py"

if __name__ == "__main__":
    if not _TARGET.exists():
        sys.stderr.write(f"[v18-shim] underlying script missing: {_TARGET}\n")
        sys.exit(2)
    sys.argv[0] = str(_TARGET)
    runpy.run_path(str(_TARGET), run_name="__main__")
