"""
resistancemap/mortfm/blocks/_common.py
======================================
Shared helpers for the block adapters. Kept private — adapters import from
here, not external callers.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def read_json(path: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON %s: %s", p, exc)
        return None


def utcnow_iso() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def git_head_short() -> str:
    """Return the short git SHA of HEAD, or empty string on failure.

    No exceptions propagate — registries are documentary and should never
    block on git plumbing.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=10", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def load_torch_payload(path: str | Path, *, map_location: str = "cpu") -> Optional[dict]:
    """Load a torch ``.pt`` checkpoint and return its dict payload, or None."""
    p = Path(path)
    if not p.exists():
        return None
    import torch
    return torch.load(str(p), map_location=map_location, weights_only=False)
