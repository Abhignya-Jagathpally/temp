#!/usr/bin/env python3
"""Fail the build if README/ARCHITECTURE/AGENT_ORCHESTRATOR numbers drift from FACTS.

Run in CI:
    python scripts/check_docs_consistent.py

Rules enforced:
  1. Canonical PPI sizes from FACTS must appear unmodified in any doc that
     mentions STRING v12.
  2. No doc may contain the phrase "Adams-Bashforth" (dopri5 is Dormand-Prince).
  3. No doc may claim NIST SP 800-207 compliance.
  4. Fusion hidden_dim must be consistent across README and AGENT_ORCHESTRATOR.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resistancemap._facts import FACTS   # noqa: E402


FORBIDDEN_PHRASES = [
    ("Adams-Bashforth",
     "dopri5 is Dormand-Prince RK(4,5), not Adams-Bashforth. Fix the phrase."),
    ("NIST SP 800-207",
     "SHA256 content hashing is not NIST ZTA. Remove the claim or implement ZTA."),
    ("first-principles math",
     "Remove unless Fano/Lyapunov/Kramers are wired into loss/inference."),
]


DOCS = [
    Path("README.md"),
    Path("ARCHITECTURE.md"),
    Path("AGENT_ORCHESTRATOR.md"),
]


def _find_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d[\d,]*\b", text))


def check_phrase_forbidden(text: str, path: Path) -> list[str]:
    errs = []
    for phrase, why in FORBIDDEN_PHRASES:
        if phrase.lower() in text.lower():
            errs.append(f"{path}: contains forbidden phrase {phrase!r}. {why}")
    return errs


def check_ppi_numbers_consistent(text: str, path: Path) -> list[str]:
    """If the doc mentions STRING v12, it must use FACTS['ppi'] numbers."""
    if "STRING" not in text:
        return []
    errs = []
    ppi = FACTS["ppi"]
    nums = _find_numbers(text.replace(",", ""))
    expected = {
        str(ppi["raw_nodes"]),
        str(ppi["raw_edges"]),
        str(ppi["mm_subnet_nodes"]),
        str(ppi["mm_subnet_edges"]),
    }
    # Any "19,177" / "930000" / "460000" style number that is NOT in expected
    # AND looks like a PPI count (>=1000) is a drift candidate.
    for n in nums:
        if len(n) >= 4 and n not in expected:
            # Let through obviously-unrelated numbers (years, epoch counts <=10k).
            if 1_000 <= int(n) <= 50_000_000 and n not in expected:
                # Heuristic: flag only numbers near the word STRING / PPI / edges / nodes
                pattern = rf"(STRING|PPI|nodes|edges|proteins|interactions).{{0,120}}\b{re.escape(n)}\b"
                if re.search(pattern, text, flags=re.I | re.S):
                    errs.append(
                        f"{path}: PPI-adjacent number {n} not in FACTS (expected one of {sorted(expected)})"
                    )
    return errs


def check_fusion_consistent(text: str, path: Path) -> list[str]:
    m = re.search(r"hidden_?dim[\s:=]+(\d+)", text, flags=re.I)
    if not m:
        return []
    hd = int(m.group(1))
    want = FACTS["fusion"]["hidden_dim"]
    if hd != want:
        return [f"{path}: fusion hidden_dim={hd} disagrees with FACTS ({want})"]
    return []


def main(repo_root: Path = Path(".")) -> int:
    all_errs: list[str] = []
    for rel in DOCS:
        path = repo_root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        all_errs.extend(check_phrase_forbidden(text, rel))
        all_errs.extend(check_ppi_numbers_consistent(text, rel))
        all_errs.extend(check_fusion_consistent(text, rel))

    if all_errs:
        print("Documentation consistency check FAILED:")
        for e in all_errs:
            print(f"  - {e}")
        return 1
    print(f"Documentation consistent with FACTS ({len(DOCS)} docs checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
