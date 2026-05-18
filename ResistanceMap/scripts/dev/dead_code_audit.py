#!/usr/bin/env python3
"""
scripts/dev/dead_code_audit.py
================================
v18.5 — stdlib-only dead-code audit for the MORT-FM repo.

Catches three failure modes that have caused real-world incidents on
this project:

  1. **Unused imports** (F401-equivalent)
     – often the leftover of a refactor, sometimes the symptom of an
       entire module being orphaned.
  2. **Unused local variables** (F841-equivalent)
     – ``foo = expensive_call()`` followed by no read of ``foo`` is a
       silent metric drop and one of the failure modes the audit team
       called out in 2026-05-07.
  3. **Orphan modules**
     – ``.py`` files that are not imported by any other file in the
       repo. Often safe to archive, but every one needs human review.

Inputs:
  * --roots                directories to scan (default: resistancemap scripts tests)
  * --allowlist            YAML allowlist (default: configs/dead_code_allowlist.yaml)
  * --out                  JSON report path (default: results/dev/dead_code_report.json)

Exit codes:
  * 0  if the audit ran and the report was written
  * 2  on hard error (e.g. allowlist not found)

This script does NOT modify any files. It is a reporter only.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dead_code_audit")


# ----------------------------------------------------------------------- yaml
def _load_yaml(path: Path) -> dict:
    """Tiny YAML subset loader — handles flat key + indented list-of-strings only."""
    try:
        import yaml  # noqa
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        pass

    out: Dict[str, list] = {}
    current_key: Optional[str] = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(stripped)
            if indent == 0 and ":" in stripped:
                key, _, rest = stripped.partition(":")
                current_key = key.strip()
                out[current_key] = []
                if rest.strip():
                    out[current_key] = [rest.strip().strip("\"")]
            elif stripped.startswith("- ") and current_key is not None:
                val = stripped[2:].strip().strip("\"")
                if val:
                    out[current_key].append(val)
    return out


@dataclass
class AuditFinding:
    file: str
    line: int
    kind: str          # unused_import | unused_local | orphan_module
    symbol: str = ""
    detail: str = ""


@dataclass
class AuditReport:
    run_id: str
    n_files_scanned: int = 0
    n_files_allowlisted: int = 0
    findings: List[AuditFinding] = field(default_factory=list)
    by_kind: Dict[str, int] = field(default_factory=dict)
    wall_time_s: float = 0.0

    def add(self, f: AuditFinding) -> None:
        self.findings.append(f)
        self.by_kind[f.kind] = self.by_kind.get(f.kind, 0) + 1


# ----------------------------------------------------------------------- ast helpers
def _collect_names_loaded(tree: ast.AST) -> Set[str]:
    loaded: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)
        elif isinstance(node, ast.Attribute):
            v = node.value
            while isinstance(v, ast.Attribute):
                v = v.value
            if isinstance(v, ast.Name):
                loaded.add(v.id)
    return loaded


def _collect_strings(tree: ast.AST) -> Set[str]:
    """Collect raw strings (for __all__-style + getattr lookups)."""
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.add(node.value)
    return out


def _imported_names(tree: ast.AST) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                out.append((name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                out.append((name, node.lineno))
    return out


def _unused_locals(tree: ast.AST) -> List[Tuple[str, int]]:
    """Detect ``x = ...`` where x is never subsequently loaded in the same scope.

    This is intentionally conservative — we only flag explicit single-target
    assignments at function-body scope, and only when the value is NOT a
    call to a logger / warnings / typing helper (those have side effects).
    """
    findings: List[Tuple[str, int]] = []

    class FuncVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._scan(node.body, node.args)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self._scan(node.body, node.args)
            self.generic_visit(node)

        def _scan(self, body: list, args: ast.arguments) -> None:
            assigned: Dict[str, int] = {}
            for stmt in body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        if isinstance(stmt.value, ast.Call):
                            f = stmt.value.func
                            f_name = (
                                f.attr if isinstance(f, ast.Attribute)
                                else (f.id if isinstance(f, ast.Name) else "")
                            )
                            if f_name in {"warn", "debug", "info", "warning", "error"}:
                                continue
                        assigned[target.id] = stmt.lineno
            loaded = _collect_names_loaded(ast.Module(body=body, type_ignores=[]))
            strings = _collect_strings(ast.Module(body=body, type_ignores=[]))
            for name, lineno in assigned.items():
                if name not in loaded and name not in strings:
                    findings.append((name, lineno))

    FuncVisitor().visit(tree)
    return findings


# ----------------------------------------------------------------------- allowlist
def _path_in_allowlist(path: Path, repo_root: Path, allowlist: List[str]) -> bool:
    try:
        rel = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return False
    for entry in allowlist:
        if entry.endswith("/") and rel.startswith(entry):
            return True
        if "::" in entry:
            continue
        if rel == entry:
            return True
    return False


def _symbol_in_allowlist(file_rel: str, symbol: str, allowlist: List[str]) -> bool:
    target = f"{file_rel}::{symbol}"
    return target in allowlist


# ----------------------------------------------------------------------- main
def _iter_py_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            if ".venv" in p.parts or "venv" in p.parts:
                continue
            yield p


def _build_import_graph(py_files: List[Path], repo_root: Path) -> Set[str]:
    """Return the set of dotted module names that are imported by any file."""
    imported: Set[str] = set()
    for p in py_files:
        try:
            tree = ast.parse(p.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                # Add ancestors so e.g. ``from a.b.c import d`` allowlists ``a.b.c.d`` if it's a module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
    return imported


def _module_dotted(path: Path, repo_root: Path) -> str:
    rel = path.resolve().relative_to(repo_root.resolve())
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def run_audit(roots: List[Path], allowlist: List[str], repo_root: Path) -> AuditReport:
    t0 = time.time()
    report = AuditReport(run_id=f"r-{time.strftime('%Y-%m-%d')}-dead-code-audit")

    py_files = list(_iter_py_files(roots))
    report.n_files_scanned = len(py_files)
    imported_dotted = _build_import_graph(py_files, repo_root)

    for p in py_files:
        if _path_in_allowlist(p, repo_root, allowlist):
            report.n_files_allowlisted += 1
            continue
        rel = p.resolve().relative_to(repo_root.resolve()).as_posix()
        try:
            tree = ast.parse(p.read_text())
        except (SyntaxError, UnicodeDecodeError) as e:
            report.add(AuditFinding(
                file=rel, line=0, kind="parse_error", detail=str(e)
            ))
            continue

        loaded = _collect_names_loaded(tree)
        strings = _collect_strings(tree)
        for name, lineno in _imported_names(tree):
            if name in loaded or name in strings:
                continue
            if _symbol_in_allowlist(rel, name, allowlist):
                continue
            report.add(AuditFinding(
                file=rel, line=lineno, kind="unused_import", symbol=name
            ))

        for name, lineno in _unused_locals(tree):
            if _symbol_in_allowlist(rel, name, allowlist):
                continue
            report.add(AuditFinding(
                file=rel, line=lineno, kind="unused_local", symbol=name
            ))

    # Orphan modules: under resistancemap/ but never imported and not in allowlist
    rm_root = repo_root / "resistancemap"
    for p in py_files:
        try:
            rel = p.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            continue
        if not rel.startswith("resistancemap/"):
            continue
        if p.name == "__init__.py":
            continue
        if _path_in_allowlist(p, repo_root, allowlist):
            continue
        dotted = _module_dotted(p, repo_root)
        if any(d.startswith(dotted) for d in imported_dotted):
            continue
        report.add(AuditFinding(
            file=rel, line=0, kind="orphan_module", symbol=dotted,
            detail="module is not imported by any file in the scanned roots"
        ))

    report.wall_time_s = round(time.time() - t0, 3)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=["resistancemap", "scripts", "tests"],
                    help="Directories to scan, relative to repo root.")
    ap.add_argument("--allowlist", default="configs/dead_code_allowlist.yaml")
    ap.add_argument("--out", default="results/dev/dead_code_report.json")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    allowlist_path = repo_root / args.allowlist
    if not allowlist_path.exists():
        logger.error("allowlist not found at %s", allowlist_path)
        return 2

    raw = _load_yaml(allowlist_path)
    allowlist = list(raw.get("allowlist") or [])

    roots = [repo_root / r for r in args.roots]
    logger.info("Scanning %d roots; allowlist has %d entries", len(roots), len(allowlist))

    report = run_audit(roots, allowlist, repo_root)

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": report.run_id,
        "n_files_scanned": report.n_files_scanned,
        "n_files_allowlisted": report.n_files_allowlisted,
        "by_kind": report.by_kind,
        "wall_time_s": report.wall_time_s,
        "findings": [
            {"file": f.file, "line": f.line, "kind": f.kind,
             "symbol": f.symbol, "detail": f.detail}
            for f in report.findings
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))

    logger.info("Scanned %d files (%d allowlisted)",
                report.n_files_scanned, report.n_files_allowlisted)
    logger.info("Findings by kind: %s", report.by_kind)
    logger.info("Report: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
