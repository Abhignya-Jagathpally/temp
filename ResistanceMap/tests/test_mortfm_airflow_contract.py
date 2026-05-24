"""Contract tests for the MORT-FM Airflow DAG.

Validates structural invariants of the pipeline orchestration layer
WITHOUT requiring a running Airflow instance. Tests use skip markers when
the DAG file or Airflow itself are not yet available, so the test module
always imports cleanly and pytest collects skip-reasons instead of errors.

Checked contracts
-----------------
1. DAG module imports without errors (even without Airflow installed).
2. All tasks reference existing scripts (verify each script path on disk).
3. Task dependencies form a valid DAG (no cycles).
4. Gate task depends on all evidence-producing tasks.
5. No task reimplements logic (tasks only call subprocess).
6. Workflow YAML loads and has required keys.
7. Every numbered script (00-10) is referenced by exactly one task.
8. ``audit_leakage`` task runs BEFORE any training task.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_RM_ROOT = Path(__file__).resolve().parent.parent  # ResistanceMap/
_DAG_PATH = _RM_ROOT / "dags" / "mortfm_evidence_pipeline.py"
_WORKFLOW_YAML = _RM_ROOT / "configs" / "workflows" / "mortfm_airflow.yaml"
_SCRIPTS_DIR = _RM_ROOT / "scripts" / "mortfm"

# All numbered scripts that MUST be present (00 through 10).
_NUMBERED_SCRIPTS = sorted(
    p for p in _SCRIPTS_DIR.glob("[0-9][0-9]_*.py")
)

# Training tasks are scripts whose numeric prefix is in {04, 05, 06, 07}.
_TRAINING_PREFIXES = {"04", "05", "06", "07"}


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------
_dag_exists = _DAG_PATH.exists()
_yaml_exists = _WORKFLOW_YAML.exists()

skip_no_dag = pytest.mark.skipif(
    not _dag_exists,
    reason=f"DAG file not yet created: {_DAG_PATH}",
)
skip_no_yaml = pytest.mark.skipif(
    not _yaml_exists,
    reason=f"Workflow YAML not yet created: {_WORKFLOW_YAML}",
)

try:
    import airflow  # noqa: F401
    _has_airflow = True
except ImportError:
    _has_airflow = False

skip_no_airflow = pytest.mark.skipif(
    not _has_airflow,
    reason="Apache Airflow is not installed in this environment",
)


# ---------------------------------------------------------------------------
# Helpers: load DAG module safely
# ---------------------------------------------------------------------------
def _load_dag_module() -> Optional[types.ModuleType]:
    """Import the DAG file as a Python module, returning None on failure."""
    if not _dag_exists:
        return None
    spec = importlib.util.spec_from_file_location(
        "mortfm_evidence_pipeline", str(_DAG_PATH),
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def _load_dag_ast() -> Optional[ast.Module]:
    """Parse the DAG file into an AST (does not execute code)."""
    if not _dag_exists:
        return None
    return ast.parse(_DAG_PATH.read_text(), filename=str(_DAG_PATH))


def _load_yaml() -> Optional[dict]:
    """Load the workflow YAML as a dict, returning None if missing."""
    if not _yaml_exists:
        return None
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("PyYAML not installed")
    return yaml.safe_load(_WORKFLOW_YAML.read_text())


# ---------------------------------------------------------------------------
# Helpers: AST-based analysis
# ---------------------------------------------------------------------------
def _extract_string_literals(tree: ast.Module) -> List[str]:
    """Return every string literal found in the AST."""
    strings: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append(node.value)
    return strings


def _extract_task_ids_and_bodies(tree: ast.Module) -> Dict[str, ast.AST]:
    """Heuristically map task_id strings to the node that defines them.

    Looks for patterns like:
        BashOperator(task_id="foo", bash_command="...")
        PythonOperator(task_id="bar", python_callable=...)
    and also bare function defs decorated with @task whose name IS the task_id.
    """
    tasks: Dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "task_id" and isinstance(kw.value, ast.Constant):
                    tasks[kw.value.value] = node
        # @task decorator: the function name is the task_id
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                dec_name = None
                if isinstance(dec, ast.Name):
                    dec_name = dec.id
                elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    dec_name = dec.func.id
                if dec_name == "task":
                    tasks[node.name] = node
    return tasks


def _build_varname_to_taskid(tree: ast.Module) -> Dict[str, str]:
    """Map Python variable names to their task_id string values.

    Handles patterns like::

        t_gate = PythonOperator(task_id="gate_revalidate", ...)

    Returns ``{"t_gate": "gate_revalidate", ...}``.
    """
    mapping: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                for kw in node.value.keywords:
                    if kw.arg == "task_id" and isinstance(kw.value, ast.Constant):
                        mapping[target.id] = kw.value.value
    return mapping


def _extract_dependency_edges_as_taskids(
    source: str, tree: ast.Module,
) -> List[Tuple[str, str]]:
    """Parse ``t_a >> t_b`` and translate variable names to task_id strings.

    Returns edges as ``(upstream_task_id, downstream_task_id)`` pairs.
    """
    var_to_tid = _build_varname_to_taskid(tree)
    raw_edges: List[Tuple[str, str]] = []

    parsed = ast.parse(source)
    for node in ast.walk(parsed):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.BinOp):
            _collect_shift_edges(node.value, raw_edges)

    # Translate variable names to task_ids.
    translated: List[Tuple[str, str]] = []
    for u, v in raw_edges:
        u_tid = var_to_tid.get(u, u)
        v_tid = var_to_tid.get(v, v)
        translated.append((u_tid, v_tid))

    return translated


def _collect_shift_edges(
    node: ast.BinOp, edges: List[Tuple[str, str]],
) -> None:
    """Recursively collect edges from chained >> / << operators."""
    if isinstance(node.op, ast.RShift):
        upstreams = _resolve_task_names(node.left)
        downstreams = _resolve_task_names(node.right)
        for u in upstreams:
            for d in downstreams:
                edges.append((u, d))
    elif isinstance(node.op, ast.LShift):
        upstreams = _resolve_task_names(node.right)
        downstreams = _resolve_task_names(node.left)
        for u in upstreams:
            for d in downstreams:
                edges.append((u, d))
    # Handle chained: a >> b >> c  =>  BinOp(BinOp(a >> b) >> c)
    if isinstance(node.left, ast.BinOp):
        _collect_shift_edges(node.left, edges)
    if isinstance(node.right, ast.BinOp):
        _collect_shift_edges(node.right, edges)


def _resolve_task_names(node: ast.AST) -> List[str]:
    """Resolve a Name, List, or Tuple to a list of variable-name strings."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        names: List[str] = []
        for elt in node.elts:
            names.extend(_resolve_task_names(elt))
        return names
    if isinstance(node, ast.BinOp):
        # middle element of chained a >> b >> c
        return _resolve_task_names(node.right)
    return []


def _has_cycle(edges: List[Tuple[str, str]]) -> bool:
    """Return True if the directed graph defined by *edges* has a cycle."""
    from collections import defaultdict

    adj: Dict[str, Set[str]] = defaultdict(set)
    nodes: Set[str] = set()
    for u, v in edges:
        adj[u].add(v)
        nodes.add(u)
        nodes.add(v)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in nodes}

    def dfs(u: str) -> bool:
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(color[n] == WHITE and dfs(n) for n in nodes)


def _transitive_ancestors(
    node: str, edges: List[Tuple[str, str]],
) -> Set[str]:
    """Return all transitive upstream ancestors of *node*."""
    from collections import defaultdict, deque

    parents: Dict[str, Set[str]] = defaultdict(set)
    for u, v in edges:
        parents[v].add(u)

    visited: Set[str] = set()
    queue = deque(parents.get(node, set()))
    while queue:
        cur = queue.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        queue.extend(parents.get(cur, set()))
    return visited


def _extract_script_paths_from_run_calls(tree: ast.Module) -> List[str]:
    """Extract script-path string arguments from ``_run_script(...)`` calls
    and ``bash_command`` keyword arguments.

    The DAG uses patterns like::

        _run_script(SCRIPTS / "00_download_public_data.py", ...)
        BashOperator(bash_command="python scripts/mortfm/00_foo.py", ...)

    For the ``SCRIPTS / "name.py"`` form we extract just the basename and
    resolve it against the known scripts directories.  For full path strings
    we return them as-is.
    """
    refs: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Pattern 1: _run_script(BASE / "script.py", ...)
        func = node.func
        is_run_script = (
            isinstance(func, ast.Name) and func.id == "_run_script"
        )
        if is_run_script and node.args:
            arg0 = node.args[0]
            # SCRIPTS / "name.py" compiles to BinOp(Name / Constant)
            if isinstance(arg0, ast.BinOp) and isinstance(arg0.op, ast.Div):
                rhs = arg0.right
                if isinstance(rhs, ast.Constant) and isinstance(rhs.value, str):
                    refs.append(rhs.value)
            elif isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                refs.append(arg0.value)

        # Pattern 2: BashOperator(bash_command="python scripts/...")
        for kw in node.keywords:
            if kw.arg == "bash_command" and isinstance(kw.value, ast.Constant):
                val = kw.value.value
                # Extract .py paths from the command string.
                for match in re.finditer(r"[\w/.]+\.py", val):
                    refs.append(match.group())

    return refs


# ---------------------------------------------------------------------------
# Contract 1: DAG imports without errors
# ---------------------------------------------------------------------------
class TestDAGImport:
    """Contract 1 -- the DAG file can be parsed / imported."""

    @skip_no_dag
    def test_dag_file_parses_as_valid_python(self):
        """The DAG file must be valid Python (AST-level)."""
        tree = _load_dag_ast()
        assert tree is not None, "Failed to parse DAG file"

    @skip_no_dag
    def test_dag_module_imports_without_airflow(self):
        """DAG module should either import cleanly or use try/except for Airflow.

        If Airflow is NOT installed, it is acceptable for the module to
        degrade gracefully (not raise).  We verify the file at least has a
        try/except around the airflow import so it can be loaded for linting.
        """
        tree = _load_dag_ast()
        assert tree is not None
        source = _DAG_PATH.read_text()

        # Check that the file contains a try/except guarding Airflow imports.
        has_try_except_airflow = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                body_src = ast.dump(ast.Module(body=node.body, type_ignores=[]))
                if "airflow" in body_src.lower():
                    has_try_except_airflow = True

        # The file may not import airflow at all (pure shell-based DAGs).
        imports_airflow = "import airflow" in source or "from airflow" in source
        assert has_try_except_airflow or not imports_airflow or _has_airflow, (
            "DAG file imports airflow without a try/except guard; "
            "this will break linters when Airflow is not installed."
        )

    @skip_no_dag
    @skip_no_airflow
    def test_dag_module_loads_with_airflow(self):
        """When Airflow IS available, the module must import cleanly."""
        mod = _load_dag_module()
        assert mod is not None, "DAG module failed to import even with Airflow installed"


# ---------------------------------------------------------------------------
# Contract 2: All tasks reference existing scripts
# ---------------------------------------------------------------------------
class TestTaskScriptReferences:
    """Contract 2 -- every script path referenced by a task must exist on disk."""

    @skip_no_dag
    def test_all_referenced_scripts_exist(self):
        tree = _load_dag_ast()
        assert tree is not None

        script_refs = _extract_script_paths_from_run_calls(tree)
        assert script_refs, (
            "No script path references found in DAG file. "
            "Tasks should reference scripts/mortfm/*.py"
        )

        missing: List[str] = []
        for ref in script_refs:
            # The DAG constructs paths as SCRIPTS / "basename.py" or
            # SCRIPTS_TOP / "basename.py".  Try resolving the basename
            # against both known directories.
            candidates = [
                _SCRIPTS_DIR / ref,              # scripts/mortfm/
                _RM_ROOT / "scripts" / ref,      # scripts/
                _RM_ROOT / ref,                   # ResistanceMap root
                Path(ref),                        # absolute
            ]
            if not any(c.exists() for c in candidates):
                missing.append(ref)

        assert not missing, (
            "DAG references scripts that do not exist on disk:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


# ---------------------------------------------------------------------------
# Contract 3: Task dependencies form a valid DAG (no cycles)
# ---------------------------------------------------------------------------
class TestNoCycles:
    """Contract 3 -- the dependency graph must be acyclic."""

    @skip_no_dag
    def test_no_cycles_in_task_dependencies(self):
        tree = _load_dag_ast()
        assert tree is not None
        source = _DAG_PATH.read_text()
        edges = _extract_dependency_edges_as_taskids(source, tree)

        if not edges:
            pytest.skip(
                "No >> / << dependency edges found; DAG may use "
                "set_upstream/set_downstream or Airflow TaskFlow API "
                "and this test needs extension."
            )

        assert not _has_cycle(edges), (
            "Cycle detected in task dependency graph. Edges:\n"
            + "\n".join(f"  {u} -> {v}" for u, v in edges)
        )


# ---------------------------------------------------------------------------
# Contract 4: Gate task depends on all evidence-producing tasks
# ---------------------------------------------------------------------------
class TestGateDependencies:
    """Contract 4 -- the gate/revalidation task must depend on every
    evidence-producing task upstream of it.
    """

    @skip_no_dag
    def test_gate_depends_on_all_evidence_tasks(self):
        tree = _load_dag_ast()
        assert tree is not None
        source = _DAG_PATH.read_text()

        tasks = _extract_task_ids_and_bodies(tree)
        edges = _extract_dependency_edges_as_taskids(source, tree)

        # Identify the gate task (task_id contains "gate" or "revalidat").
        gate_tasks = [
            tid for tid in tasks
            if "gate" in tid.lower() or "revalidat" in tid.lower()
        ]
        assert gate_tasks, (
            "No gate/revalidation task found among task_ids: "
            + ", ".join(sorted(tasks))
        )

        gate_id = gate_tasks[0]
        gate_ancestors = _transitive_ancestors(gate_id, edges)

        # Evidence-producing tasks: training (train/pretrain), evaluation
        # (eval/evidence), and baselines.  The gate task itself is excluded.
        evidence_keywords = ("train", "pretrain", "eval", "evidence", "baseline")
        evidence_tasks = [
            tid for tid in tasks
            if tid != gate_id
            and any(kw in tid.lower() for kw in evidence_keywords)
        ]

        if not evidence_tasks:
            pytest.skip("Could not identify evidence-producing tasks by name convention")

        missing_deps = [t for t in evidence_tasks if t not in gate_ancestors]
        assert not missing_deps, (
            f"Gate task '{gate_id}' does not (transitively) depend on these "
            f"evidence-producing tasks: {sorted(missing_deps)}"
        )


# ---------------------------------------------------------------------------
# Contract 5: No task reimplements logic (tasks only call subprocess)
# ---------------------------------------------------------------------------
class TestNoInlineLogic:
    """Contract 5 -- DAG tasks must delegate to scripts via subprocess / bash,
    not reimplement training or evaluation logic inline.
    """

    # Imports that indicate non-trivial inline ML/data logic.
    _FORBIDDEN_IMPORTS = {"torch", "numpy", "pandas", "sklearn", "scipy"}

    @skip_no_dag
    def test_dag_does_not_import_ml_libraries(self):
        """The DAG file should not import ML libraries directly."""
        tree = _load_dag_ast()
        assert tree is not None

        imported_names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_names.add(node.module.split(".")[0])

        violations = imported_names & self._FORBIDDEN_IMPORTS
        assert not violations, (
            f"DAG file imports ML libraries directly, suggesting inline logic: "
            f"{violations}. Tasks should delegate to scripts via subprocess."
        )

    @skip_no_dag
    def test_tasks_use_subprocess_or_bash(self):
        """Each task should call an external script, not contain model logic."""
        tree = _load_dag_ast()
        assert tree is not None
        source = _DAG_PATH.read_text()

        tasks = _extract_task_ids_and_bodies(tree)
        if not tasks:
            pytest.skip("No tasks found in DAG file")

        # Verify the DAG uses BashOperator, subprocess, or similar delegation.
        uses_bash_operator = "BashOperator" in source
        uses_subprocess = "subprocess" in source
        uses_bash_command = "bash_command" in source

        assert uses_bash_operator or uses_subprocess or uses_bash_command, (
            "DAG does not appear to use BashOperator, subprocess, or "
            "bash_command. Tasks should delegate to external scripts."
        )


# ---------------------------------------------------------------------------
# Contract 6: Workflow YAML loads and has required keys
# ---------------------------------------------------------------------------
class TestWorkflowYAML:
    """Contract 6 -- the workflow YAML must parse and contain required keys.

    The YAML (``configs/workflows/mortfm_airflow.yaml``) is the pipeline
    configuration consumed by the DAG builder and gate auditor. It defines
    paths, cohorts, baselines, claim-gate thresholds, compute profile,
    evidence channels, endpoints, splits, and artifacts.
    """

    # Top-level sections that MUST be present.
    _REQUIRED_TOP_LEVEL_KEYS = {
        "paths", "cohorts", "gates", "artifacts",
    }

    @skip_no_yaml
    def test_yaml_loads_cleanly(self):
        data = _load_yaml()
        assert data is not None, "YAML loaded as None (empty file?)"
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"

    @skip_no_yaml
    def test_yaml_has_required_keys(self):
        data = _load_yaml()
        assert data is not None
        missing = self._REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
        assert not missing, (
            f"Workflow YAML is missing required top-level keys: {missing}"
        )

    @skip_no_yaml
    def test_yaml_cohorts_have_ids(self):
        """Each cohort entry must have a ``cohort_id``."""
        data = _load_yaml()
        assert data is not None
        cohorts = data.get("cohorts")
        if cohorts is None:
            pytest.skip("No 'cohorts' key in YAML")
        assert isinstance(cohorts, list), f"'cohorts' should be a list, got {type(cohorts)}"
        for i, cohort in enumerate(cohorts):
            assert "cohort_id" in cohort, f"Cohort {i} missing 'cohort_id'"

    @skip_no_yaml
    def test_yaml_gates_have_descriptions(self):
        """Each claim-gate level must have a ``description``."""
        data = _load_yaml()
        assert data is not None
        gates = data.get("gates")
        if gates is None:
            pytest.skip("No 'gates' key in YAML")
        assert isinstance(gates, dict), f"'gates' should be a dict, got {type(gates)}"
        for level, spec in gates.items():
            assert "description" in spec, (
                f"Gate level '{level}' missing 'description'"
            )

    @skip_no_yaml
    def test_yaml_artifacts_have_local_paths(self):
        """Each artifact must have a ``local`` path."""
        data = _load_yaml()
        assert data is not None
        artifacts = data.get("artifacts")
        if artifacts is None:
            pytest.skip("No 'artifacts' key in YAML")
        assert isinstance(artifacts, dict), (
            f"'artifacts' should be a dict, got {type(artifacts)}"
        )
        for name, spec in artifacts.items():
            assert "local" in spec, f"Artifact '{name}' missing 'local' path"


# ---------------------------------------------------------------------------
# Contract 7: Every numbered script (00-10) referenced by exactly one task
# ---------------------------------------------------------------------------
class TestScriptCoverage:
    """Contract 7 -- each of the 00-10 numbered scripts must be referenced
    by exactly one DAG task (no orphans, no duplicates).
    """

    @skip_no_dag
    def test_every_numbered_script_referenced_exactly_once(self):
        tree = _load_dag_ast()
        assert tree is not None

        # Collect all script basenames referenced by _run_script() calls.
        refs = _extract_script_paths_from_run_calls(tree)

        # Build a set of expected script basenames.
        expected = {p.name for p in _NUMBERED_SCRIPTS}
        assert expected, "No numbered scripts found in scripts/mortfm/"
        assert len(expected) >= 11, (
            f"Expected at least 11 numbered scripts (00-10), found {len(expected)}: "
            + ", ".join(sorted(expected))
        )

        # Count how many times each script basename appears in references.
        ref_counts: Dict[str, int] = {name: 0 for name in expected}
        for ref_path in refs:
            basename = Path(ref_path).name
            if basename in ref_counts:
                ref_counts[basename] += 1

        unreferenced = [n for n, c in ref_counts.items() if c == 0]
        duplicated = [n for n, c in ref_counts.items() if c > 1]

        errors: List[str] = []
        if unreferenced:
            errors.append(
                "Scripts not referenced by any task:\n"
                + "\n".join(f"  - {n}" for n in sorted(unreferenced))
            )
        if duplicated:
            errors.append(
                "Scripts referenced by more than one task:\n"
                + "\n".join(f"  - {n} ({ref_counts[n]}x)" for n in sorted(duplicated))
            )
        assert not errors, "\n".join(errors)


# ---------------------------------------------------------------------------
# Contract 8: audit_leakage runs BEFORE any training task
# ---------------------------------------------------------------------------
class TestLeakageBeforeTraining:
    """Contract 8 -- the leakage audit task must be an ancestor of every
    training task in the dependency graph.
    """

    @skip_no_dag
    def test_audit_leakage_before_training(self):
        tree = _load_dag_ast()
        assert tree is not None
        source = _DAG_PATH.read_text()

        tasks = _extract_task_ids_and_bodies(tree)
        edges = _extract_dependency_edges_as_taskids(source, tree)

        # Find the leakage audit task.
        leakage_tasks = [
            tid for tid in tasks
            if "leakage" in tid.lower() or "audit" in tid.lower()
        ]
        assert leakage_tasks, (
            "No audit_leakage (or similar) task found among task_ids: "
            + ", ".join(sorted(tasks))
        )

        leakage_id = leakage_tasks[0]

        # Identify training tasks (name containing "train" or "pretrain").
        training_tasks = [
            tid for tid in tasks
            if tid != leakage_id
            and any(kw in tid.lower() for kw in ("train", "pretrain"))
        ]

        if not training_tasks:
            pytest.skip("Could not identify training tasks by name convention")

        violations: List[str] = []
        for tid in training_tasks:
            ancestors = _transitive_ancestors(tid, edges)
            if leakage_id not in ancestors:
                violations.append(tid)

        assert not violations, (
            f"Leakage audit task '{leakage_id}' is NOT an ancestor of these "
            f"training tasks: {sorted(violations)}. "
            f"The audit must run before training."
        )


# ---------------------------------------------------------------------------
# Bonus: numbered-script directory sanity
# ---------------------------------------------------------------------------
class TestNumberedScriptsExist:
    """Sanity-check that all expected numbered scripts (00-10) exist on disk.

    This is independent of the DAG and always runs.
    """

    @pytest.mark.parametrize("prefix", [f"{i:02d}" for i in range(11)])
    def test_numbered_script_exists(self, prefix: str):
        matches = list(_SCRIPTS_DIR.glob(f"{prefix}_*.py"))
        assert matches, (
            f"No script with prefix {prefix}_ found in {_SCRIPTS_DIR}"
        )
        assert len(matches) == 1, (
            f"Multiple scripts with prefix {prefix}_ found: "
            + ", ".join(m.name for m in matches)
        )
