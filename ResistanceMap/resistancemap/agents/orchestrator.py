"""DAG-based orchestrator for parallel agent execution.

Manages agent dependencies and executes agents with maximum parallelism
while respecting dependency constraints. Implements zero-trust verification
and complete audit trails.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from resistancemap.agents.base import BaseAgent, AgentState, AgentResult
from resistancemap.config import ResistanceMapConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentDAG:
    """Directed acyclic graph of agent dependencies.

    Attributes:
        agents: Dict of agent_name -> BaseAgent instance
        edges: Dict of agent_name -> [dependency_names] (what each agent depends on)
    """

    agents: dict[str, BaseAgent] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)

    def add_agent(self, agent: BaseAgent) -> None:
        """Register an agent and its dependencies in the DAG.

        Args:
            agent: BaseAgent instance to add

        Raises:
            ValueError: If agent name already exists or dependency not found later
        """
        if agent.name in self.agents:
            raise ValueError(f"Agent {agent.name} already registered")

        self.agents[agent.name] = agent
        self.edges[agent.name] = agent.dependencies.copy()

    def topological_sort(self) -> list[list[str]]:
        """Topologically sort agents into layers for parallel execution.

        Returns a list of layers where each layer contains agents that can
        run in parallel (they don't depend on each other).

        Returns:
            List of lists: [[agent_names_layer_0], [agent_names_layer_1], ...]

        Raises:
            ValueError: If cycle detected in dependencies
        """
        # Check for missing dependencies
        all_agent_names = set(self.agents.keys())
        for agent_name, deps in self.edges.items():
            for dep in deps:
                if dep not in all_agent_names:
                    raise ValueError(
                        f"Agent {agent_name} depends on unknown agent {dep}"
                    )

        # Kahn's algorithm for topological sort with layer grouping
        in_degree = {name: len(self.edges[name]) for name in self.agents}
        layers = []
        processed = set()

        while len(processed) < len(self.agents):
            # Find all agents ready to execute (all deps satisfied)
            ready = [
                name
                for name in self.agents
                if name not in processed and in_degree[name] == 0
            ]

            if not ready:
                # No ready agents but not all processed = cycle
                remaining = set(self.agents.keys()) - processed
                raise ValueError(f"Cycle detected in agent dependencies: {remaining}")

            layers.append(ready)
            processed.update(ready)

            # Decrement in_degree for agents that depend on ready agents
            for ready_agent in ready:
                for other_agent, deps in self.edges.items():
                    if ready_agent in deps:
                        in_degree[other_agent] -= 1

        return layers

    def get_ready_agents(self, completed: set[str]) -> list[str]:
        """Get agents that are ready to execute given completed agents.

        Args:
            completed: Set of agent names that have completed

        Returns:
            List of agent names whose dependencies are all satisfied
        """
        ready = []
        for agent_name, deps in self.edges.items():
            if agent_name not in completed and all(dep in completed for dep in deps):
                ready.append(agent_name)
        return ready

    def validate(self) -> tuple[bool, str]:
        """Validate DAG structure (no cycles, all deps exist).

        Returns:
            (is_valid, error_message) tuple
        """
        try:
            self.topological_sort()
            return True, ""
        except ValueError as e:
            return False, str(e)


@dataclass
class ExecutionPlan:
    """Execution plan with layered agent schedule.

    Attributes:
        layers: List of layers, each containing agent names to run in parallel
        agent_order: Flattened list of all agent names in execution order
    """

    layers: list[list[str]]
    agent_order: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Compute flattened agent order."""
        self.agent_order = [name for layer in self.layers for name in layer]


class Orchestrator:
    """Zero-trust parallel agent orchestrator.

    Manages DAG-based execution with maximum parallelism while maintaining
    data integrity through verification and audit trails.

    Attributes:
        dag: AgentDAG instance
        results: Dict of agent_name -> AgentResult
        execution_plan: ExecutionPlan for the DAG
        _verification_chain: List of hashes for audit trail
        _timing_log: Dict of agent_name -> (start_time, end_time)
    """

    def __init__(
        self,
        tracer: Any | None = None,
        evaluator: Any | None = None,
        optimizer: Any | None = None,
        guardrails: Any | None = None,
    ):
        """Initialize orchestrator.

        Args:
            tracer:    optional ``agentops.Tracer`` to emit spans.
            evaluator: optional ``agentops.Evaluator`` to record task results
                       and guardrail violations.
            optimizer: optional ``agentops.Optimizer`` (handoff timing).
            guardrails: optional ``verification.GuardrailEngine``.

        v7: previously these were constructed in ``main.run_agentic_pipeline``
        but never injected here, so the dashboard always reported 0 traces. They
        are now optional kwargs; when supplied, every agent execution emits a
        span and every successful output is run through ``GuardrailEngine.check_all``.
        """
        self.dag = AgentDAG()
        self.results: dict[str, AgentResult] = {}
        self.execution_plan: ExecutionPlan | None = None
        self._verification_chain: list[str] = []
        self._timing_log: dict[str, tuple[float, float]] = {}
        self._tracer = tracer
        self._evaluator = evaluator
        self._optimizer = optimizer
        self._guardrails = guardrails
        self._trace_id: str | None = None
        self._guardrail_violations: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Run-manifest helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID combining a UTC timestamp and UUID4 suffix.

        Returns:
            String like ``20260523T143012Z-a1b2c3d4``
        """
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        short_uuid = uuid.uuid4().hex[:8]
        return f"{ts}-{short_uuid}"

    @staticmethod
    def _get_git_sha() -> str:
        """Return the current HEAD commit SHA via ``git rev-parse HEAD``.

        Returns ``"unknown"`` when git is unavailable or the working
        directory is not inside a repository.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            logger.debug("_get_git_sha: git rev-parse failed")
        return "unknown"

    @staticmethod
    def _compute_config_hash(config: ResistanceMapConfig) -> str:
        """SHA-256 of the frozen (JSON-serialised) config.

        Falls back to hashing ``repr(config)`` when JSON serialisation
        is not available on the config object.
        """
        try:
            # ResistanceMapConfig may expose .to_dict() or similar
            if hasattr(config, "to_dict"):
                blob = json.dumps(config.to_dict(), sort_keys=True)
            elif hasattr(config, "__dict__"):
                blob = json.dumps(
                    {k: repr(v) for k, v in sorted(vars(config).items())}
                )
            else:
                blob = repr(config)
        except Exception:
            blob = repr(config)
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def _compute_file_hash(path: str | Path) -> str:
        """SHA-256 of an on-disk file.  Returns ``"missing"`` if the file
        does not exist or cannot be read.
        """
        try:
            p = Path(path)
            if not p.exists():
                return "missing"
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return "missing"

    @staticmethod
    def _detect_airflow_metadata() -> dict[str, str | None] | None:
        """Return Airflow context from environment variables, or ``None``
        when not running inside an Airflow task.
        """
        dag_id = os.environ.get("AIRFLOW_CTX_DAG_ID")
        if dag_id is None:
            return None
        return {
            "dag_id": dag_id,
            "task_id": os.environ.get("AIRFLOW_CTX_TASK_ID"),
            "execution_date": os.environ.get("AIRFLOW_CTX_EXECUTION_DATE"),
        }

    def _init_manifest(self, config: ResistanceMapConfig) -> None:
        """Build the initial run manifest (before agent execution).

        Populates ``self._run_manifest`` with run_id, git_sha,
        config_hash, data/split manifest hashes, and empty containers
        for per-agent data that will be filled during execution.
        """
        # Resolve data_ready.pt and split paths from config if available
        output_dir = getattr(config, "output_dir", None) or "."
        data_ready_path = Path(output_dir) / "data_ready.pt"
        split_path = Path(output_dir) / "splits.json"

        self._run_manifest: dict[str, Any] = {
            "run_id": self._generate_run_id(),
            "git_sha": self._get_git_sha(),
            "config_hash": self._compute_config_hash(config),
            "data_manifest_hash": self._compute_file_hash(data_ready_path),
            "split_manifest_hash": self._compute_file_hash(split_path),
            "per_agent_artifacts": {},
            "per_agent_verification_hashes": {},
            "per_agent_timing": {},
            "airflow_metadata": self._detect_airflow_metadata(),
        }
        logger.info(
            f"Orchestrator: run_id={self._run_manifest['run_id']}  "
            f"git_sha={self._run_manifest['git_sha'][:8]}..."
        )

    def save_manifest(self, path: str | Path) -> Path:
        """Write the full run manifest to disk as JSON.

        Ensures the parent directory exists before writing.  Populates
        ``per_agent_verification_hashes`` and ``per_agent_timing`` from
        the existing verification chain and timing log so the manifest
        is always internally consistent.

        Args:
            path: Destination file path.

        Returns:
            Resolved ``pathlib.Path`` of the written file.

        Raises:
            RuntimeError: If called before ``run()`` has initialised the
                manifest.
        """
        if not hasattr(self, "_run_manifest"):
            raise RuntimeError(
                "No run manifest available — call run() first"
            )

        # Sync live data into the manifest
        self._run_manifest["per_agent_verification_hashes"] = {
            name: result.verification_hash
            for name, result in self.results.items()
        }
        self._run_manifest["per_agent_timing"] = self.get_timing_report()

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self._run_manifest, indent=2, default=str))
        logger.info(f"Orchestrator: manifest saved to {dest}")
        return dest

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def add_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent: BaseAgent instance
        """
        self.dag.add_agent(agent)

    def prepare_execution(self) -> tuple[bool, str]:
        """Validate DAG and prepare execution plan.

        Must be called before run().

        Returns:
            (is_valid, error_message) tuple
        """
        is_valid, error_msg = self.dag.validate()
        if not is_valid:
            return False, error_msg

        try:
            layers = self.dag.topological_sort()
            self.execution_plan = ExecutionPlan(layers=layers)
            return True, ""
        except Exception as e:
            return False, str(e)

    async def run(self, config: ResistanceMapConfig) -> dict[str, AgentResult]:
        """Execute all agents respecting dependencies with maximum parallelism.

        Algorithm:
        1. Topological sort to identify parallelizable layers
        2. For each layer, launch agents concurrently with asyncio.gather()
        3. After each agent completes, verify output hash (zero-trust)
        4. Pass verified outputs to dependent agents
        5. Record execution traces and verification chain

        Args:
            config: ResistanceMapConfig object

        Returns:
            Dict of agent_name -> AgentResult

        Raises:
            RuntimeError: If prepare_execution() not called or DAG invalid
            Exception: If any agent fails (logged but not re-raised)
        """
        if self.execution_plan is None:
            raise RuntimeError("Call prepare_execution() before run()")

        # Initialise the run manifest before any agents execute
        self._init_manifest(config)

        start_time = time.time()
        logger.info(
            f"Orchestrator: Starting execution of {len(self.dag.agents)} agents "
            f"in {len(self.execution_plan.layers)} layers"
        )

        if self._tracer is not None:
            try:
                trace = self._tracer.start_trace()
                self._trace_id = trace.trace_id
                logger.info(f"Orchestrator: trace_id={self._trace_id}")
            except Exception:
                logger.exception("Orchestrator: failed to start trace; continuing without tracing")
                self._trace_id = None

        try:
            # Execute each layer
            for layer_idx, layer in enumerate(self.execution_plan.layers):
                logger.info(f"Orchestrator: Layer {layer_idx} ({len(layer)} agents)")

                # Run all agents in layer concurrently
                layer_results = await asyncio.gather(
                    *[
                        self._run_agent_isolated(
                            self.dag.agents[name], name, config
                        )
                        for name in layer
                    ],
                    return_exceptions=False,
                )

                # Store results and update verification chain
                for agent_name, result in zip(layer, layer_results):
                    self.results[agent_name] = result
                    self._verification_chain.append(result.verification_hash)

                    # Log result
                    if result.status == AgentState.COMPLETED:
                        logger.info(
                            f"  {agent_name}: COMPLETED "
                            f"(hash={result.verification_hash[:8]}...)"
                        )
                    else:
                        logger.error(
                            f"  {agent_name}: {result.status.value.upper()} "
                            f"({result.error or 'no error msg'})"
                        )

        except Exception as e:
            logger.exception("Orchestrator: Unexpected error during execution")
            raise

        elapsed = time.time() - start_time
        logger.info(
            f"Orchestrator: Execution complete in {elapsed:.1f}s. "
            f"Completed: {sum(1 for r in self.results.values() if r.status == AgentState.COMPLETED)}/{len(self.dag.agents)}"
        )

        if self._tracer is not None and self._trace_id is not None:
            try:
                self._tracer.end_trace(self._trace_id)
            except Exception:
                logger.debug("Orchestrator: failed to end trace cleanly")

        return self.results

    async def _run_agent_isolated(
        self, agent: BaseAgent, agent_name: str, config: ResistanceMapConfig
    ) -> AgentResult:
        """Run a single agent with input preparation and output verification.

        Simulates isolated execution by:
        1. Gathering inputs from completed dependencies
        2. Preparing input dict with verified outputs
        3. Executing agent with zero-trust wrapper
        4. Recording execution timing

        Args:
            agent: BaseAgent instance
            agent_name: Name of agent (for lookup)
            config: ResistanceMapConfig

        Returns:
            AgentResult
        """
        start_time = time.time()

        # v7: AgentOps span emission. Construct upfront so the `finally` block
        # can close it regardless of execution path.
        span = None
        if self._tracer is not None and self._trace_id is not None:
            try:
                span = self._tracer.start_span(
                    trace_id=self._trace_id,
                    agent_name=agent_name,
                    operation="agent.execute",
                )
            except Exception:
                logger.debug(f"_run_agent_isolated({agent_name}): start_span failed")
                span = None

        try:
            # Prepare inputs from dependencies
            inputs = {}
            for dep_name in agent.dependencies:
                if dep_name not in self.results:
                    return agent._make_result(
                        AgentState.FAILED,
                        error=f"Dependency {dep_name} not executed yet",
                    )

                dep_result = self.results[dep_name]
                if dep_result.status != AgentState.COMPLETED:
                    return agent._make_result(
                        AgentState.FAILED,
                        error=f"Dependency {dep_name} failed: {dep_result.error}",
                    )

                # Zero-trust: Verify dependency output before using
                is_valid, error_msg = agent.verify_output(dep_result)
                if not is_valid:
                    return agent._make_result(
                        AgentState.FAILED,
                        error=f"Dependency {dep_name} failed verification: {error_msg}",
                    )

                inputs[dep_name] = dep_result.output

            # Execute agent with zero-trust wrapper
            result = await agent._safe_execute(inputs, config)

            # Record timing
            self._timing_log[agent_name] = (start_time, time.time())

            # Record artifact paths produced by this agent (if any).
            # Agents report their output files by including an
            # ``artifact_paths`` list in ``result.metadata``.
            if hasattr(self, "_run_manifest"):
                artifact_paths = (result.metadata or {}).get("artifact_paths", [])
                self._run_manifest["per_agent_artifacts"][agent_name] = list(
                    artifact_paths
                )

            # v7: GuardrailEngine pass on the agent's output. We feed the
            # output dict in flat form; rules that aren't applicable simply
            # PASS with severity=info. Any non-passing 'error'-severity rule
            # is recorded as a violation, but does NOT fail the agent (so we
            # surface them in the dashboard rather than silently halt). The
            # user can promote any rule to a hard stop later.
            if (
                self._guardrails is not None
                and result.status == AgentState.COMPLETED
                and isinstance(result.output, dict)
            ):
                try:
                    rule_results = self._guardrails.check_all(result.output)
                    for r in rule_results:
                        if not r["passed"]:
                            v = {
                                "agent": agent_name,
                                "rule": r["guardrail"],
                                "severity": r["severity"],
                                "details": r.get("details", ""),
                            }
                            self._guardrail_violations.append(v)
                            if self._evaluator is not None:
                                try:
                                    self._evaluator.record_guardrail_violation(
                                        agent_name=agent_name,
                                        violation_type=r["guardrail"],
                                        severity=r["severity"],
                                        description=r.get("details", ""),
                                        input_hash=result.verification_hash,
                                    )
                                except Exception:
                                    logger.debug("guardrail eval record failed")
                    # attach summary on result.metadata so it's persisted
                    result.metadata = {
                        **(result.metadata or {}),
                        "guardrails_checked": len(rule_results),
                        "guardrails_failed": sum(1 for r in rule_results if not r["passed"]),
                    }
                except Exception:
                    logger.exception(
                        f"_run_agent_isolated({agent_name}): guardrail engine raised"
                    )

            # v7: AgentOps evaluator records the task outcome.
            if self._evaluator is not None:
                try:
                    self._evaluator.record_task_result(
                        task_id=agent_name,
                        status=result.status.value,
                        verified=(result.status == AgentState.COMPLETED),
                        metadata={
                            "verification_hash": result.verification_hash,
                            "elapsed_s": time.time() - start_time,
                        },
                    )
                except Exception:
                    logger.debug("evaluator.record_task_result failed")

            return result

        except Exception as e:
            logger.exception(f"_run_agent_isolated({agent_name}): Unexpected error")
            self._timing_log[agent_name] = (start_time, time.time())
            return agent._make_result(
                AgentState.FAILED,
                error=f"Unexpected error: {str(e)}",
            )
        finally:
            if span is not None and self._tracer is not None:
                try:
                    self._tracer.end_span(
                        span_id=span.span_id,
                        status="completed" if (
                            agent_name in self.results
                            and self.results[agent_name].status == AgentState.COMPLETED
                        ) else "running",
                        metadata={"elapsed_s": time.time() - start_time},
                    )
                except Exception:
                    logger.debug("end_span failed")

    def get_execution_plan(self) -> list[list[str]]:
        """Get the execution plan (agents grouped by parallelizable layers).

        Returns:
            List of layers, each containing agent names
        """
        if self.execution_plan is None:
            return []
        return self.execution_plan.layers

    def get_verification_chain(self) -> list[str]:
        """Get the audit trail of output hashes.

        Returns:
            List of SHA256 hashes in execution order
        """
        return self._verification_chain.copy()

    def get_timing_report(self) -> dict[str, dict[str, float]]:
        """Get execution timing for each agent.

        Returns:
            Dict of agent_name -> {start_time, end_time, elapsed_seconds}
        """
        report = {}
        for agent_name, (start, end) in self._timing_log.items():
            report[agent_name] = {
                "start_time": start,
                "end_time": end,
                "elapsed_seconds": end - start,
            }
        return report

    def get_results_summary(self) -> dict[str, Any]:
        """Get summary of all agent results.

        Returns:
            Dict with result counts and metadata
        """
        status_counts = defaultdict(int)
        for result in self.results.values():
            status_counts[result.status.value] += 1

        return {
            "total_agents": len(self.results),
            "status_counts": dict(status_counts),
            "verification_chain_length": len(self._verification_chain),
            "failed_agents": [
                name
                for name, result in self.results.items()
                if result.status == AgentState.FAILED
            ],
        }
