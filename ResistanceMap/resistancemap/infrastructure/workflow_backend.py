"""Thin workflow-backend adapter.

Provides a uniform interface for running pipeline steps across different
execution backends (local subprocess, Airflow, Spark).  Contains NO pipeline
logic -- only command construction, execution, and result capture.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public enums / dataclasses
# ---------------------------------------------------------------------------


class WorkflowBackend(Enum):
    """Supported execution backends."""

    LOCAL = "local"
    AIRFLOW = "airflow"
    SPARK = "spark"


@dataclass
class RunConfig:
    """Configuration for a single pipeline step execution.

    Parameters
    ----------
    backend : WorkflowBackend
        Which execution backend to use.
    repo_root : Path
        Absolute path to the repository root (used to resolve relative script
        paths).
    log_dir : Path
        Directory where stdout/stderr capture files are written.
    env_vars : dict[str, str]
        Extra environment variables injected into the subprocess.
    gpu_device : str | None
        CUDA_VISIBLE_DEVICES value.  ``None`` means inherit from parent.
    dry_run : bool
        If True, ``run_step`` prints the command but does not execute it.
    spark_master : str
        Spark master URL used when backend is SPARK.
    spark_deploy_mode : str
        Spark deploy mode (``client`` or ``cluster``).
    spark_conf : dict[str, str]
        Extra ``--conf key=value`` pairs for spark-submit.
    airflow_queue : str
        Airflow queue name emitted in BashOperator-compatible output.
    timeout : float | None
        Per-step wall-clock timeout in seconds.  ``None`` means no limit.
    """

    backend: WorkflowBackend = WorkflowBackend.LOCAL
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    env_vars: dict[str, str] = field(default_factory=dict)
    gpu_device: str | None = None
    dry_run: bool = False
    spark_master: str = "local[*]"
    spark_deploy_mode: str = "client"
    spark_conf: dict[str, str] = field(default_factory=dict)
    airflow_queue: str = "default"
    timeout: float | None = None


@dataclass
class StepResult:
    """Outcome of a single pipeline step execution.

    Attributes
    ----------
    exit_code : int
        Process return code (0 = success).  ``-1`` for dry-run.
    stdout_path : Path | None
        File where stdout was captured, if any.
    stderr_path : Path | None
        File where stderr was captured, if any.
    wall_seconds : float
        Wall-clock seconds spent in the step.
    artifact_paths : list[Path]
        Any output artefacts detected after the step completes.
    command : list[str]
        The resolved command that was (or would be) executed.
    """

    exit_code: int
    stdout_path: Path | None
    stderr_path: Path | None
    wall_seconds: float
    artifact_paths: list[Path] = field(default_factory=list)
    command: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def build_command(
    script_path: str | Path,
    args: Sequence[str] = (),
    config: RunConfig | None = None,
) -> list[str]:
    """Construct the shell command for a given backend.

    Parameters
    ----------
    script_path : str | Path
        Path to the Python script (absolute or relative to *repo_root*).
    args : Sequence[str]
        Positional / flag arguments forwarded to the script.
    config : RunConfig | None
        Execution configuration.  Uses ``RunConfig()`` defaults when *None*.

    Returns
    -------
    list[str]
        Tokenised command ready for ``subprocess.run`` (LOCAL) or
        stringification (AIRFLOW / SPARK).
    """
    cfg = config or RunConfig()
    script = Path(script_path)
    if not script.is_absolute():
        script = cfg.repo_root / script

    if cfg.backend is WorkflowBackend.LOCAL:
        return _build_local(script, args)
    elif cfg.backend is WorkflowBackend.AIRFLOW:
        return _build_airflow(script, args, cfg)
    elif cfg.backend is WorkflowBackend.SPARK:
        return _build_spark(script, args, cfg)
    else:
        raise ValueError(f"Unsupported backend: {cfg.backend!r}")


def _build_local(script: Path, args: Sequence[str]) -> list[str]:
    """Plain ``python <script> [args...]``."""
    import sys

    return [sys.executable, str(script), *args]


def _build_airflow(
    script: Path, args: Sequence[str], cfg: RunConfig
) -> list[str]:
    """Construct BashOperator-compatible command string.

    Airflow's BashOperator runs the command via ``bash -c``, so we produce
    tokens that can be joined with spaces for that purpose.
    """
    import sys

    env_prefix_parts: list[str] = []
    merged_env = dict(cfg.env_vars)
    if cfg.gpu_device is not None:
        merged_env["CUDA_VISIBLE_DEVICES"] = cfg.gpu_device
    for k, v in sorted(merged_env.items()):
        env_prefix_parts.append(f"{k}={v}")

    cmd_parts = [*env_prefix_parts, sys.executable, str(script), *args]
    return cmd_parts


def _build_spark(
    script: Path, args: Sequence[str], cfg: RunConfig
) -> list[str]:
    """Wrap a script invocation with ``spark-submit``."""
    cmd: list[str] = [
        "spark-submit",
        "--master",
        cfg.spark_master,
        "--deploy-mode",
        cfg.spark_deploy_mode,
    ]
    for k, v in sorted(cfg.spark_conf.items()):
        cmd.extend(["--conf", f"{k}={v}"])
    cmd.append(str(script))
    cmd.extend(args)
    return cmd


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------


def run_step(
    script_path: str | Path,
    args: Sequence[str] = (),
    config: RunConfig | None = None,
) -> StepResult:
    """Execute a pipeline step and capture results.

    Parameters
    ----------
    script_path : str | Path
        Script to run (absolute or relative to *repo_root*).
    args : Sequence[str]
        Arguments forwarded to the script.
    config : RunConfig | None
        Execution configuration.

    Returns
    -------
    StepResult
        Contains exit code, log paths, wall time, and detected artifacts.
    """
    cfg = config or RunConfig()
    cmd = build_command(script_path, args, cfg)

    # Derive a human-friendly step name for log files.
    step_name = Path(script_path).stem

    # Ensure log directory exists.
    log_dir = cfg.log_dir
    if not log_dir.is_absolute():
        log_dir = cfg.repo_root / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = log_dir / f"{step_name}.stdout.log"
    stderr_path = log_dir / f"{step_name}.stderr.log"

    if cfg.dry_run:
        logger.info("[dry-run] %s", " ".join(cmd))
        return StepResult(
            exit_code=-1,
            stdout_path=None,
            stderr_path=None,
            wall_seconds=0.0,
            artifact_paths=[],
            command=cmd,
        )

    # Build subprocess environment.
    env = os.environ.copy()
    env.update(cfg.env_vars)
    if cfg.gpu_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = cfg.gpu_device

    logger.info("Running step %r: %s", step_name, " ".join(cmd))
    t0 = time.monotonic()

    with open(stdout_path, "w") as f_out, open(stderr_path, "w") as f_err:
        try:
            proc = subprocess.run(
                cmd,
                stdout=f_out,
                stderr=f_err,
                env=env,
                cwd=str(cfg.repo_root),
                timeout=cfg.timeout,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            logger.error("Step %r timed out after %s s", step_name, cfg.timeout)
            exit_code = 124  # conventional timeout exit code

    wall = time.monotonic() - t0
    logger.info(
        "Step %r finished: exit_code=%d wall=%.1fs", step_name, exit_code, wall
    )

    return StepResult(
        exit_code=exit_code,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        wall_seconds=wall,
        artifact_paths=[],
        command=cmd,
    )


# ---------------------------------------------------------------------------
# Workflow YAML loader
# ---------------------------------------------------------------------------


def load_workflow_config(yaml_path: str | Path) -> dict[str, Any]:
    """Load a workflow definition from a YAML file.

    Parameters
    ----------
    yaml_path : str | Path
        Path to a YAML file (e.g. ``configs/workflows/mortfm_airflow.yaml``).

    Returns
    -------
    dict[str, Any]
        Parsed workflow configuration.

    Raises
    ------
    FileNotFoundError
        If *yaml_path* does not exist.
    """
    import yaml  # deferred -- yaml is only needed when loading configs

    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Workflow config not found: {p}")
    with open(p) as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    logger.debug("Loaded workflow config from %s (%d keys)", p, len(data))
    return data
