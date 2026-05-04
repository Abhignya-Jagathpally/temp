"""Append-only JSONL recorder for the ResistanceMap multi-agent layer.

This module is the *opt-in* sidecar described in ``SPEC.md``. It does **not**
replace the in-process tracer at :mod:`resistancemap.agentops.tracer`; it sits
alongside it and writes one JSON object per line under
``logs/<run_id>/observability.jsonl`` for downstream consumption (paper
figures, the AgentOps cloud bridge, weekly aggregation for P5).

Design constraints (intentional, see SPEC):

- **No new pip deps.** Standard library only.
- **Opt-in.** Until the orchestrator calls :func:`set_recorder`, every
  ``record_*`` helper is a no-op. Importing this module from a pipeline path
  is safe even if no run is active.
- **Append-only.** The JSONL file is opened in ``"a"`` mode; events are never
  rewritten. One file per ``run_id``.
- **Process- and thread-safe writes.** A module-level :class:`threading.Lock`
  guards each ``write`` call. Each line is flushed and ``os.fsync``-ed so a
  crash mid-run still preserves the events that were written before it.
- **No network, no GPU, no model loading.** Cheap enough to call from the
  hottest agent path.

Wire-up to the AgentOps cloud SDK is deliberately out of scope for this
module — once that lands, a separate forwarder will tail this JSONL.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class TraceContext:
    """Per-span context emitted alongside every observability event.

    Fields mirror the user-named contract in SPEC.md (Axis 1, O1/O3) so a
    downstream consumer can join JSONL events back to the in-process spans
    in ``resistancemap.agentops.tracer``.

    The two timestamp fields use ``time.time()`` (UNIX seconds, float) rather
    than ``datetime`` objects to keep the JSONL round-trippable without a
    custom decoder.
    """

    trace_id: str
    agent_name: str
    parent_span_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    tool: Optional[str] = None
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSONL (matches asdict semantics)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class Recorder:
    """JSONL recorder bound to a single run.

    Parameters
    ----------
    run_id:
        Stable identifier for the pipeline run. Becomes the directory name
        under ``log_root``.
    log_root:
        Root logs directory. Defaults to ``logs/`` relative to the current
        working directory; if the orchestrator runs from inside the
        ResistanceMap project this resolves to
        ``ResistanceMap/logs/<run_id>/observability.jsonl`` as documented.

    Notes
    -----
    The directory is created lazily on first write so importing this module
    from a unit test does not pollute ``logs/``.
    """

    def __init__(self, run_id: str, log_root: Optional[Path] = None) -> None:
        if not run_id:
            raise ValueError("run_id must be a non-empty string")
        self.run_id = run_id
        self.log_root = Path(log_root) if log_root is not None else Path("logs")
        self.run_dir = self.log_root / run_id
        self.path = self.run_dir / "observability.jsonl"
        self._lock = threading.Lock()
        self._dir_ready = False

    # -- internals ----------------------------------------------------------

    def _ensure_dir(self) -> None:
        if not self._dir_ready:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._dir_ready = True

    def _write(self, event_type: str, payload: Mapping[str, Any]) -> None:
        """Append one JSON line. Holds the lock for the duration of the write."""
        record = {
            "schema_version": 1,
            "run_id": self.run_id,
            "event_type": event_type,
            "recorded_at": time.time(),
            "payload": dict(payload),
        }
        line = json.dumps(record, default=str, sort_keys=True)
        with self._lock:
            self._ensure_dir()
            # Open per write so a crash between writes never leaves a stale
            # file handle; the cost is negligible (< 14 events / agent step).
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    # -- public API ---------------------------------------------------------

    def record_trace_context(self, ctx: TraceContext) -> None:
        """Emit a ``trace_context`` event. Called once per span end."""
        self._write("trace_context", ctx.to_dict())

    def record_handoff(
        self,
        from_agent: str,
        to_agent: str,
        payload_size_bytes: int,
        latency_ms: float,
        trace_id: Optional[str] = None,
    ) -> None:
        """Emit a ``handoff`` event (Axis 1 O2, Axis 3 P3)."""
        self._write(
            "handoff",
            {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "payload_size_bytes": int(payload_size_bytes),
                "latency_ms": float(latency_ms),
                "trace_id": trace_id,
            },
        )

    def record_tool_call(
        self,
        tool_name: str,
        latency_ms: float,
        succeeded: bool,
        trace_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> None:
        """Emit a ``tool_call`` event (Axis 1 O4)."""
        self._write(
            "tool_call",
            {
                "tool_name": tool_name,
                "latency_ms": float(latency_ms),
                "succeeded": bool(succeeded),
                "trace_id": trace_id,
                "agent_name": agent_name,
            },
        )

    def record_guardrail_violation(
        self,
        rule: str,
        evidence: Any,
        severity: str = "medium",
        agent_name: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """Emit a ``guardrail_violation`` event (Axis 2 E2)."""
        self._write(
            "guardrail_violation",
            {
                "rule": rule,
                "evidence": evidence,
                "severity": severity,
                "agent_name": agent_name,
                "trace_id": trace_id,
            },
        )


# ---------------------------------------------------------------------------
# Module-level recorder slot + convenience wrappers
# ---------------------------------------------------------------------------

_recorder_lock = threading.Lock()
_recorder: Optional[Recorder] = None


def set_recorder(recorder: Optional[Recorder]) -> None:
    """Install the active recorder. Pass ``None`` to disable.

    The orchestrator calls this once at start-of-run and once at end-of-run.
    Pipeline code that calls ``record_*`` between those two points emits to
    the JSONL; outside that window, the helpers are no-ops.
    """
    global _recorder
    with _recorder_lock:
        _recorder = recorder


def get_recorder() -> Optional[Recorder]:
    """Return the active recorder, if any. Mainly for tests."""
    with _recorder_lock:
        return _recorder


def record_handoff(
    from_agent: str,
    to_agent: str,
    payload_size_bytes: int,
    latency_ms: float,
    trace_id: Optional[str] = None,
) -> None:
    """Inert no-op unless :func:`set_recorder` has installed a Recorder."""
    rec = get_recorder()
    if rec is None:
        return
    rec.record_handoff(from_agent, to_agent, payload_size_bytes, latency_ms, trace_id)


def record_tool_call(
    tool_name: str,
    latency_ms: float,
    succeeded: bool,
    trace_id: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> None:
    """Inert no-op unless :func:`set_recorder` has installed a Recorder."""
    rec = get_recorder()
    if rec is None:
        return
    rec.record_tool_call(tool_name, latency_ms, succeeded, trace_id, agent_name)


def record_guardrail_violation(
    rule: str,
    evidence: Any,
    severity: str = "medium",
    agent_name: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> None:
    """Inert no-op unless :func:`set_recorder` has installed a Recorder."""
    rec = get_recorder()
    if rec is None:
        return
    rec.record_guardrail_violation(rule, evidence, severity, agent_name, trace_id)
