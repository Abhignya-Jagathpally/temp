"""ResistanceMap observability layer (formal spec + JSONL instrumentation).

This package defines the 14-metric spec across three axes
(Observability / Evaluation / Optimization) and provides a minimal,
opt-in JSONL recorder. It does NOT replace the in-process
``resistancemap.agentops`` tracer; it is an append-only sidecar that
will be the bridge to the AgentOps cloud SDK once that wiring lands.

Public API:
    TraceContext           — dataclass capturing per-span fields.
    Recorder               — JSONL recorder bound to a run_id.
    record_handoff(...)    — convenience wrappers using a module-level Recorder.
    record_tool_call(...)
    record_guardrail_violation(...)
    set_recorder(...)/get_recorder() — explicit lifecycle for tests/orchestrator.

See ``SPEC.md`` (sibling file) for the metric definitions.
"""

from .instrumentation import (
    Recorder,
    TraceContext,
    get_recorder,
    record_guardrail_violation,
    record_handoff,
    record_tool_call,
    set_recorder,
)

__all__ = [
    "TraceContext",
    "Recorder",
    "set_recorder",
    "get_recorder",
    "record_handoff",
    "record_tool_call",
    "record_guardrail_violation",
]
