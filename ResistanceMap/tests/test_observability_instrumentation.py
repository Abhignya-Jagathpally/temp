"""Smoke test for ``resistancemap.observability.instrumentation``.

Constraints (per the spec PR these tests gate):

- < 2 seconds, no network, no GPU, no model loading.
- Records *real* events via the public API; never fabricates JSONL on disk.
- Asserts both event count and per-event schema match SPEC.md.

Run:
    pytest -xvs tests/test_observability_instrumentation.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from resistancemap.observability.instrumentation import (
    Recorder,
    TraceContext,
    get_recorder,
    record_guardrail_violation,
    record_handoff,
    record_tool_call,
    set_recorder,
)


# Top-level event keys every JSONL line must carry.
_REQUIRED_TOP_LEVEL = {"schema_version", "run_id", "event_type", "recorded_at", "payload"}

# Required payload keys per event type, mirroring SPEC.md.
_REQUIRED_PAYLOAD = {
    "trace_context": {
        "trace_id",
        "agent_name",
        "parent_span_id",
        "started_at",
        "ended_at",
        "tool",
        "cost_usd",
        "tokens_in",
        "tokens_out",
        "span_id",
    },
    "handoff": {
        "from_agent",
        "to_agent",
        "payload_size_bytes",
        "latency_ms",
        "trace_id",
    },
    "tool_call": {
        "tool_name",
        "latency_ms",
        "succeeded",
        "trace_id",
        "agent_name",
    },
    "guardrail_violation": {
        "rule",
        "evidence",
        "severity",
        "agent_name",
        "trace_id",
    },
}


@pytest.fixture
def recorder(tmp_path: Path):
    """Install a fresh Recorder rooted at a temp dir; tear it down after."""
    rec = Recorder(run_id="smoketest_run", log_root=tmp_path)
    set_recorder(rec)
    try:
        yield rec
    finally:
        set_recorder(None)


def test_helpers_are_inert_when_no_recorder(tmp_path: Path):
    """Calling record_* with no installed recorder must be a silent no-op."""
    set_recorder(None)
    assert get_recorder() is None
    # Should not raise, should not create any files.
    record_handoff("a", "b", 0, 0.0)
    record_tool_call("t", 0.0, True)
    record_guardrail_violation("rule", "evidence")
    assert list(tmp_path.iterdir()) == []


def test_records_real_events_and_schema_matches(recorder: Recorder):
    start = time.perf_counter()

    # 1 trace_context (so the file has the per-span row too)
    ctx = TraceContext(
        trace_id="trace-001",
        agent_name="data_validation",
        parent_span_id=None,
        tool=None,
        cost_usd=0.0,
        tokens_in=0,
        tokens_out=0,
    )
    ctx.ended_at = ctx.started_at + 0.001
    recorder.record_trace_context(ctx)

    # 3 handoffs
    record_handoff("data_validation", "data_prep", payload_size_bytes=1024, latency_ms=0.5,
                   trace_id="trace-001")
    record_handoff("data_prep", "vae_pretrain", payload_size_bytes=2048, latency_ms=0.8,
                   trace_id="trace-001")
    record_handoff("vae_pretrain", "esm2_embed", payload_size_bytes=4096, latency_ms=1.2,
                   trace_id="trace-001")

    # 2 tool calls (one success, one failure — both real branches)
    record_tool_call("Bash", latency_ms=12.3, succeeded=True,
                     trace_id="trace-001", agent_name="data_prep")
    record_tool_call("Read", latency_ms=0.4, succeeded=False,
                     trace_id="trace-001", agent_name="data_prep")

    # 1 guardrail violation
    record_guardrail_violation(
        rule="no_synthetic_data",
        evidence={"path": "data/cache/fake.npy", "detector": "cache_inspector"},
        severity="high",
        agent_name="data_validation",
        trace_id="trace-001",
    )

    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"smoke test budget blown: {elapsed:.3f}s"

    # ---- read back ----
    assert recorder.path.exists()
    with open(recorder.path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]

    # 1 trace_context + 3 handoffs + 2 tool_calls + 1 guardrail = 7 events
    assert len(lines) == 7, f"expected 7 events, got {len(lines)}"

    events = [json.loads(ln) for ln in lines]

    # Top-level shape on every event
    for ev in events:
        assert _REQUIRED_TOP_LEVEL.issubset(ev.keys()), \
            f"missing top-level keys in {ev}"
        assert ev["schema_version"] == 1
        assert ev["run_id"] == "smoketest_run"
        assert ev["event_type"] in _REQUIRED_PAYLOAD
        # Payload schema for this event type
        required = _REQUIRED_PAYLOAD[ev["event_type"]]
        assert required.issubset(ev["payload"].keys()), \
            f"{ev['event_type']} payload missing keys: {required - ev['payload'].keys()}"

    # Counts per event type
    type_counts: dict[str, int] = {}
    for ev in events:
        type_counts[ev["event_type"]] = type_counts.get(ev["event_type"], 0) + 1
    assert type_counts == {
        "trace_context": 1,
        "handoff": 3,
        "tool_call": 2,
        "guardrail_violation": 1,
    }, f"unexpected event-type breakdown: {type_counts}"

    # Spot-check semantic content survived the round trip
    handoffs = [ev["payload"] for ev in events if ev["event_type"] == "handoff"]
    assert {h["from_agent"] for h in handoffs} == {
        "data_validation", "data_prep", "vae_pretrain"
    }
    tool_calls = [ev["payload"] for ev in events if ev["event_type"] == "tool_call"]
    assert {tc["succeeded"] for tc in tool_calls} == {True, False}
    [violation] = [ev["payload"] for ev in events if ev["event_type"] == "guardrail_violation"]
    assert violation["rule"] == "no_synthetic_data"
    assert violation["severity"] == "high"


def test_jsonl_is_append_only(recorder: Recorder):
    """A second batch of events must be appended, not overwritten."""
    record_tool_call("Bash", 1.0, True, trace_id="t", agent_name="a")
    record_tool_call("Bash", 2.0, True, trace_id="t", agent_name="a")
    first_size = recorder.path.stat().st_size

    record_tool_call("Bash", 3.0, True, trace_id="t", agent_name="a")
    second_size = recorder.path.stat().st_size

    assert second_size > first_size, "file must grow on subsequent writes"
    with open(recorder.path, "r", encoding="utf-8") as fh:
        assert sum(1 for _ in fh) == 3
