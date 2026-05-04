"""v11 stage observability wrapper.

Minimal context manager wiring resistancemap.agentops Tracer + Evaluator
around v11 sprint scripts. (a) span around stage, (b) start/end timestamps,
(c) exit status -> Evaluator TaskResult, (d) len(stdout)/1000 cost proxy
(no LLM in v11 stages), (e) per-stage JSON at logs/agentops/v11/.
"""

from __future__ import annotations

import io
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from resistancemap.agentops import Evaluator, Tracer


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOG_DIR = _REPO_ROOT / "logs" / "agentops" / "v11"


class _Tee(io.TextIOBase):
    """Mirror writes to a buffer + the original stream so length is measurable."""

    def __init__(self, original, buffer: io.StringIO):
        self._original = original
        self._buffer = buffer

    def write(self, s: str) -> int:
        self._buffer.write(s)
        return self._original.write(s)

    def flush(self) -> None:
        self._original.flush()


@contextmanager
def v11_stage(name: str):
    """Context manager that emits an AgentOps span + TaskResult for a v11 stage."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    tracer = Tracer.get_instance()
    evaluator = Evaluator()
    trace = tracer.start_trace()

    stdout_buf = io.StringIO()
    saved_stdout = sys.stdout
    sys.stdout = _Tee(saved_stdout, stdout_buf)

    t_start = time.time()
    status = "completed"
    error_msg = ""
    span_dict = {}
    try:
        with tracer.trace_agent(name, "v11_stage_execution", trace.trace_id) as span:
            span.metadata["stage"] = name
            span.metadata["start_wall"] = t_start
            try:
                yield span
            except BaseException as e:  # noqa: BLE001
                status = "failed"
                error_msg = f"{type(e).__name__}: {e}"
                span.metadata["error"] = error_msg
                raise
            finally:
                t_end = time.time()
                stdout_text = stdout_buf.getvalue()
                cost_proxy = len(stdout_text) / 1000.0
                span.metadata["end_wall"] = t_end
                span.metadata["duration_s"] = t_end - t_start
                span.metadata["stdout_chars"] = len(stdout_text)
                span.metadata["cost_proxy_kchar"] = cost_proxy
                span.metadata["status"] = status
                span_dict = {
                    "stage": name,
                    "trace_id": trace.trace_id,
                    "span_id": span.span_id,
                    "status": status,
                    "error": error_msg,
                    "start_wall": t_start,
                    "end_wall": t_end,
                    "duration_s": t_end - t_start,
                    "stdout_chars": len(stdout_text),
                    "cost_proxy_kchar": cost_proxy,
                    "metadata": dict(span.metadata),
                }
    finally:
        sys.stdout = saved_stdout
        tracer.end_trace(trace.trace_id)
        evaluator.record_task_result(
            task_id=f"v11::{name}",
            status=status,
            verified=(status == "completed"),
            metadata={"trace_id": trace.trace_id, "duration_s": time.time() - t_start},
        )
        if not span_dict:
            span_dict = {
                "stage": name,
                "trace_id": trace.trace_id,
                "status": status,
                "error": error_msg,
                "duration_s": time.time() - t_start,
            }
        out = _LOG_DIR / f"{name}_{trace.trace_id}.json"
        out.write_text(json.dumps(span_dict, indent=2, default=str))
