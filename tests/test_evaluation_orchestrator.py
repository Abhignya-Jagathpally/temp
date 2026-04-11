"""Unit tests for the EvalOrchestrator (evaluation governance layer).

These tests target the public surface of resistancemap.evaluation as
described in the merge contract:
- EvalAgent (subclass of BaseAgent)
- EvalFinding dataclass
- Verdict enum (PASS, FAIL, CONDITIONAL, SKIPPED, BLOCKED)
- EvalOrchestrator (tier-ordered, parallel within tier, Tier A hard stop)
- EvalReport (per_tier_verdicts, verification_chain, timing)

The evaluation package is being authored in parallel; if it is not yet
importable in the current worktree the entire module is skipped with a
clear reason. No real data files are required.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

# Skip the entire module if the evaluation package is not yet present.
evaluation = pytest.importorskip(
    "resistancemap.evaluation",
    reason="resistancemap.evaluation package not yet merged into worktree",
)

from resistancemap.agents.base import AgentResult, AgentState  # noqa: E402
from resistancemap.config import ResistanceMapConfig  # noqa: E402

EvalAgent = evaluation.EvalAgent
EvalFinding = evaluation.EvalFinding
Verdict = evaluation.Verdict
EvalOrchestrator = evaluation.EvalOrchestrator
EvalReport = evaluation.EvalReport


# ---------------------------------------------------------------------------
# Test fixtures: minimal in-process FakeEvalAgent subclasses
# ---------------------------------------------------------------------------


class FakeEvalAgent(EvalAgent):
    """Minimal EvalAgent used for orchestrator behavior tests.

    Records the wall-clock start time of execute() so tests can assert that
    two agents in the same tier overlap in time (i.e. ran in parallel).
    """

    def __init__(
        self,
        name: str,
        tier: str,
        verdict: Verdict = Verdict.PASS,
        delay: float = 0.05,
        score: float = 1.0,
    ):
        super().__init__(name=name, tier=tier)
        self._verdict = verdict
        self._delay = delay
        self._score = score
        self.start_ts: float | None = None
        self.end_ts: float | None = None

    async def execute(
        self, inputs: dict[str, Any], config: Any
    ) -> AgentResult:
        self.start_ts = time.perf_counter()
        await asyncio.sleep(self._delay)
        self.end_ts = time.perf_counter()

        finding = EvalFinding(
            agent_name=self.name,
            tier=self.tier,
            verdict=self._verdict,
            score=self._score,
            criteria_results={"fake_criterion": True},
            evidence=["in-process fixture"],
            failure_modes=[] if self._verdict == Verdict.PASS else ["fake_failure"],
            required_changes=[],
        )
        return self._make_result(
            AgentState.COMPLETED,
            output=finding,
            metadata={"tier": self.tier},
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_executes_tiers_in_order():
    """Tiers must complete strictly in A -> B -> C -> D order."""
    a1 = FakeEvalAgent("a1", tier="A", delay=0.02)
    b1 = FakeEvalAgent("b1", tier="B", delay=0.02)
    c1 = FakeEvalAgent("c1", tier="C", delay=0.02)
    d1 = FakeEvalAgent("d1", tier="D", delay=0.02)

    orch = EvalOrchestrator()
    for agent in (a1, b1, c1, d1):
        orch.add_agent(agent)

    report = await orch.run(ResistanceMapConfig())

    assert isinstance(report, EvalReport)
    # Each agent recorded its own end-time; later tiers must not start before
    # earlier tiers have finished.
    assert a1.end_ts is not None and b1.start_ts is not None
    assert a1.end_ts <= b1.start_ts + 1e-6
    assert b1.end_ts <= c1.start_ts + 1e-6
    assert c1.end_ts <= d1.start_ts + 1e-6


@pytest.mark.asyncio
async def test_orchestrator_hard_stops_on_tier_a_failure():
    """A FAIL in any Tier A agent must BLOCK all later tiers."""
    a_pass = FakeEvalAgent("a_pass", tier="A", verdict=Verdict.PASS)
    a_fail = FakeEvalAgent("a_fail", tier="A", verdict=Verdict.FAIL)
    b1 = FakeEvalAgent("b1", tier="B", verdict=Verdict.PASS)
    c1 = FakeEvalAgent("c1", tier="C", verdict=Verdict.PASS)
    d1 = FakeEvalAgent("d1", tier="D", verdict=Verdict.PASS)

    orch = EvalOrchestrator()
    for agent in (a_pass, a_fail, b1, c1, d1):
        orch.add_agent(agent)

    report = await orch.run(ResistanceMapConfig())

    # Tier A should have run; all later tiers should produce BLOCKED findings.
    findings = {f.agent_name: f for f in report.findings}
    assert findings["a_fail"].verdict == Verdict.FAIL
    assert findings["b1"].verdict == Verdict.BLOCKED
    assert findings["c1"].verdict == Verdict.BLOCKED
    assert findings["d1"].verdict == Verdict.BLOCKED

    # And the blocked agents should NOT have actually executed.
    assert b1.start_ts is None
    assert c1.start_ts is None
    assert d1.start_ts is None


@pytest.mark.asyncio
async def test_orchestrator_parallelizes_within_tier():
    """Two Tier A agents must overlap in wall-clock time."""
    delay = 0.1
    a1 = FakeEvalAgent("a1", tier="A", delay=delay)
    a2 = FakeEvalAgent("a2", tier="A", delay=delay)

    orch = EvalOrchestrator()
    orch.add_agent(a1)
    orch.add_agent(a2)

    t0 = time.perf_counter()
    await orch.run(ResistanceMapConfig())
    elapsed = time.perf_counter() - t0

    # If serial: ~2*delay; if parallel: ~delay. Allow generous slack.
    assert a1.start_ts is not None and a2.start_ts is not None
    assert a1.end_ts is not None and a2.end_ts is not None
    overlap = min(a1.end_ts, a2.end_ts) - max(a1.start_ts, a2.start_ts)
    assert overlap > 0, "Tier A agents must overlap in time (be parallel)"
    assert elapsed < 2 * delay * 0.95, (
        f"Elapsed {elapsed:.3f}s suggests serial execution (expected ~{delay:.3f}s)"
    )


@pytest.mark.asyncio
async def test_orchestrator_eval_report_per_tier_verdicts_populated():
    """EvalReport.per_tier_verdicts must include every tier that ran."""
    agents = [
        FakeEvalAgent("a1", tier="A"),
        FakeEvalAgent("b1", tier="B"),
        FakeEvalAgent("c1", tier="C"),
        FakeEvalAgent("d1", tier="D"),
    ]
    orch = EvalOrchestrator()
    for a in agents:
        orch.add_agent(a)

    report = await orch.run(ResistanceMapConfig())

    assert hasattr(report, "per_tier_verdicts")
    assert set(report.per_tier_verdicts.keys()) >= {"A", "B", "C", "D"}
    # Every tier verdict should be a Verdict enum value (PASS for this happy path).
    for tier, verdict in report.per_tier_verdicts.items():
        assert isinstance(verdict, Verdict)


@pytest.mark.asyncio
async def test_orchestrator_verification_chain_length_matches_executed_agents():
    """Verification chain length == number of agents that actually executed."""
    agents = [
        FakeEvalAgent("a1", tier="A"),
        FakeEvalAgent("a2", tier="A"),
        FakeEvalAgent("b1", tier="B"),
        FakeEvalAgent("c1", tier="C"),
        FakeEvalAgent("d1", tier="D"),
    ]
    orch = EvalOrchestrator()
    for a in agents:
        orch.add_agent(a)

    report = await orch.run(ResistanceMapConfig())

    chain = report.verification_chain
    assert isinstance(chain, list)
    assert len(chain) == len(agents)
    # SHA256 hex digests are 64 chars
    for h in chain:
        assert isinstance(h, str) and len(h) == 64


@pytest.mark.asyncio
async def test_orchestrator_timing_recorded_for_executed_agents():
    """Timing dict must contain entries for executed agents."""
    a1 = FakeEvalAgent("a1", tier="A")
    b1 = FakeEvalAgent("b1", tier="B")

    orch = EvalOrchestrator()
    orch.add_agent(a1)
    orch.add_agent(b1)

    report = await orch.run(ResistanceMapConfig())

    assert hasattr(report, "timing")
    assert "a1" in report.timing
    assert "b1" in report.timing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
