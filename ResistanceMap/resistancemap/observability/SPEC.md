# ResistanceMap Observability / Evaluation / Optimization Spec

**Status:** Draft v1 (2026-05-03). Authoritative metric definitions for the
multi-agent layer of ResistanceMap. Pairs with
`resistancemap/observability/instrumentation.py` (JSONL emitter) and
`resistancemap/agentops/` (in-process tracer/evaluator/optimizer).

This document fixes the **mathematical definition**, **data source**,
**aggregation window**, **unit**, and **emission status** for each of the
fourteen metrics the multi-agent system commits to reporting. The goal is
that any value displayed in `logs/agentops_dashboard.json`, the AgentOps
cloud dashboard, or the paper can be traced back to a single line in this
spec.

Conventions:

- **Run** = one invocation of `resistancemap.main` (or any orchestrator
  that calls `Tracer.start_trace()`). One `run_id` per run; events go to
  `logs/<run_id>/observability.jsonl`.
- **Trace** = one `Trace` object inside `resistancemap.agentops.tracer`.
  In ResistanceMap there is currently exactly one trace per run; the spec
  is written to allow multiple in the future.
- **Span** = one `Span` (one agent doing one operation) inside a trace.
- **Tool call** = leaf operation: an MCP tool, a Bash command, a
  `Read`/`Write`, an HTTP fetch, or a deterministic compute kernel.
- **Status “emitted”** values:
  - `live` — value is already written by current code.
  - `partial` — value is recorded but its definition or aggregation is
    incomplete vs. this spec.
  - `gap` — instrumentation does not yet exist; new code required.

---

## Axis 1 — Observability (4 metrics)

| # | Metric | Definition | Data source | Window | Unit | Status |
|---|---|---|---|---|---|---|
| O1 | End-to-end trace duration | For each trace `t`: `D(t) = end_time(t) − start_time(t)`. Reported per pipeline run, per agent (sum of agent's span durations), and per tool call (single span duration). | `Trace.total_duration_ms`, `Span.duration_ms` (agentops tracer) | per run; rolling p50/p95 over last N runs | milliseconds | live |
| O2 | Agent-to-agent handoff latency | Given spans of agents `A_i` and `A_{i+1}` ordered by `start_time`, `H(i) = start_time(A_{i+1}) − end_time(A_i)`. Negative values indicate overlap (parallelism) and are reported as-is, not clipped. | `Trace.get_agent_handoff_latencies()` (existing) **plus** new `record_handoff` events for explicit producer→consumer pairs | per run | milliseconds | partial — current code infers from time order, not declared edges |
| O3 | Cost per request | LLM-backed agents: `C = Σ (input_tokens · price_in + output_tokens · price_out)` from per-span metadata. Compute agents (training, embedding): `C_compute = wall_clock_seconds · $/sec` per device class (CPU/GPU). Reported per span, per agent, and per run. | `Span.metadata.{input_tokens, output_tokens, cost_usd}` and a new `compute_class` field. Pricing table lives in `configs/agentops_pricing.yaml` (TBD). | per span (raw), per run (sum) | USD | partial — token cost field exists; compute pricing table missing |
| O4 | Tool execution latency | For each leaf span whose `operation` matches a tool name (MCP, Bash, Read/Write, HTTP): `L(tool) = duration_ms(span)`. Aggregated as p50/p95/p99 per tool name. | `Trace.get_tool_execution_latencies()` plus new `record_tool_call` events tagging the tool kind explicitly | per run; per tool name | milliseconds | partial — currently aggregated by `operation` string only; no kind/tool taxonomy |

---

## Axis 2 — Evaluation (5 metrics)

| # | Metric | Definition | Data source | Window | Unit | Status |
|---|---|---|---|---|---|---|
| E1 | Task completion rate (per agent) | For agent `A`: `TCR(A) = #{tasks where A produced its declared deliverable AND verified=True} / #{tasks assigned to A}`. A deliverable is "declared" via `Agent.declared_outputs` (string list); "produced" means each declared path exists on disk OR the orchestrator's task ledger marks it as written. | `Evaluator.record_task_result(...)` plus a new `declared_outputs` registry per agent | per run; per agent | ratio in [0,1] | partial — `TaskResult.verified` exists but is not currently checked against declared outputs |
| E2 | Guardrail violation rate (per run) | `GVR = #{guardrail violations} / #{spans in run}`. Violations include: fabrication (synthetic data flagged by data-quality checks), unauthorized git commit, pip install of disallowed deps, write to forbidden paths (`/etc`, `~/.ssh`, etc.). Each violation has `severity ∈ {low, medium, high, critical}`. | `record_guardrail_violation` (this module) + existing `GuardrailViolation` records in `evaluator.py` | per run | ratio in [0,1] | partial — engine exists; rule set documented but not exhaustive |
| E3 | Factual accuracy rate | For each claim emitted by an agent, an independent verifier returns `verified ∈ {True, False}` by checking the claim against either (a) the URL the claim cites, refetched, or (b) a row in the run ledger / dataset of record. `FAR = #{verified} / #{claims}`. | `AccuracyCheck` records (existing) — but they are currently created opportunistically by agents, not by an independent verifier | per run; per agent | ratio in [0,1] | gap — needs cite-and-verify pass; no independent verifier wired today |
| E4 | Clinical appropriateness | 5-point Likert, mean over all agent outputs that present a clinical recommendation, scored against the rubric defined in `docs/CLINICAL_ACTIONABILITY_RUBRIC.md`. Until the rubric exists, this is reported as `null` and the dashboard surfaces "rubric pending". | `ClinicalCheck` records (existing) — score field currently free-form; needs rubric binding | per run | mean of {1,2,3,4,5} | gap — rubric document not yet authored |
| E5 | First-pass approval rate | `FPAR = #{outputs that passed initial review with zero rework} / #{all outputs reviewed}`. "Rework" means the orchestrator triggered a re-run of the same span with revised inputs. | `FirstPassCheck` records (existing) plus orchestrator span-replay events (new) | per run; per agent | ratio in [0,1] | partial — record type exists; replay events not yet emitted |

---

## Axis 3 — Optimization (5 metrics)

| # | Metric | Definition | Data source | Window | Unit | Status |
|---|---|---|---|---|---|---|
| P1 | Prompt token efficiency | `PTE = useful_output_tokens / (input_tokens + output_tokens)` per LLM span, averaged over the run. `useful_output_tokens` = tokens that survive into a downstream agent's input (i.e., the producer's output that the consumer actually reads). Fall back to `output_tokens / input_tokens` when downstream consumption cannot be measured. | `TokenUsageRecord` (existing) + handoff payload sizes from `record_handoff` | per run | ratio | partial — fallback formula live; "useful" half not implemented |
| P2 | Retrieval precision @ k | For each retrieval call (Open Targets, PubMed, ChEMBL, etc.) returning the top-k results: `P@k = #{relevant results in top-k} / k`. Relevance is judged by the consuming agent (it tags items it actually used). Reported for k ∈ {1, 5, 10}. | `RetrievalRecord` (existing) — needs the consumer to mark which results it used | per run; per tool | ratio | partial — record schema present; relevance feedback missing |
| P3 | Handoff success rate | For each declared edge `A_i → A_j`: `HSR_ij = 1` iff `A_j` consumed `A_i`'s output (read the file or accepted the in-memory payload) AND did not request a re-run. `HSR(run) = mean over edges`. | `record_handoff` events paired with consumer-side `record_tool_call(tool="read_handoff_payload", ...)` | per run | ratio in [0,1] | gap — neither side currently emits |
| P4 | Flow step efficiency | `FSE = on_task_wall_clock / total_wall_clock`, where `on_task_wall_clock = Σ span.duration_ms` (with overlap collapsed via interval union) and `total_wall_clock = trace.total_duration_ms`. Idle time = `1 − FSE`. | Existing `Trace.spans` + interval-union pass (new) | per run | ratio in (0,1] | partial — components exist; union pass missing |
| P5 | Improvement velocity | Week-over-week % change in any of {O1 median, O3 sum, E1 mean, E5 mean, P1 mean, P3 mean, P4 mean}. `IV(metric) = (metric_this_week − metric_last_week) / max(\|metric_last_week\|, ε)`. Used as the headline "we got better" number for paper claims. | New: `logs/observability_history.jsonl` + a weekly cron that aggregates and appends one summary row | rolling 7-day windows | ratio (signed) | gap — history file and cron do not exist |

---

## Current vs. Target — Gap Analysis

**What's already wired (live):**

- O1 end-to-end duration — `Tracer.export_metrics()` and `agentops_dashboard.json`.
- (Foundation only) E1/E2/E5 record types in `agentops/evaluator.py`, P1/P2/P3/P4 record types in `agentops/optimizer.py`.

**What's partial (definition lives in code but does not match this spec):**

- O2 handoffs are inferred by sort order, not by declared producer→consumer edges. Adds noise when agents run concurrently.
- O3 cost has token fields but no compute pricing table.
- O4 tool latencies are bucketed by free-form `operation` string, not a tool taxonomy.
- E1 verification does not check declared outputs against disk.
- E2 has the engine but not a complete rule manifest.
- E5 lacks span-replay events from the orchestrator.
- P1 uses the fallback formula only.
- P2 lacks consumer-side relevance tagging.
- P4 lacks the interval-union pass.

**What's a true gap (no code today):**

- E3 independent cite-and-verify pass. Today, agents self-attest.
- E4 the rubric document `docs/CLINICAL_ACTIONABILITY_RUBRIC.md` does not exist; clinical scores will read `null` until it does.
- P3 handoff success rate (both producer and consumer events).
- P5 weekly history file and aggregation cron.

**Closure plan (this PR):**

This PR adds the JSONL sidecar (`instrumentation.py`) that is the
common substrate for the three currently-gap metrics (E3, P3, P5) and
for the partial ones that need explicit producer/consumer events
(O2, O4, P3). It is **opt-in** — until the orchestrator calls
`set_recorder(...)`, all four `record_*` helpers are inert no-ops, so
this change cannot break any existing pipeline path.

Subsequent PRs will:

1. Author `docs/CLINICAL_ACTIONABILITY_RUBRIC.md` (closes E4).
2. Add a verifier agent that consumes claims from the JSONL and writes
   `AccuracyCheck` records (closes E3).
3. Add `configs/agentops_pricing.yaml` and a compute-cost decorator (O3).
4. Wire the orchestrator to call `record_handoff` / `record_tool_call`
   at every agent boundary (closes O2, O4, P3).
5. Add a weekly aggregation script writing
   `logs/observability_history.jsonl` (closes P5).
