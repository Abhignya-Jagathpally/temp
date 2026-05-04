# v11 Agentic-Layer Justification Audit

**Date authored:** 2026-05-03
**Status:** Honest audit. NOT promotional copy. Each section answers a specific yes/no question with evidence from the repo.
**Companion:** `docs/V11_SOTA_UPGRADE_PLAN.md` §3.

---

## §1 The skeptical question

> "What is most importantly contributing to the *agentic* aspect? Is it really required? If yes, is it perfect? What unique contribution is it bringing into the pipeline?"

Phrased that way the question splits into three independent claims:

- **C1: The agentic layer is *required* for ResistanceMap to do its job.**
- **C2: The agentic layer is currently *correct* (no silent failures, no fabrication risk, no scaffold-for-its-own-sake).**
- **C3: The agentic layer is the *unique contribution* of v11.**

This document answers each as a hard yes / no / partial, with file:line evidence.

---

## §2 What the "agentic layer" actually consists of

To audit honestly, list the components that exist on disk today, not what they are *named*:

| Layer | File | LoC | What it is | What it produces |
|---|---|---|---|---|
| Stage agents (in-pipeline) | `resistancemap/agents/specialized.py` | 475 | 10 named classes — `DataValidationAgent`, `DataPrepAgent`, `VAEPretrainAgent`, `ESM2EmbedAgent`, `VAEFinetuneAgent`, `TrajectoryAgent`, `ProteinNetAgent`, `FusionAgent`, `LandscapeAgent`, `ValidationAgent`. Each subclasses `BaseAgent`. | Runs deterministic numerical work + checkpoints. |
| Stage orchestrator | `resistancemap/agents/orchestrator.py` | 510 | DAG executor that schedules the stage agents in topological order. | Run logs. |
| Specialized review subagents | `.claude/agents/*.md` (sota-comparator, literature-deep-researcher, causal-inference-auditor, fabrication-sentinel, release-bouncer, …) | n/a | Claude-runtime subagents with scoped tool sets. | Review verdicts, citation-backed reports. |
| AgentOps tracer | `resistancemap/agentops/tracer.py` | 465 | OpenTelemetry-style spans + traces for any code that opts in. | `logs/agentops/<trace_id>.jsonl`. |
| AgentOps evaluator | `resistancemap/agentops/evaluator.py` | 449 | Task-completion / guardrail / accuracy / clinical / first-pass tracking. | Eval metrics + violation events. |
| AgentOps optimizer | `resistancemap/agentops/optimizer.py` | 508 | Token efficiency / retrieval@k / handoff success / flow efficiency / improvement velocity. | Optimization metrics. |
| AgentOps dashboard | `resistancemap/agentops/dashboard.py` | 345 | JSON + console rollup of the three pillars. | `agentops_dashboard.json`. |
| Observability instrumentation | `resistancemap/observability/instrumentation.py` | 265 | Decorators + context managers used by stage agents to emit spans. | spans → tracer. |
| MCP integration | `resistancemap/mcp_integration/connectors.py` | 806 | LRU cache + token-bucket rate limiter + 4 connector classes (PubMed, ChEMBL, ClinicalTrials, HuggingFace). | **STUBS** — every `.query()` raises `NotImplementedError`. |
| Latent compute scheduler | `resistancemap/latent_compute/scheduler.py` | 570 | Resource-aware scheduling for opt-in latent compute steps. | Scheduling decisions. |

Total: **~5,400 LoC of agent / agentops scaffolding.**

There are also Claude-runtime subagents listed in the project CLAUDE.md (sota-comparator, evidence-curator, fabrication-sentinel, release-bouncer, architecture-chair, …). These are **not** Python code — they are Claude prompts that run in the parent session.

So "agentic layer" splits into two genuinely different things:
- **(a) In-pipeline Python agents** — the 10 `*Agent` classes plus the AgentOps stack.
- **(b) Claude-runtime review subagents** — the prompt-based reviewers spawned via the `Agent` tool.

The audit must answer C1/C2/C3 separately for (a) and (b).

---

## §3 Claim C1: "The agentic layer is required."

### §3.1 Is the (a) in-pipeline Python agent layer required? — **NO, not as currently structured.**

The 10 stage agents in `specialized.py` are deterministic numerical work. Each one is a thin wrapper over a function that:

1. Loads checkpoints/data from disk.
2. Runs a forward pass / training step / metric computation.
3. Saves results.

There is **no LLM, no decision-making, no tool use** inside any of these classes. They could be Make-target shell scripts and the pipeline would behave identically. Evidence: the v10 sprints (`scripts/v10/s1*.py` … `scripts/v10/s7_*.py`, 30+ scripts) are exactly that — bare Python with no agent wrapper — and they produced the entire v10 strict-pass set without invoking anything from `specialized.py`.

So the `*Agent` naming is **architectural cosplay**: it suggests "agents" where the work is "functions in topological order". The DAG orchestrator (`orchestrator.py`, 510 lines) is functionally equivalent to `make` or `snakemake`.

**Verdict on C1(a): NOT required.** A `Makefile` or `snakemake` config would replace it without loss of capability.

There is one exception: the **fabrication-sentinel** and **release-bouncer** Python hooks (under `resistancemap/verification/` and CI scripts) are NOT in `specialized.py` but are pre-commit / pre-tag gates that DO encode the unique policy of this repo (no synthetic data, every numeric claim ties to a `RUNS.md` row). Those are required and are correctly named "agents" in the policy-enforcement sense.

### §3.2 Is the (b) Claude-runtime review subagent layer required? — **YES, partially.**

The Claude subagents (sota-comparator, literature-deep-researcher, evidence-curator, causal-inference-auditor, etc.) are **the only mechanism in the repo** that:

- Pulls live citations from PubMed / bioRxiv / ChEMBL / ClinicalTrials.gov / Open Targets / HuggingFace.
- Audits whether language has migrated from Pearl L1 to L2.
- Runs proteomics-pathway validation against KEGG/Reactome/CPTAC.
- Runs cross-disease F3 framing checks.
- Synthesizes uniqueness briefs.

Without them, the pipeline would still run and produce the same numbers — but the *paper* could not be written, because every claim in the manuscript needs citation backing and the in-pipeline `connectors.py` stubs explicitly refuse to provide it (they raise `NotImplementedError` rather than fabricate).

This was demonstrated, painfully, in the work that preceded this audit:

- The user spawned `sota-comparator` and `literature-deep-researcher` in parallel.
- Both subagents had **MCP search access denied** in their isolated session contexts.
- Both produced deliverables flagged `[UNVERIFIED]` on every line.
- The parent session (with MCP access) had to run a **post-hoc verification sweep** that overturned a major positioning claim — iMLGAM had been queued as the primary MM comparator; it is in fact pan-cancer ICB and is now rejected. The MM-specific comparator (mmSYGNAL, [PMID 40169765](https://pubmed.ncbi.nlm.nih.gov/40169765/), [DOI](https://doi.org/10.1038/s41416-025-02987-6)) was not even on the radar until the verification sweep ran.

The lesson: the review-subagent layer **is** required, but its *value* is exactly proportional to its tool access. Subagents without MCP access are decorative; subagents with MCP access settle questions the human cannot.

**Verdict on C1(b): REQUIRED, but only when MCP access is granted.**

### §3.3 Is the AgentOps three-pillar layer required? — **PARTIALLY OBSERVING (was: NOT YET).**

`tracer.py` + `evaluator.py` + `optimizer.py` + `dashboard.py` together total ~1,750 LoC. They are well-scaffolded but:

- Only the v8 evaluator is wired into stage execution.
- No v10 sprint script emits a span via `tracer`.
- The dashboard JSON `logs/agentops_dashboard.json` is git-tracked (`M ResistanceMap/logs/agentops_dashboard.json` in current diff) but its contents are likely from the v8 demo run, not from the v10 sprints that produced the strict-pass results.

So the AgentOps layer **exists as scaffolding** but does not **observe the work that actually generated the v10 results**. That's a gap, not a contradiction — the v11 plan §4 explicitly calls for wiring AgentOps into every v11 stage so observability/evaluation/optimization metrics emit automatically — but the layer is not yet earning its complexity budget.

**Verdict on C1(AgentOps): would-be-required IF the v11 wiring lands.** Today it is dead code in the critical path of any actual run.

#### §3.3.1 Update 2026-05-03 (post-v11 wiring) — **OBSERVABILITY pillar is now live for v11 stage scripts.**

The wiring task (#70) landed in this turn. Concretely:

- A 100-LoC context manager `resistancemap/observability/v11_stage_decorator.py::v11_stage(name)` now wraps the entry point of all six v11 sprint scripts: `s1f_ode_forward_diagnostic`, `s7_v11_dps_neural_ode`, `s3_gat_f6_replication`, `s5_v11_conformal_waddington_features`, `s5c_discrimination_metrics`, `s5d_cox_discrimination`.
- The wrapper opens a tracer span, records start/end timestamps, captures stdout byte-volume as a coarse cost proxy, records the exit status as an Evaluator `TaskResult`, and writes a per-stage JSON to `logs/agentops/v11/<stage>_<trace_id>.json`.
- One stage (`s1f`, the lightest) was re-run end-to-end to verify the wiring produces a real trace. It does: `logs/agentops/v11/s1f_ode_forward_diagnostic_<uuid>.json` records a 1.30 s span with 983 stdout chars and `status=completed`.
- A consolidated three-pillar report lives at `paper/v8_artifacts/v11_sprint_agentops/three_pillar_report.json`.

What changed in the verdict:

| Pillar | Before this turn | After this turn |
|---|---|---|
| Observability | Zero v10/v11 sprints emit spans. | All six v11 sprint entry points emit spans on run. One real trace exists today. |
| Evaluation | Task completion / guardrail violation tracked only by the v8 demo. | TaskResult emitted per v11 stage on run; guardrail-violation count for v11 = 0 (no fabrication-sentinel hits, no LLM in v11 stages so no LLM-side guardrails to trip). |
| Optimization | Scaffolding. | Still scaffolding — the v11 stage scripts are pure-numeric (PyTorch + lifelines), so token-efficiency / retrieval@k / handoff-success metrics have no LLM substrate. The cost-proxy `len(stdout)/1000` is a byte-volume signal, not a token cost. |

**What is still scaffolding:** the Optimizer pillar, and the wiring of the *Claude* review subagents (sota-comparator, literature-deep-researcher, causal-inference-auditor, fabrication-sentinel) into the same Tracer/Evaluator/Optimizer instance. Those subagents *do* invoke an LLM and therefore *can* emit non-trivial Optimizer metrics — but the bridge from the Claude runtime to `resistancemap.agentops` is the v12 task. The Optimizer file remains imported but unused in v11.

**What is no longer scaffolding:** the Observability pillar for v11 sprint scripts. Anyone re-running an instrumented v11 stage now leaves a trace artifact on disk that the dashboard can consume. The file `logs/agentops/v11/s1f_ode_forward_diagnostic_<uuid>.json` is the existence proof.

**Updated verdict on C1(AgentOps):** Observability pillar — REQUIRED and WIRED. Evaluation pillar — REQUIRED and PARTIALLY WIRED (TaskResult only; guardrails not exercised by pure-numeric stages). Optimization pillar — SCAFFOLDING; will require Claude-subagent wiring (v12) to earn its complexity.

---

## §4 Claim C2: "The agentic layer is currently correct."

### §4.1 Silent-failure audit — **PARTIAL.**

| Component | Silent-failure-resistant? | Evidence |
|---|---|---|
| Stage agents (`specialized.py`) | YES — exceptions propagate, checkpoints fail-loud. | `BaseAgent.execute()` does not swallow. |
| MCP connectors (`connectors.py`) | YES (post v7 disarming). | `PubMedConnector.query()` line 255-260 raises `NotImplementedError` rather than returning empty results. This was a fix; the v6 version returned `{"articles": [], "status": "success"}` which was indistinguishable from a real zero-hit query. The disarming is the right call. |
| Tracer | UNTESTED in production. | No v10 sprint has actually written a span. Whether the LRU cache + flush behavior is correct under load is unknown. |
| Evaluator / Optimizer | UNTESTED in production. | Same. |
| Subagent dispatching | **FAILS SILENTLY when MCP access is denied.** | Demonstrated 2026-05-03: both background research agents returned deliverables that *looked* complete but flagged every entry `[UNVERIFIED]`. The system did NOT fail loudly when MCP permissions were absent; it produced low-quality output that still parsed as success. |

The silent-failure mode in the subagent dispatch is the most important defect today. A subagent without its tool access should **refuse to produce output**, not produce best-effort output flagged unverified — because a downstream operator may not read the flags. The fix is to add a pre-flight check in every subagent prompt that asserts tool availability and exits with a hard error if any required MCP server is unreachable.

### §4.2 Fabrication-resistance audit — **YES (this is genuinely well-built).**

The repo has a `fabrication-sentinel` hook that runs on edits to `evaluation/`, `data/`, `scripts/`, `paper/`. It is the single feature that prevents synthetic-data leaks. Project memory says: *"ResistanceMap's generate_paper_figures.py uses np.random in most panels — do not call generate_all() for real figures"* — and that exact failure mode is what `fabrication-sentinel` blocks. Combined with `release-bouncer` (every numeric claim ties to a `RUNS.md` row) this is a working zero-trust guarantee.

This subset of the agentic layer **IS** uniquely valuable. It is also the smallest part — maybe 200 LoC across the two hooks plus the `RUNS.md` ledger.

### §4.3 Determinism audit — **YES.**

Stage agents are deterministic given fixed seeds + frozen configs. The orchestrator runs in a fixed topological order. Nothing inside the in-pipeline agent layer makes nondeterministic calls. (Claude subagents are nondeterministic by nature, but they sit *outside* the result-producing path: their outputs land in `paper/v8_artifacts/*.md`, not in the numerical results.)

### §4.4 "Is it perfect" — **NO.**

Three concrete defects, ranked by severity:

1. **`mcp_integration/connectors.py` is 806 lines of stub.** Every connector class raises `NotImplementedError`. The cache + rate-limiter scaffolding is unused because the methods that would use them never run. This is dead code competing for attention with code that actually runs.
2. **AgentOps is not wired into the v10 result-producing path.** The dashboard JSON exists; it just doesn't reflect the runs that produced the strict-pass numbers. v11 plan §4 fixes this; today it is unfixed.
3. **Subagent silent-failure on MCP denial.** Demonstrated above. The fix is one pre-flight assertion per subagent prompt.

None of these are architectural-rot-level — but the gap between "scaffolding exists" and "scaffolding earns its complexity" is real.

---

## §5 Claim C3: "The agentic layer is the unique contribution of v11."

This is the most interesting question.

### §5.1 What v10 demonstrably showed is unique

Per `docs/V10_SPRINT7_VERDICT.md` §"Aggregate v10 strict-spec status" and `RUNS.md` rows r-2026-05-03-v10s1 through v10s7:

- Pre-registered F1–F10 + F_S5 + F8-DPS gates with strict thresholds dated *before* the data was looked at.
- 7 strict-pass / 1 partial / 4 expected-fails — the expected-fails *confirm* a pre-stated identifiability-limit thesis.
- Bounded-claims rule with `release-bouncer` enforcement.
- Mondrian per-stratum conformal coverage exceeding spec (5/5 strata within ±3% of nominal 0.90).
- Cross-disease F3 refutation on Beat AML giving an empirical bound on MM-specificity.

The unique contribution of the *paper* is the falsification framework + the bounded-claims rule + the cross-disease bound + the identifiability annotation. **None of those require the in-pipeline agent layer.** They require the falsification spec doc + `RUNS.md` + `fabrication-sentinel` + `release-bouncer`.

### §5.2 What the agentic layer adds on top

The honest answer is **process, not content**. The agentic layer:

| Function | Provided by | Replaces what alternative? |
|---|---|---|
| Live evidence retrieval for paper claims | Claude review subagents + MCP servers | Manual literature search by the author |
| Independent SOTA comparison | sota-comparator subagent | Author's biased self-assessment |
| Causal-claim language audit | causal-inference-auditor subagent | Author's blind spot ("we sometimes write 'predict' where we mean 'correlate'") |
| Cross-stage observability | AgentOps tracer (when wired) | Ad-hoc print statements |
| Per-claim provenance | release-bouncer + RUNS.md | Trust the README |
| Synthetic-data leak detection | fabrication-sentinel | Trust the author |

Three of those (causal audit, fabrication detection, release gating) are **scientific contributions** because they prevent specific failure modes that *do* happen in published ML-oncology papers (overclaiming L2 from L1, np.random in figures, untraceable numbers in tables). The remaining three (evidence retrieval, SOTA comparison, observability) are **process improvements** — useful, but not novel and not the paper's headline.

### §5.3 Is the agentic layer the unique contribution? — **NO.**

The unique contribution is the falsification + bounded-claims + cross-disease-bound + identifiability-annotation framework. The agentic layer is a *quality-assurance scaffold* that helps that framework hold up. The framework can survive without the scaffold (with more effort and more author-error risk). The scaffold cannot survive without the framework — it would just be code with no claims to gate.

The honest framing for v11 is:

> "v11's scientific contribution is the v10 falsification framework, applied at higher predictive resolution via a Helmholtz-projected Neural-ODE on a foundation-model-warmed Waddington landscape. The multi-agent layer is the QA scaffold around the framework — not the contribution itself. Without the falsification frame, the agents would have nothing to verify."

---

## §6 What this means for v11 sequencing

1. **Keep the Claude review subagents.** They earn their cost when MCP access works. Add a pre-flight tool-availability check to each subagent prompt to convert silent failures into loud ones.
2. **Wire AgentOps into the v10 result-producing path** before adding more components. If the v11 sprints produce numbers and the AgentOps trace logs are still empty, the AgentOps layer remains scaffolding. (Task #70.)
3. **Either ship `connectors.py` as a real wrapper around the MCP tools, or delete the stubs and let the subagents own MCP access.** 806 lines of `NotImplementedError` is technical debt with no upside.
4. **Do NOT promote the in-pipeline `*Agent` classes as the contribution.** They are deterministic stage-runners; the falsification framework is what is novel.
5. **DO promote `fabrication-sentinel` + `release-bouncer` + `RUNS.md` + the falsification spec.** Those genuinely encode the unique zero-trust posture.

---

## §7 The bottom line

- **Is the agentic layer required?** Partially. The Claude review subagents (when MCP access works) and the policy-enforcement hooks (fabrication-sentinel, release-bouncer) ARE required. The 10 in-pipeline `*Agent` classes are NOT — they are stage functions named misleadingly. The AgentOps three-pillar stack is not yet earning its complexity budget.
- **Is it perfect?** No. Three defects: (a) `connectors.py` is 806 lines of unused stub, (b) AgentOps is not yet wired into the v10 path, (c) subagents fail silently when MCP access is denied.
- **Is it the unique contribution?** No. The unique contribution is the falsification + bounded-claims + cross-disease-bound + identifiability-annotation framework. The agentic layer is the QA scaffold around it — useful, but not the headline.

If the user pushes back on this audit and asks "then why have the agentic layer at all?", the honest answer is: **the policy-enforcement subset (fabrication-sentinel + release-bouncer) is the real win**, and the other 5,000+ lines of scaffolding need to either earn their place via the v11 wiring (task #70) or be deleted.
