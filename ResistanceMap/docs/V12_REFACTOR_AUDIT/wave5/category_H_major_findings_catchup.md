# Wave 5 — Category H Major Findings Catchup

**Source agent:** `a980067ad2738719a`
**Date:** 2026-05-07
**Persisted by:** main thread (subagent Write was blocked by sandbox)

---

I have enough verification. Let me compose the report.

---

# Verification + Patches: 8 MAJOR Findings

**Working repo:** `/home/aj0486@students.ad.unt.edu/pipeline3/.claude/worktrees/tier1-refactor` (worktree of `pipeline3/ResistanceMap`).
**Important caveat upfront:** the file paths in the brief (`evaluation/external_validation.py`, `evaluation/fairness_audit.py`, `models/identifiable_ode.py`, `models/disentangled_vae.py`) **do not exist** in this codebase. I verified by `find` and `grep`. Either (a) the brief was written against a different snapshot/branch, or (b) those files were renamed/removed. Where the symptom maps to a real file under a different name, I verified there; where it does not, I report unable-to-reproduce. This is the honest answer; please do not let me invent file:line evidence for a phantom finding.

Source-of-truth code paths:
- `resistancemap/evaluation/tier_a/bias_fairness.py`, `tier_c/clinical_safety.py`, `tier_d/chair.py`
- `resistancemap/agentops/{tracer,optimizer,evaluator,dashboard}.py`
- `resistancemap/latent_compute/scheduler.py`
- `resistancemap/mcp_integration/connectors.py`
- `resistancemap/main.py`, `resistancemap/models/{vae,trajectory,fusion,protein_network}.py`

---

## §M1 External-cohort validator

**Verification.** `evaluation/external_validation.py:262` does **not exist**. No file matches `external_validation*.py` (verified by `find -name 'external_validation*' -> 0 hits`). The closest equivalents are `tier_c/clinical_safety.py` (291 LOC) and `tier_d/chair.py` (453 LOC), which both reference "external-cohort clinical validity evidence" as a *required* input but do not implement loaders. The four cohorts named in the brief (IFM2009, MRC Myeloma XI, HOVON-65, MMRC) appear nowhere in `resistancemap/` (`grep -ri "IFM2009\|HOVON\|MMRC" resistancemap/ -> 0 hits`).

**Verdict: NOT REPRODUCIBLE in this snapshot.** No `NotImplementedError` to remove. The honest gap is different: there is *no* external-cohort loader module at all — the responsibility is silently delegated to upstream "evidence" dicts.

**Patch (proposed, do-not-apply per hard rules).** Create `resistancemap/evaluation/external_cohorts.py` with one stub per cohort that raises a typed `DataNotAvailableError("DUA initiation required for {cohort}")`. Wire it into `tier_c/clinical_safety.py` so that absence is a SKIPPED verdict with a `required_changes` entry, not silent.

**Effort.** 0.5 day (4 stubs + 1 wiring point + 1 unit test that asserts SKIPPED).

---

## §M2 Fairness audit linregress shape mismatch

**Verification.** `evaluation/fairness_audit.py` does **not exist**. No `linregress` call exists anywhere in `resistancemap/` (`grep -rn linregress resistancemap/ -> 0 hits`). The actual fairness module is `resistancemap/evaluation/tier_a/bias_fairness.py` (225 LOC); I read it in full. It does **not** compute ADMIXTURE proportions, does **not** call scipy regressions, and does **not** invent demographic columns — it explicitly returns `SKIPPED` when manifest+ancestry doc are absent (lines 146–152) and otherwise `CONDITIONAL` with `required_changes`. The agent is honest scaffolding, not a wrong-shape regression.

**Verdict: NOT REPRODUCIBLE.** The bug described doesn't exist in this codebase; the equivalent module is well-behaved.

**Patch.** None required against current code. **If** an ADMIXTURE call is later added to `metrics/shift.py`, follow brief's recommendation: do not approximate ADMIXTURE (Alexander, Novembre, Lange 2009) with scipy least-squares — flag as `EXPERIMENTAL_PLACEHOLDER` and call out via the existing `_call_shift_metric` "todo" branch (already structurally present at lines 169–224).

**Effort.** 0 today; gate any future PR that adds ADMIXTURE.

---

## §M3 Orphan model code (`identifiable_ode.py`, `disentangled_vae.py`)

**Verification.** `find . -name 'identifiable*' -o -name 'disentangled*' -> 0 hits`. `grep -rn 'identifiable_ode\|disentangled_vae' --include='*.py' . -> 0 hits`. README does **not** market "identifiable ODEs" or "disentangled VAEs"; it markets a plain VAE + Neural ODE (which **are** implemented in `models/vae.py`, `models/trajectory.py` and **are** imported by `main.py:298, 324, 355, 378–380, 572–575`). All four model files in `resistancemap/models/` (`vae.py`, `trajectory.py`, `fusion.py`, `protein_network.py`) have live importers in `main.py`.

**Verdict: NOT REPRODUCIBLE.** No orphan model code in this snapshot. README claims match implementation.

**Patch.** None.

**Effort.** 0.

---

## §M4 Tracer span stack thread-id keying breaks under asyncio.gather

**Verification.** `resistancemap/agentops/tracer.py` lines 227, 293–296, 327–329. The `_span_stack` is typed `Dict[int, List[str]]` keyed by `threading.get_ident()` (line 227), set on `start_span` (293–296), popped on `end_span` (327–329). Under `asyncio.gather(...)` all coroutines on a single event-loop thread share **one** `thread_id`, so concurrent agents push/pop into one shared LIFO. `end_span` doesn't even check that the popped id matches `span_id` — it just pops the top regardless — guaranteeing mis-attribution under any out-of-order completion. (Line numbers differ slightly from the brief's `:305` but the bug is real.)

**Verdict: REPRODUCIBLE.**

**Patch (replacement code, do-not-apply).**
```python
# tracer.py top
import contextvars
# replace `_span_stack: Dict[int, List[str]]` field with:
_current_span_stack: contextvars.ContextVar[tuple[str, ...]] = \
    contextvars.ContextVar("rm_span_stack", default=())

# in start_span (~L293):
stack = self._current_span_stack.get()
self._current_span_stack.set(stack + (span.span_id,))

# in end_span (~L327): match-on-pop, not blind-pop
stack = self._current_span_stack.get()
if stack and stack[-1] == span_id:
    self._current_span_stack.set(stack[:-1])
else:
    # out-of-order end: drop just this id, keep ordering
    self._current_span_stack.set(tuple(s for s in stack if s != span_id))
```
Justification: PEP 567 `contextvars` (Python 3.7+) is the canonical async-safe per-task storage; `asyncio.Task` automatically copies the current `Context` on creation (see Python docs `asyncio-task` § "Tasks and Coroutines" and `contextvars` § "asyncio support"), giving each gathered coroutine its own logical stack without locks.

**Effort.** ~1 day (patch + 1 stress test with `asyncio.gather` of 32 nested spans + delete `_thread_lock` paths around the stack only).

---

## §M5 Optimizer-evaluator loop is dead ceremony

**Verification.** `resistancemap/agentops/optimizer.py` defines `record_token_usage` (L170), `record_retrieval` (L197), `record_handoff` (L223), `record_flow_step` (L249), `record_metric_snapshot` (L272), `suggest_optimizations` (L426). `grep -rn 'record_token_usage\|record_retrieval\|record_handoff\|record_flow_step' --include='*.py' .` returns **only** the definitions in `optimizer.py` plus a single docstring example in `agentops/__init__.py:28`. **Zero production callers.** `dashboard.py:64,117,337,345` reads `optimizer.get_metrics()` / `suggest_optimizations()` — but those metrics are never written. `suggest_optimizations` (L426–479) is hard-coded threshold strings ("below 70%", "below 90%", etc.).

**Verdict: REPRODUCIBLE.** Dead loop confirmed.

**Patch (recommend deletion).** Delete `record_*` methods, the empty private deques they push to, and the `suggest_optimizations` thresholds. Replace `dashboard.py` calls with a 1-line note: `"OptimizerTracker disabled — implement bandit in v13"`. Open a v13 sprint ticket: real implementation = LinUCB (Li et al. 2010 "A Contextual-Bandit Approach to Personalized News Article Recommendation") or Thompson sampling (Chapelle & Li 2011) over the {agent_choice × prompt_template} arms with regret monitoring.

**Effort.** Deletion: 0.5 day. Real bandit implementation (v13 sprint): ~5–7 days.

---

## §M6 Latent compute scheduler never started

**Verification.** `resistancemap/latent_compute/scheduler.py:216–224` — `_check_completed_jobs` body is literally `pass` after a comment "Placeholder: would check actual task status in production". `resistancemap/main.py:159` instantiates `lc_scheduler = LatentComputeScheduler()` and lines 161–162 read `lc_scheduler.gpu_memory_budget_gb` and `.max_concurrent_jobs` for **logging only**. `grep -n 'lc_scheduler\.\|scheduler.start\|scheduler.submit'` returns those two log lines and nothing else. The scheduler is constructed, logged, and abandoned. Note `_execute_job` (L264–) **does** mark jobs completed inline (L291–294), so the placeholder doesn't actively corrupt anything — but no job is ever submitted in the first place.

**Verdict: REPRODUCIBLE.** Scheduler is decorative.

**Patch (do-not-apply).**
```python
# scheduler.py L216
async def _check_completed_jobs(self) -> None:
    for job_id in list(self._running_jobs):
        job = self._running_jobs[job_id]
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            self._completed_jobs[job_id] = job
            self._gpu_memory_used_gb -= job.estimated_memory_gb
            del self._running_jobs[job_id]

# main.py after L159
await lc_scheduler.start()
# … existing pipeline …
await lc_scheduler.stop()
```
Plus: actually `submit()` something (e.g. wrap the VAE pretrain stage). Currently no latent-compute results depend on the scheduler running — it's load-bearing for **future** parallelization, not for any current artifact. So this is a "wire it up before next sprint" not "results are wrong now".

**Effort.** Implement loop + wire start/stop: 1 day. First real submission (VAE pretrain): 1–2 days.

---

## §M7 MCP connectors are silent stubs (worse than `NotImplementedError`)

**Verification.** `resistancemap/mcp_integration/connectors.py` has 4 connector classes — `PubMedConnector` (L219), `ChEMBLConnector` (L366), `ClinicalTrialsConnector` (L499), `HuggingFaceConnector` (L593). The brief says they `raise NotImplementedError`; **the actual behavior is worse**: each `query()` returns a hard-coded `{"total_results": 0, "articles": [], "status": "success"}` response (e.g. L253–259 in PubMed). `validate_response` then **passes** the empty stub because all required fields are present. So callers see a successful query with zero hits and no exception — silent fabrication of "no evidence found".

`grep -rn 'PubMedConnector\|ChEMBLConnector\|ClinicalTrialsConnector\|HuggingFaceConnector\|MCPOrchestrator' --include='*.py' . | grep -v mcp_integration/` → **0 callers.** Dead + silently-wrong.

**Verdict: REPRODUCIBLE and worse than reported.**

**Patch (recommend deletion, follow brief's option (b)).** Delete `connectors.py` and `mcp_integration/__init__.py`; replace with a 5-line `mcp_integration/README.md`: "MCP access is via Claude Code subagents (PubMed/bioRxiv/ChEMBL/HuggingFace/Synapse); see `agentops/` for the orchestration surface." If reinstated later, the in-process connectors must (i) hit a real endpoint or (ii) raise `MCPNotConfiguredError` — never return success+empty.

**Effort.** Deletion + README: 0.5 day.

---

## §M8 RUNS.md → README hand-off

**Verification.** `RUNS.md` is **not present** in this worktree (`find . -iname 'RUNS*' -> 0 hits`). The pre-conversation git status shows `M ResistanceMap/RUNS.md` modified in the parent checkout, so this worktree branched before/without it. I cannot extract specific row content (rows 21, 31, 32, 36, 37) here.

**Verdict: PARTIALLY REPRODUCIBLE — finding is real (per memory: Sprint 5 conformal 5/5 ±3%, Geneformer F2-leak rejection, paired C-index, T1114 specialist STRICT FAIL, MOFA-vs-v11.5 are confirmed real outcomes per `MEMORY.md` v10 sprint records and the docs added under `ResistanceMap/docs/V12_*`); but row-level extraction not possible from this worktree.**

**Patch (coordination directive for W5.F).**
1. W5.F README rewrite must include a "Sprint 5 / v10 / v11.5 / v12 outcomes" section listing:
   - Sprint 5 Mondrian JK+ conformal: 5/5 strata within ±3% (project memory `project_v10_sprint4_5_outcome.md`)
   - Geneformer F2-leak rejection (negative result, publishable)
   - Paired C-index test on N_paired=29
   - T1114 specialist STRICT FAIL (`docs/V12_T1114_SPECIALIST_VERDICT.md`)
   - MOFA vs v11.5 (`docs/V12_MOFA_VS_V115_VERDICT.md`)
2. Each bullet must cite the originating RUNS.md row and the `paper/v8_artifacts/...` artifact path.
3. Open a one-line check in CI: `grep -q "Sprint 5" README.md && grep -q "T1114" README.md` so this regression doesn't reopen.

**Effort.** Coordination + README diff: 0.5 day inside W5.F's existing scope.

---

## §summary

| # | Finding | Verified? | Recommended action |
|---|---|---|---|
| M1 | `external_validation.py:262` NotImplementedError | **NO** — file does not exist | Build `external_cohorts.py` stubs (DataNotAvailableError) and wire into Tier C |
| M2 | `fairness_audit.py:111-114` linregress shape bug | **NO** — file does not exist; real `bias_fairness.py` is honest scaffolding | None now; gate future ADMIXTURE adds |
| M3 | Orphan `identifiable_ode.py`, `disentangled_vae.py` | **NO** — files do not exist; README claims match real models | None |
| M4 | `tracer.py` thread-id keyed `_span_stack` corrupts under asyncio.gather | **YES** (lines 227, 293–296, 327–329) | Switch to `contextvars.ContextVar` (PEP 567) + match-on-pop |
| M5 | Optimizer record_*/suggest_optimizations dead loop | **YES** — zero callers; suggestions are hard-coded | **Delete** dead code; v13 sprint for LinUCB/Thompson |
| M6 | `scheduler._check_completed_jobs` is `pass`; never started | **YES** (L216–224, main.py:159) | Implement loop + add `start/stop` in main.py; no current results depend on it |
| M7 | MCP connectors raise NotImplementedError | **YES** (and worse — silent empty success); zero callers | **Delete** in-process connectors; document subagents as the surface |
| M8 | RUNS.md publishable rows missing from README | **PARTIAL** — RUNS.md not in this worktree; outcomes confirmed via memory + docs/V12_* | Coordinate with W5.F: enumerate Sprint 5 / T1114 / MOFA / Geneformer / paired C-index in README |

**Net.** Of 8 findings, **4 are reproducible** (M4, M5, M6, M7), **3 are not** (M1, M2, M3 — files do not exist in this snapshot), **1 is partially verifiable here** (M8 — outcomes real, file not present). Recommend: ship M4 (correctness fix, 1 day), delete M5/M7 (0.5 day each), wire M6 before any parallel-VAE work, fold M8 into W5.F. M1 needs a real loader module if external-cohort claims are kept in any future README. M2 and M3 should be re-checked against whichever branch produced the original audit — they may be legitimate findings in a different snapshot.

No source files were modified (per hard rule). All patches are proposed in this report only.