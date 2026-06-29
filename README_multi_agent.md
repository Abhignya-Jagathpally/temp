# Two parallel Claude Code agents — launch guide

A WORKER (GPU builder) and an ORCHESTRATOR (integrity gate + writer) run in two terminals,
coordinating through a shared `coordination/` dir (no network). The orchestrator independently
verifies every worker result against the pre-registration — a built-in second pair of eyes.

## Why two agents (and when it's worth it)
The RQ9 build is a long GPU job; meanwhile the orchestrator enforces the stop-rule/gates and
writes the §8 ceiling paper. Parallel = faster + an independent integrity check. Cost = a
little coordination overhead; the protocol below removes the race risk.

## Launch (3 steps)
```bash
# 1. copy these into your repo, set the SHARED coord path in BOTH terminals
cp -r multi_agent/coordination <repo>/         &&  cd <repo>
export COORD_DIR=$PWD/coordination ; echo "coordination/" >> .gitignore

# 2. separate worktrees so the two agents never fight over the working tree / git index
git worktree add ../rm-worker v20-rq9
git worktree add ../rm-orch   v20-writeup

# 3. two terminals:
#   Terminal A (cd ../rm-orch):    export COORD_DIR=...  ; claude  -> paste ORCHESTRATOR_PROMPT.md
#   Terminal B (cd ../rm-worker):  export COORD_DIR=...  ; claude  -> paste WORKER_PROMPT.md
```
Both must `export COORD_DIR` to the **same absolute path**. Assign the free GPU to the worker
(`CUDA_VISIBLE_DEVICES`); the orchestrator stays CPU-only.

## How they check each other
- WORKER, every loop: `coord.py inbox` first (obey STOP/CONTINUE), then heartbeat, work, send RESULT.
- ORCHESTRATOR, every loop: `coord.py check --role worker` (liveness) + `inbox` (results) → GATE → write.
- A stalled agent is detected by `check` (stale heartbeat); a crashed agent's GPU lock auto-expires (TTL).

## Files
- `coordination/coord.py` — the primitives (verified: atomic status, inbox messaging, mkdir locks, staleness).
- `coordination/PROTOCOL.md` — the contract (ownership table, message types, rules).
- `ORCHESTRATOR_PROMPT.md`, `WORKER_PROMPT.md` — paste one into each terminal.