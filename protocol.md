# Two-agent coordination protocol (filesystem-based)

Two Claude Code agents (ORCHESTRATOR, WORKER) run in separate terminals on the same
H100 box. They cannot talk directly — they coordinate **only** through this shared
`coordination/` directory via `coord.py` (atomic ops; no daemon).

## The one rule that prevents races
**Each agent writes ONLY its own status + the OTHER agent's inbox + its own outputs.
Neither edits the other's files or branch.** Ownership:

| Path | WORKER | ORCHESTRATOR |
|---|---|---|
| `coordination/worker_status.json` | **write** | read |
| `coordination/orchestrator_status.json` | read | **write** |
| `coordination/inbox_worker/` | read (mine) | **write (to worker)** |
| `coordination/inbox_orchestrator/` | **write (to orch)** | read (mine) |
| `coordination/PLAN.md`, `RUNS.md` | read | **write** |
| `results/**` | **write** | read |
| `paper/**` (the §8 write-up) | — | **write** |
| GPU | **leases via lock** | never trains (CPU-only) |

## Setup (do once, before launching either agent)
```bash
# shared coordination dir OUTSIDE both worktrees so git never touches it:
export COORD_DIR=/abs/path/to/ResistanceMap/coordination     # SAME value in both terminals
# separate git worktrees on separate branches (shared .git, no working-tree collisions):
git worktree add ../rm-worker v20-rq9
git worktree add ../rm-orch   v20-writeup
```
Run WORKER from `../rm-worker`, ORCHESTRATOR from `../rm-orch`. Add `coordination/` to
`.gitignore` (live status files must not be committed).

## coord.py commands (call from the shell between work units)
```bash
python coordination/coord.py heartbeat --role worker --state running --msg "fold 2/5" --task rq9
python coordination/coord.py status --role worker          # read the OTHER agent's status
python coordination/coord.py check  --role worker --stale 900   # alive? exit 0 fresh / 4 stale / 2 unknown
python coordination/coord.py send  --to orchestrator --type RESULT --msg "fold0 done" --data results/rq9/fold0.json
python coordination/coord.py inbox --role worker           # read + archive my messages (use --peek to not archive)
python coordination/coord.py lock   --name gpu0 --owner worker   # exit 0 acquired / 3 busy
python coordination/coord.py unlock --name gpu0 --owner worker
```

## Message types (the shared vocabulary)
`RESULT` (worker→orch: new artifact ready) · `GATE` (orch→worker: a pre-registration gate
verdict / threshold) · `STOP` (orch→worker: halt this line, reason given) · `CONTINUE`
(orch→worker: proceed) · `BLOCKED` (worker→orch: needs human/data) · `NOTE` (either).

## Heartbeat & liveness
Every agent updates its status at least every ~5 min (and before/after any long GPU run,
with `--state running|idle|blocked|done`). The other agent uses `check` to detect a stall
(no update past `--stale`). A held lock older than its TTL (default 2 h) is auto-stealable,
so a crashed agent never deadlocks the GPU.

## Conflict & integrity rules
- Only ORCHESTRATOR commits `RUNS.md`/`PLAN.md`/`paper/`; only WORKER commits code/results on
  `v20-rq9`. Merge to `v20` happens once, by the human, after both report `done`.
- ORCHESTRATOR is the **integrity gate**: it independently re-checks every WORKER result
  against `PREREGISTRATION_LANE2.md` + `PREREG_ADDENDUM_RQ9_confounding.md` before any number
  is allowed toward the paper. A WORKER claim is not "real" until ORCHESTRATOR confirms it.