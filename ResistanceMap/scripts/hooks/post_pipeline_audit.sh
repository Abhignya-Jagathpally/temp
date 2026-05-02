#!/usr/bin/env bash
# PostToolUse hook for Bash. After a successful `python main.py` invocation,
# auto-emits a one-line health summary so the agent is forced to confront
# the dashboard reality (real spans? real metrics?) instead of trusting
# stdout.

set -uo pipefail

INPUT=$(cat)

CMD=$(printf '%s' "$INPUT" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print((d.get('tool_input') or {}).get('command', ''))
except Exception:
    pass
")

# Only act on main.py runs.
echo "$CMD" | grep -qE 'python.*main\.py' || exit 0

cd /home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap 2>/dev/null || exit 0

DASH=logs/agentops_dashboard.json
VALIDATED=checkpoints/pipeline_validated.pt

if [ ! -f "$DASH" ]; then
    echo "[post-pipeline-audit] no agentops_dashboard.json — pipeline did not complete normally" >&2
    exit 0
fi

python3 - <<'PY' 2>&1
import json, sys, os
try:
    import torch
except Exception:
    print("[post-pipeline-audit] torch unavailable in hook env"); sys.exit(0)

dash = json.load(open("logs/agentops_dashboard.json"))
obs = dash.get("observability", {})
ev = dash.get("evaluation", {})
print(f"[post-pipeline-audit] traces={obs.get('total_traces',0)} spans={obs.get('total_spans',0)} "
      f"task_completion={ev.get('task_completion_rate',0.0)} "
      f"guardrail_violations={ev.get('guardrail_violation_rate',0.0)}")
if obs.get("total_traces", 0) == 0:
    print("[post-pipeline-audit] WARNING: AgentOps emitted 0 traces — observability path is dead", file=sys.stderr)

p = "checkpoints/pipeline_validated.pt"
if os.path.exists(p):
    c = torch.load(p, map_location="cpu", weights_only=False)
    m = c.get("metrics", {}) if isinstance(c, dict) else {}
    print(f"[post-pipeline-audit] aggregate test_mse={m.get('test_mse','?')} "
          f"per_drug_present={'per_drug_metrics' in m}")
    if "per_drug_metrics" not in m:
        print("[post-pipeline-audit] WARNING: per_drug_metrics missing — v7 fix regressed", file=sys.stderr)
PY

exit 0
