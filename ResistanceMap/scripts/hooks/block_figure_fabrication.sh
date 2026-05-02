#!/usr/bin/env bash
# PreToolUse hook for Bash. Blocks calls that would invoke fabricating
# figure-generation entry points (generate_paper_figures.py:generate_all,
# or any --all flag), redirecting to scripts/run_real_figures.py.

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

# If the command does not even mention paper figures, allow.
case "$CMD" in
    *generate_paper_figures.py*) ;;
    *) exit 0 ;;
esac

# generate_all / --all / no-arg invocation hits fabricating panels.
if echo "$CMD" | grep -qE 'generate_paper_figures\.py(\s|$)|generate_all|--all|--full'; then
    {
        echo "[figure-validator] BLOCKED: $CMD"
        echo "Reason: scripts/generate_paper_figures.py uses np.random.default_rng in"
        echo "        most panels (Fig 2 heatmap+calibration, Figs 3-5, S1-S5)."
        echo "Safe alternative: bash -c 'python scripts/run_real_figures.py'"
    } >&2
    exit 2
fi

exit 0
