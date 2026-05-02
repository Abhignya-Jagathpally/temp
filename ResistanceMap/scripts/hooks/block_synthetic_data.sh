#!/usr/bin/env bash
# PreToolUse hook for Edit/Write/MultiEdit on ResistanceMap.
# Blocks edits that introduce np.random / synthetic-cohort patterns
# in non-test code, enforcing the user's hard "no synthetic data" rule.
#
# Reads tool_input from stdin (JSON). Exits 0 (allow), 2 (deny + feedback to Claude).

set -uo pipefail

INPUT=$(cat)

FILE_PATH=$(printf '%s' "$INPUT" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    ti = d.get('tool_input', {})
    print(ti.get('file_path', ''))
except Exception:
    pass
")

# Pull every text payload (new_string for Edit, content for Write, edits[*].new_string for MultiEdit).
PAYLOAD=$(printf '%s' "$INPUT" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    ti = d.get('tool_input', {})
    parts = []
    if 'content' in ti:
        parts.append(ti['content'] or '')
    if 'new_string' in ti:
        parts.append(ti['new_string'] or '')
    for e in ti.get('edits', []) or []:
        parts.append((e or {}).get('new_string', '') or '')
    print('\n'.join(parts))
except Exception:
    pass
")

# Only police paths under ResistanceMap/, and exempt tests + this hook itself + memory MD files.
case "$FILE_PATH" in
    */ResistanceMap/tests/*) exit 0 ;;
    */ResistanceMap/scripts/hooks/*) exit 0 ;;
    *.md) exit 0 ;;
    */ResistanceMap/*) ;;
    *) exit 0 ;;
esac

# Patterns to deny.
violations=()
if printf '%s' "$PAYLOAD" | grep -qE 'np\.random\.default_rng|numpy\.random\.default_rng'; then
    violations+=("np.random.default_rng — fabrication pattern")
fi
if printf '%s' "$PAYLOAD" | grep -qE 'np\.random\.seed|numpy\.random\.seed'; then
    # allow seed-setting in deterministic-mode block of main.py (very narrow)
    if ! printf '%s' "$PAYLOAD" | grep -q 'config.hardware.deterministic'; then
        violations+=("np.random.seed in non-test code")
    fi
fi
if printf '%s' "$PAYLOAD" | grep -qE '_generate_synthetic_cohort\s*\('; then
    violations+=("_generate_synthetic_cohort call — disarmed in v7")
fi

if [ ${#violations[@]} -gt 0 ]; then
    {
        echo "[fabrication-sentinel] BLOCKED edit to $FILE_PATH"
        for v in "${violations[@]}"; do echo "  - $v"; done
        echo "Reason: violates user's no-synthetic-data rule (see memory: feedback_no_synthetic_data.md)."
        echo "Fix: load real data via data/mmrf_loader.py or evaluation/external_validation real path."
    } >&2
    exit 2
fi

exit 0
