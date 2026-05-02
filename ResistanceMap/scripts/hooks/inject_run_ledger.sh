#!/usr/bin/env bash
# UserPromptSubmit hook. When the user prompt mentions "publish",
# "release", "paper", "submission", or "freeze", inject a snapshot
# of RUNS.md so the agent treats the run ledger as authoritative.

set -uo pipefail

INPUT=$(cat)

PROMPT=$(printf '%s' "$INPUT" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('prompt', '') or '')
except Exception:
    pass
")

if ! echo "$PROMPT" | grep -qiE 'publish|release|paper|submission|freeze|ICML|ICLR|NeurIPS'; then
    exit 0
fi

LEDGER=/home/aj0486@students.ad.unt.edu/pipeline3/ResistanceMap/RUNS.md
[ -f "$LEDGER" ] || exit 0

cat <<EOF
<system-reminder>
This prompt mentions release/paper/submission. The authoritative run ledger is below.
Every numeric claim must cite a row here; if a row has no W&B link or persistent log,
the number it produced cannot appear in docs. Use release-bouncer subagent before tagging.

$(head -40 "$LEDGER")
</system-reminder>
EOF
exit 0
