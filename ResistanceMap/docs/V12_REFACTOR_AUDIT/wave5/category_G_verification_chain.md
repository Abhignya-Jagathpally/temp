# Wave 5 — Category G: Verification Chain Audit (W5.G)

**Author:** general-purpose agent (verified at file:line; persisted by main thread)
**Date:** 2026-05-07

## §1 Verification of the bug

**`resistancemap/agents/base.py:137-144`** (`verify_output`):
```python
computed_hash = self.compute_hash(result.output)
if computed_hash != result.verification_hash:
    return False, f"Hash mismatch: {computed_hash} != {result.verification_hash}"
if expected_hash and expected_hash != computed_hash:
    return False, f"Expected hash {expected_hash}, got {computed_hash}"
return True, ""
```
The verifier hashes `result.output` and compares to `result.verification_hash` — both fields of the **same in-memory object**. Combined with `_make_result` at line 207 (`verification_hash = self.compute_hash(output)`), output and hash are sealed in the same scope. No external commitment, no key, no chain link.

**`resistancemap/verification/zero_trust.py:550-557`** (`create_verification_checkpoint`):
```python
log_entry = {
    "agent": agent_name,
    "data_hash": current_hash,
    "checkpoint_hash": checkpoint_hash,
    "timestamp": datetime.utcnow().isoformat(),
}
# (Could append to log file here for persistence)
return checkpoint_hash
```
`log_entry` is **constructed and discarded**. `log_verification()` (line 592-598) is the sole writer to `_verification_log`, and grep confirms it is never called by `create_verification_checkpoint`. The "audit trail" is in-memory only and evaporates on process exit.

**Confirmed:** (1) no external trust anchor, (2) audit log is a comment stub, (3) no signing, no append-only ledger, no external timestamp.

## §2 Adversarial scenario

An attacker with read/write access to a pickled `AgentResult` checkpoint (compromised CI runner, mis-permissioned NFS, insider with `paper/v8_artifacts/` write access) can swap a falsification-gate FAIL into a PASS — for instance, rewriting an F4 cosine-distance result from `0.31` (fail) to `0.014` (pass), making the v10 sprint summary appear to clear the threshold. They load the pickle, mutate `result.output["F4_cosine"] = 0.014`, then call `BaseAgent.compute_hash(result.output)` — which is a `@staticmethod` requiring no key, no salt, no nonce — and write the new digest into `result.verification_hash`. They re-pickle. On replay, `verify_output` recomputes and returns `(True, "")`. The "zero-trust" check passes.

The docstring at `zero_trust.py:26` — *"This prevents tampering even if an intermediate agent is compromised"* — is provably false. The current scheme detects bit rot, not adversarial modification.

## §3 Option A patch (apply-ready, ~1 person-day)

### New file: `resistancemap/verification/audit_log.py`
```python
"""Append-only JSONL audit log for agent verification provenance.

WARNING: This provides PROVENANCE for honest-mistake detection only.
An attacker with write access to logs/agent_audit_log.jsonl can rewrite
the entire file. For adversarial-resistant audit, use the HMAC-chain
upgrade (Option B) or external timestamping (Option C).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_PATH = Path(os.environ.get("RM_AUDIT_LOG", "logs/agent_audit_log.jsonl"))
_LOCK = threading.Lock()
_INITIALIZED = False


def _ensure_initialized() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _LOG_PATH.exists():
        fd = os.open(str(_LOG_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    else:
        try:
            os.chmod(str(_LOG_PATH), 0o600)
        except OSError:
            pass
    _INITIALIZED = True


def append_record(
    agent_name: str,
    task_id: str | None,
    output_hash: str,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one verification record. O_APPEND + fsync = crash-safe."""
    _ensure_initialized()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
        "task_id": task_id,
        "output_hash": output_hash,
        "status": status,
    }
    if extra:
        record["extra"] = extra
    line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    with _LOCK:
        fd = os.open(str(_LOG_PATH), os.O_WRONLY | os.O_APPEND)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)
```

### Modify `resistancemap/agents/base.py` `_make_result` (lines 189-215)
```python
def _make_result(
    self,
    status: AgentState,
    output: Any | None = None,
    error: str | None = None,
    metadata: dict | None = None,
) -> AgentResult:
    from resistancemap.verification.audit_log import append_record  # local import avoids cycle
    verification_hash = self.compute_hash(output) if output is not None else ""
    result = AgentResult(
        agent_name=self.name,
        status=status,
        output=output,
        verification_hash=verification_hash,
        metadata=metadata or {},
        error=error,
    )
    try:
        append_record(
            agent_name=self.name,
            task_id=(metadata or {}).get("task_id"),
            output_hash=verification_hash,
            status=status.value if hasattr(status, "value") else str(status),
        )
    except Exception as e:
        logger.warning(f"{self.name}: audit log write failed: {e}")
    return result
```

**Honest limits:** any process with write access to `logs/agent_audit_log.jsonl` can `truncate(0)` it or `sed -i` a record. Catches *accidental* divergence and provides a forensic timeline if the attacker forgets to rewrite the log. Does **not** prevent adversarial tampering.

## §4 Option B — HMAC chain (~2-3 person-days)

```python
secret = os.environ["RM_VERIFICATION_KEY"]  # 32+ bytes, never in repo
prev_hash = read_last_chain_hash() or b"\x00" * 32

mac_input = prev_hash + output_hash.encode() + ts.encode() + agent_id.encode()
chain_hash = hmac.new(secret, mac_input, hashlib.sha256).digest()
record = {"prev": prev_hash.hex(), "out": output_hash, "ts": ts,
          "agent": agent_id, "mac": chain_hash.hex()}
append_jsonl(record)
prev_hash = chain_hash
```

**Key-management caveats:**
- `RM_VERIFICATION_KEY` must live in a secret store (Vault, AWS KMS, GitHub Actions secrets), **never** in `.env`, **never** in same container as the writer
- Rotate at sprint boundaries; record key-id alongside MAC
- Consider Ed25519 (asymmetric) instead of HMAC if defense against insider matters

## §5 Option C — External trust anchor (sprint plan, 1-2 weeks)

Choose anchor — **Rekor + cosign** recommended (already used by HuggingFace model cards; aligns with v10 reproducibility story). Day 1-2 anchor selection; Day 3-5 batched Merkle root submission to Rekor on run completion; Day 6-8 standalone `verify_run.py` CLI; Day 9-10 GitHub Actions OIDC integration. Provides true non-repudiation — even repo owner cannot rewrite history.

## §6 Migration plan

| Phase | Window | Deliverable |
|---|---|---|
| 0 | Now | Land Option A. Strip "blockchain" / "immutable" / "even if compromised" language from `zero_trust.py:26-28` docstring. Add `SECURITY.md` declaring threat model: honest-mistake detection only. |
| 1 | +1 week | Add `task_id` to all `_make_result` callers. Backfill audit-log analysis script. Deploy to one CI run. |
| 2 | +2-3 weeks | Land Option B behind `RM_VERIFICATION_MODE=hmac` flag. Default remains Option A. Provision key in CI secret store. Document key rotation. |
| 3 | +4 weeks | Flip default to `hmac`. Ship `verify_chain.py` CLI. Update paper claims to "tamper-evident with key custody assumption." |
| 4 | v13 sprint | Option C: Rekor anchoring. Update paper to "non-repudiable" only after this lands. |

## §Summary

| Option | Effort | Adversarial resistance | Recommended |
|---|---|---|---|
| A — JSONL append-only | 1 person-day | Detects accidental mutation; does NOT stop attacker with disk write | Land now (1 PR) |
| B — HMAC chain + key | 2-3 person-days | Stops attacker without `RM_VERIFICATION_KEY`; key compromise = total break | **Default target** |
| C — Rekor / OpenTimestamps | 1-2 weeks | Non-repudiable even against repo owner; required for "immutable" claim | v13 sprint |

**Files cited:** `agents/base.py:137-144` (verify_output), `agents/base.py:189-215` (_make_result), `verification/zero_trust.py:26-28` (over-promising docstring), `verification/zero_trust.py:550-557` (stub log), `verification/zero_trust.py:592-598` (unused log_verification). New file proposed: `verification/audit_log.py`. No source modified per hard rules.
