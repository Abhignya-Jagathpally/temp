"""
resistancemap/mortfm/registries/artifact_registry.py
====================================================
Catalogue of every MORT-FM on-disk artifact and what it unlocks.

The registry is the *source of truth* the integrated checkpoint validator
reads. mortfm_integrated.pt can claim "Block A + B + C + D present" only if
this registry says all four are present with non-empty allowed_claims.

Honest behaviour
----------------
* :class:`ArtifactRecord` carries explicit allowed_claims and blocked_claims.
  An empty allowed_claims is legal — it means "this artifact exists but
  hasn't earned any claim yet".
* :meth:`ArtifactRegistry.save` writes a deterministic JSON; round-trip with
  :meth:`load` is lossless.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ArtifactRecord:
    block_id: str
    artifact_type: str  # checkpoint | embedding_cache | graph | manifest | report
    path: str
    data_source: List[str] = field(default_factory=list)
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    feature_space_id: Optional[str] = None
    endpoint_name: Optional[str] = None
    allowed_claims: List[str] = field(default_factory=list)
    blocked_claims: List[str] = field(default_factory=list)
    checkpoint_compatible: bool = True
    created_at: str = ""
    git_commit: str = ""
    gate_report_path: Optional[str] = None
    notes: str = ""

    @property
    def status(self) -> str:
        if not self.path:
            return "missing"
        return "present" if Path(self.path).exists() else "missing"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status
        return d


class ArtifactRegistry:
    """Container that owns a dict[block_id -> ArtifactRecord]."""

    def __init__(self, records: Optional[Iterable[ArtifactRecord]] = None) -> None:
        self._records: Dict[str, ArtifactRecord] = {}
        for r in records or []:
            self.add(r)

    def add(self, record: ArtifactRecord) -> None:
        self._records[record.block_id] = record

    def get(self, block_id: str) -> Optional[ArtifactRecord]:
        return self._records.get(block_id)

    def __contains__(self, block_id: str) -> bool:
        return block_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    def items(self):
        return self._records.items()

    def to_dict(self) -> Dict[str, Any]:
        return {bid: rec.to_dict() for bid, rec in sorted(self._records.items())}

    def save(self, out_path: str = "logs/mortfm/artifact_registry.json") -> str:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Wrote artifact registry (%d records) -> %s", len(self), out)
        return str(out)

    @classmethod
    def load(cls, path: str) -> "ArtifactRegistry":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Artifact registry not found at {p.resolve()}")
        with open(p) as f:
            raw = json.load(f)
        recs = []
        for bid, payload in raw.items():
            payload = dict(payload)
            payload.pop("status", None)
            recs.append(ArtifactRecord(**payload))
        return cls(recs)

    def all_present(self, required_block_ids: Iterable[str]) -> bool:
        return all(
            (bid in self._records) and (self._records[bid].status == "present")
            for bid in required_block_ids
        )

    def claims_unlocked(self) -> List[str]:
        """Union of allowed_claims across all *present* artifacts.

        This is the upper bound the run-time auditor can grant; the actual
        granted set is the intersection with what the run's gate report passes.
        """
        out: set[str] = set()
        for rec in self._records.values():
            if rec.status == "present":
                out.update(rec.allowed_claims)
        return sorted(out)
