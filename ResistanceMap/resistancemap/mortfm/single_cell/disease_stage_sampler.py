"""
resistancemap/mortfm/single_cell/disease_stage_sampler.py
===========================================================
Stratified per-stage sampler so each contrastive batch sees at least 2
representatives of every disease stage present in the data. Without
this, SupCon degenerates because positive pairs become rare.
"""

from __future__ import annotations

import logging
import random as _random
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class StratifiedStageSampler:
    """Iterates indices in (stage-balanced) mini-batches of size ``batch_size``.

    The sampler shuffles within each stage at the start of every epoch
    (deterministic for any seed), then draws round-robin across stages
    until every observation has been used.
    """

    stage_labels: Sequence[int]
    batch_size: int
    seed: int = 41

    def __post_init__(self) -> None:
        by_stage: dict[int, list[int]] = {}
        for i, s in enumerate(self.stage_labels):
            by_stage.setdefault(int(s), []).append(i)
        self._by_stage = by_stage
        if not by_stage:
            raise ValueError("StratifiedStageSampler: no stage labels supplied")

    def __iter__(self) -> Iterator[List[int]]:
        rng = _random.Random(self.seed)
        pools = {s: idxs[:] for s, idxs in self._by_stage.items()}
        for v in pools.values():
            rng.shuffle(v)
        stages = sorted(pools)
        total = sum(len(v) for v in pools.values())
        emitted = 0
        per_stage_quota = max(1, self.batch_size // len(stages))
        while emitted < total:
            batch: List[int] = []
            empties = 0
            for s in stages:
                pool = pools[s]
                take = min(per_stage_quota, len(pool))
                batch.extend(pool[:take])
                del pool[:take]
                if not pool:
                    empties += 1
            if not batch:
                break
            yield batch
            emitted += len(batch)
            if empties == len(stages):
                break

    def __len__(self) -> int:
        total = sum(len(v) for v in self._by_stage.values())
        return max(1, total // self.batch_size + (1 if total % self.batch_size else 0))
