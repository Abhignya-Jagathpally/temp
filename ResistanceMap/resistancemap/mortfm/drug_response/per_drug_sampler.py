"""
resistancemap/mortfm/drug_response/per_drug_sampler.py
=========================================================
Per-drug pair sampler. Emits (specimen_idx, drug_idx, family_idx,
response_value) tuples balanced over the drug-family axis so
under-represented families (BCL2_inhibitor, XPO1_inhibitor) get
enough representation in each batch.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence


@dataclass
class PerDrugPairSampler:
    """Yields batches of indices into a flat (specimen, drug) pair list.

    Stratifies on drug_family so each batch sees every family at least once
    when the family has >=2 representatives.
    """

    drug_families: Sequence[int]   # length = total pairs
    batch_size: int
    seed: int = 53

    def __post_init__(self) -> None:
        by_fam: dict[int, list[int]] = {}
        for i, f in enumerate(self.drug_families):
            by_fam.setdefault(int(f), []).append(i)
        self._by_fam = by_fam

    def __iter__(self) -> Iterator[List[int]]:
        rng = _random.Random(self.seed)
        pools = {f: idxs[:] for f, idxs in self._by_fam.items()}
        for v in pools.values():
            rng.shuffle(v)
        total = sum(len(v) for v in pools.values())
        per_fam = max(1, self.batch_size // max(len(pools), 1))
        emitted = 0
        while emitted < total:
            batch: List[int] = []
            empties = 0
            for f, pool in pools.items():
                take = min(per_fam, len(pool))
                batch.extend(pool[:take])
                del pool[:take]
                if not pool:
                    empties += 1
            if not batch:
                break
            yield batch
            emitted += len(batch)
            if empties == len(pools):
                break

    def __len__(self) -> int:
        total = sum(len(v) for v in self._by_fam.values())
        return max(1, total // self.batch_size + (1 if total % self.batch_size else 0))
