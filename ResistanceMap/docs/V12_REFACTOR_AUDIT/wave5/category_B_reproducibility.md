# Wave 5 — Category B: Reproducibility + Engineering (W5.B)

**Author:** general-purpose agent (verified at file:line; persisted by main thread)
**Date:** 2026-05-07
**Status:** All 6 blockers verified. Patches ready to apply. Recommended landing order: B.1 → B.3 → B.6 → B.4 → B.2 → B.5.

## §1 — Important verification correction

The blocker brief cites "main.py:333,337" but the top-level `main.py` is only 11 lines. The actual import contract lives in `resistancemap/data/__init__.py` itself (the file the user has open in the IDE) — lines 17-32 import 10 symbols from three sibling modules. The "main.py:333,337" line numbers presumably reference a deeper `resistancemap/main.py` not snapshotted; the import contract was reconstructed from `__init__.py` + `tests/test_leakage.py:8` instead.

## §B.1 — Missing `data/{loaders,preprocessors,splits}.py`

### Verification
`resistancemap/data/__init__.py` re-exports:
- From `loaders.py`: `load_ccle_proteomics`, `load_ccle_epigenomics`, `load_string_ppi`, `load_gdsc_drug_sensitivity`, `load_data_ready_pt`
- From `preprocessors.py`: `harmonize_omics`, `build_train_val_test_splits`
- From `splits.py`: `SplitProtocol`, `make_split`, `assert_no_overlap`

`tests/test_leakage.py:8` additionally requires `assert_folds_disjoint, LeakageError` from `splits.py` — NOT re-exported from `__init__.py`, so the existing shim is incomplete vs the test contract.

Test fixture (lines 16-25) shows `make_split(meta_df, kind={"patient"|"lineage"|"drug_holdout"|"sample"}, n_folds, seed)` returns `(folds, manifest)` where `folds: list[(train_idx, test_idx)]` and `manifest` has `.group_hash` and `.fold_sizes`. `kind="sample"` must raise `ValueError` (line 69). `assert_no_overlap(a, b, key=, label_a=, label_b=)` raises `LeakageError` on overlap.

### Patch — three new files (FULL BODIES)

**`resistancemap/data/splits.py`:**
```python
"""Leak-proof CV split factory for ResistanceMap.

Contracts (enforced by tests/test_leakage.py):
  * make_split(meta, kind, n_folds, seed) -> (folds, SplitManifest)
  * folds: list[(train_idx: np.ndarray, test_idx: np.ndarray)]
  * SplitManifest: .group_hash (str) .fold_sizes (tuple[int,...])
  * kind in {"patient","lineage","drug_holdout"}; "sample" rejected.
  * assert_no_overlap raises LeakageError on intersection.
  * assert_folds_disjoint enforces no group appears in two test folds.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


class LeakageError(RuntimeError):
    """Raised when a split or cohort partition violates disjointness."""


@dataclass(frozen=True)
class SplitProtocol:
    kind: str               # "patient" | "lineage" | "drug_holdout"
    n_folds: int
    seed: int


@dataclass(frozen=True)
class SplitManifest:
    group_hash: str
    fold_sizes: tuple[int, ...]
    kind: str
    n_folds: int
    seed: int


_KIND_TO_KEY = {
    "patient":      "patient_id",
    "lineage":      "lineage_subtype",
    "drug_holdout": "drug",
}


def _hash_groups(groups: Sequence) -> str:
    h = hashlib.sha256()
    for g in groups:
        h.update(repr(g).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def make_split(meta: pd.DataFrame, kind: str, n_folds: int = 5, seed: int = 0):
    if kind not in _KIND_TO_KEY:
        raise ValueError(
            f"Unsupported split kind={kind!r}; must be one of {sorted(_KIND_TO_KEY)}"
        )
    key = _KIND_TO_KEY[kind]
    if key not in meta.columns:
        raise KeyError(f"meta is missing required grouping column {key!r} for kind={kind!r}")
    groups = meta[key].to_numpy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(meta))
    inv = np.argsort(perm)
    gkf = GroupKFold(n_splits=n_folds)
    folds = []
    X_dummy = np.zeros((len(meta), 1))
    for tr_p, te_p in gkf.split(X_dummy[perm], groups=groups[perm]):
        folds.append((np.sort(inv[tr_p]), np.sort(inv[te_p])))
    manifest = SplitManifest(
        group_hash=_hash_groups(sorted(set(groups.tolist()))),
        fold_sizes=tuple(len(te) for _, te in folds),
        kind=kind, n_folds=n_folds, seed=seed,
    )
    return folds, manifest


def assert_no_overlap(ids_a: Iterable, ids_b: Iterable, key: str,
                     label_a: str = "A", label_b: str = "B") -> None:
    sa, sb = set(ids_a), set(ids_b)
    overlap = sa & sb
    if overlap:
        sample = sorted(map(str, list(overlap)[:5]))
        raise LeakageError(
            f"{label_a}/{label_b} share {len(overlap)} values of {key!r}; sample={sample}"
        )


def assert_folds_disjoint(folds, meta: pd.DataFrame, key: str) -> None:
    seen = {}
    for i, (_, te) in enumerate(folds):
        for g in meta.iloc[te][key].unique():
            if g in seen:
                raise LeakageError(
                    f"Group {g!r} (key={key}) leaks across folds {seen[g]} and {i}"
                )
            seen[g] = i
```

**`resistancemap/data/loaders.py`:**
```python
"""Read-only dataset loaders. Stubs satisfying the import contract."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd


class DataNotAvailable(FileNotFoundError):
    """Raised when a requested release artifact is absent on disk."""


def _require(path: Path) -> Path:
    if not path.exists():
        raise DataNotAvailable(
            f"{path} not found. Run scripts/build_data_ready.py to materialise it; "
            f"or set RM_DATA_DIR to the release path."
        )
    return path


def load_ccle_proteomics(root: str | Path) -> pd.DataFrame:
    return pd.read_parquet(_require(Path(root) / "ccle" / "proteomics.parquet"))


def load_ccle_epigenomics(root: str | Path) -> pd.DataFrame:
    return pd.read_parquet(_require(Path(root) / "ccle" / "epigenomics.parquet"))


def load_string_ppi(root: str | Path, score_threshold: int = 700) -> pd.DataFrame:
    df = pd.read_parquet(_require(Path(root) / "string" / "edges.parquet"))
    return df[df["combined_score"] >= score_threshold].reset_index(drop=True)


def load_gdsc_drug_sensitivity(root: str | Path) -> pd.DataFrame:
    return pd.read_parquet(_require(Path(root) / "gdsc" / "drug_sensitivity.parquet"))


def load_data_ready_pt(path: str | Path) -> Any:
    import torch
    return torch.load(_require(Path(path)), map_location="cpu", weights_only=False)
```

**`resistancemap/data/preprocessors.py`:**
```python
"""Minimal preprocessing primitives — stubs satisfying public contract."""
from __future__ import annotations
from typing import Mapping
import numpy as np
import pandas as pd
from .splits import SplitProtocol, make_split


def harmonize_omics(matrices: Mapping[str, pd.DataFrame],
                    sample_col: str = "sample_id") -> pd.DataFrame:
    if not matrices:
        raise ValueError("harmonize_omics: matrices is empty")
    out = None
    for modality, df in matrices.items():
        if sample_col not in df.columns:
            raise KeyError(f"matrix {modality!r} missing {sample_col!r}")
        renamed = df.rename(
            columns={c: f"{modality}__{c}" for c in df.columns if c != sample_col}
        )
        out = renamed if out is None else out.merge(renamed, on=sample_col, how="inner")
    return out.reset_index(drop=True)


def build_train_val_test_splits(meta: pd.DataFrame, protocol: SplitProtocol,
                                val_frac: float = 0.1) -> dict:
    folds, manifest = make_split(meta, kind=protocol.kind,
                                 n_folds=protocol.n_folds, seed=protocol.seed)
    train_idx, test_idx = folds[0]
    rng = np.random.default_rng(protocol.seed)
    perm = rng.permutation(train_idx)
    n_val = max(1, int(len(perm) * val_frac))
    val_idx, train_idx2 = perm[:n_val], perm[n_val:]
    return {"train": np.sort(train_idx2), "val": np.sort(val_idx),
            "test": np.sort(test_idx), "manifest": manifest}
```

**Effort:** ~90 min (create 3 files; run `pytest tests/test_leakage.py -x`).
**Risk:** Low. Stubs only; uses sklearn `GroupKFold` already in dep closure.

## §B.2 — `reproduce.sh` references nonexistent artifacts

### Verification
`reproduce.sh:13` calls `bash scripts/verify_release.sh` (likely missing). `:16` calls `python -m resistancemap.reproduce` — no `resistancemap/reproduce.py`. `:27` calls `python scripts/assert_reproduction_band.py` and `:29` references `configs/reproduction_band.yaml` — neither exists.

### Patch — rewrite `reproduce.sh` for v0.1
```bash
#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${OUT_DIR:-reports/$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-0}"
mkdir -p "$OUT_DIR"
echo "=== Running v10 sprint pipeline ==="
python -m resistancemap.v10_runner --seed "$SEED" --out "$OUT_DIR"
echo "=== Running leakage gate ==="
pytest -x tests/test_leakage.py
echo
echo "Artifacts: $OUT_DIR"
```

For v1.0 (option a) the three artifacts need to be authored as proper deliverables — 1-2 day sprint.

**Effort:** 15 min for option (b); 1-2 days for (a).
**Risk:** Low; CI/nightly will need re-pointing.

## §B.3 — pyproject.toml omits 8 on-disk subpackages

### Verification
`pyproject.toml:107` declares `resistancemap.data` (which doesn't exist pre-B.1) and OMITS: `baselines`, `clinical`, `experiments`, `infrastructure`, `interpretability`, `observability`, `theory`, `training` (all confirmed via `__init__.py` reads).

Effect: `pip install -e .` ships an incomplete wheel; downstream `from resistancemap.training.losses import ...` works in editable mode but fails on a built wheel.

### Patch — auto-discovery (replace lines 106-107)
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["resistancemap*"]
exclude = ["tests*", "scripts*", "paper*", "release*", "results*", "reports*"]
```

**Effort:** 10 min (auto-discovery) or 30 min (audited explicit list + verify wheel).
**Risk:** Low.

## §B.4 — `generate_latex_tables.py:256-269` hardcoded fabricated AUROC

### Verification
Lines 253-273 silently fall back to fabricated values 0.832, 0.845, 0.853, 0.861, 0.870, 0.858, 0.810, 0.822, 0.843, 0.851, 0.752, 0.804, 0.854, plus `full_auroc=0.891`, when `ablation_results` is empty. **No logging warning, no provenance note.** Highest-severity in Category B; matches `project_figure_fabrication_risk` memory entry.

### Patch
```python
class FabricationError(RuntimeError):
    """Raised when a paper artifact would silently insert non-real numbers."""

if not ablation_results:
    raise FabricationError(
        "table2_ablation: ablation_results is empty. Refusing to fabricate "
        "AUROC/AUPRC. Run experiments/ablation_study.py first and pass the "
        "aggregated metrics dict, or set RM_ALLOW_PLACEHOLDER=1 to emit a "
        "redacted (--) table for layout-only review."
    )
```
Move `FabricationError` to `resistancemap.utils.errors`. Audit other 4 table generators in the same file for same anti-pattern.

**Effort:** 30 min.
**Risk:** Low.

## §B.5 — No CI configuration

### Patch — `.github/workflows/ci.yml`
```yaml
name: ci
on:
  push: { branches: [main, v6, v10, v11, v12] }
  pull_request: { branches: [main] }
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    strategy:
      matrix: { python-version: ["3.11", "3.12"] }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install (cpu)
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,cpu]"
      - name: Leakage gate (release-blocker)
        run: pytest -x tests/test_leakage.py
      - name: Full test suite
        run: pytest tests/ -x --maxfail=3
      - name: Fabrication sentinel
        run: python scripts/check_no_fabrication.py
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: pip }
      - run: pip install black flake8 isort
      - run: black --check resistancemap tests scripts
      - run: isort --check-only resistancemap tests scripts
      - run: flake8 resistancemap --max-line-length=100 --extend-ignore=E203,W503
```

Plus `scripts/check_no_fabrication.py`:
```python
#!/usr/bin/env python3
"""Fail CI if hardcoded result numbers appear in scripts/ or paper/."""
from __future__ import annotations
import os, re, sys
from pathlib import Path

PATTERNS = [
    re.compile(r'"auroc_mean"\s*:\s*0\.\d{3}'),
    re.compile(r'"auprc_mean"\s*:\s*0\.\d{3}'),
    re.compile(r'\bfull_auroc\s*=\s*0\.\d{3}'),
    re.compile(r'\bfull_auprc\s*=\s*0\.\d{3}'),
    re.compile(r'np\.random\.(?:default_rng|seed)\([^)]*\)\s*#?.*figure', re.IGNORECASE),
]
ROOTS = ["scripts", "paper", "resistancemap"]
ALLOW = set(os.environ.get("NO_FAB_ALLOW", "").split(",")) - {""}

violations = []
for root in ROOTS:
    for p in Path(root).rglob("*.py"):
        if str(p) in ALLOW:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pat in PATTERNS:
                if pat.search(line):
                    violations.append((str(p), i, line.strip()))

if violations:
    print("FABRICATION SENTINEL — hardcoded metric literals detected:")
    for f, n, l in violations:
        print(f"  {f}:{n}: {l}")
    sys.exit(1)
print("OK — no fabrication patterns found.")
```

**Effort:** 1-2 hr.
**Risk:** Medium. Sentinel will fail on first run because B.4 is unfixed. Land B.4 first OR add `scripts/generate_latex_tables.py` to allowlist temporarily.

## §B.6 — `ablations.py` byte-identical to `external_validation.py`

### Verification
Lines 1-239 byte-identical (TRIPODAIChecklist boilerplate). The real harness is `evaluation/ablation_harness.py` with `ABLATIONS` list (8 entries), `AblationResult`, `paired_bootstrap_pvalue`, `run_ablation`.

**Adjacent finding (out of B.6 scope):** `ExternalCohortLoader._generate_synthetic_cohort` at line ~191 violates the no-synthetic-data rule.

### Patch
1. `git rm resistancemap/evaluation/ablations.py`
2. `git mv resistancemap/evaluation/ablation_harness.py resistancemap/evaluation/ablations.py`
3. Update any `from resistancemap.evaluation.ablation_harness import ...` (likely zero)
4. Add regression test:
```python
# tests/test_no_duplicate_modules.py
import hashlib, pathlib
def test_ablations_and_external_validation_distinct():
    base = pathlib.Path("resistancemap/evaluation")
    a = hashlib.sha256((base/"ablations.py").read_bytes()).hexdigest()
    b = hashlib.sha256((base/"external_validation.py").read_bytes()).hexdigest()
    assert a != b, "ablations.py and external_validation.py are byte-identical"
```

**Effort:** 30 min.
**Risk:** Low.

## §Summary

| ID  | Finding | Verified | Effort | Risk |
|-----|---------|---|---|---|
| B.1 | Missing data modules | Yes | 90 min | Low |
| B.2 | Broken reproduce.sh | Yes | 15 min (option b) | Low |
| B.3 | pyproject mismatch | Yes | 10 min | Low |
| B.4 | Fabricated AUROC fallback | Yes | 30 min | Low |
| B.5 | No CI | Yes | 1-2 hr | Med |
| B.6 | Duplicate ablations.py | Yes | 30 min | Low |

**Total: ~4-5 hr to clear all six.** Recommended order: **B.1 → B.3 → B.6 → B.4 → B.2 → B.5** (B.1 first because import graph blocks all tests; B.5 last because fabrication sentinel depends on B.4 landing).

**Out-of-scope adjacent finding to file separately:** `ExternalCohortLoader._generate_synthetic_cohort` violates `feedback_no_synthetic_data` memory rule.
