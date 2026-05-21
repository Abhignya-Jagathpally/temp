# Phase 10 — Baseline Comparison Suite (MORT-FM v18)

Branch: `v18-clean-canonical-mortfm`
Working dir: `/home/aj0486@students.ad.unt.edu/pipeline3`
Author scope: justify MORT-FM's complexity by code-level comparisons against
simple-to-strong baselines on the SAME splits, with the SAME metrics, on the
SAME real cohorts (MMRF CoMMpass, DepMap/PRISM, CCLE, GDSC, Beat AML).

Hard constraints honored throughout:
- No synthetic / random / hard-coded metric values appear in this plan.
- All comparison numbers are computed at run-time by the new harness.
- Every cited paper has a verified PubMed PMID (or is explicitly flagged
  unverified — see Section 7).

------------------------------------------------------------------------

## 1. Inventory — What is already implemented

Reading
`ResistanceMap/resistancemap/baselines/__init__.py:1-6` shows four public
entry points: `PERCEPTIONBaseline`, `CPABaseline`, `SimpleBaselinesSuite`,
and the MOFA+/CellRank/scVI trio plus `IntegrationBenchmark`.

| Status | Baseline | File | Class / lines |
|---|---|---|---|
| Implemented | `BaseBaseline` abstract | `resistancemap/baselines/classical_baselines.py` | `class BaseBaseline` L94 |
| Implemented | Ridge | `classical_baselines.py` | `class RidgeBaseline` L134 |
| Implemented | RandomForest (clf+reg) | `classical_baselines.py` | `class RandomForestBaseline` L165 |
| Implemented | ElasticNet | `classical_baselines.py` | `class ElasticNetBaseline` L198 |
| Implemented | Logistic L1 | `classical_baselines.py` | `class LogisticL1Baseline` L233 |
| Implemented | Logistic L2 | `classical_baselines.py` | `class LogisticL2Baseline` L264 |
| Implemented | XGBoost (optional dep) | `classical_baselines.py` | `class XGBoostBaseline` L290 |
| Implemented | SVM (RBF) | `classical_baselines.py` | `class SVMBaseline` L346 |
| Implemented | KNN | `classical_baselines.py` | `class KNNBaseline` L374 |
| Implemented | Global-mean trivial | `classical_baselines.py` | `class TrivialMeanBaseline` L400 |
| Implemented | Per-drug-mean trivial | `classical_baselines.py` | `class PerDrugMeanBaseline` L420 |
| Implemented | Suite runner | `classical_baselines.py` | `class SimpleBaselinesSuite` L462 |
| Implemented | MOFA+ wrapper | `resistancemap/baselines/integration_baselines.py` | `class MOFAPlusBaseline` L178 |
| Implemented | CellRank wrapper | `integration_baselines.py` | `class CellRankBaseline` L327 |
| Implemented | scVI wrapper | `integration_baselines.py` | `class ScVIBaseline` L484 |
| Implemented | Integration metric panel | `integration_baselines.py` | `class IntegrationBenchmark` L669 |
| Implemented | PERCEPTION (Nat Cancer 2024) | `resistancemap/baselines/perception_baseline.py` | `class PERCEPTIONBaseline` L208 |
| Implemented | CPA (chemCPA flavor) | `resistancemap/baselines/cpa_baseline.py` | `class CPABaseline` L184 |
| Empty | MORT-FM baseline package | `resistancemap/baselines/mortfm/` | (directory exists, no .py files) |

### Missing — to be added in this phase

| Missing | Reason needed |
|---|---|
| Central `BaselineRegistry` | Today each baseline is invoked ad-hoc; no shared split / metric contract |
| Sequence baselines (LSTM/GRU on visit timeline) | MMRF CoMMpass is longitudinal; we have no recurrent baseline |
| Tabular transformer (FT-Transformer) | Strong tabular SOTA missing; needed before claiming MORT-FM beats "all tabular DL" |
| Pure-MLP deep-static baseline | We compare to ElasticNet but not to a depth-matched MLP |
| CNN-over-feature-order | User explicitly asked for CNN; not present |
| Survival panel (Cox / RSF / DeepSurv / discrete-hazard MLP) | We do PFS C-index in MORT-FM but only against MOFA-style baseline — no true survival panel |
| GraphSAGE-only and graph-ablated MLP | We claim PPI helps; no isolated-graph baseline to attribute the gain |
| Unified results writer | Each baseline writes its own dict; no `parquet` comparison table |
| `scripts/mortfm/10_run_baselines.py` CLI | Today there is no canonical end-to-end baseline runner |

------------------------------------------------------------------------

## 2. `resistancemap/baselines/registry.py` — central registry

```python
# resistancemap/baselines/registry.py
from __future__ import annotations
from typing import Protocol, Callable, Mapping, Any, runtime_checkable
import numpy as np

@runtime_checkable
class BaselineModel(Protocol):
    """Every registered baseline must satisfy this contract."""
    name: str
    requires: tuple[str, ...]            # required modality keys, e.g. ("rna",) or ("rna","cna","clinical")
    supports_survival: bool
    supports_longitudinal: bool

    def fit(self, X_train: Mapping[str, np.ndarray],
            y_train: np.ndarray, *, seed: int) -> None: ...
    def predict(self, X: Mapping[str, np.ndarray]) -> np.ndarray: ...
    def predict_survival(self, X: Mapping[str, np.ndarray]) -> np.ndarray | None: ...
    def save(self, path: str) -> None: ...
    def metadata(self) -> dict[str, Any]: ...   # n_params, library, version, hyperparams

Factory = Callable[[Mapping[str, Any]], BaselineModel]

class BaselineRegistry:
    _factories: dict[str, Factory] = {}

    @classmethod
    def register(cls, name: str, factory: Factory) -> None:
        if name in cls._factories:
            raise KeyError(f"baseline {name!r} already registered")
        cls._factories[name] = factory

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(cls._factories)

    @classmethod
    def instantiate(cls, name: str, config: Mapping[str, Any]) -> BaselineModel:
        if name not in cls._factories:
            raise KeyError(f"unknown baseline {name!r}; have {cls.list_available()}")
        model = cls._factories[name](config)
        if not isinstance(model, BaselineModel):
            raise TypeError(f"{name} factory did not return a BaselineModel")
        return model
```

Registration happens at module import (each `static_ml.py`, `deep_static.py`,
etc. ends with `BaselineRegistry.register("name", factory)`).

------------------------------------------------------------------------

## 3. New / extended baseline files (signatures only — no fake metrics)

### 3.1 `resistancemap/baselines/static_ml.py`

```python
class ElasticNetBaselineV2(BaselineModel):
    name = "elasticnet"
    requires = ("rna",)
    def __init__(self, l1_ratio: float = 0.5, alphas: tuple[float, ...] = (0.01, 0.1, 1.0)): ...

class RidgeBaselineV2(BaselineModel):
    name = "ridge"
    requires = ("rna",)

class RandomForestBaselineV2(BaselineModel):
    name = "random_forest"
    requires = ("rna", "clinical")
    def __init__(self, n_estimators: int = 500, max_depth: int | None = None): ...

class XGBoostBaselineV2(BaselineModel):  # optional dep
    name = "xgboost"
    requires = ("rna", "clinical")

class LightGBMBaselineV2(BaselineModel):  # optional dep
    name = "lightgbm"
    requires = ("rna", "clinical")
```

### 3.2 `resistancemap/baselines/deep_static.py`

```python
class MLPBaseline(BaselineModel):
    name = "mlp"
    requires = ("rna",)
    def __init__(self, hidden: tuple[int, ...] = (512, 256), dropout: float = 0.2, epochs: int = 100): ...

class CNNFeatureOrderBaseline(BaselineModel):
    """1-D CNN over the gene-axis ordered by chromosomal locus (real GTF order, not random)."""
    name = "cnn1d"
    requires = ("rna",)

class FTTransformerBaseline(BaselineModel):
    """FT-Transformer (Gorishniy et al., NeurIPS 2021). arXiv:2106.11959.
    Not indexed in PubMed — see Section 7."""
    name = "ft_transformer"
    requires = ("rna", "clinical")
```

### 3.3 `resistancemap/baselines/sequence.py`

```python
class LSTMVisitBaseline(BaselineModel):
    """LSTM over per-patient visit sequence (MMRF baseline + IA visits)."""
    name = "lstm_visits"
    requires = ("rna", "clinical")
    supports_longitudinal = True

class GRUVisitBaseline(BaselineModel):
    name = "gru_visits"
    requires = ("rna", "clinical")
    supports_longitudinal = True
```

### 3.4 `resistancemap/baselines/survival.py`

```python
class CoxPHBaseline(BaselineModel):       # lifelines
    name = "cox_clinical"
    requires = ("clinical",)
    supports_survival = True

class RandomSurvivalForestBaseline(BaselineModel):  # scikit-survival
    name = "rsf"
    requires = ("rna", "clinical")
    supports_survival = True
    # Algorithm: Ishwaran et al. 2008 (PMID verified via PubMed: 20582150).

class DeepSurvBaseline(BaselineModel):    # PyCox
    name = "deepsurv"
    requires = ("rna", "clinical")
    supports_survival = True
    # Katzman et al., BMC Med Res Methodol 2018, PMID 29482517.

class DiscreteHazardMLPBaseline(BaselineModel):
    name = "discrete_hazard_mlp"
    requires = ("rna", "clinical")
    supports_survival = True
```

### 3.5 `resistancemap/baselines/graph.py`

```python
class GraphSAGEPPIBaseline(BaselineModel):
    """GraphSAGE on STRING PPI, mean-pool node embeddings, linear head."""
    name = "graphsage_ppi"
    requires = ("rna", "ppi")

class NoGraphMLPBaseline(BaselineModel):
    """Same encoder depth as MORT-FM, graph branch zeroed — for ablation parity."""
    name = "mlp_no_graph"
    requires = ("rna",)
```

------------------------------------------------------------------------

## 4. `resistancemap/evaluation/model_comparison.py`

Single-writer pattern. Every baseline AND MORT-FM appends rows to one parquet.

```python
# resistancemap/evaluation/model_comparison.py
import hashlib, json, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class ComparisonRow:
    model_name: str
    model_version: str
    split: str               # "train" | "val" | "test"
    cohort: str              # "mmrf_ia22" | "depmap_24q2" | "ccle" | "gdsc" | "beataml_1.0"
    task: str                # "drug_mse" | "pfs_cindex" | "os_cindex" | "trajectory_emd" | "binary_response"
    metric_name: str         # "mse" | "rmse" | "auroc" | "auprc" | "c_index" | "ibs" | "emd"
    metric_value: float
    ci_low: float
    ci_high: float
    n: int
    seed: int
    hash_data: str           # sha256 of (sorted) train/val/test patient IDs
    hash_code: str           # git SHA at run time
    extra: str               # JSON blob for hyperparams

def append_rows(path: Path, rows: list[ComparisonRow]) -> None:
    table = pa.Table.from_pylist([asdict(r) for r in rows])
    if path.exists():
        existing = pq.read_table(path)
        table = pa.concat_tables([existing, table])
    pq.write_table(table, path)

def hash_split(patient_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(patient_ids)).encode()).hexdigest()[:16]
```

Output: `ResistanceMap/results/baseline_comparison.parquet` (single source of
truth read by figure builders).

------------------------------------------------------------------------

## 5. `scripts/mortfm/10_run_baselines.py` — CLI skeleton

```python
#!/usr/bin/env python
"""Phase-10 baseline suite runner.

Runs every registered baseline on the SAME splits as 06_train_lens_resistance.py
and 07_train_survival.py, then appends rows to results/baseline_comparison.parquet.

USAGE
-----
    python scripts/mortfm/10_run_baselines.py \
        --cohort mmrf_ia22 \
        --split-manifest data/mmrf_ia22/splits/patient_holdout_v18.json \
        --baselines elasticnet ridge random_forest xgboost mlp cnn1d ft_transformer \
                    lstm_visits gru_visits cox_clinical rsf deepsurv \
                    graphsage_ppi mlp_no_graph \
        --tasks drug_mse pfs_cindex os_cindex binary_response \
        --seeds 0 1 2 3 4 \
        --out results/baseline_comparison.parquet
"""
import argparse, json, logging
from pathlib import Path
from resistancemap.baselines.registry import BaselineRegistry
import resistancemap.baselines.static_ml      # registers
import resistancemap.baselines.deep_static    # registers
import resistancemap.baselines.sequence       # registers
import resistancemap.baselines.survival       # registers
import resistancemap.baselines.graph          # registers
from resistancemap.evaluation.model_comparison import (
    ComparisonRow, append_rows, hash_split,
)
from resistancemap.evaluation.metrics import compute_metric_with_bootstrap_ci

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--baselines", nargs="+", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--out", type=Path, default=Path("results/baseline_comparison.parquet"))
    p.add_argument("--include-mortfm", action="store_true",
                   help="Also score the trained MORT-FM checkpoint on identical splits.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    splits = json.loads(args.split_manifest.read_text())
    data_hash = hash_split(splits["train"] + splits["val"] + splits["test"])
    # Loaders return identical Dict[str, np.ndarray] per modality, gated by baseline.requires
    # ... (load X_*, y_* per task here)
    rows: list[ComparisonRow] = []
    for name in args.baselines:
        for task in args.tasks:
            for seed in args.seeds:
                model = BaselineRegistry.instantiate(name, {"seed": seed, "task": task})
                model.fit(X_train_subset(model.requires), y_train[task], seed=seed)
                # compute metric + bootstrap CI on test split, append row.
                # NO FAKE NUMBERS: only real-evaluated metrics are written.
                ...
    append_rows(args.out, rows)

if __name__ == "__main__":
    main()
```

------------------------------------------------------------------------

## 6. Ablation list (MORT-FM minus X) — must run alongside baselines

The same CLI runs ablations when `--baselines mortfm_full mortfm_no_graph
mortfm_no_epi ...` is passed. The ablation factories live in
`resistancemap/baselines/mortfm/` (currently empty — to be populated):

| Ablation key | What is zeroed |
|---|---|
| `mortfm_full` | Reference run, all blocks active |
| `mortfm_no_graph` | GAT/GraphSAGE branch detached |
| `mortfm_no_epi` | Epigenomics encoder zeroed |
| `mortfm_no_phospho` | Phosphoproteomics encoder zeroed |
| `mortfm_no_waddington` | Potential U_θ replaced by identity |
| `mortfm_ode_not_sde` | SDE integrator swapped for ODE (no diffusion) |
| `cox_clinical` | Clinical-only Cox lower-bound |
| `elasticnet` | Static ML lower bound |
| `random_forest` | Tree lower bound |
| `xgboost` | Strong tabular |
| `mlp` | Depth-matched non-ablated tabular DL |
| `lstm_visits` / `gru_visits` | Longitudinal-aware DL on visits |

------------------------------------------------------------------------

## 7. SOTA comparator section — PubMed-verified citations

According to PubMed, the following SOTA papers are verified:

| Paper | PMID | DOI link | Why it matters as a comparator |
|---|---|---|---|
| TRACERx Lung ctDNA (Abbosh, Frankell et al., Nature 2023) | 37055640 | [10.1038/s41586-023-05776-4](https://doi.org/10.1038/s41586-023-05776-4) | Phylogenetic relapse prediction from longitudinal liquid biopsy — sets the bar for "longitudinal multi-modal cancer forecasting." |
| Pan-cancer Proteogenomics tumor immunity (Petralia et al., Cell 2024) | 38359819 | [10.1016/j.cell.2024.01.027](https://doi.org/10.1016/j.cell.2024.01.027) | CPTAC-scale integrative subtype discovery — direct comparator for our multi-omics fusion claim. |
| TrajectoryNet (Tong, Huang, Wolf, van Dijk, Krishnaswamy, ICML / PMLR 2020) | 34337419 | (no journal DOI; PMC8320749) | Dynamic OT + CNF — the recognized SOTA for cellular dynamics; the comparator for our SDE-on-latent block. |
| PathCNN (Oh et al., Bioinformatics 2021) | 34252964 | [10.1093/bioinformatics/btab285](https://doi.org/10.1093/bioinformatics/btab285) | Pathway-CNN on multi-omics with survival readout — the comparator for "interpretable multi-omics survival." |
| DeepSurv (Katzman et al., BMC Med Res Methodol 2018) | 29482517 | [10.1186/s12874-018-0482-1](https://doi.org/10.1186/s12874-018-0482-1) | The reference Cox-DL baseline for personalized hazard. |
| Random Survival Forest method note (Ishwaran/Kogalur) | 20582150 | (see PubMed) | Provides the RSF baseline algorithm. |

Unverified-via-PubMed (must NOT be cited as PMIDs in the paper):

- **mmSYGNAL** — Searches for `mmSYGNAL`, `SYGNAL myeloma`, `Plaisier SYGNAL myeloma`, and `Reiss systems genetics multiple myeloma transcription factor microRNA` all returned 0 PubMed hits. The mmSYGNAL web tool (mmsygnal.systemsbiology.net) exists, but a peer-reviewed indexed publication was not retrieved by this PubMed search session. Action: cite the web resource URL + access date only, do not invent a PMID.
- **ML-based melanoma immunotherapy signature 2024** — search returns 27 hits but none is a single canonical paper; the user must disambiguate which paper they meant before we cite it.
- **iMLGAM (2025)** — 0 PubMed hits as of 2026-05-20. Likely arXiv-only or wrong name; mark as unverified.
- **FT-Transformer (Gorishniy et al., NeurIPS 2021)** — 0 PubMed hits (NeurIPS not indexed). Cite arXiv:2106.11959 only.

------------------------------------------------------------------------

## 8. Tests — `tests/mortfm/test_baseline_registry.py`

Concrete assertions (all must pass before this phase is "done"):

```python
def test_registry_round_trip():
    from resistancemap.baselines.registry import BaselineRegistry
    import resistancemap.baselines.static_ml  # noqa: F401  (registers)
    assert "elasticnet" in BaselineRegistry.list_available()
    m = BaselineRegistry.instantiate("elasticnet", {"seed": 0})
    assert hasattr(m, "fit") and hasattr(m, "predict")

def test_all_baselines_share_split_hash():
    """Every baseline run for a given (cohort, seed) must produce identical hash_data."""
    rows = pq.read_table(RESULTS).to_pylist()
    by_seed = group_by(rows, key=lambda r: (r["cohort"], r["seed"]))
    for key, grp in by_seed.items():
        hashes = {r["hash_data"] for r in grp}
        assert len(hashes) == 1, f"split drift in {key}: {hashes}"

def test_all_baselines_share_metric_schema():
    rows = pq.read_table(RESULTS).to_pylist()
    fields = {"model_name","split","cohort","task","metric_name",
              "metric_value","ci_low","ci_high","n","seed","hash_data","hash_code"}
    for r in rows:
        assert fields.issubset(r.keys())

def test_seed_determinism():
    """Re-running the same baseline with the same seed reproduces metric_value within 1e-9."""
    ...

def test_no_synthetic_fallback_in_results():
    """Reject rows whose extra-JSON contains 'synthetic' or 'random_fallback'."""
    rows = pq.read_table(RESULTS).to_pylist()
    for r in rows:
        assert "synthetic" not in r["extra"].lower()
        assert "random_fallback" not in r["extra"].lower()
```

------------------------------------------------------------------------

## 9. HONEST-REPORTING addendum

We commit to publishing whichever baseline wins on each (cohort, task) pair,
and to keep the MORT-FM column even when it loses. Two prior project
observations make this non-negotiable:

> "RM ties baselines on BOTH drug-MSE (predict-mean wins) AND PFS C-index
> (v11.5 ties mmSYGNAL; v12 BELOW v11.5)."
> — internal memo `project_resistancemap_floor_evidence.md`.

> "v12 doesn't beat v11.5; niche A wins"
> — `project_v12_refactor_audit_outcome.md`.

Implication: the contribution we will pitch is the **falsification framework**
(F1–F10 strict gates, identifiability-aware refutation, cross-disease F3
refutation in AML) — NOT a headline metric beat. The baseline suite is the
evidence base for that honest framing, not a backstop to inflate MORT-FM.

------------------------------------------------------------------------

## 10. File-change table + Definition of Done

| File | Action |
|---|---|
| `resistancemap/baselines/registry.py` | NEW |
| `resistancemap/baselines/static_ml.py` | NEW (wraps existing classical_baselines via registry) |
| `resistancemap/baselines/deep_static.py` | NEW (MLP, CNN1D, FT-Transformer) |
| `resistancemap/baselines/sequence.py` | NEW (LSTM, GRU on visits) |
| `resistancemap/baselines/survival.py` | NEW (Cox, RSF, DeepSurv, discrete-hazard MLP) |
| `resistancemap/baselines/graph.py` | NEW (GraphSAGE-only, no-graph MLP ablation) |
| `resistancemap/baselines/mortfm/__init__.py` | NEW (registers MORT-FM ablations as baselines) |
| `resistancemap/baselines/mortfm/ablations.py` | NEW |
| `resistancemap/baselines/__init__.py` | EXTEND (re-export registry + new modules) |
| `resistancemap/evaluation/model_comparison.py` | NEW (ComparisonRow + append_rows + hash_split) |
| `scripts/mortfm/10_run_baselines.py` | NEW (CLI) |
| `tests/mortfm/test_baseline_registry.py` | NEW |
| `results/baseline_comparison.parquet` | NEW (produced by CLI, not committed) |

### Definition of Done

1. `pytest tests/mortfm/test_baseline_registry.py -q` is green.
2. `python scripts/mortfm/10_run_baselines.py --cohort mmrf_ia22 ...`
   runs end-to-end and writes a non-empty parquet whose `hash_data` matches
   the canonical MORT-FM training split.
3. The parquet contains rows for ALL of: at least one trivial baseline, at
   least one static-ML baseline, at least one deep-static baseline, at least
   one sequence baseline, at least one survival baseline, at least one
   graph baseline, and MORT-FM full + at least 3 ablations.
4. No PMID appears in the paper draft that was not retrieved in Section 7.
5. No row in the parquet has `extra` containing `"synthetic"` or
   `"random_fallback"` (Section 8 test).
