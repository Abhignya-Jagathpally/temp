# Phase 12, 13, 15 — Infrastructure design (agents, parallel compute, reproducibility)

Status: design-only. **No code is committed in this phase.** Once Phase 0
(canonical trainer, separate doc) is in, these three infrastructure phases
make MORT-FM training truly agentic, truly parallel, and truly reproducible.

The existing primitives this doc builds on:

* `resistancemap/agents/orchestrator.py` — async DAG orchestrator
  (Kahn topological sort, layered `asyncio.gather`, zero-trust hash chain).
* `resistancemap/agents/base.py:33-66` — `AgentResult` (with
  `verification_hash`) and `AgentState` enum.
* `resistancemap/mortfm/schemas.py` — `MORTBatch`, `MORTFMConfig`,
  `TemporalTrainingPair`.

No phase below introduces synthetic, random, or hardcoded metric data.

---

# Phase 12 — Make agents truly agentic (MORT-FM scientific DAG)

## 12.1 Motivation

Today's `resistancemap/agents/specialized.py` agents are largely thin
wrappers around static utilities (`DataValidationAgent`, `VAEPretrainAgent`,
etc.) — they were built for the v8–v12 cell-line pipeline. MORT-FM has a
**different scientific DAG**: it needs cohort-eligibility logic, identifier
harmonisation across MMRF / CoMMpass / BeatAML / scRNA, per-modality
preprocessing with leakage control, two foundation pretraining stages
(self-sup + graph), survival + trajectory + counterfactual heads with
strict supervision contracts, and three independent claim auditors
(Calibration, ClaimCritic, ExternalValidation).

The fix is to introduce a new module
`resistancemap/agents/mortfm/` that defines 17 MORT-FM-specific agents on
top of the existing `BaseAgent` + `Orchestrator` primitives, with a
`mortfm.scientific_dag.build_default_dag(cfg)` factory that returns a fully
wired `AgentDAG` ready for `Orchestrator.run`.

## 12.2 Agent inventory (17 nodes, 5 layers after topo-sort)

| Agent | Depends on | Output (verified by hash) |
| --- | --- | --- |
| `DataInventoryAgent` | – | `{datasets: [...], paths: {...}, n_patients_per_source}` |
| `IdentifierHarmonizationAgent` | `DataInventoryAgent` | `{patient_id_map, gene_id_map, drug_id_map}` |
| `CohortEligibilityAgent` | `IdentifierHarmonizationAgent` | `{eligible_patient_ids, exclusion_reasons}` |
| `LeakageAuditAgent` | `CohortEligibilityAgent` | `{train/val/test splits, leakage_report}` |
| `ModalityPreprocessAgent[rna]` | `LeakageAuditAgent` | `{rna_tensor, gene_index}` |
| `ModalityPreprocessAgent[atac]` | `LeakageAuditAgent` | `{atac_tensor, peak_index}` |
| `ModalityPreprocessAgent[proteomics]` | `LeakageAuditAgent` | `{prot_tensor, protein_index}` |
| `ModalityPreprocessAgent[clinical]` | `LeakageAuditAgent` | `{clinical_tensor, feature_index}` |
| `FoundationPretrainingAgent` | all `ModalityPreprocessAgent[*]` | `{checkpoint_path_stages_A_B, history}` |
| `GraphEncoderPretrainingAgent` | `IdentifierHarmonizationAgent` | `{graph_emb_path, ppi_edges}` |
| `BaselineSuiteAgent` | `LeakageAuditAgent` | `{cox_baseline_cindex, rf_baseline_cindex, ...}` |
| `TrajectorySDEAgent` | `FoundationPretrainingAgent`, `GraphEncoderPretrainingAgent` | `{checkpoint_path_stage_E, sde_diagnostics}` |
| `SurvivalAgent` | `TrajectorySDEAgent` | `{checkpoint_path_stage_F, cindex, ibs}` |
| `PathwayRouteAgent` | `SurvivalAgent` | `{checkpoint_path_stage_G, pathway_attrib}` |
| `CounterfactualAgent` | `SurvivalAgent` | `{counterfactual_rankings, oracle_corr}` |
| `AblationAgent` | `SurvivalAgent` | `{per_modality_dropouts, per_loss_dropouts}` |
| `ExternalValidationAgent` | `SurvivalAgent` | `{external_cindex_per_cohort}` |
| `CalibrationAgent` | `SurvivalAgent` | `{ece, brier, temperature}` |
| `ClaimCriticAgent` | `BaselineSuiteAgent`, `SurvivalAgent`, `ExternalValidationAgent`, `CalibrationAgent`, `AblationAgent` | `{claim_gates: {f1..f10: passed/refuted/n/a}}` |
| `FigureGenerationAgent` | `ClaimCriticAgent` | `{figure_paths}` (raises if any np.random used) |
| `PaperTableAgent` | `ClaimCriticAgent` | `{table_paths}` |

Topological layers (from `AgentDAG.topological_sort`):

```
L0: DataInventoryAgent
L1: IdentifierHarmonizationAgent
L2: CohortEligibilityAgent, GraphEncoderPretrainingAgent
L3: LeakageAuditAgent
L4: ModalityPreprocessAgent[rna|atac|proteomics|clinical], BaselineSuiteAgent
L5: FoundationPretrainingAgent
L6: TrajectorySDEAgent
L7: SurvivalAgent
L8: PathwayRouteAgent, CounterfactualAgent, AblationAgent,
    ExternalValidationAgent, CalibrationAgent
L9: ClaimCriticAgent
L10: FigureGenerationAgent, PaperTableAgent
```

(Layers L0-L4 are bounded by I/O; L5-L7 are GPU-bound; L8 is parallel-safe;
L9 is the audit gate; L10 fans out for paper artefacts.)

## 12.3 Concrete base class + scheduling sketch

```python
# resistancemap/agents/mortfm/base.py

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Mapping

from resistancemap.agents.base import AgentResult, AgentState, BaseAgent
from resistancemap.mortfm.schemas import MORTFMConfig

logger = logging.getLogger(__name__)


class MORTFMAgent(BaseAgent):
    """Base class for every MORT-FM-specific scientific agent.

    Tightens the BaseAgent contract by requiring:

      * MORTFM-typed inputs/outputs (no untyped dicts at agent boundaries)
      * Explicit no-synthetic-data guard — the agent declares which input
        keys MUST come from real data sources
      * A ``produces_real_data_only`` class attribute checked by
        FigureGenerationAgent / ClaimCriticAgent before allowing the
        run to advance past L9.
    """

    produces_real_data_only: bool = True
    real_inputs_required: tuple[str, ...] = ()  # dependency-output keys

    def __init__(self, name: str, dependencies: list[str] | None = None,
                 cfg: MORTFMConfig | None = None) -> None:
        super().__init__(name=name, dependencies=dependencies)
        self.cfg = cfg

    def verify_no_synthetic(self, inputs: Mapping[str, Any]) -> None:
        """Raise if a required real-data input carries the synthetic flag."""
        for key in self.real_inputs_required:
            dep_out = inputs.get(key, {})
            if isinstance(dep_out, dict) and dep_out.get("_synthetic", False):
                raise RuntimeError(
                    f"{self.name}: dependency '{key}' is flagged synthetic; "
                    f"refusing to run a scientific agent on synthetic input."
                )

    @abstractmethod
    async def execute(self, inputs: Mapping[str, Any],
                      cfg: MORTFMConfig) -> AgentResult: ...


# resistancemap/agents/mortfm/scientific_dag.py

from resistancemap.agents.orchestrator import AgentDAG
from resistancemap.agents.mortfm.preprocess import ModalityPreprocessAgent
from resistancemap.agents.mortfm.foundation import FoundationPretrainingAgent
# ... etc

def build_default_dag(cfg: MORTFMConfig) -> AgentDAG:
    dag = AgentDAG()
    dag.add_agent(DataInventoryAgent(cfg=cfg))
    dag.add_agent(IdentifierHarmonizationAgent(cfg=cfg,
        dependencies=["DataInventoryAgent"]))
    dag.add_agent(CohortEligibilityAgent(cfg=cfg,
        dependencies=["IdentifierHarmonizationAgent"]))
    dag.add_agent(LeakageAuditAgent(cfg=cfg,
        dependencies=["CohortEligibilityAgent"]))
    for mod in ("rna", "atac", "proteomics", "clinical"):
        dag.add_agent(ModalityPreprocessAgent(modality=mod, cfg=cfg,
            dependencies=["LeakageAuditAgent"]))
    dag.add_agent(GraphEncoderPretrainingAgent(cfg=cfg,
        dependencies=["IdentifierHarmonizationAgent"]))
    dag.add_agent(FoundationPretrainingAgent(cfg=cfg, dependencies=[
        "ModalityPreprocessAgent[rna]", "ModalityPreprocessAgent[atac]",
        "ModalityPreprocessAgent[proteomics]", "ModalityPreprocessAgent[clinical]",
    ]))
    dag.add_agent(TrajectorySDEAgent(cfg=cfg, dependencies=[
        "FoundationPretrainingAgent", "GraphEncoderPretrainingAgent",
    ]))
    dag.add_agent(SurvivalAgent(cfg=cfg, dependencies=["TrajectorySDEAgent"]))
    for name in ("PathwayRouteAgent", "CounterfactualAgent", "AblationAgent",
                 "ExternalValidationAgent", "CalibrationAgent"):
        dag.add_agent(globals()[name](cfg=cfg, dependencies=["SurvivalAgent"]))
    dag.add_agent(BaselineSuiteAgent(cfg=cfg, dependencies=["LeakageAuditAgent"]))
    dag.add_agent(ClaimCriticAgent(cfg=cfg, dependencies=[
        "BaselineSuiteAgent", "SurvivalAgent", "ExternalValidationAgent",
        "CalibrationAgent", "AblationAgent",
    ]))
    dag.add_agent(FigureGenerationAgent(cfg=cfg, dependencies=["ClaimCriticAgent"]))
    dag.add_agent(PaperTableAgent(cfg=cfg, dependencies=["ClaimCriticAgent"]))
    return dag
```

The existing `Orchestrator.run(...)` (`orchestrator.py:219-306`) already
handles layered parallel execution, hash-chained verification, and
optional AgentOps tracing — no changes there. The only new piece is the
MORT-FM agent classes themselves.

## 12.4 Most-consequential change in Phase 12

`ClaimCriticAgent` is the lock — it sits at L9 and refuses to let
`FigureGenerationAgent` / `PaperTableAgent` run if any upstream claim is
flagged refuted *or unsupported*. This is the place where the v12-audit
finding ("v12 doesn't beat v11.5") would have been caught automatically.

---

# Phase 13 — Parallel compute (`infrastructure/task_backend.py`)

## 13.1 Motivation

Today's training entry points are single-process synchronous scripts. The
agent DAG in Phase 12 is async-aware, but the *inside* of each agent
(e.g. `FoundationPretrainingAgent.execute`) still runs serially. For MORT-FM
we need:

* run **per-modality preprocessing** in parallel,
* run the **baseline suite** (Cox, RF, DeepSurv, XGB) in parallel,
* run **ablations** (drop one modality × drop one loss = 4×5 = 20 jobs) in
  parallel,
* run **Optuna HPO** with N parallel trials across GPUs.

We do **not** want to lock in any one backend (some users have SLURM,
others have a single GPU, others have a Ray cluster). The fix is a
`TaskBackend` abstraction with four implementations.

## 13.2 Concrete interface

```python
# resistancemap/infrastructure/task_backend.py

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


@dataclass
class TaskHandle:
    task_id: str
    backend: str
    submitted_ts: float
    metadata: Mapping[str, Any]


@dataclass
class TaskStatus:
    task_id: str
    state: str   # 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
    progress: Optional[float] = None    # 0.0..1.0
    error: Optional[str] = None


class TaskBackend(abc.ABC):
    """Polymorphic compute backend for ResistanceMap agents.

    Each agent that wants to fan out concurrent work calls
    ``backend.submit(fn, *args, **kwargs)`` -> TaskHandle, then
    ``backend.gather(handles)`` -> list of return values. Cancellation and
    status polling are first-class so the dashboard can show progress.
    """

    name: str

    @abc.abstractmethod
    def submit(self, fn: Callable[..., Any], *args: Any,
               resources: Optional[Mapping[str, Any]] = None,
               **kwargs: Any) -> TaskHandle: ...

    @abc.abstractmethod
    def status(self, handle: TaskHandle) -> TaskStatus: ...

    @abc.abstractmethod
    def gather(self, handles: list[TaskHandle], *,
               timeout: Optional[float] = None) -> list[Any]: ...

    @abc.abstractmethod
    def cancel(self, handle: TaskHandle) -> bool: ...

    @abc.abstractmethod
    def shutdown(self) -> None: ...


# -------------------------------------------------------------------------
# Backends
# -------------------------------------------------------------------------


class LocalAsyncBackend(TaskBackend):
    """asyncio + concurrent.futures.ProcessPoolExecutor. Default for laptops."""
    name = "local-async"
    def __init__(self, max_workers: int = 4) -> None: ...
    # implementation uses ProcessPoolExecutor under the hood; submit()
    # returns a TaskHandle whose metadata holds the Future.


class RayBackend(TaskBackend):
    """ray.remote dispatch. Imported lazily; raises on init if ray not installed."""
    name = "ray"
    def __init__(self, address: Optional[str] = None,
                 num_gpus: Optional[int] = None) -> None: ...


class DaskBackend(TaskBackend):
    """dask.distributed.Client. Same lazy import + raise pattern."""
    name = "dask"
    def __init__(self, scheduler_address: Optional[str] = None) -> None: ...


class SLURMBackend(TaskBackend):
    """sbatch-based submission. Each submit() writes a job script,
    submits via ``subprocess.run(['sbatch', ...])``, and polls via
    ``squeue -j <jobid>``."""
    name = "slurm"
    def __init__(self, partition: str, account: str,
                 sbatch_extra: Optional[list[str]] = None) -> None: ...


def make_backend(spec: str | Mapping[str, Any]) -> TaskBackend:
    """Factory: 'local' | 'ray' | 'dask' | 'slurm' or dict {'backend': ..., **kw}."""
    ...
```

## 13.3 GPU scheduler + Hydra + Optuna

```python
# resistancemap/infrastructure/gpu_scheduler.py

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

import torch


class GPUScheduler:
    """Pool-of-GPUs lock so parallel tasks don't fight over the same device.

    Used by RayBackend / DaskBackend / LocalAsyncBackend whenever a task
    declares ``resources={'gpu': 1}``. SLURMBackend ignores this (SLURM
    already assigns devices via CUDA_VISIBLE_DEVICES).
    """
    def __init__(self, gpu_ids: list[int] | None = None) -> None:
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        self._available = set(gpu_ids)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    @contextmanager
    def acquire(self, n: int = 1, timeout: float | None = None) -> Iterator[list[int]]:
        with self._cond:
            self._cond.wait_for(lambda: len(self._available) >= n, timeout=timeout)
            chosen = [self._available.pop() for _ in range(n)]
        try:
            yield chosen
        finally:
            with self._cond:
                for g in chosen:
                    self._available.add(g)
                self._cond.notify_all()


# resistancemap/infrastructure/hydra_runner.py
# Wraps hydra.main so an entry-point can sweep over configs in parallel
# via the TaskBackend.

# resistancemap/infrastructure/optuna_runner.py
# Wraps optuna.create_study with a callback that submits each trial
# through the configured TaskBackend.
```

## 13.4 Most-consequential change in Phase 13

The `TaskBackend` abstraction lets us write `ablation_runner.py` and
`hpo_runner.py` *once* and have them work locally for development and on
SLURM for the actual paper run, with no code change — just a config
swap. Today, every ablation script hard-codes its own concurrency story
(`multiprocessing.Pool`, `joblib.Parallel`, etc.), which is the main
reason ablations don't fan out properly on a cluster.

---

# Phase 15 — Reproducibility (manifests, schemas, bundles)

## 15.1 Motivation

The v12 audit found two reproducibility issues:

1. Metric files (`results/*.json`) did not record the git hash, the
   config, or the data hash they were produced from.
2. There was no single "this is everything you need to reproduce this run"
   artefact.

Phase 15 makes the result JSON a typed, versioned schema and bundles every
run into a reproducibility tarball.

## 15.2 Canonical result-JSON schema

```python
# resistancemap/reproducibility/schemas.py

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"


class ClaimGateStatus(str, enum.Enum):
    PASS = "pass"
    REFUTE = "refute"
    INCONCLUSIVE = "inconclusive"
    NOT_RUN = "not_run"


@dataclass
class GitProvenance:
    commit_sha: str
    branch: str
    dirty: bool
    remote_url: Optional[str] = None


@dataclass
class DataProvenance:
    """One entry per dataset consumed by the run."""
    dataset_name: str
    source_path: str
    n_rows: int
    sha256: str
    schema_version: Optional[str] = None
    license: Optional[str] = None


@dataclass
class MetricEntry:
    name: str            # e.g. "cindex_progression"
    value: float
    n: int               # sample size the metric was computed on
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    method: Optional[str] = None   # e.g. "bootstrap-1000"


@dataclass
class ClaimGate:
    """One of F1..F10 (or v18 LENS-claim) gates from ClaimCriticAgent."""
    claim_id: str
    description: str
    status: ClaimGateStatus
    evidence_metric: Optional[str] = None
    threshold: Optional[float] = None
    observed: Optional[float] = None
    blocking_reason: Optional[str] = None


@dataclass
class RunManifest:
    """Single source of truth for a finished MORT-FM run.

    Written by ``run_manifest.write(...)`` at the end of the orchestrator
    run. Consumed by ``reproducibility/bundle.py`` to assemble the
    tarball, and by the figure/table generators as the ONLY allowed
    source of metric values (no np.random anywhere).
    """
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""                       # e.g. "v19.0-2026-05-21T08-12-00Z"
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    config_yaml: str = ""                  # full Hydra-resolved YAML
    git: Optional[GitProvenance] = None
    datasets: List[DataProvenance] = field(default_factory=list)
    metrics: List[MetricEntry] = field(default_factory=list)
    claim_gates: List[ClaimGate] = field(default_factory=list)
    agent_hashes: Dict[str, str] = field(default_factory=dict)  # from orchestrator
    seeds: Dict[str, int] = field(default_factory=dict)         # {python, numpy, torch}
    env: Dict[str, str] = field(default_factory=dict)           # {python, torch, cuda, ...}
    artefact_paths: Dict[str, str] = field(default_factory=dict)
```

## 15.3 `run_manifest` writer

```python
# resistancemap/reproducibility/run_manifest.py

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from resistancemap.reproducibility.schemas import (
    ClaimGate, DataProvenance, GitProvenance, MetricEntry, RunManifest,
)


def _git_provenance(repo_root: Path) -> GitProvenance:
    def _g(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args]
        ).decode().strip()
    sha = _g("rev-parse", "HEAD")
    branch = _g("rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_g("status", "--porcelain"))
    remote = None
    try:
        remote = _g("config", "--get", "remote.origin.url")
    except Exception:
        pass
    return GitProvenance(commit_sha=sha, branch=branch, dirty=dirty, remote_url=remote)


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _data_provenance(dataset_specs: list[Mapping[str, Any]]) -> list[DataProvenance]:
    out = []
    for d in dataset_specs:
        p = Path(d["source_path"])
        out.append(DataProvenance(
            dataset_name=d["dataset_name"],
            source_path=str(p),
            n_rows=int(d["n_rows"]),
            sha256=_file_sha256(p) if p.is_file() else d.get("sha256", ""),
            schema_version=d.get("schema_version"),
            license=d.get("license"),
        ))
    return out


def write(
    out_dir: Path,
    *,
    run_id: str,
    config_yaml: str,
    repo_root: Path,
    datasets: list[Mapping[str, Any]],
    metrics: list[MetricEntry],
    claim_gates: list[ClaimGate],
    agent_hashes: Mapping[str, str],
    seeds: Mapping[str, int],
    artefact_paths: Mapping[str, str],
) -> Path:
    """Materialise a RunManifest JSON. Returns the path written."""
    manifest = RunManifest(
        run_id=run_id,
        config_yaml=config_yaml,
        git=_git_provenance(repo_root),
        datasets=_data_provenance(datasets),
        metrics=list(metrics),
        claim_gates=list(claim_gates),
        agent_hashes=dict(agent_hashes),
        seeds=dict(seeds),
        env={
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda or "cpu",
        },
        artefact_paths=dict(artefact_paths),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "run_manifest.json"
    out_path.write_text(json.dumps(asdict(manifest), indent=2, default=str))
    return out_path
```

## 15.4 Reproducibility bundle exporter

```python
# resistancemap/reproducibility/bundle.py

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Iterable


def export_bundle(
    run_dir: Path,
    out_tar: Path,
    *,
    include_checkpoints: bool = False,
    extra_paths: Iterable[Path] = (),
) -> Path:
    """Bundle a finished run into a single .tar.gz.

    Always included:
      * run_manifest.json (the schema-validated source of truth)
      * config.yaml      (resolved Hydra config)
      * metrics/*.json   (per-claim metric files)
      * figures/*.{png,pdf,svg}
      * tables/*.{csv,tex}
      * agent_audit/*.json  (zero-trust hash chain from orchestrator)

    Optional:
      * checkpoints/*.pt   (only if include_checkpoints=True — these are big)
      * extra_paths        (e.g. external validation result files)

    The output tarball name encodes the run_id from run_manifest.json so
    a downstream consumer can refuse to merge two bundles with the same id.
    """
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    run_id = manifest["run_id"]
    if run_id not in out_tar.name:
        out_tar = out_tar.with_name(f"{out_tar.stem}__{run_id}{out_tar.suffix}")
    with tarfile.open(out_tar, "w:gz") as tf:
        for sub in ("run_manifest.json", "config.yaml",
                    "metrics", "figures", "tables", "agent_audit"):
            p = run_dir / sub
            if p.exists():
                tf.add(p, arcname=sub)
        if include_checkpoints:
            cp = run_dir / "checkpoints"
            if cp.exists():
                tf.add(cp, arcname="checkpoints")
        for extra in extra_paths:
            if extra.exists():
                tf.add(extra, arcname=extra.name)
    return out_tar


def verify_bundle(tar_path: Path) -> dict:
    """Open a bundle, re-hash its datasets against the manifest, and report
    any mismatches. Used by ExternalValidationAgent before consuming a
    third-party bundle."""
    ...
```

## 15.5 Most-consequential change in Phase 15

The `RunManifest.claim_gates` field — combined with the rule that
`FigureGenerationAgent` is only allowed to read metrics from
`run_manifest.json` — closes the v12-audit "figure fabrication" risk
(`generate_paper_figures.py` previously used `np.random` for most panels).
After Phase 15 lands, any panel that doesn't have a corresponding
`MetricEntry` in the manifest must either be removed or rejected by
`FigureGenerationAgent` at L10.

---

## Cross-phase dependency graph

```
Phase 0 (canonical trainer)
  └─> enables: TrajectorySDEAgent / SurvivalAgent (Phase 12)

Phase 12 (agent DAG)
  └─> uses:    Orchestrator (existing)
  └─> writes:  agent_hashes -> RunManifest (Phase 15)

Phase 13 (TaskBackend)
  └─> used by: ablation / HPO / per-modality preprocessing agents (Phase 12)

Phase 15 (manifests + bundles)
  └─> consumed by: FigureGenerationAgent, PaperTableAgent (Phase 12 L10)
```

Phase 0 must land first (the agents need a canonical trainer to call).
Phase 12 and Phase 13 can land in parallel. Phase 15 must land before
Phase 12 L9 (`ClaimCriticAgent`) is wired into the default DAG, because
the gate-writing protocol depends on the manifest schema.
