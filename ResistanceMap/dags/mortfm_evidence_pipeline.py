"""
mortfm_evidence_pipeline.py — Airflow DAG for the MORT-FM evidence pipeline.

Evidence-first, not model-first: training tasks exist only to feed the
downstream evidence/evaluation/gate tasks.  Every task is a thin subprocess
wrapper around the corresponding numbered script in ``scripts/mortfm/``.
No pipeline logic is reimplemented here.

Artifact-driven scheduling uses Airflow 2.x Datasets so that downstream
DAGs (e.g. a nightly data-refresh DAG) can trigger this pipeline when
new input artifacts land.

Each task logs to ``logs/airflow/{task_id}.log`` (appended).
"""

from __future__ import annotations

import datetime
import os
import subprocess
import textwrap
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    try:
        from airflow.sdk.definitions.asset import Asset as Dataset
    except ImportError:
        from airflow.datasets import Dataset
    HAS_AIRFLOW = True
except ImportError:
    HAS_AIRFLOW = False
    Dataset = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # ResistanceMap/
SCRIPTS = ROOT / "scripts" / "mortfm"
SCRIPTS_TOP = ROOT / "scripts"
LOG_DIR = ROOT / "logs" / "airflow"

# ---------------------------------------------------------------------------
# Airflow Dataset URIs — artifact-driven scheduling
# ---------------------------------------------------------------------------
DS_RAW_DATA       = Dataset("file://ResistanceMap/data/raw") if HAS_AIRFLOW else None
DS_HARMONIZED     = Dataset("file://ResistanceMap/data/harmonized") if HAS_AIRFLOW else None
DS_LONGITUDINAL   = Dataset("file://ResistanceMap/data/longitudinal") if HAS_AIRFLOW else None
DS_MMRF_PREPARED  = Dataset("file://ResistanceMap/data/mmrf_prepared") if HAS_AIRFLOW else None
DS_AUDIT_REPORT   = Dataset("file://ResistanceMap/results/longitudinal_audit") if HAS_AIRFLOW else None
DS_FOUNDATION_CKP = Dataset("file://ResistanceMap/checkpoints/foundation") if HAS_AIRFLOW else None
DS_STATE_ENCODER  = Dataset("file://ResistanceMap/checkpoints/state_encoder") if HAS_AIRFLOW else None
DS_LENS_CKP       = Dataset("file://ResistanceMap/checkpoints/lens_resistance") if HAS_AIRFLOW else None
DS_SURVIVAL_CKP   = Dataset("file://ResistanceMap/checkpoints/survival") if HAS_AIRFLOW else None
DS_BASELINES      = Dataset("file://ResistanceMap/results/baseline_comparison") if HAS_AIRFLOW else None
DS_CAUSAL         = Dataset("file://ResistanceMap/results/causal_evidence") if HAS_AIRFLOW else None
DS_GATE_REPORT    = Dataset("file://ResistanceMap/results/gate_report") if HAS_AIRFLOW else None
DS_FIGURES        = Dataset("file://ResistanceMap/results/figures") if HAS_AIRFLOW else None
DS_PUB_BUNDLE     = Dataset("file://ResistanceMap/results/publication_bundle") if HAS_AIRFLOW else None

# ---------------------------------------------------------------------------
# Helper: run a script via subprocess
# ---------------------------------------------------------------------------

def _run_script(
    script: str | Path,
    *extra_args: str,
    task_id: str = "unknown",
) -> None:
    """Run *script* with ``python -u`` in a subprocess.

    * ``cwd`` is always ``ROOT`` so relative paths inside scripts resolve
      the same way they do when run from the repo root.
    * stdout + stderr are tee-d to ``logs/airflow/{task_id}.log``.
    * ``check=True`` ensures Airflow marks the task as FAILED on nonzero
      exit.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{task_id}.log"

    cmd = ["python", "-u", str(script), *extra_args]

    with open(log_path, "a") as log_fh:
        header = (
            f"\n{'=' * 72}\n"
            f"  task_id : {task_id}\n"
            f"  cmd     : {' '.join(cmd)}\n"
            f"  cwd     : {ROOT}\n"
            f"  started : {datetime.datetime.utcnow().isoformat()}Z\n"
            f"{'=' * 72}\n"
        )
        log_fh.write(header)
        log_fh.flush()

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    return None


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def acquire_public_data(**ctx):
    _run_script(SCRIPTS / "00_download_public_data.py", task_id="acquire_public_data")


def harmonize_identifiers(**ctx):
    _run_script(SCRIPTS / "01_harmonize_identifiers.py", task_id="harmonize_identifiers")


def build_longitudinal_dataset(**ctx):
    _run_script(SCRIPTS / "02_build_longitudinal_dataset.py", task_id="build_longitudinal_dataset")


def prepare_mmrf(**ctx):
    _run_script(
        SCRIPTS / "03_prepare_mmrf.py",
        "--data-dir", str(ROOT / "data" / "raw" / "mmrf_commpass"),
        task_id="prepare_mmrf",
    )


def audit_leakage_and_temporal_validity(**ctx):
    _run_script(
        SCRIPTS_TOP / "mortfm_longitudinal_pair_audit.py",
        task_id="audit_leakage_and_temporal_validity",
    )


def _find_pairs_pkl():
    """Locate or build temporal pairs pickle for pretrain/survival scripts."""
    candidates = [
        ROOT / "data" / "processed" / "mortfm" / "temporal_pairs.pkl",
        ROOT / "data" / "processed" / "longitudinal" / "temporal_pairs.pkl",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # Build from outcomes using a tiny inline script (no separate subprocess)
    pairs_pkl = candidates[0]
    pairs_pkl.parent.mkdir(parents=True, exist_ok=True)
    build_script = textwrap.dedent(f"""\
        import pickle, sys
        sys.path.insert(0, "{ROOT}")
        from resistancemap.data.trajectory_pair_builder import build_temporal_pairs
        from resistancemap.data.clinical_outcome_loader import load_mortfm_outcomes
        outcomes = load_mortfm_outcomes("{ROOT / 'data' / 'raw' / 'mmrf_commpass'}")
        pairs = build_temporal_pairs([], outcomes, include_survival_only=True, include_unlabelled=True)
        with open("{pairs_pkl}", "wb") as f:
            pickle.dump(pairs, f)
        print(f"Built {{len(pairs)}} pairs -> {pairs_pkl}")
    """)
    subprocess.run(
        ["python", "-c", build_script],
        cwd=str(ROOT), check=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return str(pairs_pkl)


def pretrain_foundation(**ctx):
    pairs_pkl = _find_pairs_pkl()
    _run_script(
        SCRIPTS / "04_pretrain_foundation.py",
        "--pairs", pairs_pkl,
        task_id="pretrain_foundation",
    )


def train_state_encoder(**ctx):
    _run_script(SCRIPTS / "05_train_block_c_contrastive.py", task_id="train_state_encoder")


def train_lens_resistance(**ctx):
    _run_script(
        SCRIPTS / "06_train_lens_resistance.py", "--canonical",
        task_id="train_lens_resistance",
    )


def train_survival(**ctx):
    pairs_pkl = _find_pairs_pkl()
    _run_script(
        SCRIPTS / "07_train_survival.py",
        "--pairs", pairs_pkl,
        "--canonical",
        task_id="train_survival",
    )


def run_baselines(**ctx):
    _run_script(
        SCRIPTS / "10_run_baselines.py", "--real-run",
        task_id="run_baselines",
    )


def evaluate_causal_evidence(**ctx):
    _run_script(SCRIPTS / "08_eval_causal_evidence_v2.py", task_id="evaluate_causal_evidence")


def gate_revalidate(**ctx):
    _run_script(SCRIPTS / "09_gate_revalidate.py", task_id="gate_revalidate")


def generate_figures(**ctx):
    _run_script(
        SCRIPTS_TOP / "sota_benchmark_and_visualizations.py",
        task_id="generate_figures",
    )


def export_publication_bundle(**ctx):
    """Collect gate report, figures, and key artifacts into a single bundle dir."""
    bundle_dir = ROOT / "results" / "publication_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "export_publication_bundle.log"

    # Thin: just snapshot the artifacts that the earlier tasks produced.
    artifacts = [
        ROOT / "results" / "gate_report",
        ROOT / "results" / "causal_evidence",
        ROOT / "results" / "baseline_comparison.parquet",
        ROOT / "results" / "figures",
    ]

    with open(log_path, "a") as log_fh:
        log_fh.write(
            f"\n{'=' * 72}\n"
            f"  export_publication_bundle\n"
            f"  started : {datetime.datetime.utcnow().isoformat()}Z\n"
            f"{'=' * 72}\n"
        )
        collected = 0
        for src in artifacts:
            if src.exists():
                # Use cp -r to handle both files and directories
                subprocess.run(
                    ["cp", "-r", str(src), str(bundle_dir / src.name)],
                    check=True,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                )
                log_fh.write(f"  collected: {src.name}\n")
                collected += 1
            else:
                log_fh.write(f"  MISSING (skipped): {src}\n")
        log_fh.write(f"  total artifacts collected: {collected}/{len(artifacts)}\n")
        log_fh.flush()

    if collected == 0:
        raise RuntimeError("No artifacts found to bundle; upstream tasks may have failed.")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner": "mortfm",
    "retries": 0,
    "execution_timeout": datetime.timedelta(hours=12),
}

with DAG(
    dag_id="mortfm_evidence_pipeline",
    description=(
        "End-to-end MORT-FM evidence pipeline: data acquisition through "
        "publication bundle export.  Evidence-first design."
    ),
    schedule=None,                     # manual trigger only (no self-referencing asset loop)
    start_date=datetime.datetime(2026, 5, 23),
    catchup=False,
    default_args=default_args,
    tags=["mortfm", "evidence", "pipeline"],
    doc_md=textwrap.dedent("""\
        ## MORT-FM Evidence Pipeline

        Wraps the canonical `scripts/mortfm/00-10` numbered workflow.
        Each task is a thin `subprocess.run` call -- all logic lives in
        the underlying scripts (see `scripts/mortfm/README.md`).

        ### Dependency graph (evidence-first)

        ```
        acquire_public_data
          -> harmonize_identifiers
            -> build_longitudinal_dataset
              -> prepare_mmrf
                -> audit_leakage_and_temporal_validity
                  -> pretrain_foundation
                    -> train_state_encoder
                      -> train_lens_resistance
                      -> train_survival
                      -> run_baselines
                        -> evaluate_causal_evidence
                          -> gate_revalidate  (also waits on survival + baselines)
                            -> generate_figures
                              -> export_publication_bundle
        ```
    """),
) as dag:

    t_acquire = PythonOperator(
        task_id="acquire_public_data",
        python_callable=acquire_public_data,

    )

    t_harmonize = PythonOperator(
        task_id="harmonize_identifiers",
        python_callable=harmonize_identifiers,

    )

    t_longitudinal = PythonOperator(
        task_id="build_longitudinal_dataset",
        python_callable=build_longitudinal_dataset,
    )

    t_mmrf = PythonOperator(
        task_id="prepare_mmrf",
        python_callable=prepare_mmrf,
    )

    t_audit = PythonOperator(
        task_id="audit_leakage_and_temporal_validity",
        python_callable=audit_leakage_and_temporal_validity,
    )

    t_pretrain = PythonOperator(
        task_id="pretrain_foundation",
        python_callable=pretrain_foundation,
    )

    t_state_encoder = PythonOperator(
        task_id="train_state_encoder",
        python_callable=train_state_encoder,
    )

    t_lens = PythonOperator(
        task_id="train_lens_resistance",
        python_callable=train_lens_resistance,
    )

    t_survival = PythonOperator(
        task_id="train_survival",
        python_callable=train_survival,
    )

    t_baselines = PythonOperator(
        task_id="run_baselines",
        python_callable=run_baselines,
    )

    t_causal = PythonOperator(
        task_id="evaluate_causal_evidence",
        python_callable=evaluate_causal_evidence,
    )

    t_gate = PythonOperator(
        task_id="gate_revalidate",
        python_callable=gate_revalidate,
    )

    t_figures = PythonOperator(
        task_id="generate_figures",
        python_callable=generate_figures,
    )

    t_bundle = PythonOperator(
        task_id="export_publication_bundle",
        python_callable=export_publication_bundle,
    )

    # ------------------------------------------------------------------
    # Dependencies — evidence-first ordering
    # ------------------------------------------------------------------
    # Phase 1: Data acquisition & preparation (linear chain)
    t_acquire >> t_harmonize >> t_longitudinal >> t_mmrf

    # Phase 2: Audit must pass BEFORE any training begins
    t_mmrf >> t_audit

    # Phase 3: Foundation pretraining, then state encoder
    t_audit >> t_pretrain >> t_state_encoder

    # Phase 4: Three parallel training branches after state encoder
    t_state_encoder >> t_lens
    t_state_encoder >> t_survival
    t_state_encoder >> t_baselines

    # Phase 5: Causal evidence depends on trained LENS model
    t_lens >> t_causal

    # Phase 6: Gate revalidation requires survival, baselines, AND causal
    [t_survival, t_baselines, t_causal] >> t_gate

    # Phase 7: Figures after gating, then final bundle
    t_gate >> t_figures >> t_bundle
