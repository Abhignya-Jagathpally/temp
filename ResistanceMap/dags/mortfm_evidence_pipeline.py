"""
mortfm_evidence_pipeline.py — Airflow DAG for the MORT-FM evidence pipeline.

Evidence-first, not model-first. Full Block A-D data ingestion wired in.
Each task is a thin subprocess wrapper — no pipeline logic reimplemented here.
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

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "mortfm"
SCRIPTS_TOP = ROOT / "scripts"
LOG_DIR = ROOT / "logs" / "airflow"


def _run_script(script: str | Path, *extra_args: str, task_id: str = "unknown") -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{task_id}.log"
    cmd = ["python", "-u", str(script), *extra_args]
    with open(log_path, "a") as log_fh:
        log_fh.write(
            f"\n{'=' * 72}\n"
            f"  task_id : {task_id}\n"
            f"  cmd     : {' '.join(cmd)}\n"
            f"  started : {datetime.datetime.utcnow().isoformat()}Z\n"
            f"{'=' * 72}\n"
        )
        log_fh.flush()
        subprocess.run(
            cmd, cwd=str(ROOT), check=True,
            stdout=log_fh, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )


def _run_script_allow_exit1(script, *extra_args, task_id="unknown"):
    """Run script, treating exit code 1 as honest-negative (gates failed)."""
    try:
        _run_script(script, *extra_args, task_id=task_id)
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            import logging
            logging.getLogger("airflow.task").warning(
                f"{task_id} exited 1 (honest negative / gates did not pass). Continuing."
            )
        else:
            raise


# ───────────────────────────────────────────────────────────────────────
# Phase 1: Raw data download + identifier harmonization
# ───────────────────────────────────────────────────────────────────────
def acquire_public_data(**ctx):
    _run_script(SCRIPTS / "00_download_public_data.py", task_id="acquire_public_data")

def harmonize_identifiers(**ctx):
    _run_script(SCRIPTS / "01_harmonize_identifiers.py", task_id="harmonize_identifiers")

# ───────────────────────────────────────────────────────────────────────
# Phase 2: Block A — DepMap + GDSC + PRISM + STRING + UniProt + Reactome + ChEMBL
# ───────────────────────────────────────────────────────────────────────
def ingest_depmap(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_ingest_depmap.py", task_id="ingest_depmap")

def ingest_gdsc(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_ingest_gdsc.py", task_id="ingest_gdsc")

def ingest_prism(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_ingest_prism.py", task_id="ingest_prism")

def ingest_string(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_ingest_string.py", task_id="ingest_string")

def ingest_uniprot(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_ingest_uniprot.py", task_id="ingest_uniprot")

def ingest_reactome(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_ingest_reactome.py", task_id="ingest_reactome")

def ingest_chembl(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_ingest_chembl.py", task_id="ingest_chembl")

def build_biological_graph(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_build_biological_graph.py", task_id="build_biological_graph")

def train_cellline_foundation(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_train_cellline_foundation.py", task_id="train_cellline_foundation")

# ───────────────────────────────────────────────────────────────────────
# Phase 3: Block B — BeatAML ingestion + alignment + finetune
# ───────────────────────────────────────────────────────────────────────
def ingest_beataml(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_ingest_beataml.py", "--mode", "auto", task_id="ingest_beataml")

def align_beataml_features(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_align_beataml_features.py", task_id="align_beataml_features")

def train_beataml_finetune(**ctx):
    _run_script_allow_exit1(
        SCRIPTS_TOP / "mortfm_train_beataml_finetune.py",
        "--pretrain-epochs", "30", "--finetune-epochs", "30",
        task_id="train_beataml_finetune",
    )

# ───────────────────────────────────────────────────────────────────────
# Phase 4: Block C — scRNA ingestion + preprocessing + state encoder
# ───────────────────────────────────────────────────────────────────────
def ingest_geo_singlecell(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_ingest_geo_singlecell.py", task_id="ingest_geo_singlecell")

def build_scrna_manifest(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_build_scrna_manifest.py", task_id="build_scrna_manifest")

def train_state_encoder(**ctx):
    _run_script(SCRIPTS / "05_train_block_c_contrastive.py", task_id="train_state_encoder")

# ───────────────────────────────────────────────────────────────────────
# Phase 5: Block D — ESM-2 embeddings + CRISPR external evidence
# ───────────────────────────────────────────────────────────────────────
def compute_esm2_embeddings(**ctx):
    uniprot_dir = ROOT / "data" / "raw_public" / "uniprot"
    if not uniprot_dir.exists():
        uniprot_dir = ROOT / "data" / "raw" / "uniprot"
    _run_script(
        SCRIPTS_TOP / "mortfm_compute_esm2_embeddings.py",
        "--uniprot-dir", str(uniprot_dir),
        task_id="compute_esm2_embeddings",
    )

def ingest_crispr(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_ingest_crispr.py", task_id="ingest_crispr")

# ───────────────────────────────────────────────────────────────────────
# Phase 6: Block E — Integration + longitudinal substrate
# ───────────────────────────────────────────────────────────────────────
def integrate_blocks(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_integrate_blocks_bcd.py", task_id="integrate_blocks")

def prepare_mmrf(**ctx):
    _run_script(
        SCRIPTS / "03_prepare_mmrf.py",
        "--data-dir", str(ROOT / "data" / "raw" / "mmrf_commpass"),
        task_id="prepare_mmrf",
    )

def build_longitudinal_dataset(**ctx):
    _run_script(SCRIPTS / "02_build_longitudinal_dataset.py", task_id="build_longitudinal_dataset")

def build_temporal_pairs(**ctx):
    """Build temporal pairs from outcomes. Survival-only mode when no RNA snapshots."""
    import pickle
    pairs_pkl = ROOT / "data" / "processed" / "mortfm" / "temporal_pairs.pkl"
    outcomes_pkl = ROOT / "data" / "processed" / "mortfm" / "outcomes.pkl"
    if not outcomes_pkl.exists():
        import logging
        logging.getLogger("airflow.task").warning("No outcomes.pkl — skipping build_temporal_pairs.")
        return
    # Build survival-only pairs inline (no snapshots needed)
    build_script = textwrap.dedent(f"""\
        import pickle, sys
        sys.path.insert(0, "{ROOT}")
        from resistancemap.data.trajectory_pair_builder import build_temporal_pairs
        with open("{outcomes_pkl}", "rb") as f:
            outcomes = pickle.load(f)
        pairs = build_temporal_pairs([], outcomes, include_survival_only=True, include_unlabelled=True)
        with open("{pairs_pkl}", "wb") as f:
            pickle.dump(pairs, f)
        print(f"Built {{len(pairs)}} survival-only pairs -> {pairs_pkl}")
    """)
    subprocess.run(
        ["python", "-c", build_script],
        cwd=str(ROOT), check=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

def audit_leakage(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_longitudinal_pair_audit.py", task_id="audit_leakage")

# ───────────────────────────────────────────────────────────────────────
# Phase 7: LENS training (trajectory, survival, resistance)
# ───────────────────────────────────────────────────────────────────────
def pretrain_foundation(**ctx):
    pairs_pkl = str(ROOT / "data" / "processed" / "mortfm" / "temporal_pairs.pkl")
    if Path(pairs_pkl).exists():
        import pickle
        with open(pairs_pkl, "rb") as f:
            pairs = pickle.load(f)
        if pairs:
            _run_script(SCRIPTS / "04_pretrain_foundation.py", "--pairs", pairs_pkl,
                        task_id="pretrain_foundation")
            return
    import logging
    logging.getLogger("airflow.task").warning("No temporal pairs — skipping pretrain_foundation.")

def train_lens_resistance(**ctx):
    _run_script_allow_exit1(SCRIPTS / "06_train_lens_resistance.py", task_id="train_lens_resistance")

def train_survival(**ctx):
    pairs_pkl = str(ROOT / "data" / "processed" / "mortfm" / "temporal_pairs.pkl")
    if Path(pairs_pkl).exists():
        import pickle
        with open(pairs_pkl, "rb") as f:
            pairs = pickle.load(f)
        if pairs:
            _run_script_allow_exit1(SCRIPTS / "07_train_survival.py", "--pairs", pairs_pkl,
                                    "--canonical", task_id="train_survival")
            return
    import logging
    logging.getLogger("airflow.task").warning("No temporal pairs — skipping train_survival.")

# ───────────────────────────────────────────────────────────────────────
# Phase 8: Evidence evaluation + baselines + gates
# ───────────────────────────────────────────────────────────────────────
def run_baselines(**ctx):
    _run_script_allow_exit1(SCRIPTS / "10_run_baselines.py", task_id="run_baselines")

def evaluate_causal_evidence(**ctx):
    _run_script_allow_exit1(SCRIPTS / "08_eval_causal_evidence_v2.py", task_id="evaluate_causal_evidence")

def gate_revalidate(**ctx):
    _run_script_allow_exit1(SCRIPTS / "09_gate_revalidate.py", task_id="gate_revalidate")

def emit_registries(**ctx):
    _run_script(SCRIPTS_TOP / "mortfm_emit_registries.py", task_id="emit_registries")

# ───────────────────────────────────────────────────────────────────────
# Phase 9: Figures + publication bundle
# ───────────────────────────────────────────────────────────────────────
def generate_figures(**ctx):
    _run_script(SCRIPTS_TOP / "sota_benchmark_and_visualizations.py", task_id="generate_figures")

def export_publication_bundle(**ctx):
    _run_script(SCRIPTS_TOP / "export_publication_bundle.py", task_id="export_publication_bundle")


# ═══════════════════════════════════════════════════════════════════════
# DAG definition
# ═══════════════════════════════════════════════════════════════════════
default_args = {
    "owner": "mortfm",
    "retries": 0,
    "execution_timeout": datetime.timedelta(hours=12),
}

with DAG(
    dag_id="mortfm_evidence_pipeline",
    description="End-to-end MORT-FM evidence pipeline with full Block A-D ingestion.",
    schedule=None,
    start_date=datetime.datetime(2026, 5, 23),
    catchup=False,
    default_args=default_args,
    tags=["mortfm", "evidence", "pipeline"],
    doc_md=textwrap.dedent("""\
        ## MORT-FM Evidence Pipeline (Full)

        Phases 1-9 covering data download, Block A-D ingestion, LENS training,
        evidence evaluation, and publication bundle export.
    """),
) as dag:

    # Phase 1: Download + harmonize
    t_acquire       = PythonOperator(task_id="acquire_public_data", python_callable=acquire_public_data)
    t_harmonize     = PythonOperator(task_id="harmonize_identifiers", python_callable=harmonize_identifiers)

    # Phase 2: Block A — cell-line foundation data + training
    t_depmap        = PythonOperator(task_id="ingest_depmap", python_callable=ingest_depmap)
    t_gdsc          = PythonOperator(task_id="ingest_gdsc", python_callable=ingest_gdsc)
    t_prism         = PythonOperator(task_id="ingest_prism", python_callable=ingest_prism)
    t_string        = PythonOperator(task_id="ingest_string", python_callable=ingest_string)
    t_uniprot       = PythonOperator(task_id="ingest_uniprot", python_callable=ingest_uniprot)
    t_reactome      = PythonOperator(task_id="ingest_reactome", python_callable=ingest_reactome)
    t_chembl        = PythonOperator(task_id="ingest_chembl", python_callable=ingest_chembl)
    t_bio_graph     = PythonOperator(task_id="build_biological_graph", python_callable=build_biological_graph)
    t_block_a       = PythonOperator(task_id="train_cellline_foundation", python_callable=train_cellline_foundation)

    # Phase 3: Block B — BeatAML
    t_beataml       = PythonOperator(task_id="ingest_beataml", python_callable=ingest_beataml)
    t_align_beat    = PythonOperator(task_id="align_beataml_features", python_callable=align_beataml_features)
    t_block_b       = PythonOperator(task_id="train_beataml_finetune", python_callable=train_beataml_finetune)

    # Phase 4: Block C — scRNA
    t_geo_sc        = PythonOperator(task_id="ingest_geo_singlecell", python_callable=ingest_geo_singlecell)
    t_scrna_man     = PythonOperator(task_id="build_scrna_manifest", python_callable=build_scrna_manifest)
    t_block_c       = PythonOperator(task_id="train_state_encoder", python_callable=train_state_encoder)

    # Phase 5: Block D — ESM-2 + CRISPR
    t_esm2          = PythonOperator(task_id="compute_esm2_embeddings", python_callable=compute_esm2_embeddings)
    t_crispr        = PythonOperator(task_id="ingest_crispr", python_callable=ingest_crispr)

    # Phase 6: Integration + longitudinal
    t_integrate     = PythonOperator(task_id="integrate_blocks", python_callable=integrate_blocks)
    t_mmrf          = PythonOperator(task_id="prepare_mmrf", python_callable=prepare_mmrf)
    t_longitudinal  = PythonOperator(task_id="build_longitudinal_dataset", python_callable=build_longitudinal_dataset)
    t_pairs         = PythonOperator(task_id="build_temporal_pairs", python_callable=build_temporal_pairs)
    t_audit         = PythonOperator(task_id="audit_leakage", python_callable=audit_leakage)

    # Phase 7: LENS training
    t_pretrain      = PythonOperator(task_id="pretrain_foundation", python_callable=pretrain_foundation)
    t_lens          = PythonOperator(task_id="train_lens_resistance", python_callable=train_lens_resistance)
    t_survival      = PythonOperator(task_id="train_survival", python_callable=train_survival)

    # Phase 8: Evidence + baselines + gates
    t_baselines     = PythonOperator(task_id="run_baselines", python_callable=run_baselines)
    t_causal        = PythonOperator(task_id="evaluate_causal_evidence", python_callable=evaluate_causal_evidence)
    t_registries    = PythonOperator(task_id="emit_registries", python_callable=emit_registries)
    t_gate          = PythonOperator(task_id="gate_revalidate", python_callable=gate_revalidate)

    # Phase 9: Figures + bundle
    t_figures       = PythonOperator(task_id="generate_figures", python_callable=generate_figures)
    t_bundle        = PythonOperator(task_id="export_publication_bundle", python_callable=export_publication_bundle)

    # ──────────────────────────────────────────────────────────────────
    # Dependencies
    # ──────────────────────────────────────────────────────────────────

    # Phase 1: download → harmonize
    t_acquire >> t_harmonize

    # Phase 2: Block A ingestion (parallel after harmonize) → graph → foundation training
    t_harmonize >> [t_depmap, t_gdsc, t_prism, t_string, t_uniprot, t_reactome, t_chembl]
    [t_string, t_uniprot, t_reactome, t_chembl] >> t_bio_graph
    [t_depmap, t_gdsc, t_prism, t_bio_graph] >> t_block_a

    # Phase 3: Block B (after Block A + BeatAML ingestion)
    t_harmonize >> t_beataml
    [t_block_a, t_beataml] >> t_align_beat >> t_block_b

    # Phase 4: Block C (parallel with Block A/B)
    t_harmonize >> t_geo_sc >> t_scrna_man >> t_block_c

    # Phase 5: Block D (after graph + uniprot)
    [t_bio_graph, t_uniprot] >> t_esm2
    t_depmap >> t_crispr

    # Phase 6: Integration → longitudinal → audit
    [t_block_a, t_block_b, t_block_c, t_esm2] >> t_integrate
    t_harmonize >> t_mmrf >> t_longitudinal >> t_pairs
    [t_integrate, t_pairs] >> t_audit

    # Phase 7: LENS training (after audit passes)
    t_audit >> t_pretrain >> t_lens
    t_audit >> t_survival

    # Phase 8: Evidence (after LENS + baselines)
    t_audit >> t_baselines
    t_lens >> t_causal
    [t_integrate, t_lens] >> t_registries
    [t_survival, t_baselines, t_causal, t_registries] >> t_gate

    # Phase 9: Figures + bundle
    t_gate >> t_figures >> t_bundle
