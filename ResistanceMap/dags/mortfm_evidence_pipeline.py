"""
ResistanceMap v20 — MORT-FM Evidence Pipeline (Airflow DAG)
============================================================
Refactored from v19 to support the lab-first, open-access-only data path.

Two parallel tracks:
  Track A (v19, inherited): Cell-line foundation → BeatAML → integration
  Track B (v20, new):       GDC open + GEO scRNA → scVI → PK-SSM → baselines

Both tracks converge at the evaluation/gating phase.

Track B runs entirely on open-access data (no MMRF Virtual Lab, no dbGaP).
Track A is optional — set ENABLE_CELLLINE_TRACK=False to skip.

Usage:
  airflow dags trigger mortfm_evidence_pipeline_v20
  airflow dags trigger mortfm_evidence_pipeline_v20 --conf '{"track": "open_access_only"}'
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.models.param import Param
from airflow.utils.trigger_rule import TriggerRule

# =============================================================================
# Configuration
# =============================================================================

REPO_ROOT = Path(os.environ.get(
    "RESISTANCEMAP_ROOT",
    Path(__file__).resolve().parent.parent,
))
SCRIPTS = REPO_ROOT / "scripts"
MORTFM_SCRIPTS = SCRIPTS / "mortfm"

# Toggle cell-line track (set False for open-access-only mode)
ENABLE_CELLLINE_TRACK = os.environ.get("ENABLE_CELLLINE_TRACK", "false").lower() == "true"

DEFAULT_ARGS = {
    "owner": "resistancemap",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}


# =============================================================================
# Helpers
# =============================================================================

def _run_script(script_path: str, *args, allow_exit1: bool = False, **kwargs):
    """Run a Python script as a subprocess. Raises on non-zero exit unless allow_exit1."""
    cmd = [sys.executable, "-u", str(REPO_ROOT / script_path)] + list(args)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        if allow_exit1 and result.returncode == 1:
            print(f"[WARN] {script_path} exited with code 1 (honest negative / gate not passed)")
        else:
            raise subprocess.CalledProcessError(result.returncode, cmd)


def _check_cellline_enabled(**kwargs):
    """ShortCircuit: skip cell-line track if disabled."""
    conf = kwargs.get("dag_run", {})
    if hasattr(conf, "conf") and conf.conf:
        track = conf.conf.get("track", "")
        if track == "open_access_only":
            return False
    return ENABLE_CELLLINE_TRACK


# =============================================================================
# DAG Definition
# =============================================================================

with DAG(
    dag_id="mortfm_evidence_pipeline_v20",
    default_args=DEFAULT_ARGS,
    description="MORT-FM v20 evidence pipeline — lab-first with open-access data",
    schedule=None,  # Manual trigger only (Airflow 3.x renamed schedule_interval -> schedule)
    start_date=datetime(2026, 6, 1),
    catchup=False,
    tags=["resistancemap", "v20", "open-access", "lab-first"],
    params={
        "track": Param(
            default="full",
            type="string",
            enum=["full", "open_access_only", "cellline_only"],
            description="Which track to run",
        ),
    },
    max_active_runs=1,
) as dag:

    # =================================================================
    # PHASE 0: Gate — which track(s) to run
    # =================================================================

    check_cellline = ShortCircuitOperator(
        task_id="check_cellline_track_enabled",
        python_callable=_check_cellline_enabled,
        # Airflow 3.x always injects context; `provide_context` was removed.
        # Issue 2 fix: only skip the DIRECT cell-line downstream, NOT the whole
        # subgraph. Without this, disabling the cell-line track also skips the
        # Track-B convergence + evaluation tasks (run_baselines, gate_revalidate).
        ignore_downstream_trigger_rules=False,
    )

    # =================================================================
    # PHASE 1: Common — Acquire & Harmonize (shared by both tracks)
    # =================================================================

    acquire_public = PythonOperator(
        task_id="acquire_public_data",
        python_callable=_run_script,
        op_args=["scripts/mortfm/00_download_public_data.py"],
        # Issue 1 fix: the downloader exits 1 when ANY source fails, but two
        # "failures" are by-design non-fatal for the open-access path —
        # BeatAML has no public auto-download (controlled; fetched separately
        # from AWS Open Data) and STRING can hiccup. Tolerate exit 1 so one
        # expected miss does not kill the whole pipeline at its root.
        op_kwargs={"allow_exit1": True},
    )

    harmonize_ids = PythonOperator(
        task_id="harmonize_identifiers",
        python_callable=_run_script,
        op_args=["scripts/mortfm/01_harmonize_identifiers.py"],
    )

    # P0-2 (Issue 3 fix): harmonize CONSUMES the UniProt/STRING ingest outputs,
    # so it must run AFTER them — wired below where those tasks are defined.
    # (acquire -> ingests -> harmonize, not acquire -> harmonize -> ingests.)

    # =================================================================
    # TRACK A: Cell-Line Foundation (v19 inherited, optional)
    # =================================================================

    # --- Phase 2A: Ingest cell-line data ---
    ingest_depmap = PythonOperator(
        task_id="ingest_depmap",
        python_callable=_run_script,
        op_args=["scripts/mortfm_ingest_depmap.py"],
    )
    ingest_gdsc = PythonOperator(
        task_id="ingest_gdsc",
        python_callable=_run_script,
        op_args=["scripts/mortfm_ingest_gdsc.py"],
    )
    ingest_prism = PythonOperator(
        task_id="ingest_prism",
        python_callable=_run_script,
        op_args=["scripts/mortfm_ingest_prism.py"],
    )

    # --- Phase 2A: Ingest bio knowledge (shared) ---
    ingest_string = PythonOperator(
        task_id="ingest_string",
        python_callable=_run_script,
        op_args=["scripts/mortfm_ingest_string.py"],
    )
    ingest_uniprot = PythonOperator(
        task_id="ingest_uniprot",
        python_callable=_run_script,
        op_args=["scripts/mortfm_ingest_uniprot.py"],
    )
    ingest_reactome = PythonOperator(
        task_id="ingest_reactome",
        python_callable=_run_script,
        op_args=["scripts/mortfm_ingest_reactome.py"],
    )
    ingest_chembl = PythonOperator(
        task_id="ingest_chembl",
        python_callable=_run_script,
        op_args=["scripts/mortfm_ingest_chembl.py"],
    )
    build_bio_graph = PythonOperator(
        task_id="build_biological_graph",
        python_callable=_run_script,
        op_args=["scripts/mortfm_build_biological_graph.py"],
    )

    # --- Phase 2A: Cell-line foundation training ---
    train_cellline = PythonOperator(
        task_id="train_cellline_foundation",
        python_callable=_run_script,
        op_args=["scripts/mortfm_train_cellline_foundation.py"],
    )

    # Cell-line track dependencies
    check_cellline >> [ingest_depmap, ingest_gdsc, ingest_prism]
    # P0-2 (Issue 3 fix): bio-knowledge ingests run on acquire, BEFORE harmonize,
    # because 01_harmonize_identifiers reads protein_nodes.csv (uniprot ingest) and
    # string_aliases.parquet (string ingest). harmonize then depends on those two.
    acquire_public >> [ingest_string, ingest_uniprot, ingest_reactome, ingest_chembl]
    [ingest_string, ingest_uniprot] >> harmonize_ids
    [ingest_string, ingest_uniprot, ingest_reactome, ingest_chembl] >> build_bio_graph
    harmonize_ids >> [ingest_depmap, ingest_gdsc, ingest_prism]
    [ingest_depmap, ingest_gdsc, ingest_prism, build_bio_graph] >> train_cellline

    # --- Phase 3A: BeatAML finetune (Block B) ---
    ingest_beataml = PythonOperator(
        task_id="ingest_beataml",
        python_callable=_run_script,
        op_args=["scripts/mortfm_ingest_beataml.py", "--mode", "auto"],
    )
    align_beataml = PythonOperator(
        task_id="align_beataml_features",
        python_callable=_run_script,
        op_args=["scripts/mortfm_align_beataml_features.py"],
    )
    train_beataml = PythonOperator(
        task_id="train_beataml_finetune",
        python_callable=_run_script,
        op_args=["scripts/mortfm_train_beataml_finetune.py"],
    )

    check_cellline >> ingest_beataml
    harmonize_ids >> ingest_beataml
    [train_cellline, ingest_beataml] >> align_beataml >> train_beataml

    # =================================================================
    # TRACK B: Open-Access Lab-First (v20 NEW)
    # =================================================================

    # --- Phase 2B: Download open-access data ---

    download_gdc_open = PythonOperator(
        task_id="download_gdc_open_mmrf",
        python_callable=_run_script,
        op_args=["scripts/run_first_results.py"],  # Downloads GDC clinical + runs baselines
        doc_md="""
        Downloads MMRF-COMMPASS clinical data from GDC open tier (no dbGaP).
        Runs the corrected baseline cascade (CoxPH, EN-Cox, sksurv RSF, GBM).
        Produces: results/v20_first_baselines/
        """,
    )

    download_geo_scrna = PythonOperator(
        task_id="download_geo_scrna",
        python_callable=_run_script,
        op_args=["scripts/download_geo_scrna.py"],
        doc_md="""
        Downloads open-access scRNA-seq for MM:
        - Zenodo panImmune.h5ad (~2.7 GB, CC-BY-4.0)
        - GSE161801 (RRMM pre/post-treatment)
        - GSE189460 (bortezomib responders vs non-responders)
        Produces: data/open_access/geo_scrna/
        """,
        execution_timeout=timedelta(hours=2),  # Large downloads
    )

    # Both download tasks start after harmonize_ids
    harmonize_ids >> download_gdc_open
    harmonize_ids >> download_geo_scrna

    # --- Phase 3B: scVI Integration ---

    run_scvi = PythonOperator(
        task_id="train_scvi_integration",
        python_callable=_run_script,
        op_args=["scripts/run_scvi_integration.py"],
        doc_md="""
        Trains scVI on merged MM scRNA-seq data. Produces:
        - data/processed/scvi_latent_z.npy (batch-corrected latent, R^30)
        - data/processed/chromatin_expression.csv (20 reader/writer genes for ChromatinODE)
        - data/processed/biomarker_proxy_expression.csv (Ig genes, B2M, ALB for PK decoder)
        - checkpoints/scvi/scvi_model/
        """,
        execution_timeout=timedelta(hours=3),  # GPU training
    )

    download_geo_scrna >> run_scvi

    # --- Phase 4B: PK-SSM Forward Pass ---

    run_pkssm = PythonOperator(
        task_id="run_pkssm_forward",
        python_callable=_run_script,
        op_args=["scripts/run_pkssm_forward.py"],
        doc_md="""
        Runs the PK-SSM forward pass on scVI outputs:
        scVI latent z → PK observation model → mechanism classification → hazard.
        NOTE: UNTRAINED — demonstrates architecture, not predictions.
        Training requires MMRF Virtual Lab outcome data.
        Produces: results/v20_pkssm/
        """,
    )

    run_scvi >> run_pkssm

    # =================================================================
    # PHASE 4: Shared — scRNA State Encoder (Block C)
    # Rewired to use scVI from Track B instead of custom encoder
    # =================================================================

    ingest_geo_sc = PythonOperator(
        task_id="ingest_geo_singlecell",
        python_callable=_run_script,
        op_args=["scripts/mortfm_ingest_geo_singlecell.py"],
    )
    build_scrna_manifest = PythonOperator(
        task_id="build_scrna_manifest",
        python_callable=_run_script,
        op_args=["scripts/mortfm_build_scrna_manifest.py"],
    )
    train_state_encoder = PythonOperator(
        task_id="train_state_encoder",
        python_callable=_run_script,
        op_args=["scripts/mortfm/05_train_block_c_contrastive.py"],
    )

    harmonize_ids >> ingest_geo_sc >> build_scrna_manifest >> train_state_encoder
    # scVI latent can augment the state encoder
    run_scvi >> train_state_encoder

    # =================================================================
    # PHASE 5: Shared — Protein Embeddings (Block D)
    # =================================================================

    compute_esm2 = PythonOperator(
        task_id="compute_esm2_embeddings",
        python_callable=_run_script,
        op_args=["scripts/mortfm_compute_esm2_embeddings.py"],
    )
    ingest_crispr = PythonOperator(
        task_id="ingest_crispr",
        python_callable=_run_script,
        op_args=["scripts/mortfm_ingest_crispr.py"],
    )

    [build_bio_graph, ingest_uniprot] >> compute_esm2
    ingest_depmap >> ingest_crispr

    # =================================================================
    # PHASE 6: Integration + MMRF Patient Data
    # =================================================================

    integrate_blocks = PythonOperator(
        task_id="integrate_blocks",
        python_callable=_run_script,
        op_args=["scripts/mortfm_integrate_blocks_bcd.py"],
        trigger_rule=TriggerRule.NONE_FAILED,  # Proceed even if cell-line track skipped
    )

    prepare_mmrf = PythonOperator(
        task_id="prepare_mmrf",
        python_callable=_run_script,
        op_args=["scripts/mortfm/03_prepare_mmrf.py"],
    )

    build_snapshots = PythonOperator(
        task_id="build_mmrf_snapshots",
        python_callable=lambda: _run_script(
            "scripts/mortfm/02_build_longitudinal_dataset.py", "--mode", "snapshots"
        ),
    )

    build_longitudinal = PythonOperator(
        task_id="build_longitudinal_dataset",
        python_callable=_run_script,
        op_args=["scripts/mortfm/02_build_longitudinal_dataset.py"],
    )

    build_temporal_pairs = PythonOperator(
        task_id="build_temporal_pairs",
        python_callable=lambda: _run_script(
            "scripts/mortfm/02_build_longitudinal_dataset.py", "--mode", "pairs"
        ),
    )

    audit_leakage = PythonOperator(
        task_id="audit_leakage",
        python_callable=_run_script,
        op_args=["scripts/mortfm_longitudinal_pair_audit.py"],
    )

    # Integration dependencies — converge both tracks
    [train_beataml, train_state_encoder, compute_esm2] >> integrate_blocks
    # Also allow Track B outputs to feed integration
    run_scvi >> integrate_blocks

    harmonize_ids >> prepare_mmrf >> build_snapshots >> build_longitudinal >> build_temporal_pairs
    [integrate_blocks, build_temporal_pairs] >> audit_leakage

    # =================================================================
    # PHASE 7: Training — LENS + Survival + PK-SSM
    # =================================================================

    pretrain_foundation = PythonOperator(
        task_id="pretrain_foundation",
        python_callable=_run_script,
        op_args=["scripts/mortfm/04_pretrain_foundation.py"],
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    train_lens = PythonOperator(
        task_id="train_lens_resistance",
        python_callable=_run_script,
        op_args=["scripts/mortfm/06_train_lens_resistance.py"],
    )

    train_survival = PythonOperator(
        task_id="train_survival",
        python_callable=_run_script,
        op_args=["scripts/mortfm/07_train_survival.py"],
    )

    audit_leakage >> pretrain_foundation >> train_lens
    audit_leakage >> train_survival

    # =================================================================
    # PHASE 8: Evaluation — Baselines + Causal Evidence + Gates
    # =================================================================

    run_baselines = PythonOperator(
        task_id="run_baselines",
        python_callable=_run_script,
        op_args=["scripts/mortfm/10_run_baselines.py"],
        doc_md="Corrected baselines: sksurv RSF (Bug #4 fix), EN-Cox, GBM-Surv, CoxPH (Bug #1 fix).",
    )

    eval_causal = PythonOperator(
        task_id="evaluate_causal_evidence",
        python_callable=_run_script,
        op_args=["scripts/mortfm/08_eval_causal_evidence_v2.py"],
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    gate_revalidate = PythonOperator(
        task_id="gate_revalidate",
        python_callable=_run_script,
        op_args=["scripts/mortfm/09_gate_revalidate.py"],
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    emit_registries = PythonOperator(
        task_id="emit_registries",
        python_callable=_run_script,
        op_args=["scripts/mortfm_emit_registries.py"],
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    audit_leakage >> run_baselines
    train_lens >> eval_causal
    [integrate_blocks, train_lens] >> emit_registries
    [train_survival, run_baselines, eval_causal, emit_registries] >> gate_revalidate

    # --- v20 Track B baselines also feed gating ---
    download_gdc_open >> gate_revalidate
    run_pkssm >> gate_revalidate

    # =================================================================
    # PHASE 9: Results — Figures + Report + Publication Bundle
    # =================================================================

    # v20 Phase 6 — external validation on open-access cohorts (GDC OS /
    # GSE136337 / GSE24080). Consumes model predictions if present; otherwise
    # reports each path as "skipped" (no fabrication). Runs even if upstream
    # training was blocked.
    validate_open_access = PythonOperator(
        task_id="validate_open_access",
        python_callable=_run_script,
        op_args=["scripts/mortfm/11_validate_open_access.py",
                 "--gdc-open-dir", "data/gdc_mmrf_open",
                 "--geo-bulk-dir", "data/geo_bulk"],
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    run_baselines >> validate_open_access

    gen_figures = PythonOperator(
        task_id="generate_figures",
        python_callable=_run_script,
        op_args=["scripts/sota_benchmark_and_visualizations.py"],
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    gen_v20_report = PythonOperator(
        task_id="generate_v20_report",
        python_callable=_run_script,
        op_args=["scripts/generate_report.py"],
        doc_md="Combines baseline results, scVI outputs, and PK-SSM forward pass into one report.",
    )

    export_bundle = PythonOperator(
        task_id="export_publication_bundle",
        python_callable=_run_script,
        op_args=["scripts/export_publication_bundle.py"],
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    gate_revalidate >> gen_figures
    gate_revalidate >> gen_v20_report
    [gen_figures, gen_v20_report, validate_open_access] >> export_bundle