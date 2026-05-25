"""
mortfm_spark_evidence_pipeline.py -- Spark-native Airflow DAG for MORT-FM evidence pipeline.

Replaces the legacy script-runner pattern with SparkSubmitOperator for all
data-lake stages and BashOperator for GPU/PyTorch training. Single Spark
application (`spark_jobs/mortfm_lakehouse.py`) orchestrated via --stage argument.

Dependency graph:
    raw_landing
        -> create_raw_hive_tables
            -> cleanse_standardize
                -> curate_features
                    -> confirm_analytical_dataset
                        -> audit_leakage_temporal_validity
                            -> [train_mortfm, run_baselines]  (parallel)
                                -> evaluate_causal_evidence
                                    -> strict_claim_gate
"""

from __future__ import annotations

import datetime
from pathlib import Path

from airflow import DAG
try:
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:
    from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

import os
ROOT = os.environ.get("MORTFM_ROOT", str(Path(__file__).resolve().parents[1]))
SPARK_APP = f"{ROOT}/spark_jobs/mortfm_lakehouse.py"
CONFIG_PATH = f"{ROOT}/configs/workflows/mortfm_spark_airflow.yaml"

SPARK_CONF = {
    "spark.sql.catalogImplementation": "hive",
    "spark.sql.shuffle.partitions": "400",
    "spark.sql.caseSensitive": "true",
    "spark.sql.warehouse.dir": f"{ROOT}/data/lake/warehouse",
    "javax.jdo.option.ConnectionURL": (
        f"jdbc:derby:;databaseName={ROOT}/data/lake/metastore_db;create=true"
    ),
    "hive.metastore.schema.verification": "false",
    "datanucleus.autoCreateSchema": "true",
    "datanucleus.autoCreateTables": "true",
}
SPARK_DRIVER_MEMORY = "8g"

# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------

default_args = {
    "owner": "mortfm",
    "retries": 0,
    "execution_timeout": datetime.timedelta(hours=12),
}

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="mortfm_spark_evidence_pipeline",
    description=(
        "Spark-native MORT-FM evidence pipeline: lakehouse ETL stages via "
        "SparkSubmitOperator, GPU training via BashOperator, evidence-gated claims."
    ),
    schedule=None,
    start_date=datetime.datetime(2026, 5, 24),
    catchup=False,
    default_args=default_args,
    tags=["mortfm", "spark", "airflow", "evidence-gated"],
) as dag:

    # ------------------------------------------------------------------
    # Stage 1: Raw Landing
    # Ingest raw public data into the bronze layer of the data lake.
    # ------------------------------------------------------------------
    raw_landing = SparkSubmitOperator(
        task_id="raw_landing",
        application=SPARK_APP,
        application_args=["--stage", "raw", "--config", CONFIG_PATH],
        conf=SPARK_CONF,
        driver_memory=SPARK_DRIVER_MEMORY,
        name="mortfm_raw_landing",
    )

    # ------------------------------------------------------------------
    # Stage 2: Create Raw Hive Tables
    # Register raw parquet/CSV as external Hive-managed tables.
    # ------------------------------------------------------------------
    create_raw_hive_tables = SparkSubmitOperator(
        task_id="create_raw_hive_tables",
        application=SPARK_APP,
        application_args=["--stage", "raw_hive", "--config", CONFIG_PATH],
        conf=SPARK_CONF,
        driver_memory=SPARK_DRIVER_MEMORY,
        name="mortfm_create_raw_hive_tables",
    )

    # ------------------------------------------------------------------
    # Stage 3: Cleanse and Standardize
    # Apply QC filters, type casting, identifier normalization.
    # ------------------------------------------------------------------
    cleanse_standardize = SparkSubmitOperator(
        task_id="cleanse_standardize",
        application=SPARK_APP,
        application_args=["--stage", "cleansed", "--config", CONFIG_PATH],
        conf=SPARK_CONF,
        driver_memory=SPARK_DRIVER_MEMORY,
        name="mortfm_cleanse_standardize",
    )

    # ------------------------------------------------------------------
    # Stage 4: Curate Features
    # Build analytical features: gene selection, cross-modality alignment,
    # identifier harmonization, graph edge assembly.
    # ------------------------------------------------------------------
    curate_features = SparkSubmitOperator(
        task_id="curate_features",
        application=SPARK_APP,
        application_args=["--stage", "curated", "--config", CONFIG_PATH],
        conf=SPARK_CONF,
        driver_memory=SPARK_DRIVER_MEMORY,
        name="mortfm_curate_features",
    )

    # ------------------------------------------------------------------
    # Stage 5: Confirm Analytical Dataset
    # Validate feature completeness, produce patient-disjoint splits,
    # write confirmed dataset tables and split manifest.
    # ------------------------------------------------------------------
    confirm_analytical_dataset = SparkSubmitOperator(
        task_id="confirm_analytical_dataset",
        application=SPARK_APP,
        application_args=[
            "--stage", "confirmed",
            "--config", CONFIG_PATH,
        ],
        conf=SPARK_CONF,
        driver_memory=SPARK_DRIVER_MEMORY,
        name="mortfm_confirm_analytical_dataset",
    )

    # ------------------------------------------------------------------
    # Stage 6: Audit Leakage and Temporal Validity
    # Patient-disjoint split verification, temporal leakage detection,
    # censoring-policy application. Hard gate: pipeline halts if leakage found.
    # ------------------------------------------------------------------
    audit_leakage_temporal_validity = SparkSubmitOperator(
        task_id="audit_leakage_temporal_validity",
        application=SPARK_APP,
        application_args=[
            "--stage", "audit",
            "--config", CONFIG_PATH,
        ],
        conf=SPARK_CONF,
        driver_memory=SPARK_DRIVER_MEMORY,
        name="mortfm_audit_leakage_temporal_validity",
    )

    # ------------------------------------------------------------------
    # Stage 7a: Train MORT-FM (GPU / PyTorch)
    # LENS resistance head on confirmed analytical dataset.
    # ------------------------------------------------------------------
    train_mortfm = BashOperator(
        task_id="train_mortfm",
        bash_command=(
            "cd {{ params.root }} && "
            "python -u scripts/mortfm/06_train_lens_resistance.py "
            "  --confirmed-dataset {{ params.root }}/data/lake/confirmed/mortfm_training_dataset "
            "  --split-manifest {{ params.root }}/data/lake/confirmed/split_manifest.json "
            "  --graph-embedding-table {{ params.root }}/data/processed/graphs/biological_edges.parquet "
            "  --use-graph-projector "
            "  --use-clinical"
        ),
        params={"root": ROOT},
        cwd=ROOT,
    )

    # ------------------------------------------------------------------
    # Stage 7b: Run Baselines (GPU / PyTorch, parallel with training)
    # Patient-longitudinal baselines on same splits for honest comparison.
    # ------------------------------------------------------------------
    run_baselines = BashOperator(
        task_id="run_baselines",
        bash_command=(
            "cd {{ params.root }} && "
            "python -u scripts/mortfm/10_run_baselines.py "
            "  --mode patient_longitudinal "
            "  --confirmed-dataset {{ params.root }}/data/lake/confirmed/mortfm_training_dataset "
            "  --split-manifest {{ params.root }}/data/lake/confirmed/split_manifest.json "
            "  --baselines all "
            "  --seeds 0 1 2"
        ),
        params={"root": ROOT},
        cwd=ROOT,
    )

    # ------------------------------------------------------------------
    # Stage 8: Evaluate Causal Evidence
    # Three-way evidence join (CRISPR + drug-target + Reactome) with
    # permutation-test enrichment on counterfactual edge effects.
    # ------------------------------------------------------------------
    evaluate_causal_evidence = BashOperator(
        task_id="evaluate_causal_evidence",
        bash_command=(
            "cd {{ params.root }} && "
            "python -u scripts/mortfm/08_eval_causal_evidence_v2.py "
            "  --edges {{ params.root }}/results/mortfm/causal_edge_effects.csv "
            "  --out-csv {{ params.root }}/results/mortfm/causal_edge_evidence_v2.csv "
            "  --out-summary {{ params.root }}/logs/mortfm/causal_evidence_v2_summary.json "
            "  --top-k 20"
        ),
        params={"root": ROOT},
        cwd=ROOT,
    )

    # ------------------------------------------------------------------
    # Stage 9: Strict Claim Gate
    # Final 12-level claim-gate revalidation. Fails the DAG if required
    # claims are blocked.
    # ------------------------------------------------------------------
    strict_claim_gate = BashOperator(
        task_id="strict_claim_gate",
        bash_command=(
            "cd {{ params.root }} && "
            "python -u scripts/mortfm/09_gate_revalidate.py "
            "  --out {{ params.root }}/logs/mortfm/v17_gate_revalidate.json"
        ),
        params={"root": ROOT},
        cwd=ROOT,
    )

    # ------------------------------------------------------------------
    # Dependency graph
    # ------------------------------------------------------------------

    # Linear data-lake stages
    (
        raw_landing
        >> create_raw_hive_tables
        >> cleanse_standardize
        >> curate_features
        >> confirm_analytical_dataset
        >> audit_leakage_temporal_validity
    )

    # Parallel GPU tasks after audit passes
    audit_leakage_temporal_validity >> train_mortfm
    audit_leakage_temporal_validity >> run_baselines

    # Evidence evaluation requires both training and baselines to complete
    [train_mortfm, run_baselines] >> evaluate_causal_evidence

    # Final claim gate after evidence evaluation
    evaluate_causal_evidence >> strict_claim_gate
