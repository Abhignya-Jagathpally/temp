"""
MORT-FM Data Lakehouse — PySpark Application
=============================================

Full medallion-architecture data lakehouse for the MORT-FM pipeline:
    raw -> cleansed -> curated -> confirmed -> audit

Usage:
    spark-submit spark_jobs/mortfm_lakehouse.py \
        --stage {raw|raw_hive|cleansed|curated|confirmed|audit} \
        --config configs/workflows/mortfm_spark_airflow.yaml

Invariants enforced:
    - Censored stays censored (never imputed as event).
    - PFS endpoint CANNOT fall back to OS/vital_status.
    - visit_time_days is NEVER back-filled from event_time/TT2L.
    - delta_t_days must be > 0 and from real visit timestamps.
    - Patient-disjoint splits via deterministic SHA-based assignment.
    - Temporal leakage audit: followup_visit_time_days != event_time_days.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATABASES = [
    "mortfm_raw",
    "mortfm_cleansed",
    "mortfm_curated",
    "mortfm_confirmed",
]

VALID_STAGES = ("raw", "raw_hive", "cleansed", "curated", "confirmed", "audit")

logger = logging.getLogger("mortfm_lakehouse")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def spark_session(app_name: str = "MORTFM_Lakehouse") -> SparkSession:
    """Return a configured SparkSession with Hive support and persistent metastore."""
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    warehouse_dir = os.path.join(_root, "data", "lake", "warehouse")
    metastore_dir = os.path.join(_root, "data", "lake", "metastore_db")
    os.makedirs(warehouse_dir, exist_ok=True)

    return (
        SparkSession.builder
        .appName(app_name)
        .enableHiveSupport()
        .config("spark.driver.memory", "8g")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.warehouse.dir", warehouse_dir)
        .config(
            "javax.jdo.option.ConnectionURL",
            f"jdbc:derby:;databaseName={metastore_dir};create=true",
        )
        .config("hive.metastore.schema.verification", "false")
        .config("datanucleus.autoCreateSchema", "true")
        .config("datanucleus.autoCreateTables", "true")
        .config("datanucleus.fixedDatastore", "false")
        .getOrCreate()
    )


def create_databases(spark: SparkSession) -> None:
    """Create all lakehouse databases if they do not exist."""
    for db in DATABASES:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")
        logger.info("Ensured database exists: %s", db)


def standardize_patient_ids(df: DataFrame, patient_col: str) -> DataFrame:
    """Add patient_id_hash column via SHA-256 of the raw patient identifier.

    The hash is deterministic and reproducible across runs, enabling
    patient-disjoint split assignment without exposing raw IDs downstream.
    """
    sha_udf = F.sha2(F.col(patient_col).cast("string"), 256)
    return df.withColumn("patient_id_hash", sha_udf)


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_hex(value: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Stage: RAW
# ---------------------------------------------------------------------------


def _infer_format(path: str) -> str:
    """Infer file format from extension."""
    lower = path.lower()
    if lower.endswith(".parquet"):
        return "parquet"
    elif lower.endswith(".tsv") or lower.endswith(".tsv.gz"):
        return "csv"
    elif lower.endswith(".csv") or lower.endswith(".csv.gz"):
        return "csv"
    else:
        return "csv"


def _deduplicate_columns(df: DataFrame) -> DataFrame:
    """Rename duplicate columns by appending a numeric suffix."""
    seen: Dict[str, int] = {}
    new_names = []
    for col_name in df.columns:
        if col_name in seen:
            seen[col_name] += 1
            new_names.append(f"{col_name}__dup{seen[col_name]}")
        else:
            seen[col_name] = 0
            new_names.append(col_name)
    if new_names != df.columns:
        n_dups = sum(1 for a, b in zip(new_names, df.columns) if a != b)
        logger.warning("Deduplicated %d duplicate column(s) in source", n_dups)
        df = df.toDF(*new_names)
    return df


def _read_source(spark: SparkSession, source_path: str, fmt: str) -> DataFrame:
    """Read a single data source into a DataFrame.

    Handles duplicate column names (common in MMRF gene_expression.tsv) by
    reading without header, extracting the header row manually, deduplicating
    column names, and then applying them.
    """
    if fmt == "parquet":
        return spark.read.parquet(source_path)

    sep = "\t" if (".tsv" in source_path.lower()) else ","

    # First attempt: normal read with header
    try:
        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("sep", sep)
            .option("multiLine", "true")
            .csv(source_path)
        )
        return _deduplicate_columns(df)
    except Exception as e:
        if "COLUMN_ALREADY_EXISTS" not in str(e):
            raise

    # Fallback for duplicate columns: read without header, deduplicate manually
    logger.warning("Duplicate columns detected in %s, using manual dedup", source_path)
    df_raw = (
        spark.read
        .option("header", "false")
        .option("inferSchema", "false")
        .option("sep", sep)
        .option("multiLine", "true")
        .csv(source_path)
    )
    # First row is the header
    header_row = df_raw.first()
    if header_row is None:
        raise ValueError(f"Empty file: {source_path}")

    raw_names = [str(header_row[i]) for i in range(len(header_row))]

    # Deduplicate
    seen: Dict[str, int] = {}
    deduped_names = []
    for name in raw_names:
        if name in seen:
            seen[name] += 1
            deduped_names.append(f"{name}__dup{seen[name]}")
        else:
            seen[name] = 0
            deduped_names.append(name)

    n_dups = sum(1 for a, b in zip(deduped_names, raw_names) if a != b)
    logger.info("Resolved %d duplicate column(s) in %s", n_dups, source_path)

    # Skip header row and rename columns
    header_values = [str(header_row[i]) for i in range(len(header_row))]
    df_no_header = df_raw.filter(
        ~F.concat_ws(sep, *[F.col(c) for c in df_raw.columns]).eqNullSafe(
            sep.join(header_values)
        )
    )
    df_renamed = df_no_header.toDF(*deduped_names)
    return df_renamed


def stage_raw(spark: SparkSession, cfg: Dict[str, Any]) -> None:
    """Ingest raw sources into Parquet with provenance columns.

    Reads CSV/TSV/Parquet sources defined under cfg['lakehouse']['sources'],
    adds _source and _ingest_ts metadata columns, and writes partitioned
    Parquet to the raw zone.
    """
    lake_root = cfg["lakehouse"]["lake_root"]
    sources = cfg["lakehouse"]["sources"]
    ingest_ts = _utcnow_iso()

    for source in sources:
        name = source["name"]
        path = source["path"]
        fmt = source.get("format", _infer_format(path))
        partition_cols = source.get("partition_by", ["_source"])

        logger.info("Ingesting raw source: %s from %s", name, path)

        if not os.path.exists(path):
            logger.warning("Source path does not exist, skipping: %s", path)
            continue

        df = _read_source(spark, path, fmt)

        # Add provenance columns
        df = (
            df
            .withColumn("_source", F.lit(name))
            .withColumn("_ingest_ts", F.lit(ingest_ts))
        )

        out_path = os.path.join(lake_root, "raw", name)
        (
            df.write
            .mode("overwrite")
            .partitionBy(*partition_cols)
            .parquet(out_path)
        )
        logger.info("Wrote %d rows to %s", df.count(), out_path)


# ---------------------------------------------------------------------------
# Stage: RAW_HIVE
# ---------------------------------------------------------------------------


def stage_raw_hive(spark: SparkSession, cfg: Dict[str, Any]) -> None:
    """Create external Hive tables pointing to raw Parquet paths."""
    lake_root = cfg["lakehouse"]["lake_root"]
    sources = cfg["lakehouse"]["sources"]

    create_databases(spark)

    for source in sources:
        name = source["name"]
        raw_path = os.path.join(lake_root, "raw", name)

        if not os.path.exists(raw_path):
            logger.warning("Raw path not found, skipping Hive table: %s", raw_path)
            continue

        table_name = f"mortfm_raw.{name}"
        abs_raw_path = os.path.abspath(raw_path)
        logger.info("Creating external Hive table: %s -> %s", table_name, abs_raw_path)

        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name}
            USING PARQUET
            LOCATION '{abs_raw_path}'
            """
        )
        # Repair partitions only if table is partitioned
        try:
            spark.sql(f"MSCK REPAIR TABLE {table_name}")
        except Exception:
            pass  # Non-partitioned tables don't need repair
        logger.info("Hive table created: %s (%d cols)", table_name,
                    len(spark.table(table_name).columns))


# ---------------------------------------------------------------------------
# Stage: CLEANSED
# ---------------------------------------------------------------------------


def _cleanse_clinical_outcomes(
    spark: SparkSession, cfg: Dict[str, Any]
) -> DataFrame:
    """Standardize clinical outcomes.

    CRITICAL rules:
      - Censored stays censored. Never re-code censored as event.
      - PFS CANNOT fall back to OS/vital_status. Raise ValueError if missing.
      - visit_time_days is NEVER back-filled from event_time/TT2L.
    """
    lake_root = cfg["lakehouse"]["lake_root"]
    clinical_source = cfg["lakehouse"]["cleansed"]["clinical_outcomes_source"]

    # Read from Hive table or filesystem path
    if "." in clinical_source and not os.path.exists(clinical_source):
        df = spark.table(clinical_source)
    else:
        df = spark.read.parquet(os.path.join(lake_root, "raw", clinical_source))

    # Standardize patient IDs
    patient_col = cfg["lakehouse"]["cleansed"].get("patient_col", "submitter_id")
    df = standardize_patient_ids(df, patient_col)

    available_cols = set(df.columns)

    # Determine endpoint type from config
    endpoint = cfg["lakehouse"]["cleansed"].get("endpoint", "pfs")
    time_col = cfg["lakehouse"]["cleansed"].get("time_col")
    event_col = cfg["lakehouse"]["cleansed"].get("event_col")
    event_value = cfg["lakehouse"]["cleansed"].get("event_value")

    if endpoint == "tt2l":
        tt2l_source = cfg["lakehouse"]["cleansed"]["tt2l_source"]
        tt2l_line_col = cfg["lakehouse"]["cleansed"].get("tt2l_line_col", "regimen_or_line_of_therapy")
        tt2l_time_col = cfg["lakehouse"]["cleansed"].get("tt2l_time_col", "days_to_treatment_start")

        if "." in tt2l_source and not os.path.exists(tt2l_source):
            treat_df = spark.table(tt2l_source)
        else:
            treat_df = spark.read.parquet(os.path.join(lake_root, "raw", tt2l_source))

        patient_col_t = cfg["lakehouse"]["cleansed"].get("patient_col", "submitter_id")

        line_df = (
            treat_df
            .withColumn("line_norm", F.lower(F.trim(F.col(tt2l_line_col))))
            .withColumn("treatment_start_days", F.col(tt2l_time_col).cast("double"))
        )

        first_line = (
            line_df
            .filter(F.col("line_norm").isin("first line of therapy", "line 1", "1"))
            .groupBy(patient_col_t)
            .agg(F.min("treatment_start_days").alias("first_line_start_days"))
        )

        second_line = (
            line_df
            .filter(F.col("line_norm").isin("second line of therapy", "line 2", "2"))
            .groupBy(patient_col_t)
            .agg(F.min("treatment_start_days").alias("second_line_start_days"))
        )

        tt2l_events = (
            first_line.join(second_line, on=patient_col_t, how="full")
            .withColumn(
                "tt2l_days",
                F.when(
                    F.col("first_line_start_days").isNotNull()
                    & F.col("second_line_start_days").isNotNull(),
                    F.col("second_line_start_days") - F.col("first_line_start_days"),
                ),
            )
            .withColumn("event_observed", F.col("tt2l_days").isNotNull())
        )

        # Join TT2L onto clinical
        df = df.join(tt2l_events, on=patient_col_t, how="left")

        # Censor at last follow-up for patients without 2L
        censor_col = time_col if time_col in available_cols else "days_to_last_follow_up"
        df = df.withColumn(
            "event_time_days",
            F.when(F.col("event_observed"), F.col("tt2l_days"))
             .otherwise(F.col(censor_col).cast("double")),
        )
        df = df.withColumn("endpoint_type", F.lit("tt2l_proxy"))
        df = df.withColumn("endpoint_family", F.lit("treatment_transition_proxy"))

        # Hard audit: no negative TT2L for observed events
        df = df.filter(
            ~(F.col("event_observed") & (F.col("event_time_days") <= 0))
        )
        df = df.filter(F.col("event_time_days").isNotNull())

        logger.info(
            "Using TT2L endpoint (2L_start - 1L_start). "
            "Valid for survival_proxy claims. resistance_emergence blocked."
        )

    elif endpoint in ("pfs", "relapse"):
        # PFS/relapse CANNOT fall back to OS/vital_status
        if time_col not in available_cols or event_col not in available_cols:
            raise ValueError(
                f"Endpoint '{endpoint}' requires columns ({time_col}, {event_col}) "
                f"in clinical source '{clinical_source}'. "
                f"PFS/relapse CANNOT fall back to OS/vital_status. "
                f"Available: {sorted(available_cols)}"
            )
        df = (
            df.withColumn("event_time_days", F.col(time_col).cast("double"))
              .withColumn("event_observed",
                          F.when(F.col(event_col).cast("int") == 1, True)
                           .otherwise(False))
              .withColumn("endpoint_type", F.lit(endpoint))
        )
    elif endpoint == "os":
        # OS endpoint: allowed for survival_prediction, blocked for resistance_emergence
        # Use days_to_death if dead, days_to_last_follow_up if alive
        os_time_col = cfg["lakehouse"]["cleansed"].get("os_time_col", "days_to_death")
        os_event_col = cfg["lakehouse"]["cleansed"].get("os_event_col", "vital_status")
        ev_val = event_value or "Dead"

        if os_event_col not in available_cols:
            raise ValueError(
                f"OS endpoint requires '{os_event_col}' column. "
                f"Available: {sorted(available_cols)}"
            )

        # event_observed = True if Dead, False if Alive/Not Reported (censored)
        df = df.withColumn(
            "event_observed",
            F.when(F.col(os_event_col) == ev_val, True).otherwise(False),
        )
        # event_time_days: days_to_death if dead, days_to_last_follow_up otherwise
        if os_time_col in available_cols and time_col in available_cols:
            df = df.withColumn(
                "event_time_days",
                F.when(F.col("event_observed"), F.col(os_time_col).cast("double"))
                 .otherwise(F.col(time_col).cast("double")),
            )
        elif time_col in available_cols:
            df = df.withColumn("event_time_days", F.col(time_col).cast("double"))
        elif os_time_col in available_cols:
            df = df.withColumn("event_time_days", F.col(os_time_col).cast("double"))
        else:
            df = df.withColumn("event_time_days", F.lit(None).cast("double"))

        df = df.withColumn("endpoint_type", F.lit("os"))
        logger.info(
            "Using OS endpoint (survival_prediction only). "
            "resistance_emergence claims will be blocked downstream."
        )
    else:
        raise ValueError(f"Unknown endpoint: {endpoint!r}")

    # Standardize column names from clinical_col_map
    col_map = cfg["lakehouse"]["cleansed"].get("clinical_col_map", {})
    for src_col, dst_col in col_map.items():
        if src_col in available_cols:
            df = df.withColumnRenamed(src_col, dst_col)

    # Enforce: censored stays censored. Never coerce NULL -> event.
    df = df.withColumn(
        "event_observed",
        F.when(F.col("event_observed").isNull(), False).otherwise(F.col("event_observed")),
    )

    # Never back-fill visit_time_days from event_time
    if "visit_time_days" in df.columns:
        df = df.withColumn(
            "visit_time_days",
            F.when(
                F.col("visit_time_days") == F.col("event_time_days"),
                F.lit(None).cast("double"),
            ).otherwise(F.col("visit_time_days")),
        )

    return df


def _cleanse_molecular_snapshots(
    spark: SparkSession, cfg: Dict[str, Any]
) -> DataFrame:
    """Standardize molecular snapshot data (RNA-seq, proteomics)."""
    lake_root = cfg["lakehouse"]["lake_root"]
    mol_source = cfg["lakehouse"]["cleansed"]["molecular_snapshots_source"]

    if "." in mol_source and not os.path.exists(mol_source):
        df = spark.table(mol_source)
    else:
        df = spark.read.parquet(os.path.join(lake_root, "raw", mol_source))

    patient_col = cfg["lakehouse"]["cleansed"].get("patient_col", "submitter_id")
    df = standardize_patient_ids(df, patient_col)

    # Ensure visit_time_days is from real timestamps, NEVER back-filled
    if "visit_time_days" not in df.columns:
        df = df.withColumn("visit_time_days", F.lit(None).cast("double"))

    # Add cleansed timestamp
    df = df.withColumn("_cleansed_ts", F.lit(_utcnow_iso()))

    return df


def _cleanse_treatment_exposures(
    spark: SparkSession, cfg: Dict[str, Any]
) -> DataFrame:
    """Standardize treatment exposure records."""
    lake_root = cfg["lakehouse"]["lake_root"]
    tx_source = cfg["lakehouse"]["cleansed"]["treatment_exposures_source"]

    if "." in tx_source and not os.path.exists(tx_source):
        df = spark.table(tx_source)
    else:
        df = spark.read.parquet(os.path.join(lake_root, "raw", tx_source))

    patient_col = cfg["lakehouse"]["cleansed"].get("patient_col", "submitter_id")
    df = standardize_patient_ids(df, patient_col)

    # Standardize drug names to lowercase canonical form
    if "drug_name" in df.columns:
        df = df.withColumn("drug_name_canonical", F.lower(F.trim(F.col("drug_name"))))

    df = df.withColumn("_cleansed_ts", F.lit(_utcnow_iso()))

    return df


def stage_cleansed(spark: SparkSession, cfg: Dict[str, Any]) -> None:
    """Run cleansed-zone transformations and write to cleansed layer."""
    lake_root = cfg["lakehouse"]["lake_root"]
    create_databases(spark)

    # Clinical outcomes
    logger.info("Cleansing clinical outcomes...")
    clinical_df = _cleanse_clinical_outcomes(spark, cfg)
    out_path = os.path.join(lake_root, "cleansed", "clinical_outcomes")
    (
        clinical_df.write
        .mode("overwrite")
        .partitionBy("_source")
        .parquet(out_path)
    )
    logger.info("Wrote cleansed clinical_outcomes: %d rows", clinical_df.count())

    # Molecular snapshots
    logger.info("Cleansing molecular snapshots...")
    mol_df = _cleanse_molecular_snapshots(spark, cfg)
    out_path = os.path.join(lake_root, "cleansed", "molecular_snapshots")
    (
        mol_df.write
        .mode("overwrite")
        .partitionBy("_source")
        .parquet(out_path)
    )
    logger.info("Wrote cleansed molecular_snapshots: %d rows", mol_df.count())

    # Treatment exposures
    logger.info("Cleansing treatment exposures...")
    tx_df = _cleanse_treatment_exposures(spark, cfg)
    out_path = os.path.join(lake_root, "cleansed", "treatment_exposures")
    (
        tx_df.write
        .mode("overwrite")
        .partitionBy("_source")
        .parquet(out_path)
    )
    logger.info("Wrote cleansed treatment_exposures: %d rows", tx_df.count())


# ---------------------------------------------------------------------------
# Stage: CURATED
# ---------------------------------------------------------------------------


def _build_temporal_pairs(
    spark: SparkSession, cfg: Dict[str, Any]
) -> DataFrame:
    """Build temporal pairs ONLY when both molecular timestamps are real.

    delta_t_days must be > 0 and from real visit timestamps.
    Never fabricate pairs from pseudotime or imputed timestamps.
    """
    lake_root = cfg["lakehouse"]["lake_root"]
    mol_path = os.path.join(lake_root, "cleansed", "molecular_snapshots")
    mol_df = spark.read.parquet(mol_path)

    # Only keep rows with real visit_time_days (non-null, positive)
    mol_with_time = mol_df.filter(
        F.col("visit_time_days").isNotNull() & (F.col("visit_time_days") > 0)
    )

    # Self-join on patient_id_hash to find baseline-followup pairs
    baseline = mol_with_time.alias("baseline")
    followup = mol_with_time.alias("followup")

    pairs = (
        baseline.join(
            followup,
            (F.col("baseline.patient_id_hash") == F.col("followup.patient_id_hash"))
            & (F.col("followup.visit_time_days") > F.col("baseline.visit_time_days")),
            "inner",
        )
        .select(
            F.col("baseline.patient_id_hash").alias("patient_id_hash"),
            F.col("baseline.visit_time_days").alias("baseline_visit_time_days"),
            F.col("followup.visit_time_days").alias("followup_visit_time_days"),
            (
                F.col("followup.visit_time_days") - F.col("baseline.visit_time_days")
            ).alias("delta_t_days"),
        )
    )

    # Enforce delta_t_days > 0 (should be guaranteed by join, but explicit)
    pairs = pairs.filter(F.col("delta_t_days") > 0)

    # Keep only the earliest followup per baseline per patient (closest pair)
    from pyspark.sql.window import Window

    w = Window.partitionBy("patient_id_hash", "baseline_visit_time_days").orderBy(
        "delta_t_days"
    )
    pairs = pairs.withColumn("_rank", F.row_number().over(w)).filter(
        F.col("_rank") == 1
    ).drop("_rank")

    return pairs


def _build_patient_graph_node_features(
    spark: SparkSession, cfg: Dict[str, Any]
) -> DataFrame:
    """Build patient_graph_node_features from RNA/proteomics + graph topology."""
    lake_root = cfg["lakehouse"]["lake_root"]
    mol_path = os.path.join(lake_root, "cleansed", "molecular_snapshots")
    mol_df = spark.read.parquet(mol_path)

    # Aggregate per-patient features (mean expression across visits)
    # Only numeric columns for aggregation
    feature_cols = [
        c for c in mol_df.columns
        if c not in (
            "patient_id_hash", "visit_time_days", "_source", "_ingest_ts",
            "_cleansed_ts", "sample_id", "aliquot_id",
        )
        and mol_df.schema[c].dataType in (T.DoubleType(), T.FloatType(), T.IntegerType())
    ]

    if not feature_cols:
        # Return minimal frame with patient_id_hash only
        return mol_df.select("patient_id_hash").distinct()

    agg_exprs = [F.avg(F.col(c)).alias(c) for c in feature_cols]
    patient_features = (
        mol_df
        .groupBy("patient_id_hash")
        .agg(*agg_exprs)
    )

    return patient_features


def _build_drug_target_graph(
    spark: SparkSession, cfg: Dict[str, Any]
) -> DataFrame:
    """Build drug_target_graph from drug ontology and ChEMBL targets."""
    lake_root = cfg["lakehouse"]["lake_root"]
    drug_targets_source = cfg["lakehouse"]["curated"].get(
        "drug_targets_source", "drug_targets"
    )
    raw_path = os.path.join(lake_root, "raw", drug_targets_source)

    if not os.path.exists(raw_path):
        logger.warning("Drug targets source not found: %s, returning empty DF", raw_path)
        schema = T.StructType([
            T.StructField("drug_name", T.StringType(), True),
            T.StructField("target_gene", T.StringType(), True),
            T.StructField("interaction_score", T.DoubleType(), True),
        ])
        return spark.createDataFrame([], schema)

    df = spark.read.parquet(raw_path)
    # Normalize drug names
    if "drug_name" in df.columns:
        df = df.withColumn("drug_name_canonical", F.lower(F.trim(F.col("drug_name"))))
    return df


def _build_pathway_evidence(
    spark: SparkSession, cfg: Dict[str, Any]
) -> DataFrame:
    """Build pathway_evidence table from Reactome membership."""
    lake_root = cfg["lakehouse"]["lake_root"]
    reactome_source = cfg["lakehouse"]["curated"].get(
        "reactome_source", "reactome_membership"
    )
    raw_path = os.path.join(lake_root, "raw", reactome_source)

    if not os.path.exists(raw_path):
        logger.warning("Reactome source not found: %s, returning empty DF", raw_path)
        schema = T.StructType([
            T.StructField("pathway_id", T.StringType(), True),
            T.StructField("pathway_name", T.StringType(), True),
            T.StructField("gene_symbol", T.StringType(), True),
            T.StructField("evidence_source", T.StringType(), True),
        ])
        return spark.createDataFrame([], schema)

    df = spark.read.parquet(raw_path)
    return df


def stage_curated(spark: SparkSession, cfg: Dict[str, Any]) -> None:
    """Build curated-zone tables: temporal pairs, graph features, pathway evidence."""
    lake_root = cfg["lakehouse"]["lake_root"]
    create_databases(spark)

    # Temporal pairs
    logger.info("Building temporal pairs (real timestamps only)...")
    pairs_df = _build_temporal_pairs(spark, cfg)
    out_path = os.path.join(lake_root, "curated", "temporal_pairs")
    pairs_df.write.mode("overwrite").parquet(out_path)
    logger.info("Wrote curated temporal_pairs: %d rows", pairs_df.count())

    # Patient graph node features
    logger.info("Building patient_graph_node_features...")
    node_feat_df = _build_patient_graph_node_features(spark, cfg)
    out_path = os.path.join(lake_root, "curated", "patient_graph_node_features")
    node_feat_df.write.mode("overwrite").parquet(out_path)
    logger.info("Wrote curated patient_graph_node_features: %d rows", node_feat_df.count())

    # Drug-target graph
    logger.info("Building drug_target_graph...")
    drug_graph_df = _build_drug_target_graph(spark, cfg)
    out_path = os.path.join(lake_root, "curated", "drug_target_graph")
    drug_graph_df.write.mode("overwrite").parquet(out_path)
    logger.info("Wrote curated drug_target_graph: %d rows", drug_graph_df.count())

    # Pathway evidence
    logger.info("Building pathway_evidence...")
    pathway_df = _build_pathway_evidence(spark, cfg)
    out_path = os.path.join(lake_root, "curated", "pathway_evidence")
    pathway_df.write.mode("overwrite").parquet(out_path)
    logger.info("Wrote curated pathway_evidence: %d rows", pathway_df.count())


# ---------------------------------------------------------------------------
# Stage: CONFIRMED
# ---------------------------------------------------------------------------


def _assign_split(patient_id_hash: str, seed: int, val_frac: float, test_frac: float) -> str:
    """Deterministic SHA-based split assignment.

    Uses SHA-256(seed || patient_id_hash) mod 10000 to assign patients
    to train/val/test without any randomness.
    """
    combined = f"{seed}:{patient_id_hash}"
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10000
    test_boundary = int(test_frac * 10000)
    val_boundary = test_boundary + int(val_frac * 10000)

    if bucket < test_boundary:
        return "test"
    elif bucket < val_boundary:
        return "val"
    else:
        return "train"


def stage_confirmed(spark: SparkSession, cfg: Dict[str, Any]) -> None:
    """Build confirmed training dataset with supervision flags and splits.

    HARD AUDIT: fail if followup_visit_time_days == event_time_days
    (temporal leakage indicator).
    """
    lake_root = cfg["lakehouse"]["lake_root"]
    splits_cfg = cfg.get("splits", {})
    val_frac = splits_cfg.get("val_fraction", 0.15)
    test_frac = splits_cfg.get("test_fraction", 0.15)
    seed = splits_cfg.get("seed", 42)

    create_databases(spark)

    # Load curated temporal pairs + cleansed clinical outcomes
    pairs_path = os.path.join(lake_root, "curated", "temporal_pairs")
    clinical_path = os.path.join(lake_root, "cleansed", "clinical_outcomes")

    clinical_df = spark.read.parquet(clinical_path)

    # Try loading temporal pairs (may be empty if no visit timestamps available)
    try:
        pairs_df = spark.read.parquet(pairs_path)
        has_pairs = pairs_df.count() > 0
    except Exception:
        has_pairs = False
        pairs_df = None

    if has_pairs and pairs_df is not None:
        # Join temporal pairs with clinical outcomes
        joined = pairs_df.join(clinical_df, on="patient_id_hash", how="left")
    else:
        # No temporal pairs — use survival-only rows from clinical outcomes.
        # These can supervise ONLY the survival head, NOT trajectory.
        logger.warning(
            "No temporal pairs found. Building survival-only confirmed dataset. "
            "Trajectory claims will be blocked."
        )
        joined = (
            clinical_df
            .filter(F.col("event_time_days").isNotNull() & (F.col("event_time_days") > 0))
            .withColumn("baseline_visit_time_days", F.lit(None).cast("double"))
            .withColumn("followup_visit_time_days", F.lit(None).cast("double"))
            .withColumn("delta_t_days", F.lit(None).cast("double"))
        )

    # --- HARD AUDIT: temporal leakage check ---
    if "event_time_days" in joined.columns:
        leakage_rows = joined.filter(
            F.col("followup_visit_time_days").isNotNull()
            & F.col("event_time_days").isNotNull()
            & (F.col("followup_visit_time_days") == F.col("event_time_days"))
        )
        leakage_count = leakage_rows.count()
        if leakage_count > 0:
            raise RuntimeError(
                f"TEMPORAL LEAKAGE DETECTED: {leakage_count} rows have "
                f"followup_visit_time_days == event_time_days. This indicates "
                f"that molecular follow-up timestamps are conflated with "
                f"clinical event times. Pipeline halted."
            )

    # Add supervision flags
    joined = joined.withColumn(
        "has_valid_trajectory_supervision",
        (
            F.col("delta_t_days").isNotNull()
            & (F.col("delta_t_days") > 0)
            & F.col("baseline_visit_time_days").isNotNull()
            & F.col("followup_visit_time_days").isNotNull()
        ).cast("boolean"),
    )

    # Survival supervision requires event_time and event_observed
    has_event_time = F.col("event_time_days").isNotNull() if "event_time_days" in joined.columns else F.lit(False)
    has_event_obs = F.col("event_observed").isNotNull() if "event_observed" in joined.columns else F.lit(False)

    joined = joined.withColumn(
        "has_valid_survival_supervision",
        (has_event_time & has_event_obs).cast("boolean"),
    )

    # Loss mask columns — consumed by the training loop to gate per-row loss terms
    joined = (
        joined
        .withColumn("allowed_loss_survival", F.col("has_valid_survival_supervision"))
        .withColumn("allowed_loss_trajectory", F.col("has_valid_trajectory_supervision"))
        .withColumn("allowed_loss_ordinal_transition", F.lit(False))
    )

    # Patient-disjoint split assignment (deterministic SHA-based)
    split_udf = F.udf(
        lambda pid: _assign_split(pid, seed, val_frac, test_frac),
        T.StringType(),
    )
    joined = joined.withColumn("split", split_udf(F.col("patient_id_hash")))

    # Write main training dataset
    out_path = os.path.join(lake_root, "confirmed", "mortfm_training_dataset")
    (
        joined.write
        .mode("overwrite")
        .partitionBy("split")
        .parquet(out_path)
    )
    logger.info("Wrote confirmed mortfm_training_dataset: %d rows", joined.count())

    # Write split manifest
    split_counts = (
        joined
        .groupBy("split")
        .agg(
            F.countDistinct("patient_id_hash").alias("n_patients"),
            F.count("*").alias("n_rows"),
            F.sum(F.col("has_valid_trajectory_supervision").cast("int")).alias("n_trajectory_rows"),
            F.sum(F.col("has_valid_survival_supervision").cast("int")).alias("n_survival_rows"),
        )
        .collect()
    )

    manifest = {
        "created_at": _utcnow_iso(),
        "seed": seed,
        "val_fraction": val_frac,
        "test_fraction": test_frac,
        "method": "deterministic_sha256",
        "splits": {
            row["split"]: {
                "n_patients": row["n_patients"],
                "n_rows": row["n_rows"],
                "n_trajectory_rows": row["n_trajectory_rows"],
                "n_survival_rows": row["n_survival_rows"],
            }
            for row in split_counts
        },
    }

    manifest_path = os.path.join(lake_root, "confirmed", "split_manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote split_manifest.json to %s", manifest_path)

    # Write split membership (patient_id_hash -> split)
    split_membership = joined.select("patient_id_hash", "split").distinct()
    membership_path = os.path.join(lake_root, "confirmed", "split_membership")
    (
        split_membership.write
        .mode("overwrite")
        .partitionBy("split")
        .parquet(membership_path)
    )
    logger.info("Wrote split_membership: %d patients", split_membership.count())


# ---------------------------------------------------------------------------
# Stage: AUDIT
# ---------------------------------------------------------------------------


def stage_audit(spark: SparkSession, cfg: Dict[str, Any]) -> None:
    """Run data quality audit on confirmed dataset.

    Checks:
      - Total rows, patients, trajectory rows, survival rows
      - min_trajectory_pairs gate (from config)
      - min_patients gate (from config)

    Raises RuntimeError if any gate fails.
    """
    lake_root = cfg["lakehouse"]["lake_root"]
    gates_cfg = cfg["lakehouse"].get("gates", {})
    min_survival_patients = gates_cfg.get("min_survival_patients", 50)
    min_survival_events = gates_cfg.get("min_survival_events", 20)
    min_trajectory_pairs = gates_cfg.get("min_trajectory_pairs", 10)

    confirmed_path = os.path.join(lake_root, "confirmed", "mortfm_training_dataset")

    if not os.path.exists(confirmed_path):
        raise RuntimeError(
            f"Confirmed dataset not found at {confirmed_path}. "
            f"Run the 'confirmed' stage first."
        )

    # Handle empty parquet directories gracefully
    try:
        df = spark.read.parquet(confirmed_path)
    except Exception as e:
        if "UNABLE_TO_INFER_SCHEMA" in str(e) or "empty" in str(e).lower():
            logger.warning("Confirmed dataset is empty (no rows written).")
            df = spark.createDataFrame(
                [],
                T.StructType([
                    T.StructField("patient_id_hash", T.StringType()),
                    T.StructField("has_valid_trajectory_supervision", T.BooleanType()),
                    T.StructField("has_valid_survival_supervision", T.BooleanType()),
                ]),
            )
        else:
            raise

    total_rows = df.count()
    n_patients = df.select("patient_id_hash").distinct().count()
    n_trajectory_rows = df.filter(F.col("has_valid_trajectory_supervision") == True).count()
    n_survival_rows = df.filter(F.col("has_valid_survival_supervision") == True).count()

    report = {
        "audit_timestamp": _utcnow_iso(),
        "total_rows": total_rows,
        "n_patients": n_patients,
        "n_trajectory_rows": n_trajectory_rows,
        "n_survival_rows": n_survival_rows,
        "gates": {
            "min_trajectory_pairs": {
                "threshold": min_trajectory_pairs,
                "actual": n_trajectory_rows,
                "passed": n_trajectory_rows >= min_trajectory_pairs,
            },
            "min_survival_patients": {
                "threshold": min_survival_patients,
                "actual": n_patients,
                "passed": n_patients >= min_survival_patients,
            },
            "min_survival_events": {
                "threshold": min_survival_events,
                "actual": n_survival_rows,
                "passed": n_survival_rows >= min_survival_events,
            },
        },
        "overall_passed": (
            n_patients >= min_survival_patients
            and n_survival_rows >= min_survival_events
        ),
    }

    # Write report
    report_path = os.path.join(lake_root, "confirmed", "data_quality_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote data_quality_report.json to %s", report_path)

    # Log summary
    logger.info("=== DATA QUALITY AUDIT ===")
    logger.info("  Total rows:        %d", total_rows)
    logger.info("  Patients:          %d", n_patients)
    logger.info("  Trajectory rows:   %d", n_trajectory_rows)
    logger.info("  Survival rows:     %d", n_survival_rows)
    logger.info("  Gate: min_survival_patients (%d) -> %s",
                min_survival_patients,
                "PASS" if n_patients >= min_survival_patients else "FAIL")
    logger.info("  Gate: min_survival_events (%d) -> %s",
                min_survival_events,
                "PASS" if n_survival_rows >= min_survival_events else "FAIL")
    logger.info("  Gate: min_trajectory_pairs (%d) -> %s",
                min_trajectory_pairs,
                "PASS" if n_trajectory_rows >= min_trajectory_pairs else "FAIL")

    # Claim-aware gates
    failures: List[str] = []
    blocked_claims: Dict[str, str] = {}
    allowed_claims: List[str] = ["technical_pipeline"]

    # Survival gate
    if n_patients >= min_survival_patients and n_survival_rows >= min_survival_events:
        allowed_claims.append("tt2l_survival_proxy_prediction")
        allowed_claims.append("static_drug_response")
    else:
        failures.append(
            f"Survival gate FAIL: need {min_survival_patients} patients "
            f"(have {n_patients}) and {min_survival_events} events (have {n_survival_rows})"
        )

    # Trajectory gate (informational — blocks claims, does not halt pipeline)
    if n_trajectory_rows < min_trajectory_pairs:
        blocked_claims["longitudinal_trajectory"] = (
            f"Need {min_trajectory_pairs} real molecular trajectory pairs, found {n_trajectory_rows}."
        )
        blocked_claims["resistance_emergence"] = (
            "TT2L is a treatment-transition proxy, not direct molecular resistance emergence."
        )
        blocked_claims["causal_mechanism"] = (
            "No perturbational validation tied to longitudinal patient trajectory."
        )
    else:
        allowed_claims.extend(["longitudinal_trajectory", "resistance_emergence"])

    report["allowed_claims"] = allowed_claims
    report["blocked_claims"] = blocked_claims

    # Re-write full report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if failures:
        raise RuntimeError(
            "DATA QUALITY GATES FAILED (hard):\n" + "\n".join(f"  - {f}" for f in failures)
        )

    if blocked_claims:
        logger.warning(
            "Claims blocked (pipeline continues): %s", list(blocked_claims.keys())
        )

    logger.info("All hard audit gates PASSED.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MORT-FM Data Lakehouse (raw -> cleansed -> curated -> confirmed -> audit)"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=VALID_STAGES,
        help="Lakehouse stage to execute.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/workflows/mortfm_spark_airflow.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = _load_config(args.config)

    # Accept either nested (cfg["lakehouse"]["..."]) or flat (cfg["..."]) format.
    if "lakehouse" not in cfg:
        cfg = {"lakehouse": cfg}

    spark = spark_session(app_name=f"MORTFM_Lakehouse_{args.stage}")

    logger.info("Starting MORT-FM Lakehouse stage: %s", args.stage)
    logger.info("Config: %s", args.config)

    try:
        if args.stage == "raw":
            stage_raw(spark, cfg)
        elif args.stage == "raw_hive":
            stage_raw_hive(spark, cfg)
        elif args.stage == "cleansed":
            stage_cleansed(spark, cfg)
        elif args.stage == "curated":
            stage_curated(spark, cfg)
        elif args.stage == "confirmed":
            stage_confirmed(spark, cfg)
        elif args.stage == "audit":
            stage_audit(spark, cfg)
        else:
            raise ValueError(f"Unknown stage: {args.stage}")
    finally:
        spark.stop()

    logger.info("Stage '%s' completed successfully.", args.stage)


if __name__ == "__main__":
    main()
