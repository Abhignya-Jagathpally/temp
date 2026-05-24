"""
resistancemap/data/spark_longitudinal.py
========================================
PySpark-backed alternatives for the heaviest longitudinal data operations.

Gated behind ``--engine spark`` (default stays pandas). Every function
accepts a ``SparkSession`` as its first argument so the caller controls
cluster configuration. Output is always partitioned Parquet so downstream
code (pandas or Spark) can read individual partitions cheaply.

Usage from ``scripts/mortfm_build_longitudinal_dataset.py``::

    python scripts/mortfm_build_longitudinal_dataset.py --engine spark

If ``pyspark`` is not installed, each function raises ``ImportError``
with a helpful install hint.

Canonical column schema for ``longitudinal_panel.parquet``
----------------------------------------------------------
patient_id_hash, cohort, visit_id, visit_time_days, treatment_line,
drug_exposure_start, drug_exposure_end, modality_available_rna_bulk,
modality_available_scrna, modality_available_proteomics,
modality_available_atac, modality_available_methylation,
modality_available_cite_seq, event_type, event_time_days,
event_observed, resistance_state_label, baseline_snapshot_uri,
followup_snapshot_uri
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy PySpark import helper
# ---------------------------------------------------------------------------

_INSTALL_MSG = (
    "PySpark is required for --engine spark. "
    "Install with:  pip install 'pyspark>=3.4'  or  conda install pyspark"
)


def _require_pyspark():
    """Import and return core PySpark modules, or raise ImportError."""
    try:
        from pyspark.sql import SparkSession, DataFrame  # noqa: F811
        import pyspark.sql.functions as F  # noqa: F811
        import pyspark.sql.types as T  # noqa: F811
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc
    return SparkSession, DataFrame, F, T


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_MODALITY_BOOLEANS = [
    "modality_available_rna_bulk",
    "modality_available_scrna",
    "modality_available_proteomics",
    "modality_available_atac",
    "modality_available_methylation",
    "modality_available_cite_seq",
]

_PANEL_COLUMNS = [
    "patient_id_hash",
    "cohort",
    "visit_id",
    "visit_time_days",
    "treatment_line",
    "drug_exposure_start",
    "drug_exposure_end",
    *_MODALITY_BOOLEANS,
    "event_type",
    "event_time_days",
    "event_observed",
    "resistance_state_label",
    "baseline_snapshot_uri",
    "followup_snapshot_uri",
]


def _longitudinal_panel_schema():
    """Return the StructType for longitudinal_panel.parquet."""
    _, _, _, T = _require_pyspark()
    return T.StructType([
        T.StructField("patient_id_hash", T.StringType(), False),
        T.StructField("cohort", T.StringType(), False),
        T.StructField("visit_id", T.StringType(), False),
        T.StructField("visit_time_days", T.DoubleType(), True),
        T.StructField("treatment_line", T.IntegerType(), True),
        T.StructField("drug_exposure_start", T.DoubleType(), True),
        T.StructField("drug_exposure_end", T.DoubleType(), True),
        T.StructField("modality_available_rna_bulk", T.BooleanType(), True),
        T.StructField("modality_available_scrna", T.BooleanType(), True),
        T.StructField("modality_available_proteomics", T.BooleanType(), True),
        T.StructField("modality_available_atac", T.BooleanType(), True),
        T.StructField("modality_available_methylation", T.BooleanType(), True),
        T.StructField("modality_available_cite_seq", T.BooleanType(), True),
        T.StructField("event_type", T.StringType(), True),
        T.StructField("event_time_days", T.DoubleType(), True),
        T.StructField("event_observed", T.BooleanType(), True),
        T.StructField("resistance_state_label", T.StringType(), True),
        T.StructField("baseline_snapshot_uri", T.StringType(), True),
        T.StructField("followup_snapshot_uri", T.StringType(), True),
    ])


def _hash_patient_id(patient_id: str) -> str:
    """SHA-256 hash of patient_id for de-identification in Spark outputs."""
    return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:16]


# ===================================================================
# 1. build_longitudinal_panel_spark
# ===================================================================

def build_longitudinal_panel_spark(
    spark: Any,
    data_dir: str,
    out_path: str,
    *,
    partition_cols: Optional[List[str]] = None,
) -> str:
    """Build the canonical ``longitudinal_panel.parquet`` using PySpark.

    Reads raw cohort tables from *data_dir* (paired-patient TSVs,
    outcomes TSVs, sample manifests) and assembles the wide panel that
    downstream trainers and audit gates consume.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    data_dir : str
        Root directory containing ``mmrf_paired_patients.tsv``,
        ``mmrf_outcomes_treatment.tsv``, and sample manifests.
    out_path : str
        Destination path for the partitioned Parquet output.
    partition_cols : list[str], optional
        Partition columns (default ``["cohort"]``).

    Returns
    -------
    str
        The *out_path* written to.
    """
    _, DataFrame, F, T = _require_pyspark()

    if partition_cols is None:
        partition_cols = ["cohort"]

    data_root = Path(data_dir)

    # --- Load paired-patient table ----------------------------------------
    paired_path = data_root / "mmrf_paired_patients.tsv"
    if not paired_path.exists():
        raise FileNotFoundError(
            f"MMRF paired-patient table not found at {paired_path}"
        )
    paired_df = spark.read.csv(
        str(paired_path), sep="\t", header=True, inferSchema=True,
    )
    logger.info("Loaded paired-patient table: %d rows", paired_df.count())

    # --- Load outcomes table (if available) --------------------------------
    outcomes_path = data_root / "mmrf_outcomes_treatment.tsv"
    outcomes_df = None
    if outcomes_path.exists():
        outcomes_df = spark.read.csv(
            str(outcomes_path), sep="\t", header=True, inferSchema=True,
        )
        logger.info("Loaded outcomes table: %d rows", outcomes_df.count())

    # --- Hash patient IDs for de-identification ----------------------------
    hash_udf = F.udf(_hash_patient_id, T.StringType())

    # --- Build the panel rows ---------------------------------------------
    # Start from paired patients: each row becomes one visit-pair record.
    panel = paired_df.withColumn(
        "patient_id_hash", hash_udf(F.col("patient_id").cast("string"))
    ).withColumn(
        "cohort", F.lit("MMRF")
    ).withColumn(
        "visit_id",
        F.concat_ws(
            "_",
            F.col("patient_id").cast("string"),
            F.coalesce(F.col("t0_aliquot").cast("string"), F.lit("unknown")),
        ),
    )

    # visit_time_days: MMRF timepoints are enum strings, not real days.
    # Back-fill from outcomes when a 2L event exists.
    if outcomes_df is not None and "tt2L_days" in outcomes_df.columns:
        outcomes_slim = outcomes_df.select(
            F.col("submitter_id").alias("_oc_pid"),
            F.col("tt2L_days").cast("double").alias("visit_time_days"),
            F.col("had_2L").alias("_had_2L"),
        ).dropDuplicates(["_oc_pid"])

        panel = panel.join(
            outcomes_slim,
            panel["patient_id"].cast("string") == outcomes_slim["_oc_pid"],
            "left",
        ).drop("_oc_pid")

        # If no 2L event, visit_time_days stays null.
        panel = panel.withColumn(
            "visit_time_days",
            F.when(
                F.col("_had_2L").cast("boolean"), F.col("visit_time_days")
            ).otherwise(F.lit(None).cast("double")),
        ).drop("_had_2L")
    else:
        panel = panel.withColumn(
            "visit_time_days", F.lit(None).cast("double")
        )

    # treatment_line: best-effort from bort_1L column.
    if "bort_1L" in [f.name for f in paired_df.schema.fields]:
        panel = panel.withColumn(
            "treatment_line",
            F.when(F.col("bort_1L").cast("boolean"), F.lit(1)).otherwise(
                F.lit(None).cast("int")
            ),
        )
    else:
        panel = panel.withColumn(
            "treatment_line", F.lit(None).cast("int")
        )

    # Drug exposure windows: from outcomes if available.
    if outcomes_df is not None and "trtstdy" in outcomes_df.columns:
        exposure_slim = outcomes_df.select(
            F.col("submitter_id").alias("_ex_pid"),
            F.col("trtstdy").cast("double").alias("drug_exposure_start"),
            F.col("trtendy").cast("double").alias("drug_exposure_end"),
        ).dropDuplicates(["_ex_pid"])
        panel = panel.join(
            exposure_slim,
            panel["patient_id"].cast("string") == exposure_slim["_ex_pid"],
            "left",
        ).drop("_ex_pid")
    else:
        panel = panel.withColumn(
            "drug_exposure_start", F.lit(None).cast("double")
        ).withColumn(
            "drug_exposure_end", F.lit(None).cast("double")
        )

    # Modality booleans: derived from which aliquot columns are non-null.
    # MMRF paired tables have rna_bulk by definition; others default to False.
    panel = panel.withColumn(
        "modality_available_rna_bulk",
        F.col("t0_aliquot").isNotNull() & F.col("t1_aliquot").isNotNull(),
    )
    for mod in _MODALITY_BOOLEANS[1:]:
        # Check if a column matching the modality name exists in the source.
        mod_short = mod.replace("modality_available_", "")
        if mod_short in [f.name for f in paired_df.schema.fields]:
            panel = panel.withColumn(mod, F.col(mod_short).cast("boolean"))
        else:
            panel = panel.withColumn(mod, F.lit(False))

    # Event columns from outcomes.
    if outcomes_df is not None:
        event_slim = outcomes_df.select(
            F.col("submitter_id").alias("_ev_pid"),
            F.lit("progression").alias("event_type"),
            F.col("tt2L_days").cast("double").alias("event_time_days"),
            F.col("had_2L").cast("boolean").alias("event_observed"),
        ).dropDuplicates(["_ev_pid"])
        panel = panel.join(
            event_slim,
            panel["patient_id"].cast("string") == event_slim["_ev_pid"],
            "left",
        ).drop("_ev_pid")
    else:
        panel = panel.withColumn(
            "event_type", F.lit(None).cast("string")
        ).withColumn(
            "event_time_days", F.lit(None).cast("double")
        ).withColumn(
            "event_observed", F.lit(None).cast("boolean")
        )

    # Resistance state label: derive from event.
    panel = panel.withColumn(
        "resistance_state_label",
        F.when(
            F.col("event_observed") == True,  # noqa: E712
            F.lit("resistant"),
        ).when(
            F.col("event_observed") == False,  # noqa: E712
            F.lit("sensitive_or_censored"),
        ).otherwise(F.lit(None)),
    )

    # Snapshot URIs: point to baseline and followup aliquot paths.
    panel = panel.withColumn(
        "baseline_snapshot_uri",
        F.concat(F.lit("data/processed/snapshots/"), F.col("t0_aliquot").cast("string")),
    ).withColumn(
        "followup_snapshot_uri",
        F.concat(F.lit("data/processed/snapshots/"), F.col("t1_aliquot").cast("string")),
    )

    # Select only canonical columns (dropping intermediate join artifacts).
    final_cols = [c for c in _PANEL_COLUMNS if c in panel.columns]
    panel = panel.select(final_cols)

    # --- Write partitioned Parquet ----------------------------------------
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    panel.write.mode("overwrite").partitionBy(*partition_cols).parquet(out_path)
    n_rows = panel.count()
    logger.info(
        "build_longitudinal_panel_spark: wrote %d rows -> %s (partitioned by %s)",
        n_rows, out_path, partition_cols,
    )
    return out_path


# ===================================================================
# 2. audit_leakage_spark
# ===================================================================

def audit_leakage_spark(
    spark: Any,
    panel_path: str,
    split_manifest_path: str,
) -> dict:
    """Anti-join train/test patients and verify temporal ordering.

    Checks performed:

    1. **Patient overlap** -- No ``patient_id_hash`` appears in both
       train and test partitions (anti-join must be empty).
    2. **Future covariate leakage** -- For each patient, no covariate
       timestamp in the test window appears before that patient's
       latest train-window timestamp.
    3. **Temporal ordering** -- Within every patient, ``visit_time_days``
       is weakly monotone across visit records.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
    panel_path : str
        Path to the ``longitudinal_panel.parquet`` directory.
    split_manifest_path : str
        Path to a JSON split manifest with ``train_patient_hashes``
        and ``test_patient_hashes`` lists.

    Returns
    -------
    dict
        ``{"patient_overlap": [...], "future_leaks": [...],
        "temporal_violations": [...], "pass": bool}``
    """
    _, DataFrame, F, T = _require_pyspark()
    import json as _json

    panel = spark.read.parquet(panel_path)

    # Load split manifest.
    manifest_p = Path(split_manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_p}")
    with open(manifest_p) as fh:
        manifest = _json.load(fh)

    train_hashes = set(manifest.get("train_patient_hashes", []))
    test_hashes = set(manifest.get("test_patient_hashes", []))

    # ------------------------------------------------------------------
    # Check 1: Patient overlap via anti-join
    # ------------------------------------------------------------------
    train_ids = spark.createDataFrame(
        [(h,) for h in train_hashes], schema=["patient_id_hash"]
    )
    test_ids = spark.createDataFrame(
        [(h,) for h in test_hashes], schema=["patient_id_hash"]
    )
    overlap_df = train_ids.join(test_ids, on="patient_id_hash", how="inner")
    overlap_hashes = [
        row.patient_id_hash for row in overlap_df.collect()
    ]

    # ------------------------------------------------------------------
    # Check 2: Future covariate leakage
    # ------------------------------------------------------------------
    # For patients in the test set, ensure no covariate time is earlier
    # than the latest train covariate time for the same patient.
    train_panel = panel.filter(F.col("patient_id_hash").isin(train_hashes))
    test_panel = panel.filter(F.col("patient_id_hash").isin(test_hashes))

    # Per-patient max visit_time_days in train.
    train_max_time = train_panel.groupBy("patient_id_hash").agg(
        F.max("visit_time_days").alias("train_max_time")
    )

    # Join to test panel and flag any test row with visit_time < train_max.
    future_leak_df = test_panel.join(
        train_max_time, on="patient_id_hash", how="inner"
    ).filter(
        F.col("visit_time_days") < F.col("train_max_time")
    )
    future_leak_rows = future_leak_df.select(
        "patient_id_hash", "visit_id", "visit_time_days", "train_max_time"
    ).limit(100).collect()

    future_leaks = [
        {
            "patient_id_hash": r.patient_id_hash,
            "visit_id": r.visit_id,
            "test_time": r.visit_time_days,
            "train_max_time": r.train_max_time,
        }
        for r in future_leak_rows
    ]

    # ------------------------------------------------------------------
    # Check 3: Temporal ordering within patients
    # ------------------------------------------------------------------
    from pyspark.sql.window import Window

    w = Window.partitionBy("patient_id_hash").orderBy("visit_time_days")
    panel_ordered = panel.filter(
        F.col("visit_time_days").isNotNull()
    ).withColumn(
        "prev_time", F.lag("visit_time_days").over(w)
    ).filter(
        F.col("prev_time").isNotNull()
    ).filter(
        F.col("visit_time_days") < F.col("prev_time")
    )
    temporal_violations_rows = panel_ordered.select(
        "patient_id_hash", "visit_id", "visit_time_days", "prev_time"
    ).limit(100).collect()

    temporal_violations = [
        {
            "patient_id_hash": r.patient_id_hash,
            "visit_id": r.visit_id,
            "visit_time_days": r.visit_time_days,
            "prev_time": r.prev_time,
        }
        for r in temporal_violations_rows
    ]

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    passed = (
        len(overlap_hashes) == 0
        and len(future_leaks) == 0
        and len(temporal_violations) == 0
    )
    result = {
        "patient_overlap": overlap_hashes,
        "future_leaks": future_leaks,
        "temporal_violations": temporal_violations,
        "n_train_patients": len(train_hashes),
        "n_test_patients": len(test_hashes),
        "pass": passed,
    }
    logger.info(
        "audit_leakage_spark: pass=%s (overlap=%d, future_leaks=%d, temporal=%d)",
        passed, len(overlap_hashes), len(future_leaks), len(temporal_violations),
    )
    return result


# ===================================================================
# 3. build_drug_response_long_spark
# ===================================================================

def build_drug_response_long_spark(
    spark: Any,
    gdsc_path: str,
    prism_path: str,
    out_path: str,
    *,
    partition_cols: Optional[List[str]] = None,
) -> str:
    """Build long-format drug-response table from GDSC + PRISM sources.

    Reads GDSC and PRISM dose-response matrices, melts to long format,
    harmonises column names, and writes a partitioned Parquet.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
    gdsc_path : str
        Path to GDSC drug-sensitivity CSV/TSV.
    prism_path : str
        Path to PRISM drug-sensitivity CSV/TSV.
    out_path : str
        Destination for partitioned Parquet.
    partition_cols : list[str], optional
        Columns to partition by (default:
        ``["cohort", "drug_family", "drug_name", "split_id"]``).

    Returns
    -------
    str
        The *out_path* written to.
    """
    _, DataFrame, F, T = _require_pyspark()

    if partition_cols is None:
        partition_cols = ["cohort", "drug_family", "drug_name", "split_id"]

    frames = []

    # --- GDSC -----------------------------------------------------------
    if Path(gdsc_path).exists():
        gdsc_raw = spark.read.csv(gdsc_path, header=True, inferSchema=True)
        logger.info("GDSC raw: %d rows", gdsc_raw.count())

        # Normalise column names (GDSC uses DRUG_NAME / LN_IC50 / ...).
        gdsc_col_map = {c: c.lower().strip() for c in gdsc_raw.columns}
        for old, new in gdsc_col_map.items():
            gdsc_raw = gdsc_raw.withColumnRenamed(old, new)

        # Identify key columns with fallback names.
        sample_col = _first_matching(gdsc_raw.columns, [
            "cell_line_name", "cosmic_sample_id", "sanger_model_id", "sample_id",
        ])
        drug_col = _first_matching(gdsc_raw.columns, [
            "drug_name", "compound_name", "drug",
        ])
        ic50_col = _first_matching(gdsc_raw.columns, [
            "ln_ic50", "ic50", "auc", "log_ic50",
        ])
        drug_family_col = _first_matching(gdsc_raw.columns, [
            "target_pathway", "drug_family", "pathway_name",
        ])

        if sample_col and drug_col and ic50_col:
            gdsc = gdsc_raw.select(
                F.col(sample_col).cast("string").alias("sample_id"),
                F.col(drug_col).cast("string").alias("drug_name"),
                F.col(ic50_col).cast("double").alias("response_value"),
                F.col(drug_family_col).cast("string").alias("drug_family")
                if drug_family_col else F.lit("unknown").alias("drug_family"),
                F.lit("GDSC").alias("cohort"),
                F.lit("0").alias("split_id"),
            )
            frames.append(gdsc)
        else:
            logger.warning(
                "GDSC: could not identify required columns "
                "(sample=%s, drug=%s, ic50=%s) from %s",
                sample_col, drug_col, ic50_col, gdsc_raw.columns,
            )
    else:
        logger.warning("GDSC path not found: %s", gdsc_path)

    # --- PRISM ----------------------------------------------------------
    if Path(prism_path).exists():
        prism_raw = spark.read.csv(prism_path, header=True, inferSchema=True)
        logger.info("PRISM raw: %d rows", prism_raw.count())

        prism_col_map = {c: c.lower().strip() for c in prism_raw.columns}
        for old, new in prism_col_map.items():
            prism_raw = prism_raw.withColumnRenamed(old, new)

        sample_col = _first_matching(prism_raw.columns, [
            "depmap_id", "cell_line", "broad_id", "sample_id",
        ])
        drug_col = _first_matching(prism_raw.columns, [
            "name", "drug_name", "compound_name", "broad_cpd_id",
        ])
        response_col = _first_matching(prism_raw.columns, [
            "auc", "ic50", "log2_fold_change", "ec50",
        ])
        drug_family_col = _first_matching(prism_raw.columns, [
            "moa", "target", "drug_family",
        ])

        if sample_col and drug_col and response_col:
            prism = prism_raw.select(
                F.col(sample_col).cast("string").alias("sample_id"),
                F.col(drug_col).cast("string").alias("drug_name"),
                F.col(response_col).cast("double").alias("response_value"),
                F.col(drug_family_col).cast("string").alias("drug_family")
                if drug_family_col else F.lit("unknown").alias("drug_family"),
                F.lit("PRISM").alias("cohort"),
                F.lit("0").alias("split_id"),
            )
            frames.append(prism)
        else:
            logger.warning(
                "PRISM: could not identify required columns "
                "(sample=%s, drug=%s, response=%s) from %s",
                sample_col, drug_col, response_col, prism_raw.columns,
            )
    else:
        logger.warning("PRISM path not found: %s", prism_path)

    if not frames:
        raise FileNotFoundError(
            f"Neither GDSC ({gdsc_path}) nor PRISM ({prism_path}) could be loaded."
        )

    # Union all sources.
    combined = frames[0]
    for f in frames[1:]:
        combined = combined.unionByName(f, allowMissingColumns=True)

    # Drop rows with null response.
    combined = combined.filter(F.col("response_value").isNotNull())

    # Fill null drug_family.
    combined = combined.fillna({"drug_family": "unknown"})

    # --- Write partitioned Parquet ----------------------------------------
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    combined.write.mode("overwrite").partitionBy(*partition_cols).parquet(out_path)

    n_rows = combined.count()
    logger.info(
        "build_drug_response_long_spark: wrote %d rows -> %s (partitioned by %s)",
        n_rows, out_path, partition_cols,
    )
    return out_path


# ===================================================================
# 4. join_evidence_spark
# ===================================================================

def join_evidence_spark(
    spark: Any,
    attributions_path: str,
    crispr_path: str,
    chembl_path: str,
    reactome_path: str,
) -> Any:
    """Join pathway attributions to external biological evidence.

    Performs a multi-way outer join across:
      - Model pathway attributions (gene-level importance scores)
      - CRISPR gene-effect (DepMap fitness deltas)
      - ChEMBL drug-target annotations
      - Reactome pathway memberships

    The resulting DataFrame lets downstream analysis test whether
    model-attributed genes have independent biological evidence
    (CRISPR essentiality, known drug targets, pathway membership).

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
    attributions_path : str
        Parquet/CSV with columns: gene_symbol, pathway, attribution_score.
    crispr_path : str
        DepMap CRISPR gene-effect CSV (rows=cell lines, cols=genes).
    chembl_path : str
        ChEMBL drug-target CSV with columns including gene_symbol,
        drug_name, mechanism_of_action.
    reactome_path : str
        Reactome pathway membership TSV (gene_symbol, pathway_id,
        pathway_name).

    Returns
    -------
    pyspark.sql.DataFrame
        Joined evidence table, keyed by gene_symbol.
    """
    _, DataFrame, F, T = _require_pyspark()

    # --- Attributions -----------------------------------------------------
    if attributions_path.endswith(".parquet"):
        attr_df = spark.read.parquet(attributions_path)
    else:
        attr_df = spark.read.csv(attributions_path, header=True, inferSchema=True)

    # Normalise to gene_symbol key.
    attr_col_map = {c: c.lower().strip() for c in attr_df.columns}
    for old, new in attr_col_map.items():
        attr_df = attr_df.withColumnRenamed(old, new)

    gene_col = _first_matching(attr_df.columns, [
        "gene_symbol", "gene", "symbol", "gene_name",
    ])
    if gene_col and gene_col != "gene_symbol":
        attr_df = attr_df.withColumnRenamed(gene_col, "gene_symbol")

    # Aggregate attributions to gene level if there are pathway-level rows.
    if "attribution_score" in attr_df.columns:
        attr_agg = attr_df.groupBy("gene_symbol").agg(
            F.mean("attribution_score").alias("mean_attribution"),
            F.max("attribution_score").alias("max_attribution"),
            F.count("*").alias("n_pathways_attributed"),
        )
    else:
        attr_agg = attr_df.select("gene_symbol").distinct().withColumn(
            "mean_attribution", F.lit(None).cast("double")
        ).withColumn(
            "max_attribution", F.lit(None).cast("double")
        ).withColumn(
            "n_pathways_attributed", F.lit(0)
        )

    # --- CRISPR gene-effect -----------------------------------------------
    crispr_evidence = None
    if Path(crispr_path).exists():
        crispr_raw = spark.read.csv(crispr_path, header=True, inferSchema=True)
        # DepMap format: rows=cell lines, columns=genes.
        # We want per-gene summary: mean effect, fraction essential.
        # Melt wide to long.
        gene_cols = [
            c for c in crispr_raw.columns
            if c.lower() not in ("depmap_id", "cell_line_name", "stripped_cell_line_name")
        ]
        if gene_cols:
            # Use stack to unpivot.
            stack_expr = ", ".join(
                [f"'{g}', `{g}`" for g in gene_cols]
            )
            n_genes = len(gene_cols)
            crispr_long = crispr_raw.selectExpr(
                f"stack({n_genes}, {stack_expr}) as (gene_symbol, gene_effect)"
            )
            crispr_evidence = crispr_long.groupBy("gene_symbol").agg(
                F.mean("gene_effect").alias("crispr_mean_effect"),
                F.expr(
                    "sum(case when gene_effect < -0.5 then 1 else 0 end) / count(*)"
                ).alias("crispr_frac_essential"),
            )
    if crispr_evidence is None:
        logger.warning("CRISPR evidence not loaded from %s", crispr_path)

    # --- ChEMBL drug-target -----------------------------------------------
    chembl_evidence = None
    if Path(chembl_path).exists():
        chembl_raw = spark.read.csv(chembl_path, header=True, inferSchema=True)
        chembl_col_map = {c: c.lower().strip() for c in chembl_raw.columns}
        for old, new in chembl_col_map.items():
            chembl_raw = chembl_raw.withColumnRenamed(old, new)

        gene_col_c = _first_matching(chembl_raw.columns, [
            "gene_symbol", "target_gene", "gene_name", "symbol",
        ])
        if gene_col_c:
            if gene_col_c != "gene_symbol":
                chembl_raw = chembl_raw.withColumnRenamed(gene_col_c, "gene_symbol")
            chembl_evidence = chembl_raw.groupBy("gene_symbol").agg(
                F.countDistinct("drug_name").alias("chembl_n_drugs")
                if "drug_name" in chembl_raw.columns
                else F.lit(0).alias("chembl_n_drugs"),
                F.collect_set("mechanism_of_action").alias("chembl_moa_set")
                if "mechanism_of_action" in chembl_raw.columns
                else F.array().alias("chembl_moa_set"),
            )
    if chembl_evidence is None:
        logger.warning("ChEMBL evidence not loaded from %s", chembl_path)

    # --- Reactome pathway membership --------------------------------------
    reactome_evidence = None
    if Path(reactome_path).exists():
        reactome_raw = spark.read.csv(
            reactome_path, sep="\t", header=True, inferSchema=True,
        )
        reactome_col_map = {c: c.lower().strip() for c in reactome_raw.columns}
        for old, new in reactome_col_map.items():
            reactome_raw = reactome_raw.withColumnRenamed(old, new)

        gene_col_r = _first_matching(reactome_raw.columns, [
            "gene_symbol", "gene", "symbol",
        ])
        if gene_col_r:
            if gene_col_r != "gene_symbol":
                reactome_raw = reactome_raw.withColumnRenamed(gene_col_r, "gene_symbol")
            reactome_evidence = reactome_raw.groupBy("gene_symbol").agg(
                F.countDistinct("pathway_id").alias("reactome_n_pathways")
                if "pathway_id" in reactome_raw.columns
                else F.countDistinct("pathway_name").alias("reactome_n_pathways"),
                F.collect_set("pathway_name").alias("reactome_pathways")
                if "pathway_name" in reactome_raw.columns
                else F.array().alias("reactome_pathways"),
            )
    if reactome_evidence is None:
        logger.warning("Reactome evidence not loaded from %s", reactome_path)

    # --- Multi-way outer join on gene_symbol ------------------------------
    result = attr_agg
    if crispr_evidence is not None:
        result = result.join(crispr_evidence, on="gene_symbol", how="full_outer")
    if chembl_evidence is not None:
        result = result.join(chembl_evidence, on="gene_symbol", how="full_outer")
    if reactome_evidence is not None:
        result = result.join(reactome_evidence, on="gene_symbol", how="full_outer")

    result = result.filter(F.col("gene_symbol").isNotNull())
    n_genes = result.count()
    logger.info(
        "join_evidence_spark: %d genes with at least one evidence source", n_genes
    )
    return result


# ===================================================================
# 5. aggregate_bootstrap_metrics_spark
# ===================================================================

def aggregate_bootstrap_metrics_spark(
    spark: Any,
    predictions_path: str,
    n_boot: int = 2000,
    *,
    seed: int = 42,
    ci_level: float = 0.95,
) -> dict:
    """Per-drug bootstrap confidence intervals for prediction metrics.

    Reads a predictions Parquet with columns: ``drug_name``,
    ``y_true``, ``y_pred`` (and optionally ``sample_id``, ``split_id``).
    For each drug, draws ``n_boot`` bootstrap resamples and computes
    MSE and Pearson correlation, returning point estimates and CI bounds.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
    predictions_path : str
        Parquet directory with prediction results.
    n_boot : int
        Number of bootstrap resamples per drug (default 2000).
    seed : int
        RNG seed for reproducibility.
    ci_level : float
        Confidence level (default 0.95 for 95% CIs).

    Returns
    -------
    dict
        ``{drug_name: {"n": int, "mse": float, "mse_ci_lo": float,
        "mse_ci_hi": float, "pearson_r": float, "pearson_r_ci_lo": float,
        "pearson_r_ci_hi": float}, ...}``
    """
    _, DataFrame, F, T = _require_pyspark()
    import numpy as np

    preds = spark.read.parquet(predictions_path)

    # Validate required columns.
    required = {"drug_name", "y_true", "y_pred"}
    available = set(preds.columns)
    missing = required - available
    if missing:
        raise ValueError(
            f"Predictions Parquet missing required columns: {missing}. "
            f"Available: {sorted(available)}"
        )

    # Collect per-drug arrays. For datasets fitting in driver memory this
    # is efficient; for very large tables, a UDF-based approach would be
    # needed, but bootstrap resampling inherently requires the full array.
    drug_groups = (
        preds.select("drug_name", "y_true", "y_pred")
        .groupBy("drug_name")
        .agg(
            F.collect_list("y_true").alias("y_true_list"),
            F.collect_list("y_pred").alias("y_pred_list"),
        )
        .collect()
    )

    rng = np.random.RandomState(seed)
    alpha = 1.0 - ci_level
    results: Dict[str, dict] = {}

    for row in drug_groups:
        drug = row.drug_name
        y_true = np.array(row.y_true_list, dtype=np.float64)
        y_pred = np.array(row.y_pred_list, dtype=np.float64)

        # Drop NaN pairs.
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        n = len(y_true)

        if n < 3:
            results[drug] = {
                "n": n,
                "mse": float("nan"),
                "mse_ci_lo": float("nan"),
                "mse_ci_hi": float("nan"),
                "pearson_r": float("nan"),
                "pearson_r_ci_lo": float("nan"),
                "pearson_r_ci_hi": float("nan"),
                "note": "Too few samples for bootstrap",
            }
            continue

        boot_mse = np.empty(n_boot)
        boot_r = np.empty(n_boot)

        for b in range(n_boot):
            idx = rng.randint(0, n, size=n)
            yt = y_true[idx]
            yp = y_pred[idx]
            boot_mse[b] = np.mean((yt - yp) ** 2)
            # Pearson r: guard against constant arrays.
            std_t = np.std(yt)
            std_p = np.std(yp)
            if std_t > 0 and std_p > 0:
                boot_r[b] = np.corrcoef(yt, yp)[0, 1]
            else:
                boot_r[b] = 0.0

        results[drug] = {
            "n": int(n),
            "mse": float(np.mean((y_true - y_pred) ** 2)),
            "mse_ci_lo": float(np.percentile(boot_mse, 100 * alpha / 2)),
            "mse_ci_hi": float(np.percentile(boot_mse, 100 * (1 - alpha / 2))),
            "pearson_r": float(
                np.corrcoef(y_true, y_pred)[0, 1]
                if np.std(y_true) > 0 and np.std(y_pred) > 0
                else 0.0
            ),
            "pearson_r_ci_lo": float(np.percentile(boot_r, 100 * alpha / 2)),
            "pearson_r_ci_hi": float(np.percentile(boot_r, 100 * (1 - alpha / 2))),
        }

    logger.info(
        "aggregate_bootstrap_metrics_spark: %d drugs, n_boot=%d, ci=%.0f%%",
        len(results), n_boot, ci_level * 100,
    )
    return results


# ===================================================================
# Internal helpers
# ===================================================================

def _first_matching(columns: list, candidates: List[str]) -> Optional[str]:
    """Return the first column name from *candidates* found in *columns*."""
    col_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in col_lower:
            return col_lower[cand.lower()]
    return None
