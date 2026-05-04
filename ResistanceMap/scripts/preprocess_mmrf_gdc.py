"""Merge per-sample MMRF-COMMPASS GDC downloads into the single-file schema
that ``resistancemap/data/loaders.py:load_mmrf_data`` consumes.

Inputs (already on disk under ``data/raw/mmrf_commpass/``):

    rna/  859  <file_uuid>.rna_seq.augmented_star_gene_counts.tsv
    snv/  1091 <file_uuid>.wxs.aliquot_ensemble_masked.maf.gz
    cnv/  1010 <file_uuid>_wgs_gdc_realn.cr.igv.reheader.seg.txt

with manifests at ``~/.gdc/mmrf_{rna,snv,cnv}.tsv`` (id/filename/md5/size/state).

Outputs (written next to the per-sample dirs):

    file_to_case.tsv     # file_name -> patient/sample/aliquot/sample_type
    gene_expression.tsv  # genes (Ensembl, version-stripped) x aliquot_submitter_id
    mutations.tsv        # long: Hugo_Symbol + variant fields + submitter_id
    copy_number.tsv      # long: per-segment, with submitter_id

The script never fabricates data. Files whose file_id cannot be resolved via
the GDC ``/files`` API are logged and skipped. STAR rows have their gene_id
version suffix stripped (``ENSG00000000003.15`` -> ``ENSG00000000003``) for
cross-file alignment; expression values are raw TPM (``tpm_unstranded``) so
the downstream loader can decide whether to log-transform.

Run:

    python scripts/preprocess_mmrf_gdc.py
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("preprocess_mmrf_gdc")

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
BATCH_SIZE = 200          # file_ids per /files request
API_RETRY_LIMIT = 4
API_RETRY_BACKOFF = 3.0   # seconds

# MAF columns we keep (long-format mutations.tsv schema)
MAF_KEEP_COLUMNS = [
    "Hugo_Symbol",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Variant_Classification",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
    "Tumor_Sample_Barcode",
    "t_alt_count",
    "t_ref_count",
    "t_depth",
]


# --------------------------------------------------------------------------- #
# GDC API: file_id -> (patient, sample, aliquot, sample_type)
# --------------------------------------------------------------------------- #
def _post_files(file_ids: List[str]) -> List[dict]:
    body = json.dumps(
        {
            "filters": {
                "op": "in",
                "content": {"field": "file_id", "value": file_ids},
            },
            "fields": (
                "file_id,file_name,"
                "cases.submitter_id,"
                "cases.samples.submitter_id,"
                "cases.samples.sample_type,"
                "cases.samples.tissue_type,"
                "cases.samples.portions.analytes.aliquots.submitter_id"
            ),
            "size": len(file_ids) + 50,
        }
    ).encode()

    last_err: Optional[Exception] = None
    for attempt in range(1, API_RETRY_LIMIT + 1):
        try:
            req = urllib.request.Request(
                GDC_FILES_ENDPOINT,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
            return payload.get("data", {}).get("hits", [])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            if attempt < API_RETRY_LIMIT:
                wait = API_RETRY_BACKOFF * attempt
                logger.warning(
                    f"GDC /files request failed (attempt {attempt}/{API_RETRY_LIMIT}): {e}; "
                    f"retrying in {wait:.0f}s"
                )
                time.sleep(wait)
    raise RuntimeError(f"GDC /files request exhausted retries: {last_err}")


def _pick_tumor_aliquot(case: dict) -> Tuple[str, str, str, str]:
    """Choose (patient, sample, aliquot, sample_type) preferring Tumor samples.

    Falls back to the first sample if no Tumor sample is present (rare for
    open-access MMRF, where the per-file association is to a single tumor
    aliquot).
    """
    patient = str(case.get("submitter_id", "") or "")
    samples = case.get("samples", []) or []
    chosen: Optional[dict] = None
    # Prefer tumor / non-normal samples
    for s in samples:
        if str(s.get("tissue_type", "")).lower() == "tumor":
            chosen = s
            break
    if chosen is None:
        for s in samples:
            stype = str(s.get("sample_type", "")).lower()
            if "normal" not in stype:
                chosen = s
                break
    if chosen is None and samples:
        chosen = samples[0]
    if chosen is None:
        return patient, "", "", ""

    sample_id = str(chosen.get("submitter_id", "") or "")
    sample_type = str(chosen.get("sample_type", "") or "")

    aliquot = ""
    for portion in chosen.get("portions", []) or []:
        for analyte in portion.get("analytes", []) or []:
            aliquots = analyte.get("aliquots", []) or []
            if aliquots:
                aliquot = str(aliquots[0].get("submitter_id", "") or "")
                if aliquot:
                    break
        if aliquot:
            break
    return patient, sample_id, aliquot, sample_type


def build_file_to_case_lookup(
    manifest_paths: Dict[str, Path],
    out_path: Path,
) -> pd.DataFrame:
    """Resolve file_name -> (patient, sample, aliquot, sample_type) for every
    file in the manifests, via batched GDC /files queries.

    Writes a TSV at ``out_path`` for auditability and returns the same frame.
    """
    rows: List[dict] = []
    seen_ids: set[str] = set()

    file_id_to_modality: Dict[str, str] = {}
    file_id_to_filename: Dict[str, str] = {}

    for modality, mpath in manifest_paths.items():
        if not mpath.exists():
            logger.warning(f"Manifest for {modality} not found at {mpath}; skipping")
            continue
        m = pd.read_csv(mpath, sep="\t", dtype=str)
        for fid, fname in zip(m["id"], m["filename"]):
            file_id_to_modality[fid] = modality
            file_id_to_filename[fid] = fname

    all_ids = list(file_id_to_modality.keys())
    logger.info(
        f"Resolving {len(all_ids)} file_ids across "
        f"{len(manifest_paths)} manifests via GDC /files"
    )

    for start in range(0, len(all_ids), BATCH_SIZE):
        batch = all_ids[start : start + BATCH_SIZE]
        hits = _post_files(batch)
        for h in hits:
            fid = h.get("id") or h.get("file_id") or ""
            fname = h.get("file_name", "")
            cases = h.get("cases", []) or []
            if not cases:
                continue
            patient, sample, aliquot, sample_type = _pick_tumor_aliquot(cases[0])
            rows.append(
                {
                    "file_id": fid,
                    "file_name": fname,
                    "modality": file_id_to_modality.get(fid, ""),
                    "patient_submitter_id": patient,
                    "sample_submitter_id": sample,
                    "aliquot_submitter_id": aliquot,
                    "sample_type": sample_type,
                }
            )
            seen_ids.add(fid)
        logger.info(
            f"  resolved {min(start + BATCH_SIZE, len(all_ids))}/{len(all_ids)} file_ids"
        )

    missing = [fid for fid in all_ids if fid not in seen_ids]
    if missing:
        logger.warning(
            f"GDC /files returned no hit for {len(missing)} file_ids "
            f"(first 3: {missing[:3]})"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"Wrote file->case lookup ({len(df)} rows) to {out_path}")
    return df


# --------------------------------------------------------------------------- #
# Step 2: gene_expression.tsv
# --------------------------------------------------------------------------- #
def _read_star_counts(path: Path) -> Optional[pd.Series]:
    """Read STAR ``augmented_star_gene_counts`` TSV.

    Returns a Series indexed by version-stripped Ensembl gene_id with raw
    ``tpm_unstranded`` as the value (float32). Returns None if the file is
    malformed.
    """
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            usecols=["gene_id", "tpm_unstranded"],
            dtype={"gene_id": str, "tpm_unstranded": "float32"},
            low_memory=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read STAR counts {path.name}: {exc}")
        return None

    # Drop the leading N_unmapped / N_multimapping / N_noFeature / N_ambiguous
    # bookkeeping rows. They sort to the top of the file and have NaN tpm.
    df = df[~df["gene_id"].str.startswith("N_", na=False)]
    df = df.dropna(subset=["gene_id"])
    df["gene_id"] = df["gene_id"].str.split(".", n=1).str[0]
    s = df.set_index("gene_id")["tpm_unstranded"].astype("float32")
    # Some Ensembl IDs can repeat after version strip (PAR_Y duplicates);
    # take the first observation deterministically.
    s = s[~s.index.duplicated(keep="first")]
    return s


def build_gene_expression(
    rna_dir: Path,
    file_lookup: Dict[str, dict],
    out_path: Path,
) -> Tuple[int, int, List[str]]:
    """Merge all STAR TSVs into a (genes x aliquots) matrix.

    Streams each file, accumulating one column per aliquot_submitter_id.
    Memory: a single dict-of-dicts, ~60k genes * ~860 columns * 4 bytes
    -> ~200 MB.
    """
    files = sorted(rna_dir.glob("*.rna_seq.augmented_star_gene_counts.tsv"))
    logger.info(f"  Building gene_expression.tsv from {len(files)} STAR files")

    # gene -> {col_name: tpm}
    matrix: Dict[str, Dict[str, np.float32]] = defaultdict(dict)
    columns: List[str] = []
    column_set: set[str] = set()
    skipped = 0
    skipped_examples: List[str] = []

    for i, fpath in enumerate(files, 1):
        meta = file_lookup.get(fpath.name)
        if not meta or not meta.get("aliquot_submitter_id"):
            skipped += 1
            if len(skipped_examples) < 3:
                skipped_examples.append(fpath.name)
            continue
        col = meta["aliquot_submitter_id"]
        # Disambiguate if the same aliquot ever appears twice (longitudinal
        # re-runs / replicate sequencing). We don't silently overwrite.
        if col in column_set:
            suffix = 2
            base = col
            while f"{base}__rep{suffix}" in column_set:
                suffix += 1
            col = f"{base}__rep{suffix}"
        s = _read_star_counts(fpath)
        if s is None:
            skipped += 1
            if len(skipped_examples) < 3:
                skipped_examples.append(fpath.name)
            continue
        columns.append(col)
        column_set.add(col)
        for gid, val in s.items():
            matrix[gid][col] = val
        if i % 100 == 0 or i == len(files):
            logger.info(f"    processed {i}/{len(files)} STAR files")

    # Build DataFrame: rows = sorted genes, cols = aliquots in load order.
    if not columns:
        raise RuntimeError("No STAR files mapped to aliquots; cannot build matrix")

    gene_index = sorted(matrix.keys())
    arr = np.zeros((len(gene_index), len(columns)), dtype="float32")
    col_index = {c: i for i, c in enumerate(columns)}
    for ri, gid in enumerate(gene_index):
        row = matrix[gid]
        for c, v in row.items():
            arr[ri, col_index[c]] = v

    df = pd.DataFrame(arr, index=pd.Index(gene_index, name="gene_id"), columns=columns)
    df.to_csv(out_path, sep="\t", float_format="%.4f")
    logger.info(
        f"  Wrote {out_path} shape={df.shape}; first 5 cols={columns[:5]}; "
        f"skipped {skipped} files (examples: {skipped_examples})"
    )
    return df.shape[0], df.shape[1], columns


# --------------------------------------------------------------------------- #
# Step 3: mutations.tsv
# --------------------------------------------------------------------------- #
def build_mutations(
    snv_dir: Path,
    file_lookup: Dict[str, dict],
    out_path: Path,
) -> Tuple[int, Counter]:
    files = sorted(snv_dir.glob("*.maf.gz"))
    logger.info(f"  Building mutations.tsv from {len(files)} MAF.gz files")

    chunks: List[pd.DataFrame] = []
    skipped = 0
    skipped_examples: List[str] = []
    gene_counter: Counter = Counter()
    total_rows = 0

    for i, fpath in enumerate(files, 1):
        meta = file_lookup.get(fpath.name)
        if not meta:
            skipped += 1
            if len(skipped_examples) < 3:
                skipped_examples.append(fpath.name)
            continue
        submitter = meta.get("aliquot_submitter_id") or meta.get(
            "patient_submitter_id"
        )
        if not submitter:
            skipped += 1
            if len(skipped_examples) < 3:
                skipped_examples.append(fpath.name)
            continue
        try:
            with gzip.open(fpath, "rt") as fh:
                df = pd.read_csv(fh, sep="\t", comment="#", low_memory=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to read MAF {fpath.name}: {exc}")
            skipped += 1
            if len(skipped_examples) < 3:
                skipped_examples.append(fpath.name)
            continue
        keep = [c for c in MAF_KEEP_COLUMNS if c in df.columns]
        df = df[keep].copy()
        df["submitter_id"] = submitter
        df["patient_submitter_id"] = meta.get("patient_submitter_id", "")
        chunks.append(df)
        total_rows += len(df)
        if "Hugo_Symbol" in df.columns:
            gene_counter.update(df["Hugo_Symbol"].astype(str).tolist())
        if i % 100 == 0 or i == len(files):
            logger.info(
                f"    processed {i}/{len(files)} MAFs; cumulative rows={total_rows}"
            )

    if not chunks:
        raise RuntimeError("No MAFs successfully parsed; cannot build mutations.tsv")

    big = pd.concat(chunks, ignore_index=True)
    big.to_csv(out_path, sep="\t", index=False)
    logger.info(
        f"  Wrote {out_path} rows={len(big)}; skipped {skipped} files "
        f"(examples: {skipped_examples})"
    )
    top = gene_counter.most_common(10)
    logger.info(f"  Top-10 Hugo_Symbol counts: {top}")
    return len(big), gene_counter


# --------------------------------------------------------------------------- #
# Step 4: copy_number.tsv
# --------------------------------------------------------------------------- #
def build_copy_number(
    cnv_dir: Path,
    file_lookup: Dict[str, dict],
    out_path: Path,
) -> Tuple[int, Counter]:
    files = sorted(cnv_dir.glob("*_wgs_gdc_realn.cr.igv.reheader.seg.txt"))
    logger.info(f"  Building copy_number.tsv from {len(files)} segment files")

    chunks: List[pd.DataFrame] = []
    skipped = 0
    skipped_examples: List[str] = []
    chrom_counter: Counter = Counter()
    total_rows = 0

    for i, fpath in enumerate(files, 1):
        meta = file_lookup.get(fpath.name)
        if not meta:
            skipped += 1
            if len(skipped_examples) < 3:
                skipped_examples.append(fpath.name)
            continue
        submitter = meta.get("aliquot_submitter_id") or meta.get(
            "patient_submitter_id"
        )
        if not submitter:
            skipped += 1
            if len(skipped_examples) < 3:
                skipped_examples.append(fpath.name)
            continue
        try:
            df = pd.read_csv(fpath, sep="\t", low_memory=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to read CNV seg {fpath.name}: {exc}")
            skipped += 1
            if len(skipped_examples) < 3:
                skipped_examples.append(fpath.name)
            continue

        # Spec lists "GDC_Aliquot"; the on-disk header is "GDC_Aliquot_ID"
        # whose value is the aliquot submitter id (matches our lookup).
        # Normalise to a single column name for the merged output.
        if "GDC_Aliquot_ID" in df.columns and "GDC_Aliquot" not in df.columns:
            df = df.rename(columns={"GDC_Aliquot_ID": "GDC_Aliquot"})

        df["submitter_id"] = submitter
        df["patient_submitter_id"] = meta.get("patient_submitter_id", "")
        chunks.append(df)
        total_rows += len(df)
        if "Chromosome" in df.columns:
            chrom_counter.update(df["Chromosome"].astype(str).tolist())
        if i % 200 == 0 or i == len(files):
            logger.info(
                f"    processed {i}/{len(files)} CNV segs; cumulative rows={total_rows}"
            )

    if not chunks:
        raise RuntimeError("No CNV segs successfully parsed; cannot build copy_number.tsv")

    big = pd.concat(chunks, ignore_index=True)
    big.to_csv(out_path, sep="\t", index=False)
    logger.info(
        f"  Wrote {out_path} rows={len(big)}; skipped {skipped} files "
        f"(examples: {skipped_examples})"
    )
    return len(big), chrom_counter


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmrf-dir",
        type=Path,
        default=Path("data/raw/mmrf_commpass"),
        help="Root MMRF directory holding rna/, snv/, cnv/ subdirs",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path.home() / ".gdc",
        help="Directory holding mmrf_{rna,snv,cnv}.tsv manifests",
    )
    parser.add_argument(
        "--reuse-lookup",
        action="store_true",
        help="If file_to_case.tsv already exists, reuse it instead of re-querying GDC",
    )
    parser.add_argument(
        "--skip",
        choices=("rna", "snv", "cnv"),
        action="append",
        default=[],
        help="Skip a modality (repeatable). Useful for incremental re-runs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    mmrf_dir: Path = args.mmrf_dir.resolve()
    if not mmrf_dir.exists():
        logger.error(f"MMRF dir not found: {mmrf_dir}")
        return 2

    manifest_paths = {
        "rna": args.manifest_dir / "mmrf_rna.tsv",
        "snv": args.manifest_dir / "mmrf_snv.tsv",
        "cnv": args.manifest_dir / "mmrf_cnv.tsv",
    }
    lookup_path = mmrf_dir / "file_to_case.tsv"

    timings: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Step 1: file -> case lookup
    # ------------------------------------------------------------------ #
    t0 = time.time()
    if args.reuse_lookup and lookup_path.exists():
        logger.info(f"Reusing existing lookup at {lookup_path}")
        lookup_df = pd.read_csv(lookup_path, sep="\t", dtype=str).fillna("")
    else:
        lookup_df = build_file_to_case_lookup(manifest_paths, lookup_path)
    timings["lookup"] = time.time() - t0

    # Convert to file_name -> meta dict for O(1) per-file access.
    file_lookup: Dict[str, dict] = {
        row["file_name"]: row.to_dict() for _, row in lookup_df.iterrows()
    }

    # Manifest size vs lookup size diagnostic
    expected_total = 0
    for mname, mp in manifest_paths.items():
        if mp.exists():
            expected_total += sum(1 for _ in mp.open()) - 1
    if expected_total and len(lookup_df) < expected_total:
        miss = expected_total - len(lookup_df)
        logger.warning(
            f"file->case lookup missing {miss} of {expected_total} manifest entries"
        )

    # ------------------------------------------------------------------ #
    # Step 2: gene expression
    # ------------------------------------------------------------------ #
    if "rna" not in args.skip:
        t0 = time.time()
        out = mmrf_dir / "gene_expression.tsv"
        n_genes, n_aliquots, _cols = build_gene_expression(
            mmrf_dir / "rna", file_lookup, out
        )
        timings["gene_expression"] = time.time() - t0
        logger.info(
            f"  gene_expression.tsv: {n_genes} genes x {n_aliquots} aliquots, "
            f"{out.stat().st_size / 1e6:.1f} MB, "
            f"{timings['gene_expression']:.1f} s"
        )

    # ------------------------------------------------------------------ #
    # Step 3: mutations
    # ------------------------------------------------------------------ #
    if "snv" not in args.skip:
        t0 = time.time()
        out = mmrf_dir / "mutations.tsv"
        n_rows, gene_counter = build_mutations(mmrf_dir / "snv", file_lookup, out)
        timings["mutations"] = time.time() - t0
        # Sanity check for top-30 gene presence
        top30 = [g for g, _ in gene_counter.most_common(30)]
        canonical = ["TP53", "KRAS", "NRAS", "BRAF", "MAF", "FGFR3", "NSD2"]
        present = [g for g in canonical if g in top30]
        logger.info(
            f"  mutations.tsv: {n_rows} rows, "
            f"{out.stat().st_size / 1e6:.1f} MB, "
            f"{timings['mutations']:.1f} s; "
            f"canonical MM drivers in top30: {present}"
        )

    # ------------------------------------------------------------------ #
    # Step 4: copy number
    # ------------------------------------------------------------------ #
    if "cnv" not in args.skip:
        t0 = time.time()
        out = mmrf_dir / "copy_number.tsv"
        n_rows, chrom_counter = build_copy_number(mmrf_dir / "cnv", file_lookup, out)
        timings["copy_number"] = time.time() - t0
        # Show segment counts per chromosome (chr1 + chr17 should be high).
        ordered = sorted(
            chrom_counter.items(),
            key=lambda kv: (
                # Numeric chroms first, sorted; X/Y/M last
                (0, int(kv[0][3:]))
                if kv[0].startswith("chr") and kv[0][3:].isdigit()
                else (1, kv[0])
            ),
        )
        logger.info(f"  copy_number.tsv: {n_rows} rows, "
                    f"{out.stat().st_size / 1e6:.1f} MB, "
                    f"{timings['copy_number']:.1f} s")
        logger.info(f"  Segments per chromosome: {ordered}")

    # ------------------------------------------------------------------ #
    # Final summary
    # ------------------------------------------------------------------ #
    logger.info("Wall-clock summary (seconds):")
    for k, v in timings.items():
        logger.info(f"  {k}: {v:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
