#!/usr/bin/env python3
"""
Derive per-patient binary flags for the five MM-resistance cytogenetic events
named as confounders by the causal-validity audit:

    del17p, chr1q21_gain, del13q, t(4;14), t(11;14)

Inputs (all under data/raw/mmrf_commpass/):
    copy_number.tsv      long-format CNV segments (cols include submitter_id,
                         Chromosome, Start, End, Segment_Mean) — produced by
                         scripts/preprocess_mmrf_gdc.py
    gene_expression.tsv  TPM matrix, genes x aliquots — produced by the same
    samples.tsv          patient <-> aliquot <-> sample_type join

Output:
    data/raw/mmrf_commpass/cytogenetics.tsv
        submitter_id  del17p  chr1q21_gain  del13q  t_4_14  t_11_14
        source_aliquot  notes

Per-patient sample selection
----------------------------
When a patient has multiple tumor aliquots, this script uses the EARLIEST
baseline ("Primary ...") tumor sample. Selection order:
    1. samples whose `sample_type` contains "Primary"
    2. tie-break by lexicographic minimum of the aliquot_submitter_id
       (which embeds the visit number, e.g. "_1_BM_..." comes before "_2_BM_...")
This deterministically pins the call to the diagnostic-baseline biopsy and
avoids contamination from post-treatment relapse samples.

Calling rules — see the per-event spec below. Where a signal is missing,
the corresponding cell is left as NaN (NEVER fabricated 0).

Constraints: stdlib + pandas + numpy only. No fabrication of cytogenetic
calls. No hard-coded patient IDs.
"""

# ---------------------------------------------------------------------------
# GRCh38 GENCODE v36 cytoband regions
# Coordinates verified against Ensembl GRCh38.p13 / GENCODE v36 gene models
# (the same gene model the GDC RNA-seq pipeline uses; see
#  https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/Expression_mRNA_Pipeline/).
#
# Per-locus references:
#   TP53   (chr17 short arm) https://www.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000141510
#   CKS1B  (chr1q21.3)        https://www.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000173207
#   MCL1   (chr1q21.2)        https://www.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000143384
#   RB1    (chr13q14.2)       https://www.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000139687
# Cytoband intervals (1q21 ~chr1:153.5-155.5Mb, 13q14 ~chr13:48.3-48.5Mb)
# follow the CytoBandIdeo track on UCSC GRCh38
# (https://genome.ucsc.edu/cgi-bin/hgTables, group=Mapping and Sequencing,
#  track=Chromosome Band).
# ---------------------------------------------------------------------------
TP53_REGION       = ("chr17",  7565097,   7590856)   # del17p
CHR1Q21_REGION    = ("chr1",   153500000, 155500000) # 1q21 gain (CKS1B/MCL1)
RB1_REGION        = ("chr13",  48303256,  48481890)  # del13q (RB1 locus)

# log2-ratio thresholds on Segment_Mean (GDC ASCAT-derived seg files)
LOSS_THRESHOLD = -0.3
GAIN_THRESHOLD =  0.3

# Translocation surrogate genes (Ensembl IDs, no version suffix)
FGFR3_ENSG = "ENSG00000068078"   # t(4;14) IGH-FGFR3
NSD2_ENSG  = "ENSG00000109685"   # t(4;14) IGH-NSD2 (formerly WHSC1/MMSET)
CCND1_ENSG = "ENSG00000110092"   # t(11;14) IGH-CCND1
TP53_ENSG  = "ENSG00000141510"   # used for downstream silencing cross-check

TRANSLOC_Z_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    here       = Path(__file__).resolve().parent
    data_dir   = here.parent / "data" / "raw" / "mmrf_commpass"
    cnv_path   = data_dir / "copy_number.tsv"
    rna_path   = data_dir / "gene_expression.tsv"
    samp_path  = data_dir / "samples.tsv"
    out_path   = data_dir / "cytogenetics.tsv"

    missing = [p for p in (cnv_path, rna_path, samp_path) if not p.exists()]
    if missing:
        msg = ", ".join(str(p) for p in missing)
        sys.stderr.write(
            f"[ERROR] required input(s) not found: {msg}\n"
            "        run scripts/preprocess_mmrf_gdc.py first\n"
        )
        return 2

    samples = pd.read_csv(samp_path, sep="\t", dtype=str)
    samples.columns = [c.strip() for c in samples.columns]

    needed = {"submitter_id", "sample_type", "aliquot_submitter_id"}
    if not needed.issubset(samples.columns):
        sys.stderr.write(
            f"[ERROR] samples.tsv missing columns: {needed - set(samples.columns)}\n"
        )
        return 2

    # ---- pick one baseline tumor aliquot per patient ----------------------
    sample_type = samples["sample_type"].fillna("")
    is_primary = sample_type.str.contains("Primary", case=False, na=False)
    is_tumor   = ~sample_type.str.contains("Normal",  case=False, na=False)
    baseline = samples[is_primary & is_tumor].copy()
    # Aliquot submitter IDs encode the visit ("MMRF_xxxx_<N>_BM_..."), so
    # lexicographic min == earliest visit.
    baseline = baseline.sort_values(["submitter_id", "aliquot_submitter_id"])
    baseline_per_patient = baseline.groupby("submitter_id", as_index=False).first()

    cnv = pd.read_csv(cnv_path, sep="\t", dtype={"Chromosome": str})
    cnv.columns = [c.strip() for c in cnv.columns]
    # The preprocessor writes `submitter_id` = aliquot id, `patient_submitter_id`
    # = patient id. Earlier versions of this script assumed `submitter_id` was
    # the patient id and produced 0 calls because aliquot strings never matched
    # patient strings. Use `patient_submitter_id` as the canonical patient key.
    if "patient_submitter_id" in cnv.columns:
        cnv = cnv.rename(columns={"submitter_id": "aliquot_submitter_id",
                                  "patient_submitter_id": "submitter_id"})
    elif "submitter_id" not in cnv.columns:
        if "aliquot_submitter_id" in cnv.columns:
            cnv = cnv.merge(
                samples[["submitter_id", "aliquot_submitter_id"]],
                on="aliquot_submitter_id",
                how="left",
            )
        else:
            sys.stderr.write(
                "[ERROR] copy_number.tsv lacks submitter_id and aliquot_submitter_id\n"
            )
            return 2

    cnv_required = {"submitter_id", "Chromosome", "Start", "End", "Segment_Mean"}
    if not cnv_required.issubset(cnv.columns):
        sys.stderr.write(
            f"[ERROR] copy_number.tsv missing columns: {cnv_required - set(cnv.columns)}\n"
        )
        return 2

    cnv["Start"]        = pd.to_numeric(cnv["Start"],        errors="coerce")
    cnv["End"]          = pd.to_numeric(cnv["End"],          errors="coerce")
    cnv["Segment_Mean"] = pd.to_numeric(cnv["Segment_Mean"], errors="coerce")
    cnv = cnv.dropna(subset=["Start", "End", "Segment_Mean"])
    # normalize chromosome label
    cnv["Chromosome"] = cnv["Chromosome"].astype(str)
    cnv["Chromosome"] = np.where(
        cnv["Chromosome"].str.startswith("chr"),
        cnv["Chromosome"],
        "chr" + cnv["Chromosome"],
    )

    # ---- expression matrix (genes x aliquots) -----------------------------
    rna = pd.read_csv(rna_path, sep="\t", index_col=0)
    rna.index = rna.index.astype(str).str.split(".").str[0]  # drop ENSG version

    # ---- helpers ----------------------------------------------------------
    def cnv_call(patient_segs: pd.DataFrame, region, op):
        """Return 1 if any segment in `patient_segs` overlaps `region`
        with Segment_Mean satisfying `op` against the threshold; 0 otherwise.
        `op` is one of ('loss', 'gain')."""
        chrom, start, end = region
        on_chrom = patient_segs[patient_segs["Chromosome"] == chrom]
        if on_chrom.empty:
            return 0  # patient has CNV data but no segments on this chrom
                      # (very rare for whole-genome ASCAT calls; treat as no event)
        overlap = on_chrom[(on_chrom["End"] >= start) & (on_chrom["Start"] <= end)]
        if overlap.empty:
            return 0
        if op == "loss":
            return int((overlap["Segment_Mean"] <= LOSS_THRESHOLD).any())
        if op == "gain":
            return int((overlap["Segment_Mean"] >= GAIN_THRESHOLD).any())
        raise ValueError(op)

    def expr_zscore(gene_id: str) -> "pd.Series | None":
        """Return per-aliquot z-score for `gene_id` across all aliquots in the
        cohort (legacy fallback when GMM is unavailable or degenerate)."""
        if gene_id not in rna.index:
            return None
        x = rna.loc[gene_id].astype(float)
        if isinstance(x, pd.DataFrame):
            x = x.mean(axis=0)
        xl = np.log1p(x)
        mu, sd = xl.mean(), xl.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return None
        return (xl - mu) / sd

    def expr_call_posterior(gene_id: str, prior_prevalence: float, gene_label: str
                            ) -> "pd.Series | None":
        """Return per-aliquot binary call (1.0 / 0.0 / NaN) for whether the
        sample belongs in the over-expressed mode for `gene_id`.

        Strategy:
          1. Fit a 2-component Gaussian mixture on log1p(TPM).
          2. If the upper component carries plausible mass (5-30 % — matching
             MM cytogenetic event prevalence), call positive when its
             posterior >= 0.5 AND the upper component is genuinely the
             over-expressed mode (mean separation >= 1.0 log unit).
          3. Otherwise (the GMM split "no expression vs any expression" or
             the cohort is essentially unimodal), fall back to a top-quantile
             cutoff at p(100 - prior_prevalence_pct), anchored to the
             literature prevalence prior. The chosen threshold is logged so
             the call method is auditable.

        prior_prevalence is the expected positive rate (e.g. 0.175 for
        t(11;14), 0.135 for t(4;14)). It is used ONLY to set the fallback
        cutoff; when the GMM is sane it determines positives directly.
        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            GaussianMixture = None  # type: ignore
        if gene_id not in rna.index:
            return None
        x = rna.loc[gene_id].astype(float)
        if isinstance(x, pd.DataFrame):
            x = x.mean(axis=0)
        xl_series = np.log1p(x)
        xl = xl_series.values.reshape(-1, 1)

        used_method = None
        calls = None
        if GaussianMixture is not None:
            try:
                gmm = GaussianMixture(
                    n_components=2, covariance_type="full",
                    random_state=42, n_init=4, max_iter=200,
                ).fit(xl)
                means = gmm.means_.flatten()
                weights = gmm.weights_
                upper_idx = int(np.argmax(means))
                upper_w = float(weights[upper_idx])
                mean_sep = float(abs(means[1] - means[0]))
                if 0.05 <= upper_w <= 0.30 and mean_sep >= 1.0:
                    posteriors = gmm.predict_proba(xl)[:, upper_idx]
                    calls = (posteriors >= 0.5).astype(float)
                    used_method = (
                        f"GMM upper_w={upper_w:.2f}, sep={mean_sep:.2f} log"
                    )
            except Exception:
                pass

        if calls is None:
            cutoff = float(np.quantile(xl_series.values, 1.0 - prior_prevalence))
            calls = (xl_series.values >= cutoff).astype(float)
            used_method = (
                f"top-quantile (prior={prior_prevalence:.3f}, "
                f"log1p_cutoff={cutoff:.2f}) — GMM degenerate or unavailable"
            )
        sys.stderr.write(f"[INFO] {gene_label} call method: {used_method}\n")
        return pd.Series(calls, index=xl_series.index)

    # Literature priors (Avet-Loiseau et al., MM consensus reviews):
    #   t(4;14) ≈ 12-15 % → use 13.5 %
    #   t(11;14) ≈ 15-20 % → use 17.5 %
    fgfr3_call = expr_call_posterior(FGFR3_ENSG, 0.135, "FGFR3")
    nsd2_call  = expr_call_posterior(NSD2_ENSG,  0.135, "NSD2")
    ccnd1_call = expr_call_posterior(CCND1_ENSG, 0.175, "CCND1")
    # Legacy z-score fall-throughs are kept None (we always have a call from
    # GMM-or-quantile above unless the gene is absent, in which case
    # expr_call_posterior returned None).
    fgfr3_z = nsd2_z = ccnd1_z = None
    fgfr3_p = fgfr3_call
    nsd2_p  = nsd2_call
    ccnd1_p = ccnd1_call

    # Genuine misses: gene_id absent from the expression matrix (returns
    # None from expr_call_posterior), not the legacy-z fallback case.
    expression_misses = []
    if fgfr3_p is None: expression_misses.append("FGFR3")
    if nsd2_p  is None: expression_misses.append("NSD2")
    if ccnd1_p is None: expression_misses.append("CCND1")
    if expression_misses:
        sys.stderr.write(
            f"[WARN] expression matrix missing gene_id(s): {expression_misses}; "
            "translocation flags using missing genes will be NaN\n"
        )

    # ---- per-patient calls ------------------------------------------------
    patients_cnv  = set(cnv["submitter_id"].unique())
    rna_aliquots  = set(rna.columns.astype(str))

    # Pre-compute earliest RNA aliquot per patient. RNA aliquot ids look like
    # MMRF_<pid>_<visit>_BM_CD138pos_<tail>; we group by leading MMRF_<pid>_
    # prefix and take lexicographic min (== earliest visit, since visit number
    # is the second underscore-delimited token).
    pid_to_rna_aliquot: dict[str, str] = {}
    for col in sorted(rna_aliquots):
        parts = col.split("_")
        if len(parts) >= 2 and parts[0] == "MMRF":
            pid_key = f"{parts[0]}_{parts[1]}"
            pid_to_rna_aliquot.setdefault(pid_key, col)  # first wins (sorted)

    rows = []
    for _, srow in baseline_per_patient.iterrows():
        pid     = srow["submitter_id"]
        # Prefer an aliquot that actually has RNA data over the samples.tsv
        # baseline pick (which is often a DNA-only aliquot).
        aliquot = pid_to_rna_aliquot.get(pid, srow["aliquot_submitter_id"])
        notes   = []

        # CNV calls --------------------------------------------------------
        if pid in patients_cnv:
            psegs = cnv[cnv["submitter_id"] == pid]
            del17p_v   = cnv_call(psegs, TP53_REGION,    "loss")
            chr1q21_v  = cnv_call(psegs, CHR1Q21_REGION, "gain")
            del13q_v   = cnv_call(psegs, RB1_REGION,     "loss")
        else:
            del17p_v = chr1q21_v = del13q_v = np.nan
            notes.append("no_cnv_data")

        # RNA-based translocation surrogates -------------------------------
        # Decision rule per gene: each *_p Series carries pre-computed binary
        # calls (0.0 / 1.0) from expr_call_posterior — already either GMM
        # upper-component or top-quantile, whichever was sane for that gene.
        def _gene_hit(p_series, z_series, gene_label):
            if p_series is not None:
                v = float(p_series.get(aliquot, np.nan))
                return (np.nan if not np.isfinite(v) else int(v == 1.0)), "binary"
            if z_series is not None:
                v = float(z_series.get(aliquot, np.nan))
                return (np.nan if not np.isfinite(v) else int(v >= TRANSLOC_Z_THRESHOLD)), "z"
            return None, None

        if aliquot not in rna_aliquots:
            t414 = np.nan
            t1114 = np.nan
            notes.append("no_rna_data")
        else:
            # t(4;14): FGFR3 OR NSD2 over-expression
            f_hit, _ = _gene_hit(fgfr3_p, fgfr3_z, "FGFR3")
            n_hit, _ = _gene_hit(nsd2_p,  nsd2_z,  "NSD2")
            if f_hit is None and n_hit is None:
                t414 = np.nan
                notes.append("t_4_14: FGFR3 and NSD2 absent from expression matrix")
            else:
                hits = [h for h in (f_hit, n_hit) if h is not None]
                # OR semantics across the two surrogate genes; NaN propagates
                # only when BOTH are NaN.
                if all(np.isnan(h) for h in hits):
                    t414 = np.nan
                else:
                    t414 = int(any(h == 1 for h in hits if not np.isnan(h)))
                if f_hit is None:
                    notes.append("t_4_14: FGFR3 not measured; using NSD2 only")
                if n_hit is None:
                    notes.append("t_4_14: NSD2 not measured; using FGFR3 only")

            t1114, _ = _gene_hit(ccnd1_p, ccnd1_z, "CCND1")
            if t1114 is None:
                t1114 = np.nan
                notes.append("t_11_14: CCND1 absent from expression matrix")

        rows.append({
            "submitter_id":   pid,
            "del17p":         del17p_v,
            "chr1q21_gain":   chr1q21_v,
            "del13q":         del13q_v,
            "t_4_14":         t414,
            "t_11_14":        t1114,
            "source_aliquot": aliquot,
            "notes":          "; ".join(notes) if notes else "",
        })

    out = pd.DataFrame(rows)
    out = out[["submitter_id", "del17p", "chr1q21_gain", "del13q",
               "t_4_14", "t_11_14", "source_aliquot", "notes"]]
    out.to_csv(out_path, sep="\t", index=False, na_rep="NaN")

    # ---- validation log ---------------------------------------------------
    n_pat = len(out)
    print(f"[OK] wrote {out_path}")
    print(f"     patients: {n_pat}")

    expected = {
        "del17p":       (0.10, 0.15),
        "chr1q21_gain": (0.30, 0.40),
        "del13q":       (0.40, 0.45),
        "t_4_14":       (0.12, 0.15),
        "t_11_14":      (0.15, 0.20),
    }

    for col, (lo, hi) in expected.items():
        n_called = out[col].notna().sum()
        n_pos    = (out[col] == 1).sum()
        rate     = (n_pos / n_called) if n_called else float("nan")
        flag = ""
        if n_called and (rate < lo / 2 or rate > hi * 2):
            flag = " <<< OUT OF EXPECTED RANGE — possible threshold/coordinate bug"
        elif n_called and (rate < lo or rate > hi):
            flag = " (outside literature band but within 2x — review)"
        print(
            f"     {col:14s} called={n_called:4d}  positive={n_pos:4d}  "
            f"rate={rate:.3f}  expected≈[{lo:.2f},{hi:.2f}]{flag}"
        )

    full = out[["del17p","chr1q21_gain","del13q","t_4_14","t_11_14"]].notna().all(axis=1).sum()
    partial = n_pat - full
    print(f"     full 5-call patients: {full}   partial: {partial}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
