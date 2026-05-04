"""V10 sprint orchestrator — runs the v10 paper-spec pipeline end-to-end.

Stages are named semantically (what they DO), not numerically. The mapping
between semantic names and the original sprint numbers is preserved in
the docstrings and verdict docs.

Stage names (in dependency order):
    landscape_dsm        : Sprint 1 — U_θ scalar potential via DSM + PPI Tikhonov; F4/F5 gates
    hbayes_strata        : Sprint 2 — full-channel STRING + 5-stratum hierarchical Bayes; F1/F2/F5-paired gates
    propagation_rwr      : Sprint 3 — Random Walk with Restart driver pathway + NPI z-score + CRISPR oracle; F3/F6/F7 + v2/v3 strict-spec hardening
    mediation_nie        : Sprint 4 — Lange-Hansen Cox NIE for proteasome → time-to-2nd-line Bortezomib; F8/F9/F10 gates + focused-mediator falsification
    conformal_mondrian   : Sprint 5 — Barber-Candès-Ramdas-Tibshirani 2021 jackknife+ with 5-cyto-strata partition; F_S5 per-stratum coverage
    beataml_xdisease     : Sprint 6 — Beat AML 1.0 (Tyner 2018) integration as second hematologic disease; cross-disease F3 replication
    dps_singlesnapshot   : Sprint 7 — Single-snapshot diffusion-posterior wrapper with U_θ as prior; F8 LOO energy-distance gate
    fixes                : F7-Vorinostat (PRISM oracle) + F10 (richer mediators + tipping-point) — structural-failure fix attempts

Usage:
    python main.py --v10-sprint all               # full S1 → S7 pipeline
    python main.py --v10-sprint propagation_rwr   # just Sprint 3
    python main.py --v10-sprint fixes             # just the fix attempts

Each stage's logs are written to logs/v10_e2e/<stage_name>.log; final timing
and exit-code summary is printed to stdout.

Raw data prerequisites (already on disk; verified by validate_data_files in
the legacy main()):
    data/raw/mmrf_commpass/                        — MMRF CoMMpass IA22
    data/raw/string/                                — STRING v12
    data/raw/depmap/CRISPRGeneEffect.csv           — DepMap Chronos
    data/raw/ccle_proteomics.csv                    — CCLE proteomics
    data/raw/prism/secondary-screen-...csv          — PRISM Repurposing
    data/raw/beataml/tyner2018_supplement.xlsx      — Beat AML 1.0 (Tyner 2018)
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]


# Semantic stage name → ordered list of script-arg lists (each entry is the
# CLI invocation tail; "scripts/v10/sNx_..." is implied to be relative to ROOT)
V10_STAGES: list[tuple[str, list[list[str]]]] = [
    ("landscape_dsm", [
        ["scripts/v10/s1a_build_mmrf_baseline_matrix.py"],
        ["scripts/v10/s1c_encode_mmrf_z64.py"],
        ["scripts/v10/s1b_build_ppi_laplacian.py", "--tag", "v10s1"],
        # s1d uses --laplacian-suffix for input PPI path (separate from --tag
        # which is the output-checkpoint name); s1b emits `ppi_*_v10s1.{npz,tsv}`
        # so the suffix is "_v10s1"
        ["scripts/v10/s1d_train_scalar_potential.py", "--tag", "v10s1",
         "--laplacian-suffix", "_v10s1"],
        ["scripts/v10/s1e_falsification_gates.py", "--tag", "v10s1",
         "--sprint-dir", "v10_sprint1"],
    ]),
    ("hbayes_strata", [
        ["scripts/v10/s2b_paired_patients.py"],
        ["scripts/v10/s2c_encode_paired.py"],
        ["scripts/v10/s1b_build_ppi_laplacian.py", "--tag", "v10s2"],
        ["scripts/v10/s1d_train_scalar_potential.py", "--tag", "v10s2",
         "--laplacian-suffix", "_v10s2"],
        ["scripts/v10/s1e_falsification_gates.py", "--tag", "v10s2",
         "--sprint-dir", "v10_sprint2"],
        ["scripts/v10/s2d_f5_paired.py", "--tag", "v10s2"],
        ["scripts/v10/s2e_hbayes_inference.py"],
        ["scripts/v10/s2f_f1_f2_gates.py"],
    ]),
    ("propagation_rwr", [
        # Setup: PPI adjacency + DepMap haem subset + Chronos rank-sum oracle
        ["scripts/v10/s3a_build_ppi_adjacency.py"],
        ["scripts/v10/s3prep_depmap_haem_subset.py"],
        ["scripts/v10/s3prep_crispr_oracle.py"],
        # Original v1 pipeline (preserved for audit)
        ["scripts/v10/s3c_drug_seed_propagation.py"],
        ["scripts/v10/s3d_f3_crispr_oracle.py"],
        ["scripts/v10/s3e_f6_driver_recall.py"],
        ["scripts/v10/s3f_f7_multiseed.py"],
        # v2/v3 strict-spec hardening (final paper-grade results)
        ["scripts/v10/s3fix_v2.py"],
        ["scripts/v10/s3fix_v3.py"],
        ["scripts/v10/s3fix_v3_f7.py"],
    ]),
    ("mediation_nie", [
        ["scripts/v10/s4a_build_outcome_treatment.py"],
        ["scripts/v10/s4b_build_mediator.py"],
        ["scripts/v10/s4c_merge_analysis_table.py"],
        ["scripts/v10/s4d_nie_estimation.py"],
        ["scripts/v10/s4e_falsification_neg_control.py"],
        ["scripts/v10/s4e_sensitivity_alt_mediators.py"],
        ["scripts/v10/s4e_v2_focused_neg_control.py"],
    ]),
    ("conformal_mondrian", [
        ["scripts/v10/s5_mondrian_jackknife_plus.py"],
    ]),
    ("beataml_xdisease", [
        ["scripts/v10/s6a_extract_beataml.py"],
        ["scripts/v10/s6b_cross_disease_f3.py"],
        ["scripts/v10/s6b_v2_aml_native.py"],
        ["scripts/v10/s6c_conformal_beataml.py"],
    ]),
    ("dps_singlesnapshot", [
        ["scripts/v10/s7_diffusion_posterior.py"],
    ]),
    ("fixes", [
        ["scripts/v10/s3_fix_f7_prism_oracle.py"],
        ["scripts/v10/s4_fix_f10_richer_mediator.py"],
        ["scripts/v10/s4_fix_f10_tipping_point.py"],
    ]),
]


VALID_STAGE_NAMES = ["all", *(n for n, _ in V10_STAGES)]


def _run_one_script(args: Sequence[str], log_path: Path) -> int:
    """Run a single python script with stdout+stderr appended to log_path.

    Uses unbuffered Python for live log streaming. Returns exit code.
    """
    cmd = [sys.executable, "-u", *args]
    with log_path.open("a") as f:
        f.write(f"\n========== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"$ {shlex.join(cmd)} ==========\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def run_v10_e2e(stage_name: str, log_dir: Path | None = None) -> int:
    """Run a v10 stage (or 'all'). Returns 0 on success, non-zero otherwise."""
    log = logging.getLogger("v10_runner")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s",
                                          datefmt="%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)

    if stage_name not in VALID_STAGE_NAMES:
        log.error(f"Unknown v10 stage '{stage_name}'. Valid: {VALID_STAGE_NAMES}")
        return 2

    log_dir = Path(log_dir) if log_dir else ROOT / "logs" / "v10_e2e"
    log_dir.mkdir(parents=True, exist_ok=True)

    if stage_name == "all":
        stages_to_run = V10_STAGES
    else:
        stages_to_run = [(n, s) for n, s in V10_STAGES if n == stage_name]

    summary: list[tuple[str, int, float, int]] = []
    for name, scripts in stages_to_run:
        log_path = log_dir / f"{name}.log"
        log_path.write_text("")  # truncate prior runs
        log.info(f"=== v10 stage: {name}  ({len(scripts)} script(s))  →  {log_path} ===")
        t0 = time.time()
        n_ok = 0
        rc_total = 0
        for args in scripts:
            log.info(f"  $ python -u {' '.join(args)}")
            rc = _run_one_script(args, log_path)
            if rc == 0:
                n_ok += 1
            else:
                log.error(f"  exit {rc} on {args[0]}; aborting stage '{name}'")
                rc_total = rc
                break
        elapsed = time.time() - t0
        summary.append((name, rc_total, elapsed, n_ok))
        if rc_total != 0:
            log.error(f"=== Stage '{name}' FAILED ({n_ok}/{len(scripts)} scripts ok in {elapsed:.0f}s) ===")
            break
        log.info(f"=== Stage '{name}' OK ({n_ok}/{len(scripts)} scripts in {elapsed:.0f}s) ===")

    # Summary table
    log.info("─" * 70)
    log.info(f"{'STAGE':<22}{'EXIT':>8}{'WALL':>10}{'SCRIPTS_OK':>14}")
    log.info("─" * 70)
    for name, rc, elapsed, n_ok in summary:
        log.info(f"{name:<22}{rc:>8}{elapsed:>9.1f}s  {n_ok:>10}")
    overall = 0 if all(rc == 0 for _, rc, _, _ in summary) else 1
    log.info("─" * 70)
    log.info(f"OVERALL: {'PASS' if overall == 0 else 'FAIL'}  "
              f"(stages: {len(summary)}/{len(stages_to_run)})")
    return overall


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=VALID_STAGE_NAMES)
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()
    sys.exit(run_v10_e2e(args.stage, log_dir=args.log_dir))
