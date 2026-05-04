"""S4 F10 supplemental: tipping-point sensitivity beyond E-value.

E-value gives a single summary of confounding sensitivity but reduces a 2-D
problem (RR_AU on the A→U arm, RR_UY on the U→Y arm, where U is unmeasured)
to a 1-D minimum-joint-strength bound. The tipping-point analysis shows the
full 2-D contour: for each (RR_AU, RR_UY) pair, does that strength of
unmeasured confounding nullify the lower CI bound NIE?

VanderWeele-Ding bias formula:
    bias_factor(RR_AU, RR_UY) = (RR_AU * RR_UY) / (RR_AU + RR_UY - 1)
    NIE_corrected(RR_AU, RR_UY) = NIE_observed / bias_factor

Tipping point: the (RR_AU, RR_UY) curve where NIE_corrected = 1 (null).

For the focused mediator's lower-CI-bound NIE HR = 1.029:
    we need bias_factor ≥ 1.029 to nullify
    Tipping at RR_AU = 1.5: requires RR_UY ≥ ?
    Tipping at RR_AU = 2.0: requires RR_UY ≥ ?
    Tipping at RR_AU = 3.0: requires RR_UY ≥ ?

If even moderate joint strengths (e.g., RR_AU = 1.5, RR_UY = 1.5) ARE enough
to nullify, then F10 is genuinely unprotected. If only implausible strengths
(e.g., RR_AU = 3, RR_UY = 3) suffice, F10 has more nuance than the E-value
suggests.

Output: paper/v8_artifacts/v10_sprint4/nie_tipping_point.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPRINT4 = ROOT / "paper" / "v8_artifacts" / "v10_sprint4"


def bias_factor(rr_au: float, rr_uy: float) -> float:
    return (rr_au * rr_uy) / (rr_au + rr_uy - 1)


def required_rr_uy_to_nullify(observed_rr: float, rr_au: float) -> float:
    """Solve bias_factor(rr_au, rr_uy) = observed_rr for rr_uy.

    bias_factor = (rr_au * rr_uy) / (rr_au + rr_uy - 1) = observed_rr
        rr_au * rr_uy = observed_rr * (rr_au + rr_uy - 1)
        rr_au * rr_uy - observed_rr * rr_uy = observed_rr * (rr_au - 1)
        rr_uy * (rr_au - observed_rr) = observed_rr * (rr_au - 1)
        rr_uy = observed_rr * (rr_au - 1) / (rr_au - observed_rr)
    """
    if rr_au <= observed_rr:
        return float("inf")  # Can't nullify with this rr_au
    return observed_rr * (rr_au - 1) / (rr_au - observed_rr)


def main() -> None:
    print("=== S4 F10 tipping-point sensitivity ===")
    nie = json.loads((SPRINT4 / "nie_focused_falsification.json").read_text())
    nie_hr_lo = nie["evalue_sensitivity"]["NIE_HR_lower_CI"]
    nie_hr_pt = nie["evalue_sensitivity"]["NIE_HR_point"]
    print(f"focused NIE HR point: {nie_hr_pt:.3f}")
    print(f"focused NIE HR lower CI: {nie_hr_lo:.3f}")

    rr_grid = [1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
    grid = []
    for rr_au in rr_grid:
        rr_uy_lo = required_rr_uy_to_nullify(nie_hr_lo, rr_au)
        rr_uy_pt = required_rr_uy_to_nullify(nie_hr_pt, rr_au)
        grid.append({
            "rr_au": rr_au,
            "rr_uy_to_nullify_lower_CI": rr_uy_lo,
            "rr_uy_to_nullify_point": rr_uy_pt,
        })
        print(f"  RR_AU = {rr_au:.2f}  →  RR_UY to nullify lower-CI: "
              f"{rr_uy_lo:.3f}   to nullify point: {rr_uy_pt:.3f}")

    # Plausibility benchmarks for MM
    benchmarks = {
        "ISS_stage_III_vs_I  (RR ≈ 2-3 for OS, ≈1.8-2.5 for TT2L)": "moderate-strong",
        "del17p positive  (RR ≈ 1.5-2)": "moderate",
        "del17p × ISS interaction": "potentially RR ≈ 2.5",
        "high-LDH  (RR ≈ 1.3-1.7)": "modest",
        "performance-status ECOG ≥2 (RR ≈ 1.4-2 in MM)": "moderate",
        "frailty (RR ≈ 1.2-1.5)": "modest",
    }

    # Joint confounder magnitude needed to fully explain NIE = 1.029 (lower CI):
    #   minimum joint = E-value at lower bound = 1.20
    #   tipping at RR_AU = 1.5 needs RR_UY = 1.029 * 0.5 / 0.471 = 1.092
    #   tipping at RR_AU = 2.0 needs RR_UY = 1.029 * 1.0 / 0.971 = 1.060
    #
    # With realistic medical confounders (ISS, del17p, LDH, performance status)
    # routinely showing RR ≥ 1.3 on both arms, the lower-CI NIE is *easily*
    # nullifiable. This confirms the strict-FAIL F10 verdict.
    #
    # Nevertheless, the F8 (CI excludes zero) and F9 (specificity) gates pass,
    # so the directional finding is preserved as supporting evidence.

    summary = {
        "design": "Tipping-point sensitivity for unmeasured-confounder strength on (A→U, U→Y) arms",
        "NIE_HR_point": nie_hr_pt,
        "NIE_HR_lower_CI": nie_hr_lo,
        "tipping_grid": grid,
        "MM_specific_benchmarks": benchmarks,
        "interpretation": (
            "The NIE HR lower-CI bound of 1.029 can be nullified by an unmeasured "
            "confounder with strengths (RR_AU, RR_UY) jointly as small as (1.5, 1.10) "
            "or (2.0, 1.06). Standard MM prognostic factors (ISS, del17p, LDH, "
            "performance status) routinely display joint strengths ≥ (1.3, 1.3), "
            "which is sufficient to explain the lower-CI NIE. The F10 strict-fail "
            "is a *correct* expression of this vulnerability. F8+F9 pass establishes "
            "directional/specific evidence; F10 fail establishes that we cannot "
            "rule out a moderate unmeasured confounder explanation."
        ),
        "actionable_conclusion": (
            "L2 NIE direction (proteasome → TT2L mediation) is detected and "
            "specific (F8, F9 PASS) but is sensitivity-limited (F10 FAIL); the "
            "v10 paper claim stays at L1-with-structural-prior with this NIE "
            "as supporting directional evidence, exactly as pre-declared in spec "
            "§11 abstract."
        ),
    }
    out_path = SPRINT4 / "nie_tipping_point.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved → {out_path}")
    print("\n=== Final F10 reading: structural REFUTATION not fixable by mediator construction ===")


if __name__ == "__main__":
    main()
