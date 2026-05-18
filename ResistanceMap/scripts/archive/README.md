# `scripts/archive/`

Scripts archived during the v18 cleanup. They are preserved here because:

  1. They may still be referenced by historical RUNS.md rows we want to
     reproduce, AND
  2. They have a clear v17/v18 successor that should be used for new work.

| Archived script | Successor |
|-----------------|-----------|
| `mortfm_smoke_test.py` | `scripts/mortfm_v17_unified_smoke.py` (also wired as a pytest in `tests/mortfm/test_mortfm_trainer_smoke.py`) |
| `mortfm_lens_smoke_test.py` | `scripts/mortfm_v17_unified_smoke.py` (LENS + legacy paths exercised together) |
| `mortfm_causal_evidence_report.py` | `scripts/mortfm_causal_evidence_report_v2.py` |

Do not invoke these for new training or paper-ready runs. If you need to
reproduce a RUNS.md row that names one of them, run the archived path
directly — Python imports inside still resolve from the project root.
