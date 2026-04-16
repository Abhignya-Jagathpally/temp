# ResistanceMap v3-remediation — file manifest

New and modified code to close the 12 remediation sections in
`ResistanceMap_Remediation_Plan.html`. Drop-in over the `v3-fixes` tree.

## New source modules

| File | Section | Purpose |
|------|---------|---------|
| `resistancemap/_facts.py` | §1.1 | Single source of truth for sizes/topology |
| `resistancemap/data/splits.py` | §2.1, §4.1 | Hierarchical group splits (patient/lineage/drug) |
| `resistancemap/data/targets.py` | §3.2 | Target registry with units + primary metric |
| `resistancemap/evaluation/metrics/per_drug.py` | §3.1 | Per-drug bootstrap CIs + pooled contract |
| `resistancemap/evaluation/baselines.py` | §5.1 | Required baseline panel |
| `resistancemap/evaluation/calibration.py` | §7.1, §7.2 | Empirical coverage + conformal intervals |
| `resistancemap/evaluation/ablation_harness.py` | §6.1 | Frozen-upstream ablations + paired bootstrap |
| `resistancemap/models/rl/ope.py` | §8.1 | Off-policy evaluation gate (WIS + DR + positivity) |
| `resistancemap/infrastructure/guardrails.py` | §12.1 | 8 executable guardrails |

## Tests

| File | Covers |
|------|--------|
| `tests/test_leakage.py` | §4.1 leakage audit harness |
| `tests/test_metrics_and_calibration.py` | §3.1, §7.1, §7.2 |
| `tests/test_ope_and_guardrails.py` | §8.1, §12.1 |

## Scripts

| File | Purpose |
|------|---------|
| `scripts/check_docs_consistent.py` | §1.1 CI doc consistency check (forbids "Adams-Bashforth", NIST ZTA claim, PPI drift) |
| `scripts/verify_release.sh` | §11.2 concatenate + SHA256 verify + list |
| `reproduce.sh` | §11.3 one-command headline reproducer |

## Docs / release

| File | Purpose |
|------|---------|
| `release/manifest.json` | §11.1 per-checkpoint stage/SHA/cohort mapping |
| `RUNS.md` | §1.3 run ledger; every quoted number cites a row |
| `README_PATCHES.md` | §1.2 diffs to apply to README.md |
| `PIPELINE_INTEGRATION.md` | Integration diffs into existing v3-fixes code |

## Verification

```bash
cd ResistanceMap_v3_remediation
python -m py_compile $(find resistancemap -name "*.py")
python -m pytest tests/ -v
python scripts/check_docs_consistent.py ../ResistanceMap
```

## Apply order

1. Copy `resistancemap/` subtree over your `v3-fixes` checkout.
2. Copy `tests/` additions in.
3. Copy `scripts/` and `reproduce.sh`.
4. Apply diffs from `README_PATCHES.md` and `PIPELINE_INTEGRATION.md`.
5. Run `pytest tests/ -v` — all must pass.
6. Run `python scripts/check_docs_consistent.py` — must exit 0.
7. Populate `RUNS.md` with real W&B / log links before quoting any number publicly.