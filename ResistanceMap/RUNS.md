# Run Ledger

Every number quoted in `README.md` or `ARCHITECTURE.md` must cite a row in this
table. If a row has no W&B link or persistent log, the number it produced must
not appear in the documentation.

## Run index

| Run ID              | Date       | Hardware    | Data fraction       | Stages run   | Wall-time | W&B | Notes                                  |
|---------------------|------------|-------------|---------------------|--------------|-----------|-----|----------------------------------------|
| r-2026-04-14-smoke  | 2026-04-14 | 2x A100 40GB| 10% public subsample| all          | 21.3 min  | TBD | Populates README "Latest run" line     |
| r-2026-03-02-full   | 2026-03-02 | 1x H100 80GB| 100%                | all          | 26.9 h    | TBD | Populates README Performance table     |
| r-2026-03-05-ddp8   | 2026-03-05 | 8x H100     | 100%                | all          | 4.1 h     | TBD | Distributed-training reference         |

## Required artifacts per row

1. `logs/<run-id>/stdout.log` and `logs/<run-id>/stderr.log`
2. `logs/<run-id>/config.yaml` (frozen config at launch)
3. `logs/<run-id>/verification_chain.json` (SHA256 per stage)
4. `logs/<run-id>/per_drug_metrics.csv` (required for any AUROC quoted)
5. W&B run URL (or MLflow tracking URI)

## Rule

Metrics without a complete artifact bundle may NOT be copied into docs. The CI
check `scripts/check_docs_consistent.py` greps README/ARCHITECTURE for numeric
claims and refuses the build if they cannot be traced to a row here.
