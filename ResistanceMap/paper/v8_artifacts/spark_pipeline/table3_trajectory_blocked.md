# Table 3: Trajectory Forecasting -- BLOCKED

**Status**: Trajectory baselines are BLOCKED.

**Reason**: No real molecular visit timestamps are available in the
GDC open-access MMRF CoMMpass download. Trajectory forecasting
requires paired multi-omic snapshots at known calendar-time
intervals (e.g., baseline + 6-month follow-up).

**Required data**: dbGaP phs000748 controlled-access visit-level
aliquot tables, or EGA longitudinal scRNA/multiome.

**Blocked methods**: PRESCIENT, scNODE, TrajectoryNet, CellRank 2
cannot run on fabricated visit_time_days.
