"""
resistancemap.legacy.v15_planned_mortfm
========================================
The v15 "planned MORT-FM" tree — TrajectorySampler + 7 task heads
(resistance state, time-to-resistance, trajectory distribution, pathway
route, drug-specific risk, evidential uncertainty, counterfactual
intervention) that predated the v17 LENS unification.

The canonical v18 ``forward()`` does NOT use these. The legacy
``forward_legacy()`` path still does, so they are kept as importable
modules under this namespaced location.
"""
