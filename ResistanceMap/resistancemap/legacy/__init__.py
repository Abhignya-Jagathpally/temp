"""
resistancemap.legacy
=====================
Archived modules from prior MORT-FM development phases (v8 / v15 / v16).
These are preserved for:
  * loading historical checkpoints that reference these classes by path,
  * running ``forward_legacy()`` on the canonical v18 model,
  * reproducing RUNS.md rows that exercised these modules.

NEW patient-level training MUST go through ``resistancemap.mortfm`` —
the canonical surface — not through these legacy heads/dynamics.
"""
