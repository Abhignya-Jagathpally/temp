"""tests/mortfm/test_intervention_graph_determinism.py

Regression test for Bug B3: the per-edge basis-vector seed must be
deterministic across processes. The previous implementation used Python's
process-randomised ``hash()`` which (without ``PYTHONHASHSEED=0``) returns
different values in every subprocess and silently destroyed the
reproducibility of every causal counterfactual run.

The fix uses SHA-256 truncated to 64 bits — deterministic by definition.
"""

from __future__ import annotations

import subprocess
import sys

import torch

from resistancemap.mortfm.causal.intervention_graph import (
    _deterministic_edge_seed,
    deterministic_edge_basis,
)


def test_same_process_same_eid_returns_equal_seed():
    eid = "P00533::P42345::activates"
    s1 = _deterministic_edge_seed(eid)
    s2 = _deterministic_edge_seed(eid)
    assert s1 == s2


def test_same_process_same_eid_returns_equal_basis():
    eid = "P00533::P42345::activates"
    v1 = deterministic_edge_basis(eid, d_graph=16)
    v2 = deterministic_edge_basis(eid, d_graph=16)
    assert torch.equal(v1, v2)


def test_different_eid_returns_different_seed():
    s1 = _deterministic_edge_seed("a::b::c")
    s2 = _deterministic_edge_seed("a::b::d")
    assert s1 != s2


def test_subprocess_returns_same_seed_as_parent():
    """The decisive cross-process check: spawn a fresh Python process and
    confirm SHA-256 returns the identical 64-bit integer.

    With the old ``hash()`` implementation this would be flaky / wrong
    because PYTHONHASHSEED randomises hash() per process.
    """
    eid = "P00533::P42345::activates"
    parent_seed = _deterministic_edge_seed(eid)

    snippet = (
        "from resistancemap.mortfm.causal.intervention_graph "
        "import _deterministic_edge_seed; "
        f"print(_deterministic_edge_seed({eid!r}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
    )
    child_seed = int(result.stdout.strip())
    assert child_seed == parent_seed, (
        f"Seed differs across processes: parent={parent_seed} child={child_seed}. "
        "This indicates the hash randomization bug has regressed."
    )


def test_subprocess_returns_same_basis_as_parent():
    """End-to-end: the basis vector itself must be reproducible across processes."""
    eid = "abc::def::regulates"
    parent_vec = deterministic_edge_basis(eid, d_graph=8)

    snippet = (
        "import torch; "
        "from resistancemap.mortfm.causal.intervention_graph "
        "import deterministic_edge_basis; "
        f"v = deterministic_edge_basis({eid!r}, d_graph=8); "
        "print(','.join(f'{x:.10f}' for x in v.tolist()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
    )
    child_vec = torch.tensor([float(x) for x in result.stdout.strip().split(",")])
    assert torch.allclose(parent_vec, child_vec, atol=1e-9)
