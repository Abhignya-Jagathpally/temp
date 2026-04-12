"""Tests for the evaluation Charter and Rubric.

The Charter encodes the three testable scientific claims for ResistanceMap
(state, when, pathway) along with the explicit epistemic limits of any
"before it happens" claim made on cross-sectional data.

The Rubric stores agent x criterion x weight cells with a deterministic
score() that applies weights multiplicatively, plus a YAML round-trip.
"""

from __future__ import annotations

import pytest

evaluation = pytest.importorskip(
    "resistancemap.evaluation",
    reason="resistancemap.evaluation package not yet merged into worktree",
)


# ---------------------------------------------------------------------------
# Charter
# ---------------------------------------------------------------------------


def _claim_name(claim) -> str:
    """Pull the canonical name from a Claim object regardless of attribute."""
    for attr in ("name", "id", "key"):
        if hasattr(claim, attr):
            value = getattr(claim, attr)
            if isinstance(value, str):
                return value.lower()
    raise AssertionError(f"Claim has no name/id/key attribute: {claim!r}")


def test_default_charter_has_three_claims():
    from resistancemap.evaluation import load_default_charter

    charter = load_default_charter()
    claims = list(charter.claims)
    assert len(claims) == 3, f"expected 3 claims, got {len(claims)}"

    names = {_claim_name(c) for c in claims}
    expected = {"state", "when", "pathway"}
    assert expected.issubset(names) or names == expected, (
        f"claim names {names} should match {expected}"
    )


def test_charter_documents_prospective_epistemic_limit_for_when_claim():
    """The 'before it happens' caveat must be present, in writing."""
    from resistancemap.evaluation import load_default_charter

    charter = load_default_charter()
    # Concatenate all epistemic_limits text on every claim, normalize.
    pieces: list[str] = []
    for claim in charter.claims:
        limits = getattr(claim, "epistemic_limits", None)
        if limits is None:
            continue
        if isinstance(limits, (list, tuple)):
            pieces.extend(str(x) for x in limits)
        else:
            pieces.append(str(limits))
    blob = " ".join(pieces).lower()

    assert "prospective" in blob or "time-ordered" in blob or "time ordered" in blob, (
        "Charter epistemic_limits must explicitly mention prospective / time-ordered "
        f"data; got: {blob!r}"
    )


def test_every_claim_has_at_least_one_falsifier():
    from resistancemap.evaluation import load_default_charter

    charter = load_default_charter()
    for claim in charter.claims:
        falsifiers = getattr(claim, "falsifiers", None)
        assert falsifiers is not None, f"Claim {_claim_name(claim)} has no falsifiers"
        falsifier_list = list(falsifiers)
        assert len(falsifier_list) >= 1, (
            f"Claim {_claim_name(claim)} must declare at least one falsifier"
        )


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------


def test_rubric_score_applies_weights_multiplicatively():
    """A weighted criterion's contribution must equal score * weight."""
    from resistancemap.evaluation import Rubric

    spec = {
        "agents": {
            "data_adequacy": {
                "criteria": {
                    "coverage": {"weight": 2.0},
                    "balance": {"weight": 0.5},
                }
            }
        }
    }
    rubric = Rubric.from_dict(spec) if hasattr(Rubric, "from_dict") else Rubric(spec)

    # Score with raw criterion outcomes 1.0 and 0.0
    raw = {"data_adequacy": {"coverage": 1.0, "balance": 0.0}}
    weighted = rubric.score(raw)

    # Weighted-sum / weight-sum convention: (1*2 + 0*0.5) / (2 + 0.5) = 0.8
    expected = (1.0 * 2.0 + 0.0 * 0.5) / (2.0 + 0.5)
    actual = (
        weighted["data_adequacy"]
        if isinstance(weighted, dict)
        else weighted
    )
    assert actual == pytest.approx(expected, abs=1e-9)


def test_rubric_yaml_roundtrip_preserves_structure(tmp_path):
    from resistancemap.evaluation import Rubric

    spec = {
        "agents": {
            "data_adequacy": {
                "criteria": {
                    "coverage": {"weight": 1.0, "evidence_type": "report"},
                    "balance": {"weight": 0.5, "evidence_type": "table"},
                }
            },
            "calibration": {
                "criteria": {"ece": {"weight": 1.0, "evidence_type": "metric"}}
            },
        }
    }
    original = Rubric.from_dict(spec) if hasattr(Rubric, "from_dict") else Rubric(spec)

    path = tmp_path / "rubric.yaml"
    original.to_yaml(path)
    assert path.exists()

    reloaded = Rubric.from_yaml(path)
    # Round-trip equality is checked at the structural (dict) level so we do
    # not couple to a particular __eq__ implementation.
    if hasattr(reloaded, "to_dict") and hasattr(original, "to_dict"):
        assert reloaded.to_dict() == original.to_dict()
    else:
        # Fallback: hit the same scoring on a known input.
        raw = {
            "data_adequacy": {"coverage": 1.0, "balance": 1.0},
            "calibration": {"ece": 0.5},
        }
        assert reloaded.score(raw) == original.score(raw)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
