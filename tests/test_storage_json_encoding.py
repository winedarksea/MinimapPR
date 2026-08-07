"""Stored JSON columns must be parseable by strict (non-Python) JSON readers."""

from __future__ import annotations

import json

import pytest

from minimappr.storage.db import _json_dumps


def _strict_loads(raw: str):
    """Parse rejecting Infinity/NaN, the way serde_json and JSON.parse do."""

    def _reject(constant: str):
        raise ValueError(f"non-JSON constant {constant!r}")

    return json.loads(raw, parse_constant=_reject)


def test_non_finite_floats_are_encoded_as_null() -> None:
    """Omni detections carry ``gdop=inf``, which json.dumps writes as ``Infinity``.

    That token is a Python extension, not JSON, so the Rust frontend and sidecar
    (serde_json) and any browser ``JSON.parse`` reject the whole blob. On a live
    site 1794 of 1842 stored ``feature_summary_json`` values contained it.
    """
    raw = _json_dumps({"gdop": float("inf"), "residual": float("nan")})

    assert _strict_loads(raw) == {"gdop": None, "residual": None}


def test_non_finite_floats_are_scrubbed_when_nested() -> None:
    """gdop arrives nested under branch_evidence, not just at the top level."""
    raw = _json_dumps(
        {
            "branch_evidence": {"omni": {"gdop": float("inf"), "confidence": 0.25}},
            "covariance": [[1.0, float("-inf")], [0.0, 2.0]],
        }
    )

    assert _strict_loads(raw) == {
        "branch_evidence": {"omni": {"gdop": None, "confidence": 0.25}},
        "covariance": [[1.0, None], [0.0, 2.0]],
    }


def test_finite_values_round_trip_unchanged() -> None:
    payload = {"a": 1.5, "b": [1, 2, 3], "c": "text", "d": None, "e": True}

    assert _strict_loads(_json_dumps(payload)) == payload


def test_encoder_refuses_rather_than_emitting_python_constants() -> None:
    """allow_nan=False is the backstop if a future value slips past the scrub."""
    with pytest.raises(ValueError):
        json.dumps({"x": float("inf")}, allow_nan=False)
