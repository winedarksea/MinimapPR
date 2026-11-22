"""Structural JSON comparison for the cross-language parity gates.

Exact for decisions (strings, booleans, integers, None, list lengths, keys);
floats within ``relative_tolerance`` of the larger of the value and its
enclosing array's largest magnitude, so near-zero components do not demand
impossible relative precision.
"""

from __future__ import annotations

import math


def largest_magnitude(value) -> float:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return 0.0
    if isinstance(value, (int, float)):
        return abs(float(value)) if math.isfinite(float(value)) else 0.0
    if isinstance(value, list):
        return max((largest_magnitude(item) for item in value), default=0.0)
    return 0.0


def assert_json_matches(actual, expected, label: str, relative_tolerance: float, scale: float | None = None) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), f"{label}: keys {actual!r} vs {expected!r}"
        for key in expected:
            assert_json_matches(actual[key], expected[key], f"{label}.{key}", relative_tolerance)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{label}: {actual!r} vs {expected!r}"
        scale = largest_magnitude(expected) if scale is None else scale
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_json_matches(left, right, f"{label}[{index}]", relative_tolerance, scale)
        return
    if isinstance(expected, float) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), f"{label}: {actual!r} vs {expected!r}"
        tolerance = relative_tolerance * max(abs(expected), scale or 0.0)
        assert abs(float(actual) - expected) <= tolerance, f"{label}: {actual!r} vs {expected!r}"
        return
    if isinstance(expected, int) and not isinstance(expected, bool) and isinstance(actual, float):
        assert actual == expected, f"{label}: {actual!r} vs {expected!r}"
        return
    assert actual == expected, f"{label}: {actual!r} vs {expected!r}"
