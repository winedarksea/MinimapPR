"""Lenient value coercion for Rust DSP manifest fields.

Legacy manifest fields are optional diagnostics: a malformed value degrades to
``None`` rather than failing the render. Contract-bearing fields
(``solve_provenance``) are parsed strictly in ``rust_localization_solve_contract``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_nonempty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def optional_nonnegative_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def optional_finite_float(value: Any) -> float | None:
    """Coerce to a finite float allowing negatives (e.g. dBFS received levels)."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def optional_bounded_float(value: Any) -> float | None:
    parsed = optional_nonnegative_float(value)
    if parsed is None or parsed > 1.0:
        return None
    return parsed


def coerce_covariance_matrix(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            return None
        try:
            parsed_row = [float(cell) for cell in row]
        except (TypeError, ValueError):
            return None
        if any(not np.isfinite(cell) for cell in parsed_row):
            return None
        matrix.append(parsed_row)
    return matrix


def coerce_vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        parsed = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    if any(not np.isfinite(component) for component in parsed):
        return None
    return parsed  # type: ignore[return-value]


def coerce_pair_tdoas(value: Any) -> list[dict] | None:
    if not isinstance(value, list) or not value:
        return None
    pairs: list[dict] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        try:
            pair = {
                "ch_a": int(entry["ch_a"]),
                "ch_b": int(entry["ch_b"]),
                "lag_seconds": float(entry["lag_seconds"]),
                "confidence": float(entry.get("confidence", 0.0)),
            }
            # Contract additions (docs/localization_solve_contract.md section 8);
            # omitted by the sidecar while unset.
            if isinstance(entry.get("is_cross_node"), bool):
                pair["is_cross_node"] = entry["is_cross_node"]
            if entry.get("timing_uncertainty_seconds") is not None:
                pair["timing_uncertainty_seconds"] = float(entry["timing_uncertainty_seconds"])
            pairs.append(pair)
        except (KeyError, TypeError, ValueError):
            continue
    return pairs or None
