"""Numerical parity: Rust ``minimappr_core::tdoa_solver`` vs Python goldens.

Goldens (``tests/fixtures/tdoa_solver_parity_goldens.npz``) come from Python
``solve_cartesian_tdoa`` / ``fuse_bearings`` via
``python -m tests.tdoa_solver_parity_cases``. The Rust side runs through the
sidecar's hidden ``tdoa-solver-oracle`` subcommand. Tolerance is 1e-5
relative; each array also gets an absolute floor of 1e-5 × its largest
magnitude so near-zero components (off-diagonals, a coordinate at 0) do not
demand impossible relative precision.

A second test re-runs Python against the same goldens so a server-side
change to the solver shows up here instead of as a mysterious Rust failure.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from tests.rust_sidecar_binary import build_rust_sidecar_binary, cargo_is_available
from tests.tdoa_solver_parity_cases import (
    CARTESIAN_SCALAR_NAMES,
    FUSION_SCALAR_NAMES,
    GOLDEN_PATH,
    run_python_case,
)

RELATIVE_TOLERANCE = 1.0e-5


def _load_goldens() -> tuple[list[dict], dict[str, np.ndarray]]:
    with np.load(GOLDEN_PATH) as archive:
        arrays = {key: archive[key] for key in archive.files}
    cases = [json.loads(str(raw)) for raw in arrays["case_inputs_json"]]
    return cases, arrays


def _none_to_inf(value: float | None) -> float:
    return float("inf") if value is None else float(value)


def _assert_close(actual, expected, label: str) -> None:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    finite_expected = expected_array[np.isfinite(expected_array)]
    scale = float(np.max(np.abs(finite_expected))) if finite_expected.size else 0.0
    np.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=RELATIVE_TOLERANCE,
        atol=RELATIVE_TOLERANCE * scale,
        err_msg=label,
    )


def _axisymmetric_invariants(position_m, invariant: dict) -> np.ndarray:
    """(coordinate along the axis, distance from the axis) for a symmetric layout."""
    axis_point = np.asarray(invariant["axis_point_m"], dtype=np.float64)
    axis_direction = np.asarray(invariant["axis_direction"], dtype=np.float64)
    offset = np.asarray(position_m, dtype=np.float64) - axis_point
    along = float(np.dot(offset, axis_direction))
    return np.asarray([along, float(np.linalg.norm(offset - along * axis_direction))])


def _compare_result(index: int, case: dict, result: dict, arrays: dict[str, np.ndarray]) -> None:
    label = f"case {index} ({case['kind']})"
    rejection_key = f"case{index}_rejection"
    if rejection_key in arrays:
        assert result.get("rejection") == str(arrays[rejection_key]), label
        return
    assert "error" not in result and "rejection" not in result, f"{label}: {result}"
    invariant = case.get("parity_invariant")
    if invariant is not None:
        assert invariant["kind"] == "axisymmetric"
        _assert_close(
            _axisymmetric_invariants(result["position_m"], invariant),
            _axisymmetric_invariants(arrays[f"case{index}_position_m"], invariant),
            f"{label} axis invariants",
        )
        _assert_close(
            np.linalg.eigvalsh(np.asarray(result["covariance_m2"])),
            np.linalg.eigvalsh(arrays[f"case{index}_covariance_m2"]),
            f"{label} covariance eigenvalues",
        )
    else:
        _assert_close(result["position_m"], arrays[f"case{index}_position_m"], f"{label} position")
        _assert_close(result["covariance_m2"], arrays[f"case{index}_covariance_m2"], f"{label} covariance")
    names = FUSION_SCALAR_NAMES if case["kind"] == "bearing_fusion" else CARTESIAN_SCALAR_NAMES
    expected_scalars = arrays[f"case{index}_scalars"]
    for name, expected in zip(names, expected_scalars):
        actual = _none_to_inf(result[name])
        if np.isinf(expected):
            assert np.isinf(actual), f"{label} {name}: expected inf, got {actual}"
        else:
            np.testing.assert_allclose(actual, expected, rtol=RELATIVE_TOLERANCE, err_msg=f"{label} {name}")
    mode_key = f"case{index}_mode"
    if mode_key in arrays:
        assert result["range_projection_mode"] == str(arrays[mode_key]), label


@pytest.mark.skipif(not cargo_is_available(), reason="cargo toolchain required to build the Rust sidecar oracle")
def test_rust_tdoa_solver_matches_python_goldens() -> None:
    cases, arrays = _load_goldens()
    completed = subprocess.run(
        [str(build_rust_sidecar_binary()), "tdoa-solver-oracle"],
        input=json.dumps({"cases": cases}),
        capture_output=True,
        text=True,
        check=True,
    )
    results = json.loads(completed.stdout)["results"]
    assert len(results) == len(cases)
    for index, (case, result) in enumerate(zip(cases, results)):
        _compare_result(index, case, result, arrays)


def test_python_solver_still_matches_goldens() -> None:
    cases, arrays = _load_goldens()
    for index, case in enumerate(cases):
        _compare_result(index, case, run_python_case(case), arrays)
