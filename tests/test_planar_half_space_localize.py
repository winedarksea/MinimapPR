"""End-to-end half-space (D7) reflection through LocalizationEngine.localize()
for a coplanar 5-mic planar array: an unconstrained solve is mirror-symmetric
across the array's own plane, and the configured half_space must pick the
physically valid side."""

from __future__ import annotations

import math

import numpy as np
import pytest

from minimappr.core.localization import LocalizationEngine
from tests.helpers import synthesize_delayed_array_channels

SAMPLE_RATE_HZ = 48_000
_R = 0.025 * math.sqrt(0.5)
_PLANAR_OFFSETS_M: tuple[tuple[float, float, float], ...] = (
    (_R, _R, 0.0),
    (-_R, _R, 0.0),
    (-_R, -_R, 0.0),
    (_R, -_R, 0.0),
    (0.0, 0.0, 0.0),
)


def _planar_positions() -> dict[str, np.ndarray]:
    return {
        f"mic{i}": np.asarray(offset, dtype=np.float64)
        for i, offset in enumerate(_PLANAR_OFFSETS_M)
    }


def _planar_windows(source_position_m: tuple[float, float, float]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    mono = rng.standard_normal(4096).astype(np.float64)
    channels = synthesize_delayed_array_channels(
        mono,
        SAMPLE_RATE_HZ,
        source_position_m=source_position_m,
        sensor_offsets_m=_PLANAR_OFFSETS_M,
    )
    return {f"mic{i}": channels[i] for i in range(channels.shape[0])}


def test_half_space_upper_reflects_source_below_plane_to_above() -> None:
    # Source physically below the array plane (z=0): a bare-metal solve is free
    # to converge on either mirror image, but half_space="upper" must always
    # report a position with z >= the array's own plane.
    source_below = (0.5, 0.3, -1.5)
    positions = _planar_positions()
    windows = _planar_windows(source_below)
    engine = LocalizationEngine(max_tau_s=0.02, interp_factor=4)

    result_upper = engine.localize(
        sensor_positions=positions,
        sensor_windows=windows,
        sample_rate_hz=SAMPLE_RATE_HZ,
        temperature_c=20.0,
        humidity_fraction=0.5,
        half_space="upper",
    )
    assert result_upper.position_m[2] >= -1e-6
    assert result_upper.half_space_applied is True


def test_half_space_lower_reflects_source_above_plane_to_below() -> None:
    source_above = (0.5, 0.3, 1.5)
    positions = _planar_positions()
    windows = _planar_windows(source_above)
    engine = LocalizationEngine(max_tau_s=0.02, interp_factor=4)

    result_lower = engine.localize(
        sensor_positions=positions,
        sensor_windows=windows,
        sample_rate_hz=SAMPLE_RATE_HZ,
        temperature_c=20.0,
        humidity_fraction=0.5,
        half_space="lower",
    )
    assert result_lower.position_m[2] <= 1e-6


def test_half_space_none_leaves_solve_unconstrained() -> None:
    # Baseline: without a half_space constraint, the solver is free to return
    # either side — this just proves the parameter is opt-in and doesn't
    # silently activate for tetra/unconstrained arrays.
    source = (0.5, 0.3, -1.5)
    positions = _planar_positions()
    windows = _planar_windows(source)
    engine = LocalizationEngine(max_tau_s=0.02, interp_factor=4)

    result_default = engine.localize(
        sensor_positions=positions,
        sensor_windows=windows,
        sample_rate_hz=SAMPLE_RATE_HZ,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )
    result_explicit_none = engine.localize(
        sensor_positions=positions,
        sensor_windows=windows,
        sample_rate_hz=SAMPLE_RATE_HZ,
        temperature_c=20.0,
        humidity_fraction=0.5,
        half_space=None,
    )
    assert result_default.position_m == pytest.approx(result_explicit_none.position_m)
    assert result_default.half_space_applied is False
    assert result_explicit_none.half_space_applied is False
