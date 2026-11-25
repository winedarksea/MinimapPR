"""Unit tests for the Rust-TDOA -> Python Cartesian solve bridge."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from minimappr.core.range_projection import (
    BEARING_PROJECTED_CONFIDENCE_CAP,
    RANGE_BEARING_PROJECTED,
    UNOBSERVABLE_CONFIDENCE_CAP,
    normalize_range_mode,
    range_mode_is_unobservable,
)
from minimappr.core.rust_tdoa_solve import solve_localization_from_rust_tdoas


SOUND_SPEED_MPS = 343.2
SAMPLE_RATE_HZ = 48_000

# Sirith tetrahedral mic offsets (same fixture used by the Rust oracle test).
MIC_OFFSETS_M = [
    (0.0, 0.050, 0.0),
    (0.0433, 0.025, 0.0),
    (0.0, 0.0, 0.0),
    (0.02165, 0.025, 0.04082),
]


def _sensor_positions(node_base=(0.0, 0.0, 0.0)) -> dict[str, np.ndarray]:
    base = np.asarray(node_base, dtype=np.float64)
    return {
        f"ch{index}": base + np.asarray(offset, dtype=np.float64)
        for index, offset in enumerate(MIC_OFFSETS_M)
    }


def _ideal_pair_tdoas(source_m, sensor_positions, ordered_ids) -> list[dict]:
    """Pairwise TDOAs in the solver's convention: (‖s-a‖ - ‖s-b‖)/c."""
    source = np.asarray(source_m, dtype=np.float64)
    pairs: list[dict] = []
    for channel_a, channel_b in itertools.combinations(range(len(ordered_ids)), 2):
        position_a = sensor_positions[ordered_ids[channel_a]]
        position_b = sensor_positions[ordered_ids[channel_b]]
        lag = (
            float(np.linalg.norm(source - position_a))
            - float(np.linalg.norm(source - position_b))
        ) / SOUND_SPEED_MPS
        pairs.append(
            {"ch_a": channel_a, "ch_b": channel_b, "lag_seconds": lag, "confidence": 0.8}
        )
    return pairs


def _bearing(centroid: np.ndarray, position: np.ndarray) -> np.ndarray:
    offset = position - centroid
    return offset / max(float(np.linalg.norm(offset)), 1e-12)


def test_bridge_recovers_bearing_for_near_field_source() -> None:
    sensor_positions = _sensor_positions()
    ordered_ids = [f"ch{i}" for i in range(4)]
    centroid = np.mean([sensor_positions[s] for s in ordered_ids], axis=0)
    source_m = np.array([0.20, 0.10, 0.15])
    pair_tdoas = _ideal_pair_tdoas(source_m, sensor_positions, ordered_ids)
    steering = tuple(_bearing(centroid, source_m))

    result = solve_localization_from_rust_tdoas(
        sensor_positions=sensor_positions,
        ordered_sensor_ids=ordered_ids,
        pair_tdoas=pair_tdoas,
        steering_direction=steering,
        sound_speed_mps=SOUND_SPEED_MPS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        interpolation_factor=4,
        far_field_default_range_m=50.0,
    )

    assert result is not None
    recovered = np.asarray(result.position_m)
    recovered_bearing = _bearing(centroid, recovered)
    true_bearing = _bearing(centroid, source_m)
    # Tight array: range is uncertain, but the bearing must agree closely.
    assert float(np.dot(recovered_bearing, true_bearing)) > 0.95


def test_bridge_recovers_bearing_for_far_source() -> None:
    sensor_positions = _sensor_positions()
    ordered_ids = [f"ch{i}" for i in range(4)]
    centroid = np.mean([sensor_positions[s] for s in ordered_ids], axis=0)
    source_m = np.array([60.0, 25.0, 10.0])  # far field for a ~6 cm aperture
    pair_tdoas = _ideal_pair_tdoas(source_m, sensor_positions, ordered_ids)
    steering = tuple(_bearing(centroid, source_m))

    result = solve_localization_from_rust_tdoas(
        sensor_positions=sensor_positions,
        ordered_sensor_ids=ordered_ids,
        pair_tdoas=pair_tdoas,
        steering_direction=steering,
        sound_speed_mps=SOUND_SPEED_MPS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        interpolation_factor=4,
        far_field_default_range_m=50.0,
    )

    assert result is not None
    # Range is ambiguous for a tight array, but the bearing must agree closely.
    recovered_bearing = _bearing(centroid, np.asarray(result.position_m))
    assert float(np.dot(recovered_bearing, _bearing(centroid, source_m))) > 0.95


def test_bridge_applies_confidence_cap_when_range_unobservable() -> None:
    # A degenerate, low-confidence cluster of TDOAs drives the solver into its
    # asymptotic range mode; assert the unobservable haircut is in effect.
    sensor_positions = _sensor_positions()
    ordered_ids = [f"ch{i}" for i in range(4)]
    centroid = np.mean([sensor_positions[s] for s in ordered_ids], axis=0)
    source_m = np.array([400.0, 0.0, 0.0])
    pair_tdoas = _ideal_pair_tdoas(source_m, sensor_positions, ordered_ids)
    steering = tuple(_bearing(centroid, source_m))

    result = solve_localization_from_rust_tdoas(
        sensor_positions=sensor_positions,
        ordered_sensor_ids=ordered_ids,
        pair_tdoas=pair_tdoas,
        steering_direction=steering,
        sound_speed_mps=SOUND_SPEED_MPS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        interpolation_factor=4,
        far_field_default_range_m=50.0,
    )

    assert result is not None
    # A 400 m source seen by a 5 cm array has no observable range. Before the
    # far-field model-selection fix this collapsed to a 6 cm "range_refined" fix
    # (observability 1.0) and the cap assertions below never ran.
    assert range_mode_is_unobservable(result.range_projection_mode)
    mode = normalize_range_mode(result.range_projection_mode)
    confidence_cap = (
        BEARING_PROJECTED_CONFIDENCE_CAP
        if mode == RANGE_BEARING_PROJECTED
        else UNOBSERVABLE_CONFIDENCE_CAP
    )
    assert result.confidence <= confidence_cap
    assert result.range_observability is None or result.range_observability <= 0.05
    recovered_bearing = _bearing(centroid, np.asarray(result.position_m))
    assert float(np.dot(recovered_bearing, _bearing(centroid, source_m))) > 0.99


@pytest.mark.parametrize(
    "pair_tdoas",
    [
        [],  # no measurements
        [{"ch_a": 0, "ch_b": 1, "lag_seconds": 1e-4, "confidence": 0.5}],  # too few
    ],
)
def test_bridge_returns_none_for_insufficient_measurements(pair_tdoas: list[dict]) -> None:
    sensor_positions = _sensor_positions()
    ordered_ids = [f"ch{i}" for i in range(4)]
    result = solve_localization_from_rust_tdoas(
        sensor_positions=sensor_positions,
        ordered_sensor_ids=ordered_ids,
        pair_tdoas=pair_tdoas,
        steering_direction=None,
        sound_speed_mps=SOUND_SPEED_MPS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        interpolation_factor=4,
        far_field_default_range_m=50.0,
    )
    assert result is None


def test_bridge_returns_none_when_fewer_than_four_sensors() -> None:
    sensor_positions = _sensor_positions()
    ordered_ids = ["ch0", "ch1", "ch2"]
    result = solve_localization_from_rust_tdoas(
        sensor_positions=sensor_positions,
        ordered_sensor_ids=ordered_ids,
        pair_tdoas=[{"ch_a": 0, "ch_b": 1, "lag_seconds": 1e-4, "confidence": 0.5}],
        steering_direction=None,
        sound_speed_mps=SOUND_SPEED_MPS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        interpolation_factor=4,
        far_field_default_range_m=50.0,
    )
    assert result is None


# Pair TDOAs a real sidecar measured for a lone Sirith tetra and a band-limited
# burst 100 m out at azimuth 35 deg, elevation 10 deg (E5 range matrix cell
# 100m-single_tetra-gps_locked-gps, R4 onset trigger: ~8 ms of burst in the
# 80 ms window). Their exact 3-D fit sits 0.216 m from the centroid, 4.1
# apertures out, and was published as range_refined with a 0.2 m covariance.
_MEASURED_ONSET_WINDOW_PAIR_TDOAS = [
    {"ch_a": 0, "ch_b": 1, "lag_seconds": 5.933148e-05, "confidence": 0.5008448},
    {"ch_a": 0, "ch_b": 2, "lag_seconds": -7.956549e-05, "confidence": 0.5010929},
    {"ch_a": 0, "ch_b": 3, "lag_seconds": 3.3073404e-05, "confidence": 0.5008651},
    {"ch_a": 1, "ch_b": 2, "lag_seconds": -0.00013799986, "confidence": 0.50099057},
    {"ch_a": 1, "ch_b": 3, "lag_seconds": -2.5358957e-05, "confidence": 0.50106305},
    {"ch_a": 2, "ch_b": 3, "lag_seconds": 0.000113565264, "confidence": 0.5009882},
]
_MEASURED_ONSET_WINDOW_STEERING = (0.80438995, 0.55921805, 0.20057929)


def test_lone_tetra_never_claims_range_for_a_100m_source_from_measured_tdoas() -> None:
    sensor_positions = _sensor_positions()
    ordered_ids = [f"ch{i}" for i in range(4)]
    azimuth, elevation = np.radians(35.0), np.radians(10.0)
    true_bearing = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])

    result = solve_localization_from_rust_tdoas(
        sensor_positions=sensor_positions,
        ordered_sensor_ids=ordered_ids,
        pair_tdoas=_MEASURED_ONSET_WINDOW_PAIR_TDOAS,
        steering_direction=_MEASURED_ONSET_WINDOW_STEERING,
        sound_speed_mps=SOUND_SPEED_MPS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        interpolation_factor=4,
        far_field_default_range_m=50.0,
    )

    assert result is not None
    assert range_mode_is_unobservable(result.range_projection_mode)
    centroid = np.mean([sensor_positions[s] for s in ordered_ids], axis=0)
    recovered_bearing = _bearing(centroid, np.asarray(result.position_m))
    assert float(np.degrees(np.arccos(np.clip(recovered_bearing @ true_bearing, -1.0, 1.0)))) < 5.0
    # The stated uncertainty must reach the true 100 m range along the bearing.
    radial_std_m = float(np.sqrt(true_bearing @ np.asarray(result.position_covariance_m2) @ true_bearing))
    solved_range_m = float(np.linalg.norm(np.asarray(result.position_m) - centroid))
    assert abs(solved_range_m - 100.0) < 4.0 * radial_std_m
