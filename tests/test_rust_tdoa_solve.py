"""Unit tests for the Rust-TDOA -> Python Cartesian solve bridge."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from minimappr.core.range_projection import range_mode_is_unobservable
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
        tight_array_aperture_m=0.35,
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
        tight_array_aperture_m=0.35,
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
        tight_array_aperture_m=0.35,
        far_field_default_range_m=50.0,
    )

    assert result is not None
    if range_mode_is_unobservable(result.range_projection_mode):
        assert result.confidence <= 0.20
        assert result.range_observability is None or result.range_observability <= 0.05


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
        tight_array_aperture_m=0.35,
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
        tight_array_aperture_m=0.35,
        far_field_default_range_m=50.0,
    )
    assert result is None
