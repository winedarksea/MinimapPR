from __future__ import annotations

import numpy as np
import pytest

import minimappr.core.advanced_localization as advanced_localization
from minimappr.core.advanced_localization import (
    EspritLocalizer,
    MusicLocalizer,
    SRPPhatLocalizer,
)
from minimappr.core.spatial_constraints import build_node_spatial_constraints
from minimappr.core.subspace_bearing import BearingPrior
from minimappr.core.tdoa_measurements import measure_pair_tdoas
from tests.helpers import SIRITH_TETRA_SENSOR_OFFSETS_M, shift_signal


def test_free_running_cross_node_tdoa_pairs_are_excluded() -> None:
    sensor_positions = {
        **{
            f"tetra-a:ch{index}": np.asarray([index * 0.01, 0.0, 0.0])
            for index in range(4)
        },
        **{
            f"tetra-b:ch{index}": np.asarray([20.0 + index * 0.01, 0.0, 0.0])
            for index in range(4)
        },
    }
    sensor_windows = {
        sensor_id: np.ones(128, dtype=np.float32)
        for sensor_id in sensor_positions
    }
    sensor_node_ids = {
        sensor_id: sensor_id.split(":")[0]
        for sensor_id in sensor_positions
    }
    measurements = measure_pair_tdoas(
        sensor_positions=sensor_positions,
        sensor_windows=sensor_windows,
        sensor_ids=sorted(sensor_positions),
        sample_rate_hz=16_000,
        sound_speed_mps=343.2,
        max_tau_s=0.1,
        interpolation_factor=1,
        sensor_weights={sensor_id: 0.05 for sensor_id in sensor_positions},
        gcc_phat_function=lambda **_kwargs: (0.0, 1.0),
        sensor_node_ids=sensor_node_ids,
    )

    assert len(measurements) == 12
    assert all(
        sensor_node_ids[measurement.sensor_a]
        == sensor_node_ids[measurement.sensor_b]
        for measurement in measurements
    )


def test_two_free_running_tetra_bearings_localize_with_point_node_present() -> None:
    sample_rate_hz = 48_000
    sample_count = int(sample_rate_hz * 0.2)
    rng = np.random.default_rng(123)
    excitation = (
        rng.standard_normal(sample_count) * np.hanning(sample_count)
    ).astype(np.float32)
    source_position_m = np.asarray([8.0, 12.0, 3.0], dtype=np.float64)
    node_origins_m = {
        "tetra-a": np.asarray([0.0, 0.0, 1.5], dtype=np.float64),
        "tetra-b": np.asarray([20.0, 0.0, 1.5], dtype=np.float64),
    }
    sensor_positions: dict[str, np.ndarray] = {}
    sensor_windows: dict[str, np.ndarray] = {}
    sensor_node_ids: dict[str, str] = {}
    sensor_weights: dict[str, float] = {}

    for node_id, node_origin_m in node_origins_m.items():
        node_distances: list[tuple[str, float]] = []
        for index, sensor_offset_m in enumerate(SIRITH_TETRA_SENSOR_OFFSETS_M):
            sensor_id = f"{node_id}:ch{index}"
            sensor_position_m = node_origin_m + np.asarray(
                sensor_offset_m,
                dtype=np.float64,
            )
            distance_m = float(np.linalg.norm(source_position_m - sensor_position_m))
            sensor_positions[sensor_id] = sensor_position_m
            sensor_node_ids[sensor_id] = node_id
            sensor_weights[sensor_id] = 0.05
            node_distances.append((sensor_id, distance_m))
        minimum_distance_m = min(distance_m for _, distance_m in node_distances)
        for sensor_id, distance_m in node_distances:
            sensor_windows[sensor_id] = shift_signal(
                excitation * (1.0 / distance_m),
                sample_rate_hz,
                (distance_m - minimum_distance_m) / 343.2,
            )

    point_sensor_id = "point-c:ch0"
    point_position_m = np.asarray([10.0, 20.0, 1.5], dtype=np.float64)
    point_distance_m = float(np.linalg.norm(source_position_m - point_position_m))
    sensor_positions[point_sensor_id] = point_position_m
    sensor_windows[point_sensor_id] = excitation * (1.0 / point_distance_m)
    sensor_node_ids[point_sensor_id] = "point-c"
    sensor_weights[point_sensor_id] = 0.05

    result = SRPPhatLocalizer(
        max_tau_s=0.003,
        interp=8,
        node_bearing_strength=1.0,
        amplitude_ratio_strength=0.15,
    ).localize(
        sensor_positions=sensor_positions,
        sensor_windows=sensor_windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
        sensor_weights=sensor_weights,
        sensor_node_ids=sensor_node_ids,
    )

    position_error_m = float(
        np.linalg.norm(np.asarray(result.position_m) - source_position_m)
    )
    assert position_error_m < 1.0
    assert result.range_observability is not None
    assert result.range_observability > 0.25


def test_amplitude_ratio_applies_sensor_gain_calibration() -> None:
    sensor_positions = {
        "point-a:ch0": np.asarray([0.0, 0.0, 0.0]),
        "point-b:ch0": np.asarray([10.0, 0.0, 0.0]),
    }
    sensor_windows = {
        "point-a:ch0": np.full(128, 0.5, dtype=np.float32),
        "point-b:ch0": np.ones(128, dtype=np.float32),
    }
    _, amplitude_constraints = build_node_spatial_constraints(
        sensor_positions=sensor_positions,
        sensor_windows=sensor_windows,
        sensor_node_ids={
            "point-a:ch0": "point-a",
            "point-b:ch0": "point-b",
        },
        sample_rate_hz=16_000,
        sound_speed_mps=343.2,
        max_tau_s=0.1,
        interpolation_factor=1,
        gcc_phat_function=lambda **_kwargs: (0.0, 1.0),
        bearing_strength=1.0,
        amplitude_ratio_strength=0.15,
        sensor_gain_offsets_db={"point-a:ch0": 6.020599913, "point-b:ch0": 0.0},
    )

    assert len(amplitude_constraints) == 1
    assert amplitude_constraints[0].measured_log_amplitude_ratio == pytest.approx(
        0.0,
        abs=1.0e-6,
    )


@pytest.mark.parametrize(
    ("localizer", "estimator_name"),
    [
        (
            MusicLocalizer(
                max_tau_s=0.003,
                azimuth_step_deg=7.0,
                elevation_step_deg=9.0,
                source_count=1,
                interp=8,
            ),
            "music_bearing_prior",
        ),
        (
            EspritLocalizer(max_tau_s=0.003, interp=8),
            "esprit_bearing_prior",
        ),
    ],
)
def test_subspace_bearings_are_estimated_per_tetra_node(
    monkeypatch: pytest.MonkeyPatch,
    localizer,
    estimator_name: str,
) -> None:
    sample_rate_hz = 48_000
    sample_count = int(sample_rate_hz * 0.2)
    rng = np.random.default_rng(321)
    excitation = (
        rng.standard_normal(sample_count) * np.hanning(sample_count)
    ).astype(np.float32)
    source_position_m = np.asarray([8.0, 12.0, 3.0], dtype=np.float64)
    sensor_positions: dict[str, np.ndarray] = {}
    sensor_windows: dict[str, np.ndarray] = {}
    sensor_node_ids: dict[str, str] = {}
    estimator_calls: list[list[str]] = []

    for node_id, origin_m in {
        "tetra-a": np.asarray([0.0, 0.0, 1.5]),
        "tetra-b": np.asarray([20.0, 0.0, 1.5]),
    }.items():
        node_distances: list[tuple[str, float]] = []
        for index, offset_m in enumerate(SIRITH_TETRA_SENSOR_OFFSETS_M):
            sensor_id = f"{node_id}:ch{index}"
            sensor_position_m = origin_m + np.asarray(offset_m)
            distance_m = float(np.linalg.norm(source_position_m - sensor_position_m))
            sensor_positions[sensor_id] = sensor_position_m
            sensor_node_ids[sensor_id] = node_id
            node_distances.append((sensor_id, distance_m))
        minimum_distance_m = min(distance_m for _, distance_m in node_distances)
        for sensor_id, distance_m in node_distances:
            sensor_windows[sensor_id] = shift_signal(
                excitation,
                sample_rate_hz,
                (distance_m - minimum_distance_m) / 343.2,
            )

    def exact_node_bearing_prior(**kwargs) -> BearingPrior:
        sensor_ids = list(kwargs["sensor_ids"])
        estimator_calls.append(sensor_ids)
        centroid_m = np.mean(
            np.vstack([kwargs["sensor_positions"][sensor_id] for sensor_id in sensor_ids]),
            axis=0,
        )
        direction = source_position_m - centroid_m
        direction /= np.linalg.norm(direction)
        return BearingPrior(direction=direction, strength=1.0, quality=1.0)

    monkeypatch.setattr(
        advanced_localization,
        estimator_name,
        exact_node_bearing_prior,
    )
    result = localizer.localize(
        sensor_positions=sensor_positions,
        sensor_windows=sensor_windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
        sensor_weights={sensor_id: 0.05 for sensor_id in sensor_positions},
        sensor_node_ids=sensor_node_ids,
    )

    assert len(estimator_calls) == 2
    assert all(
        len({sensor_node_ids[sensor_id] for sensor_id in sensor_ids}) == 1
        for sensor_ids in estimator_calls
    )
    assert float(
        np.linalg.norm(np.asarray(result.position_m) - source_position_m)
    ) < 1.0
