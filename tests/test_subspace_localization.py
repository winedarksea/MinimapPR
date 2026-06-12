from __future__ import annotations

import numpy as np
import pytest

from minimappr.config import Settings
from minimappr.core.advanced_localization import EspritLocalizer, MusicLocalizer
from minimappr.core.localization_dispatch import build_localizer_from_settings
from minimappr.core.subspace_bearing import music_bearing_prior
from tests.helpers import shift_signal


SENSOR_OFFSETS_M = (
    (-0.016238, 0.025000, -0.010205),
    (0.027063, 0.000000, -0.010205),
    (-0.016238, -0.025000, -0.010205),
    (0.005413, 0.000000, 0.030615),
)


def _sensor_positions() -> dict[str, np.ndarray]:
    return {
        f"s{index}": np.asarray(offset_m, dtype=np.float64)
        for index, offset_m in enumerate(SENSOR_OFFSETS_M)
    }


@pytest.mark.parametrize("algorithm_name", ["music", "esprit"])
def test_subspace_frequency_band_changes_configured_result(algorithm_name: str) -> None:
    sample_rate_hz = 16_000
    sample_times_s = np.arange(4096, dtype=np.float64) / sample_rate_hz
    sound_speed_mps = 343.2
    sensor_positions = _sensor_positions()
    low_direction = np.asarray([1.0, 0.2, 0.1], dtype=np.float64)
    low_direction /= np.linalg.norm(low_direction)
    high_direction = np.asarray([-0.2, 0.9, 0.15], dtype=np.float64)
    high_direction /= np.linalg.norm(high_direction)
    sensor_windows = {}
    for sensor_id, sensor_position in sensor_positions.items():
        low_delay_s = -float(sensor_position @ low_direction) / sound_speed_mps
        high_delay_s = -float(sensor_position @ high_direction) / sound_speed_mps
        sensor_windows[sensor_id] = (
            np.sin(2.0 * np.pi * 500.0 * (sample_times_s - low_delay_s))
            + 0.9 * np.sin(2.0 * np.pi * 2500.0 * (sample_times_s - high_delay_s))
        ).astype(np.float32)

    common_settings = {
        "localization_algorithm": algorithm_name,
        "localization_strategy": "fixed",
        "localization_max_tau_seconds": 0.003,
        "localization_music_azimuth_step_deg": 3.0,
        "localization_music_elevation_step_deg": 3.0,
    }
    low_band_result = build_localizer_from_settings(
        Settings(
            **common_settings,
            localization_subspace_freq_min_hz=400.0,
            localization_subspace_freq_max_hz=700.0,
        )
    ).localize(sensor_positions, sensor_windows, sample_rate_hz, 20.0, 0.5)
    high_band_result = build_localizer_from_settings(
        Settings(
            **common_settings,
            localization_subspace_freq_min_hz=2200.0,
            localization_subspace_freq_max_hz=2800.0,
        )
    ).localize(sensor_positions, sensor_windows, sample_rate_hz, 20.0, 0.5)

    assert low_band_result.resolved_algorithm == algorithm_name
    assert high_band_result.resolved_algorithm == algorithm_name
    assert not np.allclose(
        low_band_result.position_m,
        high_band_result.position_m,
        atol=1.0e-4,
    )


def test_music_grid_and_source_count_change_bearing_prior() -> None:
    sample_rate_hz = 16_000
    sample_count = 8192
    sample_times_s = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
    sound_speed_mps = 343.2
    sensor_positions = _sensor_positions()
    direction_a = np.asarray([1.0, 0.2, 0.1], dtype=np.float64)
    direction_a /= np.linalg.norm(direction_a)
    direction_b = np.asarray([-0.2, 0.9, 0.15], dtype=np.float64)
    direction_b /= np.linalg.norm(direction_b)
    envelope_a = np.zeros(sample_count, dtype=np.float64)
    envelope_b = np.zeros(sample_count, dtype=np.float64)
    envelope_a[: sample_count // 2] = 1.0
    envelope_b[sample_count // 2 :] = 0.9
    sensor_windows = {}
    for sensor_id, sensor_position in sensor_positions.items():
        delay_a_s = -float(sensor_position @ direction_a) / sound_speed_mps
        delay_b_s = -float(sensor_position @ direction_b) / sound_speed_mps
        sensor_windows[sensor_id] = (
            envelope_a * np.sin(2.0 * np.pi * 700.0 * (sample_times_s - delay_a_s))
            + envelope_b * np.sin(2.0 * np.pi * 700.0 * (sample_times_s - delay_b_s))
        ).astype(np.float32)
    common_arguments = {
        "sensor_positions": sensor_positions,
        "sensor_windows": sensor_windows,
        "sensor_ids": sorted(sensor_positions),
        "sample_rate_hz": sample_rate_hz,
        "sound_speed_mps": sound_speed_mps,
        "minimum_frequency_hz": 600.0,
        "maximum_frequency_hz": 800.0,
    }

    one_source_prior = music_bearing_prior(
        **common_arguments,
        azimuth_step_deg=3.0,
        elevation_step_deg=3.0,
        source_count=1,
    )
    two_source_prior = music_bearing_prior(
        **common_arguments,
        azimuth_step_deg=3.0,
        elevation_step_deg=3.0,
        source_count=2,
    )
    coarse_grid_prior = music_bearing_prior(
        **common_arguments,
        azimuth_step_deg=40.0,
        elevation_step_deg=30.0,
        source_count=2,
    )

    assert float(np.dot(one_source_prior.direction, two_source_prior.direction)) < 0.95
    assert not np.allclose(
        two_source_prior.direction,
        coarse_grid_prior.direction,
        atol=1.0e-3,
    )


@pytest.mark.parametrize(
    "localizer",
    [
        MusicLocalizer(
            max_tau_s=0.003,
            azimuth_step_deg=5.0,
            elevation_step_deg=5.0,
            freq_min_hz=300.0,
            freq_max_hz=700.0,
            source_count=1,
        ),
        EspritLocalizer(
            max_tau_s=0.003,
            freq_min_hz=300.0,
            freq_max_hz=700.0,
        ),
    ],
)
def test_subspace_compact_far_field_preserves_distant_estimate(localizer) -> None:
    sample_rate_hz = 48_000
    sample_count = int(sample_rate_hz * 0.18)
    rng = np.random.default_rng(194)
    excitation = (rng.standard_normal(sample_count) * np.hanning(sample_count)).astype(
        np.float32
    )
    sensor_positions = _sensor_positions()
    source_direction = np.asarray([0.89, 0.41, 0.18], dtype=np.float64)
    source_direction /= np.linalg.norm(source_direction)
    source_position = source_direction * 100.0
    distances_m = {
        sensor_id: float(np.linalg.norm(source_position - sensor_position))
        for sensor_id, sensor_position in sensor_positions.items()
    }
    minimum_delay_s = min(distances_m.values()) / 343.2
    sensor_windows = {
        sensor_id: shift_signal(
            excitation,
            sample_rate_hz,
            (distance_m / 343.2) - minimum_delay_s,
        )
        for sensor_id, distance_m in distances_m.items()
    }

    result = localizer.localize(
        sensor_positions,
        sensor_windows,
        sample_rate_hz,
        20.0,
        0.5,
    )
    centroid_m = np.mean(np.vstack(list(sensor_positions.values())), axis=0)
    estimated_range_m = float(np.linalg.norm(np.asarray(result.position_m) - centroid_m))

    assert estimated_range_m > 20.0
    assert result.range_observability is not None
    assert result.range_observability < 0.10
    assert result.range_projection_mode == "range_asymptotic"
