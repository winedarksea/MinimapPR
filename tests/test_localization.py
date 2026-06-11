from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.localization import LocalizationEngine, LocalizationError
from minimappr.core.tdoa_measurements import measure_pair_tdoas
from tests.helpers import shift_signal


def test_tdoa_localization_recovers_source_position() -> None:
    sample_rate_hz = 16000
    duration_s = 0.2
    n = int(sample_rate_hz * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz

    rng = np.random.default_rng(5)
    excitation = rng.standard_normal(n) * np.hanning(n)
    sound_speed = 343.2

    sensor_positions = {
        "s0": np.array([0.0, 0.0, 2.0]),
        "s1": np.array([6.0, 0.0, 2.0]),
        "s2": np.array([6.0, 0.4, 2.0]),
        "s3": np.array([6.2, 0.2, 2.2]),
        "s4": np.array([5.8, 0.2, 1.8]),
    }
    source = np.array([2.7, 3.3, 1.4])

    distances = {sensor_id: float(np.linalg.norm(source - pos)) for sensor_id, pos in sensor_positions.items()}
    windows = {}
    for sensor_id, distance in distances.items():
        windows[sensor_id] = shift_signal(excitation, sample_rate_hz, distance / sound_speed)

    engine = LocalizationEngine(max_tau_s=0.03)
    result = engine.localize(
        sensor_positions=sensor_positions,
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )

    estimate = np.array(result.position_m)
    error = float(np.linalg.norm(estimate - source))

    assert error < 0.9
    assert result.confidence > 0.1
    assert np.isfinite(result.gdop)
    assert result.position_covariance_m2 is not None
    assert result.range_observability is not None
    assert result.range_observability > 0.1
    assert result.residual_rms_seconds is not None
    assert result.range_projection_mode is None


def test_localization_rejects_non_finite_windows() -> None:
    sample_rate_hz = 16_000
    sensor_positions = {
        "s0": np.array([0.0, 0.0, 0.0]),
        "s1": np.array([1.0, 0.0, 0.0]),
        "s2": np.array([0.0, 1.0, 0.0]),
        "s3": np.array([0.0, 0.0, 1.0]),
    }
    windows = {
        "s0": np.array([0.0, 1.0, np.nan, 0.0], dtype=np.float32),
        "s1": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "s2": np.array([0.0, 1.0, np.inf, 0.0], dtype=np.float32),
        "s3": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }

    engine = LocalizationEngine(max_tau_s=0.03)
    with pytest.raises(LocalizationError):
        _ = engine.localize(
            sensor_positions=sensor_positions,
            sensor_windows=windows,
            sample_rate_hz=sample_rate_hz,
            temperature_c=20.0,
            humidity_fraction=0.5,
        )


def test_large_cluster_limits_pair_correlations_before_measurement() -> None:
    sensor_positions = {
        f"s{index}": np.asarray(
            [
                10.0 * np.cos(2.0 * np.pi * index / 16.0),
                10.0 * np.sin(2.0 * np.pi * index / 16.0),
                float(index % 3),
            ],
            dtype=np.float64,
        )
        for index in range(16)
    }
    sensor_windows = {
        sensor_id: np.ones(128, dtype=np.float32)
        for sensor_id in sensor_positions
    }
    correlation_call_count = 0

    def counting_gcc_phat(**_kwargs) -> tuple[float, float]:
        nonlocal correlation_call_count
        correlation_call_count += 1
        return 0.0, 1.0

    measurements = measure_pair_tdoas(
        sensor_positions=sensor_positions,
        sensor_windows=sensor_windows,
        sensor_ids=sorted(sensor_positions),
        sample_rate_hz=48_000,
        sound_speed_mps=343.2,
        max_tau_s=0.1,
        interpolation_factor=4,
        sensor_weights=None,
        gcc_phat_function=counting_gcc_phat,
    )

    assert correlation_call_count == 32
    assert len(measurements) == 32
