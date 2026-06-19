from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import minimappr.core.cartesian_tdoa as cartesian_tdoa
from minimappr.core.cartesian_tdoa import solve_cartesian_tdoa
from minimappr.core.localization import LocalizationEngine, LocalizationError, speed_of_sound_mps
from minimappr.core.tdoa_measurements import PairTdoaMeasurement, measure_pair_tdoas
from tests.helpers import SIRITH_TETRA_SENSOR_OFFSETS_M, shift_signal, synthesize_delayed_array_channels


def test_tdoa_localization_recovers_source_position() -> None:
    sample_rate_hz = 16000
    duration_s = 0.2
    n = int(sample_rate_hz * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz

    rng = np.random.default_rng(5)
    excitation = rng.standard_normal(n) * np.hanning(n)
    sound_speed = 343.2

    sensor_positions = {
        "s0": np.array([0.0, 0.0, 0.0]),
        "s1": np.array([6.0, 0.0, 0.0]),
        "s2": np.array([0.0, 6.0, 0.0]),
        "s3": np.array([0.0, 0.0, 6.0]),
        "s4": np.array([6.0, 6.0, 6.0]),
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
    assert result.range_projection_mode == "range_refined"


@pytest.mark.parametrize("distance_m", [1.0, 6.0])
def test_compact_array_bearing_projected_without_bearing_prior(distance_m: float) -> None:
    """A ~5cm-aperture array cannot observe range curvature, but bearing is
    well-determined. The solver now emits range_bearing_projected (bearing
    observed, range uncertain) rather than a plain range_asymptotic, so that
    the detection renders as localized on the COP with an elongated covariance.
    """
    sample_rate_hz = 48_000
    temperature_c = 20.0
    humidity_fraction = 0.0
    sound_speed_mps = speed_of_sound_mps(temperature_c=temperature_c, humidity_fraction=humidity_fraction)

    mic_positions_m = np.asarray(SIRITH_TETRA_SENSOR_OFFSETS_M, dtype=np.float64)
    centroid_m = mic_positions_m.mean(axis=0)
    direction = np.array([20.0, 10.0, 3.0])
    direction = direction / np.linalg.norm(direction)
    source_m = centroid_m + direction * distance_m

    rng = np.random.default_rng(20260613)
    mono = rng.normal(0.0, 0.3, size=8192).astype(np.float32)
    channels = synthesize_delayed_array_channels(
        mono,
        sample_rate_hz,
        source_position_m=tuple(source_m),
        sensor_offsets_m=SIRITH_TETRA_SENSOR_OFFSETS_M,
        sound_speed_mps=sound_speed_mps,
    )
    sensor_positions = {f"ch{i}": mic_positions_m[i] for i in range(len(mic_positions_m))}
    sensor_windows = {f"ch{i}": channels[i].astype(np.float32) for i in range(len(channels))}

    engine = LocalizationEngine(max_tau_s=0.02)
    result = engine.localize(sensor_positions, sensor_windows, sample_rate_hz, temperature_c, humidity_fraction)

    bearing = np.asarray(result.position_m, dtype=np.float64) - centroid_m
    bearing /= np.linalg.norm(bearing)
    assert float(np.dot(bearing, direction)) > 0.9
    # Bearing is well-determined → bearing_projected instead of plain asymptotic
    assert result.range_projection_mode == "range_bearing_projected"
    # Range axis is still not observable; observability must be low
    assert result.range_observability is not None
    assert result.range_observability < 0.10
    # Confidence must exceed the old UNOBSERVABLE_CONFIDENCE_CAP (0.20) since
    # bearing is well-determined. The exact value depends on GCC-PHAT peak quality.
    assert result.confidence > 0.20


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


def _solver_test_measurements() -> tuple[
    dict[str, np.ndarray],
    list[PairTdoaMeasurement],
]:
    sensor_positions = {
        "s0": np.asarray([0.0, 0.0, 0.0]),
        "s1": np.asarray([1.0, 0.0, 0.0]),
        "s2": np.asarray([0.0, 1.0, 0.0]),
        "s3": np.asarray([0.0, 0.0, 1.0]),
    }
    measurements = [
        PairTdoaMeasurement("s0", "s1", 1.0e-4, 1.0, 1.0),
        PairTdoaMeasurement("s0", "s2", 1.5e-4, 1.0, 1.0),
        PairTdoaMeasurement("s0", "s3", 2.0e-4, 1.0, 1.0),
    ]
    return sensor_positions, measurements


def test_cartesian_solver_retries_then_rejects_unconverged_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensor_positions, measurements = _solver_test_measurements()
    evaluation_budgets: list[int] = []

    def unconverged_least_squares(*_args, **kwargs):
        evaluation_budgets.append(kwargs["max_nfev"])
        return SimpleNamespace(
            success=False,
            x=np.asarray(kwargs["x0"], dtype=np.float64),
            jac=np.eye(3, dtype=np.float64),
        )

    monkeypatch.setattr(
        cartesian_tdoa,
        "least_squares",
        unconverged_least_squares,
    )

    with pytest.raises(ValueError, match="did not converge"):
        solve_cartesian_tdoa(
            sensor_positions=sensor_positions,
            measurements=measurements,
            sound_speed_mps=343.2,
            sample_rate_hz=48_000,
            interpolation_factor=4,
            reference_sensor="s0",
            reference_tdoa_s={"s1": 1.0e-4, "s2": 1.5e-4, "s3": 2.0e-4},
            radial_refinement_enabled=False,
        )

    assert cartesian_tdoa.MAX_REFINEMENT_EVALUATIONS in evaluation_budgets
    assert cartesian_tdoa.MAX_RETRY_REFINEMENT_EVALUATIONS in evaluation_budgets


def test_cartesian_solver_accepts_successful_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensor_positions, measurements = _solver_test_measurements()
    call_count = 0

    def retrying_least_squares(*_args, **kwargs):
        nonlocal call_count
        call_count += 1
        return SimpleNamespace(
            success=call_count % 2 == 0,
            x=np.asarray([2.0, 2.0, 2.0], dtype=np.float64),
            jac=np.eye(3, dtype=np.float64),
        )

    monkeypatch.setattr(
        cartesian_tdoa,
        "least_squares",
        retrying_least_squares,
    )

    result = solve_cartesian_tdoa(
        sensor_positions=sensor_positions,
        measurements=measurements,
        sound_speed_mps=343.2,
        sample_rate_hz=48_000,
        interpolation_factor=4,
        reference_sensor="s0",
        reference_tdoa_s={"s1": 1.0e-4, "s2": 1.5e-4, "s3": 2.0e-4},
        radial_refinement_enabled=False,
    )

    assert call_count >= 2
    assert np.all(np.isfinite(result.position_m))
