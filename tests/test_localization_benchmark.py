from __future__ import annotations

import numpy as np

from minimappr.core.localization import LocalizationEngine
from minimappr.sim.benchmark import EventRecord, compute_event_metrics
from minimappr.sim.geometry import regular_tetrahedron_offsets_m


def _shift_signal(signal: np.ndarray, sample_rate_hz: int, delay_s: float) -> np.ndarray:
    n = signal.size
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    shifted_t = t - delay_s
    return np.interp(shifted_t, t, signal, left=0.0, right=0.0).astype(np.float32)


def test_localization_benchmark_tight_geometry_tetra_case() -> None:
    sample_rate_hz = 48_000
    n = int(sample_rate_hz * 0.12)
    rng = np.random.default_rng(42)
    excitation = rng.standard_normal(n) * np.hanning(n)
    sound_speed = 343.2

    offsets = regular_tetrahedron_offsets_m(edge_m=0.05)
    sensor_positions = {
        f"s{i}": np.asarray(offset, dtype=np.float64) for i, offset in enumerate(offsets)
    }
    source = np.asarray([0.35, 0.18, 0.07], dtype=np.float64)

    windows = {}
    for sensor_id, position in sensor_positions.items():
        distance = float(np.linalg.norm(source - position))
        windows[sensor_id] = _shift_signal(excitation, sample_rate_hz, distance / sound_speed)

    engine = LocalizationEngine(max_tau_s=0.003)
    result = engine.localize(
        sensor_positions=sensor_positions,
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )

    estimate = np.asarray(result.position_m, dtype=np.float64)
    error_m = float(np.linalg.norm(estimate - source))
    assert error_m < 0.45
    assert np.isfinite(result.gdop)


def test_event_metrics_overlap_and_confusion_reporting() -> None:
    truth = [
        EventRecord(label="bird", timestamp_ns=1_000_000_000, position_m=(0.0, 0.0, 0.0)),
        EventRecord(label="bird", timestamp_ns=1_005_000_000, position_m=(1.0, 0.0, 0.0)),
        EventRecord(label="vehicle", timestamp_ns=1_020_000_000, position_m=(4.0, 0.5, 0.0)),
    ]
    predicted = [
        EventRecord(label="bird", timestamp_ns=1_000_500_000, position_m=(0.1, 0.0, 0.0)),
        EventRecord(label="human", timestamp_ns=1_005_000_000, position_m=(1.1, 0.0, 0.0)),
        EventRecord(label="vehicle", timestamp_ns=1_020_100_000, position_m=(4.1, 0.6, 0.0)),
        EventRecord(label="noise", timestamp_ns=1_200_000_000, position_m=(8.0, 8.0, 0.0)),
    ]

    metrics = compute_event_metrics(
        truth,
        predicted,
        max_time_delta_ns=2_000_000,
        max_distance_m=0.6,
    )
    assert metrics["true_positives"] == 3
    assert metrics["missed_detections"] == 0
    assert metrics["false_positives"] == 1
    assert metrics["class_confusions"] == 1
    assert float(metrics["confusion_rate"]) > 0.0
