"""Pair selection and GCC-PHAT measurement for Cartesian localization."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

from minimappr.utils.audio import rms


MAX_SELECTED_PAIR_COUNT = 32


@dataclass(frozen=True, slots=True)
class PairTdoaMeasurement:
    sensor_a: str
    sensor_b: str
    tdoa_seconds: float
    correlation_peak: float
    weight: float


def normalized_pair_quality(correlation_peak: float) -> float:
    if not np.isfinite(correlation_peak):
        return 0.0
    return float(np.clip(abs(correlation_peak), 0.05, 1.0))


def measure_pair_tdoas(
    *,
    sensor_positions: dict[str, np.ndarray],
    sensor_windows: dict[str, np.ndarray],
    sensor_ids: list[str],
    sample_rate_hz: int,
    sound_speed_mps: float,
    max_tau_s: float,
    interpolation_factor: int,
    sensor_weights: dict[str, float] | None,
    gcc_phat_function: Callable[..., tuple[float, float]],
) -> list[PairTdoaMeasurement]:
    sensor_quality = sensor_weights or {}
    pair_candidates: list[tuple[float, str, str, float, float]] = []
    for sensor_a, sensor_b in itertools.combinations(sensor_ids, 2):
        baseline_m = float(np.linalg.norm(sensor_positions[sensor_a] - sensor_positions[sensor_b]))
        synchronization_weight = min(
            float(np.clip(sensor_quality.get(sensor_a, 1.0), 0.0, 1.0)),
            float(np.clip(sensor_quality.get(sensor_b, 1.0), 0.0, 1.0)),
        )
        pair_candidates.append(
            (
                synchronization_weight * max(baseline_m, 0.01),
                sensor_a,
                sensor_b,
                baseline_m,
                synchronization_weight,
            )
        )
    if len(sensor_ids) > 8 and len(pair_candidates) > MAX_SELECTED_PAIR_COUNT:
        pair_candidates.sort(key=lambda item: item[0], reverse=True)
        selected_pairs: list[tuple[float, str, str, float, float]] = []
        selected_pair_ids: set[tuple[str, str]] = set()
        covered_sensor_ids: set[str] = set()
        for candidate in pair_candidates:
            _, sensor_a, sensor_b, _, _ = candidate
            if sensor_a in covered_sensor_ids and sensor_b in covered_sensor_ids:
                continue
            selected_pairs.append(candidate)
            selected_pair_ids.add((sensor_a, sensor_b))
            covered_sensor_ids.update((sensor_a, sensor_b))
            if (
                len(covered_sensor_ids) == len(sensor_ids)
                or len(selected_pairs) == MAX_SELECTED_PAIR_COUNT
            ):
                break
        for candidate in pair_candidates:
            pair_id = (candidate[1], candidate[2])
            if pair_id in selected_pair_ids:
                continue
            selected_pairs.append(candidate)
            if len(selected_pairs) == MAX_SELECTED_PAIR_COUNT:
                break
        pair_candidates = selected_pairs

    measurements: list[PairTdoaMeasurement] = []
    for _, sensor_a, sensor_b, baseline_m, synchronization_weight in pair_candidates:
        physical_max_tau_s = (baseline_m / sound_speed_mps) + (1.0 / sample_rate_hz)
        tdoa_seconds, correlation_peak = gcc_phat_function(
            signal=sensor_windows[sensor_a],
            reference_signal=sensor_windows[sensor_b],
            sample_rate_hz=sample_rate_hz,
            max_tau_s=min(max_tau_s, max(physical_max_tau_s, 1.0 / sample_rate_hz)),
            interp=max(1, interpolation_factor),
        )
        if not np.isfinite(tdoa_seconds) or not np.isfinite(correlation_peak):
            continue
        measurements.append(
            PairTdoaMeasurement(
                sensor_a=sensor_a,
                sensor_b=sensor_b,
                tdoa_seconds=float(tdoa_seconds),
                correlation_peak=float(correlation_peak),
                weight=max(
                    0.01,
                    synchronization_weight * normalized_pair_quality(correlation_peak),
                ),
            )
        )
    return measurements


def reference_tdoas(
    *,
    measurements: list[PairTdoaMeasurement],
    sensor_windows: dict[str, np.ndarray],
    sensor_ids: list[str],
) -> tuple[str, dict[str, float]]:
    reference_sensor = max(sensor_ids, key=lambda sensor_id: rms(sensor_windows[sensor_id]))
    measured_sensor_ids = [sensor_id for sensor_id in sensor_ids if sensor_id != reference_sensor]
    sensor_column = {
        sensor_id: column_index
        for column_index, sensor_id in enumerate(measured_sensor_ids)
    }
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for measurement in measurements:
        row = np.zeros(len(measured_sensor_ids), dtype=np.float64)
        if measurement.sensor_a != reference_sensor:
            row[sensor_column[measurement.sensor_a]] = 1.0
        if measurement.sensor_b != reference_sensor:
            row[sensor_column[measurement.sensor_b]] = -1.0
        weight = np.sqrt(measurement.weight)
        rows.append(row * weight)
        targets.append(measurement.tdoa_seconds * weight)
    if not rows:
        return reference_sensor, {}
    try:
        relative_delays, *_ = np.linalg.lstsq(
            np.vstack(rows),
            np.asarray(targets, dtype=np.float64),
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return reference_sensor, {}
    return reference_sensor, {
        sensor_id: float(relative_delays[column_index])
        for sensor_id, column_index in sensor_column.items()
    }
