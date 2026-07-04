"""Pair selection and GCC-PHAT measurement for Cartesian localization."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

from minimappr.utils.audio import rms


MAX_SELECTED_PAIR_COUNT = 32
MINIMUM_CROSS_NODE_SYNCHRONIZATION_WEIGHT = 0.25


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
    sensor_node_ids: dict[str, str] | None = None,
    cross_node_max_tau_s: float | None = None,
    cross_node_min_sync_weight: float | None = None,
    node_position_std_m: dict[str, float] | None = None,
) -> list[PairTdoaMeasurement]:
    """Measure pairwise TDOAs, optionally enabling true cross-node correlation.

    ``cross_node_max_tau_s`` (Phase 5) lifts the tau search cap for *cross-node*
    pairs so wide baselines (up to ~120 m at 0.35 s) can be correlated; same-node
    pairs keep the tight ``max_tau_s``. ``cross_node_min_sync_weight`` overrides the
    default cross-node sync gate (recommend 1.0 in the field → PPS/PTP-only TDOA).
    ``node_position_std_m`` inflates cross-node pair weight by
    ``baseline/(baseline + σ_a + σ_b)`` so poorly-surveyed nodes contribute less. All
    default to ``None`` → behaviour unchanged.
    """
    sensor_quality = sensor_weights or {}
    cross_node_sync_gate = (
        float(cross_node_min_sync_weight)
        if cross_node_min_sync_weight is not None
        else MINIMUM_CROSS_NODE_SYNCHRONIZATION_WEIGHT
    )
    pair_candidates: list[tuple[float, str, str, float, float, bool]] = []
    for sensor_a, sensor_b in itertools.combinations(sensor_ids, 2):
        baseline_m = float(np.linalg.norm(sensor_positions[sensor_a] - sensor_positions[sensor_b]))
        same_node = (
            sensor_node_ids is not None
            and sensor_node_ids.get(sensor_a) is not None
            and sensor_node_ids.get(sensor_a) == sensor_node_ids.get(sensor_b)
        )
        synchronization_weight = (
            1.0
            if same_node
            else min(
                float(np.clip(sensor_quality.get(sensor_a, 1.0), 0.0, 1.0)),
                float(np.clip(sensor_quality.get(sensor_b, 1.0), 0.0, 1.0)),
            )
        )
        if (
            sensor_node_ids is not None
            and not same_node
            and synchronization_weight < cross_node_sync_gate
        ):
            continue
        # Phase 5: geometry weighting — a wide baseline surveyed with small position
        # std is far more informative than a short/uncertain one.
        geometry_weight = 1.0
        if not same_node and node_position_std_m is not None:
            sigma_a = float(node_position_std_m.get(sensor_a, 0.0))
            sigma_b = float(node_position_std_m.get(sensor_b, 0.0))
            geometry_weight = baseline_m / max(baseline_m + sigma_a + sigma_b, 1e-6)
        pair_candidates.append(
            (
                synchronization_weight * geometry_weight * max(baseline_m, 0.01),
                sensor_a,
                sensor_b,
                baseline_m,
                synchronization_weight * geometry_weight,
                same_node,
            )
        )
    if len(sensor_ids) > 8 and len(pair_candidates) > MAX_SELECTED_PAIR_COUNT:
        pair_candidates.sort(key=lambda item: item[0], reverse=True)
        selected_pairs: list[tuple[float, str, str, float, float, bool]] = []
        selected_pair_ids: set[tuple[str, str]] = set()
        covered_sensor_ids: set[str] = set()
        for candidate in pair_candidates:
            _, sensor_a, sensor_b, _, _, _ = candidate
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
    for _, sensor_a, sensor_b, baseline_m, synchronization_weight, same_node in pair_candidates:
        physical_max_tau_s = (baseline_m / sound_speed_mps) + (1.0 / sample_rate_hz)
        # Cross-node pairs may correlate over a much wider lag than the tight
        # same-node cap when a cross-node tau ceiling is supplied (Phase 5).
        effective_max_tau_s = max_tau_s
        if not same_node and cross_node_max_tau_s is not None:
            effective_max_tau_s = max(max_tau_s, float(cross_node_max_tau_s))
        tdoa_seconds, correlation_peak = gcc_phat_function(
            signal=sensor_windows[sensor_a],
            reference_signal=sensor_windows[sensor_b],
            sample_rate_hz=sample_rate_hz,
            max_tau_s=min(effective_max_tau_s, max(physical_max_tau_s, 1.0 / sample_rate_hz)),
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
    return reference_sensor, solve_reference_tdoas(
        measurements=measurements,
        sensor_ids=sensor_ids,
        reference_sensor=reference_sensor,
    )


def solve_reference_tdoas(
    *,
    measurements: list[PairTdoaMeasurement],
    sensor_ids: list[str],
    reference_sensor: str,
) -> dict[str, float]:
    """Least-squares reference-relative delays from pairwise TDOAs.

    Windowless counterpart of :func:`reference_tdoas`: the caller chooses the
    reference sensor (e.g. for the Rust middle path, where per-sensor audio
    windows are not available — only the pairwise TDOAs the sidecar measured).
    """
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
        return {}
    try:
        relative_delays, *_ = np.linalg.lstsq(
            np.vstack(rows),
            np.asarray(targets, dtype=np.float64),
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return {}
    return {
        sensor_id: float(relative_delays[column_index])
        for sensor_id, column_index in sensor_column.items()
    }
