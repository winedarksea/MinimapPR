"""Pair selection and GCC-PHAT measurement for Cartesian localization."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

from minimappr.core.localization_band_policy import SensorPairFrontEnds, tukey_taper
from minimappr.core.pair_timing_uncertainty import (
    MINIMUM_PAIR_MEASUREMENT_QUALITY,
    inverse_variance_pair_weight,
    pair_information_score,
    pair_timing_uncertainty_seconds,
    solver_reference_time_std_seconds,
)
from minimappr.utils.audio import rms


MAX_SELECTED_PAIR_COUNT = 32


@dataclass(frozen=True, slots=True)
class PairTdoaMeasurement:
    sensor_a: str
    sensor_b: str
    tdoa_seconds: float
    correlation_peak: float
    weight: float


@dataclass(slots=True)
class PairMeasurementStats:
    """Mutable counters filled in by ``measure_pair_tdoas`` so callers can
    surface cross-node activity. Until these were wired, the corresponding
    FusionMetrics counters read 0 forever and a dead cross-node path was
    indistinguishable from a working one."""

    cross_node_pairs_measured: int = 0


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
    sensor_timing_uncertainty_seconds: dict[str, float] | None,
    gcc_phat_function: Callable[..., tuple[float, float]],
    sensor_node_ids: dict[str, str] | None = None,
    cross_node_max_tau_s: float | None = None,
    stats: PairMeasurementStats | None = None,
    pair_front_ends: SensorPairFrontEnds | None = None,
) -> list[PairTdoaMeasurement]:
    """Measure pairwise TDOAs, optionally enabling true cross-node correlation.

    ``cross_node_max_tau_s`` (Phase 5) lifts the tau search cap for *cross-node*
    pairs so wide baselines (up to ~120 m at 0.35 s) can be correlated; same-node
    pairs keep the tight ``max_tau_s``. ``sensor_timing_uncertainty_seconds`` is
    each sensor's node clock uncertainty (``core/pair_timing_uncertainty.py``):
    every cross-node pair is kept and inverse-variance weighted by it, never
    rejected by clock class. Node position accuracy never enters the weights
    (the sidecar's pairs carry none either, so both ingest paths weight a pair
    set alike); it widens the solve covariance instead
    (``core/node_position_covariance.py``).

    ``pair_front_ends`` gives each pair its band and window taper
    (``core/localization_band_policy.py``); ``None`` correlates every bin of
    the windows as given.
    """
    reference_time_std_s = solver_reference_time_std_seconds(sample_rate_hz, interpolation_factor)
    pair_candidates: list[tuple[float, str, str, float, bool, float]] = []
    for sensor_a, sensor_b in itertools.combinations(sensor_ids, 2):
        baseline_m = float(np.linalg.norm(sensor_positions[sensor_a] - sensor_positions[sensor_b]))
        same_node = (
            sensor_node_ids is not None
            and sensor_node_ids.get(sensor_a) is not None
            and sensor_node_ids.get(sensor_a) == sensor_node_ids.get(sensor_b)
        )
        timing_uncertainty_s = pair_timing_uncertainty_seconds(
            sensor_a,
            sensor_b,
            sensor_node_ids=sensor_node_ids,
            sensor_timing_uncertainty_seconds=sensor_timing_uncertainty_seconds,
        )
        pair_candidates.append(
            (
                pair_information_score(
                    baseline_m=max(baseline_m, 0.01),
                    timing_uncertainty_seconds=timing_uncertainty_s,
                    reference_time_std_seconds=reference_time_std_s,
                ),
                sensor_a,
                sensor_b,
                baseline_m,
                same_node,
                timing_uncertainty_s,
            )
        )
    if len(sensor_ids) > 8 and len(pair_candidates) > MAX_SELECTED_PAIR_COUNT:
        pair_candidates.sort(key=lambda item: item[0], reverse=True)
        selected_pairs: list[tuple[float, str, str, float, bool, float]] = []
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
    for _, sensor_a, sensor_b, baseline_m, same_node, timing_uncertainty_s in pair_candidates:
        physical_max_tau_s = (baseline_m / sound_speed_mps) + (1.0 / sample_rate_hz)
        # Cross-node pairs may correlate over a much wider lag than the tight
        # same-node cap when a cross-node tau ceiling is supplied (Phase 5).
        effective_max_tau_s = max_tau_s
        if not same_node and cross_node_max_tau_s is not None:
            effective_max_tau_s = max(max_tau_s, float(cross_node_max_tau_s))
        signal = sensor_windows[sensor_a]
        reference_signal = sensor_windows[sensor_b]
        front_end_kwargs: dict[str, object] = {}
        if pair_front_ends is not None:
            front_end = pair_front_ends.pair_front_end(sensor_a, sensor_b)
            signal = tukey_taper(signal, front_end.taper_fraction)
            reference_signal = tukey_taper(reference_signal, front_end.taper_fraction)
            front_end_kwargs["band_hz"] = front_end.band_hz
            front_end_kwargs["phat_beta"] = front_end.phat_beta
        tdoa_seconds, correlation_peak = gcc_phat_function(
            signal=signal,
            reference_signal=reference_signal,
            sample_rate_hz=sample_rate_hz,
            max_tau_s=min(effective_max_tau_s, max(physical_max_tau_s, 1.0 / sample_rate_hz)),
            interp=max(1, interpolation_factor),
            **front_end_kwargs,
        )
        if not np.isfinite(tdoa_seconds) or not np.isfinite(correlation_peak):
            continue
        if stats is not None and sensor_node_ids is not None and not same_node:
            stats.cross_node_pairs_measured += 1
        measurements.append(
            PairTdoaMeasurement(
                sensor_a=sensor_a,
                sensor_b=sensor_b,
                tdoa_seconds=float(tdoa_seconds),
                correlation_peak=float(correlation_peak),
                weight=inverse_variance_pair_weight(
                    measurement_quality=max(
                        MINIMUM_PAIR_MEASUREMENT_QUALITY,
                        normalized_pair_quality(correlation_peak),
                    ),
                    timing_uncertainty_seconds=timing_uncertainty_s,
                    reference_time_std_seconds=reference_time_std_s,
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
