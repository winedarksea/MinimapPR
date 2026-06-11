"""Node-local bearing and cross-node amplitude constraints."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from minimappr.core.tdoa_measurements import (
    PairTdoaMeasurement,
    measure_pair_tdoas,
    normalized_pair_quality,
)
from minimappr.utils.audio import rms


EPSILON = 1.0e-12
MINIMUM_BEARING_SENSOR_COUNT = 4
MINIMUM_ARRAY_APERTURE_M = 0.02
MINIMUM_NODE_BASELINE_M = 1.0
MAXIMUM_AMPLITUDE_LOG_RATIO = math.log(20.0)


@dataclass(frozen=True, slots=True)
class SpatialBearingConstraint:
    node_id: str
    origin_m: np.ndarray
    direction: np.ndarray
    strength: float
    quality: float


@dataclass(frozen=True, slots=True)
class AmplitudeRatioConstraint:
    node_a: str
    node_b: str
    origin_a_m: np.ndarray
    origin_b_m: np.ndarray
    measured_log_amplitude_ratio: float
    strength: float


def broadband_direction_from_tdoas(
    *,
    sensor_positions: dict[str, np.ndarray],
    measurements: list[PairTdoaMeasurement],
    sound_speed_mps: float,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    for measurement in measurements:
        rows.append(sensor_positions[measurement.sensor_b] - sensor_positions[measurement.sensor_a])
        targets.append(sound_speed_mps * measurement.tdoa_seconds)
        weights.append(math.sqrt(measurement.weight))
    if len(rows) < 3:
        raise ValueError("Bearing solve needs at least three finite TDOA pairs")
    weighted_rows = np.vstack(rows) * np.asarray(weights)[:, None]
    weighted_targets = np.asarray(targets, dtype=np.float64) * np.asarray(weights)
    try:
        direction, *_ = np.linalg.lstsq(weighted_rows, weighted_targets, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError("TDOA bearing solve is singular") from exc
    norm = float(np.linalg.norm(direction))
    if norm < EPSILON:
        raise ValueError("TDOA bearing solve is degenerate")
    return direction / norm


def build_node_spatial_constraints(
    *,
    sensor_positions: dict[str, np.ndarray],
    sensor_windows: dict[str, np.ndarray],
    sensor_node_ids: dict[str, str] | None,
    sample_rate_hz: int,
    sound_speed_mps: float,
    max_tau_s: float,
    interpolation_factor: int,
    gcc_phat_function: Callable[..., tuple[float, float]],
    bearing_strength: float,
    amplitude_ratio_strength: float,
    pair_measurements: list[PairTdoaMeasurement] | None = None,
    sensor_gain_offsets_db: dict[str, float] | None = None,
    node_bearing_estimator: (
        Callable[
            [dict[str, np.ndarray], dict[str, np.ndarray], list[str]],
            tuple[np.ndarray, float],
        ]
        | None
    ) = None,
) -> tuple[list[SpatialBearingConstraint], list[AmplitudeRatioConstraint]]:
    if not sensor_node_ids:
        return [], []

    sensors_by_node: dict[str, list[str]] = {}
    for sensor_id in sensor_positions:
        if sensor_id not in sensor_windows:
            continue
        node_id = sensor_node_ids.get(sensor_id)
        if node_id is not None:
            sensors_by_node.setdefault(node_id, []).append(sensor_id)

    bearings: list[SpatialBearingConstraint] = []
    node_origins: dict[str, np.ndarray] = {}
    node_amplitudes: dict[str, float] = {}
    gain_offsets_db = sensor_gain_offsets_db or {}
    for node_id, node_sensor_ids in sorted(sensors_by_node.items()):
        node_positions = {
            sensor_id: sensor_positions[sensor_id]
            for sensor_id in node_sensor_ids
        }
        origin_m = np.mean(np.vstack(list(node_positions.values())), axis=0)
        node_origins[node_id] = origin_m
        node_amplitudes[node_id] = float(
            np.sqrt(
                np.mean(
                    [
                        rms(sensor_windows[sensor_id]) ** 2
                        * (10.0 ** (gain_offsets_db.get(sensor_id, 0.0) / 10.0))
                        for sensor_id in node_sensor_ids
                    ]
                )
            )
        )
        if len(node_sensor_ids) < MINIMUM_BEARING_SENSOR_COUNT:
            continue
        aperture_m = max(
            (
                float(np.linalg.norm(node_positions[a] - node_positions[b]))
                for a, b in itertools.combinations(node_sensor_ids, 2)
            ),
            default=0.0,
        )
        if aperture_m < MINIMUM_ARRAY_APERTURE_M:
            continue
        measurements = (
            [
                measurement
                for measurement in pair_measurements
                if measurement.sensor_a in node_positions
                and measurement.sensor_b in node_positions
            ]
            if pair_measurements is not None
            else measure_pair_tdoas(
                sensor_positions=node_positions,
                sensor_windows=sensor_windows,
                sensor_ids=sorted(node_sensor_ids),
                sample_rate_hz=sample_rate_hz,
                sound_speed_mps=sound_speed_mps,
                max_tau_s=max_tau_s,
                interpolation_factor=interpolation_factor,
                sensor_weights=None,
                gcc_phat_function=gcc_phat_function,
                sensor_node_ids=sensor_node_ids,
            )
        )
        try:
            if node_bearing_estimator is not None:
                direction, quality = node_bearing_estimator(
                    node_positions,
                    sensor_windows,
                    sorted(node_sensor_ids),
                )
            else:
                direction = broadband_direction_from_tdoas(
                    sensor_positions=node_positions,
                    measurements=measurements,
                    sound_speed_mps=sound_speed_mps,
                )
                quality = float(
                    np.mean(
                        [
                            normalized_pair_quality(measurement.correlation_peak)
                            for measurement in measurements
                        ]
                    )
                )
        except (ValueError, np.linalg.LinAlgError):
            continue
        direction = np.asarray(direction, dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < EPSILON:
            continue
        direction = direction / direction_norm
        quality = float(np.clip(quality, 0.0, 1.0))
        bearings.append(
            SpatialBearingConstraint(
                node_id=node_id,
                origin_m=origin_m,
                direction=direction,
                strength=max(bearing_strength * quality, 0.01),
                quality=quality,
            )
        )

    amplitude_constraints: list[AmplitudeRatioConstraint] = []
    if amplitude_ratio_strength <= 0.0:
        return bearings, amplitude_constraints
    for node_a, node_b in itertools.combinations(sorted(node_origins), 2):
        origin_a_m = node_origins[node_a]
        origin_b_m = node_origins[node_b]
        if float(np.linalg.norm(origin_a_m - origin_b_m)) < MINIMUM_NODE_BASELINE_M:
            continue
        amplitude_a = node_amplitudes[node_a]
        amplitude_b = node_amplitudes[node_b]
        if amplitude_a <= EPSILON or amplitude_b <= EPSILON:
            continue
        measured_log_ratio = float(
            np.clip(
                math.log(amplitude_a / amplitude_b),
                -MAXIMUM_AMPLITUDE_LOG_RATIO,
                MAXIMUM_AMPLITUDE_LOG_RATIO,
            )
        )
        amplitude_constraints.append(
            AmplitudeRatioConstraint(
                node_a=node_a,
                node_b=node_b,
                origin_a_m=origin_a_m,
                origin_b_m=origin_b_m,
                measured_log_amplitude_ratio=measured_log_ratio,
                strength=amplitude_ratio_strength,
            )
        )
    return bearings, amplitude_constraints


def bearing_intersection_initial_position(
    constraints: list[SpatialBearingConstraint],
) -> np.ndarray | None:
    if len(constraints) < 2:
        return None
    information = np.zeros((3, 3), dtype=np.float64)
    target = np.zeros(3, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for constraint in constraints:
        projection = identity - np.outer(constraint.direction, constraint.direction)
        weighted_projection = projection * max(constraint.strength, 0.01)
        information += weighted_projection
        target += weighted_projection @ constraint.origin_m
    try:
        position_m, *_ = np.linalg.lstsq(information, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return position_m if np.all(np.isfinite(position_m)) else None


def spatial_constraint_residual_seconds(
    position_m: np.ndarray,
    *,
    bearing_constraints: list[SpatialBearingConstraint],
    amplitude_constraints: list[AmplitudeRatioConstraint],
    residual_scale_seconds: float,
) -> np.ndarray:
    rows: list[float] = []
    for constraint in bearing_constraints:
        offset_m = position_m - constraint.origin_m
        offset_norm_m = float(np.linalg.norm(offset_m))
        if offset_norm_m < EPSILON:
            rows.extend([0.0, 0.0, 0.0])
            continue
        estimated_direction = offset_m / offset_norm_m
        rows.extend(
            (
                (estimated_direction - constraint.direction)
                * residual_scale_seconds
                * math.sqrt(constraint.strength)
            ).tolist()
        )
    for constraint in amplitude_constraints:
        distance_a_m = max(float(np.linalg.norm(position_m - constraint.origin_a_m)), EPSILON)
        distance_b_m = max(float(np.linalg.norm(position_m - constraint.origin_b_m)), EPSILON)
        predicted_log_ratio = math.log(distance_b_m / distance_a_m)
        rows.append(
            (predicted_log_ratio - constraint.measured_log_amplitude_ratio)
            * residual_scale_seconds
            * math.sqrt(constraint.strength)
        )
    return np.asarray(rows, dtype=np.float64)


def spatial_constraint_jacobian(
    position_m: np.ndarray,
    *,
    bearing_constraints: list[SpatialBearingConstraint],
    amplitude_constraints: list[AmplitudeRatioConstraint],
    residual_scale_seconds: float,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    identity = np.eye(3, dtype=np.float64)
    for constraint in bearing_constraints:
        offset_m = position_m - constraint.origin_m
        distance_m = max(float(np.linalg.norm(offset_m)), EPSILON)
        direction = offset_m / distance_m
        rows.extend(
            (
                (identity - np.outer(direction, direction))
                / distance_m
                * residual_scale_seconds
                * math.sqrt(constraint.strength)
            )
        )
    for constraint in amplitude_constraints:
        offset_a = position_m - constraint.origin_a_m
        offset_b = position_m - constraint.origin_b_m
        distance_a_squared = max(float(np.dot(offset_a, offset_a)), EPSILON)
        distance_b_squared = max(float(np.dot(offset_b, offset_b)), EPSILON)
        rows.append(
            ((offset_b / distance_b_squared) - (offset_a / distance_a_squared))
            * residual_scale_seconds
            * math.sqrt(constraint.strength)
        )
    return np.vstack(rows) if rows else np.zeros((0, 3), dtype=np.float64)
