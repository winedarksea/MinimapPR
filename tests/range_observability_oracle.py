"""Physical observability oracle for the realistic range matrix (plan item E5).

Answers, from geometry, timing and SNR alone (never from the pipeline), which
claim a localizer can honestly make about one static source:

    NOT_HEARD < HEARD < BEARING_CORRECT < RANGE_RESOLVED < FIX_3D

Model. Sensor ``i`` hears the source at ``t_i = |p - s_i| / c + t0 + d_n + a_i
- u_i . e_n / c`` where ``t0`` is the unknown emission time, ``a_i`` the
per-sensor arrival-time estimation noise (lag quantisation plus the high-SNR
time-delay CRLB), and ``d_n`` / ``e_n`` the clock offset and rigid position
error of the node ``n`` that owns the sensor. ``d_n`` and ``e_n`` are shared
by every channel of that node, so they are *correlated* across all of the
node's pairs: within-node pairs cancel them, cross-node pairs carry them in
full. A model that treated each pair's timing error as independent would let
many correlated cross-node pairs average a 2 ms NTP offset away; this one
cannot. The position covariance is the inverse Fisher information of
``(p, t0)`` under that full arrival covariance (with a vanishing prior so
unobservable directions stay finite).

Coplanar mirror: every TDOA of a source and of its reflection across the plane
of the sensors is identical when the sensors are coplanar. The oracle reflects
the source across the best-fit sensor plane and measures how far apart the
two arrival vectors are in the same noise metric (with ``t0`` projected out).
A mirror closer than ``k`` sigma blocks a 3-D fix even when the local Fisher
information looks sharp; half-space priors are configuration, not evidence.

Two verdicts per cell, from the same covariance at two sigma multipliers:
``guaranteed_class`` (``k_hi`` sigma inside every tolerance: the pipeline must
reach it) and ``max_honest_class`` (``k_lo`` sigma: claiming more than this is
an over-claim). Tolerances come from the tier scorer's ``TierScoringConfig``
so the oracle and the scoring judge the same thing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from minimappr.sim.replay_gate_config import TierScoringConfig

DECIBELS_PER_POWER_DECADE = 10.0


class PassClass(IntEnum):
    NOT_HEARD = 0
    HEARD = 1
    BEARING_CORRECT = 2
    RANGE_RESOLVED = 3
    FIX_3D = 4


@dataclass(frozen=True)
class OracleNode:
    node_id: str
    sensor_positions_m: tuple[tuple[float, float, float], ...]
    # Per-sensor in-band SNR (signal power over noise power in the band), dB.
    sensor_in_band_snr_db: tuple[float, ...]
    # 1-sigma clock offset shared by every channel of the node.
    clock_offset_std_seconds: float
    # 1-sigma rigid translation error of the whole node, per axis.
    position_std_m: float


@dataclass(frozen=True)
class ObservabilityOracleConfig:
    sound_speed_mps: float
    # GCC-PHAT lag grid: 1 / (sample_rate * interpolation factor).
    lag_quantum_seconds: float
    correlation_window_seconds: float
    band_hz: tuple[float, float]
    heard_min_in_band_snr_db: float
    tiers: TierScoringConfig
    guaranteed_sigma_multiplier: float = 3.0
    honest_sigma_multiplier: float = 1.0
    # Vanishing priors keep the information matrix invertible along directions
    # the data does not constrain (far-field range) without informing them.
    prior_position_std_m: float = 1.0e5
    prior_emission_time_std_seconds: float = 1.0e3


@dataclass(frozen=True)
class ObservabilityVerdict:
    guaranteed_class: PassClass
    max_honest_class: PassClass
    observer_position_m: tuple[float, float, float]
    true_range_m: float
    range_tolerance_m: float
    bearing_std_deg: float
    range_std_m: float
    max_axis_std_m: float
    # Distance (in sigma) between the source's and its mirror's arrival
    # vectors; None when the mirror is within tolerance of the source itself.
    mirror_separation_sigma: float | None
    position_covariance_m2: np.ndarray = field(repr=False)


def sensor_arrival_noise_std_seconds(in_band_snr_db: float, config: ObservabilityOracleConfig) -> float:
    """Per-sensor share of a pair's lag error: half the quantisation variance plus
    this sensor's half of the two-sided high-SNR time-delay CRLB
    ``var = 3 / (8 pi^2 T s (f2^3 - f1^3))`` for a flat in-band SNR ``s``."""
    snr_linear = 10.0 ** (in_band_snr_db / DECIBELS_PER_POWER_DECADE)
    low_hz, high_hz = config.band_hz
    crlb_variance = 3.0 / (
        8.0 * np.pi**2 * config.correlation_window_seconds * snr_linear * (high_hz**3 - low_hz**3)
    )
    quantisation_variance_share = config.lag_quantum_seconds**2 / 24.0
    return float(np.sqrt(quantisation_variance_share + crlb_variance))


def _stack_sensors(nodes: Sequence[OracleNode]) -> tuple[np.ndarray, list[int]]:
    positions: list[tuple[float, float, float]] = []
    owner_index: list[int] = []
    for node_index, node in enumerate(nodes):
        if len(node.sensor_positions_m) != len(node.sensor_in_band_snr_db):
            raise ValueError(f"node {node.node_id}: one SNR per sensor required")
        positions.extend(node.sensor_positions_m)
        owner_index.extend([node_index] * len(node.sensor_positions_m))
    return np.asarray(positions, dtype=np.float64), owner_index


def arrival_noise_covariance_s2(
    nodes: Sequence[OracleNode],
    source_position_m: np.ndarray,
    config: ObservabilityOracleConfig,
) -> np.ndarray:
    """Full arrival-time covariance: independent estimation noise on the diagonal,
    clock and rigid-position error shared inside each node's block."""
    sensors_m, owner_index = _stack_sensors(nodes)
    unit_from_sensor = source_position_m - sensors_m
    unit_from_sensor /= np.linalg.norm(unit_from_sensor, axis=1, keepdims=True)
    estimation_std = [
        sensor_arrival_noise_std_seconds(snr_db, config) for node in nodes for snr_db in node.sensor_in_band_snr_db
    ]
    covariance = np.diag(np.square(estimation_std))
    for row in range(len(owner_index)):
        for column in range(len(owner_index)):
            if owner_index[row] != owner_index[column]:
                continue
            node = nodes[owner_index[row]]
            covariance[row, column] += node.clock_offset_std_seconds**2
            covariance[row, column] += (
                node.position_std_m**2 * float(unit_from_sensor[row] @ unit_from_sensor[column]) / config.sound_speed_mps**2
            )
    return covariance


def _cholesky_factor(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError as error:
        raise ValueError("arrival noise covariance is not positive definite") from error


def position_covariance_m2(
    nodes: Sequence[OracleNode],
    source_position_m: np.ndarray,
    config: ObservabilityOracleConfig,
) -> np.ndarray:
    """Inverse Fisher information of the source position, emission time marginalised.

    Solved as least squares on the noise-whitened design (SVD) rather than by
    inverting the information matrix: far-field range information is ~1e14
    below bearing information, past what forming ``G^T C^-1 G`` survives.
    Times are carried in metres (x sound speed) so the columns are balanced.
    """
    sensors_m, _owner_index = _stack_sensors(nodes)
    unit_from_sensor = source_position_m - sensors_m
    unit_from_sensor /= np.linalg.norm(unit_from_sensor, axis=1, keepdims=True)
    design_m = np.hstack([unit_from_sensor, np.ones((len(sensors_m), 1))])
    noise_covariance_m2 = arrival_noise_covariance_s2(nodes, source_position_m, config) * config.sound_speed_mps**2
    whitened_design = np.linalg.solve(_cholesky_factor(noise_covariance_m2), design_m)
    prior_rows = np.diag(
        [1.0 / config.prior_position_std_m] * 3
        + [1.0 / (config.prior_emission_time_std_seconds * config.sound_speed_mps)]
    )
    try:
        _u, singular_values, right_singular_vectors = np.linalg.svd(np.vstack([whitened_design, prior_rows]))
    except np.linalg.LinAlgError as error:
        raise ValueError("observability design matrix did not decompose") from error
    covariance = (right_singular_vectors.T / singular_values**2) @ right_singular_vectors
    return covariance[:3, :3]


def mirror_across_sensor_plane(sensors_m: np.ndarray, source_position_m: np.ndarray) -> np.ndarray:
    """Reflection of the source across the least-squares plane of the sensors."""
    centroid = sensors_m.mean(axis=0)
    _u, _s, right_singular_vectors = np.linalg.svd(sensors_m - centroid)
    plane_normal = right_singular_vectors[-1]
    return source_position_m - 2.0 * float((source_position_m - centroid) @ plane_normal) * plane_normal


def mirror_separation_sigma(
    nodes: Sequence[OracleNode],
    source_position_m: np.ndarray,
    config: ObservabilityOracleConfig,
    distinct_beyond_m: float,
) -> float | None:
    """How many sigma separate the arrival vectors of the source and its mirror,
    after projecting out the unknown emission time. None when the mirror lies
    within ``distinct_beyond_m`` of the source (the ambiguity is harmless)."""
    sensors_m, _owner_index = _stack_sensors(nodes)
    mirror_m = mirror_across_sensor_plane(sensors_m, source_position_m)
    if float(np.linalg.norm(mirror_m - source_position_m)) <= distinct_beyond_m:
        return None
    arrival_difference_s = (
        np.linalg.norm(source_position_m - sensors_m, axis=1) - np.linalg.norm(mirror_m - sensors_m, axis=1)
    ) / config.sound_speed_mps
    noise_factor = _cholesky_factor(arrival_noise_covariance_s2(nodes, source_position_m, config))
    whitened_difference = np.linalg.solve(noise_factor, arrival_difference_s)
    whitened_ones = np.linalg.solve(noise_factor, np.ones(len(sensors_m)))
    # Residual after the best common emission-time shift, in whitened units.
    emission_time_shift = float(whitened_ones @ whitened_difference) / float(whitened_ones @ whitened_ones)
    return float(np.linalg.norm(whitened_difference - emission_time_shift * whitened_ones))


def observer_position_m(nodes: Sequence[OracleNode], node_positions_m: Sequence[np.ndarray]) -> np.ndarray:
    """The tier scorer's observer: the node itself for a lone node, the sensor
    centroid for a multi-node cluster (no single observing array)."""
    if len(nodes) == 1:
        return np.asarray(node_positions_m[0], dtype=np.float64)
    sensors_m, _owner_index = _stack_sensors(nodes)
    return sensors_m.mean(axis=0)


def _class_at_sigma_multiplier(
    multiplier: float,
    *,
    bearing_std_deg: float,
    range_std_m: float,
    max_axis_std_m: float,
    range_tolerance_m: float,
    mirror_sigma: float | None,
    tiers: TierScoringConfig,
) -> PassClass:
    if multiplier * bearing_std_deg > tiers.bearing_correct_max_error_deg:
        return PassClass.HEARD
    if multiplier * range_std_m > range_tolerance_m:
        return PassClass.BEARING_CORRECT
    mirror_blocks_fix = mirror_sigma is not None and mirror_sigma < multiplier
    if multiplier * max_axis_std_m > range_tolerance_m or mirror_blocks_fix:
        return PassClass.RANGE_RESOLVED
    return PassClass.FIX_3D


def assess_observability(
    nodes: Sequence[OracleNode],
    node_positions_m: Sequence[np.ndarray],
    source_position_m: Sequence[float],
    config: ObservabilityOracleConfig,
) -> ObservabilityVerdict:
    """Guaranteed and maximum honest pass class for one static source."""
    source_m = np.asarray(source_position_m, dtype=np.float64)
    observer_m = observer_position_m(nodes, node_positions_m)
    range_axes = slice(0, 2) if config.tiers.horizontal_only else slice(0, 3)
    offset_m = source_m - observer_m
    true_range_m = float(np.linalg.norm(offset_m[range_axes]))
    range_tolerance_m = max(
        config.tiers.range_resolved_max_absolute_error_m, config.tiers.range_resolved_max_relative_error * true_range_m
    )
    covariance = position_covariance_m2(nodes, source_m, config)
    horizontal_distance_m = float(np.linalg.norm(offset_m[:2]))
    tangential_unit = np.array([-offset_m[1], offset_m[0], 0.0]) / horizontal_distance_m
    radial_unit = np.zeros(3)
    radial_unit[range_axes] = offset_m[range_axes] / true_range_m
    bearing_std_deg = float(np.degrees(np.sqrt(tangential_unit @ covariance @ tangential_unit) / horizontal_distance_m))
    range_std_m = float(np.sqrt(radial_unit @ covariance @ radial_unit))
    max_axis_std_m = float(np.sqrt(np.max(np.linalg.eigvalsh(covariance))))
    mirror_sigma = mirror_separation_sigma(nodes, source_m, config, distinct_beyond_m=range_tolerance_m)
    heard = max(snr for node in nodes for snr in node.sensor_in_band_snr_db) >= config.heard_min_in_band_snr_db
    common = {
        "bearing_std_deg": bearing_std_deg,
        "range_std_m": range_std_m,
        "max_axis_std_m": max_axis_std_m,
        "range_tolerance_m": range_tolerance_m,
        "mirror_sigma": mirror_sigma,
        "tiers": config.tiers,
    }
    if heard:
        guaranteed = _class_at_sigma_multiplier(config.guaranteed_sigma_multiplier, **common)
        max_honest = _class_at_sigma_multiplier(config.honest_sigma_multiplier, **common)
    else:
        guaranteed = max_honest = PassClass.NOT_HEARD
    return ObservabilityVerdict(
        guaranteed_class=guaranteed,
        max_honest_class=max_honest,
        observer_position_m=tuple(float(axis) for axis in observer_m),
        true_range_m=true_range_m,
        range_tolerance_m=range_tolerance_m,
        bearing_std_deg=bearing_std_deg,
        range_std_m=range_std_m,
        max_axis_std_m=max_axis_std_m,
        mirror_separation_sigma=mirror_sigma,
        position_covariance_m2=covariance,
    )
