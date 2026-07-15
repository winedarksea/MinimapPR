"""Bridge: solve a tetrahedral position from Rust-measured pairwise TDOAs.

The Rust sidecar already measures pairwise TDOAs and a steering direction for the
single-node tetrahedral array, but historically it also produced its own SRP-PHAT
*position*, bypassing the Python Cartesian solver (and its observability/range
handling). This module re-homes the *solve* to Python: feed Rust's TDOAs + bearing
into :func:`solve_cartesian_tdoa` so the production single-node path and the Python
multi-node fuser share one estimator. The sidecar keeps its TDOA front-end.

Channel indices in the Rust ``pair_tdoas`` are positional and align with the node's
ordered sensors (channel ``i`` -> ``ordered_sensor_ids[i]``); the sign/order of the
lag matches the Python GCC-PHAT convention (guarded by the Rust oracle test
``sirith_fixture_pair_tdoas_match_python_oracle_signs_and_order``).
"""

from __future__ import annotations

import numpy as np

from minimappr.core.cartesian_tdoa import (
    localization_result_from_cartesian_solve,
    solve_cartesian_tdoa,
)
from minimappr.core.subspace_bearing import BearingPrior
from minimappr.core.tdoa_measurements import (
    PairTdoaMeasurement,
    normalized_pair_quality,
    solve_reference_tdoas,
)
from minimappr.models import LocalizationResult


MIN_PAIR_MEASUREMENTS = 3


def _bearing_prior_from_direction(
    steering_direction: tuple[float, float, float] | None,
    *,
    quality: float,
    bearing_strength: float = 1.0,
) -> BearingPrior | None:
    """Build a ``BearingPrior`` from the sidecar's single-node steering direction.

    ``quality`` is the TDOA-pair-derived confidence in that direction (mean
    GCC-PHAT correlation peak); ``bearing_strength`` is the operator-configured
    ``localization_node_bearing_strength`` knob. Combining them the same way
    ``core/spatial_constraints.py`` does (``strength = bearing_strength *
    quality``) keeps this single-node Rust-fed path responsive to the same
    "how much do I trust the bearing" dial as the direct multi-node Python
    solve path (``core/localization_dispatch.py``) — previously this path
    ignored the config knob entirely and always solved as if strength == 1.0.
    """
    if steering_direction is None:
        return None
    direction = np.asarray(steering_direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1.0e-9:
        return None
    clipped_quality = float(np.clip(quality, 0.0, 1.0))
    strength = max(float(bearing_strength) * clipped_quality, 0.01)
    return BearingPrior(
        direction=direction / norm,
        strength=strength,
        quality=clipped_quality,
    )


def solve_localization_from_rust_tdoas(
    *,
    sensor_positions: dict[str, np.ndarray],
    ordered_sensor_ids: list[str],
    pair_tdoas: list[dict],
    steering_direction: tuple[float, float, float] | None,
    sound_speed_mps: float,
    sample_rate_hz: int,
    interpolation_factor: int,
    far_field_default_range_m: float,
    far_field_prior_radial_std_m: float | None = None,
    bearing_strength: float = 1.0,
) -> LocalizationResult | None:
    """Return a Python Cartesian solve from Rust TDOAs, or None if not solvable.

    None signals the caller to fall back to the sidecar's own estimate (missing
    TDOAs/geometry, degenerate input, or solver failure).

    ``bearing_strength`` should be ``Settings.localization_node_bearing_strength``
    so this single-node path honours the same operator-configured DOA-vs-TDOA
    trust dial as the multi-node dispatch path in ``core/localization_dispatch.py``.
    """
    if sample_rate_hz <= 0 or sound_speed_mps <= 0.0:
        return None
    sensor_count = len(ordered_sensor_ids)
    if sensor_count < 4:
        return None

    measurements: list[PairTdoaMeasurement] = []
    confidence_by_sensor: dict[str, float] = {sensor_id: 0.0 for sensor_id in ordered_sensor_ids}
    for pair in pair_tdoas:
        try:
            channel_a = int(pair["ch_a"])
            channel_b = int(pair["ch_b"])
            lag_seconds = float(pair["lag_seconds"])
            confidence = float(pair.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= channel_a < sensor_count and 0 <= channel_b < sensor_count):
            continue
        if channel_a == channel_b or not np.isfinite(lag_seconds):
            continue
        sensor_a = ordered_sensor_ids[channel_a]
        sensor_b = ordered_sensor_ids[channel_b]
        quality = normalized_pair_quality(confidence)
        measurements.append(
            PairTdoaMeasurement(
                sensor_a=sensor_a,
                sensor_b=sensor_b,
                tdoa_seconds=lag_seconds,
                correlation_peak=confidence,
                weight=max(0.01, quality),
            )
        )
        confidence_by_sensor[sensor_a] += quality
        confidence_by_sensor[sensor_b] += quality

    if len(measurements) < MIN_PAIR_MEASUREMENTS:
        return None

    solve_positions = {sensor_id: sensor_positions[sensor_id] for sensor_id in ordered_sensor_ids}
    reference_sensor = max(ordered_sensor_ids, key=lambda sensor_id: confidence_by_sensor[sensor_id])
    reference_tdoa_s = solve_reference_tdoas(
        measurements=measurements,
        sensor_ids=ordered_sensor_ids,
        reference_sensor=reference_sensor,
    )

    mean_confidence = float(np.mean([m.correlation_peak for m in measurements]))
    bearing_prior = _bearing_prior_from_direction(
        steering_direction,
        quality=mean_confidence,
        bearing_strength=bearing_strength,
    )

    try:
        solve = solve_cartesian_tdoa(
            sensor_positions=solve_positions,
            measurements=measurements,
            sound_speed_mps=sound_speed_mps,
            sample_rate_hz=sample_rate_hz,
            interpolation_factor=max(1, interpolation_factor),
            reference_sensor=reference_sensor,
            reference_tdoa_s=reference_tdoa_s,
            bearing_prior=bearing_prior,
            far_field_initial_range_m=far_field_default_range_m,
            far_field_prior_radial_std_m=far_field_prior_radial_std_m,
            radial_refinement_enabled=True,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None

    return localization_result_from_cartesian_solve(
        solve=solve,
        reference_sensor=reference_sensor,
        reference_tdoa_s=reference_tdoa_s,
    )
