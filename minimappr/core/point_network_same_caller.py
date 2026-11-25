"""Whether two localized solves are one caller (plan item P7).

In a point network every reporter can deliver its own solve of one call
(different solve ids). Close solves are one caller by separation alone (the
duplicate-merge distance, which also covers the next note of a perched bird).
Farther apart they are still one caller when they are the same event: same
site cluster (when both name one), overlapping solve windows, and positions
that agree within the two solves' stated covariances (a chi-square gate on
the Mahalanobis distance of the separation under the summed covariance). At
hundreds of metres independent solves differ by tens of metres, so a fixed
distance alone would make one call mask itself.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.stats import chi2

from minimappr.core.point_network_clip_config import PointNetworkClipConfig
from minimappr.core.point_network_localized_calls import LocalizedCall, seconds_to_nanoseconds


@lru_cache(maxsize=16)
def _chi_square_threshold(probability: float, degrees_of_freedom: int) -> float:
    return float(chi2.ppf(probability, degrees_of_freedom))


def same_event_mahalanobis_squared(call: LocalizedCall, other: LocalizedCall) -> tuple[float, int] | None:
    """(squared Mahalanobis distance, dimensions) of the separation under the
    summed covariance; None when either solve stated none or the sum is singular."""
    if call.position_covariance_m2 is None or other.position_covariance_m2 is None:
        return None
    covariance = np.asarray(call.position_covariance_m2, dtype=np.float64)
    other_covariance = np.asarray(other.position_covariance_m2, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape != other_covariance.shape or covariance.shape[0] != covariance.shape[1]:
        return None
    dimensions = min(covariance.shape[0], 3)
    separation_m = (np.asarray(other.source_position_m, dtype=np.float64) - call.source_position_m)[:dimensions]
    combined_covariance = covariance[:dimensions, :dimensions] + other_covariance[:dimensions, :dimensions]
    if not np.all(np.isfinite(combined_covariance)):
        return None
    try:
        weighted = np.linalg.solve(combined_covariance, separation_m)
    except np.linalg.LinAlgError:
        return None
    return float(separation_m @ weighted), dimensions


def _solve_windows_overlap(call: LocalizedCall, other: LocalizedCall, separation_m: float) -> bool:
    """``other``'s window, moved onto ``call``'s reference timeline, overlaps ``call``'s.

    The move uses ``other``'s own position, which is off by up to about the
    separation, so the windows are widened by that much propagation time.
    """
    slack_ns = seconds_to_nanoseconds(separation_m / call.sound_speed_mps)
    other_start_ns, other_end_ns = other.event_span_at(call.alignment_reference_position_m)
    return other_end_ns + slack_ns > call.event_span_start_ns and other_start_ns - slack_ns < call.event_span_end_ns


def is_same_caller(call: LocalizedCall, other: LocalizedCall, config: PointNetworkClipConfig) -> bool:
    separation_m = float(np.linalg.norm(np.asarray(other.source_position_m) - call.source_position_m))
    if separation_m <= config.distinct_caller_min_separation_m:
        return True
    if call.cluster_id is not None and other.cluster_id is not None and call.cluster_id != other.cluster_id:
        return False
    if not _solve_windows_overlap(call, other, separation_m):
        return False
    gate = same_event_mahalanobis_squared(call, other)
    if gate is None:
        return False
    mahalanobis_squared, dimensions = gate
    return mahalanobis_squared <= _chi_square_threshold(config.same_event_gate_probability, dimensions)


__all__ = ["is_same_caller", "same_event_mahalanobis_squared"]
