"""Which other localized callers compete with a call in one node's clip (plan item P7).

Measuring two callers' levels inside one clip cannot separate them once their
audio overlaps there: each caller's span holds the other's energy too. So each
caller's level is measured once, at the network node nearest to it (where it
dominates by spreading), and carried to any other node by spherical spreading
(-20 log10 of the distance ratio). A competitor counts when, at the clip's
node, its predicted level is within the comparable margin of the call's own
predicted level and above that node's noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minimappr.core.point_network_clip_config import PointNetworkClipConfig
from minimappr.core.point_network_localized_calls import LocalizedCall

# Spherical spreading is undefined at the source; below a metre the caller is
# effectively on the microphone, so distances are floored there.
MIN_SPREADING_DISTANCE_M = 1.0


@dataclass(frozen=True, slots=True)
class CallerReferenceLevel:
    """A caller's in-band level measured at its nearest network node."""

    call: LocalizedCall
    reference_position_m: np.ndarray
    level_db: float

    def predicted_level_db_at(self, position_m: np.ndarray) -> float:
        source = self.call.source_position_m
        reference_distance_m = max(MIN_SPREADING_DISTANCE_M, float(np.linalg.norm(source - self.reference_position_m)))
        distance_m = max(MIN_SPREADING_DISTANCE_M, float(np.linalg.norm(source - position_m)))
        return self.level_db - 20.0 * float(np.log10(distance_m / reference_distance_m))


@dataclass(frozen=True, slots=True)
class CompetingCaller:
    call_id: str
    separation_m: float
    level_relative_to_call_db: float


def nearest_position_id(source_position_m: np.ndarray, positions_m: dict[str, np.ndarray]) -> str:
    return min(positions_m, key=lambda key: float(np.linalg.norm(source_position_m - positions_m[key])))


def competing_callers_at_node(
    *,
    call_level: CallerReferenceLevel,
    other_levels: list[CallerReferenceLevel],
    node_position_m: np.ndarray,
    node_noise_floor_db: float,
    clip_span_ns: tuple[int, int],
    config: PointNetworkClipConfig,
) -> list[CompetingCaller]:
    """Other callers whose audio at this node overlaps the clip at a comparable, audible level."""
    call_level_db = call_level.predicted_level_db_at(node_position_m)
    clip_start_ns, clip_end_ns = clip_span_ns
    competing: list[CompetingCaller] = []
    for other in other_levels:
        span_start_ns, span_end_ns = other.call.event_span_at(node_position_m)
        if span_end_ns <= clip_start_ns or span_start_ns >= clip_end_ns:
            continue
        other_level_db = other.predicted_level_db_at(node_position_m)
        level_margin_db = other_level_db - call_level_db
        if (
            level_margin_db >= -config.overlapping_caller_comparable_level_db
            and other_level_db - node_noise_floor_db >= config.overlapping_caller_min_audible_snr_db
        ):
            competing.append(
                CompetingCaller(
                    call_id=other.call.call_id,
                    separation_m=float(np.linalg.norm(other.call.source_position_m - call_level.call.source_position_m)),
                    level_relative_to_call_db=level_margin_db,
                )
            )
    return competing


__all__ = [
    "CallerReferenceLevel",
    "CompetingCaller",
    "competing_callers_at_node",
    "nearest_position_id",
]
