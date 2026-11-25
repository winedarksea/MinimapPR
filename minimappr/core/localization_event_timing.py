"""Event span, coverage-wait cap and same-event window derived from node geometry.

Mirror of ``minimappr-core/src/localization_event_timing.rs``; both are pinned
by ``tests/fixtures/localization_event_timing_vectors.json``.

Why geometry: one sound reaches every node within ``baseline / c`` of the
first, whatever the source range. The largest cross-node baseline, not a fixed
cap, bounds how much audio an event solve reads, how long the coverage wait
holds for the far node, and how long after one node's onset another node's
onset is still the same event. A compact site keeps a short span and wait; a
kilometre point network gets a multi-second span with no cap to set. An
explicit ``localization_coverage_total_cap_seconds`` > 0 overrides the cap and
also bounds the lag (the pre-derivation behaviour).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# The far sensor may hear the event up to the lag before or after the trigger.
_LAG_EXTENT_SIDES = 2.0
# A sensor lagging by this fraction of the node-local window still holds at
# least half of a trigger-length event inside it.
_NODE_LOCAL_WINDOW_TOLERATED_LAG_FRACTION = 0.5
_MINIMUM_SOUND_SPEED_MPS = 1.0


@dataclass(frozen=True)
class EventTimingConfig:
    node_local_window_seconds: float
    coverage_wait_enabled: bool
    # <= 0: derive the cap from geometry; > 0: explicit override.
    explicit_coverage_total_cap_seconds: float
    coverage_delivery_margin_seconds: float
    # Same-event window only: onsets are stamped by each node's own clock.
    clock_margin_sigma: float
    # Hard safety bound on the derived lag (and so on span and wait).
    max_cross_node_lag_seconds: float
    max_buffer_seconds: float

    @classmethod
    def from_settings(cls, settings, *, node_local_window_seconds: float) -> EventTimingConfig:
        """The sensor buffer is the effective one: ``max_sensor_buffer_seconds``
        raised to hold the safety bound's largest span (as both runtimes size it)."""
        configured = cls(
            node_local_window_seconds=float(node_local_window_seconds),
            coverage_wait_enabled=bool(settings.localization_coverage_wait_all_streaming_nodes_enabled),
            explicit_coverage_total_cap_seconds=float(settings.localization_coverage_total_cap_seconds),
            coverage_delivery_margin_seconds=float(settings.localization_coverage_delivery_margin_seconds),
            clock_margin_sigma=float(settings.localization_event_span_clock_margin_sigma),
            max_cross_node_lag_seconds=float(settings.localization_event_span_max_cross_node_lag_seconds),
            max_buffer_seconds=float(settings.max_sensor_buffer_seconds),
        )
        return dataclasses.replace(
            configured, max_buffer_seconds=max(configured.max_buffer_seconds, required_sensor_buffer_seconds(configured))
        )

    def _explicit_cap_seconds(self) -> float | None:
        cap = self.explicit_coverage_total_cap_seconds
        return cap if math.isfinite(cap) and cap > 0.0 else None

    def lag_bound_seconds(self) -> float:
        # The buffer must hold the whole span plus the delivery skew.
        buffer_bound_seconds = max(
            0.0,
            (self.max_buffer_seconds - max(0.0, self.coverage_delivery_margin_seconds) - self.node_local_window_seconds)
            / _LAG_EXTENT_SIDES,
        )
        bound_seconds = min(max(0.0, self.max_cross_node_lag_seconds), buffer_bound_seconds)
        explicit_cap_seconds = self._explicit_cap_seconds()
        if explicit_cap_seconds is not None:
            bound_seconds = min(bound_seconds, explicit_cap_seconds)
        return bound_seconds

    def coverage_cap_for_lag(self, lag_seconds: float) -> float:
        explicit_cap_seconds = self._explicit_cap_seconds()
        if explicit_cap_seconds is not None:
            return explicit_cap_seconds
        return lag_seconds + max(0.0, self.coverage_delivery_margin_seconds)


@dataclass(frozen=True)
class CrossNodeGeometry:
    # Largest distance between sensors on different nodes.
    max_cross_node_distance_m: float


@dataclass(frozen=True)
class DerivedEventTiming:
    cross_node_lag_seconds: float
    event_span_seconds: float
    coverage_total_cap_seconds: float
    # The geometric lag exceeded a bound: the far node may miss the event.
    lag_bounded: bool


def pair_arrival_lag_seconds(
    distance_m: float, pair_clock_uncertainty_seconds: float, sound_speed_mps: float, clock_margin_sigma: float
) -> float:
    return max(0.0, distance_m) / max(_MINIMUM_SOUND_SPEED_MPS, sound_speed_mps) + max(0.0, clock_margin_sigma) * max(
        0.0, pair_clock_uncertainty_seconds
    )


def derive_event_timing(
    geometry: CrossNodeGeometry | None, sound_speed_mps: float, config: EventTimingConfig
) -> DerivedEventTiming:
    window_seconds = config.node_local_window_seconds
    if geometry is None:
        return DerivedEventTiming(0.0, window_seconds, config.coverage_cap_for_lag(0.0), False)
    # No clock margin on the span: GCC-PHAT searches a pair only to baseline / c,
    # so audio past that lag cannot move a peak (see the Rust module docs).
    geometric_lag_seconds = pair_arrival_lag_seconds(geometry.max_cross_node_distance_m, 0.0, sound_speed_mps, 0.0)
    lag_bound_seconds = config.lag_bound_seconds()
    lag_seconds = min(geometric_lag_seconds, lag_bound_seconds)
    if config.coverage_wait_enabled:
        lag_beyond_window_seconds = max(0.0, lag_seconds - _NODE_LOCAL_WINDOW_TOLERATED_LAG_FRACTION * window_seconds)
        span_seconds = max(
            min(window_seconds + _LAG_EXTENT_SIDES * lag_beyond_window_seconds, config.max_buffer_seconds),
            window_seconds,
        )
    else:
        # Without the coverage wait the later half never arrives before the solve.
        span_seconds = window_seconds
    return DerivedEventTiming(
        cross_node_lag_seconds=lag_seconds,
        event_span_seconds=span_seconds,
        coverage_total_cap_seconds=config.coverage_cap_for_lag(lag_seconds),
        lag_bounded=geometric_lag_seconds > lag_bound_seconds,
    )


def required_sensor_buffer_seconds(config: EventTimingConfig) -> float:
    """Buffer that holds the largest span the safety bound allows plus the delivery skew."""
    return (
        config.node_local_window_seconds
        + _LAG_EXTENT_SIDES * max(0.0, config.max_cross_node_lag_seconds)
        + max(0.0, config.coverage_delivery_margin_seconds)
    )


def same_event_trigger_window_seconds(
    trigger_cooldown_seconds: float,
    node_distance_m: float,
    pair_clock_uncertainty_seconds: float,
    sound_speed_mps: float,
    config: EventTimingConfig,
) -> float:
    """Two triggers this close (either order) on nodes this far apart are one event.

    A cooldown of 0 switches the cooldown off entirely (every trigger opens a solve).
    """
    if not trigger_cooldown_seconds > 0.0:
        return 0.0
    pair_lag_seconds = min(
        pair_arrival_lag_seconds(node_distance_m, pair_clock_uncertainty_seconds, sound_speed_mps, config.clock_margin_sigma),
        config.lag_bound_seconds(),
    )
    return trigger_cooldown_seconds + pair_lag_seconds


def cross_node_geometry(
    sensor_positions_m: Mapping[str, np.ndarray], sensor_node_ids: Mapping[str, str]
) -> CrossNodeGeometry | None:
    """Largest cross-node sensor distance; None when every sensor is on one node."""
    sensor_ids = [sensor_id for sensor_id in sensor_positions_m if sensor_id in sensor_node_ids]
    node_ids = {sensor_node_ids[sensor_id] for sensor_id in sensor_ids}
    if len(node_ids) < 2:
        return None
    max_distance_m = 0.0
    for index, sensor_a in enumerate(sensor_ids):
        position_a = np.asarray(sensor_positions_m[sensor_a], dtype=np.float64)
        for sensor_b in sensor_ids[index + 1 :]:
            if sensor_node_ids[sensor_a] == sensor_node_ids[sensor_b]:
                continue
            distance_m = float(np.linalg.norm(position_a - np.asarray(sensor_positions_m[sensor_b], dtype=np.float64)))
            max_distance_m = max(max_distance_m, distance_m)
    return CrossNodeGeometry(max_cross_node_distance_m=max_distance_m)


def worst_two_node_clock_uncertainty_seconds(node_uncertainties_seconds: list[float]) -> float:
    """Root-sum-square of the two largest node clock uncertainties (the worst pair)."""
    largest_two = sorted((max(0.0, value) for value in node_uncertainties_seconds), reverse=True)[:2]
    return math.sqrt(sum(value * value for value in largest_two))
