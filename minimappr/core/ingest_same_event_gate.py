"""Geometry-aware trigger cooldown for the Python ingest path.

Mirror of the sidecar's per-buffer cooldown (``dsp_localization_trigger.rs``):
a trigger on another node is the same event while it falls within the
cooldown plus the sound's travel time between the two nodes (plus a clock
margin), in either order (``localization_event_timing.same_event_trigger_window_seconds``).
So a far node's onset of one sound never opens a second solve of it, while a
second event on the same node still only waits out the plain cooldown.
Python ingest keeps one site-wide last trigger, as before.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minimappr.core.localization_event_timing import (
    EventTimingConfig,
    same_event_trigger_window_seconds,
    worst_two_node_clock_uncertainty_seconds,
)
from minimappr.core.pair_timing_uncertainty import ClockUncertaintyPolicy

_NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True, slots=True)
class IngestTrigger:
    event_time_ns: int
    node_id: str
    # None: the node reported no position; the site's largest baseline is assumed.
    node_position_m: tuple[float, float, float] | None
    clock_uncertainty_seconds: float


class SameEventTriggerGate:
    def __init__(
        self,
        *,
        trigger_cooldown_seconds: float,
        event_timing_config: EventTimingConfig,
        sound_speed_mps: float,
        clock_policy: ClockUncertaintyPolicy,
    ) -> None:
        self._trigger_cooldown_seconds = float(trigger_cooldown_seconds)
        self._event_timing_config = event_timing_config
        self._sound_speed_mps = float(sound_speed_mps)
        self._clock_policy = clock_policy
        self._last_trigger: IngestTrigger | None = None

    @property
    def last_trigger_ns(self) -> int:
        return self._last_trigger.event_time_ns if self._last_trigger is not None else 0

    def trigger_for(
        self,
        *,
        event_time_ns: int,
        node_id: str,
        node_position_m,
        time_quality: str | None,
    ) -> IngestTrigger:
        position = None if node_position_m is None else tuple(float(axis) for axis in node_position_m)
        return IngestTrigger(
            event_time_ns=int(event_time_ns),
            node_id=node_id,
            node_position_m=position,
            clock_uncertainty_seconds=self._clock_policy.class_uncertainty_for(time_quality),
        )

    def is_same_event_as_last(self, trigger: IngestTrigger) -> bool:
        last = self._last_trigger
        if last is None:
            return False
        if last.node_id == trigger.node_id:
            node_distance_m, pair_clock_uncertainty_seconds = 0.0, 0.0
        else:
            node_distance_m = self._node_distance_m(last, trigger)
            pair_clock_uncertainty_seconds = worst_two_node_clock_uncertainty_seconds(
                [last.clock_uncertainty_seconds, trigger.clock_uncertainty_seconds]
            )
        window_seconds = same_event_trigger_window_seconds(
            self._trigger_cooldown_seconds,
            node_distance_m,
            pair_clock_uncertainty_seconds,
            self._sound_speed_mps,
            self._event_timing_config,
        )
        return abs(trigger.event_time_ns - last.event_time_ns) < window_seconds * _NANOSECONDS_PER_SECOND

    def commit(self, trigger: IngestTrigger) -> None:
        self._last_trigger = trigger

    def _node_distance_m(self, last: IngestTrigger, trigger: IngestTrigger) -> float:
        if last.node_position_m is None or trigger.node_position_m is None:
            # Unknown placement: the safety-bounded lag (never a new event early).
            return float("inf")
        return float(np.linalg.norm(np.subtract(last.node_position_m, trigger.node_position_m)))


def same_event_gate_from_settings(settings) -> SameEventTriggerGate:
    return SameEventTriggerGate(
        trigger_cooldown_seconds=settings.trigger_cooldown_seconds,
        event_timing_config=EventTimingConfig.from_settings(
            settings, node_local_window_seconds=settings.localization_window_seconds
        ),
        sound_speed_mps=_default_sound_speed_mps(settings),
        clock_policy=ClockUncertaintyPolicy.from_settings(settings),
    )


def default_same_event_gate(localization_config) -> SameEventTriggerGate:
    """For an ingest processor built without Settings (tests): Settings defaults
    for everything but the cooldown and window it was given."""
    from minimappr.config import Settings

    defaults = Settings()
    return SameEventTriggerGate(
        trigger_cooldown_seconds=localization_config.trigger_cooldown_seconds,
        event_timing_config=EventTimingConfig.from_settings(
            defaults, node_local_window_seconds=localization_config.localization_window_seconds
        ),
        sound_speed_mps=_default_sound_speed_mps(defaults),
        clock_policy=ClockUncertaintyPolicy.from_settings(defaults),
    )


def _default_sound_speed_mps(settings) -> float:
    from minimappr.core.localization import speed_of_sound_mps

    return speed_of_sound_mps(settings.default_temperature_c, settings.default_humidity)
