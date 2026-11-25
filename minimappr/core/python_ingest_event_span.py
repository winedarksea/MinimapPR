"""Geometry-derived event span and coverage cap for a Python-ingest solve.

The sidecar derives these per layout (``localization_event_span.rs``); the
Python ingest path derives them per candidate from the registry's sensor
positions, with the same ``localization_event_timing`` rules, so a km-spread
cluster's far node is waited for and read on either path.
"""

from __future__ import annotations

import numpy as np

from minimappr.core.localization import speed_of_sound_mps
from minimappr.core.localization_event_timing import (
    DerivedEventTiming,
    EventTimingConfig,
    cross_node_geometry,
    derive_event_timing,
)


async def python_ingest_event_timing(
    *,
    settings,
    registry,
    environment_provider,
    sensor_positions: dict[str, np.ndarray],
    node_local_window_seconds: float,
) -> DerivedEventTiming:
    sensor_ids = sorted(sensor_positions)
    sensor_node_ids = await registry.sensor_node_ids(sensor_ids)
    centroid_m = np.mean(np.stack([np.asarray(sensor_positions[sensor_id], dtype=np.float64) for sensor_id in sensor_ids]), axis=0)
    conditions = environment_provider.get_conditions(tuple(float(axis) for axis in centroid_m))
    return derive_event_timing(
        cross_node_geometry(sensor_positions, sensor_node_ids),
        speed_of_sound_mps(conditions.temperature_c, conditions.humidity_fraction),
        EventTimingConfig.from_settings(settings, node_local_window_seconds=node_local_window_seconds),
    )


def required_coverage_deadline_seconds(base_deadline_seconds: float, event_timing: DerivedEventTiming, node_local_window_seconds: float) -> float:
    """The span reaches past the trigger by (span - window) / 2 of audio that has
    not been recorded yet; even the trigger node needs that long to cover it."""
    future_audio_seconds = max(0.0, (event_timing.event_span_seconds - node_local_window_seconds) / 2.0)
    return base_deadline_seconds + future_audio_seconds


def event_span_sensor_buffer_seconds(settings) -> float:
    """``max_sensor_buffer_seconds``, raised to hold the largest span the safety
    bound allows (the sidecar applies the same rule at startup)."""
    return EventTimingConfig.from_settings(
        settings, node_local_window_seconds=settings.localization_window_seconds
    ).max_buffer_seconds
