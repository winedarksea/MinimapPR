"""Geometry-derived event span / coverage cap / same-event window (Python side).

The vectors are shared with ``minimappr-core/src/localization_event_timing_tests.rs``,
so both languages reproduce every case.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from minimappr.config import Settings
from minimappr.core.ingest_same_event_gate import SameEventTriggerGate, same_event_gate_from_settings
from minimappr.core.localization_event_timing import (
    CrossNodeGeometry,
    EventTimingConfig,
    cross_node_geometry,
    derive_event_timing,
    required_sensor_buffer_seconds,
    same_event_trigger_window_seconds,
)
from minimappr.core.pair_timing_uncertainty import ClockUncertaintyPolicy

VECTORS = json.loads((Path(__file__).parent / "fixtures" / "localization_event_timing_vectors.json").read_text())
SOUND_SPEED_MPS = 343.0
NANOSECONDS_PER_SECOND = 1_000_000_000


def _config(raw: dict) -> EventTimingConfig:
    return EventTimingConfig(**raw)


@pytest.mark.parametrize("case", VECTORS["event_timing_cases"], ids=lambda case: case["name"])
def test_derived_event_timing_matches_the_shared_vectors(case) -> None:
    geometry = CrossNodeGeometry(**case["geometry"]) if case["geometry"] is not None else None
    derived = derive_event_timing(geometry, case["sound_speed_mps"], _config(case["config"]))
    expected = case["expected"]
    for key in ("cross_node_lag_seconds", "event_span_seconds", "coverage_total_cap_seconds"):
        assert getattr(derived, key) == pytest.approx(expected[key], abs=1e-9), key
    assert derived.lag_bounded is expected["lag_bounded"]


@pytest.mark.parametrize("case", VECTORS["same_event_window_cases"], ids=lambda case: case["name"])
def test_same_event_window_matches_the_shared_vectors(case) -> None:
    window_seconds = same_event_trigger_window_seconds(
        case["trigger_cooldown_seconds"],
        case["node_distance_m"],
        case["pair_clock_uncertainty_seconds"],
        case["sound_speed_mps"],
        _config(case["config"]),
    )
    assert window_seconds == pytest.approx(case["expected_window_seconds"], abs=1e-9)


def _settings_config(**overrides) -> EventTimingConfig:
    return EventTimingConfig.from_settings(Settings(**overrides), node_local_window_seconds=0.08)


def test_compact_site_keeps_a_short_span_and_the_old_wait() -> None:
    derived = derive_event_timing(CrossNodeGeometry(20.0), SOUND_SPEED_MPS, _settings_config())
    assert derived.event_span_seconds < 0.15
    assert derived.coverage_total_cap_seconds == pytest.approx(1.0 + 20.0 / SOUND_SPEED_MPS)


def test_kilometre_spread_gets_a_multi_second_span() -> None:
    derived = derive_event_timing(CrossNodeGeometry(1_500.0), SOUND_SPEED_MPS, _settings_config())
    assert derived.event_span_seconds == pytest.approx(2.0 * 1_500.0 / SOUND_SPEED_MPS)
    assert derived.coverage_total_cap_seconds > 5.0
    assert not derived.lag_bounded


def test_explicit_override_and_safety_bound() -> None:
    explicit = derive_event_timing(
        CrossNodeGeometry(1_500.0), SOUND_SPEED_MPS, _settings_config(localization_coverage_total_cap_seconds=1.0)
    )
    assert explicit.event_span_seconds == pytest.approx(2.0)
    assert explicit.coverage_total_cap_seconds == pytest.approx(1.0)
    bounded = derive_event_timing(CrossNodeGeometry(50_000.0), SOUND_SPEED_MPS, _settings_config())
    assert bounded.lag_bounded
    assert bounded.cross_node_lag_seconds == pytest.approx(Settings().localization_event_span_max_cross_node_lag_seconds)


def test_default_buffer_holds_the_largest_span_the_safety_bound_allows() -> None:
    settings = Settings()
    config = EventTimingConfig.from_settings(settings, node_local_window_seconds=settings.localization_window_seconds)
    assert settings.max_sensor_buffer_seconds >= required_sensor_buffer_seconds(config)


def test_cross_node_geometry_uses_only_cross_node_pairs() -> None:
    positions = {
        "a:0": np.array([0.0, 0.0, 0.0]),
        "a:1": np.array([0.05, 0.0, 0.0]),
        "b:0": np.array([30.0, 40.0, 0.0]),
    }
    assert cross_node_geometry(positions, {"a:0": "a", "a:1": "a", "b:0": "b"}).max_cross_node_distance_m == pytest.approx(
        np.linalg.norm([30.0, 40.0, 0.0])
    )
    assert cross_node_geometry(positions, {"a:0": "a", "a:1": "a", "b:0": "a"}) is None


def _gate(**overrides) -> SameEventTriggerGate:
    return same_event_gate_from_settings(Settings(trigger_cooldown_seconds=0.8, **overrides))


def _seconds_ns(seconds: float) -> int:
    return int(round(seconds * NANOSECONDS_PER_SECOND))


def test_far_nodes_onset_inside_the_inter_node_lag_is_the_same_event() -> None:
    gate = _gate()
    near = gate.trigger_for(event_time_ns=_seconds_ns(10.0), node_id="near", node_position_m=(0.0, 0.0, 0.0), time_quality="gps_locked")
    gate.commit(near)
    far_position = (1_500.0, 0.0, 0.0)
    lag_seconds = 1_500.0 / 343.2
    for offset_seconds, same_event in ((lag_seconds, True), (lag_seconds + 0.85, False), (-lag_seconds, True)):
        far = gate.trigger_for(
            event_time_ns=_seconds_ns(10.0 + offset_seconds), node_id="far", node_position_m=far_position, time_quality="gps_locked"
        )
        assert gate.is_same_event_as_last(far) is same_event, offset_seconds
    # The trigger node itself fires again after the plain cooldown.
    again = gate.trigger_for(event_time_ns=_seconds_ns(10.81), node_id="near", node_position_m=(0.0, 0.0, 0.0), time_quality="gps_locked")
    assert not gate.is_same_event_as_last(again)


def test_compact_pair_keeps_distinct_events_distinct() -> None:
    gate = _gate()
    gate.commit(gate.trigger_for(event_time_ns=0, node_id="a", node_position_m=(0.0, 0.0, 0.0), time_quality="gps_locked"))
    second = gate.trigger_for(event_time_ns=_seconds_ns(0.9), node_id="b", node_position_m=(20.0, 0.0, 0.0), time_quality="gps_locked")
    assert not gate.is_same_event_as_last(second)


def test_weak_clock_widens_the_same_event_window() -> None:
    policy = ClockUncertaintyPolicy.from_settings(Settings())
    margin_seconds = Settings().localization_event_span_clock_margin_sigma * np.hypot(
        policy.class_uncertainty_for("free_running"), policy.class_uncertainty_for("free_running")
    )
    gate = _gate()
    gate.commit(gate.trigger_for(event_time_ns=0, node_id="a", node_position_m=(0.0, 0.0, 0.0), time_quality="free_running"))
    just_past_cooldown = gate.trigger_for(
        event_time_ns=_seconds_ns(0.8 + 0.5 * margin_seconds), node_id="b", node_position_m=(0.0, 0.0, 0.0), time_quality="free_running"
    )
    assert gate.is_same_event_as_last(just_past_cooldown)


def test_python_ingest_span_follows_the_registry_geometry() -> None:
    import asyncio
    from types import SimpleNamespace

    from minimappr.core.python_ingest_event_span import python_ingest_event_timing, required_coverage_deadline_seconds

    class _Registry:
        async def sensor_node_ids(self, sensor_ids):
            return {sensor_id: sensor_id.split(":")[0] for sensor_id in sensor_ids}

    environment = SimpleNamespace(
        get_conditions=lambda location_m: SimpleNamespace(temperature_c=20.0, humidity_fraction=0.5)
    )
    positions = {"a:0": np.zeros(3), "a:1": np.array([0.05, 0.0, 0.0]), "b:0": np.array([1_500.0, 0.0, 0.0])}
    timing = asyncio.run(
        python_ingest_event_timing(
            settings=Settings(),
            registry=_Registry(),
            environment_provider=environment,
            sensor_positions=positions,
            node_local_window_seconds=0.08,
        )
    )
    assert timing.event_span_seconds > 8.0
    assert timing.coverage_total_cap_seconds > timing.cross_node_lag_seconds
    # Even the trigger node needs the span's future half before it covers it.
    assert required_coverage_deadline_seconds(0.3, timing, 0.08) == pytest.approx(0.3 + (timing.event_span_seconds - 0.08) / 2)
