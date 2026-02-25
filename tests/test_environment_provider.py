from __future__ import annotations

import time

import pytest

from minimappr.core.environment import LiveEnvironmentProvider


def test_live_environment_provider_uses_fallback_without_samples() -> None:
    provider = LiveEnvironmentProvider(
        fallback_temperature_c=18.0,
        fallback_humidity_fraction=0.45,
        max_reading_age_seconds=300.0,
    )
    conditions = provider.get_conditions()
    assert conditions.temperature_c == pytest.approx(18.0, abs=1e-6)
    assert conditions.humidity_fraction == pytest.approx(0.45, abs=1e-6)
    assert conditions.metadata["source"] == "static_fallback"


def test_live_environment_provider_prefers_nearest_temperature_sample() -> None:
    provider = LiveEnvironmentProvider(
        fallback_temperature_c=20.0,
        fallback_humidity_fraction=0.5,
        max_reading_age_seconds=300.0,
    )
    now_ns = time.time_ns()
    provider.ingest_sample(
        node_id="node-a",
        timestamp_ns=now_ns,
        temperature_c=10.0,
        humidity_fraction=0.2,
        pressure_pa=None,
        wind_speed_mps=None,
        wind_dir_deg=None,
        solar_lux=None,
        location_m=(0.0, 0.0, 0.0),
        metadata={"source": "a"},
    )
    provider.ingest_sample(
        node_id="node-b",
        timestamp_ns=now_ns,
        temperature_c=30.0,
        humidity_fraction=0.7,
        pressure_pa=None,
        wind_speed_mps=None,
        wind_dir_deg=None,
        solar_lux=None,
        location_m=(100.0, 0.0, 0.0),
        metadata={"source": "b"},
    )
    near_a = provider.get_conditions(location_m=(1.0, 0.0, 0.0))
    near_b = provider.get_conditions(location_m=(99.0, 0.0, 0.0))
    assert near_a.temperature_c == pytest.approx(10.0, abs=1e-6)
    assert near_b.temperature_c == pytest.approx(30.0, abs=1e-6)


def test_live_environment_provider_ignores_stale_samples() -> None:
    provider = LiveEnvironmentProvider(
        fallback_temperature_c=13.0,
        fallback_humidity_fraction=0.4,
        max_reading_age_seconds=0.01,
    )
    provider.ingest_sample(
        node_id="node-a",
        timestamp_ns=time.time_ns() - 2_000_000_000,
        temperature_c=29.0,
        humidity_fraction=0.5,
        pressure_pa=101000.0,
        wind_speed_mps=None,
        wind_dir_deg=None,
        solar_lux=None,
        location_m=(0.0, 0.0, 0.0),
        metadata={},
    )
    conditions = provider.get_conditions()
    assert conditions.temperature_c == pytest.approx(13.0, abs=1e-6)
    assert conditions.humidity_fraction == pytest.approx(0.4, abs=1e-6)
    assert conditions.metadata["source"] == "static_fallback"
