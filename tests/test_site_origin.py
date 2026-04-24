from __future__ import annotations

import time

import pytest

from minimappr.config import Settings
from minimappr.core.site_origin import (
    resolve_site_origin_from_nodes,
    should_schedule_deferred_site_origin_reconciliation,
)


def test_resolve_site_origin_uses_midpoint_of_active_node_geo_positions() -> None:
    now_ns = time.time_ns()
    settings = Settings(
        site_origin_source="auto",
        site_origin_lat=10.0,
        site_origin_lon=20.0,
        site_origin_alt_m=5.0,
        node_offline_after_seconds=45.0,
    )

    resolution = resolve_site_origin_from_nodes(
        settings,
        now_ns=now_ns,
        nodes=[
            {
                "id": "node-a",
                "last_seen_ns": now_ns,
                "position_geo": {"lat": 44.0, "lon": -93.0, "alt_m": 100.0},
            },
            {
                "id": "node-b",
                "last_seen_ns": now_ns - 5_000_000_000,
                "position_geo": {"lat": 46.0, "lon": -95.0, "alt_m": 120.0},
            },
            {
                "id": "node-stale",
                "last_seen_ns": now_ns - 100_000_000_000,
                "position_geo": {"lat": 0.0, "lon": 0.0, "alt_m": 0.0},
            },
        ],
    )

    assert resolution.source == "active_node_midpoint"
    assert resolution.origin.lat == pytest.approx(45.0)
    assert resolution.origin.lon == pytest.approx(-94.0)
    assert resolution.origin.alt_m == pytest.approx(110.0)
    assert resolution.contributing_node_ids == ("node-a", "node-b")


def test_resolve_site_origin_falls_back_to_configured_origin_when_auto_has_no_active_geo() -> None:
    settings = Settings(
        site_origin_source="auto",
        site_origin_lat=11.0,
        site_origin_lon=22.0,
        site_origin_alt_m=33.0,
    )

    resolution = resolve_site_origin_from_nodes(settings, now_ns=time.time_ns(), nodes=[])

    assert resolution.source == "configured_fallback"
    assert resolution.origin.lat == pytest.approx(11.0)
    assert resolution.origin.lon == pytest.approx(22.0)
    assert resolution.origin.alt_m == pytest.approx(33.0)


def test_resolve_site_origin_respects_manual_mode() -> None:
    settings = Settings(
        site_origin_source="manual",
        site_origin_lat=1.25,
        site_origin_lon=2.5,
        site_origin_alt_m=3.75,
    )

    resolution = resolve_site_origin_from_nodes(
        settings,
        now_ns=time.time_ns(),
        nodes=[
            {
                "id": "node-a",
                "last_seen_ns": time.time_ns(),
                "position_geo": {"lat": 44.0, "lon": -93.0, "alt_m": 100.0},
            }
        ],
    )

    assert resolution.source == "manual_config"
    assert resolution.origin.lat == pytest.approx(1.25)
    assert resolution.origin.lon == pytest.approx(2.5)
    assert resolution.origin.alt_m == pytest.approx(3.75)


def test_deferred_site_origin_reconciliation_only_runs_for_auto_fallback_mode() -> None:
    auto_settings = Settings(site_origin_source="auto", site_origin_reconcile_delay_seconds=30.0)

    assert should_schedule_deferred_site_origin_reconciliation(
        auto_settings,
        initial_resolution_source="configured_fallback",
    )
    assert not should_schedule_deferred_site_origin_reconciliation(
        auto_settings,
        initial_resolution_source="active_node_midpoint",
    )

    manual_settings = Settings(site_origin_source="manual", site_origin_reconcile_delay_seconds=30.0)
    assert not should_schedule_deferred_site_origin_reconciliation(
        manual_settings,
        initial_resolution_source="configured_fallback",
    )


def test_deferred_site_origin_reconciliation_can_be_disabled() -> None:
    settings = Settings(site_origin_source="auto", site_origin_reconcile_delay_seconds=0.0)

    assert not should_schedule_deferred_site_origin_reconciliation(
        settings,
        initial_resolution_source="configured_fallback",
    )