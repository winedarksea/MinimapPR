from __future__ import annotations

import time

import pytest

from minimappr.config import Settings
from minimappr.core.site_origin import (
    SOURCE_ACTIVE_NODE_MIDPOINT,
    SOURCE_CONFIGURED_FALLBACK,
    SOURCE_GPS_ANCHOR,
    SOURCE_MANUAL,
    SOURCE_PERSISTED,
    load_or_resolve_site_origin,
    node_has_trusted_gps_position,
    origins_differ,
    resolve_site_origin_from_nodes,
)
from minimappr.models import GeoPoint

GPS_METADATA = {"gps": {"position_source": "gps_nmea_uart", "signal": "fix_2d"}}


class _FakeStorage:
    """Minimal storage double covering the site-origin accessors."""

    def __init__(self, *, nodes: list[dict] | None = None, origin: dict | None = None) -> None:
        self._nodes = nodes or []
        self.origin = origin
        self.cleared_estimator_states = False

    async def get_site_origin(self) -> dict | None:
        return self.origin

    async def upsert_site_origin(self, *, lat, lon, alt_m, source, contributing_node_ids=()) -> None:
        self.origin = {
            "lat": lat,
            "lon": lon,
            "alt_m": alt_m,
            "source": source,
            "contributing_node_ids": list(contributing_node_ids),
            "resolved_ns": time.time_ns(),
        }

    async def clear_site_origin(self) -> None:
        self.origin = None

    async def list_nodes(self, limit: int | None = None) -> list[dict]:
        return self._nodes

    async def clear_node_position_estimator_states(self) -> None:
        self.cleared_estimator_states = True


def _node(node_id: str, lat: float, lon: float, alt_m: float, *, gps: bool, age_ns: int = 0) -> dict:
    return {
        "id": node_id,
        "last_seen_ns": time.time_ns() - age_ns,
        "position_geo": {"lat": lat, "lon": lon, "alt_m": alt_m},
        "metadata": dict(GPS_METADATA) if gps else {},
    }


# --- trusted-GPS gate ------------------------------------------------------


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"gps": {"position_source": "gps_nmea_uart"}}, True),
        ({"gps": {"position_source": "gps_fallback"}}, False),
        ({"position_source": "gps_nmea_uart"}, True),
        ({"gps": {"position_source": "configured"}}, False),
        ({}, False),
        (None, False),
    ],
)
def test_node_has_trusted_gps_position(metadata, expected) -> None:
    assert node_has_trusted_gps_position(metadata) is expected


def test_two_d_fix_still_anchors() -> None:
    """A 2D fix solves lat/lon, which is all an origin needs."""
    assert node_has_trusted_gps_position({"gps": {"position_source": "gps_nmea_uart", "signal": "fix_2d"}})


# --- resolution from nodes -------------------------------------------------


def test_resolve_prefers_gps_nodes_and_marks_result_anchored() -> None:
    now_ns = time.time_ns()
    settings = Settings(site_origin_source="auto", node_offline_after_seconds=45.0)

    resolution = resolve_site_origin_from_nodes(
        settings,
        now_ns=now_ns,
        nodes=[
            _node("node-gps-a", 44.0, -93.0, 100.0, gps=True),
            _node("node-gps-b", 46.0, -95.0, 120.0, gps=True),
            # No GPS: must not drag the origin even though it has a position.
            _node("node-configured", 10.0, 10.0, 0.0, gps=False),
        ],
    )

    assert resolution.source == SOURCE_GPS_ANCHOR
    assert resolution.is_anchored
    assert resolution.origin.lat == pytest.approx(45.0)
    assert resolution.origin.lon == pytest.approx(-94.0)
    assert resolution.contributing_node_ids == ("node-gps-a", "node-gps-b")


def test_resolve_uses_midpoint_of_active_node_geo_positions_without_gps() -> None:
    now_ns = time.time_ns()
    settings = Settings(site_origin_source="auto", node_offline_after_seconds=45.0)

    resolution = resolve_site_origin_from_nodes(
        settings,
        now_ns=now_ns,
        nodes=[
            _node("node-a", 44.0, -93.0, 100.0, gps=False),
            _node("node-b", 46.0, -95.0, 120.0, gps=False, age_ns=5_000_000_000),
            _node("node-stale", 0.0, 0.0, 0.0, gps=False, age_ns=100_000_000_000),
        ],
    )

    assert resolution.source == SOURCE_ACTIVE_NODE_MIDPOINT
    # Not anchored: a configured position is not a survey, so it is never persisted
    # and a later real GPS fix can still take over.
    assert not resolution.is_anchored
    assert resolution.origin.lat == pytest.approx(45.0)
    assert resolution.contributing_node_ids == ("node-a", "node-b")


def test_resolve_falls_back_to_configured_origin_when_auto_has_no_active_geo() -> None:
    settings = Settings(
        site_origin_source="auto", site_origin_lat=11.0, site_origin_lon=22.0, site_origin_alt_m=33.0
    )

    resolution = resolve_site_origin_from_nodes(settings, now_ns=time.time_ns(), nodes=[])

    assert resolution.source == SOURCE_CONFIGURED_FALLBACK
    assert not resolution.is_anchored
    assert resolution.origin.lat == pytest.approx(11.0)


def test_resolve_respects_manual_mode() -> None:
    settings = Settings(
        site_origin_source="manual", site_origin_lat=1.25, site_origin_lon=2.5, site_origin_alt_m=3.75
    )

    resolution = resolve_site_origin_from_nodes(
        settings, now_ns=time.time_ns(), nodes=[_node("node-a", 44.0, -93.0, 100.0, gps=True)]
    )

    assert resolution.source == SOURCE_MANUAL
    assert resolution.is_anchored
    assert resolution.origin.lat == pytest.approx(1.25)


# --- persistence / cross-process agreement ---------------------------------


@pytest.mark.asyncio
async def test_persisted_origin_wins_over_node_derived_resolution() -> None:
    """The regression that mattered: two processes must not resolve independently.

    Ingest anchored at 44.39/-92.08; a later api-process boot that sees only stale
    nodes must adopt that, not fall back to the configured default 113 km away.
    """
    settings = Settings(
        site_origin_source="auto",
        site_origin_lat=44.98698840878797,
        site_origin_lon=-93.2579197515542,
    )
    storage = _FakeStorage(
        nodes=[],
        origin={
            "lat": 44.39053726,
            "lon": -92.08418655,
            "alt_m": 237.85,
            "source": SOURCE_GPS_ANCHOR,
            "contributing_node_ids": ["sirith-tetra-6b9b"],
            "resolved_ns": time.time_ns(),
        },
    )

    resolution = await load_or_resolve_site_origin(settings, storage=storage, now_ns=time.time_ns())

    assert resolution.source == SOURCE_PERSISTED
    assert resolution.is_anchored
    assert resolution.origin.lat == pytest.approx(44.39053726)
    assert resolution.origin.lon == pytest.approx(-92.08418655)


@pytest.mark.asyncio
async def test_gps_resolution_is_persisted_on_first_boot() -> None:
    settings = Settings(site_origin_source="auto", node_offline_after_seconds=45.0)
    storage = _FakeStorage(nodes=[_node("node-gps", 44.39, -92.08, 237.0, gps=True)])

    resolution = await load_or_resolve_site_origin(settings, storage=storage, now_ns=time.time_ns())

    assert resolution.source == SOURCE_GPS_ANCHOR
    assert storage.origin is not None
    assert storage.origin["lat"] == pytest.approx(44.39)


@pytest.mark.asyncio
async def test_unanchored_fallback_is_not_persisted() -> None:
    """A fallback origin must stay un-persisted so real GPS can still claim the site."""
    settings = Settings(site_origin_source="auto", site_origin_lat=11.0, site_origin_lon=22.0)
    storage = _FakeStorage(nodes=[])

    resolution = await load_or_resolve_site_origin(settings, storage=storage, now_ns=time.time_ns())

    assert resolution.source == SOURCE_CONFIGURED_FALLBACK
    assert storage.origin is None


@pytest.mark.asyncio
async def test_manual_config_overrides_and_repersists_a_bad_anchor() -> None:
    settings = Settings(
        site_origin_source="manual", site_origin_lat=1.5, site_origin_lon=2.5, site_origin_alt_m=3.5
    )
    storage = _FakeStorage(
        origin={
            "lat": 44.0,
            "lon": -93.0,
            "alt_m": 0.0,
            "source": SOURCE_GPS_ANCHOR,
            "contributing_node_ids": [],
            "resolved_ns": time.time_ns(),
        }
    )

    resolution = await load_or_resolve_site_origin(settings, storage=storage, now_ns=time.time_ns())

    assert resolution.source == SOURCE_MANUAL
    assert resolution.origin.lat == pytest.approx(1.5)
    assert storage.origin["lat"] == pytest.approx(1.5)


def test_origins_differ_tolerates_float_noise() -> None:
    a = GeoPoint(lat=44.39053726, lon=-92.08418655, alt_m=237.85)
    assert not origins_differ(a, GeoPoint(lat=44.39053726, lon=-92.08418655, alt_m=237.85))
    assert origins_differ(a, GeoPoint(lat=44.39053726, lon=-92.08418000, alt_m=237.85))
