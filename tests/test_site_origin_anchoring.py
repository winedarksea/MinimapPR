"""Coverage for GPS-driven site-origin anchoring on the ingest path.

Regression context: the api and ingest processes each resolved the site origin
independently. Ingest kept the hardcoded fallback (Minneapolis) while api
reconciled to the real site, leaving node positions 113 km from the origin. Every
localization solve then failed the 5 km sanity backstop, no localization branch
was built, and — because beamformed classification only runs on a localized
branch — zero detections were emitted despite healthy audio.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from minimappr.core.site_origin import SOURCE_GPS_ANCHOR
from minimappr.models import GeoPoint
from minimappr.storage.db import Storage


@pytest.fixture
async def storage():
    with tempfile.TemporaryDirectory() as tmp:
        store = Storage(Path(tmp) / "test.db")
        await store.initialize()
        try:
            yield store
        finally:
            await store.close()


@pytest.mark.asyncio
async def test_site_origin_round_trips(storage) -> None:
    assert await storage.get_site_origin() is None

    await storage.upsert_site_origin(
        lat=44.39053726,
        lon=-92.08418655,
        alt_m=237.85,
        source=SOURCE_GPS_ANCHOR,
        contributing_node_ids=["sirith-tetra-6b9b"],
    )

    persisted = await storage.get_site_origin()
    assert persisted["lat"] == pytest.approx(44.39053726)
    assert persisted["lon"] == pytest.approx(-92.08418655)
    assert persisted["alt_m"] == pytest.approx(237.85)
    assert persisted["source"] == SOURCE_GPS_ANCHOR
    assert persisted["contributing_node_ids"] == ["sirith-tetra-6b9b"]
    assert persisted["resolved_ns"] > 0


@pytest.mark.asyncio
async def test_site_origin_upsert_replaces_single_row(storage) -> None:
    await storage.upsert_site_origin(lat=1.0, lon=2.0, alt_m=3.0, source="a")
    await storage.upsert_site_origin(lat=4.0, lon=5.0, alt_m=6.0, source="b")

    persisted = await storage.get_site_origin()
    assert persisted["lat"] == pytest.approx(4.0)
    assert persisted["source"] == "b"


@pytest.mark.asyncio
async def test_clear_site_origin_unanchors(storage) -> None:
    await storage.upsert_site_origin(lat=1.0, lon=2.0, alt_m=3.0, source=SOURCE_GPS_ANCHOR)
    await storage.clear_site_origin()

    assert await storage.get_site_origin() is None


# --- the ingest-side anchor hook -------------------------------------------


class _StubIngestProcessor:
    """Exercises IngestProcessor's anchor gate without a full ingest stack."""

    def __init__(self):
        from minimappr.core.ingest import IngestProcessor

        self._maybe_anchor = IngestProcessor._maybe_anchor_site_origin.__get__(self)
        self._site_origin_anchor = None
        self.calls: list[tuple[str, GeoPoint]] = []

    async def anchor(self, node_id: str, geo: GeoPoint) -> None:
        self.calls.append((node_id, geo))


def _spec(node_id: str, *, geo: GeoPoint | None, metadata: dict):
    from minimappr.models import NodeSpec, NodeType

    return NodeSpec(
        id=node_id,
        node_type=NodeType.SIRITH_TETRA,
        position_geo=geo,
        position_m=(0.0, 0.0, 0.0) if geo is None else None,
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_anchor_fires_for_trusted_gps_fix() -> None:
    stub = _StubIngestProcessor()
    stub._site_origin_anchor = stub.anchor
    geo = GeoPoint(lat=44.39, lon=-92.08, alt_m=237.0)

    await stub._maybe_anchor(
        _spec("node-a", geo=geo, metadata={"gps": {"position_source": "gps_nmea_uart", "signal": "fix_2d"}})
    )

    assert stub.calls == [("node-a", geo)]


@pytest.mark.asyncio
async def test_anchor_skipped_without_trusted_gps() -> None:
    stub = _StubIngestProcessor()
    stub._site_origin_anchor = stub.anchor

    await stub._maybe_anchor(
        _spec(
            "node-a",
            geo=GeoPoint(lat=44.39, lon=-92.08, alt_m=237.0),
            metadata={"gps": {"position_source": "gps_fallback"}},
        )
    )

    assert stub.calls == []


@pytest.mark.asyncio
async def test_anchor_skipped_once_disarmed() -> None:
    """The runtime clears the callback after anchoring; steady state must be inert."""
    stub = _StubIngestProcessor()
    stub._site_origin_anchor = None

    await stub._maybe_anchor(
        _spec(
            "node-a",
            geo=GeoPoint(lat=44.39, lon=-92.08, alt_m=237.0),
            metadata={"gps": {"position_source": "gps_nmea_uart"}},
        )
    )

    assert stub.calls == []


@pytest.mark.asyncio
async def test_reset_position_estimators_clears_enu_state() -> None:
    """Estimator state is ENU metres and must not survive an origin change."""
    from minimappr.core.ingest import IngestProcessor
    from minimappr.core.node_position_estimator import StationaryKdeState

    processor = object.__new__(IngestProcessor)
    processor._position_kalman = {"n": object()}
    processor._position_kde = {"n": StationaryKdeState()}
    processor._last_trusted_altitude_m = {"n": 237.0}
    processor._untrusted_altitude_warned_node_ids = {"n"}

    processor.reset_position_estimators()

    assert processor._position_kalman == {}
    assert processor._position_kde == {}
    assert processor._last_trusted_altitude_m == {}
    assert processor._untrusted_altitude_warned_node_ids == set()


@pytest.mark.asyncio
async def test_clear_node_position_estimator_states(storage) -> None:
    from minimappr.models import NodeSpec, NodeType

    await storage.upsert_node(
        NodeSpec(
            id="node-a",
            node_type=NodeType.SIRITH_TETRA,
            position_m=(0.0, 0.0, 0.0),
        ),
        time.time_ns(),
    )
    await storage.upsert_node_position_estimator_state("node-a", {"x": 1.0})
    assert await storage.list_node_position_estimator_states() != {}

    await storage.clear_node_position_estimator_states()

    assert await storage.list_node_position_estimator_states() == {}
