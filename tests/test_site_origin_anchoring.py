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
    processor._kde_rejection_streak = {"n": 3}
    processor._last_trusted_altitude_m = {"n": 237.0}
    processor._untrusted_altitude_warned_node_ids = {"n"}

    processor.reset_position_estimators()

    assert processor._position_kalman == {}
    assert processor._position_kde == {}
    assert processor._kde_rejection_streak == {}
    assert processor._last_trusted_altitude_m == {}
    assert processor._untrusted_altitude_warned_node_ids == set()


# --- estimator checkpoints are bound to the origin they were built under ---


def _kde_processor(origin: GeoPoint, storage=None):
    """An IngestProcessor with only the position-estimator collaborators wired."""
    from minimappr.core.geo import LocalCoordinateFrame
    from minimappr.core.ingest import IngestProcessor

    p = object.__new__(IngestProcessor)
    p._coordinate_frame = LocalCoordinateFrame(origin=origin, mode="flat")
    p._storage = storage
    p._position_kde = {}
    p._position_kalman = {}
    p._kde_rejection_streak = {}
    p._position_filter_by_node = {}
    p._kde_evaluation_tasks = {}
    p._kde_reservoir_capacity = 2048
    p._kde_warmup_fixes = 30
    p._kde_recompute_seconds = 30.0
    p._kde_checkpoint_seconds = 60.0
    p._kde_acceptance_radius_m = 100.0
    p._kde_bandwidth_m = 2.5
    return p


def test_snapshot_is_stamped_with_origin() -> None:
    from minimappr.core.node_position_estimator import StationaryKdeState

    origin = GeoPoint(lat=44.39, lon=-92.08, alt_m=237.0)
    state = StationaryKdeState()
    state.add((1.0, 2.0, 3.0), 128)

    snapshot = state.snapshot(origin)

    assert snapshot["origin"] == {"lat": 44.39, "lon": -92.08, "alt_m": 237.0}
    assert StationaryKdeState.snapshot_matches_origin(snapshot, origin)
    assert not StationaryKdeState.snapshot_matches_origin(
        snapshot, GeoPoint(lat=44.98, lon=-93.25, alt_m=0.0)
    )


def test_unstamped_legacy_snapshot_is_treated_as_mismatched() -> None:
    """Pre-stamp checkpoints cannot be proven safe, so they must be rebuilt."""
    from minimappr.core.node_position_estimator import StationaryKdeState

    legacy = StationaryKdeState().snapshot()

    assert "origin" not in legacy
    assert not StationaryKdeState.snapshot_matches_origin(
        legacy, GeoPoint(lat=44.39, lon=-92.08, alt_m=237.0)
    )


@pytest.mark.asyncio
async def test_hydrate_discards_checkpoint_from_a_different_origin(storage) -> None:
    """The regression: a checkpoint in the old frame pins the node 113 km out.

    Ingest anchored the origin correctly at startup, but then restored KDE state
    holding old-frame ENU. Every correct fix then sat 113 km from the restored
    estimate, tripping the 100 m acceptance radius, so the node's position never
    moved and its reported geo drifted to a point 113 km from the real site.
    """
    from minimappr.core.node_position_estimator import StationaryKdeState
    from minimappr.models import NodeSpec, NodeType

    await storage.upsert_node(
        NodeSpec(id="node-a", node_type=NodeType.SIRITH_TETRA, position_m=(0.0, 0.0, 0.0)),
        time.time_ns(),
    )
    stale = StationaryKdeState()
    stale.estimate = (92563.54, -66285.84, 236.8)
    await storage.upsert_node_position_estimator_state(
        "node-a", stale.snapshot(GeoPoint(lat=44.98698840878797, lon=-93.2579197515542, alt_m=0.0))
    )

    processor = _kde_processor(GeoPoint(lat=44.39053726, lon=-92.08418655, alt_m=237.85), storage)
    await processor.hydrate_position_estimator_states()

    assert processor._position_kde == {}
    assert await storage.list_node_position_estimator_states() == {}


@pytest.mark.asyncio
async def test_hydrate_keeps_checkpoint_from_the_same_origin(storage) -> None:
    from minimappr.core.node_position_estimator import StationaryKdeState
    from minimappr.models import NodeSpec, NodeType

    origin = GeoPoint(lat=44.39053726, lon=-92.08418655, alt_m=237.85)
    await storage.upsert_node(
        NodeSpec(id="node-a", node_type=NodeType.SIRITH_TETRA, position_m=(0.0, 0.0, 0.0)),
        time.time_ns(),
    )
    good = StationaryKdeState()
    good.add((1.0, 2.0, 3.0), 128)
    good.estimate = (1.0, 2.0, 3.0)
    await storage.upsert_node_position_estimator_state("node-a", good.snapshot(origin))

    processor = _kde_processor(origin, storage)
    await processor.hydrate_position_estimator_states()

    assert processor._position_kde["node-a"].estimate == (1.0, 2.0, 3.0)
    assert processor._position_filter_by_node["node-a"] == "kde"


def test_sustained_rejections_reset_a_wrong_estimate() -> None:
    """A node whose estimate is wrong must re-converge, not reject fixes forever."""
    from minimappr.core.ingest import _KDE_REJECTION_STREAK_RESET
    from minimappr.core.node_position_estimator import StationaryKdeState

    processor = _kde_processor(GeoPoint(lat=44.39, lon=-92.08, alt_m=237.0))
    state = StationaryKdeState()
    state.estimate = (92563.54, -66285.84, 236.8)
    processor._position_kde["node-a"] = state
    true_fix = (-2.6, -1.3, -1.0)

    for _ in range(_KDE_REJECTION_STREAK_RESET - 1):
        assert processor._update_stationary_kde("node-a", true_fix) == state.estimate

    # The streak crosses the threshold: the estimator resets and adopts the fix.
    assert processor._update_stationary_kde("node-a", true_fix) == true_fix
    assert processor._position_kde["node-a"].estimate != (92563.54, -66285.84, 236.8)
    assert processor._kde_rejection_streak == {}


def test_accepted_fix_clears_the_rejection_streak() -> None:
    """Isolated multipath outliers must not accumulate toward a reset."""
    from minimappr.core.node_position_estimator import StationaryKdeState

    processor = _kde_processor(GeoPoint(lat=44.39, lon=-92.08, alt_m=237.0))
    state = StationaryKdeState()
    state.estimate = (0.0, 0.0, 0.0)
    processor._position_kde["node-a"] = state

    processor._update_stationary_kde("node-a", (5000.0, 0.0, 0.0))  # outlier
    assert processor._kde_rejection_streak["node-a"] == 1

    processor._update_stationary_kde("node-a", (1.0, 1.0, 0.0))  # in-radius
    assert processor._kde_rejection_streak == {}


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
