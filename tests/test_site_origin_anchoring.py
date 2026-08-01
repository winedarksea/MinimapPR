"""Coverage for GPS-driven site-origin anchoring and frame-safe estimator state.

Regression context: the api and ingest processes each resolved the site origin
independently. Ingest kept the hardcoded fallback (Minneapolis) while api
reconciled to the real site, leaving node positions 113 km from the origin. Every
localization solve then failed the 5 km sanity backstop, no localization branch
was built, and — because beamformed classification only runs on a localized
branch — zero detections were emitted despite healthy audio.

Fixing the origin alone was not enough: persisted position-estimator checkpoints
held ENU metres in the *old* frame, so restoring them put the node's estimate
113 km from every incoming fix, which the acceptance radius then rejected as an
outlier forever. Checkpoints are now stored geodetically, which is what the fixes
actually were, so they survive any origin change.
"""

from __future__ import annotations

import time

import pytest

from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.node_position_estimator import StationaryKdeState
from minimappr.core.site_origin import SOURCE_GPS_ANCHOR
from minimappr.models import GeoPoint, NodeSpec, NodeType

# The real deployment values this was debugged against.
DEFAULT_ORIGIN = GeoPoint(lat=44.98698840878797, lon=-93.2579197515542, alt_m=0.0)
SITE_ORIGIN = GeoPoint(lat=44.39053726, lon=-92.08418655, alt_m=237.85)
NODE_GEO = GeoPoint(lat=44.390525226692056, lon=-92.08421968801866, alt_m=236.8)


def _frame(origin: GeoPoint) -> LocalCoordinateFrame:
    return LocalCoordinateFrame(origin=origin, mode="flat")


def _kde_processor(origin: GeoPoint, storage=None):
    """An IngestProcessor with only the position-estimator collaborators wired."""
    from minimappr.core.ingest import IngestProcessor

    processor = object.__new__(IngestProcessor)
    processor._coordinate_frame = _frame(origin)
    processor._storage = storage
    processor._position_kde = {}
    processor._position_kalman = {}
    processor._kde_rejection_streak = {}
    processor._persisted_node_ids = set()
    processor._position_filter_by_node = {}
    processor._kde_evaluation_tasks = {}
    processor._last_trusted_altitude_m = {}
    processor._kde_reservoir_capacity = 2048
    processor._kde_warmup_fixes = 30
    processor._kde_recompute_seconds = 30.0
    processor._kde_checkpoint_seconds = 60.0
    processor._kde_acceptance_radius_m = 100.0
    processor._kde_rejection_streak_reset = 60
    processor._kde_bandwidth_m = 2.5
    return processor


async def _register_node(storage, node_id: str = "node-a") -> None:
    await storage.upsert_node(
        NodeSpec(id=node_id, node_type=NodeType.SIRITH_TETRA, position_m=(0.0, 0.0, 0.0)),
        time.time_ns(),
    )


# --- site origin persistence ------------------------------------------------


@pytest.mark.asyncio
async def test_site_origin_round_trips(temp_storage) -> None:
    assert await temp_storage.get_site_origin() is None

    await temp_storage.upsert_site_origin(
        lat=SITE_ORIGIN.lat,
        lon=SITE_ORIGIN.lon,
        alt_m=SITE_ORIGIN.alt_m,
        source=SOURCE_GPS_ANCHOR,
        contributing_node_ids=["sirith-tetra-6b9b"],
    )

    persisted = await temp_storage.get_site_origin()
    assert persisted["lat"] == pytest.approx(SITE_ORIGIN.lat)
    assert persisted["lon"] == pytest.approx(SITE_ORIGIN.lon)
    assert persisted["source"] == SOURCE_GPS_ANCHOR
    assert persisted["contributing_node_ids"] == ["sirith-tetra-6b9b"]
    assert persisted["resolved_ns"] > 0


@pytest.mark.asyncio
async def test_site_origin_upsert_replaces_single_row(temp_storage) -> None:
    await temp_storage.upsert_site_origin(lat=1.0, lon=2.0, alt_m=3.0, source="a")
    await temp_storage.upsert_site_origin(lat=4.0, lon=5.0, alt_m=6.0, source="b")

    persisted = await temp_storage.get_site_origin()
    assert persisted["lat"] == pytest.approx(4.0)
    assert persisted["source"] == "b"


@pytest.mark.asyncio
async def test_clear_site_origin_unanchors(temp_storage) -> None:
    await temp_storage.upsert_site_origin(lat=1.0, lon=2.0, alt_m=3.0, source=SOURCE_GPS_ANCHOR)
    await temp_storage.clear_site_origin()

    assert await temp_storage.get_site_origin() is None


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


def _spec(node_id: str, *, geo: GeoPoint | None, metadata: dict) -> NodeSpec:
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

    await stub._maybe_anchor(
        _spec(
            "node-a",
            geo=NODE_GEO,
            metadata={"gps": {"position_source": "gps_nmea_uart", "signal": "fix_2d"}},
        )
    )

    assert stub.calls == [("node-a", NODE_GEO)]


@pytest.mark.asyncio
async def test_anchor_skipped_without_trusted_gps() -> None:
    stub = _StubIngestProcessor()
    stub._site_origin_anchor = stub.anchor

    await stub._maybe_anchor(
        _spec("node-a", geo=NODE_GEO, metadata={"gps": {"position_source": "gps_fallback"}})
    )

    assert stub.calls == []


@pytest.mark.asyncio
async def test_anchor_skipped_once_disarmed() -> None:
    """The runtime clears the callback after anchoring; steady state must be inert."""
    stub = _StubIngestProcessor()
    stub._site_origin_anchor = None

    await stub._maybe_anchor(
        _spec("node-a", geo=NODE_GEO, metadata={"gps": {"position_source": "gps_nmea_uart"}})
    )

    assert stub.calls == []


# --- checkpoints are geodetic, so they outlive any origin change -----------


def test_snapshot_is_geodetic_and_round_trips_through_its_own_frame() -> None:
    frame = _frame(SITE_ORIGIN)
    state = StationaryKdeState()
    state.add(frame.geo_to_local(NODE_GEO), 128)
    state.estimate = frame.geo_to_local(NODE_GEO)

    snapshot = state.snapshot(frame)

    # Persisted as latitude/longitude, not site-relative metres.
    assert snapshot["samples"][0][0] == pytest.approx(NODE_GEO.lat, abs=1e-8)
    assert snapshot["samples"][0][1] == pytest.approx(NODE_GEO.lon, abs=1e-8)

    restored = StationaryKdeState.from_snapshot(snapshot, frame)
    assert restored.estimate == pytest.approx(state.estimate, abs=1e-3)


def test_snapshot_reprojects_correctly_into_a_different_frame() -> None:
    """The property that makes the whole class of bug impossible."""
    old_frame, new_frame = _frame(DEFAULT_ORIGIN), _frame(SITE_ORIGIN)
    state = StationaryKdeState()
    state.estimate = old_frame.geo_to_local(NODE_GEO)
    assert state.estimate[0] == pytest.approx(92563.5, abs=1.0)  # 113 km out

    restored = StationaryKdeState.from_snapshot(state.snapshot(old_frame), new_frame)

    # Same physical place, now expressed in the new frame — metres, not kilometres.
    assert restored.estimate == pytest.approx(new_frame.geo_to_local(NODE_GEO), abs=1e-3)
    assert abs(restored.estimate[0]) < 10.0


def test_pre_geodetic_snapshot_is_rejected_rather_than_misread() -> None:
    """Legacy ENU rows have no recoverable origin, so they must not be restored."""
    legacy = {"samples": [[92563.54, -66285.84, 236.8]], "seen_count": 1, "estimate": None}

    assert StationaryKdeState.from_snapshot(legacy, _frame(SITE_ORIGIN)) is None


@pytest.mark.asyncio
async def test_hydrate_discards_pre_geodetic_checkpoint(temp_storage) -> None:
    """The exact production state: an ENU-era row that pinned the node 113 km out."""
    await _register_node(temp_storage)
    await temp_storage.upsert_node_position_estimator_state(
        "node-a", {"samples": [[92563.54, -66285.84, 236.8]], "seen_count": 1, "estimate": None}
    )

    processor = _kde_processor(SITE_ORIGIN, temp_storage)
    await processor.hydrate_position_estimator_states()

    assert processor._position_kde == {}
    assert await temp_storage.list_node_position_estimator_states() == {}


@pytest.mark.asyncio
async def test_hydrate_restores_geodetic_checkpoint_under_a_changed_origin(temp_storage) -> None:
    """Averaging accumulated under one origin is kept, not thrown away, after a re-anchor."""
    await _register_node(temp_storage)
    old_frame = _frame(DEFAULT_ORIGIN)
    state = StationaryKdeState()
    state.add(old_frame.geo_to_local(NODE_GEO), 128)
    state.estimate = old_frame.geo_to_local(NODE_GEO)
    await temp_storage.upsert_node_position_estimator_state("node-a", state.snapshot(old_frame))

    processor = _kde_processor(SITE_ORIGIN, temp_storage)
    await processor.hydrate_position_estimator_states()

    restored = processor._position_kde["node-a"]
    assert restored.estimate == pytest.approx(_frame(SITE_ORIGIN).geo_to_local(NODE_GEO), abs=1e-3)
    assert processor._position_filter_by_node["node-a"] == "kde"
    # Hydrated nodes already have a row, so checkpointing is unblocked immediately.
    assert "node-a" in processor._persisted_node_ids


def test_reproject_carries_estimator_state_across_an_origin_change() -> None:
    from minimappr.core.ingest import _NodePositionKalman

    old_frame = _frame(DEFAULT_ORIGIN)
    processor = _kde_processor(SITE_ORIGIN)
    kde = StationaryKdeState()
    kde.add(old_frame.geo_to_local(NODE_GEO), 128)
    kde.estimate = old_frame.geo_to_local(NODE_GEO)
    processor._position_kde["node-a"] = kde
    processor._position_kalman["node-a"] = _NodePositionKalman(
        *old_frame.geo_to_local(NODE_GEO), px=1.0, py=2.0, pz=3.0, initialized=True
    )
    processor._kde_rejection_streak["node-a"] = 4

    processor.reproject_position_estimators(old_frame)

    expected = _frame(SITE_ORIGIN).geo_to_local(NODE_GEO)
    assert processor._position_kde["node-a"].estimate == pytest.approx(expected, abs=1e-3)
    assert len(processor._position_kde["node-a"].samples) == 1
    kalman = processor._position_kalman["node-a"]
    assert (kalman.x, kalman.y, kalman.z) == pytest.approx(expected, abs=1e-3)
    assert (kalman.px, kalman.py, kalman.pz) == (1.0, 2.0, 3.0)  # variances are frame-invariant
    assert processor._kde_rejection_streak == {}


# --- a wrong estimate must never be permanent ------------------------------


def test_sustained_rejections_reset_a_wrong_estimate() -> None:
    """A node whose estimate is wrong must re-converge, not reject fixes forever."""
    processor = _kde_processor(SITE_ORIGIN)
    state = StationaryKdeState()
    state.estimate = (92563.54, -66285.84, 236.8)
    processor._position_kde["node-a"] = state
    true_fix = (-2.6, -1.3, -1.0)

    for _ in range(processor._kde_rejection_streak_reset - 1):
        assert processor._update_stationary_kde("node-a", true_fix) == state.estimate

    # The streak crosses the threshold: the estimator resets and adopts the fix.
    assert processor._update_stationary_kde("node-a", true_fix) == true_fix
    assert processor._position_kde["node-a"].estimate != (92563.54, -66285.84, 236.8)
    assert processor._kde_rejection_streak == {}


def test_accepted_fix_clears_the_rejection_streak() -> None:
    """Isolated multipath outliers must not accumulate toward a reset."""
    processor = _kde_processor(SITE_ORIGIN)
    state = StationaryKdeState()
    state.estimate = (0.0, 0.0, 0.0)
    processor._position_kde["node-a"] = state

    processor._update_stationary_kde("node-a", (5000.0, 0.0, 0.0))  # outlier
    assert processor._kde_rejection_streak["node-a"] == 1

    processor._update_stationary_kde("node-a", (1.0, 1.0, 0.0))  # in-radius
    assert processor._kde_rejection_streak == {}


# --- checkpoint writes must not violate the estimator-state foreign key ----


@pytest.mark.asyncio
async def test_checkpoint_skipped_until_the_node_row_exists() -> None:
    """node_position_estimator_states has a FK onto nodes(id).

    The estimator runs during spec normalization, before the frame's node row is
    written, so an unguarded checkpoint raised IntegrityError inside a storage
    batch — and a failed commit there leaves the transaction open, stalling every
    later write.
    """
    import asyncio

    writes: list[str] = []
    written = asyncio.Event()

    class _RecordingStorage:
        async def upsert_node_position_estimator_state(self, node_id, state):
            writes.append(node_id)
            written.set()

    processor = _kde_processor(SITE_ORIGIN, _RecordingStorage())
    state = StationaryKdeState()

    processor._maybe_checkpoint_kde("node-a", state, time.monotonic(), force=True)
    await asyncio.sleep(0)
    assert writes == []

    processor._persisted_node_ids.add("node-a")
    processor._maybe_checkpoint_kde("node-a", state, time.monotonic(), force=True)
    await asyncio.wait_for(written.wait(), timeout=5.0)
    assert writes == ["node-a"]


@pytest.mark.asyncio
async def test_checkpoint_serialization_runs_off_the_frame_path(monkeypatch) -> None:
    """Converting a full reservoir to geodetic must not block the audio path."""
    import asyncio
    import threading

    loop_thread_id = threading.get_ident()
    snapshot_thread_ids: list[int] = []
    written = asyncio.Event()
    original_snapshot = StationaryKdeState.snapshot

    def _tracking_snapshot(self, coordinate_frame):
        snapshot_thread_ids.append(threading.get_ident())
        return original_snapshot(self, coordinate_frame)

    monkeypatch.setattr(StationaryKdeState, "snapshot", _tracking_snapshot)

    class _RecordingStorage:
        async def upsert_node_position_estimator_state(self, node_id, state):
            written.set()

    processor = _kde_processor(SITE_ORIGIN, _RecordingStorage())
    processor._persisted_node_ids.add("node-a")

    processor._maybe_checkpoint_kde("node-a", StationaryKdeState(), time.monotonic(), force=True)
    await asyncio.wait_for(written.wait(), timeout=5.0)

    assert snapshot_thread_ids and loop_thread_id not in snapshot_thread_ids
