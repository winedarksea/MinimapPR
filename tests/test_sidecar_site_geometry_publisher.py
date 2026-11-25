"""Geometry publisher (P3): revisions, triggers, estimate threshold, reconcile, HTTP transport."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from minimappr.core.geo import LocalCoordinateFrame
from minimappr.models import GeoPoint
from minimappr.sidecar_json_http import SidecarRequestError
from minimappr.sidecar_site_geometry_publisher import (
    PUSH_REASON_NODE_OVERRIDE_CHANGED,
    AppliedSiteGeometry,
    HttpSiteGeometryTransport,
    SidecarSiteGeometryPublisher,
)
from tests.site_geometry_fixtures import (
    SITE_ORIGIN,
    FakeSidecarGeometryEndpoint,
    FakeSiteGeometryStorage,
    estimator_checkpoint_row,
    node_row,
)

ESTIMATE_MOVE_THRESHOLD_M = 0.5


class PublisherHarness:
    def __init__(self) -> None:
        self.storage = FakeSiteGeometryStorage()
        self.sidecar = FakeSidecarGeometryEndpoint()
        self.coordinate_frame = LocalCoordinateFrame(origin=SITE_ORIGIN, mode="flat")
        self.sidecar_running = True
        self.reconcile_interval_seconds = 3600.0
        self.publisher = SidecarSiteGeometryPublisher(
            storage=self.storage,
            coordinate_frame_provider=lambda: self.coordinate_frame,
            transport=self.sidecar,
            sidecar_is_running=lambda: self.sidecar_running,
            estimate_move_threshold_m_provider=lambda: ESTIMATE_MOVE_THRESHOLD_M,
            reconcile_interval_seconds_provider=lambda: self.reconcile_interval_seconds,
        )

    def set_estimate(self, node_id: str, east_m: float, north_m: float) -> None:
        if not any(row["id"] == node_id for row in self.storage.node_rows):
            self.storage.node_rows.append(node_row(node_id))
        estimate_geo = self.coordinate_frame.local_to_geo((east_m, north_m, 0.0))
        self.storage.estimator_states[node_id] = estimator_checkpoint_row(estimate_geo, horizontal_std_m=0.4)


@pytest.fixture
def harness() -> PublisherHarness:
    return PublisherHarness()


@pytest.mark.asyncio
async def test_launch_body_is_contract_json_and_revision_starts_at_one(harness: PublisherHarness) -> None:
    harness.storage.node_rows.append(node_row("pinned", overrides={"position_m": [1.0, 2.0, 3.0]}))

    launch_json = await harness.publisher.launch_geometry_json()

    body = json.loads(launch_json)
    assert body["geometry_push_revision"] == 1
    assert body["node_placements"]["pinned"]["position_m"] == [1.0, 2.0, 3.0]
    assert harness.publisher.adopted_revision == 1


@pytest.mark.asyncio
async def test_revision_is_monotonic_and_only_bumps_on_change(harness: PublisherHarness) -> None:
    harness.storage.node_rows.append(node_row("pinned", overrides={"position_m": [1.0, 2.0, 3.0]}))
    await harness.publisher.launch_geometry_json()
    await harness.publisher.push_current_geometry(reasons={"unchanged"})
    assert harness.sidecar.pushed_bodies[-1]["geometry_push_revision"] == 1

    harness.storage.node_rows[0] = node_row("pinned", overrides={"position_m": [1.0, 2.0, 4.0]})
    await harness.publisher.push_current_geometry(reasons={PUSH_REASON_NODE_OVERRIDE_CHANGED})
    harness.storage.node_rows.clear()
    await harness.publisher.push_current_geometry(reasons={"node_deleted"})

    revisions = [body["geometry_push_revision"] for body in harness.sidecar.pushed_bodies]
    assert revisions == [1, 2, 3]
    assert harness.sidecar.pushed_bodies[-1]["node_placements"] == {}
    assert harness.publisher.sidecar_acknowledged_revision == 3


@pytest.mark.asyncio
async def test_request_push_runs_in_background_and_coalesces(harness: PublisherHarness) -> None:
    harness.publisher.start()
    try:
        harness.storage.node_rows.append(node_row("n", overrides={"position_m": [0.0, 0.0, 0.0]}))
        harness.publisher.request_push(PUSH_REASON_NODE_OVERRIDE_CHANGED)
        harness.publisher.request_push("node_deleted")
        await asyncio.wait_for(harness.publisher.wait_until_idle(), timeout=5.0)
    finally:
        await harness.publisher.stop()

    assert len(harness.sidecar.pushed_bodies) == 1
    assert "n" in harness.sidecar.pushed_bodies[0]["node_placements"]


@pytest.mark.asyncio
async def test_site_origin_rebind_changes_frame_id_and_revision(harness: PublisherHarness) -> None:
    await harness.publisher.push_current_geometry(reasons={"launch"})
    first_frame_id = harness.sidecar.pushed_bodies[-1]["site_frame_id"]

    harness.coordinate_frame = LocalCoordinateFrame(
        origin=GeoPoint(lat=SITE_ORIGIN.lat + 0.01, lon=SITE_ORIGIN.lon, alt_m=SITE_ORIGIN.alt_m), mode="flat"
    )
    await harness.publisher.push_current_geometry(reasons={"site_origin_rebound"})

    assert harness.sidecar.pushed_bodies[-1]["site_frame_id"] != first_frame_id
    assert harness.sidecar.pushed_bodies[-1]["geometry_push_revision"] == 2


@pytest.mark.asyncio
async def test_estimate_drift_below_threshold_is_not_pushed_and_above_is(harness: PublisherHarness) -> None:
    harness.set_estimate("gps-node", 10.0, 20.0)
    await harness.publisher.push_current_geometry(reasons={"launch"})

    harness.set_estimate("gps-node", 10.3, 20.0)
    await harness.publisher.reconcile_once()
    assert len(harness.sidecar.pushed_bodies) == 1

    harness.set_estimate("gps-node", 10.8, 20.0)
    await harness.publisher.reconcile_once()

    assert len(harness.sidecar.pushed_bodies) == 2
    moved = harness.sidecar.pushed_bodies[-1]
    assert moved["geometry_push_revision"] == 2
    assert moved["node_placements"]["gps-node"]["position_m"][0] == pytest.approx(10.8, abs=1e-6)


@pytest.mark.asyncio
async def test_reconcile_repushes_after_a_sidecar_restart(harness: PublisherHarness) -> None:
    harness.storage.node_rows.append(node_row("pinned", overrides={"position_m": [1.0, 2.0, 3.0]}))
    await harness.publisher.push_current_geometry(reasons={"launch"})
    await harness.publisher.reconcile_once()
    assert len(harness.sidecar.pushed_bodies) == 1  # in sync: nothing to do

    harness.sidecar.simulate_restart_without_geometry()
    await harness.publisher.reconcile_once()

    assert len(harness.sidecar.pushed_bodies) == 2
    assert harness.sidecar.pushed_bodies[-1]["geometry_push_revision"] == 1
    assert harness.sidecar.applied.revision == 1


@pytest.mark.asyncio
async def test_python_restart_under_a_running_sidecar_pushes_above_its_revision(harness: PublisherHarness) -> None:
    # A fresh Python process starts at revision 1; the surviving sidecar holds 5.
    harness.sidecar.applied = AppliedSiteGeometry(revision=5, site_frame_id="site:previous-process")
    harness.storage.node_rows.append(node_row("pinned", overrides={"position_m": [1.0, 2.0, 3.0]}))

    assert await harness.publisher.push_current_geometry(reasons={"launch"})

    assert [body["geometry_push_revision"] for body in harness.sidecar.rejected_bodies] == [1]
    assert harness.sidecar.pushed_bodies[-1]["geometry_push_revision"] == 6
    assert harness.publisher.adopted_revision == 6
    assert harness.publisher.sidecar_acknowledged_revision == 6

    harness.storage.node_rows[0] = node_row("pinned", overrides={"position_m": [1.0, 2.0, 9.0]})
    await harness.publisher.push_current_geometry(reasons={PUSH_REASON_NODE_OVERRIDE_CHANGED})
    assert harness.sidecar.pushed_bodies[-1]["geometry_push_revision"] == 7


@pytest.mark.asyncio
async def test_reconcile_recovers_when_the_sidecar_is_ahead(harness: PublisherHarness) -> None:
    await harness.publisher.push_current_geometry(reasons={"launch"})
    harness.sidecar.applied = AppliedSiteGeometry(revision=40, site_frame_id="site:other")

    await harness.publisher.reconcile_once()

    assert harness.sidecar.pushed_bodies[-1]["geometry_push_revision"] == 41
    assert harness.publisher.sidecar_acknowledged_revision == 41


@pytest.mark.asyncio
async def test_reconcile_repushes_on_frame_mismatch(harness: PublisherHarness) -> None:
    await harness.publisher.push_current_geometry(reasons={"launch"})
    harness.sidecar.applied = AppliedSiteGeometry(revision=1, site_frame_id="site:other")

    await harness.publisher.reconcile_once()

    assert len(harness.sidecar.pushed_bodies) == 2


@pytest.mark.asyncio
async def test_reconcile_without_applied_revision_logs_and_retries_without_pushing(
    harness: PublisherHarness, caplog: pytest.LogCaptureFixture
) -> None:
    await harness.publisher.push_current_geometry(reasons={"launch"})
    harness.sidecar.applied = None  # sidecar build exposes no applied revision (GET 404)

    with caplog.at_level(logging.WARNING, logger="minimappr.sidecar_site_geometry_publisher"):
        await harness.publisher.reconcile_once()
        await harness.publisher.reconcile_once()

    assert len(harness.sidecar.pushed_bodies) == 1
    assert harness.sidecar.fetch_count == 2
    assert sum("no applied_geometry_push_revision" in message for message in caplog.messages) == 1


@pytest.mark.asyncio
async def test_unreachable_sidecar_keeps_adopted_geometry_for_the_next_tick(harness: PublisherHarness) -> None:
    harness.storage.node_rows.append(node_row("pinned", overrides={"position_m": [1.0, 2.0, 3.0]}))
    harness.sidecar.push_error = SidecarRequestError("/api/v1/dsp/geometry unreachable: refused")

    assert not await harness.publisher.push_current_geometry(reasons={"launch"})
    assert harness.publisher.adopted_revision == 1
    assert harness.publisher.sidecar_acknowledged_revision is None

    harness.sidecar.push_error = None
    harness.sidecar.simulate_restart_without_geometry()
    await harness.publisher.reconcile_once()

    assert harness.sidecar.pushed_bodies[-1]["geometry_push_revision"] == 1
    assert harness.publisher.sidecar_acknowledged_revision == 1


@pytest.mark.asyncio
async def test_mismatched_acknowledgement_is_not_counted_as_applied(harness: PublisherHarness) -> None:
    harness.sidecar.acknowledge_revision_override = 99

    assert not await harness.publisher.push_current_geometry(reasons={"launch"})
    assert harness.publisher.sidecar_acknowledged_revision is None


@pytest.mark.asyncio
async def test_nothing_is_pushed_while_the_sidecar_is_not_running(harness: PublisherHarness) -> None:
    harness.sidecar_running = False
    harness.storage.node_rows.append(node_row("pinned", overrides={"position_m": [1.0, 2.0, 3.0]}))

    await harness.publisher.push_current_geometry(reasons={"launch"})
    await harness.publisher.reconcile_once()

    assert harness.sidecar.pushed_bodies == []
    assert harness.sidecar.fetch_count == 0
    assert harness.publisher.adopted_revision == 1  # a relaunch would carry it


@pytest.mark.asyncio
async def test_reconcile_tick_runs_on_the_configured_interval(harness: PublisherHarness) -> None:
    harness.reconcile_interval_seconds = 0.01
    harness.publisher.start()
    try:
        for _ in range(200):
            if harness.sidecar.pushed_bodies and harness.sidecar.fetch_count >= 1:
                break
            await asyncio.sleep(0.01)
    finally:
        await harness.publisher.stop()

    # First tick adopts and pushes; later ticks verify the applied revision.
    assert harness.sidecar.pushed_bodies[0]["geometry_push_revision"] == 1
    assert harness.sidecar.fetch_count >= 1


@pytest.mark.asyncio
async def test_launch_without_buildable_geometry_does_not_block_the_sidecar(harness: PublisherHarness) -> None:
    async def failing_list_nodes(limit=None):
        raise RuntimeError("Storage is not initialized")

    harness.storage.list_nodes = failing_list_nodes  # type: ignore[method-assign]

    assert await harness.publisher.launch_geometry_json() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_overrides",
    [
        {"position_m": [float("nan"), 2.0, 3.0]},
        {"position_m": [1.0, 2.0, 3.0], "position_std_m": float("inf")},
    ],
    ids=["nan_position", "infinite_accuracy"],
)
async def test_non_finite_stored_override_skips_that_node_and_still_launches(
    harness: PublisherHarness, stored_overrides: dict, caplog: pytest.LogCaptureFixture
) -> None:
    harness.storage.node_rows.append(node_row("broken", overrides=stored_overrides))
    harness.storage.node_rows.append(node_row("pinned", overrides={"position_m": [1.0, 2.0, 3.0]}))

    with caplog.at_level(logging.WARNING):
        launch_json = await harness.publisher.launch_geometry_json()

    body = json.loads(launch_json)
    assert list(body["node_placements"]) == ["pinned"]
    assert any("broken" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_launch_body_that_cannot_serialize_does_not_block_the_sidecar(harness: PublisherHarness) -> None:
    harness.storage.node_rows.append(node_row("pinned", overrides={"position_m": [1.0, 2.0, 3.0]}))
    harness.publisher.adopted_push_body = lambda: {"site_origin": {"alt_m": float("nan")}}  # type: ignore[method-assign]

    assert await harness.publisher.launch_geometry_json() is None


class _FakeSidecarHttpHandler(BaseHTTPRequestHandler):
    server_version = "fake-sidecar"
    routes: dict[tuple[str, str], tuple[int, dict]] = {}
    received_bodies: list[dict] = []

    def _reply(self, method: str) -> None:
        status, payload = self.routes.get((method, self.path), (404, {"error": "not found"}))
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        self._reply("GET")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.received_bodies.append(json.loads(self.rfile.read(length)))
        self._reply("POST")

    def log_message(self, format, *args) -> None:  # noqa: A002
        return


@pytest.fixture
def fake_sidecar_http_server():
    _FakeSidecarHttpHandler.routes = {}
    _FakeSidecarHttpHandler.received_bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeSidecarHttpHandler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server, _FakeSidecarHttpHandler
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_http_transport_posts_body_and_reads_acknowledgement(fake_sidecar_http_server) -> None:
    server, handler = fake_sidecar_http_server
    handler.routes[("POST", "/api/v1/dsp/geometry")] = (
        200,
        {"site_frame_id": "site:x", "applied_geometry_push_revision": 4, "node_placement_count": 0},
    )
    transport = HttpSiteGeometryTransport(f"http://127.0.0.1:{server.server_address[1]}")

    acknowledgement = await transport.push_site_geometry({"site_frame_id": "site:x", "geometry_push_revision": 4})

    assert acknowledgement["applied_geometry_push_revision"] == 4
    assert handler.received_bodies == [{"site_frame_id": "site:x", "geometry_push_revision": 4}]


@pytest.mark.asyncio
async def test_http_transport_reads_applied_revision_from_get_then_health(fake_sidecar_http_server) -> None:
    server, handler = fake_sidecar_http_server
    transport = HttpSiteGeometryTransport(f"http://127.0.0.1:{server.server_address[1]}")
    handler.routes[("GET", "/healthz")] = (200, {"status": "ok"})
    assert await transport.fetch_applied_site_geometry() is None  # 404 on GET, nothing on health

    handler.routes[("GET", "/healthz")] = (200, {"status": "ok", "applied_geometry_push_revision": 2})
    assert await transport.fetch_applied_site_geometry() == AppliedSiteGeometry(revision=2, site_frame_id=None)

    handler.routes[("GET", "/api/v1/dsp/geometry")] = (
        200,
        {
            "coordinate_frame": {"frame_id": None, "kind": "sidecar_automatic_enu", "origin": None, "projection": None},
            "applied_geometry_push_revision": None,
            "node_placement_count": 0,
        },
    )
    assert await transport.fetch_applied_site_geometry() == AppliedSiteGeometry(revision=None, site_frame_id=None)

    handler.routes[("GET", "/api/v1/dsp/geometry")] = (
        200,
        {
            "coordinate_frame": {
                "frame_id": "site:x",
                "kind": "site_enu",
                "origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0},
                "projection": "flat_wgs84_series",
            },
            "applied_geometry_push_revision": 3,
            "node_placement_count": 2,
        },
    )
    assert await transport.fetch_applied_site_geometry() == AppliedSiteGeometry(revision=3, site_frame_id="site:x")


@pytest.mark.asyncio
async def test_http_transport_carries_the_stale_revision_rejection_body(fake_sidecar_http_server) -> None:
    server, handler = fake_sidecar_http_server
    handler.routes[("POST", "/api/v1/dsp/geometry")] = (
        409,
        {"error": "stale_geometry_push_revision", "detail": "lower", "applied_geometry_push_revision": 7},
    )
    transport = HttpSiteGeometryTransport(f"http://127.0.0.1:{server.server_address[1]}")

    with pytest.raises(SidecarRequestError) as raised:
        await transport.push_site_geometry({"site_frame_id": "site:x", "geometry_push_revision": 1})

    assert raised.value.status_code == 409
    assert raised.value.response_body["applied_geometry_push_revision"] == 7


@pytest.mark.asyncio
async def test_http_transport_raises_on_rejected_push(fake_sidecar_http_server) -> None:
    server, handler = fake_sidecar_http_server
    handler.routes[("POST", "/api/v1/dsp/geometry")] = (422, {"error": "unknown projection"})
    transport = HttpSiteGeometryTransport(f"http://127.0.0.1:{server.server_address[1]}")

    with pytest.raises(SidecarRequestError) as raised:
        await transport.push_site_geometry({"site_frame_id": "site:x"})

    assert raised.value.status_code == 422
    assert "unknown projection" in str(raised.value)
