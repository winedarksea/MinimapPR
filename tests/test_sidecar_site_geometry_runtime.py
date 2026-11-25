"""Geometry push wiring (P3): launch env, endpoint triggers, real storage, real sidecar parse."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from minimappr import main
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.node_position_estimator import StationaryKdeState
from minimappr.core.site_origin import SOURCE_MANUAL, SiteOriginResolution
from minimappr.ingest_sidecar_runtime import (
    SITE_GEOMETRY_LAUNCH_ENVIRONMENT_VARIABLE,
    build_ingest_sidecar_environment,
)
from minimappr.main import app
from minimappr.models import GeoPoint, NodeSpec, NodeType
from minimappr.sidecar_site_geometry_publisher import (
    PUSH_REASON_NODE_DELETED,
    PUSH_REASON_NODE_OVERRIDE_CHANGED,
    PUSH_REASON_POSITION_ESTIMATOR_RESET,
    PUSH_REASON_SITE_ORIGIN_REBOUND,
    SidecarSiteGeometryPublisher,
)
from tests.site_geometry_fixtures import SITE_ORIGIN, FakeSidecarGeometryEndpoint


class RecordingSiteGeometryPublisher:
    def __init__(self) -> None:
        self.requested_reasons: list[str] = []
        self.started = False
        self.stopped = False

    def request_push(self, reason: str) -> None:
        self.requested_reasons.append(reason)

    def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def launch_geometry_json(self) -> str | None:
        return '{"recorded": true}'


def _configure_managed_rust_sidecar_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "geometry.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_INGEST_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "rust")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "true")
    monkeypatch.setenv("MINIMAPPR_DIRECT_INGEST_ENABLED", "false")
    # Missing binary: the managed launch path runs, the process does not.
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_BINARY_PATH", str(tmp_path / "missing-sidecar"))
    monkeypatch.setenv("MINIMAPPR_SITE_ORIGIN_SOURCE", "manual")
    monkeypatch.setenv("MINIMAPPR_SITE_ORIGIN_LAT", repr(SITE_ORIGIN.lat))
    monkeypatch.setenv("MINIMAPPR_SITE_ORIGIN_LON", repr(SITE_ORIGIN.lon))
    monkeypatch.setenv("MINIMAPPR_SITE_ORIGIN_ALT_M", repr(SITE_ORIGIN.alt_m))


@pytest.fixture
def recording_publisher(monkeypatch: pytest.MonkeyPatch) -> RecordingSiteGeometryPublisher:
    publisher = RecordingSiteGeometryPublisher()
    monkeypatch.setattr(main, "_build_site_geometry_publisher", lambda *args, **kwargs: publisher)
    return publisher


def _register_offline_node(client: TestClient, node_id: str) -> None:
    response = client.post(
        "/api/v1/nodes",
        json={"id": node_id, "node_type": "point", "position_m": [1.0, 2.0, 0.0], "capabilities": ["audio"]},
    )
    assert response.status_code == 201, response.text


def test_node_override_patch_requests_a_push(monkeypatch, tmp_path, recording_publisher) -> None:
    _configure_managed_rust_sidecar_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        _register_offline_node(client, "pinned")
        assert recording_publisher.started

        response = client.patch(
            "/api/v1/nodes/pinned",
            json={"overrides": {"position_m": [3.0, 4.0, 1.0], "position_std_m": 0.02, "orientation": {"yaw_deg": 30}}},
        )
        assert response.status_code == 200, response.text
        assert response.json()["overrides"]["position_std_m"] == 0.02
        assert recording_publisher.requested_reasons == [PUSH_REASON_NODE_OVERRIDE_CHANGED]

        client.patch("/api/v1/nodes/pinned", json={"clear_overrides": ["orientation"]})
        assert recording_publisher.requested_reasons[-1] == PUSH_REASON_NODE_OVERRIDE_CHANGED
        assert len(recording_publisher.requested_reasons) == 2
    assert recording_publisher.stopped


def test_non_finite_override_is_rejected_at_the_api(monkeypatch, tmp_path, recording_publisher) -> None:
    # Starlette parses NaN/Infinity; Rust would refuse the geometry at launch.
    _configure_managed_rust_sidecar_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        _register_offline_node(client, "pinned")
        for overrides in (
            {"position_m": [3.0, 4.0, 1.0], "position_std_m": float("inf")},
            {"position_m": [float("nan"), 4.0, 1.0]},
        ):
            # httpx refuses to encode non-finite JSON; send what a client could.
            response = client.patch(
                "/api/v1/nodes/pinned",
                content=json.dumps({"overrides": overrides}).encode(),
                headers={"content-type": "application/json"},
            )
            assert response.status_code == 422, response.text
    assert recording_publisher.requested_reasons == []


def test_non_geometry_patch_does_not_push(monkeypatch, tmp_path, recording_publisher) -> None:
    _configure_managed_rust_sidecar_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        _register_offline_node(client, "plain")
        response = client.patch("/api/v1/nodes/plain", json={"metadata": {"note": "mast B"}})
        assert response.status_code == 200, response.text
    assert recording_publisher.requested_reasons == []


def test_node_delete_and_estimator_reset_request_pushes(monkeypatch, tmp_path, recording_publisher) -> None:
    _configure_managed_rust_sidecar_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        _register_offline_node(client, "stale")
        assert client.post("/api/v1/nodes/stale/position-estimator/reset").status_code == 200
        assert client.delete("/api/v1/nodes/stale").status_code == 200
    assert recording_publisher.requested_reasons == [PUSH_REASON_POSITION_ESTIMATOR_RESET, PUSH_REASON_NODE_DELETED]


def test_site_origin_rebind_requests_a_push_and_rebinds_the_registry(monkeypatch, tmp_path, recording_publisher) -> None:
    _configure_managed_rust_sidecar_env(monkeypatch, tmp_path)
    new_origin = GeoPoint(lat=SITE_ORIGIN.lat + 0.01, lon=SITE_ORIGIN.lon, alt_m=SITE_ORIGIN.alt_m)
    with TestClient(app) as client:
        client.portal.call(main._rebind_site_origin, app, SiteOriginResolution(origin=new_origin, source=SOURCE_MANUAL))
        assert app.state.registry._coordinate_frame.origin == new_origin
    assert recording_publisher.requested_reasons == [PUSH_REASON_SITE_ORIGIN_REBOUND]


def test_publisher_exists_only_where_this_process_owns_the_sidecar(monkeypatch, tmp_path) -> None:
    _configure_managed_rust_sidecar_env(monkeypatch, tmp_path)
    with TestClient(app):
        assert isinstance(app.state.site_geometry_publisher, SidecarSiteGeometryPublisher)
    assert getattr(app.state, "site_geometry_publisher", None) is None

    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    with TestClient(app):
        assert app.state.site_geometry_publisher is None


@pytest.mark.asyncio
async def test_start_ingest_sidecar_passes_the_publisher_geometry_to_the_launch_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary_path = tmp_path / "minimappr-ingest-sidecar"
    binary_path.write_text("stub", encoding="utf-8")
    observed: dict[str, object] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        observed["env"] = kwargs["env"]
        return SimpleNamespace(pid=1, returncode=None)

    async def fake_wait_for_ready(process, **kwargs) -> None:
        return None

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(main, "_wait_for_ingest_sidecar_ready", fake_wait_for_ready)
    monkeypatch.setattr(main, "_probe_ingest_sidecar_ready", lambda *args, **kwargs: False)
    monkeypatch.setattr(app.state, "site_geometry_publisher", RecordingSiteGeometryPublisher(), raising=False)
    settings = SimpleNamespace(
        ingest_sidecar_binary_path=binary_path,
        ingest_spool_dir=tmp_path / "spool",
        ingest_consumer_name="python-ingest",
        ingest_sidecar_port=18089,
        ingest_storage_mode="journal",
        ingest_sidecar_total_journal_budget_bytes=1024,
        ingest_sidecar_admission_reserve_bytes=128,
        ingest_sidecar_allow_non_tmpfs_journal=True,
    )

    await main._start_ingest_sidecar(settings)

    assert observed["env"][SITE_GEOMETRY_LAUNCH_ENVIRONMENT_VARIABLE] == '{"recorded": true}'


def test_launch_env_omits_geometry_when_none_is_supplied() -> None:
    settings = SimpleNamespace(ingest_spool_dir=Path("/tmp/spool"))
    with_geometry = build_ingest_sidecar_environment(
        settings, default_classifier_command_json_builder=lambda _settings: None, site_geometry_json='{"a":1}'
    )
    without_geometry = build_ingest_sidecar_environment(
        settings, default_classifier_command_json_builder=lambda _settings: None
    )
    assert with_geometry[SITE_GEOMETRY_LAUNCH_ENVIRONMENT_VARIABLE] == '{"a":1}'
    assert without_geometry.get(SITE_GEOMETRY_LAUNCH_ENVIRONMENT_VARIABLE) == main.os.environ.get(
        SITE_GEOMETRY_LAUNCH_ENVIRONMENT_VARIABLE
    )


@pytest.mark.asyncio
async def test_publisher_builds_from_real_storage_rows_and_checkpoints(temp_storage) -> None:
    frame = LocalCoordinateFrame(origin=SITE_ORIGIN, mode="flat")
    now_ns = time.time_ns()
    for node_id in ("pinned", "gps-node", "mobile-node"):
        await temp_storage.upsert_node(
            NodeSpec(id=node_id, node_type=NodeType.POINT, position_m=(50.0, 50.0, 0.0)), now_ns
        )
    await temp_storage.set_node_overrides("pinned", {"position_m": [1.0, 2.0, 3.0], "position_std_m": 0.03})
    await temp_storage.set_node_overrides("mobile-node", {"mobility": "mobile"})
    estimate_m = (12.0, -7.0, 1.0)
    kde_state = StationaryKdeState(samples=[estimate_m], seen_count=40, estimate=estimate_m, horizontal_std_m=0.6)
    for node_id in ("gps-node", "mobile-node"):
        await temp_storage.upsert_node_position_estimator_state(node_id, kde_state.snapshot(frame))
    sidecar = FakeSidecarGeometryEndpoint()
    publisher = SidecarSiteGeometryPublisher(
        storage=temp_storage,
        coordinate_frame_provider=lambda: frame,
        transport=sidecar,
        sidecar_is_running=lambda: True,
        estimate_move_threshold_m_provider=lambda: 0.5,
        reconcile_interval_seconds_provider=lambda: 30.0,
    )

    await publisher.push_current_geometry(reasons={"test"})

    placements = sidecar.pushed_bodies[0]["node_placements"]
    assert set(placements) == {"pinned", "gps-node"}
    assert placements["pinned"] == {
        "position_m": [1.0, 2.0, 3.0],
        "position_std_m": 0.03,
        "placement_source": "operator_override",
    }
    assert placements["gps-node"]["placement_source"] == "stationary_estimate"
    assert placements["gps-node"]["position_m"] == pytest.approx(list(estimate_m), abs=1e-4)
    assert placements["gps-node"]["position_std_m"] == 0.6


def test_real_sidecar_accepts_the_launch_geometry_body(tmp_path: Path) -> None:
    """The body Python builds must parse under the Rust serde types, or the
    sidecar refuses to start (contract section 10)."""
    from tests.rust_sidecar_e2e_helpers import http_json, running_rust_sidecar
    from tests.site_geometry_fixtures import node_row
    from minimappr.sidecar_site_geometry import build_site_geometry_snapshot

    frame = LocalCoordinateFrame(origin=SITE_ORIGIN, mode="geodetic")
    snapshot = build_site_geometry_snapshot(
        node_rows=[
            node_row("planar-a", overrides={"position_m": [1.0, 2.0, 1.5], "orientation": {"yaw_deg": 35.0}},
                     orientation={"yaw_deg": 35.0}),
            node_row("point-b", overrides={"position_geo": SITE_ORIGIN.model_dump(mode="json")}),
        ],
        estimator_states_by_node_id={},
        coordinate_frame=frame,
    )
    launch_json = json.dumps(snapshot.to_push_body(1), separators=(",", ":"))

    with running_rust_sidecar(tmp_path, extra_env={SITE_GEOMETRY_LAUNCH_ENVIRONMENT_VARIABLE: launch_json}) as (port, _):
        assert http_json(port, "/healthz")["status"] == "ok"


def test_real_sidecar_reads_the_launch_geometry_env(tmp_path: Path) -> None:
    """Negative control for the parse test: a body Rust cannot parse stops startup,
    so the passing test above really exercised the env var."""
    from tests.rust_sidecar_e2e_helpers import running_rust_sidecar

    malformed_json = json.dumps({"site_frame_id": "site:x", "projection": "sphere_6371km"})
    with pytest.raises(AssertionError, match="exited before becoming ready"):
        with running_rust_sidecar(tmp_path, extra_env={SITE_GEOMETRY_LAUNCH_ENVIRONMENT_VARIABLE: malformed_json}):
            pass
