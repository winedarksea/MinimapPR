"""MinimapPR application entrypoint."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from minimappr.api.live import LiveEventHub
from minimappr.api.transports import HttpIngestTransport
from minimappr.classifiers.factory import create_classifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.environment import LiveEnvironmentProvider
from minimappr.core.federation import FederationCoordinator
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization_dispatch import build_localizer_from_settings
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import (
    AlertStatus,
    CopStatusResponse,
    FederationAck,
    FederationHeartbeat,
    FederationStatusResponse,
    FederationTrackSnapshot,
    FusionStatusResponse,
    GeoPoint,
    IngestFrameRequest,
    IngestFrameResponse,
    StoreForwardIngestRequest,
    StoreForwardIngestResponse,
    TrackState,
    ZoneSpec,
)
from minimappr.storage.db import Storage


logger = logging.getLogger(__name__)
frontend_dir = Path(__file__).parent / "frontend"


def _require_state(request: Request):
    if not hasattr(request.app.state, "storage"):
        raise RuntimeError("Storage is not initialized")
    return request.app.state


def _require_ws_state(websocket: WebSocket):
    if not hasattr(websocket.app.state, "storage"):
        raise RuntimeError("Storage is not initialized")
    return websocket.app.state


async def _cleanup_loop(app: FastAPI) -> None:
    while True:
        state = app.state
        settings: Settings = state.settings
        now_ns = time.time_ns()
        await state.storage.cleanup_expired_snippets(now_ns=now_ns)
        await state.storage.cleanup_retention(
            now_ns=now_ns,
            tier_ttls_seconds={
                "ephemeral": settings.retention_ephemeral_seconds,
                "short": settings.retention_short_seconds,
                "long": settings.retention_long_seconds,
                "experiment": settings.retention_experiment_seconds,
            },
            operational_ttls_seconds={
                "track_updates": settings.retention_track_updates_seconds,
                "alerts": settings.retention_alerts_seconds,
                "environment": settings.retention_environment_seconds,
                "dropped_tracks": settings.retention_dropped_tracks_seconds,
            },
        )
        await state.fusion_node.housekeeping_tick(now_ns=now_ns)
        await asyncio.sleep(settings.cleanup_interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    settings.federation_peers_config_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    settings.large_artifact_dir.mkdir(parents=True, exist_ok=True)

    storage_cfg = settings.storage_config()
    localization_cfg = settings.localization_config()
    tracking_cfg = settings.tracking_config()

    storage = Storage(storage_cfg.db_path)
    registry = NodeRegistry()
    audio_buffer = MultiSensorBuffer(max_duration_seconds=localization_cfg.max_sensor_buffer_seconds)
    localizer = build_localizer_from_settings(localization_cfg)
    classifier = create_classifier(settings)
    tracker = TrackManager(tracking_cfg)
    live_hub = LiveEventHub()
    coordinate_frame = LocalCoordinateFrame(
        origin=GeoPoint(
            lat=settings.site_origin_lat,
            lon=settings.site_origin_lon,
            alt_m=settings.site_origin_alt_m,
        ),
        mode=settings.coordinate_mode,
    )
    zone_matcher = ZoneMatcher(storage=storage)
    environment_provider = LiveEnvironmentProvider(
        fallback_temperature_c=settings.default_temperature_c,
        fallback_humidity_fraction=settings.default_humidity,
        max_reading_age_seconds=settings.environment_reading_max_age_seconds,
    )
    fusion_node = FusionNode(
        settings=settings,
        registry=registry,
        buffer=audio_buffer,
        localizer=localizer,
        classifier=classifier,
        tracker=tracker,
        storage=storage,
        live_callback=live_hub.broadcast,
        coordinate_frame=coordinate_frame,
        zone_matcher=zone_matcher,
        environment_provider=environment_provider,
    )
    ingest_transport = HttpIngestTransport(fusion_node)

    async def _federation_local_tracks(now_ns: int) -> list[TrackState]:
        tracks = await tracker.snapshot(now_ns=now_ns)
        active: list[TrackState] = []
        for track in tracks:
            if track.status not in {"tentative", "confirmed", "coasting"}:
                continue
            track.position_geo = coordinate_frame.local_to_geo(track.position_m)
            active.append(track)
        return active

    federation = FederationCoordinator(
        settings=settings,
        track_supplier=_federation_local_tracks,
        live_callback=live_hub.broadcast,
    )

    app.state.settings = settings
    app.state.storage = storage
    app.state.registry = registry
    app.state.audio_buffer = audio_buffer
    app.state.localizer = localizer
    app.state.classifier = classifier
    app.state.tracker = tracker
    app.state.live_hub = live_hub
    app.state.coordinate_frame = coordinate_frame
    app.state.zone_matcher = zone_matcher
    app.state.environment_provider = environment_provider
    app.state.fusion_node = fusion_node
    app.state.ingest_transport = ingest_transport
    app.state.federation = federation

    cleanup_task: asyncio.Task | None = None
    await storage.initialize()
    environment_provider.bootstrap(await storage.list_latest_environment_per_node(limit=1024))
    await fusion_node.start()
    await federation.start()
    cleanup_task = asyncio.create_task(_cleanup_loop(app))
    app.state.cleanup_task = cleanup_task

    try:
        yield
    finally:
        if cleanup_task is not None:
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
        await federation.stop()
        await fusion_node.stop()
        await storage.close()


app = FastAPI(title="MinimapPR", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.middleware("http")
async def dynamic_cors_headers(request: Request, call_next):
    settings: Settings | None = getattr(request.app.state, "settings", None)
    allowed_origins = settings.cors_allow_origins if settings is not None else ("http://localhost:8080", "http://127.0.0.1:8080")
    allow_credentials = bool(settings.cors_allow_credentials) if settings is not None else False
    origin = request.headers.get("origin")
    is_preflight = request.method == "OPTIONS" and "access-control-request-method" in request.headers

    if is_preflight:
        response = Response(status_code=204)
    else:
        response = await call_next(request)

    if origin and ("*" in allowed_origins or origin in allowed_origins):
        response.headers["Access-Control-Allow-Origin"] = "*" if "*" in allowed_origins else origin
        response.headers.setdefault("Vary", "Origin")
        response.headers["Access-Control-Allow-Methods"] = request.headers.get("access-control-request-method", "*")
        request_headers = request.headers.get("access-control-request-headers")
        response.headers["Access-Control-Allow-Headers"] = request_headers if request_headers else "*"
        if allow_credentials and "*" not in allowed_origins:
            response.headers["Access-Control-Allow-Credentials"] = "true"

    return response


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    if str(exc) == "Storage is not initialized":
        return JSONResponse(status_code=503, content={"detail": "Service unavailable"})
    logger.exception("Runtime error for %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error for %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
async def health(request: Request) -> dict:
    state = _require_state(request)
    settings: Settings = state.settings
    status = await state.fusion_node.status()
    workers = status.get("workers", {})
    running = int(workers.get("localization_running", 0)) + int(workers.get("classification_running", 0)) + int(
        workers.get("rules_running", 0)
    )
    federation_status = await state.federation.status()
    return {
        "status": "ok",
        "time_ns": time.time_ns(),
        "classifier": settings.classifier_backend,
        "fusion_queue_depth": status["queue"]["localization_depth"],
        "fusion_workers_running": running,
        "federation_enabled": federation_status["enabled"],
        "federation_peer_count": federation_status["peer_count"],
    }


@app.post("/api/v1/ingest/frame", response_model=IngestFrameResponse)
async def ingest_frame(payload: IngestFrameRequest, request: Request) -> IngestFrameResponse:
    state = _require_state(request)
    try:
        return await state.ingest_transport.deliver_frame(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/ingest/store-forward", response_model=StoreForwardIngestResponse)
async def ingest_store_forward(payload: StoreForwardIngestRequest, request: Request) -> StoreForwardIngestResponse:
    state = _require_state(request)
    try:
        return await state.ingest_transport.deliver_store_forward(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/nodes")
async def list_nodes(
    request: Request,
    limit: int = Query(default=500, ge=1, le=5000),
) -> list[dict]:
    state = _require_state(request)
    settings: Settings = state.settings
    now_ns = time.time_ns()
    nodes = await state.storage.list_nodes(limit=limit)
    for node in nodes:
        if node.get("position_geo") is None and node.get("position_m"):
            local = node["position_m"]
            geo = state.coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
            node["position_geo"] = geo.model_dump(mode="json")

        age_s = max(0.0, (now_ns - int(node["last_seen_ns"])) / 1_000_000_000.0)
        if age_s >= settings.node_offline_after_seconds:
            health_status = "offline"
        elif age_s >= settings.node_degraded_after_seconds:
            health_status = "degraded"
        else:
            health_status = "online"
        node["health_status"] = health_status
    return nodes


@app.get("/api/v1/detections")
async def list_detections(request: Request, limit: int = Query(default=100, ge=1, le=1000)) -> list[dict]:
    state = _require_state(request)
    detections = await state.storage.list_detections(limit=limit)
    for detection in detections:
        if detection.get("position_geo") is None and detection.get("position_m"):
            local = detection["position_m"]
            geo = state.coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
            detection["position_geo"] = geo.model_dump(mode="json")
    return detections


@app.get("/api/v1/tracks")
async def list_tracks(
    request: Request,
    limit: int = Query(default=200, ge=1, le=1000),
    include_standby: bool = Query(default=False),
) -> list[dict]:
    state = _require_state(request)
    now_ns = time.time_ns()
    _ = await state.tracker.snapshot(now_ns=now_ns)
    tracks = await state.storage.list_tracks(limit=limit)
    for track in tracks:
        if track.get("position_geo") is None and track.get("position_m"):
            local = track["position_m"]
            geo = state.coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
            track["position_geo"] = geo.model_dump(mode="json")
    if state.federation.enabled:
        tracks = await state.federation.merged_tracks(
            local_tracks=tracks,
            now_ns=now_ns,
            limit=limit,
            include_standby=include_standby,
        )
    for track in tracks:
        if track.get("position_geo") is None and track.get("position_m"):
            local = track["position_m"]
            geo = state.coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
            track["position_geo"] = geo.model_dump(mode="json")
    return tracks


@app.get("/api/v1/config")
async def get_config(request: Request) -> dict:
    state = _require_state(request)
    settings: Settings = state.settings
    return {
        "trigger_rms": settings.trigger_rms,
        "trigger_cooldown_seconds": settings.trigger_cooldown_seconds,
        "localization_window_seconds": settings.localization_window_seconds,
        "snippet_retention_seconds": settings.snippet_retention_seconds,
        "retention": {
            "ephemeral_seconds": settings.retention_ephemeral_seconds,
            "short_seconds": settings.retention_short_seconds,
            "long_seconds": settings.retention_long_seconds,
            "experiment_seconds": settings.retention_experiment_seconds,
            "track_updates_seconds": settings.retention_track_updates_seconds,
            "alerts_seconds": settings.retention_alerts_seconds,
            "environment_seconds": settings.retention_environment_seconds,
            "dropped_tracks_seconds": settings.retention_dropped_tracks_seconds,
        },
        "default_temperature_c": settings.default_temperature_c,
        "default_humidity": settings.default_humidity,
        "environment_reading_max_age_seconds": settings.environment_reading_max_age_seconds,
        "preprocess_enabled": settings.preprocess_enabled,
        "audio_highpass_hz": settings.audio_highpass_hz,
        "audio_lowpass_hz": settings.audio_lowpass_hz,
        "min_sensors_for_3d": settings.min_sensors_for_3d,
        "min_sensors_for_2d": settings.min_sensors_for_2d,
        "localization_max_tau_s": settings.localization_max_tau_s,
        "localization_algorithm": settings.localization_algorithm,
        "localization_strategy": settings.localization_strategy,
        "localization_srp_grid_resolution_m": settings.localization_srp_grid_resolution_m,
        "localization_search_padding_m": settings.localization_search_padding_m,
        "localization_music_azimuth_step_deg": settings.localization_music_azimuth_step_deg,
        "localization_music_elevation_step_deg": settings.localization_music_elevation_step_deg,
        "localization_subspace_freq_min_hz": settings.localization_subspace_freq_min_hz,
        "localization_subspace_freq_max_hz": settings.localization_subspace_freq_max_hz,
        "localization_refine_confidence_threshold": settings.localization_refine_confidence_threshold,
        "localization_tight_array_aperture_m": settings.localization_tight_array_aperture_m,
        "beamformed_classification_enabled": settings.beamformed_classification_enabled,
        "beamformer_type": settings.beamformer_type,
        "beamformed_classification_min_sensor_count": settings.beamformed_classification_min_sensor_count,
        "beamformed_classification_confidence_margin": settings.beamformed_classification_confidence_margin,
        "mvdr_diagonal_loading": settings.mvdr_diagonal_loading,
        "tracking_filter": settings.tracking_filter,
        "fusion_worker_count": settings.fusion_worker_count,
        "fusion_event_queue_size": settings.fusion_event_queue_size,
        "fusion_localization_queue_size": settings.fusion_localization_queue_size,
        "fusion_classification_queue_size": settings.fusion_classification_queue_size,
        "fusion_rules_queue_size": settings.fusion_rules_queue_size,
        "fusion_drop_on_backpressure": settings.fusion_drop_on_backpressure,
        "fusion_offline_replay_mode": settings.fusion_offline_replay_mode,
        "rules_config_path": str(settings.rules_config_path),
        "taxonomy_config_path": str(settings.taxonomy_config_path),
        "model_chain_config_path": str(settings.model_chain_config_path),
        "site_origin": {
            "lat": settings.site_origin_lat,
            "lon": settings.site_origin_lon,
            "alt_m": settings.site_origin_alt_m,
        },
        "coordinate_mode": settings.coordinate_mode,
        "node_degraded_after_seconds": settings.node_degraded_after_seconds,
        "node_offline_after_seconds": settings.node_offline_after_seconds,
        "federation": {
            "enabled": settings.federation_enabled and bool(settings.federation_peers),
            "server_id": settings.federation_server_id,
            "peer_count": len(settings.federation_peers),
            "publish_interval_seconds": settings.federation_publish_interval_seconds,
            "heartbeat_interval_seconds": settings.federation_heartbeat_interval_seconds,
            "link_timeout_seconds": settings.federation_link_timeout_seconds,
            "track_ttl_seconds": settings.federation_track_ttl_seconds,
            "deconflict_mahalanobis_gate": settings.federation_deconflict_mahalanobis_gate,
            "tqi_hysteresis": settings.federation_tqi_hysteresis,
        },
    }


@app.get("/api/v1/fusion/status", response_model=FusionStatusResponse)
async def fusion_status(request: Request) -> dict:
    state = _require_state(request)
    return await state.fusion_node.status()


@app.get("/api/v1/federation/status", response_model=FederationStatusResponse)
async def federation_status(request: Request) -> dict:
    state = _require_state(request)
    return await state.federation.status()


def _federation_headers(request: Request) -> tuple[str | None, str | None]:
    return request.headers.get("authorization"), request.headers.get("x-minimappr-token")


@app.post("/api/v1/federation/heartbeat", response_model=FederationAck)
async def federation_heartbeat(payload: FederationHeartbeat, request: Request) -> FederationAck:
    state = _require_state(request)
    settings: Settings = state.settings
    if not state.federation.enabled:
        raise HTTPException(status_code=503, detail="Federation is disabled")
    authorization, token_header = _federation_headers(request)
    if not await state.federation.validate_inbound_auth(
        peer_id=payload.server_id,
        authorization_header=authorization,
        token_header=token_header,
    ):
        raise HTTPException(status_code=401, detail="Invalid federation credentials")
    accepted = await state.federation.handle_incoming_heartbeat(payload)
    if not accepted:
        raise HTTPException(status_code=403, detail="Unknown federation peer")
    return FederationAck(
        ok=True,
        server_id=settings.federation_server_id,
        received_ns=time.time_ns(),
        peer_state="active",
    )


@app.post("/api/v1/federation/snapshot", response_model=FederationAck)
async def federation_snapshot(payload: FederationTrackSnapshot, request: Request) -> FederationAck:
    state = _require_state(request)
    settings: Settings = state.settings
    if not state.federation.enabled:
        raise HTTPException(status_code=503, detail="Federation is disabled")
    authorization, token_header = _federation_headers(request)
    if not await state.federation.validate_inbound_auth(
        peer_id=payload.server_id,
        authorization_header=authorization,
        token_header=token_header,
    ):
        raise HTTPException(status_code=401, detail="Invalid federation credentials")
    accepted = await state.federation.handle_incoming_snapshot(payload)
    if not accepted:
        raise HTTPException(status_code=403, detail="Unknown federation peer")
    return FederationAck(
        ok=True,
        server_id=settings.federation_server_id,
        received_ns=time.time_ns(),
        peer_state="active",
    )


@app.get("/api/v1/cop/status", response_model=CopStatusResponse)
async def cop_status(request: Request) -> CopStatusResponse:
    state = _require_state(request)
    settings: Settings = state.settings
    now_ns = time.time_ns()
    node_counts = await state.storage.count_nodes_by_status(
        now_ns=now_ns,
        degraded_after_seconds=settings.node_degraded_after_seconds,
        offline_after_seconds=settings.node_offline_after_seconds,
    )
    active_tracks = await state.storage.count_active_tracks()
    recent_window_ns = now_ns - 300_000_000_000
    recent_alert_count = await state.storage.recent_alert_count(since_ns=recent_window_ns)
    return CopStatusResponse(
        active_nodes=node_counts["online_nodes"],
        degraded_nodes=node_counts["degraded_nodes"],
        offline_nodes=node_counts["offline_nodes"],
        active_tracks=active_tracks,
        recent_alert_count=recent_alert_count,
    )


@app.get("/api/v1/alerts")
async def list_alerts(request: Request, limit: int = Query(default=100, ge=1, le=1000)) -> list[dict]:
    state = _require_state(request)
    return await state.storage.list_alerts(limit=limit)


@app.patch("/api/v1/alerts/{alert_id}")
async def update_alert_status(
    alert_id: str,
    status: AlertStatus,
    request: Request,
    reason: str | None = Query(default=None),
) -> dict:
    state = _require_state(request)
    ok = await state.storage.update_alert_status(
        alert_id=alert_id,
        status=status.value,
        updated_ns=time.time_ns(),
        payload_patch={"operator_reason": reason} if reason else None,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"ok": True, "alert_id": alert_id, "status": status.value}


@app.get("/api/v1/pings")
async def list_pings(request: Request, limit: int = Query(default=500, ge=1, le=5000)) -> list[dict]:
    state = _require_state(request)
    return await state.storage.list_pings(limit=limit)


@app.get("/api/v1/environment")
async def list_environment(
    request: Request,
    limit: int = Query(default=500, ge=1, le=5000),
    node_id: str | None = Query(default=None),
) -> list[dict]:
    state = _require_state(request)
    return await state.storage.list_environment(limit=limit, node_id=node_id)


@app.get("/api/v1/environment/current")
async def current_environment(
    request: Request,
    x: float | None = Query(default=None),
    y: float | None = Query(default=None),
    z: float | None = Query(default=None),
) -> dict:
    state = _require_state(request)
    if any(value is None for value in (x, y, z)) and any(value is not None for value in (x, y, z)):
        raise HTTPException(status_code=400, detail="x, y, and z must all be provided together")
    location = (float(x), float(y), float(z)) if x is not None and y is not None and z is not None else None
    conditions = state.environment_provider.get_conditions(location_m=location)
    return {
        "temperature_c": conditions.temperature_c,
        "humidity_fraction": conditions.humidity_fraction,
        "pressure_pa": conditions.pressure_pa,
        "wind_speed_mps": conditions.wind_speed_mps,
        "wind_dir_deg": conditions.wind_dir_deg,
        "speed_of_sound_mps": state.environment_provider.get_speed_of_sound(location_m=location),
        "metadata": conditions.metadata,
    }


@app.get("/api/v1/zones")
async def list_zones(request: Request) -> list[dict]:
    state = _require_state(request)
    return await state.storage.list_zones()


@app.put("/api/v1/zones/{zone_id}")
async def upsert_zone(zone_id: str, payload: ZoneSpec, request: Request) -> dict:
    state = _require_state(request)
    if payload.id != zone_id:
        raise HTTPException(status_code=400, detail="zone_id path must match payload.id")
    await state.storage.upsert_zone(
        zone_id=payload.id,
        name=payload.name,
        zone_type=payload.zone_type.value,
        polygon_geo=payload.polygon_geo,
        properties=payload.properties,
        created_ns=time.time_ns(),
    )
    return {"ok": True, "zone_id": zone_id}


@app.delete("/api/v1/zones/{zone_id}")
async def delete_zone(zone_id: str, request: Request) -> dict:
    state = _require_state(request)
    deleted = await state.storage.delete_zone(zone_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Zone not found")
    return {"ok": True, "zone_id": zone_id}


@app.get("/api/v1/detections/{detection_id}/audio")
async def get_detection_audio(detection_id: str, request: Request) -> FileResponse:
    state = _require_state(request)
    settings: Settings = state.settings
    snippet_path = await state.storage.snippet_path_for_detection(detection_id)
    if not snippet_path:
        raise HTTPException(status_code=404, detail="Snippet not found for detection")

    snippet_file = Path(snippet_path).resolve()
    if not snippet_file.exists():
        raise HTTPException(status_code=404, detail="Snippet file no longer exists")

    snippet_root = settings.snippet_dir.resolve()
    if not snippet_file.is_relative_to(snippet_root):
        raise HTTPException(status_code=403, detail="Snippet path is outside snippet directory")

    return FileResponse(
        path=snippet_file,
        media_type="audio/wav",
        filename=f"{detection_id}.wav",
    )


@app.websocket("/ws/live")
async def live_events(websocket: WebSocket) -> None:
    state = _require_ws_state(websocket)
    live_hub: LiveEventHub = state.live_hub
    await live_hub.connect(websocket)
    await websocket.send_json(
        {
            "type": "subscription_ack",
            "filter": {
                "zone_ids": [],
                "label_categories": [],
                "labels": [],
                "track_status": [],
                "min_confidence": None,
            },
        }
    )
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect

            text = message.get("text")
            if not text:
                continue

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            sub_filter = await live_hub.update_filter(websocket=websocket, payload=payload)
            await websocket.send_json(
                {
                    "type": "subscription_ack",
                    "filter": {
                        "zone_ids": sorted(sub_filter.zone_ids),
                        "label_categories": sorted(sub_filter.label_categories),
                        "labels": sorted(sub_filter.labels),
                        "track_status": sorted(sub_filter.track_statuses),
                        "min_confidence": sub_filter.min_confidence,
                    },
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        await live_hub.disconnect(websocket)
