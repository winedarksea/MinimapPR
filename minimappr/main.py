"""MinimapPR application entrypoint."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from minimappr.api.live import LiveEventHub
from minimappr.api.transports import HttpIngestTransport
from minimappr.classifiers.factory import create_classifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import AlertStatus, GeoPoint, IngestFrameRequest, IngestFrameResponse, ZoneSpec
from minimappr.storage.db import Storage


settings = Settings.from_env()
storage = Storage(settings.db_path)
registry = NodeRegistry()
audio_buffer = MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds)
localizer = LocalizationEngine()
classifier = create_classifier(settings)
tracker = TrackManager(settings)
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
)
ingest_transport = HttpIngestTransport(fusion_node)

cleanup_task: asyncio.Task | None = None


async def _cleanup_loop() -> None:
    while True:
        now_ns = time.time_ns()
        await storage.cleanup_expired_snippets(now_ns=now_ns)
        await storage.cleanup_retention(
            now_ns=now_ns,
            tier_ttls_seconds={
                "ephemeral": settings.retention_ephemeral_seconds,
                "short": settings.retention_short_seconds,
                "long": settings.retention_long_seconds,
                "experiment": settings.retention_experiment_seconds,
            },
        )
        await fusion_node.housekeeping_tick(now_ns=now_ns)
        await asyncio.sleep(settings.cleanup_interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cleanup_task
    await storage.initialize()
    await fusion_node.start()
    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        if cleanup_task is not None:
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
        await fusion_node.stop()
        await storage.close()


app = FastAPI(title="MinimapPR", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
async def health() -> dict:
    status = await fusion_node.status()
    workers = status.get("workers", {})
    running = int(workers.get("localization_running", 0)) + int(workers.get("classification_running", 0)) + int(
        workers.get("rules_running", 0)
    )
    return {
        "status": "ok",
        "time_ns": time.time_ns(),
        "classifier": settings.classifier_backend,
        "fusion_queue_depth": status["queue"]["localization_depth"],
        "fusion_workers_running": running,
    }


@app.post("/api/v1/ingest/frame", response_model=IngestFrameResponse)
async def ingest_frame(payload: IngestFrameRequest) -> IngestFrameResponse:
    try:
        return await ingest_transport.deliver_frame(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/nodes")
async def list_nodes() -> list[dict]:
    now_ns = time.time_ns()
    nodes = await storage.list_nodes()
    for node in nodes:
        if node.get("position_geo") is None and node.get("position_m"):
            local = node["position_m"]
            geo = coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
            node["position_geo"] = geo.model_dump(mode="json")

        age_s = max(0.0, (now_ns - int(node["last_seen_ns"])) / 1_000_000_000.0)
        if age_s >= settings.node_offline_after_seconds:
            health = "offline"
        elif age_s >= settings.node_degraded_after_seconds:
            health = "degraded"
        else:
            health = "online"
        node["health_status"] = health
    return nodes


@app.get("/api/v1/detections")
async def list_detections(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict]:
    detections = await storage.list_detections(limit=limit)
    for detection in detections:
        if detection.get("position_geo") is None and detection.get("position_m"):
            local = detection["position_m"]
            geo = coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
            detection["position_geo"] = geo.model_dump(mode="json")
    return detections


@app.get("/api/v1/tracks")
async def list_tracks(limit: int = Query(default=200, ge=1, le=1000)) -> list[dict]:
    now_ns = time.time_ns()
    _ = await tracker.snapshot(now_ns=now_ns)
    tracks = await storage.list_tracks(limit=limit)
    for track in tracks:
        if track.get("position_geo") is None and track.get("position_m"):
            local = track["position_m"]
            geo = coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
            track["position_geo"] = geo.model_dump(mode="json")
    return tracks


@app.get("/api/v1/config")
async def get_config() -> dict:
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
        },
        "default_temperature_c": settings.default_temperature_c,
        "default_humidity": settings.default_humidity,
        "preprocess_enabled": settings.preprocess_enabled,
        "audio_highpass_hz": settings.audio_highpass_hz,
        "audio_lowpass_hz": settings.audio_lowpass_hz,
        "min_sensors_for_3d": settings.min_sensors_for_3d,
        "min_sensors_for_2d": settings.min_sensors_for_2d,
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
    }


@app.get("/api/v1/fusion/status")
async def fusion_status() -> dict:
    return await fusion_node.status()


@app.get("/api/v1/cop/status")
async def cop_status() -> dict:
    now_ns = time.time_ns()
    nodes = await list_nodes()
    tracks = await list_tracks(limit=500)
    recent_window_ns = now_ns - 300_000_000_000
    recent_alert_count = await storage.recent_alert_count(since_ns=recent_window_ns)
    return {
        "active_nodes": sum(1 for node in nodes if node["health_status"] == "online"),
        "degraded_nodes": sum(1 for node in nodes if node["health_status"] == "degraded"),
        "offline_nodes": sum(1 for node in nodes if node["health_status"] == "offline"),
        "active_tracks": sum(1 for track in tracks if track["status"] in {"tentative", "confirmed", "coasting"}),
        "recent_alert_count": recent_alert_count,
    }


@app.get("/api/v1/alerts")
async def list_alerts(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict]:
    return await storage.list_alerts(limit=limit)


@app.patch("/api/v1/alerts/{alert_id}")
async def update_alert_status(
    alert_id: str,
    status: AlertStatus,
    reason: str | None = Query(default=None),
) -> dict:
    ok = await storage.update_alert_status(
        alert_id=alert_id,
        status=status.value,
        updated_ns=time.time_ns(),
        payload_patch={"operator_reason": reason} if reason else None,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"ok": True, "alert_id": alert_id, "status": status.value}


@app.get("/api/v1/pings")
async def list_pings(limit: int = Query(default=500, ge=1, le=5000)) -> list[dict]:
    return await storage.list_pings(limit=limit)


@app.get("/api/v1/zones")
async def list_zones() -> list[dict]:
    return await storage.list_zones()


@app.put("/api/v1/zones/{zone_id}")
async def upsert_zone(zone_id: str, payload: ZoneSpec) -> dict:
    if payload.id != zone_id:
        raise HTTPException(status_code=400, detail="zone_id path must match payload.id")
    await storage.upsert_zone(
        zone_id=payload.id,
        name=payload.name,
        zone_type=payload.zone_type.value,
        polygon_geo=payload.polygon_geo,
        properties=payload.properties,
        created_ns=time.time_ns(),
    )
    return {"ok": True, "zone_id": zone_id}


@app.delete("/api/v1/zones/{zone_id}")
async def delete_zone(zone_id: str) -> dict:
    deleted = await storage.delete_zone(zone_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Zone not found")
    return {"ok": True, "zone_id": zone_id}


@app.get("/api/v1/detections/{detection_id}/audio")
async def get_detection_audio(detection_id: str) -> FileResponse:
    snippet_path = await storage.snippet_path_for_detection(detection_id)
    if not snippet_path:
        raise HTTPException(status_code=404, detail="Snippet not found for detection")

    snippet_file = Path(snippet_path).resolve()
    if not snippet_file.exists():
        raise HTTPException(status_code=404, detail="Snippet file no longer exists")

    snippet_root = settings.snippet_dir.resolve()
    if snippet_root not in snippet_file.parents:
        raise HTTPException(status_code=403, detail="Snippet path is outside snippet directory")

    return FileResponse(
        path=snippet_file,
        media_type="audio/wav",
        filename=f"{detection_id}.wav",
    )


@app.websocket("/ws/live")
async def live_events(websocket: WebSocket) -> None:
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
