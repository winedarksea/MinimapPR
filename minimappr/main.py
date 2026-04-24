"""MinimapPR application entrypoint."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from minimappr.api.live import LiveEventHub
from minimappr.api.transports import HttpIngestTransport
from minimappr.classifiers.factory import create_classifier
from minimappr.cleanup_service import CleanupService
from minimappr.config import Settings
from minimappr.core.ambisonics import (
    AmbisonicSpatialEncoder,
    SoundscapeRenderer,
    SpatialSourceFrame,
    foa_to_5_1,
    wav_multichannel_bytes,
)
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.auth import extract_federation_token
from minimappr.core.bit_report import BITReportEvaluator
from minimappr.core.diagnostics import DiagnosticsService
from minimappr.core.environment import LiveEnvironmentProvider
from minimappr.core.federation import FederationCoordinator
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization_dispatch import build_localizer_from_settings
from minimappr.core.logging_ring import install_global as install_log_ring, process_start_ns
from minimappr.core.site_origin import (
    resolve_site_origin_from_nodes,
    should_schedule_deferred_site_origin_reconciliation,
)
from minimappr.core import system_info
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import (
    AlertStatus,
    BITReport,
    BITReportIn,
    BITType,
    ContextSnapshot,
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
    ZoneOccupancyState,
    ZoneSpec,
)
from minimappr.storage.db import Storage
from minimappr.utils.audio import mono_mix, read_wav_mono


logger = logging.getLogger(__name__)
frontend_dir = Path(__file__).parent / "frontend"


def _parse_window_ns(window: str) -> int:
    """Parse a compact window string (e.g. '24h', '7d', '30m', '1y') into nanoseconds."""
    if not window:
        raise HTTPException(status_code=400, detail="Empty window")
    unit = window[-1].lower()
    if unit.isdigit():
        # Bare seconds
        try:
            return int(window) * 1_000_000_000
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid window '{window}'") from exc
    try:
        amount = int(window[:-1])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid window '{window}'") from exc
    multipliers = {
        "s": 1_000_000_000,
        "m": 60 * 1_000_000_000,
        "h": 3600 * 1_000_000_000,
        "d": 86400 * 1_000_000_000,
        "w": 7 * 86400 * 1_000_000_000,
        "y": 365 * 86400 * 1_000_000_000,
    }
    if unit not in multipliers:
        raise HTTPException(status_code=400, detail=f"Unknown window unit '{unit}'")
    return amount * multipliers[unit]


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
        cleanup_summary = await state.cleanup_service.run_housekeeping_cycle(now_ns=now_ns)
        if any(cleanup_summary["partial_cleanup"].values()) or any(cleanup_summary["retention_cleanup"].values()):
            logger.info("Cleanup cycle removed data: %s", cleanup_summary)
        await state.fusion_node.housekeeping_tick(now_ns=now_ns)
        await asyncio.sleep(settings.cleanup_interval_seconds)


def _apply_site_origin_resolution(state, resolved_site_origin) -> None:
    settings: Settings = state.settings
    settings.site_origin_lat = resolved_site_origin.origin.lat
    settings.site_origin_lon = resolved_site_origin.origin.lon
    settings.site_origin_alt_m = resolved_site_origin.origin.alt_m
    state.site_origin_resolution_source = resolved_site_origin.source
    state.site_origin_contributing_node_ids = resolved_site_origin.contributing_node_ids


async def _reconcile_site_origin_after_startup(app: FastAPI) -> None:
    state = app.state
    settings: Settings = state.settings
    delay_seconds = settings.site_origin_reconcile_delay_seconds
    if delay_seconds <= 0.0:
        return

    await asyncio.sleep(delay_seconds)

    if state.fusion_node.accepted_frame_count > 0:
        logger.info(
            "Skipping delayed site-origin reconciliation after %.1fs because ingest already accepted %d frames",
            delay_seconds,
            state.fusion_node.accepted_frame_count,
        )
        return

    resolved_site_origin = resolve_site_origin_from_nodes(
        settings,
        nodes=await state.storage.list_nodes(limit=4096),
        now_ns=time.time_ns(),
    )
    current_origin = (
        settings.site_origin_lat,
        settings.site_origin_lon,
        settings.site_origin_alt_m,
        getattr(state, "site_origin_resolution_source", settings.site_origin_source),
    )
    next_origin = (
        resolved_site_origin.origin.lat,
        resolved_site_origin.origin.lon,
        resolved_site_origin.origin.alt_m,
        resolved_site_origin.source,
    )
    if next_origin == current_origin:
        logger.info("Delayed site-origin reconciliation after %.1fs found no change", delay_seconds)
        return

    candidate_settings = replace(
        settings,
        site_origin_lat=resolved_site_origin.origin.lat,
        site_origin_lon=resolved_site_origin.origin.lon,
        site_origin_alt_m=resolved_site_origin.origin.alt_m,
    )
    new_classifier = create_classifier(candidate_settings)
    new_coordinate_frame = LocalCoordinateFrame(
        origin=resolved_site_origin.origin,
        mode=settings.coordinate_mode,
    )
    previous_classifier = state.classifier
    state.fusion_node.rebind_runtime_dependencies(
        classifier=new_classifier,
        coordinate_frame=new_coordinate_frame,
    )
    state.classifier = new_classifier
    state.coordinate_frame = new_coordinate_frame
    state.diagnostics.replace_classifier(new_classifier)
    _apply_site_origin_resolution(state, resolved_site_origin)
    logger.info(
        "Reconciled site origin after %.1fs via %s: lat=%.6f lon=%.6f alt=%.2f nodes=%s",
        delay_seconds,
        resolved_site_origin.source,
        resolved_site_origin.origin.lat,
        resolved_site_origin.origin.lon,
        resolved_site_origin.origin.alt_m,
        list(resolved_site_origin.contributing_node_ids),
    )
    try:
        previous_classifier.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Previous classifier close failed after site-origin reconciliation: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    install_log_ring()
    settings = Settings.from_env()
    settings.federation_peers_config_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    settings.large_artifact_dir.mkdir(parents=True, exist_ok=True)

    storage_cfg = settings.storage_config()
    localization_cfg = settings.localization_config()
    tracking_cfg = settings.tracking_config()

    storage = Storage(storage_cfg.db_path)
    await storage.initialize()
    resolved_site_origin = resolve_site_origin_from_nodes(
        settings,
        nodes=await storage.list_nodes(limit=4096),
        now_ns=time.time_ns(),
    )
    settings.site_origin_lat = resolved_site_origin.origin.lat
    settings.site_origin_lon = resolved_site_origin.origin.lon
    settings.site_origin_alt_m = resolved_site_origin.origin.alt_m
    logger.info(
        "Resolved site origin via %s: lat=%.6f lon=%.6f alt=%.2f nodes=%s",
        resolved_site_origin.source,
        settings.site_origin_lat,
        settings.site_origin_lon,
        settings.site_origin_alt_m,
        list(resolved_site_origin.contributing_node_ids),
    )

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
    bit_evaluator = BITReportEvaluator()
    diagnostics = DiagnosticsService(
        settings=settings,
        storage=storage,
        fusion_node=fusion_node,
        classifier=classifier,
    )
    cleanup_service = CleanupService(settings=settings, storage=storage)

    async def _federation_local_tracks(now_ns: int) -> list[TrackState]:
        tracks = await tracker.snapshot(now_ns=now_ns)
        active: list[TrackState] = []
        for track in tracks:
            if track.status not in {"tentative", "confirmed", "coasting"}:
                continue
            track.position_geo = app.state.coordinate_frame.local_to_geo(track.position_m)
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
    app.state.bit_evaluator = bit_evaluator
    app.state.diagnostics = diagnostics
    app.state.cleanup_service = cleanup_service
    _apply_site_origin_resolution(app.state, resolved_site_origin)

    cleanup_task: asyncio.Task | None = None
    site_origin_reconcile_task: asyncio.Task | None = None
    environment_provider.bootstrap(await storage.list_latest_environment_per_node(limit=1024))
    await fusion_node.start()
    await federation.start()
    cleanup_task = asyncio.create_task(_cleanup_loop(app))
    app.state.cleanup_task = cleanup_task
    if should_schedule_deferred_site_origin_reconciliation(
        settings,
        initial_resolution_source=resolved_site_origin.source,
    ):
        site_origin_reconcile_task = asyncio.create_task(_reconcile_site_origin_after_startup(app))
    app.state.site_origin_reconcile_task = site_origin_reconcile_task

    try:
        yield
    finally:
        shutdown_timeout_s = 5.0
        if site_origin_reconcile_task is not None:
            site_origin_reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(site_origin_reconcile_task, timeout=shutdown_timeout_s)
        if cleanup_task is not None:
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(cleanup_task, timeout=shutdown_timeout_s)

        try:
            await asyncio.wait_for(federation.stop(), timeout=shutdown_timeout_s)
        except Exception as exc:
            logger.warning("Federation stop failed during shutdown: %s", exc)

        # Cancel any in-flight BirdNET predictions and terminate their worker
        # subprocesses BEFORE stopping the fusion node.  If a SIGINT kills BirdNET
        # workers mid-init, the Consumer blocks on queue.get() indefinitely; closing
        # the classifier first lets the thread unblock and the queue to drain.
        try:
            classifier.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Classifier close failed during shutdown: %s", exc)

        try:
            await asyncio.wait_for(fusion_node.stop(), timeout=shutdown_timeout_s)
        except Exception as exc:
            logger.warning("Fusion node stop failed during shutdown: %s", exc)

        try:
            await asyncio.wait_for(storage.close(), timeout=shutdown_timeout_s)
        except Exception as exc:
            logger.warning("Storage close failed during shutdown: %s", exc)


app = FastAPI(title="MinimapPR", version="0.1.0", lifespan=lifespan)


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


@app.exception_handler(StarletteHTTPException)
async def spa_fallback_404_handler(request: Request, exc: StarletteHTTPException):
    """Serve the SPA entrypoint for non-API browser navigations.

    StaticFiles does not automatically map arbitrary history-routed paths
    (for example "/analysis/labels") to index.html, so refresh on nested
    routes can otherwise return FastAPI's JSON 404 response.
    """
    if exc.status_code != 404:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    index = frontend_dir / "index.html"
    request_path = request.url.path
    is_frontend_navigation = (
        request.method == "GET"
        and index.is_file()
        and not request_path.startswith("/api/")
        and not request_path.startswith("/ws/")
        and "." not in Path(request_path).name
    )
    if is_frontend_navigation:
        return FileResponse(index)

    return JSONResponse(status_code=404, content={"detail": exc.detail})


@app.get("/", response_model=None)
async def root() -> Response:
    index = frontend_dir / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse({"status": "ok", "message": "MinimapPR API is running. No frontend installed."})


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
    bit_evaluator: BITReportEvaluator = state.bit_evaluator
    now_ns = time.time_ns()
    nodes = await state.storage.list_nodes(limit=limit)
    latest_environment_rows = await state.storage.list_latest_environment_per_node(limit=limit)
    latest_environment_by_node = {
        row["node_id"]: row for row in latest_environment_rows if row.get("node_id") is not None
    }
    for node in nodes:
        if node.get("position_geo") is None and node.get("position_m"):
            local = node["position_m"]
            geo = state.coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
            node["position_geo"] = geo.model_dump(mode="json")

        age_s = max(0.0, (now_ns - int(node["last_seen_ns"])) / 1_000_000_000.0)
        if age_s >= settings.node_offline_after_seconds:
            heartbeat_health = "offline"
        elif age_s >= settings.node_degraded_after_seconds:
            heartbeat_health = "degraded"
        else:
            heartbeat_health = "online"

        # Merge BIT status with heartbeat staleness
        node["health_status"] = await bit_evaluator.derive_health_status(
            node_id=node["id"],
            heartbeat_health=heartbeat_health,
            now_ns=now_ns,
        )

        # Attach latest BIT failure codes for UI consumption
        bit_reports = await bit_evaluator.latest_reports_for_node(node["id"])
        failure_codes: list[str] = []
        for report in bit_reports:
            failure_codes.extend(report.failure_codes)
        if failure_codes:
            node["bit_failure_codes"] = failure_codes

        sensor_descriptors = await state.registry.sensors_for_node(node["id"])
        sensor_ids = [descriptor.sensor_id for descriptor in sensor_descriptors]
        audio_summary = await state.audio_buffer.summarize_sensors(sensor_ids=sensor_ids, now_ns=now_ns)

        age_seconds = audio_summary["age_seconds"]
        if age_seconds is None:
            audio_status = "no_audio"
        elif age_seconds <= settings.node_degraded_after_seconds:
            audio_status = "recent"
        else:
            audio_status = "stale"

        node["audio_debug"] = {
            "sensor_count": len(sensor_ids),
            "active_sensor_count": int(audio_summary["active_sensor_count"] or 0),
            "sample_rate_hz": audio_summary["sample_rate_hz"],
            "last_sample_time_ns": audio_summary["last_sample_time_ns"],
            "age_seconds": age_seconds,
            "rms": audio_summary["rms"],
            "status": audio_status,
        }

        latest_environment = latest_environment_by_node.get(node["id"])
        if latest_environment is not None:
            node["latest_environment"] = latest_environment
        # Yield after each node so pipeline lock holders (NodeRegistry, MultiSensorBuffer)
        # get a chance to run between per-node awaits.
        await asyncio.sleep(0)
    return nodes


# ------------------------------------------------------------------
# BIT (Built-In Test) Endpoints
# ------------------------------------------------------------------


@app.post("/api/v1/nodes/{node_id}/bit", response_model=BITReport)
async def submit_bit_report(
    node_id: str,
    report_in: BITReportIn,
    request: Request,
) -> BITReport:
    """Accept a BIT report from a sensor node.

    Evaluates the overall pass/fail status, persists to storage, and
    caches in the in-memory evaluator for real-time health derivation.
    """
    state = _require_state(request)
    bit_evaluator: BITReportEvaluator = state.bit_evaluator
    now_ns = time.time_ns()

    report = await bit_evaluator.submit_report(
        node_id=node_id,
        report_in=report_in,
        received_ns=now_ns,
    )

    # Persist to durable storage
    await state.storage.insert_bit_report(
        report_id=report.id,
        node_id=report.node_id,
        report_type=report.report_type.value,
        overall_status=report.overall_status.value,
        timestamp_ns=report.timestamp_ns,
        received_ns=report.received_ns,
        results_json=json.dumps([r.model_dump(mode="json") for r in report.results]),
        failure_codes_json=json.dumps(report.failure_codes),
        firmware_version=report.firmware_version,
        uptime_seconds=report.uptime_seconds,
        metadata_json=json.dumps(report.metadata),
    )

    # Broadcast BIT report via WebSocket for live dashboard
    await state.live_hub.broadcast(
        {
            "type": "bit_report",
            "node_id": node_id,
            "report_type": report.report_type.value,
            "overall_status": report.overall_status.value,
            "failure_codes": report.failure_codes,
            "timestamp_ns": report.timestamp_ns,
        }
    )

    return report


@app.get("/api/v1/nodes/{node_id}/bit", response_model=list[BITReport])
async def get_node_bit_reports(
    node_id: str,
    request: Request,
    report_type: BITType | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict]:
    """Retrieve BIT report history for a specific node."""
    state = _require_state(request)
    return await state.storage.list_bit_reports(
        node_id=node_id,
        report_type=report_type.value if report_type else None,
        limit=limit,
    )


@app.get("/api/v1/nodes/{node_id}/bit/latest")
async def get_node_latest_bit(
    node_id: str,
    request: Request,
) -> list[dict]:
    """Return the latest BIT report per type (PBIT/CBIT/IBIT) for a node."""
    state = _require_state(request)
    return await state.storage.latest_bit_report_per_type(node_id)


@app.get("/api/v1/bit/failures")
async def list_bit_failures(request: Request) -> dict[str, list[str]]:
    """Return all nodes with active BIT failures and their failure codes."""
    state = _require_state(request)
    bit_evaluator: BITReportEvaluator = state.bit_evaluator
    return await bit_evaluator.all_nodes_with_bit_failures()


@app.get("/api/v1/detections")
async def list_detections(request: Request, limit: int = Query(default=100, ge=1, le=1000)) -> list[dict]:
    state = _require_state(request)
    settings: Settings = state.settings
    effective_limit = min(limit, settings.cop_detections_max_items)
    cutoff_ns = time.time_ns() - int(settings.cop_detections_max_age_seconds * 1_000_000_000)
    detections = await state.storage.list_detections(
        limit=effective_limit,
        since_ns=cutoff_ns,
        min_label_confidence=settings.detection_min_confidence,
    )
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
    effective_limit = min(limit, state.settings.cop_tracks_max_items)
    cutoff_ns = now_ns - int(state.settings.cop_tracks_max_age_seconds * 1_000_000_000)
    tracks = await state.storage.list_tracks(limit=effective_limit, since_ns=cutoff_ns)
    for track in tracks:
        if track.get("position_geo") is None and track.get("position_m"):
            local = track["position_m"]
            geo = state.coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
            track["position_geo"] = geo.model_dump(mode="json")
    if state.federation.enabled:
        tracks = await state.federation.merged_tracks(
            local_tracks=tracks,
            now_ns=now_ns,
            limit=effective_limit,
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
    site_origin_resolution_source = getattr(state, "site_origin_resolution_source", settings.site_origin_source)
    return {
        "trigger_rms": settings.trigger_rms,
        "trigger_cooldown_seconds": settings.trigger_cooldown_seconds,
        "localization_window_seconds": settings.localization_window_seconds,
        "snippet_retention_seconds": settings.snippet_retention_seconds,
        "retention_policy_path": str(settings.retention_policy_path),
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
        "classifier_backend": settings.classifier_backend,
        "yamnet_min_confidence": settings.yamnet_min_confidence,
        "detection_min_confidence": settings.detection_min_confidence,
        "cop": {
            "detections_max_items": settings.cop_detections_max_items,
            "tracks_max_items": settings.cop_tracks_max_items,
            "detections_max_age_seconds": settings.cop_detections_max_age_seconds,
            "tracks_max_age_seconds": settings.cop_tracks_max_age_seconds,
        },
        "yamnet_input_target_rms": settings.yamnet_input_target_rms,
        "yamnet_max_input_gain": settings.yamnet_max_input_gain,
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
            "reconcile_delay_seconds": settings.site_origin_reconcile_delay_seconds,
            "mode": settings.site_origin_source,
            "source": site_origin_resolution_source,
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


_CONFIG_PATCH_ALLOWLIST: dict[str, type] = {
    "trigger_rms": float,
    "trigger_cooldown_seconds": float,
    "localization_window_seconds": float,
    "preprocess_enabled": bool,
    "audio_highpass_hz": float,
    "audio_lowpass_hz": float,
    "localization_algorithm": str,
    "localization_strategy": str,
    "classifier_backend": str,
    "yamnet_min_confidence": float,
    "detection_min_confidence": float,
    "cop_detections_max_items": int,
    "cop_tracks_max_items": int,
    "cop_detections_max_age_seconds": float,
    "cop_tracks_max_age_seconds": float,
    "beamformer_type": str,
    "tracking_filter": str,
    "fusion_worker_count": int,
    "coordinate_mode": str,
}

_LOCALIZATION_ALGORITHMS = {"gcc_phat", "srp_phat", "music", "esprit"}
_LOCALIZATION_STRATEGIES = {"fixed", "geometry_aware", "cascade"}
_BEAMFORMER_TYPES = {"delay_and_sum", "das", "freq_domain_das", "mvdr", "superdirective", "gevd"}
_CLASSIFIER_BACKENDS = {"yamnet", "birdnet", "heuristic"}
_TRACKING_FILTERS = {"linear", "kalman"}
_COORDINATE_MODES = {"flat", "geodetic"}


@app.patch("/api/v1/config")
async def patch_config(request: Request) -> dict:
    state = _require_state(request)
    body: dict = await request.json()

    unknown = set(body) - set(_CONFIG_PATCH_ALLOWLIST)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown or read-only fields: {sorted(unknown)}")

    errors: list[str] = []
    coerced: dict[str, object] = {}

    for key, raw in body.items():
        target_type = _CONFIG_PATCH_ALLOWLIST[key]
        try:
            if target_type is bool:
                if not isinstance(raw, bool):
                    raise ValueError("must be a boolean")
                value: object = raw
            elif target_type is float:
                value = float(raw)
            elif target_type is int:
                value = int(raw)
            else:
                value = str(raw)
        except (TypeError, ValueError) as exc:
            errors.append(f"{key}: {exc}")
            continue

        if key == "trigger_rms" and value <= 0.0:  # type: ignore[operator]
            errors.append("trigger_rms: must be > 0")
        elif key == "trigger_cooldown_seconds" and value < 0.0:  # type: ignore[operator]
            errors.append("trigger_cooldown_seconds: must be >= 0")
        elif key == "localization_window_seconds" and value <= 0.0:  # type: ignore[operator]
            errors.append("localization_window_seconds: must be > 0")
        elif key == "audio_highpass_hz" and value < 0.0:  # type: ignore[operator]
            errors.append("audio_highpass_hz: must be >= 0")
        elif key == "audio_lowpass_hz" and value < 0.0:  # type: ignore[operator]
            errors.append("audio_lowpass_hz: must be >= 0")
        elif key == "yamnet_min_confidence" and not (0.0 <= value <= 1.0):  # type: ignore[operator]
            errors.append("yamnet_min_confidence: must be in [0, 1]")
        elif key == "detection_min_confidence" and not (0.0 <= value <= 1.0):  # type: ignore[operator]
            errors.append("detection_min_confidence: must be in [0, 1]")
        elif key in {"cop_detections_max_items", "cop_tracks_max_items"} and value < 1:  # type: ignore[operator]
            errors.append(f"{key}: must be >= 1")
        elif key in {"cop_detections_max_age_seconds", "cop_tracks_max_age_seconds"} and value <= 0.0:  # type: ignore[operator]
            errors.append(f"{key}: must be > 0")
        elif key == "fusion_worker_count" and value < 1:  # type: ignore[operator]
            errors.append("fusion_worker_count: must be >= 1")
        elif key == "localization_algorithm" and value not in _LOCALIZATION_ALGORITHMS:
            errors.append(f"localization_algorithm: must be one of {sorted(_LOCALIZATION_ALGORITHMS)}")
        elif key == "localization_strategy" and value not in _LOCALIZATION_STRATEGIES:
            errors.append(f"localization_strategy: must be one of {sorted(_LOCALIZATION_STRATEGIES)}")
        elif key == "beamformer_type":
            v = str(value).strip().lower()
            if v == "das":
                v = "delay_and_sum"
            if v not in _BEAMFORMER_TYPES:
                errors.append(f"beamformer_type: must be one of {sorted(_BEAMFORMER_TYPES)}")
            else:
                value = v
        elif key == "classifier_backend" and value not in _CLASSIFIER_BACKENDS:
            errors.append(f"classifier_backend: must be one of {sorted(_CLASSIFIER_BACKENDS)}")
        elif key == "tracking_filter" and value not in _TRACKING_FILTERS:
            errors.append(f"tracking_filter: must be one of {sorted(_TRACKING_FILTERS)}")
        elif key == "coordinate_mode":
            v = str(value).strip().lower()
            if v not in _COORDINATE_MODES:
                errors.append(f"coordinate_mode: must be one of {sorted(_COORDINATE_MODES)}")
            else:
                value = v

        if not errors or key not in {e.split(":")[0] for e in errors}:
            coerced[key] = value

    if errors:
        raise HTTPException(status_code=422, detail=errors)

    settings: Settings = state.settings
    for key, value in coerced.items():
        object.__setattr__(settings, key, value)

    snapshot = await get_config(request)
    await state.live_hub.broadcast({"type": "config_updated", "config": snapshot})
    return snapshot


@app.get("/api/v1/fusion/status", response_model=FusionStatusResponse)
async def fusion_status(request: Request) -> dict:
    state = _require_state(request)
    return await state.fusion_node.status()


@app.get("/api/v1/debug/config")
async def debug_config(request: Request) -> dict:
    state = _require_state(request)
    return await state.diagnostics.config_snapshot()


@app.get("/api/v1/debug/event/{event_id}")
async def debug_event_snapshot(event_id: str, request: Request) -> dict:
    state = _require_state(request)
    snapshot = await state.diagnostics.event_snapshot(event_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return snapshot


@app.get("/api/v1/debug/selftest")
async def debug_selftest(request: Request) -> dict:
    state = _require_state(request)
    return await state.diagnostics.selftest()


@app.get("/api/v1/federation/status", response_model=FederationStatusResponse)
async def federation_status(request: Request) -> dict:
    state = _require_state(request)
    return await state.federation.status()


@app.post("/api/v1/federation/heartbeat", response_model=FederationAck)
async def federation_heartbeat(payload: FederationHeartbeat, request: Request) -> FederationAck:
    state = _require_state(request)
    settings: Settings = state.settings
    if not state.federation.enabled:
        raise HTTPException(status_code=503, detail="Federation is disabled")
    
    # Extract headers for validation by the coordinator
    auth_header = request.headers.get("authorization")
    token_header = request.headers.get("x-minimappr-token")
    
    if not await state.federation.validate_inbound_auth(
        peer_id=payload.server_id,
        authorization_header=auth_header,
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

    # Extract headers for validation by the coordinator
    auth_header = request.headers.get("authorization")
    token_header = request.headers.get("x-minimappr-token")

    if not await state.federation.validate_inbound_auth(
        peer_id=payload.server_id,
        authorization_header=auth_header,
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


@app.get("/api/v1/analytics/daily")
async def get_analytics_daily(
    request: Request,
    end: str | None = Query(default=None, description="ISO 8601 end timestamp (rolling mode)"),
    date: str | None = Query(default=None, description="YYYY-MM-DD local calendar date"),
    hours: int = Query(default=24, ge=1, le=168),
    tz: str = Query(default="UTC", description="IANA tz for bucket edges and 'calendar' mode"),
    max_labels: int = Query(default=64, ge=1, le=256),
) -> dict:
    """Daily detection activity matrix.

    Two modes:
    - rolling (default): `[end-hours, end]` snapped to hour boundaries;
      `end` defaults to now.
    - calendar: `date=YYYY-MM-DD` → 00:00 to 24:00 local on that date.

    Response cache-control: 30s (set by the client via frontend `fetch`).
    """
    from datetime import datetime, timedelta, timezone as _tz
    try:
        from zoneinfo import ZoneInfo
        tz_info = ZoneInfo(tz)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid tz '{tz}': {exc}") from exc

    state = _require_state(request)
    settings: Settings = state.settings

    bucket_ns = 3600 * 1_000_000_000
    mode: Literal["rolling", "calendar"]
    if date is not None:
        mode = "calendar"
        try:
            y, m, d = (int(p) for p in date.split("-"))
            start_local = datetime(y, m, d, tzinfo=tz_info)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid date '{date}': {exc}") from exc
        end_local = start_local + timedelta(hours=24)
        num_buckets = 24
    else:
        mode = "rolling"
        now_utc = datetime.now(tz=_tz.utc)
        if end is None:
            end_local = now_utc.astimezone(tz_info)
        else:
            try:
                parsed = datetime.fromisoformat(end.replace("Z", "+00:00"))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid end '{end}': {exc}") from exc
            end_local = parsed.astimezone(tz_info) if parsed.tzinfo else parsed.replace(tzinfo=tz_info)
        # Snap end up to the next hour boundary (rolling window ends at "current hour").
        end_local = end_local.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        num_buckets = hours
        start_local = end_local - timedelta(hours=num_buckets)

    start_ns = int(start_local.astimezone(_tz.utc).timestamp() * 1e9)
    end_ns = int(end_local.astimezone(_tz.utc).timestamp() * 1e9)

    labels, counts, bucket_totals = await state.storage.detection_matrix(
        start_ns=start_ns,
        end_ns=end_ns,
        bucket_ns=bucket_ns,
        min_label_confidence=settings.detection_min_confidence,
        max_labels=max_labels,
    )

    # Back-fill empty matrix shape when there's nothing to show, so the client
    # still gets a 24-wide row of zeros rather than length-0 arrays.
    if not labels:
        counts = []
        bucket_totals = [0] * num_buckets

    # Generate bucket_starts (ISO in the requested tz for display).
    bucket_starts = [
        (start_local + timedelta(hours=i)).isoformat(timespec="seconds")
        for i in range(num_buckets)
    ]
    label_totals = [sum(row) for row in counts]

    return {
        "mode": mode,
        "tz": tz,
        "hours": num_buckets,
        "start": start_local.isoformat(timespec="seconds"),
        "end": end_local.isoformat(timespec="seconds"),
        "bucket_starts": bucket_starts,
        "labels": labels,
        "counts": counts,
        "label_totals": label_totals,
        "bucket_totals": bucket_totals,
    }


@app.get("/api/v1/analytics/labels")
async def get_analytics_labels(
    request: Request,
    window: str = Query(default="30d", description="Window: e.g. 1h, 24h, 7d, 30d, 365d"),
) -> dict:
    """List of labels seen within `window`, with counts + first/last-seen."""
    state = _require_state(request)
    settings: Settings = state.settings
    window_ns = _parse_window_ns(window)
    now_ns = time.time_ns()
    rows = await state.storage.label_summaries(
        now_ns=now_ns,
        window_ns=window_ns,
        min_label_confidence=settings.detection_min_confidence,
    )
    return {"window": window, "now_ns": now_ns, "labels": rows}


@app.get("/api/v1/analytics/labels/{label}")
async def get_analytics_label_detail(
    request: Request,
    label: str,
    window: str = Query(default="30d"),
    recent_limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    """Per-label aggregates: hour-of-day, day-of-week, month, recent detections."""
    state = _require_state(request)
    settings: Settings = state.settings
    window_ns = _parse_window_ns(window)
    now_ns = time.time_ns()
    return await state.storage.label_detail(
        label,
        now_ns=now_ns,
        window_ns=window_ns,
        recent_limit=recent_limit,
        min_label_confidence=settings.detection_min_confidence,
    )


@app.get("/api/v1/analytics/heatmap")
async def get_analytics_heatmap(
    request: Request,
    window: str = Query(default="24h"),
    labels: str | None = Query(default=None, description="comma-separated label list"),
    bin: float = Query(default=0.0001, ge=0.00001, le=0.1, description="lat/lon rounding resolution in deg"),
    max_bins: int = Query(default=5000, ge=1, le=20000),
) -> dict:
    """Pre-binned geo density for the heatmap view — `[lat, lon, weight]` triplets."""
    state = _require_state(request)
    settings: Settings = state.settings
    window_ns = _parse_window_ns(window)
    now_ns = time.time_ns()
    label_list = [s.strip() for s in labels.split(",") if s.strip()] if labels else None
    bins = await state.storage.heatmap_bins(
        now_ns=now_ns,
        window_ns=window_ns,
        bin_deg=float(bin),
        labels=label_list,
        min_label_confidence=settings.detection_min_confidence,
        max_bins=max_bins,
    )
    return {"window": window, "bin_deg": float(bin), "now_ns": now_ns, "bins": bins}


@app.get("/api/v1/system/diagnostics")
async def get_system_diagnostics(request: Request) -> dict:
    """Runtime facts (uptime, CPU, memory, disk, load) for the Server page."""
    state = _require_state(request)
    settings: Settings = state.settings
    diagnostics = system_info.collect(db_path=settings.db_path, start_ns=process_start_ns())
    fusion_status = await state.fusion_node.status()
    diagnostics["pipeline"] = {
        "queue": fusion_status["queue"],
        "workers": fusion_status["workers"],
        "realtime": fusion_status["realtime"],
        "drop_on_backpressure": fusion_status["drop_on_backpressure"],
        "metrics": {
            "triggers_enqueued": fusion_status["metrics"].get("triggers_enqueued", 0),
            "triggers_dropped_queue_full": fusion_status["metrics"].get("triggers_dropped_queue_full", 0),
            "stage_drops_backpressure": fusion_status["metrics"].get("stage_drops_backpressure", 0),
            "classification_reuse_hits": fusion_status["metrics"].get("classification_reuse_hits", 0),
            "birdnet_chunk_dispatches_suppressed": (
                fusion_status["metrics"].get("birdnet_chunk_dispatches_suppressed", 0)
            ),
            "detections_emitted": fusion_status["metrics"].get("detections_emitted", 0),
        },
    }
    return diagnostics


@app.get("/api/v1/system/logs")
async def get_system_logs(
    request: Request,
    limit: int = Query(default=500, ge=1, le=5000),
    level: str = Query(default="INFO", description="Min level: DEBUG/INFO/WARNING/ERROR/CRITICAL"),
    logger_prefix: str | None = Query(default=None),
    since_seq: int | None = Query(default=None, description="Tail: records with seq > since_seq"),
) -> dict:
    """Recent log records from the in-process ring buffer."""
    from minimappr.core.logging_ring import global_handler
    handler = global_handler()
    if handler is None:
        return {"records": [], "capacity": 0}
    level_no = logging.getLevelName(level.upper()) if isinstance(level, str) else logging.INFO
    if not isinstance(level_no, int):
        level_no = logging.INFO
    records = handler.snapshot(
        limit=limit,
        min_level=level_no,
        logger_prefix=logger_prefix,
        since_seq=since_seq,
    )
    return {"records": records, "capacity": handler._buffer.maxlen or 0}


@app.get("/api/v1/zones/occupancy", response_model=list[ZoneOccupancyState])
async def get_zone_occupancy(request: Request) -> list[ZoneOccupancyState]:
    """Return occupancy state for all zones based on the current active track snapshot."""
    state = _require_state(request)
    now_ns = time.time_ns()
    tracks = await state.tracker.snapshot(now_ns=now_ns)
    for track in tracks:
        if track.position_geo is None:
            track.position_geo = state.coordinate_frame.local_to_geo(track.position_m)
    return await state.zone_matcher.compute_occupancy(
        tracks=tracks,
        coordinate_frame=state.coordinate_frame,
        now_ns=now_ns,
    )


@app.get("/api/v1/context/current", response_model=ContextSnapshot)
async def get_context_current(
    request: Request,
    recent_alert_window_seconds: float = Query(default=300.0, ge=10.0, le=3600.0),
    track_limit: int = Query(default=200, ge=1, le=1000),
    alert_limit: int = Query(default=50, ge=1, le=500),
) -> ContextSnapshot:
    """Return a unified operational snapshot for downstream automation and reasoning.

    Combines active tracks, zone occupancy, recent alerts, node health, and
    environment conditions into a single structured response.
    """
    state = _require_state(request)
    settings: Settings = state.settings
    now_ns = time.time_ns()

    # Active tracks with geographic positions
    tracks = await state.tracker.snapshot(now_ns=now_ns)
    for track in tracks:
        if track.position_geo is None:
            track.position_geo = state.coordinate_frame.local_to_geo(track.position_m)
    active_tracks = [
        t.model_dump(mode="json")
        for t in tracks
        if t.status in {"tentative", "confirmed", "coasting"}
    ][:track_limit]

    # Zone occupancy
    zone_occupancy = await state.zone_matcher.compute_occupancy(
        tracks=tracks,
        coordinate_frame=state.coordinate_frame,
        now_ns=now_ns,
    )

    # Recent alerts
    recent_window_ns = now_ns - int(recent_alert_window_seconds * 1_000_000_000)
    all_alerts = await state.storage.list_alerts(limit=alert_limit)
    recent_alerts = [a for a in all_alerts if int(a.get("timestamp_ns", 0)) >= recent_window_ns]

    # Node health counts
    node_counts = await state.storage.count_nodes_by_status(
        now_ns=now_ns,
        degraded_after_seconds=settings.node_degraded_after_seconds,
        offline_after_seconds=settings.node_offline_after_seconds,
    )

    # Environment snapshot at origin
    conditions = state.environment_provider.get_conditions(location_m=None)
    environment = {
        "temperature_c": conditions.temperature_c,
        "humidity_fraction": conditions.humidity_fraction,
        "pressure_pa": conditions.pressure_pa,
        "wind_speed_mps": conditions.wind_speed_mps,
        "wind_dir_deg": conditions.wind_dir_deg,
        "speed_of_sound_mps": state.environment_provider.get_speed_of_sound(location_m=None),
    }

    # Derive system health from node and track state
    offline_nodes = node_counts.get("offline_nodes", 0)
    degraded_nodes = node_counts.get("degraded_nodes", 0)
    total_nodes = offline_nodes + degraded_nodes + node_counts.get("online_nodes", 0)
    if total_nodes > 0 and offline_nodes == total_nodes:
        system_health = "error"
    elif degraded_nodes > 0 or offline_nodes > 0:
        system_health = "degraded"
    else:
        system_health = "ok"

    return ContextSnapshot(
        generated_ns=now_ns,
        active_tracks=active_tracks,
        zone_occupancy=zone_occupancy,
        recent_alerts=recent_alerts,
        node_health={
            "online": node_counts.get("online_nodes", 0),
            "degraded": degraded_nodes,
            "offline": offline_nodes,
        },
        environment=environment,
        system_health=system_health,
    )


def _resolve_snippet_file(snippet_path: str | None, snippet_root: Path) -> Path | None:
    if not snippet_path:
        return None
    snippet_file = Path(snippet_path).resolve()
    if not snippet_file.exists():
        return None
    if not snippet_file.is_relative_to(snippet_root):
        raise HTTPException(status_code=403, detail="Snippet path is outside snippet directory")
    return snippet_file


@app.get("/api/v1/detections/{detection_id}")
async def get_detection_by_id(detection_id: str, request: Request) -> dict:
    state = _require_state(request)
    detection = await state.storage.get_detection(detection_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    if detection.get("position_geo") is None and detection.get("position_m"):
        local = detection["position_m"]
        geo = state.coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
        detection["position_geo"] = geo.model_dump(mode="json")
    return detection


@app.get("/api/v1/detections/{detection_id}/audio")
async def get_detection_audio(
    detection_id: str,
    request: Request,
    download: bool = Query(default=False),
) -> FileResponse:
    state = _require_state(request)
    settings: Settings = state.settings
    snippet_path = await state.storage.snippet_path_for_detection(detection_id)
    if not snippet_path:
        raise HTTPException(status_code=404, detail="Snippet not found for detection")

    snippet_file = _resolve_snippet_file(snippet_path, settings.snippet_dir.resolve())
    if snippet_file is None:
        raise HTTPException(status_code=404, detail="Snippet file no longer exists")

    filename = f"{detection_id}.wav"
    content_disposition = "attachment" if download else "inline"
    return FileResponse(
        path=snippet_file,
        media_type="audio/wav",
        filename=filename,
        headers={"Content-Disposition": f'{content_disposition}; filename="{filename}"'},
    )


@app.get("/api/v1/tracks/{track_id}/audio")
async def get_track_audio(
    track_id: str,
    request: Request,
    download: bool = Query(default=False),
) -> FileResponse:
    state = _require_state(request)
    settings: Settings = state.settings
    latest = await state.storage.latest_detection_audio_for_track(track_id)
    if latest is None:
        raise HTTPException(status_code=404, detail="No audio snippet is available for this track")

    detection_id, snippet_path = latest
    snippet_file = _resolve_snippet_file(snippet_path, settings.snippet_dir.resolve())
    if snippet_file is None:
        raise HTTPException(status_code=404, detail="Snippet file no longer exists")

    filename = f"track_{track_id}__detection_{detection_id}.wav"
    content_disposition = "attachment" if download else "inline"
    return FileResponse(
        path=snippet_file,
        media_type="audio/wav",
        filename=filename,
        headers={"Content-Disposition": f'{content_disposition}; filename="{filename}"'},
    )


@app.get("/api/v1/nodes/{node_id}/audio/recent")
async def get_recent_node_audio(
    node_id: str,
    request: Request,
    seconds: float = Query(default=10.0, ge=1.0, le=30.0),
    channel: int | None = Query(default=None, ge=0),
    render: Literal["auto", "mix", "multichannel"] = Query(default="auto"),
) -> Response:
    state = _require_state(request)
    settings: Settings = state.settings

    sensor_descriptors = await state.registry.sensors_for_node(node_id)
    if not sensor_descriptors:
        raise HTTPException(status_code=404, detail="Node has no live sensors available for audio debug")

    sensor_ids = [descriptor.sensor_id for descriptor in sensor_descriptors]
    recent = await state.audio_buffer.get_recent_window_for_sensors(
        sensor_ids=sensor_ids,
        window_seconds=float(seconds),
    )
    if recent is None:
        raise HTTPException(status_code=404, detail="No recent audio available for node")

    windows, sample_rate_hz, latest_sample_time_ns = recent
    age_seconds = max(0.0, (time.time_ns() - latest_sample_time_ns) / 1_000_000_000.0)
    if age_seconds > settings.node_degraded_after_seconds:
        raise HTTPException(status_code=404, detail="No recent audio available for node")

    ordered_descriptors = sorted(sensor_descriptors, key=lambda descriptor: descriptor.channel_index)
    raw_channels: list[np.ndarray] = [
        windows[d.sensor_id]
        for d in ordered_descriptors
        if d.sensor_id in windows
    ]
    if not raw_channels:
        raise HTTPException(status_code=404, detail="No recent audio available for node")

    common_samples = min(ch.size for ch in raw_channels)
    if common_samples <= 0:
        raise HTTPException(status_code=404, detail="No recent audio available for node")

    _channel = channel
    _render = render
    _sample_rate_hz = sample_rate_hz
    _node_id = node_id

    # numpy stacking and WAV encoding are CPU-bound; run them off the event loop.
    def _encode_audio():
        channels_first = np.vstack([ch[-common_samples:] for ch in raw_channels])
        if _channel is not None:
            if _channel >= channels_first.shape[0]:
                return None, None, None, None
            r_mode = "single_channel"
            rendered = channels_first[_channel : _channel + 1]
            sel_ch = str(_channel)
        elif _render == "multichannel":
            r_mode = "multichannel"
            rendered = channels_first
            sel_ch = "all"
        elif _render == "mix":
            r_mode = "mix"
            rendered = mono_mix(channels_first)[None, :]
            sel_ch = "mix"
        elif channels_first.shape[0] > 2:
            # Blindly summing a compact array's raw channels adds comb filtering to
            # listen-check audio. Default multichannel arrays to ch0 unless the
            # caller explicitly requests a mix or multichannel WAV.
            r_mode = "auto_first_channel"
            rendered = channels_first[0:1]
            sel_ch = "0"
        else:
            r_mode = "auto_mix"
            rendered = mono_mix(channels_first)[None, :]
            sel_ch = "mix"
        return wav_multichannel_bytes(rendered, sample_rate_hz=_sample_rate_hz), r_mode, sel_ch, len(raw_channels)

    wav_bytes, render_mode, selected_channel, n_channels = await asyncio.to_thread(_encode_audio)

    if wav_bytes is None:
        raise HTTPException(status_code=404, detail="Requested audio channel is unavailable for node")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="{_node_id}_recent.wav"',
            "X-Minimappr-Node-Id": _node_id,
            "X-Minimappr-Sample-Rate": str(_sample_rate_hz),
            "X-Minimappr-Source-Channels": str(n_channels),
            "X-Minimappr-Rendered-Channel": selected_channel,
            "X-Minimappr-Render-Mode": render_mode,
            "X-Minimappr-Clip-Seconds": f"{common_samples / float(_sample_rate_hz):.3f}",
            "X-Minimappr-Audio-Age-Seconds": f"{age_seconds:.3f}",
        },
    )


@app.get("/api/v1/soundscape/render")
async def render_soundscape(
    request: Request,
    limit: int = Query(default=32, ge=1, le=256),
    render_format: str = Query(default="bformat", pattern="^(bformat|surround_5_1)$"),
    listener_x: float = Query(default=0.0),
    listener_y: float = Query(default=0.0),
    listener_z: float = Query(default=0.0),
    suppress_label: list[str] | None = Query(default=None),
) -> Response:
    state = _require_state(request)
    settings: Settings = state.settings
    detections = await state.storage.list_detections(
        limit=limit,
        min_label_confidence=settings.detection_min_confidence,
    )
    snippet_root = settings.snippet_dir.resolve()
    blocked_labels = {label.strip().lower() for label in (suppress_label or []) if label.strip()}
    _render_format = render_format
    _listener = (float(listener_x), float(listener_y), float(listener_z))

    # WAV reads (blocking file I/O) and all numpy/encoding work run in a thread
    # so the event loop stays free for the pipeline during this potentially slow operation.
    def _load_and_render():
        sample_rate_hz: int | None = None
        sources: list[SpatialSourceFrame] = []
        n_skipped = 0
        for detection in detections:
            snippet_file = _resolve_snippet_file(detection.get("snippet_path"), snippet_root)
            if snippet_file is None:
                n_skipped += 1
                continue

            position = detection.get("position_m")
            if not isinstance(position, list) or len(position) != 3:
                n_skipped += 1
                continue

            try:
                samples, snippet_rate_hz = read_wav_mono(snippet_file)
            except Exception:
                n_skipped += 1
                continue
            if samples.size == 0:
                n_skipped += 1
                continue

            if sample_rate_hz is None:
                sample_rate_hz = snippet_rate_hz
            elif snippet_rate_hz != sample_rate_hz:
                n_skipped += 1
                continue

            sources.append(
                SpatialSourceFrame(
                    samples=samples,
                    position_m=(float(position[0]), float(position[1]), float(position[2])),
                    label=str(detection.get("label") or ""),
                    gain=max(0.0, min(1.0, float(detection.get("label_confidence") or 1.0))),
                    source_id=str(detection.get("id") or ""),
                )
            )

        if not sources or sample_rate_hz is None:
            return None, None, n_skipped, None

        renderer = SoundscapeRenderer(
            encoder=AmbisonicSpatialEncoder(),
            suppress_labels=blocked_labels if blocked_labels else None,
        )
        result = renderer.render(sources, listener_position_m=_listener)
        ch = result.bformat
        if _render_format == "surround_5_1":
            ch = foa_to_5_1(ch)
        return wav_multichannel_bytes(ch, sample_rate_hz=sample_rate_hz), sample_rate_hz, n_skipped, result

    wav_bytes, sample_rate_hz, skipped_sources, rendered = await asyncio.to_thread(_load_and_render)

    if wav_bytes is None or rendered is None:
        raise HTTPException(status_code=404, detail="No compatible detection snippets available for rendering")

    filename_suffix = "surround_5_1" if render_format == "surround_5_1" else "bformat"
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename=\"soundscape_{filename_suffix}.wav\"',
            "X-Minimappr-Rendered-Sources": str(rendered.rendered_sources),
            "X-Minimappr-Suppressed-Sources": str(rendered.suppressed_sources),
            "X-Minimappr-Skipped-Sources": str(skipped_sources),
            "X-Minimappr-Sample-Rate": str(sample_rate_hz),
        },
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


if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")
