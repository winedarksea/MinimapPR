"""MinimapPR application entrypoint."""

from __future__ import annotations

import asyncio
import contextlib
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from minimappr.api.live import LiveEventHub
from minimappr.classifiers.factory import create_classifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.localization import LocalizationEngine
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.models import IngestFrameRequest, IngestFrameResponse
from minimappr.storage.db import Storage


settings = Settings.from_env()
storage = Storage(settings.db_path)
registry = NodeRegistry()
audio_buffer = MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds)
localizer = LocalizationEngine()
classifier = create_classifier(settings)
tracker = TrackManager(settings)
live_hub = LiveEventHub()
fusion_node = FusionNode(
    settings=settings,
    registry=registry,
    buffer=audio_buffer,
    localizer=localizer,
    classifier=classifier,
    tracker=tracker,
    storage=storage,
    live_callback=live_hub.broadcast,
)

cleanup_task: asyncio.Task | None = None


async def _cleanup_loop() -> None:
    while True:
        now_ns = time.time_ns()
        await storage.cleanup_expired_snippets(now_ns=now_ns)
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
    return {
        "status": "ok",
        "time_ns": time.time_ns(),
        "classifier": settings.classifier_backend,
        "fusion_queue_depth": status["queue"]["depth"],
        "fusion_workers_running": status["workers"]["running"],
    }


@app.post("/api/v1/ingest/frame", response_model=IngestFrameResponse)
async def ingest_frame(payload: IngestFrameRequest) -> IngestFrameResponse:
    try:
        return await fusion_node.ingest(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/nodes")
async def list_nodes() -> list[dict]:
    return await storage.list_nodes()


@app.get("/api/v1/detections")
async def list_detections(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict]:
    return await storage.list_detections(limit=limit)


@app.get("/api/v1/tracks")
async def list_tracks(limit: int = Query(default=200, ge=1, le=1000)) -> list[dict]:
    now_ns = time.time_ns()
    _ = await tracker.snapshot(now_ns=now_ns)
    return await storage.list_tracks(limit=limit)


@app.get("/api/v1/config")
async def get_config() -> dict:
    return {
        "trigger_rms": settings.trigger_rms,
        "trigger_cooldown_seconds": settings.trigger_cooldown_seconds,
        "localization_window_seconds": settings.localization_window_seconds,
        "snippet_retention_seconds": settings.snippet_retention_seconds,
        "default_temperature_c": settings.default_temperature_c,
        "default_humidity": settings.default_humidity,
        "tracking_filter": settings.tracking_filter,
        "fusion_worker_count": settings.fusion_worker_count,
        "fusion_event_queue_size": settings.fusion_event_queue_size,
    }


@app.get("/api/v1/fusion/status")
async def fusion_status() -> dict:
    return await fusion_node.status()


@app.websocket("/ws/live")
async def live_events(websocket: WebSocket) -> None:
    await live_hub.connect(websocket)
    try:
        while True:
            await websocket.receive()
    except WebSocketDisconnect:
        pass
    finally:
        await live_hub.disconnect(websocket)
