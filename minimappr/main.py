"""MinimapPR application entrypoint."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import time
import multiprocessing
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Literal
import urllib.error
import urllib.request

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect

from minimappr.api.binary_ingest import parse_binary_ingest_payload
from minimappr.api.live import LiveEventHub
from minimappr.api.spool_consumer import IngestSpoolConfig, IngestSpoolConsumer
from minimappr.api.transports import HttpIngestTransport
from minimappr.classifiers.factory import create_classifier
from minimappr.cleanup_service import CleanupService
from minimappr.config import Settings
from minimappr.core.capture_session import (
    CaptureSessionManager,
    CaptureStartRequest,
    CaptureState,
)
from minimappr.core.iamf_pipeline import IamfPipeline
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
    EnvironmentSampleIn,
    FederationAck,
    FederationHeartbeat,
    FederationStatusResponse,
    FederationTrackSnapshot,
    FusionStatusResponse,
    GeoPoint,
    IngestFrameRequest,
    IngestFrameResponse,
    StoreForwardBufferedFrameResponse,
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
_SIDECAR_READY_TIMEOUT_SECONDS = 5.0
_SIDECAR_READY_POLL_INTERVAL_SECONDS = 0.1
_SIDECAR_HEALTHCHECK_TIMEOUT_SECONDS = 0.5
_INGEST_PATH_PREFIXES = (
    "/api/v1/ingest",
    "/api/v1/fusion/status",
    "/api/v1/system/diagnostics",
)


def _default_sidecar_classifier_command_json(settings: "Settings") -> str | None:
    if getattr(settings, "classifier_backend", "").lower() == "birdnet":
        return json.dumps([sys.executable, "-m", "minimappr.sidecar_classifier_helper"])
    return None


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


async def _api_live_db_poll_loop(app: FastAPI) -> None:
    """Bridge ingest-process DB writes into API-process websocket updates."""
    state = app.state
    settings: Settings = state.settings
    last_detection_ts = time.time_ns()
    last_track_ts = last_detection_ts
    seen_detection_ids: set[str] = set()
    seen_track_ids: set[str] = set()
    while True:
        try:
            detections = await state.storage.list_detections(
                limit=100,
                since_ns=last_detection_ts,
                min_label_confidence=settings.detection_min_confidence,
            )
            for detection in sorted(detections, key=lambda item: int(item.get("timestamp_ns") or 0)):
                detection_id = str(detection.get("id") or detection.get("event_id") or "")
                timestamp_ns = int(detection.get("timestamp_ns") or last_detection_ts)
                if detection_id and detection_id not in seen_detection_ids:
                    await state.live_hub.broadcast(
                        {
                            "type": "detection",
                            "event_id": detection.get("event_id"),
                            "event_type": "detection",
                            "detection": detection,
                            "track": None,
                            "server_time_ns": time.time_ns(),
                        }
                    )
                    seen_detection_ids.add(detection_id)
                    if len(seen_detection_ids) > 512:
                        seen_detection_ids = set(list(seen_detection_ids)[-256:])
                last_detection_ts = max(last_detection_ts, timestamp_ns)

            tracks = await state.storage.list_tracks(limit=100, since_ns=last_track_ts)
            for track in sorted(tracks, key=lambda item: int(item.get("last_seen_ns") or 0)):
                track_id = str(track.get("id") or "")
                last_seen_ns = int(track.get("last_seen_ns") or last_track_ts)
                dedupe_key = f"{track_id}:{last_seen_ns}"
                if track_id and dedupe_key not in seen_track_ids:
                    await state.live_hub.broadcast(
                        {
                            "type": "track_updated",
                            "track": track,
                            "server_time_ns": time.time_ns(),
                        }
                    )
                    seen_track_ids.add(dedupe_key)
                    if len(seen_track_ids) > 512:
                        seen_track_ids = set(list(seen_track_ids)[-256:])
                last_track_ts = max(last_track_ts, last_seen_ns)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("API live DB poll failed: %s", exc)
        await asyncio.sleep(1.0)


class _SidecarState:
    """Mutable runtime health for the supervised ingest sidecar process.

    Shared between the supervision task and the diagnostics endpoint so the
    Server page always reflects live process state.
    """

    __slots__ = ("status", "pid", "restart_count", "last_exit_code", "_current_process")

    def __init__(self) -> None:
        self.status: str = "disabled"
        self.pid: int | None = None
        self.restart_count: int = 0
        self.last_exit_code: int | None = None
        self._current_process: "asyncio.subprocess.Process | None" = None


class _EnvironmentIngestSample(BaseModel):
    node_id: str
    sample: EnvironmentSampleIn


class _EnvironmentIngestBody(BaseModel):
    samples: list[_EnvironmentIngestSample]


def _ingest_sidecar_is_running(state) -> bool:
    sidecar_state: _SidecarState | None = getattr(state, "sidecar_state", None)
    return bool(
        state.settings.ingest_sidecar_enabled
        and sidecar_state is not None
        and sidecar_state.status == "running"
    )


def _should_block_direct_ingest(state) -> bool:
    """Return True when direct ingest must be rejected in favor of sidecar ingest.

    Direct ingest should only be hard-blocked when operators disabled it and the
    sidecar is confirmed running. If sidecar startup fails (missing binary,
    crash, misconfiguration), we fail open to keep nodes reporting instead of
    creating a complete ingest outage.
    """
    if state.settings.direct_ingest_enabled:
        return False
    return _ingest_sidecar_is_running(state)


def _should_autostart_ingest_sidecar(settings: "Settings") -> bool:
    """Return True when the managed sidecar process should be started.

    In Python-direct ingest mode, firmware payloads are accepted directly by the
    API process, so auto-launching the sidecar is unnecessary and can introduce
    unrelated startup failures.
    """
    return (
        getattr(settings, "process_role", "combined") != "ingest"
        and getattr(settings, "ingest_backend", "rust") == "rust"
        and settings.ingest_sidecar_enabled
        and not settings.direct_ingest_enabled
    )


def _sidecar_healthcheck_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/healthz"


def _ingest_runtime_base_url(settings: "Settings") -> str:
    return settings.ingest_base_url.rstrip("/")


def _capture_pipeline_available(settings: "Settings") -> bool:
    return settings.ingest_backend == "rust" and settings.ingest_storage_mode == "journal"


def _has_live_ingest_runtime(state) -> bool:
    settings = getattr(state, "settings", None)
    if settings is not None and getattr(settings, "process_role", "combined") == "api":
        return False
    return hasattr(state, "registry") and hasattr(state, "audio_buffer")


def _sensor_ids_from_node_row(node: dict) -> list[str]:
    offsets = node.get("sensor_offsets_m")
    if not isinstance(offsets, list):
        return []
    return [f"{node['id']}:ch{index}" for index in range(len(offsets))]


def _fetch_ingest_sidecar_health(port: int) -> dict[str, object] | None:
    request = urllib.request.Request(_sidecar_healthcheck_url(port), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=_SIDECAR_HEALTHCHECK_TIMEOUT_SECONDS) as response:
            if getattr(response, "status", None) != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, ValueError, urllib.error.HTTPError, urllib.error.URLError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _probe_ingest_sidecar_ready(port: int) -> bool:
    payload = _fetch_ingest_sidecar_health(port)
    return isinstance(payload, dict) and payload.get("status") == "ok"


async def _wait_for_ingest_sidecar_ready(
    process: "asyncio.subprocess.Process",
    *,
    port: int,
    timeout_seconds: float = _SIDECAR_READY_TIMEOUT_SECONDS,
    poll_interval_seconds: float = _SIDECAR_READY_POLL_INTERVAL_SECONDS,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while True:
        if process.returncode is not None:
            if await asyncio.to_thread(_probe_ingest_sidecar_ready, port):
                raise RuntimeError(
                    "Ingest sidecar child process exited, but readiness endpoint is already healthy on "
                    f"port {port}. Another ingest worker is likely already running."
                )
            raise RuntimeError(
                f"Ingest sidecar exited before readiness check completed (code {process.returncode})"
            )
        if await asyncio.to_thread(_probe_ingest_sidecar_ready, port):
            return
        if loop.time() >= deadline:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=1.0)
            raise RuntimeError(
                f"Ingest sidecar did not become ready on port {port} within {timeout_seconds:.1f}s"
            )
        await asyncio.sleep(poll_interval_seconds)


async def _supervise_ingest_sidecar(
    settings: "Settings",
    initial_process: "asyncio.subprocess.Process",
    state: _SidecarState,
) -> None:
    """Watch the ingest sidecar and restart it on unexpected exit.

    A clean exit (returncode 0) or SIGTERM (−15) is treated as an intentional
    shutdown and terminates supervision.  Any other exit code triggers an
    exponential-backoff restart loop.
    """
    process = initial_process
    state._current_process = process
    state.status = "running"
    state.pid = process.pid
    while True:
        returncode = await process.wait()
        state.last_exit_code = returncode
        if returncode in (0, -15):
            # Intentional shutdown; stop supervising.
            state.status = "stopped"
            state._current_process = None
            return
        state.restart_count += 1
        state.status = "restarting"
        logger.warning(
            "Ingest sidecar exited unexpectedly (code %d); restarting (attempt %d)",
            returncode,
            state.restart_count,
        )
        await asyncio.sleep(2.0)
        try:
            new_process = await _start_ingest_sidecar(settings)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to restart ingest sidecar: %s", exc)
            state.status = "crashed"
            state._current_process = None
            return
        if new_process is None:
            state.status = "binary_not_found"
            state._current_process = None
            return
        process = new_process
        state._current_process = process
        state.status = "running"
        state.pid = process.pid


def _count_failed_spool_manifest_items(spool_dir: "Path") -> int:
    """Count manifest files (.json) in the spool failed directory."""
    failed_dir = spool_dir / "failed"
    if not failed_dir.exists():
        return 0
    return sum(1 for p in failed_dir.glob("*.json") if p.is_file())


async def _start_ingest_sidecar(
    settings: "Settings",
) -> "asyncio.subprocess.Process | None":
    """Launch the Rust ingest sidecar as a managed child process.

    Returns the process if started, or None when the binary is absent or the
    feature is disabled.  The process inherits stdout/stderr so its log lines
    appear in the same terminal as the Python server.
    """
    binary = settings.ingest_sidecar_binary_path
    if not binary.exists():
        logger.warning(
            "Ingest sidecar binary not found at %s; sidecar will not start. "
            "Build it with: scripts/build_rust.sh --sidecar",
            binary,
        )
        return None
    if await asyncio.to_thread(_probe_ingest_sidecar_ready, settings.ingest_sidecar_port):
        raise RuntimeError(
            "Ingest sidecar readiness endpoint is already healthy on "
            f"port {settings.ingest_sidecar_port}. Another ingest worker is likely already running."
        )
    env = {
        **os.environ,
        # Keep spool dir in sync with the Python consumer regardless of what
        # env vars the operator may have set for the Rust binary directly.
        "MINIMAPPR_INGEST_SPOOL_DIR": str(settings.ingest_spool_dir),
        "MINIMAPPR_INGEST_CONSUMER_NAME": settings.ingest_consumer_name,
        "MINIMAPPR_INGEST_PORT": str(getattr(settings, "ingest_port", settings.ingest_sidecar_port)),
        "MINIMAPPR_SIDECAR_PORT": str(settings.ingest_sidecar_port),
        "MINIMAPPR_SIDECAR_STORAGE_MODE": settings.ingest_storage_mode,
        "MINIMAPPR_SIDECAR_TOTAL_JOURNAL_BUDGET_BYTES": str(
            settings.ingest_sidecar_total_journal_budget_bytes
        ),
        "MINIMAPPR_SIDECAR_ADMISSION_RESERVE_BYTES": str(
            settings.ingest_sidecar_admission_reserve_bytes
        ),
        "MINIMAPPR_SIDECAR_ALLOW_NON_TMPFS_JOURNAL": str(
            bool(getattr(settings, "ingest_sidecar_allow_non_tmpfs_journal", False))
        ).lower(),
        "MINIMAPPR_RUNTIME_PROFILE": str(getattr(settings, "runtime_profile", "default")),
        "MINIMAPPR_LOCALIZATION_WINDOW_SECONDS": str(
            getattr(settings, "localization_window_seconds", 0.08)
        ),
        "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS": str(
            getattr(settings, "classification_window_seconds", 30.0)
        ),
        "MINIMAPPR_CLASSIFIER_RENDER_MIN_INTERVAL_SECONDS": str(
            getattr(settings, "classifier_render_min_interval_seconds", 0.0)
        ),
        "MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS": str(
            getattr(settings, "max_sensor_buffer_seconds", 32.0)
        ),
        "MINIMAPPR_DSP_LOCALIZATION_RMS_GATE": str(
            getattr(settings, "trigger_rms", 0.0015)
        ),
        "MINIMAPPR_TRIGGER_COOLDOWN_SECONDS": str(
            getattr(settings, "trigger_cooldown_seconds", 0.8)
        ),
        "MINIMAPPR_LOCALIZATION_BAND_MIN_HZ": str(
            getattr(settings, "localization_band_min_hz", 300.0)
        ),
        "MINIMAPPR_LOCALIZATION_BAND_MAX_HZ": str(
            getattr(settings, "localization_band_max_hz", 3500.0)
        ),
        "MINIMAPPR_DEFAULT_TEMPERATURE_C": str(getattr(settings, "default_temperature_c", 20.0)),
        "MINIMAPPR_DEFAULT_HUMIDITY": str(getattr(settings, "default_humidity", 0.5)),
        "MINIMAPPR_SITE_ORIGIN_LAT": str(getattr(settings, "site_origin_lat", 0.0)),
        "MINIMAPPR_SITE_ORIGIN_LON": str(getattr(settings, "site_origin_lon", 0.0)),
        "MINIMAPPR_SITE_ORIGIN_ALT_M": str(getattr(settings, "site_origin_alt_m", 0.0)),
        "MINIMAPPR_BIRDNET_TRIGGER_MIN_CONFIDENCE": str(
            getattr(settings, "birdnet_trigger_min_confidence", 0.4)
        ),
        "MINIMAPPR_BIRDNET_GEO_MIN_CONFIDENCE": str(
            getattr(settings, "birdnet_geo_min_confidence", 0.03)
        ),
    }
    classifier_command_json = os.environ.get("MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON")
    if classifier_command_json is None:
        classifier_command_json = _default_sidecar_classifier_command_json(settings)
    if classifier_command_json is not None:
        env["MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON"] = classifier_command_json
    logger.info(
        "Starting ingest sidecar: %s (port %d, storage %s, spool %s, journal budget %d, reserve %d, allow non-tmpfs %s)",
        binary,
        settings.ingest_sidecar_port,
        settings.ingest_storage_mode,
        settings.ingest_spool_dir,
        settings.ingest_sidecar_total_journal_budget_bytes,
        settings.ingest_sidecar_admission_reserve_bytes,
        bool(getattr(settings, "ingest_sidecar_allow_non_tmpfs_journal", False)),
    )
    process = await asyncio.create_subprocess_exec(str(binary), env=env)
    await _wait_for_ingest_sidecar_ready(process, port=settings.ingest_sidecar_port)
    logger.info("Ingest sidecar ready on port %d", settings.ingest_sidecar_port)
    return process


@asynccontextmanager
async def _api_only_lifespan(app: FastAPI, settings: Settings):
    """Initialize the API/UI process without DSP, classifiers, or live ingest."""
    install_log_ring()
    settings.federation_peers_config_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    settings.large_artifact_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.storage_config().db_path)
    await storage.initialize()
    resolved_site_origin = resolve_site_origin_from_nodes(
        settings,
        nodes=await storage.list_nodes(limit=4096),
        now_ns=time.time_ns(),
    )
    settings.site_origin_lat = resolved_site_origin.origin.lat
    settings.site_origin_lon = resolved_site_origin.origin.lon
    settings.site_origin_alt_m = resolved_site_origin.origin.alt_m

    live_hub = LiveEventHub()
    coordinate_frame = LocalCoordinateFrame(
        origin=GeoPoint(
            lat=settings.site_origin_lat,
            lon=settings.site_origin_lon,
            alt_m=settings.site_origin_alt_m,
        ),
        mode=settings.coordinate_mode,
    )
    environment_provider = LiveEnvironmentProvider(
        fallback_temperature_c=settings.default_temperature_c,
        fallback_humidity_fraction=settings.default_humidity,
        max_reading_age_seconds=settings.environment_reading_max_age_seconds,
    )
    cleanup_service = CleanupService(settings=settings, storage=storage)
    bit_evaluator = BITReportEvaluator()

    async def _empty_local_tracks(now_ns: int) -> list[TrackState]:
        del now_ns
        return []

    federation = FederationCoordinator(
        settings=settings,
        track_supplier=_empty_local_tracks,
        live_callback=live_hub.broadcast,
    )

    capture_manager = CaptureSessionManager()
    iamf_pipeline = IamfPipeline(
        sidecar_url=_ingest_runtime_base_url(settings),
        db_storage=storage,
    )

    async def _run_capture_post_processing(record):
        await iamf_pipeline.run(record)
        await storage.upsert_capture_session(record)

    capture_manager.set_post_process_callback(_run_capture_post_processing)

    sidecar_state = _SidecarState()
    sidecar_supervision_task: asyncio.Task | None = None
    if _should_autostart_ingest_sidecar(settings):
        try:
            sidecar_process = await _start_ingest_sidecar(settings)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Ingest sidecar startup failed in API role: %s", exc)
            sidecar_state.status = "startup_failed"
        else:
            if sidecar_process is not None:
                sidecar_state._current_process = sidecar_process
                sidecar_state.status = "running"
                sidecar_state.pid = sidecar_process.pid
                sidecar_supervision_task = asyncio.create_task(
                    _supervise_ingest_sidecar(settings, sidecar_process, sidecar_state)
                )
            else:
                sidecar_state.status = "binary_not_found"
    else:
        sidecar_state.status = "disabled"

    app.state.settings = settings
    app.state.storage = storage
    app.state.live_hub = live_hub
    app.state.coordinate_frame = coordinate_frame
    app.state.environment_provider = environment_provider
    app.state.federation = federation
    app.state.bit_evaluator = bit_evaluator
    app.state.cleanup_service = cleanup_service
    app.state.sidecar_state = sidecar_state
    app.state.capture_manager = capture_manager
    _apply_site_origin_resolution(app.state, resolved_site_origin)

    live_db_poll_task: asyncio.Task | None = None
    try:
        environment_provider.bootstrap(await storage.list_latest_environment_per_node(limit=1024))
        await federation.start()
        live_db_poll_task = asyncio.create_task(_api_live_db_poll_loop(app))
        yield
    finally:
        shutdown_timeout_s = 15.0
        if sidecar_supervision_task is not None:
            sidecar_supervision_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(sidecar_supervision_task, timeout=shutdown_timeout_s)
        current_sidecar = sidecar_state._current_process
        if current_sidecar is not None and current_sidecar.returncode is None:
            logger.info("Stopping ingest sidecar (pid %d)", current_sidecar.pid)
            current_sidecar.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(current_sidecar.wait(), timeout=shutdown_timeout_s)
        if live_db_poll_task is not None:
            live_db_poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(live_db_poll_task, timeout=shutdown_timeout_s)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(federation.stop(), timeout=shutdown_timeout_s)
        await storage.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    if settings.process_role == "api":
        async with _api_only_lifespan(app, settings):
            yield
        return

    install_log_ring()
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
    ingest_spool_consumer = IngestSpoolConsumer(
        config=IngestSpoolConfig(
            spool_dir=settings.ingest_spool_dir,
            ready_ttl_seconds=settings.ingest_spool_ready_ttl_seconds,
            failed_ttl_seconds=settings.ingest_spool_failed_ttl_seconds,
            tmp_ttl_seconds=settings.ingest_spool_tmp_ttl_seconds,
            poll_interval_seconds=settings.ingest_spool_poll_interval_seconds,
            worker_count=settings.ingest_spool_worker_count,
            storage_mode=settings.ingest_storage_mode,
            consumer_name=settings.ingest_consumer_name,
            runtime_profile=settings.runtime_profile,
        ),
        ingest_transport=ingest_transport,
    )
    ingest_spool_consumer.ensure_directories()

    sidecar_state = _SidecarState()
    sidecar_supervision_task: asyncio.Task | None = None
    if _should_autostart_ingest_sidecar(settings):
        try:
            sidecar_process = await _start_ingest_sidecar(settings)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Ingest sidecar startup failed; continuing with direct ingest fallback: %s", exc)
            sidecar_state.status = "startup_failed"
        else:
            if sidecar_process is not None:
                sidecar_state._current_process = sidecar_process
                sidecar_state.status = "running"
                sidecar_state.pid = sidecar_process.pid
                sidecar_supervision_task = asyncio.create_task(
                    _supervise_ingest_sidecar(settings, sidecar_process, sidecar_state)
                )
            else:
                sidecar_state.status = "binary_not_found"
    else:
        sidecar_state.status = "disabled"

    if not settings.direct_ingest_enabled and sidecar_state.status != "running":
        logger.warning(
            "Direct ingest is disabled but sidecar is not running (status=%s). "
            "Falling back to direct ingest to avoid node ingest outage.",
            sidecar_state.status,
        )

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

    capture_manager = CaptureSessionManager()
    iamf_pipeline = IamfPipeline(
        sidecar_url=_ingest_runtime_base_url(settings),
        db_storage=storage,
    )

    async def _run_capture_post_processing(record):
        await iamf_pipeline.run(record)
        await storage.upsert_capture_session(record)

    capture_manager.set_post_process_callback(_run_capture_post_processing)

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
    app.state.ingest_spool_consumer = ingest_spool_consumer
    app.state.sidecar_state = sidecar_state
    app.state.capture_manager = capture_manager
    _apply_site_origin_resolution(app.state, resolved_site_origin)

    cleanup_task: asyncio.Task | None = None
    site_origin_reconcile_task: asyncio.Task | None = None
    ingest_spool_tasks: list[asyncio.Task] = []
    try:
        environment_provider.bootstrap(await storage.list_latest_environment_per_node(limit=1024))
        await fusion_node.start()
        await federation.start()
        cleanup_task = asyncio.create_task(_cleanup_loop(app))
        app.state.cleanup_task = cleanup_task
        ingest_spool_tasks = [
            asyncio.create_task(ingest_spool_consumer.run_forever(worker_name=f"spool-{index + 1}"))
            for index in range(settings.ingest_spool_worker_count)
        ]
        app.state.ingest_spool_tasks = ingest_spool_tasks
        if should_schedule_deferred_site_origin_reconciliation(
            settings,
            initial_resolution_source=resolved_site_origin.source,
        ):
            site_origin_reconcile_task = asyncio.create_task(_reconcile_site_origin_after_startup(app))
        app.state.site_origin_reconcile_task = site_origin_reconcile_task

        yield
    finally:
        shutdown_timeout_s = 15.0
        # Stop the ingest sidecar first so startup/bind failures do not leave
        # the Rust process running after the Python server begins teardown.
        if sidecar_supervision_task is not None:
            sidecar_supervision_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(sidecar_supervision_task, timeout=shutdown_timeout_s)
        current_sidecar = sidecar_state._current_process
        if current_sidecar is not None and current_sidecar.returncode is None:
            logger.info("Stopping ingest sidecar (pid %d)", current_sidecar.pid)
            current_sidecar.terminate()
            try:
                await asyncio.wait_for(current_sidecar.wait(), timeout=shutdown_timeout_s)
            except asyncio.TimeoutError:
                logger.warning("Ingest sidecar did not exit cleanly; sending SIGKILL")
                current_sidecar.kill()
                await current_sidecar.wait()

        if site_origin_reconcile_task is not None:
            site_origin_reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(site_origin_reconcile_task, timeout=shutdown_timeout_s)
        if cleanup_task is not None:
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(cleanup_task, timeout=shutdown_timeout_s)
        for task in ingest_spool_tasks:
            task.cancel()
        if ingest_spool_tasks:
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.gather(*ingest_spool_tasks), timeout=shutdown_timeout_s)

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

        # After closing the classifier, explicitly clean up multiprocessing resources
        # to avoid "resource_tracker: There appear to be X leaked shared_memory objects" warnings.
        # Give multiprocessing time to fully clean up before the event loop closes.
        try:
            # Wait for any active processes to finish with an explicit timeout.
            # This allows the resource_tracker to properly unregister shared_memory objects.
            for proc in multiprocessing.active_children():
                try:
                    proc.join(timeout=1.0)
                except Exception:  # noqa: BLE001
                    pass
            logger.info("Multiprocessing cleanup completed during shutdown")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Multiprocessing cleanup warning: %s", exc)


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


@app.middleware("http")
async def process_role_route_guard(request: Request, call_next):
    settings: Settings | None = getattr(request.app.state, "settings", None)
    role = settings.process_role if settings is not None else os.getenv("MINIMAPPR_PROCESS_ROLE", "combined")
    path = request.url.path
    if role == "api" and path.startswith("/api/v1/ingest"):
        return JSONResponse(status_code=404, content={"detail": "Ingest endpoints are served by the ingest process"})
    if role == "ingest":
        allowed = path == "/health" or path.startswith(_INGEST_PATH_PREFIXES)
        if not allowed:
            return JSONResponse(status_code=404, content={"detail": "Endpoint is not served by the ingest process"})
    return await call_next(request)


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
    if hasattr(state, "fusion_node"):
        status = await state.fusion_node.status()
        workers = status.get("workers", {})
        running = int(workers.get("localization_running", 0)) + int(workers.get("classification_running", 0)) + int(
            workers.get("rules_running", 0)
        )
        fusion_queue_depth = status["queue"]["localization_depth"]
    else:
        running = 0
        fusion_queue_depth = 0
    federation_status = await state.federation.status()
    return {
        "status": "ok",
        "time_ns": time.time_ns(),
        "process_role": settings.process_role,
        "ingest_backend": settings.ingest_backend,
        "ingest_port": settings.ingest_port,
        "classifier": settings.classifier_backend,
        "fusion_queue_depth": fusion_queue_depth,
        "fusion_workers_running": running,
        "federation_enabled": federation_status["enabled"],
        "federation_peer_count": federation_status["peer_count"],
    }


@app.post("/api/v1/ingest/frame", response_model=IngestFrameResponse)
async def ingest_frame(payload: IngestFrameRequest, request: Request) -> IngestFrameResponse:
    state = _require_state(request)
    try:
        return await state.fusion_node.ingest(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/ingest/store-forward", response_model=StoreForwardIngestResponse)
async def ingest_store_forward(payload: StoreForwardIngestRequest, request: Request) -> StoreForwardIngestResponse:
    state = _require_state(request)
    if _should_block_direct_ingest(state):
        raise HTTPException(status_code=410, detail="Direct ingest is disabled; send firmware ingest to the Rust sidecar")
    try:
        results = []
        for item in payload.buffered_frames:
            req = IngestFrameRequest(
                node=payload.node,
                frame=item.frame,
                environment=item.environment,
            )
            resp = await state.fusion_node.ingest(req)
            results.append(StoreForwardBufferedFrameResponse(
                sequence=item.frame.sequence,
                start_time_ns=item.frame.start_time_ns,
                accepted=resp.accepted,
                duplicate=resp.duplicate,
                triggered=resp.triggered,
                frame_energy=resp.frame_energy,
                queued_event_id=resp.queued_event_id,
            ))
        return StoreForwardIngestResponse(
            accepted=True,
            total_frames=len(payload.buffered_frames),
            accepted_frames=sum(1 for r in results if r.accepted),
            duplicate_frames=sum(1 for r in results if r.duplicate),
            rejected_frames=sum(1 for r in results if not r.accepted),
            queued_events=sum(1 for r in results if r.queued_event_id),
            results=results,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/ingest/binary", response_model=StoreForwardIngestResponse)
async def ingest_binary(request: Request) -> StoreForwardIngestResponse:
    state = _require_state(request)
    if _should_block_direct_ingest(state):
        raise HTTPException(status_code=410, detail="Direct ingest is disabled; send firmware ingest to the Rust sidecar")
    try:
        body = await request.body()
        payload = await asyncio.to_thread(parse_binary_ingest_payload, body)
        results = []
        for item in payload.buffered_frames:
            req = IngestFrameRequest(
                node=payload.node,
                frame=item.frame,
                environment=item.environment,
            )
            resp = await state.fusion_node.ingest(req)
            results.append(StoreForwardBufferedFrameResponse(
                sequence=item.frame.sequence,
                start_time_ns=item.frame.start_time_ns,
                accepted=resp.accepted,
                duplicate=resp.duplicate,
                triggered=resp.triggered,
                frame_energy=resp.frame_energy,
                queued_event_id=resp.queued_event_id,
            ))
        return StoreForwardIngestResponse(
            accepted=True,
            total_frames=len(payload.buffered_frames),
            accepted_frames=sum(1 for r in results if r.accepted),
            duplicate_frames=sum(1 for r in results if r.duplicate),
            rejected_frames=sum(1 for r in results if not r.accepted),
            queued_events=sum(1 for r in results if r.queued_event_id),
            results=results,
        )
    except ClientDisconnect as exc:
        raise HTTPException(status_code=499, detail="Client disconnected while uploading binary ingest payload") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/ingest/env")
async def ingest_environment(payload: _EnvironmentIngestBody, request: Request) -> dict:
    state = _require_state(request)
    if len(payload.samples) > 64:
        raise HTTPException(status_code=413, detail="environment batch exceeds 64 samples")

    accepted = 0
    now_ns = time.time_ns()
    for item in payload.samples:
        sample = item.sample
        if not sample.has_any_measurement():
            continue
        timestamp_ns = sample.timestamp_ns or now_ns
        await state.storage.insert_environment(
            node_id=item.node_id,
            timestamp_ns=timestamp_ns,
            temperature_c=sample.temperature_c,
            pressure_pa=sample.pressure_pa,
            humidity_fraction=sample.humidity_fraction,
            wind_speed_mps=sample.wind_speed_mps,
            wind_dir_deg=sample.wind_dir_deg,
            solar_lux=sample.solar_lux,
            metadata={"source": sample.source, **sample.metadata} if sample.source else sample.metadata,
        )
        environment_provider = getattr(state, "environment_provider", None)
        if environment_provider is not None and hasattr(environment_provider, "ingest_sample"):
            environment_provider.ingest_sample(
                node_id=item.node_id,
                timestamp_ns=timestamp_ns,
                temperature_c=sample.temperature_c,
                humidity_fraction=sample.humidity_fraction,
                pressure_pa=sample.pressure_pa,
                wind_speed_mps=sample.wind_speed_mps,
                wind_dir_deg=sample.wind_dir_deg,
                solar_lux=sample.solar_lux,
                location_m=None,
                metadata={"source": sample.source, **sample.metadata} if sample.source else sample.metadata,
            )
        accepted += 1
    return {"accepted": accepted, "queued": False}


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
    latest_time_quality_by_node = await state.storage.list_latest_time_quality_per_node()
    latest_observation_metadata_by_node = await state.storage.list_latest_observation_metadata_per_node()
    latest_audio_summary_rows = await state.storage.list_node_audio_summaries(limit=limit)
    latest_environment_by_node = {
        row["node_id"]: row for row in latest_environment_rows if row.get("node_id") is not None
    }
    latest_audio_summary_by_node = {
        row["node_id"]: row for row in latest_audio_summary_rows if row.get("node_id") is not None
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

        if _has_live_ingest_runtime(state):
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
                "recent_coverage_ratio": audio_summary["recent_coverage_ratio"],
                "recent_missing_ratio": audio_summary["recent_missing_ratio"],
                "recent_max_gap_seconds": audio_summary["recent_max_gap_seconds"],
                "max_buffer_samples": audio_summary["max_buffer_samples"],
                "max_buffer_seconds": audio_summary["max_buffer_seconds"],
                "status": audio_status,
            }
        else:
            sensor_ids = _sensor_ids_from_node_row(node)
            persisted_summary = latest_audio_summary_by_node.get(node["id"])
            if persisted_summary is not None:
                last_sample_time_ns = persisted_summary.get("last_sample_time_ns")
                age_seconds = persisted_summary.get("age_seconds")
                if isinstance(last_sample_time_ns, int):
                    age_seconds = max(0.0, (now_ns - last_sample_time_ns) / 1_000_000_000.0)
                if age_seconds is None:
                    audio_status = "external_ingest_process"
                elif float(age_seconds) <= settings.node_degraded_after_seconds:
                    audio_status = "recent"
                else:
                    audio_status = "stale"
                persisted_summary["sensor_count"] = int(persisted_summary.get("sensor_count") or len(sensor_ids))
                persisted_summary["active_sensor_count"] = int(persisted_summary.get("active_sensor_count") or 0)
                persisted_summary["age_seconds"] = age_seconds
                persisted_summary["status"] = audio_status
                node["audio_debug"] = persisted_summary
            else:
                node["audio_debug"] = {
                    "sensor_count": len(sensor_ids),
                    "active_sensor_count": 0,
                    "sample_rate_hz": None,
                    "last_sample_time_ns": None,
                    "age_seconds": None,
                    "rms": None,
                    "recent_coverage_ratio": None,
                    "recent_missing_ratio": None,
                    "recent_max_gap_seconds": None,
                    "max_buffer_samples": None,
                    "max_buffer_seconds": None,
                    "status": "external_ingest_process",
                }

        latest_environment = latest_environment_by_node.get(node["id"])
        if latest_environment is not None:
            node["latest_environment"] = latest_environment

        tq = latest_time_quality_by_node.get(node["id"])
        if tq is None and isinstance(node.get("metadata"), dict):
            metadata_time_quality = node["metadata"].get("time_quality")
            if isinstance(metadata_time_quality, str) and metadata_time_quality:
                tq = metadata_time_quality
        if tq is not None:
            node["latest_time_quality"] = tq
        latest_observation_metadata = latest_observation_metadata_by_node.get(node["id"], {})
        if isinstance(latest_observation_metadata, dict):
            timing_diagnostics = latest_observation_metadata.get("timing_diagnostics")
            if isinstance(timing_diagnostics, dict):
                node["latest_timing_diagnostics"] = timing_diagnostics
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
    if hasattr(state, "tracker"):
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
        "process_role": settings.process_role,
        "ingest_backend": settings.ingest_backend,
        "ingest_host": settings.ingest_host,
        "ingest_port": settings.ingest_port,
        "ingest_base_url": settings.ingest_base_url,
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
    if not hasattr(state, "fusion_node"):
        raise HTTPException(status_code=503, detail="Fusion pipeline runs in the ingest process")
    return await state.fusion_node.status()


@app.get("/api/v1/debug/config")
async def debug_config(request: Request) -> dict:
    state = _require_state(request)
    if not hasattr(state, "diagnostics"):
        raise HTTPException(status_code=503, detail="Diagnostics requiring DSP runtime run in the ingest process")
    return await state.diagnostics.config_snapshot()


@app.get("/api/v1/debug/event/{event_id}")
async def debug_event_snapshot(event_id: str, request: Request) -> dict:
    state = _require_state(request)
    if not hasattr(state, "diagnostics"):
        raise HTTPException(status_code=503, detail="Diagnostics requiring DSP runtime run in the ingest process")
    snapshot = await state.diagnostics.event_snapshot(event_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return snapshot


@app.get("/api/v1/debug/selftest")
async def debug_selftest(request: Request) -> dict:
    state = _require_state(request)
    if not hasattr(state, "diagnostics"):
        raise HTTPException(status_code=503, detail="Diagnostics requiring DSP runtime run in the ingest process")
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
    diagnostics = await asyncio.to_thread(system_info.collect, db_path=settings.db_path, start_ns=process_start_ns())
    diagnostics["process_role"] = settings.process_role
    diagnostics["ingest"] = {
        "backend": settings.ingest_backend,
        "host": settings.ingest_host,
        "port": settings.ingest_port,
        "base_url": settings.ingest_base_url,
        "capture_available": _capture_pipeline_available(settings),
        "capture_unavailable_reason": None
        if _capture_pipeline_available(settings)
        else "Ambisonic/IAMF capture requires Rust ingest journal mode",
    }
    if hasattr(state, "fusion_node"):
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
    else:
        diagnostics["pipeline"] = {
            "status": "external_ingest_process",
            "queue": {},
            "workers": {},
            "realtime": {},
            "drop_on_backpressure": None,
            "metrics": {},
        }
    sidecar_state: _SidecarState | None = getattr(state, "sidecar_state", None)
    failed_spool_count = await asyncio.to_thread(
        _count_failed_spool_manifest_items,
        settings.ingest_spool_dir,
    )
    sidecar_health: dict[str, object] | None = None
    if settings.ingest_sidecar_enabled and sidecar_state is not None and sidecar_state.status == "running":
        sidecar_health = await asyncio.to_thread(
            _fetch_ingest_sidecar_health,
            settings.ingest_port,
        )
    diagnostics["sidecar"] = {
        "enabled": settings.ingest_sidecar_enabled,
        "status": sidecar_state.status if sidecar_state is not None else (
            "disabled" if not settings.ingest_sidecar_enabled else "unknown"
        ),
        "pid": sidecar_state.pid if sidecar_state is not None else None,
        "restart_count": sidecar_state.restart_count if sidecar_state is not None else 0,
        "last_exit_code": sidecar_state.last_exit_code if sidecar_state is not None else None,
        "failed_spool_items": failed_spool_count,
        "health": sidecar_health,
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
    if not hasattr(state, "tracker") or not hasattr(state, "zone_matcher"):
        return []
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
    if hasattr(state, "tracker") and hasattr(state, "zone_matcher"):
        tracks = await state.tracker.snapshot(now_ns=now_ns)
        for track in tracks:
            if track.position_geo is None:
                track.position_geo = state.coordinate_frame.local_to_geo(track.position_m)
        active_tracks = [
            t.model_dump(mode="json")
            for t in tracks
            if t.status in {"tentative", "confirmed", "coasting"}
        ][:track_limit]
        zone_occupancy = await state.zone_matcher.compute_occupancy(
            tracks=tracks,
            coordinate_frame=state.coordinate_frame,
            now_ns=now_ns,
        )
    else:
        track_rows = await state.storage.list_tracks(limit=track_limit)
        active_tracks = [
            track
            for track in track_rows
            if track.get("status") in {"tentative", "confirmed", "coasting"}
        ][:track_limit]
        zone_occupancy = []

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
    if not _has_live_ingest_runtime(state):
        raise HTTPException(
            status_code=404,
            detail="Recent raw node audio is held by the ingest process and is not available from the API process",
        )

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


# ── Capture API ────────────────────────────────────────────────────────────────

class _CaptureStartBody(BaseModel):
    stream_key: str
    max_duration_s: float = 300.0
    video_source: str | None = None
    libcamera_mode: bool = False
    deployment_profile: str = "auto"
    work_dir: str | None = None


@app.post("/api/v1/capture/start")
async def capture_start(request: Request, body: _CaptureStartBody):
    state = request.app.state
    manager: CaptureSessionManager = state.capture_manager
    settings: Settings = state.settings
    if not _capture_pipeline_available(settings):
        raise HTTPException(
            status_code=503,
            detail=(
                "Ambisonic/IAMF capture requires Rust ingest journal mode; "
                "Python ingest keeps live raw audio in memory and does not expose journal range leases"
            ),
        )

    sidecar_url = _ingest_runtime_base_url(settings)
    work_dir_path = (
        Path(body.work_dir) if body.work_dir else Path("data/captures")
    )

    req = CaptureStartRequest(
        stream_key=body.stream_key,
        sidecar_url=sidecar_url,
        work_dir=work_dir_path,
        max_duration_s=body.max_duration_s,
        video_source=body.video_source,
        libcamera_mode=body.libcamera_mode,
        deployment_profile=body.deployment_profile,
    )
    record = await manager.start(req)

    if record.state == CaptureState.FAILED:
        raise HTTPException(status_code=500, detail=record.error or "capture start failed")

    storage: Storage = state.storage
    await storage.upsert_capture_session(record)

    return {
        "session_id": record.session_id,
        "state": record.state.value,
        "stream_key": record.stream_key,
        "range_lease_id": record.range_lease_id,
        "start_time_ns": record.start_time_ns,
    }


@app.post("/api/v1/capture/{session_id}/stop")
async def capture_stop(session_id: str, request: Request):
    state = request.app.state
    manager: CaptureSessionManager = state.capture_manager
    settings: Settings = state.settings
    if not _capture_pipeline_available(settings):
        raise HTTPException(
            status_code=503,
            detail="Ambisonic/IAMF capture requires Rust ingest journal mode",
        )
    sidecar_url = _ingest_runtime_base_url(settings)

    try:
        record = await manager.stop(session_id, sidecar_url)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    storage: Storage = state.storage
    await storage.upsert_capture_session(record)

    return {
        "session_id": record.session_id,
        "state": record.state.value,
        "end_time_ns": record.end_time_ns,
    }


@app.get("/api/v1/capture/{session_id}/status")
async def capture_status(session_id: str, request: Request):
    state = request.app.state
    manager: CaptureSessionManager = state.capture_manager
    record = manager.get(session_id)
    if record is None:
        # Fall back to DB for completed / failed sessions that left memory.
        storage: Storage = state.storage
        row = await storage.get_capture_session(session_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"session {session_id} not found")
        return row
    return {
        "session_id": record.session_id,
        "state": record.state.value,
        "stream_key": record.stream_key,
        "range_lease_id": record.range_lease_id,
        "start_time_ns": record.start_time_ns,
        "end_time_ns": record.end_time_ns,
        "first_frame_pts_ns": record.first_frame_pts_ns,
        "iamf_path": str(record.iamf_path) if record.iamf_path else None,
        "youtube_path": str(record.youtube_path) if record.youtube_path else None,
        "error": record.error,
        "created_ns": record.created_ns,
    }


@app.get("/api/v1/capture")
async def capture_list(request: Request):
    state = request.app.state
    manager: CaptureSessionManager = state.capture_manager
    records = manager.list_sessions()
    return [
        {
            "session_id": r.session_id,
            "state": r.state.value,
            "stream_key": r.stream_key,
            "created_ns": r.created_ns,
            "error": r.error,
        }
        for r in records
    ]


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
