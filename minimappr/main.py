"""MinimapPR application entrypoint."""

from __future__ import annotations

import asyncio
import contextvars
import contextlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal
import urllib.error
import urllib.request

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, model_validator
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect

from minimappr.api.binary_ingest import parse_binary_ingest_payload
from minimappr.api.live import LiveEventHub
from minimappr.api.spool_consumer import IngestSpoolConsumer
from minimappr.api.stream_consumer import IngestStreamConsumer, StreamConsumerConfig
from minimappr.classifiers.availability import probe_backends
from minimappr.classifiers.factory import create_context_classifier
from minimappr.classifiers.routing import (
    CONTEXT_LOCALIZED_RENDER,
    load_routing,
    load_routing_file,
    parse_routing_document,
    routing_to_dict,
)
from minimappr.core.config_groups import group_flat_config
from minimappr.core.pipeline_graph import build_pipeline_graph
from minimappr.config import IngestSidecarProcessConfig, IngestSidecarStartupConfig, Settings
from minimappr.settings_store import CONFIG_PATCH_ALLOWLIST, load_overrides, save_overrides
from minimappr.ingest_sidecar_runtime import (
    IngestSidecarRuntimeState as _RuntimeSidecarState,
    build_ingest_stream_consumer as _runtime_build_ingest_stream_consumer,
    build_ingest_sidecar_environment as _runtime_build_ingest_sidecar_environment,
    ensure_ingest_stream_consumer_running as _runtime_ensure_ingest_stream_consumer_running,
    fetch_ingest_sidecar_health as _runtime_fetch_ingest_sidecar_health,
    ingest_stream_consumer_runtime as _runtime_ingest_stream_consumer_runtime,
    ingest_sidecar_is_running as _runtime_ingest_sidecar_is_running,
    ingest_sidecar_process_config as _runtime_ingest_sidecar_process_config,
    ingest_sidecar_startup_config as _runtime_ingest_sidecar_startup_config,
    launch_managed_ingest_sidecar as _runtime_launch_managed_ingest_sidecar,
    probe_ingest_sidecar_ready as _runtime_probe_ingest_sidecar_ready,
    shutdown_managed_ingest_sidecar as _runtime_shutdown_managed_ingest_sidecar,
    should_autostart_ingest_sidecar as _runtime_should_autostart_ingest_sidecar,
    sidecar_classification_window_seconds as _runtime_sidecar_classification_window_seconds,
    sidecar_classifier_render_min_interval_seconds as _runtime_sidecar_classifier_render_min_interval_seconds,
    start_ingest_sidecar as _runtime_start_ingest_sidecar,
    supervise_ingest_sidecar as _runtime_supervise_ingest_sidecar,
    wait_for_ingest_sidecar_ready as _runtime_wait_for_ingest_sidecar_ready,
)
from minimappr.runtime_bootstrap import (
    _ApiOnlyRuntimeTaskHandles,
    _CombinedRuntimeTaskHandles,
    _apply_settings_site_origin_resolution,
    _build_api_only_runtime_federation,
    _build_capture_manager,
    _build_combined_runtime_core_services,
    _build_combined_runtime_federation,
    _build_common_live_runtime_services,
    _build_effector_manager,
    _build_hass_bridge,
    _initialize_storage_and_resolve_site_origin,
    _start_api_only_runtime_services,
    _stop_api_only_runtime_services,
    _stop_combined_ingest_stream_consumer,
    _shutdown_combined_runtime_services,
    _start_combined_runtime_background_tasks,
    _stop_combined_runtime_background_tasks,
    _wire_effector_zone_interlocks,
    _wire_effector_rules_handler,
    _wire_hass_live_event_tee,
    _wire_hass_rules_handler,
)
from minimappr.core.capture_session import (
    CaptureSessionManager,
    CaptureSessionRecord,
    CaptureStartRequest,
    CaptureState,
)
from minimappr.core.ambisonics import (
    AmbisonicSpatialEncoder,
    SoundscapeRenderer,
    SpatialSourceFrame,
    foa_to_5_1,
    wav_multichannel_bytes,
)
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.audio_processing.levels import apply_level_profile
from minimappr.audio_processing.profiles import LISTENING_PROFILE_NAME, load_audio_processing_configuration
from minimappr.audio_processing.wav_serving import listening_wav_bytes, level_report_headers
from minimappr.core.auth import extract_federation_token
from minimappr.core.ble_multilateration import estimate_ble_device_position
from minimappr.core.ble_observations import BleObservationStore
from minimappr.core.ble_tracking import BleTracker
from minimappr.core.bit_report import BITReportEvaluator
from minimappr.core.cluster_registry import ClusterRegistry
from minimappr.core.effectors.registry import EffectorManager
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.environment import LiveEnvironmentProvider
from minimappr.core.federation import ACTIVE_TRACK_STATUSES, FederationCoordinator
from minimappr.core.hass.state_mapper import (
    HassStateSnapshot,
    NodeStateInput,
    SystemStateInput,
    ZoneStateInput,
)
from minimappr.core.hass.topics import is_valid_topic_level
from minimappr.core.hass.track_slots import TrackSlotCandidate
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo_restriction import excludes_audio_ingest
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.logging_ring import install_global as install_log_ring, process_start_ns
from minimappr.core.rules import ConfigRuleEngine, RuleDef, default_rules_as_dicts
from minimappr.core.site_origin import (
    SOURCE_GPS_ANCHOR,
    SOURCE_PERSISTED,
    SiteOriginResolution,
    origins_differ,
    persist_site_origin,
)
from minimappr.core.zones import ZoneMatcher
from minimappr.core import system_info
from minimappr.calibration.bundle import build_ground_truth_payload, write_bundle_zip
from minimappr.calibration.embeddings import extract_embedding_npy
from minimappr.models import (
    AlertStatus,
    BITReport,
    CalibrationGroundTruthEvent,
    CalibrationGroundTruthIn,
    CalibrationGroundTruthUpdate,
    BITReportIn,
    BITStatus,
    BITTestResult,
    BITType,
    BleIngestRequest,
    ClassifierRoutingConfigResponse,
    ClassifierRoutingConfigUpdate,
    ClusterSpec,
    ContextSnapshot,
    CopStatusResponse,
    DetectionReviewState,
    DetectionReviewUpdateRequest,
    EnvironmentSampleIn,
    FederationAck,
    FederationHeartbeat,
    FederationStatusResponse,
    HassBridgeStatusResponse,
    FederationTrackSnapshot,
    FusionStatusResponse,
    MapOverlayKind,
    MapOverlaySpec,
    MapOverlayUpdate,
    GeoPoint,
    IngestFrameRequest,
    IngestFrameResponse,
    MicView,
    NodeCapability,
    NodeHealthStatus,
    NodeAudioOverride,
    NodeOverrides,
    NodePatchRequest,
    NodeRegistrationRequest,
    NodeSafetyConfig,
    NodeSpec,
    PipelineGraph,
    PipelineNodeView,
    PipelineNodesResponse,
    PipelineStageView,
    RulesConfigResponse,
    RulesConfigUpdate,
    ReviewedDetectionExportItem,
    ReviewedDetectionExportPackage,
    StoreForwardBufferedFrameResponse,
    StoreForwardIngestRequest,
    StoreForwardIngestResponse,
    TrackState,
    TrainingExampleKind,
    Vec3,
    ZoneOccupancyState,
    ZoneSpec,
)
from minimappr.training_dataset import (
    TrainingDatasetError,
    delete_training_example_files,
    materialize_training_example,
)
from minimappr.storage.db import Storage
from minimappr.utils.audio import mono_mix, read_wav_mono


logger = logging.getLogger(__name__)
frontend_dir = Path(__file__).parent / "frontend"
_INGEST_STREAM_CONSUMER_WATCHDOG_INTERVAL_SECONDS = 1.0
_INGEST_PATH_PREFIXES = (
    "/api/v1/ingest",
    "/api/v1/capture",
    "/api/v1/recordings",
    "/api/v1/fusion/status",
    "/api/v1/system/diagnostics",
    "/api/v1/system/logs",
)


# Default ingest concurrency ceiling. Mirrors the Rust sidecar's bounded MPSC
# strategy ([ingest_backend.rs] raw_manifest_channel_capacity = 2048) but
# scaled down because each Python ingest task does much heavier work (numpy
# merge + classifier wakeup) than a Rust ingest task (just queue + write).
# Tuned in conjunction with the to_thread merge in audio_buffer.py.
_DEFAULT_INGEST_MAX_CONCURRENT = 64


class _IngestConcurrencyLimit:
    """Bounded-concurrency gate on the FastAPI ingest endpoints.

    Mirrors the Rust sidecar's HTTP-503-with-`Retry-After` shape at
    [main.rs] so a burst of slow concurrent requests cannot saturate the
    worker pool. We *shed* immediately on overload (no buffering) — the
    firmware client retries on `Retry-After`, which is the same wait-and-
    retry contract the sidecar already enforces, just on the Python lane.

    Single-threaded asyncio makes the counter-check and increment safe
    without a lock (no yield points between them).
    """

    def __init__(self, max_concurrent: int, lease_timeout_seconds: float = 5.0) -> None:
        self._max = max(1, int(max_concurrent))
        self._lease_timeout_seconds = max(0.1, float(lease_timeout_seconds))
        self._active_leases: dict[int, float] = {}
        self._next_lease_id = 0
        self._lease_context: contextvars.ContextVar[int | None] = contextvars.ContextVar(
            "minimappr_ingest_lease_id",
            default=None,
        )
        # Atomic-from-asyncio's perspective: count never escapes the event loop.
        # `total_admissions` / `total_shed` are exposed via /api/v1/system/diagnostics
        # so operators can confirm backpressure is firing under load.
        self.total_admissions = 0
        self.total_shed = 0

    @property
    def max_concurrent(self) -> int:
        return self._max

    @property
    def active(self) -> int:
        self._evict_expired_leases(time.monotonic())
        return len(self._active_leases)

    async def __aenter__(self) -> "_IngestConcurrencyLimit":
        now = time.monotonic()
        self._evict_expired_leases(now)
        if len(self._active_leases) >= self._max:
            self.total_shed += 1
            raise HTTPException(
                status_code=503,
                detail=(
                    f"ingest backpressure: {len(self._active_leases)}/{self._max} slots in use"
                ),
                headers={"Retry-After": "1"},
            )
        self._next_lease_id += 1
        lease_id = self._next_lease_id
        self._active_leases[lease_id] = now + self._lease_timeout_seconds
        self._lease_context.set(lease_id)
        self.total_admissions += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        lease_id = self._lease_context.get()
        if lease_id is not None:
            self._active_leases.pop(lease_id, None)
            self._lease_context.set(None)

    def _evict_expired_leases(self, now: float) -> None:
        expired = [
            lease_id
            for lease_id, deadline in self._active_leases.items()
            if deadline <= now
        ]
        for lease_id in expired:
            self._active_leases.pop(lease_id, None)


def _require_ingest_concurrency(request: Request) -> _IngestConcurrencyLimit:
    """Fetch the per-app ingest concurrency limiter. Falls through to a fresh
    unbounded limiter only in tests where lifespan setup is skipped — in
    production, the lifespan binds the configured limit on app.state."""
    limit = getattr(request.app.state, "ingest_concurrency", None)
    if isinstance(limit, _IngestConcurrencyLimit):
        return limit
    # Defensive fallback for tests that bypass lifespan: a limiter with the
    # default ceiling, attached to app.state so subsequent requests share it.
    limit = _IngestConcurrencyLimit(_DEFAULT_INGEST_MAX_CONCURRENT)
    request.app.state.ingest_concurrency = limit
    return limit


def _ingest_request_timeout_seconds(request: Request) -> float:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    if settings is None:
        return 5.0
    return settings.ingest_request_timeout_seconds


async def _run_ingest_with_timeout(request: Request, operation) -> Any:
    return await asyncio.wait_for(
        operation,
        timeout=_ingest_request_timeout_seconds(request),
    )


def _default_sidecar_classifier_command_json(settings: "Settings") -> str | None:
    # Availability is evaluated inside the helper's deployment environment.
    # The API process may intentionally have fewer model extras installed than
    # the ingest/classifier process, so probing imports here creates false
    # negatives and silently disables the configured classifier bridge.
    del settings
    return json.dumps([sys.executable, "-m", "minimappr.sidecar_classifier_helper"])


def _build_runtime_classifier(settings: Settings):
    return create_context_classifier(settings, CONTEXT_LOCALIZED_RENDER)


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
    except (TypeError, ValueError) as exc:
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


async def _ble_tracking_loop(app: FastAPI) -> None:
    """Periodically trilaterate BLE observations into first-class tracks.

    Runs in whichever process owns the BLE observation store + tracks storage
    (both the combined runtime and the API-only role). Direct-broadcasts each
    updated track so WS clients see BLE devices in all deployment modes.
    """
    state = app.state
    settings: Settings = state.settings
    ble_tracker: BleTracker = state.ble_tracker
    period_s = max(float(settings.ble_tracking_period_s), 0.1)
    while True:
        try:
            now_ns = time.time_ns()
            node_positions = await _ble_node_positions(state)
            await ble_tracker.run_tick(
                storage=state.storage,
                observation_store=_ble_observation_store(state),
                node_positions=node_positions,
                now_ns=now_ns,
                live_hub=state.live_hub,
                coordinate_frame=getattr(state, "coordinate_frame", None),
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("BLE tracking tick failed")
        await asyncio.sleep(period_s)


def _apply_site_origin_resolution(state, resolved_site_origin) -> None:
    settings: Settings = state.settings
    _apply_settings_site_origin_resolution(settings, resolved_site_origin)
    state.site_origin_resolution_source = resolved_site_origin.source
    state.site_origin_contributing_node_ids = resolved_site_origin.contributing_node_ids
    state.site_origin_anchored = resolved_site_origin.is_anchored


def _clear_transient_ingest_runtime_state(state) -> None:
    _clear_state_attrs(
        state,
        "ingest_stream_consumer",
        "ingest_stream_consumer_watchdog_task",
        "ingest_spool_tasks",
    )


def _clear_bound_runtime_state(state) -> None:
    _clear_state_attrs(
        state,
        "settings",
        "storage",
        "registry",
        "cluster_registry",
        "ble_observation_store",
        "ble_tracker",
        "audio_buffer",
        "localizer",
        "classifier",
        "tracker",
        "live_hub",
        "coordinate_frame",
        "zone_matcher",
        "environment_provider",
        "fusion_node",
        "ingest_transport",
        "federation",
        "bit_evaluator",
        "diagnostics",
        "cleanup_service",
        "ingest_spool_consumer",
        "sidecar_state",
        "capture_manager",
        "ingest_concurrency",
        "ingest_stream_consumer_enabled",
        "site_origin_resolution_source",
        "site_origin_contributing_node_ids",
        "effector_manager",
        "hass_bridge",
    )


def _ensure_lifespan_runtime_directories(settings: Settings) -> None:
    settings.federation_peers_config_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    settings.large_artifact_dir.mkdir(parents=True, exist_ok=True)
    settings.map_overlay_dir.mkdir(parents=True, exist_ok=True)
    settings.effector_snapshot_dir.mkdir(parents=True, exist_ok=True)


def _prepare_lifespan_runtime(state, settings: Settings) -> None:
    install_log_ring()
    _clear_bound_runtime_state(state)
    _clear_transient_ingest_runtime_state(state)
    _ensure_lifespan_runtime_directories(settings)


def _bind_runtime_state(state, *, resolved_site_origin=None, **state_fields) -> None:
    for name, value in state_fields.items():
        setattr(state, name, value)
    if resolved_site_origin is not None:
        _apply_site_origin_resolution(state, resolved_site_origin)


def _request_hass_reconcile(state) -> None:
    """Nudge the HA bridge after zone/node CRUD so entities appear/vanish now.

    Synchronous by design: these are request handlers, and the bridge only sets a
    flag its publisher picks up next cycle. The periodic reconcile is the safety
    net if a call site is ever missed.
    """
    bridge = getattr(state, "hass_bridge", None)
    if bridge is not None:
        bridge.request_reconcile()


def _warn_when_direct_ingest_falls_back(settings: Settings, sidecar_state: _SidecarState) -> None:
    if settings.direct_ingest_enabled or sidecar_state.status == "running":
        return
    logger.warning(
        "Direct ingest is disabled but sidecar is not running (status=%s). "
        "Falling back to direct ingest to avoid node ingest outage.",
        sidecar_state.status,
    )


async def _rebind_site_origin(app: FastAPI, resolved_site_origin) -> None:
    """Swap the process onto a new site origin.

    In-memory node position estimators hold ENU metres and are reprojected into
    the new frame rather than reset, so a stationary node keeps however many
    hours of GNSS averaging it has accumulated. Persisted checkpoints are stored
    geodetically and need no migration at all.
    """
    state = app.state
    settings: Settings = state.settings
    candidate_settings = replace(
        settings,
        site_origin_lat=resolved_site_origin.origin.lat,
        site_origin_lon=resolved_site_origin.origin.lon,
        site_origin_alt_m=resolved_site_origin.origin.alt_m,
    )
    new_classifier = _build_runtime_classifier(candidate_settings)
    new_coordinate_frame = LocalCoordinateFrame(
        origin=resolved_site_origin.origin,
        mode=settings.coordinate_mode,
    )
    previous_coordinate_frame = state.coordinate_frame
    previous_classifier = state.classifier
    fusion_node = getattr(state, "fusion_node", None)
    if fusion_node is not None:
        fusion_node.rebind_runtime_dependencies(
            classifier=new_classifier,
            coordinate_frame=new_coordinate_frame,
        )
        fusion_node.reproject_position_estimators(previous_coordinate_frame)
    state.classifier = new_classifier
    state.coordinate_frame = new_coordinate_frame
    state.diagnostics.replace_classifier(new_classifier)
    _apply_site_origin_resolution(state, resolved_site_origin)

    if previous_classifier is not None and previous_classifier is not new_classifier:
        try:
            previous_classifier.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Previous classifier close failed after site-origin change: %s", exc)


def _make_site_origin_anchor(app: FastAPI):
    """Build the ingest callback that anchors the site origin on a trusted GPS fix.

    Anchoring is one-shot per site: the first node to report a real fix defines
    the origin, it is persisted so every process and every restart agrees, and
    the callback uninstalls itself. Sites with no GPS simply never anchor and
    keep running on the configured fallback.
    """
    lock = asyncio.Lock()

    async def anchor(node_id: str, geo: GeoPoint) -> None:
        state = app.state
        async with lock:
            if getattr(state, "site_origin_anchored", False):
                return
            resolution = SiteOriginResolution(
                origin=geo,
                source=SOURCE_GPS_ANCHOR,
                contributing_node_ids=(node_id,),
            )
            # Persist before adopting. This runs on the ingest frame path, so a
            # storage failure must neither reject the frame nor latch the anchor
            # shut — leaving it armed lets the next frame retry.
            try:
                await persist_site_origin(state.storage, resolution)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Persisting GPS-anchored site origin from node %s failed; "
                    "will retry on the next frame: %s",
                    node_id,
                    exc,
                )
                return
            state.site_origin_anchored = True
            previous = GeoPoint(
                lat=state.settings.site_origin_lat,
                lon=state.settings.site_origin_lon,
                alt_m=state.settings.site_origin_alt_m,
            )
            if origins_differ(previous, geo):
                await _rebind_site_origin(app, resolution)
            else:
                _apply_site_origin_resolution(state, resolution)
            _install_site_origin_anchor(app)
            logger.info(
                "Anchored site origin from node %s GPS fix: lat=%.6f lon=%.6f alt=%.2f "
                "(was lat=%.6f lon=%.6f)",
                node_id,
                geo.lat,
                geo.lon,
                geo.alt_m,
                previous.lat,
                previous.lon,
            )

    return anchor


def _install_site_origin_anchor(app: FastAPI) -> None:
    """Arm or disarm GPS anchoring to match the current anchored state."""
    state = app.state
    fusion_node = getattr(state, "fusion_node", None)
    if fusion_node is None:
        return
    if getattr(state, "site_origin_anchored", False):
        fusion_node.set_site_origin_anchor(None)
        return
    fusion_node.set_site_origin_anchor(_make_site_origin_anchor(app))
    logger.info(
        "Site origin is un-anchored (source=%s); awaiting a trusted GPS fix from any node",
        getattr(state, "site_origin_resolution_source", "unknown"),
    )


async def _sync_site_origin_from_storage(app: FastAPI) -> None:
    """Adopt an origin anchored by the ingest process.

    Only the ingest process sees frames, so it is the one that anchors. The api
    process reads the persisted result and rebinds, which is what keeps the two
    from drifting into disagreeing coordinate frames.
    """
    state = app.state
    settings: Settings = state.settings
    if settings.site_origin_source == "manual":
        return
    persisted = await state.storage.get_site_origin()
    if persisted is None:
        return
    origin = GeoPoint(
        lat=float(persisted["lat"]),
        lon=float(persisted["lon"]),
        alt_m=float(persisted["alt_m"]),
    )
    current = GeoPoint(
        lat=settings.site_origin_lat,
        lon=settings.site_origin_lon,
        alt_m=settings.site_origin_alt_m,
    )
    if not origins_differ(current, origin) and getattr(state, "site_origin_anchored", False):
        return
    await _rebind_site_origin(
        app,
        SiteOriginResolution(
            origin=origin,
            source=SOURCE_PERSISTED,
            contributing_node_ids=tuple(persisted.get("contributing_node_ids") or ()),
        ),
    )
    logger.info(
        "Adopted persisted site origin: lat=%.6f lon=%.6f alt=%.2f", origin.lat, origin.lon, origin.alt_m
    )


async def _api_live_db_poll_loop(app: FastAPI) -> None:
    """Bridge ingest-process DB writes into API-process websocket updates."""
    state = app.state
    settings: Settings = state.settings
    last_detection_ts = time.time_ns()
    last_track_ts = last_detection_ts
    last_environment_ts = 0
    seen_detection_ids: set[str] = set()
    seen_track_ids: set[str] = set()
    while True:
        try:
            await _sync_site_origin_from_storage(app)

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

            # Hydrate the API-process environment provider from storage. In the
            # split api/ingest deployment, environment ingest is proxied to the
            # ingest worker, so this process's provider would otherwise stay
            # empty and /api/v1/environment/current would report static_fallback.
            environment_provider = getattr(state, "environment_provider", None)
            if environment_provider is not None and hasattr(
                environment_provider, "ingest_sample"
            ):
                latest_environment = await state.storage.list_latest_environment_per_node()
                for reading in latest_environment:
                    timestamp_ns = int(reading.get("timestamp_ns") or 0)
                    if timestamp_ns <= last_environment_ts:
                        continue
                    position = reading.get("position_m")
                    location_m = (
                        tuple(float(value) for value in position)
                        if position is not None
                        else None
                    )
                    environment_provider.ingest_sample(
                        node_id=str(reading.get("node_id") or ""),
                        timestamp_ns=timestamp_ns,
                        temperature_c=reading.get("temperature_c"),
                        humidity_fraction=reading.get("humidity_fraction"),
                        pressure_pa=reading.get("pressure_pa"),
                        wind_speed_mps=reading.get("wind_speed_mps"),
                        wind_dir_deg=reading.get("wind_dir_deg"),
                        solar_lux=reading.get("solar_lux"),
                        location_m=location_m,
                        metadata=reading.get("metadata") or {},
                    )
                    last_environment_ts = max(last_environment_ts, timestamp_ns)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("API live DB poll failed: %s", exc)
        await asyncio.sleep(1.0)


# Keep the historical private name stable for local call sites and tests while
# delegating the implementation to the dedicated sidecar runtime module.
_SidecarState = _RuntimeSidecarState


class _EnvironmentIngestSample(BaseModel):
    node_id: str
    sample: EnvironmentSampleIn


class _EnvironmentIngestBody(BaseModel):
    samples: list[_EnvironmentIngestSample]


def _ble_observation_store(state) -> BleObservationStore:
    store = getattr(state, "ble_observation_store", None)
    if store is None:
        store = BleObservationStore()
        setattr(state, "ble_observation_store", store)
    return store


def _ingest_sidecar_is_running(state) -> bool:
    return _runtime_ingest_sidecar_is_running(state)


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
    return _runtime_should_autostart_ingest_sidecar(settings)


def _ingest_sidecar_startup_config(settings) -> IngestSidecarStartupConfig:
    return _runtime_ingest_sidecar_startup_config(settings)


def _ingest_sidecar_process_config(settings) -> IngestSidecarProcessConfig:
    return _runtime_ingest_sidecar_process_config(settings)


def _sidecar_classification_window_seconds(settings) -> float:
    return _runtime_sidecar_classification_window_seconds(settings)


def _sidecar_classifier_render_min_interval_seconds(
    settings,
    *,
    classification_window_seconds: float,
) -> float:
    return _runtime_sidecar_classifier_render_min_interval_seconds(
        settings,
        classification_window_seconds=classification_window_seconds,
    )


def _sidecar_classifier_command_json(settings) -> str | None:
    classifier_command_json = os.environ.get("MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON")
    if classifier_command_json is not None:
        return classifier_command_json
    return _default_sidecar_classifier_command_json(settings)


def _build_ingest_sidecar_environment(settings) -> dict[str, str]:
    return _runtime_build_ingest_sidecar_environment(
        settings,
        default_classifier_command_json_builder=_default_sidecar_classifier_command_json,
    )


def _ingest_runtime_base_url(settings: "Settings") -> str:
    return settings.ingest_base_url.rstrip("/")


def _should_proxy_ingest_to_python_worker(state) -> bool:
    settings = state.settings
    return (
        getattr(settings, "process_role", "combined") == "api"
        and getattr(settings, "ingest_backend", "python") == "python"
        and settings.ingest_port != settings.port
        and os.getenv("MINIMAPPR_INGEST_PORT") is not None
    )


async def _proxy_json_to_python_worker(
    state,
    *,
    method: str,
    endpoint_path: str,
    json_body: object | None = None,
) -> dict | list:
    settings = state.settings
    if settings.ingest_port == settings.port:
        raise HTTPException(
            status_code=503,
            detail="Ingest proxy is misconfigured: ingest_port matches API port",
        )
    target_url = f"{_ingest_runtime_base_url(settings)}{endpoint_path}"
    payload = None if json_body is None else json.dumps(json_body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}

    def _request() -> tuple[int, bytes]:
        request = urllib.request.Request(
            target_url,
            data=payload,
            method=method,
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=30.0) as response:
            status = int(getattr(response, "status", 200))
            return status, response.read()

    try:
        status, response_payload = await asyncio.to_thread(_request)
    except urllib.error.HTTPError as exc:
        detail = exc.reason or "Ingest worker error"
        error_payload = exc.read()
        try:
            parsed = json.loads(error_payload)
            if isinstance(parsed, dict) and parsed.get("detail"):
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"Ingest worker unreachable: {exc}") from exc

    if not response_payload:
        return {}
    try:
        decoded = json.loads(response_payload.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Invalid JSON response from ingest worker") from exc
    if not isinstance(decoded, (dict, list)):
        raise HTTPException(status_code=502, detail="Unexpected response shape from ingest worker")
    if status >= 400:
        detail = decoded.get("detail") if isinstance(decoded, dict) else "Ingest worker error"
        raise HTTPException(status_code=status, detail=str(detail or "Ingest worker error"))
    return decoded


async def _proxy_ingest_post(
    state,
    *,
    endpoint_path: str,
    body: bytes,
    content_type: str,
) -> dict:
    settings = state.settings
    if settings.ingest_port == settings.port:
        raise HTTPException(
            status_code=503,
            detail="Ingest proxy is misconfigured: ingest_port matches API port",
        )
    target_url = f"{_ingest_runtime_base_url(settings)}{endpoint_path}"

    def _post() -> tuple[int, bytes]:
        request = urllib.request.Request(
            target_url,
            data=body,
            method="POST",
            headers={"Content-Type": content_type},
        )
        with urllib.request.urlopen(request, timeout=15.0) as response:
            status = int(getattr(response, "status", 200))
            payload = response.read()
            return status, payload

    try:
        status, payload = await asyncio.to_thread(_post)
    except urllib.error.HTTPError as exc:
        detail = f"Ingest worker returned HTTP {exc.code}"
        try:
            error_payload = exc.read().decode("utf-8")
            parsed = json.loads(error_payload)
            if isinstance(parsed, dict) and parsed.get("detail"):
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"Ingest worker unreachable: {exc}") from exc

    if not payload:
        return {}
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Invalid JSON response from ingest worker") from exc
    if not isinstance(decoded, dict):
        raise HTTPException(status_code=502, detail="Unexpected response shape from ingest worker")
    if status >= 400:
        raise HTTPException(status_code=status, detail=str(decoded.get("detail") or "Ingest worker error"))
    return decoded


def _capture_pipeline_status(state) -> tuple[bool, str | None]:
    settings: Settings | None = getattr(state, "settings", None)
    if settings is None:
        return False, "Capture is unavailable because runtime settings are not initialized"

    if settings.ingest_backend == "python":
        if _should_proxy_ingest_to_python_worker(state):
            return True, None
        if getattr(state, "audio_buffer", None) is not None:
            return True, None
        return False, "Python ingest capture requires the combined process role or a configured ingest worker proxy"

    if settings.ingest_backend != "rust":
        return False, f"Unsupported ingest backend for capture: {settings.ingest_backend}"

    if getattr(state, "audio_buffer", None) is not None:
        return True, None

    return (
        False,
        "Rust ingest capture currently requires the combined process role so raw audio can be mirrored into the in-memory live buffer before IAMF post-processing",
    )


def _capture_pipeline_available(state) -> bool:
    return _capture_pipeline_status(state)[0]


def _capture_pipeline_unavailable_reason(state) -> str:
    return _capture_pipeline_status(state)[1] or "Capture pipeline is unavailable"


async def _capture_channel_sensor_ids(storage: Storage, stream_key: str) -> list[str]:
    node_row = await storage.get_node_by_id(stream_key)
    channel_sensor_ids = _sensor_ids_from_node_row(node_row) if node_row is not None else []
    if channel_sensor_ids:
        return channel_sensor_ids
    return [f"{stream_key}:ch{i}" for i in range(4)]


async def _build_capture_start_request(
    state,
    *,
    stream_key: str,
    work_dir_path: Path,
    max_duration_s: float,
    video_source: str | None,
    libcamera_mode: bool = False,
    deployment_profile: str = "auto",
    record_video: bool = True,
    include_iamf: bool = True,
    capture_kind: str = "recording",
) -> CaptureStartRequest:
    settings: Settings = state.settings
    storage: Storage = state.storage
    audio_buffer = getattr(state, "audio_buffer", None)

    if audio_buffer is not None:
        if capture_kind == "calibration":
            # Capture every registered audio-capable node through one session
            # buffer; extraction is per node because sample rates may differ.
            node_channel_map: dict[str, list[str]] = {}
            for node in await storage.list_nodes(limit=4096):
                sensor_ids = _sensor_ids_from_node_row(node)
                if sensor_ids:
                    node_channel_map[str(node["id"])] = sensor_ids
            if not node_channel_map:
                raise HTTPException(
                    status_code=409,
                    detail="Calibration capture requires at least one registered node with sensor offsets",
                )
            return CaptureStartRequest(
                stream_key="calibration",
                work_dir=work_dir_path,
                sidecar_url=None,
                multi_sensor_buffer=audio_buffer,
                max_duration_s=min(max_duration_s, settings.calibration_max_duration_s),
                record_video=False,
                include_iamf=False,
                capture_kind="calibration",
                node_channel_map=node_channel_map,
            )
        return CaptureStartRequest(
            stream_key=stream_key,
            work_dir=work_dir_path,
            sidecar_url=None,
            multi_sensor_buffer=audio_buffer,
            channel_sensor_ids=await _capture_channel_sensor_ids(storage, stream_key),
            max_duration_s=max_duration_s,
            video_source=video_source,
            libcamera_mode=libcamera_mode,
            deployment_profile=deployment_profile,
            record_video=record_video,
            include_iamf=include_iamf,
        )

    if settings.ingest_backend == "python":
        raise HTTPException(
            status_code=503,
            detail="Python ingest capture requires the combined process role or a configured ingest worker proxy",
        )

    raise HTTPException(
        status_code=503,
        detail=_capture_pipeline_unavailable_reason(state),
    )


def _has_live_ingest_runtime(state) -> bool:
    settings = getattr(state, "settings", None)
    if settings is not None and getattr(settings, "process_role", "combined") == "api":
        return False
    return hasattr(state, "registry") and hasattr(state, "audio_buffer")


def _clear_state_attrs(state, *names: str) -> None:
    for name in names:
        with contextlib.suppress(AttributeError, KeyError):
            delattr(state, name)


def _sidecar_stream_consumer_snapshots(state) -> dict[str, object]:
    consumer = getattr(state, "ingest_stream_consumer", None)
    snapshot_nodes = getattr(consumer, "snapshot_nodes", None)
    if not callable(snapshot_nodes):
        return {}
    snapshots = snapshot_nodes()
    return snapshots if isinstance(snapshots, dict) else {}


def _node_row_from_sidecar_snapshot(snapshot: object) -> dict[str, object] | None:
    node_payload = getattr(snapshot, "node_payload", None)
    last_seen_ns = getattr(snapshot, "last_seen_ns", None)
    if not isinstance(node_payload, dict) or not isinstance(last_seen_ns, int):
        return None
    row = dict(node_payload)
    row["last_seen_ns"] = last_seen_ns
    return row


def _merge_nodes_with_sidecar_snapshots(
    nodes: list[dict],
    sidecar_snapshots: dict[str, object],
    *,
    limit: int,
) -> list[dict]:
    nodes_by_id = {
        str(node.get("id")): node
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    for node_id, snapshot in sidecar_snapshots.items():
        snapshot_row = _node_row_from_sidecar_snapshot(snapshot)
        if snapshot_row is None:
            continue
        existing = nodes_by_id.get(node_id)
        if existing is None:
            nodes.append(snapshot_row)
            nodes_by_id[node_id] = snapshot_row
            continue
        existing_last_seen_ns = existing.get("last_seen_ns")
        existing_last_seen_ns = existing_last_seen_ns if isinstance(existing_last_seen_ns, int) else 0
        snapshot_last_seen_ns = int(snapshot_row.get("last_seen_ns") or 0)
        if snapshot_last_seen_ns >= existing_last_seen_ns:
            existing.update(snapshot_row)
    nodes.sort(key=lambda node: str(node.get("id") or ""))
    return nodes[:limit]


def _sidecar_snapshot_audio_debug(
    snapshot: object,
    *,
    now_ns: int,
    degraded_after_seconds: float,
) -> dict[str, object] | None:
    node_payload = getattr(snapshot, "node_payload", None)
    if not isinstance(node_payload, dict):
        return None
    sensor_offsets_m = node_payload.get("sensor_offsets_m")
    sensor_count = len(sensor_offsets_m) if isinstance(sensor_offsets_m, list) else 0
    last_sample_time_ns = getattr(snapshot, "last_sample_time_ns", None)
    age_seconds = None
    if isinstance(last_sample_time_ns, int):
        age_seconds = max(0.0, (now_ns - last_sample_time_ns) / 1_000_000_000.0)
    if age_seconds is None:
        status = "external_ingest_process"
    elif age_seconds <= degraded_after_seconds:
        status = "recent"
    else:
        status = "stale"
    sample_rate_hz = getattr(snapshot, "sample_rate_hz", None)
    active_sensor_count = getattr(snapshot, "active_sensor_count", None)
    rms = getattr(snapshot, "rms", None)
    return {
        "sensor_count": sensor_count,
        "active_sensor_count": int(active_sensor_count or 0),
        "sample_rate_hz": sample_rate_hz if isinstance(sample_rate_hz, int) else None,
        "last_sample_time_ns": last_sample_time_ns if isinstance(last_sample_time_ns, int) else None,
        "age_seconds": age_seconds,
        "rms": rms if isinstance(rms, (int, float)) else None,
        "recent_coverage_ratio": None,
        "recent_missing_ratio": None,
        "recent_max_gap_seconds": None,
        "max_buffer_samples": None,
        "max_buffer_seconds": None,
        "status": status,
    }


def _sidecar_snapshot_latest_environment(snapshot: object, *, node_id: str) -> dict[str, object] | None:
    latest_environment = getattr(snapshot, "latest_environment", None)
    if not isinstance(latest_environment, dict) or not latest_environment:
        return None
    payload = dict(latest_environment)
    payload.setdefault("node_id", node_id)
    return payload


def _ingest_stream_consumer_runtime(state):
    return _runtime_ingest_stream_consumer_runtime(
        state,
        ingest_runtime_base_url_builder=_ingest_runtime_base_url,
        stream_consumer_config_class=StreamConsumerConfig,
    )


def _build_ingest_stream_consumer(state) -> IngestStreamConsumer | None:
    return _runtime_build_ingest_stream_consumer(
        state,
        ingest_stream_consumer_class=IngestStreamConsumer,
        ingest_runtime_base_url_builder=_ingest_runtime_base_url,
        stream_consumer_config_class=StreamConsumerConfig,
    )


async def _ensure_ingest_stream_consumer_running(state) -> bool:
    return await _runtime_ensure_ingest_stream_consumer_running(
        state,
        clear_state_attrs=_clear_state_attrs,
        ingest_stream_consumer_class=IngestStreamConsumer,
        ingest_runtime_base_url_builder=_ingest_runtime_base_url,
        logger=logger,
        stream_consumer_config_class=StreamConsumerConfig,
    )


async def _maintain_ingest_stream_consumer(app: FastAPI) -> None:
    while True:
        await _ensure_ingest_stream_consumer_running(app.state)
        await asyncio.sleep(_INGEST_STREAM_CONSUMER_WATCHDOG_INTERVAL_SECONDS)


def _heartbeat_health_status(
    *,
    last_seen_ns: int,
    now_ns: int,
    degraded_after_seconds: float,
    offline_after_seconds: float,
) -> str:
    age_s = max(0.0, (now_ns - last_seen_ns) / 1_000_000_000.0)
    if age_s >= offline_after_seconds:
        return NodeHealthStatus.OFFLINE.value
    if age_s >= degraded_after_seconds:
        return NodeHealthStatus.DEGRADED.value
    return NodeHealthStatus.ONLINE.value


async def _apply_runtime_health_statuses(
    nodes: list[dict],
    *,
    bit_evaluator: BITReportEvaluator,
    now_ns: int,
    degraded_after_seconds: float,
    offline_after_seconds: float,
) -> None:
    for node in nodes:
        node_id = str(node.get("id") or "")
        last_seen_ns = int(node.get("last_seen_ns") or 0)
        heartbeat_health = _heartbeat_health_status(
            last_seen_ns=last_seen_ns,
            now_ns=now_ns,
            degraded_after_seconds=degraded_after_seconds,
            offline_after_seconds=offline_after_seconds,
        )
        node["health_status"] = await bit_evaluator.derive_health_status(
            node_id=node_id,
            heartbeat_health=heartbeat_health,
            now_ns=now_ns,
        )


def _sensor_ids_from_node_row(node: dict) -> list[str]:
    offsets = node.get("sensor_offsets_m")
    if not isinstance(offsets, list):
        return []
    return [f"{node['id']}:ch{index}" for index in range(len(offsets))]


def _fetch_ingest_sidecar_health(port: int, timeout_seconds: float = 0.5) -> dict[str, object] | None:
    return _runtime_fetch_ingest_sidecar_health(port, timeout_seconds)


def _probe_ingest_sidecar_ready(port: int, timeout_seconds: float = 0.5) -> bool:
    return _runtime_probe_ingest_sidecar_ready(port, timeout_seconds)


async def _wait_for_ingest_sidecar_ready(
    process: "asyncio.subprocess.Process",
    *,
    port: int,
    timeout_seconds: float = 5.0,
    poll_interval_seconds: float = 0.1,
    probe_ready: Callable[[int], bool] | None = None,
) -> None:
    if probe_ready is None:
        probe_ready = _probe_ingest_sidecar_ready
    await _runtime_wait_for_ingest_sidecar_ready(
        process,
        port=port,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        probe_ready=probe_ready,
    )


async def _supervise_ingest_sidecar(
    settings: "Settings",
    initial_process: "asyncio.subprocess.Process",
    state: _SidecarState,
) -> None:
    await _runtime_supervise_ingest_sidecar(
        settings,
        initial_process,
        state,
        start_ingest_sidecar=_start_ingest_sidecar,
        logger=logger,
    )


def _count_failed_spool_manifest_items(spool_dir: "Path") -> int:
    """Count manifest files (.json) in the spool failed directory."""
    failed_dir = spool_dir / "failed"
    if not failed_dir.exists():
        return 0
    return sum(1 for p in failed_dir.glob("*.json") if p.is_file())


async def _start_ingest_sidecar(
    settings: "Settings",
) -> "asyncio.subprocess.Process | None":
    return await _runtime_start_ingest_sidecar(
        settings,
        default_classifier_command_json_builder=_default_sidecar_classifier_command_json,
        probe_ready=_probe_ingest_sidecar_ready,
        wait_for_ready=_wait_for_ingest_sidecar_ready,
        create_subprocess_exec=asyncio.create_subprocess_exec,
        logger=logger,
    )


async def _launch_managed_ingest_sidecar(
    settings: "Settings",
    *,
    startup_failure_log_message: str,
) -> tuple[_SidecarState, asyncio.Task | None]:
    return await _runtime_launch_managed_ingest_sidecar(
        settings,
        logger=logger,
        sidecar_state_class=_SidecarState,
        start_ingest_sidecar=_start_ingest_sidecar,
        startup_failure_log_message=startup_failure_log_message,
        supervise_ingest_sidecar=_supervise_ingest_sidecar,
    )


async def _shutdown_managed_ingest_sidecar(
    sidecar_state: _SidecarState,
    sidecar_supervision_task: asyncio.Task | None,
    *,
    force_kill_on_timeout: bool,
    shutdown_timeout_seconds: float,
) -> None:
    await _runtime_shutdown_managed_ingest_sidecar(
        sidecar_state,
        sidecar_supervision_task,
        force_kill_on_timeout=force_kill_on_timeout,
        logger=logger,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
    )


@asynccontextmanager
async def _api_only_lifespan(app: FastAPI, settings: Settings):
    """Initialize the API/UI process without DSP, classifiers, or live ingest."""
    _prepare_lifespan_runtime(app.state, settings)

    storage, resolved_site_origin = await _initialize_storage_and_resolve_site_origin(
        settings,
        log_resolution=False,
    )

    common_live_runtime_services = _build_common_live_runtime_services(settings, storage=storage)
    federation = _build_api_only_runtime_federation(
        settings,
        live_hub=common_live_runtime_services.live_hub,
    )
    effector_manager = _build_effector_manager(
        settings,
        storage=storage,
        live_hub=common_live_runtime_services.live_hub,
    )
    _wire_effector_zone_interlocks(
        effector_manager,
        coordinate_frame=common_live_runtime_services.coordinate_frame,
        zone_matcher=ZoneMatcher(storage=storage),
    )

    capture_manager = _build_capture_manager(
        settings,
        live_hub=common_live_runtime_services.live_hub,
        sidecar_url=_ingest_runtime_base_url(settings),
        storage=storage,
        coordinate_frame=common_live_runtime_services.coordinate_frame,
        environment_provider=common_live_runtime_services.environment_provider,
    )

    sidecar_state, sidecar_supervision_task = await _launch_managed_ingest_sidecar(
        settings,
        startup_failure_log_message="Ingest sidecar startup failed in API role",
    )

    registry = NodeRegistry()
    await registry.load_overrides(await storage.list_node_overrides())

    _bind_runtime_state(
        app.state,
        resolved_site_origin=resolved_site_origin,
        settings=settings,
        storage=storage,
        live_hub=common_live_runtime_services.live_hub,
        coordinate_frame=common_live_runtime_services.coordinate_frame,
        environment_provider=common_live_runtime_services.environment_provider,
        federation=federation,
        bit_evaluator=common_live_runtime_services.bit_evaluator,
        cleanup_service=common_live_runtime_services.cleanup_service,
        # API-only role still serves the cluster/node CRUD endpoints; without
        # in-memory registries here those handlers raise AttributeError on
        # app.state.  In split mode these are local to the API process and do
        # not influence the ingest-process localizer (no cross-process sync yet).
        registry=registry,
        cluster_registry=ClusterRegistry(),
        ble_observation_store=BleObservationStore(),
        ble_tracker=BleTracker(settings.ble_tracking_config()),
        sidecar_state=sidecar_state,
        capture_manager=capture_manager,
        ingest_concurrency=_IngestConcurrencyLimit(
            settings.ingest_max_concurrent,
            lease_timeout_seconds=settings.ingest_request_timeout_seconds,
        ),
        effector_manager=effector_manager,
    )

    task_handles = _ApiOnlyRuntimeTaskHandles()
    try:
        task_handles = await _start_api_only_runtime_services(
            app,
            api_live_db_poll_loop=_api_live_db_poll_loop,
            ble_tracking_loop=_ble_tracking_loop,
            environment_provider=common_live_runtime_services.environment_provider,
            federation=federation,
            storage=storage,
        )
        if settings.effectors_enabled:
            await effector_manager.start()
        yield
    finally:
        shutdown_timeout_s = 15.0
        await _shutdown_managed_ingest_sidecar(
            sidecar_state,
            sidecar_supervision_task,
            force_kill_on_timeout=False,
            shutdown_timeout_seconds=shutdown_timeout_s,
        )
        await effector_manager.stop()
        await _stop_api_only_runtime_services(
            app,
            task_handles,
            clear_transient_ingest_runtime_state=_clear_transient_ingest_runtime_state,
            federation=federation,
            shutdown_timeout_seconds=shutdown_timeout_s,
            storage=storage,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    if settings.process_role == "api":
        async with _api_only_lifespan(app, settings):
            yield
        return

    _prepare_lifespan_runtime(app.state, settings)

    localization_cfg = settings.localization_config()
    tracking_cfg = settings.tracking_config()

    storage, resolved_site_origin = await _initialize_storage_and_resolve_site_origin(
        settings,
        log_resolution=True,
    )

    common_live_runtime_services = _build_common_live_runtime_services(settings, storage=storage)
    combined_runtime_core_services = _build_combined_runtime_core_services(
        settings,
        classifier_factory=_build_runtime_classifier,
        common_live_runtime_services=common_live_runtime_services,
        localization_cfg=localization_cfg,
        storage=storage,
        tracking_cfg=tracking_cfg,
    )
    await combined_runtime_core_services.registry.load_overrides(await storage.list_node_overrides())

    sidecar_state, sidecar_supervision_task = await _launch_managed_ingest_sidecar(
        settings,
        startup_failure_log_message="Ingest sidecar startup failed; continuing with direct ingest fallback",
    )
    _warn_when_direct_ingest_falls_back(settings, sidecar_state)

    federation = _build_combined_runtime_federation(
        settings,
        coordinate_frame=common_live_runtime_services.coordinate_frame,
        live_hub=common_live_runtime_services.live_hub,
        tracker=combined_runtime_core_services.tracker,
    )
    effector_manager = _build_effector_manager(
        settings,
        storage=storage,
        live_hub=common_live_runtime_services.live_hub,
    )
    _wire_effector_zone_interlocks(
        effector_manager,
        coordinate_frame=common_live_runtime_services.coordinate_frame,
        zone_matcher=combined_runtime_core_services.zone_matcher,
    )
    if settings.effectors_enabled:
        _wire_effector_rules_handler(combined_runtime_core_services.fusion_node, effector_manager)

    hass_bridge = _build_hass_bridge(settings, live_hub=common_live_runtime_services.live_hub)
    unsubscribe_hass_tee = _wire_hass_live_event_tee(
        common_live_runtime_services.live_hub, hass_bridge
    )
    if settings.hass_enabled:
        _wire_hass_rules_handler(combined_runtime_core_services.fusion_node, hass_bridge)

    _python_ingest = settings.ingest_backend == "python"
    capture_manager = _build_capture_manager(
        settings,
        live_hub=common_live_runtime_services.live_hub,
        sidecar_url=None if _python_ingest else _ingest_runtime_base_url(settings),
        storage=storage,
        multi_sensor_buffer=combined_runtime_core_services.audio_buffer if _python_ingest else None,
        coordinate_frame=common_live_runtime_services.coordinate_frame,
        environment_provider=common_live_runtime_services.environment_provider,
    )

    _bind_runtime_state(
        app.state,
        resolved_site_origin=resolved_site_origin,
        settings=settings,
        storage=storage,
        registry=combined_runtime_core_services.registry,
        cluster_registry=combined_runtime_core_services.cluster_registry,
        ble_observation_store=BleObservationStore(),
        ble_tracker=BleTracker(settings.ble_tracking_config()),
        audio_buffer=combined_runtime_core_services.audio_buffer,
        localizer=combined_runtime_core_services.localizer,
        classifier=combined_runtime_core_services.classifier,
        tracker=combined_runtime_core_services.tracker,
        live_hub=common_live_runtime_services.live_hub,
        coordinate_frame=common_live_runtime_services.coordinate_frame,
        zone_matcher=combined_runtime_core_services.zone_matcher,
        environment_provider=common_live_runtime_services.environment_provider,
        fusion_node=combined_runtime_core_services.fusion_node,
        ingest_transport=combined_runtime_core_services.ingest_transport,
        federation=federation,
        bit_evaluator=common_live_runtime_services.bit_evaluator,
        diagnostics=combined_runtime_core_services.diagnostics,
        cleanup_service=common_live_runtime_services.cleanup_service,
        ingest_spool_consumer=combined_runtime_core_services.ingest_spool_consumer,
        sidecar_state=sidecar_state,
        capture_manager=capture_manager,
        ingest_concurrency=_IngestConcurrencyLimit(
            settings.ingest_max_concurrent,
            lease_timeout_seconds=settings.ingest_request_timeout_seconds,
        ),
        effector_manager=effector_manager,
        hass_bridge=hass_bridge,
    )
    # Late-bound, mirroring set_target_zone_resolver: the provider closes over
    # runtime state (tracker, zone_matcher, storage) that only exists post-bind.
    hass_bridge.set_state_snapshot_provider(_build_hass_state_snapshot_provider(app.state))

    ingest_stream_consumer_enabled = await _ensure_ingest_stream_consumer_running(app.state)

    task_handles = _CombinedRuntimeTaskHandles()
    try:
        task_handles = await _start_combined_runtime_background_tasks(
            app,
            cleanup_loop=_cleanup_loop,
            ble_tracking_loop=_ble_tracking_loop,
            environment_provider=common_live_runtime_services.environment_provider,
            federation=federation,
            fusion_node=combined_runtime_core_services.fusion_node,
            ingest_spool_consumer=combined_runtime_core_services.ingest_spool_consumer,
            ingest_stream_consumer_enabled=ingest_stream_consumer_enabled,
            install_site_origin_anchor=_install_site_origin_anchor,
            maintain_ingest_stream_consumer=_maintain_ingest_stream_consumer,
            settings=settings,
            storage=storage,
        )
        if settings.effectors_enabled:
            await effector_manager.start()
        if settings.hass_enabled:
            # Only outside the api-only role: two processes publishing the same
            # retained topics would fight over every entity's state.
            await hass_bridge.start()

        yield
    finally:
        shutdown_timeout_s = 15.0
        # The sidecar's Axum shutdown waits for open SSE requests.  Release our
        # stream before SIGTERM so the sidecar can drain normally instead of
        # requiring the 15-second SIGKILL fallback.
        await _stop_combined_ingest_stream_consumer(
            app,
            task_handles,
            shutdown_timeout_seconds=shutdown_timeout_s,
        )

        # Stop the ingest sidecar before tearing down the rest of the Python
        # runtime so startup/bind failures do not leave it running.
        await _shutdown_managed_ingest_sidecar(
            sidecar_state,
            sidecar_supervision_task,
            force_kill_on_timeout=True,
            shutdown_timeout_seconds=shutdown_timeout_s,
        )

        unsubscribe_hass_tee()
        await hass_bridge.stop()
        await effector_manager.stop()
        await _stop_combined_runtime_background_tasks(
            app,
            task_handles,
            clear_transient_ingest_runtime_state=_clear_transient_ingest_runtime_state,
            shutdown_timeout_seconds=shutdown_timeout_s,
        )
        await _shutdown_combined_runtime_services(
            classifier=combined_runtime_core_services.classifier,
            federation=federation,
            fusion_node=combined_runtime_core_services.fusion_node,
            shutdown_timeout_seconds=shutdown_timeout_s,
            storage=storage,
        )

        # Regression tripwire: if a non-daemon, non-main thread is still
        # alive here, the interpreter will hang at exit waiting to join it.
        surviving = [
            t
            for t in threading.enumerate()
            if t is not threading.main_thread() and not t.daemon and t.is_alive()
        ]
        if surviving:
            logger.warning(
                "Shutdown teardown finished but %d non-daemon thread(s) are still alive: %s",
                len(surviving),
                [t.name for t in surviving],
            )


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
        if (
            settings is not None
            and settings.ingest_backend == "python"
            and settings.ingest_port != settings.port
            and os.getenv("MINIMAPPR_INGEST_PORT") is not None
        ):
            return await call_next(request)
        if settings is not None and settings.ingest_backend != "python":
            return JSONResponse(status_code=404, content={"detail": "Ingest endpoints are not served by this API process"})
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
        "classifier": "routing",
        "fusion_queue_depth": fusion_queue_depth,
        "fusion_workers_running": running,
        "federation_enabled": federation_status["enabled"],
        "federation_peer_count": federation_status["peer_count"],
    }


@app.post("/api/v1/ingest/frame", response_model=IngestFrameResponse)
async def ingest_frame(payload: IngestFrameRequest, request: Request) -> IngestFrameResponse:
    state = _require_state(request)
    try:
        async with _require_ingest_concurrency(request):
            return await _run_ingest_with_timeout(
                request,
                _ingest_frame_impl(state, payload),
            )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="ingest request timed out",
            headers={"Retry-After": "1"},
        ) from exc


async def _ingest_frame_impl(state, payload: IngestFrameRequest) -> IngestFrameResponse:
    if excludes_audio_ingest(payload.node.position_geo):
        return IngestFrameResponse(
            accepted=True,
            triggered=False,
            frame_energy=0.0,
            detail="incorrect geo",
        )
    if _should_proxy_ingest_to_python_worker(state):
        forwarded = await _proxy_ingest_post(
            state,
            endpoint_path="/api/v1/ingest/frame",
            body=payload.model_dump_json().encode("utf-8"),
            content_type="application/json",
        )
        return IngestFrameResponse(**forwarded)
    try:
        return await state.fusion_node.ingest(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/ingest/store-forward", response_model=StoreForwardIngestResponse)
async def ingest_store_forward(payload: StoreForwardIngestRequest, request: Request) -> StoreForwardIngestResponse:
    state = _require_state(request)
    try:
        async with _require_ingest_concurrency(request):
            return await _run_ingest_with_timeout(
                request,
                _ingest_store_forward_impl(state, payload),
            )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="ingest request timed out",
            headers={"Retry-After": "1"},
        ) from exc


async def _ingest_store_forward_impl(
    state, payload: StoreForwardIngestRequest
) -> StoreForwardIngestResponse:
    if excludes_audio_ingest(payload.node.position_geo):
        results = [
            StoreForwardBufferedFrameResponse(
                sequence=item.frame.sequence,
                start_time_ns=item.frame.start_time_ns,
                accepted=False,
                detail="incorrect geo",
            )
            for item in payload.buffered_frames
        ]
        return StoreForwardIngestResponse(
            accepted=True,
            total_frames=len(results),
            accepted_frames=0,
            duplicate_frames=0,
            rejected_frames=len(results),
            queued_events=0,
            results=results,
        )
    if _should_proxy_ingest_to_python_worker(state):
        forwarded = await _proxy_ingest_post(
            state,
            endpoint_path="/api/v1/ingest/store-forward",
            body=payload.model_dump_json().encode("utf-8"),
            content_type="application/json",
        )
        return StoreForwardIngestResponse(**forwarded)
    if _should_block_direct_ingest(state):
        raise HTTPException(
            status_code=410,
            detail="Direct ingest is disabled; send firmware ingest to the Rust sidecar",
        )
    try:
        frames = payload.buffered_frames
        if payload.sort_by_toa:
            frames = sorted(frames, key=lambda item: item.frame.toa_ns or item.frame.start_time_ns)
        results = []
        for item in frames:
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
    try:
        async with _require_ingest_concurrency(request):
            return await _run_ingest_with_timeout(
                request,
                _ingest_binary_impl(state, request),
            )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="ingest request timed out",
            headers={"Retry-After": "1"},
        ) from exc


async def _emit_runner_cbit(state, node_id: str, delta: dict, timing_diag: dict) -> None:
    """Submit an edge-triggered CBIT degradation or recovery for firmware transport."""
    if delta.get("state") == "recovered":
        results = [
            BITTestResult(
                test_name="publish_queue",
                status=BITStatus.PASS,
                detail="No new firmware queue overflows or capture drops during recovery window",
                subsystem="audio",
            ),
            BITTestResult(
                test_name="audio_capture",
                status=BITStatus.PASS,
                detail="Firmware capture transport recovered",
                subsystem="audio",
            ),
        ]
    else:
        results = _runner_degraded_cbit_results(delta, timing_diag)
    now_ns = time.time_ns()
    report_in = BITReportIn(
        report_type=BITType.CBIT,
        timestamp_ns=now_ns,
        results=results,
    )
    report = await state.bit_evaluator.submit_report(node_id, report_in, received_ns=now_ns)
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


def _runner_degraded_cbit_results(delta: dict, timing_diag: dict) -> list[BITTestResult]:
    """Build degradation results without retaining a stale CBIT after recovery."""
    new_overflows = delta["new_queue_overflows"]
    total_overflows = delta["total_queue_overflows"]
    results: list[BITTestResult] = []
    if new_overflows:
        results.append(BITTestResult(
            test_name="publish_queue",
            status=BITStatus.DEGRADED,
            failure_code="CBIT_WARN: PUBLISH_QUEUE_OVERFLOW",
            detail=(
                f"{new_overflows} new firmware publish-queue overflow(s) detected "
                f"({total_overflows} total since boot)"
            ),
            measured_value=float(new_overflows),
            subsystem="audio",
        ))
    frames_dropped = int(timing_diag.get("runner_frames_dropped") or 0)
    if delta.get("new_frames_dropped", 0) > 0:
        results.append(BITTestResult(
            test_name="audio_capture",
            status=BITStatus.DEGRADED,
            failure_code="CBIT_WARN: AUDIO_CAPTURE_DROPPED",
            detail=f"{frames_dropped} total audio capture drops since boot",
            measured_value=float(frames_dropped),
            subsystem="audio",
        ))
    return results


async def _ingest_binary_impl(state, request: Request) -> StoreForwardIngestResponse:
    if _should_proxy_ingest_to_python_worker(state):
        body = await request.body()
        forwarded = await _proxy_ingest_post(
            state,
            endpoint_path="/api/v1/ingest/binary",
            body=body,
            content_type="application/octet-stream",
        )
        return StoreForwardIngestResponse(**forwarded)
    if _should_block_direct_ingest(state):
        raise HTTPException(status_code=410, detail="Direct ingest is disabled; send firmware ingest to the Rust sidecar")
    try:
        body = await request.body()
        payload = await asyncio.to_thread(
            parse_binary_ingest_payload,
            body,
            fallback_position_m=state.settings.legacy_ingest_fallback_position_m,
        )
        if excludes_audio_ingest(payload.node.position_geo):
            results = [
                StoreForwardBufferedFrameResponse(
                    sequence=item.frame.sequence,
                    start_time_ns=item.frame.start_time_ns,
                    accepted=False,
                    detail="incorrect geo",
                )
                for item in payload.buffered_frames
            ]
            return StoreForwardIngestResponse(
                accepted=True,
                total_frames=len(results),
                accepted_frames=0,
                duplicate_frames=0,
                rejected_frames=len(results),
                queued_events=0,
                results=results,
            )
        results = []
        if payload.buffered_frames:
            last_timing_diag: dict | None = None
            for item in payload.buffered_frames:
                req = IngestFrameRequest(
                    node=payload.node,
                    frame=item.frame,
                    environment=item.environment,
                )
                resp = await state.fusion_node.ingest_decoded(req, item.decoded_audio)
                results.append(StoreForwardBufferedFrameResponse(
                    sequence=item.frame.sequence,
                    start_time_ns=item.frame.start_time_ns,
                    accepted=resp.accepted,
                    duplicate=resp.duplicate,
                    triggered=resp.triggered,
                    frame_energy=resp.frame_energy,
                    queued_event_id=resp.queued_event_id,
                ))
                if item.frame.timing_diagnostics:
                    last_timing_diag = item.frame.timing_diagnostics
            if last_timing_diag and payload.node.id:
                drop_delta = state.fusion_node.observe_firmware_runner_stats(
                    payload.node.id, last_timing_diag
                )
                if drop_delta:
                    await _emit_runner_cbit(state, payload.node.id, drop_delta, last_timing_diag)
        else:
            # Heartbeats stay in the live registry; the node record is written
            # only when this process first sees a registration change.
            await state.fusion_node.refresh_live_node(payload.node)
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
    if _should_proxy_ingest_to_python_worker(state):
        return await _proxy_ingest_post(
            state,
            endpoint_path="/api/v1/ingest/env",
            body=payload.model_dump_json().encode("utf-8"),
            content_type="application/json",
        )
    if len(payload.samples) > 64:
        raise HTTPException(status_code=413, detail="environment batch exceeds 64 samples")

    accepted = 0
    for item in payload.samples:
        sample = item.sample
        if not sample.has_any_measurement():
            continue
        await state.fusion_node.ingest_environment_sample(node_id=item.node_id, sample=sample)
        accepted += 1
    return {"accepted": accepted, "queued": False}


@app.post("/api/v1/ingest/ble")
async def ingest_ble(payload: BleIngestRequest, request: Request) -> dict:
    state = _require_state(request)
    if len(payload.observations) > 256:
        raise HTTPException(status_code=413, detail="BLE batch exceeds 256 observations")
    now_ns = time.time_ns()
    accepted = await _ble_observation_store(state).ingest(
        node_id=payload.node_id,
        observations=payload.observations,
        recv_ns=now_ns,
        boot_count=payload.boot_count,
        boot_id=payload.boot_id,
        monotonic_ms=payload.monotonic_ms,
    )
    return {"accepted": accepted, "queued": False}


async def _ble_node_positions(state, *, limit: int = 5000) -> dict[str, tuple[float, float, float]]:
    node_rows = await state.storage.list_nodes(limit=limit)
    positions: dict[str, tuple[float, float, float]] = {}
    for node in node_rows:
        position = node.get("position_m")
        if position is None:
            continue
        try:
            positions[str(node["id"])] = (
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return positions


@app.get("/api/v1/ble/devices")
async def list_ble_devices(
    request: Request,
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    state = _require_state(request)
    now_ns = time.time_ns()
    grouped = await _ble_observation_store(state).latest_by_device(now_ns=now_ns)
    node_positions = await _ble_node_positions(state)
    devices: list[dict] = []
    for mac, observations in sorted(grouped.items()):
        raw_observations = [observation.as_raw_api_dict() for observation in observations]
        estimate = estimate_ble_device_position(
            mac,
            observations,
            node_positions,
            now_ns=now_ns,
        )
        devices.append({
            "mac": mac,
            "receiver_count": len({observation.node_id for observation in observations}),
            "last_seen_ns": max(observation.recv_ns for observation in observations),
            "observations": raw_observations,
            "estimate": estimate.as_api_dict() if estimate is not None else None,
        })
    devices.sort(key=lambda item: item["last_seen_ns"], reverse=True)
    return {"devices": devices[:limit]}


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
    if _has_live_ingest_runtime(state):
        live_nodes = {
            runtime.spec.id: runtime
            for runtime in await state.registry.list_nodes()
        }
        for node in nodes:
            runtime = live_nodes.get(str(node.get("id")))
            if runtime is None:
                continue
            node.update(runtime.spec.model_dump(mode="json"))
            node["last_seen_ns"] = runtime.last_seen_ns
    latest_environment_rows = await state.storage.list_latest_environment_per_node(limit=limit)
    latest_time_quality_by_node = await state.storage.list_latest_time_quality_per_node()
    latest_observation_metadata_by_node = await state.storage.list_latest_observation_metadata_per_node()
    latest_environment_by_node = {
        row["node_id"]: row for row in latest_environment_rows if row.get("node_id") is not None
    }
    effector_manager: EffectorManager = state.effector_manager
    ptz_status_by_node = {
        status.node_id: status
        for status in await effector_manager.list_status()
    }
    sidecar_node_snapshots = (
        _sidecar_stream_consumer_snapshots(state)
        if settings.ingest_backend == "rust"
        else {}
    )
    if sidecar_node_snapshots:
        nodes = _merge_nodes_with_sidecar_snapshots(nodes, sidecar_node_snapshots, limit=limit)
        for node_id, snapshot in sidecar_node_snapshots.items():
            latest_environment = _sidecar_snapshot_latest_environment(snapshot, node_id=node_id)
            if latest_environment is not None:
                latest_environment_by_node[node_id] = latest_environment
    await _apply_runtime_health_statuses(
        nodes,
        bit_evaluator=bit_evaluator,
        now_ns=now_ns,
        degraded_after_seconds=settings.node_degraded_after_seconds,
        offline_after_seconds=settings.node_offline_after_seconds,
    )
    for node in nodes:
        if node.get("position_geo") is None and node.get("position_m"):
            local = node["position_m"]
            geo = state.coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
            node["position_geo"] = geo.model_dump(mode="json")

        # Attach latest BIT failure codes for UI consumption
        bit_reports = await bit_evaluator.latest_reports_for_node(node["id"])
        failure_codes: list[str] = []
        for report in bit_reports:
            failure_codes.extend(report.failure_codes)
        if failure_codes:
            node["bit_failure_codes"] = failure_codes

        if _node_has_capability(node, NodeCapability.PTZ_CAMERA):
            ptz_status = ptz_status_by_node.get(node["id"])
            node["ptz_status"] = ptz_status.model_dump(mode="json") if ptz_status is not None else None

        ble_observations = await _ble_observation_store(state).latest_for_node(node["id"], now_ns=now_ns)
        if ble_observations:
            node["ble_observations"] = [observation.as_raw_api_dict() for observation in ble_observations]

        node_capabilities = {str(capability) for capability in node.get("capabilities", [])}
        should_enrich_audio = not node_capabilities or NodeCapability.AUDIO.value in node_capabilities
        if not should_enrich_audio:
            await asyncio.sleep(0)
            continue

        use_live_audio_buffer_summary = _has_live_ingest_runtime(state) and (
            settings.ingest_backend == "python" or settings.direct_ingest_enabled
        )
        live_summary = (
            await state.fusion_node.live_audio_summary(node["id"])
            if _has_live_ingest_runtime(state)
            else None
        )
        if use_live_audio_buffer_summary:
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
                # Window and per-sensor breakdown: the recent_* aggregates are
                # short-window means and hid a failed microphone for weeks.
                "recent_coverage_window_seconds": audio_summary.get(
                    "recent_coverage_window_seconds"
                ),
                "recent_min_coverage_ratio": audio_summary.get("recent_min_coverage_ratio"),
                "per_sensor": audio_summary.get("per_sensor"),
                "max_buffer_samples": audio_summary["max_buffer_samples"],
                "max_buffer_seconds": audio_summary["max_buffer_seconds"],
                "status": audio_status,
            }
            if isinstance(live_summary, dict):
                for key in ("runner_stats", "ingest_health"):
                    if key in live_summary:
                        node["audio_debug"][key] = live_summary[key]
        else:
            sensor_ids = _sensor_ids_from_node_row(node)
            sidecar_snapshot = sidecar_node_snapshots.get(node["id"])
            sidecar_audio_debug = (
                _sidecar_snapshot_audio_debug(
                    sidecar_snapshot,
                    now_ns=now_ns,
                    degraded_after_seconds=settings.node_degraded_after_seconds,
                )
                if sidecar_snapshot is not None
                else None
            )
            if sidecar_audio_debug is not None:
                node["audio_debug"] = sidecar_audio_debug
                latest_sidecar_timing = getattr(
                    sidecar_snapshot, "latest_timing_diagnostics", None
                )
                if isinstance(latest_sidecar_timing, dict) and latest_sidecar_timing:
                    node["latest_timing_diagnostics"] = latest_sidecar_timing
            elif isinstance(live_summary, dict):
                node["audio_debug"] = live_summary
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


@app.get("/api/v1/nodes/omni-detection-summary")
async def list_node_omni_detection_summary(
    request: Request,
    active_seconds: float = Query(default=60.0, gt=0.0, le=3600.0),
    recent_seconds: float = Query(default=600.0, gt=0.0, le=86400.0),
    limit_per_node: int = Query(default=5, ge=1, le=20),
) -> list[dict]:
    state = _require_state(request)
    settings: Settings = state.settings
    now_ns = time.time_ns()
    active_window_ns = int(active_seconds * 1_000_000_000)
    recent_window_ns = max(int(recent_seconds * 1_000_000_000), active_window_ns)
    return await state.storage.omni_detection_summary_by_node(
        now_ns=now_ns,
        active_window_ns=active_window_ns,
        recent_window_ns=recent_window_ns,
        limit_per_node=limit_per_node,
        min_label_confidence=settings.detection_min_confidence,
    )


def _node_has_capability(node: dict[str, Any], capability: NodeCapability) -> bool:
    return capability.value in {str(item) for item in node.get("capabilities", [])}


def _redact_node_transport(node: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(node)
    redacted["transport"] = {}
    return redacted


class _EffectorAimBody(BaseModel):
    track_id: str | None = None
    target: Vec3 | None = None

    @model_validator(mode="after")
    def _validate(self) -> "_EffectorAimBody":
        if self.track_id is None and self.target is None:
            raise ValueError("Either track_id or target must be provided")
        return self


class _EffectorSnapshotBody(BaseModel):
    track_id: str | None = None
    detection_id: str | None = None


class _EffectorArmBody(BaseModel):
    zone_id: str | None = None


def _node_row_to_runtime_spec(node: dict[str, Any]) -> NodeSpec:
    return NodeSpec(
        id=node["id"],
        node_type=node["node_type"],
        position_m=tuple(node["position_m"]) if node.get("position_m") else None,
        position_geo=node.get("position_geo"),
        sensor_offsets_m=node.get("sensor_offsets_m") or [(0.0, 0.0, 0.0)],
        orientation=node.get("orientation") or {},
        capabilities=node.get("capabilities") or [],
        mobility=node.get("mobility") or "stationary",
        metadata=node.get("metadata") or {},
        properties=node.get("properties") or {},
        transport=node.get("transport") or {},
        capability_config=node.get("capability_config") or {},
        safety=node.get("safety") or {},
        permissions=node.get("permissions") or {},
    )


async def _broadcast_node_updated(state, node_id: str) -> None:
    live_hub = getattr(state, "live_hub", None)
    if live_hub is None:
        return
    try:
        await live_hub.broadcast({"type": "node_updated", "node_id": node_id})
    except Exception:
        logger.debug("node_updated broadcast failed for %s", node_id, exc_info=True)


async def _enriched_node_detail(state, node_id: str) -> dict[str, Any] | None:
    node = await state.storage.get_node_by_id(node_id)
    if node is None:
        return None
    if node.get("position_geo") is None and node.get("position_m"):
        local = node["position_m"]
        geo = state.coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
        node["position_geo"] = geo.model_dump(mode="json")
    now_ns = time.time_ns()
    await _apply_runtime_health_statuses(
        [node],
        bit_evaluator=state.bit_evaluator,
        now_ns=now_ns,
        degraded_after_seconds=state.settings.node_degraded_after_seconds,
        offline_after_seconds=state.settings.node_offline_after_seconds,
    )
    bit_reports = await state.bit_evaluator.latest_reports_for_node(node_id)
    failure_codes: list[str] = []
    for report in bit_reports:
        failure_codes.extend(report.failure_codes)
    if failure_codes:
        node["bit_failure_codes"] = failure_codes
    if _node_has_capability(node, NodeCapability.PTZ_CAMERA):
        manager: EffectorManager = state.effector_manager
        status = await manager.get_status(node_id)
        node["ptz_status"] = status.model_dump(mode="json") if status is not None else None
    ble_observations = await _ble_observation_store(state).latest_for_node(node_id, now_ns=now_ns)
    if ble_observations:
        node["ble_observations"] = [observation.as_raw_api_dict() for observation in ble_observations]
    fusion_node = getattr(state, "fusion_node", None)
    if fusion_node is not None:
        node["position_estimator"] = fusion_node.node_position_estimator_diagnostics(node_id)
    return _redact_node_transport(node)


@app.get("/api/v1/nodes/{node_id}")
async def get_node_detail(node_id: str, request: Request) -> dict:
    state = _require_state(request)
    node = await _enriched_node_detail(state, node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@app.post("/api/v1/nodes", status_code=201)
async def register_node(payload: NodeRegistrationRequest, request: Request) -> dict:
    state = _require_state(request)
    existing = await state.storage.get_node_by_id(payload.id)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Node already exists")
    spec = payload.to_node_spec()
    if spec.position_m is None and spec.position_geo is not None:
        spec = spec.model_copy(update={"position_m": state.coordinate_frame.geo_to_local(spec.position_geo)})
    await state.storage.insert_node_registration(spec, origin="operator")
    await state.registry.upsert(spec, last_seen_ns=0)
    if NodeCapability.PTZ_CAMERA in spec.capabilities:
        manager: EffectorManager = state.effector_manager
        node_row = await state.storage.get_node_by_id(spec.id)
        if node_row is not None:
            await manager.register_node(node_row)
    await _broadcast_node_updated(state, spec.id)
    _request_hass_reconcile(state)
    node = await _enriched_node_detail(state, spec.id)
    if node is None:
        raise HTTPException(status_code=500, detail="Node registration did not persist")
    return node


@app.patch("/api/v1/nodes/{node_id}")
async def patch_node(node_id: str, payload: NodePatchRequest, request: Request) -> dict:
    state = _require_state(request)
    node = await state.storage.get_node_by_id(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    changed_transport = payload.transport is not None
    previous_effective_filter = node.get("effective_position_filter")
    await state.storage.update_node_operator_fields(
        node_id,
        transport=payload.transport,
        capability_config=payload.capability_config,
        safety=payload.safety.model_dump(mode="json") if payload.safety is not None else None,
        permissions=payload.permissions,
        metadata=payload.metadata,
    )
    existing_overrides = dict(node.get("overrides") or {})
    for field_name in payload.clear_overrides:
        existing_overrides.pop(field_name, None)
    if payload.overrides is not None:
        override_payload = payload.overrides.model_dump(mode="json", exclude_none=True)
        existing_overrides.update(override_payload)
    if payload.overrides is not None or payload.clear_overrides:
        if existing_overrides:
            existing_overrides["updated_ns"] = time.time_ns()
        await state.storage.set_node_overrides(node_id, existing_overrides)
        await state.registry.set_overrides(node_id, existing_overrides)
    updated = await state.storage.get_node_by_id(node_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Node not found")
    if updated.get("effective_position_filter") == "kde" and previous_effective_filter != "kde":
        fusion_node = getattr(state, "fusion_node", None)
        if fusion_node is not None:
            await fusion_node.reset_node_position_estimator(node_id)
        else:
            await state.storage.delete_node_position_estimator_state(node_id)
    if changed_transport and _node_has_capability(updated, NodeCapability.PTZ_CAMERA):
        manager: EffectorManager = state.effector_manager
        await manager.detach(node_id)
        await manager.register_node(updated)
    await _broadcast_node_updated(state, node_id)
    return _redact_node_transport(updated)


@app.post("/api/v1/nodes/{node_id}/position-estimator/reset")
async def reset_node_position_estimator(node_id: str, request: Request) -> dict:
    state = _require_state(request)
    if await state.storage.get_node_by_id(node_id) is None:
        raise HTTPException(status_code=404, detail="Node not found")
    fusion_node = getattr(state, "fusion_node", None)
    if fusion_node is not None:
        await fusion_node.reset_node_position_estimator(node_id)
    else:
        await state.storage.delete_node_position_estimator_state(node_id)
    await _broadcast_node_updated(state, node_id)
    return {"node_id": node_id, "reset": True}


@app.get("/api/v1/nodes/{node_id}/safety", response_model=NodeSafetyConfig)
async def get_node_safety(node_id: str, request: Request) -> NodeSafetyConfig:
    state = _require_state(request)
    node = await state.storage.get_node_by_id(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return NodeSafetyConfig.model_validate(node.get("safety") or {})


@app.patch("/api/v1/nodes/{node_id}/safety", response_model=NodeSafetyConfig)
async def patch_node_safety(node_id: str, payload: NodeSafetyConfig, request: Request) -> NodeSafetyConfig:
    state = _require_state(request)
    updated = await state.storage.update_node_operator_fields(
        node_id,
        safety=payload.model_dump(mode="json"),
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Node not found")
    await _broadcast_node_updated(state, node_id)
    return payload


async def _require_ptz_node(state, node_id: str) -> None:
    node = await state.storage.get_node_by_id(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    if not _node_has_capability(node, NodeCapability.PTZ_CAMERA):
        raise HTTPException(status_code=404, detail="node has no effector capability")


@app.get("/api/v1/nodes/{node_id}/effector/status")
async def get_node_effector_status(node_id: str, request: Request) -> dict:
    state = _require_state(request)
    await _require_ptz_node(state, node_id)
    manager: EffectorManager = state.effector_manager
    status = await manager.get_status(node_id)
    capabilities = await manager.get_capabilities(node_id)
    return {
        "node_id": node_id,
        "status": status.model_dump(mode="json") if status is not None else None,
        "capabilities": capabilities,
    }


@app.post("/api/v1/nodes/{node_id}/effector/arm")
async def arm_node_effector(node_id: str, payload: _EffectorArmBody, request: Request) -> JSONResponse:
    state = _require_state(request)
    await _require_ptz_node(state, node_id)
    result = await state.effector_manager.arm(node_id, zone_id=payload.zone_id)
    status_code = 200 if result.status == "COMPLETED" else 409
    return JSONResponse(status_code=status_code, content=result.model_dump(mode="json"))


@app.post("/api/v1/nodes/{node_id}/effector/disarm")
async def disarm_node_effector(node_id: str, request: Request) -> JSONResponse:
    state = _require_state(request)
    await _require_ptz_node(state, node_id)
    result = await state.effector_manager.disarm(node_id)
    status_code = 200 if result.status == "COMPLETED" else 409
    return JSONResponse(status_code=status_code, content=result.model_dump(mode="json"))


@app.post("/api/v1/nodes/{node_id}/effector/aim")
async def aim_node_effector(node_id: str, payload: _EffectorAimBody, request: Request) -> JSONResponse:
    state = _require_state(request)
    await _require_ptz_node(state, node_id)
    target_pos = payload.target
    if target_pos is None:
        tracks = await state.storage.list_tracks(limit=1000)
        track = next((t for t in tracks if t["id"] == payload.track_id), None)
        if track is None:
            raise HTTPException(status_code=404, detail="Track not found")
        target_pos = tuple(float(v) for v in track["position_m"])
    result = await state.effector_manager.slew_to_target(node_id, target_pos, track_id=payload.track_id)
    status_code = 200 if result.status == "COMPLETED" else 409
    return JSONResponse(status_code=status_code, content=result.model_dump(mode="json"))


@app.post("/api/v1/nodes/{node_id}/effector/snapshot")
async def snapshot_node_effector(node_id: str, payload: _EffectorSnapshotBody, request: Request) -> JSONResponse:
    state = _require_state(request)
    await _require_ptz_node(state, node_id)
    result = await state.effector_manager.capture(
        node_id,
        track_id=payload.track_id,
        detection_id=payload.detection_id,
    )
    status_code = 200 if result.status == "COMPLETED" else 409
    return JSONResponse(status_code=status_code, content=result.model_dump(mode="json"))


@app.get("/api/v1/nodes/{node_id}/effector/snapshot.jpg")
async def node_effector_snapshot_live(node_id: str, request: Request) -> FileResponse:
    state = _require_state(request)
    await _require_ptz_node(state, node_id)
    try:
        path = await state.effector_manager.snapshot_live(node_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Snapshot failed: {exc}") from exc
    if path is None:
        raise HTTPException(status_code=404, detail="Effector not found or snapshot unsupported")
    return FileResponse(path=path, media_type="image/jpeg")


@app.get("/api/v1/node-artifacts/{artifact_id}")
async def get_node_artifact(artifact_id: str, request: Request) -> FileResponse:
    state = _require_state(request)
    row = await state.storage.get_node_artifact(artifact_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    artifact_path = Path(row["path"]).resolve()
    snapshot_root: Path = state.settings.effector_snapshot_dir.resolve()
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file no longer exists")
    if not artifact_path.is_relative_to(snapshot_root):
        raise HTTPException(status_code=403, detail="Artifact path is outside snapshot directory")
    return FileResponse(path=artifact_path, media_type="image/jpeg")


async def _runtime_node_health_counts(
    state,
    *,
    now_ns: int,
    limit: int = 5000,
) -> dict[str, int]:
    settings: Settings = state.settings
    bit_evaluator: BITReportEvaluator = state.bit_evaluator
    nodes = await state.storage.list_nodes(limit=limit)
    sidecar_node_snapshots = (
        _sidecar_stream_consumer_snapshots(state)
        if settings.ingest_backend == "rust"
        else {}
    )
    if sidecar_node_snapshots:
        nodes = _merge_nodes_with_sidecar_snapshots(nodes, sidecar_node_snapshots, limit=limit)
    await _apply_runtime_health_statuses(
        nodes,
        bit_evaluator=bit_evaluator,
        now_ns=now_ns,
        degraded_after_seconds=settings.node_degraded_after_seconds,
        offline_after_seconds=settings.node_offline_after_seconds,
    )
    counts = {
        "online_nodes": 0,
        "degraded_nodes": 0,
        "offline_nodes": 0,
    }
    for node in nodes:
        health_status = str(node.get("health_status") or "")
        if health_status == NodeHealthStatus.ONLINE.value:
            counts["online_nodes"] += 1
        elif health_status == NodeHealthStatus.OFFLINE.value:
            counts["offline_nodes"] += 1
        else:
            counts["degraded_nodes"] += 1
    return counts


# ------------------------------------------------------------------
# BIT (Built-In Test) Endpoints
# ------------------------------------------------------------------


@app.delete("/api/v1/nodes/{node_id}")
async def delete_node(node_id: str, request: Request) -> dict:
    """Delete a stale node and its records from storage and live caches.

    Only nodes that are currently offline may be deleted: an active node would
    immediately repopulate from its next heartbeat, so deleting it is a no-op
    that only confuses operators. The check uses the same effective
    ``last_seen_ns`` (DB row merged with any live sidecar snapshot) as the node
    list, making the backend authoritative even if a client bypasses the UI.
    """
    state = _require_state(request)
    settings: Settings = state.settings
    now_ns = time.time_ns()

    db_node = await state.storage.get_node_by_id(node_id)
    snapshot = _sidecar_stream_consumer_snapshots(state).get(node_id)
    snapshot_last_seen = getattr(snapshot, "last_seen_ns", None) if snapshot is not None else None

    last_seen_candidates = [
        value
        for value in (
            int(db_node["last_seen_ns"]) if db_node and db_node.get("last_seen_ns") is not None else None,
            int(snapshot_last_seen) if isinstance(snapshot_last_seen, int) else None,
        )
        if value is not None
    ]
    effective_last_seen_ns = max(last_seen_candidates) if last_seen_candidates else 0

    if effective_last_seen_ns:
        health = _heartbeat_health_status(
            last_seen_ns=effective_last_seen_ns,
            now_ns=now_ns,
            degraded_after_seconds=settings.node_degraded_after_seconds,
            offline_after_seconds=settings.node_offline_after_seconds,
        )
        if health != NodeHealthStatus.OFFLINE.value:
            raise HTTPException(
                status_code=409,
                detail="Node is still active; only offline nodes can be deleted.",
            )

    deleted = await state.storage.delete_node(node_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Node not found")

    # Purge in-memory caches so the deleted node cannot be re-served before the
    # next poll. A truly-stale node stays gone; a live one would re-register.
    await state.effector_manager.detach(node_id)
    await state.registry.delete_node(node_id)
    consumer = getattr(state, "ingest_stream_consumer", None)
    purge = getattr(consumer, "purge_node", None)
    if callable(purge):
        purge(node_id)
    _request_hass_reconcile(state)

    return {"ok": True, "node_id": node_id}


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
async def list_detections(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
    review_state: DetectionReviewState | None = Query(default=None),
) -> list[dict]:
    state = _require_state(request)
    settings: Settings = state.settings
    effective_limit = min(limit, settings.cop_detections_max_items)
    cutoff_ns = time.time_ns() - int(settings.cop_detections_max_age_seconds * 1_000_000_000)
    review_state_value = review_state.value if review_state is not None else None
    detections = await state.storage.list_detections(
        limit=effective_limit,
        since_ns=cutoff_ns,
        min_label_confidence=(None if review_state_value is not None else settings.detection_min_confidence),
        review_state=review_state_value,
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
    include_dropped: bool = Query(default=False),
) -> list[dict]:
    state = _require_state(request)
    now_ns = time.time_ns()
    if hasattr(state, "tracker"):
        _ = await state.tracker.snapshot(now_ns=now_ns)
    effective_limit = min(limit, state.settings.cop_tracks_max_items)
    cutoff_ns = now_ns - int(state.settings.cop_tracks_max_age_seconds * 1_000_000_000)
    # Dropped tracks were previously filtered ONLY by the federation merge path
    # below, so a deployment with federation disabled served every dead track it
    # had — 149 of 150 rows in one live sample. Filtering in the query keeps the
    # endpoint consistent regardless of federation state, and keeps the LIMIT from
    # being consumed by dropped rows.
    tracks = await state.storage.list_tracks(
        limit=effective_limit,
        since_ns=cutoff_ns,
        statuses=None if include_dropped else sorted(ACTIVE_TRACK_STATUSES),
    )
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
        "capture_final_tracks_settle_seconds": settings.capture_final_tracks_settle_seconds,
        "snippet_retention_seconds": settings.snippet_retention_seconds,
        "retention_yamnet_audio_seconds": settings.retention_yamnet_audio_seconds,
        "retention_birdnet_audio_seconds": settings.retention_birdnet_audio_seconds,
        "retention_drone_audio_seconds": settings.retention_drone_audio_seconds,
        "retention_alert_audio_seconds": settings.retention_alert_audio_seconds,
        "retention_detection_metadata_seconds": settings.retention_detection_metadata_seconds,
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
        "localization_max_tau_seconds": settings.localization_max_tau_seconds,
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
        "localization_min_reportable_confidence": settings.localization_min_reportable_confidence,
        "localization_max_reportable_gdop": settings.localization_max_reportable_gdop,
        "classifier_routing_config_path": str(settings.classifier_routing_config_path),
        "birdnet_enabled": settings.birdnet_enabled,
        "drone_head_enabled": settings.drone_head_enabled,
        "drone_head_model_path": str(settings.drone_head_model_path),
        "drone_head_min_confidence": settings.drone_head_min_confidence,
        "drone_head_min_frame_fraction": settings.drone_head_min_frame_fraction,
        "stt_enabled": settings.stt_enabled,
        "stt_model_id": settings.stt_model_id,
        "stt_model_cache_dir": str(settings.stt_model_cache_dir),
        "stt_trigger_min_confidence": settings.stt_trigger_min_confidence,
        "transcript_retention_seconds": settings.transcript_retention_seconds,
        "omni_scan_enabled": settings.omni_scan_enabled,
        "omni_scan_interval_seconds": settings.omni_scan_interval_seconds,
        "omni_scan_window_seconds": settings.omni_scan_window_seconds,
        "omni_scan_min_rms": settings.omni_scan_min_rms,
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
        "classification_audio_source": settings.classification_audio_source,
        "min_localization_confidence": settings.min_localization_confidence,
        "skip_localization_for_classification": settings.skip_localization_for_classification,
        "classifier_backends_available": [
            {"name": entry.name, "available": entry.available, "reason": entry.reason}
            for entry in probe_backends()
        ],
        "localization_band_min_hz": settings.localization_band_min_hz,
        "localization_band_max_hz": settings.localization_band_max_hz,
        "birdnet_chunked_dispatch_enabled": settings.birdnet_chunked_dispatch_enabled,
        "birdnet_trigger_min_confidence": settings.birdnet_trigger_min_confidence,
        "birdnet_geo_min_confidence": settings.birdnet_geo_min_confidence,
        "persisted_override_keys": sorted(load_overrides(settings.config_overrides_path)),
        "beamformer_type": settings.beamformer_type,
        "beamformed_classification_min_sensor_count": settings.beamformed_classification_min_sensor_count,
        "beamformed_classification_confidence_margin": settings.beamformed_classification_confidence_margin,
        "mvdr_diagonal_loading": settings.mvdr_diagonal_loading,
        "tracking_filter": settings.tracking_filter,
        "association_distance_m": settings.association_distance_m,
        "association_max_gate_m": settings.association_max_gate_m,
        "association_chi2_gate": settings.association_chi2_gate,
        "kalman_process_noise": settings.kalman_process_noise,
        "kalman_measurement_noise": settings.kalman_measurement_noise,
        "track_stale_seconds": settings.track_stale_seconds,
        "localization_node_bearing_strength": settings.localization_node_bearing_strength,
        "multi_node_bearing_window_seconds": settings.multi_node_bearing_window_seconds,
        "multi_node_bearing_min_separation_deg": settings.multi_node_bearing_min_separation_deg,
        "multi_node_bearing_ttl_seconds": settings.multi_node_bearing_ttl_seconds,
        "multi_node_bearing_max_condition": settings.multi_node_bearing_max_condition,
        "classifier_stage_timeout_seconds": settings.classifier_stage_timeout_seconds,
        "classification_stage_timeout_seconds": settings.classification_stage_timeout_seconds,
        "fusion_worker_count": settings.fusion_worker_count,
        "fusion_event_queue_size": settings.fusion_event_queue_size,
        "fusion_localization_queue_size": settings.fusion_localization_queue_size,
        "fusion_classification_queue_size": settings.fusion_classification_queue_size,
        "fusion_rules_queue_size": settings.fusion_rules_queue_size,
        "drop_on_backpressure": settings.drop_on_backpressure,
        "fusion_drop_on_backpressure": settings.fusion_drop_on_backpressure,
        "fusion_offline_replay_mode": settings.fusion_offline_replay_mode,
        "rules_config_path": str(settings.rules_config_path),
        "taxonomy_config_path": str(settings.taxonomy_config_path),
        "site_origin": {
            "lat": settings.site_origin_lat,
            "lon": settings.site_origin_lon,
            "alt_m": settings.site_origin_alt_m,
            "mode": settings.site_origin_source,
            "source": site_origin_resolution_source,
            # False means the site is still on the configured fallback and is
            # waiting for a trusted GPS fix. Surfaced because an un-anchored origin
            # silently rejects every localization once nodes are far from it.
            "anchored": bool(getattr(state, "site_origin_anchored", False)),
            "contributing_node_ids": list(
                getattr(state, "site_origin_contributing_node_ids", ()) or ()
            ),
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
        # Reads stay nested (matching the `federation` block above) while writes
        # are flat `hass_*` keys — so adding fields here changes nothing about
        # flat-key group coverage in /api/v1/config/structured.
        "hass": {
            "enabled": settings.hass_enabled,
            "base_url": settings.hass_base_url,
            "token": _REDACTED_SECRET_PLACEHOLDER if settings.hass_token else "",
            "mqtt_host": settings.hass_mqtt_host,
            "mqtt_port": settings.hass_mqtt_port,
            "mqtt_username": settings.hass_mqtt_username,
            "mqtt_password": _REDACTED_SECRET_PLACEHOLDER if settings.hass_mqtt_password else "",
            "mqtt_client_id": settings.hass_mqtt_client_id,
            "mqtt_keepalive_seconds": settings.hass_mqtt_keepalive_seconds,
            "mqtt_tls_enabled": settings.hass_mqtt_tls_enabled,
            "mqtt_tls_insecure": settings.hass_mqtt_tls_insecure,
            "discovery_prefix": settings.hass_discovery_prefix,
            "base_topic": settings.hass_base_topic,
            "device_id": settings.hass_device_id,
            "device_name": settings.hass_device_name,
            "publish_interval_seconds": settings.hass_publish_interval_seconds,
            "publish_min_interval_seconds": settings.hass_publish_min_interval_seconds,
            "reconcile_interval_seconds": settings.hass_reconcile_interval_seconds,
            "queue_size": settings.hass_queue_size,
            "reconnect_backoff_initial_seconds": settings.hass_reconnect_backoff_initial_seconds,
            "reconnect_backoff_max_seconds": settings.hass_reconnect_backoff_max_seconds,
            "detection_off_delay_seconds": settings.hass_detection_off_delay_seconds,
            "detection_classes": list(settings.hass_detection_classes),
            "track_slot_count": settings.hass_track_slot_count,
            "zone_spl_window_seconds": settings.hass_zone_spl_window_seconds,
            "publish_zone_occupancy": settings.hass_publish_zone_occupancy,
            "publish_zone_spl": settings.hass_publish_zone_spl,
            "publish_detection_classes": settings.hass_publish_detection_classes,
            "publish_node_status": settings.hass_publish_node_status,
            "publish_system_health": settings.hass_publish_system_health,
            "publish_events": settings.hass_publish_events,
            "publish_track_slots": settings.hass_publish_track_slots,
        },
    }


@app.get("/api/v1/config/structured")
async def get_config_structured(request: Request) -> dict:
    """Read-only stage-grouped projection of ``GET /api/v1/config``.

    Additive: calls ``get_config`` once and regroups the flat keys into the
    pipeline-stage vocabulary (see ``core/config_groups.py``). The flat
    GET/PATCH ``/api/v1/config`` surface is unchanged — this is purely a
    presentation projection used for the DAG's ``/settings/config#{group}``
    deep links.
    """
    flat = await get_config(request)
    return group_flat_config(flat)


# Single source of truth lives in settings_store so the persisted-overrides file
# and this HTTP allowlist can never drift apart.
_CONFIG_PATCH_ALLOWLIST = CONFIG_PATCH_ALLOWLIST

# Patched keys that only reach the Rust sidecar via spawn env — a running sidecar
# must be restarted to pick them up (surfaced as ``restart_required`` in PATCH).
_SIDECAR_RESTART_REQUIRED_KEYS = {
    "classification_audio_source",
    "birdnet_enabled",
    "min_localization_confidence",
    "localization_band_min_hz",
    "localization_band_max_hz",
    "localization_window_seconds",
    "birdnet_trigger_min_confidence",
    "birdnet_geo_min_confidence",
    "trigger_rms",
    "trigger_cooldown_seconds",
}

_LOCALIZATION_ALGORITHMS = {"gcc_phat", "srp_phat", "music", "esprit"}
_LOCALIZATION_STRATEGIES = {"fixed", "geometry_aware", "cascade"}
_BEAMFORMER_TYPES = {
    "delay_and_sum",
    "das",
    "freq_domain_das",
    "band_split_das",
    "band_split",
    "mvdr",
    "superdirective",
    "gevd",
}
_CLASSIFICATION_AUDIO_SOURCES = {"beamformed", "omni", "nearest_node_omni"}
_TRACKING_FILTERS = {"linear", "kalman"}
_COORDINATE_MODES = {"flat", "geodetic"}

# Secrets are redacted to this placeholder in GET /api/v1/config. A PATCH
# carrying it back means "unchanged", never "set the secret to '***'".
_REDACTED_SECRET_PLACEHOLDER = "***"
_SECRET_CONFIG_KEYS = {"hass_token", "hass_mqtt_password"}
_HASS_TOPIC_LEVEL_KEYS = {"hass_discovery_prefix", "hass_base_topic", "hass_device_id"}


def _rules_config_payload_from_file(config_path: Path) -> tuple[list[dict[str, Any]], str]:
    if not config_path.exists():
        return default_rules_as_dicts(), "default"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("rules config must be a JSON object")
        rules: list[dict[str, Any]] = []
        for item in raw.get("rules", []):
            if not isinstance(item, dict):
                continue
            rule = RuleDef.from_dict(item)
            if rule is not None:
                rules.append(rule.to_dict())
        return (rules or default_rules_as_dicts()), "file"
    except Exception as exc:
        logger.warning("Failed to read rules config %s: %s", config_path, exc)
        return default_rules_as_dicts(), "default"


def _rules_response(settings: Settings) -> RulesConfigResponse:
    rules, source = _rules_config_payload_from_file(settings.rules_config_path)
    return RulesConfigResponse(rules=rules, path=str(settings.rules_config_path), source=source)


def _classifier_routing_response(
    settings: Settings,
    *,
    restart_required: bool = False,
) -> ClassifierRoutingConfigResponse:
    config_path = settings.classifier_routing_config_path
    source = "file" if config_path.exists() else "default"
    try:
        routing = load_routing_file(config_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid classifier routing config at {config_path}: {exc}",
        ) from exc
    return ClassifierRoutingConfigResponse(
        routing=routing_to_dict(routing),
        path=str(config_path),
        source=source,
        restart_required=restart_required,
    )


@app.get("/api/v1/classifier-routing", response_model=ClassifierRoutingConfigResponse)
async def get_classifier_routing_config(request: Request) -> ClassifierRoutingConfigResponse:
    """Return the canonical routing document, without applying kill switches."""
    state = _require_state(request)
    return _classifier_routing_response(state.settings)


@app.put("/api/v1/classifier-routing", response_model=ClassifierRoutingConfigResponse)
async def put_classifier_routing_config(
    payload: ClassifierRoutingConfigUpdate,
    request: Request,
) -> ClassifierRoutingConfigResponse:
    """Validate and atomically save routing; callers restart runtime processes to apply it."""
    state = _require_state(request)
    try:
        routing = parse_routing_document(payload.routing, source="request body")
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    config_path = state.settings.classifier_routing_config_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config_path.with_name(f".{config_path.name}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(routing_to_dict(routing), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, config_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # The FusionNode and Rust helper own instantiated models. Updating that graph
    # live risks concurrent model teardown on the audio path, so API clients must
    # restart both processes after this durable configuration change.
    return _classifier_routing_response(state.settings, restart_required=True)


@app.get("/api/v1/rules", response_model=RulesConfigResponse)
async def get_rules_config(request: Request) -> RulesConfigResponse:
    state = _require_state(request)
    settings: Settings = state.settings
    return _rules_response(settings)


@app.put("/api/v1/rules", response_model=RulesConfigResponse)
async def put_rules_config(payload: RulesConfigUpdate, request: Request) -> RulesConfigResponse:
    state = _require_state(request)
    settings: Settings = state.settings

    canonical_rules: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, rule_model in enumerate(payload.rules):
        raw_rule = rule_model.model_dump(mode="json")
        parsed = RuleDef.from_dict(raw_rule)
        if parsed is None:
            errors.append(f"rules[{index}]: invalid rule")
        else:
            canonical_rules.append(parsed.to_dict())
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    config_path = settings.rules_config_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config_path.with_name(f".{config_path.name}.tmp")
    tmp_path.write_text(
        json.dumps({"rules": canonical_rules}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, config_path)

    # The engine throttles its own config stat, so an explicit edit must force a
    # reload rather than wait out the TTL before the new rules can fire.
    rules_engine = getattr(getattr(state, "fusion_node", None), "rules_engine", None)
    if rules_engine is not None and hasattr(rules_engine, "reload"):
        rules_engine.reload(force=True)

    await state.live_hub.broadcast({"type": "rules_updated"})
    return _rules_response(settings)


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
        if key in _SECRET_CONFIG_KEYS and str(raw) == _REDACTED_SECRET_PLACEHOLDER:
            # GET redacts secrets to "***"; a UI that round-trips the whole block
            # would otherwise overwrite the real secret with the redaction.
            # Treat the placeholder as "unchanged" rather than as a new value.
            continue
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
        elif key == "classification_audio_source":
            v = str(value).strip().lower()
            if v not in _CLASSIFICATION_AUDIO_SOURCES:
                errors.append(
                    f"classification_audio_source: must be one of {sorted(_CLASSIFICATION_AUDIO_SOURCES)}"
                )
            else:
                value = v
        elif key == "min_localization_confidence" and not (0.0 <= value <= 1.0):  # type: ignore[operator]
            errors.append("min_localization_confidence: must be in [0, 1]")
        elif key == "localization_min_reportable_confidence" and not (
            0.0 <= value <= 1.0  # type: ignore[operator]
        ):
            errors.append("localization_min_reportable_confidence: must be in [0, 1]")
        elif key == "localization_max_reportable_gdop" and value < 0.0:  # type: ignore[operator]
            errors.append("localization_max_reportable_gdop: must be >= 0 (0 disables the gate)")
        elif key in {"birdnet_trigger_min_confidence", "birdnet_geo_min_confidence"} and not (
            0.0 <= value <= 1.0  # type: ignore[operator]
        ):
            errors.append(f"{key}: must be in [0, 1]")
        elif key in {
            "drone_head_min_confidence",
            "drone_head_min_frame_fraction",
            "stt_trigger_min_confidence",
        } and not (
            0.0 <= value <= 1.0  # type: ignore[operator]
        ):
            errors.append(f"{key}: must be in [0, 1]")
        elif key in {
            "transcript_retention_seconds",
            "omni_scan_interval_seconds",
            "omni_scan_window_seconds",
        } and value <= 0.0:  # type: ignore[operator]
            errors.append(f"{key}: must be > 0")
        elif key in {
            "retention_yamnet_audio_seconds",
            "retention_birdnet_audio_seconds",
            "retention_drone_audio_seconds",
            "retention_alert_audio_seconds",
            "retention_detection_metadata_seconds",
        } and value <= 0:
            errors.append(f"{key}: must be > 0")
        elif key in {"localization_band_min_hz", "localization_band_max_hz"} and value < 0.0:  # type: ignore[operator]
            errors.append(f"{key}: must be >= 0")
        elif key == "tracking_filter" and value not in _TRACKING_FILTERS:
            errors.append(f"tracking_filter: must be one of {sorted(_TRACKING_FILTERS)}")
        elif key == "association_distance_m" and value <= 0.0:  # type: ignore[operator]
            errors.append("association_distance_m: must be > 0")
        elif key == "association_max_gate_m" and value <= 0.0:  # type: ignore[operator]
            errors.append("association_max_gate_m: must be > 0")
        elif key == "association_chi2_gate" and value <= 0.0:  # type: ignore[operator]
            errors.append("association_chi2_gate: must be > 0")
        elif key == "kalman_process_noise" and value < 0.0:  # type: ignore[operator]
            errors.append("kalman_process_noise: must be >= 0")
        elif key == "kalman_measurement_noise" and value <= 0.0:  # type: ignore[operator]
            errors.append("kalman_measurement_noise: must be > 0")
        elif key == "track_stale_seconds" and value <= 0.0:  # type: ignore[operator]
            errors.append("track_stale_seconds: must be > 0")
        elif key == "localization_node_bearing_strength" and value < 0.0:  # type: ignore[operator]
            errors.append("localization_node_bearing_strength: must be >= 0")
        elif key == "multi_node_bearing_window_seconds" and value <= 0.0:  # type: ignore[operator]
            errors.append("multi_node_bearing_window_seconds: must be > 0")
        elif key == "multi_node_bearing_min_separation_deg" and not (0.0 <= value <= 90.0):  # type: ignore[operator]
            errors.append("multi_node_bearing_min_separation_deg: must be in [0, 90]")
        elif key == "multi_node_bearing_ttl_seconds" and value <= 0.0:  # type: ignore[operator]
            errors.append("multi_node_bearing_ttl_seconds: must be > 0")
        elif key == "multi_node_bearing_max_condition" and value <= 0.0:  # type: ignore[operator]
            errors.append("multi_node_bearing_max_condition: must be > 0")
        elif key == "coordinate_mode":
            v = str(value).strip().lower()
            if v not in _COORDINATE_MODES:
                errors.append(f"coordinate_mode: must be one of {sorted(_COORDINATE_MODES)}")
            else:
                value = v
        elif key == "hass_mqtt_port" and (value < 1 or value > 65535):  # type: ignore[operator]
            errors.append("hass_mqtt_port: must be in [1, 65535]")
        elif key == "hass_mqtt_keepalive_seconds" and value < 1:  # type: ignore[operator]
            errors.append("hass_mqtt_keepalive_seconds: must be >= 1")
        elif key == "hass_publish_interval_seconds" and value < 1.0:  # type: ignore[operator]
            # Each cycle recomputes zone occupancy (O(zones x tracks) point-in-polygon).
            errors.append("hass_publish_interval_seconds: must be >= 1.0")
        elif key == "hass_publish_min_interval_seconds" and value < 0.0:  # type: ignore[operator]
            errors.append("hass_publish_min_interval_seconds: must be >= 0")
        elif key == "hass_reconcile_interval_seconds" and value <= 0.0:  # type: ignore[operator]
            errors.append("hass_reconcile_interval_seconds: must be > 0")
        elif key == "hass_queue_size" and value < 1:  # type: ignore[operator]
            errors.append("hass_queue_size: must be >= 1")
        elif key == "hass_reconnect_backoff_initial_seconds" and value <= 0.0:  # type: ignore[operator]
            errors.append("hass_reconnect_backoff_initial_seconds: must be > 0")
        elif key == "hass_reconnect_backoff_max_seconds" and value <= 0.0:  # type: ignore[operator]
            errors.append("hass_reconnect_backoff_max_seconds: must be > 0")
        elif key == "hass_detection_off_delay_seconds" and value < 1:  # type: ignore[operator]
            errors.append("hass_detection_off_delay_seconds: must be >= 1")
        elif key == "hass_track_slot_count" and (value < 0 or value > 64):  # type: ignore[operator]
            errors.append("hass_track_slot_count: must be in [0, 64]")
        elif key == "hass_zone_spl_window_seconds" and value <= 0.0:  # type: ignore[operator]
            errors.append("hass_zone_spl_window_seconds: must be > 0")
        elif key in _HASS_TOPIC_LEVEL_KEYS and not is_valid_topic_level(str(value)):
            errors.append(f"{key}: must be a single non-empty topic level (no +, #, or /)")

        if not errors or key not in {e.split(":")[0] for e in errors}:
            coerced[key] = value

    settings: Settings = state.settings
    # Cross-field: localization band max must exceed min when the band is enabled.
    band_min = coerced.get("localization_band_min_hz", settings.localization_band_min_hz)
    band_max = coerced.get("localization_band_max_hz", settings.localization_band_max_hz)
    if float(band_max) > 0.0 and float(band_max) <= float(band_min):
        errors.append("localization_band_max_hz: must be > localization_band_min_hz when enabled")

    backoff_initial = float(
        coerced.get(
            "hass_reconnect_backoff_initial_seconds",
            settings.hass_reconnect_backoff_initial_seconds,
        )
    )
    backoff_max = float(
        coerced.get("hass_reconnect_backoff_max_seconds", settings.hass_reconnect_backoff_max_seconds)
    )
    if backoff_max < backoff_initial:
        errors.append(
            "hass_reconnect_backoff_max_seconds: must be >= hass_reconnect_backoff_initial_seconds"
        )
    hass_enabled = bool(coerced.get("hass_enabled", settings.hass_enabled))
    hass_host = str(coerced.get("hass_mqtt_host", settings.hass_mqtt_host)).strip()
    if hass_enabled and not hass_host:
        errors.append("hass_enabled: requires a non-empty hass_mqtt_host")

    if errors:
        raise HTTPException(status_code=422, detail=errors)

    for key, value in coerced.items():
        object.__setattr__(settings, key, value)

    # Persist UI-set overrides (UI-saved value wins over env on restart).
    persisted = load_overrides(settings.config_overrides_path)
    persisted.update(coerced)
    save_overrides(settings.config_overrides_path, persisted)

    restart_required = sorted(
        key
        for key in coerced
        if key in _SIDECAR_RESTART_REQUIRED_KEYS and settings.ingest_backend == "rust"
    )

    snapshot = await get_config(request)
    await state.live_hub.broadcast({"type": "config_updated", "config": snapshot})
    return {**snapshot, "restart_required": restart_required}


@app.get("/api/v1/fusion/status", response_model=FusionStatusResponse)
async def fusion_status(request: Request) -> dict:
    state = _require_state(request)
    if not hasattr(state, "fusion_node"):
        raise HTTPException(status_code=503, detail="Fusion pipeline runs in the ingest process")
    return await state.fusion_node.status()


@app.get("/api/v1/diagnostics/summary")
async def diagnostics_summary(request: Request) -> dict:
    """Simple, side-by-side-comparable summary for benchmarking this backend
    against the Rust sidecar's `GET /api/v1/diagnostics/summary`. Field names
    match the Rust response exactly. Pull-based aggregation only — reads
    `FusionMetrics` counters that are already maintained on the hot path, does
    not add any new work there beyond the counters themselves.
    """
    state = _require_state(request)
    if not hasattr(state, "fusion_node"):
        raise HTTPException(status_code=503, detail="Fusion pipeline runs in the ingest process")
    fusion_status_snapshot = await state.fusion_node.status()
    metrics = fusion_status_snapshot["metrics"]
    queue = fusion_status_snapshot["queue"]

    ingest_concurrency = getattr(state, "ingest_concurrency", None)
    overload_rejections = (
        ingest_concurrency.total_shed if isinstance(ingest_concurrency, _IngestConcurrencyLimit) else 0
    )

    uptime_seconds = max(0.0, (time.time_ns() - process_start_ns()) / 1_000_000_000.0)

    def _rate(count: int) -> float:
        return count / uptime_seconds if uptime_seconds > 0 else 0.0

    def _avg(total: float, count: int) -> float:
        return total / count if count > 0 else 0.0

    frames_received = metrics["frames_accepted"]
    frames_dropped_total = (
        metrics["frames_rejected"] + metrics["stage_drops_backpressure"] + metrics["triggers_dropped_queue_full"]
    )
    packet_loss_total = metrics["frame_sequence_gaps"]

    return {
        "backend": "python",
        "uptime_seconds": uptime_seconds,
        "frames_received": frames_received,
        "frames_processed": metrics["ingest_processing_count"],
        "frames_dropped_total": frames_dropped_total,
        "overload_rejections": overload_rejections,
        "packet_loss_total": packet_loss_total,
        "queue_depth": queue["localization_depth"],
        "queue_capacity": queue["localization_max"],
        "queue_wait_avg_ms": _avg(metrics["ingest_queue_wait_total_ms"], metrics["ingest_queue_wait_count"]),
        "queue_wait_max_ms": metrics["ingest_queue_wait_max_ms"],
        "processing_avg_ms": _avg(metrics["ingest_processing_total_ms"], metrics["ingest_processing_count"]),
        "processing_max_ms": metrics["ingest_processing_max_ms"],
        "frames_per_s": _rate(frames_received),
        "packet_loss_per_s": _rate(packet_loss_total),
    }


# ---------------------------------------------------------------------------
# Pipeline / Nodes view
# ---------------------------------------------------------------------------


def _mic_labels(count: int) -> list[str]:
    _TETRA_LABELS = ["FL", "FR", "BL", "BR"]
    if count == 4:
        return _TETRA_LABELS
    return [f"CH{i}" for i in range(count)]



def _build_rust_stages(dsp_status: dict) -> list[PipelineStageView]:
    if not isinstance(dsp_status, dict):
        return []
    total_loc = int(dsp_status.get("total_localization_attempts") or 0)
    total_loc_out = int(dsp_status.get("total_localization_results") or 0)
    total_cls = int(dsp_status.get("total_classification_attempts") or 0)
    total_cls_out = int(dsp_status.get("total_classifier_renders") or 0)
    total_fails = int(dsp_status.get("total_failures") or 0)
    total_drops = int(dsp_status.get("total_classification_drops") or 0)
    queue_depth = int(dsp_status.get("raw_manifest_queue_depth") or 0)
    pending = int(dsp_status.get("pending_manifest_count") or 0)
    return [
        PipelineStageView(name="ingest", count_in=total_loc, count_out=total_loc, drops=0, queue_depth=queue_depth),
        PipelineStageView(name="localization", count_in=total_loc, count_out=total_loc_out, drops=total_fails, queue_depth=pending),
        PipelineStageView(name="classification", count_in=total_cls, count_out=total_cls_out, drops=total_drops),
        PipelineStageView(name="track", count_in=total_cls_out, count_out=total_cls_out, drops=0),
        PipelineStageView(name="rules", count_in=total_cls_out, count_out=total_cls_out, drops=0),
    ]


def _fetch_json_from_sidecar(base_url: str, path: str, body: dict | None = None) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json"},
            )
        else:
            req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


@app.get("/api/v1/pipeline/nodes", response_model=PipelineNodesResponse)
async def get_pipeline_nodes(request: Request) -> PipelineNodesResponse:
    state = _require_state(request)
    settings: Settings = state.settings
    if _ingest_sidecar_is_running(state) and settings.ingest_backend == "rust":
        dsp_status = await asyncio.to_thread(
            _fetch_json_from_sidecar,
            _ingest_runtime_base_url(settings),
            "/api/v1/dsp/status",
        )
        sidecar_snapshots = _sidecar_stream_consumer_snapshots(state)
        now_ns = time.time_ns()
        nodes_out: list[PipelineNodeView] = []
        for node_id, snapshot in sidecar_snapshots.items():
            audio_debug = _sidecar_snapshot_audio_debug(
                snapshot,
                now_ns=now_ns,
                degraded_after_seconds=settings.node_degraded_after_seconds,
            ) or {}
            sensor_count = int(audio_debug.get("sensor_count") or 1)
            sample_rate_hz = audio_debug.get("sample_rate_hz")
            last_ns = audio_debug.get("last_sample_time_ns")
            audio_status = str(audio_debug.get("status") or "unknown")
            rms_val = audio_debug.get("rms")
            rms_list = [float(rms_val)] if isinstance(rms_val, (int, float)) else []

            overrides: dict = settings.node_audio_overrides.get(node_id) or {}
            mic_count = max(1, sensor_count)
            labels = _mic_labels(mic_count)
            mic_gains_db: list[float] = overrides.get("mic_gains_db") or [0.0] * mic_count
            hp = float(overrides.get("hp_hz", settings.audio_highpass_hz))
            lp = float(overrides.get("lp_hz", settings.audio_lowpass_hz))
            smoothing = overrides.get("smoothing") or "off"

            mics = [
                MicView(
                    index=i,
                    label=labels[i] if i < len(labels) else f"CH{i}",
                    gain_db=float(mic_gains_db[i]) if i < len(mic_gains_db) else 0.0,
                    hp_hz=hp,
                    lp_hz=lp,
                    smoothing=smoothing,
                    rms_recent=rms_list if i == 0 else [],
                )
                for i in range(mic_count)
            ]
            nodes_out.append(PipelineNodeView(
                node_id=node_id,
                node_type="sirith_tetra",
                mics=mics,
                stages=_build_rust_stages(dsp_status),
                audio_override=NodeAudioOverride.model_validate(overrides) if overrides else None,
                last_frame_ns=int(last_ns) if isinstance(last_ns, int) else None,
                sample_rate_hz=int(sample_rate_hz) if isinstance(sample_rate_hz, int) else None,
                audio_status=audio_status,
            ))
        return PipelineNodesResponse(active_pipeline="rust", nodes=nodes_out)

    # Python ingest path.
    fusion_node: FusionNode | None = getattr(state, "fusion_node", None)
    realtime: dict = {}
    node_frame_metrics: dict = {}
    queue_sizes: dict = {}
    if fusion_node is not None:
        realtime = fusion_node._realtime_tracker.snapshot(now_ns=time.time_ns())
        node_frame_metrics = fusion_node.node_frame_metrics()
        queue_sizes = {
            "localization": (fusion_node._localization_queue.qsize(), fusion_node._localization_queue.maxsize),
            "classification": (fusion_node._classification_queue.qsize(), fusion_node._classification_queue.maxsize),
            "rules": (fusion_node._rules_queue.qsize(), fusion_node._rules_queue.maxsize),
        }

    def _lag(stage: str) -> float | None:
        stages_rt = realtime.get("stages") or {}
        info = stages_rt.get(stage) or {}
        v = info.get("seconds_behind_realtime")
        return float(v) if v is not None else None

    m = fusion_node._metrics if fusion_node is not None else None
    loc_q, loc_max = queue_sizes.get("localization", (0, 0))
    cls_q, cls_max = queue_sizes.get("classification", (0, 0))
    rls_q, rls_max = queue_sizes.get("rules", (0, 0))
    stages = [
        PipelineStageView(
            name="ingest",
            count_in=m.ingest_requests if m else 0,
            count_out=m.frames_accepted if m else 0,
            drops=m.frames_rejected if m else 0,
        ),
        PipelineStageView(
            name="localization",
            count_in=m.localization_stage_in if m else 0,
            count_out=m.localization_stage_out if m else 0,
            drops=m.localization_failures if m else 0,
            queue_depth=loc_q,
            queue_max=loc_max,
            lag_s=_lag("localization"),
        ),
        PipelineStageView(
            name="classification",
            count_in=m.classification_stage_in if m else 0,
            count_out=m.classification_stage_out if m else 0,
            drops=m.classification_failures if m else 0,
            queue_depth=cls_q,
            queue_max=cls_max,
            lag_s=_lag("classification"),
        ),
        PipelineStageView(
            name="track",
            count_in=m.rules_stage_in if m else 0,
            count_out=m.rules_stage_out if m else 0,
            drops=m.rules_failures if m else 0,
            queue_depth=rls_q,
            queue_max=rls_max,
            lag_s=_lag("rules"),
        ),
        PipelineStageView(
            name="rules",
            count_in=m.detections_emitted if m else 0,
            count_out=m.detections_emitted if m else 0,
            drops=0,
        ),
    ]

    overall_lag = realtime.get("pipeline_seconds_behind_realtime")
    audio_buffer: MultiSensorBuffer | None = getattr(state, "audio_buffer", None)
    nodes_out = []
    seen_ids: set = set()
    for node_id, pnm in node_frame_metrics.items():
        seen_ids.add(node_id)
        overrides = settings.node_audio_overrides.get(node_id) or {}

        # Resolve actual mic count from registry; fall back to override hints.
        sensor_descriptors = []
        if fusion_node is not None:
            sensor_descriptors = await fusion_node.registry.sensors_for_node(node_id)
        mic_count = max(1, len(sensor_descriptors))

        labels = _mic_labels(mic_count)
        mic_gains_db: list[float] = overrides.get("mic_gains_db") or [0.0] * mic_count
        hp = float(overrides.get("hp_hz", settings.audio_highpass_hz))
        lp = float(overrides.get("lp_hz", settings.audio_lowpass_hz))
        smoothing = overrides.get("smoothing") or "off"

        mics = []
        for i in range(mic_count):
            sensor_id = f"{node_id}:ch{i}"
            rms_history: list[float] = []
            if audio_buffer is not None:
                rms_history = await audio_buffer.get_sensor_rms_history(sensor_id)
            mics.append(MicView(
                index=i,
                label=labels[i] if i < len(labels) else f"CH{i}",
                gain_db=float(mic_gains_db[i]) if i < len(mic_gains_db) else 0.0,
                hp_hz=hp,
                lp_hz=lp,
                smoothing=smoothing,
                rms_recent=rms_history,
            ))

        last_ns = pnm.get("last_frame_ns") or None
        if last_ns == 0:
            last_ns = None
        nodes_out.append(PipelineNodeView(
            node_id=node_id,
            node_type="unknown",
            mics=mics,
            stages=stages,
            audio_override=NodeAudioOverride.model_validate(overrides) if overrides else None,
            frame_gaps=pnm.get("frame_gaps", 0),
            last_frame_ns=last_ns,
            audio_status="recent" if last_ns is not None else "unknown",
        ))
    for node_id in list(settings.node_audio_overrides.keys()):
        if node_id not in seen_ids:
            seen_ids.add(node_id)
            overrides = settings.node_audio_overrides.get(node_id) or {}
            hp = float(overrides.get("hp_hz", settings.audio_highpass_hz))
            lp = float(overrides.get("lp_hz", settings.audio_lowpass_hz))
            smoothing = overrides.get("smoothing") or "off"
            mics = [MicView(index=0, label="CH0", hp_hz=hp, lp_hz=lp, smoothing=smoothing)]
            nodes_out.append(PipelineNodeView(
                node_id=node_id,
                node_type="unknown",
                mics=mics,
                stages=stages,
                audio_override=NodeAudioOverride.model_validate(overrides) if overrides else None,
            ))
    return PipelineNodesResponse(
        active_pipeline="python",
        nodes=nodes_out,
        pipeline_seconds_behind_realtime=float(overall_lag) if overall_lag is not None else None,
    )

@app.get("/api/v1/pipeline/graph", response_model=PipelineGraph)
async def get_pipeline_graph(request: Request) -> PipelineGraph:
    """Read-only pipeline-flow DAG (structure + live status overlays).

    Single endpoint for initial load and status polling; ``structure_hash``
    lets the frontend skip re-layout when only status changed. Always returns
    200 — when no fusion node / sidecar is available the graph still renders
    with ``fusion_available=False`` and ``unknown`` health.
    """
    state = _require_state(request)
    settings: Settings = state.settings
    now_ns = time.time_ns()

    # Nodes: prefer live registry specs, fall back to stored specs.
    node_specs: dict[str, NodeSpec] = {}
    for row in await state.storage.list_nodes(limit=4096):
        try:
            spec = NodeSpec.model_validate(row)
        except Exception:
            continue
        node_specs[spec.id] = spec
    if _has_live_ingest_runtime(state):
        for runtime in await state.registry.list_nodes():
            node_specs[runtime.spec.id] = runtime.spec

    routing = load_routing(settings)

    fusion_node: FusionNode | None = getattr(state, "fusion_node", None)
    fusion_status: dict | None = None
    rules_engine = None
    if fusion_node is not None:
        fusion_status = await fusion_node.status()
        rules_engine = fusion_node.rules_engine
    if rules_engine is None:
        rules_engine = ConfigRuleEngine(settings.rules_config_path)
    rules = rules_engine.rules()

    sidecar_dsp_status: dict | None = None
    active_pipeline = "python"
    if _ingest_sidecar_is_running(state) and settings.ingest_backend == "rust":
        active_pipeline = "rust"
        sidecar_dsp_status = await asyncio.to_thread(
            _fetch_json_from_sidecar,
            _ingest_runtime_base_url(settings),
            "/api/v1/dsp/status",
        )

    return build_pipeline_graph(
        settings=settings,
        nodes=list(node_specs.values()),
        routing=routing,
        rules=rules,
        fusion_status=fusion_status,
        sidecar_dsp_status=sidecar_dsp_status,
        active_pipeline=active_pipeline,
        now_ns=now_ns,
    )


def _validate_stage_float(raw_value: object, *, stage_index: int, field_name: str) -> float:
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise HTTPException(
            status_code=422,
            detail=f"stages[{stage_index}].{field_name} must be a number",
        )
    value = float(raw_value)
    if not np.isfinite(value):
        raise HTTPException(
            status_code=422,
            detail=f"stages[{stage_index}].{field_name} must be finite",
        )
    return value


def _validate_stage_order(raw_value: object | None, *, stage_index: int) -> int:
    if raw_value is None:
        return 4
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise HTTPException(
            status_code=422,
            detail=f"stages[{stage_index}].order must be an integer in [2, 8]",
        )
    order = float(raw_value)
    if not np.isfinite(order) or not order.is_integer():
        raise HTTPException(
            status_code=422,
            detail=f"stages[{stage_index}].order must be an integer in [2, 8]",
        )
    order_int = int(order)
    if order_int < 2 or order_int > 8:
        raise HTTPException(
            status_code=422,
            detail=f"stages[{stage_index}].order must be in [2, 8]",
        )
    return order_int


def _canonicalize_preprocess_stages(raw_stages: list[dict[str, Any]]) -> list[dict[str, object]]:
    canonical_stages: list[dict[str, object]] = []
    for stage_index, raw_stage in enumerate(raw_stages):
        if not isinstance(raw_stage, dict):
            raise HTTPException(
                status_code=422,
                detail=f"stages[{stage_index}] must be an object",
            )
        stage_type_raw = raw_stage.get("type")
        if not isinstance(stage_type_raw, str) or not stage_type_raw.strip():
            raise HTTPException(
                status_code=422,
                detail=f"stages[{stage_index}].type must be a non-empty string",
            )
        stage_type = stage_type_raw.strip().lower()
        if stage_type == "gain":
            db = _validate_stage_float(raw_stage.get("db"), stage_index=stage_index, field_name="db")
            if db < -60.0 or db > 60.0:
                raise HTTPException(
                    status_code=422,
                    detail=f"stages[{stage_index}].db must be in [-60, 60]",
                )
            canonical_stages.append({"type": "gain", "db": db})
            continue
        if stage_type == "channel_gain":
            raw_gains = raw_stage.get("db_by_channel")
            if not isinstance(raw_gains, list) or not raw_gains:
                raise HTTPException(
                    status_code=422,
                    detail=f"stages[{stage_index}].db_by_channel must be a non-empty list",
                )
            gains = [
                _validate_stage_float(value, stage_index=stage_index, field_name="db_by_channel")
                for value in raw_gains
            ]
            if any(value < -60.0 or value > 60.0 for value in gains):
                raise HTTPException(
                    status_code=422,
                    detail=f"stages[{stage_index}].db_by_channel values must be in [-60, 60]",
                )
            canonical_stages.append({"type": "channel_gain", "db_by_channel": gains})
            continue
        if stage_type in {"highpass", "lowpass"}:
            cutoff_hz = _validate_stage_float(
                raw_stage.get("cutoff_hz"),
                stage_index=stage_index,
                field_name="cutoff_hz",
            )
            if cutoff_hz <= 0.0:
                raise HTTPException(
                    status_code=422,
                    detail=f"stages[{stage_index}].cutoff_hz must be > 0",
                )
            canonical_stages.append(
                {
                    "type": stage_type,
                    "cutoff_hz": cutoff_hz,
                    "order": _validate_stage_order(raw_stage.get("order"), stage_index=stage_index),
                }
            )
            continue
        if stage_type == "bandpass":
            low_hz = _validate_stage_float(raw_stage.get("low_hz"), stage_index=stage_index, field_name="low_hz")
            high_hz = _validate_stage_float(raw_stage.get("high_hz"), stage_index=stage_index, field_name="high_hz")
            if low_hz <= 0.0:
                raise HTTPException(
                    status_code=422,
                    detail=f"stages[{stage_index}].low_hz must be > 0",
                )
            if high_hz <= low_hz:
                raise HTTPException(
                    status_code=422,
                    detail=f"stages[{stage_index}].high_hz must be > low_hz",
                )
            canonical_stages.append(
                {
                    "type": "bandpass",
                    "low_hz": low_hz,
                    "high_hz": high_hz,
                    "order": _validate_stage_order(raw_stage.get("order"), stage_index=stage_index),
                }
            )
            continue
        if stage_type in {"dc_block", "passthrough"}:
            canonical_stages.append({"type": stage_type})
            continue
        raise HTTPException(
            status_code=422,
            detail=(
                f"stages[{stage_index}].type must be one of: gain, channel_gain, highpass, lowpass, "
                "bandpass, dc_block, passthrough"
            ),
        )
    return canonical_stages


def _build_legacy_node_audio_override(body: NodeAudioOverride) -> dict[str, object]:
    override_dict: dict[str, object] = {}
    channel_gains_db = body.channel_gains_db if body.channel_gains_db is not None else body.mic_gains_db
    if channel_gains_db is not None:
        for db in channel_gains_db:
            if db < -60.0 or db > 60.0:
                raise HTTPException(status_code=422, detail="channel gain values must be in [-60, 60] dB")
        override_dict["mic_gains_db"] = channel_gains_db
        override_dict["channel_gains_db"] = channel_gains_db
        override_dict["stages"] = [{"type": "channel_gain", "db_by_channel": channel_gains_db}]
    if body.hp_hz is not None:
        if body.hp_hz < 0.0:
            raise HTTPException(status_code=422, detail="hp_hz must be >= 0")
        override_dict["hp_hz"] = body.hp_hz
        if "stages" in override_dict and body.hp_hz > 0.0:
            override_dict["stages"].append(
                {"type": "highpass", "cutoff_hz": body.hp_hz, "order": 4}
            )
    if body.lp_hz is not None:
        if body.lp_hz < 0.0:
            raise HTTPException(status_code=422, detail="lp_hz must be >= 0")
        override_dict["lp_hz"] = body.lp_hz
        if "stages" in override_dict and body.lp_hz > 0.0:
            override_dict["stages"].append(
                {"type": "lowpass", "cutoff_hz": body.lp_hz, "order": 4}
            )
    if body.smoothing is not None:
        override_dict["smoothing"] = body.smoothing
    return override_dict


def _canonicalize_node_audio_override(
    body: NodeAudioOverride,
    existing_override: dict[str, object] | None,
) -> dict[str, object]:
    legacy_override = _build_legacy_node_audio_override(body)
    if body.stages is not None:
        canonical_stages = _canonicalize_preprocess_stages(body.stages)
        if canonical_stages:
            return {"stages": canonical_stages}
        return legacy_override
    if "stages" in legacy_override:
        return legacy_override
    existing_stages = existing_override.get("stages") if isinstance(existing_override, dict) else None
    if isinstance(existing_stages, list) and existing_stages:
        return dict(existing_override)
    return legacy_override


@app.patch("/api/v1/pipeline/nodes/{node_id}/audio")
async def patch_node_audio(node_id: str, body: NodeAudioOverride, request: Request) -> dict:
    state = _require_state(request)
    settings: Settings = state.settings

    existing_override = settings.node_audio_overrides.get(node_id)
    override_dict = _canonicalize_node_audio_override(body, existing_override)

    if override_dict:
        settings.node_audio_overrides[node_id] = override_dict
    else:
        settings.node_audio_overrides.pop(node_id, None)

    # Apply immediately to the Python ingest preprocessor if running.
    fusion_node: FusionNode | None = getattr(state, "fusion_node", None)
    rust_active = _ingest_sidecar_is_running(state) and settings.ingest_backend == "rust"
    sidecar_forward_ok = False

    if fusion_node is not None:
        # The Python-side factory remains the source of diagnostics and digital-trim
        # compensation even when Rust owns the actual ingest processing.
        fusion_node.apply_node_audio_override(node_id, override_dict if override_dict else None)
    if rust_active:
        # Forward to the Rust sidecar so it takes effect in the live DSP path.
        sidecar_payload = {"node_id": node_id, **override_dict}
        # Remove None values — sidecar treats missing keys as "unchanged".
        sidecar_payload = {k: v for k, v in sidecar_payload.items() if v is not None}
        try:
            sidecar_forward_ok = await asyncio.to_thread(
                _fetch_json_from_sidecar,
                _ingest_runtime_base_url(settings),
                "/api/v1/dsp/config",
                sidecar_payload,
            ) is not None
        except Exception:
            sidecar_forward_ok = False

    return {
        "node_id": node_id,
        "override": override_dict,
        "applied_to_pipeline": (fusion_node is not None) or sidecar_forward_ok,
        "rust_sidecar_active": rust_active,
        "rust_sidecar_forwarded": sidecar_forward_ok,
    }


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


@app.get("/api/v1/integrations/hass/status", response_model=HassBridgeStatusResponse)
async def hass_status(request: Request) -> dict:
    """Live bridge state. 200 with ``connection_state="disabled"`` when absent —
    the Settings page needs something to render in every deployment shape,
    including the api-only role where no bridge is built."""
    state = _require_state(request)
    bridge = getattr(state, "hass_bridge", None)
    if bridge is None:
        from minimappr.core.hass.aiomqtt_transport import aiomqtt_available

        settings: Settings = state.settings
        return {
            "enabled": False,
            "connection_state": "disabled",
            "transport": None,
            "transport_available": aiomqtt_available(),
            "mqtt_host": settings.hass_mqtt_host,
            "mqtt_port": settings.hass_mqtt_port,
            "mqtt_tls_enabled": settings.hass_mqtt_tls_enabled,
            "discovery_prefix": settings.hass_discovery_prefix,
            "base_topic": settings.hass_base_topic,
            "device_id": settings.hass_device_id,
        }
    return bridge.status()


def _require_hass_bridge(state):
    """503 rather than a silent success: an operator clicking "purge" needs to
    know nothing happened."""
    bridge = getattr(state, "hass_bridge", None)
    if bridge is None or not bridge.enabled:
        raise HTTPException(status_code=503, detail="The Home Assistant bridge is not enabled")
    return bridge


@app.post("/api/v1/integrations/hass/republish-discovery")
async def hass_republish_discovery(request: Request) -> dict:
    """Force a full reconcile + snapshot next cycle.

    Recovery path after purging retained messages on the broker by hand: the
    ledger would otherwise consider every entity already-published and skip it.
    """
    bridge = _require_hass_bridge(_require_state(request))
    bridge.forget_published_state()
    bridge.request_reconcile()
    return {"ok": True, "connection_state": bridge.connection_state}


@app.post("/api/v1/integrations/hass/purge-discovery")
async def hass_purge_discovery(request: Request) -> dict:
    """Blank every retained topic we published and clear the ledger.

    Run this before uninstalling, or HA keeps the entities forever as
    permanently-unavailable rows in its registry.
    """
    bridge = _require_hass_bridge(_require_state(request))
    queued = await bridge.purge_discovery()
    return {"ok": True, "queued_removals": queued}


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
    now_ns = time.time_ns()
    node_counts = await _runtime_node_health_counts(state, now_ns=now_ns)
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


@app.get("/api/v1/transcripts")
async def list_transcripts(
    request: Request,
    since_ns: int | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
) -> list[dict]:
    state = _require_state(request)
    return await state.storage.list_transcripts(since_ns=since_ns, limit=limit)


@app.get("/api/v1/transcripts/{transcript_id}")
async def get_transcript(transcript_id: str, request: Request) -> dict:
    """Return one persisted transcript for the transcript-review view."""
    state = _require_state(request)
    row = await state.storage.get_transcript(transcript_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return row


@app.get("/api/v1/transcripts/{transcript_id}/audio")
async def get_transcript_audio(transcript_id: str, request: Request) -> FileResponse:
    state = _require_state(request)
    row = await state.storage.get_transcript(transcript_id)
    if row is None or not row.get("audio_path"):
        raise HTTPException(status_code=404, detail="Transcript audio not found")
    audio_path = Path(str(row["audio_path"]))
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Transcript audio not found")
    return FileResponse(path=audio_path, media_type="audio/wav")


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


_OVERLAY_MAX_BYTES = 20 * 1024 * 1024
_OVERLAY_EXTENSIONS = {
    MapOverlayKind.IMAGE: {".png", ".jpg", ".jpeg", ".webp"},
    MapOverlayKind.SVG: {".svg"},
    MapOverlayKind.GEOJSON: {".json", ".geojson"},
}
_OVERLAY_MIME_PREFIXES = {
    MapOverlayKind.IMAGE: ("image/png", "image/jpeg", "image/webp"),
    MapOverlayKind.SVG: ("image/svg+xml",),
    MapOverlayKind.GEOJSON: ("application/json", "application/geo+json", "text/plain"),
}


def _overlay_row_to_spec(row: dict[str, Any]) -> MapOverlaySpec:
    return MapOverlaySpec(
        id=row["id"],
        name=row["name"],
        kind=row["kind"],
        content_url=f"/api/v1/overlays/{row['id']}/content",
        mime=row["mime"],
        bounds=row.get("bounds") or [],
        opacity=float(row.get("opacity", 0.75)),
        storey=row.get("storey"),
        enabled=bool(row.get("enabled", True)),
        created_ns=int(row["created_ns"]),
        metadata=row.get("metadata") or {},
    )


def _parse_overlay_bounds(bounds: str | None) -> list[list[float]]:
    if bounds is None or not bounds.strip():
        return []
    try:
        parsed = json.loads(bounds)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid bounds JSON: {exc}") from exc
    update = MapOverlayUpdate(bounds=parsed)
    return update.bounds or []


def _overlay_extension_for_upload(filename: str, kind: MapOverlayKind) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in _OVERLAY_EXTENSIONS[kind]:
        raise HTTPException(status_code=415, detail=f"{kind.value} overlays do not accept {suffix or 'extensionless'} files")
    return suffix


def _validate_overlay_mime(content_type: str | None, kind: MapOverlayKind) -> str:
    mime = (content_type or "application/octet-stream").split(";", 1)[0].strip().lower()
    if mime not in _OVERLAY_MIME_PREFIXES[kind]:
        raise HTTPException(status_code=415, detail=f"{kind.value} overlay MIME {mime!r} is not accepted")
    return mime


async def _broadcast_overlay_updated(state, overlay_id: str | None = None) -> None:
    await state.live_hub.broadcast({"type": "overlay_updated", "overlay_id": overlay_id})


@app.get("/api/v1/overlays", response_model=list[MapOverlaySpec])
async def list_overlays(request: Request) -> list[MapOverlaySpec]:
    state = _require_state(request)
    return [_overlay_row_to_spec(row) for row in await state.storage.list_map_overlays()]


@app.post("/api/v1/overlays", response_model=MapOverlaySpec, status_code=201)
async def upload_overlay(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(...),
    kind: MapOverlayKind = Form(...),
    bounds: str | None = Form(default=None),
    storey: str | None = Form(default=None),
    opacity: float = Form(default=0.75, ge=0.0, le=1.0),
) -> MapOverlaySpec:
    state = _require_state(request)
    filename = file.filename or ""
    extension = _overlay_extension_for_upload(filename, kind)
    mime = _validate_overlay_mime(file.content_type, kind)
    data = await file.read()
    if len(data) > _OVERLAY_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Overlay file exceeds 20 MB")
    if not name.strip():
        raise HTTPException(status_code=422, detail="Overlay name is required")
    overlay_id = f"ovl-{uuid.uuid4().hex[:16]}"
    overlay_dir: Path = state.settings.map_overlay_dir
    overlay_dir.mkdir(parents=True, exist_ok=True)
    file_path = overlay_dir / f"{overlay_id}{extension}"
    file_path.write_bytes(data)
    created_ns = time.time_ns()
    parsed_bounds = _parse_overlay_bounds(bounds)
    await state.storage.upsert_map_overlay(
        overlay_id=overlay_id,
        name=name.strip(),
        kind=kind.value,
        file_path=str(file_path),
        mime=mime,
        bounds=parsed_bounds,
        opacity=float(opacity),
        storey=storey.strip() if storey and storey.strip() else None,
        enabled=True,
        created_ns=created_ns,
        metadata={},
    )
    await _broadcast_overlay_updated(state, overlay_id)
    row = await state.storage.get_map_overlay(overlay_id)
    if row is None:
        raise HTTPException(status_code=500, detail="Overlay upload did not persist")
    return _overlay_row_to_spec(row)


@app.patch("/api/v1/overlays/{overlay_id}", response_model=MapOverlaySpec)
async def patch_overlay(overlay_id: str, payload: MapOverlayUpdate, request: Request) -> MapOverlaySpec:
    state = _require_state(request)
    row = await state.storage.get_map_overlay(overlay_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Overlay not found")
    await state.storage.upsert_map_overlay(
        overlay_id=overlay_id,
        name=payload.name.strip() if payload.name and payload.name.strip() else row["name"],
        kind=row["kind"],
        file_path=row["file_path"],
        mime=row["mime"],
        bounds=payload.bounds if payload.bounds is not None else row.get("bounds", []),
        opacity=payload.opacity if payload.opacity is not None else float(row["opacity"]),
        storey=payload.storey if payload.storey is not None else row.get("storey"),
        enabled=payload.enabled if payload.enabled is not None else bool(row.get("enabled", True)),
        created_ns=int(row["created_ns"]),
        metadata=payload.metadata if payload.metadata is not None else row.get("metadata", {}),
    )
    await _broadcast_overlay_updated(state, overlay_id)
    updated = await state.storage.get_map_overlay(overlay_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Overlay update did not persist")
    return _overlay_row_to_spec(updated)


@app.get("/api/v1/overlays/{overlay_id}/content")
async def get_overlay_content(overlay_id: str, request: Request) -> FileResponse:
    state = _require_state(request)
    row = await state.storage.get_map_overlay(overlay_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Overlay not found")
    overlay_root: Path = state.settings.map_overlay_dir.resolve()
    overlay_path = Path(row["file_path"]).resolve()
    if not overlay_path.exists():
        raise HTTPException(status_code=404, detail="Overlay file no longer exists")
    if not overlay_path.is_relative_to(overlay_root):
        raise HTTPException(status_code=403, detail="Overlay path is outside overlay directory")
    return FileResponse(path=overlay_path, media_type=row["mime"])


@app.delete("/api/v1/overlays/{overlay_id}")
async def delete_overlay(overlay_id: str, request: Request) -> dict:
    state = _require_state(request)
    row = await state.storage.get_map_overlay(overlay_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Overlay not found")
    deleted = await state.storage.delete_map_overlay(overlay_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Overlay not found")
    overlay_root: Path = state.settings.map_overlay_dir.resolve()
    overlay_path = Path(row["file_path"]).resolve()
    if overlay_path.is_relative_to(overlay_root):
        overlay_path.unlink(missing_ok=True)
    await _broadcast_overlay_updated(state, overlay_id)
    return {"ok": True, "overlay_id": overlay_id}


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
    _request_hass_reconcile(state)
    return {"ok": True, "zone_id": zone_id}


@app.delete("/api/v1/zones/{zone_id}")
async def delete_zone(zone_id: str, request: Request) -> dict:
    state = _require_state(request)
    deleted = await state.storage.delete_zone(zone_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Zone not found")
    _request_hass_reconcile(state)
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


@app.get("/api/v1/labels")
async def list_known_labels(request: Request) -> dict:
    """Return every classifier- or operator-known label for editable UI suggestions."""
    state = _require_state(request)
    labels = await state.storage.list_labels()
    return {
        "labels": [
            {
                "name": row["name"],
                "category": row.get("category") or "unknown",
                "source": row.get("source"),
            }
            for row in labels
        ]
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
    ingest_concurrency = getattr(state, "ingest_concurrency", None)
    capture_available, capture_unavailable_reason = _capture_pipeline_status(state)
    diagnostics["ingest"] = {
        "backend": settings.ingest_backend,
        "host": settings.ingest_host,
        "port": settings.ingest_port,
        "base_url": settings.ingest_base_url,
        "capture_available": capture_available,
        "capture_unavailable_reason": None if capture_available else capture_unavailable_reason,
        "concurrency_limit": {
            "max_concurrent": (
                ingest_concurrency.max_concurrent
                if isinstance(ingest_concurrency, _IngestConcurrencyLimit)
                else None
            ),
            "active": (
                ingest_concurrency.active
                if isinstance(ingest_concurrency, _IngestConcurrencyLimit)
                else None
            ),
            "total_admissions": (
                ingest_concurrency.total_admissions
                if isinstance(ingest_concurrency, _IngestConcurrencyLimit)
                else None
            ),
            "total_shed": (
                ingest_concurrency.total_shed
                if isinstance(ingest_concurrency, _IngestConcurrencyLimit)
                else None
            ),
        },
        "request_timeout_seconds": settings.ingest_request_timeout_seconds,
    }
    if hasattr(state, "fusion_node"):
        fusion_status = await state.fusion_node.status()
        diagnostics["pipeline"] = {
            "queue": fusion_status["queue"],
            "workers": fusion_status["workers"],
            "realtime": fusion_status["realtime"],
            "drop_on_backpressure": fusion_status["drop_on_backpressure"],
            "buffer_state": fusion_status.get("buffer_state", []),
            "health": fusion_status.get("health", {}),
            "metrics": {
                "triggers_enqueued": fusion_status["metrics"].get("triggers_enqueued", 0),
                "triggers_dropped_queue_full": fusion_status["metrics"].get("triggers_dropped_queue_full", 0),
                "stage_drops_backpressure": fusion_status["metrics"].get("stage_drops_backpressure", 0),
                "birdnet_chunk_dispatches_suppressed": (
                    fusion_status["metrics"].get("birdnet_chunk_dispatches_suppressed", 0)
                ),
                "detections_emitted": fusion_status["metrics"].get("detections_emitted", 0),
                # Silent-drop visibility — these counters are the canonical
                # "pipeline is silently dropping" signal for the Python ingest.
                # See FusionMetrics in core/fusion_node.py for full set.
                "localization_drops_by_reason": (
                    fusion_status["metrics"].get("localization_drops_by_reason", {})
                ),
                "classification_drops_by_reason": (
                    fusion_status["metrics"].get("classification_drops_by_reason", {})
                ),
                "rules_drops_by_reason": (
                    fusion_status["metrics"].get("rules_drops_by_reason", {})
                ),
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
    sidecar_dsp_metrics: dict[str, object] | None = None
    if settings.ingest_sidecar_enabled and sidecar_state is not None and sidecar_state.status == "running":
        sidecar_startup = _ingest_sidecar_startup_config(settings)
        sidecar_health = await asyncio.to_thread(
            _fetch_ingest_sidecar_health,
            settings.ingest_port,
            sidecar_startup.healthcheck_timeout_seconds,
        )
        # Pull DSP counters so the diagnostics page can surface the Rust-side
        # silent-drop signals (total_buffer_reanchors,
        # total_window_underrun_drops). The endpoint is the same one
        # `_build_rust_stages` consumes; the fetch is best-effort.
        dsp_status_raw = await asyncio.to_thread(
            _fetch_json_from_sidecar,
            _ingest_runtime_base_url(settings),
            "/api/v1/dsp/status",
        )
        if isinstance(dsp_status_raw, dict) and dsp_status_raw:
            sidecar_dsp_metrics = {
                "total_failures": dsp_status_raw.get("total_failures", 0),
                "total_classification_drops": dsp_status_raw.get(
                    "total_classification_drops", 0
                ),
                "total_stale_manifest_skips": dsp_status_raw.get(
                    "total_stale_manifest_skips", 0
                ),
                # Mirrors of Python silent-drop counters — names kept stable so
                # an alert query (`sidecar.dsp_metrics.total_window_underrun_drops`
                # vs. `pipeline.metrics.localization_drops_by_reason.no_window`)
                # works against either backend.
                "total_buffer_reanchors": dsp_status_raw.get(
                    "total_buffer_reanchors", 0
                ),
                "total_window_underrun_drops": dsp_status_raw.get(
                    "total_window_underrun_drops", 0
                ),
            }
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
        "dsp_metrics": sidecar_dsp_metrics,
        "log_source": "stderr_only",  # Rust sidecar has no /api/v1/system/logs.
        "stream_consumer": {
            "configured": bool(
                settings.ingest_backend == "rust"
                and sidecar_state is not None
                and sidecar_state.status == "running"
            ),
            "present": getattr(state, "ingest_stream_consumer", None) is not None,
            "running": bool(getattr(getattr(state, "ingest_stream_consumer", None), "is_running", False)),
            "last_event_id": getattr(getattr(state, "ingest_stream_consumer", None), "_last_event_id", None),
            "sidecar_base_url": getattr(
                getattr(getattr(state, "ingest_stream_consumer", None), "_config", None),
                "sidecar_base_url",
                None,
            ),
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
    """Recent log records from the in-process ring buffer.

    In split-process mode (process_role == "api") the ingest process owns the
    fusion pipeline and therefore the only records that diagnose pipeline
    issues. To keep operators from having to curl two ports, the API process
    merges its own ring with a proxy fetch of the ingest process's ring and
    tags each record with `source: "local" | "ingest"`. The two processes
    maintain independent `seq` counters, so callers using `since_seq` should
    track per-source cursors via `sources.local_max_seq` / `sources.ingest_max_seq`.
    """
    from minimappr.core.logging_ring import global_handler

    handler = global_handler()
    level_no = logging.getLevelName(level.upper()) if isinstance(level, str) else logging.INFO
    if not isinstance(level_no, int):
        level_no = logging.INFO

    local_records: list[dict] = []
    local_capacity = 0
    if handler is not None:
        raw = handler.snapshot(
            limit=limit,
            min_level=level_no,
            logger_prefix=logger_prefix,
            since_seq=since_seq,
        )
        for record in raw:
            record["source"] = "local"
        local_records = raw
        local_capacity = handler._buffer.maxlen or 0

    state = _require_state(request)
    settings: Settings = state.settings
    process_role = getattr(settings, "process_role", "combined")
    should_proxy = (
        process_role == "api"
        and settings.ingest_port != settings.port
    )
    ingest_records: list[dict] = []
    ingest_capacity = 0
    proxy_error: str | None = None
    if should_proxy:
        proxy_params: dict[str, Any] = {
            "limit": limit,
            "level": level,
        }
        if logger_prefix is not None:
            proxy_params["logger_prefix"] = logger_prefix
        if since_seq is not None:
            proxy_params["since_seq"] = since_seq
        try:
            import httpx

            url = f"http://127.0.0.1:{settings.ingest_port}/api/v1/system/logs"
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(url, params=proxy_params)
            response.raise_for_status()
            payload = response.json()
            ingest_capacity = int(payload.get("capacity", 0))
            for record in payload.get("records", []):
                # Avoid overwriting the ingest's own "local" tag if it ever
                # proxies further; always retag as "ingest" from this hop.
                record["source"] = "ingest"
                ingest_records.append(record)
        except Exception as exc:  # pragma: no cover - network resilience path
            proxy_error = f"{type(exc).__name__}: {exc}"

    merged = local_records + ingest_records
    return {
        "records": merged,
        "capacity": local_capacity,
        "sources": {
            "local": len(local_records),
            "local_capacity": local_capacity,
            "ingest": len(ingest_records),
            "ingest_capacity": ingest_capacity,
            "proxy_error": proxy_error,
        },
    }


@app.post("/api/v1/clusters", response_model=ClusterSpec, status_code=201)
async def create_or_update_cluster(request: Request, spec: ClusterSpec) -> ClusterSpec:
    """Create or replace a node cluster definition.

    Cluster membership is server-authoritative: nodes never self-declare.
    After upsert the registry propagates cluster_id back into NodeRegistry.
    """
    state = _require_state(request)
    try:
        await state.cluster_registry.upsert(spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await state.cluster_registry.update_node_memberships(state.registry)
    return spec


@app.get("/api/v1/clusters", response_model=list[ClusterSpec])
async def list_clusters(request: Request) -> list[ClusterSpec]:
    """List all registered node clusters."""
    state = _require_state(request)
    return await state.cluster_registry.list_all()


@app.get("/api/v1/clusters/{cluster_id}", response_model=ClusterSpec)
async def get_cluster(request: Request, cluster_id: str) -> ClusterSpec:
    """Return a single cluster by ID."""
    state = _require_state(request)
    spec = await state.cluster_registry.get(cluster_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_id}' not found")
    return spec


@app.delete("/api/v1/clusters/{cluster_id}", status_code=204)
async def delete_cluster(request: Request, cluster_id: str) -> None:
    """Remove a cluster definition. Member nodes revert to independent operation."""
    state = _require_state(request)
    deleted = await state.cluster_registry.delete(cluster_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_id}' not found")
    await state.cluster_registry.update_node_memberships(state.registry)


def _build_hass_state_snapshot_provider(state) -> Callable[[], Awaitable[HassStateSnapshot]]:
    """Poll-side state gathering for the HA bridge.

    Zone occupancy, node health, and track counts are never broadcast over the
    live hub, so they cannot be tee'd — they have to be pulled. This is the cost
    driver of the publish interval (``compute_occupancy`` is O(zones x tracks)
    with a point-in-polygon test per pair), which is why the interval has a
    validated 1.0 s floor.
    """

    async def provider() -> HassStateSnapshot:
        now_ns = time.time_ns()
        tracks = await state.tracker.snapshot(now_ns=now_ns)
        for track in tracks:
            if track.position_geo is None:
                track.position_geo = state.coordinate_frame.local_to_geo(track.position_m)
        occupancy = await state.zone_matcher.compute_occupancy(
            tracks=tracks,
            coordinate_frame=state.coordinate_frame,
            now_ns=now_ns,
        )
        node_rows = await state.storage.list_nodes(limit=5000)
        await _apply_runtime_health_statuses(
            node_rows,
            bit_evaluator=state.bit_evaluator,
            now_ns=now_ns,
            degraded_after_seconds=state.settings.node_degraded_after_seconds,
            offline_after_seconds=state.settings.node_offline_after_seconds,
        )
        counts = {"online_nodes": 0, "degraded_nodes": 0, "offline_nodes": 0}
        nodes: list[NodeStateInput] = []
        for row in node_rows:
            health = str(row.get("health_status") or "")
            if health == NodeHealthStatus.ONLINE.value:
                counts["online_nodes"] += 1
            elif health == NodeHealthStatus.OFFLINE.value:
                counts["offline_nodes"] += 1
            else:
                counts["degraded_nodes"] += 1
            nodes.append(
                NodeStateInput(
                    node_id=str(row["id"]),
                    node_name=str(row.get("name") or row["id"]),
                    health_status=health,
                    last_seen_ns=row.get("last_seen_ns"),
                    detail={"capabilities": list(row.get("capabilities") or [])},
                )
            )

        return HassStateSnapshot(
            zones=tuple(
                ZoneStateInput(
                    zone_id=item.zone_id,
                    zone_name=item.zone_name,
                    zone_type=item.zone_type,
                    occupied=item.occupied,
                    occupying_track_ids=tuple(item.occupying_track_ids),
                    occupying_labels=tuple(item.occupying_labels),
                    updated_ns=item.updated_ns,
                )
                for item in occupancy
            ),
            nodes=tuple(nodes),
            system=SystemStateInput(
                system_health=_derive_system_health(counts),
                active_track_count=len(tracks),
                online_nodes=counts["online_nodes"],
                degraded_nodes=counts["degraded_nodes"],
                offline_nodes=counts["offline_nodes"],
                generated_ns=now_ns,
            ),
            tracks=tuple(
                TrackSlotCandidate(
                    track_id=track.id,
                    tqi=track.tqi,
                    label=track.label,
                    lat=track.position_geo.lat if track.position_geo else None,
                    lon=track.position_geo.lon if track.position_geo else None,
                    altitude_m=track.position_m[2],
                    status=track.status,
                    confidence=track.confidence,
                )
                for track in tracks
            ),
        )

    return provider


def _derive_system_health(counts: dict[str, int]) -> str:
    """Same derivation as GET /api/v1/context/current, so HA and the COP agree."""
    offline = counts.get("offline_nodes", 0)
    degraded = counts.get("degraded_nodes", 0)
    total = offline + degraded + counts.get("online_nodes", 0)
    if total > 0 and offline == total:
        return "error"
    if degraded > 0 or offline > 0:
        return "degraded"
    return "ok"


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
    node_counts = await _runtime_node_health_counts(state, now_ns=now_ns)

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


def _resolve_export_snippet_file(snippet_path: str | None, snippet_root: Path) -> Path | None:
    if not snippet_path:
        return None
    try:
        return _resolve_snippet_file(snippet_path, snippet_root)
    except HTTPException:
        return None


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


@app.patch("/api/v1/detections/{detection_id}/review")
async def update_detection_review(
    detection_id: str,
    payload: DetectionReviewUpdateRequest,
    request: Request,
) -> dict:
    state = _require_state(request)
    if not payload.model_fields_set:
        raise HTTPException(status_code=400, detail="Review update must include at least one field")

    detection = await state.storage.get_detection(detection_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="Detection not found")

    fields_set = payload.model_fields_set
    now_ns = time.time_ns()
    review_state = detection.get("review_state") or DetectionReviewState.UNREVIEWED.value
    review_label_id = detection.get("review_label_id")
    review_label = detection.get("review_label")
    review_label_category = detection.get("review_label_category")
    detection_label_category = detection.get("label_category") or "unknown"
    review_notes = detection.get("review_notes")
    promote_to_training = bool(detection.get("promote_to_training") or False)
    training_example_kind = detection.get("training_example_kind")

    if "review_state" in fields_set:
        review_state = (
            payload.review_state.value
            if payload.review_state is not None
            else DetectionReviewState.UNREVIEWED.value
        )
        if review_state in {
            DetectionReviewState.UNREVIEWED.value,
            DetectionReviewState.REJECTED.value,
        }:
            if "review_label" not in fields_set:
                review_label_id = None
                review_label = None
                review_label_category = None
            if "promote_to_training" not in fields_set:
                promote_to_training = False
                training_example_kind = None
        if review_state == DetectionReviewState.UNREVIEWED.value:
            if "review_notes" not in fields_set:
                review_notes = None

    if "review_label" in fields_set:
        previous_review_label = detection.get("review_label")
        review_label = payload.review_label
        if review_label is None:
            review_label_id = None
            review_label_category = None
        elif "review_label_category" not in fields_set:
            if review_label != previous_review_label:
                review_label_category = detection_label_category
            else:
                review_label_category = review_label_category or detection_label_category

    if "review_label_category" in fields_set:
        review_label_category = payload.review_label_category

    if "review_notes" in fields_set:
        review_notes = payload.review_notes

    if "promote_to_training" in fields_set:
        promote_to_training = bool(payload.promote_to_training) if payload.promote_to_training is not None else False
        if not promote_to_training:
            training_example_kind = None

    if "training_example_kind" in fields_set:
        training_example_kind = (
            payload.training_example_kind.value if payload.training_example_kind is not None else None
        )

    if review_state == DetectionReviewState.REJECTED.value and review_label is not None:
        raise HTTPException(status_code=400, detail="Rejected detections cannot carry a review label")
    if promote_to_training and review_state != DetectionReviewState.CONFIRMED.value:
        raise HTTPException(status_code=400, detail="Training promotion requires confirmed review state")
    if promote_to_training:
        training_example_kind = training_example_kind or TrainingExampleKind.POSITIVE.value

    if review_label is not None:
        review_label_category = review_label_category or detection_label_category
        review_label_id = await state.storage.upsert_label(
            name=review_label,
            category=review_label_category,
            source="review",
            created_ns=now_ns,
        )

    effective_label = review_label or detection.get("label")
    effective_label_category = review_label_category or detection_label_category
    materialized_example: tuple[str, str] | None = None
    if promote_to_training:
        if not effective_label or not effective_label.strip():
            raise HTTPException(status_code=400, detail="Training promotion requires an effective label")
        source_audio = _resolve_snippet_file(
            detection.get("snippet_path"), state.settings.snippet_dir.resolve()
        )
        if source_audio is None:
            raise HTTPException(
                status_code=409,
                detail="Detection audio is unavailable; it may have expired before promotion",
            )
        manifest = {
            "schema_version": 1,
            "detection_id": detection_id,
            "audio_filename": f"{detection_id}.wav",
            "training": {
                "label": effective_label,
                "label_category": effective_label_category,
                "example_kind": training_example_kind,
                "promoted_at_ns": now_ns,
            },
            "review": {
                "state": review_state,
                "label": review_label,
                "label_category": review_label_category,
                "notes": review_notes,
            },
            "detection": {
                "event_id": detection.get("event_id") or detection_id,
                "timestamp_ns": detection.get("timestamp_ns"),
                "source_node_id": detection.get("source_node_id"),
                "reporting_modality": detection.get("reporting_modality"),
                "position_geo": detection.get("position_geo"),
                "position_m": detection.get("position_m"),
                "label": detection.get("label"),
                "label_category": detection.get("label_category"),
                "label_confidence": detection.get("label_confidence"),
                "classifier_scores": detection.get("classifier_scores"),
                "feature_summary": detection.get("feature_summary"),
            },
        }
        embedding_info = await extract_embedding_npy(
            source_audio,
            state.settings.training_dataset_dir / f"{detection_id}.npy",
        )
        if embedding_info is not None:
            manifest["embedding"] = embedding_info
        try:
            materialized_example = await materialize_training_example(
                dataset_dir=state.settings.training_dataset_dir,
                detection_id=detection_id,
                source_audio_path=source_audio,
                manifest=manifest,
            )
        except TrainingDatasetError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except OSError as error:
            raise HTTPException(status_code=500, detail=f"Could not save training example: {error}") from error

    updated = await state.storage.update_detection_review(
        detection_id=detection_id,
        review_state=review_state,
        review_label_id=review_label_id,
        review_label=review_label,
        review_label_category=review_label_category,
        review_notes=review_notes,
        promote_to_training=promote_to_training,
        training_example_kind=training_example_kind,
        review_updated_ns=now_ns,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Detection not found")

    if materialized_example is not None:
        audio_path, manifest_path = materialized_example
        existing_example = await state.storage.get_training_example(detection_id)
        await state.storage.upsert_training_example(
            detection_id=detection_id,
            label=effective_label,
            label_category=effective_label_category,
            example_kind=training_example_kind,
            audio_path=audio_path,
            manifest_path=manifest_path,
            created_ns=(existing_example or {}).get("created_ns", now_ns),
            updated_ns=now_ns,
            embedding_path=(manifest.get("embedding") or {}).get("path"),
        )
    elif not promote_to_training:
        removed_example = await state.storage.delete_training_example(detection_id)
        if removed_example is not None:
            await delete_training_example_files(state.settings.training_dataset_dir, removed_example)

    refreshed = await state.storage.get_detection(detection_id)
    if refreshed is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    if refreshed.get("position_geo") is None and refreshed.get("position_m"):
        local = refreshed["position_m"]
        geo = state.coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
        refreshed["position_geo"] = geo.model_dump(mode="json")
    return refreshed


@app.get("/api/v1/detections/{detection_id}/audio")
async def get_detection_audio(
    detection_id: str,
    request: Request,
    download: bool = Query(default=False),
    level: Literal["listening", "canonical"] = Query(default="listening"),
) -> Response:
    state = _require_state(request)
    settings: Settings = state.settings
    snippet_path = await state.storage.snippet_path_for_detection(detection_id)
    if not snippet_path:
        raise HTTPException(status_code=404, detail="Snippet not found for detection")

    snippet_file = _resolve_snippet_file(snippet_path, settings.snippet_dir.resolve())
    if snippet_file is None:
        raise HTTPException(status_code=404, detail="Snippet file no longer exists")

    filename = f"{detection_id}_{level}.wav"
    content_disposition = "attachment" if download else "inline"
    if level == "listening":
        wav_bytes, report = await asyncio.to_thread(
            listening_wav_bytes, snippet_file, settings.audio_processing_config_path
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'{content_disposition}; filename="{filename}"',
                **level_report_headers(report),
            },
        )
    return FileResponse(
        path=snippet_file,
        media_type="audio/wav",
        filename=filename,
        headers={
            "Content-Disposition": f'{content_disposition}; filename="{filename}"',
            "X-Minimappr-Audio-Level-Profile": "canonical",
        },
    )


@app.get("/api/v1/exports/ebird")
async def export_ebird_review_package(
    request: Request,
    format: Literal["json", "csv"] = Query(default="json"),
    limit: int = Query(default=500, ge=1, le=5000),
    since_hours: float = Query(default=24.0, ge=0.0, le=24.0 * 365.0),
):
    import csv
    import io
    from datetime import datetime, timezone

    state = _require_state(request)
    since_ns = time.time_ns() - int(since_hours * 3_600 * 1_000_000_000)
    detections = await state.storage.list_detections(
        limit=limit,
        since_ns=since_ns,
        review_state=DetectionReviewState.CONFIRMED.value,
    )
    base_url = str(request.base_url).rstrip("/")
    snippet_root = state.settings.snippet_dir.resolve()
    exported_items: list[ReviewedDetectionExportItem] = []

    for detection in detections:
        review_state = detection.get("review_state") or DetectionReviewState.UNREVIEWED.value
        effective_label = detection.get("review_label") or detection.get("label")
        effective_label_category = detection.get("review_label_category") or detection.get("label_category") or "unknown"
        if review_state != DetectionReviewState.CONFIRMED.value or effective_label_category != "bird":
            continue

        if detection.get("position_geo") is None and detection.get("position_m"):
            local = detection["position_m"]
            geo = state.coordinate_frame.local_to_geo((float(local[0]), float(local[1]), float(local[2])))
            detection["position_geo"] = geo.model_dump(mode="json")

        timestamp_ns = int(detection["timestamp_ns"])
        observed_at_iso = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        has_audio = _resolve_export_snippet_file(detection.get("snippet_path"), snippet_root) is not None
        audio_url = f"{base_url}/api/v1/detections/{detection['id']}/audio?download=true" if has_audio else None
        exported_items.append(
            ReviewedDetectionExportItem(
                detection_id=detection["id"],
                event_id=detection.get("event_id") or detection["id"],
                timestamp_ns=timestamp_ns,
                observed_at_iso=observed_at_iso,
                track_id=detection.get("track_id"),
                source_node_id=detection.get("source_node_id"),
                reporting_modality=detection.get("reporting_modality") or "localized",
                position_geo=GeoPoint.model_validate(detection["position_geo"]) if detection.get("position_geo") else None,
                original_label=detection.get("label") or "unknown",
                original_label_category=detection.get("label_category") or "unknown",
                reviewed_label=detection.get("review_label"),
                reviewed_label_category=detection.get("review_label_category"),
                effective_label=effective_label,
                effective_label_category=effective_label_category,
                review_state=review_state,
                review_notes=detection.get("review_notes"),
                promote_to_training=bool(detection.get("promote_to_training") or False),
                audio_url=audio_url,
                has_audio=has_audio,
            )
        )

    generated_at_ns = time.time_ns()
    generated_at_iso = datetime.fromtimestamp(generated_at_ns / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    export_package = ReviewedDetectionExportPackage(
        generated_at_ns=generated_at_ns,
        generated_at_iso=generated_at_iso,
        detection_count=len(exported_items),
        detections=exported_items,
    )
    if format == "json":
        return export_package.model_dump(mode="json")

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "detection_id",
            "event_id",
            "observed_at_iso",
            "effective_label",
            "effective_label_category",
            "original_label",
            "reviewed_label",
            "review_state",
            "track_id",
            "source_node_id",
            "latitude",
            "longitude",
            "alt_m",
            "audio_url",
            "promote_to_training",
            "review_notes",
        ],
    )
    writer.writeheader()
    for item in exported_items:
        writer.writerow(
            {
                "detection_id": item.detection_id,
                "event_id": item.event_id,
                "observed_at_iso": item.observed_at_iso,
                "effective_label": item.effective_label,
                "effective_label_category": item.effective_label_category,
                "original_label": item.original_label,
                "reviewed_label": item.reviewed_label or "",
                "review_state": item.review_state.value,
                "track_id": item.track_id or "",
                "source_node_id": item.source_node_id or "",
                "latitude": item.position_geo.lat if item.position_geo else "",
                "longitude": item.position_geo.lon if item.position_geo else "",
                "alt_m": item.position_geo.alt_m if item.position_geo else "",
                "audio_url": item.audio_url or "",
                "promote_to_training": item.promote_to_training,
                "review_notes": item.review_notes or "",
            }
        )
    filename = f"minimappr-ebird-review-export-{generated_at_ns}.csv"
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/v1/tracks/{track_id}/audio")
async def get_track_audio(
    track_id: str,
    request: Request,
    download: bool = Query(default=False),
    level: Literal["listening", "canonical"] = Query(default="listening"),
) -> Response:
    state = _require_state(request)
    settings: Settings = state.settings
    latest = await state.storage.latest_detection_audio_for_track(track_id)
    if latest is None:
        raise HTTPException(status_code=404, detail="No audio snippet is available for this track")

    detection_id, snippet_path = latest
    snippet_file = _resolve_snippet_file(snippet_path, settings.snippet_dir.resolve())
    if snippet_file is None:
        raise HTTPException(status_code=404, detail="Snippet file no longer exists")

    filename = f"track_{track_id}__detection_{detection_id}_{level}.wav"
    content_disposition = "attachment" if download else "inline"
    if level == "listening":
        wav_bytes, report = await asyncio.to_thread(
            listening_wav_bytes, snippet_file, settings.audio_processing_config_path
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'{content_disposition}; filename="{filename}"',
                **level_report_headers(report),
            },
        )
    return FileResponse(
        path=snippet_file,
        media_type="audio/wav",
        filename=filename,
        headers={
            "Content-Disposition": f'{content_disposition}; filename="{filename}"',
            "X-Minimappr-Audio-Level-Profile": "canonical",
        },
    )


@app.get("/api/v1/nodes/{node_id}/audio/recent")
async def get_recent_node_audio(
    node_id: str,
    request: Request,
    seconds: float = Query(default=10.0, ge=1.0, le=30.0),
    channel: int | None = Query(default=None, ge=0),
    render: Literal["auto", "mix", "multichannel"] = Query(default="auto"),
    level: Literal["listening", "canonical"] = Query(default="listening"),
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
    _level = level

    # numpy stacking and WAV encoding are CPU-bound; run them off the event loop.
    def _encode_audio():
        channels_first = np.vstack([ch[-common_samples:] for ch in raw_channels])
        if _channel is not None:
            if _channel >= channels_first.shape[0]:
                return None, None, None, None, None
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
        report = None
        if _level == "listening":
            profile = load_audio_processing_configuration(
                settings.audio_processing_config_path
            ).profile(LISTENING_PROFILE_NAME)
            rendered, report = apply_level_profile(rendered, profile)
        return (
            wav_multichannel_bytes(rendered, sample_rate_hz=_sample_rate_hz),
            r_mode,
            sel_ch,
            len(raw_channels),
            report,
        )

    wav_bytes, render_mode, selected_channel, n_channels, level_report = await asyncio.to_thread(_encode_audio)

    if wav_bytes is None:
        raise HTTPException(status_code=404, detail="Requested audio channel is unavailable for node")

    level_headers = (
        level_report_headers(level_report)
        if level_report is not None
        else {"X-Minimappr-Audio-Level-Profile": "canonical"}
    )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="{_node_id}_recent_{_level}.wav"',
            **level_headers,
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


# ── Camera Discovery API ──────────────────────────────────────────────────────

@app.get("/api/v1/cameras")
async def list_cameras():
    """Enumerate available video capture devices for the recording UI."""
    from minimappr.core.camera_discovery import discover_cameras
    cameras = await discover_cameras()
    return [
        {"id": cam.id, "label": cam.label, "platform": cam.platform}
        for cam in cameras
    ]


# PTZ camera controls live under each node's effector capability subroutes.


# ── Capture API ────────────────────────────────────────────────────────────────

class _CaptureStartBody(BaseModel):
    stream_key: str
    max_duration_s: float = 300.0
    video_source: str | None = None
    libcamera_mode: bool = False
    deployment_profile: str = "auto"
    work_dir: str | None = None
    capture_kind: str = "recording"


@app.post("/api/v1/capture/start")
async def capture_start(request: Request, body: _CaptureStartBody):
    state = request.app.state
    manager: CaptureSessionManager = state.capture_manager
    if _should_proxy_ingest_to_python_worker(state):
        return await _proxy_json_to_python_worker(
            state,
            method="POST",
            endpoint_path="/api/v1/capture/start",
            json_body=body.model_dump(mode="json"),
        )
    if not _capture_pipeline_available(state):
        raise HTTPException(
            status_code=503,
            detail=_capture_pipeline_unavailable_reason(state),
        )

    work_dir_path = (
        Path(body.work_dir) if body.work_dir else Path("data/captures")
    )
    req = await _build_capture_start_request(
        state,
        stream_key=body.stream_key,
        work_dir_path=work_dir_path,
        max_duration_s=body.max_duration_s,
        video_source=body.video_source,
        libcamera_mode=body.libcamera_mode,
        deployment_profile=body.deployment_profile,
        capture_kind=body.capture_kind,
    )
    record = await manager.start(req)

    if record.state == CaptureState.FAILED:
        raise HTTPException(status_code=500, detail=record.error or "capture start failed")

    storage: Storage = state.storage
    await storage.upsert_capture_session(record)
    await _broadcast_recording_status(state, record)

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
    if _should_proxy_ingest_to_python_worker(state):
        return await _proxy_json_to_python_worker(
            state,
            method="POST",
            endpoint_path=f"/api/v1/capture/{session_id}/stop",
        )
    if not _capture_pipeline_available(state):
        raise HTTPException(
            status_code=503,
            detail=_capture_pipeline_unavailable_reason(state),
        )
    sidecar_url = "" if settings.ingest_backend == "python" else _ingest_runtime_base_url(settings)

    try:
        record = await manager.stop(session_id, sidecar_url)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    except ValueError as exc:
        record = manager.get(session_id)
        if record is not None and record.state in {
            CaptureState.AWAITING_FINAL_TRACKS,
            CaptureState.PROCESSING,
            CaptureState.COMPLETED,
            CaptureState.FAILED,
        }:
            return {
                "session_id": record.session_id,
                "state": record.state.value,
                "end_time_ns": record.end_time_ns,
            }
        raise HTTPException(status_code=409, detail=str(exc))

    storage: Storage = state.storage
    await storage.upsert_capture_session(record)
    await _broadcast_recording_status(state, record)

    return {
        "session_id": record.session_id,
        "state": record.state.value,
        "end_time_ns": record.end_time_ns,
    }


@app.get("/api/v1/capture/{session_id}/status")
async def capture_status(session_id: str, request: Request):
    state = request.app.state
    if _should_proxy_ingest_to_python_worker(state):
        return await _proxy_json_to_python_worker(
            state,
            method="GET",
            endpoint_path=f"/api/v1/capture/{session_id}/status",
        )
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
        "ambix_path": str(record.ambix_path) if record.ambix_path else None,
        "iamf_path": str(record.iamf_path) if record.iamf_path else None,
        "object_path": str(record.object_path) if record.object_path else None,
        "visual_path": str(record.visual_path) if record.visual_path else None,
        "youtube_path": str(record.youtube_path) if record.youtube_path else None,
        "error": record.error,
        "created_ns": record.created_ns,
    }


@app.get("/api/v1/capture")
async def capture_list(request: Request):
    state = request.app.state
    if _should_proxy_ingest_to_python_worker(state):
        return await _proxy_json_to_python_worker(
            state,
            method="GET",
            endpoint_path="/api/v1/capture",
        )
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


# ── Recordings API (Frontend Adapter) ─────────────────────────────────────────
# Maps the frontend's RecordingSession / RecordingLibraryEntry shapes to the
# existing CaptureSessionManager and Storage layer.

def _capture_state_to_recording_status(state: CaptureState) -> str:
    """Map CaptureState to the frontend's RecordingStatus snake_case string."""
    mapping = {
        CaptureState.PENDING: "starting",
        CaptureState.RECORDING: "active",
        CaptureState.AWAITING_FINAL_TRACKS: "awaiting_final_tracks",
        CaptureState.PROCESSING: "stopping",
        CaptureState.COMPLETED: "completed",
        CaptureState.FAILED: "failed",
    }
    return mapping.get(state, "idle")


def _session_record_to_recording_session(record: CaptureSessionRecord) -> dict:
    """Convert a CaptureSessionRecord to the frontend's RecordingSession shape."""
    started_at_ms = (record.start_time_ns / 1_000_000) if record.start_time_ns else None
    ended_at_ms = (record.end_time_ns / 1_000_000) if record.end_time_ns else None
    duration_seconds = None
    if record.start_time_ns and record.end_time_ns:
        duration_seconds = (record.end_time_ns - record.start_time_ns) / 1_000_000_000.0

    return {
        "session_id": record.session_id,
        "status": _capture_state_to_recording_status(record.state),
        "listener_node_id": record.stream_key,
        "include_ambisonics": True,
        "include_iamf": record.include_iamf,
        "include_video": record.include_video,
        "camera_source": None,
        "started_at_ms": started_at_ms or 0.0,
        "ended_at_ms": ended_at_ms,
        "duration_seconds": duration_seconds,
        "ambisonics_ready": record.ambix_path is not None and record.ambix_path.exists() if record.ambix_path else False,
        "iamf_ready": record.iamf_path is not None and record.iamf_path.exists() if record.iamf_path else False,
        "object_ready": record.object_path is not None and record.object_path.exists() if record.object_path else False,
        "visual_ready": record.visual_path is not None and record.visual_path.exists() if record.visual_path else False,
        "video_ready": record.youtube_path is not None and record.youtube_path.exists() if record.youtube_path else False,
        "error_message": record.error,
    }


async def _broadcast_recording_status(state, record: CaptureSessionRecord) -> None:
    await state.live_hub.broadcast(
        {
            "type": "recording_status",
            "session": _session_record_to_recording_session(record),
        }
    )


class _StartRecordingBody(BaseModel):
    listener_node_id: str
    include_ambisonics: bool = True
    include_iamf: bool = True
    include_video: bool = True
    camera_source: str | None = None
    capture_kind: str = "recording"


@app.post("/api/v1/recordings")
async def recordings_start(request: Request, body: _StartRecordingBody):
    """Start a new recording session (frontend adapter).

    Translates the frontend's StartRecordingRequest into a CaptureStartRequest,
    resolving listener_node_id → stream_key and deriving channel_sensor_ids
    from the node's sensor_offsets_m.
    """
    state = request.app.state
    manager: CaptureSessionManager = state.capture_manager
    settings: Settings = state.settings
    storage: Storage = state.storage
    if _should_proxy_ingest_to_python_worker(state):
        return await _proxy_json_to_python_worker(
            state,
            method="POST",
            endpoint_path="/api/v1/recordings",
            json_body=body.model_dump(mode="json"),
        )

    if not _capture_pipeline_available(state):
        raise HTTPException(
            status_code=503,
            detail=_capture_pipeline_unavailable_reason(state),
        )

    stream_key = body.listener_node_id
    work_dir_path = Path("data/captures")
    req = await _build_capture_start_request(
        state,
        stream_key=stream_key,
        work_dir_path=work_dir_path,
        max_duration_s=(
            settings.calibration_max_duration_s if body.capture_kind == "calibration" else 300.0
        ),
        video_source=body.camera_source,
        record_video=body.include_video,
        include_iamf=body.include_iamf,
        capture_kind=body.capture_kind,
    )

    record = await manager.start(req)
    if record.state == CaptureState.FAILED:
        raise HTTPException(status_code=500, detail=record.error or "capture start failed")

    await storage.upsert_capture_session(record)
    await _broadcast_recording_status(state, record)
    return _session_record_to_recording_session(record)


@app.patch("/api/v1/recordings/{session_id}/stop")
async def recordings_stop(session_id: str, request: Request):
    """Stop an active recording session (frontend adapter)."""
    state = request.app.state
    manager: CaptureSessionManager = state.capture_manager
    settings: Settings = state.settings
    if _should_proxy_ingest_to_python_worker(state):
        return await _proxy_json_to_python_worker(
            state,
            method="PATCH",
            endpoint_path=f"/api/v1/recordings/{session_id}/stop",
        )

    if not _capture_pipeline_available(state):
        raise HTTPException(status_code=503, detail=_capture_pipeline_unavailable_reason(state))

    sidecar_url = "" if settings.ingest_backend == "python" else _ingest_runtime_base_url(settings)

    try:
        record = await manager.stop(session_id, sidecar_url)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    except ValueError as exc:
        record = manager.get(session_id)
        if record is not None and record.state in {
            CaptureState.AWAITING_FINAL_TRACKS,
            CaptureState.PROCESSING,
            CaptureState.COMPLETED,
            CaptureState.FAILED,
        }:
            return _session_record_to_recording_session(record)
        raise HTTPException(status_code=409, detail=str(exc))

    storage: Storage = state.storage
    await storage.upsert_capture_session(record)
    await _broadcast_recording_status(state, record)
    return _session_record_to_recording_session(record)


@app.get("/api/v1/recordings/{session_id}")
async def recordings_get(session_id: str, request: Request):
    """Get a single recording session status (frontend adapter)."""
    state = request.app.state
    manager: CaptureSessionManager = state.capture_manager

    record = manager.get(session_id)
    if record is not None:
        return _session_record_to_recording_session(record)

    # Fall back to DB for completed/failed sessions no longer in memory.
    storage: Storage = state.storage
    row = await storage.get_capture_session(session_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")

    # Reconstruct a minimal CaptureSessionRecord from the DB row for shape mapping.
    record = CaptureSessionRecord(
        session_id=row["session_id"],
        state=CaptureState(row["state"]),
        stream_key=row["stream_key"],
        range_lease_id=row.get("range_lease_id"),
        start_time_ns=row.get("start_time_ns"),
        end_time_ns=row.get("end_time_ns"),
        first_frame_pts_ns=row.get("first_frame_pts_ns"),
        work_dir=Path(row["work_dir"]),
        video_path=Path(row["video_path"]) if row.get("video_path") else None,
        ambix_path=Path(row["ambix_path"]) if row.get("ambix_path") else None,
        iamf_path=Path(row["iamf_path"]) if row.get("iamf_path") else None,
        object_path=Path(row["object_path"]) if row.get("object_path") else None,
        visual_path=Path(row["visual_path"]) if row.get("visual_path") else None,
        youtube_path=Path(row["youtube_path"]) if row.get("youtube_path") else None,
        error=row.get("error"),
        created_ns=row.get("created_ns", 0),
        capture_kind=row.get("capture_kind") or "recording",
    )
    return _session_record_to_recording_session(record)


@app.get("/api/v1/recordings")
async def recordings_list(request: Request):
    """List all recording sessions (frontend adapter)."""
    state = request.app.state
    storage: Storage = state.storage
    rows = await storage.list_capture_sessions(limit=200)

    results = []
    for row in rows:
        started_at_ms = (row["start_time_ns"] / 1_000_000) if row.get("start_time_ns") else None
        ended_at_ms = (row["end_time_ns"] / 1_000_000) if row.get("end_time_ns") else None
        duration_seconds = None
        if row.get("start_time_ns") and row.get("end_time_ns"):
            duration_seconds = (row["end_time_ns"] - row["start_time_ns"]) / 1_000_000_000.0

        iamf_path = Path(row["iamf_path"]) if row.get("iamf_path") else None
        ambix_path = Path(row["ambix_path"]) if row.get("ambix_path") else None
        object_path = Path(row["object_path"]) if row.get("object_path") else None
        visual_path = Path(row["visual_path"]) if row.get("visual_path") else None
        youtube_path = Path(row["youtube_path"]) if row.get("youtube_path") else None

        results.append({
            "session_id": row["session_id"],
            "started_at_ms": started_at_ms or 0.0,
            "ended_at_ms": ended_at_ms,
            "duration_seconds": duration_seconds,
            "listener_node_id": row["stream_key"],
            "capture_kind": row.get("capture_kind") or "recording",
            "ambisonics_available": ambix_path is not None and ambix_path.exists() if ambix_path else False,
            "iamf_available": iamf_path is not None and iamf_path.exists() if iamf_path else False,
            "object_available": object_path is not None and object_path.exists() if object_path else False,
            "visual_available": visual_path is not None and visual_path.exists() if visual_path else False,
            "video_available": youtube_path is not None and youtube_path.exists() if youtube_path else False,
            "size_bytes": None,
            "status": _capture_state_to_recording_status(CaptureState(row["state"])),
            "error_message": row.get("error"),
        })

    return results


@app.delete("/api/v1/recordings/{session_id}")
async def recordings_delete(session_id: str, request: Request):
    """Delete a recording session and its artifacts."""
    state = request.app.state
    if _should_proxy_ingest_to_python_worker(state):
        return await _proxy_json_to_python_worker(
            state,
            method="DELETE",
            endpoint_path=f"/api/v1/recordings/{session_id}",
        )
    manager: CaptureSessionManager = state.capture_manager
    storage: Storage = state.storage

    # If the session is still active, stop it first.
    record = manager.get(session_id)
    if record is not None and record.state == CaptureState.RECORDING:
        settings: Settings = state.settings
        sidecar_url = "" if settings.ingest_backend == "python" else _ingest_runtime_base_url(settings)
        try:
            await manager.stop(session_id, sidecar_url)
        except (KeyError, ValueError):
            pass

    # Clean up artifacts from disk.
    row = await storage.get_capture_session(session_id)
    if row is not None:
        work_dir = Path(row["work_dir"]) if row.get("work_dir") else None
        if work_dir and work_dir.exists():
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

        for path_key in ("ambix_path", "iamf_path", "object_path", "visual_path", "youtube_path"):
            artifact_path = row.get(path_key)
            if artifact_path:
                p = Path(artifact_path)
                if p.exists():
                    p.unlink(missing_ok=True)

    # Remove from in-memory manager.
    manager._sessions.pop(session_id, None)
    await storage.delete_large_artifacts_for_session(session_id)
    await storage.delete_capture_session(session_id)

    return Response(status_code=204)


@app.get("/api/v1/recordings/{session_id}/download")
async def recordings_download(session_id: str, format: str = Query(...), request: Request = None):
    """Download a recording artifact by format (ambisonics, iamf, video)."""
    state = request.app.state
    storage: Storage = state.storage

    row = await storage.get_capture_session(session_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")

    if format == "iamf":
        path = row.get("iamf_path")
        if not path or not Path(path).exists():
            raise HTTPException(status_code=404, detail="IAMF file not available")
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=f"{session_id}_audio.iamf",
        )
    elif format == "video":
        path = row.get("youtube_path")
        if not path or not Path(path).exists():
            raise HTTPException(status_code=404, detail="Video file not available")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"{session_id}_youtube.mp4",
        )
    elif format in ("ambisonics", "ambix"):
        path = row.get("ambix_path")
        if path and Path(path).exists():
            return FileResponse(
                path,
                media_type="audio/wav",
                filename=f"{session_id}_ambix.wav",
            )
        raise HTTPException(status_code=404, detail="Ambisonics file not available")
    elif format in ("object", "object-wav", "selected-object"):
        path = row.get("object_path")
        if path and Path(path).exists():
            return FileResponse(
                path,
                media_type="audio/wav",
                filename=f"{session_id}_object.wav",
            )
        raise HTTPException(status_code=404, detail="Selected object file not available")
    elif format in ("visual", "recording-visual"):
        path = row.get("visual_path")
        if path and Path(path).exists():
            return FileResponse(
                path,
                media_type="video/mp4",
                filename=f"{session_id}_visual.mp4",
            )
        raise HTTPException(status_code=404, detail="Recording visual not available")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown format: {format}. Use ambisonics, object, iamf, visual, or video.")


# ── Calibration ground truth + bundle export ──────────────────────────────────

def _calibration_gt_row_to_event(row: dict) -> CalibrationGroundTruthEvent:
    return CalibrationGroundTruthEvent(
        event_id=row["id"],
        session_id=row["session_id"],
        label=row["label"],
        label_category=row.get("label_category") or "unknown",
        geometry_kind=row.get("geometry_kind") or "static",
        lat=row.get("lat"),
        lon=row.get("lon"),
        alt_m=row.get("alt_m"),
        source_node_id=row.get("source_node_id"),
        start_ns=row["start_ns"],
        end_ns=row["end_ns"],
        notes=row.get("notes"),
        created_ns=row.get("created_ns") or 0,
        updated_ns=row.get("updated_ns") or 0,
    )


async def _require_calibration_session(storage: Storage, session_id: str) -> dict:
    row = await storage.get_capture_session(session_id)
    if row is None or (row.get("capture_kind") or "recording") != "calibration":
        raise HTTPException(
            status_code=404, detail=f"calibration session {session_id} not found"
        )
    return row


async def _resolve_ground_truth_node_position(
    request: Request, session_row: dict, node_id: str
) -> tuple[float, float, float]:
    """Resolve a ground-truth node's lat/lon/alt.

    Prefers the session's calibration manifest (capture-time snapshot, and
    proof the node has audio in the bundle); falls back to the live node
    registry. Snapshotting into the row keeps exported bundles self-contained.
    """
    manifest_path = session_row.get("calibration_manifest_path")
    if manifest_path and Path(manifest_path).exists():
        try:
            manifest = json.loads(Path(manifest_path).read_text())
        except (OSError, json.JSONDecodeError):
            manifest = {}
        for node in manifest.get("nodes", []):
            if node.get("node_id") != node_id:
                continue
            geo = node.get("position_geo")
            if isinstance(geo, dict) and geo.get("lat") is not None:
                return float(geo["lat"]), float(geo["lon"]), float(geo.get("alt_m") or 0.0)
            break

    storage: Storage = request.app.state.storage
    node_row = await storage.get_node_by_id(node_id)
    if node_row is not None:
        geo = node_row.get("position_geo")
        if isinstance(geo, dict) and geo.get("lat") is not None:
            return float(geo["lat"]), float(geo["lon"]), float(geo.get("alt_m") or 0.0)
        raw_local = node_row.get("position_m")
        frame = getattr(request.app.state, "coordinate_frame", None)
        if frame is not None and isinstance(raw_local, (list, tuple)) and len(raw_local) >= 3:
            geo_point = frame.local_to_geo(
                (float(raw_local[0]), float(raw_local[1]), float(raw_local[2]))
            )
            return geo_point.lat, geo_point.lon, geo_point.alt_m
    raise HTTPException(
        status_code=422,
        detail=f"ground-truth node {node_id} not found or has no position",
    )


@app.post("/api/v1/calibration/{session_id}/ground-truth")
async def calibration_ground_truth_add(
    session_id: str, body: CalibrationGroundTruthIn, request: Request
):
    storage: Storage = request.app.state.storage
    session_row = await _require_calibration_session(storage, session_id)
    lat, lon, alt_m = body.lat, body.lon, body.alt_m
    if body.source_node_id is not None:
        lat, lon, alt_m = await _resolve_ground_truth_node_position(
            request, session_row, body.source_node_id
        )
    row = await storage.insert_calibration_ground_truth(
        session_id=session_id,
        label=body.label,
        label_category=body.label_category,
        lat=lat,
        lon=lon,
        alt_m=alt_m,
        source_node_id=body.source_node_id,
        start_ns=body.start_ns,
        end_ns=body.end_ns,
        notes=body.notes,
    )
    return _calibration_gt_row_to_event(row)


@app.get("/api/v1/calibration/{session_id}/ground-truth")
async def calibration_ground_truth_list(session_id: str, request: Request):
    storage: Storage = request.app.state.storage
    await _require_calibration_session(storage, session_id)
    rows = await storage.list_calibration_ground_truth(session_id)
    return [_calibration_gt_row_to_event(row) for row in rows]


@app.patch("/api/v1/calibration/ground-truth/{event_id}")
async def calibration_ground_truth_update(
    event_id: str, body: CalibrationGroundTruthUpdate, request: Request
):
    storage: Storage = request.app.state.storage
    existing = await storage.get_calibration_ground_truth(event_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"ground-truth event {event_id} not found")
    updates = body.model_dump(exclude_none=True)
    start_ns = updates.get("start_ns", existing["start_ns"])
    end_ns = updates.get("end_ns", existing["end_ns"])
    if end_ns < start_ns:
        raise HTTPException(status_code=422, detail="end_ns must be >= start_ns")
    new_node_id = updates.get("source_node_id")
    if new_node_id is not None and new_node_id != existing.get("source_node_id"):
        session_row = await storage.get_capture_session(existing["session_id"])
        lat, lon, alt_m = await _resolve_ground_truth_node_position(
            request, session_row or {}, new_node_id
        )
        updates.update(lat=lat, lon=lon, alt_m=alt_m)
    row = await storage.update_calibration_ground_truth(event_id, updates)
    return _calibration_gt_row_to_event(row)


@app.delete("/api/v1/calibration/ground-truth/{event_id}")
async def calibration_ground_truth_delete(event_id: str, request: Request):
    storage: Storage = request.app.state.storage
    deleted = await storage.delete_calibration_ground_truth(event_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"ground-truth event {event_id} not found")
    return Response(status_code=204)


def _load_calibration_manifest(session_row: dict) -> dict:
    manifest_path = session_row.get("calibration_manifest_path")
    if not manifest_path or not Path(manifest_path).exists():
        raise HTTPException(
            status_code=409,
            detail="calibration session has no artifact yet (still processing or failed)",
        )
    return json.loads(Path(manifest_path).read_text())


@app.get("/api/v1/calibration/{session_id}/manifest")
async def calibration_manifest_get(session_id: str, request: Request):
    """Lightweight manifest view for the frontend waveform trimmer.

    Exposes only what the trimmer needs (node id, audio window, sample rate)
    without requiring a full bundle export.
    """
    storage: Storage = request.app.state.storage
    session_row = await _require_calibration_session(storage, session_id)
    manifest = _load_calibration_manifest(session_row)
    return {
        "session_id": session_id,
        "time_window": manifest.get("time_window"),
        "nodes": [
            {
                "node_id": node["node_id"],
                "audio_start_time_ns": node.get("audio_start_time_ns") or 0,
                "sample_rate_hz": node.get("sample_rate_hz"),
            }
            for node in manifest.get("nodes", [])
        ],
    }


@app.get("/api/v1/calibration/{session_id}/audio/{node_id}")
async def calibration_node_audio_get(session_id: str, node_id: str, request: Request):
    """Stream one node's raw multichannel WAV for browser-side waveform decode."""
    storage: Storage = request.app.state.storage
    session_row = await _require_calibration_session(storage, session_id)
    manifest = _load_calibration_manifest(session_row)
    node = next((n for n in manifest.get("nodes", []) if n.get("node_id") == node_id), None)
    if node is None:
        raise HTTPException(status_code=404, detail=f"node {node_id} not in session {session_id}")
    artifact_dir = Path(session_row["calibration_manifest_path"]).parent
    wav_path = artifact_dir / node["audio_file"]
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"audio for node {node_id} not found")
    return FileResponse(wav_path, media_type="audio/wav", filename=wav_path.name)


@app.get("/api/v1/calibration/{session_id}/bundle")
async def calibration_bundle_download(session_id: str, request: Request):
    storage: Storage = request.app.state.storage
    row = await _require_calibration_session(storage, session_id)
    _load_calibration_manifest(row)  # raises 409 while processing
    artifact_dir = Path(row["calibration_manifest_path"]).parent
    ground_truth_rows = await storage.list_calibration_ground_truth(session_id)
    ground_truth_payload = build_ground_truth_payload(ground_truth_rows)
    bundle_path = artifact_dir / f"calibration_{session_id}.zip"
    await asyncio.to_thread(
        write_bundle_zip, artifact_dir, ground_truth_payload, bundle_path
    )
    return FileResponse(
        bundle_path,
        media_type="application/zip",
        filename=bundle_path.name,
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
