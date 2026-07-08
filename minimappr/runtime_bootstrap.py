"""Application runtime bootstrap helpers.

This module owns the pure startup construction helpers used by the FastAPI
entrypoint so `main.py` can focus on orchestration rather than object graph
assembly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import time
from dataclasses import dataclass, field
from typing import Callable

from minimappr.api.live import LiveEventHub
from minimappr.api.spool_consumer import IngestSpoolConfig, IngestSpoolConsumer
from minimappr.api.stream_consumer import IngestStreamConsumer
from minimappr.api.transports import HttpIngestTransport
from minimappr.classifiers.factory import create_classifier
from minimappr.cleanup_service import CleanupService
from minimappr.config import Settings
from minimappr.core.bit_report import BITReportEvaluator
from minimappr.core.capture_session import CaptureSessionManager
from minimappr.core.capture_session import CaptureSessionRecord, CaptureState
from minimappr.core.diagnostics import DiagnosticsService
from minimappr.core.effector_rules import EffectorRuleActionHandler
from minimappr.core.effectors.registry import EffectorManager
from minimappr.core.environment import LiveEnvironmentProvider
from minimappr.core.federation import FederationCoordinator
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.iamf_pipeline import IamfPipeline
from minimappr.core.localization_dispatch import build_localizer_from_settings
from minimappr.core.site_origin import resolve_site_origin_from_nodes
from minimappr.core.cluster_registry import ClusterRegistry
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.models import GeoPoint, TrackState
from minimappr.storage.db import Storage


logger = logging.getLogger(__name__)


def _apply_settings_site_origin_resolution(settings: Settings, resolved_site_origin) -> None:
    settings.site_origin_lat = resolved_site_origin.origin.lat
    settings.site_origin_lon = resolved_site_origin.origin.lon
    settings.site_origin_alt_m = resolved_site_origin.origin.alt_m


@dataclass
class _CommonLiveRuntimeServices:
    live_hub: LiveEventHub
    coordinate_frame: LocalCoordinateFrame
    environment_provider: LiveEnvironmentProvider
    cleanup_service: CleanupService
    bit_evaluator: BITReportEvaluator


@dataclass
class _CombinedRuntimeCoreServices:
    registry: NodeRegistry
    cluster_registry: ClusterRegistry
    audio_buffer: MultiSensorBuffer
    localizer: object
    classifier: object
    tracker: TrackManager
    zone_matcher: ZoneMatcher
    fusion_node: FusionNode
    ingest_transport: HttpIngestTransport
    diagnostics: DiagnosticsService
    ingest_spool_consumer: IngestSpoolConsumer


@dataclass
class _CombinedRuntimeTaskHandles:
    cleanup_task: asyncio.Task | None = None
    site_origin_reconcile_task: asyncio.Task | None = None
    ingest_spool_tasks: list[asyncio.Task] = field(default_factory=list)
    ingest_stream_consumer_watchdog_task: asyncio.Task | None = None


@dataclass
class _ApiOnlyRuntimeTaskHandles:
    live_db_poll_task: asyncio.Task | None = None


def _build_common_live_runtime_services(
    settings: Settings,
    *,
    storage: Storage,
) -> _CommonLiveRuntimeServices:
    return _CommonLiveRuntimeServices(
        live_hub=LiveEventHub(),
        coordinate_frame=LocalCoordinateFrame(
            origin=GeoPoint(
                lat=settings.site_origin_lat,
                lon=settings.site_origin_lon,
                alt_m=settings.site_origin_alt_m,
            ),
            mode=settings.coordinate_mode,
        ),
        environment_provider=LiveEnvironmentProvider(
            fallback_temperature_c=settings.default_temperature_c,
            fallback_humidity_fraction=settings.default_humidity,
            max_reading_age_seconds=settings.environment_reading_max_age_seconds,
        ),
        cleanup_service=CleanupService(settings=settings, storage=storage),
        bit_evaluator=BITReportEvaluator(),
    )


def _build_capture_manager(
    settings: Settings,
    *,
    live_hub: LiveEventHub,
    sidecar_url: str | None,
    storage: Storage,
    multi_sensor_buffer: MultiSensorBuffer | None = None,
) -> CaptureSessionManager:
    capture_manager = CaptureSessionManager()
    capture_manager.set_final_track_settle_seconds(settings.capture_final_tracks_settle_seconds)
    iamf_pipeline = IamfPipeline(
        sidecar_url=sidecar_url,
        db_storage=storage,
        multi_sensor_buffer=multi_sensor_buffer,
        artifact_dir=settings.large_artifact_dir,
        iamf_ambi_profile=settings.iamf_ambi_profile,
    )

    async def _run_capture_post_processing(record):
        await iamf_pipeline.run(record)

    async def _persist_and_broadcast_capture_session(record: CaptureSessionRecord) -> None:
        await storage.upsert_capture_session(record)
        await live_hub.broadcast(
            {
                "type": "recording_status",
                "session": _session_record_to_recording_session(record),
            }
        )

    capture_manager.set_post_process_callback(_run_capture_post_processing)
    capture_manager.set_session_update_callback(_persist_and_broadcast_capture_session)
    return capture_manager


def _capture_state_to_recording_status(state: CaptureState) -> str:
    mapping = {
        CaptureState.PENDING: "starting",
        CaptureState.RECORDING: "recording",
        CaptureState.AWAITING_FINAL_TRACKS: "stopping",
        CaptureState.PROCESSING: "stopping",
        CaptureState.COMPLETED: "completed",
        CaptureState.FAILED: "failed",
    }
    return mapping.get(state, "idle")


def _session_record_to_recording_session(record: CaptureSessionRecord) -> dict:
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


def _build_combined_runtime_core_services(
    settings: Settings,
    *,
    classifier_factory: Callable[[Settings], object],
    common_live_runtime_services: _CommonLiveRuntimeServices,
    localization_cfg,
    storage: Storage,
    tracking_cfg,
) -> _CombinedRuntimeCoreServices:
    registry = NodeRegistry()
    cluster_registry = ClusterRegistry()
    audio_buffer = MultiSensorBuffer(max_duration_seconds=localization_cfg.max_sensor_buffer_seconds)
    localizer = build_localizer_from_settings(localization_cfg)
    classifier = classifier_factory(settings)
    tracker = TrackManager(tracking_cfg)
    zone_matcher = ZoneMatcher(storage=storage)
    fusion_node = FusionNode(
        settings=settings,
        registry=registry,
        buffer=audio_buffer,
        localizer=localizer,
        classifier=classifier,
        tracker=tracker,
        storage=storage,
        live_callback=common_live_runtime_services.live_hub.broadcast,
        coordinate_frame=common_live_runtime_services.coordinate_frame,
        zone_matcher=zone_matcher,
        environment_provider=common_live_runtime_services.environment_provider,
        cluster_registry=cluster_registry,
    )
    ingest_transport = HttpIngestTransport(fusion_node)
    diagnostics = DiagnosticsService(
        settings=settings,
        storage=storage,
        fusion_node=fusion_node,
        classifier=classifier,
    )
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
    return _CombinedRuntimeCoreServices(
        registry=registry,
        cluster_registry=cluster_registry,
        audio_buffer=audio_buffer,
        localizer=localizer,
        classifier=classifier,
        tracker=tracker,
        zone_matcher=zone_matcher,
        fusion_node=fusion_node,
        ingest_transport=ingest_transport,
        diagnostics=diagnostics,
        ingest_spool_consumer=ingest_spool_consumer,
    )


def _build_combined_runtime_federation(
    settings: Settings,
    *,
    coordinate_frame: LocalCoordinateFrame,
    live_hub: LiveEventHub,
    tracker: TrackManager,
) -> FederationCoordinator:
    async def _federation_local_tracks(now_ns: int) -> list[TrackState]:
        tracks = await tracker.snapshot(now_ns=now_ns)
        active: list[TrackState] = []
        for track in tracks:
            if track.status not in {"tentative", "confirmed", "coasting"}:
                continue
            track.position_geo = coordinate_frame.local_to_geo(track.position_m)
            active.append(track)
        return active

    return FederationCoordinator(
        settings=settings,
        track_supplier=_federation_local_tracks,
        live_callback=live_hub.broadcast,
    )


def _build_api_only_runtime_federation(
    settings: Settings,
    *,
    live_hub: LiveEventHub,
) -> FederationCoordinator:
    async def _empty_local_tracks(now_ns: int) -> list[TrackState]:
        del now_ns
        return []

    return FederationCoordinator(
        settings=settings,
        track_supplier=_empty_local_tracks,
        live_callback=live_hub.broadcast,
    )


def _build_effector_manager(
    settings: Settings,
    *,
    storage: Storage,
    live_hub: LiveEventHub,
) -> EffectorManager:
    """Build the (optional) effector subsystem.

    Always constructed — start() is cheap/dormant when the `effectors` DB
    table is empty, so no separate enable path is needed beyond the
    `effectors_enabled` kill-switch checked by the caller before start().
    """
    return EffectorManager(
        storage=storage,
        live_callback=live_hub.broadcast,
        config=settings.effector_manager_config(),
    )


def _wire_effector_zone_interlocks(
    effector_manager: EffectorManager,
    *,
    coordinate_frame: LocalCoordinateFrame,
    zone_matcher: ZoneMatcher,
) -> None:
    async def _resolve_target_zone_ids(target_pos) -> set[str]:
        geo = coordinate_frame.local_to_geo(target_pos)
        return set(await zone_matcher.match_geo_point(geo.lat, geo.lon))

    effector_manager.set_target_zone_resolver(_resolve_target_zone_ids)


def _wire_effector_rules_handler(fusion_node: FusionNode, effector_manager: EffectorManager) -> None:
    fusion_node.set_action_handler("effector", EffectorRuleActionHandler(effector_manager))


async def _initialize_storage_and_resolve_site_origin(
    settings: Settings,
    *,
    log_resolution: bool,
):
    storage = Storage(settings.storage_config().db_path)
    await storage.initialize()
    resolved_site_origin = resolve_site_origin_from_nodes(
        settings,
        nodes=await storage.list_nodes(limit=4096),
        now_ns=time.time_ns(),
    )
    _apply_settings_site_origin_resolution(settings, resolved_site_origin)
    if log_resolution:
        logger.info(
            "Resolved site origin via %s: lat=%.6f lon=%.6f alt=%.2f nodes=%s",
            resolved_site_origin.source,
            settings.site_origin_lat,
            settings.site_origin_lon,
            settings.site_origin_alt_m,
            list(resolved_site_origin.contributing_node_ids),
        )
    return storage, resolved_site_origin


async def _start_api_only_runtime_services(
    app,
    *,
    api_live_db_poll_loop,
    environment_provider: LiveEnvironmentProvider,
    federation: FederationCoordinator,
    storage: Storage,
) -> _ApiOnlyRuntimeTaskHandles:
    task_handles = _ApiOnlyRuntimeTaskHandles()
    environment_provider.bootstrap(await storage.list_latest_environment_per_node(limit=1024))
    await federation.start()
    task_handles.live_db_poll_task = asyncio.create_task(api_live_db_poll_loop(app))
    return task_handles


async def _stop_api_only_runtime_services(
    app,
    task_handles: _ApiOnlyRuntimeTaskHandles,
    *,
    clear_transient_ingest_runtime_state,
    federation: FederationCoordinator,
    shutdown_timeout_seconds: float,
    storage: Storage,
) -> None:
    if task_handles.live_db_poll_task is not None:
        task_handles.live_db_poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(task_handles.live_db_poll_task, timeout=shutdown_timeout_seconds)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(federation.stop(), timeout=shutdown_timeout_seconds)
    await storage.close()
    clear_transient_ingest_runtime_state(app.state)


async def _start_combined_runtime_background_tasks(
    app,
    *,
    cleanup_loop,
    environment_provider: LiveEnvironmentProvider,
    federation: FederationCoordinator,
    fusion_node: FusionNode,
    ingest_spool_consumer: IngestSpoolConsumer,
    ingest_stream_consumer_enabled: bool,
    initial_resolution_source: str,
    maintain_ingest_stream_consumer,
    reconcile_site_origin_after_startup,
    settings: Settings,
    should_schedule_site_origin_reconciliation,
    storage: Storage,
) -> _CombinedRuntimeTaskHandles:
    task_handles = _CombinedRuntimeTaskHandles()
    environment_provider.bootstrap(await storage.list_latest_environment_per_node(limit=1024))
    await fusion_node.start()
    await federation.start()

    task_handles.cleanup_task = asyncio.create_task(cleanup_loop(app))
    app.state.cleanup_task = task_handles.cleanup_task
    if ingest_stream_consumer_enabled:
        task_handles.ingest_stream_consumer_watchdog_task = asyncio.create_task(
            maintain_ingest_stream_consumer(app)
        )
        app.state.ingest_stream_consumer_watchdog_task = (
            task_handles.ingest_stream_consumer_watchdog_task
        )
    else:
        task_handles.ingest_spool_tasks = [
            asyncio.create_task(ingest_spool_consumer.run_forever(worker_name=f"spool-{index + 1}"))
            for index in range(settings.ingest_spool_worker_count)
        ]
        app.state.ingest_spool_tasks = task_handles.ingest_spool_tasks
    if should_schedule_site_origin_reconciliation(
        settings,
        initial_resolution_source=initial_resolution_source,
    ):
        task_handles.site_origin_reconcile_task = asyncio.create_task(
            reconcile_site_origin_after_startup(app)
        )
    app.state.site_origin_reconcile_task = task_handles.site_origin_reconcile_task
    return task_handles


async def _stop_combined_runtime_background_tasks(
    app,
    task_handles: _CombinedRuntimeTaskHandles,
    *,
    clear_transient_ingest_runtime_state,
    shutdown_timeout_seconds: float,
) -> None:
    if task_handles.site_origin_reconcile_task is not None:
        task_handles.site_origin_reconcile_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(
                task_handles.site_origin_reconcile_task,
                timeout=shutdown_timeout_seconds,
            )
    if task_handles.cleanup_task is not None:
        task_handles.cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(task_handles.cleanup_task, timeout=shutdown_timeout_seconds)
    for task in task_handles.ingest_spool_tasks:
        task.cancel()
    if task_handles.ingest_spool_tasks:
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*task_handles.ingest_spool_tasks),
                timeout=shutdown_timeout_seconds,
            )
    if task_handles.ingest_stream_consumer_watchdog_task is not None:
        task_handles.ingest_stream_consumer_watchdog_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(
                task_handles.ingest_stream_consumer_watchdog_task,
                timeout=shutdown_timeout_seconds,
            )
    ingest_stream_consumer: IngestStreamConsumer | None = getattr(app.state, "ingest_stream_consumer", None)
    if ingest_stream_consumer is not None:
        await ingest_stream_consumer.stop()
    clear_transient_ingest_runtime_state(app.state)


def _cleanup_multiprocessing_shutdown_resources() -> None:
    # Wait briefly for active children so resource_tracker can unregister
    # shared_memory objects before the event loop closes.
    for proc in multiprocessing.active_children():
        try:
            proc.join(timeout=1.0)
        except Exception:  # noqa: BLE001
            pass


async def _shutdown_combined_runtime_services(
    *,
    classifier,
    federation: FederationCoordinator,
    fusion_node: FusionNode,
    shutdown_timeout_seconds: float,
    storage: Storage,
) -> None:
    try:
        await asyncio.wait_for(federation.stop(), timeout=shutdown_timeout_seconds)
    except Exception as exc:
        logger.warning("Federation stop failed during shutdown: %s", exc)

    # Cancel any in-flight BirdNET predictions and terminate their worker
    # subprocesses before stopping the fusion node so queue consumers can drain.
    try:
        classifier.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Classifier close failed during shutdown: %s", exc)

    try:
        await asyncio.wait_for(fusion_node.stop(), timeout=shutdown_timeout_seconds)
    except Exception as exc:
        logger.warning("Fusion node stop failed during shutdown: %s", exc)

    try:
        await asyncio.wait_for(storage.close(), timeout=shutdown_timeout_seconds)
    except Exception as exc:
        logger.warning("Storage close failed during shutdown: %s", exc)

    try:
        _cleanup_multiprocessing_shutdown_resources()
        logger.info("Multiprocessing cleanup completed during shutdown")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Multiprocessing cleanup warning: %s", exc)
