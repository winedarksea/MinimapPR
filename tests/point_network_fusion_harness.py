"""A started ``FusionNode`` for point-network tests (P7 clip classification, P8 skips).

The classifier records every clip it is handed so tests can check what was
classified; ``TruthLocalizer`` stands in for the solver on the Python-ingest
path, where the solve itself is not under test. A point-network Rust render's
clip is classified on a classification worker, so ``stored_detections_by_solve``
drains the stage queues before reading.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest, load_localized_render_manifest_bundle
from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.fusion_node import EventCandidate, FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine
from minimappr.core.localization_provenance import site_frame_id_for_coordinate_frame
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import GeoPoint, LocalizationResult, NodeSpec, NodeType, TimeQuality
from minimappr.sim.sim_scenario_scoring import drain_fusion_queues
from minimappr.storage.db import Storage
from tests.point_network_scene import POINT_NODE_POSITIONS_M, Caller, point_network_render_manifest

# Wide enough for the network's ~85 m baselines.
LOCALIZATION_ENGINE_MAX_TAU_SECONDS = 0.3
SOLVE_WINDOW_SECONDS = 0.2
RENDER_HALF_SPAN_SECONDS = 0.5


class RecordingClassifier(HeuristicClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[np.ndarray] = []

    def classify(self, samples, sample_rate_hz):
        self.inputs.append(np.asarray(samples, dtype=np.float64).copy())
        return super().classify(samples, sample_rate_hz)


class TruthLocalizer:
    """Returns a fixed (true) position for every solve."""

    def __init__(self, position_m) -> None:
        self.position_m = tuple(float(axis) for axis in position_m)

    def localize(self, sensor_positions, sensor_windows, sample_rate_hz, temperature_c, humidity_fraction, **kwargs):
        del sensor_windows, sample_rate_hz, temperature_c, humidity_fraction, kwargs
        return LocalizationResult(
            position_m=self.position_m, confidence=0.9, gdop=1.0, reference_sensor=sorted(sensor_positions)[0], tdoa_s={}
        )


async def started_point_network_fusion(
    tmp_path: Path, *, localizer=None, **settings_overrides
) -> tuple[FusionNode, Storage, RecordingClassifier]:
    settings = Settings(
        db_path=tmp_path / "point_network.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        fusion_worker_count=1,
        fusion_event_queue_size=8,
        **settings_overrides,
    )
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    storage = Storage(settings.db_path)
    await storage.initialize()
    classifier = RecordingClassifier()
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=localizer if localizer is not None else LocalizationEngine(max_tau_s=LOCALIZATION_ENGINE_MAX_TAU_SECONDS),
        classifier=classifier,
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()
    return fusion, storage, classifier


async def register_point_nodes(fusion: FusionNode, storage: Storage) -> None:
    """Registry entries for the solve, storage rows for the stored detection's node key."""
    for node_id, position_m in POINT_NODE_POSITIONS_M.items():
        spec = NodeSpec(
            id=node_id,
            node_type=NodeType.POINT,
            position_m=position_m,
            sensor_offsets_m=[(0.0, 0.0, 0.0)],
            capabilities=["audio"],
        )
        await fusion.registry.upsert(spec, last_seen_ns=0)
        await storage.upsert_node(spec, last_seen_ns=0)


def python_ingest_candidate(
    *, candidate_id: str, triggering_node_id: str, event_time_ns: int, sample_rate_hz: int
) -> EventCandidate:
    return EventCandidate(
        id=candidate_id,
        source_node_id=triggering_node_id,
        event_time_ns=event_time_ns,
        sample_rate_hz=sample_rate_hz,
        source_type="raw_sensor",
        time_quality=TimeQuality.GPS_LOCKED,
        source_observation_ids=[],
    )


def point_network_render_request(
    fusion: FusionNode, caller: Caller, *, reporting_node_id: str, solve_id: str
) -> LocalizedClassifierRenderRequest:
    manifest = point_network_render_manifest(
        caller=caller,
        reporting_node_id=reporting_node_id,
        frame_id=site_frame_id_for_coordinate_frame(fusion.coordinate_frame),
        solve_id=solve_id,
        manifest_id=f"render-{solve_id}",
        solve_window_seconds=SOLVE_WINDOW_SECONDS,
        render_half_span_seconds=RENDER_HALF_SPAN_SECONDS,
    )
    return load_localized_render_manifest_bundle(manifest_payload=manifest).request


async def stored_detections_by_solve(storage: Storage, fusion: FusionNode | None = None) -> dict[str, dict]:
    """Stored detections keyed by their Rust solve id; drains ``fusion``'s stage queues first."""
    if fusion is not None:
        await drain_fusion_queues(fusion)
    detections = await storage.list_detections(limit=20)
    return {
        detection["feature_summary"]["localization_solve"]["solve_id"]: detection
        for detection in detections
        if "localization_solve" in detection["feature_summary"]
    }
