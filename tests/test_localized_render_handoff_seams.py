"""Collaborator seams around the Rust classifier-render handoff (plan P0).

The default seams must leave a legacy (provenance-less) render on the
pre-overhaul path; the hooks must fire once per solve, in the order later
milestones rely on (observation before classification/reporting).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.fusion_node import EventCandidate, FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine
from minimappr.core.localized_render_handoff_hooks import (
    SOLVE_ORIGIN_PYTHON_INGEST,
    SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER,
    HandoffSolveGeometry,
    LocalizationSeamCollaborators,
)
from minimappr.core.localization_provenance import ManifestLocalizationProvenanceRecorder
from minimappr.core.localized_handoff_geometry import ManifestSolveGeometryResolver
from minimappr.core.localized_handoff_solution import SnapshotGeometrySolutionResolver
from minimappr.core.localized_render_localization import HandoffLocalizationEstimate
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.position_observation_pipeline import PositionObservationPipeline
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import GeoPoint, NodeSpec, NodeType, TimeQuality
from minimappr.storage.db import Storage

_TETRA_OFFSETS_M = [
    (0.0, 0.05, 0.0),
    (0.0433, 0.025, 0.0),
    (0.0, 0.0, 0.0),
    (0.02165, 0.025, 0.04082),
]


async def _started_fusion(tmp_path: Path) -> tuple[FusionNode, Storage]:
    settings = Settings(
        db_path=tmp_path / "handoff_seams.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        fusion_worker_count=1,
        fusion_event_queue_size=8,
        localization_single_node_solver="rust",
    )
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    storage = Storage(settings.db_path)
    await storage.initialize()
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=LocalizationEngine(max_tau_s=0.03),
        classifier=HeuristicClassifier(),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()
    return fusion, storage


def _render(manifest_id: str, **overrides) -> LocalizedClassifierRenderRequest:
    node = NodeSpec(
        id="sirith-seam",
        node_type=NodeType.SIRITH_TETRA,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=_TETRA_OFFSETS_M,
        capabilities=["audio"],
        metadata={},
    )
    fields = dict(
        manifest_id=manifest_id,
        node=node,
        event_time_ns=1_739_950_000_000_000_000,
        sample_rate_hz=48_000,
        decoded_audio=np.random.default_rng(77).normal(0.0, 0.3, size=48_000).astype(np.float32),
        localization_position_m=(3.0, 4.0, 0.5),
        localization_confidence=0.9,
        localization_gdop=1.1,
        localization_range_projection_mode="range_refined",
        localization_method="rust_srp_phat",
        environment={"temperature_c": 18.0, "humidity_fraction": 0.4},
    )
    fields.update(overrides)
    return LocalizedClassifierRenderRequest(**fields)


class _RecordingSeams:
    """Implements every seam; records calls and optionally overrides results."""

    def __init__(self, storage: Storage, *, geometry_override=None, estimate_override=None) -> None:
        self.storage = storage
        self.geometry_override = geometry_override
        self.estimate_override = estimate_override
        self.events: list[tuple[str, dict]] = []

    def resolve_handoff_geometry(self, *, payload, node, registry_geometry):
        self.events.append(("geometry", {"registry_geometry": registry_geometry}))
        return self.geometry_override

    def resolve_handoff_solution(self, *, payload, geometry, reporting_modality):
        self.events.append(("solution", {"geometry": geometry, "reporting_modality": reporting_modality}))
        return self.estimate_override

    def record_handoff_provenance(self, *, payload, geometry, localized_product, reporting_modality):
        localized_product.extra_classification_features["seam_provenance_marker"] = payload.manifest_id
        self.events.append(("provenance", {"reporting_modality": reporting_modality}))

    async def observe_localized_solve(self, *, localized_product, reporting_modality, solve_origin):
        stored_before_observation = await self.storage.list_detections(limit=10)
        self.events.append(
            (
                "observation",
                {
                    "candidate_id": localized_product.candidate.id,
                    "reporting_modality": reporting_modality,
                    "solve_origin": solve_origin,
                    "stored_before_observation": len(stored_before_observation),
                },
            )
        )

    def as_collaborators(self) -> LocalizationSeamCollaborators:
        return LocalizationSeamCollaborators(
            geometry_resolver=self,
            solution_resolver=self,
            provenance_recorder=self,
            position_observation_sink=self,
        )


def _install(fusion: FusionNode, seams: _RecordingSeams) -> None:
    collaborators = seams.as_collaborators()
    fusion._localization_seam_collaborators = collaborators
    fusion._localized_render_handoff.collaborators = collaborators


async def test_default_seams_keep_the_legacy_path_for_renders_without_provenance(tmp_path: Path) -> None:
    fusion, storage = await _started_fusion(tmp_path)
    collaborators = fusion._localization_seam_collaborators
    assert isinstance(collaborators.geometry_resolver, ManifestSolveGeometryResolver)
    assert isinstance(collaborators.solution_resolver, SnapshotGeometrySolutionResolver)
    assert isinstance(collaborators.provenance_recorder, ManifestLocalizationProvenanceRecorder)
    assert isinstance(collaborators.position_observation_sink, PositionObservationPipeline)
    assert fusion._localized_render_handoff.collaborators is collaborators

    await fusion.ingest_localized_render(_render("manifest-seam-default"))
    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    assert detections[0]["feature_summary"]["localization_method"] == "rust_srp_phat"
    assert tuple(detections[0]["position_m"]) == (3.0, 4.0, 0.5)
    assert "localization_solve" not in detections[0]["feature_summary"]
    await fusion.stop()
    await storage.close()


async def test_seams_fire_once_per_render_with_observation_before_reporting(tmp_path: Path) -> None:
    fusion, storage = await _started_fusion(tmp_path)
    seams = _RecordingSeams(storage)
    _install(fusion, seams)

    await fusion.ingest_localized_render(_render("manifest-seam-order"))

    assert [name for name, _ in seams.events] == ["geometry", "solution", "provenance", "observation"]
    registry_geometry = seams.events[0][1]["registry_geometry"]
    assert len(registry_geometry.selected_sensor_ids) == len(_TETRA_OFFSETS_M)
    assert registry_geometry.reference_sensor == registry_geometry.selected_sensor_ids[0]
    observation = seams.events[3][1]
    assert observation["solve_origin"] == SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER
    assert observation["reporting_modality"] == "localized"
    assert observation["candidate_id"] == "rust-manifest-seam-order"
    assert observation["stored_before_observation"] == 0

    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    # Provenance written onto the product reaches the stored detection features.
    assert detections[0]["feature_summary"]["seam_provenance_marker"] == "manifest-seam-order"
    await fusion.stop()
    await storage.close()


async def test_observation_still_fires_when_chunked_classification_suppresses(tmp_path: Path) -> None:
    fusion, storage = await _started_fusion(tmp_path)
    seams = _RecordingSeams(storage)
    _install(fusion, seams)
    fusion._classification_chunking_policy = object()
    fusion._should_suppress_chunked_classification = lambda product: True

    await fusion.ingest_localized_render(_render("manifest-seam-suppressed"))

    assert [name for name, _ in seams.events][-1] == "observation"
    assert fusion._metrics.birdnet_chunk_dispatches_suppressed == 1
    assert await storage.list_detections(limit=10) == []
    await fusion.stop()
    await storage.close()


async def test_geometry_and_solution_overrides_replace_the_defaults(tmp_path: Path) -> None:
    fusion, storage = await _started_fusion(tmp_path)
    override_estimate = HandoffLocalizationEstimate(
        position_m=(6.0, -2.0, 1.0),
        confidence=0.85,
        gdop=1.0,
        position_covariance_m2=None,
        range_observability=None,
        residual_rms_seconds=None,
        range_projection_mode="range_refined",
        method="seam_override_solver",
    )
    override_geometry = HandoffSolveGeometry(
        selected_sensor_ids=["manifest-sensor-a", "manifest-sensor-b", "manifest-sensor-c", "manifest-sensor-d"],
        selected_positions={
            f"manifest-sensor-{suffix}": np.asarray(offset, dtype=np.float64)
            for suffix, offset in zip("abcd", _TETRA_OFFSETS_M)
        },
        reference_sensor="manifest-sensor-a",
    )
    seams = _RecordingSeams(storage, geometry_override=override_geometry, estimate_override=override_estimate)
    _install(fusion, seams)

    await fusion.ingest_localized_render(_render("manifest-seam-override"))

    assert seams.events[1][1]["geometry"] is override_geometry
    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    assert detections[0]["feature_summary"]["localization_method"] == "seam_override_solver"
    assert tuple(detections[0]["position_m"]) == (6.0, -2.0, 1.0)
    await fusion.stop()
    await storage.close()


async def test_python_ingest_path_observes_each_localized_product(tmp_path: Path) -> None:
    fusion, storage = await _started_fusion(tmp_path)
    seams = _RecordingSeams(storage)
    _install(fusion, seams)
    rendered_product_holder: list = []

    async def _capture_product(*, localized_product, reporting_modality, solve_origin):
        rendered_product_holder.append(localized_product)

    # Reuse a real LocalizedCandidate: capture one from the render path, then
    # feed it back through the Python localization worker.
    seams.observe_localized_solve = _capture_product
    await fusion.ingest_localized_render(_render("manifest-seam-product-source"))
    product = rendered_product_holder[0]
    seams.events.clear()
    seams.observe_localized_solve = _RecordingSeams.observe_localized_solve.__get__(seams)

    async def _localize_stub(candidate):
        return product

    fusion._localize_candidate = _localize_stub
    candidate = EventCandidate(
        id="python-ingest-seam",
        source_node_id="sirith-seam",
        event_time_ns=1_739_950_000_000_000_000,
        sample_rate_hz=48_000,
        source_type="raw_sensor",
        time_quality=TimeQuality.GPS_LOCKED,
        source_observation_ids=[],
    )
    await fusion._localization_queue.put(candidate)
    await asyncio.wait_for(fusion._localization_queue.join(), timeout=5.0)

    observations = [details for name, details in seams.events if name == "observation"]
    assert len(observations) == 1
    assert observations[0]["solve_origin"] == SOLVE_ORIGIN_PYTHON_INGEST
    assert observations[0]["reporting_modality"] == "localized"
    await fusion.stop()
    await storage.close()
