"""An omni-audio render keeps its solve's bearing (review finding 5).

The render's audio modality (omni fallback clip) is separate from its spatial
status: an srp_phat solve with a steering direction is re-solved, gated and
tracked exactly like a localized render's, and the stored detection carries
the solve-time bearing. Fallbacks with no solve, and solves below
``omni_audio_bearing_min_localization_confidence``, stay omni at the node.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import numpy as np
import pytest

from minimappr.api.rust_dsp_manifests import load_localized_render_manifest_bundle
from minimappr.api.rust_event_localization_manifest import load_event_localization_request
from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.handoff_spatial_status import (
    CLASSIFICATION_AUDIO_MODALITY_FEATURE_KEY,
    LOCALIZATION_SOLVE_BEARING_FEATURE_KEY,
    solve_time_bearing,
)
from minimappr.core.localization import LocalizationEngine
from minimappr.core.localization_provenance import site_frame_id_for_coordinate_frame
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import DetectionSolveBearing, GeoPoint
from minimappr.storage.db import Storage
from tests.localization_solve_manifest_fixtures import (
    PLANAR_NODE_ID,
    POINT_NODE_ID,
    classifier_render_manifest,
    mixed_cluster_sensor_positions_m,
    mixed_cluster_solve_localization_payload,
    planar_node_payload,
)

SOURCE_POSITION_M = (44.0, 37.0, 6.0)
ARRIVAL_ORDER = (PLANAR_NODE_ID, POINT_NODE_ID)
PACKET_TOA_NS = 1_764_000_000_016_000_000 - 3_000_000_000
LOW_CONFIDENCE_FALLBACK_REASON = "low_localization_confidence"
OMNI_FALLBACK_RENDER_KIND = "birdnet_omni_fallback"
BEARING_TOLERANCE_DEG = 1e-6
RENDER_ONLY_MANIFEST_KEYS = frozenset({"classifier_render", "raw_render_bytes", "derived_handle"})


async def _started_fusion(tmp_path: Path, *, solver: str = "python_cartesian") -> tuple[FusionNode, Storage]:
    settings = Settings(
        db_path=tmp_path / "omni_audio_bearing.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        fusion_worker_count=1,
        fusion_event_queue_size=8,
        localization_single_node_solver=solver,
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


def _steering_direction() -> list[float]:
    centroid = np.mean(np.asarray(list(mixed_cluster_sensor_positions_m().values()), dtype=np.float64), axis=0)
    offset = np.asarray(SOURCE_POSITION_M) - centroid
    return (offset / np.linalg.norm(offset)).tolist()


def _render_manifest(
    fusion: FusionNode,
    *,
    omni_audio: bool,
    with_steering: bool = True,
    fallback_reason: str = LOW_CONFIDENCE_FALLBACK_REASON,
    manifest_id: str = "render-omni-audio-1",
    localization_confidence: float | None = None,
) -> dict:
    localization = mixed_cluster_solve_localization_payload(
        arrival_order=ARRIVAL_ORDER,
        source_position_m=SOURCE_POSITION_M,
        frame_id=site_frame_id_for_coordinate_frame(fusion.coordinate_frame),
        reporting_node_id=PLANAR_NODE_ID,
        solve_id="solve-omni-audio-1",
    )
    if with_steering:
        localization["steering_direction"] = _steering_direction()
    if localization_confidence is not None:
        localization["confidence"] = localization_confidence
    manifest = classifier_render_manifest(
        node_payload=planar_node_payload(), localization=localization, toa_ns=PACKET_TOA_NS, manifest_id=manifest_id
    )
    if omni_audio:
        manifest["classifier_render"]["render_kind"] = OMNI_FALLBACK_RENDER_KIND
        manifest["classifier_render"]["fallback_reason"] = fallback_reason
    return manifest


async def _stored_detection(tmp_path: Path, **manifest_kwargs) -> tuple[dict, FusionNode]:
    fusion, storage = await _started_fusion(tmp_path)
    try:
        manifest = _render_manifest(fusion, **manifest_kwargs)
        await fusion.ingest_localized_render(load_localized_render_manifest_bundle(manifest_payload=manifest).request)
        detections = await storage.list_detections(limit=10)
        assert len(detections) == 1
        return detections[0], fusion
    finally:
        await fusion.stop()
        await storage.close()


async def test_omni_audio_render_with_a_qualified_solve_is_placed_like_a_localized_render(tmp_path: Path) -> None:
    localized, _ = await _stored_detection(tmp_path / "localized", omni_audio=False)
    omni_audio, fusion = await _stored_detection(tmp_path / "omni", omni_audio=True)

    assert omni_audio["reporting_modality"] == "localized"
    assert omni_audio["spatial_display_mode"] != "node_only"
    assert omni_audio["feature_summary"][CLASSIFICATION_AUDIO_MODALITY_FEATURE_KEY] == "omni"
    assert localized["feature_summary"][CLASSIFICATION_AUDIO_MODALITY_FEATURE_KEY] == "localized"
    assert omni_audio["feature_summary"]["localization_method"] == localized["feature_summary"]["localization_method"]
    assert omni_audio["position_m"] == pytest.approx(localized["position_m"])
    omni_bearing = omni_audio["feature_summary"][LOCALIZATION_SOLVE_BEARING_FEATURE_KEY]
    localized_bearing = localized["feature_summary"][LOCALIZATION_SOLVE_BEARING_FEATURE_KEY]
    for key in ("bearing_deg", "half_width_deg", "elevation_deg"):
        assert omni_bearing[key] == pytest.approx(localized_bearing[key])
    assert omni_bearing["origin_position_m"] == pytest.approx(localized_bearing["origin_position_m"])
    assert omni_bearing["range_projection_mode"] == localized_bearing["range_projection_mode"]
    assert fusion._metrics.omni_audio_render_bearing_carried_count == 1


async def test_stored_bearing_starts_at_the_contributing_centroid(tmp_path: Path) -> None:
    detection, _ = await _stored_detection(tmp_path, omni_audio=True)
    bearing = DetectionSolveBearing.model_validate(detection["feature_summary"][LOCALIZATION_SOLVE_BEARING_FEATURE_KEY])
    centroid = np.mean(np.asarray(list(mixed_cluster_sensor_positions_m().values()), dtype=np.float64), axis=0)
    assert bearing.origin_position_m == pytest.approx(tuple(centroid))
    assert bearing.origin_node_id == PLANAR_NODE_ID
    assert bearing.origin_geo is not None
    east, north = SOURCE_POSITION_M[0] - centroid[0], SOURCE_POSITION_M[1] - centroid[1]
    assert bearing.bearing_deg == pytest.approx(math.degrees(math.atan2(east, north)) % 360.0, abs=0.5)
    assert bearing.range_observable


@pytest.mark.parametrize("fallback_reason", ["single_point_node", "insufficient_channels"])
async def test_no_solve_fallbacks_stay_at_the_node(tmp_path: Path, fallback_reason: str) -> None:
    detection, fusion = await _stored_detection(
        tmp_path, omni_audio=True, with_steering=False, fallback_reason=fallback_reason
    )
    assert detection["reporting_modality"] == "omni"
    assert detection["spatial_display_mode"] == "node_only"
    assert LOCALIZATION_SOLVE_BEARING_FEATURE_KEY not in detection["feature_summary"]
    assert detection["feature_summary"][CLASSIFICATION_AUDIO_MODALITY_FEATURE_KEY] == "omni"
    assert fusion._metrics.omni_audio_render_bearing_carried_count == 0


def test_omni_audio_bearing_floor_defaults_to_the_sidecar_render_threshold() -> None:
    settings = Settings()
    assert settings.omni_audio_bearing_min_localization_confidence == settings.min_localization_confidence
    with pytest.raises(ValueError, match="omni_audio_bearing_min_localization_confidence"):
        Settings(omni_audio_bearing_min_localization_confidence=1.5)


@pytest.mark.parametrize(
    ("confidence_offset_from_floor", "expect_promoted"),
    [(-0.01, False), (0.0, True), (0.01, True)],
    ids=["below_floor", "at_floor", "above_floor"],
)
async def test_omni_audio_solve_is_promoted_only_at_or_above_the_confidence_floor(
    tmp_path: Path, confidence_offset_from_floor: float, expect_promoted: bool
) -> None:
    floor = Settings().omni_audio_bearing_min_localization_confidence
    detection, fusion = await _stored_detection(
        tmp_path, omni_audio=True, localization_confidence=floor + confidence_offset_from_floor
    )
    metrics = fusion._metrics
    assert detection["feature_summary"][CLASSIFICATION_AUDIO_MODALITY_FEATURE_KEY] == "omni"
    if expect_promoted:
        assert detection["reporting_modality"] == "localized"
        assert LOCALIZATION_SOLVE_BEARING_FEATURE_KEY in detection["feature_summary"]
        assert metrics.omni_audio_render_bearing_carried_count == 1
        assert metrics.omni_audio_render_bearing_below_confidence_floor_count == 0
    else:
        # Exactly as the sidecar rendered it: omni at the node, no direction stored, no track.
        assert detection["reporting_modality"] == "omni"
        assert detection["spatial_display_mode"] == "node_only"
        assert detection["track_id"] is None
        assert LOCALIZATION_SOLVE_BEARING_FEATURE_KEY not in detection["feature_summary"]
        assert metrics.omni_audio_render_bearing_carried_count == 0
        assert metrics.omni_audio_render_bearing_below_confidence_floor_count == 1


async def test_a_gate_rejection_still_places_the_omni_audio_render_at_its_node(tmp_path: Path) -> None:
    fusion, storage = await _started_fusion(tmp_path)
    fusion._localization_quality_is_reportable = lambda **kwargs: "test_low_quality"
    try:
        manifest = _render_manifest(fusion, omni_audio=True)
        await fusion.ingest_localized_render(load_localized_render_manifest_bundle(manifest_payload=manifest).request)
        [detection] = await storage.list_detections(limit=10)
        assert detection["reporting_modality"] == "omni"
        assert LOCALIZATION_SOLVE_BEARING_FEATURE_KEY not in detection["feature_summary"]
        assert fusion._metrics.localization_rejected_low_quality_count == 1
    finally:
        await fusion.stop()
        await storage.close()


async def test_covering_omni_audio_render_contributes_its_event_solve_once(tmp_path: Path) -> None:
    # The sidecar's own estimate (no Python re-solve) states no weak range, so
    # the solve is tracker-eligible and the ledger join is what is tested.
    fusion, storage = await _started_fusion(tmp_path, solver="rust")
    try:
        render_manifest = _render_manifest(fusion, omni_audio=True)
        event_manifest = {key: value for key, value in render_manifest.items() if key not in RENDER_ONLY_MANIFEST_KEYS}
        event_manifest["manifest_type"] = "localization_result"
        event_manifest["manifest_id"] = "event-omni-audio-1"
        await fusion.observe_event_localization(load_event_localization_request(event_manifest))
        await fusion.ingest_localized_render(
            load_localized_render_manifest_bundle(manifest_payload=render_manifest).request
        )
        for _ in range(20):
            await asyncio.sleep(0)
        metrics = fusion._metrics
        assert metrics.position_observations_received == 2
        assert metrics.position_observations_not_tracker_eligible == 0
        assert metrics.position_observations_duplicate_solve == 1
    finally:
        await fusion.stop()
        await storage.close()


def test_solve_time_bearing_width_follows_the_tangential_spread() -> None:
    settings = Settings()
    frame = LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat")
    # Source due north at 100 m, 10 m up; 5 m tangential (east) sigma, 80 m radial.
    covariance = [[25.0, 0.0, 0.0], [0.0, 6400.0, 0.0], [0.0, 0.0, 25.0]]
    bearing = solve_time_bearing(
        reporting_node_id="node-a",
        contributing_centroid_m=np.zeros(3),
        position_m=(0.0, 100.0, 10.0),
        covariance_m2=covariance,
        range_projection_mode="range_boundary",
        coordinate_frame=frame,
        settings=settings,
    )
    assert bearing is not None
    assert bearing.bearing_deg == pytest.approx(0.0, abs=BEARING_TOLERANCE_DEG)
    expected_half_width = math.degrees(math.atan2(settings.entity_uncertainty_sigma_multiplier * 5.0, 100.0))
    assert bearing.half_width_deg == pytest.approx(
        min(max(expected_half_width, settings.entity_bearing_min_half_width_deg), settings.entity_bearing_max_half_width_deg)
    )
    assert bearing.elevation_deg == pytest.approx(math.degrees(math.atan2(10.0, 100.0)))
    assert not bearing.range_observable
    assert bearing.range_min_m is None and bearing.range_max_m is None
    assert bearing.range_projection_mode == "range_boundary"
    assert solve_time_bearing(
        reporting_node_id="node-a",
        contributing_centroid_m=np.zeros(3),
        position_m=(0.0, 0.0, 0.0),
        covariance_m2=covariance,
        range_projection_mode="range_refined",
        coordinate_frame=frame,
        settings=settings,
    ) is None
