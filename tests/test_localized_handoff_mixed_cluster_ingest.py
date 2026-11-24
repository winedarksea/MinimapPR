"""Mixed planar + point cluster renders through ``FusionNode.ingest_localized_render``
down to the stored detection (plan item P2, the Python half of E2).

Manifests are built per ``docs/localization_solve_contract.md`` because the
sidecar does not emit ``solve_provenance`` until R6; both node arrival orders,
both reporters and both ``localization_single_node_solver`` values are covered.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import numpy as np
import pytest

from minimappr.api.rust_dsp_manifests import load_localized_render_manifest_bundle
from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine
from minimappr.core.localization_provenance import LOCALIZATION_SOLVE_FEATURE_KEY, site_frame_id_for_coordinate_frame
from minimappr.core.localized_handoff_geometry import (
    SOLVE_REJECTED_COORDINATE_FRAME_MISMATCH,
    SOLVE_REJECTED_LEGACY_SENSOR_MAP_UNAVAILABLE,
)
from minimappr.core.localized_render_handoff import CROSS_REPORTER_DUPLICATE_MERGED_REASON
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import GeoPoint
from minimappr.storage.db import Storage
from tests.localization_solve_manifest_fixtures import (
    MIXED_CLUSTER_ID,
    MIXED_WINDOW_CENTER_TIME_NS,
    PLANAR_NODE_ID,
    POINT_NODE_ID,
    POINT_NODE_POSITION_M,
    classifier_render_manifest,
    mixed_cluster_sensor_positions_m,
    mixed_cluster_solve_localization_payload,
    planar_node_payload,
    point_node_payload,
)

SOURCE_POSITION_M = (44.0, 37.0, 6.0)
POSITION_ERROR_BOUND_M = 0.5
# Packet time sits seconds before the solve window centre so a receipt- or
# packet-start-timed detection cannot pass the timestamp check by accident.
PACKET_TOA_NS = MIXED_WINDOW_CENTER_TIME_NS - 3_000_000_000
ARRIVAL_ORDERS = {
    "planar_first": (PLANAR_NODE_ID, POINT_NODE_ID),
    "point_first": (POINT_NODE_ID, PLANAR_NODE_ID),
}
REPORTER_PAYLOADS = {PLANAR_NODE_ID: planar_node_payload, POINT_NODE_ID: point_node_payload}


async def _started_fusion(tmp_path: Path, *, solver: str) -> tuple[FusionNode, Storage]:
    settings = Settings(
        db_path=tmp_path / "mixed_cluster_handoff.db",
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


async def _ingest_mixed_render(
    fusion: FusionNode,
    *,
    reporter_id: str,
    arrival_order: tuple[str, ...],
    frame_id: str | None = None,
    solve_id: str = "solve-mixed-1",
    sidecar_position_m=SOURCE_POSITION_M,
    manifest_id: str = "render-mixed-1",
) -> None:
    localization = mixed_cluster_solve_localization_payload(
        arrival_order=arrival_order,
        source_position_m=SOURCE_POSITION_M,
        frame_id=site_frame_id_for_coordinate_frame(fusion.coordinate_frame) if frame_id is None else frame_id,
        reporting_node_id=reporter_id,
        solve_id=solve_id,
        sidecar_position_m=sidecar_position_m,
    )
    manifest = classifier_render_manifest(
        node_payload=REPORTER_PAYLOADS[reporter_id](),
        localization=localization,
        toa_ns=PACKET_TOA_NS,
        manifest_id=manifest_id,
    )
    await fusion.ingest_localized_render(load_localized_render_manifest_bundle(manifest_payload=manifest).request)


@pytest.mark.parametrize("solver", ["rust", "python_cartesian"])
@pytest.mark.parametrize("reporter_id", [PLANAR_NODE_ID, POINT_NODE_ID])
@pytest.mark.parametrize("arrival_order_name", sorted(ARRIVAL_ORDERS))
async def test_mixed_cluster_render_stores_the_solve_contract(
    tmp_path: Path, arrival_order_name: str, reporter_id: str, solver: str
) -> None:
    fusion, storage = await _started_fusion(tmp_path, solver=solver)
    # The sidecar's own estimate is deliberately off for python_cartesian, so a
    # pass proves the Python re-solve on the snapshot geometry produced it.
    sidecar_position_m = SOURCE_POSITION_M if solver == "rust" else (0.0, 0.0, 0.0)
    await _ingest_mixed_render(
        fusion, reporter_id=reporter_id, arrival_order=ARRIVAL_ORDERS[arrival_order_name],
        sidecar_position_m=sidecar_position_m,
    )

    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    detection = detections[0]
    feature_summary = detection["feature_summary"]
    assert detection["reporting_modality"] == "localized", feature_summary.get("localization_rejected_reason")
    assert detection["source_node_id"] == reporter_id
    assert math.dist(detection["position_m"], SOURCE_POSITION_M) < POSITION_ERROR_BOUND_M
    assert int(detection["timestamp_ns"]) == MIXED_WINDOW_CENTER_TIME_NS

    solve = feature_summary[LOCALIZATION_SOLVE_FEATURE_KEY]
    assert sorted(solve["contributing_sensor_ids"]) == sorted(mixed_cluster_sensor_positions_m())
    assert solve["contributing_node_ids"] == sorted([PLANAR_NODE_ID, POINT_NODE_ID])
    assert solve["cluster_id"] == MIXED_CLUSTER_ID
    assert solve["reporting_node_id"] == reporter_id
    assert solve["admitted_pair_count"] == 15
    assert solve["cross_node_pair_count"] == 5
    assert len(solve["pair_arrival_lag_seconds_by_sensor_pair"]) == 15
    assert solve["coordinate_frame_id"] == site_frame_id_for_coordinate_frame(fusion.coordinate_frame)
    assert solve["window_center_time_ns"] == int(detection["timestamp_ns"])
    assert solve["solve_id"] == "solve-mixed-1"
    await fusion.stop()
    await storage.close()


async def test_solve_from_another_site_frame_is_reported_omni_at_the_reporter(tmp_path: Path) -> None:
    fusion, storage = await _started_fusion(tmp_path, solver="rust")
    await _ingest_mixed_render(
        fusion, reporter_id=POINT_NODE_ID, arrival_order=ARRIVAL_ORDERS["planar_first"], frame_id="site-elsewhere"
    )

    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    feature_summary = detections[0]["feature_summary"]
    assert detections[0]["reporting_modality"] == "omni"
    assert feature_summary["localization_rejected_reason"] == SOLVE_REJECTED_COORDINATE_FRAME_MISMATCH
    assert tuple(detections[0]["position_m"]) == pytest.approx(POINT_NODE_POSITION_M)
    assert feature_summary[LOCALIZATION_SOLVE_FEATURE_KEY]["stored_position_source"] == "reporting_node"
    await fusion.stop()
    await storage.close()


@pytest.mark.parametrize("solver", ["rust", "python_cartesian"])
async def test_legacy_cluster_render_is_never_solved_on_a_partial_sensor_map(tmp_path: Path, solver: str) -> None:
    """No provenance + pair indices past the reporter's single channel: the old
    path mapped them onto the wrong microphones; now the render is omni."""
    fusion, storage = await _started_fusion(tmp_path, solver=solver)
    localization = mixed_cluster_solve_localization_payload(
        arrival_order=ARRIVAL_ORDERS["planar_first"],
        source_position_m=SOURCE_POSITION_M,
        frame_id=None,
        reporting_node_id=POINT_NODE_ID,
    )
    del localization["solve_provenance"]
    manifest = classifier_render_manifest(node_payload=point_node_payload(), localization=localization, toa_ns=PACKET_TOA_NS)
    request = load_localized_render_manifest_bundle(manifest_payload=manifest).request
    assert request.localization_solve is None and request.event_time_ns == PACKET_TOA_NS

    await fusion.ingest_localized_render(request)

    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    assert detections[0]["reporting_modality"] == "omni"
    assert detections[0]["feature_summary"]["localization_rejected_reason"] == SOLVE_REJECTED_LEGACY_SENSOR_MAP_UNAVAILABLE
    assert fusion._metrics.localization_single_node_python_solved_count == 0
    await fusion.stop()
    await storage.close()


@pytest.mark.parametrize(
    ("second_reporter_offset_m", "expected_detection_count"),
    [((1.0, 0.5, 0.0), 1), ((40.0, 0.0, 0.0), 2)],
    ids=["consistent_positions_merge", "distinct_positions_stay_distinct"],
)
async def test_cross_reporter_duplicates_merge_only_when_spatially_consistent(
    tmp_path: Path, second_reporter_offset_m, expected_detection_count: int
) -> None:
    fusion, storage = await _started_fusion(tmp_path, solver="rust")
    await _ingest_mixed_render(fusion, reporter_id=PLANAR_NODE_ID, arrival_order=ARRIVAL_ORDERS["planar_first"])
    await _ingest_mixed_render(
        fusion,
        reporter_id=POINT_NODE_ID,
        arrival_order=ARRIVAL_ORDERS["planar_first"],
        solve_id="solve-mixed-2",
        manifest_id="render-mixed-2",
        sidecar_position_m=tuple(np.add(SOURCE_POSITION_M, second_reporter_offset_m)),
    )

    detections = await storage.list_detections(limit=10)
    assert len(detections) == expected_detection_count
    merged_count = fusion._metrics.classification_drops_by_reason.get(CROSS_REPORTER_DUPLICATE_MERGED_REASON, 0)
    assert merged_count == 2 - expected_detection_count
    if expected_detection_count == 1:
        assert detections[0]["source_node_id"] == PLANAR_NODE_ID
    await fusion.stop()
    await storage.close()
