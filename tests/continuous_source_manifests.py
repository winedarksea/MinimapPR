"""Sidecar manifests for continuous-source scenes (E6).

Each scene solve becomes the manifests the sidecar publishes for it, built per
``docs/localization_solve_contract.md`` on the planar + point cluster of
``tests/localization_solve_manifest_fixtures.py``:

- event: a ``localization_result`` with no render;
- covering render: a ``localization_result`` paired to the render through
  ``derived_handle``, then the ``classifier_render`` with the classifier's
  label (authoritative, as the sidecar classifier helper sends it).

Both repeat the solve's ``solve_id`` and window centre. Pair lags are exact for
the solve's *measured* position, so the Python re-solve and the sidecar
estimate agree on it and the fix error stays the scene's.
"""

from __future__ import annotations

from typing import Any

from minimappr.core.birdnet_label_categories import BIRDNET_V24_MODEL_NAME
from minimappr.sim.continuous_source_scenes import (
    DELIVERY_COVERING_CLASSIFIER_RENDER,
    DELIVERY_EVENT_LOCALIZATION_RESULT,
    DELIVERY_RENDER_LOCALIZATION_RESULT,
    SOURCE_KIND_BIRD,
    ContinuousSourceScene,
    SceneDelivery,
    SceneSolve,
)
from tests.localization_solve_manifest_fixtures import (
    PLANAR_NODE_ID,
    POINT_NODE_ID,
    classifier_render_manifest,
    mixed_cluster_solve_localization_payload,
    planar_node_payload,
    point_node_payload,
)

NANOSECONDS_PER_SECOND = 1_000_000_000
SCENE_START_NS = 1_764_000_000_000_000_000
SCENE_REPORTER_IDS = (PLANAR_NODE_ID, POINT_NODE_ID)
CLUSTER_ARRIVAL_ORDER = (PLANAR_NODE_ID, POINT_NODE_ID)
NODE_PAYLOAD_BUILDERS = {PLANAR_NODE_ID: planar_node_payload, POINT_NODE_ID: point_node_payload}
# Packet start sits before the window centre, as for a real upload.
PACKET_LEAD_BEFORE_WINDOW_CENTRE_NS = 40_000_000
RENDER_CLASSIFIER_CONFIDENCE = 0.9
EVENT_MANIFEST_ONLY_KEYS = ("classifier_render", "raw_render_bytes", "derived_handle")
RENDER_LOCALIZATION_RESULT_DROPPED_KEYS = ("classifier_render", "raw_render_bytes")


def window_centre_ns(solve: SceneSolve, scene_start_ns: int = SCENE_START_NS) -> int:
    return scene_start_ns + int(round(solve.solve_offset_seconds * NANOSECONDS_PER_SECOND))


def solve_localization_payload(
    scene: ContinuousSourceScene, solve: SceneSolve, *, frame_id: str, scene_start_ns: int = SCENE_START_NS
) -> dict[str, Any]:
    payload = mixed_cluster_solve_localization_payload(
        arrival_order=CLUSTER_ARRIVAL_ORDER,
        source_position_m=solve.measured_position_m,
        frame_id=frame_id,
        reporting_node_id=scene.sources[solve.source_id].reporting_node_id,
        solve_id=solve.solve_id,
        sidecar_position_m=solve.measured_position_m,
    )
    payload["solve_provenance"]["window_center_time_ns"] = window_centre_ns(solve, scene_start_ns)
    variance_m2 = solve.position_std_m**2
    payload["position_covariance_m2"] = [
        [variance_m2 if row == column else 0.0 for column in range(3)] for row in range(3)
    ]
    return payload


def _render_classification(scene: ContinuousSourceScene, solve: SceneSolve) -> dict[str, Any]:
    source = scene.sources[solve.source_id]
    # BirdNET provenance is what files a species label under wildlife.
    features = {"model": BIRDNET_V24_MODEL_NAME} if source.kind == SOURCE_KIND_BIRD else {}
    return {
        "label": source.label,
        "label_confidence": RENDER_CLASSIFIER_CONFIDENCE,
        "scores": {source.label: RENDER_CLASSIFIER_CONFIDENCE},
        "features": features,
        "classifier_source_node": source.reporting_node_id,
    }


def delivery_manifest(
    scene: ContinuousSourceScene,
    delivery: SceneDelivery,
    *,
    delivery_index: int,
    frame_id: str,
    scene_start_ns: int = SCENE_START_NS,
) -> dict[str, Any]:
    solve = scene.solve(delivery.solve_id)
    reporting_node_id = scene.sources[solve.source_id].reporting_node_id
    manifest = classifier_render_manifest(
        node_payload=NODE_PAYLOAD_BUILDERS[reporting_node_id](),
        localization=solve_localization_payload(scene, solve, frame_id=frame_id, scene_start_ns=scene_start_ns),
        toa_ns=window_centre_ns(solve, scene_start_ns) - PACKET_LEAD_BEFORE_WINDOW_CENTRE_NS,
        manifest_id=f"{scene.name}-{delivery_index:04d}-{delivery.kind}",
        audio_seed=delivery_index,
    )
    if delivery.kind == DELIVERY_EVENT_LOCALIZATION_RESULT:
        manifest["manifest_type"] = "localization_result"
        for key in EVENT_MANIFEST_ONLY_KEYS:
            manifest.pop(key)
    elif delivery.kind == DELIVERY_RENDER_LOCALIZATION_RESULT:
        manifest["manifest_type"] = "localization_result"
        for key in RENDER_LOCALIZATION_RESULT_DROPPED_KEYS:
            manifest.pop(key)
    elif delivery.kind == DELIVERY_COVERING_CLASSIFIER_RENDER:
        manifest["birdnet"] = _render_classification(scene, solve)
    else:
        raise ValueError(f"unknown delivery kind {delivery.kind!r}")
    return manifest
