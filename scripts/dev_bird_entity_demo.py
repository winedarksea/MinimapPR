"""Dev demo: run the server with bird-entity reporting on and feed it birds.

    python scripts/dev_bird_entity_demo.py [--port 8765] [--data-dir /tmp/entity-demo]

Starts the real app (python ingest, no sidecar) in-process, registers three
nodes, then drives synthetic detections through the production seams —
``EntityReportingCoordinator.decide`` (absorption) and
``FusionNode._process_rules_and_delivery`` (observe, WS events, grace window) —
so the Leptos "Heard nearby" tray and the spatial-only map can be checked in a
browser. Nothing here is used by the server itself; data goes to --data-dir.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NODE_POSITIONS_M = {"node-a": (0.0, 0.0, 0.0), "node-b": (20.0, 0.0, 0.0), "node-c": (10.0, 18.0, 0.0)}
ROBIN_PERCH_M = (6.0, 12.0, 2.0)
TANAGER_PROJECTED_M = (55.0, 35.0, 0.0)
FIX_COVARIANCE_M2 = [[2.0, 0.4, 0.0], [0.4, 1.2, 0.0], [0.0, 0.0, 1.0]]
BOUT_PERIOD_SECONDS = 6.0


def _configure_environment(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "MINIMAPPR_DB_PATH": str(data_dir / "demo.db"),
        "MINIMAPPR_SNIPPET_DIR": str(data_dir / "snippets"),
        "MINIMAPPR_LARGE_ARTIFACT_DIR": str(data_dir / "artifacts"),
        "MINIMAPPR_CONFIG_PATH": str(data_dir / "config.yml"),
        "MINIMAPPR_INGEST_SIDECAR_ENABLED": "false",
        "MINIMAPPR_INGEST_BACKEND": "python",
        "MINIMAPPR_ENTITY_REPORTING_ENABLED": "true",
        "MINIMAPPR_ENTITY_NOTIFICATION_GRACE_SECONDS": "2",
    })


async def _register_nodes(state) -> None:
    from minimappr.models import NodeSpec, NodeType

    now_ns = time.time_ns()
    for node_id, position in NODE_POSITIONS_M.items():
        spec = NodeSpec(id=node_id, node_type=NodeType.POINT, position_m=position,
                        sensor_offsets_m=[(0.0, 0.0, 0.0)], capabilities=["audio"])
        await state.registry.upsert(spec, last_seen_ns=now_ns)
        await state.storage.upsert_node(spec, last_seen_ns=now_ns)


def _detection(state, det_id: str, *, label: str, node_id: str, modality: str, position_m, features=None):
    from minimappr.models import DetectionEvent, spatial_display_mode_for_detection

    now_ns = time.time_ns()
    features = features or {}
    return DetectionEvent(
        id=det_id, timestamp_ns=now_ns, source_node_id=node_id, reporting_modality=modality,
        spatial_display_mode=spatial_display_mode_for_detection(reporting_modality=modality, feature_summary=features),
        position_m=position_m, position_geo=state.coordinate_frame.local_to_geo(position_m),
        position_covariance_m2=None if modality == "omni" else FIX_COVARIANCE_M2,
        confidence=0.8, gdop=1.2, label=label, label_category="wildlife", label_confidence=0.82,
        reference_sensor=f"{node_id}:m0", feature_summary=features,
    )


async def _surface(state, detection, *, track=None, entity_id=None) -> None:
    from minimappr.core.fusion_node import DetectionProduct

    await state.storage.insert_detection(detection, snippet_path=None, snippet_expires_ns=None)
    await state.fusion_node._process_rules_and_delivery(DetectionProduct(
        detection=detection, track=track, suppressed_by_zone=False, suppression_reasons=[],
        pipeline_item_id=detection.id, event_time_ns=detection.timestamp_ns, entity_id=entity_id,
    ))


async def _drive_birds(state) -> None:
    coordinator = state.fusion_node.entity_reporting
    await _register_nodes(state)
    bout = 0
    while True:
        bout += 1
        now_ns = time.time_ns()
        robin_track = await state.tracker.update(
            timestamp_ns=now_ns, position_m=ROBIN_PERCH_M, label="american robin", confidence=0.82,
            label_category="wildlife", measurement_covariance_m2=FIX_COVARIANCE_M2, source_node_id="node-a",
        )
        robin_track.position_geo = state.coordinate_frame.local_to_geo(robin_track.position_m)
        await _surface(state, _detection(state, f"det-robin-{bout}", label="american robin", node_id="node-a",
                                         modality="localized", position_m=ROBIN_PERCH_M), track=robin_track)
        await _surface(state, _detection(state, f"det-tanager-{bout}", label="scarlet tanager", node_id="node-b",
                                         modality="localized", position_m=TANAGER_PROJECTED_M,
                                         features={"localization_range_projection_mode": "range_asymptotic"}))
        await _surface(state, _detection(state, f"det-wren-{bout}", label="house wren", node_id="node-c",
                                         modality="omni", position_m=NODE_POSITIONS_M["node-c"]))
        if bout == 1:
            await _surface(state, _detection(state, "det-chickadee", label="black-capped chickadee",
                                             node_id="node-a", modality="omni", position_m=NODE_POSITIONS_M["node-a"]))
        await asyncio.sleep(BOUT_PERIOD_SECONDS / 2)
        # A robin bout that fails to localize: absorbed into the mapped entity (pulse, no new card).
        await coordinator.decide(
            event_time_ns=time.time_ns(), source_node_id="node-b", label="american robin",
            reporting_modality="omni", spatial_display_mode="node_only",
            branch_details={"label": "american robin", "confidence": 0.7},
            localization_position_m=None, below_display_threshold=False,
        )
        await asyncio.sleep(BOUT_PERIOD_SECONDS / 2)


async def main_async(port: int) -> None:
    import uvicorn

    from minimappr.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    serve_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.1)
    print(f"bird-entity demo on http://127.0.0.1:{port}/", flush=True)
    await asyncio.gather(serve_task, _drive_birds(app.state))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/minimappr-entity-demo"))
    args = parser.parse_args()
    _configure_environment(args.data_dir)
    asyncio.run(main_async(args.port))


if __name__ == "__main__":
    main()
