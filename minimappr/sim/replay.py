"""Calibration bundle replay harness.

Rebuilds a FusionNode from a bundle's site frame and streams the bundle's
per-node raw audio back through ingest, then scores the resulting detections
against operator ground truth (scoring lives in
`minimappr/sim/replay_scoring.py` + `replay_entity_metrics.py`). Mirrors the threshold/report style of
`minimappr/sim/soak.py`: `evaluate_bundle` returns a report whose `errors`
list is empty on pass.
"""

from __future__ import annotations

import asyncio
import heapq
from pathlib import Path
from typing import Any

import numpy as np

from minimappr.calibration.bundle import CalibrationBundle
from minimappr.classifiers.factory import create_classifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.timeline_discipline import TimelineDisciplinePolicy
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization_dispatch import build_localizer_from_settings
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.core.environment import StaticEnvironmentProvider
from minimappr.models import (
    EnvironmentSampleIn,
    GeoPoint,
    IngestFrameRequest,
    NodeSpec,
    NodeType,
)
from minimappr.sim.replay_scoring import (  # re-exported: replay API surface
    DEFAULT_EXPECTATIONS,
    MATCH_SLOP_NS,
    CalibrationReport,
    EventResult,
    evaluate_bundle,
)
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64

__all__ = [
    "DEFAULT_EXPECTATIONS",
    "MATCH_SLOP_NS",
    "CalibrationReport",
    "EventResult",
    "build_fusion_for_bundle",
    "evaluate_bundle",
    "replay_bundle",
]

async def build_fusion_for_bundle(
    bundle: CalibrationBundle,
    tmp_path: Path,
    *,
    overrides: dict[str, Any] | None = None,
    classifier: Any | None = None,
) -> tuple[FusionNode, Storage, Settings]:
    """Construct a started FusionNode whose frame matches the bundle's site."""
    site = bundle.manifest["site"]
    environment = bundle.manifest.get("environment", {})
    settings_kwargs: dict[str, Any] = dict(
        db_path=tmp_path / "replay.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        fusion_worker_count=1,
        fusion_localization_queue_size=64,
        fusion_classification_queue_size=64,
        fusion_rules_queue_size=64,
        site_origin_lat=site["origin"]["lat"],
        site_origin_lon=site["origin"]["lon"],
        site_origin_alt_m=site["origin"].get("alt_m", 0.0),
        coordinate_mode=site.get("coordinate_mode", "flat"),
        default_temperature_c=environment.get("temperature_c", 20.0),
        default_humidity=environment.get("humidity_fraction", 0.5),
    )
    expectations = bundle.expectations or DEFAULT_EXPECTATIONS
    runtime = expectations.get("runtime", {})
    settings_kwargs.update(runtime.get("settings_overrides", {}))
    if overrides:
        settings_kwargs.update(overrides)
    settings = Settings(**settings_kwargs)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    Path(settings.snippet_dir).mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()
    coordinate_frame = LocalCoordinateFrame(
        origin=GeoPoint(
            lat=settings.site_origin_lat,
            lon=settings.site_origin_lon,
            alt_m=settings.site_origin_alt_m,
        ),
        mode=settings.coordinate_mode,
    )
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(
            max_duration_seconds=settings.max_sensor_buffer_seconds,
            timeline_discipline_policy=TimelineDisciplinePolicy.from_settings(settings),
        ),
        localizer=build_localizer_from_settings(settings),
        classifier=classifier if classifier is not None else create_classifier(settings),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=coordinate_frame,
        zone_matcher=ZoneMatcher(storage=storage),
        environment_provider=StaticEnvironmentProvider(
            temperature_c=settings.default_temperature_c,
            humidity_fraction=settings.default_humidity,
        ),
    )
    await fusion.start()
    return fusion, storage, settings


def _node_spec_from_manifest(node: dict) -> NodeSpec:
    offsets = [tuple(offset) for offset in (node.get("sensor_offsets_m") or [])]
    if not offsets:
        offsets = [(0.0, 0.0, 0.0)] * len(node["channel_sensor_ids"])
    node_type = NodeType.SIRITH_TETRA if len(offsets) == 4 else NodeType.POINT
    position_geo = node.get("position_geo")
    return NodeSpec(
        id=node["node_id"],
        node_type=node_type,
        position_m=tuple(node["position_m"]) if node.get("position_m") else None,
        position_geo=GeoPoint(**position_geo) if position_geo else None,
        sensor_offsets_m=offsets,
        orientation=node.get("orientation") or {},
        capabilities=["audio", "array_localization"],
        metadata={},
    )


def _split_channels_into_frames(
    channels_first: np.ndarray,
    *,
    sample_rate_hz: int,
    start_time_ns: int,
    frame_samples: int,
) -> list[tuple[int, np.ndarray]]:
    """Split channels-first audio into fixed-size frames, zero-padding the tail."""
    _, total_samples = channels_first.shape
    frame_duration_ns = int(round((frame_samples / sample_rate_hz) * 1_000_000_000))
    frames: list[tuple[int, np.ndarray]] = []
    for frame_index, start_sample in enumerate(range(0, total_samples, frame_samples)):
        frame = channels_first[:, start_sample : start_sample + frame_samples].astype(
            np.float32, copy=True
        )
        if frame.shape[1] < frame_samples:
            padding = np.zeros((frame.shape[0], frame_samples - frame.shape[1]), dtype=np.float32)
            frame = np.concatenate([frame, padding], axis=1)
        frames.append((start_time_ns + frame_index * frame_duration_ns, frame))
    return frames


async def replay_bundle(
    fusion: FusionNode,
    bundle: CalibrationBundle,
    *,
    frame_samples: int = 1024,
    include_reference_nodes: bool | None = None,
) -> int:
    """Stream every node's audio through fusion.ingest, interleaved by timestamp.

    Nodes referenced as a ground-truth source sit at the sound source, so they
    are excluded from localization input by default; opt back in via the
    bundle's expectations runtime `replay_reference_nodes` or the keyword.
    """
    if include_reference_nodes is None:
        runtime = (bundle.expectations or DEFAULT_EXPECTATIONS).get("runtime", {})
        include_reference_nodes = bool(runtime.get("replay_reference_nodes", False))
    reference_node_ids = {
        event["source_node_id"]
        for event in bundle.events
        if event.get("source_node_id")
    }
    manifest_nodes = bundle.manifest.get("nodes", [])
    excluded = set() if include_reference_nodes else reference_node_ids
    if excluded and all(node.get("node_id") in excluded for node in manifest_nodes):
        # Excluding every node would make the bundle unreplayable.
        excluded = set()

    environment = bundle.manifest.get("environment", {})
    environment_in = EnvironmentSampleIn(
        temperature_c=environment.get("temperature_c"),
        humidity_fraction=environment.get("humidity_fraction"),
        source="calibration_bundle",
    )

    replay_nodes = [
        node for node in manifest_nodes if node.get("node_id") not in excluded
    ]

    # heap entries: (frame_start_ns, tiebreak, node_index, frame)
    heap: list[tuple[int, int, int, np.ndarray]] = []
    node_specs: list[NodeSpec] = []
    sequences: list[int] = []
    tiebreak = 0
    for node_index, node in enumerate(replay_nodes):
        node_specs.append(_node_spec_from_manifest(node))
        sequences.append(0)
        channels, sample_rate_hz = bundle.node_audio(node["node_id"])
        start_time_ns = int(node.get("audio_start_time_ns") or 0)
        for frame_start_ns, frame in _split_channels_into_frames(
            channels,
            sample_rate_hz=sample_rate_hz,
            start_time_ns=start_time_ns,
            frame_samples=frame_samples,
        ):
            heapq.heappush(heap, (frame_start_ns, tiebreak, node_index, frame))
            tiebreak += 1
        # sample rate rides on the node manifest for the frame payloads below
        node["_replay_sample_rate_hz"] = sample_rate_hz

    accepted = 0
    while heap:
        frame_start_ns, _, node_index, frame = heapq.heappop(heap)
        node = replay_nodes[node_index]
        sequences[node_index] += 1
        response = await fusion.ingest(
            IngestFrameRequest(
                node=node_specs[node_index],
                frame={
                    "start_time_ns": frame_start_ns,
                    "sample_rate_hz": node["_replay_sample_rate_hz"],
                    "channels": int(frame.shape[0]),
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(frame),
                    "sequence": sequences[node_index],
                },
                environment=environment_in,
            )
        )
        accepted += int(response.accepted)
    return accepted
