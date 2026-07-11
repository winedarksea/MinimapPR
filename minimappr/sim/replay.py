"""Calibration bundle replay harness.

Rebuilds a FusionNode from a bundle's site frame and streams the bundle's
per-node raw audio back through ingest, then scores the resulting detections
against operator ground truth. Mirrors the threshold/report style of
`minimappr/sim/soak.py`: `evaluate_bundle` returns a report whose `errors`
list is empty on pass.
"""

from __future__ import annotations

import asyncio
import heapq
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from minimappr.calibration.bundle import CalibrationBundle
from minimappr.classifiers.factory import create_classifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
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
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64

DEFAULT_EXPECTATIONS: dict[str, Any] = {
    "schema_version": 1,
    "runtime": {"classifier_backend": "yamnet", "settings_overrides": {}},
    "classification": {
        "min_label_accuracy": 0.5,
        "match": "category",
        "per_label": {},
    },
    "localization": {
        "min_event_recall": 0.5,
        "max_median_position_error_m": 20.0,
        "max_p90_position_error_m": 40.0,
        "far_field_distance_m": 120.0,
        "far_field": {"max_median_bearing_error_deg": 10.0},
        "per_algorithm": {},
    },
}

MATCH_SLOP_NS = 2_000_000_000
"""Detections within ±2 s of a ground-truth window still count as matches."""


@dataclass
class EventResult:
    event_id: str
    label: str
    matched: bool = False
    label_correct: bool | None = None
    position_error_m: float | None = None
    bearing_error_deg: float | None = None
    far_field: bool = False
    algorithm: str | None = None


@dataclass
class CalibrationReport:
    errors: list[str] = field(default_factory=list)
    event_results: list[EventResult] = field(default_factory=list)
    event_recall: float = 0.0
    label_accuracy: float = 0.0
    median_position_error_m: float | None = None
    p90_position_error_m: float | None = None
    median_bearing_error_deg: float | None = None

    @property
    def passed(self) -> bool:
        return not self.errors


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
        model_chain_config_path=tmp_path / "missing_model_chain.json",
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
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
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
) -> int:
    """Stream every node's audio through fusion.ingest, interleaved by timestamp."""
    environment = bundle.manifest.get("environment", {})
    environment_in = EnvironmentSampleIn(
        temperature_c=environment.get("temperature_c"),
        humidity_fraction=environment.get("humidity_fraction"),
        source="calibration_bundle",
    )

    # heap entries: (frame_start_ns, tiebreak, node_index, frame)
    heap: list[tuple[int, int, int, np.ndarray]] = []
    node_specs: list[NodeSpec] = []
    sequences: list[int] = []
    tiebreak = 0
    for node_index, node in enumerate(bundle.manifest.get("nodes", [])):
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
        node = bundle.manifest["nodes"][node_index]
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


def _bearing_deg(origin: np.ndarray, target: np.ndarray) -> float:
    delta = target - origin
    return float(np.degrees(np.arctan2(delta[0], delta[1])) % 360.0)


def _angular_difference_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _detection_algorithm(detection: dict) -> str | None:
    feature_summary = detection.get("feature_summary") or {}
    return (
        feature_summary.get("resolved_algorithm")
        or feature_summary.get("localization_method")
        or None
    )


def evaluate_bundle(
    detections: list[dict],
    bundle: CalibrationBundle,
    expectations: dict[str, Any] | None = None,
) -> CalibrationReport:
    """Score replayed detections against the bundle's ground truth.

    Position error uses ‖detection.position_m − geo_to_local(event)‖; events
    farther than `far_field_distance_m` from the array centroid switch to
    bearing error (small tetra arrays are DOA sensors beyond their range
    resolution — see tests/test_localization_distance_matrix.py).
    """
    expectations = expectations or bundle.expectations or DEFAULT_EXPECTATIONS
    classification_cfg = {**DEFAULT_EXPECTATIONS["classification"], **expectations.get("classification", {})}
    localization_cfg = {**DEFAULT_EXPECTATIONS["localization"], **expectations.get("localization", {})}

    site = bundle.manifest["site"]
    frame = LocalCoordinateFrame(
        origin=GeoPoint(
            lat=site["origin"]["lat"],
            lon=site["origin"]["lon"],
            alt_m=site["origin"].get("alt_m", 0.0),
        ),
        mode=site.get("coordinate_mode", "flat"),
    )
    node_positions = [
        np.asarray(node["position_m"], dtype=np.float64)
        for node in bundle.manifest.get("nodes", [])
        if node.get("position_m")
    ]
    centroid = (
        np.mean(node_positions, axis=0) if node_positions else np.zeros(3, dtype=np.float64)
    )

    report = CalibrationReport()
    far_field_distance_m = float(localization_cfg["far_field_distance_m"])
    match_mode = classification_cfg.get("match", "category")

    position_errors: list[float] = []
    bearing_errors: list[float] = []
    errors_by_algorithm: dict[str, list[float]] = {}

    for event in bundle.events:
        geometry = event["geometry"]["position_geo"]
        gt_local = np.asarray(
            frame.geo_to_local(
                GeoPoint(
                    lat=geometry["lat"],
                    lon=geometry["lon"],
                    alt_m=geometry.get("alt_m") or 0.0,
                )
            ),
            dtype=np.float64,
        )
        result = EventResult(event_id=event["event_id"], label=event["label"])
        result.far_field = float(np.linalg.norm(gt_local - centroid)) > far_field_distance_m

        window_start = event["start_ns"] - MATCH_SLOP_NS
        window_end = event["end_ns"] + MATCH_SLOP_NS
        candidates = [
            det
            for det in detections
            if window_start <= int(det.get("timestamp_ns") or 0) <= window_end
        ]
        if candidates:
            result.matched = True
            if match_mode == "exact":
                result.label_correct = any(
                    det.get("label") == event["label"] for det in candidates
                )
            else:
                event_category = event.get("label_category") or "unknown"
                result.label_correct = any(
                    det.get("label_category") == event_category for det in candidates
                )

            best_error = None
            best_det = None
            for det in candidates:
                position = det.get("position_m")
                if not position:
                    continue
                error = float(np.linalg.norm(np.asarray(position, dtype=np.float64) - gt_local))
                if best_error is None or error < best_error:
                    best_error = error
                    best_det = det
            if best_det is not None:
                result.algorithm = _detection_algorithm(best_det)
                det_position = np.asarray(best_det["position_m"], dtype=np.float64)
                if result.far_field:
                    result.bearing_error_deg = _angular_difference_deg(
                        _bearing_deg(centroid, det_position),
                        _bearing_deg(centroid, gt_local),
                    )
                    bearing_errors.append(result.bearing_error_deg)
                else:
                    result.position_error_m = best_error
                    position_errors.append(best_error)
                    if result.algorithm:
                        errors_by_algorithm.setdefault(result.algorithm, []).append(best_error)
        report.event_results.append(result)

    n_events = len(report.event_results)
    if n_events == 0:
        report.errors.append("bundle contains no ground-truth events")
        return report

    matched = [r for r in report.event_results if r.matched]
    report.event_recall = len(matched) / n_events
    labelled = [r for r in matched if r.label_correct is not None]
    report.label_accuracy = (
        sum(1 for r in labelled if r.label_correct) / len(labelled) if labelled else 0.0
    )
    if position_errors:
        report.median_position_error_m = float(statistics.median(position_errors))
        report.p90_position_error_m = float(np.percentile(position_errors, 90))
    if bearing_errors:
        report.median_bearing_error_deg = float(statistics.median(bearing_errors))

    if report.event_recall < float(localization_cfg["min_event_recall"]):
        report.errors.append(
            f"event recall {report.event_recall:.2f} below "
            f"{localization_cfg['min_event_recall']}"
        )
    if labelled and report.label_accuracy < float(classification_cfg["min_label_accuracy"]):
        report.errors.append(
            f"label accuracy {report.label_accuracy:.2f} below "
            f"{classification_cfg['min_label_accuracy']}"
        )
    per_label = classification_cfg.get("per_label") or {}
    for label, cfg in per_label.items():
        label_results = [
            r for r in labelled if r.label == label
        ]
        if not label_results:
            continue
        accuracy = sum(1 for r in label_results if r.label_correct) / len(label_results)
        if accuracy < float(cfg.get("min_label_accuracy", 0.0)):
            report.errors.append(
                f"label '{label}' accuracy {accuracy:.2f} below {cfg['min_label_accuracy']}"
            )
    if report.median_position_error_m is not None:
        if report.median_position_error_m > float(localization_cfg["max_median_position_error_m"]):
            report.errors.append(
                f"median position error {report.median_position_error_m:.1f} m above "
                f"{localization_cfg['max_median_position_error_m']} m"
            )
        if report.p90_position_error_m > float(localization_cfg["max_p90_position_error_m"]):
            report.errors.append(
                f"p90 position error {report.p90_position_error_m:.1f} m above "
                f"{localization_cfg['max_p90_position_error_m']} m"
            )
    far_field_cfg = localization_cfg.get("far_field") or {}
    if report.median_bearing_error_deg is not None and "max_median_bearing_error_deg" in far_field_cfg:
        if report.median_bearing_error_deg > float(far_field_cfg["max_median_bearing_error_deg"]):
            report.errors.append(
                f"median far-field bearing error {report.median_bearing_error_deg:.1f}° above "
                f"{far_field_cfg['max_median_bearing_error_deg']}°"
            )
    per_algorithm = localization_cfg.get("per_algorithm") or {}
    for algorithm, cfg in per_algorithm.items():
        algorithm_errors = errors_by_algorithm.get(algorithm)
        if not algorithm_errors:
            continue
        median_error = float(statistics.median(algorithm_errors))
        limit = float(cfg.get("max_median_position_error_m", float("inf")))
        if median_error > limit:
            report.errors.append(
                f"{algorithm} median position error {median_error:.1f} m above {limit} m"
            )
    return report
