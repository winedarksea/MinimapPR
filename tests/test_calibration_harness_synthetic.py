"""End-to-end synthetic validation of the calibration replay harness.

Ensures the harness is never dead code even when no field bundles exist:
synthesizes a two-node capture, builds a real bundle zip via the same
pipeline + bundle writers used in production, replays it through a full
FusionNode (stub classifier — no TF/BirdNET needed), and evaluates it.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pytest

from minimappr.calibration.bundle import (
    build_ground_truth_payload,
    load_bundle,
    write_bundle_zip,
)
from minimappr.calibration.pipeline import CalibrationPipeline
from minimappr.classifiers.base import AudioClassifier
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.capture_session import CaptureSessionRecord, CaptureState
from minimappr.core.environment import StaticEnvironmentProvider
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.models import ClassificationResult, GeoPoint
from minimappr.sim.replay import build_fusion_for_bundle, evaluate_bundle, replay_bundle
from tests.helpers import synthesize_multinode_windows

SAMPLE_RATE = 16_000
ORIGIN = GeoPoint(lat=45.0, lon=-93.0, alt_m=250.0)
SOURCE_POSITION_M = (0.6, 0.4, 0.2)
START_NS = 1_739_920_000_000_000_000

NODE_ORIGINS_M = {
    "node-a": (0.0, 0.0, 0.0),
    "node-b": (12.0, 4.0, 0.0),
}

EXPECTATIONS = {
    "schema_version": 1,
    "runtime": {"classifier_backend": "stub", "settings_overrides": {}},
    "classification": {"min_label_accuracy": 0.5, "match": "exact"},
    "localization": {
        # Loop-validation thresholds: a single 5 cm tetra aperture is a DOA
        # sensor, so absolute range error can be meters even when the harness
        # works perfectly. Field bundles carry tighter expectations.
        "min_event_recall": 0.5,
        "max_median_position_error_m": 50.0,
        "max_p90_position_error_m": 75.0,
        "far_field_distance_m": 120.0,
    },
}


class _StubToneClassifier(AudioClassifier):
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        return ClassificationResult(
            label="calib_tone", confidence=0.9, scores={"calib_tone": 0.9}, features={}
        )


class _StubStorage:
    def __init__(self, frame: LocalCoordinateFrame) -> None:
        self._frame = frame

    async def get_node_by_id(self, node_id: str) -> dict | None:
        origin = NODE_ORIGINS_M.get(node_id)
        if origin is None:
            return None
        geo = self._frame.local_to_geo(origin)
        return {
            "id": node_id,
            "position_geo": {"lat": geo.lat, "lon": geo.lon, "alt_m": geo.alt_m},
            "position_m": list(origin),
            "sensor_offsets_m": None,  # filled below per node via tetra offsets
            "orientation": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
        }

    async def list_detections(self, limit: int = 100, **_kw):
        return []

    async def insert_large_artifact(self, **kwargs) -> str:
        return "lar-synthetic"


async def _build_synthetic_bundle(tmp_path: Path) -> Path:
    from tests.helpers import SIRITH_TETRA_SENSOR_OFFSETS_M

    frame = LocalCoordinateFrame(origin=ORIGIN, mode="flat")
    storage = _StubStorage(frame)

    # Patch tetra offsets into the stub node rows.
    original_get = storage.get_node_by_id

    async def _get_node(node_id: str):
        row = await original_get(node_id)
        if row is not None:
            row["sensor_offsets_m"] = [list(o) for o in SIRITH_TETRA_SENSOR_OFFSETS_M]
        return row

    storage.get_node_by_id = _get_node  # type: ignore[method-assign]

    buffer = MultiSensorBuffer(max_duration_seconds=8.0)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    excitation = (np.sin(2 * np.pi * 900.0 * t) * 0.4).astype(np.float32)
    _, sensor_windows, _ = synthesize_multinode_windows(
        excitation,
        SAMPLE_RATE,
        source_position_m=SOURCE_POSITION_M,
        node_origins_m=NODE_ORIGINS_M,
        tetra_node_ids=tuple(NODE_ORIGINS_M),
    )

    node_channel_map = {
        node_id: [f"{node_id}:ch{i}" for i in range(4)] for node_id in NODE_ORIGINS_M
    }
    all_sensor_ids = [sid for sids in node_channel_map.values() for sid in sids]
    capture_buffer = await buffer.start_capture(
        "synthetic-1", all_sensor_ids, max_duration_seconds=8.0
    )
    n_samples = next(iter(sensor_windows.values())).size
    end_ns = START_NS + int(n_samples / SAMPLE_RATE * 1_000_000_000)
    for sensor_id, samples in sensor_windows.items():
        capture_buffer.append(
            sensor_id=sensor_id,
            sample_rate_hz=SAMPLE_RATE,
            start_time_ns=START_NS,
            samples=samples,
        )

    work_dir = tmp_path / "work" / "synthetic-1"
    work_dir.mkdir(parents=True)
    record = CaptureSessionRecord(
        session_id="synthetic-1",
        state=CaptureState.PROCESSING,
        stream_key="calibration",
        range_lease_id=None,
        start_time_ns=START_NS,
        end_time_ns=end_ns,
        first_frame_pts_ns=None,
        work_dir=work_dir,
        video_path=None,
        ambix_path=None,
        iamf_path=None,
        youtube_path=None,
        error=None,
        use_python_ingest=True,
        capture_audio_buffer=capture_buffer,
        capture_kind="calibration",
        node_channel_map=node_channel_map,
    )
    pipeline = CalibrationPipeline(
        storage=storage,
        coordinate_frame=frame,
        environment_provider=StaticEnvironmentProvider(temperature_c=20.0, humidity_fraction=0.5),
        artifact_dir=tmp_path / "artifacts",
    )
    await pipeline.run(record)

    source_geo = frame.local_to_geo(SOURCE_POSITION_M)
    ground_truth = build_ground_truth_payload(
        [
            {
                "id": "cgt-synth-1",
                "session_id": "synthetic-1",
                "label": "calib_tone",
                "label_category": "unknown",
                "geometry_kind": "static",
                "lat": source_geo.lat,
                "lon": source_geo.lon,
                "alt_m": source_geo.alt_m,
                "start_ns": START_NS,
                "end_ns": end_ns,
                "notes": "synthetic 900 Hz tone",
            }
        ]
    )
    return write_bundle_zip(
        record.work_dir,
        ground_truth,
        tmp_path / "calibration_synthetic-1.zip",
        expectations=EXPECTATIONS,
    )


@pytest.mark.asyncio
async def test_synthetic_bundle_full_replay_loop(tmp_path: Path) -> None:
    bundle_path = await _build_synthetic_bundle(tmp_path)
    bundle = load_bundle(bundle_path)
    assert len(bundle.manifest["nodes"]) == 2

    fusion, storage, settings = await build_fusion_for_bundle(
        bundle,
        tmp_path / "replay",
        overrides={"trigger_rms": 0.006, "classification_window_seconds": 1.0},
        classifier=_StubToneClassifier(),
    )
    try:
        accepted = await replay_bundle(fusion, bundle)
        assert accepted > 0

        detections: list[dict] = []
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            detections = await storage.list_detections(limit=10_000)
            if detections:
                break
            await asyncio.sleep(0.2)
    finally:
        await fusion.stop()
        await storage.close()

    assert detections, "replay produced no detections"
    report = evaluate_bundle(detections, bundle, EXPECTATIONS)
    assert report.event_recall == 1.0
    assert report.label_accuracy == 1.0
    assert report.median_position_error_m is not None
    assert report.passed, "\n".join(report.errors)


def test_evaluate_bundle_flags_failures(tmp_path: Path) -> None:
    """Threshold violations must surface as errors (no silent passes)."""
    bundle_path = asyncio.run(_build_synthetic_bundle(tmp_path))
    bundle = load_bundle(bundle_path)
    event = bundle.events[0]
    # A detection far from the truth with the wrong label.
    detections = [
        {
            "timestamp_ns": event["start_ns"] + 1,
            "label": "wrong",
            "label_category": "unknown",
            "position_m": [500.0, 500.0, 0.0],
            "feature_summary": {"localization_method": "gcc_phat"},
        }
    ]
    strict = json.loads(json.dumps(EXPECTATIONS))
    strict["localization"]["max_median_position_error_m"] = 10.0
    strict["localization"]["per_algorithm"] = {
        "gcc_phat": {"max_median_position_error_m": 10.0}
    }
    report = evaluate_bundle(detections, bundle, strict)
    assert not report.passed
    joined = "\n".join(report.errors)
    assert "median position error" in joined
    assert "label accuracy" in joined
    assert "gcc_phat" in joined

    # No detections at all → recall failure.
    report_empty = evaluate_bundle([], bundle, strict)
    assert any("recall" in error for error in report_empty.errors)
