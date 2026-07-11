"""Calibration capture pipeline tests on synthetic multi-node audio.

Self-contained: a MultiSensorBuffer capture session is filled with synthetic
tetra-node windows, then CalibrationPipeline renders per-node WAVs, the
manifest, and detections.json into a tmp artifact directory.
"""

from __future__ import annotations

import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from minimappr.calibration.pipeline import CalibrationPipeline
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.capture_session import CaptureSessionRecord, CaptureState
from minimappr.core.environment import StaticEnvironmentProvider
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.models import GeoPoint
from tests.helpers import synthesize_multinode_windows

SAMPLE_RATE = 16_000
ORIGIN = GeoPoint(lat=45.0, lon=-93.0, alt_m=250.0)

NODE_ORIGINS_M = {
    "node-a": (0.0, 0.0, 1.5),
    "node-b": (20.0, 5.0, 1.5),
}


class _StubStorage:
    def __init__(self, frame: LocalCoordinateFrame) -> None:
        self._frame = frame
        self.artifacts: list[dict] = []

    async def get_node_by_id(self, node_id: str) -> dict | None:
        origin = NODE_ORIGINS_M.get(node_id)
        if origin is None:
            return None
        geo = self._frame.local_to_geo(origin)
        return {
            "id": node_id,
            "position_geo": {"lat": geo.lat, "lon": geo.lon, "alt_m": geo.alt_m},
            "position_m": list(origin),
            "sensor_offsets_m": [[0.0, 0.0, 0.0]] * 4,
            "orientation": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
        }

    async def list_detections(self, limit: int = 100, *, since_ns=None, until_ns=None, **_kw):
        return []

    async def insert_large_artifact(self, **kwargs) -> str:
        self.artifacts.append(kwargs)
        return "lar-test"


def _build_session(tmp_path: Path, capture_buffer, node_channel_map, start_ns, end_ns):
    work_dir = tmp_path / "work" / "session-1"
    work_dir.mkdir(parents=True)
    return CaptureSessionRecord(
        session_id="session-1",
        state=CaptureState.PROCESSING,
        stream_key="calibration",
        range_lease_id=None,
        start_time_ns=start_ns,
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


def test_calibration_pipeline_end_to_end(tmp_path: Path) -> None:
    async def _run() -> None:
        frame = LocalCoordinateFrame(origin=ORIGIN, mode="flat")
        storage = _StubStorage(frame)
        buffer = MultiSensorBuffer(max_duration_seconds=8.0)

        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        excitation = (np.sin(2 * np.pi * 700.0 * t) * 0.4).astype(np.float32)
        _, sensor_windows, sensor_node_ids = synthesize_multinode_windows(
            excitation,
            SAMPLE_RATE,
            source_position_m=(10.0, 2.0, 2.0),
            node_origins_m=NODE_ORIGINS_M,
            tetra_node_ids=("node-a", "node-b"),
        )

        node_channel_map = {
            node_id: [f"{node_id}:ch{i}" for i in range(4)] for node_id in NODE_ORIGINS_M
        }
        all_sensor_ids = [sid for sids in node_channel_map.values() for sid in sids]
        capture_buffer = await buffer.start_capture(
            "session-1", all_sensor_ids, max_duration_seconds=8.0
        )

        start_ns = time.time_ns()
        n_samples = next(iter(sensor_windows.values())).size
        end_ns = start_ns + int(n_samples / SAMPLE_RATE * 1_000_000_000)
        for sensor_id, samples in sensor_windows.items():
            capture_buffer.append(
                sensor_id=sensor_id,
                sample_rate_hz=SAMPLE_RATE,
                start_time_ns=start_ns,
                samples=samples,
            )
        assert sensor_node_ids  # geometry helper sanity

        record = _build_session(tmp_path, capture_buffer, node_channel_map, start_ns, end_ns)
        pipeline = CalibrationPipeline(
            storage=storage,
            coordinate_frame=frame,
            environment_provider=StaticEnvironmentProvider(
                temperature_c=21.0, humidity_fraction=0.5
            ),
            artifact_dir=tmp_path / "artifacts",
        )
        await pipeline.run(record)

        final_dir = tmp_path / "artifacts" / "session-1_calibration"
        assert record.work_dir == final_dir
        assert record.calibration_manifest_path == final_dir / "manifest.json"

        manifest = json.loads((final_dir / "manifest.json").read_text())
        assert manifest["kind"] == "minimappr_calibration_bundle"
        assert manifest["schema_version"] == 1
        assert manifest["site"]["origin"]["lat"] == pytest.approx(ORIGIN.lat)
        assert manifest["site"]["coordinate_mode"] == "flat"
        assert manifest["environment"]["speed_of_sound_mps"] > 300.0
        assert manifest["reference_audio"] is None
        assert len(manifest["nodes"]) == 2

        for node in manifest["nodes"]:
            node_id = node["node_id"]
            assert node["channel_sensor_ids"] == node_channel_map[node_id]
            # geo ↔ local round-trip consistency
            local = frame.geo_to_local(GeoPoint(**node["position_geo"]))
            assert np.allclose(local, node["position_m"], atol=0.01)
            assert np.allclose(node["position_m"], NODE_ORIGINS_M[node_id], atol=0.01)
            wav_path = final_dir / node["audio_file"]
            with wave.open(str(wav_path), "rb") as w:
                assert w.getnchannels() == 4
                assert w.getframerate() == SAMPLE_RATE
                assert abs(w.getnframes() - n_samples) <= SAMPLE_RATE // 100
            assert node["sync_diag"]["capture_rate_hz"] == SAMPLE_RATE

        detections = json.loads((final_dir / "detections.json").read_text())
        assert detections["detections"] == []
        assert storage.artifacts and storage.artifacts[0]["artifact_type"] == "calibration"

    asyncio.run(_run())


def test_calibration_pipeline_omits_silent_node(tmp_path: Path) -> None:
    async def _run() -> None:
        frame = LocalCoordinateFrame(origin=ORIGIN, mode="flat")
        storage = _StubStorage(frame)
        buffer = MultiSensorBuffer(max_duration_seconds=8.0)

        node_channel_map = {
            "node-a": [f"node-a:ch{i}" for i in range(4)],
            "node-b": [f"node-b:ch{i}" for i in range(4)],
        }
        all_sensor_ids = [sid for sids in node_channel_map.values() for sid in sids]
        capture_buffer = await buffer.start_capture(
            "session-1", all_sensor_ids, max_duration_seconds=8.0
        )

        start_ns = time.time_ns()
        samples = np.random.default_rng(0).normal(0, 0.1, SAMPLE_RATE).astype(np.float32)
        for i in range(4):
            capture_buffer.append(
                sensor_id=f"node-a:ch{i}",
                sample_rate_hz=SAMPLE_RATE,
                start_time_ns=start_ns,
                samples=samples,
            )
        end_ns = start_ns + 1_000_000_000

        record = _build_session(tmp_path, capture_buffer, node_channel_map, start_ns, end_ns)
        pipeline = CalibrationPipeline(
            storage=storage,
            coordinate_frame=frame,
            environment_provider=StaticEnvironmentProvider(
                temperature_c=21.0, humidity_fraction=0.5
            ),
            artifact_dir=tmp_path / "artifacts",
        )
        await pipeline.run(record)

        manifest = json.loads((record.calibration_manifest_path).read_text())
        assert [n["node_id"] for n in manifest["nodes"]] == ["node-a"]
        assert not (record.work_dir / "audio" / "node-b.wav").exists()

    asyncio.run(_run())
