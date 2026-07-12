"""Calibration capture post-processing pipeline.

Runs after a `capture_kind == "calibration"` session stops:

  1. Extract  — per node in node_channel_map, render that node's sensor
                channels from the session capture buffer and write a
                multichannel PCM16 WAV (channel i == channel_sensor_ids[i]).
  2. Manifest — node geometry (geo + local ENU + sensor offsets + orientation),
                site origin/coordinate mode, environment snapshot, per-node
                sync diagnostics.
  3. Detections — system outputs recorded live during the window (reference
                only; never replayed as inputs).
  4. Register — promote the work dir to data/artifacts/{session_id}_calibration
                and insert a large_artifacts row.

Raw-audio disk contract (see minimappr/core/iamf_pipeline.py): the raw
real-time audio path must never land on disk. Like the IAMF record path,
this explicit-capture post-process is the sanctioned exception — audio is
written only for the operator-initiated calibration window.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np

from minimappr.core.capture_session import CaptureSessionRecord
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import speed_of_sound_mps
from minimappr.models import GeoPoint

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
BUNDLE_KIND = "minimappr_calibration_bundle"


def write_multichannel_wav(path: Path, channels_first: np.ndarray, sample_rate_hz: int) -> None:
    """Write a channels-first float32 array as multichannel PCM16 WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    interleaved = np.clip(channels_first.T, -1.0, 1.0)
    pcm = (interleaved * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(int(channels_first.shape[0]))
        w.setsampwidth(2)
        w.setframerate(int(sample_rate_hz))
        w.writeframes(pcm.tobytes())


class CalibrationPipeline:
    """Post-processes one calibration capture session into an artifact directory."""

    def __init__(
        self,
        *,
        storage: Any,
        coordinate_frame: LocalCoordinateFrame,
        environment_provider: Any,
        artifact_dir: Path,
        settings: Any = None,
    ) -> None:
        self._storage = storage
        self._coordinate_frame = coordinate_frame
        self._environment_provider = environment_provider
        self._artifact_dir = Path(artifact_dir)
        self._settings = settings

    async def run(self, record: CaptureSessionRecord) -> None:
        if record.capture_audio_buffer is None:
            raise RuntimeError("calibration session has no capture audio buffer")
        start_ns = record.start_time_ns
        end_ns = record.end_time_ns
        if start_ns is None or end_ns is None:
            raise RuntimeError("calibration session is missing its time window")

        audio_dir = record.work_dir / "audio"
        node_manifests: list[dict[str, Any]] = []
        for node_id, sensor_ids in record.node_channel_map.items():
            if not sensor_ids:
                continue
            try:
                channels, sample_rate_hz, sync_diag = await asyncio.to_thread(
                    record.capture_audio_buffer.extract_range,
                    list(sensor_ids),
                    start_ns,
                    end_ns,
                )
            except RuntimeError as exc:
                logger.warning(
                    "calibration session %s: node %s omitted (no audio coverage): %s",
                    record.session_id,
                    node_id,
                    exc,
                )
                continue

            wav_path = audio_dir / f"{node_id}.wav"
            await asyncio.to_thread(write_multichannel_wav, wav_path, channels, sample_rate_hz)
            node_manifests.append(
                await self._build_node_manifest(
                    node_id=node_id,
                    sensor_ids=list(sensor_ids),
                    sample_rate_hz=sample_rate_hz,
                    sync_diag=sync_diag,
                )
            )

        if not node_manifests:
            raise RuntimeError("calibration session captured no audio from any node")

        manifest = self._build_manifest(record, node_manifests, start_ns=start_ns, end_ns=end_ns)
        manifest_path = record.work_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        detections = await self._storage.list_detections(
            limit=10_000, since_ns=start_ns, until_ns=end_ns
        )
        (record.work_dir / "detections.json").write_text(
            json.dumps({"schema_version": SCHEMA_VERSION, "detections": detections}, indent=2)
        )

        final_dir = self._artifact_dir / f"{record.session_id}_calibration"
        await asyncio.to_thread(self._promote_work_dir, record.work_dir, final_dir)
        record.work_dir = final_dir
        record.calibration_manifest_path = final_dir / "manifest.json"

        await self._storage.insert_large_artifact(
            artifact_type="calibration",
            path=str(record.calibration_manifest_path),
            retention_tier="permanent",
            source_detection_id=None,
            source_track_id=None,
            created_ns=time.time_ns(),
            expires_ns=None,
            metadata={
                "session_id": record.session_id,
                "manifest_path": str(record.calibration_manifest_path),
                "node_ids": [node["node_id"] for node in node_manifests],
            },
        )
        logger.info(
            "calibration session %s: %d node(s) captured → %s",
            record.session_id,
            len(node_manifests),
            final_dir,
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _promote_work_dir(work_dir: Path, final_dir: Path) -> None:
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.move(str(work_dir), str(final_dir))

    async def _build_node_manifest(
        self,
        *,
        node_id: str,
        sensor_ids: list[str],
        sample_rate_hz: int,
        sync_diag: dict,
    ) -> dict[str, Any]:
        node_row: Optional[dict] = await self._storage.get_node_by_id(node_id)
        position_geo: Optional[dict] = None
        position_m: Optional[list[float]] = None
        sensor_offsets_m: list = []
        orientation: dict = {}
        if node_row is not None:
            sensor_offsets_m = node_row.get("sensor_offsets_m") or []
            raw_orientation = node_row.get("orientation")
            orientation = raw_orientation if isinstance(raw_orientation, dict) else {}
            geo = node_row.get("position_geo")
            if isinstance(geo, dict) and geo.get("lat") is not None:
                position_geo = {
                    "lat": float(geo["lat"]),
                    "lon": float(geo["lon"]),
                    "alt_m": float(geo.get("alt_m") or 0.0),
                }
                local = self._coordinate_frame.geo_to_local(
                    GeoPoint(lat=position_geo["lat"], lon=position_geo["lon"], alt_m=position_geo["alt_m"])
                )
                position_m = [float(local[0]), float(local[1]), float(local[2])]
            else:
                raw_local = node_row.get("position_m")
                if isinstance(raw_local, (list, tuple)) and len(raw_local) >= 3:
                    position_m = [float(raw_local[0]), float(raw_local[1]), float(raw_local[2])]
                    geo_point = self._coordinate_frame.local_to_geo(
                        (position_m[0], position_m[1], position_m[2])
                    )
                    position_geo = {
                        "lat": geo_point.lat,
                        "lon": geo_point.lon,
                        "alt_m": geo_point.alt_m,
                    }
        return {
            "node_id": node_id,
            "audio_file": f"audio/{node_id}.wav",
            "sample_rate_hz": int(sample_rate_hz),
            "audio_start_time_ns": int(sync_diag.get("requested_start_ns") or 0),
            "channel_sensor_ids": sensor_ids,
            "position_geo": position_geo,
            "position_m": position_m,
            "sensor_offsets_m": sensor_offsets_m,
            "orientation": orientation,
            "sync_diag": sync_diag,
        }

    def _build_manifest(
        self,
        record: CaptureSessionRecord,
        node_manifests: list[dict[str, Any]],
        *,
        start_ns: int,
        end_ns: int,
    ) -> dict[str, Any]:
        conditions = self._environment_provider.get_conditions(location_m=None)
        environment = {
            "temperature_c": float(conditions.temperature_c),
            "humidity_fraction": float(conditions.humidity_fraction),
            "speed_of_sound_mps": speed_of_sound_mps(
                temperature_c=conditions.temperature_c,
                humidity_fraction=conditions.humidity_fraction,
            ),
            "source": "live" if conditions.metadata.get("source") == "live" else "fallback",
        }
        origin = self._coordinate_frame.origin
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": BUNDLE_KIND,
            "session_id": record.session_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "time_window": {"start_ns": int(start_ns), "end_ns": int(end_ns)},
            "site": {
                "origin": {"lat": origin.lat, "lon": origin.lon, "alt_m": origin.alt_m},
                "coordinate_mode": self._coordinate_frame.mode,
            },
            "environment": environment,
            "nodes": node_manifests,
            # RESERVED for a future ground-truth reference recording: audio from
            # a node flagged as reference (placed at the source, excluded from
            # localization) or an operator-uploaded omni WAV, used as a
            # timing/label oracle during replay and for beamforming tuning.
            "reference_audio": None,
        }
