"""Calibration bundle read/write helpers.

Pure functions shared by the API export endpoint and the pytest replay
harness. A bundle is a zip per field scenario:

    manifest.json          session + nodes + environment
    ground_truth.json      operator-entered events
    detections.json        live system outputs during the window (reference)
    expectations.json      optional pass thresholds (harness defaults if absent)
    audio/{node_id}.wav    per-node multichannel PCM16; channel i == channel_sensor_ids[i]
    reference_audio/       RESERVED, empty in v1. Future: audio from a node
                           flagged as ground-truth reference (placed at the
                           source, excluded from localization) or an
                           operator-uploaded omni WAV, used as a timing/label
                           oracle during replay.
"""

from __future__ import annotations

import json
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

_REFERENCE_AUDIO_README = (
    "Reserved for v2: ground-truth reference audio captured at the source\n"
    "(reference node or operator-uploaded omni WAV) for replay timing/label\n"
    "oracles and beamforming tuning.\n"
)


def build_ground_truth_payload(rows: list[dict]) -> dict:
    """Convert calibration_ground_truth DB rows to the bundle's ground_truth.json."""
    events = []
    for row in rows:
        geometry_kind = row.get("geometry_kind") or "static"
        events.append(
            {
                "event_id": row["id"],
                "label": row["label"],
                "label_category": row.get("label_category") or "unknown",
                "source": "manual",
                "geometry": {
                    "type": geometry_kind,
                    "position_geo": {
                        "lat": row.get("lat"),
                        "lon": row.get("lon"),
                        "alt_m": row.get("alt_m") or 0.0,
                    },
                },
                "start_ns": int(row["start_ns"]),
                "end_ns": int(row["end_ns"]),
                "notes": row.get("notes"),
            }
        )
    return {"schema_version": SCHEMA_VERSION, "events": events}


def write_bundle_zip(
    session_artifact_dir: Path,
    ground_truth_payload: dict,
    out_path: Path,
    *,
    expectations: dict | None = None,
) -> Path:
    """Assemble a calibration bundle zip from a session artifact directory."""
    session_artifact_dir = Path(session_artifact_dir)
    manifest_path = session_artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {session_artifact_dir}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, "manifest.json")
        zf.writestr("ground_truth.json", json.dumps(ground_truth_payload, indent=2))
        detections_path = session_artifact_dir / "detections.json"
        if detections_path.exists():
            zf.write(detections_path, "detections.json")
        else:
            zf.writestr(
                "detections.json",
                json.dumps({"schema_version": SCHEMA_VERSION, "detections": []}),
            )
        if expectations is not None:
            zf.writestr("expectations.json", json.dumps(expectations, indent=2))
        audio_dir = session_artifact_dir / "audio"
        if audio_dir.is_dir():
            for wav_path in sorted(audio_dir.glob("*.wav")):
                zf.write(wav_path, f"audio/{wav_path.name}")
        zf.writestr("reference_audio/README.txt", _REFERENCE_AUDIO_README)
    return out_path


@dataclass
class CalibrationBundle:
    """In-memory view of a calibration bundle; audio is loaded lazily per node."""

    zip_path: Path
    manifest: dict
    events: list[dict]
    expectations: dict | None
    detections: list[dict]
    _audio_cache: dict[str, tuple[np.ndarray, int]] = field(default_factory=dict, repr=False)

    def node_audio(self, node_id: str) -> tuple[np.ndarray, int]:
        """Return (channels_first float32 in [-1, 1], sample_rate_hz) for a node."""
        cached = self._audio_cache.get(node_id)
        if cached is not None:
            return cached
        node = next(
            (n for n in self.manifest.get("nodes", []) if n.get("node_id") == node_id), None
        )
        if node is None:
            raise KeyError(f"node {node_id} not in bundle manifest")
        with zipfile.ZipFile(self.zip_path) as zf:
            with zf.open(node["audio_file"]) as raw:
                with wave.open(raw) as w:
                    n_channels = w.getnchannels()
                    sample_rate_hz = w.getframerate()
                    if w.getsampwidth() != 2:
                        raise ValueError("bundle audio must be PCM16")
                    frames = w.readframes(w.getnframes())
        interleaved = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
        channels_first = interleaved.reshape(-1, n_channels).T.copy()
        result = (channels_first, sample_rate_hz)
        self._audio_cache[node_id] = result
        return result


def load_bundle(zip_path: Path) -> CalibrationBundle:
    """Load and validate a calibration bundle zip (audio stays lazy)."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
        ground_truth = json.loads(zf.read("ground_truth.json"))
        expectations = (
            json.loads(zf.read("expectations.json")) if "expectations.json" in names else None
        )
        detections_payload = (
            json.loads(zf.read("detections.json")) if "detections.json" in names else {}
        )

    if manifest.get("kind") != "minimappr_calibration_bundle":
        raise ValueError(f"{zip_path} is not a minimappr calibration bundle")
    if int(manifest.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported bundle schema_version: {manifest.get('schema_version')}")

    events = list(ground_truth.get("events", []))
    for event in events:
        geometry = event.get("geometry") or {}
        # v1 readers reject unknown geometry types (trajectory import comes later).
        if geometry.get("type") != "static":
            raise ValueError(
                f"unsupported ground-truth geometry type: {geometry.get('type')!r}"
            )

    return CalibrationBundle(
        zip_path=zip_path,
        manifest=manifest,
        events=events,
        expectations=expectations,
        detections=list(detections_payload.get("detections", [])),
    )
