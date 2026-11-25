"""Calibration bundle read/write helpers.

Pure functions shared by the API export endpoint and the pytest replay
harness. A bundle is a zip per field scenario:

    manifest.json          session + nodes + environment
    ground_truth.json      ground-truth events (schema v1 or v2; v2 adds
                           individual_id, species lists, echo sources and
                           trajectories — see calibration/ground_truth_events.py
                           and docs/field_mode_progress/bundle_format_v2.md)
    detections.json        live system outputs during the window (reference)
    expectations.json      optional pass thresholds (harness defaults if absent)
    audio/{node_id}.wav    per-node multichannel PCM16; channel i == channel_sensor_ids[i]
    frame_timing/{node_id}.json
                           optional per-frame packet timing and time_quality
                           (calibration/frame_timing.py); named by the node
                           entry's ``frame_timing_file``, absent in older bundles
    reference_audio/{event_id}.wav
                           mono mixdown of the event's ground-truth node
                           (a node placed at the source, excluded from
                           localization during replay by default), used as a
                           timing/label oracle and for beamforming tuning.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from minimappr.calibration.frame_timing import FRAME_TIMING_BUNDLE_DIRECTORY, NodeFrameTimingTable
from minimappr.calibration.ground_truth_events import (
    GROUND_TRUTH_SCHEMA_VERSION,
    validate_ground_truth_payload,
)

# Manifest / detections container version. The manifest layout is unchanged by
# ground-truth v2, so it stays at 1; ground_truth.json carries its own version.
SCHEMA_VERSION = 1

_REFERENCE_AUDIO_README = (
    "Ground-truth reference audio captured at the source: one mono WAV per\n"
    "node-sourced ground-truth event ({event_id}.wav, mean of the node's\n"
    "channels over the event window), used as a replay timing/label oracle\n"
    "and for beamforming tuning.\n"
)


def build_ground_truth_payload(rows: list[dict]) -> dict:
    """Convert calibration_ground_truth DB rows to the bundle's ground_truth.json."""
    events = []
    for row in rows:
        geometry_kind = row.get("geometry_kind") or "static"
        source_node_id = row.get("source_node_id")
        event = {
            "event_id": row["id"],
            "label": row["label"],
            "label_category": row.get("label_category") or "unknown",
            "source": "node" if source_node_id else "manual",
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
        if source_node_id:
            event["source_node_id"] = source_node_id
            event["reference_audio"] = f"reference_audio/{row['id']}.wav"
        events.append(event)
    return {"schema_version": GROUND_TRUTH_SCHEMA_VERSION, "events": events}


def _slice_reference_audio(
    session_artifact_dir: Path,
    manifest: dict,
    event: dict,
) -> bytes | None:
    """Mono-mix the ground-truth node's audio over the event window as PCM16 WAV.

    Returns None when the node has no audio in the bundle or the window falls
    entirely outside the recording.
    """
    node = next(
        (
            n
            for n in manifest.get("nodes", [])
            if n.get("node_id") == event.get("source_node_id")
        ),
        None,
    )
    if node is None:
        return None
    wav_path = session_artifact_dir / node["audio_file"]
    if not wav_path.exists():
        return None
    with wave.open(str(wav_path)) as w:
        n_channels = w.getnchannels()
        sample_rate_hz = w.getframerate()
        total_samples = w.getnframes()
        audio_start_ns = int(node.get("audio_start_time_ns") or 0)
        start_sample = int((int(event["start_ns"]) - audio_start_ns) * sample_rate_hz // 1_000_000_000)
        end_sample = int((int(event["end_ns"]) - audio_start_ns) * sample_rate_hz // 1_000_000_000)
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(0, min(end_sample, total_samples))
        if end_sample <= start_sample:
            return None
        w.setpos(start_sample)
        frames = w.readframes(end_sample - start_sample)
    interleaved = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
    mono = interleaved.reshape(-1, n_channels).mean(axis=1)
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate_hz)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def write_bundle_zip(
    session_artifact_dir: Path,
    ground_truth_payload: dict,
    out_path: Path,
    *,
    expectations: dict | None = None,
) -> Path:
    """Assemble a calibration bundle zip from a session artifact directory.

    Written to a temporary file beside ``out_path`` and moved into place, so a
    concurrent reader (a download streaming the previous export) never sees a
    truncated zip.
    """
    session_artifact_dir = Path(session_artifact_dir)
    manifest_path = session_artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {session_artifact_dir}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{out_path.name}.", suffix=".partial", dir=out_path.parent)
    os.close(file_descriptor)
    try:
        _write_bundle_members(Path(temporary_name), session_artifact_dir, manifest_path, ground_truth_payload, expectations)
        os.replace(temporary_name, out_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return out_path


def _write_bundle_members(
    zip_path: Path,
    session_artifact_dir: Path,
    manifest_path: Path,
    ground_truth_payload: dict,
    expectations: dict | None,
) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
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
        frame_timing_dir = session_artifact_dir / FRAME_TIMING_BUNDLE_DIRECTORY
        if frame_timing_dir.is_dir():
            for timing_path in sorted(frame_timing_dir.glob("*.json")):
                zf.write(timing_path, f"{FRAME_TIMING_BUNDLE_DIRECTORY}/{timing_path.name}")
        manifest = json.loads(manifest_path.read_text())
        for event in ground_truth_payload.get("events", []):
            if not event.get("source_node_id"):
                continue
            wav_bytes = _slice_reference_audio(session_artifact_dir, manifest, event)
            if wav_bytes is not None:
                zf.writestr(f"reference_audio/{event['event_id']}.wav", wav_bytes)
        zf.writestr("reference_audio/README.txt", _REFERENCE_AUDIO_README)


@dataclass
class CalibrationBundle:
    """In-memory view of a calibration bundle; audio is loaded lazily per node."""

    zip_path: Path
    manifest: dict
    events: list[dict]
    expectations: dict | None
    detections: list[dict]
    _audio_cache: dict[str, tuple[np.ndarray, int]] = field(default_factory=dict, repr=False)

    def node_manifest(self, node_id: str) -> dict:
        node = next(
            (n for n in self.manifest.get("nodes", []) if n.get("node_id") == node_id), None
        )
        if node is None:
            raise KeyError(f"node {node_id} not in bundle manifest")
        return node

    def node_frame_timing(self, node_id: str) -> NodeFrameTimingTable | None:
        """The node's per-frame packet timing; None for a bundle written before the table existed."""
        timing_file = self.node_manifest(node_id).get("frame_timing_file")
        if not timing_file:
            return None
        with zipfile.ZipFile(self.zip_path) as zf:
            table = NodeFrameTimingTable.from_json_payload(json.loads(zf.read(timing_file)))
        if table.node_id != node_id:
            raise ValueError(f"{timing_file} describes node {table.node_id!r}, not {node_id!r}")
        return table

    def node_audio(self, node_id: str) -> tuple[np.ndarray, int]:
        """Return (channels_first float32 in [-1, 1], sample_rate_hz) for a node."""
        cached = self._audio_cache.get(node_id)
        if cached is not None:
            return cached
        node = self.node_manifest(node_id)
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

    def event_reference_audio(self, event_id: str) -> tuple[np.ndarray, int] | None:
        """Return (mono float32 in [-1, 1], sample_rate_hz) for a node-sourced event."""
        event = next((e for e in self.events if e.get("event_id") == event_id), None)
        if event is None or not event.get("reference_audio"):
            return None
        with zipfile.ZipFile(self.zip_path) as zf:
            if event["reference_audio"] not in zf.namelist():
                return None
            with zf.open(event["reference_audio"]) as raw:
                with wave.open(raw) as w:
                    sample_rate_hz = w.getframerate()
                    if w.getsampwidth() != 2 or w.getnchannels() != 1:
                        raise ValueError("reference audio must be mono PCM16")
                    frames = w.readframes(w.getnframes())
        mono = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
        return mono, sample_rate_hz


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

    events = validate_ground_truth_payload(ground_truth)

    return CalibrationBundle(
        zip_path=zip_path,
        manifest=manifest,
        events=events,
        expectations=expectations,
        detections=list(detections_payload.get("detections", [])),
    )
