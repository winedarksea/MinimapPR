"""Promotion helpers for journal-backed ingest artifacts."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minimappr.api.journal_reader import JournalPayloadHandle
from minimappr.interfaces import StorageBackend


@dataclass(frozen=True, slots=True)
class JournalPromotionRequest:
    handle: JournalPayloadHandle
    journal_entry: dict[str, Any]
    artifact_type: str
    retention_tier: str
    large_artifact_dir: Path
    source_detection_id: str | None = None
    source_track_id: str | None = None
    expires_ns: int | None = None


@dataclass(frozen=True, slots=True)
class JournalPromotionResult:
    observation_id: str
    artifact_id: str
    artifact_path: Path


async def promote_journal_handle(
    *,
    storage: StorageBackend,
    request: JournalPromotionRequest,
    now_ns: int | None = None,
) -> JournalPromotionResult:
    """Materialize a retained artifact and its durable observation metadata.

    The source journal remains ephemeral. Promotion copies only the requested
    handle range to managed artifact storage, then creates the observation row
    needed for downstream provenance.
    """

    created_ns = time.time_ns() if now_ns is None else now_ns
    payload = request.handle.read_bytes()
    request.large_artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = request.large_artifact_dir / f"{request.handle.journal_epoch}-{request.handle.segment_id}-{request.handle.payload_offset_bytes}.bin"
    tmp_path = artifact_path.with_name(f".{artifact_path.name}.tmp")
    tmp_path.write_bytes(payload)
    shutil.move(str(tmp_path), artifact_path)

    entry = request.journal_entry
    observation_id = await storage.insert_observation(
        node_id=str(entry["node_id"]),
        sensor_id=str(entry.get("stream_id") or entry["stream_key"]),
        sensor_type=str(entry.get("sensor_type") or "audio"),
        source_type=str(entry.get("source_type") or "raw_sensor"),
        toa_ns=int(entry.get("toa_ns") or entry.get("ingest_received_ns") or created_ns),
        tor_ns=int(entry.get("tor_ns") or entry.get("ingest_received_ns") or created_ns),
        time_quality=str(entry.get("time_quality") or "freerunning"),
        sample_rate_hz=_optional_int(entry.get("sample_rate_hz")),
        channel_index=None,
        frame_sequence=_optional_int(entry.get("journal_sequence")),
        retention_tier=request.retention_tier,
        event_id=str(entry["observation_id"]) if entry.get("observation_id") else None,
        metadata={
            "journal_epoch": request.handle.journal_epoch,
            "journal_sequence": entry.get("journal_sequence"),
            "stream_key": request.handle.stream_key,
            "segment_id": request.handle.segment_id,
            "payload_offset_bytes": request.handle.payload_offset_bytes,
            "payload_length_bytes": request.handle.payload_length_bytes,
            "sample_index_start": request.handle.sample_index_start,
            "sample_count": request.handle.sample_count,
            "integrity_hash": request.handle.integrity_hash,
            "artifact_type": request.artifact_type,
            "source_detection_id": request.source_detection_id,
            "source_track_id": request.source_track_id,
            "artifact_expires_ns": request.expires_ns,
            "promotion_source": "journal_handle",
        },
    )
    artifact_id = await storage.insert_large_artifact(
        artifact_type=request.artifact_type,
        path=str(artifact_path),
        retention_tier=request.retention_tier,
        source_detection_id=request.source_detection_id,
        source_track_id=request.source_track_id,
        created_ns=created_ns,
        expires_ns=request.expires_ns,
        metadata={
            "source_observation_id": observation_id,
            "artifact_type": request.artifact_type,
            "source_detection_id": request.source_detection_id,
            "source_track_id": request.source_track_id,
            "expires_ns": request.expires_ns,
            "journal_epoch": request.handle.journal_epoch,
            "stream_key": request.handle.stream_key,
            "segment_id": request.handle.segment_id,
        },
    )
    return JournalPromotionResult(
        observation_id=observation_id,
        artifact_id=artifact_id,
        artifact_path=artifact_path,
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
