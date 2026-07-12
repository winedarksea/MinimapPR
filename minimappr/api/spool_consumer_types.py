"""Shared types for filesystem ingest spool consumption."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from minimappr.api.journal_reader import JournalPayloadHandle
from minimappr.api.rust_dsp_manifests import JournalCursorUpdate, LocalizedClassifierRenderRequest
from minimappr.models import NodeSpec

IngestStorageMode = Literal["spool", "journal"]
SpoolEndpoint = Literal["/api/v1/ingest/binary", "/api/v1/ingest/store-forward"]


@dataclass(frozen=True, slots=True)
class IngestSpoolConfig:
    spool_dir: Path
    ready_ttl_seconds: float
    failed_ttl_seconds: float
    tmp_ttl_seconds: float
    poll_interval_seconds: float
    worker_count: int
    storage_mode: IngestStorageMode = "spool"
    consumer_name: str = "python-ingest"
    rust_dsp_claim_batch_size: int = 2

    @property
    def tmp_dir(self) -> Path:
        return self.spool_dir / "tmp"

    @property
    def ready_dir(self) -> Path:
        return self.spool_dir / "ready"

    @property
    def processing_dir(self) -> Path:
        return self.spool_dir / "processing"

    @property
    def failed_dir(self) -> Path:
        return self.spool_dir / "failed"

    @property
    def journal_dir(self) -> Path:
        return self.spool_dir / "journal"

    @property
    def journal_streams_dir(self) -> Path:
        return self.journal_dir / "streams"

    @property
    def journal_segments_dir(self) -> Path:
        return self.journal_streams_dir

    @property
    def journal_consumer_state_path(self) -> Path:
        return self.journal_dir / "consumer_state.sqlite3"

    @property
    def journal_manifest_dir(self) -> Path:
        return self.journal_dir / "manifests" / "pending"


@dataclass(frozen=True, slots=True)
class SpoolProcessingSummary:
    processed: int = 0
    expired: int = 0
    failed: int = 0
    cleaned_tmp: int = 0
    cleaned_orphan_ready: int = 0
    cleaned_failed: int = 0
    cleaned_processing: int = 0


@dataclass(frozen=True, slots=True)
class ClaimedIngestItem:
    ingest_id: str
    source_kind: IngestStorageMode
    endpoint: str
    received_ns: int
    cleanup_paths: tuple[Path, ...]
    stream_key: str | None = None
    journal_epoch: int | None = None
    journal_sequence: int | None = None
    body_path: Path | None = None
    segment_path: Path | None = None
    body_offset_bytes: int = 0
    body_length_bytes: int = 0
    integrity_hash: str = ""
    sample_index_start: int | None = None
    sample_count: int | None = None

    def read_payload_bytes(self) -> bytes:
        if self.body_path is not None:
            return self.body_path.read_bytes()
        if self.segment_path is None:
            raise ValueError(f"No payload source configured for ingest item {self.ingest_id}")
        return JournalPayloadHandle(
            journal_epoch=int(self.journal_epoch or 0),
            segment_id=self.segment_path.stem,
            stream_key=str(self.stream_key or ""),
            payload_offset_bytes=self.body_offset_bytes,
            payload_length_bytes=self.body_length_bytes,
            sample_index_start=self.sample_index_start,
            sample_count=self.sample_count,
            integrity_hash=self.integrity_hash,
            segment_path=self.segment_path,
        ).read_bytes()


@dataclass(frozen=True, slots=True)
class ClaimedLocalizedRenderItem:
    ingest_id: str
    source_kind: Literal["journal_localized_render_manifest"]
    received_ns: int
    cleanup_paths: tuple[Path, ...]
    cursor_updates: tuple[JournalCursorUpdate, ...]
    localized_render_request: LocalizedClassifierRenderRequest


@dataclass(frozen=True, slots=True)
class ClaimedLocalizedNodeHeartbeatItem:
    ingest_id: str
    source_kind: Literal["journal_localized_render_manifest"]
    received_ns: int
    cleanup_paths: tuple[Path, ...]
    cursor_updates: tuple[JournalCursorUpdate, ...]
    node: NodeSpec


ClaimedItem = ClaimedIngestItem | ClaimedLocalizedRenderItem | ClaimedLocalizedNodeHeartbeatItem
