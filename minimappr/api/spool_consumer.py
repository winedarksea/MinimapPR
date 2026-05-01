"""Filesystem ingest consumer for sidecar-backed ingest.

The Rust sidecar can publish either per-request spool manifests or append-only
journal entries. Journal replay state is durable per consumer and stream, while
short-lived claim files remain the in-flight lock so partial writes are never
parsed and concurrent workers do not race the same stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from minimappr.api.journal_reader import JournalPayloadHandle
from minimappr.api.rust_dsp_manifests import (
    JournalCursorUpdate,
    LocalizedClassifierRenderRequest,
    LocalizedRenderManifestBundle,
    load_localized_render_manifest_bundle,
    manifest_source_key,
)
from minimappr.api.binary_ingest import parse_binary_ingest_payload
from minimappr.api.journal_state import JournalConsumerCursor, JournalConsumerStateStore
from minimappr.interfaces import IngestTransport
from minimappr.models import StoreForwardIngestRequest

logger = logging.getLogger(__name__)

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
    runtime_profile: str = "default"

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


class IngestSpoolConsumer:
    def __init__(self, *, config: IngestSpoolConfig, ingest_transport: IngestTransport) -> None:
        self._config = config
        self._ingest_transport = ingest_transport

    def ensure_directories(self) -> None:
        directories = [
            self._config.tmp_dir,
            self._config.ready_dir,
            self._config.processing_dir,
            self._config.failed_dir,
        ]
        if self._config.storage_mode == "journal":
            directories.append(self._config.journal_streams_dir)
            directories.append(self._config.journal_manifest_dir)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        if self._config.storage_mode == "journal":
            JournalConsumerStateStore(self._config.journal_consumer_state_path).ensure_initialized()

    async def run_forever(self, *, worker_name: str) -> None:
        self.ensure_directories()
        # One-time startup pass: reclaim items orphaned in processing/ by a
        # previous crash.  Done before the live loop so no in-flight workers
        # are racing against the deletion.
        startup_cleaned_processing = await asyncio.to_thread(
            _delete_files_older_than,
            self._config.processing_dir,
            self._config.tmp_ttl_seconds,
            time.time_ns(),
        )
        if startup_cleaned_processing:
            logger.info(
                "Ingest spool worker %s cleaned %d stale processing item(s) at startup",
                worker_name,
                startup_cleaned_processing,
            )
        while True:
            try:
                summary = await self.run_once(now_ns=time.time_ns())
                if (
                    summary.processed
                    or summary.expired
                    or summary.failed
                    or summary.cleaned_tmp
                    or summary.cleaned_orphan_ready
                    or summary.cleaned_failed
                    or summary.cleaned_processing
                ):
                    logger.info("Ingest spool worker %s summary: %s", worker_name, summary)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Ingest spool worker %s failed a polling cycle: %s", worker_name, exc)
            await asyncio.sleep(self._config.poll_interval_seconds)

    async def run_once(self, *, now_ns: int | None = None) -> SpoolProcessingSummary:
        self.ensure_directories()
        effective_now_ns = time.time_ns() if now_ns is None else now_ns
        cleaned_tmp = 0
        cleaned_orphan_ready = 0
        if self._config.storage_mode == "spool":
            cleaned_tmp = await asyncio.to_thread(
                _delete_files_older_than,
                self._config.tmp_dir,
                self._config.tmp_ttl_seconds,
                effective_now_ns,
            )
            cleaned_orphan_ready = await asyncio.to_thread(
                _delete_orphan_ready_bodies,
                self._config.ready_dir,
                self._config.tmp_ttl_seconds,
                effective_now_ns,
            )
        cleaned_failed = await asyncio.to_thread(
            _delete_files_older_than,
            self._config.failed_dir,
            self._config.failed_ttl_seconds,
            effective_now_ns,
        )
        # Processing-dir cleanup is intentionally skipped here: items claimed
        # into processing/ may still be in-flight on a concurrent worker.  Stale
        # items from crashes are cleaned once at worker startup (run_forever).
        cleaned_processing = 0

        processed = 0
        expired = 0
        failed = 0
        lease_owner = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        leased_processing_paths = await asyncio.to_thread(
            _lease_existing_processing_claims_oldest_first,
            self._config.processing_dir,
            lease_owner,
        )
        claimed_paths = leased_processing_paths + await self._claim_ready_items()
        for claimed_path in claimed_paths:

            try:
                item = await self._load_claimed_item(claimed_path)
                # Re-sample now so items that waited in a large backlog are
                # judged by the time they are actually reached, not the single
                # snapshot taken at the start of this pass.
                item_check_now_ns = time.time_ns()
                age_seconds = max(0.0, (item_check_now_ns - item.received_ns) / 1_000_000_000.0)
                if age_seconds > self._config.ready_ttl_seconds:
                    await asyncio.to_thread(
                        _complete_claimed_item,
                        item,
                        self._config.journal_consumer_state_path,
                        self._config.consumer_name,
                        "expired",
                    )
                    expired += 1
                    continue

                await self._deliver_item(item)
                await asyncio.to_thread(
                    _complete_claimed_item,
                    item,
                    self._config.journal_consumer_state_path,
                    self._config.consumer_name,
                    "processed",
                )
                processed += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning("Moving ingest item %s to failed: %s", claimed_path.name, exc)
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(
                        _move_claimed_item_to_failed,
                        claimed_path,
                        self._config.failed_dir,
                        self._config.journal_consumer_state_path,
                        self._config.consumer_name,
                        str(exc),
                    )

        return SpoolProcessingSummary(
            processed=processed,
            expired=expired,
            failed=failed,
            cleaned_tmp=cleaned_tmp,
            cleaned_orphan_ready=cleaned_orphan_ready,
            cleaned_failed=cleaned_failed,
            cleaned_processing=cleaned_processing,  # always 0 in live loop
        )

    async def _claim_ready_items(self) -> list[Path]:
        if self._should_consume_rust_dsp_manifests():
            return await asyncio.to_thread(
                _claim_rust_dsp_manifests_oldest_first,
                self._config.journal_manifest_dir,
                self._config.journal_streams_dir,
                self._config.processing_dir,
                self._config.journal_consumer_state_path,
                self._config.consumer_name,
            )
        if self._config.storage_mode == "journal":
            return await asyncio.to_thread(
                _claim_journal_entries_oldest_first,
                self._config.journal_streams_dir,
                self._config.processing_dir,
                self._config.journal_consumer_state_path,
                self._config.consumer_name,
            )

        manifests = await asyncio.to_thread(_list_ready_manifests_oldest_first, self._config.ready_dir)
        claimed_paths: list[Path] = []
        for manifest_path in manifests:
            claimed_path = await asyncio.to_thread(
                _claim_manifest,
                manifest_path,
                self._config.processing_dir,
            )
            if claimed_path is not None:
                claimed_paths.append(claimed_path)
        return claimed_paths

    async def _load_claimed_item(self, claimed_path: Path) -> "ClaimedIngestItem":
        if claimed_path.name.endswith(".rustdsp.json"):
            return await asyncio.to_thread(
                _load_claimed_rust_dsp_item,
                claimed_path,
                self._config.journal_streams_dir,
            )
        if self._config.storage_mode == "journal":
            return await asyncio.to_thread(
                _load_claimed_journal_item,
                claimed_path,
                self._config.journal_streams_dir,
            )
        return await asyncio.to_thread(_load_claimed_spool_item, claimed_path)

    async def _deliver_item(self, item: "ClaimedIngestItem") -> None:
        if isinstance(item, ClaimedLocalizedRenderItem):
            await self._ingest_transport.deliver_localized_render(item.localized_render_request)
            return
        raw_payload = await asyncio.to_thread(item.read_payload_bytes)
        if item.endpoint == "/api/v1/ingest/binary":
            payload = parse_binary_ingest_payload(raw_payload)
            await self._ingest_transport.deliver_binary(payload)
            return
        if item.endpoint == "/api/v1/ingest/store-forward":
            try:
                decoded: Any = json.loads(raw_payload.decode("utf-8"))
                payload = StoreForwardIngestRequest.model_validate(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"Invalid store-forward spool payload: {exc}") from exc
            await self._ingest_transport.deliver_store_forward(payload)
            return
        raise ValueError(f"Unsupported ingest spool endpoint {item.endpoint!r}")

    def _should_consume_rust_dsp_manifests(self) -> bool:
        return (
            self._config.storage_mode == "journal"
            and self._config.runtime_profile == "birdnet_hybrid_production"
        )


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


def _list_ready_manifests_oldest_first(ready_dir: Path) -> list[Path]:
    manifests = [path for path in ready_dir.glob("*.json") if path.is_file()]

    def _sort_key(path: Path) -> str:
        stem = path.stem
        # Spool IDs begin with a 19-digit epoch-nanosecond prefix, so lexicographic
        # filename order is equivalent to received_ns timestamp order — no file I/O needed.
        # Fall back to mtime string for files that don't follow the naming convention.
        if stem and stem[0].isdigit():
            return stem
        try:
            return str(path.stat().st_mtime_ns)
        except Exception:  # noqa: BLE001
            return stem

    return sorted(manifests, key=_sort_key)


def _claim_manifest(manifest_path: Path, processing_dir: Path) -> Path | None:
    processing_dir.mkdir(parents=True, exist_ok=True)
    destination = processing_dir / manifest_path.name
    try:
        manifest_path.replace(destination)
    except FileNotFoundError:
        return None
    return destination


def _load_claimed_spool_item(manifest_path: Path) -> ClaimedIngestItem:
    with manifest_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    body_filename = str(metadata["body_filename"])
    if Path(body_filename).name != body_filename:
        raise ValueError(f"Invalid spool body filename {body_filename!r}")
    body_path = manifest_path.parent / body_filename
    ready_body_path = manifest_path.parent.parent / "ready" / body_filename
    if not body_path.exists() and ready_body_path.exists():
        ready_body_path.replace(body_path)
    if not body_path.is_file():
        raise FileNotFoundError(f"Spool body file missing for {manifest_path.name}: {body_filename}")
    return ClaimedIngestItem(
        ingest_id=str(metadata.get("spool_id") or manifest_path.stem),
        source_kind="spool",
        endpoint=str(metadata["endpoint"]),
        received_ns=int(metadata["received_ns"]),
        body_path=body_path,
        cleanup_paths=(body_path, manifest_path),
    )


def _load_claimed_journal_item(claim_path: Path, journal_streams_dir: Path) -> ClaimedIngestItem:
    with claim_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if str(metadata.get("source_kind", "")).strip().lower() != "journal":
        raise ValueError(f"Invalid journal claim file {claim_path.name}: missing source_kind=journal")

    journal_id = str(metadata.get("journal_id") or _journal_id_from_claim_path(claim_path))
    stream_key = str(metadata["stream_key"])
    if Path(stream_key).name != stream_key:
        raise ValueError(f"Invalid journal stream key {stream_key!r}")

    segment_id = str(metadata["segment_id"])
    if Path(segment_id).name != segment_id:
        raise ValueError(f"Invalid journal segment id {segment_id!r}")

    journal_epoch = int(metadata["journal_epoch"])
    journal_sequence = int(metadata["journal_sequence"])
    body_offset_bytes = int(metadata["body_offset_bytes"])
    body_length_bytes = int(metadata["body_length_bytes"])
    if body_offset_bytes < 0 or body_length_bytes < 0:
        raise ValueError(f"Journal entry {journal_id} has invalid byte range")

    segment_path = journal_streams_dir / stream_key / "segments" / f"{segment_id}.bin"
    if not segment_path.is_file():
        raise FileNotFoundError(
            f"Journal segment file missing for {claim_path.name}: {segment_path.name}"
        )

    return ClaimedIngestItem(
        ingest_id=journal_id,
        source_kind="journal",
        endpoint=str(metadata["endpoint"]),
        received_ns=int(metadata.get("ingest_received_ns") or metadata["received_ns"]),
        cleanup_paths=(claim_path,),
        stream_key=stream_key,
        journal_epoch=journal_epoch,
        journal_sequence=journal_sequence,
        segment_path=segment_path,
        body_offset_bytes=body_offset_bytes,
        body_length_bytes=body_length_bytes,
        integrity_hash=str(metadata.get("integrity_hash") or ""),
        sample_index_start=_optional_int(metadata.get("sample_index_start")),
        sample_count=_optional_int(metadata.get("sample_count")),
    )


def _delete_item_files(paths: tuple[Path, ...]) -> None:
    for path in paths:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def _complete_claimed_item(
    item: ClaimedIngestItem,
    journal_consumer_state_path: Path,
    consumer_name: str,
    status: Literal["processed", "expired"],
) -> None:
    if isinstance(item, ClaimedLocalizedRenderItem):
        state_store = JournalConsumerStateStore(journal_consumer_state_path)
        handled_ns = time.time_ns()
        for cursor_update in item.cursor_updates:
            state_store.mark_handled(
                consumer_name=consumer_name,
                stream_key=cursor_update.stream_key,
                journal_epoch=cursor_update.journal_epoch,
                journal_sequence=cursor_update.journal_sequence,
                journal_id=cursor_update.journal_id,
                status=status,
                handled_ns=handled_ns,
            )
        _delete_item_files(item.cleanup_paths)
        return
    if item.source_kind == "journal":
        if item.stream_key is None or item.journal_epoch is None or item.journal_sequence is None:
            raise ValueError(f"Journal item {item.ingest_id} is missing durable cursor metadata")
        JournalConsumerStateStore(journal_consumer_state_path).mark_handled(
            consumer_name=consumer_name,
            stream_key=item.stream_key,
            journal_epoch=item.journal_epoch,
            journal_sequence=item.journal_sequence,
            journal_id=item.ingest_id,
            status=status,
            handled_ns=time.time_ns(),
        )
    _delete_item_files(item.cleanup_paths)


def _move_claimed_item_to_failed(
    manifest_path: Path,
    failed_dir: Path,
    journal_consumer_state_path: Path,
    consumer_name: str,
    detail: str,
) -> None:
    failed_dir.mkdir(parents=True, exist_ok=True)
    body_path: Path | None = None
    with contextlib.suppress(Exception):
        with manifest_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if str(metadata.get("source_kind", "")).strip().lower() == "journal_localized_render_manifest":
            state_store = JournalConsumerStateStore(journal_consumer_state_path)
            handled_ns = time.time_ns()
            for payload in metadata.get("cursor_updates", []):
                state_store.mark_handled(
                    consumer_name=consumer_name,
                    stream_key=str(payload["stream_key"]),
                    journal_epoch=int(payload["journal_epoch"]),
                    journal_sequence=int(payload["journal_sequence"]),
                    journal_id=str(payload["journal_id"]),
                    status="failed",
                    handled_ns=handled_ns,
                    detail=detail,
                )
            for filename in metadata.get("cleanup_filenames", []):
                path = manifest_path.parent / str(filename)
                if path.exists():
                    path.replace(failed_dir / path.name)
            if manifest_path.exists():
                manifest_path.replace(failed_dir / manifest_path.name)
            return
        if str(metadata.get("source_kind", "")).strip().lower() == "journal":
            JournalConsumerStateStore(journal_consumer_state_path).mark_handled(
                consumer_name=consumer_name,
                stream_key=str(metadata["stream_key"]),
                journal_epoch=int(metadata["journal_epoch"]),
                journal_sequence=int(metadata["journal_sequence"]),
                journal_id=str(metadata.get("journal_id") or _journal_id_from_claim_path(manifest_path)),
                status="failed",
                handled_ns=time.time_ns(),
                detail=detail,
            )
            if manifest_path.exists():
                manifest_path.replace(failed_dir / manifest_path.name)
            return
        body_filename = str(metadata["body_filename"])
        if Path(body_filename).name == body_filename:
            body_path = manifest_path.parent / body_filename
    if body_path is not None and body_path.exists():
        body_path.replace(failed_dir / body_path.name)
    if manifest_path.exists():
        manifest_path.replace(failed_dir / manifest_path.name)


def _claim_journal_entries_oldest_first(
    journal_streams_dir: Path,
    processing_dir: Path,
    journal_consumer_state_path: Path,
    consumer_name: str,
) -> list[Path]:
    if not journal_streams_dir.exists():
        return []

    processing_dir.mkdir(parents=True, exist_ok=True)
    state_store = JournalConsumerStateStore(journal_consumer_state_path)
    state_store.ensure_initialized()

    cursor_by_stream = state_store.load_cursors(consumer_name)
    exception_ids = state_store.load_exception_ids(consumer_name)
    active_stream_keys = _processing_stream_keys(processing_dir)

    entries: list[tuple[tuple[int, str, int, int], dict[str, Any]]] = []
    for index_path in sorted(journal_streams_dir.glob("*/segments/*.index.jsonl")):
        default_stream_key = index_path.parent.parent.name
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw_entry = json.loads(line)
                stream_key = str(raw_entry.get("stream_key") or default_stream_key)
                journal_sequence = int(raw_entry["journal_sequence"])
                journal_epoch = int(raw_entry.get("journal_epoch") or 0)
                journal_id = str(raw_entry["journal_id"])
                if _cursor_covers_entry(cursor_by_stream.get(stream_key), journal_epoch, journal_sequence):
                    continue
                if journal_id in exception_ids:
                    continue
                entry = {
                    "source_kind": "journal",
                    "journal_id": journal_id,
                    "journal_epoch": journal_epoch,
                    "journal_sequence": journal_sequence,
                    "stream_key": stream_key,
                    "segment_id": str(raw_entry["segment_id"]),
                    "endpoint": str(raw_entry["endpoint"]),
                    "received_ns": int(raw_entry.get("ingest_received_ns") or raw_entry["received_ns"]),
                    "body_offset_bytes": int(raw_entry.get("payload_offset_bytes") or raw_entry["body_offset_bytes"]),
                    "body_length_bytes": int(raw_entry.get("payload_length_bytes") or raw_entry["body_length_bytes"]),
                    "integrity_hash": str(raw_entry.get("integrity_hash") or ""),
                    "sample_index_start": raw_entry.get("sample_index_start"),
                    "sample_count": raw_entry.get("sample_count"),
                }
                entries.append(
                    (
                        (entry["received_ns"], stream_key, journal_epoch, journal_sequence),
                        entry,
                    )
                )

    claimed_paths: list[Path] = []
    claimed_stream_keys = set(active_stream_keys)
    for _, entry in sorted(entries, key=lambda item: item[0]):
        stream_key = str(entry["stream_key"])
        if stream_key in claimed_stream_keys:
            continue

        journal_id = str(entry["journal_id"])
        claim_path = processing_dir / f"{journal_id}.journal.json"
        try:
            with claim_path.open("x", encoding="utf-8") as handle:
                json.dump(entry, handle)
                handle.write("\n")
        except FileExistsError:
            claimed_stream_keys.add(stream_key)
            continue

        if _cursor_covers_entry(
            state_store.get_cursor(consumer_name, stream_key),
            int(entry["journal_epoch"]),
            int(entry["journal_sequence"]),
        ):
            with contextlib.suppress(FileNotFoundError):
                claim_path.unlink()
            continue

        claimed_paths.append(claim_path)
        claimed_stream_keys.add(stream_key)

    return claimed_paths


def _processing_stream_keys(processing_dir: Path) -> set[str]:
    if not processing_dir.exists():
        return set()
    stream_keys: set[str] = set()
    for claim_path in processing_dir.glob("*.journal.json"):
        with contextlib.suppress(Exception):
            metadata = json.loads(claim_path.read_text(encoding="utf-8"))
            stream_key = str(metadata.get("stream_key") or "")
            if stream_key:
                stream_keys.add(stream_key)
    for claim_path in processing_dir.glob("*.rustdsp.json"):
        with contextlib.suppress(Exception):
            metadata = json.loads(claim_path.read_text(encoding="utf-8"))
            for stream_key in metadata.get("stream_keys", []):
                stream_key_text = str(stream_key or "")
                if stream_key_text:
                    stream_keys.add(stream_key_text)
    return stream_keys


def _lease_existing_processing_claims_oldest_first(
    processing_dir: Path,
    lease_owner: str,
) -> list[Path]:
    """Atomically lease pre-existing processing claim files for retry.

    Claims are the control files consumed by ``_load_claimed_item``:
    - ``*.journal.json`` for raw journal entries
    - ``*.rustdsp.json`` for paired Rust DSP manifests
    """
    if not processing_dir.exists():
        return []

    candidates: list[Path] = []
    candidates.extend(processing_dir.glob("*.journal.json"))
    candidates.extend(processing_dir.glob("*.rustdsp.json"))

    leased: list[Path] = []
    for claim_path in sorted(candidates, key=lambda path: path.stat().st_mtime_ns):
        lease_path = claim_path.with_name(f"lease-{lease_owner}-{claim_path.name}")
        try:
            claim_path.replace(lease_path)
        except FileNotFoundError:
            continue
        leased.append(lease_path)
    return leased


def _load_claimed_rust_dsp_item(
    claim_path: Path,
    journal_streams_dir: Path,
) -> ClaimedLocalizedRenderItem:
    with claim_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    primary_manifest_path = claim_path.parent / str(metadata["primary_manifest_filename"])
    paired_manifest_filename = metadata.get("paired_manifest_filename")
    paired_manifest_path = claim_path.parent / str(paired_manifest_filename) if paired_manifest_filename else None
    primary_manifest_payload = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    paired_manifest_payload = (
        json.loads(paired_manifest_path.read_text(encoding="utf-8"))
        if paired_manifest_path is not None and paired_manifest_path.exists()
        else None
    )
    bundle = load_localized_render_manifest_bundle(
        manifest_payload=primary_manifest_payload,
        journal_streams_dir=journal_streams_dir,
        paired_classifier_manifest_payload=paired_manifest_payload,
    )
    cleanup_paths: list[Path] = [claim_path, primary_manifest_path]
    if paired_manifest_path is not None:
        cleanup_paths.append(paired_manifest_path)
    return ClaimedLocalizedRenderItem(
        ingest_id=str(metadata.get("ingest_id") or primary_manifest_payload.get("manifest_id") or claim_path.stem),
        source_kind="journal_localized_render_manifest",
        received_ns=int(metadata.get("received_ns") or primary_manifest_payload.get("created_ns") or time.time_ns()),
        cleanup_paths=tuple(cleanup_paths),
        cursor_updates=bundle.cursor_updates,
        localized_render_request=bundle.request,
    )


def _claim_rust_dsp_manifests_oldest_first(
    journal_manifest_dir: Path,
    journal_streams_dir: Path,
    processing_dir: Path,
    journal_consumer_state_path: Path,
    consumer_name: str,
) -> list[Path]:
    if not journal_manifest_dir.exists():
        return []

    processing_dir.mkdir(parents=True, exist_ok=True)
    state_store = JournalConsumerStateStore(journal_consumer_state_path)
    state_store.ensure_initialized()
    cursor_by_stream = state_store.load_cursors(consumer_name)
    active_stream_keys = _processing_stream_keys(processing_dir)

    localization_candidates: dict[tuple[tuple[Any, ...], ...], tuple[Path, dict[str, Any]]] = {}
    classifier_candidates: dict[tuple[tuple[Any, ...], ...], tuple[Path, dict[str, Any]]] = {}

    for manifest_path in sorted(journal_manifest_dir.glob("*.json")):
        if not manifest_path.is_file() or manifest_path.name.startswith("."):
            continue
        with contextlib.suppress(Exception):
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_type = str(manifest_payload.get("manifest_type") or "")
            source_key = manifest_source_key(manifest_payload)
            if not source_key:
                continue
            if manifest_type == "localization_result" and manifest_payload.get("classifier_render") is not None:
                localization_candidates[source_key] = (manifest_path, manifest_payload)
            elif manifest_type == "classifier_render" and manifest_payload.get("derived_handle") is not None:
                classifier_candidates[source_key] = (manifest_path, manifest_payload)

    claims: list[tuple[int, Path, dict[str, Any], Path | None, dict[str, Any] | None]] = []
    for source_key, (localization_path, localization_payload) in localization_candidates.items():
        classifier_path, classifier_payload = classifier_candidates.get(source_key, (None, None))
        claims.append(
            (
                int(localization_payload.get("created_ns") or 0),
                localization_path,
                localization_payload,
                classifier_path,
                classifier_payload,
            )
        )
    for source_key, (classifier_path, classifier_payload) in classifier_candidates.items():
        if source_key in localization_candidates:
            continue
        claims.append(
            (
                int(classifier_payload.get("created_ns") or 0),
                classifier_path,
                classifier_payload,
                None,
                None,
            )
        )

    claimed_paths: list[Path] = []
    claimed_stream_keys = set(active_stream_keys)
    for _, primary_path, primary_payload, paired_path, paired_payload in sorted(claims, key=lambda item: item[0]):
        with contextlib.suppress(Exception):
            bundle = load_localized_render_manifest_bundle(
                manifest_payload=primary_payload,
                journal_streams_dir=journal_streams_dir,
                paired_classifier_manifest_payload=paired_payload,
            )
            stream_keys = {cursor_update.stream_key for cursor_update in bundle.cursor_updates}
            if stream_keys & claimed_stream_keys:
                continue
            if bundle.cursor_updates and all(
                _cursor_covers_entry(
                    cursor_by_stream.get(cursor_update.stream_key),
                    cursor_update.journal_epoch,
                    cursor_update.journal_sequence,
                )
                for cursor_update in bundle.cursor_updates
            ):
                with contextlib.suppress(FileNotFoundError):
                    primary_path.unlink()
                if paired_path is not None:
                    with contextlib.suppress(FileNotFoundError):
                        paired_path.unlink()
                continue

            primary_processing_path = processing_dir / primary_path.name
            paired_processing_path: Path | None = None
            primary_path.replace(primary_processing_path)
            if paired_path is not None:
                paired_processing_path = processing_dir / paired_path.name
                paired_path.replace(paired_processing_path)

            claim_payload = {
                "source_kind": "journal_localized_render_manifest",
                "ingest_id": bundle.request.manifest_id,
                "received_ns": int(primary_payload.get("created_ns") or time.time_ns()),
                "stream_keys": sorted(stream_keys),
                "cursor_updates": [
                    {
                        "journal_id": cursor_update.journal_id,
                        "stream_key": cursor_update.stream_key,
                        "journal_epoch": cursor_update.journal_epoch,
                        "journal_sequence": cursor_update.journal_sequence,
                    }
                    for cursor_update in bundle.cursor_updates
                ],
                "primary_manifest_filename": primary_processing_path.name,
                "paired_manifest_filename": paired_processing_path.name if paired_processing_path is not None else None,
                "cleanup_filenames": [
                    name
                    for name in (
                        primary_processing_path.name,
                        paired_processing_path.name if paired_processing_path is not None else None,
                    )
                    if name is not None
                ],
            }
            claim_path = processing_dir / f"{bundle.request.manifest_id}.rustdsp.json"
            with claim_path.open("x", encoding="utf-8") as handle:
                json.dump(claim_payload, handle)
                handle.write("\n")
            claimed_paths.append(claim_path)
            claimed_stream_keys.update(stream_keys)

    return claimed_paths


def _cursor_covers_entry(
    cursor: JournalConsumerCursor | None,
    journal_epoch: int,
    journal_sequence: int,
) -> bool:
    if cursor is None:
        return False
    if cursor.journal_epoch != journal_epoch:
        return cursor.journal_epoch > journal_epoch
    return cursor.last_fully_processed_journal_sequence >= journal_sequence


def _journal_id_from_claim_path(claim_path: Path) -> str:
    return claim_path.name.removesuffix(".journal.json")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _delete_files_older_than(directory: Path, ttl_seconds: float, now_ns: int) -> int:
    if ttl_seconds < 0.0 or not directory.exists():
        return 0
    cutoff_ns = now_ns - int(ttl_seconds * 1_000_000_000)
    deleted = 0
    for path in directory.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime_ns <= cutoff_ns:
                path.unlink()
                deleted += 1
        except FileNotFoundError:
            continue
    return deleted


def _delete_orphan_ready_bodies(ready_dir: Path, ttl_seconds: float, now_ns: int) -> int:
    if ttl_seconds < 0.0 or not ready_dir.exists():
        return 0
    cutoff_ns = now_ns - int(ttl_seconds * 1_000_000_000)
    deleted = 0
    for body_path in ready_dir.glob("*.body"):
        if not body_path.is_file():
            continue
        manifest_path = body_path.with_suffix(".json")
        try:
            if manifest_path.exists() or body_path.stat().st_mtime_ns > cutoff_ns:
                continue
            body_path.unlink()
            deleted += 1
        except FileNotFoundError:
            continue
    return deleted
