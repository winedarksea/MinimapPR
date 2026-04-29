"""Filesystem spool consumer for sidecar-backed ingest.

The Rust sidecar publishes a body file and then atomically publishes a JSON
manifest.  The consumer scans manifests only, so incomplete uploads in tmp/ are
never parsed by Python.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from minimappr.api.binary_ingest import parse_binary_ingest_payload
from minimappr.interfaces import IngestTransport
from minimappr.models import StoreForwardIngestRequest

logger = logging.getLogger(__name__)

SpoolEndpoint = Literal["/api/v1/ingest/binary", "/api/v1/ingest/store-forward"]


@dataclass(frozen=True, slots=True)
class IngestSpoolConfig:
    spool_dir: Path
    ready_ttl_seconds: float
    failed_ttl_seconds: float
    tmp_ttl_seconds: float
    poll_interval_seconds: float
    worker_count: int

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
        for directory in (
            self._config.tmp_dir,
            self._config.ready_dir,
            self._config.processing_dir,
            self._config.failed_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

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
        manifests = await asyncio.to_thread(_list_ready_manifests_oldest_first, self._config.ready_dir)
        for manifest_path in manifests:
            claimed_manifest_path = await asyncio.to_thread(
                _claim_manifest,
                manifest_path,
                self._config.processing_dir,
            )
            if claimed_manifest_path is None:
                continue

            try:
                item = await asyncio.to_thread(_load_claimed_item, claimed_manifest_path)
                # Re-sample now so items that waited in a large backlog are
                # judged by the time they are actually reached, not the single
                # snapshot taken at the start of this pass.
                item_check_now_ns = time.time_ns()
                age_seconds = max(0.0, (item_check_now_ns - item.received_ns) / 1_000_000_000.0)
                if age_seconds > self._config.ready_ttl_seconds:
                    await asyncio.to_thread(_delete_item_files, claimed_manifest_path, item.body_path)
                    expired += 1
                    continue

                await self._deliver_item(item)
                await asyncio.to_thread(_delete_item_files, claimed_manifest_path, item.body_path)
                processed += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning("Moving ingest spool item %s to failed: %s", claimed_manifest_path.name, exc)
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(_move_claimed_item_to_failed, claimed_manifest_path, self._config.failed_dir)

        return SpoolProcessingSummary(
            processed=processed,
            expired=expired,
            failed=failed,
            cleaned_tmp=cleaned_tmp,
            cleaned_orphan_ready=cleaned_orphan_ready,
            cleaned_failed=cleaned_failed,
            cleaned_processing=cleaned_processing,  # always 0 in live loop
        )

    async def _deliver_item(self, item: "ClaimedSpoolItem") -> None:
        raw_payload = await asyncio.to_thread(item.body_path.read_bytes)
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


@dataclass(frozen=True, slots=True)
class ClaimedSpoolItem:
    spool_id: str
    endpoint: str
    received_ns: int
    body_path: Path


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


def _load_claimed_item(manifest_path: Path) -> ClaimedSpoolItem:
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
    return ClaimedSpoolItem(
        spool_id=str(metadata.get("spool_id") or manifest_path.stem),
        endpoint=str(metadata["endpoint"]),
        received_ns=int(metadata["received_ns"]),
        body_path=body_path,
    )


def _delete_item_files(manifest_path: Path, body_path: Path) -> None:
    for path in (body_path, manifest_path):
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def _move_claimed_item_to_failed(manifest_path: Path, failed_dir: Path) -> None:
    failed_dir.mkdir(parents=True, exist_ok=True)
    body_path: Path | None = None
    with contextlib.suppress(Exception):
        with manifest_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        body_filename = str(metadata["body_filename"])
        if Path(body_filename).name == body_filename:
            body_path = manifest_path.parent / body_filename
    if body_path is not None and body_path.exists():
        body_path.replace(failed_dir / body_path.name)
    if manifest_path.exists():
        manifest_path.replace(failed_dir / manifest_path.name)


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
