"""Filesystem ingest consumer for sidecar-backed ingest.

The Rust sidecar can publish either per-request spool manifests or append-only
journal entries. Journal replay state is durable per consumer and stream, while
short-lived claim files remain the in-flight lock so partial writes are never
parsed and concurrent workers do not race the same stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from minimappr.api.binary_ingest import parse_binary_ingest_payload
from minimappr.api.spool_consumer_claims import (
    _claim_rust_dsp_manifests_oldest_first,
    _complete_claimed_item,
    _load_claimed_rust_dsp_item,
    _localized_manifest_item_is_cursor_covered,
    _move_claimed_item_to_failed,
)
from minimappr.api.spool_consumer_cleanup import _delete_files_older_than, _delete_orphan_ready_bodies
from minimappr.api.spool_consumer_strategy import IngestStorageStrategy
from minimappr.api.spool_consumer_types import (
    ClaimedIngestItem,
    ClaimedItem,
    ClaimedLocalizedNodeHeartbeatItem,
    ClaimedLocalizedRenderItem,
    IngestSpoolConfig,
    IngestStorageMode,
    SpoolEndpoint,
    SpoolProcessingSummary,
)
from minimappr.interfaces import IngestTransport
from minimappr.models import StoreForwardIngestRequest

logger = logging.getLogger(__name__)


class IngestSpoolConsumer:
    def __init__(self, *, config: IngestSpoolConfig, ingest_transport: IngestTransport) -> None:
        self._config = config
        self._ingest_transport = ingest_transport
        self._storage_strategy = IngestStorageStrategy(
            config=config,
            consume_rust_dsp_manifests=self._should_consume_rust_dsp_manifests(),
        )

    def ensure_directories(self) -> None:
        self._storage_strategy.ensure_directories()

    async def run_forever(self, *, worker_name: str) -> None:
        self.ensure_directories()
        startup_cleaned_processing = await asyncio.to_thread(
            self._storage_strategy.cleanup_startup_processing_directory,
            now_ns=time.time_ns(),
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
        cleanup_summary = await asyncio.to_thread(
            self._storage_strategy.run_cleanup_policy,
            now_ns=effective_now_ns,
        )

        processed = 0
        expired = 0
        failed = 0
        lease_owner = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        leased_processing_paths = await asyncio.to_thread(
            self._storage_strategy.lease_existing_processing_claims,
            lease_owner=lease_owner,
        )
        claimed_paths = leased_processing_paths + await asyncio.to_thread(
            self._storage_strategy.claim_ready_items,
        )
        for claimed_path in claimed_paths:
            try:
                item = await self._load_claimed_item(claimed_path)
                if isinstance(item, (ClaimedLocalizedRenderItem, ClaimedLocalizedNodeHeartbeatItem)) and await asyncio.to_thread(
                    _localized_manifest_item_is_cursor_covered,
                    item,
                    self._config.journal_consumer_state_path,
                    self._config.consumer_name,
                ):
                    await asyncio.to_thread(
                        _complete_claimed_item,
                        item,
                        self._config.journal_consumer_state_path,
                        self._config.consumer_name,
                        "processed",
                    )
                    processed += 1
                    continue
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
            cleaned_tmp=cleanup_summary.cleaned_tmp,
            cleaned_orphan_ready=cleanup_summary.cleaned_orphan_ready,
            cleaned_failed=cleanup_summary.cleaned_failed,
            cleaned_processing=cleanup_summary.cleaned_processing,
        )

    async def _load_claimed_item(self, claimed_path: Path) -> ClaimedItem:
        return await asyncio.to_thread(self._storage_strategy.load_claimed_item, claimed_path)

    async def _deliver_item(
        self,
        item: ClaimedIngestItem | ClaimedLocalizedRenderItem | ClaimedLocalizedNodeHeartbeatItem,
    ) -> None:
        if isinstance(item, ClaimedLocalizedRenderItem):
            await self._ingest_transport.deliver_localized_render(item.localized_render_request)
            return
        if isinstance(item, ClaimedLocalizedNodeHeartbeatItem):
            await self._ingest_transport.deliver_node_heartbeat(item.node)
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
        # In birdnet_hybrid_production mode, Rust DSP results used to be consumed via
        # disk manifests. With the SSE-based stream_consumer delivering results inline
        # (including BirdNET labels via ClassificationWorker -> dsp_result_tx), the disk
        # manifest path is no longer needed and would cause double-delivery to the fusion
        # node. Always return False so only the stream_consumer handles delivery.
        return False
