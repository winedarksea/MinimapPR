"""Storage strategy helpers for filesystem ingest spool consumption."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from minimappr.api.journal_state import JournalConsumerStateStore
from minimappr.api.spool_consumer_claims import _claim_journal_entries_oldest_first, _claim_manifest, _claim_rust_dsp_manifests_oldest_first, _lease_existing_processing_claims_oldest_first, _list_ready_manifests_oldest_first, _load_claimed_journal_item, _load_claimed_rust_dsp_item, _load_claimed_spool_item
from minimappr.api.spool_consumer_cleanup import SpoolCleanupSummary, cleanup_startup_processing_directory, run_spool_cleanup_policy
from minimappr.api.spool_consumer_types import ClaimedItem, IngestSpoolConfig


@dataclass(frozen=True, slots=True)
class IngestStorageStrategy:
    config: IngestSpoolConfig
    consume_rust_dsp_manifests: bool = False

    def ensure_directories(self) -> None:
        directories = [
            self.config.tmp_dir,
            self.config.ready_dir,
            self.config.processing_dir,
            self.config.failed_dir,
        ]
        if self.config.storage_mode == "journal":
            directories.append(self.config.journal_streams_dir)
            directories.append(self.config.journal_manifest_dir)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        if self.config.storage_mode == "journal":
            JournalConsumerStateStore(self.config.journal_consumer_state_path).ensure_initialized()

    def cleanup_startup_processing_directory(self, *, now_ns: int) -> int:
        return cleanup_startup_processing_directory(
            self.config.processing_dir,
            self.config.tmp_ttl_seconds,
            now_ns,
        )

    def run_cleanup_policy(self, *, now_ns: int) -> SpoolCleanupSummary:
        return run_spool_cleanup_policy(self.config, now_ns=now_ns)

    def lease_existing_processing_claims(self, *, lease_owner: str) -> list[Path]:
        return _lease_existing_processing_claims_oldest_first(
            self.config.processing_dir,
            lease_owner,
        )

    def claim_ready_items(self) -> list[Path]:
        if self.consume_rust_dsp_manifests:
            return _claim_rust_dsp_manifests_oldest_first(
                self.config.journal_manifest_dir,
                self.config.journal_streams_dir,
                self.config.processing_dir,
                self.config.journal_consumer_state_path,
                self.config.consumer_name,
                self.config.rust_dsp_claim_batch_size,
            )
        if self.config.storage_mode == "journal":
            return _claim_journal_entries_oldest_first(
                self.config.journal_streams_dir,
                self.config.processing_dir,
                self.config.journal_consumer_state_path,
                self.config.consumer_name,
            )

        manifests = _list_ready_manifests_oldest_first(self.config.ready_dir)
        claimed_paths: list[Path] = []
        for manifest_path in manifests:
            claimed_path = _claim_manifest(
                manifest_path,
                self.config.processing_dir,
            )
            if claimed_path is not None:
                claimed_paths.append(claimed_path)
        return claimed_paths

    def load_claimed_item(self, claimed_path: Path) -> ClaimedItem:
        if claimed_path.name.endswith(".rustdsp.json"):
            return _load_claimed_rust_dsp_item(
                claimed_path,
                self.config.journal_streams_dir,
            )
        if self.config.storage_mode == "journal":
            return _load_claimed_journal_item(
                claimed_path,
                self.config.journal_streams_dir,
            )
        return _load_claimed_spool_item(claimed_path)
