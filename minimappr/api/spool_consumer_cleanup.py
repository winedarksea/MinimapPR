"""Cleanup policy helpers for filesystem ingest spool consumption."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from minimappr.api.spool_consumer_types import IngestSpoolConfig


@dataclass(frozen=True, slots=True)
class SpoolCleanupSummary:
    cleaned_tmp: int = 0
    cleaned_orphan_ready: int = 0
    cleaned_failed: int = 0
    cleaned_processing: int = 0


def cleanup_startup_processing_directory(
    processing_dir: Path,
    tmp_ttl_seconds: float,
    now_ns: int,
) -> int:
    return _delete_files_older_than(processing_dir, tmp_ttl_seconds, now_ns)


def run_spool_cleanup_policy(config: IngestSpoolConfig, *, now_ns: int) -> SpoolCleanupSummary:
    cleaned_tmp = 0
    cleaned_orphan_ready = 0
    if config.storage_mode == "spool":
        cleaned_tmp = _delete_files_older_than(
            config.tmp_dir,
            config.tmp_ttl_seconds,
            now_ns,
        )
        cleaned_orphan_ready = _delete_orphan_ready_bodies(
            config.ready_dir,
            config.tmp_ttl_seconds,
            now_ns,
        )

    cleaned_failed = _delete_files_older_than(
        config.failed_dir,
        config.failed_ttl_seconds,
        now_ns,
    )
    return SpoolCleanupSummary(
        cleaned_tmp=cleaned_tmp,
        cleaned_orphan_ready=cleaned_orphan_ready,
        cleaned_failed=cleaned_failed,
        cleaned_processing=0,
    )


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
