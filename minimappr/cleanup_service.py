"""Shared cleanup service for CLI and server housekeeping."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from minimappr.cleanup_policy import CleanupPolicy, RetentionActions, RetentionMatchCriteria, RetentionRule
from minimappr.config import Settings
from minimappr.interfaces import StorageBackend


_KNOWN_RUNTIME_CACHE_PATHS: tuple[Path, ...] = (
    Path("data/yamnet_class_map.csv"),
)


class CleanupService:
    """Coordinates destructive and policy-driven cleanup operations."""

    def __init__(self, *, settings: Settings, storage: StorageBackend | None = None) -> None:
        self._settings = settings
        self._storage = storage
        self._last_sqlite_maintenance_ns: int | None = None

    @property
    def settings(self) -> Settings:
        return self._settings

    def with_overrides(
        self,
        *,
        db_path: Path | None = None,
        snippet_dir: Path | None = None,
        artifact_dir: Path | None = None,
        training_dataset_dir: Path | None = None,
        retention_policy_path: Path | None = None,
    ) -> "CleanupService":
        return CleanupService(
            settings=replace(
                self._settings,
                db_path=db_path or self._settings.db_path,
                snippet_dir=snippet_dir or self._settings.snippet_dir,
                large_artifact_dir=artifact_dir or self._settings.large_artifact_dir,
                training_dataset_dir=training_dataset_dir or self._settings.training_dataset_dir,
                retention_policy_path=retention_policy_path or self._settings.retention_policy_path,
            ),
            storage=self._storage,
        )

    def load_policy(self, policy_path: Path | None = None) -> CleanupPolicy:
        path = policy_path or self._settings.retention_policy_path
        policy = CleanupPolicy.from_file(
            path,
            default_snippet_max_age_seconds=self._settings.retention_short_seconds,
            default_artifact_max_age_seconds=self._settings.retention_experiment_seconds,
        )
        # Classifier defaults are policy rules so user-managed label rules in
        # retention_policy.json remain an additive, backwards-compatible escape
        # hatch.  The alert rule is deliberately last: alert evidence wins.
        defaults = [
            RetentionRule("builtin-yamnet", -100, RetentionMatchCriteria(classifier="yamnet"), RetentionActions(snippet_max_age_seconds=self._settings.retention_yamnet_audio_seconds, artifact_max_age_seconds=self._settings.retention_yamnet_audio_seconds)),
            RetentionRule("builtin-birdnet", -100, RetentionMatchCriteria(classifier="birdnet"), RetentionActions(snippet_max_age_seconds=self._settings.retention_birdnet_audio_seconds, artifact_max_age_seconds=self._settings.retention_birdnet_audio_seconds)),
            RetentionRule("builtin-drone", -100, RetentionMatchCriteria(classifier="drone_head"), RetentionActions(snippet_max_age_seconds=self._settings.retention_drone_audio_seconds, artifact_max_age_seconds=self._settings.retention_drone_audio_seconds)),
            RetentionRule("builtin-alert", 10_000, RetentionMatchCriteria(trigger_source="alert"), RetentionActions(snippet_max_age_seconds=self._settings.retention_alert_audio_seconds, artifact_max_age_seconds=self._settings.retention_alert_audio_seconds)),
        ]
        return CleanupPolicy(version=policy.version, defaults=policy.defaults, rules=[*defaults, *policy.rules])

    async def run_partial_cleanup(
        self,
        *,
        now_ns: int | None = None,
        policy_path: Path | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        if self._storage is None:
            raise RuntimeError("Partial cleanup requires an initialized storage backend")
        effective_now_ns = now_ns if now_ns is not None else time.time_ns()
        policy = self.load_policy(policy_path)
        summary = await self._storage.cleanup_policy_managed_files(
            now_ns=effective_now_ns,
            policy=policy,
            dry_run=dry_run,
        )
        return {
            "mode": "partial",
            "dry_run": dry_run,
            "now_ns": effective_now_ns,
            "policy_path": str(policy_path or self._settings.retention_policy_path),
            "summary": summary,
        }

    async def run_housekeeping_cycle(
        self,
        *,
        now_ns: int | None = None,
        dry_run: bool = False,
        force_sqlite_maintenance: bool = False,
    ) -> dict[str, Any]:
        if self._storage is None:
            raise RuntimeError("Housekeeping requires an initialized storage backend")
        effective_now_ns = now_ns if now_ns is not None else time.time_ns()
        partial_summary = await self.run_partial_cleanup(
            now_ns=effective_now_ns,
            dry_run=dry_run,
        )
        retention_summary = await self._storage.cleanup_retention(
            now_ns=effective_now_ns,
            tier_ttls_seconds={
                # Detection rows are intentionally lightweight evidence.  Audio
                # is pruned by the policy pass above; retain the metadata here.
                "ephemeral": self._settings.retention_detection_metadata_seconds,
                "short": self._settings.retention_detection_metadata_seconds,
                "long": self._settings.retention_detection_metadata_seconds,
                "experiment": self._settings.retention_detection_metadata_seconds,
            },
            operational_ttls_seconds={
                "bit_reports": self._settings.retention_bit_reports_seconds,
                "pings": self._settings.retention_pings_seconds,
                "track_updates": self._settings.retention_track_updates_seconds,
                "alerts": self._settings.retention_alerts_seconds,
                "environment": self._settings.retention_environment_seconds,
                "dropped_tracks": self._settings.retention_dropped_tracks_seconds,
            },
        )
        transcript_cutoff_ns = effective_now_ns - int(
            self._settings.transcript_retention_seconds * 1_000_000_000
        )
        transcripts_deleted = 0
        if not dry_run:
            audio_paths = await self._storage.delete_transcripts_older_than(transcript_cutoff_ns)
            transcripts_deleted = len(audio_paths)
            for audio_path in audio_paths:
                with contextlib.suppress(OSError):
                    Path(audio_path).unlink(missing_ok=True)
        sqlite_maintenance_due = self._sqlite_maintenance_due(
            effective_now_ns,
            force=force_sqlite_maintenance,
        )
        maintenance_summary = None
        if sqlite_maintenance_due:
            maintenance_summary = await self._storage.run_sqlite_maintenance()
            self._last_sqlite_maintenance_ns = effective_now_ns
        return {
            "mode": "housekeeping",
            "dry_run": dry_run,
            "now_ns": effective_now_ns,
            "partial_cleanup": partial_summary["summary"],
            "retention_cleanup": retention_summary,
            "transcripts_deleted": transcripts_deleted,
            "sqlite_maintenance": maintenance_summary,
            "sqlite_maintenance_due": sqlite_maintenance_due,
            "sqlite_maintenance_interval_seconds": self._settings.sqlite_maintenance_interval_seconds,
        }

    async def run_full_cleanup(self, *, dry_run: bool = False) -> dict[str, Any]:
        if self._storage is not None:
            with contextlib.suppress(Exception):
                await self._storage.close()
        db_removed = False
        if self._settings.db_path.exists():
            if not dry_run:
                await asyncio.to_thread(self._settings.db_path.unlink, missing_ok=True)
            db_removed = True
        snippet_summary, speech_summary, artifact_summary, training_dataset_summary, spool_summary, cache_summaries = await asyncio.gather(
            _remove_tree_async(self._settings.snippet_dir, dry_run=dry_run),
            _remove_tree_async(self._settings.speech_audio_dir, dry_run=dry_run),
            _remove_tree_async(self._settings.large_artifact_dir, dry_run=dry_run),
            _remove_tree_async(self._settings.training_dataset_dir, dry_run=dry_run),
            _remove_tree_async(self._settings.ingest_spool_dir, dry_run=dry_run),
            _remove_known_runtime_cache_paths(dry_run=dry_run),
        )
        return {
            "mode": "full",
            "dry_run": dry_run,
            "db_removed": db_removed,
            "snippet_dir": snippet_summary,
            "speech_audio_dir": speech_summary,
            "artifact_dir": artifact_summary,
            "training_dataset_dir": training_dataset_summary,
            "spool_dir": spool_summary,
            "cache_paths": cache_summaries,
        }

    def _sqlite_maintenance_due(self, now_ns: int, *, force: bool) -> bool:
        if force:
            return True
        if self._last_sqlite_maintenance_ns is None:
            return True
        interval_ns = int(self._settings.sqlite_maintenance_interval_seconds * 1_000_000_000)
        return now_ns - self._last_sqlite_maintenance_ns >= interval_ns


async def _remove_known_runtime_cache_paths(*, dry_run: bool) -> list[dict[str, Any]]:
    return list(
        await asyncio.gather(
            *[
                _remove_path_async(path, dry_run=dry_run)
                for path in _KNOWN_RUNTIME_CACHE_PATHS
            ]
        )
    )


async def _remove_tree_async(path: Path, *, dry_run: bool) -> dict[str, Any]:
    return await asyncio.to_thread(_remove_tree, path, dry_run=dry_run)


async def _remove_path_async(path: Path, *, dry_run: bool) -> dict[str, Any]:
    return await asyncio.to_thread(_remove_path, path, dry_run=dry_run)


def _remove_tree(path: Path, *, dry_run: bool) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "removed": False, "file_count": 0}
    file_count = sum(1 for child in path.rglob("*") if child.is_file())
    if not dry_run:
        shutil.rmtree(path)
    return {"path": str(path), "removed": True, "file_count": file_count}


def _remove_path(path: Path, *, dry_run: bool) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "removed": False, "kind": "missing"}
    if path.is_dir():
        file_count = sum(1 for child in path.rglob("*") if child.is_file())
        if not dry_run:
            shutil.rmtree(path)
        return {"path": str(path), "removed": True, "kind": "directory", "file_count": file_count}
    if not dry_run:
        path.unlink(missing_ok=True)
    return {"path": str(path), "removed": True, "kind": "file"}
