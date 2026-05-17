"""Shared cleanup service for CLI and server housekeeping."""

from __future__ import annotations

import contextlib
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from minimappr.cleanup_policy import CleanupPolicy
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

    @property
    def settings(self) -> Settings:
        return self._settings

    def with_overrides(
        self,
        *,
        db_path: Path | None = None,
        snippet_dir: Path | None = None,
        artifact_dir: Path | None = None,
        retention_policy_path: Path | None = None,
    ) -> "CleanupService":
        return CleanupService(
            settings=replace(
                self._settings,
                db_path=db_path or self._settings.db_path,
                snippet_dir=snippet_dir or self._settings.snippet_dir,
                large_artifact_dir=artifact_dir or self._settings.large_artifact_dir,
                retention_policy_path=retention_policy_path or self._settings.retention_policy_path,
            ),
            storage=self._storage,
        )

    def load_policy(self, policy_path: Path | None = None) -> CleanupPolicy:
        path = policy_path or self._settings.retention_policy_path
        return CleanupPolicy.from_file(
            path,
            default_snippet_max_age_seconds=self._settings.retention_short_seconds,
            default_artifact_max_age_seconds=self._settings.retention_experiment_seconds,
        )

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
                "ephemeral": self._settings.retention_ephemeral_seconds,
                "short": self._settings.retention_short_seconds,
                "long": self._settings.retention_long_seconds,
                "experiment": self._settings.retention_experiment_seconds,
            },
            operational_ttls_seconds={
                "ingested_frames": self._settings.retention_ingested_frames_seconds,
                "bit_reports": self._settings.retention_bit_reports_seconds,
                "pings": self._settings.retention_pings_seconds,
                "track_updates": self._settings.retention_track_updates_seconds,
                "alerts": self._settings.retention_alerts_seconds,
                "environment": self._settings.retention_environment_seconds,
                "dropped_tracks": self._settings.retention_dropped_tracks_seconds,
            },
        )
        maintenance_summary = await self._storage.run_sqlite_maintenance()
        return {
            "mode": "housekeeping",
            "dry_run": dry_run,
            "now_ns": effective_now_ns,
            "partial_cleanup": partial_summary["summary"],
            "retention_cleanup": retention_summary,
            "sqlite_maintenance": maintenance_summary,
        }

    async def run_full_cleanup(self, *, dry_run: bool = False) -> dict[str, Any]:
        if self._storage is not None:
            with contextlib.suppress(Exception):
                await self._storage.close()
        db_removed = False
        if self._settings.db_path.exists():
            if not dry_run:
                self._settings.db_path.unlink(missing_ok=True)
            db_removed = True
        snippet_summary = _remove_tree(self._settings.snippet_dir, dry_run=dry_run)
        artifact_summary = _remove_tree(self._settings.large_artifact_dir, dry_run=dry_run)
        cache_summaries = [
            _remove_path(path, dry_run=dry_run)
            for path in _KNOWN_RUNTIME_CACHE_PATHS
        ]
        return {
            "mode": "full",
            "dry_run": dry_run,
            "db_removed": db_removed,
            "snippet_dir": snippet_summary,
            "artifact_dir": artifact_summary,
            "cache_paths": cache_summaries,
        }


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
