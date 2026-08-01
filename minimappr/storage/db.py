"""SQLite persistence layer."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite
from pydantic import ValidationError

from minimappr.cleanup_policy import CleanupPolicy
from minimappr.models import (
    ContributorSummary,
    DetectionEvent,
    GeoPoint,
    LabelId,
    NodeCapability,
    NodeOverrides,
    NodeSpec,
    NodeType,
    TrackState,
)

logger = logging.getLogger(__name__)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _json_loads(raw: str | None, fallback: Any) -> Any:
    if raw is None:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _row_value(row: aiosqlite.Row, key: str, default: Any = None) -> Any:
    """Read a column from a sqlite Row, tolerating rows that predate the column."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    cleaned = cleaned.strip("-")
    return cleaned or "unknown"


def _round_digits(bin_deg: float) -> int:
    """Decimal places for SQLite ROUND() given a bin resolution in degrees.

    e.g. 0.0001° → 4, 0.01° → 2, 1° → 0. Clamped to [0, 6].
    """
    import math
    if bin_deg <= 0:
        return 4
    return max(0, min(6, int(round(-math.log10(bin_deg)))))


def _contributor_role_priority(roles: list[str]) -> int:
    if "source" in roles:
        return 0
    if "reference" in roles:
        return 1
    if "tdoa" in roles:
        return 2
    return 3


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._write_owner: asyncio.Task[Any] | None = None
        self._batch_depth = 0
        self.stale_recoveries_count = 0

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async def _open_and_configure() -> None:
            await self._open_connection()
            await self._configure_connection()

        try:
            await _open_and_configure()
        except sqlite3.OperationalError as exc:
            if not self._is_recoverable_open_error(exc) or not await self._recover_stale_sqlite_state(exc):
                raise
            await _open_and_configure()

        try:
            await self._initialize_schema_and_migrations()
        except sqlite3.OperationalError as exc:
            if not self._is_recoverable_open_error(exc) or not await self._recover_stale_sqlite_state(exc):
                raise
            await _open_and_configure()
            await self._initialize_schema_and_migrations()

        await self._ensure_incremental_auto_vacuum()

    async def _initialize_schema_and_migrations(self) -> None:
        db = self._require_db()

        # Receipt and heartbeat tables used to write at packet cadence.  Live
        # ingest now owns those concerns in bounded process memory, so discard
        # old diagnostic state as part of the forward-only schema migration.
        await db.execute("DROP TABLE IF EXISTS ingested_frames")
        await db.execute("DROP TABLE IF EXISTS node_audio_summaries")

        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                lat REAL,
                lon REAL,
                alt REAL,
                sensor_offsets_json TEXT NOT NULL,
                capabilities_json TEXT NOT NULL,
                mobility TEXT NOT NULL DEFAULT 'stationary',
                orientation_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL,
                properties_json TEXT NOT NULL DEFAULT '{}',
                transport_json TEXT NOT NULL DEFAULT '{}',
                capability_config_json TEXT NOT NULL DEFAULT '{}',
                safety_json TEXT NOT NULL DEFAULT '{}',
                permissions_json TEXT NOT NULL DEFAULT '{}',
                overrides_json TEXT NOT NULL DEFAULT '{}',
                origin TEXT NOT NULL DEFAULT 'ingest',
                last_seen_ns INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS node_position_estimator_states (
                node_id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
                version INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                updated_ns INTEGER NOT NULL
            );

            -- The resolved site origin, anchored once from a trusted GPS fix and then
            -- held. Single row (id=1). This is deliberately persisted rather than
            -- re-derived per process: every stored `position_m` is ENU relative to it,
            -- so two processes resolving it independently silently disagree about where
            -- every node and track is. See core/site_origin.py.
            CREATE TABLE IF NOT EXISTS site_origin (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                alt_m REAL NOT NULL,
                source TEXT NOT NULL,
                contributing_node_ids_json TEXT NOT NULL DEFAULT '[]',
                resolved_ns INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                sensor_id TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                source_type TEXT NOT NULL,
                toa_ns INTEGER NOT NULL,
                tor_ns INTEGER NOT NULL,
                time_quality TEXT NOT NULL,
                sample_rate_hz INTEGER,
                channel_index INTEGER,
                frame_sequence INTEGER,
                retention_tier TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_observations_toa ON observations(toa_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_observations_sensor ON observations(sensor_id, toa_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_observations_event_id ON observations(event_id);
            CREATE INDEX IF NOT EXISTS idx_observations_node_id ON observations(node_id);

            CREATE TABLE IF NOT EXISTS labels (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                parent_label_id TEXT,
                source TEXT NOT NULL,
                external_taxonomy TEXT,
                external_label_id TEXT,
                created_ns INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_labels_category ON labels(category);

            CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_node_id TEXT REFERENCES nodes(id) ON DELETE SET NULL,
                timestamp_ns INTEGER NOT NULL,
                toa_ns INTEGER NOT NULL,
                tor_ns INTEGER NOT NULL,
                time_quality TEXT NOT NULL,
                stale_ns INTEGER,
                report_window_start_ns INTEGER,
                report_window_end_ns INTEGER,
                reporting_modality TEXT NOT NULL DEFAULT 'localized',
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                lat REAL,
                lon REAL,
                alt REAL,
                covariance_json TEXT,
                confidence REAL NOT NULL,
                gdop REAL NOT NULL,
                label_id TEXT REFERENCES labels(id) ON DELETE SET NULL,
                label TEXT NOT NULL,
                label_category TEXT NOT NULL DEFAULT 'unknown',
                iff_category TEXT NOT NULL DEFAULT 'unknown',
                label_confidence REAL NOT NULL,
                received_level_db REAL,
                track_id TEXT REFERENCES tracks(id) ON DELETE SET NULL,
                reference_sensor TEXT NOT NULL,
                source_sensors_json TEXT NOT NULL,
                source_observation_ids_json TEXT NOT NULL DEFAULT '[]',
                zone_ids_json TEXT NOT NULL DEFAULT '[]',
                tdoa_json TEXT NOT NULL,
                classifier_scores_json TEXT NOT NULL,
                feature_summary_json TEXT NOT NULL,
                review_state TEXT NOT NULL DEFAULT 'unreviewed',
                review_label_id TEXT REFERENCES labels(id) ON DELETE SET NULL,
                review_label TEXT,
                review_label_category TEXT,
                review_notes TEXT,
                review_updated_ns INTEGER,
                promote_to_training INTEGER NOT NULL DEFAULT 0,
                training_example_kind TEXT,
                retention_tier TEXT NOT NULL DEFAULT 'short',
                snippet_path TEXT,
                snippet_expires_ns INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_detections_track ON detections(track_id, timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label_category, timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_detections_review_state ON detections(review_state, timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS ix_detections_retention_tier_created
                ON detections(retention_tier, timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_detections_reporting_window
                ON detections(source_node_id, label, report_window_start_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_detections_label_window
                ON detections(label, report_window_start_ns DESC);

            CREATE TABLE IF NOT EXISTS training_examples (
                detection_id TEXT PRIMARY KEY REFERENCES detections(id) ON DELETE CASCADE,
                label TEXT NOT NULL,
                label_category TEXT NOT NULL,
                example_kind TEXT NOT NULL CHECK(example_kind IN ('positive', 'negative')),
                audio_path TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                updated_ns INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_training_examples_updated ON training_examples(updated_ns DESC);

            CREATE TABLE IF NOT EXISTS tracks (
                id TEXT PRIMARY KEY,
                first_seen_ns INTEGER NOT NULL,
                last_seen_ns INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                lat REAL,
                lon REAL,
                alt REAL,
                covariance_json TEXT,
                vx REAL NOT NULL,
                vy REAL NOT NULL,
                vz REAL NOT NULL,
                label_id TEXT,
                label TEXT NOT NULL,
                label_category TEXT NOT NULL DEFAULT 'unknown',
                iff_category TEXT NOT NULL DEFAULT 'unknown',
                confidence REAL NOT NULL,
                update_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                capability_tier TEXT NOT NULL DEFAULT 'full_3d',
                tqi REAL NOT NULL DEFAULT 0.0
            );
            CREATE INDEX IF NOT EXISTS idx_tracks_last_seen ON tracks(last_seen_ns DESC);

            CREATE TABLE IF NOT EXISTS track_updates (
                id TEXT PRIMARY KEY,
                track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
                timestamp_ns INTEGER NOT NULL,
                event_id TEXT,
                update_type TEXT NOT NULL,
                status TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                lat REAL,
                lon REAL,
                alt REAL,
                vx REAL NOT NULL,
                vy REAL NOT NULL,
                vz REAL NOT NULL,
                covariance_json TEXT,
                tqi REAL NOT NULL,
                confidence REAL NOT NULL,
                label TEXT NOT NULL,
                label_category TEXT NOT NULL,
                detection_id TEXT REFERENCES detections(id) ON DELETE SET NULL,
                observation_ids_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_track_updates_track_ts ON track_updates(track_id, timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS ix_track_updates_retention_tier_created
                ON track_updates(timestamp_ns DESC);

            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                timestamp_ns INTEGER NOT NULL,
                rule_id TEXT,
                detection_id TEXT REFERENCES detections(id) ON DELETE SET NULL,
                track_id TEXT REFERENCES tracks(id) ON DELETE SET NULL,
                destination TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_ns INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp_ns DESC);

            CREATE TABLE IF NOT EXISTS pings (
                id TEXT PRIMARY KEY,
                timestamp_ns INTEGER NOT NULL,
                lat REAL,
                lon REAL,
                alt REAL,
                x REAL,
                y REAL,
                z REAL,
                ping_type TEXT NOT NULL,
                label_id TEXT REFERENCES labels(id) ON DELETE SET NULL,
                label TEXT,
                received_level_db REAL,
                source_detection_id TEXT REFERENCES detections(id) ON DELETE SET NULL,
                source_observation_id TEXT REFERENCES observations(id) ON DELETE SET NULL,
                source_track_id TEXT REFERENCES tracks(id) ON DELETE SET NULL,
                retention_tier TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pings_ts ON pings(timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_pings_type ON pings(ping_type, timestamp_ns DESC);

            CREATE TABLE IF NOT EXISTS large_artifacts (
                id TEXT PRIMARY KEY,
                artifact_type TEXT NOT NULL,
                path TEXT NOT NULL,
                retention_tier TEXT NOT NULL,
                source_detection_id TEXT REFERENCES detections(id) ON DELETE SET NULL,
                source_track_id TEXT REFERENCES tracks(id) ON DELETE SET NULL,
                metadata_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                expires_ns INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_large_artifacts_expires ON large_artifacts(expires_ns);
            CREATE INDEX IF NOT EXISTS ix_large_artifacts_retention_tier_expires
                ON large_artifacts(retention_tier, expires_ns, created_ns DESC);

            CREATE TABLE IF NOT EXISTS transcripts (
                id TEXT PRIMARY KEY,
                node_id TEXT,
                sensor_id TEXT,
                start_ns INTEGER NOT NULL,
                end_ns INTEGER NOT NULL,
                text TEXT NOT NULL,
                model TEXT,
                trigger_confidence REAL,
                audio_path TEXT,
                detection_id TEXT REFERENCES detections(id) ON DELETE SET NULL,
                created_ns INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_transcripts_created ON transcripts(created_ns DESC);

            CREATE TABLE IF NOT EXISTS zones (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                zone_type TEXT NOT NULL,
                polygon_geo_json TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS map_overlays (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                file_path TEXT NOT NULL,
                mime TEXT NOT NULL,
                bounds_json TEXT NOT NULL,
                opacity REAL NOT NULL,
                storey TEXT,
                enabled INTEGER NOT NULL,
                created_ns INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS environment (
                id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                timestamp_ns INTEGER NOT NULL,
                temperature_k REAL,
                pressure_pa REAL,
                humidity REAL,
                wind_speed_mps REAL,
                wind_dir_deg REAL,
                solar_lux REAL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_environment_ts ON environment(timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_environment_node_ts ON environment(node_id, timestamp_ns DESC);

            CREATE TABLE IF NOT EXISTS bit_reports (
                id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                report_type TEXT NOT NULL,
                overall_status TEXT NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                received_ns INTEGER NOT NULL,
                results_json TEXT NOT NULL,
                failure_codes_json TEXT NOT NULL DEFAULT '[]',
                firmware_version TEXT,
                uptime_seconds REAL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_bit_reports_node_ts ON bit_reports(node_id, timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_bit_reports_type ON bit_reports(report_type, timestamp_ns DESC);

            CREATE TABLE IF NOT EXISTS capture_sessions (
                session_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                stream_key TEXT NOT NULL,
                range_lease_id TEXT,
                start_time_ns INTEGER,
                end_time_ns INTEGER,
                first_frame_pts_ns INTEGER,
                work_dir TEXT NOT NULL,
                video_path TEXT,
                ambix_path TEXT,
                iamf_path TEXT,
                object_path TEXT,
                visual_path TEXT,
                youtube_path TEXT,
                error TEXT,
                created_ns INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_capture_sessions_created ON capture_sessions(created_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_capture_sessions_state ON capture_sessions(state);

            CREATE TABLE IF NOT EXISTS calibration_ground_truth (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES capture_sessions(session_id) ON DELETE CASCADE,
                label TEXT NOT NULL,
                label_category TEXT NOT NULL DEFAULT 'unknown',
                geometry_kind TEXT NOT NULL DEFAULT 'static',
                lat REAL,
                lon REAL,
                alt_m REAL,
                source_node_id TEXT,  -- node placed at the source; position resolved from it
                trajectory_json TEXT,  -- reserved for future GPX/CSV trajectory import
                start_ns INTEGER NOT NULL,
                end_ns INTEGER NOT NULL,
                notes TEXT,
                created_ns INTEGER NOT NULL,
                updated_ns INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_calibration_gt_session
                ON calibration_ground_truth(session_id, start_ns);

            -- NOTE: the legacy `effectors` / `effector_artifacts` tables are intentionally
            -- NOT created here. They are migrated into `nodes` / `node_artifacts` by
            -- `_migrate_effectors_into_nodes()` and renamed to `*_backup_v1`. Recreating
            -- them here would resurrect empty tables every boot and re-arm the migration.
            CREATE TABLE IF NOT EXISTS node_artifacts (
                id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                track_id TEXT,
                detection_id TEXT,
                kind TEXT NOT NULL DEFAULT 'snapshot',
                path TEXT NOT NULL,
                created_ns INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_node_artifacts_node ON node_artifacts(node_id, created_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_node_artifacts_track ON node_artifacts(track_id, created_ns DESC);
            """
        )

        # Migration-safe additions for older DBs.
        await self._migrate_spl_db_to_received_level()
        await self._ensure_columns(
            "nodes",
            {
                "lat": "REAL",
                "lon": "REAL",
                "alt": "REAL",
                "mobility": "TEXT NOT NULL DEFAULT 'stationary'",
                "orientation_json": "TEXT NOT NULL DEFAULT '{}'",
                "properties_json": "TEXT NOT NULL DEFAULT '{}'",
                "transport_json": "TEXT NOT NULL DEFAULT '{}'",
                "capability_config_json": "TEXT NOT NULL DEFAULT '{}'",
                "safety_json": "TEXT NOT NULL DEFAULT '{}'",
                "permissions_json": "TEXT NOT NULL DEFAULT '{}'",
                "overrides_json": "TEXT NOT NULL DEFAULT '{}'",
                "origin": "TEXT NOT NULL DEFAULT 'ingest'",
            },
        )
        await self._ensure_columns(
            "detections",
            {
                "event_id": "TEXT",
                "source_type": "TEXT NOT NULL DEFAULT 'raw_sensor'",
                "source_node_id": "TEXT",
                "toa_ns": "INTEGER",
                "tor_ns": "INTEGER",
                "time_quality": "TEXT NOT NULL DEFAULT 'free_running'",
                "stale_ns": "INTEGER",
                "report_window_start_ns": "INTEGER",
                "report_window_end_ns": "INTEGER",
                "reporting_modality": "TEXT NOT NULL DEFAULT 'localized'",
                "lat": "REAL",
                "lon": "REAL",
                "alt": "REAL",
                "covariance_json": "TEXT",
                "label_id": "TEXT",
                "label_category": "TEXT NOT NULL DEFAULT 'unknown'",
                "iff_category": "TEXT NOT NULL DEFAULT 'unknown'",
                "received_level_db": "REAL",
                "source_observation_ids_json": "TEXT NOT NULL DEFAULT '[]'",
                "zone_ids_json": "TEXT NOT NULL DEFAULT '[]'",
                "review_state": "TEXT NOT NULL DEFAULT 'unreviewed'",
                "review_label_id": "TEXT",
                "review_label": "TEXT",
                "review_label_category": "TEXT",
                "review_notes": "TEXT",
                "review_updated_ns": "INTEGER",
                "promote_to_training": "INTEGER NOT NULL DEFAULT 0",
                "training_example_kind": "TEXT",
                "retention_tier": "TEXT NOT NULL DEFAULT 'short'",
            },
        )
        await self._ensure_columns(
            "alerts",
            {
                "updated_ns": "INTEGER NOT NULL DEFAULT 0",
            },
        )
        await self._ensure_columns(
            "tracks",
            {
                "lat": "REAL",
                "lon": "REAL",
                "alt": "REAL",
                "covariance_json": "TEXT",
                "label_id": "TEXT",
                "label_category": "TEXT NOT NULL DEFAULT 'unknown'",
                "iff_category": "TEXT NOT NULL DEFAULT 'unknown'",
                "capability_tier": "TEXT NOT NULL DEFAULT 'full_3d'",
                "track_kind": "TEXT NOT NULL DEFAULT 'acoustic'",
            },
        )
        await self._ensure_columns(
            "environment",
            {
                "temperature_k": "REAL",
                "pressure_pa": "REAL",
                "humidity": "REAL",
                "wind_speed_mps": "REAL",
                "wind_dir_deg": "REAL",
                "solar_lux": "REAL",
                "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
            },
        )
        await self._ensure_columns(
            "capture_sessions",
            {
                "ambix_path": "TEXT",
                "object_path": "TEXT",
                "visual_path": "TEXT",
                "capture_kind": "TEXT NOT NULL DEFAULT 'recording'",
                "calibration_manifest_path": "TEXT",
            },
        )
        await self._ensure_columns(
            "training_examples",
            {
                "embedding_path": "TEXT",
            },
        )
        await self._ensure_columns(
            "calibration_ground_truth",
            {
                "source_node_id": "TEXT",
            },
        )
        await self._deduplicate_reporting_window_canonicals()
        await self._ensure_reporting_window_uniqueness_index()
        await self._migrate_effectors_into_nodes()
        await db.commit()

    async def _open_connection(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row

    async def _configure_connection(self) -> None:
        db = self._require_db()
        # Establish WAL mode first so the subsequent checkpoint has something to operate on.
        # Then drain any stale WAL frames left by an unclean prior shutdown — without this,
        # write-side PRAGMAs below can hit "database is locked" because SQLite still treats
        # the abandoned -wal/-shm pair as belonging to a live writer.
        #
        # Each PRAGMA that returns rows must have its cursor drained and closed before the
        # next statement, or SQLite reports SQLITE_LOCKED on the connection.
        async with db.execute("PRAGMA journal_mode=WAL;") as cursor:
            await cursor.fetchall()
        async with db.execute("PRAGMA wal_checkpoint(TRUNCATE);") as cursor:
            await cursor.fetchall()
        await db.execute("PRAGMA foreign_keys=ON;")
        await db.execute("PRAGMA auto_vacuum=INCREMENTAL;")
        await db.execute("PRAGMA synchronous=NORMAL;")

    async def _ensure_incremental_auto_vacuum(self) -> None:
        db = self._db
        if db is None:
            return
        row = await (await db.execute("PRAGMA auto_vacuum;")).fetchone()
        current_mode = int(row[0]) if row is not None else 0
        if current_mode == 2:
            return
        logger.info("Enabling incremental auto_vacuum for %s", self.db_path)
        await db.execute("PRAGMA auto_vacuum=INCREMENTAL;")
        await db.commit()
        await db.execute("VACUUM;")
        await db.commit()

    async def run_sqlite_maintenance(self) -> dict[str, int]:
        db = self._require_db()
        async with self._write_guard():
            freelist_before_row = await (await db.execute("PRAGMA freelist_count;")).fetchone()
            freelist_before = int(freelist_before_row[0]) if freelist_before_row is not None else 0
            await db.execute("PRAGMA incremental_vacuum;")
            await db.execute("PRAGMA optimize;")
            await self._commit_if_needed(db)
            freelist_after_row = await (await db.execute("PRAGMA freelist_count;")).fetchone()
            freelist_after = int(freelist_after_row[0]) if freelist_after_row is not None else 0
        return {
            "freelist_before": freelist_before,
            "freelist_after": freelist_after,
        }

    @staticmethod
    def _is_disk_io_error(exc: sqlite3.OperationalError) -> bool:
        return "disk I/O error" in str(exc)

    @staticmethod
    def _is_database_locked(exc: sqlite3.OperationalError) -> bool:
        return "database is locked" in str(exc)

    @classmethod
    def _is_recoverable_open_error(cls, exc: sqlite3.OperationalError) -> bool:
        return cls._is_disk_io_error(exc) or cls._is_database_locked(exc)

    @staticmethod
    def _file_size_or_none(path: Path) -> int | None:
        try:
            return path.stat().st_size
        except OSError:
            return None

    async def _recover_stale_sqlite_state(self, trigger_exc: sqlite3.OperationalError) -> bool:
        """Graduated recovery for stale WAL/SHM state from an unclean shutdown.

        Returns True if recovery succeeded and the caller should retry opening the
        database. False propagates the original error to the caller.

        Safety invariants:
          * The ``-shm`` file is never load-bearing; it is rebuilt from ``-wal`` on
            open, so deleting it cannot lose data.
          * The ``-wal`` file is only deleted when a checkpoint has proven it
            contains zero committed frames. Any path that would lose committed
            data returns False and re-raises instead.
        """
        if not self.db_path.exists():
            return False

        wal_path = Path(f"{self.db_path}-wal")
        shm_path = Path(f"{self.db_path}-shm")

        await self._close_connection()

        def _sizes() -> str:
            return (
                f"db={self._file_size_or_none(self.db_path)} "
                f"wal={self._file_size_or_none(wal_path)} "
                f"shm={self._file_size_or_none(shm_path)}"
            )

        logger.warning(
            "SQLite stale-state recovery starting for %s after %s (%s)",
            self.db_path,
            type(trigger_exc).__name__,
            _sizes(),
        )

        # Step A: fresh connection + TRUNCATE checkpoint.
        if await self._try_checkpoint_truncate():
            logger.warning("SQLite stale-state recovery step A succeeded for %s (%s)", self.db_path, _sizes())
            self.stale_recoveries_count += 1
            return True

        # Step B: delete -shm (always safe — rebuilt from -wal) and retry checkpoint.
        if shm_path.exists():
            try:
                shm_path.unlink()
            except OSError as exc:
                logger.warning("SQLite stale-state recovery could not remove %s: %s", shm_path, exc)
                return False
            if await self._try_checkpoint_truncate():
                logger.warning("SQLite stale-state recovery step B succeeded for %s (%s)", self.db_path, _sizes())
                self.stale_recoveries_count += 1
                return True

        # Step C: only if we can prove -wal has zero committed frames, delete it.
        if wal_path.exists():
            frames_in_wal = await self._wal_committed_frames()
            if frames_in_wal == 0:
                try:
                    wal_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("SQLite stale-state recovery could not remove %s: %s", wal_path, exc)
                    return False
                logger.warning(
                    "SQLite stale-state recovery step C succeeded for %s — removed empty WAL (%s)",
                    self.db_path,
                    _sizes(),
                )
                self.stale_recoveries_count += 1
                return True
            logger.error(
                "SQLite stale-state recovery refusing to delete -wal with %s committed frames for %s (%s)",
                frames_in_wal,
                self.db_path,
                _sizes(),
            )

        return False

    async def _close_connection(self) -> None:
        db = self._db
        if db is None:
            return
        self._db = None
        try:
            await db.close()
        except Exception as exc:  # noqa: BLE001 — defensive: closing a half-broken conn
            logger.debug("Ignoring error while closing stale SQLite connection: %s", exc)

    async def _try_checkpoint_truncate(self) -> bool:
        """Open a short-lived connection and attempt a TRUNCATE checkpoint."""
        try:
            conn = await aiosqlite.connect(self.db_path, timeout=2.0)
        except sqlite3.OperationalError as exc:
            logger.debug("Recovery checkpoint: could not open %s: %s", self.db_path, exc)
            return False
        try:
            try:
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                await conn.commit()
            except sqlite3.OperationalError as exc:
                logger.debug("Recovery checkpoint failed for %s: %s", self.db_path, exc)
                return False
        finally:
            try:
                await conn.close()
            except Exception:  # noqa: BLE001
                pass
        return True

    async def _wal_committed_frames(self) -> int | None:
        """Return the number of committed frames currently in the WAL, or None on error.

        Uses ``PRAGMA wal_checkpoint(PASSIVE)`` which returns ``(busy, log, checkpointed)``;
        ``log`` is the count of frames in the WAL after the call (0 means the WAL contains
        no committed data).
        """
        try:
            conn = await aiosqlite.connect(self.db_path, timeout=2.0)
        except sqlite3.OperationalError as exc:
            logger.debug("Recovery probe: could not open %s: %s", self.db_path, exc)
            return None
        try:
            try:
                await conn.execute("PRAGMA journal_mode=WAL;")
                cursor = await conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
                row = await cursor.fetchone()
                await cursor.close()
            except sqlite3.OperationalError as exc:
                logger.debug("Recovery probe failed for %s: %s", self.db_path, exc)
                return None
            if row is None or len(row) < 2:
                return None
            try:
                return int(row[1])
            except (TypeError, ValueError):
                return None
        finally:
            try:
                await conn.close()
            except Exception:  # noqa: BLE001
                pass

    async def _deduplicate_reporting_window_canonicals(self) -> None:
        """Keep one canonical row per reporting key before enforcing uniqueness."""
        db = self._require_db()
        await db.execute(
            """
            WITH ranked AS (
                SELECT
                    rowid,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            COALESCE(source_node_id, '__none__'),
                            lower(label),
                            report_window_start_ns,
                            report_window_end_ns
                        ORDER BY
                            CASE WHEN reporting_modality = 'localized' THEN 1 ELSE 0 END DESC,
                            timestamp_ns DESC,
                            rowid DESC
                    ) AS duplicate_rank
                FROM detections
                WHERE report_window_start_ns IS NOT NULL
                  AND report_window_end_ns IS NOT NULL
            )
            DELETE FROM detections
            WHERE rowid IN (
                SELECT rowid FROM ranked WHERE duplicate_rank > 1
            )
            """
        )

    async def _ensure_reporting_window_uniqueness_index(self) -> None:
        """Enforce one canonical detection per node/label/reporting window."""
        db = self._require_db()
        await db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_detections_reporting_window_canonical
                ON detections(
                    COALESCE(source_node_id, '__none__'),
                    lower(label),
                    report_window_start_ns,
                    report_window_end_ns
                )
            WHERE report_window_start_ns IS NOT NULL
              AND report_window_end_ns IS NOT NULL
            """
        )

    async def _table_exists(self, table: str) -> bool:
        db = self._require_db()
        row = await (
            await db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
                (table,),
            )
        ).fetchone()
        return row is not None

    async def _migrate_effectors_into_nodes(self) -> None:
        db = self._require_db()
        if not await self._table_exists("effectors"):
            return
        valid_capabilities = {capability.value for capability in NodeCapability}
        rows = await (await db.execute("SELECT * FROM effectors ORDER BY id ASC")).fetchall()
        for row in rows:
            raw_capabilities = _json_loads(row["capabilities_json"], [])
            capabilities = [
                str(capability)
                for capability in raw_capabilities
                if str(capability) in valid_capabilities
            ]
            if NodeCapability.PTZ_CAMERA.value not in capabilities:
                capabilities.append(NodeCapability.PTZ_CAMERA.value)
            metadata = _json_loads(row["metadata_json"], {})
            properties = _json_loads(row["properties_json"], {})
            safety = {}
            if isinstance(properties, dict) and isinstance(properties.get("safety"), dict):
                safety = dict(properties["safety"])
            existing = await (
                await db.execute("SELECT * FROM nodes WHERE id = ? LIMIT 1", (row["id"],))
            ).fetchone()
            if existing is None:
                await db.execute(
                    """
                    INSERT INTO nodes (
                        id, node_type, x, y, z, lat, lon, alt,
                        sensor_offsets_json, capabilities_json, mobility,
                        orientation_json, metadata_json, properties_json,
                        transport_json, capability_config_json, safety_json,
                        permissions_json, overrides_json, origin, last_seen_ns
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'stationary', ?, ?, ?, ?, '{}', ?, '{}', '{}', 'operator', ?)
                    """,
                    (
                        row["id"],
                        NodeType.POINT.value,
                        row["x"],
                        row["y"],
                        row["z"],
                        row["lat"],
                        row["lon"],
                        row["alt"],
                        _json_dumps([[0.0, 0.0, 0.0]]),
                        _json_dumps(capabilities),
                        _json_dumps(
                            {
                                "yaw_deg": row["yaw_deg"],
                                "pitch_deg": row["pitch_deg"],
                                "roll_deg": 0.0,
                            }
                        ),
                        _json_dumps(metadata if isinstance(metadata, dict) else {}),
                        _json_dumps(properties if isinstance(properties, dict) else {}),
                        row["transport_json"],
                        _json_dumps(safety),
                        row["last_seen_ns"],
                    ),
                )
            else:
                existing_capabilities = _json_loads(existing["capabilities_json"], [])
                merged_capabilities = sorted({*map(str, existing_capabilities), *capabilities})
                await db.execute(
                    """
                    UPDATE nodes
                    SET capabilities_json = ?,
                        transport_json = CASE WHEN transport_json = '{}' THEN ? ELSE transport_json END,
                        safety_json = CASE WHEN safety_json = '{}' THEN ? ELSE safety_json END
                    WHERE id = ?
                    """,
                    (
                        _json_dumps(merged_capabilities),
                        row["transport_json"],
                        _json_dumps(safety),
                        row["id"],
                    ),
                )

        if await self._table_exists("effector_artifacts"):
            await db.execute(
                """
                INSERT OR IGNORE INTO node_artifacts (
                    id, node_id, track_id, detection_id, kind, path, created_ns
                )
                SELECT id, effector_id, track_id, detection_id, kind, path, created_ns
                FROM effector_artifacts
                """
            )
        if not await self._table_exists("effectors_backup_v1"):
            await db.execute("ALTER TABLE effectors RENAME TO effectors_backup_v1")
        if await self._table_exists("effector_artifacts") and not await self._table_exists("effector_artifacts_backup_v1"):
            await db.execute("ALTER TABLE effector_artifacts RENAME TO effector_artifacts_backup_v1")

    async def _table_columns(self, table: str) -> set[str]:
        db = self._require_db()
        rows = await (await db.execute(f"PRAGMA table_info({table})")).fetchall()
        return {row["name"] for row in rows}

    async def _migrate_spl_db_to_received_level(self) -> None:
        """Rename the legacy ``spl_db`` column to ``received_level_db``.

        The value was never sound pressure level — it is 20*log10(rms) plus the
        node's calibration gain offset, so it is a level relative to full scale and
        normally negative. The old name invited rules written against real SPL
        thresholds, which could never match.

        Uses ALTER TABLE ... RENAME COLUMN so existing rows keep their values; the
        additive ``_ensure_columns`` pass would otherwise add an empty
        ``received_level_db`` beside the populated ``spl_db`` and orphan the data.
        Must therefore run before that pass. No-op once migrated.
        """
        db = self._require_db()
        for table in ("detections", "pings"):
            columns = await self._table_columns(table)
            if "spl_db" not in columns or "received_level_db" in columns:
                continue
            await db.execute(
                f"ALTER TABLE {table} RENAME COLUMN spl_db TO received_level_db"
            )

    async def _ensure_columns(self, table: str, expected: dict[str, str]) -> None:
        db = self._require_db()
        current = await self._table_columns(table)
        for name, definition in expected.items():
            if name in current:
                continue
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    @asynccontextmanager
    async def _write_guard(self):
        current = asyncio.current_task()
        if current is not None and self._write_owner is current:
            yield
            return
        async with self._write_lock:
            self._write_owner = asyncio.current_task()
            try:
                yield
            finally:
                self._write_owner = None

    async def _commit_if_needed(self, db: aiosqlite.Connection) -> None:
        if self._batch_depth == 0:
            await db.commit()

    @staticmethod
    def _in_transaction(db: aiosqlite.Connection) -> bool:
        try:
            return bool(db.in_transaction)
        except (ValueError, AttributeError):
            # The connection was closed underneath us; there is no transaction
            # to reason about and the caller's next statement will say so.
            return False

    @staticmethod
    async def _rollback_quietly(db: aiosqlite.Connection) -> None:
        try:
            await db.rollback()
        except Exception:
            # A rollback that fails leaves the transaction open, which the next
            # batch clears. It must never mask the error that prompted it.
            logger.exception("SQLite rollback failed")

    async def _discard_stale_transaction(self, db: aiosqlite.Connection) -> None:
        """Clear a transaction left open by a failed or interrupted commit.

        sqlite3 keeps its implicit DML transaction open when the commit that
        should have closed it raises -- ``database is locked`` is the usual
        cause once a second process shares the file. ``BEGIN`` on top of that
        raises "cannot start a transaction within a transaction" for every
        batch that follows, so one transient lock wedges the connection for the
        life of the process. Whatever was pending is already unrecoverable, so
        drop it loudly and let the caller start clean.
        """
        if not self._in_transaction(db):
            return
        logger.warning(
            "Discarding a SQLite transaction left open by an earlier failed commit; "
            "uncommitted writes from that batch are lost",
            extra={"db_path": str(self.db_path)},
        )
        await self._rollback_quietly(db)

    @asynccontextmanager
    async def begin_batch(self):
        db = self._require_db()
        async with self._write_guard():
            owns_transaction = self._batch_depth == 0
            if owns_transaction:
                await self._discard_stale_transaction(db)
                # If the rollback above could not clear it either, join the open
                # transaction rather than raising: a batch that commits someone
                # else's orphaned writes beats a connection that never works again.
                if not self._in_transaction(db):
                    await db.execute("BEGIN")
            self._batch_depth += 1
            try:
                yield
            except BaseException:
                # BaseException, not Exception: a cancelled batch used to skip
                # this branch entirely, pinning _batch_depth above zero so
                # _commit_if_needed silently dropped every later write.
                self._batch_depth = max(0, self._batch_depth - 1)
                if owns_transaction:
                    await self._rollback_quietly(db)
                raise
            else:
                self._batch_depth = max(0, self._batch_depth - 1)
                if owns_transaction:
                    await db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def upsert_node(
        self,
        spec: NodeSpec,
        last_seen_ns: int,
        position_geo: GeoPoint | None = None,
    ) -> None:
        db = self._require_db()
        if spec.position_m is None:
            raise ValueError("NodeSpec.position_m is required for persistence")
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO nodes (
                    id, node_type, x, y, z, lat, lon, alt,
                    sensor_offsets_json, capabilities_json, mobility,
                    orientation_json, metadata_json, properties_json,
                    transport_json, capability_config_json, safety_json,
                    permissions_json, overrides_json, origin, last_seen_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', '{}', '{}', '{}', '{}', 'ingest', ?)
                ON CONFLICT(id) DO UPDATE SET
                    node_type=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.node_type
                        ELSE nodes.node_type
                    END,
                    x=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.x
                        ELSE nodes.x
                    END,
                    y=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.y
                        ELSE nodes.y
                    END,
                    z=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.z
                        ELSE nodes.z
                    END,
                    lat=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.lat
                        ELSE nodes.lat
                    END,
                    lon=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.lon
                        ELSE nodes.lon
                    END,
                    alt=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.alt
                        ELSE nodes.alt
                    END,
                    sensor_offsets_json=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.sensor_offsets_json
                        ELSE nodes.sensor_offsets_json
                    END,
                    capabilities_json=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.capabilities_json
                        ELSE nodes.capabilities_json
                    END,
                    mobility=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.mobility
                        ELSE nodes.mobility
                    END,
                    orientation_json=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.orientation_json
                        ELSE nodes.orientation_json
                    END,
                    metadata_json=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.metadata_json
                        ELSE nodes.metadata_json
                    END,
                    properties_json=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.properties_json
                        ELSE nodes.properties_json
                    END,
                    last_seen_ns=CASE
                        WHEN excluded.last_seen_ns > nodes.last_seen_ns THEN excluded.last_seen_ns
                        ELSE nodes.last_seen_ns
                    END
                """,
                (
                    spec.id,
                    spec.node_type.value,
                    spec.position_m[0],
                    spec.position_m[1],
                    spec.position_m[2],
                    position_geo.lat if position_geo is not None else None,
                    position_geo.lon if position_geo is not None else None,
                    position_geo.alt_m if position_geo is not None else None,
                    _json_dumps(spec.sensor_offsets_m),
                    _json_dumps(spec.capabilities),
                    spec.mobility,
                    _json_dumps(spec.orientation.model_dump(mode="json")),
                    _json_dumps(spec.metadata),
                    _json_dumps(spec.properties),
                    last_seen_ns,
                ),
            )
            await self._commit_if_needed(db)

    # ------------------------------------------------------------------
    # BIT Reports
    # ------------------------------------------------------------------

    async def insert_bit_report(
        self,
        *,
        report_id: str,
        node_id: str,
        report_type: str,
        overall_status: str,
        timestamp_ns: int,
        received_ns: int,
        results_json: str,
        failure_codes_json: str,
        firmware_version: str | None,
        uptime_seconds: float | None,
        metadata_json: str,
    ) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO bit_reports (
                    id, node_id, report_type, overall_status,
                    timestamp_ns, received_ns, results_json,
                    failure_codes_json, firmware_version,
                    uptime_seconds, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    node_id,
                    report_type,
                    overall_status,
                    timestamp_ns,
                    received_ns,
                    results_json,
                    failure_codes_json,
                    firmware_version,
                    uptime_seconds,
                    metadata_json,
                ),
            )
            await self._commit_if_needed(db)

    async def list_bit_reports(
        self,
        node_id: str | None = None,
        report_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        db = self._require_db()
        clauses: list[str] = []
        params: list[object] = []
        if node_id is not None:
            clauses.append("node_id = ?")
            params.append(node_id)
        if report_type is not None:
            clauses.append("report_type = ?")
            params.append(report_type)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = await (
            await db.execute(
                f"SELECT * FROM bit_reports{where} ORDER BY timestamp_ns DESC LIMIT ?",
                (*params, limit),
            )
        ).fetchall()
        result: list[dict] = []
        for row in rows:
            result.append(
                {
                    "id": row["id"],
                    "node_id": row["node_id"],
                    "report_type": row["report_type"],
                    "overall_status": row["overall_status"],
                    "timestamp_ns": row["timestamp_ns"],
                    "received_ns": row["received_ns"],
                    "results": _json_loads(row["results_json"], []),
                    "failure_codes": _json_loads(row["failure_codes_json"], []),
                    "firmware_version": row["firmware_version"],
                    "uptime_seconds": row["uptime_seconds"],
                    "metadata": _json_loads(row["metadata_json"], {}),
                }
            )
        return result

    async def latest_bit_report_per_type(self, node_id: str) -> list[dict]:
        """Return the single newest BIT report per report_type for a node."""
        db = self._require_db()
        rows = await (
            await db.execute(
                """
                SELECT b.* FROM bit_reports b
                INNER JOIN (
                    SELECT report_type, MAX(timestamp_ns) AS max_ts
                    FROM bit_reports
                    WHERE node_id = ?
                    GROUP BY report_type
                ) latest ON b.report_type = latest.report_type
                    AND b.timestamp_ns = latest.max_ts
                    AND b.node_id = ?
                ORDER BY b.report_type
                """,
                (node_id, node_id),
            )
        ).fetchall()
        return [
            {
                "id": row["id"],
                "node_id": row["node_id"],
                "report_type": row["report_type"],
                "overall_status": row["overall_status"],
                "timestamp_ns": row["timestamp_ns"],
                "received_ns": row["received_ns"],
                "results": _json_loads(row["results_json"], []),
                "failure_codes": _json_loads(row["failure_codes_json"], []),
                "firmware_version": row["firmware_version"],
                "uptime_seconds": row["uptime_seconds"],
                "metadata": _json_loads(row["metadata_json"], {}),
            }
            for row in rows
        ]

    async def insert_observation(
        self,
        *,
        node_id: str,
        sensor_id: str,
        sensor_type: str,
        source_type: str,
        toa_ns: int,
        tor_ns: int,
        time_quality: str,
        sample_rate_hz: int | None,
        channel_index: int | None,
        frame_sequence: int | None,
        retention_tier: str = "short",
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> str:
        db = self._require_db()
        observation_id = event_id or f"obs-{uuid.uuid4().hex[:16]}"
        payload = metadata or {}
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO observations (
                    id, event_id, node_id, sensor_id, sensor_type, source_type,
                    toa_ns, tor_ns, time_quality, sample_rate_hz, channel_index,
                    frame_sequence, retention_tier, metadata_json, created_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    observation_id,
                    node_id,
                    sensor_id,
                    sensor_type,
                    source_type,
                    toa_ns,
                    tor_ns,
                    time_quality,
                    sample_rate_hz,
                    channel_index,
                    frame_sequence,
                    retention_tier,
                    _json_dumps(payload),
                    tor_ns,
                ),
            )
            await self._commit_if_needed(db)
        return observation_id

    async def upsert_label(
        self,
        *,
        name: str,
        category: str,
        source: str = "runtime",
        parent_label_id: LabelId | None = None,
        external_taxonomy: str | None = None,
        external_label_id: str | None = None,
        created_ns: int,
    ) -> LabelId:
        db = self._require_db()
        label_id = f"lbl-{_slugify(name)[:48]}"
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO labels (
                    id, name, category, parent_label_id, source,
                    external_taxonomy, external_label_id, created_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    category=excluded.category,
                    parent_label_id=COALESCE(excluded.parent_label_id, labels.parent_label_id),
                    source=excluded.source,
                    external_taxonomy=COALESCE(excluded.external_taxonomy, labels.external_taxonomy),
                    external_label_id=COALESCE(excluded.external_label_id, labels.external_label_id)
                """,
                (
                    label_id,
                    name,
                    category,
                    parent_label_id,
                    source,
                    external_taxonomy,
                    external_label_id,
                    created_ns,
                ),
            )
            await self._commit_if_needed(db)
        return label_id

    async def insert_detection(
        self,
        detection: DetectionEvent,
        snippet_path: str | None,
        snippet_expires_ns: int | None,
        retention_tier: str = "short",
    ) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO detections (
                    id, event_id, source_type, source_node_id,
                    timestamp_ns, toa_ns, tor_ns, time_quality, stale_ns,
                    report_window_start_ns, report_window_end_ns, reporting_modality,
                    x, y, z, lat, lon, alt, covariance_json,
                    confidence, gdop, label_id, label, label_category, iff_category,
                    label_confidence, received_level_db, track_id,
                    reference_sensor, source_sensors_json, source_observation_ids_json,
                    zone_ids_json, tdoa_json, classifier_scores_json, feature_summary_json,
                    retention_tier, snippet_path, snippet_expires_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    detection.id,
                    detection.event_id or detection.id,
                    detection.source_type,
                    detection.source_node_id,
                    detection.timestamp_ns,
                    detection.toa_ns,
                    detection.tor_ns,
                    detection.time_quality.value,
                    detection.stale_ns,
                    detection.report_window_start_ns,
                    detection.report_window_end_ns,
                    detection.reporting_modality,
                    detection.position_m[0],
                    detection.position_m[1],
                    detection.position_m[2],
                    detection.position_geo.lat if detection.position_geo else None,
                    detection.position_geo.lon if detection.position_geo else None,
                    detection.position_geo.alt_m if detection.position_geo else None,
                    _json_dumps(detection.position_covariance_m2)
                    if detection.position_covariance_m2 is not None
                    else None,
                    detection.confidence,
                    detection.gdop,
                    detection.label_id,
                    detection.label,
                    detection.label_category,
                    detection.iff_category,
                    detection.label_confidence,
                    detection.received_level_db,
                    detection.track_id,
                    detection.reference_sensor,
                    _json_dumps(detection.source_sensors),
                    _json_dumps(detection.source_observation_ids),
                    _json_dumps(detection.zone_ids),
                    _json_dumps(detection.tdoa_s),
                    _json_dumps(detection.classifier_scores),
                    _json_dumps(detection.feature_summary),
                    retention_tier,
                    snippet_path,
                    snippet_expires_ns,
                ),
            )
            await self._commit_if_needed(db)

    async def update_detection(
        self,
        detection: DetectionEvent,
        snippet_path: str | None,
        snippet_expires_ns: int | None,
        retention_tier: str = "short",
    ) -> bool:
        db = self._require_db()
        async with self._write_guard():
            cursor = await db.execute(
                """
                UPDATE detections
                SET event_id = ?, source_type = ?, source_node_id = ?,
                    timestamp_ns = ?, toa_ns = ?, tor_ns = ?, time_quality = ?, stale_ns = ?,
                    report_window_start_ns = ?, report_window_end_ns = ?, reporting_modality = ?,
                    x = ?, y = ?, z = ?, lat = ?, lon = ?, alt = ?, covariance_json = ?,
                    confidence = ?, gdop = ?, label_id = ?, label = ?, label_category = ?, iff_category = ?,
                    label_confidence = ?, received_level_db = ?, track_id = ?,
                    reference_sensor = ?, source_sensors_json = ?, source_observation_ids_json = ?,
                    zone_ids_json = ?, tdoa_json = ?, classifier_scores_json = ?, feature_summary_json = ?,
                    retention_tier = ?, snippet_path = COALESCE(?, snippet_path),
                    snippet_expires_ns = COALESCE(?, snippet_expires_ns)
                WHERE id = ?
                """,
                (
                    detection.event_id or detection.id,
                    detection.source_type,
                    detection.source_node_id,
                    detection.timestamp_ns,
                    detection.toa_ns,
                    detection.tor_ns,
                    detection.time_quality.value,
                    detection.stale_ns,
                    detection.report_window_start_ns,
                    detection.report_window_end_ns,
                    detection.reporting_modality,
                    detection.position_m[0],
                    detection.position_m[1],
                    detection.position_m[2],
                    detection.position_geo.lat if detection.position_geo else None,
                    detection.position_geo.lon if detection.position_geo else None,
                    detection.position_geo.alt_m if detection.position_geo else None,
                    _json_dumps(detection.position_covariance_m2)
                    if detection.position_covariance_m2 is not None
                    else None,
                    detection.confidence,
                    detection.gdop,
                    detection.label_id,
                    detection.label,
                    detection.label_category,
                    detection.iff_category,
                    detection.label_confidence,
                    detection.received_level_db,
                    detection.track_id,
                    detection.reference_sensor,
                    _json_dumps(detection.source_sensors),
                    _json_dumps(detection.source_observation_ids),
                    _json_dumps(detection.zone_ids),
                    _json_dumps(detection.tdoa_s),
                    _json_dumps(detection.classifier_scores),
                    _json_dumps(detection.feature_summary),
                    retention_tier,
                    snippet_path,
                    snippet_expires_ns,
                    detection.id,
                ),
            )
            await self._commit_if_needed(db)
        return bool(cursor.rowcount)

    async def find_detection_for_reporting_window(
        self,
        *,
        source_node_id: str | None,
        label: str,
        report_window_start_ns: int,
        report_window_end_ns: int,
        any_node: bool = False,
    ) -> dict[str, Any] | None:
        db = self._require_db()
        if any_node:
            # Site-wide lookup: ignore node keying entirely; prefer localized
            # detections so an omni row never masks a localized one elsewhere.
            row = await (
                await db.execute(
                    """
                    SELECT * FROM detections
                    WHERE label = ?
                      AND report_window_start_ns = ?
                      AND report_window_end_ns = ?
                    ORDER BY CASE WHEN reporting_modality = 'localized' THEN 0 ELSE 1 END,
                             timestamp_ns DESC
                    LIMIT 1
                    """,
                    (label, report_window_start_ns, report_window_end_ns),
                )
            ).fetchone()
        elif source_node_id is None:
            row = await (
                await db.execute(
                    """
                    SELECT * FROM detections
                    WHERE source_node_id IS NULL
                      AND label = ?
                      AND report_window_start_ns = ?
                      AND report_window_end_ns = ?
                    ORDER BY timestamp_ns DESC
                    LIMIT 1
                    """,
                    (label, report_window_start_ns, report_window_end_ns),
                )
            ).fetchone()
        else:
            row = await (
                await db.execute(
                    """
                    SELECT * FROM detections
                    WHERE source_node_id = ?
                      AND label = ?
                      AND report_window_start_ns = ?
                      AND report_window_end_ns = ?
                    ORDER BY timestamp_ns DESC
                    LIMIT 1
                    """,
                    (source_node_id, label, report_window_start_ns, report_window_end_ns),
                )
            ).fetchone()
        if row is None:
            return None
        return self._row_to_detection(row).model_dump(mode="json")

    async def insert_track_update(
        self,
        *,
        track: TrackState,
        timestamp_ns: int,
        event_id: str | None,
        update_type: str,
        detection_id: str | None,
        observation_ids: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        db = self._require_db()
        update_id = f"tup-{uuid.uuid4().hex[:16]}"
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO track_updates (
                    id, track_id, timestamp_ns, event_id, update_type, status,
                    x, y, z, lat, lon, alt, vx, vy, vz, covariance_json,
                    tqi, confidence, label, label_category,
                    detection_id, observation_ids_json, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    update_id,
                    track.id,
                    timestamp_ns,
                    event_id,
                    update_type,
                    track.status,
                    track.position_m[0],
                    track.position_m[1],
                    track.position_m[2],
                    track.position_geo.lat if track.position_geo else None,
                    track.position_geo.lon if track.position_geo else None,
                    track.position_geo.alt_m if track.position_geo else None,
                    track.velocity_mps[0],
                    track.velocity_mps[1],
                    track.velocity_mps[2],
                    _json_dumps(track.position_covariance_m2)
                    if track.position_covariance_m2 is not None
                    else None,
                    track.tqi,
                    track.confidence,
                    track.label,
                    track.label_category,
                    detection_id,
                    _json_dumps(observation_ids),
                    _json_dumps(metadata or {}),
                ),
            )
            await self._commit_if_needed(db)
        return update_id

    async def upsert_track(self, track: TrackState) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO tracks (
                    id, first_seen_ns, last_seen_ns,
                    x, y, z, lat, lon, alt, covariance_json,
                    vx, vy, vz,
                    label_id, label, label_category, iff_category,
                    confidence, update_count, status, capability_tier, tqi, track_kind
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_seen_ns=excluded.last_seen_ns,
                    x=excluded.x,
                    y=excluded.y,
                    z=excluded.z,
                    lat=excluded.lat,
                    lon=excluded.lon,
                    alt=excluded.alt,
                    covariance_json=excluded.covariance_json,
                    vx=excluded.vx,
                    vy=excluded.vy,
                    vz=excluded.vz,
                    label_id=excluded.label_id,
                    label=excluded.label,
                    label_category=excluded.label_category,
                    iff_category=excluded.iff_category,
                    confidence=excluded.confidence,
                    update_count=excluded.update_count,
                    status=excluded.status,
                    capability_tier=excluded.capability_tier,
                    tqi=excluded.tqi,
                    track_kind=excluded.track_kind
                """,
                (
                    track.id,
                    track.first_seen_ns,
                    track.last_seen_ns,
                    track.position_m[0],
                    track.position_m[1],
                    track.position_m[2],
                    track.position_geo.lat if track.position_geo else None,
                    track.position_geo.lon if track.position_geo else None,
                    track.position_geo.alt_m if track.position_geo else None,
                    _json_dumps(track.position_covariance_m2)
                    if track.position_covariance_m2 is not None
                    else None,
                    track.velocity_mps[0],
                    track.velocity_mps[1],
                    track.velocity_mps[2],
                    track.label_id,
                    track.label,
                    track.label_category,
                    track.iff_category,
                    track.confidence,
                    track.update_count,
                    track.status,
                    track.capability_tier,
                    track.tqi,
                    track.track_kind,
                ),
            )
            await self._commit_if_needed(db)

    def _row_to_node(self, row: aiosqlite.Row) -> dict[str, Any]:
        reported_position_m = [row["x"], row["y"], row["z"]]
        reported_position_geo = None
        if row["lat"] is not None and row["lon"] is not None:
            reported_position_geo = {"lat": row["lat"], "lon": row["lon"], "alt_m": row["alt"] or 0.0}
        reported_orientation = _json_loads(row["orientation_json"], {})
        reported_capabilities = _json_loads(row["capabilities_json"], [])
        reported_mobility = row["mobility"] or "stationary"
        overrides = _json_loads(row["overrides_json"], {})
        if not isinstance(overrides, dict):
            overrides = {}
        effective_position_m = overrides.get("position_m", reported_position_m)
        effective_position_geo = overrides.get("position_geo", reported_position_geo)
        effective_orientation = overrides.get("orientation", reported_orientation)
        effective_capabilities = overrides.get("capabilities", reported_capabilities)
        effective_mobility = overrides.get("mobility", reported_mobility)
        requested_position_filter = overrides.get("position_filter")
        effective_position_filter = requested_position_filter or (
            "kalman" if effective_mobility == "mobile" else "kde"
        )
        return {
            "id": row["id"],
            "node_type": row["node_type"],
            "position_m": effective_position_m,
            "position_geo": effective_position_geo,
            "reported_position_m": reported_position_m,
            "reported_position_geo": reported_position_geo,
            "sensor_offsets_m": _json_loads(row["sensor_offsets_json"], []),
            "capabilities": effective_capabilities,
            "reported_capabilities": reported_capabilities,
            "mobility": effective_mobility,
            "reported_mobility": reported_mobility,
            "position_filter": requested_position_filter,
            "effective_position_filter": effective_position_filter,
            "orientation": effective_orientation,
            "reported_orientation": reported_orientation,
            "metadata": _json_loads(row["metadata_json"], {}),
            "properties": _json_loads(row["properties_json"], {}),
            "transport": _json_loads(row["transport_json"], {}),
            "capability_config": _json_loads(row["capability_config_json"], {}),
            "safety": _json_loads(row["safety_json"], {}),
            "permissions": _json_loads(row["permissions_json"], {}),
            "overrides": overrides,
            "origin": row["origin"] or "ingest",
            "last_seen_ns": row["last_seen_ns"],
        }

    async def list_node_position_estimator_states(self) -> dict[str, dict[str, Any]]:
        db = self._require_db()
        rows = await (await db.execute(
            "SELECT node_id, version, state_json, updated_ns FROM node_position_estimator_states"
        )).fetchall()
        return {
            row["node_id"]: {
                "version": row["version"], "state": _json_loads(row["state_json"], {}),
                "updated_ns": row["updated_ns"],
            }
            for row in rows
        }

    async def upsert_node_position_estimator_state(self, node_id: str, state: dict[str, Any]) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                """INSERT INTO node_position_estimator_states(node_id, version, state_json, updated_ns)
                   VALUES (?, 1, ?, ?)
                   ON CONFLICT(node_id) DO UPDATE SET version=excluded.version,
                       state_json=excluded.state_json, updated_ns=excluded.updated_ns""",
                (node_id, _json_dumps(state), time.time_ns()),
            )
            await self._commit_if_needed(db)

    async def delete_node_position_estimator_state(self, node_id: str) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute("DELETE FROM node_position_estimator_states WHERE node_id = ?", (node_id,))
            await self._commit_if_needed(db)

    async def get_site_origin(self) -> dict[str, Any] | None:
        db = self._require_db()
        row = await (await db.execute(
            """SELECT lat, lon, alt_m, source, contributing_node_ids_json, resolved_ns
               FROM site_origin WHERE id = 1"""
        )).fetchone()
        if row is None:
            return None
        return {
            "lat": row["lat"],
            "lon": row["lon"],
            "alt_m": row["alt_m"],
            "source": row["source"],
            "contributing_node_ids": _json_loads(row["contributing_node_ids_json"], []),
            "resolved_ns": row["resolved_ns"],
        }

    async def upsert_site_origin(
        self,
        *,
        lat: float,
        lon: float,
        alt_m: float,
        source: str,
        contributing_node_ids: list[str] | tuple[str, ...] = (),
    ) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                """INSERT INTO site_origin(id, lat, lon, alt_m, source, contributing_node_ids_json, resolved_ns)
                   VALUES (1, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET lat=excluded.lat, lon=excluded.lon,
                       alt_m=excluded.alt_m, source=excluded.source,
                       contributing_node_ids_json=excluded.contributing_node_ids_json,
                       resolved_ns=excluded.resolved_ns""",
                (
                    float(lat),
                    float(lon),
                    float(alt_m),
                    source,
                    _json_dumps(list(contributing_node_ids)),
                    time.time_ns(),
                ),
            )
            await self._commit_if_needed(db)

    async def clear_site_origin(self) -> None:
        """Un-anchor the site so the next trusted GPS fix re-resolves the origin."""
        db = self._require_db()
        async with self._write_guard():
            await db.execute("DELETE FROM site_origin WHERE id = 1")
            await self._commit_if_needed(db)

    async def list_nodes(self, limit: int | None = None) -> list[dict]:
        db = self._require_db()
        if limit is not None and limit > 0:
            rows = await (await db.execute("SELECT * FROM nodes ORDER BY id ASC LIMIT ?", (limit,))).fetchall()
        else:
            rows = await (await db.execute("SELECT * FROM nodes ORDER BY id ASC")).fetchall()
        return [self._row_to_node(row) for row in rows]

    async def get_node_by_id(self, node_id: str) -> dict | None:
        db = self._require_db()
        row = await (await db.execute("SELECT * FROM nodes WHERE id = ? LIMIT 1", (node_id,))).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    async def insert_node_registration(
        self,
        spec: NodeSpec,
        *,
        origin: str = "operator",
        last_seen_ns: int | None = None,
    ) -> None:
        db = self._require_db()
        if spec.position_m is None:
            raise ValueError("NodeSpec.position_m is required for operator registration")
        position_geo = spec.position_geo
        created_ns = last_seen_ns if last_seen_ns is not None else 0
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO nodes (
                    id, node_type, x, y, z, lat, lon, alt,
                    sensor_offsets_json, capabilities_json, mobility,
                    orientation_json, metadata_json, properties_json,
                    transport_json, capability_config_json, safety_json,
                    permissions_json, overrides_json, origin, last_seen_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)
                """,
                (
                    spec.id,
                    spec.node_type.value,
                    spec.position_m[0],
                    spec.position_m[1],
                    spec.position_m[2],
                    position_geo.lat if position_geo is not None else None,
                    position_geo.lon if position_geo is not None else None,
                    position_geo.alt_m if position_geo is not None else None,
                    _json_dumps(spec.sensor_offsets_m),
                    _json_dumps([capability.value for capability in spec.capabilities]),
                    spec.mobility,
                    _json_dumps(spec.orientation.model_dump(mode="json")),
                    _json_dumps(spec.metadata),
                    _json_dumps(spec.properties),
                    _json_dumps(spec.transport),
                    _json_dumps(spec.capability_config),
                    _json_dumps(spec.safety.model_dump(mode="json")),
                    _json_dumps(spec.permissions),
                    origin,
                    created_ns,
                ),
            )
            await self._commit_if_needed(db)

    async def update_node_operator_fields(
        self,
        node_id: str,
        *,
        transport: dict[str, Any] | None = None,
        capability_config: dict[str, Any] | None = None,
        safety: dict[str, Any] | None = None,
        permissions: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        assignments: list[str] = []
        params: list[Any] = []
        if transport is not None:
            assignments.append("transport_json = ?")
            params.append(_json_dumps(transport))
        if capability_config is not None:
            assignments.append("capability_config_json = ?")
            params.append(_json_dumps(capability_config))
        if safety is not None:
            assignments.append("safety_json = ?")
            params.append(_json_dumps(safety))
        if permissions is not None:
            assignments.append("permissions_json = ?")
            params.append(_json_dumps(permissions))
        if metadata is not None:
            assignments.append("metadata_json = ?")
            params.append(_json_dumps(metadata))
        if not assignments:
            return await self.get_node_by_id(node_id) is not None
        db = self._require_db()
        async with self._write_guard():
            cursor = await db.execute(
                f"UPDATE nodes SET {', '.join(assignments)} WHERE id = ?",
                (*params, node_id),
            )
            await self._commit_if_needed(db)
        return cursor.rowcount > 0

    async def set_node_overrides(self, node_id: str, overrides: NodeOverrides | dict[str, Any]) -> bool:
        payload = overrides.model_dump(mode="json", exclude_none=True) if isinstance(overrides, NodeOverrides) else dict(overrides)
        db = self._require_db()
        async with self._write_guard():
            cursor = await db.execute(
                "UPDATE nodes SET overrides_json = ? WHERE id = ?",
                (_json_dumps(payload), node_id),
            )
            await self._commit_if_needed(db)
        return cursor.rowcount > 0

    async def list_node_overrides(self) -> dict[str, dict[str, Any]]:
        db = self._require_db()
        rows = await (
            await db.execute("SELECT id, overrides_json FROM nodes WHERE overrides_json != '{}'")
        ).fetchall()
        return {
            row["id"]: overrides
            for row in rows
            if isinstance((overrides := _json_loads(row["overrides_json"], {})), dict) and overrides
        }

    async def get_detection_by_event_id(self, event_id: str) -> dict | None:
        db = self._require_db()
        row = await (
            await db.execute(
                """
                SELECT * FROM detections
                WHERE event_id = ? OR id = ?
                ORDER BY timestamp_ns DESC
                LIMIT 1
                """,
                (event_id, event_id),
            )
        ).fetchone()
        if row is None:
            return None
        return self._row_to_detection(row).model_dump(mode="json")

    async def update_detection_review(
        self,
        *,
        detection_id: str,
        review_state: str,
        review_label_id: str | None,
        review_label: str | None,
        review_label_category: str | None,
        review_notes: str | None,
        promote_to_training: bool,
        training_example_kind: str | None,
        review_updated_ns: int,
    ) -> bool:
        db = self._require_db()
        async with self._write_guard():
            cursor = await db.execute(
                """
                UPDATE detections
                SET review_state = ?,
                    review_label_id = ?,
                    review_label = ?,
                    review_label_category = ?,
                    review_notes = ?,
                    review_updated_ns = ?,
                    promote_to_training = ?,
                    training_example_kind = ?
                WHERE id = ?
                """,
                (
                    review_state,
                    review_label_id,
                    review_label,
                    review_label_category,
                    review_notes,
                    review_updated_ns,
                    1 if promote_to_training else 0,
                    training_example_kind,
                    detection_id,
                ),
            )
            await self._commit_if_needed(db)
        return bool(cursor.rowcount)

    async def get_training_example(self, detection_id: str) -> dict | None:
        db = self._require_db()
        row = await (
            await db.execute(
                "SELECT * FROM training_examples WHERE detection_id = ? LIMIT 1",
                (detection_id,),
            )
        ).fetchone()
        return dict(row) if row is not None else None

    async def upsert_training_example(
        self,
        *,
        detection_id: str,
        label: str,
        label_category: str,
        example_kind: str,
        audio_path: str,
        manifest_path: str,
        created_ns: int,
        updated_ns: int,
        embedding_path: str | None = None,
    ) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO training_examples (
                    detection_id, label, label_category, example_kind, audio_path,
                    manifest_path, created_ns, updated_ns, embedding_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(detection_id) DO UPDATE SET
                    label = excluded.label,
                    label_category = excluded.label_category,
                    example_kind = excluded.example_kind,
                    audio_path = excluded.audio_path,
                    manifest_path = excluded.manifest_path,
                    updated_ns = excluded.updated_ns,
                    embedding_path = excluded.embedding_path
                """,
                (
                    detection_id,
                    label,
                    label_category,
                    example_kind,
                    audio_path,
                    manifest_path,
                    created_ns,
                    updated_ns,
                    embedding_path,
                ),
            )
            await self._commit_if_needed(db)

    async def delete_training_example(self, detection_id: str) -> dict | None:
        db = self._require_db()
        async with self._write_guard():
            row = await (
                await db.execute(
                    "SELECT * FROM training_examples WHERE detection_id = ? LIMIT 1",
                    (detection_id,),
                )
            ).fetchone()
            if row is None:
                return None
            await db.execute("DELETE FROM training_examples WHERE detection_id = ?", (detection_id,))
            await self._commit_if_needed(db)
        return dict(row)

    async def list_detections(
        self,
        limit: int = 100,
        *,
        since_ns: int | None = None,
        until_ns: int | None = None,
        min_label_confidence: float | None = None,
        review_state: str | None = None,
    ) -> list[dict]:
        db = self._require_db()
        clauses: list[str] = []
        params: list[object] = []
        if since_ns is not None:
            clauses.append("timestamp_ns >= ?")
            params.append(int(since_ns))
        if until_ns is not None:
            clauses.append("timestamp_ns <= ?")
            params.append(int(until_ns))
        if min_label_confidence is not None:
            clauses.append("label_confidence >= ?")
            params.append(float(min_label_confidence))
        if review_state is not None:
            clauses.append("review_state = ?")
            params.append(str(review_state))

        where_clause = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = await (
            await db.execute(
                f"SELECT * FROM detections{where_clause} ORDER BY timestamp_ns DESC LIMIT ?",
                (*params, limit),
            )
        ).fetchall()
        detections = [self._row_to_detection(row).model_dump(mode="json") for row in rows]
        return await self.enrich_detections_with_contributor_summaries(detections)

    async def omni_detection_summary_by_node(
        self,
        *,
        now_ns: int,
        active_window_ns: int,
        recent_window_ns: int,
        limit_per_node: int = 5,
        min_label_confidence: float | None = None,
    ) -> list[dict[str, Any]]:
        db = self._require_db()
        recent_start_ns = int(now_ns - recent_window_ns)
        active_start_ns = int(now_ns - active_window_ns)
        clauses = [
            "reporting_modality = 'omni'",
            "source_node_id IS NOT NULL",
            "timestamp_ns >= ?",
            "timestamp_ns <= ?",
        ]
        params: list[object] = [recent_start_ns, int(now_ns)]
        if min_label_confidence is not None:
            clauses.append("label_confidence >= ?")
            params.append(float(min_label_confidence))
        where = " AND ".join(clauses)
        rows = await (
            await db.execute(
                f"""
                SELECT
                    source_node_id,
                    id,
                    event_id,
                    timestamp_ns,
                    COALESCE(label, 'unknown') AS label,
                    COALESCE(label_category, 'unknown') AS label_category,
                    confidence,
                    label_confidence
                FROM detections
                WHERE {where}
                ORDER BY source_node_id ASC, timestamp_ns DESC
                """,
                tuple(params),
            )
        ).fetchall()

        summaries_by_node: dict[str, dict[str, Any]] = {}
        label_counts_by_node: dict[str, dict[str, int]] = {}
        category_counts_by_node: dict[str, dict[str, int]] = {}
        for row in rows:
            node_id = str(row["source_node_id"])
            summary = summaries_by_node.setdefault(
                node_id,
                {
                    "node_id": node_id,
                    "active_count": 0,
                    "recent_count": 0,
                    "last_detection_ns": None,
                    "top_labels": [],
                    "top_categories": [],
                    "max_confidence": None,
                    "sample_detection_ids": [],
                },
            )
            label_counts = label_counts_by_node.setdefault(node_id, {})
            category_counts = category_counts_by_node.setdefault(node_id, {})
            timestamp_ns = int(row["timestamp_ns"])
            confidence = float(
                row["label_confidence"]
                if row["label_confidence"] is not None
                else row["confidence"]
            )

            summary["recent_count"] += 1
            if timestamp_ns >= active_start_ns:
                summary["active_count"] += 1
            if summary["last_detection_ns"] is None or timestamp_ns > int(summary["last_detection_ns"]):
                summary["last_detection_ns"] = timestamp_ns
            if summary["max_confidence"] is None or confidence > float(summary["max_confidence"]):
                summary["max_confidence"] = confidence
            if len(summary["sample_detection_ids"]) < limit_per_node:
                summary["sample_detection_ids"].append(row["event_id"] or row["id"])

            label = str(row["label"] or "unknown")
            category = str(row["label_category"] or "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1

        for node_id, summary in summaries_by_node.items():
            label_counts = label_counts_by_node[node_id]
            category_counts = category_counts_by_node[node_id]
            summary["top_labels"] = [
                {"label": label, "count": count}
                for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[:limit_per_node]
            ]
            summary["top_categories"] = [
                {"category": category, "count": count}
                for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))[:limit_per_node]
            ]

        return sorted(
            summaries_by_node.values(),
            key=lambda item: (-(item["active_count"] or 0), -(item["recent_count"] or 0), item["node_id"]),
        )

    async def detection_matrix(
        self,
        *,
        start_ns: int,
        end_ns: int,
        bucket_ns: int,
        min_label_confidence: float | None = None,
        max_labels: int = 64,
    ) -> tuple[list[str], list[list[int]], list[int]]:
        """Aggregate detections into a (label × hour-bucket) count matrix.

        Returns (labels, counts[label_idx][bucket_idx], bucket_totals[bucket_idx]).
        Buckets are [start_ns, end_ns) divided into fixed-width intervals of
        `bucket_ns` — typically 3600 * 1e9 ns (one hour). Labels are selected
        from the top `max_labels` by total count in the window.
        """
        db = self._require_db()
        if bucket_ns <= 0 or end_ns <= start_ns:
            return ([], [], [])
        num_buckets = max(1, int((end_ns - start_ns) // bucket_ns))

        clauses = ["timestamp_ns >= ?", "timestamp_ns < ?"]
        params: list[object] = [int(start_ns), int(end_ns)]
        if min_label_confidence is not None:
            clauses.append("label_confidence >= ?")
            params.append(float(min_label_confidence))
        where = " AND ".join(clauses)

        # Pick top-N labels by count in window — keeps payload small for "noisy" days.
        top_rows = await (
            await db.execute(
                f"""
                SELECT COALESCE(label, 'unknown') AS lbl, COUNT(*) AS c
                FROM detections
                WHERE {where}
                GROUP BY lbl
                ORDER BY c DESC
                LIMIT ?
                """,
                (*params, int(max_labels)),
            )
        ).fetchall()
        if not top_rows:
            return ([], [[0] * num_buckets for _ in ()], [0] * num_buckets)

        labels = [row["lbl"] for row in top_rows]
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        counts = [[0] * num_buckets for _ in labels]
        bucket_totals = [0] * num_buckets

        placeholders = ",".join("?" for _ in labels)
        rows = await (
            await db.execute(
                f"""
                SELECT
                    CAST((timestamp_ns - ?) / ? AS INTEGER) AS bucket,
                    COALESCE(label, 'unknown') AS lbl,
                    COUNT(*) AS c
                FROM detections
                WHERE {where} AND COALESCE(label, 'unknown') IN ({placeholders})
                GROUP BY bucket, lbl
                """,
                (int(start_ns), int(bucket_ns), *params, *labels),
            )
        ).fetchall()
        for row in rows:
            b = int(row["bucket"])
            if b < 0 or b >= num_buckets:
                continue
            idx = label_to_idx.get(row["lbl"])
            if idx is None:
                continue
            c = int(row["c"])
            counts[idx][b] = c
            bucket_totals[b] += c
        return labels, counts, bucket_totals

    async def label_summaries(
        self,
        *,
        now_ns: int,
        window_ns: int,
        min_label_confidence: float | None = None,
    ) -> list[dict]:
        """Return one row per label: count, first_seen, last_seen, distinct-node count.

        Window = `[now_ns - window_ns, now_ns]`. Cheap aggregate query — one pass
        over the timestamp index; no heavy joins.
        """
        db = self._require_db()
        start_ns = int(now_ns - window_ns)
        clauses = ["timestamp_ns >= ?", "timestamp_ns <= ?"]
        params: list[object] = [start_ns, int(now_ns)]
        if min_label_confidence is not None:
            clauses.append("label_confidence >= ?")
            params.append(float(min_label_confidence))
        where = " AND ".join(clauses)
        rows = await (
            await db.execute(
                f"""
                SELECT
                    COALESCE(label, 'unknown')         AS label,
                    COUNT(*)                           AS count,
                    MIN(timestamp_ns)                  AS first_seen_ns,
                    MAX(timestamp_ns)                  AS last_seen_ns,
                    COUNT(DISTINCT source_node_id)     AS node_count,
                    COALESCE(label_category, 'unknown') AS category
                FROM detections
                WHERE {where}
                GROUP BY label
                ORDER BY count DESC
                """,
                tuple(params),
            )
        ).fetchall()
        return [dict(row) for row in rows]

    async def label_detail(
        self,
        label: str,
        *,
        now_ns: int,
        window_ns: int,
        recent_limit: int = 50,
        min_label_confidence: float | None = None,
    ) -> dict:
        """Per-label: hour-of-day histogram, day-of-week histogram, recent detections.

        Returned ints cover labels matching SQL `COALESCE(label, 'unknown') = ?`,
        so callers can pass `"unknown"` to query the bucket for detections without
        a label.
        """
        db = self._require_db()
        start_ns = int(now_ns - window_ns)
        clauses = ["timestamp_ns >= ?", "timestamp_ns <= ?", "COALESCE(label, 'unknown') = ?"]
        params: list[object] = [start_ns, int(now_ns), label]
        if min_label_confidence is not None:
            clauses.append("label_confidence >= ?")
            params.append(float(min_label_confidence))
        where = " AND ".join(clauses)

        # strftime on unix-seconds (converted from ns). SQLite handles this natively.
        hour_rows = await (
            await db.execute(
                f"""
                SELECT
                    CAST(strftime('%H', timestamp_ns / 1000000000, 'unixepoch') AS INTEGER) AS h,
                    COUNT(*) AS c
                FROM detections
                WHERE {where}
                GROUP BY h
                """,
                tuple(params),
            )
        ).fetchall()
        hours = [0] * 24
        for row in hour_rows:
            h = int(row["h"] or 0)
            if 0 <= h < 24:
                hours[h] = int(row["c"])

        dow_rows = await (
            await db.execute(
                f"""
                SELECT
                    CAST(strftime('%w', timestamp_ns / 1000000000, 'unixepoch') AS INTEGER) AS d,
                    COUNT(*) AS c
                FROM detections
                WHERE {where}
                GROUP BY d
                """,
                tuple(params),
            )
        ).fetchall()
        dow = [0] * 7
        for row in dow_rows:
            d = int(row["d"] or 0)
            if 0 <= d < 7:
                dow[d] = int(row["c"])

        month_rows = await (
            await db.execute(
                f"""
                SELECT
                    CAST(strftime('%m', timestamp_ns / 1000000000, 'unixepoch') AS INTEGER) AS m,
                    COUNT(*) AS c
                FROM detections
                WHERE {where}
                GROUP BY m
                """,
                tuple(params),
            )
        ).fetchall()
        months = [0] * 12
        for row in month_rows:
            m = int(row["m"] or 0)
            if 1 <= m <= 12:
                months[m - 1] = int(row["c"])

        recent = await (
            await db.execute(
                f"SELECT * FROM detections WHERE {where} ORDER BY timestamp_ns DESC LIMIT ?",
                tuple([*params, int(recent_limit)]),
            )
        ).fetchall()
        recent_out = [self._row_to_detection(row).model_dump(mode="json") for row in recent]

        total_row = await (
            await db.execute(
                f"SELECT COUNT(*) AS c, MIN(timestamp_ns) AS first_ns, MAX(timestamp_ns) AS last_ns FROM detections WHERE {where}",
                tuple(params),
            )
        ).fetchone()
        total = int(total_row["c"]) if total_row else 0
        first_ns = total_row["first_ns"] if total_row else None
        last_ns = total_row["last_ns"] if total_row else None

        return {
            "label": label,
            "total": total,
            "first_seen_ns": first_ns,
            "last_seen_ns": last_ns,
            "hour_histogram": hours,
            "dow_histogram": dow,
            "month_histogram": months,
            "recent": recent_out,
        }

    async def heatmap_bins(
        self,
        *,
        now_ns: int,
        window_ns: int,
        bin_deg: float = 0.0001,
        labels: list[str] | None = None,
        min_label_confidence: float | None = None,
        max_bins: int = 5000,
    ) -> list[dict]:
        """Pre-binned lat/lon/weight triplets for the geo heatmap view.

        `bin_deg` = rounding resolution in degrees (0.0001 ≈ 11 m at the equator).
        Hard `max_bins` cap on the payload — we fall back to larger bins if needed.
        """
        db = self._require_db()
        start_ns = int(now_ns - window_ns)
        clauses = [
            "timestamp_ns >= ?",
            "timestamp_ns <= ?",
            "lat IS NOT NULL",
            "lon IS NOT NULL",
        ]
        params: list[object] = [start_ns, int(now_ns)]
        if min_label_confidence is not None:
            clauses.append("label_confidence >= ?")
            params.append(float(min_label_confidence))
        if labels:
            placeholders = ",".join("?" for _ in labels)
            clauses.append(f"label IN ({placeholders})")
            params.extend(labels)
        where = " AND ".join(clauses)

        # Single grouped query; SQLite handles ROUND efficiently.
        rows = await (
            await db.execute(
                f"""
                SELECT
                    ROUND(lat, ?) AS lat,
                    ROUND(lon, ?) AS lon,
                    COUNT(*) AS weight
                FROM detections
                WHERE {where}
                GROUP BY lat, lon
                ORDER BY weight DESC
                LIMIT ?
                """,
                (_round_digits(bin_deg), _round_digits(bin_deg), *params, int(max_bins)),
            )
        ).fetchall()
        return [{"lat": float(row["lat"]), "lon": float(row["lon"]), "weight": int(row["weight"])} for row in rows]

    async def list_tracks(
        self,
        limit: int = 200,
        *,
        since_ns: int | None = None,
        statuses: list[str] | None = None,
    ) -> list[dict]:
        """List tracks, newest first.

        ``statuses`` restricts to the given track statuses (e.g. the active set,
        excluding ``dropped``). Filtering in SQL rather than after the fact matters:
        applied post-query the LIMIT is consumed by dropped rows, so an active track
        can be pushed out of the result entirely.
        """
        db = self._require_db()
        clauses: list[str] = []
        params: list[object] = []
        if since_ns is not None:
            clauses.append("last_seen_ns >= ?")
            params.append(int(since_ns))
        if statuses is not None:
            if not statuses:
                return []
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend(str(status) for status in statuses)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = await (
            await db.execute(
                f"SELECT * FROM tracks{where} ORDER BY last_seen_ns DESC LIMIT ?",
                tuple(params),
            )
        ).fetchall()
        tracks = [self._row_to_track(row).model_dump(mode="json") for row in rows]
        return await self.enrich_tracks_with_contributor_summaries(tracks)

    async def list_labels(self) -> list[dict]:
        db = self._require_db()
        rows = await (await db.execute("SELECT * FROM labels ORDER BY name ASC")).fetchall()
        return [dict(row) for row in rows]

    async def list_alerts(
        self,
        limit: int = 100,
        *,
        detection_id: str | None = None,
        track_id: str | None = None,
    ) -> list[dict]:
        db = self._require_db()
        clauses: list[str] = []
        params: list[object] = []
        if detection_id is not None:
            clauses.append("detection_id = ?")
            params.append(detection_id)
        if track_id is not None:
            clauses.append("track_id = ?")
            params.append(track_id)
        where = f" WHERE {' OR '.join(clauses)}" if clauses else ""
        rows = await (
            await db.execute(
                f"SELECT * FROM alerts{where} ORDER BY timestamp_ns DESC LIMIT ?",
                (*params, limit),
            )
        ).fetchall()
        result: list[dict] = []
        for row in rows:
            item = dict(row)
            item["payload"] = _json_loads(item.pop("payload_json"), {})
            if "updated_ns" not in item or item["updated_ns"] is None:
                item["updated_ns"] = item["timestamp_ns"]
            result.append(item)
        return result

    async def list_observations_by_ids(self, observation_ids: list[str]) -> list[dict]:
        db = self._require_db()
        if not observation_ids:
            return []
        placeholders = ",".join("?" for _ in observation_ids)
        rows = await (
            await db.execute(
                f"SELECT * FROM observations WHERE id IN ({placeholders})",
                tuple(observation_ids),
            )
        ).fetchall()
        by_id = {
            row["id"]: {
                "id": row["id"],
                "event_id": row["event_id"],
                "node_id": row["node_id"],
                "sensor_id": row["sensor_id"],
                "sensor_type": row["sensor_type"],
                "source_type": row["source_type"],
                "toa_ns": row["toa_ns"],
                "tor_ns": row["tor_ns"],
                "time_quality": row["time_quality"],
                "sample_rate_hz": row["sample_rate_hz"],
                "channel_index": row["channel_index"],
                "frame_sequence": row["frame_sequence"],
                "retention_tier": row["retention_tier"],
                "metadata": _json_loads(row["metadata_json"], {}),
                "created_ns": row["created_ns"],
            }
            for row in rows
        }
        return [by_id[observation_id] for observation_id in observation_ids if observation_id in by_id]

    async def list_track_updates(
        self,
        *,
        track_id: str | None = None,
        detection_id: str | None = None,
        event_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        db = self._require_db()
        clauses: list[str] = []
        params: list[object] = []
        if track_id is not None:
            clauses.append("track_id = ?")
            params.append(track_id)
        if detection_id is not None:
            clauses.append("detection_id = ?")
            params.append(detection_id)
        if event_id is not None:
            clauses.append("event_id = ?")
            params.append(event_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = await (
            await db.execute(
                f"SELECT * FROM track_updates{where} ORDER BY timestamp_ns DESC LIMIT ?",
                (*params, limit),
            )
        ).fetchall()
        return [
            {
                "id": row["id"],
                "track_id": row["track_id"],
                "timestamp_ns": row["timestamp_ns"],
                "event_id": row["event_id"],
                "update_type": row["update_type"],
                "status": row["status"],
                "position_m": [row["x"], row["y"], row["z"]],
                "position_geo": self._row_to_geo(row).model_dump(mode="json") if self._row_to_geo(row) else None,
                "velocity_mps": [row["vx"], row["vy"], row["vz"]],
                "covariance": _json_loads(row["covariance_json"], None),
                "tqi": row["tqi"],
                "confidence": row["confidence"],
                "label": row["label"],
                "label_category": row["label_category"],
                "detection_id": row["detection_id"],
                "observation_ids": _json_loads(row["observation_ids_json"], []),
                "metadata": _json_loads(row["metadata_json"], {}),
            }
            for row in rows
        ]

    async def latest_detection_audio_for_track(self, track_id: str) -> tuple[str, str] | None:
        db = self._require_db()
        row = await (
            await db.execute(
                """
                SELECT id, snippet_path
                FROM detections
                WHERE track_id = ? AND snippet_path IS NOT NULL
                ORDER BY timestamp_ns DESC
                LIMIT 1
                """,
                (track_id,),
            )
        ).fetchone()
        if row is None:
            return None
        detection_id = str(row["id"])
        snippet_path = row["snippet_path"]
        if not snippet_path:
            return None
        return detection_id, str(snippet_path)

    async def insert_alert(
        self,
        *,
        timestamp_ns: int,
        rule_id: str | None,
        detection_id: str | None,
        track_id: str | None,
        destination: str,
        priority: str,
        status: str,
        payload: dict[str, Any] | None = None,
        alert_id: str | None = None,
    ) -> str:
        db = self._require_db()
        final_id = alert_id or f"alr-{uuid.uuid4().hex[:16]}"
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO alerts (
                    id, timestamp_ns, rule_id, detection_id, track_id,
                    destination, priority, status, updated_ns, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    final_id,
                    timestamp_ns,
                    rule_id,
                    detection_id,
                    track_id,
                    destination,
                    priority,
                    status,
                    timestamp_ns,
                    _json_dumps(payload or {}),
                ),
            )
            await self._commit_if_needed(db)
        return final_id

    async def update_alert_status(
        self,
        *,
        alert_id: str,
        status: str,
        updated_ns: int,
        payload_patch: dict[str, Any] | None = None,
    ) -> bool:
        db = self._require_db()
        async with self._write_guard():
            row = await (
                await db.execute("SELECT payload_json FROM alerts WHERE id = ? LIMIT 1", (alert_id,))
            ).fetchone()
            if row is None:
                return False
            payload = _json_loads(row["payload_json"], {})
            if payload_patch:
                payload.update(payload_patch)
            await db.execute(
                """
                UPDATE alerts
                SET status = ?, updated_ns = ?, payload_json = ?
                WHERE id = ?
                """,
                (status, updated_ns, _json_dumps(payload), alert_id),
            )
            await self._commit_if_needed(db)
        return True

    async def insert_transcript(
        self,
        *,
        node_id: str | None,
        sensor_id: str | None,
        start_ns: int,
        end_ns: int,
        text: str,
        model: str | None,
        trigger_confidence: float | None,
        audio_path: str | None,
        detection_id: str | None,
        created_ns: int,
        transcript_id: str | None = None,
    ) -> str:
        db = self._require_db()
        final_id = transcript_id or f"txt-{uuid.uuid4().hex[:16]}"
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO transcripts (
                    id, node_id, sensor_id, start_ns, end_ns, text, model,
                    trigger_confidence, audio_path, detection_id, created_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    final_id,
                    node_id,
                    sensor_id,
                    start_ns,
                    end_ns,
                    text,
                    model,
                    trigger_confidence,
                    audio_path,
                    detection_id,
                    created_ns,
                ),
            )
            await self._commit_if_needed(db)
        return final_id

    async def list_transcripts(
        self, *, since_ns: int | None = None, limit: int = 200
    ) -> list[dict]:
        db = self._require_db()
        if since_ns is not None:
            rows = await (
                await db.execute(
                    "SELECT * FROM transcripts WHERE created_ns >= ? "
                    "ORDER BY created_ns DESC LIMIT ?",
                    (since_ns, limit),
                )
            ).fetchall()
        else:
            rows = await (
                await db.execute(
                    "SELECT * FROM transcripts ORDER BY created_ns DESC LIMIT ?", (limit,)
                )
            ).fetchall()
        return [dict(row) for row in rows]

    async def get_transcript(self, transcript_id: str) -> dict | None:
        db = self._require_db()
        row = await (
            await db.execute(
                "SELECT * FROM transcripts WHERE id = ? LIMIT 1", (transcript_id,)
            )
        ).fetchone()
        return dict(row) if row is not None else None

    async def delete_transcripts_older_than(self, cutoff_ns: int) -> list[str]:
        """Delete transcript rows older than cutoff_ns; return their audio_paths."""
        db = self._require_db()
        async with self._write_guard():
            rows = await (
                await db.execute(
                    "SELECT audio_path FROM transcripts WHERE created_ns < ?", (cutoff_ns,)
                )
            ).fetchall()
            await db.execute("DELETE FROM transcripts WHERE created_ns < ?", (cutoff_ns,))
            await self._commit_if_needed(db)
        return [row["audio_path"] for row in rows if row["audio_path"]]

    async def list_pings(self, limit: int = 500) -> list[dict]:
        db = self._require_db()
        rows = await (
            await db.execute("SELECT * FROM pings ORDER BY timestamp_ns DESC LIMIT ?", (limit,))
        ).fetchall()
        result: list[dict] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = _json_loads(item.pop("metadata_json"), {})
            result.append(item)
        return result

    async def insert_ping(
        self,
        *,
        timestamp_ns: int,
        ping_type: str,
        label: str | None,
        label_id: LabelId | None,
        received_level_db: float | None,
        position_m: tuple[float, float, float] | None,
        position_geo: GeoPoint | None,
        source_detection_id: str | None,
        source_observation_id: str | None,
        source_track_id: str | None,
        retention_tier: str,
        metadata: dict[str, Any] | None = None,
        ping_id: str | None = None,
    ) -> str:
        db = self._require_db()
        final_id = ping_id or f"png-{uuid.uuid4().hex[:16]}"
        x = y = z = None
        if position_m is not None:
            x, y, z = position_m
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO pings (
                    id, timestamp_ns, lat, lon, alt, x, y, z, ping_type,
                    label_id, label, received_level_db, source_detection_id, source_observation_id,
                    source_track_id, retention_tier, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    final_id,
                    timestamp_ns,
                    position_geo.lat if position_geo is not None else None,
                    position_geo.lon if position_geo is not None else None,
                    position_geo.alt_m if position_geo is not None else None,
                    x,
                    y,
                    z,
                    ping_type,
                    label_id,
                    label,
                    received_level_db,
                    source_detection_id,
                    source_observation_id,
                    source_track_id,
                    retention_tier,
                    _json_dumps(metadata or {}),
                ),
            )
            await self._commit_if_needed(db)
        return final_id

    async def insert_environment(
        self,
        *,
        node_id: str,
        timestamp_ns: int,
        temperature_c: float | None,
        pressure_pa: float | None,
        humidity_fraction: float | None,
        wind_speed_mps: float | None,
        wind_dir_deg: float | None,
        solar_lux: float | None,
        metadata: dict[str, Any] | None = None,
        environment_id: str | None = None,
    ) -> str:
        db = self._require_db()
        final_id = environment_id or f"env-{uuid.uuid4().hex[:16]}"
        temperature_k = (float(temperature_c) + 273.15) if temperature_c is not None else None
        humidity_value = None if humidity_fraction is None else max(0.0, min(1.0, float(humidity_fraction)))
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO environment (
                    id, node_id, timestamp_ns, temperature_k, pressure_pa, humidity,
                    wind_speed_mps, wind_dir_deg, solar_lux, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    final_id,
                    node_id,
                    timestamp_ns,
                    temperature_k,
                    pressure_pa,
                    humidity_value,
                    wind_speed_mps,
                    wind_dir_deg,
                    solar_lux,
                    _json_dumps(metadata or {}),
                ),
            )
            await self._commit_if_needed(db)
        return final_id

    async def list_environment(self, limit: int = 500, node_id: str | None = None) -> list[dict]:
        db = self._require_db()
        if node_id:
            rows = await (
                await db.execute(
                    """
                    SELECT e.*, n.x AS x, n.y AS y, n.z AS z
                    FROM environment e
                    LEFT JOIN nodes n ON n.id = e.node_id
                    WHERE e.node_id = ?
                    ORDER BY e.timestamp_ns DESC
                    LIMIT ?
                    """,
                    (node_id, limit),
                )
            ).fetchall()
        else:
            rows = await (
                await db.execute(
                    """
                    SELECT e.*, n.x AS x, n.y AS y, n.z AS z
                    FROM environment e
                    LEFT JOIN nodes n ON n.id = e.node_id
                    ORDER BY e.timestamp_ns DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            ).fetchall()
        return [self._row_to_environment(row) for row in rows]

    async def list_latest_time_quality_per_node(self) -> dict[str, str]:
        db = self._require_db()
        rows = await (
            await db.execute(
                """
                SELECT o.node_id, o.time_quality
                FROM observations o
                INNER JOIN (
                    SELECT node_id, MAX(toa_ns) AS max_toa
                    FROM observations
                    GROUP BY node_id
                ) latest
                ON o.node_id = latest.node_id AND o.toa_ns = latest.max_toa
                """
            )
        ).fetchall()
        return {row["node_id"]: row["time_quality"] for row in rows}

    async def list_latest_observation_metadata_per_node(self) -> dict[str, dict[str, Any]]:
        db = self._require_db()
        rows = await (
            await db.execute(
                """
                SELECT o.node_id, o.metadata_json
                FROM observations o
                INNER JOIN (
                    SELECT node_id, MAX(toa_ns) AS max_toa
                    FROM observations
                    GROUP BY node_id
                ) latest
                ON o.node_id = latest.node_id AND o.toa_ns = latest.max_toa
                """
            )
        ).fetchall()
        return {
            row["node_id"]: _json_loads(row["metadata_json"], {})
            for row in rows
        }

    async def list_latest_environment_per_node(self, limit: int = 500) -> list[dict[str, Any]]:
        db = self._require_db()
        rows = await (
            await db.execute(
                """
                SELECT e.*, n.x AS x, n.y AS y, n.z AS z
                FROM environment e
                INNER JOIN (
                    SELECT node_id, MAX(timestamp_ns) AS latest_ts
                    FROM environment
                    GROUP BY node_id
                ) latest
                ON latest.node_id = e.node_id AND latest.latest_ts = e.timestamp_ns
                LEFT JOIN nodes n ON n.id = e.node_id
                ORDER BY e.timestamp_ns DESC
                LIMIT ?
                """,
                (limit,),
            )
        ).fetchall()
        return [self._row_to_environment(row) for row in rows]

    async def insert_large_artifact(
        self,
        *,
        artifact_type: str,
        path: str,
        retention_tier: str,
        source_detection_id: str | None,
        source_track_id: str | None,
        created_ns: int,
        expires_ns: int | None,
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> str:
        db = self._require_db()
        final_id = artifact_id or f"lar-{uuid.uuid4().hex[:16]}"
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO large_artifacts (
                    id, artifact_type, path, retention_tier, source_detection_id,
                    source_track_id, metadata_json, created_ns, expires_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    final_id,
                    artifact_type,
                    path,
                    retention_tier,
                    source_detection_id,
                    source_track_id,
                    _json_dumps(metadata or {}),
                    created_ns,
                    expires_ns,
                ),
            )
            await self._commit_if_needed(db)
        return final_id

    # ── Capture sessions ───────────────────────────────────────────────────────

    async def upsert_capture_session(self, record: "Any") -> None:
        """Insert or replace a CaptureSessionRecord row."""
        from minimappr.core.capture_session import CaptureSessionRecord
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                """
                INSERT OR REPLACE INTO capture_sessions (
                session_id, state, stream_key, range_lease_id,
                start_time_ns, end_time_ns, first_frame_pts_ns,
                    work_dir, video_path, ambix_path, iamf_path, object_path, visual_path, youtube_path,
                    error, created_ns, capture_kind, calibration_manifest_path
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.session_id,
                    record.state.value if hasattr(record.state, "value") else str(record.state),
                    record.stream_key,
                    record.range_lease_id,
                    record.start_time_ns,
                    record.end_time_ns,
                    record.first_frame_pts_ns,
                    str(record.work_dir),
                    str(record.video_path) if record.video_path else None,
                    str(record.ambix_path) if getattr(record, "ambix_path", None) else None,
                    str(record.iamf_path) if record.iamf_path else None,
                    str(record.object_path) if getattr(record, "object_path", None) else None,
                    str(record.visual_path) if getattr(record, "visual_path", None) else None,
                    str(record.youtube_path) if record.youtube_path else None,
                    record.error,
                    record.created_ns,
                    getattr(record, "capture_kind", "recording") or "recording",
                    str(record.calibration_manifest_path)
                    if getattr(record, "calibration_manifest_path", None)
                    else None,
                ),
            )
            await self._commit_if_needed(db)

    async def delete_capture_session(self, session_id: str) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute("DELETE FROM capture_sessions WHERE session_id=?", (session_id,))
            await self._commit_if_needed(db)

    async def delete_large_artifacts_for_session(self, session_id: str) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                "DELETE FROM large_artifacts WHERE metadata_json LIKE ?",
                (f'%"session_id":"{session_id}"%',),
            )
            await db.execute(
                "DELETE FROM large_artifacts WHERE metadata_json LIKE ?",
                (f'%"session_id": "{session_id}"%',),
            )
            await self._commit_if_needed(db)

    async def get_capture_session(self, session_id: str) -> dict | None:
        db = self._require_db()
        row = await (
            await db.execute(
                "SELECT * FROM capture_sessions WHERE session_id=?", (session_id,)
            )
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    async def list_capture_sessions(self, limit: int = 100) -> list[dict]:
        db = self._require_db()
        rows = await (
            await db.execute(
                "SELECT * FROM capture_sessions ORDER BY created_ns DESC LIMIT ?",
                (limit,),
            )
        ).fetchall()
        return [dict(r) for r in rows]

    async def insert_large_artifact_for_session(
        self,
        *,
        session_id: str,
        artifact_type: str,
        ambix_path: str | None = None,
        iamf_path: str | None = None,
        object_path: str | None = None,
        visual_path: str | None = None,
        youtube_path: str | None = None,
        positions_path: str | None = None,
        created_ns: int,
    ) -> str:
        """Insert a large_artifacts row for a completed capture session."""
        path = iamf_path or youtube_path or visual_path or ambix_path or object_path or ""
        metadata: dict[str, Any] = {"session_id": session_id}
        if ambix_path:
            metadata["ambix_path"] = ambix_path
        if iamf_path:
            metadata["iamf_path"] = iamf_path
        if object_path:
            metadata["object_path"] = object_path
        if visual_path:
            metadata["visual_path"] = visual_path
        if youtube_path:
            metadata["youtube_path"] = youtube_path
        if positions_path:
            metadata["positions_path"] = positions_path
        return await self.insert_large_artifact(
            artifact_type=artifact_type,
            path=path,
            # Capture exports are operator-authored recordings, not disposable
            # detection evidence.  Only explicit deletion/full purge removes them.
            retention_tier="permanent",
            source_detection_id=None,
            source_track_id=None,
            created_ns=created_ns,
            expires_ns=None,
            metadata=metadata,
        )

    # ── Calibration ground truth ───────────────────────────────────────────────

    async def insert_calibration_ground_truth(
        self,
        *,
        session_id: str,
        label: str,
        label_category: str,
        lat: float | None,
        lon: float | None,
        alt_m: float | None,
        start_ns: int,
        end_ns: int,
        notes: str | None = None,
        geometry_kind: str = "static",
        source_node_id: str | None = None,
        event_id: str | None = None,
    ) -> dict:
        db = self._require_db()
        now_ns = time.time_ns()
        final_id = event_id or f"cgt-{uuid.uuid4().hex[:16]}"
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO calibration_ground_truth (
                    id, session_id, label, label_category, geometry_kind,
                    lat, lon, alt_m, source_node_id, trajectory_json,
                    start_ns, end_ns, notes, created_ns, updated_ns
                ) VALUES (?,?,?,?,?,?,?,?,?,NULL,?,?,?,?,?)
                """,
                (
                    final_id,
                    session_id,
                    label,
                    label_category,
                    geometry_kind,
                    lat,
                    lon,
                    alt_m,
                    source_node_id,
                    int(start_ns),
                    int(end_ns),
                    notes,
                    now_ns,
                    now_ns,
                ),
            )
            await self._commit_if_needed(db)
        row = await self.get_calibration_ground_truth(final_id)
        assert row is not None
        return row

    async def get_calibration_ground_truth(self, event_id: str) -> dict | None:
        db = self._require_db()
        row = await (
            await db.execute(
                "SELECT * FROM calibration_ground_truth WHERE id=?", (event_id,)
            )
        ).fetchone()
        return dict(row) if row is not None else None

    async def list_calibration_ground_truth(self, session_id: str) -> list[dict]:
        db = self._require_db()
        rows = await (
            await db.execute(
                "SELECT * FROM calibration_ground_truth WHERE session_id=? ORDER BY start_ns ASC",
                (session_id,),
            )
        ).fetchall()
        return [dict(r) for r in rows]

    async def update_calibration_ground_truth(
        self, event_id: str, updates: dict[str, Any]
    ) -> dict | None:
        allowed = {
            "label", "label_category", "lat", "lon", "alt_m",
            "start_ns", "end_ns", "notes",
        }
        fields = {k: v for k, v in updates.items() if k in allowed}
        if not fields:
            return await self.get_calibration_ground_truth(event_id)
        db = self._require_db()
        set_clause = ", ".join(f"{k}=?" for k in fields)
        async with self._write_guard():
            await db.execute(
                f"UPDATE calibration_ground_truth SET {set_clause}, updated_ns=? WHERE id=?",
                (*fields.values(), time.time_ns(), event_id),
            )
            await self._commit_if_needed(db)
        return await self.get_calibration_ground_truth(event_id)

    async def delete_calibration_ground_truth(self, event_id: str) -> bool:
        db = self._require_db()
        async with self._write_guard():
            cursor = await db.execute(
                "DELETE FROM calibration_ground_truth WHERE id=?", (event_id,)
            )
            await self._commit_if_needed(db)
        return cursor.rowcount > 0

    # ── Track / detection range queries (for iamf_pipeline) ──────────────────

    async def query_tracks_in_window(
        self, start_ns: int, end_ns: int
    ) -> list[dict]:
        """Return tracks that were active during [start_ns, end_ns]."""
        db = self._require_db()
        rows = await (
            await db.execute(
                """
                SELECT id AS track_id, x, y, z, label, status, tqi, confidence
                FROM tracks
                WHERE status IN ('confirmed', 'dropped')
                  AND (
                    (last_seen_ns >= ? AND first_seen_ns <= ?)
                    OR EXISTS (
                        SELECT 1
                        FROM detections
                        WHERE detections.track_id = tracks.id
                          AND COALESCE(report_window_end_ns, toa_ns, timestamp_ns) >= ?
                          AND COALESCE(report_window_start_ns, toa_ns, timestamp_ns) <= ?
                    )
                  )
                ORDER BY first_seen_ns ASC
                """,
                (start_ns, end_ns, start_ns, end_ns),
            )
        ).fetchall()
        return [dict(r) for r in rows]

    async def query_detections_for_track(
        self, track_id: str, start_ns: int, end_ns: int
    ) -> list[dict]:
        """Return in-window detections plus nearest bracketing detections for a track."""
        db = self._require_db()
        rows = await (
            await db.execute(
                """
                SELECT
                    toa_ns,
                    timestamp_ns,
                    report_window_start_ns,
                    report_window_end_ns,
                    x,
                    y,
                    z,
                    confidence,
                    label_confidence,
                    snippet_path,
                    feature_summary_json
                FROM detections
                WHERE track_id = ?
                  AND (
                    (toa_ns >= ? AND toa_ns <= ?)
                    OR (
                        COALESCE(report_window_end_ns, toa_ns, timestamp_ns) >= ?
                        AND COALESCE(report_window_start_ns, toa_ns, timestamp_ns) <= ?
                    )
                    OR toa_ns = (
                        SELECT max(toa_ns)
                        FROM detections
                        WHERE track_id = ? AND toa_ns < ?
                    )
                    OR toa_ns = (
                        SELECT min(toa_ns)
                        FROM detections
                        WHERE track_id = ? AND toa_ns > ?
                    )
                  )
                ORDER BY toa_ns ASC
                """,
                (
                    track_id,
                    start_ns,
                    end_ns,
                    start_ns,
                    end_ns,
                    track_id,
                    start_ns,
                    track_id,
                    end_ns,
                ),
            )
        ).fetchall()
        return [dict(r) for r in rows]

    def _row_to_map_overlay(self, row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "kind": row["kind"],
            "file_path": row["file_path"],
            "mime": row["mime"],
            "bounds": _json_loads(row["bounds_json"], []),
            "opacity": row["opacity"],
            "storey": row["storey"],
            "enabled": bool(row["enabled"]),
            "created_ns": row["created_ns"],
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    async def upsert_map_overlay(
        self,
        *,
        overlay_id: str,
        name: str,
        kind: str,
        file_path: str,
        mime: str,
        bounds: list[list[float]],
        opacity: float,
        storey: str | None,
        enabled: bool,
        created_ns: int,
        metadata: dict[str, Any] | None,
    ) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO map_overlays (
                    id, name, kind, file_path, mime, bounds_json, opacity,
                    storey, enabled, created_ns, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    kind=excluded.kind,
                    file_path=excluded.file_path,
                    mime=excluded.mime,
                    bounds_json=excluded.bounds_json,
                    opacity=excluded.opacity,
                    storey=excluded.storey,
                    enabled=excluded.enabled,
                    metadata_json=excluded.metadata_json
                """,
                (
                    overlay_id,
                    name,
                    kind,
                    file_path,
                    mime,
                    _json_dumps(bounds),
                    opacity,
                    storey,
                    1 if enabled else 0,
                    created_ns,
                    _json_dumps(metadata or {}),
                ),
            )
            await self._commit_if_needed(db)

    async def get_map_overlay(self, overlay_id: str) -> dict[str, Any] | None:
        db = self._require_db()
        row = await (
            await db.execute("SELECT * FROM map_overlays WHERE id = ? LIMIT 1", (overlay_id,))
        ).fetchone()
        if row is None:
            return None
        return self._row_to_map_overlay(row)

    async def list_map_overlays(self) -> list[dict[str, Any]]:
        db = self._require_db()
        rows = await (await db.execute("SELECT * FROM map_overlays ORDER BY created_ns DESC")).fetchall()
        return [self._row_to_map_overlay(row) for row in rows]

    async def delete_map_overlay(self, overlay_id: str) -> bool:
        db = self._require_db()
        async with self._write_guard():
            cursor = await db.execute("DELETE FROM map_overlays WHERE id = ?", (overlay_id,))
            await self._commit_if_needed(db)
            return cursor.rowcount > 0

    async def list_zones(self) -> list[dict]:
        db = self._require_db()
        rows = await (await db.execute("SELECT * FROM zones ORDER BY id ASC")).fetchall()
        result: list[dict] = []
        for row in rows:
            result.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "zone_type": row["zone_type"],
                    "polygon_geo": _json_loads(row["polygon_geo_json"], []),
                    "properties": _json_loads(row["properties_json"], {}),
                    "created_ns": row["created_ns"],
                }
            )
        return result

    async def upsert_zone(
        self,
        *,
        zone_id: str,
        name: str,
        zone_type: str,
        polygon_geo: list[list[float]],
        properties: dict[str, Any] | None,
        created_ns: int,
    ) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO zones (id, name, zone_type, polygon_geo_json, properties_json, created_ns)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    zone_type=excluded.zone_type,
                    polygon_geo_json=excluded.polygon_geo_json,
                    properties_json=excluded.properties_json
                """,
                (zone_id, name, zone_type, _json_dumps(polygon_geo), _json_dumps(properties or {}), created_ns),
            )
            await self._commit_if_needed(db)

    async def delete_zone(self, zone_id: str) -> bool:
        db = self._require_db()
        async with self._write_guard():
            cursor = await db.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
            await self._commit_if_needed(db)
            return cursor.rowcount > 0

    # ── Node Artifacts ───────────────────────────────────────────────────────

    async def insert_node_artifact(
        self,
        *,
        node_id: str,
        track_id: str | None,
        detection_id: str | None,
        kind: str = "snapshot",
        path: str,
        created_ns: int,
        artifact_id: str | None = None,
    ) -> str:
        db = self._require_db()
        final_id = artifact_id or f"na-{uuid.uuid4().hex[:16]}"
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO node_artifacts (
                    id, node_id, track_id, detection_id, kind, path, created_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (final_id, node_id, track_id, detection_id, kind, path, created_ns),
            )
            await self._commit_if_needed(db)
        return final_id

    async def get_node_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        db = self._require_db()
        row = await (
            await db.execute(
                "SELECT * FROM node_artifacts WHERE id = ? LIMIT 1", (artifact_id,)
            )
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result.setdefault("effector_id", result.get("node_id"))
        return result

    async def list_node_artifacts(
        self,
        *,
        node_id: str | None = None,
        track_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        db = self._require_db()
        clauses: list[str] = []
        params: list[object] = []
        if node_id is not None:
            clauses.append("node_id = ?")
            params.append(node_id)
        if track_id is not None:
            clauses.append("track_id = ?")
            params.append(track_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = await (
            await db.execute(
                f"SELECT * FROM node_artifacts{where} ORDER BY created_ns DESC LIMIT ?",
                (*params, limit),
            )
        ).fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result.setdefault("effector_id", result.get("node_id"))
            results.append(result)
        return results

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and all of its node-keyed records.

        With ``PRAGMA foreign_keys=ON`` this cascades to ``observations``,
        ``environment`` and ``bit_reports`` and NULLs ``detections.source_node_id``.
        Returns True if the node row existed.
        """
        db = self._require_db()
        async with self._write_guard():
            cursor = await db.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            await self._commit_if_needed(db)
            return cursor.rowcount > 0

    async def recent_alert_count(self, since_ns: int) -> int:
        db = self._require_db()
        row = await (
            await db.execute("SELECT COUNT(1) AS c FROM alerts WHERE timestamp_ns >= ?", (since_ns,))
        ).fetchone()
        return int(row["c"] if row is not None else 0)

    async def count_nodes_by_status(
        self,
        *,
        now_ns: int,
        degraded_after_seconds: float,
        offline_after_seconds: float,
    ) -> dict[str, int]:
        db = self._require_db()
        degraded_ns = int(degraded_after_seconds * 1_000_000_000)
        offline_ns = int(offline_after_seconds * 1_000_000_000)
        row = await (
            await db.execute(
                """
                SELECT
                    SUM(CASE WHEN (? - last_seen_ns) < ? THEN 1 ELSE 0 END) AS online_nodes,
                    SUM(CASE WHEN (? - last_seen_ns) >= ? AND (? - last_seen_ns) < ? THEN 1 ELSE 0 END) AS degraded_nodes,
                    SUM(CASE WHEN (? - last_seen_ns) >= ? THEN 1 ELSE 0 END) AS offline_nodes
                FROM nodes
                """,
                (
                    now_ns,
                    degraded_ns,
                    now_ns,
                    degraded_ns,
                    now_ns,
                    offline_ns,
                    now_ns,
                    offline_ns,
                ),
            )
        ).fetchone()
        if row is None:
            return {"online_nodes": 0, "degraded_nodes": 0, "offline_nodes": 0}
        return {
            "online_nodes": int(row["online_nodes"] or 0),
            "degraded_nodes": int(row["degraded_nodes"] or 0),
            "offline_nodes": int(row["offline_nodes"] or 0),
        }

    async def count_active_tracks(self) -> int:
        db = self._require_db()
        row = await (
            await db.execute(
                """
                SELECT COUNT(1) AS c
                FROM tracks
                WHERE status IN ('tentative', 'confirmed', 'coasting')
                """,
            )
        ).fetchone()
        return int(row["c"] if row is not None else 0)

    async def get_detection(self, detection_id: str) -> dict | None:
        db = self._require_db()
        row = await (
            await db.execute("SELECT * FROM detections WHERE id = ? LIMIT 1", (detection_id,))
        ).fetchone()
        if row is None:
            return None
        return self._row_to_detection(row).model_dump(mode="json")

    async def snippet_path_for_detection(self, detection_id: str) -> str | None:
        db = self._require_db()
        row = await (
            await db.execute(
                "SELECT snippet_path FROM detections WHERE id = ? LIMIT 1",
                (detection_id,),
            )
        ).fetchone()
        if row is None:
            return None
        return row["snippet_path"]

    async def cleanup_expired_snippets(self, now_ns: int) -> int:
        db = self._require_db()
        async with self._write_guard():
            rows = await (
                await db.execute(
                    """
                    SELECT id, snippet_path
                    FROM detections
                    WHERE snippet_path IS NOT NULL
                    AND snippet_expires_ns IS NOT NULL
                    AND snippet_expires_ns <= ?
                    """,
                    (now_ns,),
                )
            ).fetchall()

            for row in rows:
                snippet_path = row["snippet_path"]
                if snippet_path:
                    path = Path(snippet_path)
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                await db.execute(
                    "UPDATE detections SET snippet_path = NULL, snippet_expires_ns = NULL WHERE id = ?",
                    (row["id"],),
                )

            await self._commit_if_needed(db)
        return len(rows)

    async def cleanup_policy_managed_files(
        self,
        *,
        now_ns: int,
        policy: CleanupPolicy,
        dry_run: bool = False,
    ) -> dict[str, int]:
        db = self._require_db()
        summary = {
            "snippet_files_deleted": 0,
            "snippet_records_cleared": 0,
            "artifact_files_deleted": 0,
            "artifact_rows_deleted": 0,
        }
        protected_tiers = {"config", "permanent"}

        async with self._write_guard():
            snippet_rows = await (
                await db.execute(
                    """
                    SELECT d.id, d.snippet_path, d.timestamp_ns, d.label, d.retention_tier,
                           d.feature_summary_json,
                           EXISTS(SELECT 1 FROM alerts a WHERE a.detection_id = d.id) AS has_alert
                    FROM detections d
                    WHERE d.snippet_path IS NOT NULL
                    """
                )
            ).fetchall()
            snippet_ids_to_clear: list[str] = []
            for row in snippet_rows:
                if row["retention_tier"] in protected_tiers:
                    continue
                features = _json_loads(row["feature_summary_json"], {})
                max_age_seconds = policy.resolve_for_context({
                    "label": row["label"],
                    "classifier": features.get("winner_member"),
                    "retention_tier": row["retention_tier"],
                    "trigger_source": "alert" if row["has_alert"] else None,
                }).snippet_max_age_seconds
                if max_age_seconds is None:
                    continue
                threshold_ns = now_ns - int(max_age_seconds * 1_000_000_000)
                if int(row["timestamp_ns"]) > threshold_ns:
                    continue
                snippet_ids_to_clear.append(str(row["id"]))
                snippet_path = row["snippet_path"]
                if snippet_path and Path(snippet_path).exists():
                    summary["snippet_files_deleted"] += 1
                    if not dry_run:
                        try:
                            Path(snippet_path).unlink(missing_ok=True)
                        except OSError:
                            pass
            summary["snippet_records_cleared"] = len(snippet_ids_to_clear)
            if snippet_ids_to_clear and not dry_run:
                await db.executemany(
                    "UPDATE detections SET snippet_path = NULL, snippet_expires_ns = NULL WHERE id = ?",
                    ((detection_id,) for detection_id in snippet_ids_to_clear),
                )

            artifact_rows = await (
                await db.execute(
                    """
                    SELECT la.id, la.path, la.created_ns, la.retention_tier,
                           COALESCE(d.label, t.label) AS label,
                           d.feature_summary_json,
                           EXISTS(SELECT 1 FROM alerts a WHERE a.detection_id = d.id) AS has_alert
                    FROM large_artifacts la
                    LEFT JOIN detections d ON d.id = la.source_detection_id
                    LEFT JOIN tracks t ON t.id = la.source_track_id
                    """
                )
            ).fetchall()
            artifact_ids_to_delete: list[str] = []
            for row in artifact_rows:
                if row["retention_tier"] in protected_tiers:
                    continue
                features = _json_loads(row["feature_summary_json"], {})
                max_age_seconds = policy.resolve_for_context({
                    "label": row["label"],
                    "classifier": features.get("winner_member"),
                    "retention_tier": row["retention_tier"],
                    "trigger_source": "alert" if row["has_alert"] else None,
                }).artifact_max_age_seconds
                if max_age_seconds is None:
                    continue
                threshold_ns = now_ns - int(max_age_seconds * 1_000_000_000)
                if int(row["created_ns"]) > threshold_ns:
                    continue
                artifact_ids_to_delete.append(str(row["id"]))
                artifact_path = row["path"]
                if artifact_path and Path(artifact_path).exists():
                    summary["artifact_files_deleted"] += 1
                    if not dry_run:
                        try:
                            Path(artifact_path).unlink(missing_ok=True)
                        except OSError:
                            pass
            summary["artifact_rows_deleted"] = len(artifact_ids_to_delete)
            if artifact_ids_to_delete and not dry_run:
                await db.executemany(
                    "DELETE FROM large_artifacts WHERE id = ?",
                    ((artifact_id,) for artifact_id in artifact_ids_to_delete),
                )

            if not dry_run:
                await self._commit_if_needed(db)
        return summary

    async def cleanup_retention(
        self,
        *,
        now_ns: int,
        tier_ttls_seconds: dict[str, int],
        operational_ttls_seconds: dict[str, int] | None = None,
    ) -> dict[str, int]:
        db = self._require_db()
        summary = {
            "observations": 0,
            "detections": 0,
            "pings": 0,
            "large_artifacts": 0,
            "bit_reports": 0,
            "track_updates": 0,
            "alerts": 0,
            "environment": 0,
            "dropped_tracks": 0,
        }
        protected_tiers = {"config", "permanent"}
        operational_ttls = operational_ttls_seconds or {}
        prune_pings_by_operational_ttl = "pings" in operational_ttls

        async with self._write_guard():
            for tier, ttl_s in tier_ttls_seconds.items():
                if tier in protected_tiers or ttl_s < 0:
                    continue
                threshold_ns = now_ns - int(ttl_s * 1_000_000_000)

                obs_cursor = await db.execute(
                    """
                    DELETE FROM observations
                    WHERE retention_tier = ?
                    AND created_ns <= ?
                    """,
                    (tier, threshold_ns),
                )
                summary["observations"] += max(0, obs_cursor.rowcount)

                det_rows = await (
                    await db.execute(
                        """
                        SELECT id, snippet_path
                        FROM detections
                        WHERE retention_tier = ?
                        AND timestamp_ns <= ?
                        """,
                        (tier, threshold_ns),
                    )
                ).fetchall()
                for row in det_rows:
                    snippet_path = row["snippet_path"]
                    if snippet_path:
                        try:
                            Path(snippet_path).unlink(missing_ok=True)
                        except OSError:
                            pass
                det_cursor = await db.execute(
                    """
                    DELETE FROM detections
                    WHERE retention_tier = ?
                    AND timestamp_ns <= ?
                    """,
                    (tier, threshold_ns),
                )
                summary["detections"] += max(0, det_cursor.rowcount)

                if not prune_pings_by_operational_ttl:
                    ping_cursor = await db.execute(
                        """
                        DELETE FROM pings
                        WHERE retention_tier = ?
                        AND timestamp_ns <= ?
                        """,
                        (tier, threshold_ns),
                    )
                    summary["pings"] += max(0, ping_cursor.rowcount)

                artifact_rows = await (
                    await db.execute(
                        """
                        SELECT id, path
                        FROM large_artifacts
                        WHERE retention_tier = ?
                        AND ((expires_ns IS NOT NULL AND expires_ns <= ?) OR created_ns <= ?)
                        """,
                        (tier, now_ns, threshold_ns),
                    )
                ).fetchall()
                for row in artifact_rows:
                    try:
                        Path(row["path"]).unlink(missing_ok=True)
                    except OSError:
                        pass
                artifact_cursor = await db.execute(
                    """
                    DELETE FROM large_artifacts
                    WHERE retention_tier = ?
                    AND ((expires_ns IS NOT NULL AND expires_ns <= ?) OR created_ns <= ?)
                    """,
                    (tier, now_ns, threshold_ns),
                )
                summary["large_artifacts"] += max(0, artifact_cursor.rowcount)

            for key, sql in (
                (
                    "pings",
                    """
                    DELETE FROM pings
                    WHERE retention_tier NOT IN ('config', 'permanent')
                    AND timestamp_ns <= ?
                    """,
                ),
                (
                    "bit_reports",
                    """
                    DELETE FROM bit_reports
                    WHERE timestamp_ns <= ?
                    """,
                ),
                (
                    "track_updates",
                    """
                    DELETE FROM track_updates
                    WHERE timestamp_ns <= ?
                    """,
                ),
                (
                    "alerts",
                    """
                    DELETE FROM alerts
                    WHERE timestamp_ns <= ?
                    """,
                ),
                (
                    "environment",
                    """
                    DELETE FROM environment
                    WHERE timestamp_ns <= ?
                    """,
                ),
                (
                    "dropped_tracks",
                    """
                    DELETE FROM tracks
                    WHERE status = 'dropped'
                    AND last_seen_ns <= ?
                    """,
                ),
            ):
                ttl_s = operational_ttls.get(key)
                if ttl_s is None or ttl_s < 0:
                    continue
                threshold_ns = now_ns - int(ttl_s * 1_000_000_000)
                cursor = await db.execute(sql, (threshold_ns,))
                summary[key] += max(0, cursor.rowcount)

            await self._commit_if_needed(db)

        return summary

    @staticmethod
    def _row_to_geo(row: aiosqlite.Row) -> GeoPoint | None:
        if row["lat"] is None or row["lon"] is None:
            return None
        try:
            return GeoPoint(lat=float(row["lat"]), lon=float(row["lon"]), alt_m=float(row["alt"] or 0.0))
        except ValidationError:
            # A corrupt local->geographic conversion (e.g. an out-of-envelope
            # altitude) may have been persisted by an older write path. Drop the
            # geo rather than 500 the read endpoint; lat/lon/alt outside the
            # GeoPoint envelope are not physically meaningful anyway.
            return None

    @staticmethod
    def _row_to_environment(row: aiosqlite.Row) -> dict[str, Any]:
        position_m = None
        if row["x"] is not None and row["y"] is not None and row["z"] is not None:
            position_m = [float(row["x"]), float(row["y"]), float(row["z"])]
        temperature_c = None
        if row["temperature_k"] is not None:
            temperature_c = float(row["temperature_k"]) - 273.15
        return {
            "id": row["id"],
            "node_id": row["node_id"],
            "timestamp_ns": int(row["timestamp_ns"]),
            "temperature_c": temperature_c,
            "pressure_pa": float(row["pressure_pa"]) if row["pressure_pa"] is not None else None,
            "humidity_fraction": float(row["humidity"]) if row["humidity"] is not None else None,
            "wind_speed_mps": float(row["wind_speed_mps"]) if row["wind_speed_mps"] is not None else None,
            "wind_dir_deg": float(row["wind_dir_deg"]) if row["wind_dir_deg"] is not None else None,
            "solar_lux": float(row["solar_lux"]) if row["solar_lux"] is not None else None,
            "position_m": position_m,
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def _row_to_detection(self, row: aiosqlite.Row) -> DetectionEvent:
        return DetectionEvent(
            id=row["id"],
            event_id=row["event_id"] or row["id"],
            source_type=row["source_type"] or "raw_sensor",
            source_node_id=row["source_node_id"],
            timestamp_ns=int(row["timestamp_ns"]),
            toa_ns=int(row["toa_ns"] or row["timestamp_ns"]),
            tor_ns=int(row["tor_ns"] or row["timestamp_ns"]),
            time_quality=row["time_quality"] or "free_running",
            stale_ns=row["stale_ns"],
            report_window_start_ns=row["report_window_start_ns"],
            report_window_end_ns=row["report_window_end_ns"],
            reporting_modality=row["reporting_modality"] or "localized",
            position_m=(float(row["x"]), float(row["y"]), float(row["z"])),
            position_geo=self._row_to_geo(row),
            position_covariance_m2=_json_loads(row["covariance_json"], None),
            confidence=float(row["confidence"]),
            gdop=float(row["gdop"]),
            label_id=row["label_id"],
            label=row["label"],
            label_category=row["label_category"] or "unknown",
            iff_category=row["iff_category"] or "unknown",
            label_confidence=float(row["label_confidence"]),
            received_level_db=row["received_level_db"],
            track_id=row["track_id"],
            reference_sensor=row["reference_sensor"],
            source_sensors=_json_loads(row["source_sensors_json"], []),
            source_observation_ids=_json_loads(row["source_observation_ids_json"], []),
            zone_ids=_json_loads(row["zone_ids_json"], []),
            tdoa_s=_json_loads(row["tdoa_json"], {}),
            classifier_scores=_json_loads(row["classifier_scores_json"], {}),
            feature_summary=_json_loads(row["feature_summary_json"], {}),
            review_state=row["review_state"] or "unreviewed",
            review_label_id=row["review_label_id"],
            review_label=row["review_label"],
            review_label_category=row["review_label_category"],
            review_notes=row["review_notes"],
            review_updated_ns=row["review_updated_ns"],
            promote_to_training=bool(row["promote_to_training"] or 0),
            training_example_kind=_row_value(row, "training_example_kind"),
            retention_tier=row["retention_tier"] or "short",
            snippet_path=row["snippet_path"],
        )

    def _row_to_track(self, row: aiosqlite.Row) -> TrackState:
        return TrackState(
            id=row["id"],
            first_seen_ns=int(row["first_seen_ns"]),
            last_seen_ns=int(row["last_seen_ns"]),
            position_m=(float(row["x"]), float(row["y"]), float(row["z"])),
            position_geo=self._row_to_geo(row),
            position_covariance_m2=_json_loads(row["covariance_json"], None),
            velocity_mps=(float(row["vx"]), float(row["vy"]), float(row["vz"])),
            label_id=row["label_id"],
            label=row["label"],
            label_category=row["label_category"] or "unknown",
            iff_category=row["iff_category"] or "unknown",
            confidence=float(row["confidence"]),
            update_count=int(row["update_count"]),
            status=row["status"],
            capability_tier=row["capability_tier"] or "full_3d",
            tqi=float(row["tqi"]),
            track_kind=(_row_value(row, "track_kind") or "acoustic"),
        )

    @staticmethod
    def _ensure_contributor_entry(
        contributors_by_node_id: dict[str, dict[str, Any]],
        node_id: str,
    ) -> dict[str, Any]:
        return contributors_by_node_id.setdefault(
            node_id,
            {
                "node_id": node_id,
                "roles": set(),
                "sensor_ids": set(),
                "contribution_count": 0,
                "last_contributed_ns": None,
            },
        )

    @classmethod
    def _record_contributor_roles(
        cls,
        contributors_by_node_id: dict[str, dict[str, Any]],
        *,
        node_id: str | None,
        roles: list[str],
        sensor_id: str | None = None,
        timestamp_ns: int | None = None,
        increment_count: bool,
    ) -> None:
        if not node_id:
            return
        entry = cls._ensure_contributor_entry(contributors_by_node_id, node_id)
        entry["roles"].update(role for role in roles if role)
        if sensor_id:
            entry["sensor_ids"].add(sensor_id)
        if increment_count:
            entry["contribution_count"] += 1
        else:
            entry["contribution_count"] = max(1, int(entry["contribution_count"]))
        if timestamp_ns is not None:
            existing = entry["last_contributed_ns"]
            entry["last_contributed_ns"] = timestamp_ns if existing is None else max(existing, timestamp_ns)

    @staticmethod
    def _serialize_contributor_summaries(
        contributors_by_node_id: dict[str, dict[str, Any]],
    ) -> list[ContributorSummary]:
        contributor_summaries = [
            ContributorSummary(
                node_id=entry["node_id"],
                roles=sorted(entry["roles"]),
                sensor_ids=sorted(entry["sensor_ids"]),
                contribution_count=int(entry["contribution_count"]),
                last_contributed_ns=entry["last_contributed_ns"],
            )
            for entry in contributors_by_node_id.values()
        ]
        contributor_summaries.sort(
            key=lambda contributor: (
                _contributor_role_priority(contributor.roles),
                -(contributor.last_contributed_ns or 0),
                -contributor.contribution_count,
                contributor.node_id,
            )
        )
        return contributor_summaries

    def _build_detection_contributor_summaries(
        self,
        detection: dict[str, Any],
        observations_by_id: dict[str, dict[str, Any]],
    ) -> list[ContributorSummary]:
        contributors_by_node_id: dict[str, dict[str, Any]] = {}
        detection_timestamp_ns = detection.get("timestamp_ns")
        detection_timestamp_ns = detection_timestamp_ns if isinstance(detection_timestamp_ns, int) else None
        source_node_id = detection.get("source_node_id")
        self._record_contributor_roles(
            contributors_by_node_id,
            node_id=source_node_id,
            roles=["source"],
            timestamp_ns=detection_timestamp_ns,
            increment_count=False,
        )

        reference_sensor_id = detection.get("reference_sensor")
        localized_sensor_ids = set(detection.get("source_sensors") or [])
        localized_sensor_ids.update((detection.get("tdoa_s") or {}).keys())

        for observation_id in detection.get("source_observation_ids") or []:
            observation = observations_by_id.get(observation_id)
            if observation is None:
                continue
            observation_roles: list[str] = []
            sensor_id = observation.get("sensor_id")
            if sensor_id == reference_sensor_id:
                observation_roles.append("reference")
            if sensor_id in localized_sensor_ids:
                observation_roles.append("tdoa")
            if not observation_roles:
                observation_roles.append("support")
            self._record_contributor_roles(
                contributors_by_node_id,
                node_id=observation.get("node_id"),
                roles=observation_roles,
                sensor_id=sensor_id,
                timestamp_ns=detection_timestamp_ns,
                increment_count=True,
            )

        return self._serialize_contributor_summaries(contributors_by_node_id)

    async def enrich_detections_with_contributor_summaries(self, detections: list[dict]) -> list[dict]:
        observation_ids: list[str] = []
        for detection in detections:
            observation_ids.extend(detection.get("source_observation_ids") or [])
        observations = await self.list_observations_by_ids(list(dict.fromkeys(observation_ids)))
        observations_by_id = {observation["id"]: observation for observation in observations}

        for detection in detections:
            contributors = self._build_detection_contributor_summaries(detection, observations_by_id)
            detection["contributors"] = [contributor.model_dump(mode="json") for contributor in contributors]
        return detections

    async def enrich_tracks_with_contributor_summaries(self, tracks: list[dict]) -> list[dict]:
        for track in tracks:
            track_updates = await self.list_track_updates(track_id=track["id"], limit=1)
            if not track_updates:
                track["contributors"] = []
                track["contributor_count"] = 0
                continue

            latest_track_update = track_updates[0]
            contributors: list[ContributorSummary] = []
            detection_id = latest_track_update.get("detection_id")
            if isinstance(detection_id, str) and detection_id:
                detection = await self.get_detection(detection_id)
                if detection is not None:
                    observations = await self.list_observations_by_ids(detection.get("source_observation_ids") or [])
                    observations_by_id = {observation["id"]: observation for observation in observations}
                    contributors = self._build_detection_contributor_summaries(detection, observations_by_id)

            if not contributors:
                contributors_by_node_id: dict[str, dict[str, Any]] = {}
                observation_ids = latest_track_update.get("observation_ids") or []
                observations = await self.list_observations_by_ids(observation_ids)
                reference_sensor_id = (latest_track_update.get("metadata") or {}).get("reference_sensor")
                update_timestamp_ns = latest_track_update.get("timestamp_ns")
                update_timestamp_ns = update_timestamp_ns if isinstance(update_timestamp_ns, int) else None
                for observation in observations:
                    observation_roles = ["tdoa"]
                    if observation.get("sensor_id") == reference_sensor_id:
                        observation_roles.append("reference")
                    self._record_contributor_roles(
                        contributors_by_node_id,
                        node_id=observation.get("node_id"),
                        roles=observation_roles,
                        sensor_id=observation.get("sensor_id"),
                        timestamp_ns=update_timestamp_ns,
                        increment_count=True,
                    )
                contributors = self._serialize_contributor_summaries(contributors_by_node_id)

            track["contributors"] = [contributor.model_dump(mode="json") for contributor in contributors]
            track["contributor_count"] = len(contributors)
        return tracks

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Storage is not initialized")
        return self._db
