"""SQLite persistence layer."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

from minimappr.models import DetectionEvent, GeoPoint, NodeSpec, TrackState


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _json_loads(raw: str | None, fallback: Any) -> Any:
    if raw is None:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    cleaned = cleaned.strip("-")
    return cleaned or "unknown"


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA synchronous=NORMAL;")

        await self._db.executescript(
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
                metadata_json TEXT NOT NULL,
                properties_json TEXT NOT NULL DEFAULT '{}',
                last_seen_ns INTEGER NOT NULL
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
                created_ns INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_observations_toa ON observations(toa_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_observations_sensor ON observations(sensor_id, toa_ns DESC);

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
                source_node_id TEXT,
                timestamp_ns INTEGER NOT NULL,
                toa_ns INTEGER NOT NULL,
                tor_ns INTEGER NOT NULL,
                time_quality TEXT NOT NULL,
                stale_ns INTEGER,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                lat REAL,
                lon REAL,
                alt REAL,
                covariance_json TEXT,
                confidence REAL NOT NULL,
                gdop REAL NOT NULL,
                label_id TEXT,
                label TEXT NOT NULL,
                label_category TEXT NOT NULL DEFAULT 'unknown',
                iff_category TEXT NOT NULL DEFAULT 'unknown',
                label_confidence REAL NOT NULL,
                spl_db REAL,
                track_id TEXT,
                reference_sensor TEXT NOT NULL,
                source_sensors_json TEXT NOT NULL,
                source_observation_ids_json TEXT NOT NULL DEFAULT '[]',
                zone_ids_json TEXT NOT NULL DEFAULT '[]',
                tdoa_json TEXT NOT NULL,
                classifier_scores_json TEXT NOT NULL,
                feature_summary_json TEXT NOT NULL,
                retention_tier TEXT NOT NULL DEFAULT 'short',
                snippet_path TEXT,
                snippet_expires_ns INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_detections_track ON detections(track_id, timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label_category, timestamp_ns DESC);

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
                track_id TEXT NOT NULL,
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
                detection_id TEXT,
                observation_ids_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_track_updates_track_ts ON track_updates(track_id, timestamp_ns DESC);

            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                timestamp_ns INTEGER NOT NULL,
                rule_id TEXT,
                detection_id TEXT,
                track_id TEXT,
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
                label_id TEXT,
                label TEXT,
                spl_db REAL,
                source_detection_id TEXT,
                source_observation_id TEXT,
                source_track_id TEXT,
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
                source_detection_id TEXT,
                source_track_id TEXT,
                metadata_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                expires_ns INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_large_artifacts_expires ON large_artifacts(expires_ns);

            CREATE TABLE IF NOT EXISTS zones (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                zone_type TEXT NOT NULL,
                polygon_geo_json TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS environment (
                id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                temperature_k REAL,
                pressure_pa REAL,
                humidity REAL,
                wind_speed_mps REAL,
                wind_dir_deg REAL,
                solar_lux REAL,
                metadata_json TEXT NOT NULL
            );
            """
        )

        # Migration-safe additions for older DBs.
        await self._ensure_columns(
            "nodes",
            {
                "lat": "REAL",
                "lon": "REAL",
                "alt": "REAL",
                "mobility": "TEXT NOT NULL DEFAULT 'stationary'",
                "properties_json": "TEXT NOT NULL DEFAULT '{}'",
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
                "time_quality": "TEXT NOT NULL DEFAULT 'freerunning'",
                "stale_ns": "INTEGER",
                "lat": "REAL",
                "lon": "REAL",
                "alt": "REAL",
                "covariance_json": "TEXT",
                "label_id": "TEXT",
                "label_category": "TEXT NOT NULL DEFAULT 'unknown'",
                "iff_category": "TEXT NOT NULL DEFAULT 'unknown'",
                "spl_db": "REAL",
                "source_observation_ids_json": "TEXT NOT NULL DEFAULT '[]'",
                "zone_ids_json": "TEXT NOT NULL DEFAULT '[]'",
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
            },
        )
        await self._db.commit()

    async def _table_columns(self, table: str) -> set[str]:
        db = self._require_db()
        rows = await (await db.execute(f"PRAGMA table_info({table})")).fetchall()
        return {row["name"] for row in rows}

    async def _ensure_columns(self, table: str, expected: dict[str, str]) -> None:
        db = self._require_db()
        current = await self._table_columns(table)
        for name, definition in expected.items():
            if name in current:
                continue
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

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
        async with self._lock:
            await db.execute(
                """
                INSERT INTO nodes (
                    id, node_type, x, y, z, lat, lon, alt,
                    sensor_offsets_json, capabilities_json, mobility,
                    metadata_json, properties_json, last_seen_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    node_type=excluded.node_type,
                    x=excluded.x,
                    y=excluded.y,
                    z=excluded.z,
                    lat=excluded.lat,
                    lon=excluded.lon,
                    alt=excluded.alt,
                    sensor_offsets_json=excluded.sensor_offsets_json,
                    capabilities_json=excluded.capabilities_json,
                    mobility=excluded.mobility,
                    metadata_json=excluded.metadata_json,
                    properties_json=excluded.properties_json,
                    last_seen_ns=excluded.last_seen_ns
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
                    _json_dumps(spec.metadata),
                    _json_dumps(spec.properties),
                    last_seen_ns,
                ),
            )
            await db.commit()

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
        async with self._lock:
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
            await db.commit()
        return observation_id

    async def upsert_label(
        self,
        *,
        name: str,
        category: str,
        source: str = "runtime",
        parent_label_id: str | None = None,
        external_taxonomy: str | None = None,
        external_label_id: str | None = None,
        created_ns: int,
    ) -> str:
        db = self._require_db()
        label_id = f"lbl-{_slugify(name)[:48]}"
        async with self._lock:
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
            await db.commit()
        return label_id

    async def insert_detection(
        self,
        detection: DetectionEvent,
        snippet_path: str | None,
        snippet_expires_ns: int | None,
        retention_tier: str = "short",
    ) -> None:
        db = self._require_db()
        async with self._lock:
            await db.execute(
                """
                INSERT INTO detections (
                    id, event_id, source_type, source_node_id,
                    timestamp_ns, toa_ns, tor_ns, time_quality, stale_ns,
                    x, y, z, lat, lon, alt, covariance_json,
                    confidence, gdop, label_id, label, label_category, iff_category,
                    label_confidence, spl_db, track_id,
                    reference_sensor, source_sensors_json, source_observation_ids_json,
                    zone_ids_json, tdoa_json, classifier_scores_json, feature_summary_json,
                    retention_tier, snippet_path, snippet_expires_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    detection.spl_db,
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
            await db.commit()

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
        async with self._lock:
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
            await db.commit()
        return update_id

    async def upsert_track(self, track: TrackState) -> None:
        db = self._require_db()
        async with self._lock:
            await db.execute(
                """
                INSERT INTO tracks (
                    id, first_seen_ns, last_seen_ns,
                    x, y, z, lat, lon, alt, covariance_json,
                    vx, vy, vz,
                    label_id, label, label_category, iff_category,
                    confidence, update_count, status, capability_tier, tqi
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    tqi=excluded.tqi
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
                ),
            )
            await db.commit()

    async def list_nodes(self) -> list[dict]:
        db = self._require_db()
        async with self._lock:
            rows = await (await db.execute("SELECT * FROM nodes ORDER BY id ASC")).fetchall()

        result = []
        for row in rows:
            position_geo = None
            if row["lat"] is not None and row["lon"] is not None:
                position_geo = {"lat": row["lat"], "lon": row["lon"], "alt_m": row["alt"] or 0.0}
            result.append(
                {
                    "id": row["id"],
                    "node_type": row["node_type"],
                    "position_m": [row["x"], row["y"], row["z"]],
                    "position_geo": position_geo,
                    "sensor_offsets_m": _json_loads(row["sensor_offsets_json"], []),
                    "capabilities": _json_loads(row["capabilities_json"], []),
                    "mobility": row["mobility"] or "stationary",
                    "metadata": _json_loads(row["metadata_json"], {}),
                    "properties": _json_loads(row["properties_json"], {}),
                    "last_seen_ns": row["last_seen_ns"],
                }
            )
        return result

    async def list_detections(self, limit: int = 100) -> list[dict]:
        db = self._require_db()
        async with self._lock:
            rows = await (
                await db.execute(
                    "SELECT * FROM detections ORDER BY timestamp_ns DESC LIMIT ?",
                    (limit,),
                )
            ).fetchall()

        result = []
        for row in rows:
            position_geo = None
            if row["lat"] is not None and row["lon"] is not None:
                position_geo = {"lat": row["lat"], "lon": row["lon"], "alt_m": row["alt"] or 0.0}
            result.append(
                {
                    "id": row["id"],
                    "event_id": row["event_id"] or row["id"],
                    "source_type": row["source_type"] or "raw_sensor",
                    "source_node_id": row["source_node_id"],
                    "timestamp_ns": row["timestamp_ns"],
                    "toa_ns": row["toa_ns"] or row["timestamp_ns"],
                    "tor_ns": row["tor_ns"] or row["timestamp_ns"],
                    "time_quality": row["time_quality"] or "freerunning",
                    "stale_ns": row["stale_ns"],
                    "position_m": [row["x"], row["y"], row["z"]],
                    "position_geo": position_geo,
                    "position_covariance_m2": _json_loads(row["covariance_json"], None),
                    "confidence": row["confidence"],
                    "gdop": row["gdop"],
                    "label_id": row["label_id"],
                    "label": row["label"],
                    "label_category": row["label_category"] or "unknown",
                    "iff_category": row["iff_category"] or "unknown",
                    "label_confidence": row["label_confidence"],
                    "spl_db": row["spl_db"],
                    "track_id": row["track_id"],
                    "reference_sensor": row["reference_sensor"],
                    "source_sensors": _json_loads(row["source_sensors_json"], []),
                    "source_observation_ids": _json_loads(row["source_observation_ids_json"], []),
                    "zone_ids": _json_loads(row["zone_ids_json"], []),
                    "tdoa_s": _json_loads(row["tdoa_json"], {}),
                    "classifier_scores": _json_loads(row["classifier_scores_json"], {}),
                    "feature_summary": _json_loads(row["feature_summary_json"], {}),
                    "retention_tier": row["retention_tier"] or "short",
                    "snippet_path": row["snippet_path"],
                }
            )
        return result

    async def list_tracks(self, limit: int = 200) -> list[dict]:
        db = self._require_db()
        async with self._lock:
            rows = await (
                await db.execute("SELECT * FROM tracks ORDER BY last_seen_ns DESC LIMIT ?", (limit,))
            ).fetchall()

        result = []
        for row in rows:
            position_geo = None
            if row["lat"] is not None and row["lon"] is not None:
                position_geo = {"lat": row["lat"], "lon": row["lon"], "alt_m": row["alt"] or 0.0}
            result.append(
                {
                    "id": row["id"],
                    "first_seen_ns": row["first_seen_ns"],
                    "last_seen_ns": row["last_seen_ns"],
                    "position_m": [row["x"], row["y"], row["z"]],
                    "position_geo": position_geo,
                    "position_covariance_m2": _json_loads(row["covariance_json"], None),
                    "velocity_mps": [row["vx"], row["vy"], row["vz"]],
                    "label_id": row["label_id"],
                    "label": row["label"],
                    "label_category": row["label_category"] or "unknown",
                    "iff_category": row["iff_category"] or "unknown",
                    "confidence": row["confidence"],
                    "update_count": row["update_count"],
                    "status": row["status"],
                    "capability_tier": row["capability_tier"] or "full_3d",
                    "tqi": row["tqi"],
                }
            )
        return result

    async def list_labels(self) -> list[dict]:
        db = self._require_db()
        async with self._lock:
            rows = await (await db.execute("SELECT * FROM labels ORDER BY name ASC")).fetchall()
        return [dict(row) for row in rows]

    async def list_alerts(self, limit: int = 100) -> list[dict]:
        db = self._require_db()
        async with self._lock:
            rows = await (
                await db.execute("SELECT * FROM alerts ORDER BY timestamp_ns DESC LIMIT ?", (limit,))
            ).fetchall()
        result: list[dict] = []
        for row in rows:
            item = dict(row)
            item["payload"] = _json_loads(item.pop("payload_json"), {})
            if "updated_ns" not in item or item["updated_ns"] is None:
                item["updated_ns"] = item["timestamp_ns"]
            result.append(item)
        return result

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
        async with self._lock:
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
            await db.commit()
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
        async with self._lock:
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
            await db.commit()
        return True

    async def list_pings(self, limit: int = 500) -> list[dict]:
        db = self._require_db()
        async with self._lock:
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
        label_id: str | None,
        spl_db: float | None,
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
        async with self._lock:
            await db.execute(
                """
                INSERT INTO pings (
                    id, timestamp_ns, lat, lon, alt, x, y, z, ping_type,
                    label_id, label, spl_db, source_detection_id, source_observation_id,
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
                    spl_db,
                    source_detection_id,
                    source_observation_id,
                    source_track_id,
                    retention_tier,
                    _json_dumps(metadata or {}),
                ),
            )
            await db.commit()
        return final_id

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
        async with self._lock:
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
            await db.commit()
        return final_id

    async def list_zones(self) -> list[dict]:
        db = self._require_db()
        async with self._lock:
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
        async with self._lock:
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
            await db.commit()

    async def delete_zone(self, zone_id: str) -> bool:
        db = self._require_db()
        async with self._lock:
            cursor = await db.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
            await db.commit()
            return cursor.rowcount > 0

    async def recent_alert_count(self, since_ns: int) -> int:
        db = self._require_db()
        async with self._lock:
            row = await (
                await db.execute("SELECT COUNT(1) AS c FROM alerts WHERE timestamp_ns >= ?", (since_ns,))
            ).fetchone()
        return int(row["c"] if row is not None else 0)

    async def get_detection(self, detection_id: str) -> dict | None:
        db = self._require_db()
        async with self._lock:
            row = await (
                await db.execute("SELECT * FROM detections WHERE id = ? LIMIT 1", (detection_id,))
            ).fetchone()
        if row is None:
            return None
        raw = dict(row)
        raw["source_sensors"] = _json_loads(raw.pop("source_sensors_json"), [])
        raw["source_observation_ids"] = _json_loads(raw.pop("source_observation_ids_json"), [])
        raw["zone_ids"] = _json_loads(raw.pop("zone_ids_json"), [])
        raw["tdoa_s"] = _json_loads(raw.pop("tdoa_json"), {})
        raw["classifier_scores"] = _json_loads(raw.pop("classifier_scores_json"), {})
        raw["feature_summary"] = _json_loads(raw.pop("feature_summary_json"), {})
        if not raw.get("retention_tier"):
            raw["retention_tier"] = "short"
        return raw

    async def snippet_path_for_detection(self, detection_id: str) -> str | None:
        db = self._require_db()
        async with self._lock:
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
        async with self._lock:
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

            await db.commit()
        return len(rows)

    async def cleanup_retention(self, *, now_ns: int, tier_ttls_seconds: dict[str, int]) -> dict[str, int]:
        db = self._require_db()
        summary = {
            "observations": 0,
            "detections": 0,
            "pings": 0,
            "large_artifacts": 0,
        }
        protected_tiers = {"config", "permanent"}

        async with self._lock:
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
                        AND (expires_ns IS NOT NULL AND expires_ns <= ? OR created_ns <= ?)
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
                    AND (expires_ns IS NOT NULL AND expires_ns <= ? OR created_ns <= ?)
                    """,
                    (tier, now_ns, threshold_ns),
                )
                summary["large_artifacts"] += max(0, artifact_cursor.rowcount)

            await db.commit()

        return summary

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Storage is not initialized")
        return self._db
