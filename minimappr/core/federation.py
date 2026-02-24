"""Federated fusion peer exchange and deconfliction runtime."""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx
import numpy as np
from pydantic import ValidationError

from minimappr.config import FederationConfig, FederationPeerConfig, Settings
from minimappr.models import FederationHeartbeat, FederationTrackSnapshot, TrackState


ACTIVE_TRACK_STATUSES = {"tentative", "confirmed", "coasting"}


TrackSupplier = Callable[[int], Awaitable[list[TrackState]]]


@dataclass(slots=True)
class PeerTrackRecord:
    track: TrackState
    received_ns: int
    generated_ns: int


@dataclass(slots=True)
class PeerLinkRuntime:
    config: FederationPeerConfig
    link_state: str = "offline"
    last_seen_ns: int | None = None
    last_heartbeat_sent_ns: int | None = None
    last_heartbeat_ack_ns: int | None = None
    last_snapshot_sent_ns: int | None = None
    last_snapshot_ack_ns: int | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    last_rtt_ms: float | None = None
    sent_track_count: int = 0
    received_track_count: int = 0


@dataclass(slots=True)
class FederationMetrics:
    heartbeats_sent: int = 0
    heartbeat_failures: int = 0
    snapshots_sent: int = 0
    snapshot_failures: int = 0
    snapshots_received: int = 0
    unknown_peer_messages: int = 0
    auth_rejections: int = 0
    peer_tracks_stale_dropped: int = 0
    deconflict_pairs_evaluated: int = 0
    deconflict_standby_local: int = 0
    deconflict_standby_peer: int = 0


class FederationCoordinator:
    def __init__(
        self,
        *,
        settings: Settings,
        track_supplier: TrackSupplier,
        live_callback: Callable[[dict], Awaitable[None]] | None = None,
    ) -> None:
        self._config: FederationConfig = settings.federation_config()
        self._track_supplier = track_supplier
        self._live_callback = live_callback
        self._links: dict[str, PeerLinkRuntime] = {
            peer.peer_id: PeerLinkRuntime(config=peer)
            for peer in self._config.peers
            if peer.enabled
        }
        self._peer_tracks: dict[str, dict[str, PeerTrackRecord]] = {
            peer_id: {} for peer_id in self._links
        }
        self._pair_owners: dict[str, str] = {}
        self._metrics = FederationMetrics()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._client = httpx.AsyncClient(timeout=self._config.request_timeout_seconds)

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def server_id(self) -> str:
        return self._config.server_id

    async def start(self) -> None:
        if not self.enabled:
            return
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop(), name="federation-coordinator")

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._client.aclose()

    async def status(self, *, now_ns: int | None = None) -> dict:
        now = now_ns if now_ns is not None else time.time_ns()
        async with self._lock:
            peers = []
            peer_track_count = 0
            for peer_id, runtime in self._links.items():
                peer_tracks = self._peer_tracks.get(peer_id, {})
                peer_track_count += len(peer_tracks)
                peers.append(
                    {
                        "peer_id": peer_id,
                        "base_url": runtime.config.base_url,
                        "link_state": self._compute_link_state(runtime, now_ns=now),
                        "last_seen_ns": runtime.last_seen_ns,
                        "last_heartbeat_sent_ns": runtime.last_heartbeat_sent_ns,
                        "last_heartbeat_ack_ns": runtime.last_heartbeat_ack_ns,
                        "last_snapshot_sent_ns": runtime.last_snapshot_sent_ns,
                        "last_snapshot_ack_ns": runtime.last_snapshot_ack_ns,
                        "last_error": runtime.last_error,
                        "last_rtt_ms": runtime.last_rtt_ms,
                        "consecutive_failures": runtime.consecutive_failures,
                        "sent_track_count": runtime.sent_track_count,
                        "received_track_count": runtime.received_track_count,
                    }
                )
            metrics = {
                "heartbeats_sent": self._metrics.heartbeats_sent,
                "heartbeat_failures": self._metrics.heartbeat_failures,
                "snapshots_sent": self._metrics.snapshots_sent,
                "snapshot_failures": self._metrics.snapshot_failures,
                "snapshots_received": self._metrics.snapshots_received,
                "unknown_peer_messages": self._metrics.unknown_peer_messages,
                "auth_rejections": self._metrics.auth_rejections,
                "peer_tracks_stale_dropped": self._metrics.peer_tracks_stale_dropped,
                "deconflict_pairs_evaluated": self._metrics.deconflict_pairs_evaluated,
                "deconflict_standby_local": self._metrics.deconflict_standby_local,
                "deconflict_standby_peer": self._metrics.deconflict_standby_peer,
            }
        return {
            "enabled": self.enabled,
            "server_id": self.server_id,
            "peer_count": len(peers),
            "peer_track_count": peer_track_count,
            "peers": peers,
            "metrics": metrics,
            "publish_interval_seconds": self._config.publish_interval_seconds,
            "heartbeat_interval_seconds": self._config.heartbeat_interval_seconds,
            "link_timeout_seconds": self._config.link_timeout_seconds,
            "track_ttl_seconds": self._config.track_ttl_seconds,
        }

    async def validate_inbound_auth(
        self,
        *,
        peer_id: str,
        authorization_header: str | None,
        token_header: str | None,
    ) -> bool:
        expected = self._expected_token_for_peer(peer_id)
        if expected is None:
            return True
        provided = _extract_token(authorization_header, token_header)
        if provided == expected:
            return True
        async with self._lock:
            self._metrics.auth_rejections += 1
        return False

    async def handle_incoming_heartbeat(self, heartbeat: FederationHeartbeat, *, now_ns: int | None = None) -> bool:
        now = now_ns if now_ns is not None else time.time_ns()
        async with self._lock:
            runtime = self._links.get(heartbeat.server_id)
            if runtime is None:
                self._metrics.unknown_peer_messages += 1
                return False
            runtime.last_seen_ns = now
            runtime.last_heartbeat_ack_ns = now
            runtime.last_error = None
            runtime.consecutive_failures = 0
            runtime.link_state = self._compute_link_state(runtime, now_ns=now)
        return True

    async def handle_incoming_snapshot(
        self,
        snapshot: FederationTrackSnapshot,
        *,
        now_ns: int | None = None,
    ) -> bool:
        now = now_ns if now_ns is not None else time.time_ns()
        async with self._lock:
            runtime = self._links.get(snapshot.server_id)
            if runtime is None:
                self._metrics.unknown_peer_messages += 1
                return False
            runtime.last_seen_ns = now
            runtime.last_snapshot_ack_ns = now
            runtime.last_error = None
            runtime.consecutive_failures = 0
            runtime.link_state = self._compute_link_state(runtime, now_ns=now)

            staged: dict[str, PeerTrackRecord] = {}
            for track in snapshot.tracks:
                if track.status not in ACTIVE_TRACK_STATUSES:
                    continue
                staged[track.id] = PeerTrackRecord(
                    track=track,
                    received_ns=now,
                    generated_ns=snapshot.generated_ns,
                )
            self._peer_tracks[snapshot.server_id] = staged
            runtime.received_track_count = len(staged)
            self._metrics.snapshots_received += 1

        if self._live_callback is not None:
            payload = {
                "event_type": "peer_track_snapshot",
                "source_type": "peer_track",
                "server_id": snapshot.server_id,
                "track_count": len(snapshot.tracks),
                "server_time_ns": now,
            }
            await self._safe_live_broadcast(payload)
        return True

    async def merged_tracks(
        self,
        *,
        local_tracks: list[dict] | list[TrackState],
        now_ns: int | None = None,
        limit: int = 200,
        include_standby: bool = False,
    ) -> list[dict]:
        now = now_ns if now_ns is not None else time.time_ns()
        local_states: list[TrackState] = []
        for entry in local_tracks:
            track = _coerce_track(entry)
            if track is None or track.status not in ACTIVE_TRACK_STATUSES:
                continue
            local_states.append(track)

        async with self._lock:
            self._drop_stale_peer_tracks(now_ns=now)
            peer_items: list[tuple[str, TrackState]] = []
            for peer_id, tracks in self._peer_tracks.items():
                for record in tracks.values():
                    if record.track.status in ACTIVE_TRACK_STATUSES:
                        peer_items.append((peer_id, record.track))
            results = self._deconflict_tracks(
                local_tracks=local_states,
                peer_tracks=peer_items,
                include_standby=include_standby,
            )
        if limit > 0:
            return results[:limit]
        return results

    async def _loop(self) -> None:
        heartbeat_ns = int(self._config.heartbeat_interval_seconds * 1_000_000_000)
        publish_ns = int(self._config.publish_interval_seconds * 1_000_000_000)
        cadence_ns = min(heartbeat_ns, publish_ns, 500_000_000)
        next_heartbeat_ns = 0
        next_publish_ns = 0

        while True:
            now_ns = time.time_ns()
            if now_ns >= next_heartbeat_ns:
                await self._send_heartbeats(now_ns=now_ns)
                next_heartbeat_ns = now_ns + heartbeat_ns
            if now_ns >= next_publish_ns:
                await self._send_snapshots(now_ns=now_ns)
                next_publish_ns = now_ns + publish_ns
            async with self._lock:
                for runtime in self._links.values():
                    runtime.link_state = self._compute_link_state(runtime, now_ns=now_ns)
                self._drop_stale_peer_tracks(now_ns=now_ns)
            await asyncio.sleep(cadence_ns / 1_000_000_000.0)

    async def _send_heartbeats(self, *, now_ns: int) -> None:
        payload = FederationHeartbeat(server_id=self.server_id, sent_ns=now_ns).model_dump(mode="json")
        tasks = [
            asyncio.create_task(self._send_heartbeat_to_peer(peer_id=peer_id, payload=payload, now_ns=now_ns))
            for peer_id in self._links
        ]
        if tasks:
            await asyncio.gather(*tasks)

    async def _send_snapshots(self, *, now_ns: int) -> None:
        local_tracks = await self._track_supplier(now_ns)
        active_tracks = [track for track in local_tracks if track.status in ACTIVE_TRACK_STATUSES]
        snapshot = FederationTrackSnapshot(
            server_id=self.server_id,
            generated_ns=now_ns,
            source_type="local_track",
            tracks=active_tracks,
        )
        payload = snapshot.model_dump(mode="json")
        tasks = [
            asyncio.create_task(self._send_snapshot_to_peer(peer_id=peer_id, payload=payload, now_ns=now_ns))
            for peer_id in self._links
        ]
        if tasks:
            await asyncio.gather(*tasks)

    async def _send_heartbeat_to_peer(self, *, peer_id: str, payload: dict, now_ns: int) -> None:
        runtime = self._links.get(peer_id)
        if runtime is None:
            return
        url = f"{runtime.config.base_url}/api/v1/federation/heartbeat"
        headers = self._auth_headers_for_peer(peer_id)
        start = time.perf_counter()
        try:
            response = await self._client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except Exception as exc:
            await self._record_peer_failure(peer_id=peer_id, now_ns=now_ns, error=f"{type(exc).__name__}: {exc}")
            async with self._lock:
                self._metrics.heartbeat_failures += 1
            return
        rtt_ms = (time.perf_counter() - start) * 1000.0
        async with self._lock:
            runtime.last_heartbeat_sent_ns = payload["sent_ns"]
            runtime.last_heartbeat_ack_ns = now_ns
            runtime.last_seen_ns = now_ns
            runtime.last_rtt_ms = rtt_ms
            runtime.last_error = None
            runtime.consecutive_failures = 0
            runtime.link_state = self._compute_link_state(runtime, now_ns=now_ns)
            self._metrics.heartbeats_sent += 1

    async def _send_snapshot_to_peer(self, *, peer_id: str, payload: dict, now_ns: int) -> None:
        runtime = self._links.get(peer_id)
        if runtime is None:
            return
        url = f"{runtime.config.base_url}/api/v1/federation/snapshot"
        headers = self._auth_headers_for_peer(peer_id)
        start = time.perf_counter()
        try:
            response = await self._client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except Exception as exc:
            await self._record_peer_failure(peer_id=peer_id, now_ns=now_ns, error=f"{type(exc).__name__}: {exc}")
            async with self._lock:
                self._metrics.snapshot_failures += 1
            return
        rtt_ms = (time.perf_counter() - start) * 1000.0
        async with self._lock:
            runtime.last_snapshot_sent_ns = payload["generated_ns"]
            runtime.last_snapshot_ack_ns = now_ns
            runtime.last_seen_ns = now_ns
            runtime.last_rtt_ms = rtt_ms
            runtime.last_error = None
            runtime.consecutive_failures = 0
            runtime.link_state = self._compute_link_state(runtime, now_ns=now_ns)
            runtime.sent_track_count = len(payload.get("tracks", []))
            self._metrics.snapshots_sent += 1

    async def _record_peer_failure(self, *, peer_id: str, now_ns: int, error: str) -> None:
        async with self._lock:
            runtime = self._links.get(peer_id)
            if runtime is None:
                return
            runtime.consecutive_failures += 1
            runtime.last_error = error
            runtime.link_state = self._compute_link_state(runtime, now_ns=now_ns)

    def _compute_link_state(self, runtime: PeerLinkRuntime, *, now_ns: int) -> str:
        if runtime.last_seen_ns is None:
            return "offline"
        age_s = max(0.0, (now_ns - runtime.last_seen_ns) / 1_000_000_000.0)
        if age_s <= self._config.link_timeout_seconds and runtime.consecutive_failures <= 1:
            return "active"
        if age_s <= (self._config.link_timeout_seconds * 2.0):
            return "degraded"
        return "offline"

    def _auth_headers_for_peer(self, peer_id: str) -> dict[str, str]:
        token = self._expected_token_for_peer(peer_id)
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}

    def _expected_token_for_peer(self, peer_id: str) -> str | None:
        runtime = self._links.get(peer_id)
        if runtime is None:
            return None
        if runtime.config.api_key:
            return runtime.config.api_key
        return self._config.auth_token

    async def _safe_live_broadcast(self, payload: dict) -> None:
        if self._live_callback is None:
            return
        try:
            await self._live_callback(payload)
        except Exception:
            return

    def _drop_stale_peer_tracks(self, *, now_ns: int) -> None:
        ttl_ns = int(self._config.track_ttl_seconds * 1_000_000_000)
        for peer_id, tracks in self._peer_tracks.items():
            stale_ids = [
                track_id
                for track_id, record in tracks.items()
                if now_ns - record.received_ns > ttl_ns
            ]
            if not stale_ids:
                continue
            for track_id in stale_ids:
                tracks.pop(track_id, None)
                self._metrics.peer_tracks_stale_dropped += 1
            runtime = self._links.get(peer_id)
            if runtime is not None:
                runtime.received_track_count = len(tracks)

    def _deconflict_tracks(
        self,
        *,
        local_tracks: list[TrackState],
        peer_tracks: list[tuple[str, TrackState]],
        include_standby: bool,
    ) -> list[dict]:
        local_by_id = {track.id: track for track in local_tracks}
        standby_local_ids: set[str] = set()
        standby_peer_ids: set[tuple[str, str]] = set()
        local_owner_reason: dict[str, str] = {}
        peer_owner_reason: dict[tuple[str, str], str] = {}
        active_pairs: set[str] = set()

        for peer_id, peer_track in peer_tracks:
            match = self._best_local_match(local_tracks=local_tracks, peer_track=peer_track)
            if match is None:
                continue
            local_track, _distance = match
            pair_key = f"{peer_id}|{peer_track.id}|{local_track.id}"
            active_pairs.add(pair_key)
            self._metrics.deconflict_pairs_evaluated += 1

            local_tqi = float(local_track.tqi)
            peer_tqi = float(peer_track.tqi)
            previous_owner = self._pair_owners.get(pair_key)
            if peer_tqi > (local_tqi + self._config.tqi_hysteresis):
                owner = "peer"
                reason = "peer_higher_tqi"
            elif local_tqi > (peer_tqi + self._config.tqi_hysteresis):
                owner = "local"
                reason = "local_higher_tqi"
            elif previous_owner in {"peer", "local"}:
                owner = previous_owner
                reason = "hysteresis_hold"
            else:
                owner = "local"
                reason = "hysteresis_default_local"

            self._pair_owners[pair_key] = owner
            if owner == "peer":
                standby_local_ids.add(local_track.id)
                local_owner_reason[local_track.id] = reason
                peer_owner_reason[(peer_id, peer_track.id)] = reason
                self._metrics.deconflict_standby_local += 1
            else:
                standby_peer_ids.add((peer_id, peer_track.id))
                local_owner_reason[local_track.id] = reason
                peer_owner_reason[(peer_id, peer_track.id)] = reason
                self._metrics.deconflict_standby_peer += 1

        stale_pairs = [key for key in self._pair_owners if key not in active_pairs]
        for key in stale_pairs:
            self._pair_owners.pop(key, None)

        merged: list[dict] = []
        for track_id, track in local_by_id.items():
            is_standby = track_id in standby_local_ids
            if is_standby and not include_standby:
                continue
            item = track.model_dump(mode="json")
            item["source_type"] = "local_track"
            item["source_server_id"] = self.server_id
            item["ownership_role"] = "standby" if is_standby else "owner"
            item["ownership_reason"] = local_owner_reason.get(track_id, "local_only")
            merged.append(item)

        for peer_id, track in peer_tracks:
            key = (peer_id, track.id)
            is_standby = key in standby_peer_ids
            if is_standby and not include_standby:
                continue
            item = track.model_dump(mode="json")
            item["peer_track_id"] = track.id
            item["id"] = f"{peer_id}:{track.id}"
            item["source_type"] = "peer_track"
            item["source_server_id"] = peer_id
            item["ownership_role"] = "standby" if is_standby else "owner"
            item["ownership_reason"] = peer_owner_reason.get(key, "peer_only")
            merged.append(item)

        merged.sort(key=lambda row: int(row.get("last_seen_ns") or 0), reverse=True)
        return merged

    def _best_local_match(
        self,
        *,
        local_tracks: list[TrackState],
        peer_track: TrackState,
    ) -> tuple[TrackState, float] | None:
        best: tuple[TrackState, float] | None = None
        for local_track in local_tracks:
            distance = _mahalanobis_distance(
                local_track=local_track,
                peer_track=peer_track,
                default_variance=max(self._config.deconflict_mahalanobis_gate, 1.0) ** 2,
            )
            if distance > self._config.deconflict_mahalanobis_gate:
                continue
            if best is None or distance < best[1]:
                best = (local_track, distance)
        return best


def _coerce_track(value: dict | TrackState) -> TrackState | None:
    if isinstance(value, TrackState):
        return value
    if not isinstance(value, dict):
        return None
    try:
        return TrackState.model_validate(value)
    except ValidationError:
        return None


def _mahalanobis_distance(
    *,
    local_track: TrackState,
    peer_track: TrackState,
    default_variance: float,
) -> float:
    local_xy = np.asarray([float(local_track.position_m[0]), float(local_track.position_m[1])], dtype=np.float64)
    peer_xy = np.asarray([float(peer_track.position_m[0]), float(peer_track.position_m[1])], dtype=np.float64)
    delta = peer_xy - local_xy
    local_cov = _covariance_2d(local_track.position_covariance_m2, default_variance)
    peer_cov = _covariance_2d(peer_track.position_covariance_m2, default_variance)
    joint_cov = local_cov + peer_cov + np.eye(2, dtype=np.float64) * 1e-6
    try:
        inv_cov = np.linalg.pinv(joint_cov)
    except np.linalg.LinAlgError:
        inv_cov = np.eye(2, dtype=np.float64) * (1.0 / max(default_variance, 1e-6))
    d2 = float(delta.T @ inv_cov @ delta)
    if d2 <= 0.0:
        return 0.0
    return math.sqrt(d2)


def _covariance_2d(matrix: list[list[float]] | None, default_variance: float) -> np.ndarray:
    fallback = np.diag([default_variance, default_variance]).astype(np.float64)
    if not matrix or len(matrix) < 2:
        return fallback
    try:
        c00 = float(matrix[0][0])
        c01 = float(matrix[0][1])
        c10 = float(matrix[1][0]) if len(matrix[1]) > 0 else c01
        c11 = float(matrix[1][1])
    except (TypeError, ValueError, IndexError):
        return fallback
    if not np.isfinite(c00) or not np.isfinite(c11):
        return fallback
    c00 = max(c00, 1e-6)
    c11 = max(c11, 1e-6)
    cov = np.asarray(
        [
            [c00, c01],
            [c10, c11],
        ],
        dtype=np.float64,
    )
    return 0.5 * (cov + cov.T)


def _extract_token(authorization_header: str | None, token_header: str | None) -> str | None:
    if authorization_header:
        raw = authorization_header.strip()
        if raw.lower().startswith("bearer "):
            token = raw[7:].strip()
            if token:
                return token
        if raw:
            return raw
    if token_header:
        token = token_header.strip()
        if token:
            return token
    return None
