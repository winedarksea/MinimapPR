"""Bounded in-memory state for the live ingest path.

Raw audio frames are intentionally not durable before they become detections or
recordings.  The small amount of state needed to reject immediate retransmits,
remember node registrations, and sample environment history belongs in memory
too; retaining it in SQLite turns packet-rate telemetry into disk churn.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from minimappr.models import NodeSpec


@dataclass(frozen=True, slots=True)
class FrameIdentity:
    """Audio-free identity used to reject a retransmitted live frame."""

    node_id: str
    boot_session: str
    source_type: str
    start_sample_index: int | None
    end_sample_index: int | None
    start_time_ns: int | None
    frame_sequence: int | None

    @classmethod
    def from_frame(
        cls,
        *,
        node_id: str,
        boot_session: str,
        source_type: str,
        start_sample_index: int | None,
        end_sample_index: int | None,
        start_time_ns: int,
        frame_sequence: int | None,
    ) -> "FrameIdentity":
        return cls(
            node_id=node_id,
            boot_session=boot_session or "boot-unknown",
            source_type=source_type,
            start_sample_index=start_sample_index,
            end_sample_index=end_sample_index,
            start_time_ns=(
                None
                if start_sample_index is not None and end_sample_index is not None
                else start_time_ns
            ),
            frame_sequence=frame_sequence,
        )


class LiveIngestState:
    """Owns non-durable replay, node, and environment persistence decisions."""

    def __init__(
        self,
        *,
        frame_dedup_window_seconds: float = 120.0,
        frame_dedup_max_entries_per_node: int = 4_096,
        environment_persistence_interval_seconds: float = 60.0,
        node_liveness_persistence_interval_seconds: float = 5.0,
    ) -> None:
        self._frame_dedup_window_ns = int(frame_dedup_window_seconds * 1_000_000_000)
        self._frame_dedup_max_entries_per_node = frame_dedup_max_entries_per_node
        self._environment_persistence_interval_ns = int(
            environment_persistence_interval_seconds * 1_000_000_000
        )
        # Comfortably below the smallest sensible node_degraded_after_seconds so a
        # healthy node never drifts into "degraded" between heartbeat writes.
        self._node_liveness_persistence_interval_ns = int(
            node_liveness_persistence_interval_seconds * 1_000_000_000
        )
        self._completed_frame_keys_by_node: dict[str, OrderedDict[FrameIdentity, int]] = {}
        self._reserved_frame_keys_by_node: dict[str, set[FrameIdentity]] = {}
        self._node_fingerprints: dict[str, str] = {}
        self._last_node_liveness_persist_ns: dict[str, int] = {}
        self._last_environment_bucket_by_node: dict[str, int] = {}
        self._audio_summaries_by_node: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def reserve_frame(self, identity: FrameIdentity) -> bool:
        """Reserve a frame until its buffer insertion commits or aborts.

        A reservation prevents concurrent retransmits from being processed
        twice, but deliberately does not turn an exception into permanent data
        loss. Callers must commit after buffer insertion or release on failure.
        """
        now_ns = time.monotonic_ns()
        async with self._lock:
            entries = self._completed_frame_keys_by_node.setdefault(identity.node_id, OrderedDict())
            reservations = self._reserved_frame_keys_by_node.setdefault(identity.node_id, set())
            expires_before_ns = now_ns - self._frame_dedup_window_ns
            while entries:
                _, inserted_ns = next(iter(entries.items()))
                if inserted_ns > expires_before_ns:
                    break
                entries.popitem(last=False)
            if identity in entries or identity in reservations:
                return False
            reservations.add(identity)
            return True

    async def commit_reserved_frame(self, identity: FrameIdentity) -> None:
        """Mark a successfully buffered frame as a completed live ingest."""
        now_ns = time.monotonic_ns()
        async with self._lock:
            reservations = self._reserved_frame_keys_by_node.setdefault(identity.node_id, set())
            reservations.discard(identity)
            entries = self._completed_frame_keys_by_node.setdefault(identity.node_id, OrderedDict())
            entries[identity] = now_ns
            while len(entries) > self._frame_dedup_max_entries_per_node:
                entries.popitem(last=False)

    async def release_reserved_frame(self, identity: FrameIdentity) -> None:
        """Allow a retransmit after processing failed before buffer insertion."""
        async with self._lock:
            reservations = self._reserved_frame_keys_by_node.get(identity.node_id)
            if reservations is not None:
                reservations.discard(identity)

    async def claim_processed_frame(self, identity: FrameIdentity) -> bool:
        """Compatibility helper for callers that have already completed work."""
        if not await self.reserve_frame(identity):
            return False
        await self.commit_reserved_frame(identity)
        return True

    async def should_persist_node_row(self, node: NodeSpec, *, now_ns: int) -> bool:
        """Persist a changed registration, or refresh a stale persisted heartbeat.

        The static fingerprint deliberately ignores position and metadata so
        per-frame telemetry does not churn the row.  On its own that freezes the
        stored ``last_seen_ns`` at the node's very first frame.  In a combined
        process nothing notices, because ``/api/v1/nodes`` overlays the live
        in-memory registry; but when the API runs as its own process
        (``process_role="api"``) that overlay is skipped and the frozen column is
        the only liveness signal there is, so every healthy node reads "offline"
        forever.  A bounded heartbeat write keeps the persisted row honest at a
        cost of one small UPDATE per node per interval.
        """
        fingerprint = _static_node_fingerprint(node)
        async with self._lock:
            previous_fingerprint = self._node_fingerprints.get(node.id)
            if previous_fingerprint != fingerprint:
                self._node_fingerprints[node.id] = fingerprint
                self._last_node_liveness_persist_ns[node.id] = now_ns
                return True
            last_persist_ns = self._last_node_liveness_persist_ns.get(node.id)
            if (
                last_persist_ns is not None
                and now_ns - last_persist_ns < self._node_liveness_persistence_interval_ns
            ):
                return False
            self._last_node_liveness_persist_ns[node.id] = now_ns
            return True

    async def should_persist_environment_sample(self, *, node_id: str, timestamp_ns: int) -> bool:
        """Allow the first source-timestamped sample in each persistence bucket."""
        bucket = timestamp_ns // self._environment_persistence_interval_ns
        async with self._lock:
            previous_bucket = self._last_environment_bucket_by_node.get(node_id)
            if previous_bucket is not None and bucket <= previous_bucket:
                return False
            self._last_environment_bucket_by_node[node_id] = bucket
            return True

    async def record_audio_summary(self, *, node_id: str, summary: dict[str, Any]) -> None:
        async with self._lock:
            self._audio_summaries_by_node[node_id] = dict(summary)

    async def audio_summary(self, node_id: str) -> dict[str, Any] | None:
        async with self._lock:
            summary = self._audio_summaries_by_node.get(node_id)
            return dict(summary) if summary is not None else None


def _static_node_fingerprint(node: NodeSpec) -> str:
    """Exclude per-frame location and telemetry from registration persistence."""
    payload: dict[str, Any] = node.model_dump(mode="json")
    payload.pop("position_m", None)
    payload.pop("position_geo", None)
    payload.pop("metadata", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def frame_identity_key(
    *,
    node_id: str,
    boot_session: str = "",
    frame_sequence: int | None,
    start_time_ns: int,
    utc_end_ns: int | None = None,
    start_sample_index: int | None,
    end_sample_index: int | None,
    source_type: str,
    time_quality: str = "",
    tor_ns: int | None = None,
) -> FrameIdentity:
    """Compatibility helper for asserting the live frame identity contract."""
    _ = (utc_end_ns, time_quality, tor_ns)
    return FrameIdentity.from_frame(
        node_id=node_id,
        boot_session=boot_session,
        source_type=source_type,
        start_sample_index=start_sample_index,
        end_sample_index=end_sample_index,
        start_time_ns=start_time_ns,
        frame_sequence=frame_sequence,
    )
