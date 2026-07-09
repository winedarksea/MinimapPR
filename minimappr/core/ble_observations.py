"""Short-TTL BLE RSSI observation store."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from minimappr.models import BleObservationIn


_HEX_RE = re.compile(r"[^0-9a-fA-F]")


@dataclass(slots=True)
class BleObservationRecord:
    node_id: str
    mac: str
    addr_type: int
    rssi_last: int
    rssi_ewma: float
    rssi_min: int
    rssi_max: int
    count: int
    recv_ns: int
    age_ms: int
    adv_type: int | None
    name: str | None
    boot_count: int | None = None
    boot_id: int | None = None
    monotonic_ms: int | None = None

    def as_raw_api_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "mac": self.mac,
            "addr_type": self.addr_type,
            "rssi_last": self.rssi_last,
            "rssi_ewma": self.rssi_ewma,
            "rssi_min": self.rssi_min,
            "rssi_max": self.rssi_max,
            "count": self.count,
            "recv_ns": self.recv_ns,
            "age_ms": self.age_ms,
            "adv_type": self.adv_type,
            "name": self.name,
            "boot_count": self.boot_count,
            "boot_id": self.boot_id,
            "monotonic_ms": self.monotonic_ms,
        }


def normalize_ble_mac(value: str) -> str:
    compact = _HEX_RE.sub("", value or "").lower()
    if len(compact) == 12:
        return ":".join(compact[index : index + 2] for index in range(0, 12, 2))
    return str(value or "").strip().lower()


class BleObservationStore:
    """Bounded latest-observation table for BLE RSSI reports.

    The firmware reports per-second aggregates, so retaining the latest aggregate
    for each `(device, node)` plus a short TTL gives the multilateration step the
    current receiver set without landing scan noise on disk.
    """

    def __init__(self, *, ttl_seconds: float = 30.0, max_entries: int = 2048) -> None:
        self._ttl_ns = int(max(ttl_seconds, 1.0) * 1_000_000_000)
        self._max_entries = max(1, int(max_entries))
        self._records_by_device_node: dict[tuple[str, str], BleObservationRecord] = {}
        self._lock = asyncio.Lock()

    async def ingest(
        self,
        *,
        node_id: str,
        observations: list[BleObservationIn],
        recv_ns: int,
        boot_count: int | None = None,
        boot_id: int | None = None,
        monotonic_ms: int | None = None,
    ) -> int:
        accepted = 0
        async with self._lock:
            self._prune_locked(recv_ns)
            for observation in observations:
                mac = normalize_ble_mac(observation.mac)
                if not mac:
                    continue
                self._records_by_device_node[(mac, node_id)] = BleObservationRecord(
                    node_id=node_id,
                    mac=mac,
                    addr_type=observation.addr_type,
                    rssi_last=observation.rssi_last,
                    rssi_ewma=float(observation.rssi_ewma),
                    rssi_min=observation.rssi_min,
                    rssi_max=observation.rssi_max,
                    count=observation.count,
                    recv_ns=recv_ns,
                    age_ms=observation.age_ms,
                    adv_type=observation.adv_type,
                    name=observation.name,
                    boot_count=boot_count,
                    boot_id=boot_id,
                    monotonic_ms=monotonic_ms,
                )
                accepted += 1
            self._evict_locked()
        return accepted

    async def latest_by_device(self, *, now_ns: int) -> dict[str, list[BleObservationRecord]]:
        async with self._lock:
            self._prune_locked(now_ns)
            grouped: dict[str, list[BleObservationRecord]] = {}
            for record in self._records_by_device_node.values():
                grouped.setdefault(record.mac, []).append(record)
            for records in grouped.values():
                records.sort(key=lambda record: record.node_id)
            return grouped

    async def latest_for_node(self, node_id: str, *, now_ns: int) -> list[BleObservationRecord]:
        async with self._lock:
            self._prune_locked(now_ns)
            records = [
                record
                for record in self._records_by_device_node.values()
                if record.node_id == node_id
            ]
            records.sort(key=lambda record: (record.mac, record.addr_type))
            return records

    async def size(self, *, now_ns: int) -> int:
        async with self._lock:
            self._prune_locked(now_ns)
            return len(self._records_by_device_node)

    def _prune_locked(self, now_ns: int) -> None:
        cutoff_ns = now_ns - self._ttl_ns
        self._records_by_device_node = {
            key: record
            for key, record in self._records_by_device_node.items()
            if record.recv_ns >= cutoff_ns
        }

    def _evict_locked(self) -> None:
        overflow = len(self._records_by_device_node) - self._max_entries
        if overflow <= 0:
            return
        oldest_keys = sorted(
            self._records_by_device_node,
            key=lambda key: self._records_by_device_node[key].recv_ns,
        )[:overflow]
        for key in oldest_keys:
            self._records_by_device_node.pop(key, None)
