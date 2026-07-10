"""Classify firmware audio gaps from node-reported transport telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


HealthVerdict = Literal[
    "HEALTHY",
    "LOSSY_RATE_LIMITED",
    "LOSSY_WIFI",
    "NODE_RESTARTED",
    "SERVER_GAP_ONLY",
]


_COUNTER_KEYS = (
    "runner_frames_dropped",
    "runner_queue_overflows",
    "runner_publish_errors",
    "runner_publish_timeout_failures",
    "runner_publish_connect_or_reset_failures",
    "runner_publish_dns_failures",
    "runner_publish_wifi_down_failures",
)


@dataclass(slots=True)
class IngestHealthSnapshot:
    boot_id: int | None = None
    counters: dict[str, int] = field(default_factory=dict)


def runner_stats_from_timing_diagnostics(timing_diagnostics: dict | None) -> dict[str, Any] | None:
    if not isinstance(timing_diagnostics, dict):
        return None

    stats: dict[str, Any] = {}
    for key, value in timing_diagnostics.items():
        if key.startswith("runner_") and value is not None:
            stats[key] = value

    transport_health = timing_diagnostics.get("transport_health")
    if isinstance(transport_health, dict):
        stats["transport_health"] = dict(transport_health)

    return stats or None


class IngestHealthClassifier:
    """Stateful per-node verdicts for sequence gaps and transport stress."""

    def __init__(self, *, saturated_current_queue_ratio: float = 0.95, weak_wifi_rssi_dbm: int = -75) -> None:
        self._snapshots: dict[str, IngestHealthSnapshot] = {}
        self._saturated_current_queue_ratio = saturated_current_queue_ratio
        self._weak_wifi_rssi_dbm = weak_wifi_rssi_dbm

    def observe(
        self,
        *,
        node_id: str,
        sequence_gap_count: int,
        timing_diagnostics: dict | None,
    ) -> dict[str, Any]:
        transport_health = {}
        if isinstance(timing_diagnostics, dict) and isinstance(timing_diagnostics.get("transport_health"), dict):
            transport_health = dict(timing_diagnostics["transport_health"])
        if isinstance(timing_diagnostics, dict):
            queue_depth = self._optional_int(timing_diagnostics.get("runner_queue_depth"))
            if queue_depth is not None:
                transport_health["current_queue_depth"] = queue_depth

        current_counters = self._extract_counters(timing_diagnostics)
        current_boot_id = self._optional_int(transport_health.get("boot_id"))
        previous = self._snapshots.get(node_id)
        deltas = self._counter_deltas(previous.counters if previous else {}, current_counters)
        boot_changed = (
            previous is not None
            and previous.boot_id is not None
            and current_boot_id is not None
            and current_boot_id != previous.boot_id
        )

        verdict = self._classify(
            sequence_gap_count=sequence_gap_count,
            boot_changed=boot_changed,
            deltas=deltas,
            transport_health=transport_health,
        )
        result = {
            "verdict": verdict,
            "sequence_gap_count": max(0, int(sequence_gap_count or 0)),
            "counter_deltas": deltas,
            "boot_id": current_boot_id,
            "transport_health": transport_health,
        }
        self._snapshots[node_id] = IngestHealthSnapshot(
            boot_id=current_boot_id if current_boot_id is not None else (previous.boot_id if previous else None),
            counters=current_counters,
        )
        return result

    def _classify(
        self,
        *,
        sequence_gap_count: int,
        boot_changed: bool,
        deltas: dict[str, int],
        transport_health: dict[str, Any],
    ) -> HealthVerdict:
        if boot_changed:
            return "NODE_RESTARTED"
        if self._rate_limited(deltas=deltas, transport_health=transport_health):
            return "LOSSY_RATE_LIMITED"
        if self._wifi_loss(deltas=deltas, transport_health=transport_health):
            return "LOSSY_WIFI"
        if sequence_gap_count > 0:
            return "SERVER_GAP_ONLY"
        return "HEALTHY"

    def _rate_limited(self, *, deltas: dict[str, int], transport_health: dict[str, Any]) -> bool:
        if deltas.get("runner_frames_dropped", 0) > 0 or deltas.get("runner_queue_overflows", 0) > 0:
            return True

        # High-water marks are boot-lifetime diagnostics. Using them as current
        # pressure keeps a node degraded forever after a single burst. The
        # firmware reports current depth on every packet and advertises a
        # capacity including the active publish batch.
        queue_depth = self._optional_int(transport_health.get("current_queue_depth"))
        capacity = self._optional_int(transport_health.get("queue_slots_capacity"))
        return (
            queue_depth is not None
            and capacity is not None
            and capacity > 0
            and queue_depth >= int(capacity * self._saturated_current_queue_ratio)
        )

    def _wifi_loss(self, *, deltas: dict[str, int], transport_health: dict[str, Any]) -> bool:
        publish_delta = (
            deltas.get("runner_publish_errors", 0)
            + deltas.get("runner_publish_timeout_failures", 0)
            + deltas.get("runner_publish_connect_or_reset_failures", 0)
            + deltas.get("runner_publish_dns_failures", 0)
            + deltas.get("runner_publish_wifi_down_failures", 0)
        )
        if publish_delta > 0:
            return True
        wifi_rssi = self._optional_int(transport_health.get("wifi_rssi_dbm"))
        return wifi_rssi is not None and wifi_rssi <= self._weak_wifi_rssi_dbm

    @classmethod
    def _extract_counters(cls, timing_diagnostics: dict | None) -> dict[str, int]:
        if not isinstance(timing_diagnostics, dict):
            return {}
        counters = {}
        for key in _COUNTER_KEYS:
            value = cls._optional_int(timing_diagnostics.get(key))
            if value is not None:
                counters[key] = value
        return counters

    @staticmethod
    def _counter_deltas(previous: dict[str, int], current: dict[str, int]) -> dict[str, int]:
        return {
            key: max(0, value - int(previous.get(key, 0)))
            for key, value in current.items()
        }

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
