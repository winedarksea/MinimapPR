"""Value types for the Home Assistant bridge: resolved config and metrics.

Separate from ``bridge.py`` so ``state_mapper`` and ``entity_set`` can import the
config type without importing the bridge itself — they need the shape, not the
machinery, and a one-way import is easier to reason about than a TYPE_CHECKING
guard in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class HassBridgeConfig:
    """Fully-resolved bridge configuration, derived from ``Settings``.

    ``enabled`` already folds in "a broker host was actually supplied", so the
    bridge itself never re-derives that condition.
    """

    enabled: bool
    mqtt_host: str
    mqtt_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_client_id: str = "minimappr"
    mqtt_keepalive_seconds: int = 60
    mqtt_tls_enabled: bool = False
    mqtt_tls_insecure: bool = False
    discovery_prefix: str = "homeassistant"
    base_topic: str = "minimappr"
    device_id: str = "minimappr"
    device_name: str = "MinimapPR"
    publish_interval_seconds: float = 5.0
    publish_min_interval_seconds: float = 1.0
    reconcile_interval_seconds: float = 60.0
    queue_size: int = 2000
    reconnect_backoff_initial_seconds: float = 1.0
    reconnect_backoff_max_seconds: float = 60.0
    detection_off_delay_seconds: int = 30
    detection_classes: tuple[str, ...] = ()
    track_slot_count: int = 8
    zone_spl_window_seconds: float = 60.0
    discovery_ledger_path: Path = Path("data/hass_discovery_ledger.json")
    publish_zone_occupancy: bool = True
    publish_zone_spl: bool = True
    publish_detection_classes: bool = True
    publish_node_status: bool = True
    publish_system_health: bool = True
    publish_events: bool = True
    publish_track_slots: bool = False
    version: str = "unknown"


@dataclass(slots=True)
class HassBridgeMetrics:
    messages_published: int = 0
    messages_failed: int = 0
    messages_dropped_queue_full: int = 0
    messages_suppressed_unchanged: int = 0
    messages_coalesced: int = 0
    live_events_dropped: int = 0
    reconnect_count: int = 0
    """Connection attempts, successful or not — the initial connect included. A
    number that keeps climbing is the signal an operator wants (a flapping link),
    which counting only *re*-connects would understate by one."""
    reconcile_count: int = 0
    rule_actions_queued: int = 0
    rule_actions_rejected: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "messages_published": self.messages_published,
            "messages_failed": self.messages_failed,
            "messages_dropped_queue_full": self.messages_dropped_queue_full,
            "messages_suppressed_unchanged": self.messages_suppressed_unchanged,
            "messages_coalesced": self.messages_coalesced,
            "live_events_dropped": self.live_events_dropped,
            "reconnect_count": self.reconnect_count,
            "reconcile_count": self.reconcile_count,
            "rule_actions_queued": self.rule_actions_queued,
            "rule_actions_rejected": self.rule_actions_rejected,
        }
