"""Pure MQTT topic builders and object-id slugification.

Zone/node ids come from operator input and may contain spaces, slashes, or
non-ASCII text; MQTT topic levels cannot contain ``+``/``#`` and HA object ids
must match ``[a-zA-Z0-9_-]``. ``slugify`` is the single normalization point:
deterministic (same input always yields the same slug, across processes and
restarts) because the slug is baked into the HA ``unique_id``, which HA's entity
registry persists forever.
"""

from __future__ import annotations

import hashlib
import re

_SLUG_MAX_LENGTH = 48
_SLUG_HASH_LENGTH = 6
_DISALLOWED_CHARS = re.compile(r"[^a-z0-9_]+")
_TOPIC_LEVEL_DISALLOWED = ("+", "#", "/")


def slugify(value: str) -> str:
    """Normalize an arbitrary id into a stable MQTT/HA-safe object id fragment.

    Truncated slugs get a sha1-derived suffix so two long ids sharing a prefix
    do not collapse onto the same entity. The hash is over the *original* value,
    so the suffix is stable for the life of that zone/node id.
    """
    raw = str(value)
    normalized = _DISALLOWED_CHARS.sub("_", raw.strip().lower()).strip("_")
    if not normalized:
        # An id made entirely of punctuation still needs a distinct, stable slug.
        return f"x_{_short_hash(raw)}"
    if len(normalized) <= _SLUG_MAX_LENGTH:
        return normalized
    keep = _SLUG_MAX_LENGTH - _SLUG_HASH_LENGTH - 1
    return f"{normalized[:keep].rstrip('_')}_{_short_hash(raw)}"


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:_SLUG_HASH_LENGTH]


def is_valid_topic_level(value: str) -> bool:
    """True when ``value`` can be used as a single MQTT topic level."""
    text = str(value)
    if not text.strip():
        return False
    return not any(char in text for char in _TOPIC_LEVEL_DISALLOWED)


class HassTopics:
    """Topic builders for one bridge instance.

    ``discovery_prefix`` is HA's config topic root (default ``homeassistant``);
    ``base_topic`` is ours (default ``minimappr``). They are deliberately
    separate roots: purging our state must never touch HA's discovery tree of
    other integrations, and the rule-action topic guard keys off ``base_topic``.
    """

    __slots__ = ("discovery_prefix", "base_topic", "device_id")

    def __init__(self, *, discovery_prefix: str, base_topic: str, device_id: str) -> None:
        self.discovery_prefix = discovery_prefix.strip().strip("/")
        self.base_topic = base_topic.strip().strip("/")
        self.device_id = slugify(device_id)

    # -- discovery ----------------------------------------------------------

    def discovery(self, component: str, object_id: str) -> str:
        return f"{self.discovery_prefix}/{component}/{self.device_id}/{object_id}/config"

    def unique_id(self, object_id: str) -> str:
        return f"minimappr_{self.device_id}_{object_id}"

    # -- availability -------------------------------------------------------

    @property
    def availability(self) -> str:
        return f"{self.base_topic}/status"

    # -- zones --------------------------------------------------------------

    def zone_occupancy(self, zone_id: str) -> str:
        return f"{self.base_topic}/zone/{slugify(zone_id)}/occupancy"

    def zone_occupancy_attributes(self, zone_id: str) -> str:
        return f"{self.base_topic}/zone/{slugify(zone_id)}/attributes"

    def zone_spl(self, zone_id: str) -> str:
        return f"{self.base_topic}/zone/{slugify(zone_id)}/spl_db"

    # -- detections ---------------------------------------------------------

    def detection_class(self, label: str) -> str:
        return f"{self.base_topic}/detection_class/{slugify(label)}"

    # -- nodes --------------------------------------------------------------

    def node_connectivity(self, node_id: str) -> str:
        return f"{self.base_topic}/node/{slugify(node_id)}/connectivity"

    def node_attributes(self, node_id: str) -> str:
        return f"{self.base_topic}/node/{slugify(node_id)}/attributes"

    # -- system -------------------------------------------------------------

    @property
    def system_health(self) -> str:
        return f"{self.base_topic}/system/health"

    @property
    def system_health_attributes(self) -> str:
        return f"{self.base_topic}/system/attributes"

    @property
    def active_track_count(self) -> str:
        return f"{self.base_topic}/system/active_track_count"

    # -- track slots --------------------------------------------------------

    def track_slot(self, slot_index: int) -> str:
        return f"{self.base_topic}/track/{slot_index:02d}/state"

    def track_slot_attributes(self, slot_index: int) -> str:
        return f"{self.base_topic}/track/{slot_index:02d}/attributes"

    # -- impulse events -----------------------------------------------------

    @property
    def detection_event(self) -> str:
        return f"{self.base_topic}/event/detection"

    @property
    def alert_event(self) -> str:
        return f"{self.base_topic}/event/alert"

    # -- rule-authored topics ----------------------------------------------

    def rule_topic(self, requested: str) -> str | None:
        """Normalize a rule-authored topic into our namespace, or None if unsafe.

        This is a safety control, not cosmetics: without it a stored rule could
        name ``homeassistant/binary_sensor/.../config`` and overwrite every
        discovery payload on the broker, or use a wildcard the broker rejects.
        Anything that is not a plain relative topic is refused outright rather
        than sanitized — silently rewriting an operator's topic would be worse
        than telling them it was invalid.
        """
        text = str(requested or "").strip()
        if not text:
            return None
        if any(char in text for char in ("+", "#")):
            return None
        if text.startswith("/"):
            # A leading slash creates an empty first level; almost always a typo.
            return None
        relative = text.strip("/")
        if not relative or ".." in relative.split("/"):
            return None
        if any(not level for level in relative.split("/")):
            return None
        # Accept an already-prefixed topic idempotently so a rule authored with
        # the full topic behaves the same as one authored relatively.
        prefix = f"{self.base_topic}/"
        if relative == self.base_topic or relative.startswith(prefix):
            return relative
        return f"{prefix}{relative}"
