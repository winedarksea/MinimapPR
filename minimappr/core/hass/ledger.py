"""Persistent record of what we last published to the broker.

Without this, a zone deleted while MinimapPR was stopped leaves its retained
discovery config on the broker forever: on restart the zone is absent from the
desired set, so nothing tells us to remove an entity we no longer know about.
HA keeps showing it as unavailable and the operator has no way to clear it short
of hand-editing the broker.

Written tmp-then-``os.replace`` so a crash mid-write cannot leave a truncated
ledger — a corrupt ledger reads as empty, which re-publishes everything (noisy
but correct) rather than orphaning entities.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_LEDGER_VERSION = 1


def payload_sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    config_topic: str
    payload_sha256: str
    state_topics: tuple[str, ...]


class HassDiscoveryLedger:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._entries: dict[str, LedgerEntry] = {}

    @property
    def path(self) -> Path:
        return self._path

    def entries(self) -> dict[str, LedgerEntry]:
        return dict(self._entries)

    def get(self, config_topic: str) -> LedgerEntry | None:
        return self._entries.get(config_topic)

    def load(self) -> None:
        """Read the ledger from disk. A missing or unreadable file reads as empty."""
        if not self._path.exists():
            self._entries = {}
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("hass discovery ledger %s unreadable (%s); treating as empty", self._path, exc)
            self._entries = {}
            return
        if not isinstance(raw, dict) or int(raw.get("version", 0)) != _LEDGER_VERSION:
            logger.warning("hass discovery ledger %s has an unexpected shape; treating as empty", self._path)
            self._entries = {}
            return
        entries: dict[str, LedgerEntry] = {}
        for topic, item in (raw.get("entities") or {}).items():
            if not isinstance(item, dict):
                continue
            entries[str(topic)] = LedgerEntry(
                config_topic=str(topic),
                payload_sha256=str(item.get("payload_sha256", "")),
                state_topics=tuple(str(value) for value in item.get("state_topics") or ()),
            )
        self._entries = entries

    def record(self, entry: LedgerEntry) -> None:
        self._entries[entry.config_topic] = entry

    def forget(self, config_topic: str) -> None:
        self._entries.pop(config_topic, None)

    def clear(self) -> None:
        self._entries = {}

    def invalidate_digests(self) -> None:
        """Blank every recorded digest so the next reconcile republishes all
        configs, while keeping the topic list that makes orphan removal possible.

        Used by the republish-discovery recovery path: dropping the entries
        outright would also forget the entities we still need to be able to delete.
        """
        self._entries = {
            topic: LedgerEntry(
                config_topic=entry.config_topic,
                payload_sha256="",
                state_topics=entry.state_topics,
            )
            for topic, entry in self._entries.items()
        }

    def save(self) -> None:
        payload = {
            "version": _LEDGER_VERSION,
            "entities": {
                topic: {
                    "payload_sha256": entry.payload_sha256,
                    "state_topics": list(entry.state_topics),
                }
                for topic, entry in sorted(self._entries.items())
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
        try:
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp_path, self._path)
        except OSError as exc:
            # A read-only data dir must not take the bridge down; the cost is
            # that orphans survive a restart, which the periodic reconcile then
            # cannot fix. Loud enough to notice, not fatal.
            logger.warning("hass discovery ledger %s could not be written: %s", self._path, exc)
            tmp_path.unlink(missing_ok=True)
