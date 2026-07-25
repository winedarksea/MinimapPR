"""Discovery reconciliation: diff the desired entity set against the ledger.

The ledger is what makes removal possible across restarts. Entities a previous
process published for zones or nodes that no longer exist are absent from the
desired set, so without a persisted record they would be invisible to us — their
retained config would sit on the broker forever and HA would keep showing an
unavailable entity with no way to clear it.
"""

from __future__ import annotations

from typing import Callable

from minimappr.core.hass.ledger import HassDiscoveryLedger, LedgerEntry, payload_sha256
from minimappr.core.hass.state_mapper import HassStateMapper, NodeStateInput, ZoneStateInput
from minimappr.core.hass.transport import MqttPublish


def reconcile_discovery(
    *,
    mapper: HassStateMapper,
    ledger: HassDiscoveryLedger,
    enqueue: Callable[[MqttPublish], bool],
    zones: tuple[ZoneStateInput, ...],
    nodes: tuple[NodeStateInput, ...],
) -> None:
    """Publish new/changed discovery configs, remove entities no longer desired.

    Unchanged entities are skipped by comparing the payload digest, so a steady
    site costs one ledger read per reconcile and no publishes at all.
    """
    desired = mapper.desired_discovery(zones=zones, nodes=nodes)

    for config_topic, entity in desired.items():
        message = mapper.discovery_publish(entity)
        digest = payload_sha256(message.payload)
        existing = ledger.get(config_topic)
        if existing is not None and existing.payload_sha256 == digest:
            continue
        enqueue(message)
        ledger.record(
            LedgerEntry(
                config_topic=config_topic,
                payload_sha256=digest,
                state_topics=entity.state_topics,
            )
        )

    for config_topic, entry in ledger.entries().items():
        if config_topic in desired:
            continue
        for message in mapper.removal_publishes(
            config_topic=config_topic, state_topics=entry.state_topics
        ):
            enqueue(message)
        ledger.forget(config_topic)

    ledger.save()
