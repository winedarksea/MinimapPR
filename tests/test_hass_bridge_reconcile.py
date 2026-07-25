"""Discovery reconciliation and the ledger that makes cross-restart removal work."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minimappr.core.hass.ledger import HassDiscoveryLedger, LedgerEntry
from minimappr.core.hass.state_mapper import ZoneStateInput
from tests.hass_helpers import (
    build_test_bridge,
    node,
    snapshot,
    static_snapshot_provider,
    zone,
)

ZONE_CONFIG_TOPIC = "homeassistant/binary_sensor/minimappr/zone_occupancy_z1/config"
NODE_CONFIG_TOPIC = "homeassistant/binary_sensor/minimappr/node_online_n1/config"


@pytest.mark.asyncio
async def test_zones_and_nodes_get_retained_discovery_configs(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(zones=(zone("z1"),), nodes=(node("n1"),)))
    )

    await bridge._connect_once()

    retained = transport.retained()
    assert ZONE_CONFIG_TOPIC in retained
    assert NODE_CONFIG_TOPIC in retained
    payload = json.loads(retained[ZONE_CONFIG_TOPIC])
    assert payload["state_topic"] == "minimappr/zone/z1/occupancy"
    assert payload["device_class"] == "safety"
    await bridge.stop()


@pytest.mark.asyncio
async def test_unchanged_entity_is_not_republished(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(zone("z1"),))))
    await bridge._connect_once()
    transport.clear()

    bridge.request_reconcile()
    await bridge._publish_cycle_once()

    assert ZONE_CONFIG_TOPIC not in transport.topics()
    await bridge.stop()


@pytest.mark.asyncio
async def test_changed_entity_payload_is_republished(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(zone("z1"),))))
    await bridge._connect_once()
    transport.clear()

    # Renaming the zone changes the discovery payload for the same entity.
    renamed = ZoneStateInput(zone_id="z1", zone_name="Renamed", zone_type="alert_zone")
    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(renamed,))))
    bridge.request_reconcile()
    await bridge._publish_cycle_once()

    assert json.loads(transport.payload_for(ZONE_CONFIG_TOPIC))["name"] == "Renamed occupancy"
    await bridge.stop()


@pytest.mark.asyncio
async def test_deleting_a_zone_blanks_config_and_state_topics(tmp_path: Path, monkeypatch) -> None:
    """A retained state left behind is a zombie value for an entity HA dropped."""
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(zones=(zone("z1"), zone("z2"))))
    )
    await bridge._connect_once()
    assert "minimappr/zone/z2/occupancy" in transport.retained()
    transport.clear()

    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(zone("z1"),))))
    bridge.request_reconcile()
    await bridge._publish_cycle_once()

    blanked = {m.topic for m in transport.published if m.payload == "" and m.retain}
    assert "homeassistant/binary_sensor/minimappr/zone_occupancy_z2/config" in blanked
    assert "minimappr/zone/z2/occupancy" in blanked
    assert "minimappr/zone/z2/attributes" in blanked
    assert "minimappr/zone/z2/spl_db" in blanked

    retained = transport.retained()
    assert not any("/z2/" in topic for topic in retained)
    await bridge.stop()


@pytest.mark.asyncio
async def test_deleting_a_node_removes_its_connectivity_entity(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(nodes=(node("n1"), node("n2"))))
    )
    await bridge._connect_once()
    transport.clear()

    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(nodes=(node("n1"),))))
    bridge.request_reconcile()
    await bridge._publish_cycle_once()

    assert transport.payload_for("homeassistant/binary_sensor/minimappr/node_online_n2/config") == ""
    await bridge.stop()


@pytest.mark.asyncio
async def test_request_reconcile_is_sync_and_applied_next_cycle(tmp_path: Path, monkeypatch) -> None:
    """Zone/node CRUD routes call this without awaiting."""
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(zone("z1"),))))
    await bridge._connect_once()
    transport.clear()

    result = bridge.request_reconcile()
    assert result is None, "must not be a coroutine — routes call it inline"
    assert transport.published == [], "nothing publishes until the next cycle"

    bridge.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(zones=(zone("z1"), zone("z9"))))
    )
    await bridge._publish_cycle_once()
    assert "homeassistant/binary_sensor/minimappr/zone_occupancy_z9/config" in transport.topics()
    await bridge.stop()


# -- ledger -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_ledger_is_written_and_records_state_topics(tmp_path: Path, monkeypatch) -> None:
    bridge, _ = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(zone("z1"),))))
    await bridge._connect_once()

    ledger_path = bridge.config.discovery_ledger_path
    assert ledger_path.exists()
    raw = json.loads(ledger_path.read_text())
    assert raw["version"] == 1
    entry = raw["entities"][ZONE_CONFIG_TOPIC]
    assert entry["payload_sha256"]
    assert "minimappr/zone/z1/occupancy" in entry["state_topics"]
    await bridge.stop()


@pytest.mark.asyncio
async def test_a_fresh_bridge_removes_orphans_from_a_previous_process(
    tmp_path: Path, monkeypatch
) -> None:
    """The whole reason the ledger is persisted: a zone deleted while we were
    stopped is absent from the desired set and otherwise invisible to us."""
    first, _ = build_test_bridge(tmp_path, monkeypatch)
    first.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(zones=(zone("z1"), zone("gone"))))
    )
    await first._connect_once()
    await first.stop()

    second, transport = build_test_bridge(tmp_path, monkeypatch)
    second.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(zone("z1"),))))
    await second.start()
    await second._connect_once()

    blanked = {m.topic for m in transport.published if m.payload == "" and m.retain}
    assert "homeassistant/binary_sensor/minimappr/zone_occupancy_gone/config" in blanked
    assert "minimappr/zone/gone/occupancy" in blanked
    await second.stop()


@pytest.mark.asyncio
async def test_purge_blanks_everything_and_empties_the_ledger(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(zones=(zone("z1"),), nodes=(node("n1"),)))
    )
    await bridge._connect_once()
    transport.clear()

    removed = await bridge.purge_discovery()
    await bridge._flush_outbound()

    assert removed > 0
    assert bridge.status()["discovery_entity_count"] == 0
    assert json.loads(bridge.config.discovery_ledger_path.read_text())["entities"] == {}
    assert all(message.payload == "" for message in transport.published)
    await bridge.stop()


def test_ledger_round_trips_atomically(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "ledger.json"
    ledger = HassDiscoveryLedger(path)
    ledger.record(LedgerEntry(config_topic="t/config", payload_sha256="abc", state_topics=("t/state",)))
    ledger.save()

    assert not list(path.parent.glob("*.tmp")), "the temp file must not survive a save"

    reloaded = HassDiscoveryLedger(path)
    reloaded.load()
    entry = reloaded.get("t/config")
    assert entry is not None
    assert entry.payload_sha256 == "abc"
    assert entry.state_topics == ("t/state",)


def test_missing_ledger_reads_as_empty(tmp_path: Path) -> None:
    ledger = HassDiscoveryLedger(tmp_path / "absent.json")
    ledger.load()
    assert ledger.entries() == {}


def test_corrupt_ledger_reads_as_empty_rather_than_raising(tmp_path: Path) -> None:
    """Re-publishing everything is noisy but correct; refusing to start is not."""
    path = tmp_path / "ledger.json"
    path.write_text("{not json")
    ledger = HassDiscoveryLedger(path)
    ledger.load()
    assert ledger.entries() == {}


def test_ledger_with_an_unknown_version_reads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"version": 99, "entities": {"a": {}}}))
    ledger = HassDiscoveryLedger(path)
    ledger.load()
    assert ledger.entries() == {}


def test_ledger_forget_and_clear(tmp_path: Path) -> None:
    ledger = HassDiscoveryLedger(tmp_path / "ledger.json")
    ledger.record(LedgerEntry(config_topic="a", payload_sha256="1", state_topics=()))
    ledger.record(LedgerEntry(config_topic="b", payload_sha256="2", state_topics=()))
    ledger.forget("a")
    assert set(ledger.entries()) == {"b"}
    ledger.clear()
    assert ledger.entries() == {}
