from __future__ import annotations

import pytest

from minimappr.core.node_registry import NodeRegistry
from minimappr.models import NodeSpec, NodeType


@pytest.mark.asyncio
async def test_node_registry_ignores_stale_runtime_updates() -> None:
    registry = NodeRegistry()
    newer_spec = NodeSpec(
        id="node-registry-1",
        node_type=NodeType.POINT,
        position_m=(9.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
    )
    stale_spec = NodeSpec(
        id="node-registry-1",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
    )

    runtime_new = await registry.upsert(newer_spec, last_seen_ns=200)
    runtime_stale = await registry.upsert(stale_spec, last_seen_ns=100)

    assert runtime_new.last_seen_ns == 200
    assert runtime_stale.last_seen_ns == 200
    positions = await registry.sensor_positions()
    assert float(positions["node-registry-1:ch0"][0]) == pytest.approx(9.0, abs=1e-9)


@pytest.mark.asyncio
async def test_node_registry_applies_pinned_position_overrides() -> None:
    registry = NodeRegistry()
    await registry.load_overrides(
        {
            "node-registry-override": {
                "position_m": [10.0, 20.0, 30.0],
                "capabilities": ["audio"],
            }
        }
    )
    spec = NodeSpec(
        id="node-registry-override",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(1.0, 0.0, 0.0)],
        capabilities=["audio"],
    )

    runtime = await registry.upsert(spec, last_seen_ns=1)
    positions = await registry.sensor_positions()

    assert runtime.spec.position_m == (10.0, 20.0, 30.0)
    assert positions["node-registry-override:ch0"].tolist() == [11.0, 20.0, 30.0]


def test_node_spec_strips_unknown_capabilities_into_metadata() -> None:
    spec = NodeSpec(
        id="node-capabilities",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        capabilities=["gps_optional", "ble_rssi", "bogus"],
    )

    assert [capability.value for capability in spec.capabilities] == ["gps_optional", "ble_rssi"]
    assert spec.metadata["unrecognized_capabilities"] == ["bogus"]
