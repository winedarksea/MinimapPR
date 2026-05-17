"""Phase E: ClusterRegistry unit tests.

Verifies CRUD, node→cluster resolution, sensor lookup, and effective-grade
computation (runtime can only degrade, never upgrade declared grade).
"""
from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.cluster_registry import ClusterRegistry, _min_grade
from minimappr.core.node_registry import NodeRegistry
from minimappr.models import (
    ClusterSpec,
    IamfRenderMode,
    NodeSpec,
    NodeType,
    SyncGrade,
)


def _node(node_id: str, x: float, y: float) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type=NodeType.POINT,
        position_m=(x, y, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
    )


@pytest.mark.asyncio
async def test_upsert_get_list_delete_roundtrip() -> None:
    reg = ClusterRegistry()
    spec = ClusterSpec(
        id="c1",
        member_node_ids=["n0", "n1"],
        declared_sync_grade=SyncGrade.GPS_PPS,
    )
    await reg.upsert(spec)

    assert await reg.get("c1") == spec
    assert await reg.list_all() == [spec]

    # Replace
    replaced = spec.model_copy(update={"iamf_render_mode": IamfRenderMode.FOA_BED})
    await reg.upsert(replaced)
    fetched = await reg.get("c1")
    assert fetched is not None
    assert fetched.iamf_render_mode == IamfRenderMode.FOA_BED

    assert await reg.delete("c1") is True
    assert await reg.get("c1") is None
    assert await reg.delete("c1") is False


@pytest.mark.asyncio
async def test_cluster_for_node_resolution() -> None:
    reg = ClusterRegistry()
    await reg.upsert(ClusterSpec(
        id="alpha", member_node_ids=["n0", "n1"], declared_sync_grade=SyncGrade.GPS_PPS,
    ))
    await reg.upsert(ClusterSpec(
        id="beta", member_node_ids=["n2"], declared_sync_grade=SyncGrade.NTP,
    ))

    found = await reg.cluster_for_node("n1")
    assert found is not None and found.id == "alpha"

    found = await reg.cluster_for_node("n2")
    assert found is not None and found.id == "beta"

    assert await reg.cluster_for_node("nonexistent") is None


@pytest.mark.asyncio
async def test_sensors_and_positions_in_cluster() -> None:
    nodes = NodeRegistry()
    await nodes.upsert(_node("n0", 0.0, 0.0), last_seen_ns=1)
    await nodes.upsert(_node("n1", 2.0, 0.0), last_seen_ns=1)
    await nodes.upsert(_node("n_other", 5.0, 5.0), last_seen_ns=1)

    reg = ClusterRegistry()
    await reg.upsert(ClusterSpec(
        id="c1", member_node_ids=["n0", "n1"], declared_sync_grade=SyncGrade.GPS_PPS,
    ))

    sensors = await reg.sensors_in_cluster("c1", nodes)
    assert set(sensors.keys()) == {"n0:ch0", "n1:ch0"}

    positions = await reg.cluster_sensor_positions("c1", nodes)
    np.testing.assert_allclose(positions["n0:ch0"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(positions["n1:ch0"], [2.0, 0.0, 0.0])


@pytest.mark.asyncio
async def test_effective_grade_min_of_declared_and_runtime() -> None:
    """Cluster declared = GPS_PPS, but runtime degraded one sensor to NTP.
    Effective grade for that sensor must be NTP (lower); the rest stay GPS_PPS.
    """
    nodes = NodeRegistry()
    await nodes.upsert(_node("n0", 0.0, 0.0), last_seen_ns=1)
    await nodes.upsert(_node("n1", 1.0, 0.0), last_seen_ns=1)
    await nodes.update_sensor_sync_grade("n0:ch0", SyncGrade.GPS_PPS)
    await nodes.update_sensor_sync_grade("n1:ch0", SyncGrade.NTP)

    reg = ClusterRegistry()
    await reg.upsert(ClusterSpec(
        id="c1", member_node_ids=["n0", "n1"], declared_sync_grade=SyncGrade.GPS_PPS,
    ))

    grades = await reg.cluster_sensor_grades("c1", nodes)
    assert grades["n0:ch0"] == SyncGrade.GPS_PPS
    assert grades["n1:ch0"] == SyncGrade.NTP


@pytest.mark.asyncio
async def test_declared_caps_effective_grade() -> None:
    """Declared NTP cluster: even a GPS_PPS sensor cannot upgrade past NTP."""
    nodes = NodeRegistry()
    await nodes.upsert(_node("n0", 0.0, 0.0), last_seen_ns=1)
    await nodes.update_sensor_sync_grade("n0:ch0", SyncGrade.GPS_PPS)

    reg = ClusterRegistry()
    await reg.upsert(ClusterSpec(
        id="c1", member_node_ids=["n0"], declared_sync_grade=SyncGrade.NTP,
    ))

    grades = await reg.cluster_sensor_grades("c1", nodes)
    assert grades["n0:ch0"] == SyncGrade.NTP


@pytest.mark.asyncio
async def test_cluster_sensor_weights_uses_sync_grade_weights() -> None:
    nodes = NodeRegistry()
    await nodes.upsert(_node("n0", 0.0, 0.0), last_seen_ns=1)
    await nodes.upsert(_node("n1", 1.0, 0.0), last_seen_ns=1)
    await nodes.update_sensor_sync_grade("n0:ch0", SyncGrade.GPS_PPS)
    await nodes.update_sensor_sync_grade("n1:ch0", SyncGrade.NTP)

    reg = ClusterRegistry()
    await reg.upsert(ClusterSpec(
        id="c1", member_node_ids=["n0", "n1"], declared_sync_grade=SyncGrade.GPS_PPS,
    ))

    weights = await reg.cluster_sensor_weights("c1", nodes)
    assert weights["n0:ch0"] == pytest.approx(1.0)
    assert weights["n1:ch0"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_update_node_memberships_propagates_to_registry() -> None:
    nodes = NodeRegistry()
    await nodes.upsert(_node("n0", 0.0, 0.0), last_seen_ns=1)
    await nodes.upsert(_node("n1", 1.0, 0.0), last_seen_ns=1)
    await nodes.upsert(_node("n2", 2.0, 0.0), last_seen_ns=1)

    reg = ClusterRegistry()
    await reg.upsert(ClusterSpec(
        id="c1", member_node_ids=["n0", "n1"], declared_sync_grade=SyncGrade.GPS_PPS,
    ))
    await reg.update_node_memberships(nodes)

    sensors_n0 = await nodes.sensors_for_node("n0")
    sensors_n2 = await nodes.sensors_for_node("n2")
    assert sensors_n0[0].cluster_id == "c1"
    assert sensors_n2[0].cluster_id is None

    # Delete cluster → memberships must clear.
    await reg.delete("c1")
    await reg.update_node_memberships(nodes)
    sensors_n0 = await nodes.sensors_for_node("n0")
    assert sensors_n0[0].cluster_id is None


def test_min_grade_helper_orders_grades_correctly() -> None:
    assert _min_grade(SyncGrade.GPS_PPS, SyncGrade.GPS_PPS) == SyncGrade.GPS_PPS
    assert _min_grade(SyncGrade.GPS_PPS, SyncGrade.NTP) == SyncGrade.NTP
    assert _min_grade(SyncGrade.NTP, SyncGrade.GPS_PPS) == SyncGrade.NTP
    assert _min_grade(SyncGrade.PTP, SyncGrade.FREE) == SyncGrade.FREE


def test_sync_grade_weight_table() -> None:
    assert SyncGrade.GPS_PPS.weight() == 1.0
    assert SyncGrade.PTP.weight() == 1.0
    assert SyncGrade.NTP.weight() == 0.25
    assert SyncGrade.FREE.weight() == 0.05
