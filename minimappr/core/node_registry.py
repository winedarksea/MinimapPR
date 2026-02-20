"""In-memory node and sensor metadata registry."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from minimappr.models import NodeSpec


@dataclass(slots=True)
class SensorDescriptor:
    sensor_id: str
    node_id: str
    channel_index: int
    position_m: np.ndarray


@dataclass(slots=True)
class NodeRuntime:
    spec: NodeSpec
    sensor_ids: list[str]
    last_seen_ns: int


class NodeRegistry:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeRuntime] = {}
        self._sensors: dict[str, SensorDescriptor] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, spec: NodeSpec, last_seen_ns: int) -> NodeRuntime:
        base = np.asarray(spec.position_m, dtype=np.float64)
        sensor_ids: list[str] = []
        sensor_descriptors: dict[str, SensorDescriptor] = {}

        for index, offset in enumerate(spec.sensor_offsets_m):
            sensor_id = f"{spec.id}:ch{index}"
            sensor_ids.append(sensor_id)
            sensor_descriptors[sensor_id] = SensorDescriptor(
                sensor_id=sensor_id,
                node_id=spec.id,
                channel_index=index,
                position_m=base + np.asarray(offset, dtype=np.float64),
            )

        runtime = NodeRuntime(spec=spec, sensor_ids=sensor_ids, last_seen_ns=last_seen_ns)
        async with self._lock:
            self._nodes[spec.id] = runtime
            for sensor_id, descriptor in sensor_descriptors.items():
                self._sensors[sensor_id] = descriptor
        return runtime

    async def list_nodes(self) -> list[NodeRuntime]:
        async with self._lock:
            return list(self._nodes.values())

    async def sensor_positions(self) -> dict[str, np.ndarray]:
        async with self._lock:
            return {sensor_id: descriptor.position_m.copy() for sensor_id, descriptor in self._sensors.items()}

    async def sensors_for_node(self, node_id: str) -> list[SensorDescriptor]:
        async with self._lock:
            return [descriptor for descriptor in self._sensors.values() if descriptor.node_id == node_id]
