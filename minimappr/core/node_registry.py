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
        self._latest_observations: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, spec: NodeSpec, last_seen_ns: int) -> NodeRuntime:
        if spec.position_m is None:
            raise ValueError("NodeSpec.position_m must be present for runtime registration")
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

    async def record_observation(self, sensor_id: str, observation_id: str) -> None:
        async with self._lock:
            self._latest_observations[sensor_id] = observation_id

    async def latest_observation_ids(self, sensor_ids: list[str]) -> list[str]:
        async with self._lock:
            values = [self._latest_observations.get(sensor_id) for sensor_id in sensor_ids]
        return [value for value in values if value is not None]

    async def node_id_for_sensor(self, sensor_id: str) -> str | None:
        async with self._lock:
            descriptor = self._sensors.get(sensor_id)
            if descriptor is None:
                return None
            return descriptor.node_id

    async def gain_offset_db_for_sensor(self, sensor_id: str) -> float:
        async with self._lock:
            descriptor = self._sensors.get(sensor_id)
            if descriptor is None:
                return 0.0
            runtime = self._nodes.get(descriptor.node_id)
            if runtime is None:
                return 0.0
            properties = runtime.spec.properties if isinstance(runtime.spec.properties, dict) else {}
            raw = properties.get("gain_offset_db")
            if raw is None and isinstance(properties.get("audio"), dict):
                raw = properties["audio"].get("gain_offset_db")
            try:
                return float(raw)
            except (TypeError, ValueError):
                return 0.0
