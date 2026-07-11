"""In-memory node and sensor metadata registry."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np

from minimappr.models import NodeOverrides, NodeSpec, SyncGrade


@dataclass(slots=True)
class SensorDescriptor:
    sensor_id: str
    node_id: str
    channel_index: int
    position_m: np.ndarray
    effective_sync_grade: SyncGrade = field(default=SyncGrade.FREE)
    cluster_id: str | None = field(default=None)


@dataclass(slots=True)
class NodeRuntime:
    spec: NodeSpec
    sensor_ids: list[str]
    last_seen_ns: int
    last_sequence_seen: int | None = None
    boot_session: str | None = None


class NodeRegistry:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeRuntime] = {}
        self._sensors: dict[str, SensorDescriptor] = {}
        self._latest_observations: dict[str, str] = {}
        # Phase 5: per-node position 1σ (m), fed from the ingest position-Kalman P.
        # Absent → 0.0 (surveyed / trusted-GPS default). Used to down-weight
        # cross-node TDOA pairs anchored on poorly-localized nodes.
        self._node_position_std_m: dict[str, float] = {}
        self._overrides: dict[str, NodeOverrides] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, spec: NodeSpec, last_seen_ns: int) -> NodeRuntime:
        override = self._overrides.get(spec.id)
        if override is not None:
            update: dict[str, object] = {}
            if override.position_m is not None:
                update["position_m"] = override.position_m
            if override.position_geo is not None:
                update["position_geo"] = override.position_geo
            if override.orientation is not None:
                update["orientation"] = override.orientation
            if override.mobility is not None:
                update["mobility"] = override.mobility
            if override.capabilities is not None:
                update["capabilities"] = override.capabilities
            if update:
                spec = spec.model_copy(update=update)
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

        async with self._lock:
            existing = self._nodes.get(spec.id)
            if existing is not None and last_seen_ns < existing.last_seen_ns:
                return existing
            for sensor_id, descriptor in sensor_descriptors.items():
                existing_descriptor = self._sensors.get(sensor_id)
                if existing_descriptor is not None:
                    descriptor.effective_sync_grade = existing_descriptor.effective_sync_grade
                    descriptor.cluster_id = existing_descriptor.cluster_id
            runtime = NodeRuntime(
                spec=spec,
                sensor_ids=sensor_ids,
                last_seen_ns=last_seen_ns,
                last_sequence_seen=existing.last_sequence_seen if existing is not None else None,
                boot_session=existing.boot_session if existing is not None else None,
            )
            self._nodes[spec.id] = runtime
            for sensor_id, descriptor in sensor_descriptors.items():
                self._sensors[sensor_id] = descriptor
        return runtime

    async def load_overrides(self, overrides_by_node_id: dict[str, dict]) -> None:
        parsed: dict[str, NodeOverrides] = {}
        for node_id, payload in overrides_by_node_id.items():
            if not isinstance(payload, dict) or not payload:
                continue
            parsed[node_id] = NodeOverrides.model_validate(payload)
        async with self._lock:
            self._overrides = parsed

    async def set_overrides(self, node_id: str, overrides: NodeOverrides | dict) -> None:
        parsed = overrides if isinstance(overrides, NodeOverrides) else NodeOverrides.model_validate(overrides)
        async with self._lock:
            if parsed.model_dump(exclude_none=True):
                self._overrides[node_id] = parsed
            else:
                self._overrides.pop(node_id, None)

    def position_policy_for(self, node_id: str, reported_mobility: str) -> tuple[str, str]:
        """Return operator-effective mobility and position filter without I/O.

        This is deliberately available before ingest normalization: applying the
        override only in ``upsert`` is too late to affect GPS processing.
        """
        override = self._overrides.get(node_id)
        mobility = override.mobility if override and override.mobility else reported_mobility
        mobility = "mobile" if mobility == "mobile" else "stationary"
        explicit_filter = override.position_filter if override else None
        return mobility, explicit_filter or ("kalman" if mobility == "mobile" else "kde")

    async def delete_node(self, node_id: str) -> None:
        """Drop a node and its sensors from the in-memory registry.

        Used after a node is deleted from storage so a stale entry cannot be
        re-served. A live node simply re-registers on its next ingest frame.
        """
        async with self._lock:
            self._nodes.pop(node_id, None)
            stale_sensor_ids = [
                sensor_id
                for sensor_id, descriptor in self._sensors.items()
                if descriptor.node_id == node_id
            ]
            for sensor_id in stale_sensor_ids:
                self._sensors.pop(sensor_id, None)
                self._latest_observations.pop(sensor_id, None)

    async def list_nodes(self) -> list[NodeRuntime]:
        async with self._lock:
            return list(self._nodes.values())

    async def sensor_positions(self) -> dict[str, np.ndarray]:
        async with self._lock:
            return {sensor_id: descriptor.position_m.copy() for sensor_id, descriptor in self._sensors.items()}

    async def set_node_position_std_m(self, node_id: str, position_std_m: float) -> None:
        """Record a node's position 1σ (m), e.g. from the ingest position-Kalman P."""
        async with self._lock:
            self._node_position_std_m[node_id] = max(float(position_std_m), 0.0)

    async def sensor_position_std_m(self) -> dict[str, float]:
        """Per-sensor position 1σ (m), inherited from the sensor's node.

        Sensors on nodes without a recorded std default to 0.0 (surveyed / trusted
        GPS). Consumed by the cross-node TDOA geometry weighting (Phase 5).
        """
        async with self._lock:
            return {
                sensor_id: self._node_position_std_m.get(descriptor.node_id, 0.0)
                for sensor_id, descriptor in self._sensors.items()
            }

    async def sensor_node_ids(
        self,
        sensor_ids: list[str] | None = None,
    ) -> dict[str, str]:
        async with self._lock:
            selected_ids = sensor_ids if sensor_ids is not None else list(self._sensors)
            return {
                sensor_id: self._sensors[sensor_id].node_id
                for sensor_id in selected_ids
                if sensor_id in self._sensors
            }

    async def sensors_for_node(self, node_id: str) -> list[SensorDescriptor]:
        async with self._lock:
            return [descriptor for descriptor in self._sensors.values() if descriptor.node_id == node_id]

    async def record_observation(self, sensor_id: str, observation_id: str) -> None:
        async with self._lock:
            self._latest_observations[sensor_id] = observation_id

    async def record_frame_sequence(
        self,
        *,
        node_id: str,
        boot_session: str,
        frame_sequence: int | None,
    ) -> int:
        if frame_sequence is None:
            return 0
        normalized_boot_session = boot_session or None
        async with self._lock:
            runtime = self._nodes.get(node_id)
            if runtime is None:
                return 0
            if runtime.boot_session != normalized_boot_session:
                runtime.boot_session = normalized_boot_session
                runtime.last_sequence_seen = frame_sequence
                return 0
            last_sequence_seen = runtime.last_sequence_seen
            gap_size = 0
            if last_sequence_seen is not None and frame_sequence > last_sequence_seen + 1:
                gap_size = frame_sequence - (last_sequence_seen + 1)
            if last_sequence_seen is None or frame_sequence > last_sequence_seen:
                runtime.last_sequence_seen = frame_sequence
            return gap_size

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

    async def update_sensor_sync_grade(self, sensor_id: str, grade: SyncGrade) -> None:
        async with self._lock:
            descriptor = self._sensors.get(sensor_id)
            if descriptor is not None:
                descriptor.effective_sync_grade = grade

    async def update_node_sensor_sync_grade(
        self,
        node_id: str,
        grade: SyncGrade,
    ) -> None:
        async with self._lock:
            for descriptor in self._sensors.values():
                if descriptor.node_id == node_id:
                    descriptor.effective_sync_grade = grade

    async def update_node_cluster(self, node_id: str, cluster_id: str | None) -> None:
        async with self._lock:
            for descriptor in self._sensors.values():
                if descriptor.node_id == node_id:
                    descriptor.cluster_id = cluster_id

    async def sensor_sync_grades(self) -> dict[str, SyncGrade]:
        async with self._lock:
            return {sid: d.effective_sync_grade for sid, d in self._sensors.items()}

    async def gain_offset_db_for_sensor(self, sensor_id: str) -> float:
        async with self._lock:
            return self._gain_offset_db_for_sensor_unlocked(sensor_id)

    async def sensor_gain_offsets_db(
        self,
        sensor_ids: list[str],
    ) -> dict[str, float]:
        async with self._lock:
            return {
                sensor_id: self._gain_offset_db_for_sensor_unlocked(sensor_id)
                for sensor_id in sensor_ids
                if sensor_id in self._sensors
            }

    def _gain_offset_db_for_sensor_unlocked(self, sensor_id: str) -> float:
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
