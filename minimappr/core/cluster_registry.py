"""In-memory registry for node cluster specifications."""

from __future__ import annotations

import asyncio

import numpy as np

from minimappr.models import ClusterSpec, SyncGrade
from minimappr.core.node_registry import NodeRegistry, SensorDescriptor


class ClusterRegistry:
    """Thread-safe registry mapping cluster IDs to ClusterSpec objects.

    Clusters are authoritative server-side: nodes never self-declare membership.
    Operator creates clusters via the REST API; this registry holds the live state.
    """

    def __init__(self) -> None:
        self._clusters: dict[str, ClusterSpec] = {}
        self._lock = asyncio.Lock()

    # ── CRUD ────────────────────────────────────────────────────────────────

    async def upsert(self, spec: ClusterSpec) -> None:
        async with self._lock:
            self._clusters[spec.id] = spec

    async def get(self, cluster_id: str) -> ClusterSpec | None:
        async with self._lock:
            return self._clusters.get(cluster_id)

    async def list_all(self) -> list[ClusterSpec]:
        async with self._lock:
            return list(self._clusters.values())

    async def delete(self, cluster_id: str) -> bool:
        async with self._lock:
            return self._clusters.pop(cluster_id, None) is not None

    # ── Node / sensor resolution ─────────────────────────────────────────────

    async def cluster_for_node(self, node_id: str) -> ClusterSpec | None:
        """Return the cluster that contains *node_id*, or None."""
        async with self._lock:
            for spec in self._clusters.values():
                if node_id in spec.member_node_ids:
                    return spec
        return None

    async def sensors_in_cluster(
        self,
        cluster_id: str,
        node_registry: NodeRegistry,
    ) -> dict[str, SensorDescriptor]:
        """Return all SensorDescriptors for every node in the cluster."""
        spec = await self.get(cluster_id)
        if spec is None:
            return {}
        result: dict[str, SensorDescriptor] = {}
        for node_id in spec.member_node_ids:
            descriptors = await node_registry.sensors_for_node(node_id)
            for d in descriptors:
                result[d.sensor_id] = d
        return result

    async def cluster_sensor_positions(
        self,
        cluster_id: str,
        node_registry: NodeRegistry,
    ) -> dict[str, np.ndarray]:
        """Return {sensor_id: position_m} for all sensors in the cluster."""
        sensors = await self.sensors_in_cluster(cluster_id, node_registry)
        return {sid: d.position_m.copy() for sid, d in sensors.items()}

    async def cluster_sensor_grades(
        self,
        cluster_id: str,
        node_registry: NodeRegistry,
    ) -> dict[str, SyncGrade]:
        """Return {sensor_id: effective_sync_grade} for the cluster.

        Grade is the minimum of the cluster's declared_sync_grade and the
        per-sensor runtime grade recorded in NodeRegistry.
        """
        spec = await self.get(cluster_id)
        if spec is None:
            return {}
        sensors = await self.sensors_in_cluster(cluster_id, node_registry)
        result: dict[str, SyncGrade] = {}
        declared = spec.declared_sync_grade
        for sid, d in sensors.items():
            result[sid] = _min_grade(declared, d.effective_sync_grade)
        return result

    async def cluster_sensor_weights(
        self,
        cluster_id: str,
        node_registry: NodeRegistry,
    ) -> dict[str, float]:
        """Return {sensor_id: localization_weight} using SYNC_GRADE_WEIGHTS."""
        grades = await self.cluster_sensor_grades(cluster_id, node_registry)
        return {sid: grade.weight() for sid, grade in grades.items()}

    async def update_node_memberships(self, node_registry: NodeRegistry) -> None:
        """Propagate cluster_id into NodeRegistry's SensorDescriptors.

        Called after upsert/delete so sensor descriptors stay consistent.
        Nodes no longer in any cluster have their cluster_id cleared to None.
        """
        # Build node → cluster_id mapping from current clusters.
        node_to_cluster: dict[str, str | None] = {}
        async with self._lock:
            for spec in self._clusters.values():
                for node_id in spec.member_node_ids:
                    node_to_cluster[node_id] = spec.id

        # Also fetch all registered nodes so we can clear stale cluster_ids.
        all_runtimes = await node_registry.list_nodes()
        for runtime in all_runtimes:
            if runtime.spec.id not in node_to_cluster:
                node_to_cluster[runtime.spec.id] = None

        # Flush memberships to NodeRegistry.
        for node_id, cluster_id in node_to_cluster.items():
            await node_registry.update_node_cluster(node_id, cluster_id)


# ── Helpers ──────────────────────────────────────────────────────────────────

_GRADE_ORDER: list[SyncGrade] = [
    SyncGrade.GPS_PPS,
    SyncGrade.PTP,
    SyncGrade.NTP,
    SyncGrade.FREE,
]


def _min_grade(a: SyncGrade, b: SyncGrade) -> SyncGrade:
    """Return the lower-quality of two SyncGrades (runtime can only degrade)."""
    ia = _GRADE_ORDER.index(a)
    ib = _GRADE_ORDER.index(b)
    return _GRADE_ORDER[max(ia, ib)]
