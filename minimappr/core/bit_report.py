"""Built-In Test (BIT) report evaluator and in-memory registry.

Tracks the latest PBIT, CBIT, and IBIT report per node and derives
composite health status by combining BIT outcomes with heartbeat
staleness.  Designed for constructor injection via ``app.state``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from minimappr.models import (
    BITReport,
    BITReportIn,
    BITStatus,
    BITTestResult,
    BITType,
    NodeHealthStatus,
)


@dataclass(slots=True)
class _NodeBITState:
    """Latest BIT reports for a single node, keyed by report type."""

    latest_by_type: dict[BITType, BITReport] = field(default_factory=dict)


class BITReportEvaluator:
    """Accepts, evaluates, and stores BIT reports; derives node health.

    Parameters
    ----------
    bit_fail_override_heartbeat:
        When ``True`` (default), a BIT failure overrides the heartbeat-based
        health status.  When ``False``, BIT failure is only reflected when the
        node is otherwise online/degraded (i.e. an offline node stays offline).
    stale_cbit_seconds:
        If the most recent CBIT is older than this many seconds, the node is
        treated as *degraded* regardless of heartbeat.  ``0`` disables.
    """

    def __init__(
        self,
        *,
        bit_fail_override_heartbeat: bool = True,
        stale_cbit_seconds: float = 0.0,
    ) -> None:
        self._states: dict[str, _NodeBITState] = {}
        self._lock = asyncio.Lock()
        self._bit_fail_override_heartbeat = bit_fail_override_heartbeat
        self._stale_cbit_seconds = stale_cbit_seconds

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def submit_report(
        self,
        node_id: str,
        report_in: BITReportIn,
        *,
        received_ns: int | None = None,
    ) -> BITReport:
        """Validate, persist in memory, and return the full ``BITReport``."""
        now_ns = received_ns or time.time_ns()
        timestamp_ns = report_in.timestamp_ns or now_ns

        overall_status = _derive_overall_status(report_in.results)
        failure_codes = [
            result.failure_code
            for result in report_in.results
            if result.failure_code is not None
        ]

        report = BITReport(
            id=uuid.uuid4().hex,
            node_id=node_id,
            report_type=report_in.report_type,
            overall_status=overall_status,
            timestamp_ns=timestamp_ns,
            received_ns=now_ns,
            results=report_in.results,
            failure_codes=failure_codes,
            firmware_version=report_in.firmware_version,
            uptime_seconds=report_in.uptime_seconds,
            metadata=report_in.metadata,
        )

        async with self._lock:
            state = self._states.setdefault(node_id, _NodeBITState())
            existing = state.latest_by_type.get(report.report_type)
            # Only keep the newest report per type
            if existing is None or report.timestamp_ns >= existing.timestamp_ns:
                state.latest_by_type[report.report_type] = report

        return report

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def latest_reports_for_node(self, node_id: str) -> list[BITReport]:
        """Return the latest report per BIT type for *node_id*."""
        async with self._lock:
            state = self._states.get(node_id)
            if state is None:
                return []
            return list(state.latest_by_type.values())

    async def latest_report_of_type(
        self,
        node_id: str,
        report_type: BITType,
    ) -> BITReport | None:
        async with self._lock:
            state = self._states.get(node_id)
            if state is None:
                return None
            return state.latest_by_type.get(report_type)

    async def all_nodes_with_bit_failures(self) -> dict[str, list[str]]:
        """Return ``{node_id: [failure_codes…]}`` for nodes with active failures."""
        result: dict[str, list[str]] = {}
        async with self._lock:
            for node_id, state in self._states.items():
                codes: list[str] = []
                for report in state.latest_by_type.values():
                    if report.overall_status in (BITStatus.FAIL, BITStatus.DEGRADED):
                        codes.extend(report.failure_codes)
                if codes:
                    result[node_id] = codes
        return result

    # ------------------------------------------------------------------
    # Health derivation
    # ------------------------------------------------------------------

    async def derive_health_status(
        self,
        node_id: str,
        heartbeat_health: str,
        now_ns: int | None = None,
    ) -> str:
        """Combine heartbeat-based health with BIT status.

        Parameters
        ----------
        node_id:
            The node to evaluate.
        heartbeat_health:
            The staleness-based health string (``"online"``, ``"degraded"``,
            ``"offline"``).
        now_ns:
            Current time (nanoseconds).  Defaults to ``time.time_ns()``.

        Returns
        -------
        str
            One of ``NodeHealthStatus`` values.
        """
        now_ns = now_ns or time.time_ns()

        async with self._lock:
            state = self._states.get(node_id)

        if state is None:
            # No BIT data — fall through to heartbeat status
            return heartbeat_health

        # Evaluate BIT outcome across all stored report types
        worst_bit_status = BITStatus.PASS
        for report in state.latest_by_type.values():
            if report.overall_status == BITStatus.FAIL:
                worst_bit_status = BITStatus.FAIL
                break
            if report.overall_status == BITStatus.DEGRADED:
                worst_bit_status = BITStatus.DEGRADED

        # Check for stale CBIT
        stale_cbit = False
        if self._stale_cbit_seconds > 0.0:
            cbit = state.latest_by_type.get(BITType.CBIT)
            if cbit is not None:
                age_s = max(0.0, (now_ns - cbit.timestamp_ns) / 1_000_000_000.0)
                if age_s > self._stale_cbit_seconds:
                    stale_cbit = True

        # Decision matrix
        if worst_bit_status == BITStatus.FAIL:
            if self._bit_fail_override_heartbeat:
                return NodeHealthStatus.BIT_FAIL.value
            # If node is already offline, keep offline; else report bit_fail
            if heartbeat_health == NodeHealthStatus.OFFLINE.value:
                return NodeHealthStatus.OFFLINE.value
            return NodeHealthStatus.BIT_FAIL.value

        if worst_bit_status == BITStatus.DEGRADED or stale_cbit:
            # BIT degradation promotes heartbeat-online to degraded
            if heartbeat_health == NodeHealthStatus.ONLINE.value:
                return NodeHealthStatus.DEGRADED.value
            return heartbeat_health

        # BIT is passing — defer to heartbeat
        return heartbeat_health


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _derive_overall_status(results: list[BITTestResult]) -> BITStatus:
    """Worst-case roll-up: any FAIL → FAIL, any DEGRADED → DEGRADED, else PASS."""
    has_degraded = False
    for result in results:
        if result.status == BITStatus.FAIL:
            return BITStatus.FAIL
        if result.status == BITStatus.DEGRADED:
            has_degraded = True
    return BITStatus.DEGRADED if has_degraded else BITStatus.PASS
