"""Merge cross-reporter duplicates of one cluster event (plan item P2).

Every node in a shared cluster can deliver a render of the same event, and the
reporting policy keys its window dedupe per reporting node, so one call heard
by a planar array and a point node would be stored twice. A later localized
render is merged into an earlier one only when all of these hold: same cluster,
same label, same reporting window, a different reporter, and positions within
``localized_handoff_duplicate_merge_max_separation_m``. Two same-label callers
at distinct positions in one cluster and window therefore stay distinct.

Memory stays bounded: only the newest few reporting windows are retained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Renders reach Python seconds after their event, so a window stays mergeable
# for a few windows after newer ones open; older windows can no longer gain a
# duplicate and are pruned.
RETAINED_REPORT_WINDOW_COUNT = 8


@dataclass(frozen=True, slots=True)
class LocalizedClusterEmission:
    reporting_node_id: str
    position_m: tuple[float, float, float]
    solve_id: str


class CrossReporterDuplicateMerger:
    def __init__(self) -> None:
        self._emissions_by_key: dict[tuple[str, str, int], list[LocalizedClusterEmission]] = {}
        self._newest_report_window_start_ns: int | None = None

    def merge_target(
        self,
        *,
        cluster_id: str,
        label: str,
        report_window_start_ns: int,
        reporting_node_id: str,
        position_m,
        max_separation_m: float,
    ) -> LocalizedClusterEmission | None:
        """The earlier emission this render duplicates, or ``None`` to keep it."""
        position = np.asarray(position_m, dtype=np.float64)
        for emission in self._emissions_by_key.get((cluster_id, label, report_window_start_ns), ()):
            if emission.reporting_node_id == reporting_node_id:
                continue
            separation_m = float(np.linalg.norm(position - np.asarray(emission.position_m, dtype=np.float64)))
            if separation_m <= max_separation_m:
                return emission
        return None

    def note_emission(
        self,
        *,
        cluster_id: str,
        label: str,
        report_window_start_ns: int,
        report_window_ns: int,
        emission: LocalizedClusterEmission,
    ) -> None:
        if (
            self._newest_report_window_start_ns is not None
            and report_window_start_ns
            < self._newest_report_window_start_ns - RETAINED_REPORT_WINDOW_COUNT * report_window_ns
        ):
            return
        self._emissions_by_key.setdefault((cluster_id, label, report_window_start_ns), []).append(emission)
        if self._newest_report_window_start_ns is None or report_window_start_ns > self._newest_report_window_start_ns:
            self._newest_report_window_start_ns = report_window_start_ns
            oldest_retained_ns = report_window_start_ns - RETAINED_REPORT_WINDOW_COUNT * report_window_ns
            self._emissions_by_key = {
                key: emissions for key, emissions in self._emissions_by_key.items() if key[2] >= oldest_retained_ns
            }

    def retained_key_count(self) -> int:
        return len(self._emissions_by_key)
