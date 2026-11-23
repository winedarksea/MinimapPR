"""Post-window label reconciliation for onset-cluster siblings (closeout B2).

Flag `onset_window_label_reconciliation_enabled` (default off). Rust port:
`minimappr-core/src/onset_clustering/window_label_reconciliation.rs`, applied in
`mesh/call_attribution.rs::attribute_window_clusters`.

The SNR-gradient tier proves a cluster's classifier mostly heard its own bird.
That justifies the cluster's top label, not the rest of the window's score list:
a loud robin's window still scores the quieter tanager beside it, and that
secondary score lets the robin's fix absorb into the tanager's entity (the
same-band two-species merge). Once every sibling cluster of an analysis window
is classified and attributed, a gradient-settled cluster drops each label that
another settled (gradient or LOCO) sibling claims as its top label. Its own top
label is always kept, so two same-species birds in one window stay labelled.

Holding never blocks a worker. A sibling that arrives before the rest is parked
here and its worker moves on; the last sibling to arrive takes the whole window
and assembles it. A window whose missing siblings never arrive (dropped by
backpressure, failed classification) is released after a bounded timeout
through an emit callback. Windows with a single product pass straight through.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from minimappr.core.onset_cluster_attribution import ATTRIBUTION_TIER_LOCO, ATTRIBUTION_TIER_SNR_GRADIENT

ATTRIBUTION_FEATURE_KEY = "onset_attribution"
UNLABELLED_LABEL = "unknown"
RECONCILED_DROPPED_LABELS_KEY = "window_reconciled_dropped_labels"
TOP_LABEL_CLAIMING_TIERS = frozenset({ATTRIBUTION_TIER_SNR_GRADIENT, ATTRIBUTION_TIER_LOCO})
# Memory bounds only: open windows beyond this are released oldest first, and this
# many finished window ids are remembered so a late sibling passes straight through.
MAX_PENDING_RECONCILIATION_WINDOWS = 256
MAX_REMEMBERED_FINISHED_WINDOWS = 1024


def labels_to_drop_by_cluster(clusters: Sequence[tuple[str | None, str | None]]) -> list[list[str]]:
    """For each sibling `(tier, top label)`, the labels to remove from its scores.

    Only gradient-settled clusters lose labels; a sibling without a top label
    (unsettled, abstained) claims nothing. Mirrors the Rust port exactly.
    """
    dropped_by_cluster: list[list[str]] = []
    for index, (tier, own_top) in enumerate(clusters):
        dropped: list[str] = []
        if tier == ATTRIBUTION_TIER_SNR_GRADIENT:
            for other_index, (other_tier, other_top) in enumerate(clusters):
                if (
                    other_index == index
                    or other_top is None
                    or other_tier not in TOP_LABEL_CLAIMING_TIERS
                    or other_top == own_top
                    or other_top in dropped
                ):
                    continue
                dropped.append(other_top)
        dropped_by_cluster.append(dropped)
    return dropped_by_cluster


def _tier_and_top_label(classified: Any) -> tuple[str | None, str | None]:
    classification = classified.classification
    tier = (classification.features.get(ATTRIBUTION_FEATURE_KEY) or {}).get("tier")
    top = classification.label if classification.label != UNLABELLED_LABEL and classification.scores else None
    return tier, top


def _without_labels(classified: Any, dropped: list[str]) -> Any:
    """The classified result minus `dropped` scores; the label itself never changes."""
    original = classified.classification
    annotation = {**original.features.get(ATTRIBUTION_FEATURE_KEY, {}), RECONCILED_DROPPED_LABELS_KEY: list(dropped)}
    reconciled = original.model_copy(
        update={
            "scores": {label: score for label, score in original.scores.items() if label not in dropped},
            "features": {**original.features, ATTRIBUTION_FEATURE_KEY: annotation},
        }
    )
    # The beamformed-only path aliases one result into all three slots; keep them in step.
    return dataclasses.replace(
        classified,
        classification=reconciled,
        omni_classification=reconciled if classified.omni_classification is original else classified.omni_classification,
        beamformed_classification=(
            reconciled if classified.beamformed_classification is original else classified.beamformed_classification
        ),
    )


def reconcile_window_classifications(classified_results: Sequence[Any]) -> tuple[list[Any], int]:
    """Apply the window rule to sibling ClassifiedResults; returns them and the dropped-label count."""
    dropped_by_cluster = labels_to_drop_by_cluster([_tier_and_top_label(c) for c in classified_results])
    reconciled: list[Any] = []
    dropped_count = 0
    for classified, dropped in zip(classified_results, dropped_by_cluster):
        present = [label for label in dropped if label in classified.classification.scores]
        dropped_count += len(present)
        reconciled.append(_without_labels(classified, present) if present else classified)
    return reconciled, dropped_count


@dataclass(slots=True)
class WindowSibling:
    """One classified sibling product waiting for (or released with) its window."""

    product: Any
    classified: Any


@dataclass(slots=True)
class _PendingWindow:
    expected_sibling_count: int
    arrived: list[WindowSibling] = field(default_factory=list)
    resolved_candidate_ids: set[str] = field(default_factory=set)
    timeout_handle: asyncio.TimerHandle | None = None


@dataclass(slots=True)
class WindowReconciliationCounters:
    windows_reconciled: int = 0
    windows_timed_out: int = 0
    labels_dropped: int = 0
    late_siblings: int = 0


class OnsetWindowLabelReconciler:
    """Collects a window's classified siblings and releases them reconciled.

    `submit` returns the siblings ready for assembly now (possibly several, when
    the last one arrives) or an empty list while the window is still open.
    Windows released by timeout, eviction or a withdrawn last sibling go to
    `emit_released_window`, run as a background task.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float,
        emit_released_window: Callable[[list[WindowSibling]], Awaitable[None]],
        max_pending_windows: int = MAX_PENDING_RECONCILIATION_WINDOWS,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._emit_released_window = emit_released_window
        self._max_pending_windows = max_pending_windows
        self._pending: OrderedDict[str, _PendingWindow] = OrderedDict()
        self._finished: OrderedDict[str, None] = OrderedDict()
        self._emit_tasks: set[asyncio.Task] = set()
        self.counters = WindowReconciliationCounters()

    def pending_window_count(self) -> int:
        return len(self._pending)

    def submit(self, *, window_id: str, sibling_count: int, candidate_id: str, sibling: WindowSibling) -> list[WindowSibling]:
        if sibling_count < 2:
            return [sibling]
        if window_id in self._finished:
            # The window was already released (timeout): reconciling now would
            # need the released siblings back, so this one stands as classified.
            self.counters.late_siblings += 1
            return [sibling]
        window = self._pending.get(window_id)
        if window is None:
            window = self._open_window(window_id, sibling_count)
        if candidate_id in window.resolved_candidate_ids:
            return [sibling]
        window.resolved_candidate_ids.add(candidate_id)
        window.arrived.append(sibling)
        if len(window.resolved_candidate_ids) < window.expected_sibling_count:
            return []
        return self._finish(window_id, timed_out=False)

    def withdraw(self, *, window_id: str, sibling_count: int, candidate_id: str) -> None:
        """A sibling will never arrive (skipped, failed): stop waiting for it."""
        if sibling_count < 2 or window_id in self._finished:
            return
        window = self._pending.get(window_id) or self._open_window(window_id, sibling_count)
        if candidate_id in window.resolved_candidate_ids:
            return
        window.resolved_candidate_ids.add(candidate_id)
        if len(window.resolved_candidate_ids) >= window.expected_sibling_count:
            self._release_in_background(self._finish(window_id, timed_out=False))

    async def release_all(self) -> None:
        """Release every open window now and wait for its emission (drain/shutdown)."""
        for window_id in list(self._pending):
            self._release_in_background(self._finish(window_id, timed_out=True))
        while self._emit_tasks:
            await asyncio.gather(*list(self._emit_tasks), return_exceptions=True)

    def _open_window(self, window_id: str, sibling_count: int) -> _PendingWindow:
        while len(self._pending) >= self._max_pending_windows:
            oldest_id = next(iter(self._pending))
            self._release_in_background(self._finish(oldest_id, timed_out=True))
        window = _PendingWindow(expected_sibling_count=sibling_count)
        window.timeout_handle = asyncio.get_running_loop().call_later(
            self._timeout_seconds, self._on_timeout, window_id
        )
        self._pending[window_id] = window
        return window

    def _on_timeout(self, window_id: str) -> None:
        if window_id in self._pending:
            self._release_in_background(self._finish(window_id, timed_out=True))

    def _finish(self, window_id: str, *, timed_out: bool) -> list[WindowSibling]:
        window = self._pending.pop(window_id)
        if window.timeout_handle is not None:
            window.timeout_handle.cancel()
        self._finished[window_id] = None
        while len(self._finished) > MAX_REMEMBERED_FINISHED_WINDOWS:
            self._finished.popitem(last=False)
        if timed_out:
            self.counters.windows_timed_out += 1
        if not window.arrived:
            return []
        reconciled, dropped_count = reconcile_window_classifications([s.classified for s in window.arrived])
        self.counters.windows_reconciled += 1
        self.counters.labels_dropped += dropped_count
        return [WindowSibling(product=s.product, classified=c) for s, c in zip(window.arrived, reconciled)]

    def _release_in_background(self, siblings: list[WindowSibling]) -> None:
        if not siblings:
            return
        task = asyncio.get_running_loop().create_task(self._emit_released_window(siblings))
        self._emit_tasks.add(task)
        task.add_done_callback(self._emit_tasks.discard)
