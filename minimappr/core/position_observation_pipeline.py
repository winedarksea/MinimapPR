"""One tracker update per solve (plan P5): the observation sink and its ledger.

Both ingest paths call ``observe_localized_solve`` exactly once per solve,
after localization quality checks and before the classification queue and
reporting. The pipeline turns the solve into a ``PositionObservation`` and
queues it; a drain task applies queued observations to the tracker in arrival
order. So every localized solve moves its track once, whatever happens to it
afterwards (classification suppression, dedupe, classification drop, backlog
drop, cross-reporter merge).

- Idempotent by ``solve_id``: a covering Rust render repeats its event's solve
  id, so the same solve can arrive twice; the second arrival is counted and
  ignored. The ledger remembers which track each solve updated, so the solve's
  later classification (``track_for_classified_solve``) attaches only the
  label. A classification that overtakes its own queued observation applies
  that observation itself, once.
- Bounded FIFO, drop-oldest, counted in FusionMetrics. Never latest-wins: two
  simultaneous solves are two observations even when they share a cluster and
  a label, because they can be two sources.
- The observation step only enqueues; tracker work happens in the drain task,
  so the localization worker and the render handoff never wait on the tracker.
- A solve whose position never reached the tracker (dropped, not allowed to
  start a track, ambiguous between tracks, unknown id) is updated by its
  classification through the label-aware ``update``: still one update.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from minimappr.core.localization_overhaul_config import POSITION_OBSERVATION_POLICY_START_HIDDEN_TENTATIVE
from minimappr.core.position_observation import PositionObservation, position_observation_from_localized_product
from minimappr.core.position_observation_config import PositionObservationConfig
from minimappr.core.tracking_position_observation_methods import TrackManagerPositionObservationMethods

if TYPE_CHECKING:
    from minimappr.core.fusion_stage_carriers import LocalizedCandidate
    from minimappr.models import TrackState

logger = logging.getLogger(__name__)


class SolveLedgerState(StrEnum):
    QUEUED = "queued"
    APPLYING = "applying"
    # The solve's position reached ``track_id``.
    APPLIED = "applied"
    # The position never reached the tracker; the classification will update it.
    DROPPED_QUEUE_FULL = "dropped_queue_full"
    TRACK_NOT_STARTED = "track_not_started"
    DEFERRED_AMBIGUOUS = "deferred_ambiguous"
    APPLY_FAILED = "apply_failed"


_POSITION_REACHED_TRACKER_STATES = frozenset(
    {SolveLedgerState.QUEUED, SolveLedgerState.APPLYING, SolveLedgerState.APPLIED}
)


# eq=False: queue membership and removal must be by identity, never by value.
@dataclass(slots=True, eq=False)
class _SolveLedgerEntry:
    observation: PositionObservation
    state: SolveLedgerState = SolveLedgerState.QUEUED
    track_id: str | None = None
    # This solve's fix started ``track_id`` (a provisional, hidden track).
    started_track: bool = False
    # Serialises the drain and this solve's classifications (event + covering render).
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class PositionObservationPipeline:
    """``PositionObservationSink`` implementation plus the classified-solve resolver."""

    def __init__(
        self,
        *,
        config: PositionObservationConfig,
        tracker_provider: Callable[[], Any],
        metrics_provider: Callable[[], Any],
    ) -> None:
        self._config = config
        # Read per call: tests and runtime rebinding may replace the tracker.
        self._tracker_provider = tracker_provider
        self._metrics_provider = metrics_provider
        self._queue: deque[_SolveLedgerEntry] = deque()
        self._ledger: OrderedDict[str, _SolveLedgerEntry] = OrderedDict()
        self._drain_task: asyncio.Task | None = None

    # -- sink --------------------------------------------------------------------

    async def observe_localized_solve(
        self,
        *,
        localized_product: "LocalizedCandidate",
        reporting_modality: str,
        solve_origin: str,
    ) -> None:
        metrics = self._metrics_provider()
        observation = position_observation_from_localized_product(
            localized_product, reporting_modality=reporting_modality, solve_origin=solve_origin
        )
        metrics.position_observations_received += 1
        if observation is None or not observation.tracker_eligible:
            # Not a trackable fix; its classification keeps the pre-P5 path
            # (partial binding at most), so nothing is recorded.
            metrics.position_observations_not_tracker_eligible += 1
            return
        if not isinstance(self._tracker_provider(), TrackManagerPositionObservationMethods):
            # An injected tracker without the two-step API keeps per-detection updates.
            return
        if observation.solve_id in self._ledger:
            metrics.position_observations_duplicate_solve += 1
            return
        entry = _SolveLedgerEntry(observation=observation)
        self._remember(entry)
        if len(self._queue) >= self._config.queue_max_items:
            dropped = self._queue.popleft()
            dropped.state = SolveLedgerState.DROPPED_QUEUE_FULL
            metrics.position_observations_dropped_queue_full += 1
        self._queue.append(entry)
        self._ensure_drain_scheduled()

    def _remember(self, entry: _SolveLedgerEntry) -> None:
        self._ledger[entry.observation.solve_id] = entry
        # Validation keeps ledger >= queue, so an evicted entry is never still queued.
        while len(self._ledger) > self._config.solve_ledger_max_entries:
            self._ledger.popitem(last=False)

    def _ensure_drain_scheduled(self) -> None:
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.get_running_loop().create_task(
                self._drain_queue(), name="position-observation-drain"
            )

    async def _drain_queue(self) -> None:
        while self._queue:
            entry = self._queue.popleft()
            async with entry.lock:
                await self._apply_queued_entry(entry)

    async def drain(self) -> None:
        """Apply everything queued (shutdown, tests)."""
        while self._drain_task is not None and not self._drain_task.done():
            await asyncio.shield(self._drain_task)
        if self._queue:
            await self._drain_queue()

    def queued_count(self) -> int:
        return len(self._queue)

    async def _apply_queued_entry(self, entry: _SolveLedgerEntry) -> None:
        """Caller holds ``entry.lock``."""
        if entry.state is not SolveLedgerState.QUEUED:
            return
        entry.state = SolveLedgerState.APPLYING
        metrics = self._metrics_provider()
        observation = entry.observation
        try:
            outcome = await self._tracker_provider().apply_position_observation(
                observation, may_start_track=self._may_start_track(observation)
            )
        except Exception:  # noqa: BLE001 - one bad fix must not stop the drain
            entry.state = SolveLedgerState.APPLY_FAILED
            metrics.position_observations_apply_failed += 1
            logger.exception("Position observation failed", extra={"solve_id": observation.solve_id})
            return
        if outcome.deferred_ambiguous:
            entry.state = SolveLedgerState.DEFERRED_AMBIGUOUS
            metrics.position_observations_deferred_ambiguous += 1
            return
        if outcome.track_id is None:
            entry.state = SolveLedgerState.TRACK_NOT_STARTED
            metrics.position_observations_track_not_started += 1
            return
        entry.state = SolveLedgerState.APPLIED
        entry.track_id = outcome.track_id
        entry.started_track = outcome.started_track
        metrics.position_observations_applied += 1

    def _may_start_track(self, observation: PositionObservation) -> bool:
        return (
            observation.event_triggered
            and self._config.unclassified_track_policy == POSITION_OBSERVATION_POLICY_START_HIDDEN_TENTATIVE
        )

    # -- classified-solve resolver -----------------------------------------------

    def solve_position_reached_tracker(self, solve_id: str | None) -> bool:
        """Whether this solve's position is (or is about to be) on a track."""
        entry = self._ledger.get(solve_id) if solve_id is not None else None
        return entry is not None and entry.state in _POSITION_REACHED_TRACKER_STATES

    async def retract_provisional_track_started_by(self, solve_id: str | None) -> None:
        """The solve's classification found nothing trackable (no label, a
        sub-display result no belief can lift, a zone-suppressed label): end
        the hidden track its fix started, if no other fix has joined it since."""
        entry = self._ledger.get(solve_id) if solve_id is not None else None
        tracker = self._tracker_provider()
        if entry is None or not isinstance(tracker, TrackManagerPositionObservationMethods):
            return
        async with entry.lock:
            if entry.state is SolveLedgerState.QUEUED:
                if entry in self._queue:
                    self._queue.remove(entry)
                await self._apply_queued_entry(entry)
            if entry.state is not SolveLedgerState.APPLIED or not entry.started_track:
                return
            if await tracker.retract_untouched_provisional_track(entry.track_id):
                self._metrics_provider().position_observation_provisional_tracks_retracted += 1
            entry.started_track = False

    async def track_for_classified_solve(
        self,
        *,
        solve_id: str | None,
        classified_update_kwargs: dict[str, Any],
    ) -> "TrackState | None":
        """The track a classified solve belongs to, its position applied exactly once.

        ``classified_update_kwargs`` are the ``TrackManager.update`` keywords the
        classified-detection path used before P5. ``None`` when the label's
        category is not tracked (tracking mode ``none``).
        """
        tracker = self._tracker_provider()
        entry = self._ledger.get(solve_id) if solve_id is not None else None
        if entry is None or not isinstance(tracker, TrackManagerPositionObservationMethods):
            return await self._classified_update(tracker, classified_update_kwargs)
        metrics = self._metrics_provider()
        async with entry.lock:
            if entry.state is SolveLedgerState.QUEUED:
                # The classification overtook its own observation: apply it now, once.
                # (The drain may already have popped it and be waiting on the lock.)
                if entry in self._queue:
                    self._queue.remove(entry)
                await self._apply_queued_entry(entry)
            if entry.state is SolveLedgerState.APPLIED:
                track = await tracker.attach_classification_to_observed_track(
                    observed_track_id=entry.track_id,
                    classified_update_kwargs=classified_update_kwargs,
                )
                metrics.position_observation_labels_attached += 1
            else:
                track = await tracker.update_with_classification(**classified_update_kwargs)
                metrics.position_observation_classified_updates += 1
            entry.started_track = False
            if track is not None:
                # Any later classification of this solve (a covering render) attaches only its label.
                entry.state = SolveLedgerState.APPLIED
                entry.track_id = track.id
            return track

    @staticmethod
    async def _classified_update(tracker: Any, classified_update_kwargs: dict[str, Any]) -> "TrackState | None":
        if isinstance(tracker, TrackManagerPositionObservationMethods):
            return await tracker.update_with_classification(**classified_update_kwargs)
        return await tracker.update(**classified_update_kwargs)


def classified_solve_resolver(position_observation_sink: object) -> PositionObservationPipeline | None:
    """The pipeline behind a seam sink, or None for a no-op / test sink."""
    return position_observation_sink if isinstance(position_observation_sink, PositionObservationPipeline) else None
