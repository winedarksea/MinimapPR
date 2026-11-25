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
  that observation itself, once. An evicted ledger entry leaves its solve id
  (and track id) in a longer-lived memory, so a late render or classification
  still never applies the position twice.
- Bounded FIFO, drop-oldest, counted in FusionMetrics. Never latest-wins: two
  simultaneous solves are two observations even when they share a cluster and
  a label, because they can be two sources.
- The observation step only enqueues; tracker work happens in the drain task,
  so the localization worker and the render handoff never wait on the tracker.
- A solve whose position never reached the tracker (dropped, not allowed to
  start a track, ambiguous between tracks, unknown id) is updated by its
  classification through the label-aware ``update``: still one update.
- Hold rule: a fix whose one gating track is a labelled stationary_hold track
  is held for its classification (not an event-only Rust solve, which has
  none). An informative label applies it to that track unless the label gates
  reject it, then through the label-aware ``update``; an abstention, a
  classification that finds nothing trackable, or a label whose category is
  not tracked, drops it; a classification
  that never comes
  (``release_held_position_without_classification``, ledger eviction, or the
  ``held_fix_max_wait_seconds`` deadline) applies it geometry-only.
- An abstention never labels or reveals a track: it retracts a hidden track
  only its own fix started, otherwise refreshes recency only.
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
from minimappr.core.localized_render_handoff_hooks import SOLVE_ORIGIN_RUST_EVENT_LOCALIZATION
from minimappr.core.position_observation import PositionObservation, position_observation_from_localized_product
from minimappr.core.position_observation_config import PositionObservationConfig
from minimappr.core.tracking_position_observation_methods import (
    TrackManagerPositionObservationMethods,
    classification_is_abstention,
)

if TYPE_CHECKING:
    from minimappr.core.fusion_stage_carriers import LocalizedCandidate
    from minimappr.models import TrackState

logger = logging.getLogger(__name__)


class SolveLedgerState(StrEnum):
    QUEUED = "queued"
    APPLYING = "applying"
    # The solve's position reached ``track_id``.
    APPLIED = "applied"
    # Waits for the classification; ``track_id`` is the one gating (stationary_hold) track.
    HELD_FOR_LABEL = "held_for_label"
    # The classification abstained, found nothing, or labelled an untracked category:
    # the held fix never reaches the tracker.
    HELD_DROPPED = "held_dropped"
    # The position never reached the tracker; the classification will update it.
    DROPPED_QUEUE_FULL = "dropped_queue_full"
    TRACK_NOT_STARTED = "track_not_started"
    DEFERRED_AMBIGUOUS = "deferred_ambiguous"
    APPLY_FAILED = "apply_failed"


_POSITION_REACHED_TRACKER_STATES = frozenset(
    {SolveLedgerState.QUEUED, SolveLedgerState.APPLYING, SolveLedgerState.APPLIED, SolveLedgerState.HELD_FOR_LABEL}
)
# States whose ``track_id`` holds (or will hold) the solve's position.
_STATES_WITH_OBSERVED_TRACK = frozenset({SolveLedgerState.APPLIED, SolveLedgerState.HELD_FOR_LABEL})


def _solve_brings_its_classification(observation: PositionObservation) -> bool:
    """An event-only Rust solve carries no audio and only one per classifier
    render interval is ever covered by a render, so holding it would only
    delay it by the deadline; it applies at once, and the attach-time gates
    still catch a covering render's contradicting label."""
    return observation.solve_origin != SOLVE_ORIGIN_RUST_EVENT_LOCALIZATION


# eq=False: queue membership and removal must be by identity, never by value.
@dataclass(slots=True, eq=False)
class _SolveLedgerEntry:
    observation: PositionObservation
    state: SolveLedgerState = SolveLedgerState.QUEUED
    track_id: str | None = None
    # This solve's fix started ``track_id`` (a provisional, hidden track).
    started_track: bool = False
    # No classification will come (queue drop, suppression, failure): never hold.
    classification_abandoned: bool = False
    held_deadline: asyncio.TimerHandle | None = None
    # Serialises the drain and this solve's classifications (event + covering render).
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def cancel_held_deadline(self) -> None:
        if self.held_deadline is not None:
            self.held_deadline.cancel()
            self.held_deadline = None


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
        self._held_release_queue: deque[_SolveLedgerEntry] = deque()
        self._ledger: OrderedDict[str, _SolveLedgerEntry] = OrderedDict()
        # solve id -> the track its position reached (None: it never did), after ledger eviction.
        self._evicted_solve_tracks: OrderedDict[str, str | None] = OrderedDict()
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
        if not self._tracker_has_two_step_api():
            # An injected tracker without the two-step API keeps per-detection updates.
            return
        if self.knows_solve(observation.solve_id):
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

    def _tracker_has_two_step_api(self) -> bool:
        return isinstance(self._tracker_provider(), TrackManagerPositionObservationMethods)

    def knows_solve(self, solve_id: str | None) -> bool:
        return solve_id is not None and (solve_id in self._ledger or solve_id in self._evicted_solve_tracks)

    def _remember(self, entry: _SolveLedgerEntry) -> None:
        self._ledger[entry.observation.solve_id] = entry
        # Validation keeps ledger >= queue, so an evicted entry is never still queued.
        while len(self._ledger) > self._config.solve_ledger_max_entries:
            solve_id, evicted = self._ledger.popitem(last=False)
            self._remember_evicted_solve(solve_id, evicted)

    def _remember_evicted_solve(self, solve_id: str, entry: _SolveLedgerEntry) -> None:
        self._evicted_solve_tracks[solve_id] = entry.track_id if entry.state in _STATES_WITH_OBSERVED_TRACK else None
        while len(self._evicted_solve_tracks) > self._config.solve_id_memory_max_entries:
            self._evicted_solve_tracks.popitem(last=False)
        if entry.state is SolveLedgerState.HELD_FOR_LABEL:
            # Its classification can no longer find it; the memory points the label at its gating track.
            self._queue_held_release(entry)

    def _ensure_drain_scheduled(self) -> None:
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.get_running_loop().create_task(
                self._drain_queue(), name="position-observation-drain"
            )

    async def _drain_queue(self) -> None:
        while self._queue or self._held_release_queue:
            # Held fixes are older than anything queued after them: release them first.
            if self._held_release_queue:
                entry = self._held_release_queue.popleft()
                async with entry.lock:
                    await self._release_held_entry(entry)
                continue
            entry = self._queue.popleft()
            async with entry.lock:
                await self._apply_queued_entry(entry)

    async def drain(self) -> None:
        """Apply everything queued (shutdown, tests). Held fixes stay held."""
        while self._drain_task is not None and not self._drain_task.done():
            await asyncio.shield(self._drain_task)
        if self._queue or self._held_release_queue:
            await self._drain_queue()

    def close(self) -> None:
        """Stop the drain task and every held-fix deadline (shutdown)."""
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
        for entry in self._ledger.values():
            entry.cancel_held_deadline()
        self._held_release_queue.clear()

    def queued_count(self) -> int:
        return len(self._queue)

    async def _apply_queued_entry(self, entry: _SolveLedgerEntry) -> None:
        """Caller holds ``entry.lock``."""
        if entry.state is not SolveLedgerState.QUEUED:
            return
        entry.state = SolveLedgerState.APPLYING
        observation = entry.observation
        outcome = await self._apply_position_or_record_failure(
            entry,
            may_start_track=self._may_start_track(observation),
            hold_for_label=not entry.classification_abandoned and _solve_brings_its_classification(observation),
        )
        if outcome is not None:
            self._record_position_outcome(entry, outcome)

    async def _apply_position_or_record_failure(self, entry: _SolveLedgerEntry, *, may_start_track: bool, hold_for_label: bool):
        try:
            return await self._tracker_provider().apply_position_observation(
                entry.observation, may_start_track=may_start_track, hold_for_label=hold_for_label
            )
        except Exception:  # noqa: BLE001 - one bad fix must not stop the drain
            entry.state = SolveLedgerState.APPLY_FAILED
            self._metrics_provider().position_observations_apply_failed += 1
            logger.exception("Position observation failed", extra={"solve_id": entry.observation.solve_id})
            return None

    def _record_position_outcome(self, entry: _SolveLedgerEntry, outcome) -> None:
        metrics = self._metrics_provider()
        if outcome.held_for_label:
            entry.state = SolveLedgerState.HELD_FOR_LABEL
            entry.track_id = outcome.track_id
            metrics.position_observations_held_for_label += 1
            entry.held_deadline = asyncio.get_running_loop().call_later(
                self._config.held_fix_max_wait_seconds, self._queue_held_release, entry
            )
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

    # -- held fixes -----------------------------------------------------------------

    def _queue_held_release(self, entry: _SolveLedgerEntry) -> None:
        """Sync (timer callback, eviction): the drain applies the held fix geometry-only."""
        entry.cancel_held_deadline()
        if entry.state is SolveLedgerState.HELD_FOR_LABEL and entry not in self._held_release_queue:
            self._held_release_queue.append(entry)
            self._ensure_drain_scheduled()

    async def _release_held_entry(self, entry: _SolveLedgerEntry) -> None:
        """Caller holds ``entry.lock``: no classification will resolve this held fix."""
        if entry.state is not SolveLedgerState.HELD_FOR_LABEL:
            return
        entry.cancel_held_deadline()
        outcome = await self._apply_position_or_record_failure(entry, may_start_track=False, hold_for_label=False)
        if outcome is None:
            return
        self._record_position_outcome(entry, outcome)
        if entry.state is SolveLedgerState.APPLIED:
            self._metrics_provider().position_observations_held_applied_geometry_only += 1
        solve_id = entry.observation.solve_id
        if solve_id in self._evicted_solve_tracks:
            self._evicted_solve_tracks[solve_id] = entry.track_id if entry.state is SolveLedgerState.APPLIED else None

    def _drop_held_entry(self, entry: _SolveLedgerEntry) -> None:
        entry.cancel_held_deadline()
        entry.state = SolveLedgerState.HELD_DROPPED
        self._metrics_provider().position_observations_held_dropped += 1

    async def release_held_position_without_classification(self, solve_id: str | None) -> None:
        """This solve will never be classified (queue drop, chunk or saturated-window
        suppression, timeout, backend failure): a held or still-queued fix is applied
        geometry-only now. No-op once its classification resolved it."""
        entry = self._ledger.get(solve_id) if solve_id is not None else None
        if entry is None or not self._tracker_has_two_step_api():
            return
        async with entry.lock:
            entry.classification_abandoned = True
            if entry.state is SolveLedgerState.QUEUED:
                if entry in self._queue:
                    self._queue.remove(entry)
                await self._apply_queued_entry(entry)
            await self._release_held_entry(entry)

    def schedule_held_position_release(self, solve_id: str | None) -> None:
        """Sync form of ``release_held_position_without_classification`` (queue eviction)."""
        entry = self._ledger.get(solve_id) if solve_id is not None else None
        if entry is None:
            return
        entry.classification_abandoned = True
        self._queue_held_release(entry)

    # -- classified-solve resolver -----------------------------------------------

    def solve_position_reached_tracker(self, solve_id: str | None) -> bool:
        """Whether this solve's position is (or is about to be) on a track."""
        if solve_id is None:
            return False
        entry = self._ledger.get(solve_id)
        if entry is not None:
            return entry.state in _POSITION_REACHED_TRACKER_STATES
        return self._evicted_solve_tracks.get(solve_id) is not None

    def solve_fix_is_held(self, solve_id: str | None) -> bool:
        entry = self._ledger.get(solve_id) if solve_id is not None else None
        return entry is not None and entry.state is SolveLedgerState.HELD_FOR_LABEL

    async def classification_found_nothing_trackable(self, solve_id: str | None) -> None:
        """The solve's classification found nothing trackable (no label, a
        sub-display result no belief can lift, a zone-suppressed label): drop a
        held fix, and end the hidden track its fix started if no other fix has
        joined it since."""
        entry = self._ledger.get(solve_id) if solve_id is not None else None
        tracker = self._tracker_provider()
        if entry is None or not isinstance(tracker, TrackManagerPositionObservationMethods):
            return
        async with entry.lock:
            if entry.state is SolveLedgerState.QUEUED:
                if entry in self._queue:
                    self._queue.remove(entry)
                await self._apply_queued_entry(entry)
            if entry.state is SolveLedgerState.HELD_FOR_LABEL:
                self._drop_held_entry(entry)
                return
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
        category is not tracked (tracking mode ``none``), or when the fix was
        dropped (an abstention on a held fix) or its provisional track retracted.
        """
        tracker = self._tracker_provider()
        entry = self._ledger.get(solve_id) if solve_id is not None else None
        if not isinstance(tracker, TrackManagerPositionObservationMethods):
            return await tracker.update(**classified_update_kwargs)
        abstention = classification_is_abstention(classified_update_kwargs)
        if entry is None:
            if solve_id is not None and solve_id in self._evicted_solve_tracks:
                return await self._classify_evicted_solve(tracker, solve_id, classified_update_kwargs, abstention)
            return await tracker.update_with_classification(**classified_update_kwargs)
        async with entry.lock:
            if entry.state is SolveLedgerState.QUEUED:
                # The classification overtook its own observation: apply it now, once.
                # (The drain may already have popped it and be waiting on the lock.)
                if entry in self._queue:
                    self._queue.remove(entry)
                await self._apply_queued_entry(entry)
            track = await self._resolve_classified_entry(tracker, entry, classified_update_kwargs, abstention)
            entry.started_track = False
            if track is not None:
                # Any later classification of this solve (a covering render) attaches only its label.
                entry.state = SolveLedgerState.APPLIED
                entry.track_id = track.id
            return track

    async def _resolve_classified_entry(
        self, tracker: Any, entry: _SolveLedgerEntry, classified_update_kwargs: dict[str, Any], abstention: bool
    ) -> "TrackState | None":
        """Caller holds ``entry.lock``."""
        metrics = self._metrics_provider()
        sensor_count = int(classified_update_kwargs.get("sensor_count", 1))
        if entry.state is SolveLedgerState.HELD_FOR_LABEL:
            if abstention:
                self._drop_held_entry(entry)
                return None
            entry.cancel_held_deadline()
            track = await tracker.apply_held_position_with_label(
                entry.observation, gating_track_id=entry.track_id, classified_update_kwargs=classified_update_kwargs
            )
            if track is None:
                # Resolved without a track (an untracked label category): no later
                # release may apply the fix geometry-only onto the perched track.
                self._drop_held_entry(entry)
                return None
            metrics.position_observations_held_applied_with_label += 1
            return track
        if entry.state is SolveLedgerState.APPLIED:
            if not abstention:
                metrics.position_observation_labels_attached += 1
                return await tracker.attach_classification_to_observed_track(
                    observed_track_id=entry.track_id, classified_update_kwargs=classified_update_kwargs
                )
            if entry.started_track and await tracker.retract_untouched_provisional_track(entry.track_id):
                metrics.position_observation_provisional_tracks_retracted += 1
                return None
            return await tracker.refresh_observed_track_recency(
                observed_track_id=entry.track_id, sensor_count=sensor_count
            )
        if entry.state is SolveLedgerState.HELD_DROPPED:
            return None
        if abstention:
            # The position never reached the tracker and the label says nothing:
            # the geometry-only step (hold rule included), never a new track.
            outcome = await self._apply_position_or_record_failure(entry, may_start_track=False, hold_for_label=True)
            if outcome is None:
                return None
            self._record_position_outcome(entry, outcome)
            if entry.state is SolveLedgerState.HELD_FOR_LABEL:
                self._drop_held_entry(entry)
                return None
            return tracker.observed_track(entry.track_id) if entry.state is SolveLedgerState.APPLIED else None
        metrics.position_observation_classified_updates += 1
        return await tracker.update_with_classification(**classified_update_kwargs)

    async def _classify_evicted_solve(
        self, tracker: Any, solve_id: str, classified_update_kwargs: dict[str, Any], abstention: bool
    ) -> "TrackState | None":
        """The solve's ledger entry is gone; its memory still says where its position went."""
        metrics = self._metrics_provider()
        observed_track_id = self._evicted_solve_tracks[solve_id]
        if observed_track_id is None:
            if abstention:
                return None
            metrics.position_observation_classified_updates += 1
            track = await tracker.update_with_classification(**classified_update_kwargs)
        elif abstention:
            return await tracker.refresh_observed_track_recency(
                observed_track_id=observed_track_id, sensor_count=int(classified_update_kwargs.get("sensor_count", 1))
            )
        else:
            metrics.position_observation_labels_attached += 1
            track = await tracker.attach_classification_to_observed_track(
                observed_track_id=observed_track_id, classified_update_kwargs=classified_update_kwargs
            )
        if track is not None and solve_id in self._evicted_solve_tracks:
            self._evicted_solve_tracks[solve_id] = track.id
        return track


def classified_solve_resolver(position_observation_sink: object) -> PositionObservationPipeline | None:
    """The pipeline behind a seam sink, or None for a no-op / test sink."""
    return position_observation_sink if isinstance(position_observation_sink, PositionObservationPipeline) else None
