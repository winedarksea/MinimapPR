"""One-observation-per-solve methods of ``TrackManager`` (plan P5).

Mixed into ``TrackManager``; relies on its ``_lock``, ``_tracks``, ``_filter``,
``_associator``, ``_field_mode``, ``_fingerprints``, ``_id_counter``, helper
methods and P4's ``apply_label_category_tracking_mode``. Split from ``update``
because a solve reaches the tracker in two steps:

1. ``apply_position_observation``: the position, at the solve's first
   arrival. The fix associates on geometry alone (the label is unknown, which
   every gate treats as a wildcard) and never touches the track's label or
   dynamics. With no match, an event-triggered fix may start a *hidden*
   tentative track. When more than one track gates the fix, geometry cannot
   pick the source (two birds a few metres apart), so the fix is left to its
   classification's label-aware ``update`` (species gates, fingerprints).
   When the one gating track is a labelled ``stationary_hold`` track, the fix
   is *held* for its classification (see ``_fix_waits_for_label``).
2. ``attach_classification_to_observed_track``: the label, later, on the track
   the position already moved, without moving it again (no change to
   ``update_count`` or ``last_seen_ns``); a hidden track becomes visible. Only
   when the classification contradicts that track (its category, or the
   label-dependent association gates: species log-odds, concurrent perch), or
   the track is gone, does the label-aware ``update`` run instead, so the fix
   still reaches a track of the right source.

An abstention (a classification with no informative category) never labels,
reveals or re-labels a track: the pipeline retracts a hidden track only this
solve's fix started, otherwise ``refresh_observed_track_recency`` refreshes
recency only.
A classification that finds nothing trackable (no label, a sub-display result
no belief can lift, a zone-suppressed label) also retracts such a track
(``retract_untouched_provisional_track``), so an unconfirmed fix cannot linger
and capture a later source.

Hidden-unclassified tracks are left out of ``snapshot`` and reported by
``is_track_hidden`` until a classification reveals them or they are reaped
(dropped ones included), so they never reach storage, the live feed,
federation, rules or the dormant registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from minimappr.interfaces import AssociationContext
from minimappr.models import TrackState, TrackStatus

if TYPE_CHECKING:
    from minimappr.core.position_observation import PositionObservation

# The label an unclassified track carries; every association gate treats it as a wildcard.
UNCLASSIFIED_TRACK_LABEL = "unknown"
UNCLASSIFIED_TRACK_CATEGORY = "unknown"
UNCLASSIFIED_TRACK_CONFIDENCE = 0.0
# Same dt floor ``TrackManager.update`` applies before a filter update.
MINIMUM_FILTER_UPDATE_INTERVAL_SECONDS = 1e-3
_VALID_IFF_CATEGORIES = frozenset({"friendly", "unknown", "hostile"})
_NON_CONFIRMING_CAPABILITY_TIERS = frozenset({"classification_only", "alerting_only"})


@dataclass(frozen=True, slots=True)
class PositionObservationTrackOutcome:
    """Which track a position observation updated; ``track_id`` None means none."""

    track_id: str | None
    started_track: bool
    # Several tracks gate the fix; it waits for its classification's label-aware update.
    deferred_ambiguous: bool = False
    # Only ``track_id``, a labelled stationary_hold track, gates the fix; it waits for its classification.
    held_for_label: bool = False


def _informative_label_category(value: str | None) -> str | None:
    normalised = (value or "").strip().lower()
    return None if not normalised or normalised == UNCLASSIFIED_TRACK_CATEGORY else normalised


def classification_is_abstention(label_category: str | None) -> bool:
    """No informative category (classifier abstention, unsettled onset attribution):
    the classification says nothing about which source the fix is."""
    return _informative_label_category(label_category) is None


class TrackManagerPositionObservationMethods:
    def _init_position_observation_state(self) -> None:
        # Tracks started by an unclassified fix and not yet classified.
        self._unclassified_track_ids: set[str] = set()

    # -- visibility -------------------------------------------------------------

    def is_track_hidden(self, track_id: str) -> bool:
        return super().is_track_hidden(track_id) or self._is_unclassified_hidden(track_id)  # type: ignore[misc]

    def _is_unclassified_hidden(self, track_id: str) -> bool:
        if track_id not in self._unclassified_track_ids:
            return False
        track = self._tracks.get(track_id)
        # Any path that gave the track a real label (e.g. partial binding) has classified it.
        return track is not None and track.label == UNCLASSIFIED_TRACK_LABEL

    def _prune_unclassified_track_ids(self) -> None:
        # Pruned only once reaped: a DROPPED track stays in ``_tracks`` until then
        # and must stay hidden there too.
        self._unclassified_track_ids = {
            track_id for track_id in self._unclassified_track_ids if track_id in self._tracks
        }

    def _parks_as_dormant(self, track: TrackState) -> bool:
        # A never-classified track has no identity to reacquire; a later
        # abstention would match its "unknown" label and revive it visible.
        return not self._is_unclassified_hidden(track.id)

    # -- step 1: position ---------------------------------------------------------

    async def apply_position_observation(
        self, observation: "PositionObservation", *, may_start_track: bool, hold_for_label: bool = True
    ) -> PositionObservationTrackOutcome:
        """``hold_for_label`` False applies a would-be-held fix geometry-only (its
        classification will never arrive)."""
        async with self._lock:
            timestamp_ns = observation.event_time_ns
            self._age_tracks(timestamp_ns)
            self._prune_unclassified_track_ids()
            gating_track_ids = self._track_ids_gating_unclassified_position(observation)
            if len(gating_track_ids) > 1:
                # Geometry alone cannot tell which source this is (two birds a few
                # metres apart, crossing drones); the label-aware update, with its
                # species gates and fingerprint ranking, decides once classified.
                return PositionObservationTrackOutcome(track_id=None, started_track=False, deferred_ambiguous=True)
            if gating_track_ids:
                [matched_id] = gating_track_ids
                matched_track = self._tracks[matched_id]
                if hold_for_label and self._fix_waits_for_label(matched_track):
                    return PositionObservationTrackOutcome(track_id=matched_id, started_track=False, held_for_label=True)
                self._apply_position_to_track(matched_track, observation)
                return PositionObservationTrackOutcome(track_id=matched_id, started_track=False)
            if not may_start_track:
                return PositionObservationTrackOutcome(track_id=None, started_track=False)
            started = self._start_unclassified_track(observation)
            return PositionObservationTrackOutcome(track_id=started.id, started_track=True)

    def _fix_waits_for_label(self, track: TrackState) -> bool:
        """Hold rule (a deliberate change to plan P5, see docs/tracking.md).

        A perched bird's filter averages every fix in for good, and only the
        label can tell this bird's call from a neighbour's overlapping call
        (species log-odds gate, concurrent-perch guard). So a fix whose one
        gating track is a labelled stationary_hold track waits for its
        classification; continuous-motion and hidden tracks update at once.
        """
        return (
            not self.is_track_hidden(track.id)
            and _informative_label_category(track.label_category) is not None
            and self.tracking_modes.holds_station(track.label_category)
        )

    def _track_ids_gating_unclassified_position(self, observation: "PositionObservation") -> list[str]:
        """Every live track the associator would accept this fix for, taken one at a time."""
        timestamp_ns = observation.event_time_ns
        context = AssociationContext(
            label=UNCLASSIFIED_TRACK_LABEL,
            label_category=UNCLASSIFIED_TRACK_CATEGORY,
            classifier_scores=None,
            track_fingerprints=self._fingerprints,
        )
        gating_track_ids: list[str] = []
        for track in self._tracks.values():
            if track.status == TrackStatus.DROPPED.value:
                continue
            predicted = self._filter.predict(track, max((timestamp_ns - track.last_seen_ns) / 1_000_000_000.0, 0.0))
            if self._associate_with(observation, [predicted], context) is not None:
                gating_track_ids.append(track.id)
        return gating_track_ids

    def _associate_with(
        self, observation: "PositionObservation", predicted_tracks: list[TrackState], context: AssociationContext
    ) -> str | None:
        timestamp_ns, position_m = observation.event_time_ns, observation.position_m
        covariance_m2 = observation.position_covariance_m2
        if self._associator_accepts_context:
            return self._associator.associate(timestamp_ns, position_m, predicted_tracks, covariance_m2, context)
        if self._associator_accepts_measurement_covariance:
            return self._associator.associate(timestamp_ns, position_m, predicted_tracks, covariance_m2)
        return self._associator.associate(timestamp_ns, position_m, predicted_tracks)

    def _apply_position_to_track(self, track: TrackState, observation: "PositionObservation") -> None:
        """Filter the fix into ``track``; label, category and dynamics stay as they are."""
        timestamp_ns = observation.event_time_ns
        dt_s = max((timestamp_ns - track.last_seen_ns) / 1_000_000_000.0, MINIMUM_FILTER_UPDATE_INTERVAL_SECONDS)
        if self._filter_accepts_measurement_covariance:
            filtered = self._filter.update(track, observation.position_m, dt_s, observation.position_covariance_m2)
        else:
            filtered = self._filter.update(track, observation.position_m, dt_s)
        # A held fix released late can be older than the track's last update.
        track.last_seen_ns = max(track.last_seen_ns, timestamp_ns)
        track.position_m = filtered.position_m
        track.velocity_mps = filtered.velocity_mps
        track.position_covariance_m2 = filtered.position_covariance_m2
        track.update_count += 1
        track.capability_tier = observation.capability_tier
        self._record_contributor_node(track, observation.reporting_node_id)
        can_confirm = observation.capability_tier not in _NON_CONFIRMING_CAPABILITY_TIERS
        if track.status == TrackStatus.COASTING.value:
            track.status = TrackStatus.CONFIRMED.value if can_confirm else TrackStatus.TENTATIVE.value
        elif track.status == TrackStatus.TENTATIVE.value:
            if can_confirm and track.update_count >= self.CONFIRM_THRESHOLD:
                track.status = TrackStatus.CONFIRMED.value
        elif can_confirm:
            track.status = TrackStatus.CONFIRMED.value
        track.tqi = self._compute_tqi(
            track.confidence,
            track.update_count,
            dt_s,
            observation.contributing_sensor_count,
            contributor_count=max(len(track.contributor_node_ids), 1),
        )

    def _record_contributor_node(self, track: TrackState, node_id: str | None) -> None:
        if not node_id or node_id in track.contributor_node_ids:
            return
        had_single_node = len(track.contributor_node_ids) <= 1
        track.contributor_node_ids.append(node_id)
        if len(track.contributor_node_ids) > self._MAX_CONTRIBUTOR_NODE_IDS:
            track.contributor_node_ids = track.contributor_node_ids[-self._MAX_CONTRIBUTOR_NODE_IDS :]
        if had_single_node and len(track.contributor_node_ids) >= 2:
            self._multi_node_association_count += 1

    def _start_unclassified_track(self, observation: "PositionObservation") -> TrackState:
        track_id = f"{self._track_id_prefix}{next(self._id_counter):05d}"
        initial_variance = self._initial_position_variance
        position_m = observation.position_m
        started = TrackState(
            id=track_id,
            first_seen_ns=observation.event_time_ns,
            last_seen_ns=observation.event_time_ns,
            position_m=position_m,
            position_covariance_m2=(
                observation.position_covariance_m2
                if observation.position_covariance_m2 is not None
                else [[initial_variance, 0.0, 0.0], [0.0, initial_variance, 0.0], [0.0, 0.0, initial_variance]]
            ),
            velocity_mps=(0.0, 0.0, 0.0),
            label=UNCLASSIFIED_TRACK_LABEL,
            label_category=UNCLASSIFIED_TRACK_CATEGORY,
            confidence=UNCLASSIFIED_TRACK_CONFIDENCE,
            update_count=1,
            status=TrackStatus.TENTATIVE.value,
            tqi=self._compute_tqi(
                UNCLASSIFIED_TRACK_CONFIDENCE, 1, 0.0, observation.contributing_sensor_count, contributor_count=1
            ),
            capability_tier=observation.capability_tier,
            track_kind=self._default_track_kind,
            contributor_node_ids=[observation.reporting_node_id] if observation.reporting_node_id else [],
        )
        self._tracks[track_id] = started
        self._filter.initialize_track(track_id, position_m)
        # No label yet: the uncategorised dynamics apply until a classification sets the mode.
        self._apply_motion_model(track_id, UNCLASSIFIED_TRACK_CATEGORY, position_m)
        self._apply_process_noise(track_id, UNCLASSIFIED_TRACK_CATEGORY)
        self._unclassified_track_ids.add(track_id)
        return started

    # -- step 2: label ------------------------------------------------------------

    async def attach_classification_to_observed_track(
        self,
        *,
        observed_track_id: str,
        classified_update_kwargs: dict[str, Any],
    ) -> TrackState | None:
        """Label the track a solve's position already updated.

        ``classified_update_kwargs`` are ``TrackManager.update`` keywords for this
        solve (position included, used only when the label-aware update must be
        re-run; see the module docstring). ``None`` when the label's category is
        not tracked (tracking mode ``none``): an unclassified provisional track
        is ended, as ``update`` would never have started it; a labelled track is
        left alone (one fix of an untracked source must not end it, which
        matters when ``association_category_gate_enabled`` is off).
        """
        async with self._lock:
            track = self._tracks.get(observed_track_id)
            attachable = track is not None and track.status != TrackStatus.DROPPED.value
            if attachable and self._classification_rejects_track(track, classified_update_kwargs):
                attachable = False
            if attachable:
                return self._apply_classification_to_track(track, classified_update_kwargs)
        return await self.update_with_classification(**classified_update_kwargs)

    async def apply_held_position_with_label(
        self,
        observation: "PositionObservation",
        *,
        gating_track_id: str,
        classified_update_kwargs: dict[str, Any],
    ) -> TrackState | None:
        """Resolve a held fix with its informative label.

        The fix goes to the track that gated it, now with the label, unless the
        label-dependent gates reject that track (another species, a concurrent
        perch) or it is gone; then the label-aware ``update`` places it (a new
        track for the other bird). Re-associating every held fix from scratch
        instead would let one poorly conditioned fix jump to a nearer
        provisional track whose filter is still loose (``gate_flock_echo``).
        """
        async with self._lock:
            self._age_tracks(observation.event_time_ns)
            track = self._tracks.get(gating_track_id)
            if (
                track is not None
                and track.status != TrackStatus.DROPPED.value
                and not self._classification_rejects_track(track, classified_update_kwargs)
            ):
                self._apply_position_to_track(track, observation)
                return self._apply_classification_to_track(track, classified_update_kwargs)
        return await self.update_with_classification(**classified_update_kwargs)

    async def refresh_observed_track_recency(self, *, observed_track_id: str | None, sensor_count: int) -> TrackState | None:
        """An abstention on the track a solve's position moved: no label, no reveal,
        only the recency the label step would refresh. ``None`` if the track is gone."""
        async with self._lock:
            track = self._tracks.get(observed_track_id) if observed_track_id is not None else None
            if track is None or track.status == TrackStatus.DROPPED.value:
                return None
            track.tqi = self._compute_tqi(
                track.confidence,
                track.update_count,
                0.0,
                sensor_count,
                contributor_count=max(len(track.contributor_node_ids), 1),
            )
            return track

    def observed_track(self, track_id: str | None) -> TrackState | None:
        return self._tracks.get(track_id) if track_id is not None else None

    async def retract_untouched_provisional_track(self, track_id: str | None) -> bool:
        """End a hidden-unclassified track that holds a single fix; False if none."""
        async with self._lock:
            track = self._tracks.get(track_id) if track_id is not None else None
            if track is None or not self._is_unclassified_hidden(track_id) or track.update_count != 1:
                return False
            self._tracks.pop(track_id, None)
            self._filter.remove_track(track_id)
            self._fingerprints.pop(track_id, None)
            self._field_mode.forget(track_id)
            self._unclassified_track_ids.discard(track_id)
            return True

    async def update_with_classification(self, **classified_update_kwargs: Any) -> TrackState | None:
        """The label-aware ``update``; whichever track it lands on is classified now."""
        track = await self.update(**classified_update_kwargs)
        if track is not None:
            self._unclassified_track_ids.discard(track.id)
        return track

    def _classification_rejects_track(self, track: TrackState, update_kwargs: dict[str, Any]) -> bool:
        """The label-dependent association gates ``update`` would apply, re-run
        now the label exists: a rejection means the geometric match was wrong."""
        label_category = update_kwargs.get("label_category", UNCLASSIFIED_TRACK_CATEGORY)
        if self._classification_contradicts_track(track, label_category):
            return True
        evidence = self._field_mode.detection_evidence(
            label=update_kwargs["label"],
            confidence=float(update_kwargs["confidence"]),
            classifier_scores=update_kwargs.get("classifier_scores"),
            species_evidence=update_kwargs.get("species_evidence"),
        )
        timestamp_ns = int(update_kwargs["timestamp_ns"])
        # Every live track, as ``update`` passes them: the concurrent-perch guard
        # needs the neighbours to see who sang since.
        live_tracks = [
            self._filter.predict(candidate, max((timestamp_ns - candidate.last_seen_ns) / 1_000_000_000.0, 0.0))
            for candidate in self._tracks.values()
            if candidate.status != TrackStatus.DROPPED.value
        ]
        admitted = self._field_mode.association_candidates(
            timestamp_ns=timestamp_ns,
            position_m=update_kwargs["position_m"],
            measurement_covariance_m2=update_kwargs.get("measurement_covariance_m2"),
            label_category=label_category,
            evidence=evidence,
            tracks=live_tracks,
            track_filter=self._filter,
        )
        return track.id not in {candidate.id for candidate in admitted}

    def _classification_contradicts_track(self, track: TrackState, label_category: str | None) -> bool:
        # The same rule the associator's category gate applies to a labeled fix.
        if not getattr(self._cfg, "association_category_gate_enabled", True):
            return False
        track_category = _informative_label_category(track.label_category)
        detection_category = _informative_label_category(label_category)
        return track_category is not None and detection_category is not None and track_category != detection_category

    def _apply_classification_to_track(self, track: TrackState, update_kwargs: dict[str, Any]) -> TrackState | None:
        """The label half of ``TrackManager.update`` for an already-matched track; caller holds the lock."""
        label = update_kwargs["label"]
        confidence = float(update_kwargs["confidence"])
        label_category = update_kwargs.get("label_category", UNCLASSIFIED_TRACK_CATEGORY)
        iff_category = update_kwargs.get("iff_category", "unknown")
        classifier_scores = update_kwargs.get("classifier_scores")
        below_display_threshold = bool(update_kwargs.get("below_display_threshold", False))
        evidence = self._field_mode.detection_evidence(
            label=label,
            confidence=confidence,
            classifier_scores=classifier_scores,
            species_evidence=update_kwargs.get("species_evidence"),
        )
        if not self.tracks_category(label_category) and _informative_label_category(track.label_category):
            return None
        label_changed = label != track.label
        category_changed = label_category != track.label_category
        track.label_id = update_kwargs.get("label_id")
        track.label = label
        track.label_category = label_category
        # Same rule as ``update``: an abstention keeps the track's dynamics.
        if category_changed and _informative_label_category(label_category) is not None:
            if not self.apply_label_category_tracking_mode(track.id, label_category):
                self._unclassified_track_ids.discard(track.id)
                return None
        track.iff_category = iff_category if iff_category in _VALID_IFF_CATEGORIES else "unknown"
        if self._field_mode.belief_applies(label_category):
            self._field_mode.apply_belief(track, evidence, display_level=not below_display_threshold)
        elif label_changed:
            track.confidence = confidence
        else:
            track.confidence = float(max(track.confidence, confidence))
        # The position step already scored this update's recency; the label only
        # changes confidence, so recency is "just updated" here.
        track.tqi = self._compute_tqi(
            track.confidence,
            track.update_count,
            0.0,
            int(update_kwargs.get("sensor_count", 1)),
            contributor_count=max(len(track.contributor_node_ids), 1),
        )
        self._update_fingerprint(track.id, classifier_scores)
        self._unclassified_track_ids.discard(track.id)
        return track
