"""Field Mode W2-tracker methods of ``TrackManager`` (kept out of tracking.py).

Mixed into ``TrackManager``; relies on its ``_lock``, ``_tracks``, ``_filter``,
``_field_mode``, ``_id_counter``, ``_track_id_prefix`` and helper methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from minimappr.core.track_partial_binding import PartialConstraint
from minimappr.core.track_perched_relocation import RelocationRevertListener
from minimappr.models import TrackState, TrackStatus

if TYPE_CHECKING:
    from minimappr.core.track_field_mode_extensions import FieldModeTrackExtensions


class TrackManagerFieldModeMethods:
    def is_track_hidden(self, track_id: str) -> bool:
        return self._field_mode.is_hidden(track_id)

    @property
    def field_mode(self) -> FieldModeTrackExtensions:
        return self._field_mode

    def add_relocation_revert_listener(self, listener: RelocationRevertListener) -> None:
        """Called (under the tracker lock, synchronously) when a perched hop is undone."""
        self._field_mode.relocation_gate.add_revert_listener(listener)

    async def bind_partial_constraints(
        self,
        *,
        timestamp_ns: int,
        label: str,
        confidence: float,
        label_category: str,
        constraints: list[PartialConstraint],
        species_evidence: dict[str, float] | None = None,
        classifier_scores: dict[str, float] | None = None,
        source_node_id: str | None = None,
        below_display_threshold: bool = False,
    ) -> TrackState | None:
        """Bind an unpositioned detection to a recent track, or return ``None``.

        On a bind the track's species belief and recency update and its position
        gets a soft nudge toward the constraint (see ``track_partial_binding``).
        """
        if not self._field_mode.partial_binding_enabled or not self.tracks_category(label_category):
            return None
        async with self._lock:
            self._age_tracks(timestamp_ns)
            evidence = self._field_mode.detection_evidence(
                label=label,
                confidence=confidence,
                classifier_scores=classifier_scores,
                species_evidence=species_evidence,
            )
            decision = self._field_mode.choose_partial_binding(
                timestamp_ns=timestamp_ns,
                evidence=evidence,
                constraints=constraints,
                tracks=list(self._tracks.values()),
            )
            if decision is None:
                return None
            track = self._tracks[decision.track_id]
            if decision.corrected_position_m is not None:
                corrected = tuple(float(value) for value in decision.corrected_position_m)
                # Nudge only when the filter takes it too; otherwise its next
                # predict would silently overwrite the corrected position.
                soft_update = getattr(self._filter, "soft_position_update", None)
                if callable(soft_update) and soft_update(track.id, corrected):
                    track.position_m = corrected  # type: ignore[assignment]
            track.last_seen_ns = max(track.last_seen_ns, timestamp_ns)
            track.update_count += 1
            if self._field_mode.belief_applies(track.label_category or label_category):
                self._field_mode.apply_belief(
                    track, evidence, display_level=not below_display_threshold
                )
            elif label == track.label:
                track.confidence = float(max(track.confidence, confidence))
            if source_node_id and source_node_id not in track.contributor_node_ids:
                track.contributor_node_ids.append(source_node_id)
                track.contributor_node_ids = track.contributor_node_ids[-self._MAX_CONTRIBUTOR_NODE_IDS :]
            if track.status == TrackStatus.COASTING.value:
                track.status = TrackStatus.CONFIRMED.value
            self._update_fingerprint(track.id, classifier_scores)
            return track

    def _revert_relocation(self, ghost, timestamp_ns: int) -> None:
        """A fix landed back on the perch a track just left: it was two birds.

        The track returns to its old perch; whatever it picked up at the new
        perch continues as a new track (see ``track_perched_relocation``).
        """
        relocated = self._tracks.get(ghost.track_id)
        if relocated is None or relocated.status == TrackStatus.DROPPED.value:
            # The filter state is gone; reviving it here would resurrect a dead track.
            self._field_mode.relocation_gate.forget(ghost.track_id)
            return
        split_id = f"{self._track_id_prefix}{next(self._id_counter):05d}"
        split = relocated.model_copy(
            update={
                "id": split_id,
                "first_seen_ns": ghost.left_at_ns,
                "update_count": 1,
                "status": TrackStatus.TENTATIVE.value,
                "contributor_node_ids": list(relocated.contributor_node_ids),
            }
        )
        self._tracks[split_id] = split
        self._filter.initialize_track(split_id, split.position_m)
        self._apply_motion_model(split_id, split.label_category, split.position_m)
        self._field_mode.clone_belief(relocated.id, split_id)
        relocated.position_m = ghost.perch_position_m
        self._filter.initialize_track(relocated.id, ghost.perch_position_m)
        self._field_mode.relocation_gate.mark_reverted(ghost, split_id, timestamp_ns)
