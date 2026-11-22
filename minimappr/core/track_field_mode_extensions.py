"""Field Mode W2-tracker behaviour owned by ``TrackManager``.

Kept out of ``tracking.py`` so the manager only calls a handful of hooks:

- species belief (per-track capped log-odds) replacing the single-label
  overwrite, plus the display threshold that hides tentative tracks;
- the perched relocation gate (a 10-30 m hop with a matching species and no
  nearer same-species track is the same bird on a new perch);
- partial-constraint binding of unpositioned detections to recent tracks.

Every behaviour is off unless its ``TrackingConfig`` flag is set, in which case
the manager behaves exactly as before.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from minimappr.core.track_filters_perched import (
    MotionModelRoutingTrackFilter,
    PerchedRandomWalkTrackFilter,
)
from minimappr.core.track_partial_binding import (
    PartialBindingCandidate,
    PartialBindingConfig,
    PartialBindingDecision,
    PartialConstraint,
    choose_track_for_partial_constraints,
)
from minimappr.core.track_partial_constraint_evidence import PartialConstraintSigmas
from minimappr.core.track_perched_concurrency import (
    ConcurrentPerchGuardConfig,
    concurrently_active_perch_conflicts,
    fix_lands_on_another_perch,
)
from minimappr.core.track_perched_relocation import PerchedRelocationGate, RelocationGhost
from minimappr.core.track_species_association_gate import (
    SpeciesAssociationGateConfig,
    species_compatible_tracks,
)
from minimappr.core.track_species_belief import (
    SpeciesBeliefConfig,
    SpeciesLogOddsBelief,
    species_evidence_from_detection,
)
from minimappr.interfaces import TrackFilter
from minimappr.models import TrackState, TrackStatus


def parse_label_category_set(value: str | Iterable[str] | None) -> frozenset[str]:
    """Comma-separated (or iterable) label categories -> normalised set."""
    if value is None:
        return frozenset()
    items = value.split(",") if isinstance(value, str) else list(value)
    return frozenset(item.strip().lower() for item in items if item and item.strip())


def _normalised_label(label: str | None) -> str:
    return (label or "").strip().lower()


class FieldModeTrackExtensions:
    """Species belief, relocation gate and partial binding for one TrackManager."""

    def __init__(self, cfg) -> None:
        self.perched_filter_enabled = bool(getattr(cfg, "tracking_perched_filter_enabled", False))
        self.species_log_odds_enabled = bool(getattr(cfg, "tracking_species_log_odds_enabled", False))
        self.partial_binding_enabled = bool(
            getattr(cfg, "tracking_partial_constraint_binding_enabled", False)
        )
        self._perched_label_categories = parse_label_category_set(
            getattr(cfg, "tracking_perched_label_categories", "wildlife")
        )
        self._belief_label_categories = parse_label_category_set(
            getattr(cfg, "tracking_species_log_odds_label_categories", "wildlife")
        )
        self.relocation_gate = PerchedRelocationGate(
            association_distance_m=float(cfg.association_distance_m),
            relocation_gate_max_m=float(getattr(cfg, "tracking_perched_relocation_gate_max_m", 30.0)),
            relocation_min_silence_seconds=float(
                getattr(cfg, "tracking_perched_relocation_min_silence_seconds", 2.0)
            ),
            relocation_ghost_seconds=float(getattr(cfg, "tracking_perched_relocation_ghost_seconds", 30.0)),
        )
        self._belief_config = SpeciesBeliefConfig(
            prior_probability=float(getattr(cfg, "tracking_species_log_odds_prior_probability", 0.1)),
            per_update_log_odds_cap=float(getattr(cfg, "tracking_species_log_odds_per_update_cap", 3.0)),
            total_log_odds_cap=float(getattr(cfg, "tracking_species_log_odds_total_cap", 6.0)),
            evidence_top_k=int(getattr(cfg, "tracking_species_log_odds_evidence_top_k", 3)),
            display_min_belief=float(getattr(cfg, "tracking_species_display_min_belief", 0.5)),
            exposed_top_n=int(getattr(cfg, "tracking_species_belief_exposed_top_n", 5)),
        )
        self._binding_config = PartialBindingConfig(
            window_seconds=float(getattr(cfg, "tracking_partial_binding_window_seconds", 25.0)),
            max_normalized_residual=float(
                getattr(cfg, "tracking_partial_binding_max_normalized_residual", 3.0)
            ),
            species_top_k=int(getattr(cfg, "tracking_partial_binding_species_top_k", 3)),
            soft_update_gain=float(getattr(cfg, "tracking_partial_binding_soft_update_gain", 0.3)),
            ambiguity_score_ratio=float(
                getattr(cfg, "tracking_partial_binding_ambiguity_score_ratio", 0.5)
            ),
        )
        self.partial_constraint_sigmas = PartialConstraintSigmas(
            bearing_sigma_deg=float(getattr(cfg, "tracking_partial_binding_bearing_sigma_deg", 10.0)),
            hyperbola_sigma_m=float(getattr(cfg, "tracking_partial_binding_hyperbola_sigma_m", 2.0)),
            snr_level_margin_db=float(getattr(cfg, "tracking_partial_binding_snr_level_margin_db", 7.0)),
        )
        # --- Field Mode S-fixes tracking ---
        self._species_association_gate = SpeciesAssociationGateConfig(
            enabled=self.species_log_odds_enabled
            and bool(getattr(cfg, "tracking_species_association_gate_enabled", True)),
            confident_probability=float(
                getattr(cfg, "tracking_species_association_confident_probability", 0.5)
            ),
            detection_claim_probability=float(
                getattr(cfg, "tracking_species_association_detection_claim_probability", 0.15)
            ),
        )
        self._concurrent_perch_guard = ConcurrentPerchGuardConfig(
            enabled=self.perched_filter_enabled
            and bool(getattr(cfg, "tracking_perched_concurrent_perch_guard_enabled", True)),
            snap_distance_m=float(getattr(cfg, "tracking_perched_relocation_snap_distance_m", 6.0)),
            min_silence_seconds=float(getattr(cfg, "tracking_perched_relocation_min_silence_seconds", 2.0)),
            concurrent_activity_seconds=float(
                getattr(cfg, "tracking_perched_concurrent_activity_seconds", 30.0)
            ),
        )
        # --- end Field Mode S-fixes tracking ---
        self._beliefs: dict[str, SpeciesLogOddsBelief] = {}
        # Tracks that received at least one display-level detection. Only tracks
        # built purely from sub-display (two-threshold) evidence can be hidden, so
        # nothing that surfaces today is ever hidden by turning these flags on.
        self._revealed_track_ids: set[str] = set()
        self.partial_bindings_accepted = 0
        self.partial_bindings_unbound = 0

    # -- filter construction --------------------------------------------------

    def wrap_filter(self, constant_velocity_filter: TrackFilter, cfg) -> TrackFilter:
        """Route perched categories to the random-walk filter when flagged."""
        if not self.perched_filter_enabled:
            return constant_velocity_filter
        perched = PerchedRandomWalkTrackFilter(
            position_diffusion_m2_per_second=float(
                getattr(cfg, "tracking_perched_position_diffusion_m2_per_second", 0.05)
            ),
            measurement_noise_m2=float(cfg.kalman_measurement_noise),
            initial_position_variance_m2=float(cfg.kalman_initial_position_variance),
            relocation_snap_distance_m=float(
                getattr(cfg, "tracking_perched_relocation_snap_distance_m", 6.0)
            ),
        )
        return MotionModelRoutingTrackFilter(
            constant_velocity_filter=constant_velocity_filter,
            perched_filter=perched,
            perched_label_categories=self._perched_label_categories,
        )

    # -- species belief ---------------------------------------------------------

    def belief_applies(self, label_category: str) -> bool:
        return self.species_log_odds_enabled and (
            _normalised_label(label_category) in self._belief_label_categories
        )

    def detection_evidence(
        self,
        *,
        label: str,
        confidence: float,
        classifier_scores: dict[str, float] | None,
        species_evidence: dict[str, float] | None,
        winner_member: str | None = None,
    ) -> dict[str, float]:
        if not (self.species_log_odds_enabled or self.perched_filter_enabled or self.partial_binding_enabled):
            # Nothing consumes evidence with every flag off; skip the work.
            return {}
        if species_evidence is not None:
            return dict(species_evidence)
        return species_evidence_from_detection(
            label=label,
            confidence=confidence,
            classifier_scores=classifier_scores,
            winner_member=winner_member,
            top_k=self._belief_config.evidence_top_k,
        )

    def apply_belief(
        self, track: TrackState, evidence: dict[str, float], *, display_level: bool = True
    ) -> None:
        """Fuse *evidence* and project top label/probability onto *track*."""
        if display_level:
            self._revealed_track_ids.add(track.id)
        belief = self._beliefs.get(track.id)
        if belief is None:
            belief = SpeciesLogOddsBelief(self._belief_config)
            self._beliefs[track.id] = belief
        belief.update(evidence)
        if belief.passes_display_threshold():
            # Once shown (and persisted), a track stays visible: hiding it again
            # on a later weak bout would make it flicker off the map.
            self._revealed_track_ids.add(track.id)
        top_label, top_probability = belief.top_label()
        if top_label is not None:
            track.label = top_label
            track.confidence = float(top_probability)
        track.species_belief = belief.exposed_probabilities()
        track.species_top_label = top_label
        track.species_top_probability = float(top_probability) if top_label is not None else None

    def track_top_species(self, track: TrackState, count: int) -> list[str]:
        belief = self._beliefs.get(track.id)
        if belief is not None:
            return belief.top_labels(count)
        label = _normalised_label(track.label)
        return [track.label] if label and label != "unknown" else []

    def belief_probabilities(self, track_id: str) -> dict[str, float]:
        belief = self._beliefs.get(track_id)
        return belief.exposed_probabilities() if belief is not None else {}

    # --- Field Mode S-fixes tracking ---
    def association_candidates(
        self,
        *,
        timestamp_ns: int,
        position_m: tuple[float, float, float],
        measurement_covariance_m2: list[list[float]] | None,
        label_category: str,
        evidence: dict[str, float],
        tracks: list[TrackState],
        track_filter: TrackFilter,
    ) -> list[TrackState]:
        """Tracks the associator may consider: species-compatible, and not a
        perch another concurrently singing bird of the species would be merged into."""
        candidates = tracks
        if self.belief_applies(label_category):
            candidates = species_compatible_tracks(
                candidates,
                detection_evidence=evidence,
                belief_for_track=self.belief_probabilities,
                weak_claim_fits_track=lambda track: not fix_lands_on_another_perch(
                    position_m=position_m,
                    measurement_covariance_m2=measurement_covariance_m2,
                    track=track,
                    snap_distance_m=self._concurrent_perch_guard.snap_distance_m,
                ),
                config=self._species_association_gate,
            )
        if (
            self._concurrent_perch_guard.enabled
            and isinstance(track_filter, MotionModelRoutingTrackFilter)
            and track_filter.category_is_perched(label_category)
        ):
            conflicts = concurrently_active_perch_conflicts(
                timestamp_ns=timestamp_ns,
                position_m=position_m,
                measurement_covariance_m2=measurement_covariance_m2,
                detection_species=max(evidence.items(), key=lambda item: item[1])[0] if evidence else None,
                tracks=candidates,
                top_species_for=lambda track: self.track_top_species(track, 1),
                is_perched=track_filter.is_perched,
                config=self._concurrent_perch_guard,
            )
            candidates = [track for track in candidates if track.id not in conflicts]
        return candidates
    # --- end Field Mode S-fixes tracking ---

    def is_hidden(self, track_id: str) -> bool:
        """A sub-display-only track whose belief never reached the display threshold."""
        return track_id in self._beliefs and track_id not in self._revealed_track_ids

    def forget(self, track_id: str) -> None:
        self._beliefs.pop(track_id, None)
        self._revealed_track_ids.discard(track_id)
        self.relocation_gate.forget(track_id)

    # -- perched relocation gate ------------------------------------------------

    def find_relocation_match(
        self,
        *,
        timestamp_ns: int,
        position_m: tuple[float, float, float],
        label_category: str,
        evidence: dict[str, float],
        tracks: Iterable[TrackState],
        track_filter: TrackFilter,
    ) -> str | None:
        if not self.perched_filter_enabled or not isinstance(track_filter, MotionModelRoutingTrackFilter):
            return None
        if not track_filter.category_is_perched(label_category) or not evidence:
            return None
        return self.relocation_gate.find_relocation_match(
            timestamp_ns=timestamp_ns,
            position_m=position_m,
            detection_species=max(evidence.items(), key=lambda item: item[1])[0],
            tracks=tracks,
            top_species_for=lambda track: self.track_top_species(track, 1),
            is_perched=track_filter.is_perched,
        )

    def find_relocation_ghost_hit(
        self,
        *,
        timestamp_ns: int,
        position_m: tuple[float, float, float],
        label_category: str,
        evidence: dict[str, float],
    ) -> RelocationGhost | None:
        if not self.perched_filter_enabled or not evidence:
            return None
        if _normalised_label(label_category) not in self._perched_label_categories:
            return None
        return self.relocation_gate.find_ghost_hit(
            timestamp_ns=timestamp_ns,
            position_m=position_m,
            detection_species=max(evidence.items(), key=lambda item: item[1])[0],
        )

    def clone_belief(self, source_track_id: str, target_track_id: str) -> None:
        belief = self._beliefs.get(source_track_id)
        if belief is not None:
            self._beliefs[target_track_id] = belief.copy()
        if source_track_id in self._revealed_track_ids:
            self._revealed_track_ids.add(target_track_id)

    # -- partial-constraint binding -------------------------------------------

    def choose_partial_binding(
        self,
        *,
        timestamp_ns: int,
        evidence: dict[str, float],
        constraints: list[PartialConstraint],
        tracks: Iterable[TrackState],
    ) -> PartialBindingDecision | None:
        top_k = self._binding_config.species_top_k
        detection_top = [
            label for label, _ in sorted(evidence.items(), key=lambda item: item[1], reverse=True)[:top_k]
        ]
        candidates = [
            PartialBindingCandidate(
                track_id=track.id,
                position_m=np.asarray(track.position_m, dtype=np.float64),
                last_seen_ns=track.last_seen_ns,
                top_species_labels=self.track_top_species(track, top_k),
            )
            for track in tracks
            if track.status != TrackStatus.DROPPED.value
        ]
        decision = choose_track_for_partial_constraints(
            timestamp_ns=timestamp_ns,
            detection_top_species_labels=detection_top,
            constraints=constraints,
            candidates=candidates,
            config=self._binding_config,
        )
        if decision is None:
            self.partial_bindings_unbound += 1
        else:
            self.partial_bindings_accepted += 1
        return decision
