"""Which track (if any) a detection feeds, and whether it may surface.

Extracted from ``DetectionAssembler.assemble``. With every Field Mode
W2-tracker flag off this is exactly the old behaviour: localized, unsuppressed,
localizable-tier detections update the tracker; nothing else touches it.

With flags on (all decided by the tracker's ``field_mode`` extensions and the
classifier's ``below_display_threshold`` feature):

- a detection without a full fix may bind to a recent track through partial
  constraints (``tracking_partial_constraint_binding_enabled``);
- a sub-display BirdNET result (``birdnet_two_threshold_enabled``) only feeds
  tracks whose category carries a species belief; it never starts a track whose
  belief cannot be tracked;
- a track whose species belief is below the display threshold is *hidden*: it
  is returned as ``hidden_track_id`` only, never persisted, broadcast or ruled on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minimappr.classifiers.birdnet_two_threshold import detection_is_below_display_threshold
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.track_partial_constraint_evidence import build_partial_constraints
from minimappr.models import LabelId, TrackState

_LOCALIZABLE_CAPABILITY_TIERS = frozenset({"2d", "full_3d"})


@dataclass(slots=True)
class TrackResolution:
    track: TrackState | None
    hidden_track_id: str | None = None
    below_display_threshold: bool = False
    partially_bound: bool = False


async def resolve_detection_track(
    *,
    tracker: Any,
    registry: NodeRegistry,
    spatial_display_mode: str,
    suppressed_by_zone: bool,
    capability_tier: str,
    event_time_ns: int,
    localization_position_m: tuple[float, float, float],
    localization_position_covariance_m2: list[list[float]] | None,
    classification_label: str,
    classification_confidence: float,
    classification_scores: dict[str, float],
    classification_features: dict[str, Any],
    label_category: str,
    iff_category: str,
    label_id: LabelId | None,
    selected_sensor_ids: list[str],
    source_node_id: str | None,
    reference_sensor: str,
    tdoa_s: dict[str, float],
    environment: dict[str, Any] | None,
) -> TrackResolution:
    field_mode = getattr(tracker, "field_mode", None)
    below_display = detection_is_below_display_threshold(classification_features)
    resolution = TrackResolution(track=None, below_display_threshold=below_display)
    if suppressed_by_zone:
        return resolution
    belief_applies = field_mode is not None and field_mode.belief_applies(label_category)
    if below_display and not belief_applies:
        # Nothing could ever lift this track over the display threshold.
        return resolution

    species_evidence = None
    if field_mode is not None:
        species_evidence = field_mode.detection_evidence(
            label=classification_label,
            confidence=classification_confidence,
            classifier_scores=classification_scores,
            species_evidence=None,
            winner_member=classification_features.get("winner_member"),
        )

    track: TrackState | None = None
    if spatial_display_mode == "localized" and capability_tier in _LOCALIZABLE_CAPABILITY_TIERS:
        update_kwargs: dict[str, Any] = {}
        if species_evidence is not None:
            update_kwargs["species_evidence"] = species_evidence
            update_kwargs["below_display_threshold"] = below_display
        track = await tracker.update(
            timestamp_ns=event_time_ns,
            position_m=localization_position_m,
            measurement_covariance_m2=localization_position_covariance_m2,
            label=classification_label,
            label_category=label_category,
            iff_category=iff_category,
            label_id=label_id,
            confidence=classification_confidence,
            sensor_count=len(selected_sensor_ids),
            capability_tier=capability_tier,
            source_node_id=source_node_id,
            classifier_scores=classification_scores,
            **update_kwargs,
        )
    elif field_mode is not None and field_mode.partial_binding_enabled:
        constraints = await build_partial_constraints(
            registry=registry,
            reference_sensor=reference_sensor,
            source_node_id=source_node_id,
            spatial_display_mode=spatial_display_mode,
            localization_position_m=localization_position_m,
            tdoa_s=tdoa_s,
            environment=environment,
            sigmas=field_mode.partial_constraint_sigmas,
        )
        track = await tracker.bind_partial_constraints(
            timestamp_ns=event_time_ns,
            label=classification_label,
            confidence=classification_confidence,
            label_category=label_category,
            constraints=constraints,
            species_evidence=species_evidence,
            classifier_scores=classification_scores,
            source_node_id=source_node_id,
            below_display_threshold=below_display,
        )
        resolution.partially_bound = track is not None

    if track is None:
        return resolution
    is_hidden = getattr(tracker, "is_track_hidden", None)
    if callable(is_hidden) and is_hidden(track.id):
        resolution.hidden_track_id = track.id
        return resolution
    resolution.track = track
    return resolution
