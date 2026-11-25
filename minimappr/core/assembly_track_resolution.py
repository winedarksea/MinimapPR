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

With the P5 observation pipeline (``position_observations``), a localized
solve's position has already reached the tracker before classification, so a
classified detection never moves a track a second time: the pipeline attaches
the label to the track that solve updated (``track_for_classified_solve``).
The same holds for an absorbed entity bout whose solve was observed: it
attaches its label instead of nudging the track through partial binding, and
for a classification reporting suppresses
(``label_observed_solve_track_without_report``), so a hidden track is still
revealed and a held fix still resolved by its label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from minimappr.classifiers.birdnet_two_threshold import detection_is_below_display_threshold
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.position_observation import TRACKER_LOCALIZABLE_CAPABILITY_TIERS
from minimappr.core.track_partial_constraint_evidence import build_partial_constraints
from minimappr.models import LabelId, TrackState

if TYPE_CHECKING:
    from minimappr.core.position_observation_pipeline import PositionObservationPipeline

# An observed solve passed the tracker-eligibility check (a localized fix) when it was observed.
_OBSERVED_SOLVE_SPATIAL_DISPLAY_MODE = "localized"
# Written into classification features by the onset stage (Field Mode W3-attribution).
NODE_BAND_SNR_FEATURE_KEY = "onset_cluster_node_band_snr_db"
AUDIBLE_SNR_FEATURE_KEY = "onset_cluster_audible_snr_db"


@dataclass(slots=True)
class TrackResolution:
    track: TrackState | None
    hidden_track_id: str | None = None
    below_display_threshold: bool = False
    partially_bound: bool = False

    @property
    def persists_detection(self) -> bool:
        """Whether the detection row may be stored.

        A sub-display result that no visible track surfaces is never stored: a
        row would leak it through the detections API, and as the canonical
        (node, label, window) row it would make the reporting policy suppress a
        later display-level detection of the same species in that window.
        """
        return not (self.below_display_threshold and self.track is None)


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
    solve_id: str | None = None,
    position_observations: "PositionObservationPipeline | None" = None,
) -> TrackResolution:
    field_mode = getattr(tracker, "field_mode", None)
    below_display = detection_is_below_display_threshold(classification_features)
    resolution = TrackResolution(track=None, below_display_threshold=below_display)
    classified_update_kwargs = _classified_update_kwargs(
        field_mode=field_mode,
        event_time_ns=event_time_ns,
        localization_position_m=localization_position_m,
        localization_position_covariance_m2=localization_position_covariance_m2,
        classification_label=classification_label,
        classification_confidence=classification_confidence,
        classification_scores=classification_scores,
        classification_features=classification_features,
        label_category=label_category,
        iff_category=iff_category,
        label_id=label_id,
        selected_sensor_ids=selected_sensor_ids,
        capability_tier=capability_tier,
        source_node_id=source_node_id,
    )
    species_evidence = classified_update_kwargs.get("species_evidence")
    if suppressed_by_zone and position_observations is not None and position_observations.solve_fix_is_held(solve_id):
        # Zone suppression hides the report, not the source: a held fix is
        # still resolved by the classifier's label (hold rule).
        await position_observations.track_for_classified_solve(
            solve_id=solve_id, classified_update_kwargs=classified_update_kwargs
        )
    # Nothing could ever lift a track over the display threshold without a belief.
    if suppressed_by_zone or _finds_nothing_trackable(field_mode, label_category, below_display):
        if position_observations is not None:
            # The classification found nothing to track here, so a provisional
            # track that only this solve's fix started must not linger and
            # capture a later source.
            await position_observations.classification_found_nothing_trackable(solve_id)
        return resolution

    track: TrackState | None = None
    if spatial_display_mode == "localized" and capability_tier in TRACKER_LOCALIZABLE_CAPABILITY_TIERS:
        if position_observations is not None:
            track = await position_observations.track_for_classified_solve(
                solve_id=solve_id, classified_update_kwargs=classified_update_kwargs
            )
        else:
            track = await tracker.update(**classified_update_kwargs)
    elif position_observations is not None and position_observations.solve_position_reached_tracker(solve_id):
        # An absorbed entity bout of an observed solve: its position is already on a track.
        track = await position_observations.track_for_classified_solve(
            solve_id=solve_id, classified_update_kwargs=classified_update_kwargs
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
            # Field Mode W3-attribution: per-node levels from the onset stage, when present.
            measured_node_band_snr_db=classification_features.get(NODE_BAND_SNR_FEATURE_KEY),
            measured_level_audible_snr_db=classification_features.get(AUDIBLE_SNR_FEATURE_KEY),
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


def _finds_nothing_trackable(field_mode: Any, label_category: str, below_display: bool) -> bool:
    belief_applies = field_mode is not None and field_mode.belief_applies(label_category)
    return below_display and not belief_applies


def _classified_update_kwargs(
    *,
    field_mode: Any,
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
    capability_tier: str,
    source_node_id: str | None,
) -> dict[str, Any]:
    """The ``TrackManager.update`` keywords of one classified solve."""
    classified_update_kwargs: dict[str, Any] = {
        "timestamp_ns": event_time_ns,
        "position_m": localization_position_m,
        "measurement_covariance_m2": localization_position_covariance_m2,
        "label": classification_label,
        "label_category": label_category,
        "iff_category": iff_category,
        "label_id": label_id,
        "confidence": classification_confidence,
        "sensor_count": len(selected_sensor_ids),
        "capability_tier": capability_tier,
        "source_node_id": source_node_id,
        "classifier_scores": classification_scores,
    }
    species_evidence = None
    if field_mode is not None:
        species_evidence = field_mode.detection_evidence(
            label=classification_label,
            confidence=classification_confidence,
            classifier_scores=classification_scores,
            species_evidence=None,
            winner_member=classification_features.get("winner_member"),
        )
    if species_evidence is not None:
        classified_update_kwargs["species_evidence"] = species_evidence
        classified_update_kwargs["below_display_threshold"] = detection_is_below_display_threshold(
            classification_features
        )
    return classified_update_kwargs


async def label_observed_solve_track_without_report(
    *,
    tracker: Any,
    registry: NodeRegistry,
    solve_id: str | None,
    position_observations: "PositionObservationPipeline | None",
    **resolve_kwargs: Any,
) -> None:
    """A classification reporting will not surface still resolves its observed
    solve: labels (and reveals) the track the position moved, resolves a held
    fix, or makes the label-aware update of a fix that never reached a track.
    Solves the pipeline never observed keep the pre-P5 behaviour: no tracker work."""
    if position_observations is None or not position_observations.knows_solve(solve_id):
        return
    await resolve_detection_track(
        tracker=tracker,
        registry=registry,
        spatial_display_mode=_OBSERVED_SOLVE_SPATIAL_DISPLAY_MODE,
        suppressed_by_zone=False,
        solve_id=solve_id,
        position_observations=position_observations,
        **resolve_kwargs,
    )
