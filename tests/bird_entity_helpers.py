"""Shared builders for the Field Mode W3-entity tests."""

from __future__ import annotations

from dataclasses import replace

from minimappr.config import Settings
from minimappr.core.bird_entity_registry import BirdEntityObservation, BirdEntityRegistry
from minimappr.core.entity_reporting_config import EntityReportingConfig
from minimappr.models import TrackState

NANOSECONDS_PER_SECOND = 1_000_000_000
START_NS = 1_700_000_000_000_000_000
ROBIN = "american robin"
TANAGER = "scarlet tanager"
NODE_POSITIONS_M = {
    "node-a": (0.0, 0.0, 0.0),
    "node-b": (20.0, 0.0, 0.0),
    "node-far": (400.0, 0.0, 0.0),
}
ROBIN_POSITION_M = (6.0, 12.0, 2.0)
TIGHT_COVARIANCE_M2 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
LOOSE_COVARIANCE_M2 = [[900.0, 0.0, 0.0], [0.0, 900.0, 0.0], [0.0, 0.0, 1.0]]


def seconds(value: float) -> int:
    return START_NS + int(value * NANOSECONDS_PER_SECOND)


def entity_config(**overrides) -> EntityReportingConfig:
    config = Settings(entity_reporting_enabled=True).entity_reporting_config()
    return replace(config, **overrides)


def sequential_registry(**overrides) -> BirdEntityRegistry:
    counter = iter(range(1, 10_000))
    return BirdEntityRegistry(entity_config(**overrides), entity_id_factory=lambda: f"ent-{next(counter)}")


def observation(
    *,
    at_s: float,
    label: str = ROBIN,
    confidence: float = 0.8,
    mode: str = "localized",
    node_id: str = "node-a",
    position_m=ROBIN_POSITION_M,
    covariance_m2=TIGHT_COVARIANCE_M2,
    track: TrackState | None = None,
    detection_id: str | None = None,
) -> BirdEntityObservation:
    return BirdEntityObservation(
        event_time_ns=seconds(at_s),
        label=label,
        confidence=confidence,
        label_category="wildlife",
        spatial_display_mode=mode,
        reporting_node_id=node_id,
        reporting_node_position_m=NODE_POSITIONS_M[node_id],
        detection_position_m=None if mode == "node_only" else position_m,
        detection_covariance_m2=None if mode == "node_only" else covariance_m2,
        detection_id=detection_id,
        track=track,
    )


def track_state(
    track_id: str = "trk-1",
    *,
    position_m=ROBIN_POSITION_M,
    covariance_m2=TIGHT_COVARIANCE_M2,
    label: str = ROBIN,
    status: str = "confirmed",
    belief: dict[str, float] | None = None,
) -> TrackState:
    top_label, top_probability = (max(belief.items(), key=lambda item: item[1]) if belief else (None, None))
    return TrackState(
        id=track_id,
        first_seen_ns=START_NS,
        last_seen_ns=START_NS,
        position_m=position_m,
        position_covariance_m2=covariance_m2,
        label=label,
        label_category="wildlife",
        confidence=0.8,
        status=status,
        species_belief=belief,
        species_top_label=top_label,
        species_top_probability=top_probability,
    )
