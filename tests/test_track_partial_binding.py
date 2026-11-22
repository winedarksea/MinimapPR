"""Partial-constraint binding of unpositioned detections (Field Mode W2-tracker, S3)."""

from __future__ import annotations

import math

import numpy as np

from minimappr.config import Settings
from minimappr.core.assembly_track_resolution import resolve_detection_track
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.track_partial_binding import (
    BearingConstraint,
    HyperbolaConstraint,
    PartialBindingCandidate,
    PartialBindingConfig,
    SnrRankingConstraint,
    choose_track_for_partial_constraints,
)
from minimappr.core.tracking import TrackManager
from minimappr.models import NodeSpec, NodeType

T0 = 1_700_000_000_000_000_000
SECOND = 1_000_000_000
NODE_POSITIONS = {"node-a": (0.0, 0.0, 0.0), "node-b": (20.0, 0.0, 0.0), "node-c": (10.0, 18.0, 0.0)}


def _bearing_from(origin, target, sigma_deg=10.0) -> BearingConstraint:
    origin = np.asarray(origin, dtype=float)
    offset = np.asarray(target, dtype=float) - origin
    return BearingConstraint(origin_m=origin, unit_direction=offset / np.linalg.norm(offset),
                             sigma_rad=math.radians(sigma_deg))


def _candidate(track_id, position, labels, last_seen_ns=T0) -> PartialBindingCandidate:
    return PartialBindingCandidate(track_id=track_id, position_m=np.asarray(position, float),
                                   last_seen_ns=last_seen_ns, top_species_labels=labels)


def test_bearing_residual_and_projection():
    constraint = _bearing_from((0, 0, 0), (10, 0, 0))
    assert constraint.normalized_residual(np.array([20.0, 0.0, 0.0])) == 0.0
    off_axis = constraint.normalized_residual(np.array([10.0, 10.0 * math.tan(math.radians(20)), 0.0]))
    assert off_axis == np.float64(off_axis) and abs(off_axis - 2.0) < 1e-6
    projected = constraint.corrected_position(np.array([10.0, 3.0, 0.0]))
    np.testing.assert_allclose(projected, [10.0, 0.0, 0.0])


def test_hyperbola_residual_zero_on_true_source():
    source = np.array([6.0, 12.0, 2.0])
    a, b = np.array(NODE_POSITIONS["node-b"]), np.array(NODE_POSITIONS["node-a"])
    range_difference = float(np.linalg.norm(source - a) - np.linalg.norm(source - b))
    constraint = HyperbolaConstraint(sensor_m=a, reference_sensor_m=b,
                                     range_difference_m=range_difference, sigma_m=2.0)
    assert constraint.normalized_residual(source) < 1e-9
    corrected = constraint.corrected_position(source + np.array([3.0, 0.0, 0.0]))
    assert constraint.normalized_residual(corrected) < constraint.normalized_residual(source + [3.0, 0, 0])


def test_snr_ranking_is_loose_but_rejects_wrong_side():
    constraint = SnrRankingConstraint(
        reporting_node_m=np.array(NODE_POSITIONS["node-a"]),
        other_node_positions_m=(np.array(NODE_POSITIONS["node-b"]), np.array(NODE_POSITIONS["node-c"])),
        level_margin_db=7.0,
    )
    assert constraint.normalized_residual(np.array([6.0, 12.0, 2.0])) <= 1.0
    # Bird beside node-b: node-a 6x farther (~15.6 dB) -> > 2 margins.
    assert constraint.normalized_residual(np.array([19.0, 1.0, 0.0])) > 2.0


def test_binds_to_track_on_bearing_with_matching_species():
    decision = choose_track_for_partial_constraints(
        timestamp_ns=T0 + 5 * SECOND,
        detection_top_species_labels=["robin", "wren"],
        constraints=[_bearing_from((0, 0, 0), (6, 12, 2))],
        candidates=[_candidate("t-robin", (6.5, 12.0, 2.0), ["robin"]),
                    _candidate("t-tanager", (18, 9, 3), ["tanager"])],
        config=PartialBindingConfig(),
    )
    assert decision is not None and decision.track_id == "t-robin"
    assert decision.corrected_position_m is not None


def test_refuses_species_mismatch_stale_track_and_ambiguity():
    bearing = [_bearing_from((0, 0, 0), (6, 12, 2))]
    config = PartialBindingConfig(window_seconds=25.0)
    assert choose_track_for_partial_constraints(
        timestamp_ns=T0, detection_top_species_labels=["wren"], constraints=bearing,
        candidates=[_candidate("t", (6, 12, 2), ["robin"])], config=config) is None
    assert choose_track_for_partial_constraints(
        timestamp_ns=T0 + 60 * SECOND, detection_top_species_labels=["robin"], constraints=bearing,
        candidates=[_candidate("t", (6, 12, 2), ["robin"])], config=config) is None
    # Two same-species tracks both on the ray: unattributable, stay unbound.
    assert choose_track_for_partial_constraints(
        timestamp_ns=T0, detection_top_species_labels=["robin"], constraints=bearing,
        candidates=[_candidate("near", (3, 6, 1), ["robin"]), _candidate("far", (9, 18, 3), ["robin"])],
        config=config) is None
    # No constraints at all: never bind on species alone.
    assert choose_track_for_partial_constraints(
        timestamp_ns=T0, detection_top_species_labels=["robin"], constraints=[],
        candidates=[_candidate("t", (6, 12, 2), ["robin"])], config=config) is None


async def _registry() -> NodeRegistry:
    registry = NodeRegistry()
    for node_id, position in NODE_POSITIONS.items():
        await registry.upsert(
            NodeSpec(id=node_id, node_type=NodeType.POINT, position_m=position,
                     sensor_offsets_m=[(0.0, 0.0, 0.0)], capabilities=["audio"]),
            last_seen_ns=T0,
        )
    return registry


async def _sensor_for(registry: NodeRegistry, node_id: str) -> str:
    return next(sensor for sensor, node in (await registry.sensor_node_ids()).items() if node == node_id)


async def _resolve_omni(manager, registry, *, label, confidence, timestamp_ns, node_id="node-a"):
    sensor = await _sensor_for(registry, node_id)
    return await resolve_detection_track(
        tracker=manager, registry=registry, spatial_display_mode="node_only", suppressed_by_zone=False,
        capability_tier="classification_only", event_time_ns=timestamp_ns,
        localization_position_m=NODE_POSITIONS[node_id], localization_position_covariance_m2=None,
        classification_label=label, classification_confidence=confidence,
        classification_scores={label: confidence}, classification_features={},
        label_category="wildlife", iff_category="unknown", label_id=None, selected_sensor_ids=[sensor],
        source_node_id=node_id, reference_sensor=sensor, tdoa_s={}, environment={},
    )


async def test_omni_detection_binds_by_snr_ranking_and_updates_belief():
    """Bout 1: localized, weak ID. Bout 2: omni only, decent ID -> one track, classified."""
    registry = await _registry()
    manager = TrackManager(Settings(tracking_partial_constraint_binding_enabled=True,
                                    tracking_species_log_odds_enabled=True))
    localized = await manager.update(timestamp_ns=T0, position_m=(6.0, 12.0, 2.0), label="robin",
                                     label_category="wildlife", confidence=0.35,
                                     below_display_threshold=True)
    assert manager.is_track_hidden(localized.id)
    resolution = await _resolve_omni(manager, registry, label="robin", confidence=0.6,
                                     timestamp_ns=T0 + 6 * SECOND)
    assert resolution.partially_bound
    assert resolution.track is not None and resolution.track.id == localized.id
    assert resolution.track.species_top_probability > 0.5
    assert not manager.is_track_hidden(localized.id)


async def test_omni_from_far_node_does_not_bind():
    registry = await _registry()
    manager = TrackManager(Settings(tracking_partial_constraint_binding_enabled=True))
    await manager.update(timestamp_ns=T0, position_m=(1.0, 1.0, 1.0), label="robin",
                         label_category="wildlife", confidence=0.8)
    resolution = await _resolve_omni(manager, registry, label="robin", confidence=0.8,
                                     timestamp_ns=T0 + 3 * SECOND, node_id="node-b")
    assert resolution.track is None and not resolution.partially_bound


async def test_flag_off_omni_never_touches_tracker():
    registry = await _registry()
    manager = TrackManager(Settings())
    track = await manager.update(timestamp_ns=T0, position_m=(6.0, 12.0, 2.0), label="robin",
                                 label_category="wildlife", confidence=0.8)
    resolution = await _resolve_omni(manager, registry, label="robin", confidence=0.8,
                                     timestamp_ns=T0 + 3 * SECOND)
    assert resolution.track is None
    assert (await manager.snapshot())[0].update_count == track.update_count == 1
