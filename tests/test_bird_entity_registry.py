"""Field Mode W3-entity: bird-entity state machine and sliding absorption window."""

from __future__ import annotations

import pytest

from minimappr.core.bird_entity_model import (
    bearing_estimate_toward,
    compass_bearing_deg,
    horizontal_major_sigma_m,
    uncertainty_ellipse,
)
from tests.bird_entity_helpers import (
    LOOSE_COVARIANCE_M2,
    NODE_POSITIONS_M,
    ROBIN,
    TANAGER,
    observation,
    seconds,
    sequential_registry,
    track_state,
)


def test_heard_nearby_then_direction_known_then_pinpointed_is_one_entity() -> None:
    registry = sequential_registry()
    created = registry.observe(observation(at_s=0, mode="node_only"))
    assert created.kind == "created"
    assert created.entity.state == "heard_nearby"
    assert created.entity.position_m is None

    # Bearing-only solve from the same node: the same bird gains a direction.
    bearing = registry.observe(observation(at_s=5, mode="bearing_only", position_m=(30.0, 60.0, 0.0)))
    assert bearing.kind == "state_changed"
    assert bearing.previous_state == "heard_nearby"
    assert bearing.entity.entity_id == created.entity.entity_id
    assert bearing.entity.state == "direction_known"
    assert bearing.entity.bearing is not None
    assert bearing.entity.bearing.range_max_m is None  # range unobservable -> default fan
    assert bearing.entity.bearing.bearing_deg == pytest.approx(compass_bearing_deg((0, 0, 0), (30, 60, 0)))

    pinpointed = registry.observe(observation(at_s=10, track=track_state()))
    assert pinpointed.kind == "state_changed"
    assert pinpointed.entity.entity_id == created.entity.entity_id
    assert pinpointed.entity.state == "pinpointed"
    assert pinpointed.entity.track_id == "trk-1"
    assert len(registry.entities()) == 1


def test_failed_localization_bout_never_demotes() -> None:
    registry = sequential_registry()
    entity = registry.observe(observation(at_s=0, track=track_state())).entity
    change = registry.observe(observation(at_s=4, mode="node_only"), preferred_entity_id=entity.entity_id)
    assert change.kind == "updated"
    assert change.entity.state == "pinpointed"
    assert change.entity.position_m is not None


def test_unpositioned_bout_absorbs_into_mapped_entity_within_window() -> None:
    registry = sequential_registry(absorption_window_seconds=60.0)
    entity = registry.observe(observation(at_s=0, track=track_state())).entity
    match = registry.find_absorbing_entity(
        label=ROBIN, event_time_ns=seconds(45), spatial_display_mode="node_only",
        reporting_node_id="node-b", reporting_node_position_m=NODE_POSITIONS_M["node-b"],
        detection_position_m=None,
    )
    assert match is entity
    change = registry.absorb_unlocalized(
        entity.entity_id, event_time_ns=seconds(45), label=ROBIN, confidence=0.6, reporting_node_id="node-b"
    )
    assert change is not None and change.kind == "absorbed"
    assert entity.absorbed_detection_count == 1
    assert entity.last_heard_ns == seconds(45) and entity.last_pulse_ns == seconds(45)
    assert entity.reporting_node_ids == ["node-a", "node-b"]


def test_window_slides_with_the_entity_not_fixed_buckets() -> None:
    registry = sequential_registry(absorption_window_seconds=60.0)
    entity = registry.observe(observation(at_s=0, track=track_state())).entity
    # Every 50 s a bout keeps the entity alive; a 30 s bucket scheme would split these.
    for at_s in (50, 100, 150):
        match = registry.find_absorbing_entity(
            label=ROBIN, event_time_ns=seconds(at_s), spatial_display_mode="node_only",
            reporting_node_id="node-a", reporting_node_position_m=NODE_POSITIONS_M["node-a"],
            detection_position_m=None,
        )
        assert match is entity
        registry.absorb_unlocalized(entity.entity_id, event_time_ns=seconds(at_s), label=ROBIN,
                                    confidence=0.6, reporting_node_id="node-a")
    # Silent past the window: a new bout is a new entity.
    assert registry.find_absorbing_entity(
        label=ROBIN, event_time_ns=seconds(150 + 61), spatial_display_mode="node_only",
        reporting_node_id="node-a", reporting_node_position_m=NODE_POSITIONS_M["node-a"],
        detection_position_m=None,
    ) is None


def test_absorption_requires_species_and_spatial_consistency() -> None:
    registry = sequential_registry()
    registry.observe(observation(at_s=0, track=track_state()))
    common = dict(event_time_ns=seconds(5), detection_position_m=None, spatial_display_mode="node_only")
    assert registry.find_absorbing_entity(
        label=TANAGER, reporting_node_id="node-a", reporting_node_position_m=NODE_POSITIONS_M["node-a"], **common
    ) is None
    # A node 400 m away cannot be hearing the bird mapped next to node-a.
    assert registry.find_absorbing_entity(
        label=ROBIN, reporting_node_id="node-far", reporting_node_position_m=NODE_POSITIONS_M["node-far"], **common
    ) is None
    # A fix 25 m from the pinpointed bird is another bird.
    assert registry.find_absorbing_entity(
        label=ROBIN, event_time_ns=seconds(5), spatial_display_mode="localized", reporting_node_id="node-a",
        reporting_node_position_m=NODE_POSITIONS_M["node-a"], detection_position_m=(31.0, 12.0, 2.0),
    ) is None


def test_track_binding_wins_and_heard_nearby_promotes_by_node_distance() -> None:
    registry = sequential_registry()
    heard = registry.observe(observation(at_s=0, mode="node_only")).entity
    # A localized fix within node distance of the heard-nearby node promotes the same entity.
    promoted = registry.observe(observation(at_s=3, track=track_state("trk-9")))
    assert promoted.entity is heard and promoted.entity.state == "pinpointed"
    # Later, the tracker says trk-9 moved 40 m (relocation): still the same entity.
    moved = registry.observe(observation(at_s=20, position_m=(46.0, 12.0, 2.0),
                                         track=track_state("trk-9", position_m=(46.0, 12.0, 2.0))))
    assert moved.entity is heard


def test_large_uncertainty_fix_is_direction_known_with_range_band() -> None:
    registry = sequential_registry(pinpointed_max_horizontal_uncertainty_m=15.0)
    entity = registry.observe(observation(at_s=0, position_m=(0.0, 80.0, 0.0),
                                          covariance_m2=[[4.0, 0, 0], [0, 900.0, 0], [0, 0, 1.0]])).entity
    assert entity.state == "direction_known"
    assert entity.bearing.bearing_deg == pytest.approx(0.0)
    assert entity.bearing.range_min_m == pytest.approx(80.0 - 2 * 30.0)
    assert entity.bearing.range_max_m == pytest.approx(80.0 + 2 * 30.0)
    # Tangential sigma 2 m at 80 m: narrow fan, clamped to the configured minimum.
    assert entity.bearing.half_width_deg == pytest.approx(5.0)


def test_track_expiry_steps_back_one_state_and_entity_expires() -> None:
    registry = sequential_registry(expiry_seconds=300.0, spatial_state_hold_seconds=120.0)
    entity = registry.observe(observation(at_s=0, track=track_state())).entity
    # Track still visible: nothing changes.
    assert registry.maintain(seconds(10), {"trk-1": track_state()}) == []
    # Track gone: one step back (pinpointed -> direction_known), not straight to the tray.
    changes = registry.maintain(seconds(20), {})
    assert [(c.kind, c.previous_state, c.entity.state) for c in changes] == [
        ("state_changed", "pinpointed", "direction_known")
    ]
    assert entity.track_id is None and entity.bearing is not None
    # No spatial update for the hold: back to heard nearby (geometry dropped).
    changes = registry.maintain(seconds(20 + 121), {})
    assert [c.entity.state for c in changes] == ["heard_nearby"]
    assert entity.position_m is None and entity.bearing is None
    changes = registry.maintain(seconds(301), {})
    assert [c.kind for c in changes] == ["expired"]
    assert registry.entities() == []


def test_uncertainty_growth_on_track_demotes_pinpointed() -> None:
    registry = sequential_registry(pinpointed_max_horizontal_uncertainty_m=15.0)
    entity = registry.observe(observation(at_s=0, track=track_state())).entity
    changes = registry.maintain(seconds(5), {"trk-1": track_state(covariance_m2=LOOSE_COVARIANCE_M2)})
    assert [c.entity.state for c in changes] == ["direction_known"]
    assert entity.horizontal_uncertainty_m == pytest.approx(30.0)


def test_species_belief_follows_track_log_odds() -> None:
    registry = sequential_registry()
    belief = {ROBIN: 0.9, TANAGER: 0.05}
    entity = registry.observe(observation(at_s=0, label=TANAGER, confidence=0.4,
                                          track=track_state(belief=belief))).entity
    assert entity.species_top_label == ROBIN
    assert entity.species_top_probability == pytest.approx(0.9)
    # Field Mode S-fixes tracking: a prior-level secondary score (0.05) is not
    # belief, so a tanager detection no longer absorbs into the robin entity...
    tanager_query = dict(
        label=TANAGER, event_time_ns=seconds(3), spatial_display_mode="node_only", reporting_node_id="node-a",
        reporting_node_position_m=NODE_POSITIONS_M["node-a"], detection_position_m=None,
    )
    assert registry.find_absorbing_entity(**tanager_query) is None
    # ...while a zero floor restores the W3 "anywhere in the belief" rule.
    permissive = sequential_registry(absorption_min_species_belief=0.0)
    permissive_entity = permissive.observe(observation(at_s=0, label=TANAGER, confidence=0.4,
                                                       track=track_state(belief=belief))).entity
    assert permissive.find_absorbing_entity(**tanager_query) is permissive_entity


def test_max_entities_evicts_least_recently_heard() -> None:
    registry = sequential_registry(max_tracked_entities=2)
    for index, label in enumerate(("a", "b", "c")):
        registry.observe(observation(at_s=index, label=label, mode="node_only"))
    assert {entity.species_top_label for entity in registry.entities()} == {"b", "c"}


def test_geometry_helpers() -> None:
    assert horizontal_major_sigma_m(LOOSE_COVARIANCE_M2) == pytest.approx(30.0)
    assert horizontal_major_sigma_m(None) is None
    assert compass_bearing_deg((0, 0, 0), (10, 0, 0)) == pytest.approx(90.0)
    semi_major, semi_minor, orientation = uncertainty_ellipse([[9.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], 2.0)
    assert (semi_major, semi_minor) == (pytest.approx(6.0), pytest.approx(2.0))
    assert orientation == pytest.approx(90.0)
    fan = bearing_estimate_toward(
        origin_node_id="n", origin_position_m=(0, 0, 0), target_position_m=(0, 50, 0), covariance_m2=None,
        range_observable=False, default_half_width_deg=15.0, min_half_width_deg=5.0,
        max_half_width_deg=45.0, sigma_multiplier=2.0,
    )
    assert fan.half_width_deg == 15.0 and fan.range_min_m is None
