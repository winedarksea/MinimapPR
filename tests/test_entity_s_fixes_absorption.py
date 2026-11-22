"""Field Mode S-fixes tracking: covariance-aware entity absorption, bearing evidence,
concurrency, and the fused canonical-row position."""

from __future__ import annotations

from pathlib import Path

from minimappr.core.entity_canonical_row_position import (
    RAW_FIX_POSITION_FEATURE_KEY,
    persisted_row_position,
)
from minimappr.core.entity_reporting_policy import EntityReportingPolicy
from minimappr.core.range_projection import RANGE_BEARING_PROJECTED
from minimappr.models import DetectionEvent, NodeSpec, NodeType
from minimappr.storage.db import Storage
from tests.bird_entity_helpers import (
    NODE_POSITIONS_M,
    ROBIN,
    TIGHT_COVARIANCE_M2,
    observation,
    seconds,
    sequential_registry,
    track_state,
)

CHICKADEE = "black-capped chickadee"
# Two perches 10 m apart with ~1 m fixes: two birds, inside the old fixed 12 m radius.
FIRST_PERCH_M = (5.0, 5.0, 4.0)
SECOND_PERCH_M = (15.0, 4.0, 3.0)
# A far bird pinpointed ~90 m out, then a fix on the same bearing at ~440 m whose
# covariance runs hundreds of metres along the ray (range unobservable).
FAR_BIRD_M = (80.0, 46.0, 10.0)
FAR_FIX_ON_BEARING_M = (391.0, 210.0, 10.0)
FAR_FIX_RADIAL_COVARIANCE_M2 = [[141757.0, 75783.0, 0.0], [75783.0, 40558.0, 0.0], [0.0, 0.0, 100.0]]
LOOSE_TRACK_COVARIANCE_M2 = [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]


def _query(position_m, *, at_s: float, covariance_m2=TIGHT_COVARIANCE_M2, label: str = CHICKADEE) -> dict:
    return dict(
        label=label, event_time_ns=seconds(at_s), spatial_display_mode="localized", reporting_node_id="node-a",
        reporting_node_position_m=NODE_POSITIONS_M["node-a"], detection_position_m=position_m,
        detection_covariance_m2=covariance_m2,
    )


def test_flock_perches_ten_metres_apart_are_two_entities() -> None:
    registry = sequential_registry()
    registry.observe(observation(at_s=0, label=CHICKADEE, position_m=FIRST_PERCH_M))
    assert registry.find_absorbing_entity(**_query(SECOND_PERCH_M, at_s=1.2)) is None
    # Jitter on the same perch still absorbs.
    assert registry.find_absorbing_entity(**_query((5.8, 5.5, 4.0), at_s=1.2)) is not None


def test_covariance_gate_off_restores_fixed_radius_merge() -> None:
    registry = sequential_registry(absorption_covariance_gate_enabled=False)
    entity = registry.observe(observation(at_s=0, label=CHICKADEE, position_m=FIRST_PERCH_M)).entity
    assert registry.find_absorbing_entity(**_query(SECOND_PERCH_M, at_s=1.2)) is entity


def test_far_fix_on_bearing_absorbs_into_the_pinpointed_bird() -> None:
    registry = sequential_registry()
    entity = registry.observe(observation(at_s=0, label=CHICKADEE, position_m=FAR_BIRD_M)).entity
    assert entity.state == "pinpointed"
    assert registry.find_absorbing_entity(
        **_query(FAR_FIX_ON_BEARING_M, at_s=5, covariance_m2=FAR_FIX_RADIAL_COVARIANCE_M2)
    ) is entity
    # Off the bearing (same range, other side) it is another bird.
    assert registry.find_absorbing_entity(
        **_query((-391.0, 210.0, 10.0), at_s=5, covariance_m2=FAR_FIX_RADIAL_COVARIANCE_M2)
    ) is None


def test_concurrent_inconsistent_fixes_are_two_birds_even_on_a_loose_track() -> None:
    registry = sequential_registry()
    loose_track = track_state(position_m=FIRST_PERCH_M, covariance_m2=LOOSE_TRACK_COVARIANCE_M2)
    entity = registry.observe(observation(at_s=0, label=CHICKADEE, position_m=FIRST_PERCH_M, track=loose_track)).entity
    # The loose track would take a fix 10 m away; the entity's own tight fix 0.5 s ago says no.
    assert registry.find_absorbing_entity(**_query(SECOND_PERCH_M, at_s=0.5)) is None
    # After the concurrency window the same fix is a plausible hop of the loose track.
    assert registry.find_absorbing_entity(**_query(SECOND_PERCH_M, at_s=5)) is entity


def test_shared_track_does_not_drag_a_concurrent_bird_into_the_entity() -> None:
    registry = sequential_registry()
    shared = track_state(position_m=FIRST_PERCH_M, covariance_m2=LOOSE_TRACK_COVARIANCE_M2)
    first = registry.observe(observation(at_s=0, label=CHICKADEE, position_m=FIRST_PERCH_M, track=shared)).entity
    second = registry.observe(
        observation(at_s=0.5, label=CHICKADEE, position_m=SECOND_PERCH_M, track=shared)
    ).entity
    assert second.entity_id != first.entity_id
    assert first.track_id == shared.id and second.track_id is None  # binding not stolen
    assert second.position_m == SECOND_PERCH_M


def _row(det_id: str) -> DetectionEvent:
    return DetectionEvent(
        id=det_id, timestamp_ns=seconds(0), source_node_id="node-a", reporting_modality="localized",
        spatial_display_mode="localized", position_m=FAR_BIRD_M, position_covariance_m2=TIGHT_COVARIANCE_M2,
        confidence=0.8, gdop=1.0, label=ROBIN, label_category="wildlife", label_confidence=0.8,
        reference_sensor="m0", report_window_start_ns=seconds(0), report_window_end_ns=seconds(60),
    )


async def test_policy_absorbs_bearing_evidence_without_upgrading_the_row(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "policy.db")
    await storage.initialize()
    await storage.upsert_node(
        NodeSpec(id="node-a", node_type=NodeType.SIRITH_TETRA, position_m=(0.0, 0.0, 0.0)), last_seen_ns=seconds(0)
    )
    try:
        registry = sequential_registry()
        policy = EntityReportingPolicy(storage=storage, registry=registry)
        await storage.insert_detection(_row("det-canon"), snippet_path=None, snippet_expires_ns=None)
        entity = registry.observe(observation(at_s=0, position_m=FAR_BIRD_M, detection_id="det-canon")).entity
        common = dict(source_node_id="node-a", reporting_node_position_m=NODE_POSITIONS_M["node-a"], label=ROBIN,
                      branch_details={"label": ROBIN, "confidence": 0.9}, below_display_threshold=False,
                      reporting_modality="localized", spatial_display_mode="localized")
        far, change = await policy.decide(
            event_time_ns=seconds(5), localization_position_m=FAR_FIX_ON_BEARING_M,
            localization_position_covariance_m2=FAR_FIX_RADIAL_COVARIANCE_M2, **common,
        )
        assert far.action == "enrich_existing" and far.entity_id == entity.entity_id
        assert change.kind == "absorbed" and policy.absorbed_bearing_evidence_count == 1
        assert entity.position_m == FAR_BIRD_M  # not moved
        # A bearing-projected range is bearing evidence even with a tight-looking covariance.
        projected, _ = await policy.decide(
            event_time_ns=seconds(6), localization_position_m=(81.0, 46.5, 10.0),
            localization_position_covariance_m2=TIGHT_COVARIANCE_M2,
            localization_range_projection_mode=RANGE_BEARING_PROJECTED, **common,
        )
        assert projected.action == "enrich_existing"
        # A good fix on the bird still upgrades the canonical row.
        good, _ = await policy.decide(
            event_time_ns=seconds(7), localization_position_m=(80.5, 46.2, 10.0),
            localization_position_covariance_m2=TIGHT_COVARIANCE_M2, **common,
        )
        assert good.action == "upgrade_existing"
    finally:
        await storage.close()


def test_canonical_row_carries_the_fused_track_position() -> None:
    track = track_state(position_m=(6.1, 12.1, 2.0))
    raw = (9.0, 15.0, 2.0)
    fused = persisted_row_position(
        use_fused_track_position=True, track=track, raw_position_m=raw, raw_covariance_m2=TIGHT_COVARIANCE_M2
    )
    assert fused.position_m == (6.1, 12.1, 2.0)
    assert fused.covariance_m2 == track.position_covariance_m2
    assert fused.raw_fix_features[RAW_FIX_POSITION_FEATURE_KEY] == list(raw)
    for use_fused, maybe_track in ((False, track), (True, None)):
        legacy = persisted_row_position(
            use_fused_track_position=use_fused, track=maybe_track, raw_position_m=raw,
            raw_covariance_m2=TIGHT_COVARIANCE_M2,
        )
        assert legacy.position_m == raw and legacy.raw_fix_features == {}
