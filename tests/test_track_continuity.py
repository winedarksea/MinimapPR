"""Tests for the acoustic track-continuity overhaul.

Three mechanisms, all on by default:

  Phase 1 — per-category lifecycle windows + Kalman coast guards, so a source
            that only re-detects every minute or two keeps its identity.
  Phase 2 — class-aware association (category hard gate + classifier-score
            fingerprint tie-break).
  Phase 3 — dormant reacquisition: a dropped identity can be revived by a later
            detection at roughly the same place with a compatible class.

Several tests compare against "legacy" settings — the pre-overhaul behaviour
reconstructed by zeroing the new knobs — so the regressions being fixed stay
visible rather than merely asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.config import Settings
from minimappr.core.track_associators import NearestNeighborAssociator
from minimappr.core.track_filters import KalmanTrackFilter
from minimappr.core.tracking import TrackManager
from minimappr.interfaces import AssociationContext
from minimappr.models import TrackState, TrackStatus

T0 = 1_700_000_000_000_000_000
SECOND = 1_000_000_000


# Pre-overhaul behaviour: base 20 s lifecycle for every category, unguarded
# constant-velocity coasting, no dormant registry.
LEGACY_KWARGS = dict(
    track_stale_seconds_wildlife=0.0,
    track_stale_seconds_vehicle=0.0,
    track_stale_seconds_human=0.0,
    track_stale_seconds_security=0.0,
    kalman_process_noise_wildlife=0.0,
    kalman_process_noise_vehicle=0.0,
    kalman_process_noise_human=0.0,
    kalman_process_noise_security=0.0,
    kalman_max_coast_process_seconds=0.0,
    kalman_coast_velocity_half_life_seconds=0.0,
    association_category_gate_enabled=False,
    association_fingerprint_weight=0.0,
    dormant_reacquire_enabled=False,
)


async def _sing(manager: TrackManager, *, gap_s: float, count: int) -> list[str]:
    """A recurrent singer at a fixed spot with small localization jitter."""
    jitter = [0.0, 1.2, -0.8, 0.9, -1.1, 0.5]
    ids: list[str] = []
    for index in range(count):
        track = await manager.update(
            timestamp_ns=T0 + int(index * gap_s * SECOND),
            position_m=(12.0 + jitter[index % len(jitter)], 3.0, 1.0),
            label="Northern Cardinal",
            label_category="wildlife",
            confidence=0.7,
        )
        ids.append(track.id)
    return ids


# ---------------------------------------------------------------------------
# Phase 1 — fragmentation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recurrent_wildlife_source_keeps_one_track_id() -> None:
    """The headline regression: a bird re-detected every ~75 s stays one track.

    75 s is past the legacy 60 s drop threshold (20 s stale x 3), which is the
    half of a ~45 s singer's real inter-detection gaps that used to lose the
    track outright once trigger cooldown and reporting dedupe stretched them.
    """
    manager = TrackManager(Settings())
    ids = await _sing(manager, gap_s=75.0, count=6)

    assert len(set(ids)) == 1, f"track fragmented into {sorted(set(ids))}"
    tracks = await manager.snapshot()
    assert len(tracks) == 1
    assert tracks[0].status == TrackStatus.CONFIRMED.value
    assert tracks[0].update_count == 6


@pytest.mark.asyncio
async def test_recurrent_wildlife_source_fragmented_before_the_fix() -> None:
    """Same scenario under pre-overhaul settings: a new id at every re-detection."""
    manager = TrackManager(Settings(**LEGACY_KWARGS))
    ids = await _sing(manager, gap_s=75.0, count=6)
    assert len(set(ids)) == 6


@pytest.mark.asyncio
async def test_security_track_coasts_while_wildlife_stays_confirmed() -> None:
    """Impulses shouldn't persist; singers should. Same gap, opposite outcome."""
    manager = TrackManager(Settings())
    for category, label, position in (
        ("security", "Gunshot", (0.0, 0.0, 0.0)),
        ("wildlife", "Northern Cardinal", (100.0, 0.0, 0.0)),
    ):
        for index in range(2):
            await manager.update(
                timestamp_ns=T0 + index * SECOND,
                position_m=position,
                label=label,
                label_category=category,
                confidence=0.8,
            )

    # 40 s later: past the 20 s security window, well inside the 120 s wildlife one.
    tracks = {t.label_category: t for t in await manager.snapshot(now_ns=T0 + 40 * SECOND)}
    assert tracks["security"].status == TrackStatus.COASTING.value
    assert tracks["wildlife"].status == TrackStatus.CONFIRMED.value


@pytest.mark.asyncio
async def test_per_category_process_noise_reaches_the_filter() -> None:
    manager = TrackManager(Settings())
    wildlife = await manager.update(
        timestamp_ns=T0,
        position_m=(0.0, 0.0, 0.0),
        label="Northern Cardinal",
        label_category="wildlife",
        confidence=0.6,
    )
    assert manager._filter._states[wildlife.id].process_noise == pytest.approx(0.5)

    # An unclassified track starts on the base q; once a category is resolved
    # (unknown is a wildcard, so this associates) the category's q is applied.
    unclassified = await manager.update(
        timestamp_ns=T0,
        position_m=(500.0, 0.0, 0.0),
        label="unknown",
        confidence=0.4,
    )
    assert manager._filter._states[unclassified.id].process_noise == pytest.approx(2.0)
    resolved = await manager.update(
        timestamp_ns=T0 + SECOND,
        position_m=(500.0, 0.0, 0.0),
        label="Truck",
        label_category="vehicle",
        confidence=0.9,
    )
    assert resolved.id == unclassified.id
    assert manager._filter._states[unclassified.id].process_noise == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Phase 1 — Kalman coast guards
# ---------------------------------------------------------------------------

def _moving_filter(**kwargs) -> tuple[KalmanTrackFilter, TrackState]:
    filt = KalmanTrackFilter(
        process_noise=2.0,
        measurement_noise=1.5,
        initial_position_variance=4.0,
        initial_velocity_variance=16.0,
        **kwargs,
    )
    state = TrackState(
        id="trk-1",
        first_seen_ns=T0,
        last_seen_ns=T0,
        position_m=(0.0, 0.0, 0.0),
    )
    filt.initialize_track(state.id, (0.0, 0.0, 0.0))
    filt._states[state.id].mean[3:] = (5.0, 0.0, 0.0)  # 5 m/s along +x
    return filt, state


def test_long_coast_bounds_displacement_and_covariance() -> None:
    filt, state = _moving_filter()
    predicted = filt.predict(state, 60.0)

    # Displacement is bounded by (half_life / ln2) * |v| instead of dt * |v|.
    max_displacement = (10.0 / np.log(2.0)) * 5.0
    assert 0.0 < predicted.position_m[0] <= max_displacement
    assert predicted.position_m[0] < 60.0 * 5.0  # unguarded CV would go 300 m

    # Q growth uses dt capped at kalman_max_coast_process_seconds (10 s), so the
    # position variance stays ~0.25*10^4*q rather than ~0.25*60^4*q.
    variance = predicted.position_covariance_m2[0][0]
    assert variance < 1.0e4
    assert variance < 0.01 * (0.25 * (60.0**4) * 2.0)


def test_long_coast_without_guards_matches_legacy_behaviour() -> None:
    filt, state = _moving_filter(
        max_coast_process_seconds=0.0,
        coast_velocity_half_life_seconds=0.0,
    )
    predicted = filt.predict(state, 60.0)
    assert predicted.position_m[0] == pytest.approx(300.0)
    assert predicted.position_covariance_m2[0][0] > 6.0e6


def test_short_gap_prediction_is_exact_constant_velocity() -> None:
    """Gaps below half the velocity half-life keep the exact CV transition, so
    the pinned dt=1 s posteriors elsewhere in the suite stay bit-identical."""
    guarded, state = _moving_filter()
    legacy, legacy_state = _moving_filter(
        max_coast_process_seconds=0.0,
        coast_velocity_half_life_seconds=0.0,
    )
    for dt_s in (0.5, 1.0, 4.0):
        a = guarded.predict(state, dt_s)
        b = legacy.predict(legacy_state, dt_s)
        assert a.position_m == b.position_m
        assert a.position_covariance_m2 == b.position_covariance_m2


@pytest.mark.asyncio
async def test_coast_guard_keeps_a_drifting_singer_after_a_long_silence() -> None:
    """With the lifecycle fix alone the track survives but its extrapolated mean
    lands out of gate; the coast guard is what actually re-associates it."""

    async def run(**overrides) -> list[str]:
        manager = TrackManager(Settings(**overrides))
        ids: list[str] = []
        for index in range(4):  # drifting at 1.5 m/s, detections 5 s apart
            track = await manager.update(
                timestamp_ns=T0 + index * 5 * SECOND,
                position_m=(1.5 * 5 * index, 0.0, 0.0),
                label="Northern Cardinal",
                label_category="wildlife",
                confidence=0.7,
            )
            ids.append(track.id)
        # 60 s of silence, then it sings again from where it stopped.
        track = await manager.update(
            timestamp_ns=T0 + 75 * SECOND,
            position_m=(22.5, 0.0, 0.0),
            label="Northern Cardinal",
            label_category="wildlife",
            confidence=0.7,
        )
        ids.append(track.id)
        return ids

    lifecycle_only = await run(
        kalman_max_coast_process_seconds=0.0,
        kalman_coast_velocity_half_life_seconds=0.0,
        dormant_reacquire_enabled=False,
    )
    assert len(set(lifecycle_only)) == 2

    guarded = await run(dormant_reacquire_enabled=False)
    assert len(set(guarded)) == 1


# ---------------------------------------------------------------------------
# Phase 2 — class-aware association
# ---------------------------------------------------------------------------

def _track(track_id: str, position, category: str) -> TrackState:
    return TrackState(
        id=track_id,
        first_seen_ns=T0,
        last_seen_ns=T0,
        position_m=position,
        label_category=category,
        status=TrackStatus.CONFIRMED.value,
    )


def test_category_gate_rejects_a_mismatched_class_inside_the_gate() -> None:
    assoc = NearestNeighborAssociator(association_distance_m=8.0)
    tracks = [_track("trk-veh", (0.0, 0.0, 0.0), "vehicle")]
    assert (
        assoc.associate(
            T0 + SECOND,
            (2.0, 0.0, 0.0),
            tracks,
            None,
            AssociationContext(label="Northern Cardinal", label_category="wildlife"),
        )
        is None
    )


def test_category_gate_treats_unknown_as_a_wildcard() -> None:
    assoc = NearestNeighborAssociator(association_distance_m=8.0)
    tracks = [_track("trk-veh", (0.0, 0.0, 0.0), "vehicle")]
    # Unknown detection against a real track category...
    assert (
        assoc.associate(
            T0 + SECOND, (2.0, 0.0, 0.0), tracks, None, AssociationContext()
        )
        == "trk-veh"
    )
    # ...and a real detection category against an unknown track.
    unknown_tracks = [_track("trk-any", (0.0, 0.0, 0.0), "unknown")]
    assert (
        assoc.associate(
            T0 + SECOND,
            (2.0, 0.0, 0.0),
            unknown_tracks,
            None,
            AssociationContext(label_category="wildlife"),
        )
        == "trk-any"
    )


def test_category_gate_can_be_disabled() -> None:
    assoc = NearestNeighborAssociator(
        association_distance_m=8.0, category_gate_enabled=False
    )
    tracks = [_track("trk-veh", (0.0, 0.0, 0.0), "vehicle")]
    assert (
        assoc.associate(
            T0 + SECOND,
            (2.0, 0.0, 0.0),
            tracks,
            None,
            AssociationContext(label_category="wildlife"),
        )
        == "trk-veh"
    )


def test_fingerprint_reranks_two_candidates_inside_the_gate() -> None:
    assoc = NearestNeighborAssociator(association_distance_m=8.0)
    tracks = [
        _track("trk-near", (1.0, 0.0, 0.0), "wildlife"),
        _track("trk-far", (2.5, 0.0, 0.0), "wildlife"),
    ]
    scores = {"Northern Cardinal": 0.9, "Blue Jay": 0.1}
    fingerprints = {
        "trk-near": {"American Robin": 1.0},
        "trk-far": {"Northern Cardinal": 1.0},
    }
    # Geometry alone would pick the nearer track.
    assert (
        assoc.associate(T0 + SECOND, (0.0, 0.0, 0.0), tracks, None, AssociationContext())
        == "trk-near"
    )
    # Score similarity re-ranks toward the track that sounds like this detection.
    assert (
        assoc.associate(
            T0 + SECOND,
            (0.0, 0.0, 0.0),
            tracks,
            None,
            AssociationContext(
                classifier_scores=scores, track_fingerprints=fingerprints
            ),
        )
        == "trk-far"
    )


def test_fingerprint_never_admits_a_match_outside_the_positional_gate() -> None:
    assoc = NearestNeighborAssociator(association_distance_m=8.0)
    tracks = [_track("trk-far", (200.0, 0.0, 0.0), "wildlife")]
    assert (
        assoc.associate(
            T0 + SECOND,
            (0.0, 0.0, 0.0),
            tracks,
            None,
            AssociationContext(
                classifier_scores={"Northern Cardinal": 1.0},
                track_fingerprints={"trk-far": {"Northern Cardinal": 1.0}},
            ),
        )
        is None
    )


def test_fingerprint_is_neutral_when_either_side_is_empty() -> None:
    assoc = NearestNeighborAssociator(association_distance_m=8.0)
    tracks = [
        _track("trk-near", (1.0, 0.0, 0.0), "wildlife"),
        _track("trk-far", (2.5, 0.0, 0.0), "wildlife"),
    ]
    assert (
        assoc.associate(
            T0 + SECOND,
            (0.0, 0.0, 0.0),
            tracks,
            None,
            AssociationContext(
                classifier_scores={},
                track_fingerprints={"trk-far": {"Northern Cardinal": 1.0}},
            ),
        )
        == "trk-near"
    )
    assert (
        assoc.associate(
            T0 + SECOND,
            (0.0, 0.0, 0.0),
            tracks,
            None,
            AssociationContext(
                classifier_scores={"Northern Cardinal": 1.0}, track_fingerprints={}
            ),
        )
        == "trk-near"
    )


@pytest.mark.asyncio
async def test_manager_keeps_a_bird_from_stealing_a_vehicle_track() -> None:
    manager = TrackManager(Settings())
    for index in range(2):
        vehicle = await manager.update(
            timestamp_ns=T0 + index * SECOND,
            position_m=(0.0, 0.0, 0.0),
            label="Truck",
            label_category="vehicle",
            confidence=0.9,
            classifier_scores={"Truck": 0.9},
        )

    bird = await manager.update(
        timestamp_ns=T0 + 3 * SECOND,
        position_m=(2.0, 0.0, 0.0),
        label="Northern Cardinal",
        label_category="wildlife",
        confidence=0.6,
        classifier_scores={"Northern Cardinal": 0.6},
    )
    assert bird.id != vehicle.id
    assert vehicle.label == "Truck"


@pytest.mark.asyncio
async def test_manager_maintains_normalized_bounded_fingerprints() -> None:
    manager = TrackManager(Settings(track_fingerprint_top_k=2))
    track = await manager.update(
        timestamp_ns=T0,
        position_m=(0.0, 0.0, 0.0),
        label="Northern Cardinal",
        label_category="wildlife",
        confidence=0.7,
        classifier_scores={"a": 0.9, "b": 0.5, "c": 0.2, "d": 0.1},
    )
    fingerprint = manager._fingerprints[track.id]
    assert set(fingerprint) == {"a", "b"}
    assert np.linalg.norm(list(fingerprint.values())) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Phase 2 — associator arity compatibility
# ---------------------------------------------------------------------------

class _ThreeArgAssociator:
    def __init__(self) -> None:
        self.calls = 0

    def associate(self, timestamp_ns, position_m, existing_tracks):
        self.calls += 1
        return existing_tracks[0].id if existing_tracks else None


class _FourArgAssociator:
    def __init__(self) -> None:
        self.covariances: list = []

    def associate(
        self, timestamp_ns, position_m, existing_tracks, measurement_covariance_m2=None
    ):
        self.covariances.append(measurement_covariance_m2)
        return existing_tracks[0].id if existing_tracks else None


@pytest.mark.asyncio
@pytest.mark.parametrize("associator_cls", [_ThreeArgAssociator, _FourArgAssociator])
async def test_custom_associators_without_context_still_work(associator_cls) -> None:
    associator = associator_cls()
    manager = TrackManager(Settings(), associator=associator)
    first = await manager.update(
        timestamp_ns=T0,
        position_m=(0.0, 0.0, 0.0),
        label="Truck",
        label_category="vehicle",
        confidence=0.8,
        classifier_scores={"Truck": 0.8},
    )
    second = await manager.update(
        timestamp_ns=T0 + SECOND,
        position_m=(1.0, 0.0, 0.0),
        label="Truck",
        label_category="vehicle",
        confidence=0.8,
        classifier_scores={"Truck": 0.8},
    )
    assert first.id == second.id


# ---------------------------------------------------------------------------
# Phase 3 — dormant reacquisition
# ---------------------------------------------------------------------------

async def _confirmed_wildlife_track(manager: TrackManager, **kwargs) -> TrackState:
    track = None
    for index in range(2):
        track = await manager.update(
            timestamp_ns=T0 + index * SECOND,
            position_m=(30.0, 0.0, 0.0),
            label="Northern Cardinal",
            label_category="wildlife",
            confidence=0.8,
            source_node_id="node-a",
            **kwargs,
        )
    assert track is not None
    return track


@pytest.mark.asyncio
async def test_dormant_track_is_reacquired_after_a_long_silence() -> None:
    manager = TrackManager(Settings())
    original = await _confirmed_wildlife_track(manager)

    # 12 minutes of silence: past the wildlife drop (360 s) and reap (600 s)
    # thresholds, inside the 1800 s dormant TTL. Reaping needs a second ageing
    # pass — the first only marks the track DROPPED.
    silent_ns = T0 + 12 * 60 * SECOND
    await manager.snapshot(now_ns=T0 + 400 * SECOND)
    assert await manager.snapshot(now_ns=silent_ns) == []

    revived = await manager.update(
        timestamp_ns=silent_ns,
        position_m=(40.0, 0.0, 0.0),
        label="Northern Cardinal",
        label_category="wildlife",
        confidence=0.1,
        source_node_id="node-b",
    )
    assert revived.id == original.id
    assert revived.status == TrackStatus.CONFIRMED.value
    assert revived.first_seen_ns == original.first_seen_ns
    assert manager.dormant_reacquired_count() == 1
    # Confidence decays from the parked value (0.8, half-life 300 s) but never
    # falls below this detection's own confidence.
    assert 0.1 < revived.confidence < 0.8
    assert revived.confidence == pytest.approx(0.8 * (0.5 ** (719.0 / 300.0)), rel=1e-6)
    assert revived.contributor_node_ids == ["node-a", "node-b"]
    assert revived.velocity_mps == (0.0, 0.0, 0.0)


@pytest.mark.asyncio
async def test_pre_reap_revive_mutates_in_place_leaving_one_track() -> None:
    manager = TrackManager(Settings())
    original = await _confirmed_wildlife_track(manager)

    # Past drop (360 s) but before reap (600 s): the DROPPED row is still there.
    silent_ns = T0 + 400 * SECOND
    dropped = await manager.snapshot(now_ns=silent_ns)
    assert [t.status for t in dropped] == [TrackStatus.DROPPED.value]

    revived = await manager.update(
        timestamp_ns=silent_ns,
        position_m=(31.0, 0.0, 0.0),
        label="Northern Cardinal",
        label_category="wildlife",
        confidence=0.5,
    )
    assert revived.id == original.id
    tracks = await manager.snapshot()
    assert len(tracks) == 1
    assert tracks[0].status == TrackStatus.CONFIRMED.value


@pytest.mark.asyncio
async def test_dormant_reacquisition_by_fingerprint_without_label_match() -> None:
    manager = TrackManager(Settings())
    original = await _confirmed_wildlife_track(
        manager, classifier_scores={"Northern Cardinal": 0.9, "Blue Jay": 0.3}
    )
    silent_ns = T0 + 12 * 60 * SECOND
    await manager.snapshot(now_ns=silent_ns)

    revived = await manager.update(
        timestamp_ns=silent_ns,
        position_m=(33.0, 0.0, 0.0),
        label="Blue Jay",  # different label, similar score profile
        label_category="wildlife",
        confidence=0.4,
        classifier_scores={"Northern Cardinal": 0.8, "Blue Jay": 0.35},
    )
    assert revived.id == original.id
    assert manager.dormant_reacquired_count() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs, offset_s",
    [
        ({"label": "Truck", "label_category": "vehicle"}, 0),  # category mismatch
        ({"position_m": (200.0, 0.0, 0.0)}, 0),                # outside radius
        ({}, 40 * 60),                                          # past the TTL
    ],
)
async def test_ineligible_dormant_records_yield_a_fresh_track(kwargs, offset_s) -> None:
    manager = TrackManager(Settings())
    original = await _confirmed_wildlife_track(manager)
    silent_ns = T0 + (12 * 60 + offset_s) * SECOND
    await manager.snapshot(now_ns=silent_ns)

    detection = dict(
        timestamp_ns=silent_ns,
        position_m=(35.0, 0.0, 0.0),
        label="Northern Cardinal",
        label_category="wildlife",
        confidence=0.6,
    )
    detection.update(kwargs)
    fresh = await manager.update(**detection)
    assert fresh.id != original.id
    assert manager.dormant_reacquired_count() == 0


@pytest.mark.asyncio
async def test_dormant_reacquisition_can_be_disabled() -> None:
    manager = TrackManager(Settings(dormant_reacquire_enabled=False))
    original = await _confirmed_wildlife_track(manager)
    silent_ns = T0 + 12 * 60 * SECOND
    await manager.snapshot(now_ns=silent_ns)

    fresh = await manager.update(
        timestamp_ns=silent_ns,
        position_m=(31.0, 0.0, 0.0),
        label="Northern Cardinal",
        label_category="wildlife",
        confidence=0.6,
    )
    assert fresh.id != original.id
    assert manager.dormant_reacquired_count() == 0


@pytest.mark.asyncio
async def test_dormant_registry_is_capped_and_evicts_oldest() -> None:
    manager = TrackManager(Settings(dormant_max_records=4))
    # 5 confirmed tracks, far enough apart that none associate with another,
    # each dropped in turn.
    for index in range(5):
        for repeat in range(2):
            await manager.update(
                timestamp_ns=T0 + (index * 10 + repeat) * SECOND,
                position_m=(1000.0 * index, 0.0, 0.0),
                label="Northern Cardinal",
                label_category="wildlife",
                confidence=0.8,
            )
    await manager.snapshot(now_ns=T0 + 600 * SECOND)
    assert len(manager._dormant) == 4
    # The first-dropped (oldest last_seen) record was evicted.
    assert "trk-00001" not in manager._dormant


@pytest.mark.asyncio
async def test_unconfirmed_clutter_is_never_parked() -> None:
    manager = TrackManager(Settings())
    await manager.update(
        timestamp_ns=T0,
        position_m=(0.0, 0.0, 0.0),
        label="Northern Cardinal",
        label_category="wildlife",
        confidence=0.4,
    )
    await manager.snapshot(now_ns=T0 + 600 * SECOND)
    assert manager._dormant == {}


# ---------------------------------------------------------------------------
# BLE: the second TrackManager consumer, always ``unknown`` category
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ble_tracks_keep_base_lifecycle_and_reacquire_their_id() -> None:
    settings = Settings()
    manager = TrackManager(
        settings.ble_tracking_config(),
        default_track_kind="ble",
        track_id_prefix="ble-",
    )
    original = None
    for index in range(2):
        original = await manager.update(
            timestamp_ns=T0 + index * SECOND,
            position_m=(5.0, 5.0, 0.0),
            label="aa:bb:cc:dd:ee:ff",
            confidence=0.6,
        )
    assert original.id.startswith("ble-")

    # BLE inherits the base (unknown-category) lifecycle: stale 20 s, drop 60 s.
    coasting = await manager.snapshot(now_ns=T0 + 30 * SECOND)
    assert coasting[0].status == TrackStatus.COASTING.value
    await manager.snapshot(now_ns=T0 + 90 * SECOND)
    assert await manager.snapshot(now_ns=T0 + 120 * SECOND) == []

    reseen = await manager.update(
        timestamp_ns=T0 + 120 * SECOND,
        position_m=(6.0, 5.0, 0.0),
        label="aa:bb:cc:dd:ee:ff",
        confidence=0.6,
    )
    assert reseen.id == original.id
    assert reseen.track_kind == "ble"
