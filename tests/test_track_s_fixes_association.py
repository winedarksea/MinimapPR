"""Field Mode S-fixes tracking: species-aware association and the concurrent-perch guard."""

from __future__ import annotations

from minimappr.config import Settings
from minimappr.core.track_perched_concurrency import (
    ConcurrentPerchGuardConfig,
    concurrently_active_perch_conflicts,
)
from minimappr.core.track_species_association_gate import species_claims_contradict
from minimappr.core.tracking import TrackManager
from minimappr.sim.tracker_scenario_catalog import WILDLIFE_CONTINUOUS_MOTION

T0 = 1_700_000_000_000_000_000
SECOND = 1_000_000_000
TANAGER = "scarlet tanager"
GROSBEAK = "rose-breasted grosbeak"
CHICKADEE = "black-capped chickadee"
ROBIN = "american robin"
TANAGER_PERCH_M = (14.0, 7.0, 3.0)
GROSBEAK_PERCH_M = (10.0, 13.0, 3.0)  # 7.2 m from the tanager (the S gate's real-audio pair)
TIGHT_FIX_COVARIANCE_M2 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]
# Three perches 12.5 m apart: beyond the plain gate, inside the relocation gate.
FLOCK_PERCHES_M = ((0.0, 0.0, 3.0), (15.0, 0.0, 3.0), (7.5, 10.0, 3.0))
FLOCK_ROTATION_SECONDS = 1.2
# Wildlife stationary_hold is the default; log-odds is the remaining opt-in.
ALL_TRACKER_FLAGS = dict(tracking_species_log_odds_enabled=True)


def test_contradiction_needs_two_confident_disagreeing_claims() -> None:
    assert species_claims_contradict({GROSBEAK: 0.9}, {TANAGER: 0.8}, 0.5, 0.15)
    assert species_claims_contradict({GROSBEAK: 0.3}, {TANAGER: 0.8}, 0.5, 0.15)  # sub-display claim still counts
    assert not species_claims_contradict({GROSBEAK: 0.1}, {TANAGER: 0.8}, 0.5, 0.15)  # under the claim floor
    assert not species_claims_contradict({GROSBEAK: 0.9}, {TANAGER: 0.4}, 0.5, 0.15)  # unsure track
    assert not species_claims_contradict({GROSBEAK: 0.9}, {}, 0.5, 0.15)  # no belief yet
    assert not species_claims_contradict({GROSBEAK: 0.9, TANAGER: 0.6}, {TANAGER: 0.8}, 0.5, 0.15)  # multi-label support


async def _tanager_then_grosbeak(
    settings: Settings, grosbeak_confidence: float, grosbeak_label: str = GROSBEAK, *, tanager_bouts: int = 1
):
    manager = TrackManager(settings)
    for bout in range(tanager_bouts):
        tanager = await manager.update(
            timestamp_ns=T0 + bout * SECOND // 2, position_m=TANAGER_PERCH_M, label=TANAGER,
            label_category="wildlife", confidence=0.9, measurement_covariance_m2=TIGHT_FIX_COVARIANCE_M2,
        )
    grosbeak = await manager.update(
        timestamp_ns=T0 + int(2.5 * SECOND), position_m=GROSBEAK_PERCH_M, label=grosbeak_label, label_category="wildlife",
        confidence=grosbeak_confidence, measurement_covariance_m2=TIGHT_FIX_COVARIANCE_M2,
    )
    return tanager, grosbeak


async def test_confident_other_species_never_joins_a_confident_track() -> None:
    tanager, grosbeak = await _tanager_then_grosbeak(Settings(**ALL_TRACKER_FLAGS), grosbeak_confidence=0.9)
    assert grosbeak.id != tanager.id
    assert tanager.label == TANAGER


async def test_unknown_label_may_still_associate() -> None:
    tanager, unknown = await _tanager_then_grosbeak(
        Settings(**ALL_TRACKER_FLAGS), grosbeak_confidence=0.0, grosbeak_label="unknown"
    )
    assert unknown.id == tanager.id


async def test_sub_display_other_species_builds_its_own_track() -> None:
    # The real-audio failure: weak grosbeak bouts must not seed a grosbeak belief
    # inside the tanager's track (which would later let confident ones in).
    tanager, weak_grosbeak = await _tanager_then_grosbeak(
        Settings(**ALL_TRACKER_FLAGS), grosbeak_confidence=0.3, tanager_bouts=3
    )
    assert weak_grosbeak.id != tanager.id
    assert GROSBEAK not in (tanager.species_belief or {})


async def test_weak_misclassified_bout_on_the_same_perch_joins_the_bird() -> None:
    manager = TrackManager(Settings(**ALL_TRACKER_FLAGS))
    for bout in range(3):
        tanager = await manager.update(
            timestamp_ns=T0 + bout * SECOND, position_m=TANAGER_PERCH_M, label=TANAGER, label_category="wildlife",
            confidence=0.9, measurement_covariance_m2=TIGHT_FIX_COVARIANCE_M2,
        )
    confused = await manager.update(
        timestamp_ns=T0 + 4 * SECOND, position_m=(14.5, 7.3, 3.0), label=GROSBEAK, label_category="wildlife",
        confidence=0.3, measurement_covariance_m2=TIGHT_FIX_COVARIANCE_M2,
    )
    assert confused.id == tanager.id and confused.species_top_label == TANAGER


async def test_species_gate_off_or_log_odds_off_keeps_legacy_merge() -> None:
    for settings in (
        Settings(**ALL_TRACKER_FLAGS, tracking_species_association_gate_enabled=False),
        Settings(tracking_perched_concurrent_perch_guard_enabled=False),
        Settings(**WILDLIFE_CONTINUOUS_MOTION),
    ):
        tanager, grosbeak = await _tanager_then_grosbeak(settings, grosbeak_confidence=0.9)
        assert grosbeak.id == tanager.id


async def _rotating_flock(settings: Settings, rounds: int = 2) -> list[str]:
    manager = TrackManager(settings)
    track_ids: list[str] = []
    for bout in range(rounds * len(FLOCK_PERCHES_M)):
        track = await manager.update(
            timestamp_ns=T0 + int(bout * FLOCK_ROTATION_SECONDS * SECOND),
            position_m=FLOCK_PERCHES_M[bout % len(FLOCK_PERCHES_M)], label=CHICKADEE, label_category="wildlife",
            confidence=0.9, measurement_covariance_m2=TIGHT_FIX_COVARIANCE_M2,
        )
        track_ids.append(track.id)
    return track_ids


async def test_rotating_same_species_flock_keeps_one_track_per_perch() -> None:
    ids = await _rotating_flock(Settings(**ALL_TRACKER_FLAGS))
    assert len(set(ids)) == len(FLOCK_PERCHES_M)
    assert ids[: len(FLOCK_PERCHES_M)] == ids[len(FLOCK_PERCHES_M):]  # every perch keeps its own track


async def test_without_guard_the_third_bird_relocates_the_first_track() -> None:
    ids = await _rotating_flock(
        Settings(**ALL_TRACKER_FLAGS, tracking_perched_concurrent_perch_guard_enabled=False)
    )
    assert ids[2] == ids[0]  # the W2 ghost-perch mistake the guard removes


async def test_single_bird_hop_keeps_identity_with_guard_on() -> None:
    manager = TrackManager(Settings(**ALL_TRACKER_FLAGS))
    first = await manager.update(timestamp_ns=T0, position_m=(0.0, 0.0, 2.0), label=ROBIN,
                                 label_category="wildlife", confidence=0.8)
    hop = await manager.update(timestamp_ns=T0 + 10 * SECOND, position_m=(15.0, 0.0, 2.0), label=ROBIN,
                               label_category="wildlife", confidence=0.8)
    assert hop.id == first.id


def test_guard_ignores_fixes_consistent_with_a_noisy_perch() -> None:
    from minimappr.models import TrackState

    noisy = TrackState(
        id="trk-far", first_seen_ns=T0, last_seen_ns=T0, position_m=(80.0, 46.0, 10.0),
        position_covariance_m2=[[900.0, 0.0, 0.0], [0.0, 900.0, 0.0], [0.0, 0.0, 900.0]],
        label=ROBIN, label_category="wildlife", confidence=0.8,
    )
    config = ConcurrentPerchGuardConfig(enabled=True, snap_distance_m=6.0, min_silence_seconds=2.0,
                                        concurrent_activity_seconds=30.0)
    conflicts = concurrently_active_perch_conflicts(
        timestamp_ns=T0 + SECOND, position_m=(95.0, 50.0, 10.0), measurement_covariance_m2=None,
        detection_species=ROBIN, tracks=[noisy], top_species_for=lambda track: [track.label],
        is_perched=lambda track_id: True, config=config,
    )
    assert conflicts == set()


async def test_weak_claim_with_an_uninformative_fix_does_not_join_a_distant_confident_track() -> None:
    # review/s-fixes (real-audio robin row labelled Harris's Sparrow): a weak contradicting
    # claim whose fix covariance spans tens of metres is statistically "on" every perch;
    # it must be geometrically on the track's perch to count as that bird's confused bout.
    manager = TrackManager(Settings(**ALL_TRACKER_FLAGS))
    for bout in range(3):
        robin = await manager.update(
            timestamp_ns=T0 + bout * SECOND, position_m=(6.0, 6.0, 3.0), label=ROBIN, label_category="wildlife",
            confidence=0.9, measurement_covariance_m2=TIGHT_FIX_COVARIANCE_M2,
        )
    huge_fix_covariance_m2 = [[2025.0, 0.0, 0.0], [0.0, 41000.0, 0.0], [0.0, 0.0, 198000.0]]
    sparrow = await manager.update(
        timestamp_ns=T0 + 20 * SECOND, position_m=(-5.8, 4.2, 3.2), label="harris's sparrow",
        label_category="wildlife", confidence=0.46, measurement_covariance_m2=huge_fix_covariance_m2,
    )
    assert sparrow.id != robin.id
