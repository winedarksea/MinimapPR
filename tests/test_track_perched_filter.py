"""Perched random-walk filter + relocation gate (Field Mode W2-tracker, S3)."""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.config import Settings
from minimappr.core.track_filters import KalmanTrackFilter
from minimappr.core.track_filters_perched import (
    MotionModelRoutingTrackFilter,
    PerchedRandomWalkTrackFilter,
)
from minimappr.core.tracking import TrackManager
from minimappr.models import TrackState

T0 = 1_700_000_000_000_000_000
SECOND = 1_000_000_000


def _state(track_id: str = "t1", position=(0.0, 0.0, 0.0)) -> TrackState:
    return TrackState(id=track_id, first_seen_ns=T0, last_seen_ns=T0, position_m=position)


def _perched(**overrides) -> PerchedRandomWalkTrackFilter:
    kwargs = dict(
        position_diffusion_m2_per_second=0.05,
        measurement_noise_m2=1.5,
        initial_position_variance_m2=4.0,
        relocation_snap_distance_m=6.0,
    )
    kwargs.update(overrides)
    return PerchedRandomWalkTrackFilter(**kwargs)


def test_variance_grows_linearly_with_silence_and_velocity_stays_zero():
    flt = _perched()
    flt.initialize_track("t1", (1.0, 2.0, 0.0))
    short = flt.predict(_state(position=(1.0, 2.0, 0.0)), 1.0)
    long = flt.predict(_state(position=(1.0, 2.0, 0.0)), 10.0)
    assert short.position_m == long.position_m == (1.0, 2.0, 0.0)
    growth_short = short.position_covariance_m2[0][0] - 4.0
    growth_long = long.position_covariance_m2[0][0] - 4.0
    assert growth_long == pytest.approx(10.0 * growth_short)
    assert long.velocity_mps == (0.0, 0.0, 0.0)


def test_jitter_is_averaged_not_turned_into_velocity():
    flt = _perched()
    state = _state()
    flt.initialize_track("t1", (0.0, 0.0, 0.0))
    for offset in (1.0, -1.0, 1.0, -1.0):
        state = flt.update(state, (offset, 0.0, 0.0), 8.0)
    assert abs(state.position_m[0]) < 0.6
    assert state.velocity_mps == (0.0, 0.0, 0.0)


def test_far_fix_snaps_to_new_perch_instead_of_smearing():
    flt = _perched()
    state = _state()
    flt.initialize_track("t1", (0.0, 0.0, 0.0))
    state = flt.update(state, (15.0, 0.0, 0.0), 8.0)
    assert state.position_m == (15.0, 0.0, 0.0)
    assert flt.relocation_count == 1


def test_routing_filter_keeps_constant_velocity_for_other_categories():
    cv = KalmanTrackFilter(process_noise=2.0, measurement_noise=1.5,
                           initial_position_variance=4.0, initial_velocity_variance=16.0)
    routing = MotionModelRoutingTrackFilter(
        constant_velocity_filter=cv, perched_filter=_perched(),
        perched_label_categories=frozenset({"wildlife"}),
    )
    routing.initialize_track("bird", (0.0, 0.0, 0.0))
    routing.set_track_motion_category("bird", "wildlife", (0.0, 0.0, 0.0))
    routing.initialize_track("drone", (0.0, 0.0, 0.0))
    routing.set_track_motion_category("drone", "vehicle", (0.0, 0.0, 0.0))
    assert routing.is_perched("bird") and not routing.is_perched("drone")
    drone = _state("drone")
    for step in range(1, 5):
        drone = routing.update(drone, (2.0 * step, 0.0, 0.0), 1.0)
    assert drone.velocity_mps[0] > 0.5


async def _intermittent_bird(manager: TrackManager, positions, gaps_s, label="robin"):
    ids, t = [], T0
    for position, gap in zip(positions, gaps_s):
        t += int(gap * SECOND)
        track = await manager.update(timestamp_ns=t, position_m=position, label=label,
                                     label_category="wildlife", confidence=0.7)
        ids.append(track.id)
    return ids


async def test_perched_track_survives_silence_without_drifting():
    """A bird singing 4 bouts 12 s apart with 1 m jitter: one track, no drift."""
    positions = [(10.0, 5.0, 2.0), (11.0, 5.5, 2.0), (9.3, 4.4, 2.0), (10.6, 5.2, 2.0)]
    gaps = [0.0, 12.0, 12.0, 12.0]
    manager = TrackManager(Settings(tracking_perched_filter_enabled=True))
    ids = await _intermittent_bird(manager, positions, gaps)
    assert len(set(ids)) == 1
    track = (await manager.snapshot())[0]
    assert np.linalg.norm(np.asarray(track.position_m) - np.array([10.2, 5.0, 2.0])) < 1.5
    assert track.velocity_mps == (0.0, 0.0, 0.0)


async def test_relocation_gate_keeps_identity_on_a_15_m_hop():
    positions = [(10.0, 5.0, 2.0), (10.5, 5.0, 2.0), (25.0, 5.0, 2.0), (25.4, 5.3, 2.0)]
    gaps = [0.0, 6.0, 8.0, 6.0]
    flag_on = TrackManager(Settings(tracking_perched_filter_enabled=True))
    ids_on = await _intermittent_bird(flag_on, positions, gaps)
    assert len(set(ids_on)) == 1
    track = (await flag_on.snapshot())[0]
    # Snapped to the new perch, not smeared between perches.
    assert track.position_m[0] > 24.0
    assert flag_on.field_mode.relocation_gate.relocations_accepted == 1

    # Legacy CV Kalman: the wide covariance gate keeps the id, but the hop is
    # read as velocity, which then coasts the bird away during the next silence.
    flag_off = TrackManager(Settings())
    await _intermittent_bird(flag_off, positions[:3], gaps[:3])
    legacy = (await flag_off.snapshot())[0]
    assert abs(legacy.velocity_mps[0]) > 0.5
    assert track.velocity_mps == (0.0, 0.0, 0.0)


async def test_relocation_refused_when_jump_follows_immediately():
    """Two birds of one species calling 15 m apart within a second are two birds."""
    positions = [(10.0, 5.0, 2.0), (25.0, 5.0, 2.0)]
    manager = TrackManager(Settings(tracking_perched_filter_enabled=True))
    ids = await _intermittent_bird(manager, positions, [0.0, 0.5])
    assert len(set(ids)) == 2


async def test_relocation_refused_when_another_same_species_track_is_nearer():
    manager = TrackManager(Settings(tracking_perched_filter_enabled=True))
    a = await manager.update(timestamp_ns=T0, position_m=(0.0, 0.0, 2.0), label="robin",
                             label_category="wildlife", confidence=0.7)
    b = await manager.update(timestamp_ns=T0 + SECOND, position_m=(40.0, 0.0, 2.0), label="robin",
                             label_category="wildlife", confidence=0.7)
    hop = await manager.update(timestamp_ns=T0 + 10 * SECOND, position_m=(28.0, 0.0, 2.0),
                               label="robin", label_category="wildlife", confidence=0.7)
    # b (12 m) is nearer than a (28 m): only the nearest same-species track can relocate.
    assert hop.id == b.id and hop.id != a.id


async def test_relocation_requires_matching_species():
    manager = TrackManager(Settings(tracking_perched_filter_enabled=True))
    first = await manager.update(timestamp_ns=T0, position_m=(0.0, 0.0, 2.0), label="robin",
                                 label_category="wildlife", confidence=0.7)
    other = await manager.update(timestamp_ns=T0 + 10 * SECOND, position_m=(15.0, 0.0, 2.0),
                                 label="wren", label_category="wildlife", confidence=0.7)
    assert other.id != first.id


async def test_drone_tracks_keep_constant_velocity_when_flag_on():
    manager = TrackManager(Settings(tracking_perched_filter_enabled=True))
    track = None
    for step in range(5):
        track = await manager.update(timestamp_ns=T0 + step * SECOND, position_m=(3.0 * step, 0.0, 20.0),
                                     label="drone", label_category="vehicle", confidence=0.8)
    assert track is not None and track.velocity_mps[0] > 1.0


async def test_flock_hop_back_to_ghost_perch_reverts_relocation():
    """Bird A sings, same-species bird B sings 20 m away, then A again: two tracks."""
    manager = TrackManager(Settings(tracking_perched_filter_enabled=True))
    a1 = await manager.update(timestamp_ns=T0, position_m=(2.0, 10.0, 2.0), label="robin",
                              label_category="wildlife", confidence=0.7)
    b1 = await manager.update(timestamp_ns=T0 + 5 * SECOND, position_m=(22.0, 12.0, 2.0), label="robin",
                              label_category="wildlife", confidence=0.7)
    assert b1.id == a1.id  # indistinguishable from a hop at this point
    a2 = await manager.update(timestamp_ns=T0 + 10 * SECOND, position_m=(2.5, 10.2, 2.0), label="robin",
                              label_category="wildlife", confidence=0.7)
    assert a2.id == a1.id
    b2 = await manager.update(timestamp_ns=T0 + 15 * SECOND, position_m=(21.6, 12.3, 2.0), label="robin",
                              label_category="wildlife", confidence=0.7)
    assert b2.id != a1.id
    assert manager.field_mode.relocation_gate.relocations_reverted == 1
    # Both barred from relocating: further alternation stays on two stable tracks.
    for step in range(4):
        timestamp = T0 + (20 + 5 * step) * SECOND
        position = (2.2, 10.1, 2.0) if step % 2 == 0 else (21.8, 12.1, 2.0)
        track = await manager.update(timestamp_ns=timestamp, position_m=position, label="robin",
                                     label_category="wildlife", confidence=0.7)
        assert track.id == (a1.id if step % 2 == 0 else b2.id)
