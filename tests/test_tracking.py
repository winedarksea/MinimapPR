from __future__ import annotations

import pytest

from minimappr.config import Settings
from minimappr.core.tracking import TrackManager
from minimappr.models import TrackStatus


@pytest.mark.asyncio
async def test_tracking_defaults_to_kalman_filter() -> None:
    manager = TrackManager(Settings(association_distance_m=20.0))
    t0 = 1_700_000_000_000_000_000

    first = await manager.update(
        timestamp_ns=t0,
        position_m=(0.0, 0.0, 0.0),
        label="bird",
        confidence=0.5,
    )
    assert first.status == TrackStatus.TENTATIVE.value

    second = await manager.update(
        timestamp_ns=t0 + 1_000_000_000,
        position_m=(10.0, 0.0, 0.0),
        label="bird",
        confidence=0.6,
    )

    # The default Kalman filter trusts this low-uncertainty position measurement
    # more than the legacy alpha-beta smoother and estimates a matching velocity.
    assert second.position_m == pytest.approx((9.318181818181818, 0.0, 0.0), abs=1e-6)
    assert second.velocity_mps == pytest.approx((7.727272727272727, 0.0, 0.0), abs=1e-6)
    assert second.status == TrackStatus.CONFIRMED.value


@pytest.mark.asyncio
async def test_tracking_kalman_mode_tracks_constant_velocity() -> None:
    settings = Settings(
        tracking_filter="kalman",
        association_distance_m=40.0,
        kalman_process_noise=1.0,
        kalman_measurement_noise=0.8,
    )
    manager = TrackManager(settings)

    t0 = 1_700_000_000_000_000_000
    vx = 2.0
    noise = [0.0, 0.25, -0.2, 0.15, -0.1, 0.05]

    track_ids: set[str] = set()
    last = None
    for idx, nx in enumerate(noise):
        measured_x = (idx * vx) + nx
        last = await manager.update(
            timestamp_ns=t0 + idx * 1_000_000_000,
            position_m=(measured_x, 0.0, 0.0),
            label="vehicle",
            confidence=0.8,
        )
        track_ids.add(last.id)

    assert last is not None
    assert len(track_ids) == 1
    assert last.update_count == len(noise)
    assert last.status == TrackStatus.CONFIRMED.value
    assert abs(last.position_m[0] - (vx * (len(noise) - 1))) < 0.8
    assert 1.2 < last.velocity_mps[0] < 2.8


def test_tracking_rejects_unsupported_filter_name() -> None:
    with pytest.raises(ValueError, match="Unsupported tracking_filter"):
        TrackManager(Settings(tracking_filter="spline"))


def test_tracking_rejects_invalid_kalman_noise() -> None:
    with pytest.raises(ValueError, match="kalman_measurement_noise"):
        TrackManager(Settings(tracking_filter="kalman", kalman_measurement_noise=0.0))


@pytest.mark.asyncio
async def test_tqi_recency_reflects_update_gap_not_total_age() -> None:
    manager = TrackManager(Settings(association_distance_m=20.0))
    t0 = 1_700_000_000_000_000_000

    last = None
    for idx in range(301):
        last = await manager.update(
            timestamp_ns=t0 + idx * 1_000_000_000,
            position_m=(0.0, 0.0, 0.0),
            label="drone",
            confidence=0.8,
        )

    assert last is not None
    assert last.update_count == 301
    # Default weights (0.3, 0.3, 0.2, 0.2): confidence 0.8, full update
    # corroboration, recency for a 1 s gap (30/31), sensor factor 0.25.
    # The 300 s total track age must NOT drag recency down (that would give
    # recency ~0.09 and TQI ~0.61).
    expected = 0.3 * 0.8 + 0.3 * 1.0 + 0.2 * (30.0 / 31.0) + 0.2 * 0.25
    assert last.tqi == pytest.approx(expected, abs=1e-6)


@pytest.mark.asyncio
async def test_tqi_penalizes_long_gap_since_previous_update() -> None:
    manager = TrackManager(Settings(association_distance_m=20.0))
    t0 = 1_700_000_000_000_000_000

    await manager.update(
        timestamp_ns=t0,
        position_m=(0.0, 0.0, 0.0),
        label="drone",
        confidence=0.8,
    )
    prompt = await manager.update(
        timestamp_ns=t0 + 1_000_000_000,
        position_m=(0.0, 0.0, 0.0),
        label="drone",
        confidence=0.8,
    )
    prompt_tqi = prompt.tqi

    stale = await manager.update(
        timestamp_ns=t0 + 61_000_000_000,
        position_m=(0.0, 0.0, 0.0),
        label="drone",
        confidence=0.8,
    )
    assert stale.id == prompt.id
    assert stale.tqi < prompt_tqi


@pytest.mark.asyncio
async def test_label_change_resets_confidence() -> None:
    manager = TrackManager(Settings(association_distance_m=20.0))
    t0 = 1_700_000_000_000_000_000

    first = await manager.update(
        timestamp_ns=t0,
        position_m=(0.0, 0.0, 0.0),
        label="gunshot",
        confidence=0.95,
    )
    assert first.confidence == pytest.approx(0.95)

    relabeled = await manager.update(
        timestamp_ns=t0 + 1_000_000_000,
        position_m=(0.0, 0.0, 0.0),
        label="dog",
        confidence=0.30,
    )
    assert relabeled.id == first.id
    assert relabeled.label == "dog"
    assert relabeled.confidence == pytest.approx(0.30)


@pytest.mark.asyncio
async def test_same_label_confidence_still_ratchets() -> None:
    manager = TrackManager(Settings(association_distance_m=20.0))
    t0 = 1_700_000_000_000_000_000

    await manager.update(
        timestamp_ns=t0,
        position_m=(0.0, 0.0, 0.0),
        label="dog",
        confidence=0.30,
    )
    raised = await manager.update(
        timestamp_ns=t0 + 1_000_000_000,
        position_m=(0.0, 0.0, 0.0),
        label="dog",
        confidence=0.50,
    )
    assert raised.confidence == pytest.approx(0.50)

    held = await manager.update(
        timestamp_ns=t0 + 2_000_000_000,
        position_m=(0.0, 0.0, 0.0),
        label="dog",
        confidence=0.20,
    )
    assert held.confidence == pytest.approx(0.50)
