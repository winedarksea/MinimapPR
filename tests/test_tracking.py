from __future__ import annotations

import pytest

from minimappr.config import Settings
from minimappr.core.tracking import TrackManager
from minimappr.models import TrackStatus


@pytest.mark.asyncio
async def test_tracking_defaults_to_linear_smoother() -> None:
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

    # Existing linear smoother behavior should remain the default.
    assert second.position_m == pytest.approx((6.0, 0.0, 0.0), abs=1e-6)
    assert second.velocity_mps == pytest.approx((5.0, 0.0, 0.0), abs=1e-6)
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
