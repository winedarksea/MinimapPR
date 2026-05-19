from __future__ import annotations

from dataclasses import dataclass

import pytest

from minimappr.core.recording_visual import build_recording_visual_timeline


@dataclass
class _Trajectory:
    track_id: str
    label: str
    waypoints: list[tuple[int, tuple[float, float, float]]]


@dataclass
class _Slot:
    unit_track_ids: list[str | None]
    active_ranges: list[tuple[int, int]]


def test_visual_timeline_interpolates_selected_track_position() -> None:
    slot = _Slot(unit_track_ids=["trk-a", "trk-a"], active_ranges=[(0, 200)])
    trajectories = [
        _Trajectory(
            track_id="trk-a",
            label="warbler",
            waypoints=[(0, (0.0, 0.0, 0.0)), (100, (10.0, 0.0, 0.0))],
        )
    ]

    frames = build_recording_visual_timeline(
        slot,
        trajectories,
        n_samples=200,
        sample_rate_hz=100,
        samples_per_unit=100,
        frame_rate_hz=2,
    )

    assert [frame.sample_offset for frame in frames] == [0, 50, 100, 150]
    assert frames[1].track_id == "trk-a"
    assert frames[1].label == "warbler"
    assert frames[1].position_m == pytest.approx((5.0, 0.0, 0.0))
    assert frames[3].position_m == pytest.approx((10.0, 0.0, 0.0))


def test_visual_timeline_marks_handoff_gaps_inactive() -> None:
    slot = _Slot(unit_track_ids=["trk-a", None, "trk-b"], active_ranges=[(0, 10), (20, 30)])
    trajectories = [
        _Trajectory("trk-a", "first", [(0, (1.0, 0.0, 0.0))]),
        _Trajectory("trk-b", "second", [(20, (2.0, 0.0, 0.0))]),
    ]

    frames = build_recording_visual_timeline(
        slot,
        trajectories,
        n_samples=30,
        sample_rate_hz=30,
        samples_per_unit=10,
        frame_rate_hz=3,
    )

    assert [frame.track_id for frame in frames] == ["trk-a", None, "trk-b"]
    assert frames[1].label is None
    assert frames[1].position_m is None
