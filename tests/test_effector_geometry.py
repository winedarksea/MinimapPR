"""Pure pan/tilt slew math tests for known camera/target geometries."""

from __future__ import annotations

import math

import pytest

from minimappr.core.effectors.geometry import compute_pan_tilt
from minimappr.models import NodeOrientation


def _home() -> NodeOrientation:
    return NodeOrientation(yaw_deg=0.0, pitch_deg=0.0)


def test_target_straight_ahead_of_home_view() -> None:
    pan, tilt = compute_pan_tilt((0.0, 0.0, 0.0), _home(), (0.0, 10.0, 0.0))
    assert pan == pytest.approx(0.0, abs=1e-6)
    assert tilt == pytest.approx(0.0, abs=1e-6)


def test_target_due_east_is_90_degrees_pan() -> None:
    pan, tilt = compute_pan_tilt((0.0, 0.0, 0.0), _home(), (10.0, 0.0, 0.0))
    assert pan == pytest.approx(90.0, abs=1e-6)
    assert tilt == pytest.approx(0.0, abs=1e-6)


def test_target_due_west_is_minus_90_degrees_pan() -> None:
    pan, tilt = compute_pan_tilt((0.0, 0.0, 0.0), _home(), (-10.0, 0.0, 0.0))
    assert pan == pytest.approx(-90.0, abs=1e-6)


def test_target_due_south_is_180_degrees_pan() -> None:
    pan, tilt = compute_pan_tilt((0.0, 0.0, 0.0), _home(), (0.0, -10.0, 0.0))
    assert abs(pan) == pytest.approx(180.0, abs=1e-6)


def test_target_directly_above_is_90_degrees_tilt() -> None:
    pan, tilt = compute_pan_tilt((0.0, 0.0, 0.0), _home(), (0.0, 0.0, 10.0))
    assert tilt == pytest.approx(90.0, abs=1e-6)


def test_target_directly_below_is_minus_90_degrees_tilt() -> None:
    pan, tilt = compute_pan_tilt((0.0, 0.0, 0.0), _home(), (0.0, 0.0, -10.0))
    assert tilt == pytest.approx(-90.0, abs=1e-6)


def test_45_degree_elevation() -> None:
    _, tilt = compute_pan_tilt((0.0, 0.0, 0.0), _home(), (0.0, 10.0, 10.0))
    assert tilt == pytest.approx(45.0, abs=1e-6)


def test_home_yaw_offset_is_subtracted_from_bearing() -> None:
    # Camera's home view already points east (yaw=90); a target due east is
    # then dead-center (pan=0).
    orientation = NodeOrientation(yaw_deg=90.0, pitch_deg=0.0)
    pan, _ = compute_pan_tilt((0.0, 0.0, 0.0), orientation, (10.0, 0.0, 0.0))
    assert pan == pytest.approx(0.0, abs=1e-6)


def test_home_pitch_offset_is_subtracted_from_elevation() -> None:
    orientation = NodeOrientation(yaw_deg=0.0, pitch_deg=20.0)
    _, tilt = compute_pan_tilt((0.0, 0.0, 0.0), orientation, (0.0, 10.0, 0.0))
    assert tilt == pytest.approx(-20.0, abs=1e-6)


def test_camera_offset_from_origin() -> None:
    pan, tilt = compute_pan_tilt((5.0, 5.0, 2.0), _home(), (5.0, 15.0, 2.0))
    assert pan == pytest.approx(0.0, abs=1e-6)
    assert tilt == pytest.approx(0.0, abs=1e-6)


def test_pan_stays_within_180_range() -> None:
    for dx in range(-10, 11):
        for dy in range(-10, 11):
            if dx == 0 and dy == 0:
                continue
            pan, _ = compute_pan_tilt((0.0, 0.0, 0.0), _home(), (float(dx), float(dy), 0.0))
            assert -180.0 <= pan <= 180.0


def test_tilt_clamped_to_90_after_pitch_correction() -> None:
    orientation = NodeOrientation(yaw_deg=0.0, pitch_deg=-45.0)
    _, tilt = compute_pan_tilt((0.0, 0.0, 0.0), orientation, (0.0, 10.0, 10.0))
    assert tilt <= 90.0
