"""Golden-fixture regression for IamfPipeline._mvdr_beamform_python.

Fixture captured BEFORE the BlockTrajectoryRenderer extraction (BEAMFORMED_RENDER_CONTRACT
Phase 7). Asserts the refactored renderer reproduces the pre-refactor output at float
tolerance, with band-split blending disabled (the fixture predates that feature).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minimappr.core.iamf_pipeline import IamfPipeline, TrackTrajectory

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mvdr_block_loop_golden_prerefactor.npz"
CAPTURE_RATE_HZ = 16_000


def test_mvdr_beamform_python_matches_prerefactor_golden() -> None:
    data = np.load(FIXTURE_PATH)
    channels = data["channels"]
    expected_output = data["output"]
    waypoint_samples = data["waypoint_samples"]
    waypoint_positions = data["waypoint_positions"]

    waypoints = [
        (int(sample), tuple(float(v) for v in pos))
        for sample, pos in zip(waypoint_samples, waypoint_positions)
    ]
    traj = TrackTrajectory(track_id="golden", waypoints=waypoints)

    pipeline = IamfPipeline(
        sidecar_url=None,
        db_storage=None,
        iamf_object_band_split_enabled=False,
    )

    actual_output = pipeline._mvdr_beamform_python(channels, traj, CAPTURE_RATE_HZ)

    assert actual_output.shape == expected_output.shape
    assert actual_output.dtype == expected_output.dtype
    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-6)
