"""Timeline discipline (step re-anchor + drift slew) on the Python ingest path.

Decision parity with ``minimappr-core/src/timeline_discipline.rs`` through the
shared fixture, then the closed loop on a real ``SensorStreamBuffer``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer, SensorStreamBuffer
from minimappr.core.timeline_discipline import (
    TimelineCorrection,
    TimelineCorrectionKind,
    TimelineDiscipline,
    TimelineDisciplinePolicy,
)

PARITY_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "timeline_discipline_parity_cases.json"
NANOSECONDS_PER_SECOND = 1_000_000_000
START_NS = 1_764_000_000_000_000_000
STEP_NS = 21_000_000

# Step and jitter runs: 16 kHz, 32 ms frames, as the firmware sends them.
RATE_HZ = 16_000
FRAME_SAMPLES = 512
FRAME_NS = 32_000_000
BUFFER_SECONDS = 4.0

# Drift run: 15 minutes at 20 ppm. One-second frames keep the run fast; the
# residual window is still 16 frames, so the slew acts on a 16 s median.
DRIFT_RATE_HZ = 8_000
DRIFT_FRAME_SAMPLES = 8_000
DRIFT_FRAME_NS = 1_000_000_000
DRIFT_BUFFER_SECONDS = 20.0
DRIFT_PARTS_PER_MILLION = 20.0
DRIFT_RUN_SECONDS = 15 * 60
SECONDS_PER_MINUTE = 60.0


def _policy_from_json(payload: dict) -> TimelineDisciplinePolicy:
    settings = Settings()
    return TimelineDisciplinePolicy(
        step_reanchor_enabled=payload["step_reanchor_enabled"],
        step_threshold_ns=payload["step_threshold_ns"],
        step_confirm_frames=payload["step_confirm_frames"],
        drift_slew_threshold_ns=payload["drift_slew_threshold_ns"],
        hard_limit_ns=payload["hard_limit_ns"],
        residual_window_frames=settings.localization_timeline_residual_window_frames,
        residual_percentile=settings.localization_timeline_residual_percentile,
    )


def _correction_from_json(payload: dict | None) -> TimelineCorrection | None:
    if payload is None:
        return None
    return TimelineCorrection(TimelineCorrectionKind(payload["kind"]), payload["origin_shift_ns"])


def test_decisions_match_the_shared_parity_fixture() -> None:
    fixture = json.loads(PARITY_FIXTURE_PATH.read_text())
    for case in fixture["cases"]:
        discipline = TimelineDiscipline(_policy_from_json(case.get("policy", fixture["default_policy"])))
        for op_index, op in enumerate(case["ops"]):
            if "frame" in op:
                actual = discipline.observe_frame(op["frame"]["residual_ns"], op["frame"]["detected_step_ns"])
            else:
                actual = discipline.drift_slew(op["slew"]["windowed_residual_ns"])
            assert actual == _correction_from_json(op["expect"]), f"{case['name']}: op {op_index}"


def test_policy_from_settings_defaults() -> None:
    policy = TimelineDisciplinePolicy.from_settings(Settings())
    assert policy.step_reanchor_enabled is True
    assert policy.step_threshold_ns == 2_000_000
    assert policy.step_confirm_frames == 3
    assert policy.drift_slew_threshold_ns == 5_000_000
    assert policy.hard_limit_ns == 10_000_000
    assert policy.residual_window_frames == 16


def _disciplined_buffer(rate_hz: int = RATE_HZ, buffer_seconds: float = BUFFER_SECONDS, **overrides) -> SensorStreamBuffer:
    return SensorStreamBuffer(rate_hz, buffer_seconds, TimelineDisciplinePolicy.from_settings(Settings(**overrides)))


def _append_frames(buffer: SensorStreamBuffer, frames, extra_ns, *, frame_samples=FRAME_SAMPLES, frame_ns=FRAME_NS) -> None:
    for frame in frames:
        start = frame * frame_samples
        buffer.append(
            START_NS + frame * frame_ns + extra_ns(frame),
            np.full(frame_samples, float(frame % 7), dtype=np.float32),
            start_sample_index=start,
            end_sample_index=start + frame_samples,
        )


def _residual_ns(buffer: SensorStreamBuffer, frame: int, extra_ns, *, frame_samples=FRAME_SAMPLES, frame_ns=FRAME_NS) -> int:
    state = buffer._state
    return START_NS + frame * frame_ns + extra_ns(frame) - buffer._sample_index_to_time_ns(state, frame * frame_samples)


def test_a_step_mid_stream_reanchors_without_dropping_audio() -> None:
    step_frame = 40
    stepped = lambda frame: STEP_NS if frame >= step_frame else 0  # noqa: E731
    buffer = _disciplined_buffer()
    _append_frames(buffer, range(step_frame + 1), stepped)
    assert buffer.timeline_step_reanchor_count == 0, "one stepped frame is not yet confirmed"
    _append_frames(buffer, range(step_frame + 1, 100), stepped)

    assert buffer.timeline_step_reanchor_count == 1
    assert buffer.timeline_drift_slew_count == 0
    assert buffer.reanchor_count == 0, "no audio was cleared"
    assert bool(buffer.coverage.all()), "no zero gap"
    assert buffer.samples.size == min(100 * FRAME_SAMPLES, buffer.max_samples)
    # Later frames map onto the corrected clock; so does the pre-step audio.
    assert _residual_ns(buffer, 99, stepped) == 0
    assert buffer._sample_index_to_time_ns(buffer._state, 0) == START_NS + STEP_NS


def test_a_late_frame_and_a_lost_frame_run_never_reanchor() -> None:
    late_frame, late_ns = 30, 6_000_000
    gap_start, gap_frames = 60, 9
    rng = np.random.default_rng(7)
    jitter_ns = {frame: int(rng.uniform(-900_000, 900_000)) for frame in range(200)}
    noisy = lambda frame: jitter_ns[frame] + (late_ns if frame == late_frame else 0)  # noqa: E731
    # Long enough to still hold the gap at the end of the run.
    buffer = _disciplined_buffer(buffer_seconds=2 * BUFFER_SECONDS)
    _append_frames(buffer, range(gap_start), noisy)
    _append_frames(buffer, range(gap_start + gap_frames, 200), noisy)
    assert buffer.timeline_step_reanchor_count == 0
    assert buffer.timeline_drift_slew_count == 0
    assert buffer.reanchor_count == 0
    # The lost frames stay a real (uncovered) gap; nothing else is missing.
    assert int(np.count_nonzero(~buffer.coverage)) == gap_frames * FRAME_SAMPLES


def test_slow_drift_at_20_ppm_for_15_minutes_stays_under_the_hard_limit() -> None:
    settings = Settings()
    hard_limit_ns = round(settings.localization_cross_node_max_timeline_residual_seconds * NANOSECONDS_PER_SECOND)
    slew_threshold_ns = round(hard_limit_ns * settings.localization_timeline_drift_slew_fraction)
    drift = lambda frame: round(frame * DRIFT_FRAME_NS * DRIFT_PARTS_PER_MILLION * 1e-6)  # noqa: E731
    frame_kwargs = {"frame_samples": DRIFT_FRAME_SAMPLES, "frame_ns": DRIFT_FRAME_NS}

    def run(buffer: SensorStreamBuffer) -> tuple[int, float | None]:
        max_abs_residual_ns, over_limit_after_minutes = 0, None
        for frame in range(DRIFT_RUN_SECONDS):
            _append_frames(buffer, [frame], drift, **frame_kwargs)
            residual_ns = abs(_residual_ns(buffer, frame, drift, **frame_kwargs))
            max_abs_residual_ns = max(max_abs_residual_ns, residual_ns)
            if residual_ns > hard_limit_ns and over_limit_after_minutes is None:
                over_limit_after_minutes = frame / SECONDS_PER_MINUTE
        return max_abs_residual_ns, over_limit_after_minutes

    _, undisciplined_over_limit_after = run(SensorStreamBuffer(DRIFT_RATE_HZ, DRIFT_BUFFER_SECONDS))
    assert undisciplined_over_limit_after is not None and 8.0 < undisciplined_over_limit_after < 9.0

    buffer = _disciplined_buffer(DRIFT_RATE_HZ, DRIFT_BUFFER_SECONDS)
    max_abs_residual_ns, over_limit_after = run(buffer)
    assert over_limit_after is None, "the residual never reaches the 10 ms limit"
    # The windowed median lags the newest frame by half the window.
    window_lag_ns = drift(settings.localization_timeline_residual_window_frames // 2)
    assert max_abs_residual_ns <= slew_threshold_ns + window_lag_ns + drift(1)
    assert buffer.timeline_step_reanchor_count == 0
    accumulated_drift_ns = drift(DRIFT_RUN_SECONDS)
    assert buffer.timeline_drift_slew_count == pytest.approx(accumulated_drift_ns / slew_threshold_ns, abs=1)
    assert buffer.reanchor_count == 0 and bool(buffer.coverage.all())


def test_without_a_policy_the_buffer_keeps_the_first_anchor() -> None:
    stepped = lambda frame: STEP_NS if frame >= 10 else 0  # noqa: E731
    buffer = SensorStreamBuffer(RATE_HZ, BUFFER_SECONDS)
    _append_frames(buffer, range(40), stepped)
    assert buffer.timeline_step_reanchor_count == 0
    assert _residual_ns(buffer, 39, stepped) == STEP_NS


def test_step_reanchor_off_still_slews_drift() -> None:
    buffer = _disciplined_buffer(localization_timeline_step_reanchor_enabled=False)
    stepped = lambda frame: STEP_NS if frame >= 10 else 0  # noqa: E731
    _append_frames(buffer, range(40), stepped)
    assert buffer.timeline_step_reanchor_count == 0
    # 21 ms is past the hard limit: not drift, so it is not slewed either.
    assert buffer.timeline_drift_slew_count == 0
    assert _residual_ns(buffer, 39, stepped) == STEP_NS


def test_a_full_reset_forgets_the_pending_step() -> None:
    buffer = _disciplined_buffer()
    _append_frames(buffer, range(10), lambda frame: 0)
    # One stepped frame opens a pending step, then a jump past the buffer
    # length re-anchors from scratch; the new timeline must not inherit it.
    _append_frames(buffer, [10], lambda frame: STEP_NS)
    far_ns = round((BUFFER_SECONDS + 60.0) * NANOSECONDS_PER_SECOND)
    _append_frames(buffer, range(11, 20), lambda frame: far_ns)
    assert buffer.reanchor_count == 1
    assert buffer.timeline_step_reanchor_count == 0


def test_shift_timeline_origin_retimes_in_place() -> None:
    buffer = SensorStreamBuffer(RATE_HZ, BUFFER_SECONDS)
    buffer.shift_timeline_origin(STEP_NS)
    assert buffer.start_time_ns is None
    _append_frames(buffer, range(4), lambda frame: 0)
    samples_before = buffer.samples.copy()
    buffer.shift_timeline_origin(STEP_NS)
    assert buffer.start_time_ns == START_NS + STEP_NS
    np.testing.assert_array_equal(buffer.samples, samples_before)


@pytest.mark.asyncio
async def test_multi_sensor_buffer_counts_corrections_per_sensor() -> None:
    buffer = MultiSensorBuffer(BUFFER_SECONDS, TimelineDisciplinePolicy.from_settings(Settings()))
    for frame in range(30):
        start = frame * FRAME_SAMPLES
        await buffer.append(
            "node:ch0",
            RATE_HZ,
            START_NS + frame * FRAME_NS + (STEP_NS if frame >= 10 else 0),
            np.zeros(FRAME_SAMPLES, dtype=np.float32),
            start_sample_index=start,
            end_sample_index=start + FRAME_SAMPLES,
        )
    (entry,) = await buffer.snapshot_state()
    assert entry["timeline_step_reanchor_count"] == 1
    assert entry["timeline_drift_slew_count"] == 0
    assert entry["reanchor_count"] == 0


def test_a_step_on_the_frame_after_the_anchor_is_seen() -> None:
    stepped = lambda frame: STEP_NS if frame >= 1 else 0  # noqa: E731
    buffer = _disciplined_buffer()
    _append_frames(buffer, range(10), stepped)
    assert buffer.timeline_step_reanchor_count == 1
    assert _residual_ns(buffer, 9, stepped) == 0
