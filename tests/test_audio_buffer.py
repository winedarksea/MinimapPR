from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.audio_buffer import AudioCoverageStats, MultiSensorBuffer, SensorStreamBuffer


@pytest.mark.asyncio
async def test_multi_sensor_buffer_skips_sample_rate_mismatch() -> None:
    buffer = MultiSensorBuffer(max_duration_seconds=1.0)
    start_time_ns = 1_739_900_000_000_000_000

    await buffer.append(
        sensor_id="s16k",
        sample_rate_hz=16_000,
        start_time_ns=start_time_ns,
        samples=np.ones(160, dtype=np.float32),
    )
    await buffer.append(
        sensor_id="s8k",
        sample_rate_hz=8_000,
        start_time_ns=start_time_ns,
        samples=np.ones(80, dtype=np.float32),
    )

    window = await buffer.get_synchronized_window(
        sensor_ids=["s16k", "s8k"],
        center_time_ns=start_time_ns + 5_000_000,
        window_seconds=0.004,
        sample_rate_hz=16_000,
    )

    assert "s16k" in window
    assert "s8k" not in window
    assert window["s16k"].ndim == 1
    assert np.all(np.isfinite(window["s16k"]))


@pytest.mark.asyncio
async def test_multi_sensor_buffer_can_fetch_trailing_window() -> None:
    buffer = MultiSensorBuffer(max_duration_seconds=2.0)
    start_time_ns = 1_739_900_000_000_000_000
    samples = np.arange(16_000, dtype=np.float32)

    await buffer.append(
        sensor_id="s16k",
        sample_rate_hz=16_000,
        start_time_ns=start_time_ns,
        samples=samples,
    )

    trailing = await buffer.get_synchronized_window_ending_at(
        sensor_ids=["s16k"],
        end_time_ns=start_time_ns + 1_000_000_000,
        window_seconds=0.25,
        sample_rate_hz=16_000,
    )

    assert trailing["s16k"].shape == (4_000,)
    assert np.array_equal(trailing["s16k"], samples[-4_000:])


@pytest.mark.asyncio
async def test_multi_sensor_buffer_session_capture_does_not_block_live_pruning() -> None:
    buffer = MultiSensorBuffer(max_duration_seconds=1.0)
    capture = await buffer.start_capture(
        "session-1",
        ["s16k"],
        max_duration_seconds=10.0,
    )
    start_time_ns = 1_739_900_000_000_000_000
    frame = np.ones(400, dtype=np.float32)
    frame_duration_ns = int(round((frame.size / 16_000) * 1_000_000_000))

    for index in range(100):
        await buffer.append(
            sensor_id="s16k",
            sample_rate_hz=16_000,
            start_time_ns=start_time_ns + index * frame_duration_ns,
            samples=frame,
        )

    live_stream = buffer._buffers["s16k"]
    assert live_stream.samples.size == 16_000

    captured, sample_rate_hz, sync_diag = capture.extract_range(
        ["s16k"],
        start_time_ns,
        start_time_ns + 100 * frame_duration_ns,
    )

    assert sample_rate_hz == 16_000
    assert captured.shape == (1, 40_000)
    assert sync_diag["source"] == "python_capture_session"
    assert sync_diag["channel_coverage_ratios"] == [1.0]


def test_sensor_stream_buffer_replaces_padded_gap_when_late_frame_arrives() -> None:
    sample_rate_hz = 16_000
    samples_per_chunk = 4
    chunk_duration_ns = int(round((samples_per_chunk / sample_rate_hz) * 1_000_000_000))
    start_time_ns = 1_000_000_000_000_000_000

    buf = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=5.0)
    buf.append(start_time_ns=start_time_ns, samples=np.ones(samples_per_chunk, dtype=np.float32))
    buf.append(
        start_time_ns=start_time_ns + (2 * chunk_duration_ns),
        samples=np.full(samples_per_chunk, 3.0, dtype=np.float32),
    )

    assert np.array_equal(
        buf.samples,
        np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32),
    )

    buf.append(
        start_time_ns=start_time_ns + chunk_duration_ns,
        samples=np.full(samples_per_chunk, 2.0, dtype=np.float32),
    )

    assert np.array_equal(
        buf.samples,
        np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32),
    )
    stats = buf.get_window_coverage_stats(
        center_time_ns=start_time_ns + int(round((6 / sample_rate_hz) * 1_000_000_000)),
        window_seconds=(12 / sample_rate_hz),
    )
    assert stats is not None
    assert stats.coverage_ratio == 1.0
    assert stats.missing_samples == 0


def test_sensor_stream_buffer_marks_explicit_sample_index_gaps_as_missing() -> None:
    sample_rate_hz = 16_000
    start_time_ns = 1_000_000_000_000_000_000
    buf = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=5.0)
    buf.append(
        start_time_ns=start_time_ns,
        samples=np.ones(4, dtype=np.float32),
        start_sample_index=0,
        end_sample_index=4,
    )
    buf.append(
        start_time_ns=start_time_ns + int(round((8 / sample_rate_hz) * 1_000_000_000)),
        samples=np.full(4, 2.0, dtype=np.float32),
        start_sample_index=8,
        end_sample_index=12,
    )

    assert np.array_equal(buf.samples, np.array([1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2], dtype=np.float32))
    stats = buf.get_window_coverage_stats(
        center_time_ns=start_time_ns + int(round((6 / sample_rate_hz) * 1_000_000_000)),
        window_seconds=(12 / sample_rate_hz),
    )
    assert stats is not None
    assert stats.covered_samples == 8
    assert stats.missing_samples == 4
    assert stats.max_gap_samples == 4
    assert stats.warning is True
    assert stats.degraded is True


def test_audio_coverage_stats_to_json_matches_manifest_shape() -> None:
    stats = AudioCoverageStats(
        sample_count=1280,
        covered_samples=1184,
        missing_samples=96,
        coverage_ratio=0.925,
        missing_ratio=0.075,
        max_gap_samples=64,
        max_gap_seconds=0.004,
        warning=True,
        degraded=True,
    )

    assert stats.to_json() == {
        "sample_count": 1280,
        "covered_samples": 1184,
        "missing_samples": 96,
        "coverage_ratio": 0.925,
        "missing_ratio": 0.075,
        "max_gap_samples": 64,
        "max_gap_seconds": 0.004,
        "warning": True,
        "degraded": True,
    }


def test_sensor_stream_buffer_prune_uses_exact_sample_time_at_48khz() -> None:
    sample_rate_hz = 48_000
    max_samples = 3
    start_time_ns = 1_000_000_000_000_000_000
    buf = SensorStreamBuffer(
        sample_rate_hz=sample_rate_hz,
        max_duration_seconds=max_samples / sample_rate_hz,
    )

    def sample_time(sample_index: int) -> int:
        return start_time_ns + int(round((sample_index / sample_rate_hz) * 1_000_000_000))

    for sample_index in range(5):
        buf.append(
            start_time_ns=sample_time(sample_index),
            samples=np.array([float(sample_index)], dtype=np.float32),
        )

    assert np.array_equal(buf.samples, np.array([2.0, 3.0, 4.0], dtype=np.float32))
    assert buf.start_time_ns == sample_time(2)


def test_sensor_stream_buffer_preserves_large_absolute_nanoseconds_as_integers() -> None:
    sample_rate_hz = 16_000
    frame_samples = 1024
    start_time_ns = 1_777_212_345_678_901_123
    expected_duration_ns = (frame_samples * 1_000_000_000 + sample_rate_hz // 2) // sample_rate_hz
    buf = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=5.0)

    buf.append(
        start_time_ns=start_time_ns,
        samples=np.ones(frame_samples, dtype=np.float32),
        start_sample_index=9_000_000_000_000,
        end_sample_index=9_000_000_000_000 + frame_samples,
    )

    assert buf.start_time_ns == start_time_ns
    assert buf.end_time_ns() == start_time_ns + expected_duration_ns


def test_get_window_ending_at_returns_partial_audio_when_buffer_shorter_than_window() -> None:
    """Regression: previously returned None when start_offset_samples < 0.

    With only 3 s in buffer and a 30 s classification window, BirdNET must still
    receive whatever audio is available rather than falling back to the 80 ms
    localization clip.
    """
    sample_rate_hz = 16_000
    # Buffer contains exactly 3 s of audio.
    three_seconds = np.ones(3 * sample_rate_hz, dtype=np.float32)
    buf = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=40.0)
    start_time_ns = 1_000_000_000_000_000_000
    buf.append(start_time_ns=start_time_ns, samples=three_seconds)

    end_time_ns = start_time_ns + 3 * 1_000_000_000
    # Request 30 s — buffer only has 3 s.
    result = buf.get_window_ending_at(end_time_ns=end_time_ns, window_seconds=30.0)

    assert result is not None, "Should return partial audio instead of None"
    assert result.shape[0] == 3 * sample_rate_hz
    assert np.all(result == 1.0)


def test_get_window_ending_at_returns_none_when_end_is_beyond_buffer() -> None:
    """End offset beyond the buffer still returns None — that would require future samples."""
    sample_rate_hz = 16_000
    buf = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=5.0)
    start_time_ns = 1_000_000_000_000_000_000
    buf.append(start_time_ns=start_time_ns, samples=np.ones(sample_rate_hz, dtype=np.float32))

    # Ask for a window ending 2 s into the future relative to the 1 s buffer.
    end_time_ns = start_time_ns + 2 * 1_000_000_000
    result = buf.get_window_ending_at(end_time_ns=end_time_ns, window_seconds=0.5)

    assert result is None


def test_sensor_stream_buffer_resets_on_large_forward_jump() -> None:
    """Large forward jump (e.g. NTP sync correcting a stale build-timestamp epoch) must
    not allocate a huge zeros array.  The buffer should reset cleanly at the new position
    so subsequent frames resume without a minutes-long silence gap."""
    sample_rate_hz = 16_000
    samples_per_chunk = 1024
    buf = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=30.0)

    t0 = 1_000_000_000_000_000_000
    buf.append(start_time_ns=t0, samples=np.ones(samples_per_chunk, dtype=np.float32))

    # Simulate NTP sync: timestamp jumps forward by 1 hour.
    one_hour_ns = 3_600 * 1_000_000_000
    t_post_ntp = t0 + one_hour_ns
    new_audio = np.full(samples_per_chunk, 2.0, dtype=np.float32)
    buf.append(start_time_ns=t_post_ntp, samples=new_audio)

    # Buffer should have reset to the new position — no huge zeros, no crash.
    assert buf.samples.size <= samples_per_chunk
    assert np.all(buf.samples == 2.0)
    assert buf.start_time_ns == t_post_ntp


def test_sensor_stream_buffer_resets_on_large_backward_jump() -> None:
    """Large backward jump (severe clock regression) also resets cleanly."""
    sample_rate_hz = 16_000
    samples_per_chunk = 1024
    buf = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=30.0)

    t0 = 1_000_000_000_000_000_000
    buf.append(start_time_ns=t0, samples=np.ones(samples_per_chunk, dtype=np.float32))

    # Clock jumps backward by 10 minutes.
    t_before = t0 - 10 * 60 * 1_000_000_000
    new_audio = np.full(samples_per_chunk, 3.0, dtype=np.float32)
    buf.append(start_time_ns=t_before, samples=new_audio)

    assert buf.samples.size <= samples_per_chunk
    assert np.all(buf.samples == 3.0)
    assert buf.start_time_ns == t_before


def test_sensor_stream_buffer_snaps_subframe_jitter_to_contiguous_audio() -> None:
    """Regression: the firmware derives each frame's start_time_ns from a monotonic
    timer captured at DMA-IRQ time, so consecutive frames carry sub-millisecond
    timing jitter even when the audio is sample-contiguous.  Without snap-to-end
    every jittered frame whose timestamp rounds to >=1 sample past the previous
    frame's end leaves a 1+ sample zero hole in the buffer that surfaces as a
    click in the snippet/debug WAV.  Over a multi-second clip these accumulate
    into the scattered <1s gaps users hear in the UI.
    """
    sample_rate_hz = 16_000
    frame_samples = 1024
    frame_duration_ns = int(round(frame_samples * 1_000_000_000 / sample_rate_hz))

    rng = np.random.default_rng(7)
    buf = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=20.0)
    origin = 1_000_000_000_000_000_000
    # Use an all-nonzero pattern so any zero in the resulting buffer is a true
    # gap rather than a recorded silence.
    content = np.linspace(0.1, 1.0, frame_samples, dtype=np.float32)
    n_frames = 200

    for i in range(n_frames):
        # +/- 2 ms jitter is realistic for a single-core firmware that interleaves
        # DMA-IRQ servicing, HTTP publishing, NTP poll, and Wi-Fi housekeeping.
        jitter_ns = int(rng.integers(-2_000_000, 2_000_001))
        start = origin + i * frame_duration_ns + jitter_ns
        buf.append(start_time_ns=start, samples=content.copy())

    assert buf.samples.size == n_frames * frame_samples
    assert int(np.sum(buf.samples == 0.0)) == 0


def test_sensor_stream_buffer_preserves_real_gap_from_dropped_frames() -> None:
    """Snap-to-contiguous must NOT mask genuine missing frames — when the firmware's
    DMA overruns or a publish fails, the resulting multi-frame gap must still be
    zero-padded so downstream timing stays anchored to wall-clock event times.
    """
    sample_rate_hz = 16_000
    frame_samples = 1024
    frame_duration_ns = int(round(frame_samples * 1_000_000_000 / sample_rate_hz))

    buf = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=10.0)
    origin = 1_000_000_000_000_000_000
    content = np.full(frame_samples, 0.5, dtype=np.float32)

    buf.append(start_time_ns=origin, samples=content.copy())
    # Five frames missing between these two (DMA overrun while publish was slow).
    buf.append(start_time_ns=origin + 6 * frame_duration_ns, samples=content.copy())

    # Buffer covers the full 7-frame span; the middle 5 frames must be zeros.
    assert buf.samples.size == 7 * frame_samples
    assert np.all(buf.samples[:frame_samples] == 0.5)
    assert np.all(buf.samples[6 * frame_samples :] == 0.5)
    assert int(np.sum(buf.samples == 0.0)) == 5 * frame_samples


def test_sensor_stream_buffer_uses_explicit_sample_coverage_without_jitter_snapping() -> None:
    sample_rate_hz = 16_000
    start_time_ns = 1_000_000_000_000_000_000
    buf = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=5.0)

    buf.append(
        start_time_ns=start_time_ns,
        samples=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        start_sample_index=100,
        end_sample_index=104,
    )
    # Timestamp is slightly late, but explicit sample coverage should still place this
    # chunk immediately after the previous one with no snapped gap or overlap.
    buf.append(
        start_time_ns=start_time_ns + 300_000,
        samples=np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        start_sample_index=104,
        end_sample_index=108,
    )

    assert np.array_equal(
        buf.samples,
        np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )
    assert buf.start_time_ns == start_time_ns


def test_sensor_stream_buffer_preserves_explicit_sample_origin_after_large_reset() -> None:
    sample_rate_hz = 16_000
    start_time_ns = 1_000_000_000_000_000_000
    buf = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=1.0)

    buf.append(
        start_time_ns=start_time_ns,
        samples=np.ones(16_000, dtype=np.float32),
        start_sample_index=0,
        end_sample_index=16_000,
    )
    # Force the reset path with explicit sample coverage far in the future.
    buf.append(
        start_time_ns=start_time_ns + 10_000_000_000,
        samples=np.full(4, 2.0, dtype=np.float32),
        start_sample_index=320_000,
        end_sample_index=320_004,
    )

    assert np.array_equal(buf.samples, np.full(4, 2.0, dtype=np.float32))
    assert buf.start_time_ns == start_time_ns + 10_000_000_000


def test_sensor_stream_buffer_reanchors_explicit_contiguous_samples_after_ntp_jump() -> None:
    sample_rate_hz = 16_000
    frame_samples = 1024
    stale_start_time_ns = 1_000_000_000_000_000_000
    corrected_start_time_ns = stale_start_time_ns + 600_000_000_000
    buf = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=30.0)

    buf.append(
        start_time_ns=stale_start_time_ns,
        samples=np.ones(frame_samples, dtype=np.float32),
        start_sample_index=0,
        end_sample_index=frame_samples,
    )
    buf.append(
        start_time_ns=corrected_start_time_ns,
        samples=np.full(frame_samples, 2.0, dtype=np.float32),
        start_sample_index=frame_samples,
        end_sample_index=frame_samples * 2,
    )

    assert np.array_equal(buf.samples, np.full(frame_samples, 2.0, dtype=np.float32))
    assert buf.start_time_ns == corrected_start_time_ns
    assert buf.end_time_ns() == corrected_start_time_ns + int(round(frame_samples * 1_000_000_000 / sample_rate_hz))


@pytest.mark.asyncio
async def test_classification_windows_use_per_sensor_fallback_when_some_sensors_have_partial_audio() -> None:
    """With a short buffer, sensors that have any audio should use their trailing window;
    sensors with no buffer at all fall back to the supplied fallback window."""
    sample_rate_hz = 16_000
    start_time_ns = 1_000_000_000_000_000_000
    buf = MultiSensorBuffer(max_duration_seconds=40.0)

    # Sensor A has 3 s of audio.
    await buf.append(
        sensor_id="a",
        sample_rate_hz=sample_rate_hz,
        start_time_ns=start_time_ns,
        samples=np.ones(3 * sample_rate_hz, dtype=np.float32),
    )
    # Sensor B has no audio at all.

    end_time_ns = start_time_ns + 3 * 1_000_000_000
    trailing = await buf.get_synchronized_window_ending_at(
        sensor_ids=["a", "b"],
        end_time_ns=end_time_ns,
        window_seconds=30.0,
        sample_rate_hz=sample_rate_hz,
    )

    # Sensor A should have returned its partial 3 s window.
    assert "a" in trailing
    assert trailing["a"].shape[0] == 3 * sample_rate_hz
    # Sensor B has nothing — no key.
    assert "b" not in trailing
