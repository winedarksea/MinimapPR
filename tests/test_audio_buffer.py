from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.audio_buffer import MultiSensorBuffer, SensorStreamBuffer


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
