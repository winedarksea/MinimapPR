from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.audio_buffer import MultiSensorBuffer


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
