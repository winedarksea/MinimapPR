"""Time-indexed rolling buffers for each sensor stream."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class SensorStreamBuffer:
    sample_rate_hz: int
    max_duration_seconds: float
    ns_per_sample: int = field(init=False)
    max_samples: int = field(init=False)
    start_time_ns: int | None = field(init=False, default=None)
    samples: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0, dtype=np.float32))

    def __post_init__(self) -> None:
        self.ns_per_sample = int(round(1_000_000_000 / self.sample_rate_hz))
        self.max_samples = max(1, int(round(self.max_duration_seconds * self.sample_rate_hz)))

    def append(self, start_time_ns: int, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("SensorStreamBuffer expects mono samples")

        if self.start_time_ns is None:
            self.start_time_ns = start_time_ns
            self.samples = samples.copy()
            self._prune()
            return

        end_time_ns = self.start_time_ns + self.samples.size * self.ns_per_sample

        if start_time_ns > end_time_ns + self.ns_per_sample:
            gap_samples = int(round((start_time_ns - end_time_ns) / self.ns_per_sample))
            if gap_samples > 0:
                self.samples = np.concatenate([self.samples, np.zeros(gap_samples, dtype=np.float32)])
        elif start_time_ns < end_time_ns:
            overlap_samples = int(round((end_time_ns - start_time_ns) / self.ns_per_sample))
            if overlap_samples >= samples.size:
                return
            samples = samples[overlap_samples:]

        if samples.size:
            self.samples = np.concatenate([self.samples, samples])
        self._prune()

    def _prune(self) -> None:
        if self.samples.size <= self.max_samples:
            return

        drop = self.samples.size - self.max_samples
        self.samples = self.samples[drop:]
        if self.start_time_ns is not None:
            self.start_time_ns += drop * self.ns_per_sample

    def get_window(self, center_time_ns: int, window_seconds: float) -> np.ndarray | None:
        if self.start_time_ns is None or self.samples.size == 0:
            return None

        window_samples = max(1, int(round(window_seconds * self.sample_rate_hz)))
        start_time_ns = center_time_ns - (window_samples // 2) * self.ns_per_sample
        offset_samples = int(round((start_time_ns - self.start_time_ns) / self.ns_per_sample))
        end_samples = offset_samples + window_samples

        if offset_samples < 0 or end_samples > self.samples.size:
            return None

        return self.samples[offset_samples:end_samples].copy()


class MultiSensorBuffer:
    def __init__(self, max_duration_seconds: float) -> None:
        self.max_duration_seconds = max_duration_seconds
        self._buffers: dict[str, SensorStreamBuffer] = {}
        self._lock = asyncio.Lock()

    async def append(self, sensor_id: str, sample_rate_hz: int, start_time_ns: int, samples: np.ndarray) -> None:
        async with self._lock:
            buffer = self._buffers.get(sensor_id)
            if buffer is None or buffer.sample_rate_hz != sample_rate_hz:
                buffer = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=self.max_duration_seconds)
                self._buffers[sensor_id] = buffer
            buffer.append(start_time_ns=start_time_ns, samples=samples)

    async def get_synchronized_window(
        self,
        sensor_ids: list[str],
        center_time_ns: int,
        window_seconds: float,
        sample_rate_hz: int,
    ) -> dict[str, np.ndarray]:
        async with self._lock:
            result: dict[str, np.ndarray] = {}
            for sensor_id in sensor_ids:
                buffer = self._buffers.get(sensor_id)
                if buffer is None or buffer.sample_rate_hz != sample_rate_hz:
                    continue
                window = buffer.get_window(center_time_ns=center_time_ns, window_seconds=window_seconds)
                if window is not None:
                    result[sensor_id] = window
            return result
