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

    async def get_recent_window_for_sensors(
        self,
        sensor_ids: list[str],
        window_seconds: float,
    ) -> tuple[dict[str, np.ndarray], int, int] | None:
        async with self._lock:
            available: list[tuple[str, SensorStreamBuffer]] = []
            sample_rate_counts: dict[int, int] = {}
            for sensor_id in sensor_ids:
                buffer = self._buffers.get(sensor_id)
                if buffer is None or buffer.start_time_ns is None or buffer.samples.size == 0:
                    continue
                available.append((sensor_id, buffer))
                sample_rate_counts[buffer.sample_rate_hz] = sample_rate_counts.get(buffer.sample_rate_hz, 0) + 1

            if not available:
                return None

            dominant_sample_rate_hz = min(sample_rate_counts.keys())
            best_count = -1
            for sample_rate_hz, count in sample_rate_counts.items():
                if count > best_count or (count == best_count and sample_rate_hz > dominant_sample_rate_hz):
                    dominant_sample_rate_hz = sample_rate_hz
                    best_count = count

            window_samples = max(1, int(round(window_seconds * dominant_sample_rate_hz)))
            latest_end_ns = 0
            recent: dict[str, np.ndarray] = {}
            for sensor_id, buffer in available:
                if buffer.sample_rate_hz != dominant_sample_rate_hz:
                    continue
                tail = buffer.samples[-window_samples:].copy()
                if tail.size == 0:
                    continue
                recent[sensor_id] = tail
                end_ns = buffer.start_time_ns + buffer.samples.size * buffer.ns_per_sample
                latest_end_ns = max(latest_end_ns, end_ns)

            if not recent:
                return None

            common_samples = min(window.size for window in recent.values())
            if common_samples <= 0:
                return None
            aligned = {sensor_id: window[-common_samples:] for sensor_id, window in recent.items()}
            return aligned, dominant_sample_rate_hz, latest_end_ns

    async def summarize_sensors(self, sensor_ids: list[str], now_ns: int) -> dict[str, float | int | None]:
        async with self._lock:
            active_sensor_count = 0
            sample_rate_counts: dict[int, int] = {}
            rms_values: list[float] = []
            latest_sample_time_ns: int | None = None
            for sensor_id in sensor_ids:
                buffer = self._buffers.get(sensor_id)
                if buffer is None or buffer.start_time_ns is None or buffer.samples.size == 0:
                    continue

                active_sensor_count += 1
                sample_rate_counts[buffer.sample_rate_hz] = sample_rate_counts.get(buffer.sample_rate_hz, 0) + 1
                end_ns = buffer.start_time_ns + buffer.samples.size * buffer.ns_per_sample
                latest_sample_time_ns = end_ns if latest_sample_time_ns is None else max(latest_sample_time_ns, end_ns)

                tail_samples = max(1, min(buffer.samples.size, buffer.sample_rate_hz // 2))
                tail = buffer.samples[-tail_samples:]
                rms_values.append(float(np.sqrt(np.mean(np.square(tail)) + 1e-12)))

            dominant_sample_rate_hz: int | None = None
            if sample_rate_counts:
                dominant_sample_rate_hz = max(sample_rate_counts.items(), key=lambda item: (item[1], item[0]))[0]

            age_seconds: float | None = None
            if latest_sample_time_ns is not None:
                age_seconds = max(0.0, (now_ns - latest_sample_time_ns) / 1_000_000_000.0)

            rms_value: float | None = None
            if rms_values:
                rms_value = float(np.mean(np.asarray(rms_values, dtype=np.float64)))

            return {
                "active_sensor_count": active_sensor_count,
                "sample_rate_hz": dominant_sample_rate_hz,
                "last_sample_time_ns": latest_sample_time_ns,
                "age_seconds": age_seconds,
                "rms": rms_value,
            }
