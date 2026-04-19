"""Time-indexed rolling buffers for each sensor stream."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class SensorStreamBuffer:
    sample_rate_hz: int
    max_duration_seconds: float
    max_samples: int = field(init=False)
    start_time_ns: int | None = field(init=False, default=None)
    samples: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0, dtype=np.float32))
    _timeline_origin_ns: int | None = field(init=False, default=None)
    _buffer_start_sample_index: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.max_samples = max(1, int(round(self.max_duration_seconds * self.sample_rate_hz)))

    def append(self, start_time_ns: int, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("SensorStreamBuffer expects mono samples")

        if self.start_time_ns is None:
            self._timeline_origin_ns = start_time_ns
            self._buffer_start_sample_index = 0
            self.start_time_ns = start_time_ns
            self.samples = samples.copy()
            self._prune()
            return

        if samples.size == 0:
            return

        current_start_sample_index = self._buffer_start_sample_index
        current_end_sample_index = current_start_sample_index + self.samples.size
        incoming_start_sample_index = self._time_to_sample_index(start_time_ns)
        incoming_end_sample_index = incoming_start_sample_index + samples.size

        merged_start_sample_index = min(current_start_sample_index, incoming_start_sample_index)
        merged_end_sample_index = max(current_end_sample_index, incoming_end_sample_index)
        merged = np.zeros(merged_end_sample_index - merged_start_sample_index, dtype=np.float32)

        existing_offset = current_start_sample_index - merged_start_sample_index
        merged[existing_offset : existing_offset + self.samples.size] = self.samples

        incoming_offset = incoming_start_sample_index - merged_start_sample_index
        # Late-arriving frames should overwrite previously padded zeros rather than
        # being discarded as overlap when HTTP delivery is slightly out of order.
        merged[incoming_offset : incoming_offset + samples.size] = samples

        self.samples = merged
        self._buffer_start_sample_index = merged_start_sample_index
        self._refresh_start_time_ns()
        self._prune()

    def _prune(self) -> None:
        if self.samples.size <= self.max_samples:
            return

        drop = self.samples.size - self.max_samples
        self.samples = self.samples[drop:]
        self._buffer_start_sample_index += drop
        self._refresh_start_time_ns()

    def get_window(self, center_time_ns: int, window_seconds: float) -> np.ndarray | None:
        if self.start_time_ns is None or self.samples.size == 0:
            return None

        window_samples = max(1, int(round(window_seconds * self.sample_rate_hz)))
        center_sample_index = self._time_to_sample_index(center_time_ns)
        start_sample_index = center_sample_index - (window_samples // 2)
        end_sample_index = start_sample_index + window_samples

        relative_start_index = start_sample_index - self._buffer_start_sample_index
        relative_end_index = end_sample_index - self._buffer_start_sample_index

        if relative_start_index < 0 or relative_end_index > self.samples.size:
            return None

        return self.samples[relative_start_index:relative_end_index].copy()

    def get_window_ending_at(self, end_time_ns: int, window_seconds: float) -> np.ndarray | None:
        if self.start_time_ns is None or self.samples.size == 0:
            return None

        window_samples = max(1, int(round(window_seconds * self.sample_rate_hz)))
        end_sample_index = self._time_to_sample_index(end_time_ns)
        start_sample_index = end_sample_index - window_samples
        end_offset_samples = end_sample_index - self._buffer_start_sample_index
        start_offset_samples = start_sample_index - self._buffer_start_sample_index

        # Clamp the start to the beginning of the buffer so partial windows (e.g.
        # when the buffer has less than classification_window_seconds of history) return
        # whatever audio IS available rather than None.  Returning None here caused every
        # sensor to fall back to the 80 ms localization window, which is far too short
        # for BirdNET and produced only "unknown" (0.0) classifications.
        # We still return None when the end is beyond what has been buffered because
        # that would require future samples.
        if end_offset_samples > self.samples.size:
            return None
        start_offset_samples = max(0, start_offset_samples)

        return self.samples[start_offset_samples:end_offset_samples].copy()

    def end_time_ns(self) -> int | None:
        if self.start_time_ns is None or self.samples.size == 0:
            return None
        return self._sample_index_to_time_ns(self._buffer_start_sample_index + self.samples.size)

    def _refresh_start_time_ns(self) -> None:
        if self._timeline_origin_ns is None:
            self.start_time_ns = None
            return
        self.start_time_ns = self._sample_index_to_time_ns(self._buffer_start_sample_index)

    def _time_to_sample_index(self, time_ns: int) -> int:
        if self._timeline_origin_ns is None:
            raise ValueError("SensorStreamBuffer timeline origin is not initialized")
        delta_ns = time_ns - self._timeline_origin_ns
        return self._round_divide(delta_ns * self.sample_rate_hz, 1_000_000_000)

    def _sample_index_to_time_ns(self, sample_index: int) -> int:
        if self._timeline_origin_ns is None:
            raise ValueError("SensorStreamBuffer timeline origin is not initialized")
        return self._timeline_origin_ns + self._round_divide(sample_index * 1_000_000_000, self.sample_rate_hz)

    @staticmethod
    def _round_divide(numerator: int, denominator: int) -> int:
        if numerator >= 0:
            return (numerator + denominator // 2) // denominator
        return -((-numerator + denominator // 2) // denominator)


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

    async def get_synchronized_window_ending_at(
        self,
        sensor_ids: list[str],
        end_time_ns: int,
        window_seconds: float,
        sample_rate_hz: int,
    ) -> dict[str, np.ndarray]:
        async with self._lock:
            result: dict[str, np.ndarray] = {}
            for sensor_id in sensor_ids:
                buffer = self._buffers.get(sensor_id)
                if buffer is None or buffer.sample_rate_hz != sample_rate_hz:
                    continue
                window = buffer.get_window_ending_at(end_time_ns=end_time_ns, window_seconds=window_seconds)
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
                end_ns = buffer.end_time_ns()
                if end_ns is None:
                    continue
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
                end_ns = buffer.end_time_ns()
                if end_ns is None:
                    continue
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
