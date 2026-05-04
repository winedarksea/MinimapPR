"""Chunked per-session audio capture buffers for IAMF recording.

These buffers mirror incoming frame-sized chunks so the live rolling
MultiSensorBuffer can keep pruning normally during a recording session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class _CapturedChunk:
    start_sample_index: int
    samples: np.ndarray


@dataclass(slots=True)
class _CapturedSensorStream:
    sample_rate_hz: int
    max_duration_seconds: float
    _timeline_origin_ns: int | None = field(init=False, default=None)
    _timeline_origin_sample_index: int = field(init=False, default=0)
    _chunks: list[_CapturedChunk] = field(init=False, default_factory=list)
    _start_sample_index: int | None = field(init=False, default=None)
    _end_sample_index: int | None = field(init=False, default=None)

    def append(
        self,
        start_time_ns: int,
        samples: np.ndarray,
        *,
        start_sample_index: int | None = None,
        end_sample_index: int | None = None,
    ) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("CapturedSensorStream expects mono samples")
        if samples.size == 0:
            return

        explicit_sample_coverage = start_sample_index is not None
        if explicit_sample_coverage and end_sample_index is None:
            end_sample_index = start_sample_index + samples.size
        if explicit_sample_coverage and end_sample_index is not None:
            if end_sample_index < start_sample_index:
                raise ValueError("end_sample_index must be >= start_sample_index")
            if (end_sample_index - start_sample_index) != samples.size:
                raise ValueError("Explicit sample coverage must match appended sample count")

        if self._timeline_origin_ns is None:
            self._timeline_origin_ns = start_time_ns
            self._timeline_origin_sample_index = start_sample_index or 0
            incoming_start_sample_index = start_sample_index or 0
        elif explicit_sample_coverage:
            expected_start_time_ns = self._sample_index_to_time_ns(start_sample_index)
            reset_threshold_ns = int(round(self.max_duration_seconds * 1_000_000_000))
            timestamp_correction_ns = abs(start_time_ns - expected_start_time_ns)
            if timestamp_correction_ns > reset_threshold_ns:
                self._timeline_origin_ns = start_time_ns
                self._timeline_origin_sample_index = start_sample_index or 0
                self._chunks.clear()
                self._start_sample_index = None
                self._end_sample_index = None
            incoming_start_sample_index = start_sample_index
        else:
            incoming_start_sample_index = self._time_to_sample_index(start_time_ns)
            if self._end_sample_index is not None:
                drift = incoming_start_sample_index - self._end_sample_index
                snap_tolerance = max(1, samples.size // 2)
                if -snap_tolerance < drift < snap_tolerance:
                    incoming_start_sample_index = self._end_sample_index

        self._chunks.append(
            _CapturedChunk(
                start_sample_index=incoming_start_sample_index,
                samples=samples.copy(),
            )
        )
        self._start_sample_index = (
            incoming_start_sample_index
            if self._start_sample_index is None
            else min(self._start_sample_index, incoming_start_sample_index)
        )
        chunk_end_sample_index = incoming_start_sample_index + samples.size
        self._end_sample_index = (
            chunk_end_sample_index
            if self._end_sample_index is None
            else max(self._end_sample_index, chunk_end_sample_index)
        )

    @property
    def start_time_ns(self) -> int | None:
        if self._timeline_origin_ns is None or self._start_sample_index is None:
            return None
        return self._sample_index_to_time_ns(self._start_sample_index)

    def end_time_ns(self) -> int | None:
        if self._timeline_origin_ns is None or self._end_sample_index is None:
            return None
        return self._sample_index_to_time_ns(self._end_sample_index)

    def render_range(self, start_ns: int, end_ns: int) -> tuple[np.ndarray, np.ndarray]:
        if self._timeline_origin_ns is None:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.bool_)

        start_idx = self._time_to_sample_index(start_ns)
        end_idx = self._time_to_sample_index(end_ns)
        n_out = max(0, end_idx - start_idx)
        if n_out == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.bool_)

        out = np.zeros(n_out, dtype=np.float32)
        coverage = np.zeros(n_out, dtype=np.bool_)

        for chunk in self._chunks:
            chunk_end = chunk.start_sample_index + chunk.samples.size
            overlap_start = max(start_idx, chunk.start_sample_index)
            overlap_end = min(end_idx, chunk_end)
            if overlap_start >= overlap_end:
                continue
            out_offset = overlap_start - start_idx
            chunk_offset = overlap_start - chunk.start_sample_index
            length = overlap_end - overlap_start
            out[out_offset : out_offset + length] = chunk.samples[chunk_offset : chunk_offset + length]
            coverage[out_offset : out_offset + length] = True

        return out, coverage

    def _time_to_sample_index(self, time_ns: int) -> int:
        if self._timeline_origin_ns is None:
            raise ValueError("CapturedSensorStream timeline origin is not initialized")
        delta_ns = time_ns - self._timeline_origin_ns
        return self._timeline_origin_sample_index + self._round_divide(
            delta_ns * self.sample_rate_hz,
            1_000_000_000,
        )

    def _sample_index_to_time_ns(self, sample_index: int) -> int:
        if self._timeline_origin_ns is None:
            raise ValueError("CapturedSensorStream timeline origin is not initialized")
        return self._timeline_origin_ns + self._round_divide(
            (sample_index - self._timeline_origin_sample_index) * 1_000_000_000,
            self.sample_rate_hz,
        )

    @staticmethod
    def _round_divide(numerator: int, denominator: int) -> int:
        if numerator >= 0:
            return (numerator + denominator // 2) // denominator
        return -((-numerator + denominator // 2) // denominator)


class ChunkedCaptureSessionBuffer:
    """Per-session raw capture mirror with O(frame) append cost."""

    def __init__(self, sensor_ids: list[str], max_duration_seconds: float) -> None:
        self.sensor_ids = list(sensor_ids)
        self._sensor_id_set = set(sensor_ids)
        self._max_duration_seconds = max_duration_seconds
        self._streams: dict[str, _CapturedSensorStream] = {}

    def wants_sensor(self, sensor_id: str) -> bool:
        return sensor_id in self._sensor_id_set

    def append(
        self,
        *,
        sensor_id: str,
        sample_rate_hz: int,
        start_time_ns: int,
        samples: np.ndarray,
        start_sample_index: int | None = None,
        end_sample_index: int | None = None,
    ) -> None:
        if sensor_id not in self._sensor_id_set:
            return
        stream = self._streams.get(sensor_id)
        if stream is None:
            stream = _CapturedSensorStream(
                sample_rate_hz=sample_rate_hz,
                max_duration_seconds=self._max_duration_seconds,
            )
            self._streams[sensor_id] = stream
        elif stream.sample_rate_hz != sample_rate_hz:
            raise ValueError(
                f"Capture session sample rate mismatch for {sensor_id}: "
                f"{stream.sample_rate_hz} != {sample_rate_hz}"
            )
        stream.append(
            start_time_ns=start_time_ns,
            samples=samples,
            start_sample_index=start_sample_index,
            end_sample_index=end_sample_index,
        )

    def extract_range(
        self,
        sensor_ids: list[str],
        start_ns: int,
        end_ns: int,
    ) -> tuple[np.ndarray, int, dict]:
        sample_rate_hz = 0
        actual_start_ns = start_ns
        actual_end_ns = start_ns
        rendered_channels: list[np.ndarray] = []
        coverages: list[np.ndarray] = []

        for sensor_id in sensor_ids:
            stream = self._streams.get(sensor_id)
            if stream is None:
                rendered_channels.append(np.zeros(0, dtype=np.float32))
                coverages.append(np.zeros(0, dtype=np.bool_))
                continue
            if sample_rate_hz == 0:
                sample_rate_hz = stream.sample_rate_hz
                if stream.start_time_ns is not None:
                    actual_start_ns = max(start_ns, stream.start_time_ns)
                stream_end_ns = stream.end_time_ns()
                if stream_end_ns is not None:
                    actual_end_ns = min(end_ns, stream_end_ns)
            channel, coverage = stream.render_range(start_ns, end_ns)
            rendered_channels.append(channel)
            coverages.append(coverage)

        if sample_rate_hz == 0:
            raise RuntimeError("no audio data found in capture session buffer for time range")
        if all(channel.size == 0 for channel in rendered_channels):
            raise RuntimeError("no audio data after range extraction")

        n_samples = max(channel.size for channel in rendered_channels)
        if n_samples == 0:
            raise RuntimeError("no audio data after range extraction")

        aligned_channels: list[np.ndarray] = []
        aligned_coverages: list[np.ndarray] = []
        for channel, coverage in zip(rendered_channels, coverages, strict=False):
            if channel.size < n_samples:
                padded_channel = np.zeros(n_samples, dtype=np.float32)
                padded_coverage = np.zeros(n_samples, dtype=np.bool_)
                padded_channel[: channel.size] = channel
                padded_coverage[: coverage.size] = coverage
                aligned_channels.append(padded_channel)
                aligned_coverages.append(padded_coverage)
            else:
                aligned_channels.append(channel[:n_samples])
                aligned_coverages.append(coverage[:n_samples])

        channels = np.vstack(aligned_channels)
        channel_coverage_ratios: list[float] = []
        coverage_warnings: list[str] = []
        for index, coverage in enumerate(aligned_coverages):
            coverage_ratio = 0.0 if coverage.size == 0 else float(np.count_nonzero(coverage)) / float(n_samples)
            channel_coverage_ratios.append(coverage_ratio)
            if coverage_ratio < 0.5:
                sensor_label = sensor_ids[index] if index < len(sensor_ids) else f"ch{index}"
                coverage_warnings.append(
                    f"sensor {sensor_label} has {coverage_ratio:.0%} coverage in the requested range"
                )

        sync_diag = {
            "requested_start_ns": start_ns,
            "requested_end_ns": end_ns,
            "actual_audio_start_ns": actual_start_ns,
            "actual_audio_end_ns": actual_end_ns,
            "capture_rate_hz": sample_rate_hz,
            "n_segments": sum(len(stream._chunks) for stream in self._streams.values()),
            "n_samples": n_samples,
            "source": "python_capture_session",
            "channel_coverage_ratios": channel_coverage_ratios,
            "coverage_warnings": coverage_warnings,
        }
        return channels, sample_rate_hz, sync_diag