"""Time-indexed rolling buffers for each sensor stream."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioCoverageStats:
    sample_count: int
    covered_samples: int
    missing_samples: int
    coverage_ratio: float
    missing_ratio: float
    max_gap_samples: int
    max_gap_seconds: float
    warning: bool
    degraded: bool

    def to_feature_summary(self, *, source_window_type: str) -> dict[str, float | int | bool | str]:
        return {
            "source_window_type": source_window_type,
            "sample_count": self.sample_count,
            "covered_samples": self.covered_samples,
            "missing_samples": self.missing_samples,
            "coverage_ratio": self.coverage_ratio,
            "missing_ratio": self.missing_ratio,
            "max_gap_seconds": self.max_gap_seconds,
            "warning": self.warning,
            "degraded": self.degraded,
        }


def _coverage_stats(coverage: np.ndarray, sample_rate_hz: int) -> AudioCoverageStats:
    coverage = np.asarray(coverage, dtype=np.bool_)
    sample_count = int(coverage.size)
    if sample_count <= 0:
        return AudioCoverageStats(
            sample_count=0,
            covered_samples=0,
            missing_samples=0,
            coverage_ratio=0.0,
            missing_ratio=1.0,
            max_gap_samples=0,
            max_gap_seconds=0.0,
            warning=True,
            degraded=True,
        )

    missing = ~coverage
    missing_samples = int(np.count_nonzero(missing))
    covered_samples = sample_count - missing_samples
    max_gap_samples = 0
    current_gap = 0
    for is_missing in missing:
        if bool(is_missing):
            current_gap += 1
            max_gap_samples = max(max_gap_samples, current_gap)
        else:
            current_gap = 0
    max_gap_seconds = max_gap_samples / float(sample_rate_hz)
    missing_ratio = missing_samples / float(sample_count)
    warning = missing_ratio > 0.01 or max_gap_seconds > 0.100
    degraded = missing_ratio > 0.05 or max_gap_seconds > 0.250
    return AudioCoverageStats(
        sample_count=sample_count,
        covered_samples=covered_samples,
        missing_samples=missing_samples,
        coverage_ratio=covered_samples / float(sample_count),
        missing_ratio=missing_ratio,
        max_gap_samples=max_gap_samples,
        max_gap_seconds=max_gap_seconds,
        warning=warning,
        degraded=degraded,
    )


@dataclass(slots=True)
class SensorStreamBuffer:
    sample_rate_hz: int
    max_duration_seconds: float
    max_samples: int = field(init=False)
    start_time_ns: int | None = field(init=False, default=None)
    samples: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0, dtype=np.float32))
    coverage: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0, dtype=np.bool_))
    _timeline_origin_ns: int | None = field(init=False, default=None)
    _timeline_origin_sample_index: int = field(init=False, default=0)
    _buffer_start_sample_index: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.max_samples = max(1, int(round(self.max_duration_seconds * self.sample_rate_hz)))

    def append(
        self,
        start_time_ns: int,
        samples: np.ndarray,
        *,
        start_sample_index: int | None = None,
        end_sample_index: int | None = None,
        end_time_ns: int | None = None,
        _protected_from_ns: Optional[int] = None,
    ) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("SensorStreamBuffer expects mono samples")
        explicit_sample_coverage = start_sample_index is not None
        if explicit_sample_coverage and end_sample_index is None:
            end_sample_index = start_sample_index + samples.size
        if explicit_sample_coverage and end_sample_index is not None:
            if end_sample_index < start_sample_index:
                raise ValueError("end_sample_index must be >= start_sample_index")
            if (end_sample_index - start_sample_index) != samples.size:
                raise ValueError("Explicit sample coverage must match appended sample count")

        if self.start_time_ns is None:
            self._timeline_origin_ns = start_time_ns
            self._timeline_origin_sample_index = start_sample_index or 0
            self._buffer_start_sample_index = start_sample_index or 0
            self.start_time_ns = start_time_ns
            self.samples = samples.copy()
            self.coverage = np.ones(samples.size, dtype=np.bool_)
            self._prune(_protected_from_ns)
            return

        if samples.size == 0:
            return

        current_start_sample_index = self._buffer_start_sample_index
        current_end_sample_index = current_start_sample_index + self.samples.size
        incoming_start_sample_index = (
            start_sample_index if explicit_sample_coverage else self._time_to_sample_index(start_time_ns)
        )
        if explicit_sample_coverage:
            expected_start_time_ns = self._sample_index_to_time_ns(incoming_start_sample_index)
            timestamp_correction_ns = abs(start_time_ns - expected_start_time_ns)
            reset_threshold_ns = int(round(self.max_duration_seconds * 1_000_000_000))
            if timestamp_correction_ns > reset_threshold_ns:
                # NTP/GPS lock can correct wall time while sample indices stay contiguous.
                # Re-anchor so recent-audio age follows the corrected node clock.
                self._timeline_origin_ns = start_time_ns
                self._timeline_origin_sample_index = start_sample_index or 0
                self._buffer_start_sample_index = start_sample_index or 0
                self.start_time_ns = start_time_ns
                self.samples = samples.copy()
                self.coverage = np.ones(samples.size, dtype=np.bool_)
                self._prune(_protected_from_ns)
                return

        # If the incoming frame is more than max_samples away from the current buffer
        # (either far ahead or far behind), reset the timeline rather than allocating
        # a huge zeros array. The common cause is an NTP sync that corrects a stale
        # build-timestamp epoch by hours, which would otherwise produce a multi-minute
        # silence gap and a potentially enormous intermediate allocation.
        if (incoming_start_sample_index > current_end_sample_index + self.max_samples
                or incoming_start_sample_index < current_start_sample_index - self.max_samples):
            self._timeline_origin_ns = start_time_ns
            self._timeline_origin_sample_index = start_sample_index or 0
            self._buffer_start_sample_index = start_sample_index or 0
            self.start_time_ns = start_time_ns
            self.samples = samples.copy()
            self.coverage = np.ones(samples.size, dtype=np.bool_)
            self._prune(_protected_from_ns)
            return

        # Snap to sample-contiguous when the apparent gap or overlap is smaller than
        # half a frame. The firmware derives each frame's start_time_ns from a
        # monotonic timer captured at DMA-IRQ time, so consecutive frames arrive with
        # sub-millisecond timing jitter. Without this snap, every frame whose jitter
        # rounds to >=1 sample later than the previous frame's end leaves a 1+ sample
        # zero hole that is audible as a click — and over a multi-second clip these
        # accumulate into the "scattered <1s gaps" reported in the UI snippet audio.
        # Drifts larger than half a frame indicate a genuine missing frame (DMA
        # overrun on the node, dropped HTTP publish, etc.) and are still zero-padded.
        if not explicit_sample_coverage:
            drift = incoming_start_sample_index - current_end_sample_index
            snap_tolerance = max(1, samples.size // 2)
            if -snap_tolerance < drift < snap_tolerance:
                incoming_start_sample_index = current_end_sample_index

        incoming_end_sample_index = end_sample_index if explicit_sample_coverage else incoming_start_sample_index + samples.size

        merged_start_sample_index = min(current_start_sample_index, incoming_start_sample_index)
        merged_end_sample_index = max(current_end_sample_index, incoming_end_sample_index)
        merged = np.zeros(merged_end_sample_index - merged_start_sample_index, dtype=np.float32)
        merged_coverage = np.zeros(merged_end_sample_index - merged_start_sample_index, dtype=np.bool_)

        existing_offset = current_start_sample_index - merged_start_sample_index
        merged[existing_offset : existing_offset + self.samples.size] = self.samples
        merged_coverage[existing_offset : existing_offset + self.coverage.size] = self.coverage

        incoming_offset = incoming_start_sample_index - merged_start_sample_index
        # Late-arriving frames should overwrite previously padded zeros rather than
        # being discarded as overlap when HTTP delivery is slightly out of order.
        merged[incoming_offset : incoming_offset + samples.size] = samples
        merged_coverage[incoming_offset : incoming_offset + samples.size] = True

        self.samples = merged
        self.coverage = merged_coverage
        self._buffer_start_sample_index = merged_start_sample_index
        self._refresh_start_time_ns()
        self._prune(_protected_from_ns)

    def get_range(self, start_ns: int, end_ns: int) -> np.ndarray:
        """Return samples for [start_ns, end_ns), zero-padding any coverage gaps.

        Returns an empty array if the entire range lies outside the buffer.
        """
        if self.start_time_ns is None or self.samples.size == 0:
            return np.zeros(0, dtype=np.float32)

        start_idx = self._time_to_sample_index(start_ns)
        end_idx = self._time_to_sample_index(end_ns)
        n_out = max(0, end_idx - start_idx)
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)

        out = np.zeros(n_out, dtype=np.float32)
        cov_out = np.zeros(n_out, dtype=np.bool_)

        buf_start = self._buffer_start_sample_index
        buf_end = buf_start + self.samples.size

        # Overlap of requested range with buffered range.
        overlap_start = max(start_idx, buf_start)
        overlap_end = min(end_idx, buf_end)

        if overlap_start < overlap_end:
            out_offset = overlap_start - start_idx
            buf_offset = overlap_start - buf_start
            length = overlap_end - overlap_start
            out[out_offset : out_offset + length] = self.samples[buf_offset : buf_offset + length]
            cov_out[out_offset : out_offset + length] = self.coverage[buf_offset : buf_offset + length]

        # Zero-out any uncovered samples (coverage gaps inside the buffered range).
        out[~cov_out] = 0.0
        return out

    def _prune(self, protected_from_ns: Optional[int] = None) -> None:
        if self.samples.size <= self.max_samples:
            return

        drop = self.samples.size - self.max_samples

        # If a pin is active, do not evict samples at or after the pin timestamp.
        if protected_from_ns is not None and self._timeline_origin_ns is not None:
            protected_idx = self._time_to_sample_index(protected_from_ns)
            max_drop = max(0, protected_idx - self._buffer_start_sample_index)
            drop = min(drop, max_drop)
            if drop <= 0:
                return

        self.samples = self.samples[drop:]
        self.coverage = self.coverage[drop:]
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

    def get_window_coverage_stats(self, center_time_ns: int, window_seconds: float) -> AudioCoverageStats | None:
        coverage = self._coverage_window(center_time_ns=center_time_ns, window_seconds=window_seconds)
        if coverage is None:
            return None
        return _coverage_stats(coverage, self.sample_rate_hz)

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

    def get_window_ending_at_coverage_stats(
        self,
        end_time_ns: int,
        window_seconds: float,
    ) -> AudioCoverageStats | None:
        coverage = self._coverage_window_ending_at(end_time_ns=end_time_ns, window_seconds=window_seconds)
        if coverage is None:
            return None
        return _coverage_stats(coverage, self.sample_rate_hz)

    def end_time_ns(self) -> int | None:
        if self.start_time_ns is None or self.samples.size == 0:
            return None
        return self._sample_index_to_time_ns(self._buffer_start_sample_index + self.samples.size)

    def _refresh_start_time_ns(self) -> None:
        if self._timeline_origin_ns is None:
            self.start_time_ns = None
            return
        self.start_time_ns = self._sample_index_to_time_ns(self._buffer_start_sample_index)

    def _coverage_window(self, center_time_ns: int, window_seconds: float) -> np.ndarray | None:
        if self.start_time_ns is None or self.coverage.size == 0:
            return None
        window_samples = max(1, int(round(window_seconds * self.sample_rate_hz)))
        center_sample_index = self._time_to_sample_index(center_time_ns)
        start_sample_index = center_sample_index - (window_samples // 2)
        end_sample_index = start_sample_index + window_samples
        relative_start_index = start_sample_index - self._buffer_start_sample_index
        relative_end_index = end_sample_index - self._buffer_start_sample_index
        if relative_start_index < 0 or relative_end_index > self.coverage.size:
            return None
        return self.coverage[relative_start_index:relative_end_index].copy()

    def _coverage_window_ending_at(self, end_time_ns: int, window_seconds: float) -> np.ndarray | None:
        if self.start_time_ns is None or self.coverage.size == 0:
            return None
        window_samples = max(1, int(round(window_seconds * self.sample_rate_hz)))
        end_sample_index = self._time_to_sample_index(end_time_ns)
        start_sample_index = end_sample_index - window_samples
        end_offset_samples = end_sample_index - self._buffer_start_sample_index
        start_offset_samples = start_sample_index - self._buffer_start_sample_index
        if end_offset_samples > self.coverage.size:
            return None
        start_offset_samples = max(0, start_offset_samples)
        return self.coverage[start_offset_samples:end_offset_samples].copy()

    def _time_to_sample_index(self, time_ns: int) -> int:
        if self._timeline_origin_ns is None:
            raise ValueError("SensorStreamBuffer timeline origin is not initialized")
        delta_ns = time_ns - self._timeline_origin_ns
        return self._timeline_origin_sample_index + self._round_divide(delta_ns * self.sample_rate_hz, 1_000_000_000)

    def _sample_index_to_time_ns(self, sample_index: int) -> int:
        if self._timeline_origin_ns is None:
            raise ValueError("SensorStreamBuffer timeline origin is not initialized")
        return self._timeline_origin_ns + self._round_divide(
            (sample_index - self._timeline_origin_sample_index) * 1_000_000_000,
            self.sample_rate_hz,
        )

    @staticmethod
    def _round_divide(numerator: int, denominator: int) -> int:
        if numerator >= 0:
            return (numerator + denominator // 2) // denominator
        return -((-numerator + denominator // 2) // denominator)


class MultiSensorBuffer:
    # Maximum duration a pin can protect (mirrors Rust sidecar's hard cap).
    _PIN_MAX_DURATION_NS: int = 300_000_000_000  # 5 minutes

    def __init__(self, max_duration_seconds: float) -> None:
        self.max_duration_seconds = max_duration_seconds
        self._buffers: dict[str, SensorStreamBuffer] = {}
        self._lock = asyncio.Lock()
        self._pins: dict[str, int] = {}  # session_id → start_ns

    def pin(self, session_id: str, start_ns: int) -> None:
        """Prevent the buffer from evicting samples at or after start_ns.

        Enforces a 5-minute hard cap so a stuck session cannot cause OOM.

        Performance note: SensorStreamBuffer uses a grow-and-prune pattern.
        While a pin is active, _prune() cannot evict old frames, so the buffer
        grows by one frame on every append() call.  Each append() allocates and
        copies (current_buffer_size + new_frame) samples — O(N) per frame.
        At 16 kHz × 4 ch × 5 min this peaks at ~19 MB per append on a 300 s
        recording, which will block the event loop for several milliseconds per
        frame on a Raspberry Pi and may cause ingest frame drops at that point.
        For production Pi captures use the Rust ingest sidecar (journal mode),
        which avoids this by writing audio directly to tmpfs.
        """
        floor = time.time_ns() - self._PIN_MAX_DURATION_NS
        self._pins[session_id] = max(start_ns, floor)
        if self.max_duration_seconds < 60.0:
            logger.warning(
                "capture pin started on a buffer configured for %.0f s of live data; "
                "buffer will grow during recording (peak ~%.0f MB per channel at 5 min / 16 kHz). "
                "Use Rust ingest+journal mode for production Pi captures.",
                self.max_duration_seconds,
                300 * 16000 * 4 / 1_000_000,
            )

    def unpin(self, session_id: str) -> None:
        """Release a pin, allowing normal pruning to resume."""
        self._pins.pop(session_id, None)

    def _oldest_pin_ns(self) -> Optional[int]:
        if not self._pins:
            return None
        earliest_recorded = min(self._pins.values())
        # Rolling TTL: even if unpin() is never called, never protect data
        # older than _PIN_MAX_DURATION_NS from the current moment.  Without
        # this, a stuck pin (unhandled exception or task cancellation before
        # unpin()) would allow the buffer to grow without bound past the
        # initial 5-minute cap enforced at pin creation time.
        ttl_floor = time.time_ns() - self._PIN_MAX_DURATION_NS
        return max(earliest_recorded, ttl_floor)

    async def extract_range(
        self,
        sensor_ids: list[str],
        start_ns: int,
        end_ns: int,
    ) -> tuple[np.ndarray, int, dict]:
        """Assemble a (N_channels, N_samples) float32 array from pinned buffers.

        Returns (channels, sample_rate_hz, sync_diagnostics) with the same
        contract as IamfPipeline._extract_audio so the pipeline step is
        a transparent drop-in replacement for the Rust journal path.

        The lock is held only briefly to snapshot array references and index
        metadata. The actual numpy slicing happens outside the lock so that
        incoming real-time frames are not blocked during post-processing of a
        potentially multi-minute recording.  CPython reference counting keeps
        the snapshotted numpy arrays alive even after _prune() replaces them.
        """
        # ── Phase 1: snapshot under lock (cheap — just captures references) ───
        # Each entry is (samples_ref, coverage_ref, buf_start_idx, sr, origin_ns, origin_sample_idx)
        # or None when the sensor has no data.
        snapshots: list[Optional[tuple]] = []
        sample_rate_hz = 0
        actual_start_ns = start_ns
        actual_end_ns = start_ns

        async with self._lock:
            for sensor_id in sensor_ids:
                buf = self._buffers.get(sensor_id)
                if buf is None or buf.samples.size == 0 or buf._timeline_origin_ns is None:
                    snapshots.append(None)
                    continue
                if sample_rate_hz == 0:
                    sample_rate_hz = buf.sample_rate_hz
                    if buf.start_time_ns is not None:
                        actual_start_ns = max(start_ns, buf.start_time_ns)
                    end_buf = buf.end_time_ns()
                    if end_buf is not None:
                        actual_end_ns = min(end_ns, end_buf)
                # Snapshot the array *references* (not copies) plus index metadata.
                # After lock release, buf.samples may be replaced by append(), but
                # CPython keeps the old array alive as long as we hold this reference.
                snapshots.append((
                    buf.samples,                       # numpy ref — safe to read after unlock
                    buf.coverage,
                    buf._buffer_start_sample_index,
                    buf.sample_rate_hz,
                    buf._timeline_origin_ns,
                    buf._timeline_origin_sample_index,
                ))

        # ── Phase 2: slice from snapshots — lock NOT held ─────────────────────
        if sample_rate_hz == 0:
            raise RuntimeError("no audio data found in Python buffer for time range")

        def _snap_get_range(snap: tuple) -> np.ndarray:
            samples_ref, coverage_ref, buf_start_idx, sr, origin_ns, origin_sample_idx = snap
            # Replicate SensorStreamBuffer._time_to_sample_index arithmetic.
            def _ts_to_idx(ns: int) -> int:
                delta = ns - origin_ns
                return origin_sample_idx + (delta * sr + 500_000_000) // 1_000_000_000
            start_idx = _ts_to_idx(start_ns)
            end_idx = _ts_to_idx(end_ns)
            n_out = max(0, end_idx - start_idx)
            if n_out == 0:
                return np.zeros(0, dtype=np.float32)
            out = np.zeros(n_out, dtype=np.float32)
            overlap_start = max(start_idx, buf_start_idx)
            overlap_end = min(end_idx, buf_start_idx + samples_ref.size)
            if overlap_start < overlap_end:
                out_off = overlap_start - start_idx
                buf_off = overlap_start - buf_start_idx
                length = overlap_end - overlap_start
                chunk = samples_ref[buf_off : buf_off + length]
                cov_chunk = coverage_ref[buf_off : buf_off + length]
                out[out_off : out_off + length] = chunk
                out[out_off : out_off + length][~cov_chunk] = 0.0
            return out

        channels_list: list[np.ndarray] = []
        for snap in snapshots:
            if snap is None:
                channels_list.append(np.zeros(0, dtype=np.float32))
            else:
                channels_list.append(_snap_get_range(snap))

        if all(c.size == 0 for c in channels_list):
            raise RuntimeError("no audio data after range extraction")

        n_samples = max(c.size for c in channels_list)
        if n_samples == 0:
            raise RuntimeError("no audio data after range extraction")

        aligned: list[np.ndarray] = []
        for ch in channels_list:
            if ch.size < n_samples:
                padded = np.zeros(n_samples, dtype=np.float32)
                padded[: ch.size] = ch
                aligned.append(padded)
            else:
                aligned.append(ch[:n_samples])

        channels = np.vstack(aligned)  # (N_ch, N_samples)

        sync_diag: dict = {
            "requested_start_ns": start_ns,
            "requested_end_ns": end_ns,
            "actual_audio_start_ns": actual_start_ns,
            "actual_audio_end_ns": actual_end_ns,
            "capture_rate_hz": sample_rate_hz,
            "n_segments": 1,
            "n_samples": n_samples,
            "source": "python_buffer",
        }
        return channels, sample_rate_hz, sync_diag

    async def append(
        self,
        sensor_id: str,
        sample_rate_hz: int,
        start_time_ns: int,
        samples: np.ndarray,
        *,
        start_sample_index: int | None = None,
        end_sample_index: int | None = None,
        end_time_ns: int | None = None,
    ) -> None:
        async with self._lock:
            buffer = self._buffers.get(sensor_id)
            if buffer is None or buffer.sample_rate_hz != sample_rate_hz:
                buffer = SensorStreamBuffer(sample_rate_hz=sample_rate_hz, max_duration_seconds=self.max_duration_seconds)
                self._buffers[sensor_id] = buffer
            buffer.append(
                start_time_ns=start_time_ns,
                samples=samples,
                start_sample_index=start_sample_index,
                end_sample_index=end_sample_index,
                end_time_ns=end_time_ns,
                _protected_from_ns=self._oldest_pin_ns(),
            )

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

    async def get_synchronized_window_coverage_stats(
        self,
        sensor_ids: list[str],
        center_time_ns: int,
        window_seconds: float,
        sample_rate_hz: int,
    ) -> dict[str, AudioCoverageStats]:
        async with self._lock:
            result: dict[str, AudioCoverageStats] = {}
            for sensor_id in sensor_ids:
                buffer = self._buffers.get(sensor_id)
                if buffer is None or buffer.sample_rate_hz != sample_rate_hz:
                    continue
                stats = buffer.get_window_coverage_stats(
                    center_time_ns=center_time_ns,
                    window_seconds=window_seconds,
                )
                if stats is not None:
                    result[sensor_id] = stats
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

    async def get_synchronized_window_ending_at_coverage_stats(
        self,
        sensor_ids: list[str],
        end_time_ns: int,
        window_seconds: float,
        sample_rate_hz: int,
    ) -> dict[str, AudioCoverageStats]:
        async with self._lock:
            result: dict[str, AudioCoverageStats] = {}
            for sensor_id in sensor_ids:
                buffer = self._buffers.get(sensor_id)
                if buffer is None or buffer.sample_rate_hz != sample_rate_hz:
                    continue
                stats = buffer.get_window_ending_at_coverage_stats(
                    end_time_ns=end_time_ns,
                    window_seconds=window_seconds,
                )
                if stats is not None:
                    result[sensor_id] = stats
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
            coverage_ratios: list[float] = []
            missing_ratios: list[float] = []
            max_gap_seconds = 0.0
            max_buffer_samples = 0
            max_buffer_seconds = 0.0
            latest_sample_time_ns: int | None = None
            for sensor_id in sensor_ids:
                buffer = self._buffers.get(sensor_id)
                if buffer is None or buffer.start_time_ns is None or buffer.samples.size == 0:
                    continue

                active_sensor_count += 1
                sample_rate_counts[buffer.sample_rate_hz] = sample_rate_counts.get(buffer.sample_rate_hz, 0) + 1
                max_buffer_samples = max(max_buffer_samples, buffer.max_samples)
                max_buffer_seconds = max(max_buffer_seconds, buffer.max_duration_seconds)
                end_ns = buffer.end_time_ns()
                if end_ns is None:
                    continue
                latest_sample_time_ns = end_ns if latest_sample_time_ns is None else max(latest_sample_time_ns, end_ns)

                tail_samples = max(1, min(buffer.samples.size, buffer.sample_rate_hz // 2))
                tail = buffer.samples[-tail_samples:]
                rms_values.append(float(np.sqrt(np.mean(np.square(tail)) + 1e-12)))
                coverage_tail = buffer.coverage[-tail_samples:]
                coverage_stats = _coverage_stats(coverage_tail, buffer.sample_rate_hz)
                coverage_ratios.append(coverage_stats.coverage_ratio)
                missing_ratios.append(coverage_stats.missing_ratio)
                max_gap_seconds = max(max_gap_seconds, coverage_stats.max_gap_seconds)

            dominant_sample_rate_hz: int | None = None
            if sample_rate_counts:
                dominant_sample_rate_hz = max(sample_rate_counts.items(), key=lambda item: (item[1], item[0]))[0]

            age_seconds: float | None = None
            if latest_sample_time_ns is not None:
                age_seconds = max(0.0, (now_ns - latest_sample_time_ns) / 1_000_000_000.0)

            rms_value: float | None = None
            if rms_values:
                rms_value = float(np.mean(np.asarray(rms_values, dtype=np.float64)))

            coverage_ratio: float | None = None
            missing_ratio: float | None = None
            if coverage_ratios:
                coverage_ratio = float(np.mean(np.asarray(coverage_ratios, dtype=np.float64)))
            if missing_ratios:
                missing_ratio = float(np.mean(np.asarray(missing_ratios, dtype=np.float64)))

            return {
                "active_sensor_count": active_sensor_count,
                "sample_rate_hz": dominant_sample_rate_hz,
                "last_sample_time_ns": latest_sample_time_ns,
                "age_seconds": age_seconds,
                "rms": rms_value,
                "recent_coverage_ratio": coverage_ratio,
                "recent_missing_ratio": missing_ratio,
                "recent_max_gap_seconds": max_gap_seconds if coverage_ratios else None,
                "max_buffer_samples": max_buffer_samples or None,
                "max_buffer_seconds": max_buffer_seconds or None,
            }
