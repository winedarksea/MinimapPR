"""Time-indexed rolling buffers for each sensor stream."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from minimappr.core.capture_audio_buffer import ChunkedCaptureSessionBuffer

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

    def to_json(self) -> dict[str, int | float | bool]:
        return {
            "sample_count": self.sample_count,
            "covered_samples": self.covered_samples,
            "missing_samples": self.missing_samples,
            "coverage_ratio": self.coverage_ratio,
            "missing_ratio": self.missing_ratio,
            "max_gap_samples": self.max_gap_samples,
            "max_gap_seconds": self.max_gap_seconds,
            "warning": self.warning,
            "degraded": self.degraded,
        }

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
    if missing_samples > 0:
        padded = np.empty(missing.size + 2, dtype=np.bool_)
        padded[0] = False
        padded[1:-1] = missing
        padded[-1] = False
        edges = np.diff(padded.view(np.int8))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        max_gap_samples = int(np.max(ends - starts))
    else:
        max_gap_samples = 0
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


@dataclass(frozen=True, slots=True)
class _BufferState:
    """Immutable snapshot of a SensorStreamBuffer's mutable timeline state.

    append()/prune compute the next state locally and publish it with a single
    attribute assignment (atomic under the GIL), so a reader on another thread
    sees either the old or the new state — never new `samples` paired with a
    stale `buffer_start_sample_index`.
    """

    samples: np.ndarray
    coverage: np.ndarray
    buffer_start_sample_index: int
    timeline_origin_ns: int | None
    timeline_origin_sample_index: int
    start_time_ns: int | None


def _empty_buffer_state() -> _BufferState:
    return _BufferState(
        samples=np.zeros(0, dtype=np.float32),
        coverage=np.zeros(0, dtype=np.bool_),
        buffer_start_sample_index=0,
        timeline_origin_ns=None,
        timeline_origin_sample_index=0,
        start_time_ns=None,
    )


class SensorStreamBuffer:
    __slots__ = ("sample_rate_hz", "max_duration_seconds", "max_samples", "reanchor_count", "_state")

    def __init__(self, sample_rate_hz: int, max_duration_seconds: float) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.max_duration_seconds = max_duration_seconds
        self.max_samples = max(1, int(round(max_duration_seconds * sample_rate_hz)))
        # Diagnostic counter: incremented when append() resets the timeline anchor
        # (NTP/GPS correction or a frame so far ahead/behind that allocating the
        # in-between zero gap would be unbounded). Long-uptime drift between this
        # and the rate-mismatch wipe count was the leading hypothesis for the
        # 2026-05 silent-window-drop incident — see the corresponding counter on
        # MultiSensorBuffer for wipes, which survive buffer replacement.
        self.reanchor_count = 0
        self._state = _empty_buffer_state()

    @property
    def samples(self) -> np.ndarray:
        return self._state.samples

    @property
    def coverage(self) -> np.ndarray:
        return self._state.coverage

    @property
    def start_time_ns(self) -> int | None:
        return self._state.start_time_ns

    @property
    def _buffer_start_sample_index(self) -> int:
        return self._state.buffer_start_sample_index

    @property
    def _timeline_origin_ns(self) -> int | None:
        return self._state.timeline_origin_ns

    @property
    def _timeline_origin_sample_index(self) -> int:
        return self._state.timeline_origin_sample_index

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

        state = self._state
        if state.start_time_ns is None:
            self._state = self._pruned(
                self._anchored_state(start_time_ns, samples, start_sample_index),
                _protected_from_ns,
            )
            return

        if samples.size == 0:
            return

        current_start_sample_index = state.buffer_start_sample_index
        current_end_sample_index = current_start_sample_index + state.samples.size
        incoming_start_sample_index = (
            start_sample_index if explicit_sample_coverage else self._time_to_sample_index(state, start_time_ns)
        )
        if explicit_sample_coverage:
            expected_start_time_ns = self._sample_index_to_time_ns(state, incoming_start_sample_index)
            timestamp_correction_ns = abs(start_time_ns - expected_start_time_ns)
            reset_threshold_ns = int(round(self.max_duration_seconds * 1_000_000_000))
            if timestamp_correction_ns > reset_threshold_ns:
                # NTP/GPS lock can correct wall time while sample indices stay contiguous.
                # Re-anchor so recent-audio age follows the corrected node clock.
                self.reanchor_count += 1
                logger.warning(
                    "SensorStreamBuffer re-anchor (timestamp correction)",
                    extra={
                        "sample_rate_hz": self.sample_rate_hz,
                        "timestamp_correction_ns": timestamp_correction_ns,
                        "reset_threshold_ns": reset_threshold_ns,
                        "new_start_time_ns": start_time_ns,
                        "reanchor_count": self.reanchor_count,
                    },
                )
                self._state = self._pruned(
                    self._anchored_state(start_time_ns, samples, start_sample_index),
                    _protected_from_ns,
                )
                return

        # If the incoming frame is more than max_samples away from the current buffer
        # (either far ahead or far behind), reset the timeline rather than allocating
        # a huge zeros array. The common cause is an NTP sync that corrects a stale
        # build-timestamp epoch by hours, which would otherwise produce a multi-minute
        # silence gap and a potentially enormous intermediate allocation.
        if (incoming_start_sample_index > current_end_sample_index + self.max_samples
                or incoming_start_sample_index < current_start_sample_index - self.max_samples):
            self.reanchor_count += 1
            logger.warning(
                "SensorStreamBuffer re-anchor (out-of-bounds frame)",
                extra={
                    "sample_rate_hz": self.sample_rate_hz,
                    "incoming_start_sample_index": incoming_start_sample_index,
                    "current_start_sample_index": current_start_sample_index,
                    "current_end_sample_index": current_end_sample_index,
                    "max_samples": self.max_samples,
                    "new_start_time_ns": start_time_ns,
                    "reanchor_count": self.reanchor_count,
                },
            )
            self._state = self._pruned(
                self._anchored_state(start_time_ns, samples, start_sample_index),
                _protected_from_ns,
            )
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
        merged[existing_offset : existing_offset + state.samples.size] = state.samples
        merged_coverage[existing_offset : existing_offset + state.coverage.size] = state.coverage

        incoming_offset = incoming_start_sample_index - merged_start_sample_index
        # Late-arriving frames should overwrite previously padded zeros rather than
        # being discarded as overlap when HTTP delivery is slightly out of order.
        merged[incoming_offset : incoming_offset + samples.size] = samples
        merged_coverage[incoming_offset : incoming_offset + samples.size] = True

        self._state = self._pruned(
            _BufferState(
                samples=merged,
                coverage=merged_coverage,
                buffer_start_sample_index=merged_start_sample_index,
                timeline_origin_ns=state.timeline_origin_ns,
                timeline_origin_sample_index=state.timeline_origin_sample_index,
                start_time_ns=self._sample_index_to_time_ns(state, merged_start_sample_index),
            ),
            _protected_from_ns,
        )

    def _anchored_state(
        self,
        start_time_ns: int,
        samples: np.ndarray,
        start_sample_index: int | None,
    ) -> _BufferState:
        anchor_sample_index = start_sample_index or 0
        return _BufferState(
            samples=samples.copy(),
            coverage=np.ones(samples.size, dtype=np.bool_),
            buffer_start_sample_index=anchor_sample_index,
            timeline_origin_ns=start_time_ns,
            timeline_origin_sample_index=anchor_sample_index,
            start_time_ns=start_time_ns,
        )

    def get_range(self, start_ns: int, end_ns: int) -> np.ndarray:
        """Return samples for [start_ns, end_ns), zero-padding any coverage gaps.

        Returns an empty array if the entire range lies outside the buffer.
        """
        state = self._state
        if state.start_time_ns is None or state.samples.size == 0:
            return np.zeros(0, dtype=np.float32)

        start_idx = self._time_to_sample_index(state, start_ns)
        end_idx = self._time_to_sample_index(state, end_ns)
        n_out = max(0, end_idx - start_idx)
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)

        out = np.zeros(n_out, dtype=np.float32)
        cov_out = np.zeros(n_out, dtype=np.bool_)

        buf_start = state.buffer_start_sample_index
        buf_end = buf_start + state.samples.size

        # Overlap of requested range with buffered range.
        overlap_start = max(start_idx, buf_start)
        overlap_end = min(end_idx, buf_end)

        if overlap_start < overlap_end:
            out_offset = overlap_start - start_idx
            buf_offset = overlap_start - buf_start
            length = overlap_end - overlap_start
            out[out_offset : out_offset + length] = state.samples[buf_offset : buf_offset + length]
            cov_out[out_offset : out_offset + length] = state.coverage[buf_offset : buf_offset + length]

        # Zero-out any uncovered samples (coverage gaps inside the buffered range).
        out[~cov_out] = 0.0
        return out

    def _pruned(self, state: _BufferState, protected_from_ns: Optional[int] = None) -> _BufferState:
        if state.samples.size <= self.max_samples:
            return state

        drop = state.samples.size - self.max_samples

        # If a pin is active, do not evict samples at or after the pin timestamp.
        if protected_from_ns is not None and state.timeline_origin_ns is not None:
            protected_idx = self._time_to_sample_index(state, protected_from_ns)
            max_drop = max(0, protected_idx - state.buffer_start_sample_index)
            drop = min(drop, max_drop)
            if drop <= 0:
                return state

        pruned_start_sample_index = state.buffer_start_sample_index + drop
        return _BufferState(
            samples=state.samples[drop:],
            coverage=state.coverage[drop:],
            buffer_start_sample_index=pruned_start_sample_index,
            timeline_origin_ns=state.timeline_origin_ns,
            timeline_origin_sample_index=state.timeline_origin_sample_index,
            start_time_ns=self._sample_index_to_time_ns(state, pruned_start_sample_index),
        )

    def get_window(self, center_time_ns: int, window_seconds: float) -> np.ndarray | None:
        state = self._state
        if state.start_time_ns is None or state.samples.size == 0:
            return None

        window_samples = max(1, int(round(window_seconds * self.sample_rate_hz)))
        center_sample_index = self._time_to_sample_index(state, center_time_ns)
        start_sample_index = center_sample_index - (window_samples // 2)
        end_sample_index = start_sample_index + window_samples

        relative_start_index = start_sample_index - state.buffer_start_sample_index
        relative_end_index = end_sample_index - state.buffer_start_sample_index

        if relative_start_index < 0 or relative_end_index > state.samples.size:
            return None

        return state.samples[relative_start_index:relative_end_index].copy()

    def get_window_coverage_stats(self, center_time_ns: int, window_seconds: float) -> AudioCoverageStats | None:
        coverage = self._coverage_window(center_time_ns=center_time_ns, window_seconds=window_seconds)
        if coverage is None:
            return None
        return _coverage_stats(coverage, self.sample_rate_hz)

    def get_window_ending_at(self, end_time_ns: int, window_seconds: float) -> np.ndarray | None:
        state = self._state
        if state.start_time_ns is None or state.samples.size == 0:
            return None

        window_samples = max(1, int(round(window_seconds * self.sample_rate_hz)))
        end_sample_index = self._time_to_sample_index(state, end_time_ns)
        start_sample_index = end_sample_index - window_samples
        end_offset_samples = end_sample_index - state.buffer_start_sample_index
        start_offset_samples = start_sample_index - state.buffer_start_sample_index

        # Clamp the start to the beginning of the buffer so partial windows (e.g.
        # when the buffer has less than classification_window_seconds of history) return
        # whatever audio IS available rather than None.  Returning None here caused every
        # sensor to fall back to the 80 ms localization window, which is far too short
        # for BirdNET and produced only "unknown" (0.0) classifications.
        # We still return None when the end is beyond what has been buffered because
        # that would require future samples.
        if end_offset_samples > state.samples.size:
            return None
        start_offset_samples = max(0, start_offset_samples)

        return state.samples[start_offset_samples:end_offset_samples].copy()

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
        return self._state_end_time_ns(self._state)

    def _state_end_time_ns(self, state: _BufferState) -> int | None:
        if state.start_time_ns is None or state.samples.size == 0:
            return None
        return self._sample_index_to_time_ns(state, state.buffer_start_sample_index + state.samples.size)

    def _coverage_window(self, center_time_ns: int, window_seconds: float) -> np.ndarray | None:
        state = self._state
        if state.start_time_ns is None or state.coverage.size == 0:
            return None
        window_samples = max(1, int(round(window_seconds * self.sample_rate_hz)))
        center_sample_index = self._time_to_sample_index(state, center_time_ns)
        start_sample_index = center_sample_index - (window_samples // 2)
        end_sample_index = start_sample_index + window_samples
        relative_start_index = start_sample_index - state.buffer_start_sample_index
        relative_end_index = end_sample_index - state.buffer_start_sample_index
        if relative_start_index < 0 or relative_end_index > state.coverage.size:
            return None
        return state.coverage[relative_start_index:relative_end_index].copy()

    def _coverage_window_ending_at(self, end_time_ns: int, window_seconds: float) -> np.ndarray | None:
        state = self._state
        if state.start_time_ns is None or state.coverage.size == 0:
            return None
        window_samples = max(1, int(round(window_seconds * self.sample_rate_hz)))
        end_sample_index = self._time_to_sample_index(state, end_time_ns)
        start_sample_index = end_sample_index - window_samples
        end_offset_samples = end_sample_index - state.buffer_start_sample_index
        start_offset_samples = start_sample_index - state.buffer_start_sample_index
        if end_offset_samples > state.coverage.size:
            return None
        start_offset_samples = max(0, start_offset_samples)
        return state.coverage[start_offset_samples:end_offset_samples].copy()

    def _time_to_sample_index(self, state: _BufferState, time_ns: int) -> int:
        if state.timeline_origin_ns is None:
            raise ValueError("SensorStreamBuffer timeline origin is not initialized")
        delta_ns = time_ns - state.timeline_origin_ns
        return state.timeline_origin_sample_index + self._round_divide(delta_ns * self.sample_rate_hz, 1_000_000_000)

    def _sample_index_to_time_ns(self, state: _BufferState, sample_index: int) -> int:
        if state.timeline_origin_ns is None:
            raise ValueError("SensorStreamBuffer timeline origin is not initialized")
        return state.timeline_origin_ns + self._round_divide(
            (sample_index - state.timeline_origin_sample_index) * 1_000_000_000,
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
        self._capture_sessions: dict[str, ChunkedCaptureSessionBuffer] = {}
        # Per-sensor mutation lock — append() takes the global `_lock` only
        # briefly to look up/init the sensor's buffer + pin horizon, then
        # releases it and acquires the per-sensor lock for the heavy numpy
        # merge in `asyncio.to_thread`. This lets concurrent appends to
        # *different* sensors run in parallel while still serializing
        # same-sensor appends to preserve the buffer's invariants.
        self._per_sensor_locks: dict[str, asyncio.Lock] = {}
        # Diagnostic counter: incremented when an append() arrives with a
        # different sample_rate_hz than the existing per-sensor buffer, which
        # silently discards prior audio. Survives the buffer replacement
        # because it's on the wrapper, unlike SensorStreamBuffer.reanchor_count.
        self._rate_mismatch_wipes: dict[str, int] = {}

    async def start_capture(
        self,
        session_id: str,
        sensor_ids: list[str],
        *,
        max_duration_seconds: float,
    ) -> ChunkedCaptureSessionBuffer:
        capture_buffer = ChunkedCaptureSessionBuffer(
            sensor_ids=sensor_ids,
            max_duration_seconds=max_duration_seconds,
        )
        async with self._lock:
            self._capture_sessions[session_id] = capture_buffer
        return capture_buffer

    async def stop_capture(self, session_id: str) -> None:
        async with self._lock:
            self._capture_sessions.pop(session_id, None)

    def pin(self, session_id: str, start_ns: int) -> None:
        """Prevent the buffer from evicting samples at or after start_ns.

        Enforces a 5-minute hard cap so a stuck session cannot cause OOM.

        Performance note: SensorStreamBuffer uses a grow-and-prune pattern.
        While a pin is active, pruning cannot evict old frames, so the buffer
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
        the snapshotted numpy arrays alive even after pruning replaces them.
        """
        # ── Phase 1: snapshot under lock (cheap — just captures references) ───
        # Each entry is (samples_ref, coverage_ref, buf_start_idx, sr, origin_ns, origin_sample_idx)
        # or None when the sensor has no data.
        snapshots: list[Optional[tuple]] = []
        sample_rate_hz = 0
        actual_start_ns = start_ns
        actual_end_ns = end_ns

        async with self._lock:
            for sensor_id in sensor_ids:
                buf = self._buffers.get(sensor_id)
                if buf is None:
                    snapshots.append(None)
                    continue
                buf_state = buf._state
                if buf_state.samples.size == 0 or buf_state.timeline_origin_ns is None:
                    snapshots.append(None)
                    continue
                if sample_rate_hz == 0:
                    sample_rate_hz = buf.sample_rate_hz
                if buf_state.start_time_ns is not None:
                    actual_start_ns = max(actual_start_ns, buf_state.start_time_ns)
                end_buf = buf._state_end_time_ns(buf_state)
                if end_buf is not None:
                    actual_end_ns = min(actual_end_ns, end_buf)
                # Snapshot the array *references* (not copies) plus index metadata,
                # all taken from one immutable _BufferState so they are mutually
                # consistent. After lock release, buf._state may be replaced by
                # append(), but CPython keeps the old arrays alive as long as we
                # hold these references.
                snapshots.append((
                    buf_state.samples,                 # numpy ref — safe to read after unlock
                    buf_state.coverage,
                    buf_state.buffer_start_sample_index,
                    buf.sample_rate_hz,
                    buf_state.timeline_origin_ns,
                    buf_state.timeline_origin_sample_index,
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
        # Track per-channel coverage ratio for diagnostic warnings.
        channel_coverage_ratios: list[float] = []
        for i, snap in enumerate(snapshots):
            if snap is None:
                channels_list.append(np.zeros(0, dtype=np.float32))
                channel_coverage_ratios.append(0.0)
            else:
                ch_data = _snap_get_range(snap)
                channels_list.append(ch_data)
                # Compute coverage ratio from the snapshot's coverage array.
                samples_ref, coverage_ref, buf_start_idx, sr, origin_ns, origin_sample_idx = snap
                def _ts_to_idx(ns: int) -> int:
                    delta = ns - origin_ns
                    return origin_sample_idx + (delta * sr + 500_000_000) // 1_000_000_000
                start_idx = _ts_to_idx(start_ns)
                end_idx = _ts_to_idx(end_ns)
                n_range = max(0, end_idx - start_idx)
                if n_range > 0 and coverage_ref.size > 0:
                    overlap_start = max(start_idx, buf_start_idx)
                    overlap_end = min(end_idx, buf_start_idx + coverage_ref.size)
                    if overlap_start < overlap_end:
                        cov_off = overlap_start - buf_start_idx
                        cov_len = overlap_end - overlap_start
                        covered = int(np.count_nonzero(coverage_ref[cov_off:cov_off + cov_len]))
                        channel_coverage_ratios.append(covered / float(n_range))
                    else:
                        channel_coverage_ratios.append(0.0)
                else:
                    channel_coverage_ratios.append(0.0)

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

        # Build warnings for channels with poor coverage (<50%).
        coverage_warnings: list[str] = []
        for i, ratio in enumerate(channel_coverage_ratios):
            if ratio < 0.5:
                sensor_label = sensor_ids[i] if i < len(sensor_ids) else f"ch{i}"
                coverage_warnings.append(
                    f"sensor {sensor_label} has {ratio:.0%} coverage in the requested range"
                )

        sync_diag: dict = {
            "requested_start_ns": start_ns,
            "requested_end_ns": end_ns,
            "actual_audio_start_ns": actual_start_ns,
            "actual_audio_end_ns": actual_end_ns,
            "capture_rate_hz": sample_rate_hz,
            "n_segments": 1,
            "n_samples": n_samples,
            "source": "python_buffer",
            "channel_coverage_ratios": channel_coverage_ratios,
            "coverage_warnings": coverage_warnings,
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
        # Per-sensor concurrency: same-sensor appends serialize, but appends
        # for *different* sensors run in parallel. This is required for the
        # plan's primary goal ("many nodes streaming concurrently") — without
        # it, a single global lock around the O(N) numpy merge would queue
        # every node's ingest behind every other node.
        #
        # Lock order is always sensor → global (never global → sensor), since
        # the per-sensor lookup happens under the global lock only once during
        # init and the subsequent buffer-state acquisition re-enters the
        # global lock *inside* the sensor lock. No other call site acquires
        # the sensor lock, so deadlock is impossible.

        # Brief global lock — just to get/init the per-sensor lock handle.
        async with self._lock:
            sensor_lock = self._per_sensor_locks.setdefault(sensor_id, asyncio.Lock())

        async with sensor_lock:
            # Re-acquire the global lock inside the sensor lock so the buffer
            # lookup and the merge stay consistent with each other. Without
            # this, a sample-rate-change race could leave one append writing
            # to a buffer the other append has already replaced in the dict,
            # silently dropping the first append's samples.
            async with self._lock:
                buffer = self._buffers.get(sensor_id)
                if buffer is None or buffer.sample_rate_hz != sample_rate_hz:
                    if buffer is not None and buffer.sample_rate_hz != sample_rate_hz:
                        self._rate_mismatch_wipes[sensor_id] = (
                            self._rate_mismatch_wipes.get(sensor_id, 0) + 1
                        )
                        logger.warning(
                            "MultiSensorBuffer rate-mismatch wipe",
                            extra={
                                "sensor_id": sensor_id,
                                "old_sample_rate_hz": buffer.sample_rate_hz,
                                "new_sample_rate_hz": sample_rate_hz,
                                "wipes_for_sensor": self._rate_mismatch_wipes[sensor_id],
                            },
                        )
                    buffer = SensorStreamBuffer(
                        sample_rate_hz=sample_rate_hz,
                        max_duration_seconds=self.max_duration_seconds,
                    )
                    self._buffers[sensor_id] = buffer
                # Pin horizon is captured once. Any pin acquired *during* the
                # merge will be honored on the next append; the numpy merge
                # itself is bounded by max_duration_seconds so a stale-by-one-
                # frame pin horizon is safe.
                protected_from_ns = self._oldest_pin_ns()
                capture_buffers: list[ChunkedCaptureSessionBuffer] = (
                    [
                        capture_buffer
                        for capture_buffer in self._capture_sessions.values()
                        if capture_buffer.wants_sensor(sensor_id)
                    ]
                    if self._capture_sessions
                    else []
                )

            # Heavy numpy merge — bounded by max_samples (~32s × 16kHz = 512K
            # floats). Runs off the event loop so other ingest tasks (and any
            # concurrent reader on a *different* sensor) keep making progress.
            # The global lock is *not* held during this call.
            await asyncio.to_thread(
                buffer.append,
                start_time_ns=start_time_ns,
                samples=samples,
                start_sample_index=start_sample_index,
                end_sample_index=end_sample_index,
                end_time_ns=end_time_ns,
                _protected_from_ns=protected_from_ns,
            )

        for capture_buffer in capture_buffers:
            capture_buffer.append(
                sensor_id=sensor_id,
                sample_rate_hz=sample_rate_hz,
                start_time_ns=start_time_ns,
                samples=samples,
                start_sample_index=start_sample_index,
                end_sample_index=end_sample_index,
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

    async def snapshot_state(
        self,
        sensor_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return a per-sensor snapshot of buffer time coverage and counters.

        Used by FusionNode.status() for /api/v1/fusion/status and by
        _localize_candidate's "no_window" silent-drop log so an operator can
        immediately see whether an event_time_ns fell behind buffer start or
        ahead of buffer end, without code spelunking.
        """
        async with self._lock:
            ids = sensor_ids if sensor_ids is not None else sorted(self._buffers.keys())
            pin_ns = self._oldest_pin_ns()
            state: list[dict[str, Any]] = []
            for sensor_id in ids:
                buffer = self._buffers.get(sensor_id)
                if buffer is None:
                    state.append({"sensor_id": sensor_id, "present": False})
                    continue
                buffer_state = buffer._state
                end_time_ns = buffer._state_end_time_ns(buffer_state)
                state.append(
                    {
                        "sensor_id": sensor_id,
                        "present": True,
                        "sample_rate_hz": buffer.sample_rate_hz,
                        "sample_count": int(buffer_state.samples.size),
                        "start_time_ns": buffer_state.start_time_ns,
                        "end_time_ns": end_time_ns,
                        "reanchor_count": int(buffer.reanchor_count),
                        "rate_mismatch_wipes": int(self._rate_mismatch_wipes.get(sensor_id, 0)),
                        "pin_protected_from_ns": pin_ns,
                    }
                )
            return state

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
            available: list[tuple[str, SensorStreamBuffer, _BufferState]] = []
            sample_rate_counts: dict[int, int] = {}
            for sensor_id in sensor_ids:
                buffer = self._buffers.get(sensor_id)
                if buffer is None:
                    continue
                buffer_state = buffer._state
                if buffer_state.start_time_ns is None or buffer_state.samples.size == 0:
                    continue
                available.append((sensor_id, buffer, buffer_state))
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
            for sensor_id, buffer, buffer_state in available:
                if buffer.sample_rate_hz != dominant_sample_rate_hz:
                    continue
                tail = buffer_state.samples[-window_samples:].copy()
                if tail.size == 0:
                    continue
                recent[sensor_id] = tail
                end_ns = buffer._state_end_time_ns(buffer_state)
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

    async def sensor_end_time_ns(self, sensor_id: str) -> int | None:
        """Return the sample-count-based end timestamp for one sensor's buffer.

        The returned value is derived from the GPS-anchored timeline origin plus
        accumulated sample counts (not from the firmware GPS clock directly).
        This is intentional: the firmware GPS clock drifts from the ADC crystal
        oscillator (~40 ppm typical), so using the GPS-derived frame timestamp
        directly for trigger placement causes buffer_lag_timeout after hours of
        uptime.  Callers that need event_time_ns consistent with buffer contents
        should use this value rather than frame.start_time_ns.
        """
        async with self._lock:
            buf = self._buffers.get(sensor_id)
            if buf is None:
                return None
            return buf.end_time_ns()

    async def get_sensor_rms_history(self, sensor_id: str, n_buckets: int = 20) -> list[float]:
        """Return a list of n_buckets RMS values spanning the stored sample window for one sensor."""
        async with self._lock:
            buffer = self._buffers.get(sensor_id)
            if buffer is None:
                return []
            samples = buffer._state.samples
            if samples.size < n_buckets:
                return []
            n = samples.size
            chunk_size = max(1, n // n_buckets)
            result: list[float] = []
            for i in range(n_buckets):
                start = i * chunk_size
                end = start + chunk_size if i < n_buckets - 1 else n
                chunk = samples[start:end]
                if chunk.size > 0:
                    result.append(float(np.sqrt(np.mean(np.square(chunk)) + 1e-12)))
                else:
                    result.append(0.0)
            return result

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
                if buffer is None:
                    continue
                buffer_state = buffer._state
                if buffer_state.start_time_ns is None or buffer_state.samples.size == 0:
                    continue

                active_sensor_count += 1
                sample_rate_counts[buffer.sample_rate_hz] = sample_rate_counts.get(buffer.sample_rate_hz, 0) + 1
                max_buffer_samples = max(max_buffer_samples, buffer.max_samples)
                max_buffer_seconds = max(max_buffer_seconds, buffer.max_duration_seconds)
                end_ns = buffer._state_end_time_ns(buffer_state)
                if end_ns is None:
                    continue
                latest_sample_time_ns = end_ns if latest_sample_time_ns is None else max(latest_sample_time_ns, end_ns)

                tail_samples = max(1, min(buffer_state.samples.size, buffer.sample_rate_hz // 2))
                tail = buffer_state.samples[-tail_samples:]
                rms_values.append(float(np.sqrt(np.mean(np.square(tail)) + 1e-12)))
                coverage_tail = buffer_state.coverage[-tail_samples:]
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
