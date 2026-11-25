"""Step re-anchor and drift slew of a buffer timeline (Python ingest path).

Mirror of ``minimappr-core/src/timeline_discipline.rs`` (owner decision
2026-11-25, "step + drift re-anchor"); both are pinned by
``tests/fixtures/timeline_discipline_parity_cases.json``.

A buffer places audio on a nominal-rate timeline: one origin (packet time at a
sample index) extrapolated by the node's own sample count. A frame's timeline
residual is its packet timestamp minus where that timeline put it: the node's
packet clock against its own sample clock, with no acoustic delay and no other
node in it, so absorbing it cannot fight a genuine TDOA.

Both corrections move only the timeline origin. The mapping stays one straight
line over the sample index, so buffered audio stays contiguous (nothing zero
filled or overwritten) and every buffered sample is retimed onto the corrected
clock, which is also the best estimate for recent pre-step audio: the sample
clock never jumped, and a step is usually a re-sync correcting an offset the
earlier packets already carried.

- Step re-anchor: a frame whose packet-time advance disagrees with its
  sample-index advance by more than the step threshold opens a pending step;
  ``step_confirm_frames`` frames on the stepped clock confirm it and the origin
  shifts by their lower-median residual. A frame back on the old timeline
  cancels it (one late-stamped frame is jitter).
- Drift slew: when the windowed residual magnitude is in
  ``(slew_fraction * hard_limit, hard_limit]`` the origin shifts by it. Above
  the hard limit is not drift (an unconfirmed step or a flapping clock).

``BufferTimelineDiscipline`` adds what the sidecar's node clock tracker does
around the decision (step detection between consecutive frames and the
windowed residual) for one Python ``SensorStreamBuffer``. Python has no step
penalty or residual gate (``pair_timing_uncertainty.py``); only the timeline is
corrected here.
"""

from __future__ import annotations

import enum
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

NANOSECONDS_PER_SECOND = 1_000_000_000
PERCENT = 100.0


def _seconds_to_ns(seconds: float) -> int:
    return int(round(max(0.0, float(seconds)) * NANOSECONDS_PER_SECOND))


@dataclass(frozen=True, slots=True)
class TimelineDisciplinePolicy:
    step_reanchor_enabled: bool
    step_threshold_ns: int
    step_confirm_frames: int
    # 0 disables the drift slew.
    drift_slew_threshold_ns: int
    hard_limit_ns: int
    residual_window_frames: int
    residual_percentile: float

    @classmethod
    def from_settings(cls, settings: Any) -> "TimelineDisciplinePolicy":
        hard_limit_seconds = float(settings.localization_cross_node_max_timeline_residual_seconds)
        return cls(
            step_reanchor_enabled=bool(settings.localization_timeline_step_reanchor_enabled),
            step_threshold_ns=_seconds_to_ns(settings.localization_timestamp_step_threshold_seconds),
            step_confirm_frames=max(1, int(settings.localization_timestamp_step_confirm_frames)),
            drift_slew_threshold_ns=_seconds_to_ns(
                max(0.0, float(settings.localization_timeline_drift_slew_fraction)) * hard_limit_seconds
            ),
            hard_limit_ns=_seconds_to_ns(hard_limit_seconds),
            residual_window_frames=max(1, int(settings.localization_timeline_residual_window_frames)),
            residual_percentile=min(PERCENT, max(0.0, float(settings.localization_timeline_residual_percentile))),
        )


class TimelineCorrectionKind(str, enum.Enum):
    STEP_REANCHOR = "step_reanchor"
    DRIFT_SLEW = "drift_slew"


@dataclass(frozen=True, slots=True)
class TimelineCorrection:
    """Add ``origin_shift_ns`` to the timeline origin time."""

    kind: TimelineCorrectionKind
    origin_shift_ns: int


def lower_median(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


@dataclass(slots=True)
class _PendingStep:
    baseline_residual_ns: int
    stepped_residuals_ns: list[int]


@dataclass(slots=True)
class TimelineDiscipline:
    policy: TimelineDisciplinePolicy
    _pending_step: _PendingStep | None = None

    @property
    def step_pending(self) -> bool:
        return self._pending_step is not None

    def cancel_pending_step(self) -> None:
        self._pending_step = None

    def observe_frame(self, frame_residual_ns: int, detected_step_ns: int | None) -> TimelineCorrection | None:
        if not self.policy.step_reanchor_enabled:
            return None
        threshold_ns = self.policy.step_threshold_ns
        pending = self._pending_step
        if pending is not None:
            if abs(frame_residual_ns - pending.baseline_residual_ns) <= threshold_ns:
                self._pending_step = None
                return None
            if abs(frame_residual_ns - lower_median(pending.stepped_residuals_ns)) <= threshold_ns:
                pending.stepped_residuals_ns.append(frame_residual_ns)
            else:
                # A further step before confirmation: restart on the newest clock.
                pending.stepped_residuals_ns = [frame_residual_ns]
        elif detected_step_ns is not None:
            pending = _PendingStep(
                baseline_residual_ns=frame_residual_ns - detected_step_ns,
                stepped_residuals_ns=[frame_residual_ns],
            )
            self._pending_step = pending
        else:
            return None
        if len(pending.stepped_residuals_ns) < self.policy.step_confirm_frames:
            return None
        origin_shift_ns = lower_median(pending.stepped_residuals_ns)
        self._pending_step = None
        if origin_shift_ns == 0:
            return None
        return TimelineCorrection(TimelineCorrectionKind.STEP_REANCHOR, origin_shift_ns)

    def drift_slew(self, windowed_residual_ns: int | None) -> TimelineCorrection | None:
        if self._pending_step is not None or self.policy.drift_slew_threshold_ns <= 0 or windowed_residual_ns is None:
            return None
        magnitude_ns = abs(windowed_residual_ns)
        if self.policy.drift_slew_threshold_ns < magnitude_ns <= self.policy.hard_limit_ns:
            return TimelineCorrection(TimelineCorrectionKind.DRIFT_SLEW, windowed_residual_ns)
        return None


def nominal_elapsed_ns(sample_index_delta: int, sample_rate_hz: int) -> int:
    """Rounded half away from zero, like the buffer's index-to-time mapping."""
    scaled = sample_index_delta * NANOSECONDS_PER_SECOND
    half = sample_rate_hz // 2
    if scaled >= 0:
        return (scaled + half) // sample_rate_hz
    return -((-scaled + half) // sample_rate_hz)


def residual_at_percentile(residuals_ns: deque[int], percentile: float) -> int | None:
    """Residual whose magnitude sits at ``percentile`` (nearest rank, sign kept),
    as the sidecar's ``residual_at_percentile``."""
    if not residuals_ns:
        return None
    by_magnitude = sorted(residuals_ns, key=abs)
    rank = math.floor((percentile / PERCENT) * (len(by_magnitude) - 1) + 0.5)
    return by_magnitude[min(rank, len(by_magnitude) - 1)]


@dataclass(slots=True)
class BufferTimelineDiscipline:
    """Per-buffer discipline state, fed one explicit-sample-index frame at a time."""

    policy: TimelineDisciplinePolicy
    sample_rate_hz: int
    step_reanchor_count: int = 0
    drift_slew_count: int = 0
    _discipline: TimelineDiscipline = field(init=False)
    _previous_point: tuple[int, int] | None = None
    _recent_residuals_ns: deque[int] = field(init=False)

    def __post_init__(self) -> None:
        self._discipline = TimelineDiscipline(self.policy)
        self._recent_residuals_ns = deque(maxlen=self.policy.residual_window_frames)

    def reset(self) -> None:
        """Nothing measured so far applies (fresh anchor, or receipt-time audio)."""
        self._discipline.cancel_pending_step()
        self._previous_point = None
        self._recent_residuals_ns.clear()

    def restart(self, *, anchor_sample_index: int | None, anchor_time_ns: int) -> None:
        """A fresh anchor: reset, then record the anchor frame itself (residual
        0) so a step on the very next frame is seen, as the sidecar's detector
        sees every frame. ``None``: the anchor has no sample index or is off
        packet time."""
        self.reset()
        if anchor_sample_index is not None:
            self.observe_frame(
                start_sample_index=anchor_sample_index, packet_timestamp_ns=anchor_time_ns, frame_residual_ns=0
            )

    def origin_shift_ns(self, *, start_sample_index: int, packet_timestamp_ns: int, frame_residual_ns: int) -> int:
        """``observe_frame``, counted and logged; the summed origin shift to apply."""
        corrections = self.observe_frame(
            start_sample_index=start_sample_index,
            packet_timestamp_ns=packet_timestamp_ns,
            frame_residual_ns=frame_residual_ns,
        )
        for correction in corrections:
            if correction.kind is TimelineCorrectionKind.STEP_REANCHOR:
                self.step_reanchor_count += 1
            else:
                self.drift_slew_count += 1
            logger.info(
                "Buffer timeline correction",
                extra={
                    "sample_rate_hz": self.sample_rate_hz,
                    "kind": correction.kind.value,
                    "origin_shift_ns": correction.origin_shift_ns,
                    "start_sample_index": start_sample_index,
                },
            )
        return sum(correction.origin_shift_ns for correction in corrections)

    @property
    def windowed_residual_ns(self) -> int | None:
        return residual_at_percentile(self._recent_residuals_ns, self.policy.residual_percentile)

    def observe_frame(
        self, *, start_sample_index: int, packet_timestamp_ns: int, frame_residual_ns: int
    ) -> list[TimelineCorrection]:
        """Corrections (in order) for one frame whose residual was measured
        against the buffer's current timeline. The caller shifts its origin by
        their sum."""
        detected_step_ns: int | None = None
        if self._previous_point is not None:
            previous_sample_index, previous_timestamp_ns = self._previous_point
            step_ns = (packet_timestamp_ns - previous_timestamp_ns) - nominal_elapsed_ns(
                start_sample_index - previous_sample_index, self.sample_rate_hz
            )
            if abs(step_ns) > self.policy.step_threshold_ns:
                detected_step_ns = step_ns
        self._previous_point = (start_sample_index, packet_timestamp_ns)

        corrections: list[TimelineCorrection] = []
        residual_ns = frame_residual_ns
        step_correction = self._discipline.observe_frame(residual_ns, detected_step_ns)
        if step_correction is not None:
            # Residuals recorded before the step were on the old clock.
            self._recent_residuals_ns.clear()
            residual_ns -= step_correction.origin_shift_ns
            corrections.append(step_correction)
        self._recent_residuals_ns.append(residual_ns)
        slew_correction = self._discipline.drift_slew(self.windowed_residual_ns)
        if slew_correction is not None:
            shifted = [recorded - slew_correction.origin_shift_ns for recorded in self._recent_residuals_ns]
            self._recent_residuals_ns.clear()
            self._recent_residuals_ns.extend(shifted)
            corrections.append(slew_correction)
        return corrections
