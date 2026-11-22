"""Audio conditioning for BirdNET+ V3.0: resample, silence guard, windowing.

Pure numpy/scipy, no model dependency, so every rule here is unit-testable
without the ONNX file.

Contract with the model (BirdNET+ V3.0.1 ONNX, ``input`` float32
``[batch, samples]`` at 32 kHz, any length; the model card says 2-12 s matches
training best):

* Clips shorter than the minimum window are padded up to it with Gaussian noise
  at the clip's *own* noise floor, never zeros. Hard zeros put broadband edges
  into the spectrogram, and V3.0.0 exports returned NaN on zero-variance input.
* Clips inside [min, max] go in whole.
* Longer clips are cut into max-length windows with overlap; the last window is
  anchored to the clip end so every window is full length (no tail padding).

The silence guard is deliberately *not* a level gate. Faint distant birds must
reach the classifier (AGENTS.md: long-range detections are quiet), so the only
rejections are numerical: non-finite samples beyond a tolerated fraction, and
(near-)exactly zero variance relative to the signal's own DC level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

BIRDNET_V3_SAMPLE_RATE_HZ = 32_000

SILENCE_GUARD_REASON_NONFINITE = "nonfinite_input"
SILENCE_GUARD_REASON_ZERO_VARIANCE = "zero_variance_input"
SILENCE_GUARD_REASON_EMPTY = "empty_input"


@dataclass(frozen=True, slots=True)
class BirdNETV3WindowPolicy:
    """Window/guard tuning; every field is a Settings key (see config.py)."""

    min_window_seconds: float = 2.0
    max_window_seconds: float = 12.0
    window_overlap_seconds: float = 1.5
    # Fraction of NaN/inf samples tolerated (replaced by noise fill) before the
    # whole clip is rejected as corrupt.
    max_nonfinite_sample_fraction: float = 0.01
    # Noise-floor estimate for padding = this percentile of short-frame RMS.
    pad_noise_floor_percentile: float = 10.0
    # Frame length for the noise-floor estimate.
    noise_floor_frame_seconds: float = 0.05
    # Zero variance is judged relative to |mean| so a pure DC offset counts as
    # silence; the absolute floor is float32 ``tiny`` - a numerical zero, not a
    # level threshold.
    zero_variance_relative_tolerance: float = 1e-7
    # Fixed seed so padding (and therefore scores) is reproducible per clip.
    pad_noise_seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.min_window_seconds <= self.max_window_seconds:
            raise ValueError("BirdNET V3 windows need 0 < min_window_seconds <= max_window_seconds")
        if not 0.0 <= self.window_overlap_seconds < self.max_window_seconds:
            raise ValueError("BirdNET V3 window_overlap_seconds must be in [0, max_window_seconds)")
        if not 0.0 <= self.max_nonfinite_sample_fraction <= 1.0:
            raise ValueError("BirdNET V3 max_nonfinite_sample_fraction must be in [0, 1]")
        if not 0.0 <= self.pad_noise_floor_percentile <= 100.0:
            raise ValueError("BirdNET V3 pad_noise_floor_percentile must be in [0, 100]")


@dataclass(frozen=True, slots=True)
class SilenceGuardVerdict:
    """Outcome of the guard: ``audio`` is finite float32 when ``accepted``."""

    accepted: bool
    audio: np.ndarray
    reason: str | None = None
    nonfinite_sample_count: int = 0


def resample_to_birdnet_v3_rate(samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """Polyphase resample to 32 kHz (48 kHz -> 32 kHz is the exact 2/3 ratio)."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if int(sample_rate_hz) == BIRDNET_V3_SAMPLE_RATE_HZ or audio.size == 0:
        return audio
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    divisor = np.gcd(BIRDNET_V3_SAMPLE_RATE_HZ, int(sample_rate_hz))
    return resample_poly(
        audio,
        up=BIRDNET_V3_SAMPLE_RATE_HZ // divisor,
        down=int(sample_rate_hz) // divisor,
    ).astype(np.float32)


def estimate_noise_floor_rms(audio: np.ndarray, policy: BirdNETV3WindowPolicy, sample_rate_hz: int) -> float:
    """Robust noise-floor RMS: low percentile of short-frame RMS over non-silent frames.

    Exact-zero frames (dropped packets zero-filled upstream) are excluded so a
    partly blank clip still pads with its real background, not with nothing.
    """
    frame_samples = max(1, int(round(policy.noise_floor_frame_seconds * sample_rate_hz)))
    usable = (audio.size // frame_samples) * frame_samples
    if usable == 0:
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    frames = audio[:usable].astype(np.float64).reshape(-1, frame_samples)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    nonzero = frame_rms[frame_rms > 0.0]
    if nonzero.size == 0:
        # Energy only in the partial tail frame: whole-clip std is the best floor left.
        return float(np.std(audio.astype(np.float64)))
    return float(np.percentile(nonzero, policy.pad_noise_floor_percentile))


def _noise_at_floor(sample_count: int, floor_rms: float, policy: BirdNETV3WindowPolicy) -> np.ndarray:
    generator = np.random.default_rng(policy.pad_noise_seed)
    return (generator.standard_normal(sample_count) * floor_rms).astype(np.float32)


def apply_silence_guard(
    audio: np.ndarray,
    policy: BirdNETV3WindowPolicy,
    sample_rate_hz: int = BIRDNET_V3_SAMPLE_RATE_HZ,
) -> SilenceGuardVerdict:
    """Reject only numerically degenerate input; repair a few non-finite samples.

    Never compares the signal level against an absolute threshold: a 1e-6
    amplitude call with background noise passes untouched.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return SilenceGuardVerdict(False, audio, SILENCE_GUARD_REASON_EMPTY)

    finite_mask = np.isfinite(audio)
    nonfinite_count = int(audio.size - np.count_nonzero(finite_mask))
    if nonfinite_count:
        if nonfinite_count / audio.size > policy.max_nonfinite_sample_fraction:
            return SilenceGuardVerdict(False, audio, SILENCE_GUARD_REASON_NONFINITE, nonfinite_count)
        audio = audio.copy()
        audio[~finite_mask] = 0.0
        floor = estimate_noise_floor_rms(audio, policy, sample_rate_hz)
        audio[~finite_mask] = _noise_at_floor(nonfinite_count, floor, policy)

    as_float64 = audio.astype(np.float64)
    mean = float(np.mean(as_float64))
    standard_deviation = float(np.std(as_float64))
    zero_variance_floor = max(
        float(np.finfo(np.float32).tiny),
        policy.zero_variance_relative_tolerance * abs(mean),
    )
    if not np.isfinite(standard_deviation) or standard_deviation <= zero_variance_floor:
        return SilenceGuardVerdict(False, audio, SILENCE_GUARD_REASON_ZERO_VARIANCE, nonfinite_count)
    return SilenceGuardVerdict(True, audio, None, nonfinite_count)


def pad_to_minimum_window(
    audio: np.ndarray,
    policy: BirdNETV3WindowPolicy,
    sample_rate_hz: int = BIRDNET_V3_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Centre a short clip inside noise at its own floor, up to the minimum window."""
    minimum_samples = int(round(policy.min_window_seconds * sample_rate_hz))
    if audio.size >= minimum_samples:
        return audio
    floor = estimate_noise_floor_rms(audio, policy, sample_rate_hz)
    padded = _noise_at_floor(minimum_samples, floor, policy)
    offset = (minimum_samples - audio.size) // 2
    padded[offset : offset + audio.size] = audio
    return padded


def plan_window_spans(
    total_samples: int,
    policy: BirdNETV3WindowPolicy,
    sample_rate_hz: int = BIRDNET_V3_SAMPLE_RATE_HZ,
) -> list[tuple[int, int]]:
    """``[start, end)`` spans covering ``total_samples`` (already >= min window).

    One span when the clip fits the max window; otherwise max-length spans at
    ``hop = max - overlap`` with the final span flush to the end.
    """
    maximum_samples = int(round(policy.max_window_seconds * sample_rate_hz))
    if total_samples <= 0:
        return []
    if total_samples <= maximum_samples:
        return [(0, total_samples)]
    overlap_samples = int(round(policy.window_overlap_seconds * sample_rate_hz))
    hop_samples = max(1, maximum_samples - overlap_samples)
    spans: list[tuple[int, int]] = []
    start = 0
    while start + maximum_samples < total_samples:
        spans.append((start, start + maximum_samples))
        start += hop_samples
    spans.append((total_samples - maximum_samples, total_samples))
    return spans


def prepare_birdnet_v3_windows(
    samples: np.ndarray,
    sample_rate_hz: int,
    policy: BirdNETV3WindowPolicy,
) -> tuple[list[tuple[float, float, np.ndarray]], SilenceGuardVerdict]:
    """Full conditioning chain: resample -> guard -> pad -> segment.

    Returns ``(windows, verdict)`` where each window is
    ``(start_seconds, end_seconds, audio)`` in the *original* clip's timeline
    (padding time is not reported). Windows whose own variance is degenerate,
    e.g. a zero-filled gap inside a long clip, are dropped individually.
    """
    audio = resample_to_birdnet_v3_rate(samples, sample_rate_hz)
    verdict = apply_silence_guard(audio, policy)
    if not verdict.accepted:
        return [], verdict
    original_samples = verdict.audio.size
    conditioned = pad_to_minimum_window(verdict.audio, policy)
    windows: list[tuple[float, float, np.ndarray]] = []
    for start, end in plan_window_spans(conditioned.size, policy):
        segment = conditioned[start:end]
        if not apply_silence_guard(segment, policy).accepted:
            continue
        clipped_end = min(end, original_samples)
        windows.append(
            (
                min(start, original_samples) / BIRDNET_V3_SAMPLE_RATE_HZ,
                clipped_end / BIRDNET_V3_SAMPLE_RATE_HZ,
                np.ascontiguousarray(segment, dtype=np.float32),
            )
        )
    return windows, verdict
