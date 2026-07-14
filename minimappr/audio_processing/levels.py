"""Bounded level conditioning that preserves dynamics and prevents clipping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minimappr.audio_processing.profiles import AudioProcessingProfile

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class AudioLevelReport:
    input_rms_dbfs: float
    input_peak_dbfs: float
    output_rms_dbfs: float
    output_peak_dbfs: float
    applied_gain_db: float
    clipping_risk_sample_count: int


def _amplitude_dbfs(value: float) -> float:
    if value <= _EPSILON:
        return -240.0
    return float(20.0 * np.log10(value))


def apply_bounded_rms_gain(
    samples: np.ndarray,
    *,
    target_rms: float,
    max_gain: float,
    peak_ceiling: float,
    center: bool = False,
    boost_only: bool = True,
) -> tuple[np.ndarray, AudioLevelReport]:
    """Apply one scalar chosen from RMS target, gain cap, and peak headroom.

    Multichannel input is conditioned with a common scalar, preserving channel
    relationships. Non-finite samples are rejected instead of contaminating a
    downstream classifier or WAV encoder.
    """
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.size == 0:
        report = AudioLevelReport(-240.0, -240.0, -240.0, -240.0, 0.0, 0)
        return waveform.astype(np.float32, copy=False), report
    if not np.all(np.isfinite(waveform)):
        raise ValueError("Audio level conditioning requires finite samples")
    if target_rms <= 0.0 or max_gain <= 0.0 or not 0.0 < peak_ceiling <= 1.0:
        raise ValueError("target_rms and max_gain must be positive; peak_ceiling must be in (0, 1]")

    working = waveform.astype(np.float32, copy=True)
    if center:
        if working.ndim == 1:
            working -= np.mean(working, dtype=np.float32)
        else:
            working -= np.mean(working, axis=-1, keepdims=True, dtype=np.float32)

    input_rms = float(np.sqrt(np.mean(np.square(working), dtype=np.float64) + _EPSILON))
    input_peak = float(np.max(np.abs(working)) + _EPSILON)
    requested_gain = float(target_rms) / max(input_rms, _EPSILON)
    if boost_only:
        requested_gain = max(1.0, requested_gain)
    peak_limited_gain = float(peak_ceiling) / input_peak
    gain = min(float(max_gain), requested_gain, peak_limited_gain)
    if boost_only:
        gain = max(gain, min(1.0, peak_limited_gain))

    output = (working * gain).astype(np.float32)
    output_rms = float(np.sqrt(np.mean(np.square(output), dtype=np.float64) + _EPSILON))
    output_peak = float(np.max(np.abs(output)) + _EPSILON)
    report = AudioLevelReport(
        input_rms_dbfs=_amplitude_dbfs(input_rms),
        input_peak_dbfs=_amplitude_dbfs(input_peak),
        output_rms_dbfs=_amplitude_dbfs(output_rms),
        output_peak_dbfs=_amplitude_dbfs(output_peak),
        applied_gain_db=_amplitude_dbfs(gain),
        clipping_risk_sample_count=int(np.count_nonzero(np.abs(output) > 1.0)),
    )
    return output, report


def apply_listening_level(samples: np.ndarray) -> tuple[np.ndarray, AudioLevelReport]:
    """Default human-listening profile: -24 dBFS RMS, +24 dB max, -1 dBFS peak."""
    return apply_bounded_rms_gain(
        samples,
        target_rms=10.0 ** (-24.0 / 20.0),
        max_gain=10.0 ** (24.0 / 20.0),
        peak_ceiling=10.0 ** (-1.0 / 20.0),
        center=True,
        boost_only=True,
    )


def apply_level_profile(
    samples: np.ndarray,
    profile: AudioProcessingProfile,
) -> tuple[np.ndarray, AudioLevelReport]:
    """Apply a declarative mean-center + bounded-RMS level profile."""
    center = any(str(stage.get("type")) == "mean_center" for stage in profile.stages)
    bounded = next(
        (stage for stage in profile.stages if str(stage.get("type")) == "bounded_rms_gain"),
        None,
    )
    if bounded is None:
        raise ValueError(f"Audio level profile {profile.name!r} has no bounded_rms_gain stage")
    return apply_bounded_rms_gain(
        samples,
        target_rms=10.0 ** (float(bounded["target_rms_dbfs"]) / 20.0),
        max_gain=10.0 ** (float(bounded["max_gain_db"]) / 20.0),
        peak_ceiling=10.0 ** (float(bounded["peak_ceiling_dbfs"]) / 20.0),
        center=center,
        boost_only=bool(bounded.get("boost_only", True)),
    )
