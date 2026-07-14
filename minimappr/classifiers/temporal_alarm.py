"""Classic-DSP T3/T4 temporal alarm cadence detector.

Detects ISO 8201-1 "temporal-three" (T3, 3 pulses/cycle — fire evacuation
signal) and "temporal-four" (T4, 4 pulses/cycle — commonly CO alarms) audible
alarm cadences. Unlike the neural classifiers in this package, this is pure
vectorized DSP: narrowband tone-energy extraction (Goertzel-as-matmul) feeds a
hysteresis on/off envelope, which is segmented into pulse runs and matched
against the two cadence templates.

The primary discriminant is cadence, not tone — most piezo sounders share the
same ~3 kHz tone for both patterns. The tone-band gate exists only to reject
broadband/rhythmic noise (footsteps, door slams) that could otherwise mimic a
pulse train; it does not by itself distinguish T3 from T4.

Runs stateless per call (one window in, one result out) because the shared
``ContinuousOmniScanner`` reuses a single classifier instance across every
registered node with no node identity passed to ``classify()`` — see
``core/omni_scanner.py``. Reliability instead comes from requiring several
consecutive matching cycles *within* one sufficiently long window (see
``min_repeats`` and the ``omni_continuous`` context's window sizing).
"""

from __future__ import annotations

import numpy as np

from minimappr.classifiers.base import AudioClassifier
from minimappr.models import ClassificationResult

# (state, nominal_duration_seconds) runs for one full cadence cycle.
# state: 1 = tone on, 0 = tone off.
T3_TEMPLATE: tuple[tuple[int, float], ...] = (
    (1, 0.5), (0, 0.5),
    (1, 0.5), (0, 0.5),
    (1, 0.5), (0, 1.5),
)
T4_TEMPLATE: tuple[tuple[int, float], ...] = (
    (1, 0.5), (0, 0.5),
    (1, 0.5), (0, 0.5),
    (1, 0.5), (0, 0.5),
    (1, 0.5), (0, 1.5),
)

_EPSILON = 1e-9


def _control_freqs_hz(band_low_hz: float, band_high_hz: float, sample_rate_hz: int, count: int) -> np.ndarray:
    """Reference frequencies spread across the spectrum, avoiding the target band.

    Used as a noise-floor estimate: a single DFT/Goertzel bin's energy for
    white noise is the *same order* as the signal's total time-domain energy
    (it does not shrink with more bins), so comparing target-band energy
    against total broadband frame energy does not discriminate a tone from
    noise. Comparing against *other individual bins* does — a pure tone
    concentrates energy in its own bin while noise spreads it evenly across
    all bins.
    """
    nyquist = 0.45 * sample_rate_hz
    candidates = np.linspace(150.0, nyquist, count + 2)
    band_margin = (band_high_hz - band_low_hz)
    mask = (candidates < band_low_hz - band_margin) | (candidates > band_high_hz + band_margin)
    filtered = candidates[mask]
    return filtered[:count] if filtered.size >= count else candidates[:count]


def _tone_envelope(
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    band_low_hz: float,
    band_high_hz: float,
    frame_seconds: float,
    num_tone_bins: int,
    num_control_bins: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Per-frame narrowband tone energy, control-bin energy, dominant bin, hop seconds.

    Frames the signal and projects each (Hann-windowed) frame onto a small
    cos/sin basis spanning ``num_tone_bins`` target frequencies plus
    ``num_control_bins`` reference frequencies elsewhere in the spectrum — one
    matrix multiply for the whole window, no per-frame Python loop (a
    Goertzel-as-matmul: cheaper than an FFT since only a few bins are needed).
    """
    frame_len = max(8, int(round(frame_seconds * sample_rate_hz)))
    hop_len = max(1, frame_len // 2)
    if samples.size < frame_len:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            hop_len / sample_rate_hz,
        )

    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_len)[::hop_len]
    window = np.hanning(frame_len).astype(np.float64)
    windowed = frames.astype(np.float64) * window

    target_freqs = np.linspace(band_low_hz, band_high_hz, num_tone_bins)
    control_freqs = _control_freqs_hz(band_low_hz, band_high_hz, sample_rate_hz, num_control_bins)
    freqs = np.concatenate([target_freqs, control_freqs])
    n = np.arange(frame_len, dtype=np.float64)
    angles = 2.0 * np.pi * freqs[:, None] * n[None, :] / sample_rate_hz  # (K, frame_len)
    basis = np.concatenate([np.cos(angles), np.sin(angles)], axis=0).T  # (frame_len, 2K)

    proj = windowed @ basis  # (num_frames, 2K)
    total_bins = freqs.size
    energy_per_bin = proj[:, :total_bins] ** 2 + proj[:, total_bins:] ** 2  # (num_frames, K)
    target_energy = energy_per_bin[:, :num_tone_bins]
    control_energy = energy_per_bin[:, num_tone_bins:]

    dominant_bin = np.argmax(target_energy, axis=1)
    tone_energy = np.take_along_axis(target_energy, dominant_bin[:, None], axis=1)[:, 0]
    # Compare the target band's *max* bin against the control band's *max*
    # bin (not the mean) so both sides carry the same max-of-N order-statistic
    # bias; otherwise white noise would look artificially "tone-like" just
    # because picking the best of several target bins beats a plain average.
    control_floor = np.max(control_energy, axis=1)

    return tone_energy, control_floor, dominant_bin, hop_len / sample_rate_hz


def _binarize_hysteresis(
    envelope: np.ndarray, *, hi_ratio: float = 4.0, lo_ratio: float = 2.0
) -> np.ndarray:
    """Hysteresis on/off gate relative to a noise-floor estimate.

    The floor is the envelope's 20th percentile (robust to the pulses
    themselves, which occupy roughly half the window at most). ``hi_ratio``
    must exceed a pulse to turn on; ``lo_ratio`` (< hi_ratio) must be crossed
    downward to turn back off, avoiding chatter at the pulse edges.
    """
    if envelope.size == 0:
        return np.zeros(0, dtype=bool)
    floor = float(np.percentile(envelope, 20.0)) + _EPSILON
    hi = floor * hi_ratio
    lo = floor * lo_ratio
    state = np.zeros(envelope.size, dtype=bool)
    on = False
    for i, value in enumerate(envelope):
        if on:
            on = value >= lo
        else:
            on = value >= hi
        state[i] = on
    return state


def _segment_runs(state: np.ndarray, hop_seconds: float) -> list[tuple[int, float]]:
    """Collapse a per-frame boolean state array into (state, duration_seconds) runs."""
    if state.size == 0:
        return []
    runs: list[tuple[int, float]] = []
    current = bool(state[0])
    count = 1
    for value in state[1:]:
        if bool(value) == current:
            count += 1
        else:
            runs.append((int(current), count * hop_seconds))
            current = bool(value)
            count = 1
    runs.append((int(current), count * hop_seconds))
    return runs


def _match_from(
    runs: list[tuple[int, float]], start: int, template: tuple[tuple[int, float], ...], tolerance: float
) -> tuple[int, float]:
    count = 0
    errors: list[float] = []
    idx = start
    while idx + len(template) <= len(runs):
        cycle_errors: list[float] = []
        ok = True
        for (t_state, t_dur), (r_state, r_dur) in zip(template, runs[idx : idx + len(template)]):
            if r_state != t_state:
                ok = False
                break
            rel_err = abs(r_dur - t_dur) / t_dur
            if rel_err > tolerance:
                ok = False
                break
            cycle_errors.append(rel_err)
        if not ok:
            break
        count += 1
        errors.extend(cycle_errors)
        idx += len(template)
    mean_err = float(np.mean(errors)) if errors else 1.0
    return count, mean_err


def _best_match(
    runs: list[tuple[int, float]], template: tuple[tuple[int, float], ...], tolerance: float
) -> tuple[int, float]:
    best_count, best_err = 0, 1.0
    template_start_state = template[0][0]
    for start in range(len(runs)):
        if runs[start][0] != template_start_state:
            continue
        count, err = _match_from(runs, start, template, tolerance)
        if count > best_count or (count == best_count and err < best_err):
            best_count, best_err = count, err
    return best_count, best_err


class TemporalAlarmClassifier(AudioClassifier):
    """Detects T3/T4 temporal alarm cadences via narrowband envelope + template matching."""

    def __init__(
        self,
        *,
        min_repeats: int = 3,
        min_confidence: float = 0.5,
        tone_band_low_hz: float = 2800.0,
        tone_band_high_hz: float = 3500.0,
        num_tone_bins: int = 5,
        frame_seconds: float = 0.02,
        tolerance: float = 0.3,
    ) -> None:
        self.min_repeats = max(2, int(min_repeats))
        self.min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self.tone_band_low_hz = float(tone_band_low_hz)
        self.tone_band_high_hz = float(tone_band_high_hz)
        self.num_tone_bins = max(1, int(num_tone_bins))
        self.frame_seconds = float(frame_seconds)
        self.tolerance = float(tolerance)

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        samples = np.asarray(samples)
        if samples.ndim > 1:
            samples = np.mean(samples, axis=0)
        samples = samples.astype(np.float64)

        empty_features = {"cycle_count": 0.0, "tone_hz": 0.0, "pattern_error": 1.0}
        if samples.size == 0:
            return ClassificationResult(label="unknown", confidence=0.0, scores={}, features=empty_features)

        tone_energy, control_floor, dominant_bin, hop_seconds = _tone_envelope(
            samples,
            sample_rate_hz,
            band_low_hz=self.tone_band_low_hz,
            band_high_hz=self.tone_band_high_hz,
            frame_seconds=self.frame_seconds,
            num_tone_bins=self.num_tone_bins,
        )
        if tone_energy.size == 0:
            return ClassificationResult(label="unknown", confidence=0.0, scores={}, features=empty_features)

        state = _binarize_hysteresis(tone_energy)
        runs = _segment_runs(state, hop_seconds)

        t3_count, t3_err = _best_match(runs, T3_TEMPLATE, self.tolerance)
        t4_count, t4_err = _best_match(runs, T4_TEMPLATE, self.tolerance)

        candidates = [
            ("alarm_t3", t3_count, t3_err),
            ("alarm_t4", t4_count, t4_err),
        ]
        # Prefer more matched cycles, then lower timing error.
        candidates.sort(key=lambda c: (-c[1], c[2]))
        label, count, err = candidates[0]

        scores = {
            "alarm_t3": 0.0,
            "alarm_t4": 0.0,
            "unknown": 1.0,
        }
        freqs = np.linspace(self.tone_band_low_hz, self.tone_band_high_hz, self.num_tone_bins)

        if count < self.min_repeats:
            features = {
                "cycle_count": float(max(t3_count, t4_count)),
                "tone_hz": 0.0,
                "pattern_error": float(min(t3_err, t4_err)),
            }
            return ClassificationResult(label="unknown", confidence=0.0, scores=scores, features=features)

        # SNR over "on" frames only: how far the target band's strongest bin
        # stands above the control bands' strongest bin. A pure tone
        # concentrates energy in one bin (high ratio); broadband/rhythmic
        # noise (e.g. footsteps) spreads energy evenly, so target and control
        # bins score similarly (ratio near 1) even if the cadence matches.
        on_mask = state
        snr = float(
            np.mean(tone_energy[on_mask] / (control_floor[on_mask] + _EPSILON))
        ) if np.any(on_mask) else 0.0
        snr_score = float(np.clip((snr - 1.0) / 4.0, 0.0, 1.0))
        repeat_score = float(np.clip((count - self.min_repeats + 1) / 3.0, 0.0, 1.0))
        timing_score = float(np.clip(1.0 - err / max(self.tolerance, _EPSILON), 0.0, 1.0))
        confidence = float(np.clip(0.15 + 0.2 * repeat_score + 0.2 * timing_score + 0.45 * snr_score, 0.0, 0.98))

        tone_hz = float(freqs[int(np.median(dominant_bin[on_mask]))]) if np.any(on_mask) else 0.0

        scores[label] = confidence
        scores["unknown"] = 1.0 - confidence

        features = {
            "cycle_count": float(count),
            "tone_hz": tone_hz,
            "pattern_error": float(err),
            "tone_snr": snr,
        }

        if confidence < self.min_confidence:
            return ClassificationResult(label="unknown", confidence=0.0, scores=scores, features=features)

        return ClassificationResult(label=label, confidence=confidence, scores=scores, features=features)


__all__ = ["TemporalAlarmClassifier", "T3_TEMPLATE", "T4_TEMPLATE"]
