"""T3/T4 temporal alarm cadence detector: synthetic pulse-train tests.

Builds synthetic tone-pulse trains at known cadence/tone and checks that the
classifier (a) detects the correct cadence once enough consecutive cycles have
been observed, and (b) rejects the main false-positive sources: silence,
a single isolated beep, off-cadence rhythmic noise, and a tone outside the
narrowband gate.
"""

from __future__ import annotations

import numpy as np

from minimappr.classifiers.temporal_alarm import T3_TEMPLATE, T4_TEMPLATE, TemporalAlarmClassifier

SAMPLE_RATE_HZ = 16_000


def _tone(duration_s: float, freq_hz: float, sample_rate_hz: int, amplitude: float = 0.5) -> np.ndarray:
    n = int(round(duration_s * sample_rate_hz))
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float64)


def _silence(duration_s: float, sample_rate_hz: int, noise_amplitude: float = 0.001) -> np.ndarray:
    n = int(round(duration_s * sample_rate_hz))
    rng = np.random.default_rng(0)
    return (noise_amplitude * rng.standard_normal(n)).astype(np.float64)


def _pulse_train(
    template,
    reps: int,
    *,
    tone_hz: float,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    on_kind: str = "tone",
) -> np.ndarray:
    segments = []
    for _ in range(reps):
        for state, duration_s in template:
            if state == 1:
                if on_kind == "tone":
                    segments.append(_tone(duration_s, tone_hz, sample_rate_hz))
                else:  # broadband noise "on" pulses
                    n = int(round(duration_s * sample_rate_hz))
                    rng = np.random.default_rng(1)
                    segments.append((0.5 * rng.standard_normal(n)).astype(np.float64))
            else:
                segments.append(_silence(duration_s, sample_rate_hz))
    return np.concatenate(segments)


def _classifier(**overrides) -> TemporalAlarmClassifier:
    kwargs = {"min_repeats": 3, "min_confidence": 0.5}
    kwargs.update(overrides)
    return TemporalAlarmClassifier(**kwargs)


def test_detects_t3_cadence_after_min_repeats() -> None:
    clf = _classifier()
    audio = _pulse_train(T3_TEMPLATE, reps=3, tone_hz=3100.0)
    result = clf.classify(audio, SAMPLE_RATE_HZ)
    assert result.label == "alarm_t3"
    assert result.confidence >= 0.5
    assert result.features["cycle_count"] >= 3


def test_detects_t4_cadence_after_min_repeats() -> None:
    clf = _classifier()
    audio = _pulse_train(T4_TEMPLATE, reps=3, tone_hz=3200.0)
    result = clf.classify(audio, SAMPLE_RATE_HZ)
    assert result.label == "alarm_t4"
    assert result.confidence >= 0.5


def test_rejects_white_noise() -> None:
    clf = _classifier()
    rng = np.random.default_rng(2)
    audio = (0.1 * rng.standard_normal(SAMPLE_RATE_HZ * 21)).astype(np.float64)
    result = clf.classify(audio, SAMPLE_RATE_HZ)
    assert result.label == "unknown"
    assert result.confidence == 0.0


def test_rejects_single_isolated_beep() -> None:
    clf = _classifier()
    audio = np.concatenate(
        [_silence(5.0, SAMPLE_RATE_HZ), _tone(0.5, 3100.0, SAMPLE_RATE_HZ), _silence(10.0, SAMPLE_RATE_HZ)]
    )
    result = clf.classify(audio, SAMPLE_RATE_HZ)
    assert result.label == "unknown"


def test_below_min_repeats_is_rejected() -> None:
    clf = _classifier(min_repeats=3)
    audio = _pulse_train(T3_TEMPLATE, reps=2, tone_hz=3100.0)
    result = clf.classify(audio, SAMPLE_RATE_HZ)
    assert result.label == "unknown"


def test_rejects_broadband_rhythmic_noise_matching_cadence() -> None:
    """Right cadence, wrong spectral content: broadband noise pulses should
    score low confidence via the tone-SNR gate even though the on/off timing
    matches the T3 template."""
    clf = _classifier()
    audio = _pulse_train(T3_TEMPLATE, reps=3, tone_hz=3100.0, on_kind="noise")
    result = clf.classify(audio, SAMPLE_RATE_HZ)
    assert result.label == "unknown" or result.confidence < 0.5


def test_rejects_off_band_tone() -> None:
    """A tone well outside the configured narrowband gate should not register
    as a pulse train even at the right cadence."""
    clf = _classifier(tone_band_low_hz=2800.0, tone_band_high_hz=3500.0)
    audio = _pulse_train(T3_TEMPLATE, reps=3, tone_hz=6500.0)
    result = clf.classify(audio, SAMPLE_RATE_HZ)
    assert result.label == "unknown"


def test_more_repeats_increase_confidence() -> None:
    clf = _classifier(min_repeats=2)
    low = clf.classify(_pulse_train(T3_TEMPLATE, reps=2, tone_hz=3100.0), SAMPLE_RATE_HZ)
    high = clf.classify(_pulse_train(T3_TEMPLATE, reps=5, tone_hz=3100.0), SAMPLE_RATE_HZ)
    assert high.confidence >= low.confidence


def test_empty_input_is_unknown() -> None:
    clf = _classifier()
    result = clf.classify(np.zeros(0, dtype=np.float64), SAMPLE_RATE_HZ)
    assert result.label == "unknown"
    assert result.confidence == 0.0
