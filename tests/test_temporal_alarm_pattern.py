"""T3/T4 template-fidelity, tuning, and efficiency tests.

Companion to ``test_temporal_alarm_classifier.py``. Where that file exercises
end-to-end detect/reject behaviour, this file pins down three separate claims:

1. The ``T3_TEMPLATE`` / ``T4_TEMPLATE`` constants encode the *correct*
   standardized cadences (ISO 8201-1 temporal-three; the 4-pulse temporal-four
   the detector is tuned for) — pulse counts, strict on/off alternation, per-run
   durations, and total cycle length — so a silent edit to a template can't
   drift the detector off-spec unnoticed.
2. A signal synthesized at *exactly* those cadences is detected with the right
   label and cycle count (the template and the matcher agree).
3. ``classify()`` stays cheap on a full-length omni window (the whole point of a
   classic-DSP detector rather than a neural one).
"""

from __future__ import annotations

import time

import numpy as np

from minimappr.classifiers.temporal_alarm import (
    T3_TEMPLATE,
    T4_TEMPLATE,
    TemporalAlarmClassifier,
)

SAMPLE_RATE_HZ = 16_000
_PULSE_S = 0.5
_GAP_S = 0.5
_TAIL_GAP_S = 1.5
_TOL = 1e-9


def _tone(duration_s: float, freq_hz: float, *, amplitude: float = 0.5, noise: float = 0.0) -> np.ndarray:
    n = int(round(duration_s * SAMPLE_RATE_HZ))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE_HZ
    out = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
    if noise:
        out = out + noise * np.random.default_rng(7).standard_normal(n)
    return out.astype(np.float64)


def _silence(duration_s: float, *, noise: float = 0.0) -> np.ndarray:
    n = int(round(duration_s * SAMPLE_RATE_HZ))
    if noise:
        return (noise * np.random.default_rng(8).standard_normal(n)).astype(np.float64)
    return np.zeros(n, dtype=np.float64)


def _train(template, reps: int, *, tone_hz: float = 3100.0, noise: float = 0.0) -> np.ndarray:
    segments = []
    for _ in range(reps):
        for state, duration_s in template:
            if state == 1:
                segments.append(_tone(duration_s, tone_hz, noise=noise))
            else:
                segments.append(_silence(duration_s, noise=noise))
    return np.concatenate(segments)


# --------------------------------------------------------------------------- #
# 1. Template fidelity — the constants encode the correct standardized cadence
# --------------------------------------------------------------------------- #
def _pulse_count(template) -> int:
    return sum(1 for state, _ in template if state == 1)


def test_templates_strictly_alternate_on_off_starting_on() -> None:
    for template in (T3_TEMPLATE, T4_TEMPLATE):
        states = [state for state, _ in template]
        assert states[0] == 1, "a cadence must begin with a tone pulse"
        assert states[-1] == 0, "a cadence must end on the inter-cycle gap"
        # No two adjacent runs share a state (otherwise they'd be one run).
        assert all(a != b for a, b in zip(states, states[1:])), states


def test_t3_template_matches_iso8201_temporal_three() -> None:
    # Temporal-three: three 0.5s pulses separated by 0.5s gaps, then a 1.5s gap.
    assert _pulse_count(T3_TEMPLATE) == 3
    pulses = [d for s, d in T3_TEMPLATE if s == 1]
    gaps = [d for s, d in T3_TEMPLATE if s == 0]
    assert all(abs(d - _PULSE_S) < _TOL for d in pulses), pulses
    assert gaps[:-1] == [_GAP_S] * (len(gaps) - 1)
    assert abs(gaps[-1] - _TAIL_GAP_S) < _TOL
    assert abs(sum(d for _, d in T3_TEMPLATE) - 4.0) < _TOL  # 4.0s cycle


def test_t4_template_is_four_pulse_five_second_cycle() -> None:
    # Temporal-four: four 0.5s pulses / 0.5s gaps, then a 1.5s gap -> 5.0s cycle.
    assert _pulse_count(T4_TEMPLATE) == 4
    pulses = [d for s, d in T4_TEMPLATE if s == 1]
    gaps = [d for s, d in T4_TEMPLATE if s == 0]
    assert all(abs(d - _PULSE_S) < _TOL for d in pulses), pulses
    assert gaps[:-1] == [_GAP_S] * (len(gaps) - 1)
    assert abs(gaps[-1] - _TAIL_GAP_S) < _TOL
    assert abs(sum(d for _, d in T4_TEMPLATE) - 5.0) < _TOL  # 5.0s cycle


def test_t4_has_exactly_one_more_pulse_than_t3() -> None:
    # The sole discriminant between the two labels is the pulse count.
    assert _pulse_count(T4_TEMPLATE) == _pulse_count(T3_TEMPLATE) + 1


# --------------------------------------------------------------------------- #
# 2. Template <-> matcher agreement on the exact cadence
# --------------------------------------------------------------------------- #
def test_exact_t3_cadence_detected_with_expected_cycle_count() -> None:
    clf = TemporalAlarmClassifier(min_repeats=3)
    result = clf.classify(_train(T3_TEMPLATE, reps=4), SAMPLE_RATE_HZ)
    assert result.label == "alarm_t3"
    assert result.features["cycle_count"] >= 3
    assert 2800.0 <= result.features["tone_hz"] <= 3500.0


def test_exact_t4_cadence_detected_and_not_confused_with_t3() -> None:
    clf = TemporalAlarmClassifier(min_repeats=3)
    result = clf.classify(_train(T4_TEMPLATE, reps=4), SAMPLE_RATE_HZ)
    assert result.label == "alarm_t4"
    assert result.scores["alarm_t4"] > result.scores["alarm_t3"]


def test_noisy_cadence_still_detected_at_min_repeats() -> None:
    # Regression: with realistic background noise a single one-frame envelope
    # blip in a long gap used to desync the fixed-stride matcher, dropping the
    # matched-cycle count below min_repeats. The despeckle pass must prevent it.
    clf = TemporalAlarmClassifier(min_repeats=3)
    result = clf.classify(_train(T3_TEMPLATE, reps=4, noise=0.003), SAMPLE_RATE_HZ)
    assert result.label == "alarm_t3"
    assert result.features["cycle_count"] >= 3


# --------------------------------------------------------------------------- #
# 3. Tuning knobs — tolerance + configurable hysteresis
# --------------------------------------------------------------------------- #
def test_hysteresis_ratios_are_sanitized() -> None:
    # lo must never exceed hi (that would invert the gate); it is clamped.
    clf = TemporalAlarmClassifier(hysteresis_hi_ratio=4.0, hysteresis_lo_ratio=9.0)
    assert clf.hysteresis_lo_ratio <= clf.hysteresis_hi_ratio


def test_tighter_tolerance_rejects_off_cadence_that_loose_tolerance_accepts() -> None:
    # A cadence stretched ~25% off-nominal: inside a loose 0.30 band, outside a
    # tight 0.12 band. Tightening tolerance is what buys the false-positive
    # reduction, so the two settings must actually diverge here.
    stretched = tuple((s, d * 1.25) for s, d in T3_TEMPLATE)
    audio = _train(stretched, reps=4)
    loose = TemporalAlarmClassifier(min_repeats=3, tolerance=0.30).classify(audio, SAMPLE_RATE_HZ)
    tight = TemporalAlarmClassifier(min_repeats=3, tolerance=0.12).classify(audio, SAMPLE_RATE_HZ)
    assert loose.label == "alarm_t3"
    assert tight.label == "unknown"


def test_default_tolerance_is_tightened_from_original() -> None:
    # Guardrail on the tuning decision: default must stay well below the old 0.3.
    assert TemporalAlarmClassifier().tolerance <= 0.2


# --------------------------------------------------------------------------- #
# 4. Efficiency — full 21s omni window must classify cheaply
# --------------------------------------------------------------------------- #
def test_classify_is_efficient_on_full_window() -> None:
    clf = TemporalAlarmClassifier()
    audio = _train(T3_TEMPLATE, reps=6)[: 21 * SAMPLE_RATE_HZ]  # 6 x 4.0s -> trim to 21s
    assert audio.size == 21 * SAMPLE_RATE_HZ
    clf.classify(audio, SAMPLE_RATE_HZ)  # warm up (basis construction, BLAS)

    best = min(_timed(clf, audio) for _ in range(5))
    # Measured ~5ms; assert a generous ceiling that still fails loudly if the
    # vectorized DSP is ever replaced by a per-frame Python loop.
    assert best < 0.10, f"classify() too slow on 21s window: {best * 1e3:.1f} ms"


def _timed(clf: TemporalAlarmClassifier, audio: np.ndarray) -> float:
    start = time.perf_counter()
    clf.classify(audio, SAMPLE_RATE_HZ)
    return time.perf_counter() - start
