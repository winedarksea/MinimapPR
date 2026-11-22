"""SNR-graded, multi-label species oracle that emulates BirdNET's output shape.

The S-gate (``scripts/server_track_gate.py``) needs a classifier whose
confidence falls with the call's in-band SNR, reports several species at once
and drops sub-floor results to "unknown" exactly like the BirdNET member does.
The spectral-peak oracle (``DominantFrequencySpeciesClassifier``) always says
0.9 for one label, which leaves species log-odds, two-threshold BirdNET and
LOCO's collapse rule with nothing to act on.

Model of the score: each species' band SNR is the mean power in its band over
the window's median bin power (a narrowband call barely moves the median, so the
median is the noise floor). Confidence is a logistic of that SNR, capped below
1. Results carry ``features["model"] = "birdnet_v2m4"`` so the production
category resolution treats them as BirdNET provenance (wildlife), and the
BirdNET floors come from ``Settings.birdnet_threshold_set()``; with
``birdnet_two_threshold_enabled`` the oracle is wrapped like the factory wraps
the real member (low floor + ``DisplayThresholdAnnotatingClassifier``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minimappr.classifiers.base import AudioClassifier, ClassificationResult
from minimappr.classifiers.birdnet_two_threshold import DisplayThresholdAnnotatingClassifier

ORACLE_MODEL_NAME = "birdnet_v2m4"
ABSTENTION_LABEL = "unknown"
# Logistic SNR -> confidence: 50 % at 12 dB band SNR, 3 dB per e-fold, capped
# where BirdNET's saturated scores sit on clean fixture calls.
CONFIDENCE_MIDPOINT_SNR_DB = 12.0
CONFIDENCE_SLOPE_DB = 3.0
CONFIDENCE_CEILING = 0.95
# Species below this confidence are left out of ``scores`` (BirdNET reports top-k).
REPORTED_SCORE_MIN_CONFIDENCE = 0.05
# Bins under this frequency (wind/HPF region) are kept out of the floor estimate.
NOISE_FLOOR_MIN_FREQUENCY_HZ = 500.0
POWER_EPSILON = 1e-30


def band_snr_db(samples: np.ndarray, sample_rate_hz: int, bands_hz: dict[str, tuple[float, float]]) -> dict[str, float]:
    mono = np.asarray(samples, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=0)
    if mono.size == 0:
        return {label: 0.0 for label in bands_hz}
    power = np.abs(np.fft.rfft(mono * np.hanning(mono.size))) ** 2
    frequencies_hz = np.fft.rfftfreq(mono.size, 1.0 / sample_rate_hz)
    floor = float(np.median(power[frequencies_hz >= NOISE_FLOOR_MIN_FREQUENCY_HZ])) + POWER_EPSILON
    snr: dict[str, float] = {}
    for label, (low_hz, high_hz) in bands_hz.items():
        in_band = (frequencies_hz >= low_hz) & (frequencies_hz <= high_hz)
        band_power = float(np.mean(power[in_band])) if np.any(in_band) else 0.0
        snr[label] = 10.0 * np.log10((band_power + POWER_EPSILON) / floor)
    return snr


def confidence_for_snr_db(snr_db: float) -> float:
    return CONFIDENCE_CEILING / (1.0 + float(np.exp(-(snr_db - CONFIDENCE_MIDPOINT_SNR_DB) / CONFIDENCE_SLOPE_DB)))


@dataclass
class SnrGradedSpeciesOracle(AudioClassifier):
    species_bands_hz: dict[str, tuple[float, float]]
    min_confidence: float

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        snr = band_snr_db(samples, sample_rate_hz, self.species_bands_hz)
        confidences = {label: confidence_for_snr_db(value) for label, value in snr.items()}
        scores = {label: c for label, c in confidences.items() if c >= REPORTED_SCORE_MIN_CONFIDENCE}
        features = {"model": ORACLE_MODEL_NAME, "band_snr_db": {k: round(v, 2) for k, v in snr.items()}}
        if not scores:
            return ClassificationResult(label=ABSTENTION_LABEL, confidence=0.0, scores={}, features=features)
        label = max(scores, key=scores.get)
        if scores[label] < self.min_confidence:
            # BirdNET's member floor: a result under min_confidence is an abstention.
            return ClassificationResult(label=ABSTENTION_LABEL, confidence=0.0, scores=scores, features=features)
        return ClassificationResult(label=label, confidence=scores[label], scores=scores, features=features)


def build_graded_oracle_for_settings(settings, species_bands_hz: dict[str, tuple[float, float]]) -> AudioClassifier:
    """Mirror ``classifiers.factory`` for the BirdNET member: one floor, or two with the wrapper."""
    thresholds = settings.birdnet_threshold_set()
    display_floor = float(thresholds.trigger_min_confidence)
    if not bool(getattr(settings, "birdnet_two_threshold_enabled", False)):
        return SnrGradedSpeciesOracle(species_bands_hz, min_confidence=display_floor)
    low_floor = min(float(thresholds.localize_to_track_min_confidence), display_floor)
    return DisplayThresholdAnnotatingClassifier(
        classifier=SnrGradedSpeciesOracle(species_bands_hz, min_confidence=low_floor),
        display_min_confidence=display_floor,
    )
