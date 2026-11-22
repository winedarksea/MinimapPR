"""Two-threshold BirdNET results (Field Mode W2-tracker, S4).

With ``birdnet_two_threshold_enabled`` the BirdNET backend is built with the
low ``birdnet_localize_to_track_min_confidence`` floor instead of its display
floor, so weak identifications are no longer dropped to "unknown". This
wrapper marks every result whose confidence is under the display floor with
``features["below_display_threshold"] = True``. Downstream, such a detection
feeds a track's species belief but is never shown, alerted on or kept as audio
by itself (see ``detection_is_below_display_threshold``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minimappr.classifiers.base import AudioClassifier, ClassificationResult

BELOW_DISPLAY_THRESHOLD_FEATURE = "below_display_threshold"
DISPLAY_MIN_CONFIDENCE_FEATURE = "display_min_confidence"


@dataclass
class DisplayThresholdAnnotatingClassifier(AudioClassifier):
    classifier: AudioClassifier
    display_min_confidence: float

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        result = self.classifier.classify(samples, sample_rate_hz)
        if result.label == "unknown":
            return result
        features = dict(result.features)
        features[DISPLAY_MIN_CONFIDENCE_FEATURE] = float(self.display_min_confidence)
        features[BELOW_DISPLAY_THRESHOLD_FEATURE] = bool(
            float(result.confidence) < float(self.display_min_confidence)
        )
        return ClassificationResult(
            label=result.label,
            confidence=result.confidence,
            scores=result.scores,
            features=features,
        )

    def close(self) -> None:
        self.classifier.close()

    def cancel_pending(self) -> None:
        cancel = getattr(self.classifier, "cancel_pending", None)
        if callable(cancel):
            cancel()

    def __getattr__(self, name: str):
        # Expose the wrapped backend's stats (batches_dispatched, labels, ...).
        if name == "classifier":
            raise AttributeError(name)
        return getattr(self.classifier, name)


def detection_is_below_display_threshold(features: dict | None) -> bool:
    """Whether the *winning* member flagged this result as below display.

    Composite feature keys carry a ``member:`` prefix for non-primary members,
    so the winner's flag is looked up under its own prefix first.
    """
    if not features:
        return False
    winner = str(features.get("winner_member") or "").strip()
    winner_prefix = f"{winner}:"
    winner_is_prefixed_member = bool(winner) and any(
        isinstance(key, str) and key.startswith(winner_prefix) for key in features
    )
    if winner_is_prefixed_member:
        return bool(features.get(f"{winner_prefix}{BELOW_DISPLAY_THRESHOLD_FEATURE}", False))
    # Winner is the primary (unprefixed) member, or there is no composite.
    return bool(features.get(BELOW_DISPLAY_THRESHOLD_FEATURE, False))
