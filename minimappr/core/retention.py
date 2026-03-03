"""Retention tier determination for detections.

Extracted from FusionNode god-class to enable independent testing and
configuration without coupling to the full ingest pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from minimappr.models import RetentionTier


@dataclass(slots=True)
class RetentionPolicy:
    """Determines the storage retention tier for a detection.

    Uses exact label matching against a configured set of permanent labels
    (no substring matching) and category-based promotion for security labels
    exceeding a confidence threshold.
    """

    permanent_labels: set[str]
    security_long_confidence: float

    def tier_for_detection(
        self,
        *,
        label: str,
        label_category: str,
        confidence: float,
        suppressed_by_zone: bool,
    ) -> str:
        if suppressed_by_zone:
            return RetentionTier.EPHEMERAL.value
        normalized = label.strip().lower()
        if normalized in self.permanent_labels:
            return RetentionTier.PERMANENT.value
        if label_category == "security" and confidence >= self.security_long_confidence:
            return RetentionTier.LONG.value
        return RetentionTier.SHORT.value
