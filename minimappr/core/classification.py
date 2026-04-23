"""Classification orchestration — omni vs. beamformed path selection, label
resolution, and confidence comparison.

Extracted from FusionNode to isolate classification logic as an independently
testable component. The orchestrator decides which audio path (omnidirectional
reference sensor vs. spatially beamformed) produces the best classification and
resolves taxonomy metadata for the winning result.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from minimappr.classifiers.base import AudioClassifier, ClassificationResult
from minimappr.config import LocalizationConfig
from minimappr.core.preprocessing import create_classification_preprocessor
from minimappr.interfaces import (
    AudioPreprocessor,
    Beamformer,
    EnvironmentProvider,
    StorageBackend,
    TaxonomyProvider,
)
from minimappr.models import LabelId

logger = logging.getLogger(__name__)

_CLASSIFICATION_TIMEOUT_S = 30.0


@dataclass(slots=True)
class ClassifiedResult:
    """Output of the classification orchestrator.

    Contains the winning classification, path metadata, and resolved taxonomy
    fields needed by downstream assembly and rules stages.
    """

    classification: ClassificationResult
    omni_classification: ClassificationResult
    beamformed_classification: ClassificationResult | None
    classification_path: str
    classification_signal: np.ndarray
    beamforming_error: str | None

    # Resolved taxonomy
    label_category: str
    iff_category: str
    label_id: LabelId | None


class ClassificationOrchestrator:
    """Runs the omni + beamformed dual classification path.

    Pipeline:
      1. Classify the omnidirectional (reference sensor) signal.
      2. If beamforming is available and meets sensor-count requirements,
         beamform toward the localized position with recall-biased settings,
         optionally apply pre-classification preprocessing, then classify again.
      3. Pick the higher-confidence result (with configurable margin).
      4. Resolve taxonomy (category, IFF, label_id) for the winning label.
    """

    def __init__(
        self,
        *,
        classifier: AudioClassifier,
        storage: StorageBackend,
        taxonomy_provider: TaxonomyProvider,
        environment_provider: EnvironmentProvider,
        beamformer: Beamformer | None = None,
        classification_preprocessor: AudioPreprocessor | None = None,
        beamformed_classification_min_sensor_count: int = 3,
        beamformed_classification_confidence_margin: float = 0.0,
        classifier_backend_name: str = "heuristic",
    ) -> None:
        self._classifier = classifier
        self._storage = storage
        self._taxonomy_provider = taxonomy_provider
        self._environment_provider = environment_provider
        self._beamformer = beamformer
        self._classification_preprocessor = classification_preprocessor
        self._beamformed_min_sensors = beamformed_classification_min_sensor_count
        self._confidence_margin = max(0.0, beamformed_classification_confidence_margin)
        self._classifier_backend_name = classifier_backend_name

    async def classify(
        self,
        *,
        reference_signal: np.ndarray,
        sample_rate_hz: int,
        capability_tier: str,
        selected_sensor_ids: list[str],
        selected_positions: dict[str, np.ndarray],
        selected_windows: dict[str, np.ndarray],
        localization_position_m: tuple[float, float, float],
        event_time_ns: int,
    ) -> ClassifiedResult:
        """Run classification pipeline and return the best result with taxonomy."""
        omni_signal = reference_signal
        if self._classification_preprocessor is not None:
            omni_signal = self._classification_preprocessor.process(
                omni_signal,
                sample_rate_hz,
            )

        omni_classification = await self._classify_with_timeout(omni_signal, sample_rate_hz)
        classification = omni_classification
        classification_signal = omni_signal
        classification_path = "omni"
        beamformed_classification: ClassificationResult | None = None
        beamforming_error: str | None = None

        if (
            self._beamformer is not None
            and capability_tier != "alerting_only"
            and len(selected_sensor_ids) >= self._beamformed_min_sensors
        ):
            try:
                sound_speed = self._environment_provider.get_speed_of_sound(
                    localization_position_m
                )
                beamformed_signal = await asyncio.to_thread(
                    self._beamformer.beamform,
                    selected_positions,
                    selected_windows,
                    sample_rate_hz,
                    localization_position_m,
                    sound_speed,
                )

                if self._classification_preprocessor is not None:
                    beamformed_signal = self._classification_preprocessor.process(
                        beamformed_signal,
                        sample_rate_hz,
                    )

                beamformed_classification = await self._classify_with_timeout(
                    beamformed_signal, sample_rate_hz
                )
                if beamformed_classification.confidence > (
                    omni_classification.confidence + self._confidence_margin
                ):
                    classification = beamformed_classification
                    classification_signal = beamformed_signal
                    classification_path = f"beamformed:{self._beamformer.__class__.__name__}"
            except Exception as exc:  # pragma: no cover - resilience path
                beamforming_error = f"{type(exc).__name__}: {exc}"

        return await self._build_result(
            classification=classification,
            omni_classification=omni_classification,
            beamformed_classification=beamformed_classification,
            classification_path=classification_path,
            classification_signal=classification_signal,
            beamforming_error=beamforming_error,
            event_time_ns=event_time_ns,
        )

    async def classify_omni_only(
        self,
        *,
        reference_signal: np.ndarray,
        sample_rate_hz: int,
        event_time_ns: int,
    ) -> ClassifiedResult:
        """Classify a full-band omni signal without beamforming."""
        omni_signal = reference_signal
        if self._classification_preprocessor is not None:
            omni_signal = self._classification_preprocessor.process(
                omni_signal,
                sample_rate_hz,
            )

        omni_classification = await self._classify_with_timeout(omni_signal, sample_rate_hz)
        return await self._build_result(
            classification=omni_classification,
            omni_classification=omni_classification,
            beamformed_classification=None,
            classification_path="omni",
            classification_signal=omni_signal,
            beamforming_error=None,
            event_time_ns=event_time_ns,
        )

    async def _classify_with_timeout(
        self, signal: np.ndarray, sample_rate_hz: int
    ) -> ClassificationResult:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._classifier.classify, signal, sample_rate_hz),
                timeout=_CLASSIFICATION_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Classification timed out after %.0fs", _CLASSIFICATION_TIMEOUT_S
            )
            return ClassificationResult(
                label="timeout",
                confidence=0.0,
                scores={},
                features={"reason": "classification_timeout"},
            )

    async def _build_result(
        self,
        *,
        classification: ClassificationResult,
        omni_classification: ClassificationResult,
        beamformed_classification: ClassificationResult | None,
        classification_path: str,
        classification_signal: np.ndarray,
        beamforming_error: str | None,
        event_time_ns: int,
    ) -> ClassifiedResult:
        label_category = self._taxonomy_provider.category_for_label(classification.label)
        iff_category = self._taxonomy_provider.iff_for_category(label_category)
        label_id = await self._storage.upsert_label(
            name=classification.label,
            category=label_category,
            source=self._classifier_backend_name,
            created_ns=event_time_ns,
        )
        if hasattr(self._taxonomy_provider, "register_label"):
            self._taxonomy_provider.register_label(classification.label, label_category)
        return ClassifiedResult(
            classification=classification,
            omni_classification=omni_classification,
            beamformed_classification=beamformed_classification,
            classification_path=classification_path,
            classification_signal=classification_signal,
            beamforming_error=beamforming_error,
            label_category=label_category,
            iff_category=iff_category,
            label_id=label_id,
        )
