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
from dataclasses import dataclass, replace
from typing import Any, Callable

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
from minimappr.utils.audio import energy_contrast_db, framed_spectral_flatness_median

logger = logging.getLogger(__name__)


_BIRDNET_MODEL_NAME = "birdnet_v2m4"

# BirdNET's entire vocabulary is bird species, so any label it produces is
# wildlife by construction. Models whose output vocabulary maps wholesale onto a
# single taxonomy category are listed here.
_MODEL_FALLBACK_CATEGORY = {
    _BIRDNET_MODEL_NAME: "wildlife",
}


def _resolve_label_category(
    classification: ClassificationResult,
    taxonomy_provider: TaxonomyProvider,
) -> str:
    """Resolve taxonomy first, falling back to the model's own vocabulary category.

    BirdNET species labels are open-ended, so name-only taxonomy heuristics cannot
    reliably identify every result as BirdNET-derived.  A configured or heuristic
    category remains authoritative; model provenance is only the unknown fallback.

    The fallback yields a real taxonomy category (``wildlife``), not the model id.
    Emitting ``birdnet_v2m4`` as a *category* meant it was absent from
    ``DEFAULT_CATEGORY_TO_IFF``, so every BirdNET detection still resolved to
    ``iff_category="unknown"`` and no rule keyed on ``wildlife`` could ever match.
    Model provenance already lives in ``classification.features["model"]``.
    """
    taxonomy_category = taxonomy_provider.category_for_label(classification.label)
    if taxonomy_category.strip().lower() != "unknown":
        return taxonomy_category
    model_name = classification.features.get("model")
    if isinstance(model_name, str):
        fallback = _MODEL_FALLBACK_CATEGORY.get(model_name.strip().lower())
        if fallback is not None:
            return fallback
    return taxonomy_category


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

    # Set True when the backend raised an exception (degraded to unknown)
    backend_failed: bool = False


@dataclass(slots=True)
class _ClassificationPathCandidate:
    classification: ClassificationResult
    signal: np.ndarray
    path: str


@dataclass(slots=True)
class _BeamformedClassificationAttempt:
    candidate: _ClassificationPathCandidate | None
    error: str | None = None


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
        beamformed_classification_min_sensor_count: int = 2,
        beamformed_classification_confidence_margin: float = 0.0,
        stage_timeout_seconds: float = 30.0,
        classifier_backend_name: str = "routing",
        on_beamform_error: Callable[[str], None] | None = None,
        texture_gate_enabled: bool = True,
        texture_gate_contrast_db: float = 8.0,
        texture_gate_flatness_min: float = 0.2,
        texture_gate_confidence_factor: float = 1.0,
        on_texture_gate_demotion: Callable[[], None] | None = None,
    ) -> None:
        self._classifier = classifier
        self._storage = storage
        self._taxonomy_provider = taxonomy_provider
        self._environment_provider = environment_provider
        self._beamformer = beamformer
        self._classification_preprocessor = classification_preprocessor
        self._beamformed_min_sensors = beamformed_classification_min_sensor_count
        self._confidence_margin = max(0.0, beamformed_classification_confidence_margin)
        self._stage_timeout_seconds = max(0.001, float(stage_timeout_seconds))
        self._classifier_backend_name = classifier_backend_name
        self._on_beamform_error = on_beamform_error
        self._texture_gate_enabled = bool(texture_gate_enabled)
        self._texture_gate_contrast_db = float(texture_gate_contrast_db)
        self._texture_gate_flatness_min = float(texture_gate_flatness_min)
        self._texture_gate_confidence_factor = float(texture_gate_confidence_factor)
        self._on_texture_gate_demotion = on_texture_gate_demotion

    def replace_classifier(self, classifier: AudioClassifier) -> None:
        self._classifier = classifier

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
        alias_cutoff_hz: float | None = None,
    ) -> ClassifiedResult:
        """Run classification pipeline and return the best result with taxonomy."""
        omni_candidate = await self._classify_omni_candidate(
            reference_signal=reference_signal,
            sample_rate_hz=sample_rate_hz,
        )
        backend_failed = self._classification_backend_failed(omni_candidate.classification)
        beamformed_attempt = await self._classify_beamformed_candidate_if_eligible(
            capability_tier=capability_tier,
            sample_rate_hz=sample_rate_hz,
            selected_sensor_ids=selected_sensor_ids,
            selected_positions=selected_positions,
            selected_windows=selected_windows,
            localization_position_m=localization_position_m,
            alias_cutoff_hz=alias_cutoff_hz,
        )
        winning_candidate = self._select_winning_candidate(
            omni_candidate=omni_candidate,
            beamformed_candidate=beamformed_attempt.candidate,
        )
        beamformed_classification = (
            None
            if beamformed_attempt.candidate is None
            else beamformed_attempt.candidate.classification
        )

        return await self._build_result(
            classification=winning_candidate.classification,
            omni_classification=omni_candidate.classification,
            beamformed_classification=beamformed_classification,
            classification_path=winning_candidate.path,
            classification_signal=winning_candidate.signal,
            sample_rate_hz=sample_rate_hz,
            beamforming_error=beamformed_attempt.error,
            event_time_ns=event_time_ns,
            backend_failed=backend_failed,
        )

    async def classify_omni_only(
        self,
        *,
        reference_signal: np.ndarray,
        sample_rate_hz: int,
        event_time_ns: int,
        path: str = "omni",
    ) -> ClassifiedResult:
        """Classify a full-band omni signal without beamforming.

        ``path`` stamps the classification provenance (e.g. ``"nearest_node_omni"``
        for the raw nearest-node source); defaults to ``"omni"``.
        """
        omni_candidate = await self._classify_omni_candidate(
            reference_signal=reference_signal,
            sample_rate_hz=sample_rate_hz,
        )
        backend_failed = self._classification_backend_failed(omni_candidate.classification)
        return await self._build_result(
            classification=omni_candidate.classification,
            omni_classification=omni_candidate.classification,
            beamformed_classification=None,
            classification_path=path,
            classification_signal=omni_candidate.signal,
            sample_rate_hz=sample_rate_hz,
            beamforming_error=None,
            event_time_ns=event_time_ns,
            backend_failed=backend_failed,
        )

    async def classify_beamformed_only(
        self,
        *,
        sample_rate_hz: int,
        capability_tier: str,
        selected_sensor_ids: list[str],
        selected_positions: dict[str, np.ndarray],
        selected_windows: dict[str, np.ndarray],
        localization_position_m: tuple[float, float, float],
        event_time_ns: int,
        alias_cutoff_hz: float | None = None,
    ) -> ClassifiedResult | None:
        """Classify only a spatial render, returning nothing when unavailable.

        The trigger pipeline must not quietly fall back to raw omni inference:
        continuous omni scanning owns that responsibility.  Returning ``None``
        also makes an insufficient array a clear no-classification outcome.
        """
        beamformed_attempt = await self._classify_beamformed_candidate_if_eligible(
            capability_tier=capability_tier,
            sample_rate_hz=sample_rate_hz,
            selected_sensor_ids=selected_sensor_ids,
            selected_positions=selected_positions,
            selected_windows=selected_windows,
            localization_position_m=localization_position_m,
            alias_cutoff_hz=alias_cutoff_hz,
        )
        candidate = beamformed_attempt.candidate
        if candidate is None:
            return None
        backend_failed = self._classification_backend_failed(candidate.classification)
        return await self._build_result(
            classification=candidate.classification,
            # Preserve the existing reporting contract's scalar omni field
            # without claiming an additional raw-omni inference occurred.
            omni_classification=candidate.classification,
            beamformed_classification=candidate.classification,
            classification_path=candidate.path,
            classification_signal=candidate.signal,
            sample_rate_hz=sample_rate_hz,
            beamforming_error=beamformed_attempt.error,
            event_time_ns=event_time_ns,
            backend_failed=backend_failed,
        )

    async def adopt_authoritative_classification(
        self,
        *,
        classification: ClassificationResult,
        event_time_ns: int,
        classification_signal: np.ndarray,
        sample_rate_hz: int,
        classification_path: str = "rust_manifest",
    ) -> ClassifiedResult:
        """Adopt a precomputed classification result without re-running inference."""
        authoritative_classification = classification.model_copy(deep=True)
        return await self._build_result(
            classification=authoritative_classification,
            omni_classification=authoritative_classification.model_copy(deep=True),
            beamformed_classification=None,
            classification_path=classification_path,
            classification_signal=classification_signal,
            sample_rate_hz=sample_rate_hz,
            beamforming_error=None,
            event_time_ns=event_time_ns,
        )

    def _prepare_signal_for_classification(
        self,
        signal: np.ndarray,
        sample_rate_hz: int,
    ) -> np.ndarray:
        if self._classification_preprocessor is None:
            return signal
        return self._classification_preprocessor.process(signal, sample_rate_hz)

    async def _classify_omni_candidate(
        self,
        *,
        reference_signal: np.ndarray,
        sample_rate_hz: int,
    ) -> _ClassificationPathCandidate:
        omni_signal = self._prepare_signal_for_classification(reference_signal, sample_rate_hz)
        return _ClassificationPathCandidate(
            classification=await self._classify_with_timeout(omni_signal, sample_rate_hz),
            signal=omni_signal,
            path="omni",
        )

    def _should_attempt_beamformed_classification(
        self,
        *,
        capability_tier: str,
        selected_sensor_ids: list[str],
    ) -> bool:
        return (
            self._beamformer is not None
            and capability_tier != "alerting_only"
            and len(selected_sensor_ids) >= self._beamformed_min_sensors
        )

    async def _classify_beamformed_candidate_if_eligible(
        self,
        *,
        capability_tier: str,
        sample_rate_hz: int,
        selected_sensor_ids: list[str],
        selected_positions: dict[str, np.ndarray],
        selected_windows: dict[str, np.ndarray],
        localization_position_m: tuple[float, float, float],
        alias_cutoff_hz: float | None = None,
    ) -> _BeamformedClassificationAttempt:
        if not self._should_attempt_beamformed_classification(
            capability_tier=capability_tier,
            selected_sensor_ids=selected_sensor_ids,
        ):
            return _BeamformedClassificationAttempt(candidate=None)

        assert self._beamformer is not None
        beamformer = self._resolve_beamformer(alias_cutoff_hz)
        try:
            sound_speed = self._environment_provider.get_speed_of_sound(localization_position_m)
            beamformed_signal = await asyncio.to_thread(
                beamformer.beamform,
                selected_positions,
                selected_windows,
                sample_rate_hz,
                localization_position_m,
                sound_speed,
            )
            prepared_signal = self._prepare_signal_for_classification(
                beamformed_signal,
                sample_rate_hz,
            )
            return _BeamformedClassificationAttempt(
                candidate=_ClassificationPathCandidate(
                    classification=await self._classify_with_timeout(prepared_signal, sample_rate_hz),
                    signal=prepared_signal,
                    path=f"beamformed:{self._beamformer.__class__.__name__}",
                )
            )
        except Exception as exc:  # pragma: no cover - resilience path
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("Beamformed classification failed; falling back to omni: %s", error)
            if self._on_beamform_error is not None:
                try:
                    self._on_beamform_error(error)
                except Exception:  # noqa: BLE001 - metrics hook must not break classification
                    logger.exception("on_beamform_error hook failed")
            return _BeamformedClassificationAttempt(candidate=None, error=error)

    def _resolve_beamformer(self, alias_cutoff_hz: float | None) -> Beamformer:
        """Return the beamformer, specialised with the event's alias cutoff.

        A per-call copy keeps the shared instance immutable under concurrent
        classification tasks.
        """
        assert self._beamformer is not None
        from minimappr.core.beamforming import BandSplitDasRenderer

        if alias_cutoff_hz is not None and isinstance(self._beamformer, BandSplitDasRenderer):
            return replace(self._beamformer, alias_cutoff_hz=float(alias_cutoff_hz))
        return self._beamformer

    def _select_winning_candidate(
        self,
        *,
        omni_candidate: _ClassificationPathCandidate,
        beamformed_candidate: _ClassificationPathCandidate | None,
    ) -> _ClassificationPathCandidate:
        if beamformed_candidate is None:
            return omni_candidate
        if beamformed_candidate.classification.confidence > (
            omni_candidate.classification.confidence + self._confidence_margin
        ):
            return beamformed_candidate
        return omni_candidate

    def _classification_backend_failed(self, classification: ClassificationResult) -> bool:
        return classification.features.get("reason") == "classification_error"

    async def _run_classifier_with_timeout(
        self,
        signal: np.ndarray,
        sample_rate_hz: int,
    ) -> ClassificationResult:
        return await asyncio.wait_for(
            asyncio.to_thread(self._classifier.classify, signal, sample_rate_hz),
            timeout=self._stage_timeout_seconds,
        )

    async def _classify_with_timeout(
        self, signal: np.ndarray, sample_rate_hz: int
    ) -> ClassificationResult:
        try:
            return await self._run_classifier_with_timeout(signal, sample_rate_hz)
        except asyncio.TimeoutError:
            return self._timeout_degradation_result(
                warning_message="Classification timed out after %.3fs",
                cancellation_failure_message="Classifier cancellation hook failed after timeout",
            )
        except Exception as exc:  # noqa: BLE001 - classifier backends are optional/runtime-boundary code
            retry_result = await self._retry_cancelled_classification(
                signal,
                sample_rate_hz,
                exc,
            )
            if retry_result is not None:
                return retry_result
            return self._exception_degradation_result(exc)

    async def _retry_cancelled_classification(
        self,
        signal: np.ndarray,
        sample_rate_hz: int,
        exc: Exception,
    ) -> ClassificationResult | None:
        message = str(exc).strip()
        if "cancel" not in message.lower():
            return None

        logger.warning("Classification backend cancelled analysis; retrying once: %s", message)
        try:
            return await self._run_classifier_with_timeout(signal, sample_rate_hz)
        except asyncio.TimeoutError:
            return self._timeout_degradation_result(
                warning_message="Classification retry timed out after %.3fs",
                cancellation_failure_message="Classifier cancellation hook failed after retry timeout",
            )
        except Exception as retry_exc:  # noqa: BLE001
            return self._exception_degradation_result(retry_exc)

    def _timeout_degradation_result(
        self,
        *,
        warning_message: str,
        cancellation_failure_message: str,
    ) -> ClassificationResult:
        logger.warning(warning_message, self._stage_timeout_seconds)
        try:
            self._classifier.cancel_pending()
        except Exception:  # noqa: BLE001 - timeout path should never crash classification
            logger.exception(cancellation_failure_message)
        return ClassificationResult(
            label="timeout",
            confidence=0.0,
            scores={},
            features={"reason": "classification_timeout"},
        )

    def _exception_degradation_result(self, exc: Exception) -> ClassificationResult:
        message = str(exc).strip() or type(exc).__name__
        logger.warning("Classification backend failed; degrading to unknown: %s", message)
        return ClassificationResult(
            label="unknown",
            confidence=0.0,
            scores={},
            features={
                "reason": "classification_error",
                "error_type": type(exc).__name__,
            },
        )

    def _apply_texture_gate(
        self,
        *,
        classification: ClassificationResult,
        signal: np.ndarray | None,
        sample_rate_hz: int,
    ) -> ClassificationResult:
        """Flag (and optionally demote) results drawn from featureless noise.

        A classification window holding nothing but the node's noise floor is
        boosted ~30 dB by the bounded-RMS AGC, and YAMNet labels the resulting
        hiss confidently ("Zipper (clothing)", "Fireworks", ...). Two
        level-invariant discriminators separate that from a real event, however
        faint: energy contrast (a real event punches above its own floor) and
        spectral flatness (a steady tonal source such as a distant drone is
        strongly non-flat even without contrast). Both must indicate noise.

        This runs after the winner is selected, so it can never change which
        path wins, and it never rewrites ``label`` or ``scores`` — a demoted
        detection is still persisted, just below the alerting and default-UI
        confidence floors.
        """
        if not self._texture_gate_enabled:
            return classification
        if classification.features.get("reason") in {
            "classification_timeout",
            "classification_error",
        }:
            return classification
        if signal is None or sample_rate_hz <= 0:
            return classification
        samples = np.asarray(signal, dtype=np.float32)
        if samples.ndim > 1:
            samples = np.mean(samples, axis=0, dtype=np.float32)
        if samples.size == 0:
            return classification

        contrast_db = energy_contrast_db(samples, int(sample_rate_hz))
        flatness = framed_spectral_flatness_median(samples, int(sample_rate_hz))
        if contrast_db is None or flatness is None:
            classification.features["texture_gate"] = {"skipped": "short_window"}
            return classification

        gated = (
            contrast_db < self._texture_gate_contrast_db
            and flatness > self._texture_gate_flatness_min
        )
        annotation: dict[str, Any] = {
            "contrast_db": round(float(contrast_db), 3),
            "flatness": round(float(flatness), 4),
            "gated": bool(gated),
        }
        if not gated:
            classification.features["texture_gate"] = annotation
            return classification

        if self._texture_gate_confidence_factor < 1.0:
            # `classify_omni_only` passes the same object as both the winner and
            # the omni result; copy so the raw omni confidence stays intact for
            # assembly reporting.
            classification = classification.model_copy(deep=True)
            annotation["original_confidence"] = float(classification.confidence)
            classification.confidence = float(
                classification.confidence * self._texture_gate_confidence_factor
            )
        classification.features["texture_gate"] = annotation
        # Fired even in annotate-only mode so the counter measures what demotion
        # would do. The detection is still persisted — this flags, never drops.
        if self._on_texture_gate_demotion is not None:
            self._on_texture_gate_demotion()
        return classification

    async def _build_result(
        self,
        *,
        classification: ClassificationResult,
        omni_classification: ClassificationResult,
        beamformed_classification: ClassificationResult | None,
        classification_path: str,
        classification_signal: np.ndarray,
        sample_rate_hz: int,
        beamforming_error: str | None,
        event_time_ns: int,
        backend_failed: bool = False,
    ) -> ClassifiedResult:
        classification = self._apply_texture_gate(
            classification=classification,
            signal=classification_signal,
            sample_rate_hz=sample_rate_hz,
        )
        label_category = _resolve_label_category(classification, self._taxonomy_provider)
        iff_category = self._taxonomy_provider.iff_for_category(label_category)
        # Provenance: the winning ensemble member (set by CompositeClassifier)
        # names the actual model; fall back to the configured pipeline name.
        winner_member = classification.features.get("winner_member")
        label_id = await self._storage.upsert_label(
            name=classification.label,
            category=label_category,
            source=str(winner_member) if winner_member else self._classifier_backend_name,
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
            backend_failed=backend_failed,
        )
