from __future__ import annotations

import time

import numpy as np
import pytest

from minimappr.classifiers.base import AudioClassifier
from minimappr.core.classification import ClassificationOrchestrator
from minimappr.models import ClassificationResult


class _HangingClassifier(AudioClassifier):
    def __init__(self) -> None:
        self.cancel_pending_calls = 0

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        time.sleep(0.2)
        return ClassificationResult(label="bird", confidence=1.0, scores={"bird": 1.0}, features={})

    def cancel_pending(self) -> None:
        self.cancel_pending_calls += 1


class _CancellingThenRecoveringClassifier(AudioClassifier):
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Analysis was cancelled")
        return ClassificationResult(label="warbler", confidence=0.72, scores={"warbler": 0.72}, features={})


class _AlwaysFailingClassifier(AudioClassifier):
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        raise RuntimeError("birdnet worker crashed")


class _StorageStub:
    async def upsert_label(self, *, name: str, category: str, source: str, created_ns: int) -> str:
        del name, category, source, created_ns
        return "label-timeout"


class _TaxonomyStub:
    def category_for_label(self, label: str) -> str:
        del label
        return "unknown"

    def iff_for_category(self, category: str) -> str:
        del category
        return "unknown"


class _EnvironmentStub:
    def get_speed_of_sound(self, position_m: tuple[float, float, float]) -> float:
        del position_m
        return 343.0


@pytest.mark.asyncio
async def test_timeout_invokes_classifier_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("minimappr.core.classification._CLASSIFICATION_TIMEOUT_S", 0.01)

    classifier = _HangingClassifier()
    orchestrator = ClassificationOrchestrator(
        classifier=classifier,
        storage=_StorageStub(),
        taxonomy_provider=_TaxonomyStub(),
        environment_provider=_EnvironmentStub(),
    )

    result = await orchestrator.classify_omni_only(
        reference_signal=np.zeros(1024, dtype=np.float32),
        sample_rate_hz=16_000,
        event_time_ns=123,
    )

    assert result.classification.label == "timeout"
    assert result.classification.features.get("reason") == "classification_timeout"
    assert classifier.cancel_pending_calls == 1


@pytest.mark.asyncio
async def test_cancelled_analysis_retries_once_and_recovers() -> None:
    classifier = _CancellingThenRecoveringClassifier()
    orchestrator = ClassificationOrchestrator(
        classifier=classifier,
        storage=_StorageStub(),
        taxonomy_provider=_TaxonomyStub(),
        environment_provider=_EnvironmentStub(),
    )

    result = await orchestrator.classify_omni_only(
        reference_signal=np.zeros(1024, dtype=np.float32),
        sample_rate_hz=16_000,
        event_time_ns=123,
    )

    assert result.classification.label == "warbler"
    assert classifier.calls == 2


@pytest.mark.asyncio
async def test_classifier_exception_degrades_to_unknown() -> None:
    orchestrator = ClassificationOrchestrator(
        classifier=_AlwaysFailingClassifier(),
        storage=_StorageStub(),
        taxonomy_provider=_TaxonomyStub(),
        environment_provider=_EnvironmentStub(),
    )

    result = await orchestrator.classify_omni_only(
        reference_signal=np.zeros(1024, dtype=np.float32),
        sample_rate_hz=16_000,
        event_time_ns=123,
    )

    assert result.classification.label == "unknown"
    assert result.classification.features.get("reason") == "classification_error"
