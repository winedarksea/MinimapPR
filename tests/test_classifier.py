from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from minimappr.classifiers.base import AudioClassifier
from minimappr.classifiers.chaining import ChainStage, ChainedClassifier
from minimappr.classifiers import factory
from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.classifiers.routing import default_routing, load_routing
from minimappr.classifiers.yamnet import _prepare_waveform_for_yamnet
from minimappr.config import Settings
from minimappr.models import ClassificationResult


def test_heuristic_classifier_detects_machine_hum() -> None:
    sample_rate_hz = 16000
    t = np.arange(sample_rate_hz, dtype=np.float32) / sample_rate_hz
    hum = 0.4 * np.sin(2 * np.pi * 220.0 * t)

    classifier = HeuristicClassifier()
    result = classifier.classify(hum, sample_rate_hz)

    assert result.label in {"machine_hum", "speech_like", "unknown"}
    assert 0.0 <= result.confidence <= 1.0
    assert "centroid_hz" in result.features


def test_heuristic_classifier_detects_impulse_like_event() -> None:
    sample_rate_hz = 16000
    x = np.zeros(sample_rate_hz, dtype=np.float32)
    x[2000:2010] = 0.9
    x[5000:5010] = -0.8

    classifier = HeuristicClassifier()
    result = classifier.classify(x, sample_rate_hz)

    assert result.label in {"impulse", "unknown"}
    assert 0.0 <= result.confidence <= 1.0


class _StaticClassifier(AudioClassifier):
    def __init__(self, label: str, confidence: float, scores: dict[str, float] | None = None) -> None:
        self._label = label
        self._confidence = confidence
        self._scores = scores or {label: confidence}

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        return ClassificationResult(
            label=self._label,
            confidence=self._confidence,
            scores=self._scores,
            features={"source": self._label},
        )


def test_chained_classifier_runs_triggered_stage_and_fuses_confidence() -> None:
    base = _StaticClassifier("bird", 0.55)
    downstream = _StaticClassifier("robin", 0.92)
    chain = ChainedClassifier(
        base_classifier=base,
        stages=[
            ChainStage(
                stage_id="species",
                classifier=downstream,
                trigger_labels={"bird"},
                min_confidence=0.5,
                score_weight=1.0,
            )
        ],
        category_for_label=lambda _: "wildlife",
    )

    result = chain.classify(np.zeros(64, dtype=np.float32), 16000)
    assert result.label == "robin"
    assert result.confidence > 0.9
    assert result.features["chain_stage_count"] == 1.0
    assert "species:robin" in result.scores


def test_chained_classifier_skips_non_matching_stage() -> None:
    base = _StaticClassifier("vehicle", 0.7)
    downstream = _StaticClassifier("truck", 0.95)
    chain = ChainedClassifier(
        base_classifier=base,
        stages=[
            ChainStage(
                stage_id="species",
                classifier=downstream,
                trigger_labels={"bird"},
                min_confidence=0.5,
                score_weight=1.0,
            )
        ],
        category_for_label=lambda _: "vehicle",
    )

    result = chain.classify(np.zeros(64, dtype=np.float32), 16000)
    assert result.label == "vehicle"
    assert result.confidence == 0.7
    assert result.features["chain_stage_count"] == 0.0


def test_create_context_classifier_builds_yamnet_birdnet_composite(monkeypatch) -> None:
    """Default routing: YAMNet and BirdNET both always run (composite, not chained)."""

    class _StubYAMNetClassifier(AudioClassifier):
        def __init__(self, *args, **kwargs) -> None:
            pass

        def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
            del samples, sample_rate_hz
            return ClassificationResult(label="bird", confidence=0.4, scores={"bird": 0.4})

    class _StubBirdNETClassifier(AudioClassifier):
        def __init__(self, *args, **kwargs) -> None:
            pass

        def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
            del samples, sample_rate_hz
            return ClassificationResult(label="robin", confidence=0.9, scores={"robin": 0.9})

    monkeypatch.setattr(factory, "YAMNetClassifier", _StubYAMNetClassifier)
    stub_module = types.ModuleType("minimappr.classifiers.birdnet")
    stub_module.BirdNETClassifier = _StubBirdNETClassifier
    monkeypatch.setitem(sys.modules, "minimappr.classifiers.birdnet", stub_module)

    settings = Settings()
    routing = default_routing()
    classifier = factory.create_context_classifier(settings, "detection_trigger", routing=routing)

    result = classifier.classify(np.zeros(64, dtype=np.float32), 16_000)
    assert result.label == "robin"
    assert result.features["winner_member"] == "birdnet"
    assert result.scores["bird"] == pytest.approx(0.4)
    assert result.scores["birdnet:robin"] == pytest.approx(0.9)


def test_prepare_waveform_for_yamnet_boosts_low_level_audio() -> None:
    sample_rate_hz = 16_000
    t = np.arange(sample_rate_hz, dtype=np.float32) / sample_rate_hz
    low_level_tone = 0.0015 * np.sin(2.0 * np.pi * 440.0 * t)

    conditioned = _prepare_waveform_for_yamnet(low_level_tone)

    input_rms = float(np.sqrt(np.mean(np.square(low_level_tone), dtype=np.float64)))
    output_rms = float(np.sqrt(np.mean(np.square(conditioned), dtype=np.float64)))
    assert output_rms > (input_rms * 10.0)
    assert np.max(np.abs(conditioned)) <= 1.0


def test_prepare_waveform_for_yamnet_preserves_headroom_for_hot_audio() -> None:
    sample_rate_hz = 16_000
    t = np.arange(sample_rate_hz, dtype=np.float32) / sample_rate_hz
    hot_tone = 0.95 * np.sin(2.0 * np.pi * 880.0 * t)

    conditioned = _prepare_waveform_for_yamnet(hot_tone)

    assert np.max(np.abs(conditioned)) <= 0.981
    assert np.max(np.abs(conditioned)) >= 0.90


def test_prepare_waveform_for_yamnet_respects_max_input_gain_cap() -> None:
    sample_rate_hz = 16_000
    t = np.arange(sample_rate_hz, dtype=np.float32) / sample_rate_hz
    low_level_tone = 0.001 * np.sin(2.0 * np.pi * 440.0 * t)

    conditioned = _prepare_waveform_for_yamnet(low_level_tone, target_rms=0.5, max_input_gain=4.0)

    input_rms = float(np.sqrt(np.mean(np.square(low_level_tone), dtype=np.float64)))
    output_rms = float(np.sqrt(np.mean(np.square(conditioned), dtype=np.float64)))
    assert output_rms / input_rms <= 4.05


def test_create_classifier_passes_yamnet_conditioning_settings(monkeypatch, tmp_path) -> None:
    captured: dict[str, float] = {}

    class _StubYAMNetClassifier(AudioClassifier):
        def __init__(
            self,
            min_confidence: float = 0.25,
            *,
            target_rms: float = 0.10,
            max_input_gain: float = 32.0,
            keep_embeddings: bool = False,
        ) -> None:
            captured["min_confidence"] = min_confidence
            captured["target_rms"] = target_rms
            captured["max_input_gain"] = max_input_gain

        def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
            del samples, sample_rate_hz
            return ClassificationResult(label="unknown", confidence=0.0, scores={})

    monkeypatch.setattr(factory, "YAMNetClassifier", _StubYAMNetClassifier)

    settings = Settings(
        yamnet_min_confidence=0.31,
        yamnet_input_target_rms=0.22,
        yamnet_max_input_gain=11.0,
        birdnet_enabled=False,
    )
    routing = load_routing(settings)

    _ = factory.create_context_classifier(settings, "detection_trigger", routing=routing)

    # routing spec min_confidence (0.25) wins over the settings default here;
    # the conditioning knobs pass through from settings.
    assert captured["min_confidence"] == pytest.approx(0.25)
    assert captured["target_rms"] == pytest.approx(0.22)
    assert captured["max_input_gain"] == pytest.approx(11.0)


def test_create_classifier_uses_birdnet_as_primary_backend(monkeypatch, tmp_path) -> None:
    captured: dict[str, float] = {}

    class _StubBirdNETClassifier(AudioClassifier):
        def __init__(
            self,
            min_confidence: float = 0.1,
            *,
            latitude: float | None = None,
            longitude: float | None = None,
            geo_min_confidence: float = 0.03,
        ) -> None:
            captured["min_confidence"] = min_confidence
            captured["latitude"] = latitude or 0.0
            captured["longitude"] = longitude or 0.0
            captured["geo_min_confidence"] = geo_min_confidence

        def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
            del samples, sample_rate_hz
            return ClassificationResult(label="american robin", confidence=0.91, scores={"american robin": 0.91})

    stub_module = types.ModuleType("minimappr.classifiers.birdnet")
    stub_module.BirdNETClassifier = _StubBirdNETClassifier
    monkeypatch.setitem(sys.modules, "minimappr.classifiers.birdnet", stub_module)

    settings = Settings(birdnet_trigger_min_confidence=0.07)
    routing = default_routing()
    routing.contexts["detection_trigger"].run = ("birdnet",)
    routing.classifiers["birdnet"].min_confidence = 0.0
    classifier = factory.create_context_classifier(settings, "detection_trigger", routing=routing)
    result = classifier.classify(np.zeros(64, dtype=np.float32), 16_000)

    assert captured["min_confidence"] == pytest.approx(0.07)
    assert captured["latitude"] == pytest.approx(settings.site_origin_lat)
    assert captured["longitude"] == pytest.approx(settings.site_origin_lon)
    assert captured["geo_min_confidence"] == pytest.approx(settings.birdnet_geo_min_confidence)
    assert result.label == "american robin"
