"""Routing config parsing/kill-switches + composite/embedding-chain behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from minimappr.classifiers.base import AudioClassifier, EmbeddingClassifier
from minimappr.classifiers.chaining import ChainStage, ChainedClassifier
from minimappr.classifiers.composite import CompositeClassifier, CompositeMember
from minimappr.classifiers.routing import (
    CONTEXT_DETECTION_TRIGGER,
    CONTEXT_OMNI_CONTINUOUS,
    apply_settings,
    default_routing,
    load_routing_file,
)
from minimappr.config import Settings
from minimappr.models import ClassificationResult


class StubClassifier(AudioClassifier):
    def __init__(self, label: str, confidence: float, features: dict | None = None) -> None:
        self.label = label
        self.confidence = confidence
        self.features = dict(features or {})
        self.calls = 0

    def classify(self, samples, sample_rate_hz):
        self.calls += 1
        return ClassificationResult(
            label=self.label,
            confidence=self.confidence,
            scores={self.label: self.confidence},
            features=dict(self.features),
        )


class StubEmbeddingHead(EmbeddingClassifier):
    def __init__(self) -> None:
        self.received: list[np.ndarray] = []

    def classify_embedding(self, frames: np.ndarray) -> ClassificationResult:
        self.received.append(np.asarray(frames))
        return ClassificationResult(
            label="drone",
            confidence=0.9,
            scores={"drone": 0.9, "no_drone": 0.1},
            features={"model": "stub_head"},
        )


# ---------------------------------------------------------------- routing


def test_default_routing_matches_shipped_json() -> None:
    shipped = load_routing_file(Path("data/classifier_routing.json"))
    default = default_routing()
    assert set(shipped.classifiers) == set(default.classifiers)
    assert set(shipped.contexts) == set(default.contexts)
    assert [c.chain_id for c in shipped.chains] == [c.chain_id for c in default.chains]
    assert [t.trigger_id for t in shipped.triggers] == [t.trigger_id for t in default.triggers]


def test_missing_file_yields_defaults(tmp_path: Path) -> None:
    routing = load_routing_file(tmp_path / "nope.json")
    assert "yamnet" in routing.classifiers
    assert routing.context(CONTEXT_DETECTION_TRIGGER).run == ("yamnet", "birdnet")


def test_malformed_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "routing.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_routing_file(path)


def test_embedding_only_backend_rejected_in_run(tmp_path: Path) -> None:
    path = tmp_path / "routing.json"
    path.write_text(
        json.dumps(
            {
                "classifiers": {"drone_head": {"backend": "drone_head"}},
                "contexts": {"detection_trigger": {"run": ["drone_head"]}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="embedding-only"):
        load_routing_file(path)


def test_unknown_backend_raises(tmp_path: Path) -> None:
    path = tmp_path / "routing.json"
    path.write_text(
        json.dumps({"classifiers": {"x": {"backend": "quantum"}}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="quantum"):
        load_routing_file(path)


def test_kill_switches_strip_members_chains_triggers() -> None:
    settings = Settings(birdnet_enabled=False, drone_head_enabled=False, stt_enabled=False)
    routing = apply_settings(default_routing(), settings)
    assert routing.context(CONTEXT_DETECTION_TRIGGER).run == ("yamnet",)
    assert routing.context(CONTEXT_OMNI_CONTINUOUS).run == ()
    assert routing.chains == ()
    assert routing.triggers == ()


def test_settings_override_thresholds_and_omni_params() -> None:
    settings = Settings(
        drone_head_min_confidence=0.7,
        stt_trigger_min_confidence=0.9,
        omni_scan_interval_seconds=60.0,
        omni_scan_window_seconds=10.0,
        omni_scan_min_rms=0.5,
    )
    routing = apply_settings(default_routing(), settings)
    assert routing.classifiers["drone_head"].min_confidence == 0.7
    assert routing.triggers[0].min_confidence == 0.9
    omni = routing.context(CONTEXT_OMNI_CONTINUOUS)
    assert omni.interval_seconds == 60.0
    assert omni.window_seconds == 10.0
    assert omni.min_rms == 0.5


# ---------------------------------------------------------------- composite


def test_composite_merges_scores_and_picks_winner() -> None:
    primary = StubClassifier("bird", 0.5)
    secondary = StubClassifier("coyote", 0.8)
    composite = CompositeClassifier(
        [
            CompositeMember("yamnet", primary),
            CompositeMember("birdnet", secondary),
        ]
    )
    result = composite.classify(np.zeros(1600, dtype=np.float32), 16000)
    assert result.label == "coyote"
    assert result.confidence == pytest.approx(0.8)
    assert result.scores["bird"] == pytest.approx(0.5)  # primary unprefixed
    assert result.scores["birdnet:coyote"] == pytest.approx(0.8)
    assert result.features["winner_member"] == "birdnet"
    ensemble = result.features["ensemble"]
    assert [entry["member_id"] for entry in ensemble] == ["yamnet", "birdnet"]


def test_composite_all_unknown_falls_back_to_primary() -> None:
    composite = CompositeClassifier(
        [
            CompositeMember("a", StubClassifier("unknown", 0.3)),
            CompositeMember("b", StubClassifier("unknown", 0.6)),
        ]
    )
    result = composite.classify(np.zeros(160, dtype=np.float32), 16000)
    assert result.label == "unknown"
    assert result.features["winner_member"] == "a"


def test_composite_member_failure_skipped() -> None:
    class Boom(AudioClassifier):
        def classify(self, samples, sample_rate_hz):
            raise RuntimeError("boom")

    composite = CompositeClassifier(
        [
            CompositeMember("boom", Boom()),
            CompositeMember("ok", StubClassifier("dog", 0.7)),
        ]
    )
    result = composite.classify(np.zeros(160, dtype=np.float32), 16000)
    assert result.label == "dog"
    assert result.features["winner_member"] == "ok"


def test_composite_features_json_serializable() -> None:
    base = StubClassifier(
        "speech",
        0.6,
        features={
            "embedding": np.ones(1024, dtype=np.float32),
            "embedding_frames": np.ones((2, 1024), dtype=np.float32),
            "embedding_model": "yamnet/1",
        },
    )
    composite = CompositeClassifier([CompositeMember("yamnet", base)])
    # single-member composite is normally unwrapped by the factory, but the
    # class itself must still strip ndarrays.
    result = composite.classify(np.zeros(160, dtype=np.float32), 16000)
    json.dumps(result.features)
    assert "embedding" not in result.features
    assert result.features["embedding_model"] == "yamnet/1"


# ---------------------------------------------------------------- embedding chain


def test_embedding_chain_stage_consumes_frames() -> None:
    frames = np.random.default_rng(0).normal(size=(3, 1024)).astype(np.float32)
    base = StubClassifier("engine", 0.6, features={"embedding_frames": frames})
    head = StubEmbeddingHead()
    chained = ChainedClassifier(
        base_classifier=base,
        stages=[ChainStage(stage_id="drone_head", classifier=head, input_kind="embedding")],
    )
    result = chained.classify(np.zeros(16000, dtype=np.float32), 16000)
    assert head.received and head.received[0].shape == (3, 1024)
    assert result.scores["drone_head:drone"] == pytest.approx(0.9)
    assert result.label == "drone"  # 0.9 beats 0.6
    json.dumps(result.features)
    assert "embedding_frames" not in result.features


def test_embedding_chain_falls_back_to_mean_embedding() -> None:
    base = StubClassifier("engine", 0.6, features={"embedding": np.ones(1024, dtype=np.float32)})
    head = StubEmbeddingHead()
    chained = ChainedClassifier(
        base_classifier=base,
        stages=[ChainStage(stage_id="drone_head", classifier=head, input_kind="embedding")],
    )
    chained.classify(np.zeros(16000, dtype=np.float32), 16000)
    assert head.received[0].shape == (1, 1024)


def test_embedding_chain_skipped_without_embeddings() -> None:
    base = StubClassifier("engine", 0.6)
    head = StubEmbeddingHead()
    chained = ChainedClassifier(
        base_classifier=base,
        stages=[ChainStage(stage_id="drone_head", classifier=head, input_kind="embedding")],
    )
    result = chained.classify(np.zeros(16000, dtype=np.float32), 16000)
    assert not head.received
    assert result.label == "engine"
