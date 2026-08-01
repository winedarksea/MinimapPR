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
    CONTEXT_LOCALIZED_RENDER,
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
    def __init__(self, label: str = "drone", confidence: float = 0.9) -> None:
        self.received: list[np.ndarray] = []
        self._label = label
        self._confidence = confidence

    def classify_embedding(self, frames: np.ndarray) -> ClassificationResult:
        self.received.append(np.asarray(frames))
        return ClassificationResult(
            label=self._label,
            confidence=self._confidence,
            scores={"drone": self._confidence, "no_drone": 1.0 - self._confidence},
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
    assert routing.context(CONTEXT_DETECTION_TRIGGER).run == ()
    assert routing.context(CONTEXT_LOCALIZED_RENDER).run == ("yamnet", "birdnet")


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


def test_min_frame_fraction_round_trips(tmp_path: Path) -> None:
    from minimappr.classifiers.routing import routing_to_dict

    path = tmp_path / "routing.json"
    path.write_text(
        json.dumps(
            {
                "classifiers": {
                    "drone_head": {"backend": "drone_head", "min_frame_fraction": 0.3}
                },
            }
        ),
        encoding="utf-8",
    )
    routing = load_routing_file(path)
    assert routing.classifiers["drone_head"].min_frame_fraction == 0.3
    assert routing_to_dict(routing)["classifiers"]["drone_head"]["min_frame_fraction"] == 0.3


def test_min_frame_fraction_out_of_range_raises(tmp_path: Path) -> None:
    path = tmp_path / "routing.json"
    path.write_text(
        json.dumps(
            {
                "classifiers": {
                    "drone_head": {"backend": "drone_head", "min_frame_fraction": 1.5}
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="min_frame_fraction"):
        load_routing_file(path)


def test_kill_switches_strip_members_chains_triggers() -> None:
    settings = Settings(
        birdnet_enabled=False, drone_head_enabled=False, stt_enabled=False, t3t4_enabled=False
    )
    routing = apply_settings(default_routing(), settings)
    assert routing.context(CONTEXT_DETECTION_TRIGGER).run == ()
    assert routing.context(CONTEXT_OMNI_CONTINUOUS).run == ()
    assert routing.chains == ()
    assert routing.triggers == ()


def test_t3t4_kill_switch_strips_omni_member_only() -> None:
    settings = Settings(t3t4_enabled=False)
    routing = apply_settings(default_routing(), settings)
    assert routing.context(CONTEXT_OMNI_CONTINUOUS).run == ("birdnet",)


def test_t3t4_threshold_override() -> None:
    settings = Settings(t3t4_min_confidence=0.8)
    routing = apply_settings(default_routing(), settings)
    assert routing.classifiers["t3t4_alarm"].min_confidence == 0.8


def test_settings_override_thresholds_and_omni_params() -> None:
    settings = Settings(
        drone_head_min_confidence=0.7,
        drone_head_min_frame_fraction=0.35,
        stt_trigger_min_confidence=0.9,
        omni_scan_interval_seconds=60.0,
        omni_scan_window_seconds=10.0,
        omni_scan_min_rms=0.5,
    )
    routing = apply_settings(default_routing(), settings)
    assert routing.classifiers["drone_head"].min_confidence == 0.7
    assert routing.classifiers["drone_head"].min_frame_fraction == 0.35
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


def test_composite_promotes_priority_label_over_higher_scorer() -> None:
    # A louder, higher-confidence bird call must not mask a safety-critical
    # alarm cadence in the same window: the priority label wins even at lower
    # confidence, so the alerting rules still see it.
    composite = CompositeClassifier(
        [
            CompositeMember("birdnet", StubClassifier("coyote", 0.9)),
            CompositeMember("t3t4_alarm", StubClassifier("alarm_t3", 0.6)),
        ],
        priority_labels=frozenset({"alarm_t3", "alarm_t4"}),
    )
    result = composite.classify(np.zeros(1600, dtype=np.float32), 16000)
    assert result.label == "alarm_t3"
    assert result.confidence == pytest.approx(0.6)
    assert result.features["winner_member"] == "t3t4_alarm"
    # The masked sibling is still recorded (birdnet is the primary/unprefixed member).
    assert result.scores["coyote"] == pytest.approx(0.9)


def test_composite_without_priority_labels_is_pure_winner_take_all() -> None:
    composite = CompositeClassifier(
        [
            CompositeMember("birdnet", StubClassifier("coyote", 0.9)),
            CompositeMember("t3t4_alarm", StubClassifier("alarm_t3", 0.6)),
        ]
    )
    result = composite.classify(np.zeros(1600, dtype=np.float32), 16000)
    assert result.label == "coyote"


def test_context_classifier_wires_alarm_priority_labels() -> None:
    # End-to-end: the omni context's composite must carry the alarm labels so
    # promotion is active in the running system, not just when hand-constructed.
    from minimappr.classifiers.factory import create_context_classifier

    classifier = create_context_classifier(Settings(), CONTEXT_OMNI_CONTINUOUS)
    if isinstance(classifier, CompositeClassifier):
        assert {"alarm_t3", "alarm_t4"} <= classifier._priority_labels
    else:  # BirdNET absent -> single t3t4 member, promotion is moot
        assert getattr(classifier, "PRIORITY_LABELS", frozenset()) >= {"alarm_t3", "alarm_t4"}


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
    assert result.features["winner_member"] == "drone_head"
    json.dumps(result.features)
    assert "embedding_frames" not in result.features


def test_unknown_chain_stage_does_not_clobber_base_label() -> None:
    # A sub-threshold head reports unknown; the base's real label must stand
    # even when the stage's raw confidence is numerically higher.
    frames = np.ones((2, 1024), dtype=np.float32)
    base = StubClassifier("engine", 0.6, features={"embedding_frames": frames})
    head = StubEmbeddingHead(label="unknown", confidence=0.7)
    chained = ChainedClassifier(
        base_classifier=base,
        stages=[ChainStage(stage_id="drone_head", classifier=head, input_kind="embedding")],
    )
    result = chained.classify(np.zeros(16000, dtype=np.float32), 16000)
    assert result.label == "engine"
    assert result.confidence == pytest.approx(0.6)
    assert "winner_member" not in result.features
    # The stage's evidence is still merged for observability.
    assert result.scores["drone_head:drone"] == pytest.approx(0.7)


def test_composite_preserves_chain_stage_attribution() -> None:
    # yamnet member whose drone_head chain stage wins -> composite must report
    # winner_member="drone_head" so classifier_source/audio retention key off it.
    frames = np.ones((2, 1024), dtype=np.float32)
    base = StubClassifier("engine", 0.6, features={"embedding_frames": frames})
    chained = ChainedClassifier(
        base_classifier=base,
        stages=[
            ChainStage(stage_id="drone_head", classifier=StubEmbeddingHead(), input_kind="embedding")
        ],
    )
    composite = CompositeClassifier(
        [
            CompositeMember("yamnet", chained),
            CompositeMember("birdnet", StubClassifier("robin", 0.4)),
        ]
    )
    result = composite.classify(np.zeros(16000, dtype=np.float32), 16000)
    assert result.label == "drone"
    assert result.features["winner_member"] == "drone_head"

    # When another member outscores the chained one, its own id is reported.
    composite_birdnet_wins = CompositeClassifier(
        [
            CompositeMember("yamnet", chained),
            CompositeMember("birdnet", StubClassifier("robin", 0.95)),
        ]
    )
    result = composite_birdnet_wins.classify(np.zeros(16000, dtype=np.float32), 16000)
    assert result.label == "robin"
    assert result.features["winner_member"] == "birdnet"


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


class StubThreeClassHead(EmbeddingClassifier):
    """Mimics an N-class drone head emitting per-class scores for all labels."""

    def classify_embedding(self, frames: np.ndarray) -> ClassificationResult:
        return ClassificationResult(
            label="coyote",
            confidence=0.8,
            scores={"ambient": 0.1, "drone": 0.3, "coyote": 0.8},
            features={"model": "drone_head"},
        )


def test_three_label_head_namespaces_all_class_scores() -> None:
    base = StubClassifier("engine", 0.6, features={"embedding": np.ones(1024, dtype=np.float32)})
    head = StubThreeClassHead()
    chained = ChainedClassifier(
        base_classifier=base,
        stages=[ChainStage(stage_id="drone_head", classifier=head, input_kind="embedding")],
    )
    result = chained.classify(np.zeros(16000, dtype=np.float32), 16000)
    assert result.scores["drone_head:coyote"] == 0.8
    assert result.scores["drone_head:drone"] == 0.3
    assert result.scores["drone_head:ambient"] == 0.1
