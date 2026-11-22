"""Two-threshold BirdNET (Field Mode W2-tracker, S4)."""

from __future__ import annotations

import numpy as np

from minimappr.classifiers import factory
from minimappr.classifiers.base import AudioClassifier
from minimappr.classifiers.birdnet_two_threshold import (
    DisplayThresholdAnnotatingClassifier,
    detection_is_below_display_threshold,
)
from minimappr.classifiers.composite import CompositeClassifier, CompositeMember
from minimappr.classifiers.routing import ClassifierSpec
from minimappr.config import Settings
from minimappr.core.assembly_track_resolution import resolve_detection_track
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.models import ClassificationResult

T0 = 1_700_000_000_000_000_000
SECOND = 1_000_000_000


class _FixedClassifier(AudioClassifier):
    def __init__(self, label: str, confidence: float, min_confidence: float = 0.0) -> None:
        self.label, self.confidence, self.min_confidence = label, confidence, min_confidence

    def classify(self, samples, sample_rate_hz) -> ClassificationResult:
        if self.confidence < self.min_confidence:
            return ClassificationResult(label="unknown", confidence=0.0, scores={}, features={})
        return ClassificationResult(label=self.label, confidence=self.confidence,
                                    scores={self.label: self.confidence}, features={"model": "stub"})


def _classify(classifier: AudioClassifier) -> ClassificationResult:
    return classifier.classify(np.zeros(16, dtype=np.float32), 16_000)


def test_wrapper_flags_only_results_under_display_floor():
    weak = _classify(DisplayThresholdAnnotatingClassifier(_FixedClassifier("robin", 0.2), 0.4))
    strong = _classify(DisplayThresholdAnnotatingClassifier(_FixedClassifier("robin", 0.6), 0.4))
    assert weak.label == "robin" and weak.features["below_display_threshold"] is True
    assert strong.features["below_display_threshold"] is False
    assert detection_is_below_display_threshold(weak.features)
    assert not detection_is_below_display_threshold(strong.features)


def test_flag_lookup_follows_composite_winner():
    birdnet = DisplayThresholdAnnotatingClassifier(_FixedClassifier("robin", 0.2), 0.4)
    yamnet_wins = CompositeClassifier([
        CompositeMember(member_id="yamnet", classifier=_FixedClassifier("Bird", 0.9)),
        CompositeMember(member_id="birdnet", classifier=birdnet),
    ])
    result = _classify(yamnet_wins)
    assert result.features["winner_member"] == "yamnet"
    assert not detection_is_below_display_threshold(result.features)

    birdnet_wins = CompositeClassifier([
        CompositeMember(member_id="yamnet", classifier=_FixedClassifier("Bird", 0.1)),
        CompositeMember(member_id="birdnet", classifier=birdnet),
    ])
    result = _classify(birdnet_wins)
    assert result.features["winner_member"] == "birdnet"
    assert detection_is_below_display_threshold(result.features)


def test_factory_builds_with_low_floor_and_wraps_when_flagged(monkeypatch):
    built_floors: list[float] = []

    def fake_backend(spec, settings, *, needs_embeddings=False):
        built_floors.append(spec.min_confidence)
        return _FixedClassifier("robin", 0.2, min_confidence=spec.min_confidence)

    monkeypatch.setattr(factory, "_build_backend", fake_backend)
    spec = ClassifierSpec(member_id="birdnet", backend="birdnet", min_confidence=0.4)

    legacy = factory._build_member(spec, Settings(), (), None)
    assert _classify(legacy).label == "unknown"  # dropped below 0.4, as today

    flagged_settings = Settings(birdnet_two_threshold_enabled=True,
                                birdnet_localize_to_track_min_confidence=0.15)
    flagged = factory._build_member(spec, flagged_settings, (), None)
    result = _classify(flagged)
    assert built_floors == [0.4, 0.15]
    assert result.label == "robin" and result.features["below_display_threshold"] is True


async def _resolve_localized(manager, *, confidence, features, timestamp_ns=T0):
    return await resolve_detection_track(
        tracker=manager, registry=NodeRegistry(), spatial_display_mode="localized",
        suppressed_by_zone=False, capability_tier="full_3d", event_time_ns=timestamp_ns,
        localization_position_m=(6.0, 12.0, 2.0), localization_position_covariance_m2=None,
        classification_label="robin", classification_confidence=confidence,
        classification_scores={"robin": confidence}, classification_features=features,
        label_category="wildlife", iff_category="unknown", label_id=None, selected_sensor_ids=["s"],
        source_node_id="node-a", reference_sensor="s", tdoa_s={}, environment={},
    )


async def test_sub_display_detection_never_starts_a_track_without_belief():
    manager = TrackManager(Settings())
    resolution = await _resolve_localized(manager, confidence=0.2,
                                          features={"below_display_threshold": True})
    assert resolution.below_display_threshold and resolution.track is None
    assert await manager.snapshot(include_hidden=True) == []


async def test_sub_display_detections_build_hidden_track_until_belief_passes():
    manager = TrackManager(Settings(tracking_species_log_odds_enabled=True))
    first = await _resolve_localized(manager, confidence=0.2, features={"below_display_threshold": True})
    assert first.track is None and first.hidden_track_id is not None
    second = await _resolve_localized(manager, confidence=0.2, timestamp_ns=T0 + 5 * SECOND,
                                      features={"below_display_threshold": True})
    assert second.track is None
    third = await _resolve_localized(manager, confidence=0.35, timestamp_ns=T0 + 10 * SECOND,
                                     features={"below_display_threshold": True})
    # 0.20 + 0.20 (belief 0.36, hidden) + 0.35 fused -> 0.73: the track surfaces, never alone before.
    assert third.track is not None and third.track.id == first.hidden_track_id
    assert third.track.species_top_probability > 0.5
