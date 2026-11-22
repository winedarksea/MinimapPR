"""Fast unit tests for the S-gate harness (configurations, row filter, verdict, oracle)."""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.classifiers.birdnet_two_threshold import DisplayThresholdAnnotatingClassifier
from minimappr.config import Settings
from minimappr.sim import server_track_gate as gate
from minimappr.sim.graded_species_oracle import (
    SnrGradedSpeciesOracle,
    build_graded_oracle_for_settings,
    confidence_for_snr_db,
)

ALL_FLAGS = {flag for flags in gate.STREAM_FLAGS.values() for flag in flags}


def test_configuration_flags_set_every_stream_flag_explicitly() -> None:
    assert gate.configuration_flags(gate.BASELINE) == {flag: False for flag in ALL_FLAGS}
    assert gate.configuration_flags(gate.ALL_ON) == {flag: True for flag in ALL_FLAGS}
    alone = gate.configuration_flags("only:S5_attribution")
    # Attribution needs onset clustering (config validation).
    assert {f for f, on in alone.items() if on} == {"onset_attribution_enabled", "localization_onset_clustering_enabled"}
    minus_onset = gate.configuration_flags("all_on-minus:S2_onset_clustering")
    assert not minus_onset["localization_onset_clustering_enabled"] and not minus_onset["onset_attribution_enabled"]
    assert minus_onset["entity_reporting_enabled"]
    with pytest.raises(ValueError):
        gate.configuration_flags("only:S9_nothing")


def test_every_gate_configuration_builds_valid_settings() -> None:
    for name in gate.all_configuration_names():
        Settings(**gate.GATE_COMMON_OVERRIDES, **gate.configuration_flags(name))


def test_surfaced_detections_drop_hidden_sub_display_rows_and_attach_entities() -> None:
    sub_display = {"below_display_threshold": True}
    shown = {"label": "robin", "label_confidence": 0.4}
    rows = [
        {"id": "a", "track_id": None, "feature_summary": sub_display, **shown},
        {"id": "b", "track_id": "trk-hidden", "feature_summary": sub_display, **shown},
        {"id": "c", "track_id": "trk-visible", "feature_summary": sub_display, **shown},
        {"id": "d", "track_id": None, "feature_summary": {}, **shown},
        {"id": "e", "track_id": "trk-visible", "feature_summary": {}, "label": "unknown", "label_confidence": 0.0},
    ]
    surfaced = gate.surfaced_detections(
        rows,
        visible_track_ids={"trk-visible"},
        entity_ids={"d": "ent-1"},
        entity_ids_by_track={"trk-visible": "ent-2"},
    )
    assert [row["id"] for row in surfaced] == ["c", "d"]
    # A row joins its entity as the canonical row or through the entity's bound track.
    assert [row["entity_id"] for row in surfaced] == ["ent-2", "ent-1"]


def _row(scene: str, recall: float, swaps=0, duplicates=0, echoes=0, error=1.0) -> dict:
    return {
        "scene": scene,
        "localized_and_classified_individual_recall": recall,
        "label_swap_count": swaps,
        "duplicate_entities": duplicates,
        "echo_promotions": echoes,
        "entities_spanning_multiple_individuals": 0,
        "median_position_error_m": error,
    }


def test_all_on_gate_failures() -> None:
    baseline = [_row("a", 0.0, error=10.0), _row("b", 0.5, error=2.0)]
    assert gate.all_on_gate_failures([_row("a", 1.0), _row("b", 1.0, error=2.2)], baseline) == []
    failures = gate.all_on_gate_failures([_row("a", 0.0, swaps=1), _row("b", 0.5, error=4.0)], baseline)
    assert any("recall" in f for f in failures)
    assert any("label_swap_count" in f for f in failures)
    assert any("position error worse" in f for f in failures)


def _tone(frequency_hz: float, amplitude: float, sample_rate_hz: int = 24000) -> np.ndarray:
    t = np.arange(sample_rate_hz) / sample_rate_hz
    noise = np.random.default_rng(0).normal(0.0, 1e-3, t.size)
    return amplitude * np.sin(2 * np.pi * frequency_hz * t) + noise


def test_graded_oracle_confidence_follows_snr_and_respects_floor() -> None:
    bands = {"robin": (2000.0, 2600.0), "tanager": (3900.0, 4500.0)}
    oracle = SnrGradedSpeciesOracle(bands, min_confidence=0.5)
    loud = oracle.classify(_tone(2300.0, 0.1), 24000)
    assert loud.label == "robin" and loud.confidence > 0.9
    assert loud.features["model"] == "birdnet_v2m4"
    faint = oracle.classify(_tone(2300.0, 2e-4), 24000)
    assert faint.label == "unknown"
    assert confidence_for_snr_db(0.0) < confidence_for_snr_db(12.0) < confidence_for_snr_db(30.0)


def test_graded_oracle_two_threshold_wrapping_mirrors_the_factory() -> None:
    bands = {"robin": (2000.0, 2600.0)}
    single = build_graded_oracle_for_settings(Settings(), bands)
    assert isinstance(single, SnrGradedSpeciesOracle) and single.min_confidence == pytest.approx(0.5)
    double = build_graded_oracle_for_settings(Settings(birdnet_two_threshold_enabled=True), bands)
    assert isinstance(double, DisplayThresholdAnnotatingClassifier)
    assert double.classifier.min_confidence == pytest.approx(0.15)
