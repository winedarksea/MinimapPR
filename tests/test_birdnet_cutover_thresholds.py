"""W3-birdnet-cutover: per-version thresholds, two-threshold constraint, calibration, gate tooling."""

from __future__ import annotations

import importlib.util
import logging
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from minimappr.classifiers import factory
from minimappr.classifiers.base import AudioClassifier
from minimappr.classifiers.birdnet_confidence_calibration import (
    ConfidenceQuantileMap,
    build_confidence_quantile_map,
    load_confidence_quantile_map,
    matched_threshold,
    write_confidence_quantile_map,
)
from minimappr.classifiers.birdnet_threshold_set import (
    birdnet_cutover_setting_errors,
    warn_on_two_threshold_without_companions,
)
from minimappr.classifiers.routing import ClassifierSpec
from minimappr.config import Settings
from minimappr.models import ClassificationResult
from minimappr.sim.fixture_audio_scenarios import (
    DEFAULT_SIM_BINARY_PATH,
    build_distance_ladder_scenario,
    build_species_gate_scenario,
    load_fixture_bird_clips,
    render_scenario_bundle,
    scenario_to_toml,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V3 = {"birdnet_model_version": "3.0"}
# Wildlife stationary_hold is the default companion; log-odds is opt-in.
COMPANIONS_ON = {"tracking_species_log_odds_enabled": True}
WILDLIFE_CONTINUOUS_MOTION = {"tracking_mode_by_category": {"wildlife": "continuous_motion", "default": "continuous_motion"}}


def _load_script(name: str):
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


# ── per-version threshold selection ─────────────────────────────────────────

def test_threshold_set_follows_model_version():
    legacy = Settings().birdnet_threshold_set()
    assert (legacy.trigger_min_confidence, legacy.localize_to_track_min_confidence, legacy.species_display_min_belief) == (0.5, 0.15, 0.5)
    v3 = Settings(**V3, birdnet_v3_trigger_min_confidence=0.7, birdnet_v3_localize_to_track_min_confidence=0.2,
                  birdnet_v3_species_display_min_belief=0.66).birdnet_threshold_set()
    assert (v3.trigger_min_confidence, v3.localize_to_track_min_confidence, v3.species_display_min_belief) == (0.7, 0.2, 0.66)
    # The v2.4 keys stay in force under 2.4 even when the V3 keys are set.
    assert Settings(birdnet_v3_trigger_min_confidence=0.9).birdnet_threshold_set().trigger_min_confidence == 0.5


def test_tracker_display_belief_swaps_with_version():
    assert Settings().tracking_config().tracking_species_display_min_belief == 0.5
    v3 = Settings(**V3)
    assert v3.tracking_config().tracking_species_display_min_belief == v3.birdnet_v3_species_display_min_belief


def test_pipeline_graph_display_belief_edits_the_live_key():
    from minimappr.core.pipeline_graph import _field_mode_tracker_params

    def display_key(settings: Settings) -> str:
        params = _field_mode_tracker_params(settings.tracking_config(), settings)
        return next(param.config_key for param in params if param.label == "Display belief")

    log_odds_on = {"tracking_species_log_odds_enabled": True}
    assert display_key(Settings(**log_odds_on)) == "tracking_species_display_min_belief"
    assert display_key(Settings(**V3, **log_odds_on)) == "birdnet_v3_species_display_min_belief"


class _FixedClassifier(AudioClassifier):
    def __init__(self, min_confidence: float) -> None:
        self.min_confidence = min_confidence

    def classify(self, samples, sample_rate_hz) -> ClassificationResult:
        confidence = 0.3
        if confidence < self.min_confidence:
            return ClassificationResult(label="unknown", confidence=0.0, scores={}, features={})
        return ClassificationResult(label="american robin", confidence=confidence, scores={}, features={})


def test_factory_two_threshold_floors_follow_v3_set(monkeypatch):
    built_floors: list[float] = []

    def fake_backend(spec, settings, *, needs_embeddings=False):
        built_floors.append(spec.min_confidence)
        return _FixedClassifier(spec.min_confidence)

    monkeypatch.setattr(factory, "_build_backend", fake_backend)
    spec = ClassifierSpec(member_id="birdnet", backend="birdnet", min_confidence=0.0)
    settings = Settings(**V3, **COMPANIONS_ON, birdnet_two_threshold_enabled=True,
                        birdnet_v3_trigger_min_confidence=0.66, birdnet_v3_localize_to_track_min_confidence=0.2)
    member = factory._build_member(spec, settings, (), None)
    result = member.classify(np.zeros(16_000, dtype=np.float32), 16_000)
    assert built_floors == [0.2]
    assert result.features["display_min_confidence"] == 0.66
    assert result.features["below_display_threshold"] is True


# ── two-threshold constraint ────────────────────────────────────────────────

def test_two_threshold_without_companions_warns_by_default(caplog):
    # Constructs fine: default is warn-only.
    settings = Settings(birdnet_two_threshold_enabled=True, **WILDLIFE_CONTINUOUS_MOTION)
    with caplog.at_level(logging.WARNING):
        warn_on_two_threshold_without_companions(settings)
    assert "tracking_mode_by_category[wildlife]=stationary_hold" in caplog.text
    assert "tracking_species_log_odds_enabled" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        warn_on_two_threshold_without_companions(Settings(birdnet_two_threshold_enabled=True, **COMPANIONS_ON))
    assert caplog.text == ""


def test_two_threshold_companions_enforced_when_required():
    with pytest.raises(ValueError, match=r"tracking_mode_by_category\[wildlife\]=stationary_hold"):
        Settings(
            birdnet_two_threshold_enabled=True,
            birdnet_two_threshold_require_tracker_companions=True,
            **COMPANIONS_ON,
            **WILDLIFE_CONTINUOUS_MOTION,
        )
    with pytest.raises(ValueError, match="tracking_species_log_odds_enabled"):
        Settings(birdnet_two_threshold_enabled=True, birdnet_two_threshold_require_tracker_companions=True)
    Settings(birdnet_two_threshold_enabled=True, birdnet_two_threshold_require_tracker_companions=True, **COMPANIONS_ON)
    Settings(birdnet_two_threshold_require_tracker_companions=True)  # two-threshold off: nothing to enforce


def test_v3_threshold_range_validated():
    with pytest.raises(ValueError, match="birdnet_v3_trigger_min_confidence"):
        Settings(birdnet_v3_trigger_min_confidence=1.5)


def test_patch_merged_view_reports_constraint():
    from minimappr.main import _birdnet_cutover_patch_errors

    settings = Settings(birdnet_two_threshold_require_tracker_companions=True)
    errors = _birdnet_cutover_patch_errors({"birdnet_two_threshold_enabled": True}, settings)
    assert errors and "tracking_species_log_odds_enabled" in errors[0]
    assert _birdnet_cutover_patch_errors({"birdnet_two_threshold_enabled": True, **COMPANIONS_ON}, settings) == []
    assert birdnet_cutover_setting_errors(Settings()) == []


# ── calibration math ────────────────────────────────────────────────────────

def test_matched_threshold_preserves_pass_fraction():
    generator = np.random.default_rng(0)
    v24 = generator.uniform(0.0, 1.0, 2000)
    v3 = np.sqrt(v24)  # monotone rescale: matching must recover it
    for threshold in (0.15, 0.4, 0.5):
        assert matched_threshold(v24, v3, threshold) == pytest.approx(np.sqrt(threshold), abs=0.01)
    assert matched_threshold(v24, v3, 1.1) > v3.max()
    with pytest.raises(ValueError):
        matched_threshold(np.array([]), v3, 0.5)


def test_confidence_map_is_monotone_and_round_trips(tmp_path):
    generator = np.random.default_rng(1)
    v24 = generator.uniform(0.0, 1.0, 500)
    confidence_map = build_confidence_quantile_map(v24, v24 ** 0.5, source_model_version="2.4", target_model_version="3.0")
    assert list(confidence_map.target_knots) == sorted(confidence_map.target_knots)
    assert confidence_map.map_confidence(0.0) == 0.0 and confidence_map.map_confidence(1.0) == 1.0
    path = tmp_path / "map.json"
    write_confidence_quantile_map(confidence_map, path)
    assert load_confidence_quantile_map(path) == confidence_map
    with pytest.raises(ValueError):
        ConfidenceQuantileMap("2.4", "3.0", (0.0, 0.5, 0.4), (0.0, 0.1, 0.2))


def test_committed_confidence_map_agrees_with_v3_defaults():
    committed = load_confidence_quantile_map()
    defaults = Settings(**V3).birdnet_threshold_set()
    # Defaults are the committed curve at the v2.4 values, rounded for config.
    assert committed.map_confidence(0.5) == pytest.approx(defaults.trigger_min_confidence, abs=0.03)
    assert committed.map_confidence(0.15) == pytest.approx(defaults.localize_to_track_min_confidence, abs=0.03)


# ── sim scenarios + gate tooling ────────────────────────────────────────────

def test_fixture_scenarios_serialise_to_valid_toml():
    clips = load_fixture_bird_clips()
    assert len(clips) == 8
    ladder = tomllib.loads(scenario_to_toml(build_distance_ladder_scenario(clips, seed=0)))
    assert len(ladder["sources"]) == 8 * 6 and ladder["devices"][0]["kind"] == "sirith_tetra"
    assert ladder["sources"][0]["calls"][0]["generator"]["kind"] == "wav_file"
    assert "noise_tail_start_s" not in ladder
    gate = tomllib.loads(scenario_to_toml(build_species_gate_scenario(clips, seed=0)))
    assert len(gate["devices"]) == 3 and all(len(source["calls"]) == 2 for source in gate["sources"])


@pytest.mark.skipif(not DEFAULT_SIM_BINARY_PATH.exists(), reason="minimappr-sim release binary not built")
def test_species_gate_bundle_renders_and_loads(tmp_path):
    from minimappr.calibration.bundle import load_bundle

    bundle = load_bundle(render_scenario_bundle(build_species_gate_scenario(load_fixture_bird_clips(), seed=0), tmp_path / "gate.zip"))
    calls = [event for event in bundle.events if not event.get("is_echo_source")]
    assert len(calls) == 16 and {event["label"] for event in calls} >= {"american robin", "blue jay"}


def test_gate_verdict_requires_real_site_evidence():
    gate = _load_script("birdnet_version_replay_gate")

    def row(correct, labelled, recall, swaps=0, duplicates=0):
        return {"correct_events": correct, "labelled_events": labelled, "event_recall": recall,
                "entity_metrics_by_radius_m": {"15.0": {"label_swap_count": swaps, "duplicate_entities": duplicates}}}

    good = {"2.4": [row(10, 14, 0.9)], "3.0": [row(0, 1, 0.0)], "3.0+regional": [row(13, 14, 1.0)]}
    verdict, _ = gate.gate_verdict(good, None)
    assert verdict.startswith("INSUFFICIENT EVIDENCE")
    verdict, _ = gate.gate_verdict(good, {"v24_vs_v3": {"top1_agreement_mapped": 0.9}})
    assert verdict.startswith("PASS")
    worse = {**good, "3.0+regional": [row(9, 14, 1.0, swaps=1)]}
    verdict, reasons = gate.gate_verdict(worse, {"v24_vs_v3": {"top1_agreement_mapped": 0.9}})
    assert verdict.startswith("FAIL") and len(reasons) == 2


def test_calibration_analysis_on_synthetic_scores():
    calibrate = _load_script("calibrate_birdnet_v3_thresholds")
    from minimappr.classifiers.birdnet_v3_labels import BirdNETV3Label

    v24_species = ["Turdus migratorius_American Robin", "Cyanocitta cristata_Blue Jay"]
    v3_labels = [BirdNETV3Label(0, "BN1", "Turdus migratorius", "American Robin", "Aves"),
                 BirdNETV3Label(1, "BN2", "Cyanocitta cristata", "Blue Jay", "Aves")]
    clips = [
        calibrate.CalibrationClip("robin", np.zeros(1), 1, "sim", "Turdus migratorius", 10.0),
        calibrate.CalibrationClip("jay", np.zeros(1), 1, "sim", "Cyanocitta cristata", 60.0),
        calibrate.CalibrationClip("noise", np.zeros(1), 1, "sim", ""),
    ]
    v24 = [np.array([0.9, 0.1]), np.array([0.2, 0.6]), np.array([0.05, 0.02])]
    v3 = [np.array([0.8, 0.1]), np.array([0.1, 0.85]), np.array([0.03, 0.01])]
    report = calibrate.analyse(clips, v24, v24_species, v3, v3_labels)
    assert report["labelled_clips"] == 2 and report["negative_clips"] == 1
    trigger = report["thresholds"]["birdnet_trigger_min_confidence (display floor, retention confident bucket)"]
    assert trigger["pooled_top1"]["v24_pass_fraction"] == pytest.approx(2 / 3)
    assert trigger["pooled_top1"]["v3_pass_fraction_at_v3_threshold"] == pytest.approx(2 / 3)
    assert report["labelled_accuracy_by_sim_distance_m"]["60.0"]["v3_top1_accuracy"] == 1.0
