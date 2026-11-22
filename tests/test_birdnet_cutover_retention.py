"""W3-birdnet-cutover review: retention never re-buckets rows across BirdNET scales.

Cleanup re-judges every stored row on each pass, so a confidence-gated
BirdNET rule must use the scale of the model that scored the row. Switching
``birdnet_model_version`` must neither delete v2.4-scored audio early nor make
other classifiers' rows keep-forever.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minimappr.classifiers.birdnet_threshold_set import birdnet_model_version_for_classification_features
from minimappr.classifiers.birdnet_v3_label_migration import (
    load_birdnet_v24_to_v3_translator,
    migrate_permanent_labels_for_birdnet_v3,
)
from minimappr.cleanup_service import CleanupService, load_cleanup_policy
from minimappr.config import Settings
from minimappr.models import DetectionEvent
from minimappr.storage.db import Storage

V3 = {"birdnet_model_version": "3.0"}
V24_ROW, V3_ROW = "2.4", "3.0"
# Between the v2.4 confident floor (0.50) and the V3 one (0.65).
CONFIDENCE_BETWEEN_FLOORS = 0.6
SECONDS_PER_DAY = 86_400
V24_POWER_TOOLS, V3_POWER_TOOL = "power tools", "power tool"  # V3 name is also a YAMNet label


def _birdnet_context(confidence: float, model_version: str, label: str = "house sparrow") -> dict:
    return {"label": label, "classifier": "birdnet", "confidence": confidence, "birdnet_model_version": model_version}


def _write_policy(tmp_path: Path, rules: list[dict]) -> Path:
    policy_path = tmp_path / "retention_policy.json"
    policy_path.write_text(json.dumps({"version": 1, "rules": rules}))
    return policy_path


def test_model_version_read_from_winner_member_features():
    assert birdnet_model_version_for_classification_features(None) == V24_ROW
    assert birdnet_model_version_for_classification_features({"winner_member": "birdnet"}) == V24_ROW
    assert birdnet_model_version_for_classification_features({"winner_member": "birdnet", "model": "birdnet_v3"}) == V3_ROW
    # Non-primary composite members carry "<member>:" prefixed features.
    composite = {"winner_member": "birdnet", "model": "yamnet", "birdnet:model": "birdnet_v3"}
    assert birdnet_model_version_for_classification_features(composite) == V3_ROW
    yamnet_won = {"winner_member": "yamnet", "model": "birdnet_v3", "yamnet:model": "yamnet"}
    assert birdnet_model_version_for_classification_features(yamnet_won) == V24_ROW


@pytest.mark.parametrize("settings_overrides", [{}, V3])
def test_confident_bucket_judges_each_row_on_its_own_scale(tmp_path, settings_overrides):
    settings = Settings(**settings_overrides)
    policy = load_cleanup_policy(settings, tmp_path / "missing.json")
    long_bucket = settings.retention_birdnet_audio_seconds
    short_bucket = settings.retention_short_seconds
    # A stored v2.4 row keeps its long bucket whatever version is configured.
    assert policy.resolve_for_context(_birdnet_context(CONFIDENCE_BETWEEN_FLOORS, V24_ROW)).snippet_max_age_seconds == long_bucket
    # A V3 row under its own (higher) floor is not "confident".
    assert policy.resolve_for_context(_birdnet_context(CONFIDENCE_BETWEEN_FLOORS, V3_ROW)).snippet_max_age_seconds == short_bucket
    assert policy.resolve_for_context(_birdnet_context(0.7, V3_ROW)).snippet_max_age_seconds == long_bucket
    # Contexts without provenance are v2.4 (all pre-V3 rows).
    legacy_context = {"label": "house sparrow", "classifier": "birdnet", "confidence": CONFIDENCE_BETWEEN_FLOORS}
    assert policy.resolve_for_context(legacy_context).snippet_max_age_seconds == long_bucket


def test_v24_decisions_identical_under_both_versions_for_user_rules(tmp_path):
    policy_path = _write_policy(tmp_path, [
        {"id": "keep-strong-birdnet", "priority": 5, "match": {"classifier": "birdnet", "confidence_min": 0.8},
         "actions": {"keep_snippets": True}},
        {"id": "drop-weak-birdnet", "priority": 6, "match": {"classifier": "birdnet", "confidence_max": 0.3},
         "actions": {"snippet_max_age_seconds": 60}},
        {"id": "keep-robin", "priority": 7, "match": {"label": "american robin", "confidence_min": 0.5},
         "actions": {"keep_snippets": True}},
    ])
    legacy = load_cleanup_policy(Settings(), policy_path)
    migrated = load_cleanup_policy(Settings(**V3), policy_path)
    for confidence in (0.1, 0.29, 0.31, 0.35, 0.5, 0.55, 0.79, 0.81, 0.83, 0.99):
        for label in ("house sparrow", "american robin"):
            context = _birdnet_context(confidence, V24_ROW, label)
            assert migrated.resolve_for_context(context) == legacy.resolve_for_context(context), (confidence, label)
    translator = load_birdnet_v24_to_v3_translator()
    clone = next(rule for rule in migrated.rules if rule.rule_id == "drop-weak-birdnet@birdnet-v3")
    assert clone.match.confidence_max == pytest.approx(translator.map_confidence(0.3))
    assert clone.match.birdnet_model_version == V3_ROW
    robin_clone = next(rule for rule in migrated.rules if rule.rule_id == "keep-robin@birdnet-v3")
    assert robin_clone.match.confidence_min == pytest.approx(translator.map_confidence(0.5))
    # V3 rows are judged on the mapped bound.
    weak_v3 = _birdnet_context(translator.map_confidence(0.3) - 0.01, V3_ROW)
    assert migrated.resolve_for_context(weak_v3).snippet_max_age_seconds == 60


def test_v3_clone_keeps_its_originals_position(tmp_path):
    # Same priority; "keep-all" sorts between "keep" and "keep@birdnet-v3:...".
    # Last-write-wins for a V3 catbird row must equal the v2.4 catbird decision.
    policy_path = _write_policy(tmp_path, [
        {"id": "keep", "priority": 5, "match": {"label": "gray catbird"}, "actions": {"snippet_max_age_seconds": 10}},
        {"id": "keep-all", "priority": 5, "match": {"classifier": "birdnet"}, "actions": {"snippet_max_age_seconds": 999}},
    ])
    migrated = load_cleanup_policy(Settings(**V3), policy_path)
    v24_decision = migrated.resolve_for_context(_birdnet_context(0.9, V24_ROW, "gray catbird"))
    v3_decision = migrated.resolve_for_context(_birdnet_context(0.9, V3_ROW, "grey catbird"))
    assert v24_decision.snippet_max_age_seconds == v3_decision.snippet_max_age_seconds == 999


def test_v3_name_shared_with_yamnet_does_not_rebucket_yamnet_rows(tmp_path):
    policy_path = _write_policy(tmp_path, [
        {"id": "keep-power-tools", "priority": 5, "match": {"label": V24_POWER_TOOLS}, "actions": {"keep_snippets": True}},
    ])
    migrated = load_cleanup_policy(Settings(**V3), policy_path)
    yamnet_row = {"label": V3_POWER_TOOL, "classifier": "yamnet", "confidence": 0.9, "birdnet_model_version": V24_ROW}
    assert migrated.resolve_for_context(yamnet_row).snippet_max_age_seconds is not None
    v3_row = _birdnet_context(0.9, V3_ROW, V3_POWER_TOOL)
    assert migrated.resolve_for_context(v3_row).snippet_max_age_seconds is None


def test_explicit_v3_label_rule_is_not_overridden_by_a_clone(tmp_path):
    policy_path = _write_policy(tmp_path, [
        {"id": "a-keep-gray", "priority": 5, "match": {"label": "gray catbird"}, "actions": {"keep_snippets": True}},
        {"id": "z-grey-short", "priority": 1, "match": {"label": "grey catbird"}, "actions": {"snippet_max_age_seconds": 60}},
    ])
    migrated = load_cleanup_policy(Settings(**V3), policy_path)
    assert not any(rule.rule_id.startswith("a-keep-gray@") for rule in migrated.rules)
    assert migrated.resolve_for_context(_birdnet_context(0.9, V3_ROW, "grey catbird")).snippet_max_age_seconds == 60


def test_permanent_labels_skip_v3_names_other_classifiers_emit():
    settings = Settings(**V3)
    labels = migrate_permanent_labels_for_birdnet_v3((V24_POWER_TOOLS, "gray catbird"), settings)
    assert V3_POWER_TOOL not in labels
    assert "grey catbird" in labels


async def _insert_birdnet_row(storage: Storage, tmp_path: Path, *, detection_id: str, model_feature: str | None,
                              timestamp_ns: int) -> Path:
    snippet_path = tmp_path / f"{detection_id}.wav"
    snippet_path.write_bytes(b"wav")
    features = {"winner_member": "birdnet"}
    if model_feature is not None:
        features["model"] = model_feature
    await storage.insert_detection(
        detection=DetectionEvent(
            id=detection_id, timestamp_ns=timestamp_ns, position_m=(0.0, 0.0, 0.0), confidence=0.5, gdop=1.5,
            label="house sparrow", label_category="wildlife", label_confidence=CONFIDENCE_BETWEEN_FLOORS,
            reference_sensor="node-a:ch0", feature_summary=features,
        ),
        snippet_path=str(snippet_path), snippet_expires_ns=None, retention_tier="short",
    )
    return snippet_path


@pytest.mark.asyncio
@pytest.mark.parametrize("settings_overrides", [{}, V3])
async def test_cleanup_pass_keeps_v24_audio_after_switching_versions(tmp_path, settings_overrides):
    settings = Settings(
        **settings_overrides,
        db_path=tmp_path / "cutover.db", snippet_dir=tmp_path, large_artifact_dir=tmp_path / "artifacts",
        retention_policy_path=tmp_path / "missing-policy.json",
    )
    storage = Storage(settings.db_path)
    await storage.initialize()
    try:
        now_ns = 5_000_000_000_000_000_000
        # Older than the short bucket (1 day), inside the long one (30 days).
        old_ns = now_ns - 2 * SECONDS_PER_DAY * 1_000_000_000
        legacy_row = await _insert_birdnet_row(storage, tmp_path, detection_id="v24", model_feature=None, timestamp_ns=old_ns)
        v24_row = await _insert_birdnet_row(storage, tmp_path, detection_id="v24m", model_feature="birdnet_v2m4", timestamp_ns=old_ns)
        v3_row = await _insert_birdnet_row(storage, tmp_path, detection_id="v3", model_feature="birdnet_v3", timestamp_ns=old_ns)
        await CleanupService(settings=settings, storage=storage).run_partial_cleanup(now_ns=now_ns)
        assert legacy_row.exists() and v24_row.exists(), "v2.4 0.6 is confident on its own scale"
        assert not v3_row.exists(), "V3 0.6 is under the V3 confident floor"
    finally:
        await storage.close()
