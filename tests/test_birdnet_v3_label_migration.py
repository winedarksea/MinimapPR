"""W3-birdnet-cutover: v2.4-keyed rules / retention / routing / taxonomy keep matching under V3."""

from __future__ import annotations

import json
import logging
import time

import pytest

from minimappr.classifiers.birdnet_v3_label_migration import (
    expand_labels_for_v3,
    load_birdnet_v24_to_v3_translator,
    migrate_rules_for_birdnet_v3,
    migrate_taxonomy_for_birdnet_v3,
)
from minimappr.classifiers.routing import (
    ChainSpec,
    ClassifierSpec,
    RoutingConfig,
    TriggerSpec,
    apply_settings,
    default_routing,
)
from minimappr.cleanup_service import load_cleanup_policy
from minimappr.config import Settings
from minimappr.core.retention import RetentionPolicy
from minimappr.core.rules import ConfigRuleEngine, RuleCondition, RuleDef, default_rules
from minimappr.core.taxonomy import RuntimeTaxonomyProvider
from minimappr.classifiers.birdnet_v3_label_migration import (
    birdnet_v3_rule_transform,
    migrate_permanent_labels_for_birdnet_v3,
)
from minimappr.interfaces import ActionDescriptor
from minimappr.models import DetectionEvent

V3 = {"birdnet_model_version": "3.0"}
# Renamed in V3 (scientific match, English name differs) and a v2.4 bird with no V3 class.
V24_CATBIRD, V3_CATBIRD = "gray catbird", "grey catbird"
V24_DOG, V3_DOG = "dog", "domestic dog"
UNMAPPED_V24_BIRD = "nihoa finch"


@pytest.fixture(scope="module")
def translator():
    return load_birdnet_v24_to_v3_translator()


def _detection(label: str, confidence: float, *, modality: str = "localized") -> DetectionEvent:
    return DetectionEvent(
        id=f"det-{label}",
        timestamp_ns=time.time_ns(),
        position_m=(0.0, 0.0, 0.0),
        confidence=0.8,
        gdop=1.2,
        reference_sensor="node-a:ch0",
        label=label,
        label_category="wildlife",
        label_confidence=confidence,
        reporting_modality=modality,
    )


def _rule(rule_id: str, labels: set[str], min_confidence: float | None) -> RuleDef:
    return RuleDef(
        rule_id=rule_id,
        enabled=True,
        scope="detection",
        condition=RuleCondition(labels=labels, min_confidence=min_confidence),
        actions=[ActionDescriptor(action_type="alert", destination="cop", priority="high")],
    )


def test_translator_maps_renames_and_reports_unmapped(translator):
    assert translator.v3_labels_for(V24_CATBIRD) == {V3_CATBIRD}
    assert translator.v3_labels_for(V24_DOG) == {V3_DOG}
    assert translator.v3_labels_for("american robin") == {"american robin"}
    expanded, added, unmapped = expand_labels_for_v3({V24_CATBIRD, UNMAPPED_V24_BIRD, "gunshot"}, translator)
    assert expanded == {V24_CATBIRD, V3_CATBIRD, UNMAPPED_V24_BIRD, "gunshot"}
    assert added == [V3_CATBIRD]
    assert unmapped == [UNMAPPED_V24_BIRD]


def test_rule_migration_expands_labels_and_reports(translator):
    rules = [
        _rule("catbird", {V24_CATBIRD}, 0.5),
        _rule("mixed", {V24_DOG, "speech"}, 0.5),
        _rule("lost", {UNMAPPED_V24_BIRD}, None),
    ]
    migrated, report = migrate_rules_for_birdnet_v3(rules, translator)
    by_id = {rule.rule_id: rule for rule in migrated}
    assert by_id["catbird"].condition.labels == {V24_CATBIRD, V3_CATBIRD}
    # BirdNET-only labels: threshold carried to the V3 scale.
    assert by_id["catbird"].condition.min_confidence == pytest.approx(translator.map_confidence(0.5))
    assert by_id["catbird"].condition.min_confidence > 0.5
    # "dog"/"speech" are YAMNet labels too: threshold left alone and reported.
    assert by_id["mixed"].condition.min_confidence == 0.5
    assert report.confidence_not_remapped == ["mixed"]
    assert report.unmapped_labels == {"lost": [UNMAPPED_V24_BIRD]}
    assert rules[0].condition.labels == {V24_CATBIRD}  # inputs untouched


async def test_rules_engine_matches_v3_labels_only_under_v3(tmp_path, caplog):
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"rules": [
        {"id": "catbird", "when": {"labels": [V24_CATBIRD], "min_confidence": 0.5}},
        {"id": "lost", "when": {"labels": [UNMAPPED_V24_BIRD]}},
    ]}))
    v3_detection = _detection(V3_CATBIRD, 0.9)

    legacy = ConfigRuleEngine(rules_path, rule_transform=birdnet_v3_rule_transform(Settings()))
    assert not any(r.matched for r in await legacy.evaluate(detection=v3_detection))

    with caplog.at_level(logging.WARNING):
        migrated = ConfigRuleEngine(rules_path, rule_transform=birdnet_v3_rule_transform(Settings(**V3)))
    results = {r.rule_id: r for r in await migrated.evaluate(detection=v3_detection)}
    assert results["catbird"].matched
    assert "no V3 class" in caplog.text and UNMAPPED_V24_BIRD in caplog.text
    # Below the V3-scale threshold (0.5 on v2.4 maps above 0.5) no longer fires.
    weak = await migrated.evaluate(detection=_detection(V3_CATBIRD, 0.55))
    assert not any(r.matched for r in weak if r.rule_id == "catbird")
    # The file itself is never rewritten.
    assert json.loads(rules_path.read_text())["rules"][0]["when"]["labels"] == [V24_CATBIRD]


def test_default_coyote_rule_threshold_follows_v3_scale(translator):
    migrated, report = migrate_rules_for_birdnet_v3(default_rules(), translator)
    coyote = next(rule for rule in migrated if rule.rule_id == "coyote_alert")
    assert coyote.condition.min_confidence == pytest.approx(translator.map_confidence(0.4))
    assert "gunshot_alert" not in report.remapped_confidences  # YAMNet vocabulary


def test_retention_label_rule_cloned_and_birdnet_bounds_mapped(tmp_path, translator):
    policy_path = tmp_path / "retention_policy.json"
    policy_path.write_text(json.dumps({"version": 1, "rules": [
        {"id": "keep-catbird", "priority": 5, "match": {"label": V24_CATBIRD}, "actions": {"keep_snippets": True}},
        {"id": "birdnet-weak", "priority": 6, "match": {"classifier": "birdnet", "confidence_max": 0.2},
         "actions": {"snippet_max_age_seconds": 60}},
    ]}))
    v3_context = {"label": V3_CATBIRD, "classifier": "birdnet", "confidence": 0.8, "birdnet_model_version": "3.0"}

    legacy = load_cleanup_policy(Settings(), policy_path)
    assert legacy.resolve_for_context(v3_context).snippet_max_age_seconds is not None

    migrated = load_cleanup_policy(Settings(**V3), policy_path)
    assert migrated.resolve_for_context(v3_context).snippet_max_age_seconds is None  # kept forever
    # The original keeps judging v2.4 rows on the v2.4 scale; a V3-scoped clone carries the mapped bound.
    weak = next(rule for rule in migrated.rules if rule.rule_id == "birdnet-weak")
    assert weak.match.confidence_max == 0.2 and weak.match.birdnet_model_version == "2.4"
    weak_v3 = next(rule for rule in migrated.rules if rule.rule_id == "birdnet-weak@birdnet-v3")
    assert weak_v3.match.confidence_max == pytest.approx(translator.map_confidence(0.2))
    # Builtin confident buckets: one per scale, not double-mapped.
    confident = {rule.rule_id: rule for rule in migrated.rules if rule.rule_id.endswith("-confident")}
    assert confident["builtin-birdnet-confident"].match.confidence_min == Settings().birdnet_trigger_min_confidence
    assert confident["builtin-birdnet-v3-confident"].match.confidence_min == Settings(**V3).birdnet_v3_trigger_min_confidence


def test_permanent_labels_gain_v3_names():
    settings = Settings(**V3, retention_permanent_labels=("gunshot", V24_CATBIRD))
    labels = migrate_permanent_labels_for_birdnet_v3(settings.retention_permanent_labels, settings)
    policy = RetentionPolicy(permanent_labels=set(labels), security_long_confidence=0.9)
    assert policy.tier_for_detection(label=V3_CATBIRD, label_category="wildlife", confidence=0.9,
                                     suppressed_by_zone=False) == "permanent"
    assert migrate_permanent_labels_for_birdnet_v3(("gunshot", V24_CATBIRD), Settings()) == ("gunshot", V24_CATBIRD)


def test_routing_birdnet_thresholds_and_labels_follow_v3(translator):
    routing = default_routing()
    routing = RoutingConfig(
        version=1,
        classifiers={**routing.classifiers, "catbird_head": ClassifierSpec("catbird_head", "yamnet", 0.3)},
        contexts=routing.contexts,
        chains=(*routing.chains, ChainSpec("catbird_head", "birdnet", trigger_labels=frozenset({V24_CATBIRD}), min_confidence=0.5)),
        triggers=(*routing.triggers, TriggerSpec("t", "birdnet", "speech_capture", frozenset({V24_DOG}), 0.5)),
    )
    legacy = apply_settings(routing, Settings())
    migrated = apply_settings(routing, Settings(**V3))
    assert legacy.classifiers["birdnet"].min_confidence == 0.4
    assert migrated.classifiers["birdnet"].min_confidence == pytest.approx(translator.map_confidence(0.4))
    assert migrated.classifiers["yamnet"].min_confidence == legacy.classifiers["yamnet"].min_confidence
    chain = next(c for c in migrated.chains if c.after == "birdnet")
    assert chain.trigger_labels == {V24_CATBIRD, V3_CATBIRD}
    assert chain.min_confidence == pytest.approx(translator.map_confidence(0.5))
    yamnet_chain = next(c for c in migrated.chains if c.after == "yamnet")
    assert yamnet_chain == next(c for c in legacy.chains if c.after == "yamnet")
    trigger = next(t for t in migrated.triggers if t.on == "birdnet")
    assert trigger.labels == {V24_DOG, V3_DOG}


def test_migration_flag_off_leaves_everything_v24():
    settings = Settings(**V3, birdnet_v3_label_migration_enabled=False)
    assert birdnet_v3_rule_transform(settings) is None
    assert apply_settings(default_routing(), settings).classifiers["birdnet"].min_confidence == 0.4


def test_taxonomy_categories_follow_v3_names():
    provider = RuntimeTaxonomyProvider(label_to_category={V24_CATBIRD: "pest"})
    report = migrate_taxonomy_for_birdnet_v3(provider, Settings(**V3))
    assert provider.category_for_label(V3_CATBIRD) == "pest"
    assert report.added_labels == {V24_CATBIRD: [V3_CATBIRD]}
    untouched = RuntimeTaxonomyProvider(label_to_category={V24_CATBIRD: "pest"})
    migrate_taxonomy_for_birdnet_v3(untouched, Settings())
    assert (V3_CATBIRD, "pest") not in untouched.label_category_items()
