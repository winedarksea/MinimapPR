"""Load-time migration of v2.4-keyed config to BirdNET+ V3.0 (W3-birdnet-cutover).

About 1,150 v2.4 species carry a different English name in V3 (Gray -> Grey,
redpoll lumps, "Dog" -> "Domestic Dog", "Human vocal" -> "Human"), and V3
confidences sit on another scale. With ``birdnet_model_version == "3.0"``
this module rewrites, *in memory only*, everything keyed on a BirdNET label
or a BirdNET confidence so existing config keeps matching:

* rules (``rules.json``): ``labels`` gain the V3 name next to the v2.4 one;
  a rule whose labels are all BirdNET-only gets its ``min_confidence``
  carried through the v2.4 -> V3 confidence map;
* retention policy rules: a ``label`` rule is cloned for the V3 name, and a
  ``classifier: birdnet`` rule's confidence bounds are carried across;
* ``retention_permanent_labels``: V3 names added;
* classifier routing: the BirdNET member's ``min_confidence``, and chain /
  trigger labels and thresholds hanging off a BirdNET member.

Files on disk are never rewritten (user-authored rules stay v2.4, and
switching back to "2.4" needs no undo). Expansion is additive: a label keeps
matching other classifiers that emit the v2.4 spelling (YAMNet "dog").
Labels with no V3 class are reported, not dropped.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

from minimappr.classifiers.birdnet_confidence_calibration import (
    DEFAULT_V24_TO_V3_CONFIDENCE_MAP_PATH,
    ConfidenceQuantileMap,
    load_confidence_quantile_map,
)
from minimappr.classifiers.birdnet_v3_labels import (
    DEFAULT_V24_TO_V3_LABEL_MAP_PATH,
    load_v24_to_v3_label_map,
)

logger = logging.getLogger(__name__)

BIRDNET_V3_MODEL_VERSION = "3.0"
BIRDNET_BACKEND_NAME = "birdnet"
YAMNET_CLASS_MAP_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "yamnet" / "assets" / "yamnet_class_map.csv"
)
V3_RETENTION_RULE_ID_SUFFIX = "@birdnet-v3"


def _normalize(label: str) -> str:
    return str(label).strip().lower()


@dataclass(frozen=True, slots=True)
class BirdNETV24ToV3LabelTranslator:
    """v2.4 pipeline label (lower-cased common name) -> V3 pipeline label(s)."""

    v3_labels_by_v24_label: dict[str, frozenset[str]]
    unmapped_v24_labels: frozenset[str]
    non_birdnet_vocabulary: frozenset[str]
    confidence_map: ConfidenceQuantileMap

    def is_birdnet_v24_label(self, label: str) -> bool:
        key = _normalize(label)
        return key in self.v3_labels_by_v24_label or key in self.unmapped_v24_labels

    def is_birdnet_only_label(self, label: str) -> bool:
        """A v2.4 BirdNET label no other bundled classifier emits."""
        key = _normalize(label)
        return self.is_birdnet_v24_label(key) and key not in self.non_birdnet_vocabulary

    def v3_labels_for(self, label: str) -> frozenset[str]:
        return self.v3_labels_by_v24_label.get(_normalize(label), frozenset())

    def map_confidence(self, v24_confidence: float) -> float:
        return round(self.confidence_map.map_confidence(v24_confidence), 4)


def _load_non_birdnet_vocabulary(yamnet_class_map_path: Path) -> frozenset[str]:
    if not yamnet_class_map_path.is_file():
        return frozenset()
    with yamnet_class_map_path.open(newline="", encoding="utf-8") as handle:
        return frozenset(_normalize(row["display_name"]) for row in csv.DictReader(handle) if row.get("display_name"))


@lru_cache(maxsize=4)
def load_birdnet_v24_to_v3_translator(
    label_map_path: Path = DEFAULT_V24_TO_V3_LABEL_MAP_PATH,
    confidence_map_path: Path = DEFAULT_V24_TO_V3_CONFIDENCE_MAP_PATH,
    yamnet_class_map_path: Path = YAMNET_CLASS_MAP_PATH,
) -> BirdNETV24ToV3LabelTranslator:
    v3_by_v24: dict[str, set[str]] = {}
    unmapped: set[str] = set()
    for mapping in load_v24_to_v3_label_map(label_map_path).values():
        key = _normalize(mapping.v24_common_name)
        if mapping.is_mapped:
            v3_by_v24.setdefault(key, set()).add(_normalize(mapping.v3_common_name_en))
        else:
            unmapped.add(key)
    # A common name shared by a mapped and an unmapped v2.4 species is mapped.
    unmapped -= set(v3_by_v24)
    return BirdNETV24ToV3LabelTranslator(
        v3_labels_by_v24_label={key: frozenset(values) for key, values in v3_by_v24.items()},
        unmapped_v24_labels=frozenset(unmapped),
        non_birdnet_vocabulary=_load_non_birdnet_vocabulary(yamnet_class_map_path),
        confidence_map=load_confidence_quantile_map(confidence_map_path),
    )


@dataclass(slots=True)
class LabelMigrationReport:
    """What a migration pass changed and what it could not map."""

    added_labels: dict[str, list[str]] = field(default_factory=dict)
    remapped_confidences: dict[str, tuple[float, float]] = field(default_factory=dict)
    unmapped_labels: dict[str, list[str]] = field(default_factory=dict)
    confidence_not_remapped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "added_labels": self.added_labels,
            "remapped_confidences": {key: list(value) for key, value in self.remapped_confidences.items()},
            "unmapped_labels": self.unmapped_labels,
            "confidence_not_remapped": self.confidence_not_remapped,
        }

    def log_unmapped(self, subject: str) -> None:
        for owner, labels in self.unmapped_labels.items():
            logger.warning(
                "BirdNET V3 migration: %s %r references v2.4 label(s) with no V3 class %s; "
                "they keep matching other classifiers only",
                subject, owner, sorted(labels),
            )


def expand_labels_for_v3(
    labels: set[str] | frozenset[str],
    translator: BirdNETV24ToV3LabelTranslator,
) -> tuple[set[str], list[str], list[str]]:
    """(labels plus V3 names, added V3 names, unmapped v2.4 BirdNET labels)."""
    expanded = {_normalize(label) for label in labels}
    added: set[str] = set()
    unmapped: list[str] = []
    for label in sorted(expanded):
        if label in translator.unmapped_v24_labels and label not in translator.non_birdnet_vocabulary:
            unmapped.append(label)
        added |= translator.v3_labels_for(label) - expanded
    return expanded | added, sorted(added), unmapped


def birdnet_v3_label_migration_active(settings) -> bool:
    return (
        str(getattr(settings, "birdnet_model_version", "2.4")) == BIRDNET_V3_MODEL_VERSION
        and bool(getattr(settings, "birdnet_v3_label_migration_enabled", True))
    )


# ── rules ──────────────────────────────────────────────────────────────────

def migrate_rules_for_birdnet_v3(rules: list, translator: BirdNETV24ToV3LabelTranslator) -> tuple[list, LabelMigrationReport]:
    """Return migrated copies of ``RuleDef``s; inputs are not mutated."""
    report = LabelMigrationReport()
    migrated = []
    for rule in rules:
        condition = rule.condition
        if not condition.labels:
            migrated.append(rule)
            continue
        expanded, added, unmapped = expand_labels_for_v3(condition.labels, translator)
        if added:
            report.added_labels[rule.rule_id] = added
        if unmapped:
            report.unmapped_labels[rule.rule_id] = unmapped
        min_confidence = condition.min_confidence
        if min_confidence is not None:
            if all(translator.is_birdnet_only_label(label) for label in condition.labels):
                mapped = translator.map_confidence(min_confidence)
                report.remapped_confidences[rule.rule_id] = (min_confidence, mapped)
                min_confidence = mapped
            elif any(translator.is_birdnet_v24_label(label) for label in condition.labels):
                # Mixed vocabularies: one threshold cannot be right on both scales.
                report.confidence_not_remapped.append(rule.rule_id)
        new_condition = replace(condition, labels=expanded, min_confidence=min_confidence)
        migrated.append(replace(rule, condition=new_condition))
    return migrated, report


def birdnet_v3_rule_transform(settings):
    """``ConfigRuleEngine`` hook: ``None`` unless V3 migration is active."""
    if not birdnet_v3_label_migration_active(settings):
        return None
    translator = load_birdnet_v24_to_v3_translator()

    def transform(rules: list) -> list:
        migrated, report = migrate_rules_for_birdnet_v3(rules, translator)
        report.log_unmapped("rule")
        for rule_id in report.confidence_not_remapped:
            logger.warning(
                "BirdNET V3 migration: rule %r mixes BirdNET and other labels; its min_confidence "
                "stays on the v2.4 scale", rule_id,
            )
        return migrated

    return transform


# ── retention ──────────────────────────────────────────────────────────────

def migrate_retention_rules_for_birdnet_v3(rules: list, translator: BirdNETV24ToV3LabelTranslator) -> tuple[list, LabelMigrationReport]:
    """Clone label rules for V3 names; carry ``classifier: birdnet`` confidence bounds."""
    report = LabelMigrationReport()
    migrated = []
    for rule in rules:
        criteria = rule.match
        if criteria.classifier == BIRDNET_BACKEND_NAME and (
            criteria.confidence_min is not None or criteria.confidence_max is not None
        ):
            new_min = None if criteria.confidence_min is None else translator.map_confidence(criteria.confidence_min)
            new_max = None if criteria.confidence_max is None else translator.map_confidence(criteria.confidence_max)
            report.remapped_confidences[rule.rule_id] = (
                criteria.confidence_min if criteria.confidence_min is not None else criteria.confidence_max,
                new_min if new_min is not None else new_max,
            )
            rule = replace(rule, match=replace(criteria, confidence_min=new_min, confidence_max=new_max))
            criteria = rule.match
        migrated.append(rule)
        if criteria.label is None:
            continue
        _, added, unmapped = expand_labels_for_v3({criteria.label}, translator)
        if unmapped:
            report.unmapped_labels[rule.rule_id] = unmapped
        for v3_label in added:
            clone_id = f"{rule.rule_id}{V3_RETENTION_RULE_ID_SUFFIX}:{v3_label}"
            migrated.append(replace(rule, rule_id=clone_id, match=replace(criteria, label=v3_label)))
            report.added_labels.setdefault(rule.rule_id, []).append(v3_label)
    return migrated, report


def migrate_permanent_labels_for_birdnet_v3(labels, settings) -> tuple[str, ...]:
    if not birdnet_v3_label_migration_active(settings):
        return tuple(labels)
    expanded, _, unmapped = expand_labels_for_v3(set(labels), load_birdnet_v24_to_v3_translator())
    if unmapped:
        logger.warning("BirdNET V3 migration: retention_permanent_labels with no V3 class %s", unmapped)
    return tuple(sorted(expanded))


# ── taxonomy ───────────────────────────────────────────────────────────────

def migrate_taxonomy_for_birdnet_v3(taxonomy_provider, settings) -> LabelMigrationReport:
    """Give each V3 name the category configured for its v2.4 label (if unset)."""
    report = LabelMigrationReport()
    items = getattr(taxonomy_provider, "label_category_items", None)
    if not birdnet_v3_label_migration_active(settings) or not callable(items):
        return report
    translator = load_birdnet_v24_to_v3_translator()
    configured = dict(items())
    for label, category in configured.items():
        for v3_label in sorted(translator.v3_labels_for(label) - set(configured)):
            taxonomy_provider.register_label(v3_label, category)
            report.added_labels.setdefault(label, []).append(v3_label)
    return report


# ── routing ────────────────────────────────────────────────────────────────

def migrate_routing_for_birdnet_v3(routing, settings):
    """Carry BirdNET member / chain / trigger thresholds and labels to V3."""
    if not birdnet_v3_label_migration_active(settings):
        return routing
    translator = load_birdnet_v24_to_v3_translator()
    birdnet_members = {
        member_id for member_id, spec in routing.classifiers.items() if spec.backend == BIRDNET_BACKEND_NAME
    }
    classifiers = dict(routing.classifiers)
    for member_id in birdnet_members:
        spec = classifiers[member_id]
        if spec.min_confidence:
            classifiers[member_id] = replace(spec, min_confidence=translator.map_confidence(spec.min_confidence))
    chains = tuple(
        replace(
            chain,
            trigger_labels=frozenset(expand_labels_for_v3(chain.trigger_labels, translator)[0]),
            min_confidence=translator.map_confidence(chain.min_confidence) if chain.min_confidence else 0.0,
        )
        if chain.after in birdnet_members
        else chain
        for chain in routing.chains
    )
    triggers = tuple(
        replace(
            trigger,
            labels=frozenset(expand_labels_for_v3(trigger.labels, translator)[0]),
            min_confidence=translator.map_confidence(trigger.min_confidence) if trigger.min_confidence else 0.0,
        )
        if trigger.on in birdnet_members
        else trigger
        for trigger in routing.triggers
    )
    return replace(routing, classifiers=classifiers, chains=chains, triggers=triggers)
