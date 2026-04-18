"""Rule-based policy models for operator-managed snippet and artifact cleanup."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()


def _optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0 when provided")
    return parsed


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


@dataclass(slots=True)
class CleanupDefaults:
    snippet_max_age_seconds: int
    artifact_max_age_seconds: int

    def __post_init__(self) -> None:
        if self.snippet_max_age_seconds < 0:
            raise ValueError("snippet_max_age_seconds must be >= 0")
        if self.artifact_max_age_seconds < 0:
            raise ValueError("artifact_max_age_seconds must be >= 0")

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        default_snippet_max_age_seconds: int,
        default_artifact_max_age_seconds: int,
    ) -> "CleanupDefaults":
        return cls(
            snippet_max_age_seconds=int(
                payload.get("snippet_max_age_seconds", payload.get("default_snippet_max_age_seconds", default_snippet_max_age_seconds))
            ),
            artifact_max_age_seconds=int(
                payload.get("artifact_max_age_seconds", payload.get("default_artifact_max_age_seconds", default_artifact_max_age_seconds))
            ),
        )


@dataclass(slots=True)
class RetentionMatchCriteria:
    label: str | None = None
    classifier: str | None = None
    label_source: str | None = None
    retention_tier: str | None = None
    trigger_source: str | None = None
    review_state: str | None = None
    confidence_min: float | None = None
    confidence_max: float | None = None

    def __post_init__(self) -> None:
        if self.label is not None:
            self.label = _normalize_text(self.label)
        if self.classifier is not None:
            self.classifier = _normalize_text(self.classifier)
        if self.label_source is not None:
            self.label_source = _normalize_text(self.label_source)
        if self.retention_tier is not None:
            self.retention_tier = _normalize_text(self.retention_tier)
        if self.trigger_source is not None:
            self.trigger_source = _normalize_text(self.trigger_source)
        if self.review_state is not None:
            self.review_state = _normalize_text(self.review_state)
        if self.confidence_min is not None and self.confidence_max is not None:
            if self.confidence_min > self.confidence_max:
                raise ValueError("confidence_min must be <= confidence_max")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetentionMatchCriteria":
        return cls(
            label=payload.get("label"),
            classifier=payload.get("classifier"),
            label_source=payload.get("label_source"),
            retention_tier=payload.get("retention_tier"),
            trigger_source=payload.get("trigger_source"),
            review_state=payload.get("review_state"),
            confidence_min=float(payload["confidence_min"]) if payload.get("confidence_min") is not None else None,
            confidence_max=float(payload["confidence_max"]) if payload.get("confidence_max") is not None else None,
        )

    def matches(self, context: dict[str, Any]) -> bool:
        if self.label is not None and _normalize_text(str(context.get("label"))) != self.label:
            return False
        if self.classifier is not None and _normalize_text(str(context.get("classifier"))) != self.classifier:
            return False
        if self.label_source is not None and _normalize_text(str(context.get("label_source"))) != self.label_source:
            return False
        if self.retention_tier is not None and _normalize_text(str(context.get("retention_tier"))) != self.retention_tier:
            return False
        if self.trigger_source is not None and _normalize_text(str(context.get("trigger_source"))) != self.trigger_source:
            return False
        if self.review_state is not None and _normalize_text(str(context.get("review_state"))) != self.review_state:
            return False
        confidence = context.get("confidence")
        if self.confidence_min is not None:
            if confidence is None or float(confidence) < self.confidence_min:
                return False
        if self.confidence_max is not None:
            if confidence is None or float(confidence) > self.confidence_max:
                return False
        return True


@dataclass(slots=True)
class RetentionActions:
    snippet_max_age_seconds: int | None = None
    artifact_max_age_seconds: int | None = None
    keep_snippets: bool | None = None
    keep_artifacts: bool | None = None

    def __post_init__(self) -> None:
        if self.snippet_max_age_seconds is not None and self.snippet_max_age_seconds < 0:
            raise ValueError("snippet_max_age_seconds must be >= 0 when provided")
        if self.artifact_max_age_seconds is not None and self.artifact_max_age_seconds < 0:
            raise ValueError("artifact_max_age_seconds must be >= 0 when provided")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetentionActions":
        return cls(
            snippet_max_age_seconds=_optional_int(payload.get("snippet_max_age_seconds"), field_name="snippet_max_age_seconds"),
            artifact_max_age_seconds=_optional_int(payload.get("artifact_max_age_seconds"), field_name="artifact_max_age_seconds"),
            keep_snippets=_optional_bool(payload.get("keep_snippets")),
            keep_artifacts=_optional_bool(payload.get("keep_artifacts")),
        )


@dataclass(slots=True)
class RetentionRule:
    rule_id: str
    priority: int = 0
    match: RetentionMatchCriteria = field(default_factory=RetentionMatchCriteria)
    actions: RetentionActions = field(default_factory=RetentionActions)

    def __post_init__(self) -> None:
        self.rule_id = self.rule_id.strip()
        if not self.rule_id:
            raise ValueError("Retention rule id must not be empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, fallback_rule_id: str) -> "RetentionRule":
        match_payload = payload.get("match", {})
        actions_payload = payload.get("actions", {})
        if not isinstance(match_payload, dict):
            raise ValueError("Retention rule match must be an object")
        if not isinstance(actions_payload, dict):
            raise ValueError("Retention rule actions must be an object")
        return cls(
            rule_id=str(payload.get("id", fallback_rule_id)),
            priority=int(payload.get("priority", 0)),
            match=RetentionMatchCriteria.from_dict(match_payload),
            actions=RetentionActions.from_dict(actions_payload),
        )


@dataclass(slots=True)
class CleanupDecision:
    snippet_max_age_seconds: int | None
    artifact_max_age_seconds: int | None


@dataclass(slots=True)
class CleanupPolicy:
    version: int
    defaults: CleanupDefaults
    rules: list[RetentionRule] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("Cleanup policy version must be >= 1")
        self.rules = sorted(self.rules, key=lambda rule: (rule.priority, rule.rule_id))

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        default_snippet_max_age_seconds: int,
        default_artifact_max_age_seconds: int,
    ) -> "CleanupPolicy":
        version = int(payload.get("version", 1))
        defaults_payload = payload.get("defaults")
        if defaults_payload is not None and not isinstance(defaults_payload, dict):
            raise ValueError("defaults must be an object when provided")
        defaults = CleanupDefaults.from_dict(
            defaults_payload or payload,
            default_snippet_max_age_seconds=default_snippet_max_age_seconds,
            default_artifact_max_age_seconds=default_artifact_max_age_seconds,
        )

        rules: list[RetentionRule] = []
        rules_payload = payload.get("rules", [])
        if rules_payload is not None:
            if not isinstance(rules_payload, list):
                raise ValueError("rules must be a list when provided")
            for index, rule_payload in enumerate(rules_payload):
                if not isinstance(rule_payload, dict):
                    raise ValueError(f"Retention rule at index {index} must be an object")
                rules.append(RetentionRule.from_dict(rule_payload, fallback_rule_id=f"rule-{index}"))

        # Backward-compatible loader for the earlier keep_labels schema.
        keep_labels_raw = payload.get("keep_labels", {})
        if keep_labels_raw:
            if not isinstance(keep_labels_raw, dict):
                raise ValueError("keep_labels must be an object when provided")
            for raw_label, override_payload in keep_labels_raw.items():
                label = _normalize_text(raw_label)
                if not label:
                    continue
                if not isinstance(override_payload, dict):
                    raise ValueError(f"keep_labels entry for {raw_label!r} must be an object")
                rules.append(
                    RetentionRule(
                        rule_id=f"legacy-label-{label}",
                        priority=0,
                        match=RetentionMatchCriteria(label=label),
                        actions=RetentionActions.from_dict(override_payload),
                    )
                )

        return cls(version=version, defaults=defaults, rules=rules)

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        default_snippet_max_age_seconds: int,
        default_artifact_max_age_seconds: int,
    ) -> "CleanupPolicy":
        if not path.exists():
            return cls(
                version=1,
                defaults=CleanupDefaults(
                    snippet_max_age_seconds=default_snippet_max_age_seconds,
                    artifact_max_age_seconds=default_artifact_max_age_seconds,
                ),
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Cleanup policy at {path} must be a JSON object")
        return cls.from_dict(
            payload,
            default_snippet_max_age_seconds=default_snippet_max_age_seconds,
            default_artifact_max_age_seconds=default_artifact_max_age_seconds,
        )

    def matching_rules(self, context: dict[str, Any]) -> list[RetentionRule]:
        return [rule for rule in self.rules if rule.match.matches(context)]

    def resolve_for_context(self, context: dict[str, Any]) -> CleanupDecision:
        snippet_max_age_seconds = self.defaults.snippet_max_age_seconds
        artifact_max_age_seconds = self.defaults.artifact_max_age_seconds
        keep_snippets = False
        keep_artifacts = False

        for rule in self.matching_rules(context):
            if rule.actions.snippet_max_age_seconds is not None:
                snippet_max_age_seconds = rule.actions.snippet_max_age_seconds
            if rule.actions.artifact_max_age_seconds is not None:
                artifact_max_age_seconds = rule.actions.artifact_max_age_seconds
            if rule.actions.keep_snippets is not None:
                keep_snippets = rule.actions.keep_snippets
            if rule.actions.keep_artifacts is not None:
                keep_artifacts = rule.actions.keep_artifacts

        return CleanupDecision(
            snippet_max_age_seconds=None if keep_snippets else snippet_max_age_seconds,
            artifact_max_age_seconds=None if keep_artifacts else artifact_max_age_seconds,
        )

    def snippet_max_age_seconds_for_label(self, label: str | None) -> int | None:
        return self.resolve_for_context({"label": label}).snippet_max_age_seconds

    def artifact_max_age_seconds_for_label(self, label: str | None) -> int | None:
        return self.resolve_for_context({"label": label}).artifact_max_age_seconds
