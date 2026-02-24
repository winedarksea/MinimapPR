"""Config-driven rules engine and action handlers."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minimappr.interfaces import ActionDescriptor, RuleActionHandler, RuleEngine, RuleEvaluationResult
from minimappr.models import DetectionEvent, TrackState


logger = logging.getLogger(__name__)


def _as_set(value: Any) -> set[str]:
    if isinstance(value, str):
        text = value.strip().lower()
        return {text} if text else set()
    if isinstance(value, (list, tuple, set)):
        out: set[str] = set()
        for item in value:
            text = str(item).strip().lower()
            if text:
                out.add(text)
        return out
    return set()


@dataclass(slots=True)
class RuleCondition:
    label_categories: set[str] = field(default_factory=set)
    labels: set[str] = field(default_factory=set)
    zone_ids: set[str] = field(default_factory=set)
    track_statuses: set[str] = field(default_factory=set)
    source_types: set[str] = field(default_factory=set)
    min_confidence: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RuleCondition":
        min_conf = raw.get("min_confidence")
        if min_conf is not None:
            try:
                min_conf = float(min_conf)
            except (TypeError, ValueError):
                min_conf = None
        return cls(
            label_categories=_as_set(raw.get("label_categories")),
            labels=_as_set(raw.get("labels")),
            zone_ids=_as_set(raw.get("zone_ids")),
            track_statuses=_as_set(raw.get("track_statuses")),
            source_types=_as_set(raw.get("source_types")),
            min_confidence=min_conf,
        )


@dataclass(slots=True)
class RuleDef:
    rule_id: str
    enabled: bool
    scope: str
    condition: RuleCondition
    actions: list[ActionDescriptor]
    cooldown_seconds: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RuleDef | None":
        rid = str(raw.get("id") or "").strip()
        if not rid:
            return None
        enabled = bool(raw.get("enabled", True))
        scope = str(raw.get("scope") or "detection").strip().lower()
        condition = RuleCondition.from_dict(raw.get("when") if isinstance(raw.get("when"), dict) else {})
        actions = []
        for item in raw.get("actions", []):
            if not isinstance(item, dict):
                continue
            action_type = str(item.get("type") or "alert").strip().lower()
            destination = str(item.get("destination") or "cop").strip().lower()
            priority = str(item.get("priority") or "normal").strip().lower()
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            actions.append(
                ActionDescriptor(
                    action_type=action_type,
                    destination=destination,
                    priority=priority,
                    payload=payload,
                )
            )
        if not actions:
            actions = [ActionDescriptor(action_type="alert", destination="cop", priority="normal", payload={})]
        cooldown_s = float(raw.get("cooldown_seconds", 0.0) or 0.0)
        return cls(
            rule_id=rid,
            enabled=enabled,
            scope=scope,
            condition=condition,
            actions=actions,
            cooldown_seconds=max(0.0, cooldown_s),
        )


class ConfigRuleEngine(RuleEngine):
    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._rules: list[RuleDef] = []
        self._last_mtime_ns: int | None = None
        self._last_fire_ns: dict[str, int] = {}
        self.reload()

    def reload(self) -> None:
        if not self._config_path.exists():
            self._rules = default_rules()
            return
        try:
            stat = self._config_path.stat()
            mtime_ns = stat.st_mtime_ns
            if self._last_mtime_ns is not None and mtime_ns == self._last_mtime_ns:
                return
            raw = json.loads(self._config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("rules config must be a JSON object")
            parsed: list[RuleDef] = []
            for item in raw.get("rules", []):
                if not isinstance(item, dict):
                    continue
                rule = RuleDef.from_dict(item)
                if rule is not None:
                    parsed.append(rule)
            self._rules = parsed or default_rules()
            self._last_mtime_ns = mtime_ns
        except Exception as exc:
            logger.warning("Failed to read rules config %s: %s", self._config_path, exc)
            self._rules = default_rules()

    async def evaluate(
        self,
        *,
        detection: DetectionEvent | None = None,
        track: TrackState | None = None,
    ) -> list[RuleEvaluationResult]:
        self.reload()
        now_ns = time.time_ns()
        out: list[RuleEvaluationResult] = []
        for rule in self._rules:
            if not rule.enabled:
                continue
            if rule.scope == "detection" and detection is None:
                continue
            if rule.scope == "track" and track is None:
                continue

            match, reason = _matches(rule, detection=detection, track=track)
            if not match:
                out.append(RuleEvaluationResult(rule_id=rule.rule_id, matched=False, reason=reason))
                continue

            if rule.cooldown_seconds > 0.0:
                key = rule.rule_id
                last = self._last_fire_ns.get(key)
                if last is not None and now_ns - last < int(rule.cooldown_seconds * 1_000_000_000):
                    out.append(RuleEvaluationResult(rule_id=rule.rule_id, matched=False, reason="cooldown"))
                    continue
                self._last_fire_ns[key] = now_ns

            out.append(
                RuleEvaluationResult(
                    rule_id=rule.rule_id,
                    matched=True,
                    descriptors=list(rule.actions),
                )
            )
        return out


def _matches(
    rule: RuleDef,
    *,
    detection: DetectionEvent | None,
    track: TrackState | None,
) -> tuple[bool, str]:
    cond = rule.condition
    label = ""
    category = "unknown"
    zone_ids: set[str] = set()
    status = ""
    confidence: float | None = None
    source_type = ""

    if detection is not None:
        label = detection.label.strip().lower()
        category = detection.label_category.strip().lower()
        zone_ids = {zone.strip().lower() for zone in detection.zone_ids}
        confidence = float(detection.label_confidence)
        source_type = detection.source_type.strip().lower()
    if track is not None:
        if not label:
            label = track.label.strip().lower()
        category = track.label_category.strip().lower() or category
        status = track.status.strip().lower()
        if confidence is None:
            confidence = float(track.confidence)

    if cond.labels and label not in cond.labels:
        return False, "label"
    if cond.label_categories and category not in cond.label_categories:
        return False, "category"
    if cond.zone_ids and zone_ids.isdisjoint(cond.zone_ids):
        return False, "zone"
    if cond.track_statuses and status not in cond.track_statuses:
        return False, "status"
    if cond.min_confidence is not None:
        value = confidence if confidence is not None else 0.0
        if value < cond.min_confidence:
            return False, "confidence"
    if cond.source_types and source_type not in cond.source_types:
        return False, "source_type"
    return True, "matched"


class LoggingRuleActionHandler(RuleActionHandler):
    async def handle(
        self,
        descriptor: ActionDescriptor,
        *,
        detection: DetectionEvent | None = None,
        track: TrackState | None = None,
    ) -> dict[str, Any]:
        logger.info(
            "rule_action type=%s destination=%s priority=%s detection=%s track=%s",
            descriptor.action_type,
            descriptor.destination,
            descriptor.priority,
            detection.id if detection else None,
            track.id if track else None,
        )
        return {"delivered": True, "handler": "log"}


class WebsocketRuleActionHandler(RuleActionHandler):
    def __init__(self, send_callback) -> None:
        self._send_callback = send_callback

    async def handle(
        self,
        descriptor: ActionDescriptor,
        *,
        detection: DetectionEvent | None = None,
        track: TrackState | None = None,
    ) -> dict[str, Any]:
        payload = {
            "event_type": "alert",
            "alert": {
                "action_type": descriptor.action_type,
                "destination": descriptor.destination,
                "priority": descriptor.priority,
                "payload": descriptor.payload,
                "detection_id": detection.id if detection else None,
                "track_id": track.id if track else None,
            },
        }
        await self._send_callback(payload)
        return {"delivered": True, "handler": "websocket"}


def default_rules() -> list[RuleDef]:
    return [
        RuleDef(
            rule_id="security_high_confidence",
            enabled=True,
            scope="detection",
            condition=RuleCondition(label_categories={"security"}, min_confidence=0.45),
            actions=[
                ActionDescriptor(action_type="alert", destination="cop", priority="high"),
                ActionDescriptor(action_type="alert", destination="log", priority="high"),
            ],
            cooldown_seconds=2.0,
        ),
        RuleDef(
            rule_id="human_perimeter",
            enabled=True,
            scope="detection",
            condition=RuleCondition(label_categories={"human"}, min_confidence=0.5),
            actions=[ActionDescriptor(action_type="alert", destination="cop", priority="normal")],
            cooldown_seconds=2.0,
        ),
    ]

