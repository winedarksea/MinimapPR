"""Config-driven rules engine and action handlers."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minimappr.interfaces import ActionDescriptor, RuleActionHandler, RuleEngine, RuleEvaluationResult
from minimappr.models import DetectionEvent, TrackState, TranscriptRecord


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
    reporting_modalities: set[str] = field(default_factory=set)
    zone_ids: set[str] = field(default_factory=set)
    track_statuses: set[str] = field(default_factory=set)
    source_types: set[str] = field(default_factory=set)
    min_confidence: float | None = None
    transcript_contains: set[str] = field(default_factory=set)

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
            reporting_modalities=_as_set(raw.get("reporting_modalities")),
            zone_ids=_as_set(raw.get("zone_ids")),
            track_statuses=_as_set(raw.get("track_statuses")),
            source_types=_as_set(raw.get("source_types")),
            min_confidence=min_conf,
            transcript_contains=_as_set(raw.get("transcript_contains")),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "label_categories": sorted(self.label_categories),
            "labels": sorted(self.labels),
            "reporting_modalities": sorted(self.reporting_modalities),
            "zone_ids": sorted(self.zone_ids),
            "track_statuses": sorted(self.track_statuses),
            "source_types": sorted(self.source_types),
            "transcript_contains": sorted(self.transcript_contains),
        }
        if self.min_confidence is not None:
            out["min_confidence"] = self.min_confidence
        return out


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.rule_id,
            "enabled": self.enabled,
            "scope": self.scope,
            "when": self.condition.to_dict(),
            "actions": [
                {
                    "type": action.action_type,
                    "destination": action.destination,
                    "priority": action.priority,
                    "payload": action.payload,
                }
                for action in self.actions
            ],
            "cooldown_seconds": self.cooldown_seconds,
        }


class ConfigRuleEngine(RuleEngine):
    def __init__(self, config_path: Path, *, reload_ttl_seconds: float = 1.0) -> None:
        self._config_path = config_path
        self._reload_ttl_s = max(0.0, float(reload_ttl_seconds))
        self._rules: list[RuleDef] = []
        self._last_mtime_ns: int | None = None
        self._last_fire_ns: dict[str, int] = {}
        self._last_check_monotonic: float | None = None
        self._default_rules: list[RuleDef] | None = None
        self.reload(force=True)

    def reload(self, *, force: bool = False) -> None:
        # evaluate() calls reload() on every detection; throttle the filesystem
        # stat to at most once per TTL so a rules-file edit may take up to
        # ~reload_ttl_seconds to be observed.
        now = time.monotonic()
        if (
            not force
            and self._last_check_monotonic is not None
            and now - self._last_check_monotonic < self._reload_ttl_s
        ):
            return
        self._last_check_monotonic = now
        if not self._config_path.exists():
            self._last_mtime_ns = None
            if self._default_rules is None:
                self._default_rules = default_rules()
            self._rules = self._default_rules
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
            if rule.scope not in {"detection", "track"}:
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

    def rules(self) -> list[RuleDef]:
        """Read-only snapshot of the parsed rule definitions (reloads first).

        Used by the pipeline-graph builder to render the rules/alerts stage.
        """
        self.reload()
        return list(self._rules)

    async def evaluate_transcript(
        self, transcript: TranscriptRecord
    ) -> list[RuleEvaluationResult]:
        self.reload()
        now_ns = time.time_ns()
        out: list[RuleEvaluationResult] = []
        lowered_text = transcript.text.strip().lower()
        for rule in self._rules:
            if not rule.enabled or rule.scope != "transcript":
                continue
            cond = rule.condition
            if not cond.transcript_contains:
                out.append(RuleEvaluationResult(rule_id=rule.rule_id, matched=False, reason="no_condition"))
                continue
            if not any(needle in lowered_text for needle in cond.transcript_contains):
                out.append(RuleEvaluationResult(rule_id=rule.rule_id, matched=False, reason="transcript_text"))
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
    reporting_modality = ""

    if detection is not None:
        label = detection.label.strip().lower()
        category = detection.label_category.strip().lower()
        zone_ids = {zone.strip().lower() for zone in detection.zone_ids}
        confidence = float(detection.label_confidence)
        source_type = detection.source_type.strip().lower()
        reporting_modality = detection.reporting_modality.strip().lower()
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
    if cond.reporting_modalities and reporting_modality not in cond.reporting_modalities:
        return False, "reporting_modality"
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
        alert_id: str | None = None,
        rule_id: str | None = None,
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
    """Push a fired alert to live websocket clients.

    The payload is **flat**, not nested under an ``"alert"`` key, and carries both
    ``type`` and ``event_type``. The frontend's ``LiveEvent`` enum is
    ``#[serde(tag = "type")]`` over an internally-tagged ``Alert`` struct, so the
    previous nested/untagged shape failed to deserialize and every live alert was
    silently dropped at the websocket boundary. Flat also means the websocket and
    the ``GET /api/v1/alerts`` poll deliver the same shape, so the two paths
    cannot drift again.
    """

    def __init__(self, send_callback) -> None:
        self._send_callback = send_callback

    async def handle(
        self,
        descriptor: ActionDescriptor,
        *,
        detection: DetectionEvent | None = None,
        track: TrackState | None = None,
        alert_id: str | None = None,
        rule_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "type": "alert",
            "event_type": "alert",
            "action_type": descriptor.action_type,
            "destination": descriptor.destination,
            "priority": descriptor.priority,
            "status": "sent",
            "timestamp_ns": time.time_ns(),
            "payload": descriptor.payload,
            "detection_id": detection.id if detection else None,
            "track_id": track.id if track else None,
        }
        # Omit rather than null-fill: the consumer types these as required
        # strings, and an explicit null fails deserialization where absence
        # falls back to a default.
        if alert_id is not None:
            payload["alert_id"] = alert_id
        if rule_id is not None:
            payload["rule_id"] = rule_id
        await self._send_callback(payload)
        return {"delivered": True, "handler": "websocket"}


# The real Home Assistant handler lives in core/hass/rules_handler.py — it needs
# the MQTT bridge, which would make this module import the whole hass package.


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
        # BirdNET exposes a direct "coyote" label for wildlife deployments.
        # Alerts are limited to canonical localized detections so omni-only
        # BirdNET hits are still stored for review without paging operators.
        RuleDef(
            rule_id="coyote_alert",
            enabled=True,
            scope="detection",
            condition=RuleCondition(
                labels={"coyote"},
                reporting_modalities={"localized"},
                min_confidence=0.4,
            ),
            actions=[
                ActionDescriptor(
                    action_type="alert",
                    destination="cop",
                    priority="high",
                    payload={"message": "Coyote detected"},
                ),
                ActionDescriptor(action_type="alert", destination="log", priority="high"),
            ],
            cooldown_seconds=30.0,
        ),
        RuleDef(
            rule_id="gunshot_alert",
            enabled=True,
            scope="detection",
            condition=RuleCondition(
                labels={"gunshot", "gunshot, gunfire", "machine gun"},
                min_confidence=0.5,
            ),
            actions=[
                ActionDescriptor(
                    action_type="alert",
                    destination="cop",
                    priority="high",
                    payload={"message": "Gunshot detected"},
                ),
                ActionDescriptor(action_type="alert", destination="log", priority="high"),
            ],
            cooldown_seconds=5.0,
        ),
        RuleDef(
            rule_id="temporal_alarm_alert",
            enabled=True,
            scope="detection",
            condition=RuleCondition(labels={"alarm_t3", "alarm_t4"}, min_confidence=0.5),
            actions=[
                ActionDescriptor(
                    action_type="alert",
                    destination="cop",
                    priority="high",
                    payload={"message": "T3/T4 temporal alarm cadence detected"},
                ),
                ActionDescriptor(action_type="alert", destination="log", priority="high"),
            ],
            cooldown_seconds=90.0,
        ),
        RuleDef(
            rule_id="drone_alert",
            enabled=True,
            scope="detection",
            condition=RuleCondition(labels={"drone"}, min_confidence=0.5),
            actions=[
                ActionDescriptor(
                    action_type="alert",
                    destination="cop",
                    priority="high",
                    payload={"message": "Drone detected"},
                ),
                ActionDescriptor(action_type="alert", destination="log", priority="high"),
            ],
            cooldown_seconds=30.0,
        ),
        RuleDef(
            rule_id="help_me_alert",
            enabled=True,
            scope="transcript",
            condition=RuleCondition(transcript_contains={"help me"}),
            actions=[
                ActionDescriptor(
                    action_type="alert",
                    destination="cop",
                    priority="critical",
                    payload={"message": "Speech transcript contained \"help me\""},
                ),
                ActionDescriptor(action_type="alert", destination="log", priority="critical"),
            ],
            cooldown_seconds=10.0,
        ),
    ]


def default_rules_as_dicts() -> list[dict[str, Any]]:
    return [rule.to_dict() for rule in default_rules()]
