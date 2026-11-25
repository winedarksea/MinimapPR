"""Field Mode W3-entity: policy decisions, grace window, FusionNode wiring, WS events."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.entity_reporting_policy import EntityReportingPolicy
from minimappr.core.fusion_node import DetectionProduct, FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.notification_grace_window import NotificationGraceWindow
from minimappr.core.reporting_fusion import ReportingFusionPolicy
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import DetectionEvent, GeoPoint, NodeSpec, NodeType
from minimappr.storage.db import Storage
from tests.bird_entity_helpers import (
    NODE_POSITIONS_M,
    ROBIN,
    ROBIN_POSITION_M,
    observation,
    seconds,
    sequential_registry,
    track_state,
)

GRACE_SECONDS = 0.05
GRACE_WAIT_SECONDS = 0.2


class RecordingRulesEngine:
    def __init__(self) -> None:
        self.evaluated: list[tuple[str, str]] = []

    async def evaluate(self, *, detection, track):
        self.evaluated.append((detection.id, detection.reporting_modality))
        return []


def _detection(det_id: str, *, at_s: float, modality: str, node_id: str = "node-a",
               position_m=ROBIN_POSITION_M, label: str = ROBIN) -> DetectionEvent:
    return DetectionEvent(
        id=det_id,
        timestamp_ns=seconds(at_s),
        source_node_id=node_id,
        reporting_modality=modality,
        spatial_display_mode="node_only" if modality == "omni" else "localized",
        position_m=NODE_POSITIONS_M[node_id] if modality == "omni" else position_m,
        position_covariance_m2=None if modality == "omni" else [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
        confidence=0.8,
        gdop=1.0,
        label=label,
        label_category="wildlife",
        label_confidence=0.8,
        reference_sensor="m0",
    )


def _product(detection: DetectionEvent, track=None, entity_id=None) -> DetectionProduct:
    return DetectionProduct(
        detection=detection, track=track, suppressed_by_zone=False, suppression_reasons=[],
        pipeline_item_id=detection.id, event_time_ns=detection.timestamp_ns, entity_id=entity_id,
    )


async def _fusion(tmp_path: Path, **overrides) -> tuple[FusionNode, Storage, list[dict], RecordingRulesEngine]:
    settings = Settings(
        db_path=tmp_path / "entity.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        fusion_worker_count=1,
        **overrides,
    )
    storage = Storage(settings.db_path)
    await storage.initialize()
    registry = NodeRegistry()
    for node_id, position in NODE_POSITIONS_M.items():
        await registry.upsert(
            NodeSpec(id=node_id, node_type=NodeType.POINT, position_m=position,
                     sensor_offsets_m=[(0.0, 0.0, 0.0)], capabilities=["audio"]),
            last_seen_ns=seconds(0),
        )
    live_payloads: list[dict] = []

    async def live_callback(payload: dict) -> None:
        live_payloads.append(payload)

    rules = RecordingRulesEngine()
    fusion = FusionNode(
        settings=settings, registry=registry,
        buffer=MultiSensorBuffer(max_duration_seconds=2.0),
        localizer=LocalizationEngine(max_tau_s=0.03), classifier=HeuristicClassifier(),
        tracker=TrackManager(settings), storage=storage, live_callback=live_callback,
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage), rules_engine=rules,
    )
    return fusion, storage, live_payloads, rules


def _entity_events(payloads: list[dict]) -> list[dict]:
    return [payload for payload in payloads if payload.get("type") == "entity"]


async def test_flag_off_keeps_legacy_policy_and_fires_rules_per_detection(tmp_path: Path) -> None:
    # Entity reporting defaults on since localization overhaul milestone 3.
    fusion, storage, payloads, rules = await _fusion(tmp_path, entity_reporting_enabled=False)
    try:
        assert fusion.entity_reporting is None
        assert isinstance(fusion.reporting_policy, ReportingFusionPolicy)
        await fusion._process_rules_and_delivery(_product(_detection("det-1", at_s=0, modality="omni")))
        await fusion._process_rules_and_delivery(_product(_detection("det-2", at_s=1, modality="localized")))
        assert rules.evaluated == [("det-1", "omni"), ("det-2", "localized")]
        assert _entity_events(payloads) == []
    finally:
        await storage.close()


async def test_unlocalized_then_localized_inside_grace_fires_once_as_entity(tmp_path: Path) -> None:
    fusion, storage, payloads, rules = await _fusion(
        tmp_path, entity_reporting_enabled=True, entity_notification_grace_seconds=GRACE_SECONDS
    )
    try:
        await fusion._process_rules_and_delivery(_product(_detection("det-omni", at_s=0, modality="omni")))
        assert rules.evaluated == []  # held for the grace window
        entity_id = _entity_events(payloads)[0]["entity"]["id"]
        localized = _product(_detection("det-loc", at_s=1, modality="localized"),
                             track=track_state(), entity_id=entity_id)
        await fusion._process_rules_and_delivery(localized)
        assert rules.evaluated == [("det-loc", "localized")]
        await asyncio.sleep(GRACE_WAIT_SECONDS)
        await fusion._process_rules_and_delivery(
            _product(_detection("det-loc-2", at_s=8, modality="localized"), track=track_state())
        )
        assert rules.evaluated == [("det-loc", "localized")]  # still once
        events = _entity_events(payloads)
        assert [event["event"] for event in events] == ["created", "state_changed", "updated"]
        assert {event["entity"]["id"] for event in events} == {entity_id}
        pinpointed = events[1]["entity"]
        assert pinpointed["state"] == "pinpointed" and events[1]["previous_state"] == "heard_nearby"
        assert pinpointed["position_geo"] is not None
        assert pinpointed["uncertainty_ellipse"]["semi_major_m"] == pytest.approx(2.0)
        assert pinpointed["species_top_label"] == ROBIN
    finally:
        await fusion.entity_reporting.stop()
        await storage.close()


async def test_never_localized_fires_unlocalized_after_grace(tmp_path: Path) -> None:
    fusion, storage, payloads, rules = await _fusion(
        tmp_path, entity_reporting_enabled=True, entity_notification_grace_seconds=GRACE_SECONDS
    )
    try:
        await fusion._process_rules_and_delivery(_product(_detection("det-omni", at_s=0, modality="omni")))
        assert rules.evaluated == []
        await asyncio.sleep(GRACE_WAIT_SECONDS)
        assert rules.evaluated == [("det-omni", "omni")]
        heard = _entity_events(payloads)[0]["entity"]
        assert heard["state"] == "heard_nearby"
        assert heard["position_m"] is None and heard["reporting_node_id"] == "node-a"
        assert fusion.entity_reporting.grace_window.fired_after_grace_count == 1
    finally:
        await fusion.entity_reporting.stop()
        await storage.close()


async def test_zero_grace_fires_immediately_and_expiry_cancels_pending(tmp_path: Path) -> None:
    fired: list[str] = []

    async def fire(detection, track):
        fired.append(detection)

    immediate = NotificationGraceWindow(grace_seconds=0.0, fire_rules=fire)
    await immediate.hold_or_fire("ent-1", "det-a", None)
    assert fired == ["det-a"]

    held = NotificationGraceWindow(grace_seconds=GRACE_SECONDS, fire_rules=fire)
    await held.hold_or_fire("ent-2", "det-b", None)
    held.forget("ent-2")  # entity expired before the window ended
    await asyncio.sleep(GRACE_WAIT_SECONDS)
    assert fired == ["det-a"]


async def test_hidden_or_suppressed_detections_never_create_entities(tmp_path: Path) -> None:
    fusion, storage, payloads, rules = await _fusion(tmp_path, entity_reporting_enabled=True)
    try:
        sub_display = _detection("det-sub", at_s=0, modality="localized")
        sub_display.feature_summary = {"below_display_threshold": True}
        await fusion._process_rules_and_delivery(_product(sub_display, track=None))
        zone = _product(_detection("det-zone", at_s=1, modality="localized"))
        zone.suppressed_by_zone = True
        await fusion._process_rules_and_delivery(zone)
        assert fusion.entity_reporting.registry.entities() == []
        assert _entity_events(payloads) == [] and rules.evaluated == []
    finally:
        await storage.close()


async def test_policy_absorbs_unpositioned_and_upgrades_positioned(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "policy.db")
    await storage.initialize()
    await storage.upsert_node(
        NodeSpec(id="node-a", node_type=NodeType.SIRITH_TETRA, position_m=(0.0, 0.0, 0.0)), last_seen_ns=seconds(0)
    )
    try:
        registry = sequential_registry()
        policy = EntityReportingPolicy(storage=storage, registry=registry)
        common = dict(source_node_id="node-a", reporting_node_position_m=NODE_POSITIONS_M["node-a"], label=ROBIN,
                      branch_details={"label": ROBIN, "confidence": 0.7}, below_display_threshold=False)
        first, change = await policy.decide(
            event_time_ns=seconds(0), reporting_modality="localized", spatial_display_mode="localized",
            localization_position_m=ROBIN_POSITION_M, **common,
        )
        assert first.action == "insert" and change is None and first.entity_id is None
        assert first.report_window_start_ns == seconds(0)
        assert first.report_window_end_ns == seconds(60)  # sliding, not a bucket edge
        canonical = _detection("det-canon", at_s=0, modality="localized")
        canonical.report_window_start_ns, canonical.report_window_end_ns = first.report_window_start_ns, first.report_window_end_ns
        await storage.insert_detection(canonical, snippet_path=None, snippet_expires_ns=None)
        entity = registry.observe(observation(at_s=0, track=track_state(), detection_id="det-canon")).entity

        absorbed, change = await policy.decide(
            event_time_ns=seconds(20), reporting_modality="omni", spatial_display_mode="node_only",
            localization_position_m=None, **common,
        )
        assert absorbed.action == "enrich_existing" and absorbed.entity_id == entity.entity_id
        assert absorbed.existing_detection["id"] == "det-canon"
        assert absorbed.branch_evidence["omni"]["entity_id"] == entity.entity_id
        assert change.kind == "absorbed" and entity.absorbed_detection_count == 1

        upgraded, _ = await policy.decide(
            event_time_ns=seconds(30), reporting_modality="localized", spatial_display_mode="localized",
            localization_position_m=(6.5, 12.5, 2.0), **common,
        )
        assert upgraded.action == "upgrade_existing" and upgraded.existing_detection["id"] == "det-canon"

        sub_display, change = await policy.decide(
            event_time_ns=seconds(31), reporting_modality="omni", spatial_display_mode="node_only",
            localization_position_m=None, **{**common, "below_display_threshold": True},
        )
        assert sub_display.action == "insert" and change is None
    finally:
        await storage.close()


async def test_maintenance_emits_state_changes_and_expiry(tmp_path: Path) -> None:
    fusion, storage, payloads, _rules = await _fusion(
        tmp_path, entity_reporting_enabled=True, entity_notification_grace_seconds=0.0,
        entity_expiry_seconds=60.0, entity_absorption_window_seconds=30.0,
    )
    try:
        await fusion._process_rules_and_delivery(
            _product(_detection("det-loc", at_s=0, modality="localized"), track=track_state("trk-gone"))
        )
        # trk-gone is not in the tracker: one step back, then expiry.
        changes = await fusion.entity_reporting.maintain_once(now_ns=seconds(5))
        assert [c.entity.state for c in changes] == ["direction_known"]
        changes = await fusion.entity_reporting.maintain_once(now_ns=seconds(61))
        assert [c.kind for c in changes] == ["expired"]
        assert [event["event"] for event in _entity_events(payloads)][-2:] == ["state_changed", "expired"]
        listing = await fusion.entity_reporting.list_entities()
        assert listing.enabled is True and listing.entities == []
    finally:
        await storage.close()


async def test_grace_window_stop_flushes_held_notifications_exactly_once() -> None:
    fired: list[str] = []

    async def fire(detection, track):
        fired.append(detection)

    window = NotificationGraceWindow(grace_seconds=60.0, fire_rules=fire)
    await window.hold_or_fire("ent-held", "det-held", None)
    assert fired == [] and window.is_pending("ent-held")
    await window.stop()  # shutdown must not drop the pending alert
    assert fired == ["det-held"]
    assert not window.is_pending("ent-held") and window.fired_on_shutdown_count == 1
    await window.hold_or_fire("ent-held", "det-later", None)
    assert fired == ["det-held"]  # already notified: never twice
    assert window._timer_tasks == set()


async def test_grace_window_stop_waits_for_in_flight_fire() -> None:
    fire_started = asyncio.Event()
    release_fire = asyncio.Event()
    fired: list[str] = []

    async def slow_fire(detection, track):
        fire_started.set()
        await release_fire.wait()
        fired.append(detection)

    window = NotificationGraceWindow(grace_seconds=GRACE_SECONDS, fire_rules=slow_fire)
    await window.hold_or_fire("ent-slow", "det-slow", None)
    await asyncio.wait_for(fire_started.wait(), timeout=2.0)
    stop_task = asyncio.create_task(window.stop())
    await asyncio.sleep(0)
    assert not stop_task.done()
    release_fire.set()
    await asyncio.wait_for(stop_task, timeout=2.0)
    assert fired == ["det-slow"] and window.fired_on_shutdown_count == 0
