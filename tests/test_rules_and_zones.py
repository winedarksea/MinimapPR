from __future__ import annotations

import json
from pathlib import Path

import pytest

from minimappr.core.rules import ConfigRuleEngine
from minimappr.core.zones import ZoneMatcher
from minimappr.models import DetectionEvent, TranscriptRecord
from minimappr.storage.db import Storage


@pytest.mark.asyncio
async def test_zone_matcher_applies_exclusion_and_only_in_zone_rules(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "zones.db")
    await storage.initialize()

    await storage.upsert_zone(
        zone_id="coop-exclusion",
        name="Chicken Coop",
        zone_type="exclusion_zone",
        polygon_geo=[[37.0, -122.0], [37.0, -121.999], [37.001, -121.999], [37.001, -122.0]],
        properties={"suppress_label_categories": ["wildlife"]},
        created_ns=1,
    )
    await storage.upsert_zone(
        zone_id="perimeter-security",
        name="Perimeter",
        zone_type="alert_zone",
        polygon_geo=[[38.0, -123.0], [38.0, -122.999], [38.001, -122.999], [38.001, -123.0]],
        properties={"only_in_zone_label_categories": ["security"]},
        created_ns=1,
    )

    matcher = ZoneMatcher(storage=storage, refresh_interval_seconds=0.0)
    zone_ids = await matcher.match_geo_point(lat=37.0005, lon=-121.9995, now_ns=10_000_000_000)
    assert "coop-exclusion" in zone_ids

    suppressed, reasons = await matcher.evaluate_detection_policies(
        zone_ids=zone_ids,
        label="bird_like",
        label_category="wildlife",
        now_ns=11_000_000_000,
    )
    assert suppressed is True
    assert any("suppressed_by_zone_rule" in reason for reason in reasons)

    outside_security_suppressed, outside_reasons = await matcher.evaluate_detection_policies(
        zone_ids=[],
        label="gunshot",
        label_category="security",
        now_ns=12_000_000_000,
    )
    assert outside_security_suppressed is True
    assert any("outside_required_zone" in reason for reason in outside_reasons)

    await storage.close()


@pytest.mark.asyncio
async def test_rules_engine_config_and_cooldown(tmp_path: Path) -> None:
    config_path = tmp_path / "rules.json"
    config_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "id": "security_perimeter",
                        "enabled": True,
                        "scope": "detection",
                        "when": {
                            "label_categories": ["security"],
                            "zone_ids": ["perimeter"],
                            "min_confidence": 0.4,
                        },
                        "cooldown_seconds": 60.0,
                        "actions": [{"type": "alert", "destination": "cop", "priority": "high"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    engine = ConfigRuleEngine(config_path=config_path)
    detection = DetectionEvent(
        id="det-001",
        timestamp_ns=1_700_000_000_000_000_000,
        position_m=(0.0, 0.0, 0.0),
        confidence=0.8,
        gdop=1.2,
        label="gunshot",
        label_category="security",
        label_confidence=0.9,
        zone_ids=["perimeter"],
        reference_sensor="node-a:ch0",
    )

    first = await engine.evaluate(detection=detection, track=None)
    assert first
    assert first[0].matched is True
    assert first[0].descriptors
    assert first[0].descriptors[0].destination == "cop"

    second = await engine.evaluate(detection=detection, track=None)
    assert second
    assert second[0].matched is False
    assert second[0].reason == "cooldown"


@pytest.mark.asyncio
async def test_default_drone_rule_alerts_at_head_acceptance_threshold(tmp_path: Path) -> None:
    engine = ConfigRuleEngine(tmp_path / "missing-rules.json")
    detection = DetectionEvent(
        id="det-drone",
        timestamp_ns=1_700_000_000_000_000_000,
        position_m=(0.0, 0.0, 0.0),
        confidence=0.5,
        gdop=1.2,
        label="drone",
        label_category="security",
        label_confidence=0.5,
        reference_sensor="node-a:ch0",
    )
    evaluations = await engine.evaluate(detection=detection, track=None)
    drone = next(item for item in evaluations if item.rule_id == "drone_alert")
    assert drone.matched is True


@pytest.mark.asyncio
async def test_transcript_rule_matches_text(tmp_path: Path) -> None:
    engine = ConfigRuleEngine(tmp_path / "missing-rules.json")
    transcript = TranscriptRecord(
        id="txt-001",
        node_id="node-a",
        sensor_id="node-a:ch0",
        start_ns=1,
        end_ns=2,
        text="please help me now",
        model="test",
        trigger_confidence=0.9,
        created_ns=3,
    )
    evaluations = await engine.evaluate_transcript(transcript)
    help_me = next(item for item in evaluations if item.rule_id == "help_me_alert")
    assert help_me.matched is True
