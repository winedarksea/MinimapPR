"""Site-wide omni→localized suppression in ReportingFusionPolicy (Phase 5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from minimappr.core.reporting_fusion import ReportingFusionPolicy
from minimappr.core.taxonomy import RuntimeTaxonomyProvider
from minimappr.models import DetectionEvent, NodeSpec, NodeType
from minimappr.storage.db import Storage

WINDOW_SECONDS = 30.0
EVENT_TIME_NS = 1_000_000_000_000


@pytest.fixture
async def storage(tmp_path: Path) -> Storage:
    store = Storage(tmp_path / "dedupe.db")
    await store.initialize()
    for node_id in ("node-a", "node-b"):
        await store.upsert_node(
            NodeSpec(id=node_id, node_type=NodeType.SIRITH_TETRA, position_m=(0.0, 0.0, 0.0)),
            last_seen_ns=EVENT_TIME_NS,
        )
    yield store
    await store.close()


def _policy(storage: Storage, **kwargs) -> ReportingFusionPolicy:
    return ReportingFusionPolicy(
        storage=storage,
        reporting_window_seconds=WINDOW_SECONDS,
        **kwargs,
    )


async def _seed_localized(
    storage: Storage,
    policy: ReportingFusionPolicy,
    *,
    label: str = "coyote",
    node_id: str = "node-a",
    position: tuple[float, float, float] = (10.0, 5.0, 0.0),
) -> DetectionEvent:
    window_start_ns, window_end_ns = policy.report_window_bounds(EVENT_TIME_NS)
    detection = DetectionEvent(
        id=f"det-{node_id}-{label}",
        timestamp_ns=EVENT_TIME_NS,
        source_node_id=node_id,
        report_window_start_ns=window_start_ns,
        report_window_end_ns=window_end_ns,
        reporting_modality="localized",
        position_m=position,
        confidence=0.8,
        gdop=1.5,
        label=label,
        label_confidence=0.9,
        reference_sensor="m0",
    )
    await storage.insert_detection(detection, snippet_path=None, snippet_expires_ns=None)
    return detection


async def test_site_scope_suppresses_omni_from_other_node(storage: Storage) -> None:
    policy = _policy(storage, omni_suppression_scope="site")
    existing = await _seed_localized(storage, policy)

    decision = await policy.decide(
        event_time_ns=EVENT_TIME_NS + 1_000_000,
        source_node_id="node-b",
        label="coyote",
        reporting_modality="omni",
        branch_details={"label": "coyote", "confidence": 0.6},
    )
    assert decision.action == "enrich_existing"
    assert decision.suppression_reason == "localized_detection_site_wide"
    assert decision.existing_detection is not None
    assert decision.existing_detection["id"] == existing.id
    omni_evidence = decision.branch_evidence["omni"]
    assert omni_evidence["suppressed"] is True
    assert omni_evidence["suppression_reason"] == "localized_detection_site_wide"


async def test_node_scope_keeps_legacy_per_node_insert(storage: Storage) -> None:
    policy = _policy(storage, omni_suppression_scope="node")
    await _seed_localized(storage, policy)

    decision = await policy.decide(
        event_time_ns=EVENT_TIME_NS + 1_000_000,
        source_node_id="node-b",
        label="coyote",
        reporting_modality="omni",
        branch_details={"label": "coyote", "confidence": 0.6},
    )
    assert decision.action == "insert"


async def test_different_label_still_inserts_site_scope(storage: Storage) -> None:
    policy = _policy(storage, omni_suppression_scope="site")
    await _seed_localized(storage, policy, label="coyote")

    decision = await policy.decide(
        event_time_ns=EVENT_TIME_NS + 1_000_000,
        source_node_id="node-b",
        label="owl",
        reporting_modality="omni",
        branch_details={"label": "owl", "confidence": 0.6},
    )
    assert decision.action == "insert"


async def test_localized_report_never_site_suppressed(storage: Storage) -> None:
    policy = _policy(storage, omni_suppression_scope="site")
    await _seed_localized(storage, policy, node_id="node-a")

    decision = await policy.decide(
        event_time_ns=EVENT_TIME_NS + 1_000_000,
        source_node_id="node-b",
        label="coyote",
        reporting_modality="localized",
        branch_details={"label": "coyote", "confidence": 0.7},
    )
    assert decision.action == "insert"


async def test_omni_then_localized_same_node_upgrades(storage: Storage) -> None:
    policy = _policy(storage, omni_suppression_scope="site")
    window_start_ns, window_end_ns = policy.report_window_bounds(EVENT_TIME_NS)
    omni = DetectionEvent(
        id="det-omni",
        timestamp_ns=EVENT_TIME_NS,
        source_node_id="node-a",
        report_window_start_ns=window_start_ns,
        report_window_end_ns=window_end_ns,
        reporting_modality="omni",
        position_m=(0.0, 0.0, 0.0),
        confidence=0.3,
        gdop=99.0,
        label="coyote",
        label_confidence=0.6,
        reference_sensor="m0",
    )
    await storage.insert_detection(omni, snippet_path=None, snippet_expires_ns=None)

    decision = await policy.decide(
        event_time_ns=EVENT_TIME_NS + 1_000_000,
        source_node_id="node-a",
        label="coyote",
        reporting_modality="localized",
        branch_details={"label": "coyote", "confidence": 0.8},
    )
    assert decision.action == "upgrade_existing"


async def test_alias_label_suppressed_via_taxonomy_canonicalization(storage: Storage) -> None:
    taxonomy = RuntimeTaxonomyProvider(label_aliases={"canis latrans": "coyote"})
    policy = _policy(storage, omni_suppression_scope="site", taxonomy_provider=taxonomy)
    await _seed_localized(storage, policy, label="coyote")

    decision = await policy.decide(
        event_time_ns=EVENT_TIME_NS + 1_000_000,
        source_node_id="node-b",
        label="Canis latrans",
        reporting_modality="omni",
        branch_details={"label": "Canis latrans", "confidence": 0.6},
    )
    assert decision.action == "enrich_existing"
    assert decision.suppression_reason == "localized_detection_site_wide"


async def test_distance_escape_hatch_allows_far_insert(storage: Storage) -> None:
    policy = _policy(
        storage,
        omni_suppression_scope="site",
        omni_suppression_max_distance_m=50.0,
    )
    await _seed_localized(storage, policy, position=(500.0, 0.0, 0.0))

    far = await policy.decide(
        event_time_ns=EVENT_TIME_NS + 1_000_000,
        source_node_id="node-b",
        label="coyote",
        reporting_modality="omni",
        branch_details={"label": "coyote", "confidence": 0.6},
        source_position_m=(0.0, 0.0, 0.0),
    )
    assert far.action == "insert"

    near = await policy.decide(
        event_time_ns=EVENT_TIME_NS + 1_000_000,
        source_node_id="node-b",
        label="coyote",
        reporting_modality="omni",
        branch_details={"label": "coyote", "confidence": 0.6},
        source_position_m=(490.0, 0.0, 0.0),
    )
    assert near.action == "enrich_existing"
