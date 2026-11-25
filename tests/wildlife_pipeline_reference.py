"""Python reference run of the wildlife entity pipeline for the parity gate.

Composes the real server pieces in ``FusionNode`` order for an entity-routed
detection, as ``minimappr_core::bird_entity::entity_pipeline`` does:

1. ``EntityReportingPolicy.decide`` (pre-assembly; a stand-in storage holds
   each inserted canonical row);
2. ``TrackManager.update`` for localized fixes (S3 flags on; dormant
   reacquire and fingerprints off, which the Rust tracker does not port),
   with the relocation-revert listener releasing post-hop entity fixes;
3. surfacing (unknown zero-confidence dropped) and
   ``BirdEntityRegistry.observe`` with the decision's entity, carrying the
   track's fused position as the canonical row does.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

from minimappr.config import Settings
from minimappr.core.bird_entity_absorption_geometry import forget_fixes_since
from minimappr.core.bird_entity_registry import BirdEntityObservation, BirdEntityRegistry
from minimappr.core.entity_reporting_policy import EntityReportingPolicy
from minimappr.core.tracking import TrackManager

# Wildlife stationary_hold (the S3 perched tracker) is the shipped default.
PIPELINE_SETTINGS = dict(
    tracking_species_log_odds_enabled=True,
    dormant_reacquire_enabled=False,
    association_fingerprint_weight=0.0,
    entity_reporting_enabled=True,
)
MODE_TO_MODALITY = {"localized": "localized", "bearing_only": "localized", "node_only": "omni"}


def pipeline_settings() -> Settings:
    return Settings(**PIPELINE_SETTINGS)


def entity_config():
    return pipeline_settings().entity_reporting_config()


def _numeric_id(identifier: str | None) -> int | None:
    if identifier is None:
        return None
    return int(identifier.rsplit("-", 1)[1])


class _CanonicalRowStore:
    """Holds the canonical rows inserted so far (what ``get_detection`` reads)."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def get_detection(self, detection_id: str) -> dict[str, Any] | None:
        return self.rows.get(detection_id)


def _entity_summary(entity) -> dict[str, Any]:
    return {
        "entity_id": _numeric_id(entity.entity_id),
        "state": entity.state,
        "top_label": entity.species_top_label,
        "top_probability": float(entity.species_top_probability),
        "position_m": list(entity.position_m) if entity.position_m is not None else None,
        "horizontal_uncertainty_m": entity.horizontal_uncertainty_m,
        "bearing_deg": entity.bearing.bearing_deg if entity.bearing is not None else None,
        "half_width_deg": entity.bearing.half_width_deg if entity.bearing is not None else None,
        "track_id": _numeric_id(entity.track_id),
        "detection_count": entity.detection_count,
        "absorbed_detection_count": entity.absorbed_detection_count,
        "recent_fix_count": len(entity.recent_fixes),
    }


async def _run(steps: list[dict]) -> dict[str, Any]:
    settings = pipeline_settings()
    counter = itertools.count(1)
    registry = BirdEntityRegistry(settings.entity_reporting_config(), entity_id_factory=lambda: f"ent-{next(counter)}")
    store = _CanonicalRowStore()
    policy = EntityReportingPolicy(storage=store, registry=registry)
    tracker = TrackManager(settings)
    reverts: list[list[int]] = []
    released: list[int] = []

    def on_revert(revert) -> None:
        reverts.append([_numeric_id(revert.track_id), _numeric_id(revert.split_track_id), revert.hop_time_ns])
        entity = registry.entity_for_track(revert.track_id)
        released.append(forget_fixes_since(entity, revert.hop_time_ns) if entity is not None else 0)

    tracker.add_relocation_revert_listener(on_revert)
    outputs = []
    for step in steps:
        if step["type"] == "maintain":
            visible = {track.id: track for track in await tracker.snapshot()}
            changes = registry.maintain(step["now_ns"], visible)
            outputs.append({"changes": [[c.kind, _numeric_id(c.entity.entity_id), c.entity.state] for c in changes]})
            continue
        outputs.append(await _process_detection(step, registry, policy, store, tracker, reverts, released))
    entities = sorted(registry.entities(), key=lambda entity: _numeric_id(entity.entity_id))
    tracks = await tracker.snapshot(include_hidden=True)
    return {
        "steps": outputs,
        "entities": [_entity_summary(entity) for entity in entities],
        "tracks": [
            {"track_id": _numeric_id(track.id), "status": track.status, "position_m": list(track.position_m),
             "label": track.label, "confidence": float(track.confidence), "update_count": track.update_count}
            for track in tracks
        ],
    }


async def _process_detection(step, registry, policy, store, tracker, reverts, released) -> dict[str, Any]:
    mode = step["mode"]
    position = tuple(step["position_m"]) if step.get("position_m") is not None else None
    decision, change = await policy.decide(
        event_time_ns=step["event_time_ns"], source_node_id=step.get("node_id"),
        reporting_node_position_m=tuple(step["node_position_m"]) if step.get("node_position_m") else None,
        label=step["label"], reporting_modality=MODE_TO_MODALITY[mode], spatial_display_mode=mode,
        branch_details={"confidence": step["confidence"]}, localization_position_m=position,
        below_display_threshold=False, localization_position_covariance_m2=step.get("covariance_m2"),
        localization_range_projection_mode=step.get("range_projection_mode"),
    )
    result: dict[str, Any] = {"action": decision.action, "decision_entity_id": _numeric_id(decision.entity_id),
                              "track_id": None, "track_created": False, "revert": None, "released": 0}
    if decision.action == "enrich_existing":
        result.update(entity_id=_numeric_id(change.entity.entity_id) if change else None,
                      change=change.kind if change else None, entity_state=change.entity.state if change else None)
        return result
    track = None
    if mode == "localized":
        reverts_before, tracks_before = len(reverts), {t.id for t in await tracker.snapshot(include_hidden=True)}
        track = await tracker.update(
            timestamp_ns=step["event_time_ns"], position_m=position, label=step["label"],
            confidence=step["confidence"], label_category="wildlife",
            measurement_covariance_m2=step.get("covariance_m2"),
            species_evidence={label: probability for label, probability in step["species_evidence"]},
        )
        result.update(track_id=_numeric_id(track.id), track_created=track.id not in tracks_before)
        if len(reverts) > reverts_before:
            result.update(revert=reverts[-1], released=released[-1])
    if step["label"] == "unknown" and step["confidence"] == 0.0:
        result.update(entity_id=None, change=None, entity_state=None)
        return result
    if decision.action == "insert":
        store.rows[step["detection_id"]] = {"id": step["detection_id"], "reporting_modality": MODE_TO_MODALITY[mode]}
    detection_position, detection_covariance = position, step.get("covariance_m2")
    if track is not None and registry.config.canonical_row_fused_position_enabled:
        detection_position, detection_covariance = tuple(track.position_m), track.position_covariance_m2
    observation = BirdEntityObservation(
        event_time_ns=step["event_time_ns"], label=step["label"], confidence=float(step["confidence"]),
        label_category="wildlife", spatial_display_mode=mode, reporting_node_id=step.get("node_id"),
        reporting_node_position_m=tuple(step["node_position_m"]) if step.get("node_position_m") else None,
        detection_position_m=detection_position, detection_covariance_m2=detection_covariance,
        detection_id=step["detection_id"], track=track,
    )
    observed = registry.observe(observation, preferred_entity_id=decision.entity_id)
    result.update(entity_id=_numeric_id(observed.entity.entity_id), change=observed.kind, entity_state=observed.entity.state)
    return result


def run_pipeline_case(steps: list[dict]) -> dict[str, Any]:
    return asyncio.run(_run(steps))
