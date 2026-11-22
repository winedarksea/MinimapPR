"""Field Mode W3-entity: entity-keyed reporting decision (replaces fixed buckets).

Same contract as ``ReportingFusionPolicy.decide`` (a ``ReportingDecision`` the
fusion node already knows how to act on), but keyed on the bird entity and its
sliding absorption window instead of ``(node, label, floor(t / window))``:

- no matching entity -> ``insert`` (assembly builds a row; the post-assembly
  observe creates the entity);
- an *unpositioned* bout matching an entity -> ``enrich_existing`` on the
  entity's canonical row: absorbed, no new row, no rules, the entity pulses;
- a *positioned* bout matching an entity -> ``upgrade_existing`` on the
  canonical row: assembly still runs (the tracker gets the fix) but the row is
  updated in place instead of duplicated.

The row's report window is ``[t, t + absorption_window)`` of the inserting
detection, so the canonical-row unique index never collides on a bucket edge.
A sub-display (two-threshold) detection is never absorbed here; it takes the
plain insert path where the existing hidden-track rules apply.
"""

from __future__ import annotations

from typing import Any, Callable

from minimappr.core.bird_entity_absorption_geometry import should_absorb_as_bearing_evidence
from minimappr.core.bird_entity_model import Vec3
from minimappr.core.bird_entity_registry import BirdEntityChange, BirdEntityRegistry
from minimappr.core.reporting_fusion import ReportingDecision, ReportingFusionPolicy

NANOSECONDS_PER_SECOND = 1_000_000_000
UNPOSITIONED_REPORTING_MODALITY = "omni"


class EntityReportingPolicy:
    def __init__(
        self,
        *,
        storage: Any,
        registry: BirdEntityRegistry,
        canonical_label: Callable[[str], str] | None = None,
    ) -> None:
        self._storage = storage
        self._registry = registry
        self._canonical_label = canonical_label or (lambda label: label)
        self._absorption_window_ns = int(registry.config.absorption_window_seconds * NANOSECONDS_PER_SECOND)
        self.absorbed_unlocalized_count = 0
        self.upgraded_existing_count = 0
        self.absorbed_bearing_evidence_count = 0

    async def decide(
        self,
        *,
        event_time_ns: int,
        source_node_id: str | None,
        reporting_node_position_m: Vec3 | None,
        label: str,
        reporting_modality: str,
        spatial_display_mode: str,
        branch_details: dict[str, Any],
        localization_position_m: Vec3 | None,
        below_display_threshold: bool,
        localization_position_covariance_m2: list[list[float]] | None = None,
        localization_range_projection_mode: str | None = None,
    ) -> tuple[ReportingDecision, BirdEntityChange | None]:
        window_start_ns = int(event_time_ns)
        window_end_ns = window_start_ns + self._absorption_window_ns
        modality = "omni" if reporting_modality == UNPOSITIONED_REPORTING_MODALITY else "localized"
        insert = ReportingDecision(
            action="insert",
            report_window_start_ns=window_start_ns,
            report_window_end_ns=window_end_ns,
            reporting_modality=modality,
            branch_evidence={modality: dict(branch_details)},
        )
        if below_display_threshold:
            return insert, None
        positioned = modality == "localized"
        entity = self._registry.find_absorbing_entity(
            label=self._canonical_label(label),
            event_time_ns=event_time_ns,
            spatial_display_mode=spatial_display_mode if positioned else "node_only",
            reporting_node_id=source_node_id,
            reporting_node_position_m=reporting_node_position_m,
            detection_position_m=localization_position_m if positioned else None,
            detection_covariance_m2=localization_position_covariance_m2 if positioned else None,
        )
        if entity is None:
            return insert, None
        # Field Mode S-fixes tracking: a fix too uncertain to pinpoint anything
        # (e.g. a far bird's unobservable range) joins a better-located entity as
        # bearing evidence: absorbed like an unpositioned bout, so it neither
        # starts a far-off track nor rewrites the entity's row.
        bearing_evidence = positioned and should_absorb_as_bearing_evidence(
            entity, localization_position_covariance_m2, localization_range_projection_mode, self._registry.config
        )
        insert.entity_id = entity.entity_id
        existing = None
        if entity.canonical_detection_id:
            existing = await self._storage.get_detection(entity.canonical_detection_id)
        if existing is None:
            # Canonical row gone (cleanup) or not yet written: a new row, same entity.
            return insert, None
        merged_branch_evidence = ReportingFusionPolicy._merged_branch_evidence(
            existing_detection=existing,
            reporting_modality=modality,
            branch_details={**branch_details, "entity_id": entity.entity_id},
        )
        existing_window_start_ns = existing.get("report_window_start_ns") or window_start_ns
        existing_window_end_ns = existing.get("report_window_end_ns") or window_end_ns
        if not positioned or bearing_evidence:
            self.absorbed_bearing_evidence_count += int(bearing_evidence)
            change = self._registry.absorb_unlocalized(
                entity.entity_id,
                event_time_ns=event_time_ns,
                label=self._canonical_label(label),
                confidence=float(branch_details.get("confidence") or 0.0),
                reporting_node_id=source_node_id,
            )
            self.absorbed_unlocalized_count += 1
            existing_modality = str(existing.get("reporting_modality") or "localized")
            return (
                ReportingDecision(
                    action="enrich_existing",
                    report_window_start_ns=int(existing_window_start_ns),
                    report_window_end_ns=int(existing_window_end_ns),
                    reporting_modality="localized" if existing_modality == "localized" else "omni",
                    branch_evidence=merged_branch_evidence,
                    existing_detection=existing,
                    suppression_reason="absorbed_into_entity",
                    entity_id=entity.entity_id,
                ),
                change,
            )
        self.upgraded_existing_count += 1
        return (
            ReportingDecision(
                action="upgrade_existing",
                report_window_start_ns=int(existing_window_start_ns),
                report_window_end_ns=int(existing_window_end_ns),
                reporting_modality="localized",
                branch_evidence=merged_branch_evidence,
                existing_detection=existing,
                entity_id=entity.entity_id,
            ),
            None,
        )

