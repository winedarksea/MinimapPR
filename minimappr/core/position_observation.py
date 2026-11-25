"""The position observation: one localized solve, as the tracker sees it (plan P5).

Built once per solve by the observation sink, before classification, from the
localized product both ingest paths hand to
``PositionObservationSink.observe_localized_solve``. Solve identity and origin
come from the per-solve provenance record that the provenance recorder stamped
onto the product (``extra_classification_features["localization_solve"]``);
a solve without provenance (Python ingest, legacy Rust render) is identified
by its candidate id.

The reporter (``reporting_node_id``) and the cluster (``cluster_id``) are kept
apart: a cluster solve is reported by one node but measured by several, and
two reporters in one cluster can localize two different sources at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from minimappr.api.rust_localization_solve_contract import EVENT_TRIGGERED_ATTEMPT_KINDS
from minimappr.core.localization_provenance import LOCALIZATION_SOLVE_FEATURE_KEY
from minimappr.core.localized_render_handoff_hooks import SOLVE_ORIGIN_PYTHON_INGEST
from minimappr.models import spatial_display_mode_for_detection

if TYPE_CHECKING:
    from minimappr.core.fusion_stage_carriers import LocalizedCandidate

# Same rule the classified-detection path used before P5 for "may this fix
# update a track": a localized display position at a localizable tier. Bearing
# only, node only and classification-only results feed partial binding at most.
TRACKER_LOCALIZABLE_CAPABILITY_TIERS = frozenset({"2d", "full_3d"})
TRACKER_ELIGIBLE_SPATIAL_DISPLAY_MODE = "localized"
_EVENT_TRIGGERED_ATTEMPT_KIND_VALUES = frozenset(kind.value for kind in EVENT_TRIGGERED_ATTEMPT_KINDS)


@dataclass(frozen=True, slots=True)
class PositionObservation:
    solve_id: str
    solve_origin: str
    reporting_node_id: str
    cluster_id: str | None
    # Packet-derived event time of the solve; the tracker update timestamp.
    event_time_ns: int
    # Centre of the solve window (provenance); equals event_time_ns when absent.
    window_center_time_ns: int
    position_m: tuple[float, float, float]
    position_covariance_m2: list[list[float]] | None
    reporting_modality: str
    spatial_display_mode: str
    capability_tier: str
    contributing_sensor_count: int
    # Provenance ``attempt_kind``; None when the solve carried no provenance.
    attempt_kind: str | None
    # Only event-triggered solves may start a provisional (hidden) track.
    event_triggered: bool

    @property
    def tracker_eligible(self) -> bool:
        return (
            self.spatial_display_mode == TRACKER_ELIGIBLE_SPATIAL_DISPLAY_MODE
            and self.capability_tier in TRACKER_LOCALIZABLE_CAPABILITY_TIERS
        )


def _solve_provenance_record(localized_product: "LocalizedCandidate") -> dict[str, Any] | None:
    record = localized_product.extra_classification_features.get(LOCALIZATION_SOLVE_FEATURE_KEY)
    return record if isinstance(record, dict) else None


def position_observation_solve_id(localized_product: "LocalizedCandidate") -> str:
    """Solve identity: the provenance ``solve_id`` (a covering render repeats
    its event's), else the candidate id (Python ingest, legacy renders)."""
    record = _solve_provenance_record(localized_product)
    if record is not None and isinstance(record.get("solve_id"), str) and record["solve_id"]:
        return record["solve_id"]
    return localized_product.candidate.id


def _solve_is_event_triggered(*, solve_origin: str, attempt_kind: str | None) -> bool:
    # Python-ingest candidates exist only because the ingest trigger fired
    # (``FusionNode._queue_trigger_candidate``); a Rust render says so in its
    # provenance, and a render without provenance cannot prove it.
    if attempt_kind is not None:
        return attempt_kind in _EVENT_TRIGGERED_ATTEMPT_KIND_VALUES
    return solve_origin == SOLVE_ORIGIN_PYTHON_INGEST


def position_observation_from_localized_product(
    localized_product: "LocalizedCandidate",
    *,
    reporting_modality: str,
    solve_origin: str,
) -> PositionObservation | None:
    """``None`` when the product carries no localization branch at all."""
    branch = localized_product.localization_branch
    if branch is None:
        return None
    record = _solve_provenance_record(localized_product) or {}
    attempt_kind = record.get("attempt_kind") if isinstance(record.get("attempt_kind"), str) else None
    candidate = localized_product.candidate
    window_center_time_ns = record.get("window_center_time_ns")
    reporting_node_id = record.get("reporting_node_id")
    return PositionObservation(
        solve_id=position_observation_solve_id(localized_product),
        solve_origin=solve_origin,
        reporting_node_id=reporting_node_id if isinstance(reporting_node_id, str) else candidate.source_node_id,
        cluster_id=record.get("cluster_id") if isinstance(record.get("cluster_id"), str) else None,
        event_time_ns=int(candidate.event_time_ns),
        window_center_time_ns=(
            int(window_center_time_ns) if isinstance(window_center_time_ns, int) else int(candidate.event_time_ns)
        ),
        position_m=tuple(float(value) for value in branch.localization_position_m),  # type: ignore[arg-type]
        position_covariance_m2=branch.localization_position_covariance_m2,
        reporting_modality=reporting_modality,
        spatial_display_mode=spatial_display_mode_for_detection(
            reporting_modality=reporting_modality,
            feature_summary={
                "localization_range_projection_mode": branch.localization_range_projection_mode,
                "localization_range_observability": branch.localization_range_observability,
            },
        ),
        capability_tier=branch.capability_tier,
        contributing_sensor_count=len(localized_product.selected_sensor_ids),
        attempt_kind=attempt_kind,
        event_triggered=_solve_is_event_triggered(solve_origin=solve_origin, attempt_kind=attempt_kind),
    )
