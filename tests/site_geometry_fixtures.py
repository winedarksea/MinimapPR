"""Shared fakes for the sidecar site-geometry tests (P3)."""

from __future__ import annotations

from typing import Any

from minimappr.models import GeoPoint
from minimappr.sidecar_json_http import SidecarRequestError
from minimappr.sidecar_site_geometry_publisher import AppliedSiteGeometry

SITE_ORIGIN = GeoPoint(lat=44.98698840878797, lon=-93.2579197515542, alt_m=285.75)


def node_row(
    node_id: str,
    *,
    overrides: dict | None = None,
    orientation: dict | None = None,
    mobility: str = "stationary",
    position_filter: str | None = None,
    reported_position_m: list[float] | None = None,
) -> dict[str, Any]:
    """A node row shaped like ``Storage._row_to_node`` (effective fields resolved)."""
    overrides = dict(overrides or {})
    effective_mobility = overrides.get("mobility", mobility)
    requested_filter = overrides.get("position_filter", position_filter)
    return {
        "id": node_id,
        "position_m": overrides.get("position_m", reported_position_m or [0.0, 0.0, 0.0]),
        "mobility": effective_mobility,
        "effective_position_filter": requested_filter or ("kalman" if effective_mobility == "mobile" else "kde"),
        "orientation": overrides.get("orientation", orientation or {}),
        "overrides": overrides,
    }


def estimator_checkpoint_row(estimate_geo: GeoPoint | None, *, horizontal_std_m: float | None) -> dict[str, Any]:
    """A ``Storage.list_node_position_estimator_states`` value (geodetic checkpoint)."""
    return {
        "version": 1,
        "updated_ns": 1,
        "state": {
            "format": "geodetic_v1",
            "samples": [],
            "seen_count": 64,
            "estimate": (
                [estimate_geo.lat, estimate_geo.lon, estimate_geo.alt_m] if estimate_geo is not None else None
            ),
            "horizontal_std_m": horizontal_std_m,
            "last_evaluated_seen_count": 64,
        },
    }


class FakeSiteGeometryStorage:
    def __init__(self) -> None:
        self.node_rows: list[dict[str, Any]] = []
        self.estimator_states: dict[str, dict[str, Any]] = {}

    async def list_nodes(self, limit: int | None = None) -> list[dict]:
        return [dict(row) for row in self.node_rows]

    async def list_node_position_estimator_states(self) -> dict[str, dict[str, Any]]:
        return dict(self.estimator_states)


class FakeSidecarGeometryEndpoint:
    """Behaves like the sidecar's geometry API: applies pushes, reports the applied revision."""

    def __init__(self) -> None:
        self.pushed_bodies: list[dict] = []
        self.rejected_bodies: list[dict] = []
        self.applied: AppliedSiteGeometry | None = AppliedSiteGeometry(revision=None, site_frame_id=None)
        self.push_error: SidecarRequestError | None = None
        self.fetch_error: SidecarRequestError | None = None
        self.acknowledge_revision_override: int | None = None
        self.fetch_count = 0

    async def push_site_geometry(self, body: dict[str, object]) -> dict[str, Any]:
        if self.push_error is not None:
            raise self.push_error
        revision = int(body["geometry_push_revision"])
        applied_revision = self.applied.revision if self.applied is not None else None
        if applied_revision is not None and revision < applied_revision:
            # Real sidecar rule: lower revisions are stale, equal is an idempotent replace.
            self.rejected_bodies.append(body)
            raise SidecarRequestError(
                "/api/v1/dsp/geometry returned HTTP 409",
                status_code=409,
                response_body={
                    "error": "stale_geometry_push_revision",
                    "detail": "revision lower than applied",
                    "applied_geometry_push_revision": applied_revision,
                },
            )
        self.pushed_bodies.append(body)
        if self.acknowledge_revision_override is not None:
            revision = self.acknowledge_revision_override
        self.applied = AppliedSiteGeometry(revision=revision, site_frame_id=str(body["site_frame_id"]))
        return {
            "site_frame_id": body["site_frame_id"],
            "applied_geometry_push_revision": revision,
            "node_placement_count": len(body["node_placements"]),
        }

    async def fetch_applied_site_geometry(self) -> AppliedSiteGeometry | None:
        self.fetch_count += 1
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.applied

    def simulate_restart_without_geometry(self) -> None:
        self.applied = AppliedSiteGeometry(revision=None, site_frame_id=None)
