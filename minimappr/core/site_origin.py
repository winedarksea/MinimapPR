"""Site-origin resolution helpers shared by startup and geo defaults."""

from __future__ import annotations

from dataclasses import dataclass

from minimappr.config import Settings
from minimappr.models import GeoPoint


@dataclass(slots=True)
class SiteOriginResolution:
    origin: GeoPoint
    source: str
    contributing_node_ids: tuple[str, ...] = ()


def should_schedule_deferred_site_origin_reconciliation(
    settings: Settings,
    *,
    initial_resolution_source: str,
) -> bool:
    return (
        settings.site_origin_source == "auto"
        and settings.site_origin_reconcile_delay_seconds > 0.0
        and initial_resolution_source == "configured_fallback"
    )


def resolve_site_origin_from_nodes(
    settings: Settings,
    *,
    nodes: list[dict],
    now_ns: int,
) -> SiteOriginResolution:
    configured_origin = GeoPoint(
        lat=settings.site_origin_lat,
        lon=settings.site_origin_lon,
        alt_m=settings.site_origin_alt_m,
    )
    if settings.site_origin_source == "manual":
        return SiteOriginResolution(origin=configured_origin, source="manual_config")

    active_points: list[tuple[str, GeoPoint]] = []
    stale_cutoff_ns = now_ns - int(settings.node_offline_after_seconds * 1_000_000_000)
    for node in nodes:
        geo_payload = node.get("position_geo")
        if not isinstance(geo_payload, dict):
            continue
        last_seen_ns = node.get("last_seen_ns")
        if last_seen_ns is None or int(last_seen_ns) < stale_cutoff_ns:
            continue
        try:
            point = GeoPoint(
                lat=float(geo_payload["lat"]),
                lon=float(geo_payload["lon"]),
                alt_m=float(geo_payload.get("alt_m") or 0.0),
            )
        except (KeyError, TypeError, ValueError):
            continue
        active_points.append((str(node.get("id") or ""), point))

    if not active_points:
        return SiteOriginResolution(origin=configured_origin, source="configured_fallback")

    count = float(len(active_points))
    mean_lat = sum(point.lat for _, point in active_points) / count
    mean_lon = sum(point.lon for _, point in active_points) / count
    mean_alt_m = sum(point.alt_m for _, point in active_points) / count
    contributing_node_ids = tuple(node_id for node_id, _ in active_points if node_id)
    return SiteOriginResolution(
        origin=GeoPoint(lat=mean_lat, lon=mean_lon, alt_m=mean_alt_m),
        source="active_node_midpoint",
        contributing_node_ids=contributing_node_ids,
    )