"""Site-origin resolution helpers shared by startup and geo defaults.

The site origin is the reference point for every ENU coordinate in the system:
node `position_m`, track positions, zone geometry. Two rules follow from that.

1. It must be *shared*. Each process resolving it independently produced a
   split brain in which the api and ingest processes disagreed by 113 km, so
   every localization solve was rejected by the sanity backstop and no
   detection was ever emitted. The resolved origin is therefore persisted to
   the database and read back by every process.

2. It must be *stable*. Stored coordinates are not re-projected when the origin
   moves, so a drifting origin silently reinterprets historical data. The origin
   is anchored once, from the first trusted GPS fix, and then held until
   explicitly reset.

The configured `site_origin_lat`/`lon` are only a fallback for sites with no
usable GPS. They are never persisted, so a site running on the fallback stays
un-anchored and adopts real GPS as soon as a node reports a trusted fix.
"""

from __future__ import annotations

from dataclasses import dataclass

from minimappr.config import Settings
from minimappr.models import GeoPoint

# Resolution sources, most to least authoritative.
SOURCE_MANUAL = "manual_config"
SOURCE_PERSISTED = "persisted"
SOURCE_GPS_ANCHOR = "gps_anchor"
SOURCE_ACTIVE_NODE_MIDPOINT = "active_node_midpoint"
SOURCE_CONFIGURED_FALLBACK = "configured_fallback"

#: Sources that represent a real survey of the site rather than a placeholder.
#: Only these are persisted, and only these stop further re-anchoring.
ANCHORED_SOURCES = frozenset({SOURCE_MANUAL, SOURCE_PERSISTED, SOURCE_GPS_ANCHOR})


@dataclass(slots=True)
class SiteOriginResolution:
    origin: GeoPoint
    source: str
    contributing_node_ids: tuple[str, ...] = ()

    @property
    def is_anchored(self) -> bool:
        """True when the origin reflects a real fix and should not be re-derived."""
        return self.source in ANCHORED_SOURCES


def node_has_trusted_gps_position(metadata: object) -> bool:
    """Whether a node's metadata reports a position from a real GNSS fix.

    Mirrors the gate in `IngestProcessor._normalize_node_spec`: a `position_source`
    beginning with `gps`, excluding the explicit `gps_fallback` sentinel that nodes
    report when echoing a configured position rather than a solved one.

    A 2D fix qualifies. It solves latitude and longitude, which is all the origin
    needs; altitude is handled separately by the ingest altitude gate, and a
    slightly wrong origin altitude costs far less than refusing to anchor at all.
    """
    if not isinstance(metadata, dict):
        return False
    gps_metadata = metadata.get("gps")
    if isinstance(gps_metadata, dict):
        position_source = gps_metadata.get("position_source")
    else:
        position_source = metadata.get("position_source")
    return (
        isinstance(position_source, str)
        and position_source.startswith("gps")
        and position_source != "gps_fallback"
    )


def _node_reports_3d_fix(metadata: object) -> bool:
    """Whether a node's GPS metadata reports a genuine 3D fix.

    Mirrors the altitude gate in `IngestProcessor._normalize_node_spec`: only a
    3D fix carries a solved altitude — receivers emit a GGA altitude even under
    a GSA 2D fix.
    """
    if not isinstance(metadata, dict):
        return False
    gps_metadata = metadata.get("gps")
    signal = gps_metadata.get("signal") if isinstance(gps_metadata, dict) else None
    return isinstance(signal, str) and signal.strip().lower() == "fix_3d"


def _geo_from_node(node: dict) -> GeoPoint | None:
    geo_payload = node.get("position_geo")
    if not isinstance(geo_payload, dict):
        return None
    try:
        return GeoPoint(
            lat=float(geo_payload["lat"]),
            lon=float(geo_payload["lon"]),
            alt_m=float(geo_payload.get("alt_m") or 0.0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _midpoint(points: list[tuple[str, GeoPoint]]) -> tuple[GeoPoint, tuple[str, ...]]:
    count = float(len(points))
    return (
        GeoPoint(
            lat=sum(point.lat for _, point in points) / count,
            lon=sum(point.lon for _, point in points) / count,
            alt_m=sum(point.alt_m for _, point in points) / count,
        ),
        tuple(node_id for node_id, _ in points if node_id),
    )


def resolve_site_origin_from_nodes(
    settings: Settings,
    *,
    nodes: list[dict],
    now_ns: int,
) -> SiteOriginResolution:
    """Resolve an origin from the current node set.

    Prefers the midpoint of nodes reporting a trusted GPS fix — that result is
    anchorable and gets persisted. Falls back to the midpoint of any node with a
    position (un-anchored, e.g. nodes running on a configured position), and
    finally to the configured fallback.
    """
    configured_origin = GeoPoint(
        lat=settings.site_origin_lat,
        lon=settings.site_origin_lon,
        alt_m=settings.site_origin_alt_m,
    )
    if settings.site_origin_source == "manual":
        return SiteOriginResolution(origin=configured_origin, source=SOURCE_MANUAL)

    gps_points: list[tuple[str, GeoPoint]] = []
    any_points: list[tuple[str, GeoPoint]] = []
    stale_cutoff_ns = now_ns - int(settings.node_offline_after_seconds * 1_000_000_000)
    for node in nodes:
        point = _geo_from_node(node)
        if point is None:
            continue
        last_seen_ns = node.get("last_seen_ns")
        if last_seen_ns is None or int(last_seen_ns) < stale_cutoff_ns:
            continue
        entry = (str(node.get("id") or ""), point)
        any_points.append(entry)
        if node_has_trusted_gps_position(node.get("metadata")):
            gps_points.append(entry)

    if gps_points:
        origin, contributing = _midpoint(gps_points)
        # A 2D fix may anchor lat/lon, but its alt_m is not a measurement. When
        # any contributor has a real 3D fix, take the origin altitude from those
        # nodes only, so 2D-fix GGA altitudes cannot skew the site datum.
        three_d_alts = [
            point.alt_m
            for node in nodes
            if (point := _geo_from_node(node)) is not None
            and node_has_trusted_gps_position(node.get("metadata"))
            and _node_reports_3d_fix(node.get("metadata"))
            and str(node.get("id") or "") in set(contributing)
        ]
        if three_d_alts:
            origin = GeoPoint(
                lat=origin.lat,
                lon=origin.lon,
                alt_m=sum(three_d_alts) / float(len(three_d_alts)),
            )
        return SiteOriginResolution(
            origin=origin, source=SOURCE_GPS_ANCHOR, contributing_node_ids=contributing
        )
    if any_points:
        origin, contributing = _midpoint(any_points)
        return SiteOriginResolution(
            origin=origin,
            source=SOURCE_ACTIVE_NODE_MIDPOINT,
            contributing_node_ids=contributing,
        )
    return SiteOriginResolution(origin=configured_origin, source=SOURCE_CONFIGURED_FALLBACK)


async def load_or_resolve_site_origin(
    settings: Settings,
    *,
    storage,
    now_ns: int,
) -> SiteOriginResolution:
    """Resolve the origin every process should boot with.

    A persisted origin always wins, so all processes and restarts agree. An
    explicit `manual` configuration overrides and re-persists, letting an
    operator correct a bad anchor without hand-editing the database.
    """
    if settings.site_origin_source == "manual":
        resolution = SiteOriginResolution(
            origin=GeoPoint(
                lat=settings.site_origin_lat,
                lon=settings.site_origin_lon,
                alt_m=settings.site_origin_alt_m,
            ),
            source=SOURCE_MANUAL,
        )
        await persist_site_origin(storage, resolution)
        return resolution

    persisted = await storage.get_site_origin()
    if persisted is not None:
        return SiteOriginResolution(
            origin=GeoPoint(
                lat=float(persisted["lat"]),
                lon=float(persisted["lon"]),
                alt_m=float(persisted["alt_m"]),
            ),
            source=SOURCE_PERSISTED,
            contributing_node_ids=tuple(persisted.get("contributing_node_ids") or ()),
        )

    resolution = resolve_site_origin_from_nodes(
        settings, nodes=await storage.list_nodes(limit=4096), now_ns=now_ns
    )
    if resolution.is_anchored:
        await persist_site_origin(storage, resolution)
    return resolution


async def persist_site_origin(storage, resolution: SiteOriginResolution) -> None:
    await storage.upsert_site_origin(
        lat=resolution.origin.lat,
        lon=resolution.origin.lon,
        alt_m=resolution.origin.alt_m,
        source=resolution.source,
        contributing_node_ids=list(resolution.contributing_node_ids),
    )


def origins_differ(a: GeoPoint, b: GeoPoint, *, tolerance_deg: float = 1e-9) -> bool:
    return (
        abs(a.lat - b.lat) > tolerance_deg
        or abs(a.lon - b.lon) > tolerance_deg
        or abs(a.alt_m - b.alt_m) > 1e-6
    )
