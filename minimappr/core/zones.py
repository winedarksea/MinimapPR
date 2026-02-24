"""Zone geometry helpers for filtering and exclusion scaffolding."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from minimappr.storage.db import Storage


def _point_in_polygon(lat: float, lon: float, polygon: list[list[float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        yi, xi = float(polygon[i][0]), float(polygon[i][1])
        yj, xj = float(polygon[j][0]), float(polygon[j][1])
        intersects = ((xi > lon) != (xj > lon)) and (
            lat < (yj - yi) * (lon - xi) / ((xj - xi) + 1e-12) + yi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


@dataclass(slots=True)
class _ZoneShape:
    zone_id: str
    zone_type: str
    polygon_geo: list[list[float]]
    properties: dict[str, Any]


class ZoneMatcher:
    def __init__(self, storage: Storage, refresh_interval_seconds: float = 5.0) -> None:
        self._storage = storage
        self._refresh_interval_seconds = max(1.0, refresh_interval_seconds)
        self._last_refresh_ns = 0
        self._zones: list[_ZoneShape] = []

    async def refresh_if_due(self, now_ns: int | None = None) -> None:
        now = now_ns if now_ns is not None else time.time_ns()
        if now - self._last_refresh_ns < int(self._refresh_interval_seconds * 1_000_000_000):
            return
        rows = await self._storage.list_zones()
        zones: list[_ZoneShape] = []
        for row in rows:
            polygon = row.get("polygon_geo", [])
            if isinstance(polygon, list) and len(polygon) >= 3:
                zones.append(
                    _ZoneShape(
                        zone_id=row["id"],
                        zone_type=str(row.get("zone_type") or ""),
                        polygon_geo=polygon,
                        properties=row.get("properties") if isinstance(row.get("properties"), dict) else {},
                    )
                )
        self._zones = zones
        self._last_refresh_ns = now

    async def match_geo_point(self, lat: float, lon: float, now_ns: int | None = None) -> list[str]:
        await self.refresh_if_due(now_ns=now_ns)
        hits: list[str] = []
        for zone in self._zones:
            if _point_in_polygon(lat=lat, lon=lon, polygon=zone.polygon_geo):
                hits.append(zone.zone_id)
        return hits

    async def evaluate_detection_policies(
        self,
        *,
        zone_ids: list[str],
        label: str,
        label_category: str,
        now_ns: int | None = None,
    ) -> tuple[bool, list[str]]:
        await self.refresh_if_due(now_ns=now_ns)
        label_key = label.strip().lower()
        category_key = label_category.strip().lower()
        zone_set = {zone_id.strip().lower() for zone_id in zone_ids}
        suppress = False
        reasons: list[str] = []

        for zone in self._zones:
            props = zone.properties
            zid = zone.zone_id.strip().lower()
            matched = zid in zone_set

            suppress_labels = _normalize(props.get("suppress_labels"))
            suppress_categories = _normalize(props.get("suppress_label_categories"))
            only_labels = _normalize(props.get("only_in_zone_labels"))
            only_categories = _normalize(props.get("only_in_zone_label_categories"))

            if matched:
                if zone.zone_type == "exclusion_zone":
                    if not suppress_labels and not suppress_categories:
                        suppress = True
                        reasons.append(f"suppressed_by_exclusion_zone:{zone.zone_id}")
                    elif label_key in suppress_labels or category_key in suppress_categories:
                        suppress = True
                        reasons.append(f"suppressed_by_zone_rule:{zone.zone_id}")
                if label_key in suppress_labels or category_key in suppress_categories:
                    suppress = True
                    reasons.append(f"suppressed_by_zone_rule:{zone.zone_id}")
            else:
                if label_key in only_labels or category_key in only_categories:
                    suppress = True
                    reasons.append(f"outside_required_zone:{zone.zone_id}")

        return suppress, sorted(set(reasons))


def _normalize(value: Any) -> set[str]:
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
