"""Zone geometry helpers for filtering and exclusion scaffolding."""

from __future__ import annotations

import time
from dataclasses import dataclass

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
    polygon_geo: list[list[float]]


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
                zones.append(_ZoneShape(zone_id=row["id"], polygon_geo=polygon))
        self._zones = zones
        self._last_refresh_ns = now

    async def match_geo_point(self, lat: float, lon: float, now_ns: int | None = None) -> list[str]:
        await self.refresh_if_due(now_ns=now_ns)
        hits: list[str] = []
        for zone in self._zones:
            if _point_in_polygon(lat=lat, lon=lon, polygon=zone.polygon_geo):
                hits.append(zone.zone_id)
        return hits
