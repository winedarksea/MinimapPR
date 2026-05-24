"""Local-Cartesian <-> geographic coordinate conversion utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from minimappr.models import GeoPoint, Vec3


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - (WGS84_E2 * sin_lat * sin_lat))
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.asarray([x, y, z], dtype=np.float64)


def _ecef_to_lla(ecef: np.ndarray) -> tuple[float, float, float]:
    x, y, z = float(ecef[0]), float(ecef[1]), float(ecef[2])
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))

    for _ in range(6):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - (WGS84_E2 * sin_lat * sin_lat))
        alt = (p / max(math.cos(lat), 1e-12)) - n
        lat = math.atan2(z, p * (1.0 - (WGS84_E2 * n / max(n + alt, 1e-12))))

    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - (WGS84_E2 * sin_lat * sin_lat))
    alt = (p / max(math.cos(lat), 1e-12)) - n

    return math.degrees(lat), math.degrees(lon), alt


def _enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    return np.asarray(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )


@dataclass(slots=True)
class LocalCoordinateFrame:
    """Reference frame with configurable conversion mode.

    Local frame convention:
        x = east (meters)
        y = north (meters)
        z = up (meters)
    """

    origin: GeoPoint
    mode: str = "flat"
    _origin_ecef: np.ndarray = field(init=False, repr=False)
    _ecef_to_enu: np.ndarray = field(init=False, repr=False)
    _enu_to_ecef: np.ndarray = field(init=False, repr=False)
    _meters_per_deg_lat: float = field(init=False, repr=False)
    _meters_per_deg_lon: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mode = self.mode.strip().lower()
        if self.mode not in {"flat", "geodetic"}:
            raise ValueError("mode must be either 'flat' or 'geodetic'")
        self._origin_ecef = _lla_to_ecef(self.origin.lat, self.origin.lon, self.origin.alt_m)
        self._ecef_to_enu = _enu_rotation(self.origin.lat, self.origin.lon)
        self._enu_to_ecef = self._ecef_to_enu.T

        lat_rad = math.radians(self.origin.lat)
        self._meters_per_deg_lat = 111132.92 - (559.82 * math.cos(2 * lat_rad)) + (1.175 * math.cos(4 * lat_rad))
        self._meters_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)

    def local_to_geo(self, position_m: Vec3) -> GeoPoint:
        east, north, up = float(position_m[0]), float(position_m[1]), float(position_m[2])
        if self.mode == "flat":
            lat = self.origin.lat + (north / max(self._meters_per_deg_lat, 1e-9))
            lon = self.origin.lon + (east / max(self._meters_per_deg_lon, 1e-9))
            alt_m = self.origin.alt_m + up
            return GeoPoint(lat=lat, lon=lon, alt_m=alt_m)

        enu = np.asarray([east, north, up], dtype=np.float64)
        ecef = self._origin_ecef + (self._enu_to_ecef @ enu)
        lat, lon, alt = _ecef_to_lla(ecef)
        return GeoPoint(lat=lat, lon=lon, alt_m=float(alt))

    def local_covariance_to_geo_uncertainty(
        self,
        covariance_m2: list[list[float]] | np.ndarray | None,
    ) -> dict[str, float] | None:
        if covariance_m2 is None:
            return None
        try:
            covariance = np.asarray(covariance_m2, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            return None
        covariance = 0.5 * (covariance + covariance.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance[:2, :2])
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(eigenvalues)):
            return None

        order = np.argsort(eigenvalues)[::-1]
        major_variance_m2 = max(float(eigenvalues[order[0]]), 0.0)
        minor_variance_m2 = max(float(eigenvalues[order[1]]), 0.0)
        major_axis = eigenvectors[:, order[0]]
        major_bearing_deg = math.degrees(math.atan2(float(major_axis[0]), float(major_axis[1])))
        east_std_m = math.sqrt(max(float(covariance[0, 0]), 0.0))
        north_std_m = math.sqrt(max(float(covariance[1, 1]), 0.0))
        up_std_m = math.sqrt(max(float(covariance[2, 2]), 0.0))
        return {
            "east_std_m": east_std_m,
            "north_std_m": north_std_m,
            "up_std_m": up_std_m,
            "lat_std_deg": north_std_m / max(self._meters_per_deg_lat, 1e-9),
            "lon_std_deg": east_std_m / max(self._meters_per_deg_lon, 1e-9),
            "horizontal_major_std_m": math.sqrt(major_variance_m2),
            "horizontal_minor_std_m": math.sqrt(minor_variance_m2),
            "horizontal_major_bearing_deg": ((major_bearing_deg + 180.0) % 360.0) - 180.0,
        }

    def geo_to_local(self, point: GeoPoint) -> Vec3:
        if self.mode == "flat":
            east = (point.lon - self.origin.lon) * self._meters_per_deg_lon
            north = (point.lat - self.origin.lat) * self._meters_per_deg_lat
            up = point.alt_m - self.origin.alt_m
            return float(east), float(north), float(up)

        ecef = _lla_to_ecef(point.lat, point.lon, point.alt_m)
        enu = self._ecef_to_enu @ (ecef - self._origin_ecef)
        return float(enu[0]), float(enu[1]), float(enu[2])
