"""Coordinate-frame identity shared by the geometry push and the solve handoff."""

# shared with P2; integrator reconciles

from __future__ import annotations

from minimappr.models import GeoPoint

# 1e-7 deg is ~1 cm and 1e-2 m is 1 cm: finer than any origin an operator or a
# GPS anchor produces, coarse enough that float noise from a JSON round trip
# never changes the id.
_FRAME_ID_LAT_LON_DECIMALS = 7
_FRAME_ID_ALT_DECIMALS = 2


def site_frame_id_for(origin: GeoPoint, projection: str) -> str:
    """Deterministic id of a site ENU frame: equal ids mean equal geometry.

    Every process computes the same id from the same origin and projection, so
    a solve whose echoed ``frame_id`` matches was computed in this frame.
    """
    return (
        f"site:{projection}:"
        f"{origin.lat:.{_FRAME_ID_LAT_LON_DECIMALS}f}:"
        f"{origin.lon:.{_FRAME_ID_LAT_LON_DECIMALS}f}:"
        f"{origin.alt_m:.{_FRAME_ID_ALT_DECIMALS}f}"
    )
