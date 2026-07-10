"""Regression: a persisted out-of-envelope geo must not 500 the read path.

An older/corrupt local->geographic conversion could have written an altitude
outside the GeoPoint envelope (e.g. 171 km). The detection read path builds a
GeoPoint from the row, which would raise a pydantic ValidationError and crash
the endpoint. _row_to_geo now drops such geos instead of raising.
"""

from __future__ import annotations

from minimappr.storage.db import Storage


def test_row_to_geo_drops_out_of_envelope_altitude() -> None:
    row = {"lat": 44.9, "lon": -93.2, "alt": 171315.0355507712}
    assert Storage._row_to_geo(row) is None


def test_row_to_geo_drops_out_of_range_latitude() -> None:
    row = {"lat": 421.0, "lon": -93.2, "alt": 0.0}
    assert Storage._row_to_geo(row) is None


def test_row_to_geo_accepts_valid_geo() -> None:
    row = {"lat": 44.9, "lon": -93.2, "alt": 280.0}
    geo = Storage._row_to_geo(row)
    assert geo is not None
    assert geo.lat == 44.9
    assert geo.lon == -93.2
    assert geo.alt_m == 280.0


def test_row_to_geo_returns_none_when_latlon_missing() -> None:
    assert Storage._row_to_geo({"lat": None, "lon": -93.2, "alt": 0.0}) is None
    assert Storage._row_to_geo({"lat": 44.9, "lon": None, "alt": 0.0}) is None
