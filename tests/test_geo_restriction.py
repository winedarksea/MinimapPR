import pytest

from minimappr.core.geo_restriction import excludes_audio_ingest
from minimappr.models import GeoPoint


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [
        (55.7558, 37.6173),
        (43.1155, 131.8855),
        (59.9343, 30.3351),
        (35.6892, 51.3890),
    ],
)
def test_restricted_regions_exclude_audio_ingest(latitude: float, longitude: float) -> None:
    assert excludes_audio_ingest(GeoPoint(lat=latitude, lon=longitude, alt_m=0.0))


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [
        (50.4501, 30.5234),
        (48.5740, 39.3078),
        (48.5, 40.2),
        (52.5, 40.5),
    ],
)
def test_ukraine_and_strict_boundaries_remain_allowed(latitude: float, longitude: float) -> None:
    assert not excludes_audio_ingest(GeoPoint(lat=latitude, lon=longitude, alt_m=0.0))


def test_missing_position_remains_allowed() -> None:
    assert not excludes_audio_ingest(None)
