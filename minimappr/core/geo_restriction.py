"""Fixed ingest admission policy for geographic restrictions."""

from minimappr.models import GeoPoint


def excludes_audio_ingest(position_geo: GeoPoint | None) -> bool:
    if position_geo is None:
        return False

    latitude = position_geo.lat
    longitude = position_geo.lon
    return (
        (latitude > 52.5 and longitude > 31.5 and longitude < 165.0)
        or (latitude > 42.0 and latitude <= 52.5 and longitude > 40.5 and longitude < 138.0)
        or (latitude > 29.5 and latitude < 36.5 and longitude > 48.5 and longitude < 54.0)
        or (latitude > 59.5 and latitude < 60.5 and longitude > 29.0 and longitude < 31.5)
    )
