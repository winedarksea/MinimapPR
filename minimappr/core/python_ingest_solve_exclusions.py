"""Per-solve sensor exclusions on the Python-ingest path (plan item P8).

The Rust solve contract lists every excluded sensor with a reason
(``solve_provenance.excluded_sensors``). The Python-ingest solve has no
provenance block, so its exclusions ride on the localized product's
classification features and reach the stored detection's feature summary
under ``localization_excluded_sensors``, with the same reason spelling.
"""

from __future__ import annotations

from typing import Any

from minimappr.core.synchronized_window_skips import SAMPLE_RATE_MISMATCH_SKIP_REASON

LOCALIZATION_EXCLUDED_SENSORS_FEATURE_KEY = "localization_excluded_sensors"


async def python_ingest_excluded_sensor_features(
    *, buffer, registry, sensor_ids: list[str], solve_sample_rate_hz: int
) -> dict[str, Any]:
    """``{"localization_excluded_sensors": [...]}`` for sensors a solve at
    ``solve_sample_rate_hz`` left out for their rate; ``{}`` when none."""
    mismatched = await buffer.sample_rate_mismatched_sensors(sensor_ids, solve_sample_rate_hz)
    if not mismatched:
        return {}
    node_ids = await registry.sensor_node_ids(sorted(mismatched))
    return {
        LOCALIZATION_EXCLUDED_SENSORS_FEATURE_KEY: [
            {
                "sensor_id": sensor_id,
                "node_id": node_ids.get(sensor_id),
                "reason": SAMPLE_RATE_MISMATCH_SKIP_REASON,
                "sensor_sample_rate_hz": mismatch.buffered_sample_rate_hz,
                "solve_sample_rate_hz": mismatch.requested_sample_rate_hz,
            }
            for sensor_id, mismatch in sorted(mismatched.items())
        ]
    }


__all__ = ["LOCALIZATION_EXCLUDED_SENSORS_FEATURE_KEY", "python_ingest_excluded_sensor_features"]
