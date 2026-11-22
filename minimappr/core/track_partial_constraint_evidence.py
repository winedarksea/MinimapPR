"""Build partial spatial constraints for a detection that has no full fix.

Called from ``DetectionAssembler`` (W2-tracker, S3) only when
``tracking_partial_constraint_binding_enabled`` is set and the detection did not
reach the tracker as a localized fix. Uses whatever the detection carries:

- bearing-only solve (unobservable range) on a node: a DOA ray from that node's
  centre through the projected position;
- cross-node TDOA entries: one hyperbola per sensor on a different node from
  the reference sensor;
- otherwise (omni): the reporting node should rank loudest, i.e. be about the
  nearest node, within a loose level margin.

Sensor and node positions come from the registry snapshot, so nothing here
blocks on I/O; it runs on the fusion worker, never the audio hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from minimappr.core.localization import speed_of_sound_mps
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.track_partial_binding import (
    BearingConstraint,
    HyperbolaConstraint,
    PartialConstraint,
    SnrRankingConstraint,
)
from minimappr.spatial_audio.geometry import SPEED_OF_SOUND_MPS

# Below this offset a projected position carries no bearing.
_MINIMUM_BEARING_OFFSET_M = 1e-3


@dataclass(frozen=True, slots=True)
class PartialConstraintSigmas:
    bearing_sigma_deg: float = 10.0
    hyperbola_sigma_m: float = 2.0
    snr_level_margin_db: float = 7.0


def _sound_speed_for_environment(environment: dict[str, Any] | None) -> float:
    environment = environment or {}
    temperature_c = environment.get("temperature_c")
    if not isinstance(temperature_c, int | float) or not math.isfinite(float(temperature_c)):
        return SPEED_OF_SOUND_MPS
    humidity = environment.get("humidity_fraction", 0.0)
    humidity_fraction = float(humidity) if isinstance(humidity, int | float) else 0.0
    return speed_of_sound_mps(float(temperature_c), humidity_fraction)


async def _node_centres_m(registry: NodeRegistry) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    sensor_positions = await registry.sensor_positions()
    sensor_node_ids = await registry.sensor_node_ids()
    grouped: dict[str, list[np.ndarray]] = {}
    for sensor_id, position in sensor_positions.items():
        node_id = sensor_node_ids.get(sensor_id)
        if node_id is not None:
            grouped.setdefault(node_id, []).append(np.asarray(position, dtype=np.float64))
    centres = {node_id: np.mean(np.vstack(points), axis=0) for node_id, points in grouped.items()}
    return centres, sensor_node_ids


async def build_partial_constraints(
    *,
    registry: NodeRegistry,
    reference_sensor: str,
    source_node_id: str | None,
    spatial_display_mode: str,
    localization_position_m: tuple[float, float, float],
    tdoa_s: dict[str, float] | None,
    environment: dict[str, Any] | None,
    sigmas: PartialConstraintSigmas,
) -> list[PartialConstraint]:
    centres, sensor_node_ids = await _node_centres_m(registry)
    constraints: list[PartialConstraint] = []
    reporting_centre = centres.get(source_node_id) if source_node_id else None

    if spatial_display_mode == "bearing_only" and reporting_centre is not None:
        offset = np.asarray(localization_position_m, dtype=np.float64) - reporting_centre
        offset_norm = float(np.linalg.norm(offset))
        if offset_norm > _MINIMUM_BEARING_OFFSET_M:
            constraints.append(
                BearingConstraint(
                    origin_m=reporting_centre,
                    unit_direction=offset / offset_norm,
                    sigma_rad=math.radians(sigmas.bearing_sigma_deg),
                )
            )

    if tdoa_s:
        sensor_positions = await registry.sensor_positions()
        reference_position = sensor_positions.get(reference_sensor)
        reference_node = sensor_node_ids.get(reference_sensor)
        sound_speed = _sound_speed_for_environment(environment)
        for sensor_id, delay_s in tdoa_s.items():
            if sensor_id == reference_sensor or reference_position is None:
                continue
            # Intra-array pairs are a bearing, already covered above.
            if sensor_node_ids.get(sensor_id) == reference_node:
                continue
            sensor_position = sensor_positions.get(sensor_id)
            if sensor_position is None or not math.isfinite(float(delay_s)):
                continue
            constraints.append(
                HyperbolaConstraint(
                    sensor_m=np.asarray(sensor_position, dtype=np.float64),
                    reference_sensor_m=np.asarray(reference_position, dtype=np.float64),
                    range_difference_m=float(delay_s) * sound_speed,
                    sigma_m=sigmas.hyperbola_sigma_m,
                )
            )

    if not constraints and reporting_centre is not None:
        others = tuple(centre for node_id, centre in centres.items() if node_id != source_node_id)
        if others:
            constraints.append(
                SnrRankingConstraint(
                    reporting_node_m=reporting_centre,
                    other_node_positions_m=others,
                    level_margin_db=sigmas.snr_level_margin_db,
                )
            )
    return constraints
