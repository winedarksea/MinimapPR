"""Capability tier derivation for graceful degradation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CapabilityDegradationModel:
    min_sensors_for_3d: int = 4
    min_sensors_for_2d: int = 3

    def tier_for_sensor_count(self, sensor_count: int) -> str:
        if sensor_count >= self.min_sensors_for_3d:
            return "full_3d"
        if sensor_count >= self.min_sensors_for_2d:
            return "2d"
        if sensor_count >= 1:
            return "classification_only"
        return "alerting_only"

