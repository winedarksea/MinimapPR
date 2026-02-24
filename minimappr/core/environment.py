"""Environment providers for localization calculations."""

from __future__ import annotations

from dataclasses import dataclass

from minimappr.interfaces import EnvironmentProvider, EnvironmentReading


def _speed_of_sound(temperature_c: float, humidity_fraction: float) -> float:
    humidity_percent = max(0.0, min(1.0, humidity_fraction)) * 100.0
    return 331.3 + (0.606 * temperature_c) + (0.0124 * humidity_percent)


@dataclass(slots=True)
class StaticEnvironmentProvider(EnvironmentProvider):
    temperature_c: float
    humidity_fraction: float

    def get_speed_of_sound(self, location_m: tuple[float, float, float] | None = None) -> float:
        del location_m
        return _speed_of_sound(self.temperature_c, self.humidity_fraction)

    def get_conditions(self, location_m: tuple[float, float, float] | None = None) -> EnvironmentReading:
        del location_m
        return EnvironmentReading(
            temperature_c=self.temperature_c,
            humidity_fraction=self.humidity_fraction,
        )

