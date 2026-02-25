"""Environment providers for localization calculations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

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
            metadata={"source": "static"},
        )


@dataclass(slots=True)
class EnvironmentSample:
    node_id: str
    timestamp_ns: int
    temperature_c: float | None = None
    humidity_fraction: float | None = None
    pressure_pa: float | None = None
    wind_speed_mps: float | None = None
    wind_dir_deg: float | None = None
    solar_lux: float | None = None
    location_m: tuple[float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LiveEnvironmentProvider(EnvironmentProvider):
    fallback_temperature_c: float
    fallback_humidity_fraction: float
    max_reading_age_seconds: float = 300.0
    _samples_by_node: dict[str, EnvironmentSample] = field(default_factory=dict, init=False, repr=False)

    def bootstrap(self, samples: list[dict[str, Any]]) -> None:
        for sample in samples:
            self.ingest_sample(
                node_id=str(sample["node_id"]),
                timestamp_ns=int(sample["timestamp_ns"]),
                temperature_c=_safe_float(sample.get("temperature_c")),
                humidity_fraction=_safe_float(sample.get("humidity_fraction")),
                pressure_pa=_safe_float(sample.get("pressure_pa")),
                wind_speed_mps=_safe_float(sample.get("wind_speed_mps")),
                wind_dir_deg=_safe_float(sample.get("wind_dir_deg")),
                solar_lux=_safe_float(sample.get("solar_lux")),
                location_m=_coerce_vec3(sample.get("position_m")),
                metadata=dict(sample.get("metadata") or {}),
            )

    def ingest_sample(
        self,
        *,
        node_id: str,
        timestamp_ns: int,
        temperature_c: float | None,
        humidity_fraction: float | None,
        pressure_pa: float | None,
        wind_speed_mps: float | None,
        wind_dir_deg: float | None,
        solar_lux: float | None,
        location_m: tuple[float, float, float] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        existing = self._samples_by_node.get(node_id)
        if existing is not None and timestamp_ns < existing.timestamp_ns:
            return
        self._samples_by_node[node_id] = EnvironmentSample(
            node_id=node_id,
            timestamp_ns=timestamp_ns,
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
            pressure_pa=pressure_pa,
            wind_speed_mps=wind_speed_mps,
            wind_dir_deg=wind_dir_deg,
            solar_lux=solar_lux,
            location_m=location_m,
            metadata=metadata or {},
        )

    def get_speed_of_sound(self, location_m: tuple[float, float, float] | None = None) -> float:
        conditions = self.get_conditions(location_m=location_m)
        return _speed_of_sound(conditions.temperature_c, conditions.humidity_fraction)

    def get_conditions(self, location_m: tuple[float, float, float] | None = None) -> EnvironmentReading:
        now_ns = time.time_ns()
        sample = self._select_sample(now_ns=now_ns, location_m=location_m)

        temperature_c = self.fallback_temperature_c
        humidity_fraction = _clamp_humidity(self.fallback_humidity_fraction)
        pressure_pa: float | None = None
        wind_speed_mps: float | None = None
        wind_dir_deg: float | None = None
        source = "static_fallback"
        metadata: dict[str, Any] = {}

        if sample is not None:
            if sample.temperature_c is not None:
                temperature_c = sample.temperature_c
                source = "live"
            if sample.humidity_fraction is not None:
                humidity_fraction = _clamp_humidity(sample.humidity_fraction)
            pressure_pa = sample.pressure_pa
            wind_speed_mps = sample.wind_speed_mps
            wind_dir_deg = sample.wind_dir_deg
            age_s = max(0.0, (now_ns - sample.timestamp_ns) / 1_000_000_000.0)
            metadata.update(sample.metadata)
            metadata.update(
                {
                    "source": source,
                    "node_id": sample.node_id,
                    "sample_timestamp_ns": sample.timestamp_ns,
                    "sample_age_s": age_s,
                    "temperature_available": sample.temperature_c is not None,
                    "humidity_available": sample.humidity_fraction is not None,
                }
            )
        else:
            metadata["source"] = source

        return EnvironmentReading(
            temperature_c=temperature_c,
            humidity_fraction=humidity_fraction,
            pressure_pa=pressure_pa,
            wind_speed_mps=wind_speed_mps,
            wind_dir_deg=wind_dir_deg,
            metadata=metadata,
        )

    def _select_sample(
        self,
        *,
        now_ns: int,
        location_m: tuple[float, float, float] | None,
    ) -> EnvironmentSample | None:
        if not self._samples_by_node:
            return None
        max_age_ns = int(max(0.0, self.max_reading_age_seconds) * 1_000_000_000)
        candidates = [
            sample
            for sample in self._samples_by_node.values()
            if max_age_ns == 0 or (now_ns - sample.timestamp_ns) <= max_age_ns
        ]
        if not candidates:
            return None

        with_temperature = [sample for sample in candidates if sample.temperature_c is not None]
        if location_m is None or not with_temperature:
            pool = with_temperature or candidates
            return max(pool, key=lambda sample: sample.timestamp_ns)

        located = [sample for sample in with_temperature if sample.location_m is not None]
        if not located:
            return max(with_temperature, key=lambda sample: sample.timestamp_ns)
        return min(
            located,
            key=lambda sample: (_distance_sq(location_m, sample.location_m), -sample.timestamp_ns),
        )


def _distance_sq(a: tuple[float, float, float], b: tuple[float, float, float] | None) -> float:
    if b is None:
        return float("inf")
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return (dx * dx) + (dy * dy) + (dz * dz)


def _clamp_humidity(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
