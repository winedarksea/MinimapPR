"""Typed parse of the sidecar's per-solve provenance (``localization.solve_provenance``)
and admitted ``pair_tdoas``, exactly per ``docs/localization_solve_contract.md``.

The wire shapes are owned by ``minimappr-core/src/localization_sensor_entry.rs``;
this module mirrors them field for field. A manifest without
``solve_provenance`` is a legacy manifest and parses to ``None``. A manifest
that carries one but breaks the contract (missing field, unknown enum value,
an index outside ``sensors``) raises ``SolveContractError``: a half-understood
solve must never reach geometry, because every index in it is meaningful only
against this solve's own ``sensors`` list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

SOLVE_PROVENANCE_SCHEMA_VERSION = 1
SOLVE_PROVENANCE_FIELD = "solve_provenance"


class SolveContractError(ValueError):
    """A manifest's solve provenance or pair TDOAs break the wire contract."""


class SolveAttemptKind(StrEnum):
    EVENT_TRIGGER_ABSOLUTE_RMS = "event_trigger_absolute_rms"
    EVENT_TRIGGER_NOISE_FLOOR_RELATIVE = "event_trigger_noise_floor_relative"
    SCHEDULED_CLASSIFIER_RENDER = "scheduled_classifier_render"
    NOT_ATTEMPTED = "not_attempted"


EVENT_TRIGGERED_ATTEMPT_KINDS = frozenset(
    {SolveAttemptKind.EVENT_TRIGGER_ABSOLUTE_RMS, SolveAttemptKind.EVENT_TRIGGER_NOISE_FLOOR_RELATIVE}
)


class SolveBufferScope(StrEnum):
    NODE_LOCAL = "node_local"
    EXPLICIT_CLUSTER = "explicit_cluster"
    AUTOMATIC_SITE_CLUSTER = "automatic_site_cluster"


class NodeLocalRoutingReason(StrEnum):
    NODE_ARRAY_GEOMETRY_UNUSABLE = "node_array_geometry_unusable"
    NODE_POSITION_UNTRUSTED_NO_OVERRIDE = "node_position_untrusted_no_override"
    EXPLICIT_CLUSTER_INCOMPLETE = "explicit_cluster_incomplete"
    EXPLICIT_CLUSTER_BELOW_MINIMUM_SENSORS = "explicit_cluster_below_minimum_sensors"
    NODE_CHANNELS_MISSING_FROM_EXPLICIT_CLUSTER = "node_channels_missing_from_explicit_cluster"


class CoordinateFrameKind(StrEnum):
    SITE_ENU = "site_enu"
    SIDECAR_AUTOMATIC_ENU = "sidecar_automatic_enu"
    EXPLICIT_CLUSTER_POSITIONS = "explicit_cluster_positions"
    NODE_BODY = "node_body"


class CoordinateProjection(StrEnum):
    FLAT_WGS84_SERIES = "flat_wgs84_series"
    GEODETIC_ECEF_ENU = "geodetic_ecef_enu"


class SensorPositionSource(StrEnum):
    OPERATOR_OVERRIDE = "operator_override"
    STATIONARY_ESTIMATE = "stationary_estimate"
    REPORTED_GPS = "reported_gps"
    FALLBACK_STATIC = "fallback_static"
    EXPLICIT_CLUSTER_MANIFEST = "explicit_cluster_manifest"
    NODE_BODY_OFFSET = "node_body_offset"


class NodeTimeQuality(StrEnum):
    GPS_LOCKED = "gps_locked"
    GPS_HOLDOVER = "gps_holdover"
    NTP_DISCIPLINED = "ntp_disciplined"
    FREE_RUNNING = "free_running"
    UNKNOWN = "unknown"


class SolveExclusionReason(StrEnum):
    SAMPLE_RATE_MISMATCH = "sample_rate_mismatch"
    PAIR_BASELINE_BELOW_MINIMUM = "pair_baseline_below_minimum"
    COVERAGE_BELOW_THRESHOLD = "coverage_below_threshold"
    TIMELINE_RESIDUAL_OVER_LIMIT = "timeline_residual_over_limit"
    NODE_POSITION_UNTRUSTED_NO_OVERRIDE = "node_position_untrusted_no_override"
    NODE_ARRAY_GEOMETRY_UNUSABLE = "node_array_geometry_unusable"
    MAX_SOLVE_SENSORS_EXCEEDED = "max_solve_sensors_exceeded"


@dataclass(frozen=True, slots=True)
class GeodeticOrigin:
    lat: float
    lon: float
    alt_m: float


@dataclass(frozen=True, slots=True)
class SolveCoordinateFrame:
    frame_id: str | None
    kind: CoordinateFrameKind
    origin: GeodeticOrigin | None
    projection: CoordinateProjection | None


@dataclass(frozen=True, slots=True)
class SolveSensorEntry:
    sensor_id: str
    node_id: str
    node_channel_index: int
    position_m: tuple[float, float, float]
    sample_rate_hz: int
    position_source: SensorPositionSource
    position_std_m: float | None
    clock_source: NodeTimeQuality
    clock_domain: str
    clock_offset_std_s: float | None


@dataclass(frozen=True, slots=True)
class ExcludedSensor:
    sensor_index: int
    sensor_id: str
    reason: SolveExclusionReason


@dataclass(frozen=True, slots=True)
class ExcludedSensorPair:
    sensor_index_a: int
    sensor_index_b: int
    reason: SolveExclusionReason


@dataclass(frozen=True, slots=True)
class NodeTimingEvidence:
    node_id: str
    time_quality: NodeTimeQuality
    clock_offset_std_s: float | None
    timeline_residual_ns: int | None
    last_timestamp_step_ns: int | None
    timestamp_step_count: int


@dataclass(frozen=True, slots=True)
class SolveProvenance:
    schema_version: int
    solve_id: str
    attempt_kind: SolveAttemptKind
    buffer_scope: SolveBufferScope
    cluster_id: str | None
    reporting_node_id: str
    # Why the reporter solved node-local; None for cluster solves.
    node_local_routing_reason: NodeLocalRoutingReason | None
    coordinate_frame: SolveCoordinateFrame
    layout_revision: int
    geometry_push_revision: int | None
    sensors: tuple[SolveSensorEntry, ...]
    active_sensor_indices: tuple[int, ...]
    excluded_sensors: tuple[ExcludedSensor, ...]
    excluded_pairs: tuple[ExcludedSensorPair, ...]
    node_timing: tuple[NodeTimingEvidence, ...]
    window_center_time_ns: int
    window_seconds: float
    solve_sample_rate_hz: int

    @property
    def ordered_sensor_ids(self) -> list[str]:
        return [sensor.sensor_id for sensor in self.sensors]


@dataclass(frozen=True, slots=True)
class AdmittedPairTdoa:
    """One admitted pair; ``ch_a``/``ch_b`` index ``SolveProvenance.sensors``.

    ``lag_seconds`` is arrival at ``ch_a`` minus arrival at ``ch_b``.
    """

    ch_a: int
    ch_b: int
    lag_seconds: float
    confidence: float
    delay_samples: float | None
    is_cross_node: bool | None
    timing_uncertainty_seconds: float | None


@dataclass(frozen=True, slots=True)
class RustLocalizationSolve:
    """A solve's provenance together with the pairs it admitted."""

    provenance: SolveProvenance
    admitted_pair_tdoas: tuple[AdmittedPairTdoa, ...]

    def contributing_sensor_indices(self) -> list[int]:
        """Sensors whose audio entered an admitted pair, in solve order.

        A solve that admitted no pairs (e.g. not attempted) reports its active
        sensors instead, so the record still names the audio it considered.
        """
        if not self.admitted_pair_tdoas:
            return sorted(set(self.provenance.active_sensor_indices))
        return sorted({pair.ch_a for pair in self.admitted_pair_tdoas} | {pair.ch_b for pair in self.admitted_pair_tdoas})


def parse_rust_localization_solve(localization_payload: Any) -> RustLocalizationSolve | None:
    """Parse ``localization.solve_provenance`` + ``pair_tdoas``; ``None`` for a legacy manifest."""
    if not isinstance(localization_payload, dict) or SOLVE_PROVENANCE_FIELD not in localization_payload:
        return None
    provenance = parse_solve_provenance(localization_payload[SOLVE_PROVENANCE_FIELD])
    admitted_pair_tdoas = parse_admitted_pair_tdoas(
        localization_payload.get("pair_tdoas") or [],
        sensor_count=len(provenance.sensors),
    )
    return RustLocalizationSolve(provenance=provenance, admitted_pair_tdoas=admitted_pair_tdoas)


def parse_solve_provenance(payload: Any) -> SolveProvenance:
    mapping = _require_mapping(payload, "solve_provenance")
    schema_version = _require_int(mapping, "schema_version")
    if schema_version != SOLVE_PROVENANCE_SCHEMA_VERSION:
        raise SolveContractError(
            f"solve_provenance.schema_version {schema_version} is not supported "
            f"(expected {SOLVE_PROVENANCE_SCHEMA_VERSION})"
        )
    sensors = tuple(
        _parse_sensor_entry(entry, f"sensors[{index}]")
        for index, entry in enumerate(_require_list(mapping, "sensors"))
    )
    sensor_count = len(sensors)
    if len({sensor.sensor_id for sensor in sensors}) != sensor_count:
        raise SolveContractError("solve_provenance.sensors repeats a sensor_id")
    active_sensor_indices = tuple(
        _require_sensor_index(value, sensor_count, f"active_sensor_indices[{position}]")
        for position, value in enumerate(_require_list(mapping, "active_sensor_indices"))
    )
    return SolveProvenance(
        schema_version=schema_version,
        solve_id=_require_nonempty_string(mapping, "solve_id"),
        attempt_kind=_require_enum(mapping, "attempt_kind", SolveAttemptKind),
        buffer_scope=_require_enum(mapping, "buffer_scope", SolveBufferScope),
        cluster_id=_optional_string(mapping, "cluster_id"),
        reporting_node_id=_require_nonempty_string(mapping, "reporting_node_id"),
        node_local_routing_reason=(
            None
            if _require_key(mapping, "node_local_routing_reason") is None
            else _require_enum(mapping, "node_local_routing_reason", NodeLocalRoutingReason)
        ),
        coordinate_frame=_parse_coordinate_frame(_require_key(mapping, "coordinate_frame")),
        layout_revision=_require_int(mapping, "layout_revision"),
        geometry_push_revision=_optional_int(mapping, "geometry_push_revision"),
        sensors=sensors,
        active_sensor_indices=active_sensor_indices,
        excluded_sensors=tuple(
            _parse_excluded_sensor(entry, sensor_count, f"excluded_sensors[{index}]")
            for index, entry in enumerate(_require_list(mapping, "excluded_sensors"))
        ),
        excluded_pairs=tuple(
            _parse_excluded_pair(entry, sensor_count, f"excluded_pairs[{index}]")
            for index, entry in enumerate(_require_list(mapping, "excluded_pairs"))
        ),
        node_timing=tuple(
            _parse_node_timing(entry, f"node_timing[{index}]")
            for index, entry in enumerate(_require_list(mapping, "node_timing"))
        ),
        window_center_time_ns=_require_int(mapping, "window_center_time_ns"),
        window_seconds=_require_finite_float(mapping, "window_seconds"),
        solve_sample_rate_hz=_require_int(mapping, "solve_sample_rate_hz"),
    )


def parse_admitted_pair_tdoas(payload: Any, *, sensor_count: int) -> tuple[AdmittedPairTdoa, ...]:
    if not isinstance(payload, list):
        raise SolveContractError("localization.pair_tdoas must be a list")
    pairs: list[AdmittedPairTdoa] = []
    for index, entry in enumerate(payload):
        context = f"pair_tdoas[{index}]"
        mapping = _require_mapping(entry, context)
        ch_a = _require_sensor_index(_require_key(mapping, "ch_a", context), sensor_count, f"{context}.ch_a")
        ch_b = _require_sensor_index(_require_key(mapping, "ch_b", context), sensor_count, f"{context}.ch_b")
        if ch_a == ch_b:
            raise SolveContractError(f"{context} pairs sensor {ch_a} with itself")
        is_cross_node = mapping.get("is_cross_node")
        if is_cross_node is not None and not isinstance(is_cross_node, bool):
            raise SolveContractError(f"{context}.is_cross_node must be a bool")
        pairs.append(
            AdmittedPairTdoa(
                ch_a=ch_a,
                ch_b=ch_b,
                lag_seconds=_require_finite_float(mapping, "lag_seconds", context),
                confidence=_require_finite_float(mapping, "confidence", context),
                delay_samples=_optional_finite_float(mapping, "delay_samples", context),
                is_cross_node=is_cross_node,
                timing_uncertainty_seconds=_optional_finite_float(mapping, "timing_uncertainty_seconds", context),
            )
        )
    return tuple(pairs)


def _parse_coordinate_frame(payload: Any) -> SolveCoordinateFrame:
    context = "coordinate_frame"
    mapping = _require_mapping(payload, context)
    origin_payload = _require_key(mapping, "origin", context)
    origin = None
    if origin_payload is not None:
        origin_mapping = _require_mapping(origin_payload, f"{context}.origin")
        origin = GeodeticOrigin(
            lat=_require_finite_float(origin_mapping, "lat", f"{context}.origin"),
            lon=_require_finite_float(origin_mapping, "lon", f"{context}.origin"),
            alt_m=_require_finite_float(origin_mapping, "alt_m", f"{context}.origin"),
        )
    projection = None
    if _require_key(mapping, "projection", context) is not None:
        projection = _require_enum(mapping, "projection", CoordinateProjection, context)
    return SolveCoordinateFrame(
        frame_id=_optional_string(mapping, "frame_id", context),
        kind=_require_enum(mapping, "kind", CoordinateFrameKind, context),
        origin=origin,
        projection=projection,
    )


def _parse_sensor_entry(payload: Any, context: str) -> SolveSensorEntry:
    mapping = _require_mapping(payload, context)
    return SolveSensorEntry(
        sensor_id=_require_nonempty_string(mapping, "sensor_id", context),
        node_id=_require_nonempty_string(mapping, "node_id", context),
        node_channel_index=_require_int(mapping, "node_channel_index", context),
        position_m=_require_vec3(mapping, "position_m", context),
        sample_rate_hz=_require_int(mapping, "sample_rate_hz", context),
        position_source=_require_enum(mapping, "position_source", SensorPositionSource, context),
        position_std_m=_optional_finite_float(mapping, "position_std_m", context, require_key=True),
        clock_source=_require_enum(mapping, "clock_source", NodeTimeQuality, context),
        clock_domain=_require_nonempty_string(mapping, "clock_domain", context),
        clock_offset_std_s=_optional_finite_float(mapping, "clock_offset_std_s", context, require_key=True),
    )


def _parse_excluded_sensor(payload: Any, sensor_count: int, context: str) -> ExcludedSensor:
    mapping = _require_mapping(payload, context)
    return ExcludedSensor(
        sensor_index=_require_sensor_index(_require_key(mapping, "sensor_index", context), sensor_count, context),
        sensor_id=_require_nonempty_string(mapping, "sensor_id", context),
        reason=_require_enum(mapping, "reason", SolveExclusionReason, context),
    )


def _parse_excluded_pair(payload: Any, sensor_count: int, context: str) -> ExcludedSensorPair:
    mapping = _require_mapping(payload, context)
    return ExcludedSensorPair(
        sensor_index_a=_require_sensor_index(_require_key(mapping, "sensor_index_a", context), sensor_count, context),
        sensor_index_b=_require_sensor_index(_require_key(mapping, "sensor_index_b", context), sensor_count, context),
        reason=_require_enum(mapping, "reason", SolveExclusionReason, context),
    )


def _parse_node_timing(payload: Any, context: str) -> NodeTimingEvidence:
    mapping = _require_mapping(payload, context)
    return NodeTimingEvidence(
        node_id=_require_nonempty_string(mapping, "node_id", context),
        time_quality=_require_enum(mapping, "time_quality", NodeTimeQuality, context),
        clock_offset_std_s=_optional_finite_float(mapping, "clock_offset_std_s", context, require_key=True),
        timeline_residual_ns=_optional_int(mapping, "timeline_residual_ns", context, require_key=True),
        last_timestamp_step_ns=_optional_int(mapping, "last_timestamp_step_ns", context, require_key=True),
        timestamp_step_count=_require_int(mapping, "timestamp_step_count", context),
    )


# --- strict scalar readers ------------------------------------------------------


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SolveContractError(f"{context} must be an object")
    return value


def _require_key(mapping: dict[str, Any], key: str, context: str = "solve_provenance") -> Any:
    if key not in mapping:
        raise SolveContractError(f"{context}.{key} is missing")
    return mapping[key]


def _require_list(mapping: dict[str, Any], key: str, context: str = "solve_provenance") -> list[Any]:
    value = _require_key(mapping, key, context)
    if not isinstance(value, list):
        raise SolveContractError(f"{context}.{key} must be a list")
    return value


def _is_json_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(mapping: dict[str, Any], key: str, context: str = "solve_provenance") -> int:
    value = _require_key(mapping, key, context)
    if not _is_json_integer(value):
        raise SolveContractError(f"{context}.{key} must be an integer")
    return value


def _optional_int(
    mapping: dict[str, Any], key: str, context: str = "solve_provenance", *, require_key: bool = True
) -> int | None:
    value = _require_key(mapping, key, context) if require_key else mapping.get(key)
    if value is None:
        return None
    if not _is_json_integer(value):
        raise SolveContractError(f"{context}.{key} must be an integer or null")
    return value


def _require_sensor_index(value: Any, sensor_count: int, context: str) -> int:
    if not _is_json_integer(value) or not 0 <= value < sensor_count:
        raise SolveContractError(f"{context} = {value!r} does not index this solve's {sensor_count} sensors")
    return value


def _require_finite_float(mapping: dict[str, Any], key: str, context: str = "solve_provenance") -> float:
    value = _require_key(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SolveContractError(f"{context}.{key} must be a finite number")
    return float(value)


def _optional_finite_float(
    mapping: dict[str, Any], key: str, context: str, *, require_key: bool = False
) -> float | None:
    value = _require_key(mapping, key, context) if require_key else mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SolveContractError(f"{context}.{key} must be a finite number or null")
    return float(value)


def _require_nonempty_string(mapping: dict[str, Any], key: str, context: str = "solve_provenance") -> str:
    value = _require_key(mapping, key, context)
    if not isinstance(value, str) or not value:
        raise SolveContractError(f"{context}.{key} must be a non-empty string")
    return value


def _optional_string(mapping: dict[str, Any], key: str, context: str = "solve_provenance") -> str | None:
    value = _require_key(mapping, key, context)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SolveContractError(f"{context}.{key} must be a non-empty string or null")
    return value


def _require_enum(mapping: dict[str, Any], key: str, enum_type: type[StrEnum], context: str = "solve_provenance"):
    value = _require_key(mapping, key, context)
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = sorted(member.value for member in enum_type)
        raise SolveContractError(f"{context}.{key} = {value!r} is not one of {allowed}") from exc


def _require_vec3(mapping: dict[str, Any], key: str, context: str) -> tuple[float, float, float]:
    value = _require_key(mapping, key, context)
    if not isinstance(value, list) or len(value) != 3:
        raise SolveContractError(f"{context}.{key} must be [x, y, z]")
    components = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)) or not math.isfinite(component):
            raise SolveContractError(f"{context}.{key} must hold finite numbers")
        components.append(float(component))
    return (components[0], components[1], components[2])
