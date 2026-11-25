"""Site-frame, node-survey, audio and sample-rate checks for bundle validation.

Node entries follow ``docs/field_mode_progress/bundle_format_v2.md`` section 2
plus the survey keys the field capture protocol adds (``position_source``,
``position_std_m``, ``body_to_world_quaternion_wxyz``, ``cluster_id``).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import ValidationError

from minimappr.api.rust_localization_solve_contract import SensorPositionSource
from minimappr.calibration.bundle_validation_findings import (
    BundleFindingCode as Code,
    BundleValidationConfig,
    BundleValidationFinding as Finding,
)
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.sensor_world_position import body_to_world_quaternion_wxyz
from minimappr.models import GeoPoint, NodeOrientation, NodeType

NANOSECONDS_PER_SECOND = 1_000_000_000
# A node placement in a bundle is one of these; the solve-only sources
# (explicit cluster, node body offset) never describe where a node stands.
BUNDLE_NODE_POSITION_SOURCES = frozenset(
    {
        SensorPositionSource.OPERATOR_OVERRIDE.value,
        SensorPositionSource.STATIONARY_ESTIMATE.value,
        SensorPositionSource.REPORTED_GPS.value,
        SensorPositionSource.FALLBACK_STATIC.value,
    }
)
# Nodes with no ``cluster_id`` share the auto-GPS cluster, i.e. the whole site.
SITE_WIDE_CLUSTER_ID = "site"
_IDENTITY_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class WavHeader:
    channel_count: int
    sample_rate_hz: int
    total_samples: int


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def capture_window_ns(manifest: Mapping[str, Any]) -> tuple[int, int] | None:
    window = manifest.get("time_window") or {}
    try:
        start_ns, end_ns = int(window["start_ns"]), int(window["end_ns"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start_ns, end_ns) if end_ns > start_ns else None


def check_site_frame(manifest: Mapping[str, Any]) -> tuple[list[Finding], LocalCoordinateFrame | None]:
    site = manifest.get("site") if isinstance(manifest.get("site"), dict) else {}
    origin = site.get("origin") if isinstance(site.get("origin"), dict) else {}
    findings: list[Finding] = []
    if not (_is_finite_number(origin.get("lat")) and _is_finite_number(origin.get("lon"))):
        findings.append(Finding(Code.SITE_ORIGIN_INVALID, "site.origin needs numeric lat and lon"))
    coordinate_mode = site.get("coordinate_mode", "flat")
    if not isinstance(coordinate_mode, str) or coordinate_mode.strip().lower() not in {"flat", "geodetic"}:
        findings.append(
            Finding(Code.SITE_COORDINATE_MODE_UNKNOWN, f"site.coordinate_mode {coordinate_mode!r} is not flat or geodetic")
        )
    if capture_window_ns(manifest) is None:
        findings.append(Finding(Code.CAPTURE_WINDOW_INVALID, "time_window needs integer start_ns < end_ns"))
    if any(finding.code is not Code.CAPTURE_WINDOW_INVALID for finding in findings):
        return findings, None
    frame = LocalCoordinateFrame(
        origin=GeoPoint(lat=origin["lat"], lon=origin["lon"], alt_m=float(origin.get("alt_m") or 0.0)),
        mode=coordinate_mode,
    )
    return findings, frame


def _finite_vector3(value: Any) -> np.ndarray | None:
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(_is_finite_number(axis) for axis in value):
        return np.asarray(value, dtype=np.float64)
    return None


def _geo_point(value: Any) -> GeoPoint | None:
    if isinstance(value, dict) and _is_finite_number(value.get("lat")) and _is_finite_number(value.get("lon")):
        return GeoPoint(lat=value["lat"], lon=value["lon"], alt_m=float(value.get("alt_m") or 0.0))
    return None


def node_position_in_site_frame(node: Mapping[str, Any], frame: LocalCoordinateFrame | None) -> np.ndarray | None:
    """``position_m`` when present (replay places nodes by it), else the geo fix projected."""
    position_m = _finite_vector3(node.get("position_m"))
    if position_m is not None:
        return position_m
    geo = _geo_point(node.get("position_geo"))
    if geo is None or frame is None:
        return None
    return np.asarray(frame.geo_to_local(geo), dtype=np.float64)


def _position_findings(node_id: str, node: Mapping[str, Any], frame: LocalCoordinateFrame | None, config: BundleValidationConfig) -> list[Finding]:
    position_m = _finite_vector3(node.get("position_m"))
    geo = _geo_point(node.get("position_geo"))
    if position_m is None and geo is None:
        return [Finding(Code.NODE_POSITION_MISSING, "node has neither position_m nor position_geo", node_id=node_id)]
    if frame is None:
        return []
    findings: list[Finding] = []
    if position_m is not None and geo is not None:
        separation_m = float(np.linalg.norm(np.asarray(frame.geo_to_local(geo)) - position_m))
        if separation_m > config.node_position_frame_tolerance_m:
            findings.append(
                Finding(
                    Code.NODE_POSITION_FRAME_MISMATCH,
                    f"position_m and position_geo are {separation_m:.2f} m apart in the site frame; "
                    "one of them was written against a different origin",
                    node_id=node_id,
                    details={"separation_m": separation_m},
                )
            )
    local = node_position_in_site_frame(node, frame)
    distance_m = float(np.linalg.norm(local))
    if distance_m > config.maximum_distance_from_site_origin_m:
        findings.append(
            Finding(
                Code.NODE_POSITION_FAR_FROM_SITE_ORIGIN,
                f"node is {distance_m:.0f} m from the site origin; check the origin and projection",
                node_id=node_id,
                details={"distance_m": distance_m},
            )
        )
    return findings


def _position_source_findings(node_id: str, node: Mapping[str, Any]) -> list[Finding]:
    position_source = node.get("position_source")
    if position_source is None:
        return [
            Finding(
                Code.NODE_POSITION_SOURCE_MISSING,
                "no position_source: the scorer cannot tell a surveyed position from a guess",
                node_id=node_id,
            )
        ]
    if position_source not in BUNDLE_NODE_POSITION_SOURCES:
        return [
            Finding(
                Code.NODE_POSITION_SOURCE_UNKNOWN,
                f"position_source {position_source!r} is not one of {sorted(BUNDLE_NODE_POSITION_SOURCES)}",
                node_id=node_id,
            )
        ]
    findings: list[Finding] = []
    if position_source == SensorPositionSource.FALLBACK_STATIC.value:
        findings.append(
            Finding(
                Code.NODE_POSITION_UNTRUSTED,
                "fallback_static position: the node solves alone and forms no cross-node baseline",
                node_id=node_id,
            )
        )
    position_std_m = node.get("position_std_m")
    if position_std_m is None:
        if position_source == SensorPositionSource.OPERATOR_OVERRIDE.value:
            findings.append(
                Finding(
                    Code.HAND_MEASURED_POSITION_STD_MISSING,
                    "hand-measured position without position_std_m: its accuracy is what widens the error",
                    node_id=node_id,
                )
            )
        elif position_source != SensorPositionSource.FALLBACK_STATIC.value:
            findings.append(
                Finding(
                    Code.NODE_POSITION_STD_MISSING,
                    f"{position_source} position without position_std_m; coverage scoring treats it as exact",
                    node_id=node_id,
                )
            )
    elif not _is_finite_number(position_std_m) or position_std_m < 0.0:
        findings.append(
            Finding(Code.NODE_POSITION_STD_INVALID, f"position_std_m {position_std_m!r} is not a finite number >= 0", node_id=node_id)
        )
    return findings


def node_is_array(node: Mapping[str, Any]) -> bool:
    """Two or more distinct mic offsets: bearing depends on the mount orientation."""
    offsets = [tuple(offset) for offset in node.get("sensor_offsets_m") or [] if isinstance(offset, (list, tuple))]
    return len(set(offsets)) >= 2


def _canonical_quaternion(quaternion_wxyz: Iterable[float]) -> np.ndarray:
    quaternion = np.asarray(list(quaternion_wxyz), dtype=np.float64)
    return -quaternion if quaternion[0] < 0.0 else quaternion


def _orientation_findings(node_id: str, node: Mapping[str, Any], config: BundleValidationConfig) -> list[Finding]:
    raw_orientation = node.get("orientation")
    raw_quaternion = node.get("body_to_world_quaternion_wxyz")
    has_orientation = isinstance(raw_orientation, dict) and bool(raw_orientation)
    if not has_orientation and raw_quaternion is None:
        if node_is_array(node):
            return [
                Finding(
                    Code.NODE_ORIENTATION_MISSING,
                    "array node without orientation or body_to_world_quaternion_wxyz; "
                    "record the mount even when it is identity",
                    node_id=node_id,
                )
            ]
        return []
    orientation_quaternion: np.ndarray | None = None
    if has_orientation:
        try:
            orientation = NodeOrientation.model_validate(raw_orientation)
        except ValidationError as exc:
            return [Finding(Code.NODE_ORIENTATION_INVALID, f"orientation is invalid: {exc.errors()[0]['msg']}", node_id=node_id)]
        orientation_quaternion = _canonical_quaternion(body_to_world_quaternion_wxyz(orientation) or _IDENTITY_QUATERNION_WXYZ)
    if raw_quaternion is None:
        return []
    if not (
        isinstance(raw_quaternion, (list, tuple))
        and len(raw_quaternion) == 4
        and all(_is_finite_number(component) for component in raw_quaternion)
        and abs(float(np.linalg.norm(raw_quaternion)) - 1.0) <= config.quaternion_unit_norm_tolerance
    ):
        return [
            Finding(
                Code.NODE_ORIENTATION_INVALID,
                f"body_to_world_quaternion_wxyz {raw_quaternion!r} is not a unit quaternion [w, x, y, z]",
                node_id=node_id,
            )
        ]
    if orientation_quaternion is not None:
        difference = float(np.max(np.abs(_canonical_quaternion(raw_quaternion) - orientation_quaternion)))
        if difference > config.orientation_agreement_tolerance:
            return [
                Finding(
                    Code.NODE_ORIENTATION_CONFLICT,
                    "orientation (yaw/pitch/roll, Rz*Ry*Rx) and body_to_world_quaternion_wxyz describe different rotations",
                    node_id=node_id,
                    details={"max_component_difference": difference},
                )
            ]
    return []


def check_node_survey(node: Mapping[str, Any], frame: LocalCoordinateFrame | None, config: BundleValidationConfig) -> list[Finding]:
    node_id = str(node.get("node_id"))
    findings: list[Finding] = []
    node_type = node.get("node_type")
    if node_type not in {member.value for member in NodeType}:
        findings.append(Finding(Code.NODE_TYPE_MISSING, f"node_type {node_type!r} is not a NodeType; binary replay needs it", node_id=node_id))
    findings += _position_findings(node_id, node, frame, config)
    findings += _position_source_findings(node_id, node)
    findings += _orientation_findings(node_id, node, config)
    channel_count = len(node.get("channel_sensor_ids") or [])
    offset_count = len(node.get("sensor_offsets_m") or [])
    if offset_count != channel_count:
        findings.append(
            Finding(
                Code.NODE_SENSOR_OFFSETS_MISMATCH,
                f"{offset_count} sensor_offsets_m for {channel_count} channel_sensor_ids",
                node_id=node_id,
            )
        )
    return findings


def check_node_audio(
    node: Mapping[str, Any],
    wav_header: WavHeader | None,
    window_ns: tuple[int, int] | None,
    config: BundleValidationConfig,
) -> list[Finding]:
    node_id = str(node.get("node_id"))
    if wav_header is None:
        return [Finding(Code.NODE_AUDIO_MISSING, f"audio file {node.get('audio_file')!r} is absent or not PCM16 WAV", node_id=node_id)]
    findings: list[Finding] = []
    channel_count = len(node.get("channel_sensor_ids") or [])
    if wav_header.channel_count != channel_count:
        findings.append(
            Finding(
                Code.NODE_AUDIO_CHANNEL_MISMATCH,
                f"WAV has {wav_header.channel_count} channels, manifest names {channel_count} sensors",
                node_id=node_id,
            )
        )
    if node.get("sample_rate_hz") != wav_header.sample_rate_hz:
        findings.append(
            Finding(
                Code.NODE_SAMPLE_RATE_INCONSISTENT,
                f"manifest says {node.get('sample_rate_hz')} Hz, WAV is {wav_header.sample_rate_hz} Hz",
                node_id=node_id,
            )
        )
    if window_ns is not None:
        window_seconds = (window_ns[1] - window_ns[0]) / NANOSECONDS_PER_SECOND
        audio_seconds = wav_header.total_samples / wav_header.sample_rate_hz
        coverage_fraction = audio_seconds / window_seconds
        if coverage_fraction < config.minimum_node_window_coverage_fraction:
            findings.append(
                Finding(
                    Code.NODE_WINDOW_COVERAGE_PARTIAL,
                    f"audio covers {audio_seconds:.1f} s of the {window_seconds:.1f} s capture window",
                    node_id=node_id,
                    details={"audio_seconds": audio_seconds, "window_seconds": window_seconds},
                )
            )
    return findings


def check_cluster_sample_rates(nodes: Iterable[Mapping[str, Any]], reference_node_ids: set[str]) -> list[Finding]:
    """A cluster mixing rates loses its cross-rate pairs (guard now, resample later)."""
    node_ids_by_rate_by_cluster: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for node in nodes:
        node_id = str(node.get("node_id"))
        if node_id in reference_node_ids or not isinstance(node.get("sample_rate_hz"), int):
            continue
        cluster_id = str(node.get("cluster_id") or SITE_WIDE_CLUSTER_ID)
        node_ids_by_rate_by_cluster[cluster_id][node["sample_rate_hz"]].append(node_id)
    findings: list[Finding] = []
    for cluster_id, node_ids_by_rate in sorted(node_ids_by_rate_by_cluster.items()):
        if len(node_ids_by_rate) < 2:
            continue
        groups = {str(rate): sorted(node_ids) for rate, node_ids in sorted(node_ids_by_rate.items())}
        findings.append(
            Finding(
                Code.CLUSTER_SAMPLE_RATE_MISMATCH,
                f"cluster {cluster_id!r} mixes sample rates {groups}: pairs across rates are excluded from "
                "cross-node solves (sample_rate_mismatch), not resampled; each rate group still solves alone",
                details={"cluster_id": cluster_id, "node_ids_by_sample_rate_hz": groups},
            )
        )
    return findings
