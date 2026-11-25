"""Finding vocabulary, report and thresholds for calibration bundle validation.

Codes are a stable public vocabulary (``docs/field_capture_protocol.md`` lists
them with the fix for each); every code has one fixed severity, so a consumer
can filter on the code alone. Severity meaning:

- ``error``: the bundle cannot be scored honestly against the protocol (replay
  would fail, or the score would rest on an unknown position, clock or truth).
- ``warning``: usable, but some evidence is weaker, excluded or missing; the
  report says which part of the score it affects.
- ``info``: statistics and protocol notes.

Validation never raises on a malformed bundle; it reports.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from minimappr.core.localization_overhaul_config import (
    DEFAULT_LOCALIZATION_CROSS_NODE_MAX_TIMELINE_RESIDUAL_SECONDS,
    DEFAULT_LOCALIZATION_TIMESTAMP_STEP_THRESHOLD_SECONDS,
)

BUNDLE_VALIDATION_REPORT_SCHEMA_VERSION = 1


class FindingSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class BundleFindingCode(StrEnum):
    # Container and site frame
    BUNDLE_UNREADABLE = "bundle_unreadable"
    SITE_ORIGIN_INVALID = "site_origin_invalid"
    SITE_COORDINATE_MODE_UNKNOWN = "site_coordinate_mode_unknown"
    CAPTURE_WINDOW_INVALID = "capture_window_invalid"
    # Node survey
    NODE_TYPE_MISSING = "node_type_missing"
    NODE_POSITION_MISSING = "node_position_missing"
    NODE_POSITION_FRAME_MISMATCH = "node_position_frame_mismatch"
    NODE_POSITION_FAR_FROM_SITE_ORIGIN = "node_position_far_from_site_origin"
    NODE_POSITION_SOURCE_MISSING = "node_position_source_missing"
    NODE_POSITION_SOURCE_UNKNOWN = "node_position_source_unknown"
    NODE_POSITION_UNTRUSTED = "node_position_untrusted"
    HAND_MEASURED_POSITION_STD_MISSING = "hand_measured_position_std_missing"
    NODE_POSITION_STD_MISSING = "node_position_std_missing"
    NODE_POSITION_STD_INVALID = "node_position_std_invalid"
    NODE_ORIENTATION_MISSING = "node_orientation_missing"
    NODE_ORIENTATION_INVALID = "node_orientation_invalid"
    NODE_ORIENTATION_CONFLICT = "node_orientation_conflict"
    NODE_SENSOR_OFFSETS_MISMATCH = "node_sensor_offsets_mismatch"
    # Node audio and rates
    NODE_AUDIO_MISSING = "node_audio_missing"
    NODE_AUDIO_CHANNEL_MISMATCH = "node_audio_channel_mismatch"
    NODE_SAMPLE_RATE_INCONSISTENT = "node_sample_rate_inconsistent"
    NODE_WINDOW_COVERAGE_PARTIAL = "node_window_coverage_partial"
    CLUSTER_SAMPLE_RATE_MISMATCH = "cluster_sample_rate_mismatch"
    # Per-frame timing and clock
    FRAME_TIMING_MISSING = "frame_timing_missing"
    FRAME_TIMING_UNREADABLE = "frame_timing_unreadable"
    FRAME_TIMING_NODE_MISMATCH = "frame_timing_node_mismatch"
    FRAME_TIMING_NOT_MONOTONIC = "frame_timing_not_monotonic"
    FRAME_TIMING_OVERLAP = "frame_timing_overlap"
    FRAME_TIMING_EMPTY_FRAME = "frame_timing_empty_frame"
    FRAME_TIMING_OUTSIDE_AUDIO = "frame_timing_outside_audio"
    TIME_QUALITY_UNREPORTED = "time_quality_unreported"
    TIME_QUALITY_UNKNOWN_VALUE = "time_quality_unknown_value"
    TIME_QUALITY_FREE_RUNNING = "time_quality_free_running"
    TIME_QUALITY_CHANGED = "time_quality_changed"
    TIMESTAMP_STEP_DETECTED = "timestamp_step_detected"
    TIMELINE_RESIDUAL_OVER_LIMIT = "timeline_residual_over_limit"
    NODE_GAP_FRACTION_HIGH = "node_gap_fraction_high"
    NODE_CLOCK_SUMMARY = "node_clock_summary"
    NODE_GAP_STATISTICS = "node_gap_statistics"
    # Ground truth
    GROUND_TRUTH_INVALID = "ground_truth_invalid"
    GROUND_TRUTH_MISSING = "ground_truth_missing"
    GROUND_TRUTH_OUTSIDE_CAPTURE = "ground_truth_outside_capture"
    GROUND_TRUTH_PARTIALLY_OUTSIDE_CAPTURE = "ground_truth_partially_outside_capture"
    GROUND_TRUTH_POSITION_OUTSIDE_SITE = "ground_truth_position_outside_site"
    GROUND_TRUTH_SOURCE_NODE_UNKNOWN = "ground_truth_source_node_unknown"
    GROUND_TRUTH_TRAJECTORY_INCOMPLETE = "ground_truth_trajectory_incomplete"
    GROUND_TRUTH_TRAJECTORY_SPARSE = "ground_truth_trajectory_sparse"
    GROUND_TRUTH_POSITION_STD_MISSING = "ground_truth_position_std_missing"
    GROUND_TRUTH_ACCURACY_COARSE_FOR_RANGE = "ground_truth_accuracy_coarse_for_range"
    # Range ladder and summary
    RANGE_LADDER_COVERAGE = "range_ladder_coverage"
    RANGE_LADDER_RUNG_EMPTY = "range_ladder_rung_empty"
    RANGE_LADDER_RUNG_UNDERCOVERED = "range_ladder_rung_undercovered"
    GROUND_TRUTH_OUTSIDE_RANGE_LADDER = "ground_truth_outside_range_ladder"
    CAPTURE_TOO_SHORT = "capture_too_short"
    CAPTURE_SUMMARY = "capture_summary"


_C = BundleFindingCode
_ERROR_CODES = frozenset(
    {
        _C.BUNDLE_UNREADABLE,
        _C.SITE_ORIGIN_INVALID,
        _C.SITE_COORDINATE_MODE_UNKNOWN,
        _C.CAPTURE_WINDOW_INVALID,
        _C.NODE_TYPE_MISSING,
        _C.NODE_POSITION_MISSING,
        _C.NODE_POSITION_FRAME_MISMATCH,
        _C.NODE_POSITION_SOURCE_MISSING,
        _C.NODE_POSITION_SOURCE_UNKNOWN,
        _C.HAND_MEASURED_POSITION_STD_MISSING,
        _C.NODE_POSITION_STD_INVALID,
        _C.NODE_ORIENTATION_MISSING,
        _C.NODE_ORIENTATION_INVALID,
        _C.NODE_ORIENTATION_CONFLICT,
        _C.NODE_SENSOR_OFFSETS_MISMATCH,
        _C.NODE_AUDIO_MISSING,
        _C.NODE_AUDIO_CHANNEL_MISMATCH,
        _C.NODE_SAMPLE_RATE_INCONSISTENT,
        _C.FRAME_TIMING_MISSING,
        _C.FRAME_TIMING_UNREADABLE,
        _C.FRAME_TIMING_NODE_MISMATCH,
        _C.FRAME_TIMING_NOT_MONOTONIC,
        _C.FRAME_TIMING_OVERLAP,
        _C.FRAME_TIMING_EMPTY_FRAME,
        _C.FRAME_TIMING_OUTSIDE_AUDIO,
        _C.TIME_QUALITY_UNREPORTED,
        _C.TIME_QUALITY_UNKNOWN_VALUE,
        _C.GROUND_TRUTH_INVALID,
        _C.GROUND_TRUTH_MISSING,
        _C.GROUND_TRUTH_OUTSIDE_CAPTURE,
        _C.GROUND_TRUTH_POSITION_OUTSIDE_SITE,
    }
)
_INFO_CODES = frozenset(
    {
        _C.TIME_QUALITY_CHANGED,
        _C.NODE_CLOCK_SUMMARY,
        _C.NODE_GAP_STATISTICS,
        _C.GROUND_TRUTH_POSITION_STD_MISSING,
        _C.RANGE_LADDER_COVERAGE,
        _C.RANGE_LADDER_RUNG_EMPTY,
        _C.GROUND_TRUTH_OUTSIDE_RANGE_LADDER,
        _C.CAPTURE_SUMMARY,
    }
)
SEVERITY_BY_FINDING_CODE: Mapping[BundleFindingCode, FindingSeverity] = {
    code: (
        FindingSeverity.ERROR
        if code in _ERROR_CODES
        else FindingSeverity.INFO
        if code in _INFO_CODES
        else FindingSeverity.WARNING
    )
    for code in BundleFindingCode
}


@dataclass(frozen=True)
class BundleValidationConfig:
    """Protocol thresholds. Clock thresholds default to the live Settings defaults
    so the report flags exactly what the sidecar would act on."""

    timestamp_step_threshold_seconds: float = DEFAULT_LOCALIZATION_TIMESTAMP_STEP_THRESHOLD_SECONDS
    cross_node_max_timeline_residual_seconds: float = DEFAULT_LOCALIZATION_CROSS_NODE_MAX_TIMELINE_RESIDUAL_SECONDS
    # position_m and position_geo of one node must describe one point; more than
    # a few centimetres apart means they were written in different site frames.
    node_position_frame_tolerance_m: float = 0.05
    # Far beyond the 1 km rung: a node or truth point this far out was surveyed
    # against a different origin.
    maximum_distance_from_site_origin_m: float = 20_000.0
    quaternion_unit_norm_tolerance: float = 1.0e-3
    orientation_agreement_tolerance: float = 1.0e-3
    # Several calls per rung plus a quiet stretch for the noise floor.
    minimum_capture_duration_seconds: float = 60.0
    # Release requirement: minimise lost packets in detection audio.
    maximum_node_gap_fraction: float = 0.02
    minimum_node_window_coverage_fraction: float = 0.9
    range_ladder_rungs_m: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)
    # An event belongs to the nearest rung within this many decades of range.
    range_rung_half_width_decades: float = 0.5
    # Coverage statistics need more than one shot per rung.
    minimum_events_per_range_rung: int = 3
    # Linear interpolation between drone GPS fixes this far apart misplaces a
    # 10 m/s drone by metres.
    maximum_trajectory_waypoint_spacing_seconds: float = 2.0
    # Truth coarser than this fraction of the range cannot grade a range-resolved fix.
    maximum_truth_position_std_relative_to_range: float = 0.1


@dataclass(frozen=True)
class BundleValidationFinding:
    code: BundleFindingCode
    message: str
    node_id: str | None = None
    event_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> FindingSeverity:
        return SEVERITY_BY_FINDING_CODE[self.code]

    def to_json_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code.value, "severity": self.severity.value, "message": self.message}
        if self.node_id is not None:
            payload["node_id"] = self.node_id
        if self.event_id is not None:
            payload["event_id"] = self.event_id
        if self.details:
            payload["details"] = dict(self.details)
        return payload

    def one_line(self) -> str:
        subject = f" [{self.node_id}]" if self.node_id else f" [{self.event_id}]" if self.event_id else ""
        return f"{self.severity.value.upper():7s} {self.code.value}{subject}: {self.message}"


@dataclass(frozen=True)
class BundleValidationReport:
    bundle_path: str
    findings: tuple[BundleValidationFinding, ...]

    def findings_with_severity(self, severity: FindingSeverity) -> tuple[BundleValidationFinding, ...]:
        return tuple(finding for finding in self.findings if finding.severity is severity)

    @property
    def has_errors(self) -> bool:
        return bool(self.findings_with_severity(FindingSeverity.ERROR))

    @property
    def has_warnings(self) -> bool:
        return bool(self.findings_with_severity(FindingSeverity.WARNING))

    def codes(self, severity: FindingSeverity | None = None) -> set[BundleFindingCode]:
        return {finding.code for finding in self.findings if severity is None or finding.severity is severity}

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "schema_version": BUNDLE_VALIDATION_REPORT_SCHEMA_VERSION,
            "bundle_path": self.bundle_path,
            "error_count": len(self.findings_with_severity(FindingSeverity.ERROR)),
            "warning_count": len(self.findings_with_severity(FindingSeverity.WARNING)),
            "info_count": len(self.findings_with_severity(FindingSeverity.INFO)),
            "findings": [finding.to_json_payload() for finding in self.findings],
        }
