"""Calibration replay scoring: per-event (v1) metrics plus entity-aware metrics.

`evaluate_bundle` returns a `CalibrationReport` whose `errors` list is empty on
pass, mirroring `minimappr/sim/soak.py`. Per-event metrics are unchanged from
v1 for v1 bundles; echo-source events (ground truth v2) are excluded from them
and trajectories are evaluated at each detection's timestamp. Entity metrics
(`replay_entity_metrics.py`) are always computed and gated only when the
bundle's expectations set a threshold under ``entities``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from minimappr.calibration.bundle import CalibrationBundle
from minimappr.calibration.ground_truth_events import (
    parse_ground_truth_events,
    site_frame_from_manifest,
)
from minimappr.sim.replay_entity_metrics import (
    DEFAULT_ENTITY_EXPECTATIONS,
    EntityAssociationConfig,
    EntityMetrics,
    angular_difference_deg,
    bearing_deg,
    compute_entity_metrics,
    entity_threshold_errors,
)

DEFAULT_EXPECTATIONS: dict[str, Any] = {
    "schema_version": 2,
    # replay_reference_nodes: when True, nodes referenced as a ground-truth
    # source (placed at the sound source) are replayed like any other node
    # instead of being excluded from localization input.
    "runtime": {
        "classifier_backend": "yamnet",
        "settings_overrides": {},
        "replay_reference_nodes": False,
    },
    "classification": {
        "min_label_accuracy": 0.5,
        "match": "category",
        "per_label": {},
    },
    "localization": {
        "min_event_recall": 0.5,
        "max_median_position_error_m": 20.0,
        "max_p90_position_error_m": 40.0,
        "far_field_distance_m": 120.0,
        "far_field": {"max_median_bearing_error_deg": 10.0},
        "per_algorithm": {},
    },
    "entities": dict(DEFAULT_ENTITY_EXPECTATIONS),
}

MATCH_SLOP_NS = 2_000_000_000
"""Detections within ±2 s of a ground-truth window still count as matches."""


@dataclass
class EventResult:
    event_id: str
    label: str
    matched: bool = False
    label_correct: bool | None = None
    position_error_m: float | None = None
    bearing_error_deg: float | None = None
    far_field: bool = False
    algorithm: str | None = None


@dataclass
class CalibrationReport:
    errors: list[str] = field(default_factory=list)
    event_results: list[EventResult] = field(default_factory=list)
    event_recall: float = 0.0
    label_accuracy: float = 0.0
    median_position_error_m: float | None = None
    p90_position_error_m: float | None = None
    median_bearing_error_deg: float | None = None
    entity_metrics: EntityMetrics | None = None

    @property
    def passed(self) -> bool:
        return not self.errors


def _detection_algorithm(detection: dict) -> str | None:
    feature_summary = detection.get("feature_summary") or {}
    return (
        feature_summary.get("resolved_algorithm")
        or feature_summary.get("localization_method")
        or None
    )


def array_centroid_m(bundle: CalibrationBundle) -> np.ndarray:
    """Centroid of replayed (non-reference) nodes; all nodes if every node is a reference."""
    reference_node_ids = {
        event.get("source_node_id") for event in bundle.events if event.get("source_node_id")
    }
    nodes_with_position = [n for n in bundle.manifest.get("nodes", []) if n.get("position_m")]
    positions = [
        np.asarray(n["position_m"], dtype=np.float64)
        for n in nodes_with_position
        if n.get("node_id") not in reference_node_ids
    ] or [np.asarray(n["position_m"], dtype=np.float64) for n in nodes_with_position]
    return np.mean(positions, axis=0) if positions else np.zeros(3, dtype=np.float64)


def _merged_section(expectations: dict[str, Any], section: str) -> dict[str, Any]:
    return {**DEFAULT_EXPECTATIONS[section], **(expectations.get(section) or {})}


def _threshold_errors(
    report: CalibrationReport,
    classification_cfg: dict[str, Any],
    localization_cfg: dict[str, Any],
    errors_by_algorithm: dict[str, list[float]],
) -> list[str]:
    errors: list[str] = []
    labelled = [r for r in report.event_results if r.matched and r.label_correct is not None]
    if report.event_recall < float(localization_cfg["min_event_recall"]):
        errors.append(
            f"event recall {report.event_recall:.2f} below {localization_cfg['min_event_recall']}"
        )
    if labelled and report.label_accuracy < float(classification_cfg["min_label_accuracy"]):
        errors.append(
            f"label accuracy {report.label_accuracy:.2f} below "
            f"{classification_cfg['min_label_accuracy']}"
        )
    for label, cfg in (classification_cfg.get("per_label") or {}).items():
        label_results = [r for r in labelled if r.label == label]
        if not label_results:
            continue
        accuracy = sum(1 for r in label_results if r.label_correct) / len(label_results)
        if accuracy < float(cfg.get("min_label_accuracy", 0.0)):
            errors.append(
                f"label '{label}' accuracy {accuracy:.2f} below {cfg['min_label_accuracy']}"
            )
    if report.median_position_error_m is not None:
        if report.median_position_error_m > float(localization_cfg["max_median_position_error_m"]):
            errors.append(
                f"median position error {report.median_position_error_m:.1f} m above "
                f"{localization_cfg['max_median_position_error_m']} m"
            )
        if report.p90_position_error_m > float(localization_cfg["max_p90_position_error_m"]):
            errors.append(
                f"p90 position error {report.p90_position_error_m:.1f} m above "
                f"{localization_cfg['max_p90_position_error_m']} m"
            )
    far_field_cfg = localization_cfg.get("far_field") or {}
    if report.median_bearing_error_deg is not None and "max_median_bearing_error_deg" in far_field_cfg:
        if report.median_bearing_error_deg > float(far_field_cfg["max_median_bearing_error_deg"]):
            errors.append(
                f"median far-field bearing error {report.median_bearing_error_deg:.1f}° above "
                f"{far_field_cfg['max_median_bearing_error_deg']}°"
            )
    for algorithm, cfg in (localization_cfg.get("per_algorithm") or {}).items():
        algorithm_errors = errors_by_algorithm.get(algorithm)
        if not algorithm_errors:
            continue
        median_error = float(statistics.median(algorithm_errors))
        limit = float(cfg.get("max_median_position_error_m", float("inf")))
        if median_error > limit:
            errors.append(f"{algorithm} median position error {median_error:.1f} m above {limit} m")
    return errors


def evaluate_bundle(
    detections: list[dict],
    bundle: CalibrationBundle,
    expectations: dict[str, Any] | None = None,
) -> CalibrationReport:
    """Score replayed detections against the bundle's ground truth.

    Per-event: any detection within ±`MATCH_SLOP_NS` of a real event's window
    matches it (overlapping events may share detections — the entity metrics
    are the overlap-aware view). Position error uses ‖detection.position_m −
    truth(t_detection)‖; events farther than `far_field_distance_m` from the
    array centroid switch to bearing error (small tetra arrays are DOA sensors
    beyond their range resolution — see tests/test_localization_distance_matrix.py).
    """
    expectations = expectations or bundle.expectations or DEFAULT_EXPECTATIONS
    classification_cfg = _merged_section(expectations, "classification")
    localization_cfg = _merged_section(expectations, "localization")
    entity_cfg = _merged_section(expectations, "entities")

    ground_truth_events = parse_ground_truth_events(
        bundle.events, site_frame_from_manifest(bundle.manifest)
    )
    call_events = [event for event in ground_truth_events if not event.is_echo_source]
    centroid = array_centroid_m(bundle)

    report = CalibrationReport()
    far_field_distance_m = float(localization_cfg["far_field_distance_m"])
    match_mode = classification_cfg.get("match", "category")

    position_errors: list[float] = []
    bearing_errors: list[float] = []
    errors_by_algorithm: dict[str, list[float]] = {}

    for event in call_events:
        result = EventResult(event_id=event.event_id, label=event.label)
        candidates = [
            det
            for det in detections
            if event.is_active_at(int(det.get("timestamp_ns") or 0), MATCH_SLOP_NS)
        ]
        result.far_field = (
            float(np.linalg.norm(event.position_at(event.midpoint_ns) - centroid))
            > far_field_distance_m
        )
        if candidates:
            result.matched = True
            result.label_correct = any(
                event.label_matches(det.get("label"), det.get("label_category"), match_mode)
                for det in candidates
            )
            best_error, best_det, best_truth = None, None, None
            for det in candidates:
                position = det.get("position_m")
                if not position:
                    continue
                truth_m = event.position_at(int(det.get("timestamp_ns") or 0))
                error = float(np.linalg.norm(np.asarray(position, dtype=np.float64) - truth_m))
                if best_error is None or error < best_error:
                    best_error, best_det, best_truth = error, det, truth_m
            if best_det is not None:
                result.algorithm = _detection_algorithm(best_det)
                det_position = np.asarray(best_det["position_m"], dtype=np.float64)
                if result.far_field:
                    result.bearing_error_deg = angular_difference_deg(
                        bearing_deg(centroid, det_position), bearing_deg(centroid, best_truth)
                    )
                    bearing_errors.append(result.bearing_error_deg)
                else:
                    result.position_error_m = best_error
                    position_errors.append(best_error)
                    if result.algorithm:
                        errors_by_algorithm.setdefault(result.algorithm, []).append(best_error)
        report.event_results.append(result)

    if not report.event_results:
        report.errors.append("bundle contains no ground-truth events")
        return report

    matched = [r for r in report.event_results if r.matched]
    report.event_recall = len(matched) / len(report.event_results)
    labelled = [r for r in matched if r.label_correct is not None]
    report.label_accuracy = (
        sum(1 for r in labelled if r.label_correct) / len(labelled) if labelled else 0.0
    )
    if position_errors:
        report.median_position_error_m = float(statistics.median(position_errors))
        report.p90_position_error_m = float(np.percentile(position_errors, 90))
    if bearing_errors:
        report.median_bearing_error_deg = float(statistics.median(bearing_errors))
    report.errors.extend(
        _threshold_errors(report, classification_cfg, localization_cfg, errors_by_algorithm)
    )

    association_config = EntityAssociationConfig.from_expectations(
        entity_cfg,
        far_field_distance_m=far_field_distance_m,
        match_mode=match_mode,
        array_centroid_m=centroid,
    )
    report.entity_metrics = compute_entity_metrics(
        detections, ground_truth_events, association_config
    )
    report.errors.extend(entity_threshold_errors(report.entity_metrics, entity_cfg))
    return report
