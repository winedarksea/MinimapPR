"""Tiered replay scoring: heard -> classified and bearing-correct -> classified and range-resolved.

Each tier is a strictly harder claim about one ground-truth emission:

* **heard**: some detection is temporally associated with the emission;
* **classified and bearing-correct**: an associated detection has an accepted
  label *and* a position whose azimuth from its observer is within
  ``bearing_correct_max_error_deg`` of the truth's azimuth;
* **classified and range-resolved**: additionally the detection *claims* a
  range (localized, range mode observable) and its radial distance from the
  observer is within the range tolerance.

A **false range claim** is a detection that claims range but whose range is
outside tolerance, or that claims range while no emission is active (a
phantom fix). Bearing-only output (omni, unobservable range mode, weak range
observability) never counts as a range claim, so an honest bearing-only array
is never penalised for not resolving range.

Association: each detection is assigned to at most one non-echo emission
active within ``association_time_slop_seconds``; label-matching emissions are
preferred, then the smallest bearing error (localized detections), then the
nearest window midpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from minimappr.calibration.ground_truth_events import GroundTruthEvent
from minimappr.core.range_projection import range_mode_is_unobservable
from minimappr.models import WEAK_RANGE_OBSERVABILITY_THRESHOLD
from minimappr.sim.replay_entity_metrics import angular_difference_deg, bearing_deg
from minimappr.sim.replay_gate_config import TierScoringConfig

NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass
class EmissionTierResult:
    event_id: str
    individual_id: str
    label: str
    true_range_m: float | None = None
    heard: bool = False
    classified: bool = False
    classified_bearing_correct: bool = False
    classified_range_resolved: bool = False
    best_bearing_error_deg: float | None = None
    best_range_error_m: float | None = None


@dataclass
class TierScoringReport:
    emission_results: list[EmissionTierResult] = field(default_factory=list)
    emission_count: int = 0
    heard_count: int = 0
    classified_count: int = 0
    classified_bearing_correct_count: int = 0
    classified_range_resolved_count: int = 0
    heard_recall: float | None = None
    classified_bearing_correct_recall: float | None = None
    classified_range_resolved_recall: float | None = None
    range_claim_count: int = 0
    out_of_tolerance_range_claim_count: int = 0
    unassociated_range_claim_count: int = 0
    false_range_claim_count: int = 0
    false_range_claim_rate: float | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


def detection_has_position_fix(detection: dict) -> bool:
    """A localized detection with a 3-vector position (omni rows carry the node position)."""
    position_m = detection.get("position_m")
    return position_m is not None and len(position_m) >= 3 and detection.get("reporting_modality") != "omni"


def detection_claims_range(detection: dict) -> bool:
    """True when the detection asserts a resolved range, not just a direction.

    Mirrors the COP's own rule (``spatial_display_mode_for_detection``) except
    that ``range_bearing_projected`` is not a range claim here: its point is a
    placeholder along a well-observed bearing, displayed but not asserted.
    """
    if not detection_has_position_fix(detection):
        return False
    feature_summary = detection.get("feature_summary") or {}
    if range_mode_is_unobservable(feature_summary.get("localization_range_projection_mode")):
        return False
    range_observability = feature_summary.get("localization_range_observability")
    if isinstance(range_observability, int | float) and range_observability <= WEAK_RANGE_OBSERVABILITY_THRESHOLD:
        return False
    return True


@dataclass(frozen=True)
class _DetectionGeometry:
    bearing_error_deg: float
    range_error_m: float
    true_range_m: float


class _TierScorer:
    def __init__(
        self,
        config: TierScoringConfig,
        observer_positions_m: dict[str, np.ndarray],
        array_centroid_m: np.ndarray,
    ) -> None:
        self.config = config
        self.observer_positions_m = {
            node_id: np.asarray(position, dtype=np.float64) for node_id, position in observer_positions_m.items()
        }
        self.array_centroid_m = np.asarray(array_centroid_m, dtype=np.float64)
        self.range_axes = slice(0, 2) if config.horizontal_only else slice(0, 3)
        self.slop_ns = int(config.association_time_slop_seconds * NANOSECONDS_PER_SECOND)

    def observer_for(self, detection: dict) -> np.ndarray:
        # Cross-node cluster solves have no single observing array; the site
        # centroid is then the common origin for both claimed and true bearing.
        return self.observer_positions_m.get(str(detection.get("source_node_id") or ""), self.array_centroid_m)

    def geometry(self, detection: dict, event: GroundTruthEvent) -> _DetectionGeometry:
        observer_m = self.observer_for(detection)
        truth_m = np.asarray(event.position_at(int(detection.get("timestamp_ns") or 0)), dtype=np.float64)
        claimed_m = np.asarray(detection["position_m"], dtype=np.float64)
        true_range_m = float(np.linalg.norm((truth_m - observer_m)[self.range_axes]))
        claimed_range_m = float(np.linalg.norm((claimed_m - observer_m)[self.range_axes]))
        return _DetectionGeometry(
            bearing_error_deg=angular_difference_deg(bearing_deg(observer_m, claimed_m), bearing_deg(observer_m, truth_m)),
            range_error_m=abs(claimed_range_m - true_range_m),
            true_range_m=true_range_m,
        )

    def range_tolerance_m(self, true_range_m: float) -> float:
        return max(
            self.config.range_resolved_max_absolute_error_m,
            self.config.range_resolved_max_relative_error * true_range_m,
        )

    def label_matches(self, detection: dict, event: GroundTruthEvent) -> bool:
        return event.label_matches(detection.get("label"), detection.get("label_category"), self.config.label_match_mode)

    def assign(self, detection: dict, events: list[GroundTruthEvent]) -> GroundTruthEvent | None:
        timestamp_ns = int(detection.get("timestamp_ns") or 0)
        candidates = [event for event in events if event.is_active_at(timestamp_ns, self.slop_ns)]
        if not candidates:
            return None
        has_fix = detection_has_position_fix(detection)

        def preference(event: GroundTruthEvent) -> tuple[int, float, int]:
            bearing_error = self.geometry(detection, event).bearing_error_deg if has_fix else 0.0
            return (0 if self.label_matches(detection, event) else 1, bearing_error, abs(event.midpoint_ns - timestamp_ns))

        return min(candidates, key=preference)


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_detection_tiers(
    detections: list[dict],
    ground_truth_events: list[GroundTruthEvent],
    config: TierScoringConfig,
    *,
    observer_positions_m: dict[str, np.ndarray],
    array_centroid_m: np.ndarray,
) -> TierScoringReport:
    """Score detection rows (stored-detection dicts) against ground-truth emissions."""
    scorer = _TierScorer(config, observer_positions_m, array_centroid_m)
    emissions = [event for event in ground_truth_events if not event.is_echo_source]
    results_by_event_id = {
        event.event_id: EmissionTierResult(event_id=event.event_id, individual_id=event.individual_id, label=event.label)
        for event in emissions
    }
    report = TierScoringReport(emission_count=len(emissions))

    for detection in detections:
        claims_range = detection_claims_range(detection)
        report.range_claim_count += int(claims_range)
        event = scorer.assign(detection, emissions)
        if event is None:
            report.unassociated_range_claim_count += int(claims_range)
            continue
        result = results_by_event_id[event.event_id]
        result.heard = True
        if not detection_has_position_fix(detection):
            result.classified = result.classified or scorer.label_matches(detection, event)
            continue
        geometry = scorer.geometry(detection, event)
        result.true_range_m = geometry.true_range_m
        within_range_tolerance = geometry.range_error_m <= scorer.range_tolerance_m(geometry.true_range_m)
        if claims_range and not within_range_tolerance:
            report.out_of_tolerance_range_claim_count += 1
        if not scorer.label_matches(detection, event):
            continue
        result.classified = True
        if result.best_bearing_error_deg is None or geometry.bearing_error_deg < result.best_bearing_error_deg:
            result.best_bearing_error_deg = geometry.bearing_error_deg
        if geometry.bearing_error_deg > config.bearing_correct_max_error_deg:
            continue
        result.classified_bearing_correct = True
        if not claims_range:
            continue
        if result.best_range_error_m is None or geometry.range_error_m < result.best_range_error_m:
            result.best_range_error_m = geometry.range_error_m
        result.classified_range_resolved = result.classified_range_resolved or within_range_tolerance

    report.emission_results = list(results_by_event_id.values())
    report.heard_count = sum(result.heard for result in report.emission_results)
    report.classified_count = sum(result.classified for result in report.emission_results)
    report.classified_bearing_correct_count = sum(r.classified_bearing_correct for r in report.emission_results)
    report.classified_range_resolved_count = sum(r.classified_range_resolved for r in report.emission_results)
    report.heard_recall = _fraction(report.heard_count, report.emission_count)
    report.classified_bearing_correct_recall = _fraction(report.classified_bearing_correct_count, report.emission_count)
    report.classified_range_resolved_recall = _fraction(report.classified_range_resolved_count, report.emission_count)
    report.false_range_claim_count = report.out_of_tolerance_range_claim_count + report.unassociated_range_claim_count
    report.false_range_claim_rate = _fraction(report.false_range_claim_count, report.range_claim_count)
    report.errors = tier_gate_errors(report, config)
    return report


def tier_gate_errors(report: TierScoringReport, config: TierScoringConfig) -> list[str]:
    errors: list[str] = []
    recall_gates = (
        ("heard recall", report.heard_recall, config.min_heard_recall),
        ("classified+bearing-correct recall", report.classified_bearing_correct_recall,
         config.min_classified_bearing_correct_recall),
        ("classified+range-resolved recall", report.classified_range_resolved_recall,
         config.min_classified_range_resolved_recall),
    )
    for name, value, minimum in recall_gates:
        if minimum is None:
            continue
        if value is None:
            errors.append(f"{name} undefined (no ground-truth emissions) but gated at {minimum}")
        elif value < minimum:
            errors.append(f"{name} {value:.3f} below {minimum}")
    if config.max_false_range_claim_count is not None and report.false_range_claim_count > config.max_false_range_claim_count:
        errors.append(
            f"false range claims {report.false_range_claim_count} above {config.max_false_range_claim_count}"
        )
    if (
        config.max_false_range_claim_rate is not None
        and report.false_range_claim_rate is not None
        and report.false_range_claim_rate > config.max_false_range_claim_rate
    ):
        errors.append(
            f"false range claim rate {report.false_range_claim_rate:.3f} above {config.max_false_range_claim_rate}"
        )
    return errors
