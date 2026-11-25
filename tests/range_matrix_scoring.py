"""Score one range-matrix cell's stored detections against its oracle verdict.

Three checks, all on detections whose classifier render overlaps the event
(renders of pre-roll noise are not detections of it; the fake classifier
labels everything):

1. **Reach**: the best class achieved (tier scorer ladder, plus a 3-D fix
   when a range-claiming detection's full 3-D error is inside the range
   tolerance) must be at least the oracle's ``guaranteed_class``.
2. **No over-claim**: a detection claiming range, or a 3-D fix (claims range
   and its stated covariance resolves every axis at the oracle's honest
   sigma), must not claim more than the oracle's ``max_honest_class``.
3. **Honest uncertainty**: every positioned detection must state a valid
   covariance that contains the truth (squared Mahalanobis error under the
   chi-square quantile of ``CONTAINMENT_CONFIDENCE``). A range claim is
   checked in 3-D. A bearing-only fix (range explicitly unobservable, its
   point a placeholder along the ray) is checked on what it asserts: the
   truth's direction from the observer against the stated covariance's
   angular spread across the ray (2 dof). For NTP-timed and hand-measured
   cells this is the check that "widen the error" is honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

from minimappr.calibration.ground_truth_events import GroundTruthEvent, GroundTruthSpecies
from minimappr.sim.replay_entity_metrics import angular_difference_deg, bearing_deg
from minimappr.sim.replay_tier_scoring import (
    detection_claims_range,
    detection_has_position_fix,
    score_detection_tiers,
)
from minimappr.sim.replay_uncertainty_coverage import squared_mahalanobis_distance
from tests.range_matrix_cells import ORACLE_CONFIG, TIER_SCORING_CONFIG, RealizedCell
from tests.range_observability_oracle import ObservabilityVerdict, PassClass
from tests.rust_sidecar_e2e_helpers import FAKE_CLASSIFIER_LABEL

NANOSECONDS_PER_SECOND = 1_000_000_000
# One deterministic sample per cell: a high quantile, so only a covariance
# that is clearly too tight fails, never an honest 2-sigma draw.
CONTAINMENT_CONFIDENCE = 0.999
POSITION_DIMENSIONS = 3
BEARING_DIMENSIONS = 2
MINIMUM_RAY_LENGTH_M = 1e-3
FAKE_LABEL_CATEGORY = "unknown"


@dataclass
class CellOutcome:
    achieved_class: PassClass
    violations: list[str] = field(default_factory=list)
    detection_lines: list[str] = field(default_factory=list)


def event_arrival_span_ns(realized: RealizedCell, emission_packet_time_ns: int, burst_seconds: float) -> tuple[int, int]:
    first_arrival_s, last_arrival_end_s = realized.scene.emission_to_arrival_span_seconds(burst_seconds)
    return (
        emission_packet_time_ns + int(first_arrival_s * NANOSECONDS_PER_SECOND),
        emission_packet_time_ns + int(last_arrival_end_s * NANOSECONDS_PER_SECOND),
    )


def detections_overlapping_span(detections: list[dict], span_ns: tuple[int, int]) -> list[dict]:
    return [
        detection
        for detection in detections
        if int((detection.get("feature_summary") or {}).get("rust_render_start_ns") or 0) <= span_ns[1]
        and int((detection.get("feature_summary") or {}).get("rust_render_end_ns") or 0) >= span_ns[0]
    ]


def _claimed_class(detection: dict, range_tolerance_m: float) -> PassClass:
    if not detection_has_position_fix(detection):
        return PassClass.HEARD
    if not detection_claims_range(detection):
        return PassClass.BEARING_CORRECT
    covariance = detection.get("position_covariance_m2")
    if covariance is not None:
        largest_variance = float(np.max(np.linalg.eigvalsh(np.asarray(covariance, dtype=np.float64)[:3, :3])))
        if ORACLE_CONFIG.honest_sigma_multiplier * np.sqrt(max(largest_variance, 0.0)) <= range_tolerance_m:
            return PassClass.FIX_3D
    return PassClass.RANGE_RESOLVED


def _stated_std_text(covariance: np.ndarray) -> str:
    return str(np.round(np.sqrt(np.clip(np.diag(covariance), 0, None)), 2).tolist())


def _range_claim_containment_violation(position_m: np.ndarray, covariance: np.ndarray, truth_m: np.ndarray) -> str | None:
    error_m = position_m - truth_m
    distance_squared = squared_mahalanobis_distance(error_m, covariance)
    if distance_squared is None:
        return "stated covariance is singular, indefinite or non-finite"
    limit = float(chi2.ppf(CONTAINMENT_CONFIDENCE, df=POSITION_DIMENSIONS))
    if distance_squared > limit:
        return (
            f"truth outside stated uncertainty: Mahalanobis^2 {distance_squared:.1f} > {limit:.1f} "
            f"(error {np.round(error_m, 2).tolist()} m, stated std {_stated_std_text(covariance)} m)"
        )
    return None


def _bearing_containment_violation(
    position_m: np.ndarray, covariance: np.ndarray, truth_m: np.ndarray, observer_m: np.ndarray
) -> str | None:
    """Truth direction vs the stated angular spread across the claimed ray."""
    claimed_offset_m = position_m - observer_m
    claimed_distance_m = float(np.linalg.norm(claimed_offset_m))
    if claimed_distance_m < MINIMUM_RAY_LENGTH_M:
        return "bearing-only fix sits on the observer, so it asserts no direction"
    ray_unit = claimed_offset_m / claimed_distance_m
    truth_offset_m = truth_m - observer_m
    truth_unit = truth_offset_m / float(np.linalg.norm(truth_offset_m))
    if float(truth_unit @ ray_unit) <= 0.0:
        return f"bearing points away from the truth (claimed ray {np.round(ray_unit, 2).tolist()})"
    # Orthonormal basis of the plane across the ray (the two directions a bearing asserts).
    _u, _s, right_singular_vectors = np.linalg.svd(ray_unit.reshape(1, 3))
    across_ray = right_singular_vectors[1:].T
    angular_error_rad = across_ray.T @ truth_unit
    angular_covariance = across_ray.T @ covariance @ across_ray / claimed_distance_m**2
    distance_squared = squared_mahalanobis_distance(angular_error_rad, angular_covariance)
    if distance_squared is None:
        return "stated covariance is singular, indefinite or non-finite across the bearing"
    limit = float(chi2.ppf(CONTAINMENT_CONFIDENCE, df=BEARING_DIMENSIONS))
    if distance_squared > limit:
        return (
            f"truth direction outside stated bearing uncertainty: Mahalanobis^2 {distance_squared:.1f} > {limit:.1f} "
            f"(angular error {np.round(np.degrees(angular_error_rad), 2).tolist()} deg, stated angular std "
            f"{np.round(np.degrees(np.sqrt(np.clip(np.diag(angular_covariance), 0, None))), 2).tolist()} deg)"
        )
    return None


def _containment_violation(detection: dict, truth_m: np.ndarray, observer_m: np.ndarray) -> str | None:
    covariance = detection.get("position_covariance_m2")
    if covariance is None:
        return "positioned detection states no covariance"
    covariance = np.asarray(covariance, dtype=np.float64)[:3, :3]
    position_m = np.asarray(detection["position_m"], dtype=np.float64)[:3]
    if not np.all(np.isfinite(covariance)):
        return "stated covariance is non-finite"
    if detection_claims_range(detection):
        return _range_claim_containment_violation(position_m, covariance, truth_m)
    return _bearing_containment_violation(position_m, covariance, truth_m, observer_m)


def _detection_line(detection: dict, truth_m: np.ndarray, claimed: PassClass) -> str:
    feature_summary = detection.get("feature_summary") or {}
    position = detection.get("position_m")
    error = (
        f"{float(np.linalg.norm(np.asarray(position, dtype=np.float64)[:3] - truth_m)):.2f} m"
        if position is not None
        else "n/a"
    )
    return (
        f"{detection.get('source_node_id')} modality={detection.get('reporting_modality')} "
        f"mode={feature_summary.get('localization_range_projection_mode')} "
        f"method={feature_summary.get('localization_method')} error={error} claimed={claimed.name}"
    )


def score_cell(
    realized: RealizedCell,
    verdict: ObservabilityVerdict,
    detections: list[dict],
    *,
    emission_packet_time_ns: int,
    burst_seconds: float,
) -> CellOutcome:
    span_ns = event_arrival_span_ns(realized, emission_packet_time_ns, burst_seconds)
    truth_m = np.asarray(realized.scene.source_position_enu_m, dtype=np.float64)
    event = GroundTruthEvent(
        event_id=realized.cell.cell_id,
        individual_id=realized.cell.cell_id,
        label=FAKE_CLASSIFIER_LABEL,
        label_category=FAKE_LABEL_CATEGORY,
        species=[GroundTruthSpecies(label=FAKE_CLASSIFIER_LABEL, label_category=FAKE_LABEL_CATEGORY)],
        start_ns=span_ns[0],
        end_ns=span_ns[1],
        waypoints_local_m=[(span_ns[0], truth_m)],
    )
    event_detections = detections_overlapping_span(detections, span_ns)
    single_node = len(realized.reported_nodes) == 1
    observers = (
        {realized.reported_nodes[0].scene_node.node_id: realized.observer_position_m} if single_node else {}
    )
    report = score_detection_tiers(
        event_detections,
        [event],
        TIER_SCORING_CONFIG,
        observer_positions_m=observers,
        array_centroid_m=realized.observer_position_m,
    )
    tier = report.emission_results[0]
    achieved = PassClass.NOT_HEARD
    if tier.heard:
        achieved = PassClass.HEARD
    if tier.classified_bearing_correct:
        achieved = PassClass.BEARING_CORRECT
    if tier.classified_range_resolved:
        achieved = PassClass.RANGE_RESOLVED
    outcome = CellOutcome(achieved_class=achieved)
    for detection in event_detections:
        claimed = _claimed_class(detection, verdict.range_tolerance_m)
        outcome.detection_lines.append(_detection_line(detection, truth_m, claimed))
        if not detection_has_position_fix(detection):
            continue
        position_m = np.asarray(detection["position_m"], dtype=np.float64)[:3]
        observer_m = np.asarray(verdict.observer_position_m)
        bearing_error_deg = angular_difference_deg(bearing_deg(observer_m, position_m), bearing_deg(observer_m, truth_m))
        if (
            achieved == PassClass.RANGE_RESOLVED
            and detection_claims_range(detection)
            and bearing_error_deg <= TIER_SCORING_CONFIG.bearing_correct_max_error_deg
            and float(np.linalg.norm(position_m - truth_m)) <= verdict.range_tolerance_m
        ):
            outcome.achieved_class = PassClass.FIX_3D
        if claimed >= PassClass.RANGE_RESOLVED and claimed > verdict.max_honest_class:
            outcome.violations.append(
                f"{detection.get('source_node_id')}: claims {claimed.name} but the oracle allows at most "
                f"{verdict.max_honest_class.name} ({_detection_line(detection, truth_m, claimed)})"
            )
        containment = _containment_violation(detection, truth_m, observer_m)
        if containment is not None:
            outcome.violations.append(f"{detection.get('source_node_id')}: {containment}")
    if outcome.achieved_class < verdict.guaranteed_class:
        outcome.violations.append(
            f"reached {outcome.achieved_class.name}, oracle guarantees {verdict.guaranteed_class.name} "
            f"(best bearing error {tier.best_bearing_error_deg}, best range error {tier.best_range_error_m}, "
            f"{len(event_detections)} event detections of {len(detections)})"
        )
    return outcome
