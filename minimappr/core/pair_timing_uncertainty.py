"""One clock policy for TDOA pairs on both ingest paths (plan item P9).

Owner decision: weak clocks admit every pair and widen the error; clock class
never excludes a pair. The rules mirror the sidecar's (``dsp_node_clock_assessment.rs``,
``dsp_solve_pair_admission.rs``) so a site behaves the same on either path:

- node timing uncertainty = the larger of its clock-class default and its
  reported holdover predicted error. A class missing from the configured map,
  or no reported class, takes ``localization_clock_unknown_uncertainty_seconds``
  (the map replaces the defaults; it does not merge, same as Rust).
- pair timing uncertainty = 0 for a same-node pair (one clock cancels out),
  quadrature of the two node uncertainties for a cross-node pair (independent
  clocks). A pair whose node membership is unknown cannot be shown to span two
  clocks and gets 0, matching the old gate, which also only acted on known nodes.
- pair weight = inverse variance, relative to the Cartesian solver's reference
  timing noise (one interpolated sample). The correlation quality ``q`` scales
  the measurement variance as ``sigma_ref**2 / q``, so with no timing
  uncertainty the weight is exactly ``q`` (the pre-P9 weight). Because weights
  are relative to the same ``sigma_ref`` the solver turns into its covariance,
  the published covariance widens by the timing variance with no extra step.

Only a *measured* timeline residual over the hard limit drops a pair, and only
the sidecar measures one; the Python ingest path has no such gate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from minimappr.models import SyncGrade, TimeQuality

NANOSECONDS_PER_SECOND = 1.0e9
# Floor on a pair's correlation/geometry quality before the timing term is
# combined, so a near-zero product cannot zero a pair. Pre-P9 weights never fell
# below it, so a pair set whose best weight does is clock-dominated.
MINIMUM_PAIR_MEASUREMENT_QUALITY = 0.01
# Firmware ``clock_holdover`` diagnostics key (binary section 0x0010 and the JSON
# suffix); ``predicted_error_ns`` is 0 unless holdover is active.
CLOCK_HOLDOVER_DIAGNOSTICS_KEY = "clock_holdover"
HOLDOVER_PREDICTED_ERROR_NS_KEY = "predicted_error_ns"

# Legacy database value; the sidecar parses it as ntp_disciplined too.
_TIME_QUALITY_ALIASES: dict[str, str] = {TimeQuality.NTP_SYNC.value: TimeQuality.NTP_DISCIPLINED.value}

# A cluster's declared sync grade is an operator ceiling on clock quality, so it
# floors each member's uncertainty at the matching clock class. PTP has no
# time_quality of its own; it is microsecond-class like a locked GPS clock.
_TIME_QUALITY_FOR_DECLARED_SYNC_GRADE: dict[SyncGrade, str] = {
    SyncGrade.GPS_PPS: TimeQuality.GPS_LOCKED.value,
    SyncGrade.PTP: TimeQuality.GPS_LOCKED.value,
    SyncGrade.NTP: TimeQuality.NTP_DISCIPLINED.value,
    SyncGrade.FREE: TimeQuality.FREE_RUNNING.value,
}


@dataclass(frozen=True, slots=True)
class ClockUncertaintyPolicy:
    class_uncertainty_seconds: Mapping[str, float]
    unknown_uncertainty_seconds: float

    @classmethod
    def from_settings(cls, settings: Any) -> "ClockUncertaintyPolicy":
        return cls(
            class_uncertainty_seconds=dict(settings.localization_clock_class_uncertainty_seconds),
            unknown_uncertainty_seconds=float(settings.localization_clock_unknown_uncertainty_seconds),
        )

    def class_uncertainty_for(self, time_quality: str | None) -> float:
        if time_quality is None:
            return self.unknown_uncertainty_seconds
        canonical_time_quality = _TIME_QUALITY_ALIASES.get(time_quality, time_quality)
        return float(self.class_uncertainty_seconds.get(canonical_time_quality, self.unknown_uncertainty_seconds))


@dataclass(frozen=True, slots=True)
class NodeClockEvidence:
    """What a node last reported about its clock."""

    time_quality: str | None
    holdover_predicted_error_seconds: float | None = None


def node_clock_evidence_from_frame(time_quality: Any, timing_diagnostics: Mapping[str, Any] | None) -> NodeClockEvidence:
    return NodeClockEvidence(
        time_quality=_time_quality_value(time_quality),
        holdover_predicted_error_seconds=holdover_predicted_error_seconds(timing_diagnostics),
    )


def holdover_predicted_error_seconds(timing_diagnostics: Mapping[str, Any] | None) -> float | None:
    if not isinstance(timing_diagnostics, Mapping):
        return None
    holdover = timing_diagnostics.get(CLOCK_HOLDOVER_DIAGNOSTICS_KEY)
    if not isinstance(holdover, Mapping):
        return None
    predicted_error_ns = holdover.get(HOLDOVER_PREDICTED_ERROR_NS_KEY)
    if isinstance(predicted_error_ns, bool) or not isinstance(predicted_error_ns, (int, float)):
        return None
    if not math.isfinite(predicted_error_ns) or predicted_error_ns < 0:
        return None
    return float(predicted_error_ns) / NANOSECONDS_PER_SECOND


def node_timing_uncertainty_seconds(evidence: NodeClockEvidence | None, policy: ClockUncertaintyPolicy) -> float:
    if evidence is None:
        return policy.unknown_uncertainty_seconds
    uncertainty_seconds = policy.class_uncertainty_for(evidence.time_quality)
    if evidence.holdover_predicted_error_seconds is not None:
        uncertainty_seconds = max(uncertainty_seconds, evidence.holdover_predicted_error_seconds)
    return uncertainty_seconds


def declared_sync_grade_uncertainty_seconds(grade: SyncGrade, policy: ClockUncertaintyPolicy) -> float:
    return policy.class_uncertainty_for(_TIME_QUALITY_FOR_DECLARED_SYNC_GRADE[grade])


def sensors_spanning_two_nodes(
    sensor_a: str,
    sensor_b: str,
    sensor_node_ids: Mapping[str, str] | None,
) -> bool:
    if sensor_node_ids is None:
        return False
    node_a = sensor_node_ids.get(sensor_a)
    node_b = sensor_node_ids.get(sensor_b)
    return node_a is not None and node_b is not None and node_a != node_b


def pair_timing_uncertainty_seconds(
    sensor_a: str,
    sensor_b: str,
    *,
    sensor_node_ids: Mapping[str, str] | None,
    sensor_timing_uncertainty_seconds: Mapping[str, float] | None,
) -> float:
    if sensor_timing_uncertainty_seconds is None or not sensors_spanning_two_nodes(sensor_a, sensor_b, sensor_node_ids):
        return 0.0
    return cross_node_pair_timing_uncertainty_seconds(
        float(sensor_timing_uncertainty_seconds.get(sensor_a, 0.0)),
        float(sensor_timing_uncertainty_seconds.get(sensor_b, 0.0)),
    )


def cross_node_pair_timing_uncertainty_seconds(node_a_uncertainty_seconds: float, node_b_uncertainty_seconds: float) -> float:
    return math.hypot(node_a_uncertainty_seconds, node_b_uncertainty_seconds)


def solver_reference_time_std_seconds(sample_rate_hz: int, interpolation_factor: int) -> float:
    """One interpolated sample: the timing noise ``solve_cartesian_tdoa`` assumes."""
    return 1.0 / max(sample_rate_hz * max(interpolation_factor, 1), 1)


def inverse_variance_pair_weight(
    *,
    measurement_quality: float,
    timing_uncertainty_seconds: float,
    reference_time_std_seconds: float,
) -> float:
    timing_variance_ratio = (timing_uncertainty_seconds / reference_time_std_seconds) ** 2
    return 1.0 / ((1.0 / measurement_quality) + timing_variance_ratio)


def optimizer_residual_scale(pair_weights: Iterable[float]) -> float:
    """Factor that lifts a clock-dominated residual vector back to unit scale.

    Weights stay absolute so the covariance widens, but scipy's ``least_squares``
    stops on absolute gradient tolerances: when every pair is weak (e.g. a
    point-only network on NTP clocks) the scaled residuals are so small the
    solver halts at its start point. A uniform factor leaves the least-squares
    minimiser unchanged, so callers multiply only the residuals they optimise.
    Sets with any pre-P9-strength pair are untouched (factor 1).
    """
    strongest_weight = max(pair_weights, default=1.0)
    if strongest_weight <= 0.0 or strongest_weight >= MINIMUM_PAIR_MEASUREMENT_QUALITY:
        return 1.0
    return 1.0 / math.sqrt(strongest_weight)


def pair_information_score(
    *,
    baseline_m: float,
    timing_uncertainty_seconds: float,
    reference_time_std_seconds: float,
) -> float:
    """Baseline over total timing std: how much a pair can constrain direction.

    Used to rank pairs before they are measured. With no timing uncertainty the
    ranking is by baseline alone, as before P9.
    """
    total_time_std_seconds = math.hypot(reference_time_std_seconds, timing_uncertainty_seconds)
    return baseline_m * (reference_time_std_seconds / total_time_std_seconds)


def _time_quality_value(time_quality: Any) -> str | None:
    if time_quality is None:
        return None
    return str(getattr(time_quality, "value", time_quality))
