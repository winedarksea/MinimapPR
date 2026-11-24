"""Compare two E10 hot-path gate results (baseline vs candidate) under the gate tolerances."""

from __future__ import annotations

from dataclasses import dataclass, field

from minimappr.sim.replay_gate_config import HotPathGateConfig, HotPathMetricTolerance

# Workload fields that must match: two results from different inputs are not comparable.
SCENARIO_IDENTITY_FIELDS = ("name", "duration_seconds", "nodes", "sample_rate_hz", "frame_samples", "frames_per_post",
                            "slow_classifier_delay_seconds", "signal_seed")


@dataclass(frozen=True)
class MetricComparison:
    metric_name: str
    baseline_value: float | None
    candidate_value: float | None
    allowed_maximum: float | None
    gated: bool
    passed: bool


@dataclass
class HotPathComparison:
    rows: list[MetricComparison] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def format_table(self) -> str:
        lines = [f"{'metric':42s} {'baseline':>12s} {'candidate':>12s} {'allowed':>12s}  verdict"]
        for row in self.rows:
            verdict = ("ok" if row.passed else "FAIL") if row.gated else "info"
            lines.append(
                f"{row.metric_name:42s} {_format(row.baseline_value):>12s} {_format(row.candidate_value):>12s} "
                f"{_format(row.allowed_maximum):>12s}  {verdict}"
            )
        lines.extend(f"WARNING: {warning}" for warning in self.warnings)
        lines.extend(f"ERROR: {error}" for error in self.errors)
        return "\n".join(lines)


def _format(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _compare_metric(
    metric_name: str, baseline_value: float | None, candidate_value: float | None, tolerance: HotPathMetricTolerance
) -> tuple[MetricComparison, str | None]:
    if baseline_value is None:
        # Nothing to be relative to (e.g. no solves at baseline): only a ceiling can judge.
        allowed = tolerance.absolute_ceiling
    else:
        allowed = tolerance.allowed_maximum(float(baseline_value))
    if not tolerance.gated:
        return MetricComparison(metric_name, baseline_value, candidate_value, allowed, False, True), None
    if candidate_value is None:
        error = f"{metric_name}: candidate has no value" if baseline_value is not None else None
        return MetricComparison(metric_name, baseline_value, candidate_value, allowed, True, error is None), error
    passed = allowed is None or float(candidate_value) <= allowed
    error = None if passed else f"{metric_name}: candidate {candidate_value:.3f} above allowed {allowed:.3f} (baseline {_format(baseline_value)})"
    return MetricComparison(metric_name, baseline_value, candidate_value, allowed, True, passed), error


def compare_hot_path_results(baseline: dict, candidate: dict, config: HotPathGateConfig) -> HotPathComparison:
    comparison = HotPathComparison()
    if baseline.get("schema_version") != candidate.get("schema_version"):
        comparison.errors.append(
            f"result schema {candidate.get('schema_version')} differs from baseline {baseline.get('schema_version')}"
        )
    for field_name in SCENARIO_IDENTITY_FIELDS:
        if baseline["scenario"].get(field_name) != candidate["scenario"].get(field_name):
            comparison.errors.append(f"scenario field {field_name!r} differs: results measure different workloads")
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        if not result.get("observer_complete", False):
            comparison.errors.append(f"{label} run's stream observer missed events; its counts are not trustworthy")
    if baseline.get("machine") != candidate.get("machine"):
        comparison.warnings.append("baseline and candidate ran on different machines; latency and RSS are not comparable")
    if baseline.get("sidecar_build_profile") != candidate.get("sidecar_build_profile"):
        comparison.warnings.append("sidecar build profiles differ")

    baseline_metrics, candidate_metrics = baseline["metrics"], candidate["metrics"]
    baseline_solves = baseline_metrics.get("solve_count") or 0
    candidate_solves = candidate_metrics.get("solve_count") or 0
    if candidate_solves < config.min_candidate_solve_count_fraction * baseline_solves:
        comparison.errors.append(
            f"candidate produced {candidate_solves} solves vs baseline {baseline_solves}: "
            "latency percentiles would describe a different workload"
        )
    metric_names = list(config.metric_tolerances) + sorted(
        (set(baseline_metrics) | set(candidate_metrics)) - set(config.metric_tolerances)
    )
    for metric_name in metric_names:
        tolerance = config.metric_tolerances.get(metric_name, HotPathMetricTolerance(gated=False))
        row, error = _compare_metric(
            metric_name, baseline_metrics.get(metric_name), candidate_metrics.get(metric_name), tolerance
        )
        comparison.rows.append(row)
        if error:
            comparison.errors.append(error)
    return comparison
