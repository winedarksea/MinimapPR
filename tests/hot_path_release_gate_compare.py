"""Compare two E10 hot-path gate results (baseline vs candidate) under the gate tolerances."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

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
    allowed_minimum: float | None = None


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
            allowed = f">={_format(row.allowed_minimum)}" if row.allowed_minimum is not None else _format(row.allowed_maximum)
            lines.append(
                f"{row.metric_name:42s} {_format(row.baseline_value):>12s} {_format(row.candidate_value):>12s} "
                f"{allowed:>12s}  {verdict}"
            )
        lines.extend(f"WARNING: {warning}" for warning in self.warnings)
        lines.extend(f"ERROR: {error}" for error in self.errors)
        return "\n".join(lines)


_HARDWARE_IDENTITY_FIELDS = ("machine", "cpu_model", "logical_cpu_count", "memory_gib")


def _hardware_identity(machine: dict | None) -> tuple:
    # Older results also carry OS/Python strings; only hardware decides comparability.
    machine = machine or {}
    return tuple(machine.get(field_name) for field_name in _HARDWARE_IDENTITY_FIELDS)


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


KNOWN_CALL_DETECTION_METRIC = "known_call_detection_count"
CLASSIFICATION_BACKLOG_DROP_METRIC = "classification_backlog_drop_count"


def _compare_absolute_detection_requirements(
    baseline_metrics: dict, candidate_metrics: dict, config: HotPathGateConfig
) -> list[tuple[MetricComparison, str | None]]:
    """Baseline-independent checks that the classification path delivered.

    A missing value fails: it means the harness did not measure the metric
    (e.g. a result recorded before it existed), and passing would let a run
    that stored nothing through.
    """
    results = []
    known_calls = candidate_metrics.get(KNOWN_CALL_DETECTION_METRIC)
    floor = float(config.min_candidate_known_call_detection_count)
    if known_calls is None:
        error = f"{KNOWN_CALL_DETECTION_METRIC}: candidate has no value; re-record it with the current harness"
    elif float(known_calls) < floor:
        error = (
            f"{KNOWN_CALL_DETECTION_METRIC}: only {known_calls:.0f} known calls reached stored detections "
            f"(need >= {floor:.0f}); solves without detections mean the classification handoff is losing calls"
        )
    else:
        error = None
    results.append((
        MetricComparison(KNOWN_CALL_DETECTION_METRIC, baseline_metrics.get(KNOWN_CALL_DETECTION_METRIC), known_calls,
                         None, True, error is None, allowed_minimum=floor),
        error,
    ))
    backlog_drops = candidate_metrics.get(CLASSIFICATION_BACKLOG_DROP_METRIC)
    budget = float(config.max_candidate_classification_backlog_drop_count)
    if backlog_drops is None:
        error = f"{CLASSIFICATION_BACKLOG_DROP_METRIC}: candidate has no value; re-record it with the current harness"
    elif float(backlog_drops) > budget:
        error = f"{CLASSIFICATION_BACKLOG_DROP_METRIC}: candidate dropped {backlog_drops:.0f} classifications (budget {budget:.0f})"
    else:
        error = None
    results.append((
        MetricComparison(CLASSIFICATION_BACKLOG_DROP_METRIC, baseline_metrics.get(CLASSIFICATION_BACKLOG_DROP_METRIC),
                         backlog_drops, budget, True, error is None),
        error,
    ))
    return results


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
    same_machine = _hardware_identity(baseline.get("machine")) == _hardware_identity(candidate.get("machine"))
    if not same_machine:
        comparison.warnings.append(
            "baseline and candidate ran on different machines; latency and RSS are reported but not gated"
        )
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
    for row, error in _compare_absolute_detection_requirements(baseline_metrics, candidate_metrics, config):
        comparison.rows.append(row)
        if error:
            comparison.errors.append(error)
    absolute_requirement_metrics = {KNOWN_CALL_DETECTION_METRIC, CLASSIFICATION_BACKLOG_DROP_METRIC}
    for metric_name in (name for name in metric_names if name not in absolute_requirement_metrics):
        tolerance = config.metric_tolerances.get(metric_name, HotPathMetricTolerance(gated=False))
        if tolerance.depends_on_machine and not same_machine:
            tolerance = replace(tolerance, gated=False)
        row, error = _compare_metric(
            metric_name, baseline_metrics.get(metric_name), candidate_metrics.get(metric_name), tolerance
        )
        comparison.rows.append(row)
        if error:
            comparison.errors.append(error)
    return comparison
