"""Thresholds for the replay scoring gates (E3) and the hot-path release gate (E10).

Every number a replay gate compares against lives here, so a gate change is a
reviewable config diff rather than an edited literal inside scoring code.

Gate thresholds typed ``X | None`` are opt-in: ``None`` means "report only".
They stay ``None`` by default until a baseline exists for the scene family,
mirroring ``DEFAULT_ENTITY_EXPECTATIONS`` in ``replay_entity_metrics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any

LABEL_MATCH_MODE_EXACT = "exact"
LABEL_MATCH_MODE_CATEGORY = "category"


@dataclass(frozen=True)
class TierScoringConfig:
    """Tier ladder: heard -> classified and bearing-correct -> classified and range-resolved.

    A detection's bearing and range are measured from its observer (reporting
    node position, or the array centroid for cross-node solves with no single
    observer). Range tolerance is ``max(absolute, relative * true_range)``:
    TDOA range error grows with distance, so a fixed metre bound would fail
    every honest far fix and pass every sloppy near one.
    """

    association_time_slop_seconds: float = 2.0
    bearing_correct_max_error_deg: float = 10.0
    range_resolved_max_absolute_error_m: float = 3.0
    range_resolved_max_relative_error: float = 0.25
    label_match_mode: str = LABEL_MATCH_MODE_CATEGORY
    # Position error on (E, N) only: planar layouts barely constrain height.
    horizontal_only: bool = False
    min_heard_recall: float | None = None
    min_classified_bearing_correct_recall: float | None = None
    min_classified_range_resolved_recall: float | None = None
    max_false_range_claim_rate: float | None = None
    max_false_range_claim_count: int | None = None

    def __post_init__(self) -> None:
        if not self.association_time_slop_seconds >= 0.0:
            raise ValueError("association_time_slop_seconds must be >= 0")
        if not self.bearing_correct_max_error_deg > 0.0:
            raise ValueError("bearing_correct_max_error_deg must be > 0")
        if not self.range_resolved_max_absolute_error_m > 0.0:
            raise ValueError("range_resolved_max_absolute_error_m must be > 0")
        if not self.range_resolved_max_relative_error >= 0.0:
            raise ValueError("range_resolved_max_relative_error must be >= 0")
        if self.label_match_mode not in (LABEL_MATCH_MODE_EXACT, LABEL_MATCH_MODE_CATEGORY):
            raise ValueError(f"label_match_mode must be 'exact' or 'category', got {self.label_match_mode!r}")


@dataclass(frozen=True)
class UncertaintyCoverageConfig:
    """Statistical check that stated covariances contain the truth as often as claimed.

    Each held-out sample is "covered" when its squared Mahalanobis error is at
    most the chi-square quantile of ``stated_confidence_level``. Under honest
    covariances the covered count is Binomial(n, level); the gate fails when the
    two-sided binomial test rejects that at ``binomial_test_significance``.
    Over-coverage (too conservative) is reported but only fails when
    ``fail_on_over_coverage`` is set: the owner policy is "widen the error"
    for weak clocks and hand-measured positions.
    """

    stated_confidence_level: float = 0.95
    binomial_test_significance: float = 0.01
    minimum_held_out_sample_count: int = 30
    horizontal_only: bool = False
    fail_on_over_coverage: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.stated_confidence_level < 1.0:
            raise ValueError("stated_confidence_level must be in (0, 1)")
        if not 0.0 < self.binomial_test_significance < 1.0:
            raise ValueError("binomial_test_significance must be in (0, 1)")
        if self.minimum_held_out_sample_count < 1:
            raise ValueError("minimum_held_out_sample_count must be >= 1")

    @property
    def position_dimension_count(self) -> int:
        return 2 if self.horizontal_only else 3


@dataclass(frozen=True)
class TrackMetricsConfig:
    """Track continuity scoring against continuous ground-truth trajectories.

    A track update is attributed to the truth individual nearest to it (within
    ``association_radius_m``) among those active at the update time.
    ``fragment_break_gap_seconds`` is the update silence that splits one
    individual's coverage into separate fragments.
    """

    association_radius_m: float = 15.0
    association_time_slop_seconds: float = 1.0
    fragment_break_gap_seconds: float = 5.0
    horizontal_only: bool = False
    update_gap_percentile: float = 95.0
    max_update_gap_seconds: float | None = None
    max_identity_switches: int | None = None
    max_fragmentations: int | None = None
    min_tracked_time_fraction: float | None = None

    def __post_init__(self) -> None:
        if not self.association_radius_m > 0.0:
            raise ValueError("association_radius_m must be > 0")
        if not self.association_time_slop_seconds >= 0.0:
            raise ValueError("association_time_slop_seconds must be >= 0")
        if not self.fragment_break_gap_seconds > 0.0:
            raise ValueError("fragment_break_gap_seconds must be > 0")
        if not 0.0 < self.update_gap_percentile <= 100.0:
            raise ValueError("update_gap_percentile must be in (0, 100]")


@dataclass(frozen=True)
class HotPathMetricTolerance:
    """How far a candidate metric may move from the baseline before the gate fails.

    The candidate passes when
    ``candidate <= baseline * (1 + max_relative_increase) + max_absolute_increase``
    and, when ``absolute_ceiling`` is set, also ``candidate <= absolute_ceiling``.
    The ceiling exists for metrics with a correct value independent of the
    baseline (raw-audio disk writes in memory-only mode must stay ~0 even if a
    baseline regressed).
    """

    max_relative_increase: float = 0.0
    max_absolute_increase: float = 0.0
    absolute_ceiling: float | None = None
    gated: bool = True
    # Zero-tolerance counters take the worst run: a packet gap or disk write in
    # one of N repeats is a real regression that a median would hide.
    aggregate_worst_case_across_runs: bool = False
    # Wall-clock and allocator metrics only compare on the same hardware; on a
    # different machine they are reported but not gated.
    depends_on_machine: bool = False

    def allowed_maximum(self, baseline_value: float) -> float:
        allowed = baseline_value * (1.0 + self.max_relative_increase) + self.max_absolute_increase
        if self.absolute_ceiling is not None:
            allowed = min(allowed, self.absolute_ceiling)
        return allowed


def _default_hot_path_tolerances() -> dict[str, HotPathMetricTolerance]:
    # Counters that should be zero on a healthy box get no or tiny absolute
    # slack. Latency and RSS are wall-clock / allocator noisy on a shared dev
    # machine (debug sidecar build, other builds running), so they get relative
    # headroom plus an absolute floor and the median of repeated runs; counters
    # take the worst run.
    counter = {"aggregate_worst_case_across_runs": True}
    timing = {"depends_on_machine": True}
    return {
        "packet_gap_count": HotPathMetricTolerance(**counter),
        "app_stream_reconnect_count": HotPathMetricTolerance(**counter),
        "ingest_rejected_post_count": HotPathMetricTolerance(**counter),
        "ingest_queue_drop_count": HotPathMetricTolerance(max_absolute_increase=2.0, **counter),
        "buffer_discontinuity_count": HotPathMetricTolerance(max_absolute_increase=2.0, **counter),
        "solve_latency_p50_ms": HotPathMetricTolerance(max_relative_increase=0.30, max_absolute_increase=75.0, **timing),
        "solve_latency_p95_ms": HotPathMetricTolerance(max_relative_increase=0.50, max_absolute_increase=250.0, **timing),
        "ingest_post_round_trip_p50_ms": HotPathMetricTolerance(max_relative_increase=0.50, max_absolute_increase=5.0, **timing),
        "ingest_post_round_trip_p95_ms": HotPathMetricTolerance(max_relative_increase=0.50, max_absolute_increase=20.0, **timing),
        "sidecar_peak_rss_mb": HotPathMetricTolerance(max_relative_increase=0.25, max_absolute_increase=32.0, **timing),
        "app_peak_rss_mb": HotPathMetricTolerance(max_relative_increase=0.25, max_absolute_increase=64.0, **timing),
        "raw_audio_disk_bytes_written": HotPathMetricTolerance(absolute_ceiling=0.0, **counter),
        "raw_audio_disk_files_written": HotPathMetricTolerance(absolute_ceiling=0.0, **counter),
        "raw_audio_out_of_order_span_count": HotPathMetricTolerance(gated=False),
        "solve_count": HotPathMetricTolerance(gated=False),
        "detection_count": HotPathMetricTolerance(gated=False),
        "detection_snippet_bytes_written": HotPathMetricTolerance(gated=False),
        "classification_backlog_drop_count": HotPathMetricTolerance(gated=False),
        "classification_max_completed_lag_seconds": HotPathMetricTolerance(gated=False),
    }


@dataclass(frozen=True)
class HotPathGateConfig:
    """E10 tolerances, keyed by the metric names the harness writes to its JSON."""

    metric_tolerances: dict[str, HotPathMetricTolerance] = field(default_factory=_default_hot_path_tolerances)
    # A candidate that lost this fraction of the baseline's solves is not
    # measuring the same workload; its latency percentiles would be misleading.
    min_candidate_solve_count_fraction: float = 0.5


@dataclass(frozen=True)
class ReplayGateConfig:
    tiers: TierScoringConfig = field(default_factory=TierScoringConfig)
    uncertainty_coverage: UncertaintyCoverageConfig = field(default_factory=UncertaintyCoverageConfig)
    tracks: TrackMetricsConfig = field(default_factory=TrackMetricsConfig)
    hot_path: HotPathGateConfig = field(default_factory=HotPathGateConfig)


def _replace_known_fields(instance: Any, overrides: dict[str, Any], section_name: str) -> Any:
    known_field_names = {dataclass_field.name for dataclass_field in fields(instance)}
    unknown_field_names = set(overrides) - known_field_names
    # A typo in a gate override would otherwise silently leave the default in force.
    if unknown_field_names:
        raise ValueError(f"unknown {section_name} gate fields: {sorted(unknown_field_names)}")
    return replace(instance, **overrides)


def replay_gate_config_from_mapping(mapping: dict[str, Any] | None) -> ReplayGateConfig:
    """Build a config from ``{"tiers": {...}, "uncertainty_coverage": {...}, "tracks": {...},
    "hot_path": {"metric_tolerances": {name: {...}}, ...}}``; omitted keys keep defaults."""
    mapping = mapping or {}
    defaults = ReplayGateConfig()
    hot_path_mapping = dict(mapping.get("hot_path") or {})
    tolerance_overrides = hot_path_mapping.pop("metric_tolerances", {}) or {}
    metric_tolerances = dict(defaults.hot_path.metric_tolerances)
    for metric_name, tolerance_mapping in tolerance_overrides.items():
        base_tolerance = metric_tolerances.get(metric_name, HotPathMetricTolerance())
        metric_tolerances[metric_name] = _replace_known_fields(
            base_tolerance, tolerance_mapping, f"hot_path.metric_tolerances.{metric_name}"
        )
    hot_path = _replace_known_fields(
        replace(defaults.hot_path, metric_tolerances=metric_tolerances), hot_path_mapping, "hot_path"
    )
    return ReplayGateConfig(
        tiers=_replace_known_fields(defaults.tiers, dict(mapping.get("tiers") or {}), "tiers"),
        uncertainty_coverage=_replace_known_fields(
            defaults.uncertainty_coverage, dict(mapping.get("uncertainty_coverage") or {}), "uncertainty_coverage"
        ),
        tracks=_replace_known_fields(defaults.tracks, dict(mapping.get("tracks") or {}), "tracks"),
        hot_path=hot_path,
    )
