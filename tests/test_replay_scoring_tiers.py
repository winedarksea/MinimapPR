"""E3 tier scoring, uncertainty coverage, track metrics and gate config on synthetic inputs."""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.calibration.ground_truth_events import GroundTruthEvent, GroundTruthSpecies
from minimappr.sim.replay_gate_config import (
    HotPathMetricTolerance,
    TierScoringConfig,
    TrackMetricsConfig,
    UncertaintyCoverageConfig,
    replay_gate_config_from_mapping,
)
from minimappr.sim.replay_tier_scoring import detection_claims_range, score_detection_tiers
from minimappr.sim.replay_track_metrics import TrackUpdate, compute_track_metrics, track_updates_from_detections
from minimappr.sim.replay_uncertainty_coverage import (
    COVERAGE_VERDICT_CONSISTENT,
    COVERAGE_VERDICT_OVER_COVERED,
    COVERAGE_VERDICT_UNDER_COVERED,
    CoverageSample,
    coverage_sample_from_detection,
    measure_uncertainty_coverage,
    partition_held_out_seeds,
    squared_mahalanobis_distance,
)

SECOND_NS = 1_000_000_000
OBSERVER_NODE_ID = "array-a"
OBSERVER_POSITION_M = np.zeros(3)


def _event(
    event_id: str,
    *,
    label: str = "robin",
    category: str = "bird",
    start_s: float = 0.0,
    end_s: float = 1.0,
    waypoints: list[tuple[float, tuple[float, float, float]]] | None = None,
    individual_id: str | None = None,
    is_echo_source: bool = False,
) -> GroundTruthEvent:
    waypoints = waypoints or [(start_s, (30.0, 0.0, 0.0))]
    return GroundTruthEvent(
        event_id=event_id,
        individual_id=individual_id or event_id,
        label=label,
        label_category=category,
        species=[GroundTruthSpecies(label=label, label_category=category)],
        start_ns=int(start_s * SECOND_NS),
        end_ns=int(end_s * SECOND_NS),
        waypoints_local_m=[(int(t * SECOND_NS), np.asarray(p, dtype=np.float64)) for t, p in waypoints],
        is_echo_source=is_echo_source,
    )


def _detection(
    *,
    timestamp_s: float = 0.5,
    label: str = "robin",
    category: str = "bird",
    position_m: tuple[float, float, float] | None = (30.0, 0.0, 0.0),
    modality: str = "localized",
    range_mode: str | None = "range_refined",
    track_id: str | None = None,
) -> dict:
    return {
        "timestamp_ns": int(timestamp_s * SECOND_NS),
        "label": label,
        "label_category": category,
        "position_m": position_m,
        "reporting_modality": modality,
        "source_node_id": OBSERVER_NODE_ID,
        "track_id": track_id,
        "feature_summary": {"localization_range_projection_mode": range_mode} if range_mode else {},
    }


def _score(detections: list[dict], events: list[GroundTruthEvent], **config_overrides):
    return score_detection_tiers(
        detections,
        events,
        TierScoringConfig(**config_overrides),
        observer_positions_m={OBSERVER_NODE_ID: OBSERVER_POSITION_M},
        array_centroid_m=np.array([100.0, 100.0, 0.0]),
    )


# ---- tier scoring ---------------------------------------------------------------------------


def test_tier_ladder_separates_heard_bearing_and_range():
    events = [_event("omni-only"), _event("bearing-only", start_s=10, end_s=11), _event("ranged", start_s=20, end_s=21),
              _event("missed", start_s=30, end_s=31)]
    detections = [
        _detection(timestamp_s=0.5, modality="omni", position_m=(0.0, 0.0, 0.0)),
        # Right direction, range unobservable: bearing tier but never a range claim.
        _detection(timestamp_s=10.5, position_m=(300.0, 2.0, 0.0), range_mode="range_asymptotic"),
        _detection(timestamp_s=20.5, position_m=(31.0, 1.0, 0.0)),
    ]
    report = _score(detections, events)
    by_id = {result.event_id: result for result in report.emission_results}
    assert by_id["omni-only"].heard and by_id["omni-only"].classified
    assert not by_id["omni-only"].classified_bearing_correct
    assert by_id["bearing-only"].classified_bearing_correct and not by_id["bearing-only"].classified_range_resolved
    assert by_id["ranged"].classified_range_resolved
    assert not by_id["missed"].heard
    assert report.heard_recall == pytest.approx(0.75)
    assert report.classified_bearing_correct_recall == pytest.approx(0.5)
    assert report.classified_range_resolved_recall == pytest.approx(0.25)
    assert report.range_claim_count == 1 and report.false_range_claim_count == 0


def test_wrong_label_is_heard_but_not_classified():
    report = _score([_detection(label="crow", category="corvid")], [_event("e")])
    assert report.emission_results[0].heard
    assert not report.emission_results[0].classified
    assert report.classified_bearing_correct_count == 0


def test_bearing_error_beyond_gate_fails_bearing_tier():
    # 30 m east truth; claimed 30 m at ~20 degrees off.
    claimed = (30.0 * np.cos(np.radians(20.0)), 30.0 * np.sin(np.radians(20.0)), 0.0)
    report = _score([_detection(position_m=claimed)], [_event("e")])
    result = report.emission_results[0]
    assert result.classified and not result.classified_bearing_correct
    assert result.best_bearing_error_deg == pytest.approx(20.0, abs=1e-6)


def test_range_tolerance_scales_with_distance():
    far_event = _event("far", waypoints=[(0.0, (400.0, 0.0, 0.0))])
    # 80 m short at 400 m is within 25 % relative tolerance; 8 m short at 30 m is not (3 m / 7.5 m).
    near_report = _score([_detection(position_m=(22.0, 0.0, 0.0))], [_event("near")])
    far_report = _score([_detection(position_m=(320.0, 0.0, 0.0))], [far_event])
    assert not near_report.emission_results[0].classified_range_resolved
    assert near_report.false_range_claim_count == 1
    assert far_report.emission_results[0].classified_range_resolved
    assert far_report.false_range_claim_count == 0


def test_false_range_claims_include_phantom_fixes_but_not_bearing_projected():
    events = [_event("e")]
    detections = [
        _detection(timestamp_s=50.0),  # range claim with nothing active: phantom
        _detection(timestamp_s=0.5, position_m=(90.0, 0.0, 0.0)),  # 60 m long
        _detection(timestamp_s=0.6, position_m=(90.0, 0.0, 0.0), range_mode="range_bearing_projected"),
    ]
    report = _score(detections, events, max_false_range_claim_count=1)
    assert report.range_claim_count == 2
    assert report.unassociated_range_claim_count == 1
    assert report.out_of_tolerance_range_claim_count == 1
    assert report.false_range_claim_count == 2
    assert report.false_range_claim_rate == pytest.approx(1.0)
    assert not report.passed and "false range claims 2 above 1" in report.errors[0]


def test_weak_range_observability_is_not_a_range_claim():
    detection = _detection()
    detection["feature_summary"]["localization_range_observability"] = 0.01
    assert not detection_claims_range(detection)
    assert not detection_claims_range(_detection(modality="omni"))
    assert detection_claims_range(_detection())


def test_echo_events_are_not_scored_and_detection_prefers_label_match():
    events = [
        _event("echo", is_echo_source=True),
        _event("crow-call", label="crow", category="corvid", waypoints=[(0.0, (30.0, 0.0, 0.0))]),
        _event("robin-call", waypoints=[(0.0, (0.0, 30.0, 0.0))]),
    ]
    report = _score([_detection(position_m=(0.0, 30.0, 0.0))], events)
    assert report.emission_count == 2
    by_id = {result.event_id: result for result in report.emission_results}
    assert by_id["robin-call"].classified_range_resolved and not by_id["crow-call"].heard


def test_unknown_observer_uses_array_centroid():
    detection = _detection(position_m=(130.0, 100.0, 0.0))
    detection["source_node_id"] = "cluster-reporter-without-position"
    report = _score([detection], [_event("e", waypoints=[(0.0, (130.0, 100.0, 0.0))])])
    assert report.emission_results[0].true_range_m == pytest.approx(30.0)


def test_tier_recall_gates_report_errors():
    report = _score([], [_event("e")], min_heard_recall=0.5, min_classified_range_resolved_recall=0.1)
    assert len(report.errors) == 2


# ---- uncertainty coverage -------------------------------------------------------------------


def _gaussian_samples(seeds: list[int], per_seed: int, stated_sigma_m: float, true_sigma_m: float) -> list[CoverageSample]:
    samples = []
    covariance = tuple(tuple(stated_sigma_m**2 if i == j else 0.0 for j in range(3)) for i in range(3))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for error in rng.normal(0.0, true_sigma_m, size=(per_seed, 3)):
            samples.append(CoverageSample(seed=seed, position_error_m=tuple(error), covariance_m2=covariance))
    return samples


def test_honest_covariance_is_consistent_on_held_out_seeds():
    _development, held_out = partition_held_out_seeds(range(20), held_out_every_nth=2)
    samples = _gaussian_samples(list(range(20)), per_seed=40, stated_sigma_m=2.0, true_sigma_m=2.0)
    report = measure_uncertainty_coverage(samples, held_out, UncertaintyCoverageConfig())
    assert report.sample_count == 10 * 40
    assert report.verdict == COVERAGE_VERDICT_CONSISTENT and report.passed
    assert report.empirical_coverage == pytest.approx(0.95, abs=0.03)
    assert set(report.coverage_by_seed) == set(held_out)


def test_overconfident_covariance_fails_and_conservative_one_passes_by_default():
    seeds = list(range(5))
    overconfident = measure_uncertainty_coverage(
        _gaussian_samples(seeds, 60, stated_sigma_m=1.0, true_sigma_m=2.0), seeds, UncertaintyCoverageConfig()
    )
    assert overconfident.verdict == COVERAGE_VERDICT_UNDER_COVERED and not overconfident.passed
    conservative_samples = _gaussian_samples(seeds, 200, stated_sigma_m=4.0, true_sigma_m=2.0)
    conservative = measure_uncertainty_coverage(conservative_samples, seeds, UncertaintyCoverageConfig())
    assert conservative.verdict == COVERAGE_VERDICT_OVER_COVERED and conservative.passed
    strict = measure_uncertainty_coverage(
        conservative_samples, seeds, UncertaintyCoverageConfig(fail_on_over_coverage=True)
    )
    assert not strict.passed


def test_development_seeds_are_ignored():
    good = _gaussian_samples([1, 3], 50, stated_sigma_m=2.0, true_sigma_m=2.0)
    bad_development = _gaussian_samples([2], 500, stated_sigma_m=0.1, true_sigma_m=2.0)
    report = measure_uncertainty_coverage(good + bad_development, [1, 3], UncertaintyCoverageConfig())
    assert report.sample_count == 100 and report.passed


def test_singular_covariance_counts_as_invalid_and_uncovered():
    assert squared_mahalanobis_distance(np.ones(3), np.zeros((3, 3))) is None
    assert squared_mahalanobis_distance(np.ones(3), -np.eye(3)) is None
    singular = [CoverageSample(seed=0, position_error_m=(0.0, 0.0, 0.0), covariance_m2=((0.0,) * 3,) * 3)] * 40
    report = measure_uncertainty_coverage(singular, [0], UncertaintyCoverageConfig())
    assert report.invalid_covariance_count == 40 and report.covered_count == 0
    assert report.verdict == COVERAGE_VERDICT_UNDER_COVERED


def test_coverage_reports_missing_seeds_and_small_samples():
    samples = _gaussian_samples([0], 5, stated_sigma_m=1.0, true_sigma_m=1.0)
    report = measure_uncertainty_coverage(samples, [0, 7], UncertaintyCoverageConfig())
    assert any("no samples" in error for error in report.errors)
    assert any("below the minimum" in error for error in report.errors)
    assert not measure_uncertainty_coverage(samples, [], UncertaintyCoverageConfig()).passed


def test_horizontal_coverage_ignores_height_error():
    covariance = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1e-6))
    samples = [CoverageSample(seed=0, position_error_m=(0.1, 0.1, 50.0), covariance_m2=covariance)] * 40
    report = measure_uncertainty_coverage(samples, [0], UncertaintyCoverageConfig(horizontal_only=True))
    assert report.covered_count == 40


def test_coverage_sample_from_detection():
    detection = {"position_m": [1.0, 2.0, 3.0], "position_covariance_m2": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    sample = coverage_sample_from_detection(4, detection, (0.0, 0.0, 0.0))
    assert sample.position_error_m == (1.0, 2.0, 3.0) and sample.seed == 4
    assert coverage_sample_from_detection(4, {"position_m": [0, 0, 0]}, (0, 0, 0)) is None


def test_partition_held_out_seeds_is_deterministic_and_disjoint():
    development, held_out = partition_held_out_seeds([5, 1, 3, 2, 4, 6], held_out_every_nth=3)
    assert held_out == (3, 6)
    assert set(development) | set(held_out) == {1, 2, 3, 4, 5, 6}
    with pytest.raises(ValueError):
        partition_held_out_seeds([1, 2], held_out_every_nth=1)


# ---- track metrics --------------------------------------------------------------------------


def _drone_flyby() -> GroundTruthEvent:
    return _event(
        "drone",
        label="drone",
        category="drone",
        start_s=0.0,
        end_s=20.0,
        waypoints=[(0.0, (-100.0, 50.0, 30.0)), (20.0, (100.0, 50.0, 30.0))],
    )


def _flyby_updates(track_ids_by_time: dict[float, str]) -> list[TrackUpdate]:
    event = _drone_flyby()
    return [
        TrackUpdate(track_id, int(t * SECOND_NS), tuple(event.position_at(int(t * SECOND_NS))))
        for t, track_id in sorted(track_ids_by_time.items())
    ]


def test_continuous_track_has_small_gaps_and_no_switches():
    updates = _flyby_updates({float(t): "T1" for t in range(0, 21)})
    report = compute_track_metrics(updates, [_drone_flyby()], TrackMetricsConfig(max_update_gap_seconds=2.0))
    assert report.max_update_gap_seconds == pytest.approx(1.0)
    assert report.total_identity_switches == 0 and report.total_fragmentations == 0
    assert report.min_tracked_time_fraction == pytest.approx(1.0)
    assert report.passed


def test_starved_track_reports_large_gap_and_fragmentation():
    updates = _flyby_updates({1.0: "T1", 2.0: "T1", 15.0: "T1", 16.0: "T1"})
    report = compute_track_metrics(
        updates, [_drone_flyby()], TrackMetricsConfig(max_update_gap_seconds=5.0, max_fragmentations=0)
    )
    assert report.max_update_gap_seconds == pytest.approx(13.0)
    assert report.total_fragmentations == 1
    # Gaps <= 5 s: 1 (lead) + 1 + 1 + 4 (trail) = 7 s of 20 s.
    assert report.min_tracked_time_fraction == pytest.approx(7.0 / 20.0)
    assert len(report.errors) == 2


def test_identity_switches_and_merged_tracks():
    updates = _flyby_updates({float(t): ("T1" if t < 10 else "T2") for t in range(0, 21)})
    updates.append(TrackUpdate("T1", 5 * SECOND_NS, (500.0, 500.0, 0.0)))  # far from any truth
    bird = _event("bird", start_s=0.0, end_s=20.0, waypoints=[(0.0, (0.0, -40.0, 5.0))])
    updates.append(TrackUpdate("T2", 12 * SECOND_NS, (0.0, -40.0, 5.0)))
    report = compute_track_metrics(updates, [_drone_flyby(), bird], TrackMetricsConfig(max_identity_switches=0))
    drone_result = next(r for r in report.individual_results if r.individual_id == "drone")
    assert drone_result.identity_switches == 1
    assert drone_result.distinct_track_ids == ("T1", "T2")
    assert report.unassociated_update_count == 1
    assert report.tracks_spanning_multiple_individuals == 1
    assert not report.passed


def test_bird_recalling_on_new_track_is_an_identity_switch_across_emissions():
    calls = [
        _event(f"call-{i}", individual_id="wren", start_s=10.0 * i, end_s=10.0 * i + 1.0,
               waypoints=[(0.0, (5.0, 5.0, 2.0))])
        for i in range(3)
    ]
    updates = [TrackUpdate(f"T{i}", int((10.0 * i + 0.5) * SECOND_NS), (5.0, 5.0, 2.0)) for i in range(3)]
    report = compute_track_metrics(updates, calls, TrackMetricsConfig())
    assert report.individual_results[0].identity_switches == 2
    # Silence between calls is not a track gap.
    assert report.max_update_gap_seconds == pytest.approx(0.5)


def test_emission_without_updates_is_one_full_gap():
    report = compute_track_metrics([], [_drone_flyby()], TrackMetricsConfig())
    assert report.max_update_gap_seconds == pytest.approx(20.0)
    assert report.min_tracked_time_fraction == pytest.approx(0.0)


def test_track_updates_from_detections_skips_untracked_rows():
    rows = [_detection(track_id="T9"), _detection(track_id=None), _detection(track_id="T9", position_m=None)]
    assert [update.track_id for update in track_updates_from_detections(rows)] == ["T9"]


# ---- gate config ----------------------------------------------------------------------------


def test_gate_config_mapping_overrides_and_rejects_typos():
    config = replay_gate_config_from_mapping(
        {
            "tiers": {"bearing_correct_max_error_deg": 5.0},
            "hot_path": {"metric_tolerances": {"solve_latency_p95_ms": {"max_relative_increase": 0.5}}},
        }
    )
    assert config.tiers.bearing_correct_max_error_deg == 5.0
    assert config.hot_path.metric_tolerances["solve_latency_p95_ms"].max_relative_increase == 0.5
    # Untouched defaults survive a partial override.
    assert config.hot_path.metric_tolerances["solve_latency_p95_ms"].max_absolute_increase > 0.0
    with pytest.raises(ValueError, match="unknown tiers"):
        replay_gate_config_from_mapping({"tiers": {"bearing_max": 1.0}})
    with pytest.raises(ValueError):
        TierScoringConfig(label_match_mode="fuzzy")
    with pytest.raises(ValueError):
        UncertaintyCoverageConfig(stated_confidence_level=1.0)


def test_metric_tolerance_allowed_maximum_applies_ceiling():
    assert HotPathMetricTolerance(max_relative_increase=0.5, max_absolute_increase=1.0).allowed_maximum(10.0) == 16.0
    assert HotPathMetricTolerance(absolute_ceiling=0.0).allowed_maximum(4096.0) == 0.0
