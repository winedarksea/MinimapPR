"""E10 hot-path release gate: comparison logic, metric accounting, a fast smoke run, and the opt-in full gate.

The full gate (``MINIMAPPR_ENABLE_HOT_PATH_RELEASE_GATE=1``) replays the full
scenario against this checkout and compares it with the committed baseline
``tests/data/hot_path_baseline.json``. Latency and RSS only compare on the
machine the baseline names; ``scripts/hot_path_release_gate.py`` re-records a
local baseline from any commit.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from minimappr.sim.replay_gate_config import HotPathGateConfig
from tests.hot_path_release_gate import run_hot_path_release_gate
from tests.hot_path_release_gate_compare import compare_hot_path_results
from tests.hot_path_release_gate_metrics import (
    account_known_call_detections,
    aggregate_repeated_run_metrics,
    classify_disk_writes,
)
from tests.hot_path_release_gate_probes import account_raw_audio_forwarding, written_files
from tests.hot_path_release_gate_scenario import (
    FULL_SCENARIO,
    SMOKE_SCENARIO,
    build_post_schedule,
    call_arrival_windows_seconds,
)
from tests.rust_sidecar_binary import REPOSITORY_ROOT, cargo_is_available

FULL_GATE_ENV = "MINIMAPPR_ENABLE_HOT_PATH_RELEASE_GATE"
FULL_GATE_REPEAT_ENV = "MINIMAPPR_HOT_PATH_GATE_REPEAT"
FULL_GATE_DEFAULT_REPEAT = 3
BASELINE_PATH = Path(__file__).resolve().parent / "data" / "hot_path_baseline.json"


def _result(**metric_overrides) -> dict:
    metrics = {
        "packet_gap_count": 0.0,
        "app_stream_reconnect_count": 0.0,
        "ingest_rejected_post_count": 0.0,
        "ingest_queue_drop_count": 0.0,
        "buffer_discontinuity_count": 0.0,
        "solve_count": 40.0,
        "solve_latency_p50_ms": 600.0,
        "solve_latency_p95_ms": 1200.0,
        "ingest_post_round_trip_p50_ms": 2.0,
        "ingest_post_round_trip_p95_ms": 6.0,
        "sidecar_peak_rss_mb": 200.0,
        "app_peak_rss_mb": 250.0,
        "raw_audio_disk_bytes_written": 0.0,
        "raw_audio_disk_files_written": 0.0,
        "detection_count": 5.0,
        "known_call_detection_count": 4.0,
        "classification_backlog_drop_count": 0.0,
    }
    metrics.update(metric_overrides)
    return {
        "schema_version": 1,
        "scenario": FULL_SCENARIO.to_json_dict(),
        "machine": {"cpu_model": "test"},
        "sidecar_build_profile": "debug",
        "observer_complete": True,
        "metrics": metrics,
    }


# ---- comparison -----------------------------------------------------------------------------


def test_identical_results_pass_and_report_every_metric():
    comparison = compare_hot_path_results(_result(), _result(), HotPathGateConfig())
    assert comparison.passed, comparison.format_table()
    assert {row.metric_name for row in comparison.rows} >= set(_result()["metrics"])


def test_new_packet_gap_or_disk_write_fails_even_from_a_dirty_baseline():
    config = HotPathGateConfig()
    assert not compare_hot_path_results(_result(), _result(packet_gap_count=1.0), config).passed
    # The raw-audio ceiling is absolute: a baseline that leaked audio does not license the candidate to.
    leaky_baseline = _result(raw_audio_disk_bytes_written=4096.0)
    comparison = compare_hot_path_results(leaky_baseline, _result(raw_audio_disk_bytes_written=4096.0), config)
    assert not comparison.passed
    assert any("raw_audio_disk_bytes_written" in error for error in comparison.errors)


def test_latency_and_memory_have_relative_headroom():
    config = HotPathGateConfig()
    assert compare_hot_path_results(_result(), _result(solve_latency_p95_ms=1500.0), config).passed
    assert not compare_hot_path_results(_result(), _result(solve_latency_p95_ms=2500.0), config).passed
    assert not compare_hot_path_results(_result(), _result(sidecar_peak_rss_mb=400.0), config).passed


def test_ungated_metrics_never_fail():
    comparison = compare_hot_path_results(_result(), _result(novel_metric=9.0), HotPathGateConfig())
    assert comparison.passed
    assert any(row.metric_name == "novel_metric" and not row.gated for row in comparison.rows)


def test_healthy_solve_stream_with_zero_stored_detections_fails():
    # Solves, latency and ingest all match the baseline; only the render handoff lost every classification.
    silent_candidate = _result(detection_count=0.0, known_call_detection_count=0.0)
    comparison = compare_hot_path_results(_result(), silent_candidate, HotPathGateConfig())
    assert not comparison.passed
    assert any("known_call_detection_count" in error for error in comparison.errors)
    # The floor is absolute: a baseline that also stored nothing does not excuse the candidate.
    assert not compare_hot_path_results(silent_candidate, silent_candidate, HotPathGateConfig()).passed


def test_detection_requirements_fail_when_the_harness_did_not_measure_them():
    for metric_name in ("known_call_detection_count", "classification_backlog_drop_count"):
        unmeasured = _result()
        del unmeasured["metrics"][metric_name]
        comparison = compare_hot_path_results(unmeasured, unmeasured, HotPathGateConfig())
        assert not comparison.passed
        assert any(metric_name in error for error in comparison.errors)


def test_classification_backlog_drops_over_budget_fail():
    config = HotPathGateConfig()
    budget = config.max_candidate_classification_backlog_drop_count
    within_budget = _result(classification_backlog_drop_count=float(budget))
    assert compare_hot_path_results(_result(), within_budget, config).passed
    over_budget = _result(classification_backlog_drop_count=float(budget + 1))
    comparison = compare_hot_path_results(_result(), over_budget, config)
    assert not comparison.passed
    assert any("classification_backlog_drop_count" in error for error in comparison.errors)


def test_committed_baseline_passes_against_itself():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    comparison = compare_hot_path_results(baseline, baseline, HotPathGateConfig())
    assert comparison.passed, comparison.format_table()


def test_incomparable_runs_are_errors():
    config = HotPathGateConfig()
    other_scenario = _result()
    other_scenario["scenario"] = SMOKE_SCENARIO.to_json_dict()
    assert not compare_hot_path_results(_result(), other_scenario, config).passed
    blind_observer = _result()
    blind_observer["observer_complete"] = False
    assert not compare_hot_path_results(_result(), blind_observer, config).passed
    assert not compare_hot_path_results(_result(), _result(solve_count=5.0), config).passed
    missing_latency = _result(solve_latency_p50_ms=None)
    assert not compare_hot_path_results(_result(), missing_latency, config).passed
    other_machine = _result()
    other_machine["machine"] = {"cpu_model": "elsewhere"}
    assert compare_hot_path_results(_result(), other_machine, config).warnings


# ---- accounting -----------------------------------------------------------------------------


def test_raw_audio_forwarding_separates_loss_reordering_and_duplicates():
    posted = {"node-a": [(0, 100), (100, 100), (200, 100)], "node-b": [(0, 100)]}
    forwarded = {
        "node-a__audio_main__1": [(100, 100), (0, 100), (100, 100)],  # 300..: lost; one reorder; one duplicate
        "node-b__audio_main__2": [(0, 100)],
        "stranger__audio_main__3": [(0, 10)],
    }
    accounting = account_raw_audio_forwarding(posted, forwarded, total_sample_count=400)
    assert accounting.missing_sample_count == 100
    assert accounting.out_of_order_span_count == 2
    assert accounting.duplicated_sample_count == 100
    assert accounting.unmapped_stream_count == 1


def test_disk_write_classification_excludes_db_and_reports_snippets_apart():
    writes = written_files(
        {"/d/minimappr.db": 100, "/d/old.wav": 10},
        {"/d/minimappr.db": 200, "/d/minimappr.db-wal": 50, "/d/old.wav": 10, "/d/snippets/a.wav": 7,
         "/d/ingest/journal/streams/x/seg.bin": 4096, "/d/config.yml": 3},
    )
    raw_audio, snippets = classify_disk_writes(writes, "/d/snippets")
    assert raw_audio == {"/d/ingest/journal/streams/x/seg.bin": 4096}
    assert snippets == {"/d/snippets/a.wav": 7}


def test_known_call_accounting_counts_calls_not_detection_rows():
    start_ns = 1_000_000_000_000
    windows = call_arrival_windows_seconds(FULL_SCENARIO)
    half_interval_s = FULL_SCENARIO.event_interval_seconds / 2.0

    def detection_at(offset_s: float) -> dict:
        return {"timestamp_ns": start_ns + int(offset_s * 1e9)}

    detections = [
        detection_at(windows[0][0]),
        detection_at(windows[0][1]),  # second row (e.g. omni + localized) on the same call
        detection_at(windows[3][0] - 0.55),  # block-quantized event time ahead of the arrival
        detection_at(windows[5][1] + 0.1),  # just after call 5: nearer it than call 6
        detection_at(windows[-1][1] + half_interval_s + 0.5),  # after the last call: no call to blame
        {"timestamp_ns": None},
    ]
    accounting = account_known_call_detections(FULL_SCENARIO, start_ns, detections)
    assert accounting.known_call_count == len(windows)
    assert accounting.detected_known_call_count == 3
    assert accounting.unmatched_detection_count == 2


def test_aggregate_metrics_ignores_missing_values():
    merged = aggregate_repeated_run_metrics([{"a": 1, "b": None}, {"a": 3, "b": 2.0}, {"a": 2, "b": None}])
    assert merged == {"a": 2.0, "b": 2.0}


def test_aggregate_metrics_takes_worst_run_for_zero_tolerance_counters():
    tolerances = HotPathGateConfig().metric_tolerances
    runs = [
        {"packet_gap_count": 0, "raw_audio_disk_bytes_written": 0, "solve_latency_p95_ms": 900.0},
        {"packet_gap_count": 3, "raw_audio_disk_bytes_written": 4096, "solve_latency_p95_ms": 5000.0},
        {"packet_gap_count": 0, "raw_audio_disk_bytes_written": 0, "solve_latency_p95_ms": 1000.0},
    ]
    merged = aggregate_repeated_run_metrics(runs, tolerances)
    assert merged["packet_gap_count"] == 3.0
    assert merged["raw_audio_disk_bytes_written"] == 4096.0
    assert merged["solve_latency_p95_ms"] == 1000.0


def test_aggregate_metrics_takes_the_fewest_known_calls_for_the_floor():
    """A floor worsens downward: one run that stored no known call must fail the
    floor, not hide under a median of (5, 0, 6)."""
    config = HotPathGateConfig()
    runs = [{"known_call_detection_count": 5}, {"known_call_detection_count": 0}, {"known_call_detection_count": 6}]
    merged = aggregate_repeated_run_metrics(runs, config.metric_tolerances)
    assert merged["known_call_detection_count"] == 0.0
    assert merged["known_call_detection_count"] < config.min_candidate_known_call_detection_count


def test_higher_is_better_tolerance_cannot_be_baseline_gated():
    from minimappr.sim.replay_gate_config import HotPathMetricTolerance

    with pytest.raises(ValueError):
        HotPathMetricTolerance(higher_is_better=True)


def test_known_call_accounting_with_no_calls_counts_detections_as_unmatched():
    no_call_scenario = dataclasses.replace(SMOKE_SCENARIO, duration_seconds=SMOKE_SCENARIO.event_interval_seconds / 4.0)
    assert call_arrival_windows_seconds(no_call_scenario) == []
    accounting = account_known_call_detections(no_call_scenario, 0, [{"timestamp_ns": 10}, {"timestamp_ns": None}])
    assert (accounting.known_call_count, accounting.detected_known_call_count) == (0, 0)
    assert accounting.unmatched_detection_count == 2


def test_post_schedule_is_deterministic_and_follows_membership():
    first = build_post_schedule(SMOKE_SCENARIO, start_time_ns=1_000)
    second = build_post_schedule(SMOKE_SCENARIO, start_time_ns=1_000)
    assert [post.body for post in first] == [post.body for post in second]
    assert [post.send_offset_seconds for post in first] == sorted(post.send_offset_seconds for post in first)
    for node in SMOKE_SCENARIO.nodes:
        node_posts = [post for post in first if post.node_id == node.node_id]
        assert node_posts[0].first_sample_index / SMOKE_SCENARIO.sample_rate_hz >= node.join_seconds
        assert node_posts[-1].end_sample_index / SMOKE_SCENARIO.sample_rate_hz <= node.leave_seconds
        # Contiguous sequences and samples: any gap the gate counts is the server's, not the input's.
        for earlier, later in zip(node_posts, node_posts[1:]):
            assert later.first_sequence == earlier.last_sequence + 1
            assert later.first_sample_index == earlier.end_sample_index
        frame = json.loads(node_posts[0].body)["buffered_frames"][0]["frame"]
        assert frame["channels"] == 4 and frame["sample_rate_hz"] == SMOKE_SCENARIO.sample_rate_hz


# ---- real server runs -----------------------------------------------------------------------


def _skip_without_cargo() -> None:
    if not cargo_is_available():
        pytest.skip("cargo toolchain not available; cannot build the Rust sidecar")


def test_smoke_gate_run_against_this_checkout(tmp_path: Path):
    _skip_without_cargo()
    result = run_hot_path_release_gate(REPOSITORY_ROOT, tmp_path, SMOKE_SCENARIO, repeat_count=1)
    metrics, components = result["metrics"], result["runs"][0]["components"]
    assert result["observer_complete"], components
    assert Path(result["runs"][0]["served_minimappr_file"]).is_relative_to(REPOSITORY_ROOT)
    assert components["sidecar_storage_mode"] == "memory_only_live_path"
    assert metrics["solve_count"] > 0 and metrics["solve_latency_p50_ms"] is not None
    assert metrics["sidecar_peak_rss_mb"] > 0 and metrics["app_peak_rss_mb"] > 0
    # Hot-path invariants that hold on any machine (timing-free).
    assert metrics["ingest_rejected_post_count"] == 0
    assert metrics["packet_gap_count"] == 0, components
    assert metrics["raw_audio_disk_bytes_written"] == 0, components["raw_audio_written_paths"]
    # The harness must emit the detection requirements the comparator gates (absent would fail the gate).
    assert metrics["known_call_detection_count"] is not None
    assert metrics["classification_backlog_drop_count"] is not None


@pytest.mark.skipif(
    os.environ.get(FULL_GATE_ENV) != "1",
    reason=f"set {FULL_GATE_ENV}=1 to run the full hot-path release gate (several minutes)",
)
def test_full_hot_path_gate_against_committed_baseline(tmp_path: Path):
    _skip_without_cargo()
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    repeat_count = int(os.environ.get(FULL_GATE_REPEAT_ENV, FULL_GATE_DEFAULT_REPEAT))
    candidate = run_hot_path_release_gate(REPOSITORY_ROOT, tmp_path, FULL_SCENARIO, repeat_count=repeat_count)
    (tmp_path / "candidate.json").write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    comparison = compare_hot_path_results(baseline, candidate, HotPathGateConfig())
    assert comparison.passed, comparison.format_table()
