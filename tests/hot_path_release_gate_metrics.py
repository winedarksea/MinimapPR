"""Turn one E10 gate run's raw captures into the metrics the release gate compares."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np

from tests.hot_path_release_gate_probes import (
    DspStreamObservations,
    account_raw_audio_forwarding,
    node_id_for_stream_key,
)
from tests.hot_path_release_gate_scenario import HotPathScenario, ScheduledPost, call_arrival_windows_seconds

NANOSECONDS_PER_MILLISECOND = 1_000_000
NANOSECONDS_PER_SECOND = 1_000_000_000
LATENCY_PERCENTILES = (50, 95)
SQLITE_SUFFIXES = (".db", ".db-wal", ".db-shm", ".sqlite3", ".sqlite3-wal", ".sqlite3-shm")
CONFIG_SUFFIXES = (".yml", ".yaml", ".json")


@dataclass
class GateRunCapture:
    """Everything one run observed; filled by the runner, consumed by ``assemble_gate_metrics``."""

    scenario: HotPathScenario
    # Wall-clock ns of the scenario's sample 0 (frame timestamps are built from it).
    scenario_start_time_ns: int
    posts: list[ScheduledPost]
    accepted_monotonic_ns_by_post: dict[tuple[str, int], int]
    post_round_trip_ms: list[float]
    rejected_post_count: int
    observations: DspStreamObservations
    sidecar_health: dict
    sidecar_dsp_status: dict
    sidecar_summary: dict
    fusion_status: dict
    app_diagnostics: dict
    detections: list
    peak_rss_mb: dict[str, float]
    app_stream_reconnect_count: int
    written_bytes_by_path: dict[str, int] = field(default_factory=dict)
    snippet_root: str = ""


def _percentiles(values: list[float], prefix: str) -> dict[str, float | None]:
    return {
        f"{prefix}_p{percentile}_ms": (float(np.percentile(values, percentile)) if values else None)
        for percentile in LATENCY_PERCENTILES
    }


def solve_latencies_ms(capture: GateRunCapture) -> tuple[list[float], int]:
    """Trigger packet accepted by the sidecar -> pair-bearing localization result on the DSP stream."""
    node_ids = [node.node_id for node in capture.scenario.nodes]
    posts_by_node: dict[str, list[ScheduledPost]] = {}
    for post in capture.posts:
        posts_by_node.setdefault(post.node_id, []).append(post)
    latencies_ms, unmapped_count = [], 0
    for solve in capture.observations.solves:
        node_id = node_id_for_stream_key(solve.stream_key, node_ids)
        last_sample = solve.trigger_sample_index_start + solve.trigger_sample_count - 1
        post = next(
            (p for p in posts_by_node.get(node_id, []) if p.first_sample_index <= last_sample < p.end_sample_index),
            None,
        )
        accepted_ns = capture.accepted_monotonic_ns_by_post.get((post.node_id, post.first_sample_index)) if post else None
        if accepted_ns is None:
            unmapped_count += 1
            continue
        latencies_ms.append((solve.arrival_monotonic_ns - accepted_ns) / NANOSECONDS_PER_MILLISECOND)
    return latencies_ms, unmapped_count


def app_counters(fusion_status: dict, diagnostics: dict) -> dict[str, int]:
    metrics = fusion_status.get("metrics") or {}
    buffer_state = (diagnostics.get("pipeline") or {}).get("buffer_state") or []
    return {
        "frame_sequence_gaps": int(metrics.get("frame_sequence_gaps", 0)),
        "frames_zero_padded_degraded": int(metrics.get("frames_zero_padded_degraded", 0)),
        "triggers_dropped_queue_full": int(metrics.get("triggers_dropped_queue_full", 0)),
        "stage_drops_backpressure": int(metrics.get("stage_drops_backpressure", 0)),
        "classification_drops": int(sum((metrics.get("classification_drops_by_reason") or {}).values())),
        "buffer_reanchors": int(sum(int(sensor.get("reanchor_count") or 0) for sensor in buffer_state)),
        "buffer_rate_mismatch_wipes": int(sum(int(sensor.get("rate_mismatch_wipes") or 0) for sensor in buffer_state)),
        "detections_emitted": int(metrics.get("detections_emitted", 0)),
    }


@dataclass(frozen=True)
class KnownCallDetectionAccounting:
    known_call_count: int
    detected_known_call_count: int
    # Stored detections attributable to no call (outside the scenario's calls, or no event time).
    unmatched_detection_count: int


def account_known_call_detections(
    scenario: HotPathScenario, scenario_start_time_ns: int, detections: list
) -> KnownCallDetectionAccounting:
    """Count the scenario's known calls that reached at least one stored detection.

    Each detection is attributed to the call whose arrival window is nearest
    its event time (``timestamp_ns``), if within half the call interval.
    Stored event times are quantized to the analysis block, not the onset:
    measured offsets from the arrival window run from ~0.6 s early to ~0.5 s
    late, so a tight window would reject real calls; nearest-call attribution
    is the resolution the server's timestamps support. Counting calls rather
    than rows keeps duplicate rows of one call (omni + localized) from
    counting twice; detections outside the scenario's calls stay unmatched.
    """
    attribution_limit_ns = int(scenario.event_interval_seconds / 2.0 * NANOSECONDS_PER_SECOND)
    windows_ns = [
        (
            scenario_start_time_ns + int(start_s * NANOSECONDS_PER_SECOND),
            scenario_start_time_ns + int(end_s * NANOSECONDS_PER_SECOND),
        )
        for start_s, end_s in call_arrival_windows_seconds(scenario)
    ]
    if not windows_ns:
        # No calls to attribute to: every stored detection is unmatched.
        return KnownCallDetectionAccounting(0, 0, len(detections))
    detected_call_indices: set[int] = set()
    unmatched_detection_count = 0
    for detection in detections:
        event_time_ns = detection.get("timestamp_ns") if isinstance(detection, dict) else None
        if event_time_ns is None:
            unmatched_detection_count += 1
            continue
        distances_ns = [max(start_ns - int(event_time_ns), int(event_time_ns) - end_ns, 0) for start_ns, end_ns in windows_ns]
        nearest_index = min(range(len(windows_ns)), key=distances_ns.__getitem__)
        if distances_ns[nearest_index] <= attribution_limit_ns:
            detected_call_indices.add(nearest_index)
        else:
            unmatched_detection_count += 1
    return KnownCallDetectionAccounting(len(windows_ns), len(detected_call_indices), unmatched_detection_count)


def classify_disk_writes(written_bytes_by_path: dict[str, int], snippet_root: str) -> tuple[dict[str, int], dict[str, int]]:
    """Split new/grown files into (raw audio, detection snippets).

    The SQLite DB and config files are state, not audio. Detection snippets are
    the retention exception ("some detections") and are reported on their own.
    Everything else under the data / spool / sidecar storage dirs is raw audio
    that the memory-only live path should never have written.
    """
    snippet_writes = {path: size for path, size in written_bytes_by_path.items() if snippet_root and path.startswith(snippet_root)}
    raw_audio_writes = {
        path: size
        for path, size in written_bytes_by_path.items()
        if path not in snippet_writes and not path.endswith(SQLITE_SUFFIXES + CONFIG_SUFFIXES)
    }
    return raw_audio_writes, snippet_writes


def assemble_gate_metrics(capture: GateRunCapture) -> tuple[dict, dict]:
    """Return (metrics compared by the gate, components kept for diagnosis)."""
    scenario = capture.scenario
    known_calls = account_known_call_detections(scenario, capture.scenario_start_time_ns, capture.detections)
    total_sample_count = int(round(scenario.duration_seconds * scenario.sample_rate_hz))
    posted_spans_by_node: dict[str, list[tuple[int, int]]] = {node.node_id: [] for node in scenario.nodes}
    for post in capture.posts:
        if (post.node_id, post.first_sample_index) in capture.accepted_monotonic_ns_by_post:
            posted_spans_by_node[post.node_id].append((post.first_sample_index, post.end_sample_index - post.first_sample_index))
    forwarding = account_raw_audio_forwarding(
        posted_spans_by_node, capture.observations.raw_audio_spans_by_stream, total_sample_count
    )
    forwarded_missing_frames = int(np.ceil(forwarding.missing_sample_count / scenario.frame_samples))
    latencies_ms, unmapped_solve_count = solve_latencies_ms(capture)
    counters = app_counters(capture.fusion_status, capture.app_diagnostics)
    sidecar_sequence_gaps = int(((capture.sidecar_health.get("backend") or {}).get("sequence_gaps") or {}).get("total_gap_count", 0))
    sidecar_queue_drops = int(capture.sidecar_summary.get("overload_rejections", 0)) + int(
        capture.sidecar_summary.get("frames_dropped_total", 0)
    )
    sidecar_discontinuities = int(capture.sidecar_dsp_status.get("total_buffer_reanchors", 0)) + int(
        capture.sidecar_dsp_status.get("total_window_underrun_drops", 0)
    )
    raw_audio_writes, snippet_writes = classify_disk_writes(capture.written_bytes_by_path, capture.snippet_root)
    metrics = {
        "packet_gap_count": sidecar_sequence_gaps
        + forwarded_missing_frames
        + counters["frame_sequence_gaps"]
        + counters["frames_zero_padded_degraded"],
        "app_stream_reconnect_count": capture.app_stream_reconnect_count,
        "ingest_rejected_post_count": capture.rejected_post_count,
        "ingest_queue_drop_count": sidecar_queue_drops + counters["triggers_dropped_queue_full"],
        "buffer_discontinuity_count": sidecar_discontinuities + counters["buffer_reanchors"] + counters["buffer_rate_mismatch_wipes"],
        "raw_audio_out_of_order_span_count": forwarding.out_of_order_span_count,
        "solve_count": len(capture.observations.solves),
        **_percentiles(latencies_ms, "solve_latency"),
        **_percentiles(capture.post_round_trip_ms, "ingest_post_round_trip"),
        "sidecar_peak_rss_mb": capture.peak_rss_mb.get("sidecar"),
        "app_peak_rss_mb": capture.peak_rss_mb.get("app"),
        "raw_audio_disk_bytes_written": int(sum(raw_audio_writes.values())),
        "raw_audio_disk_files_written": len(raw_audio_writes),
        "detection_snippet_bytes_written": int(sum(snippet_writes.values())),
        "detection_count": len(capture.detections),
        "known_call_detection_count": known_calls.detected_known_call_count,
        "classification_backlog_drop_count": counters["classification_drops"] + counters["stage_drops_backpressure"],
        "classification_max_completed_lag_seconds": (capture.fusion_status.get("realtime") or {}).get(
            "max_completed_lag_seconds"
        ),
    }
    components = {
        "sidecar_sequence_gaps": sidecar_sequence_gaps,
        "forwarded_raw_audio_missing_samples": forwarding.missing_sample_count,
        "forwarded_raw_audio_duplicated_samples": forwarding.duplicated_sample_count,
        "forwarded_raw_audio_unmapped_streams": forwarding.unmapped_stream_count,
        "sidecar_queue_drops": sidecar_queue_drops,
        "sidecar_discontinuities": sidecar_discontinuities,
        "app": counters,
        "sidecar_storage_mode": (capture.sidecar_health.get("backend") or {}).get("storage_mode"),
        "sidecar_coverage_wait_stop_reasons": capture.sidecar_dsp_status.get("localization_coverage_wait_stop_reasons"),
        "sidecar_total_localization_results": capture.sidecar_dsp_status.get("total_localization_results"),
        "unmapped_solve_count": unmapped_solve_count,
        "observed_classifier_renders": capture.observations.classifier_render_count,
        "observer_replay_gap_events": capture.observations.replay_gap_event_count,
        "observer_malformed_events": capture.observations.malformed_event_count,
        "observer_stream_error": capture.observations.stream_error,
        "raw_audio_written_paths": sorted(raw_audio_writes)[:20],
        "scenario_start_time_ns": capture.scenario_start_time_ns,
        "scenario_known_call_count": known_calls.known_call_count,
        "unmatched_detection_count": known_calls.unmatched_detection_count,
    }
    return metrics, components


def observer_was_complete(components: dict) -> bool:
    """False when the harness's own stream subscriber missed events (its metrics undercount)."""
    return (
        components["observer_replay_gap_events"] == 0
        and components["observer_stream_error"] is None
        and components["forwarded_raw_audio_unmapped_streams"] == 0
    )


def aggregate_repeated_run_metrics(per_run_metrics: list[dict], tolerances: dict | None = None) -> dict:
    """Per-metric aggregate over repeated runs.

    Median damps shared-machine timing noise; metrics whose tolerance asks for
    the worst case take the worst run (max for counters, min for floor metrics
    such as known-call detections) so one bad run fails.
    """
    tolerances = tolerances or {}
    merged: dict = {}
    for metric_name in per_run_metrics[0]:
        values = [run[metric_name] for run in per_run_metrics if run.get(metric_name) is not None]
        if not values:
            merged[metric_name] = None
            continue
        tolerance = tolerances.get(metric_name)
        worst_case = tolerance is not None and tolerance.aggregate_worst_case_across_runs
        merged[metric_name] = tolerance.worst_run_value(values) if worst_case else float(statistics.median(values))
    return merged
