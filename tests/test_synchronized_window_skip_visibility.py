"""P8: a sensor buffered at another rate is still excluded, but visibly.

A mixed-rate point network (one node at 48 kHz, the rest at 16 kHz) runs a
16 kHz Python-ingest solve: the 48 kHz sensor is counted once per solve, per
sensor and reason, in ``FusionMetrics`` / ``fusion_status`` (totals and a
recent window), shown in the pipeline graph's solve stage from the recent
window, and noted on the solve's stored detection with the Rust contract's
``sample_rate_mismatch`` reason. A matched-rate network shows none.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minimappr.core.pipeline_graph import _solve_status
from minimappr.core.python_ingest_solve_exclusions import (
    LOCALIZATION_EXCLUDED_SENSORS_FEATURE_KEY,
    SolveSensorExclusionLedger,
)
from minimappr.core.synchronized_window_skips import SAMPLE_RATE_MISMATCH_SKIP_REASON
from tests.point_network_fusion_harness import (
    TruthLocalizer,
    python_ingest_candidate,
    register_point_nodes,
    started_point_network_fusion,
)
from tests.point_network_scene import (
    CAPTURE_SECONDS,
    CAPTURE_START_NS,
    NANOSECONDS_PER_SECOND,
    SAMPLE_RATE_HZ,
    Caller,
    append_captures,
    point_sensor_id,
    render_point_node_captures,
)

MISMATCHED_SAMPLE_RATE_HZ = 48_000
MISMATCHED_NODE_ID = "pt-c"
LOCALIZATION_WINDOW_SECONDS = 0.4
CALLER = Caller(position_m=(8.0, 6.0, 3.0), emission_offset_seconds=0.5, seed=11)


async def _fusion_with_point_network(tmp_path: Path, *, mismatched_node_rate_hz: int | None):
    fusion, storage, _ = await started_point_network_fusion(
        tmp_path,
        localizer=TruthLocalizer(CALLER.position_m),
        localization_window_seconds=LOCALIZATION_WINDOW_SECONDS,
        classification_window_seconds=LOCALIZATION_WINDOW_SECONDS,
        min_sensors_for_3d=3,
        min_sensors_for_2d=3,
        trigger_rms=1e-6,
    )
    await register_point_nodes(fusion, storage)
    captures = render_point_node_captures([CALLER])
    if mismatched_node_rate_hz is not None:
        mismatched_sensor_id = point_sensor_id(MISMATCHED_NODE_ID)
        del captures[mismatched_sensor_id]
        await fusion.buffer.append(
            sensor_id=mismatched_sensor_id,
            sample_rate_hz=mismatched_node_rate_hz,
            start_time_ns=CAPTURE_START_NS,
            samples=np.zeros(int(CAPTURE_SECONDS * mismatched_node_rate_hz), dtype=np.float32),
        )
    await append_captures(fusion.buffer, captures)
    return fusion, storage


def _rate_candidate(candidate_id: str):
    return python_ingest_candidate(
        candidate_id=candidate_id,
        triggering_node_id="pt-a",
        event_time_ns=CALLER.arrival_time_ns("pt-a") + int(0.05 * NANOSECONDS_PER_SECOND),
        sample_rate_hz=SAMPLE_RATE_HZ,
    )


async def _localize_and_classify(fusion):
    product = await fusion._localize_candidate(_rate_candidate("evt-rate"))
    assert product is not None and product.localization_branch is not None
    [detection_product] = await fusion._classify_and_assemble(product)
    return product, detection_product


async def test_mixed_rate_cluster_counts_and_notes_the_excluded_sensor(tmp_path: Path) -> None:
    fusion, storage = await _fusion_with_point_network(tmp_path, mismatched_node_rate_hz=MISMATCHED_SAMPLE_RATE_HZ)
    mismatched_sensor_id = point_sensor_id(MISMATCHED_NODE_ID)
    try:
        product, detection_product = await _localize_and_classify(fusion)

        # The policy is unchanged: the 48 kHz sensor is not in the 16 kHz solve.
        assert mismatched_sensor_id not in product.selected_sensor_ids
        expected_exclusion = [
            {
                "sensor_id": mismatched_sensor_id,
                "node_id": MISMATCHED_NODE_ID,
                "reason": SAMPLE_RATE_MISMATCH_SKIP_REASON,
                "sensor_sample_rate_hz": MISMATCHED_SAMPLE_RATE_HZ,
                "solve_sample_rate_hz": SAMPLE_RATE_HZ,
            }
        ]
        assert product.extra_classification_features[LOCALIZATION_EXCLUDED_SENSORS_FEATURE_KEY] == expected_exclusion
        assert detection_product.detection.feature_summary[LOCALIZATION_EXCLUDED_SENSORS_FEATURE_KEY] == expected_exclusion

        # One solve read the sensor's windows several times (localization and
        # classification windows, P7 clip reads, omni scanner): it counts once.
        status = await fusion.status()
        metrics = status["metrics"]
        assert metrics["localization_excluded_sensor_counts_by_reason"] == {SAMPLE_RATE_MISMATCH_SKIP_REASON: 1}
        assert metrics["localization_excluded_sensor_counts_by_sensor"] == {
            mismatched_sensor_id: {SAMPLE_RATE_MISMATCH_SKIP_REASON: 1}
        }
        assert metrics["localization_excluded_sensor_recent_counts_by_reason"] == {SAMPLE_RATE_MISMATCH_SKIP_REASON: 1}

        solve_stage = _solve_status(metrics, fusion_available=True)
        recent_param = next(param for param in solve_stage.metrics if param.label == "Rate-mismatch exclusions (5 min)")
        total_param = next(param for param in solve_stage.metrics if param.label == "Rate-mismatch exclusions (total)")
        assert int(recent_param.value) == 1 and int(total_param.value) == 1
        assert "1 sensor(s) excluded in the last 5 min: sample-rate mismatch" in solve_stage.summary

        # A second solve counts once more.
        assert await fusion._localize_candidate(_rate_candidate("evt-rate-2")) is not None
        metrics = (await fusion.status())["metrics"]
        assert metrics["localization_excluded_sensor_counts_by_reason"] == {SAMPLE_RATE_MISMATCH_SKIP_REASON: 2}
    finally:
        await fusion.stop()
        await storage.close()


async def test_matched_rate_cluster_shows_no_skips(tmp_path: Path) -> None:
    fusion, storage = await _fusion_with_point_network(tmp_path, mismatched_node_rate_hz=None)
    try:
        product, detection_product = await _localize_and_classify(fusion)

        assert LOCALIZATION_EXCLUDED_SENSORS_FEATURE_KEY not in product.extra_classification_features
        assert LOCALIZATION_EXCLUDED_SENSORS_FEATURE_KEY not in detection_product.detection.feature_summary
        status = await fusion.status()
        assert status["metrics"]["localization_excluded_sensor_counts_by_reason"] == {}
        assert status["metrics"]["localization_excluded_sensor_counts_by_sensor"] == {}
        assert status["metrics"]["localization_excluded_sensor_recent_counts_by_sensor"] == {}
        solve_stage = _solve_status(status["metrics"], fusion_available=True)
        recent_param = next(param for param in solve_stage.metrics if param.label == "Rate-mismatch exclusions (5 min)")
        assert int(recent_param.value) == 0
        assert "excluded" not in solve_stage.summary
    finally:
        await fusion.stop()
        await storage.close()


class _SteppedClock:
    def __init__(self) -> None:
        self.now_seconds = 0.0

    def __call__(self) -> float:
        return self.now_seconds


def test_the_recent_window_forgets_an_old_mismatch_and_the_dag_stops_warning() -> None:
    clock = _SteppedClock()
    ledger = SolveSensorExclusionLedger(recent_window_seconds=300.0, monotonic_clock=clock)
    ledger.record_solve([{"sensor_id": "pt-c:ch0", "reason": SAMPLE_RATE_MISMATCH_SKIP_REASON}])
    assert ledger.recent_counts_by_sensor() == {"pt-c:ch0": {SAMPLE_RATE_MISMATCH_SKIP_REASON: 1}}

    clock.now_seconds = 301.0
    assert ledger.recent_counts_by_sensor() == {}
    assert ledger.counts_by_sensor() == {"pt-c:ch0": {SAMPLE_RATE_MISMATCH_SKIP_REASON: 1}}
    metrics = {
        "localization_stage_out": 0,
        "localization_excluded_sensor_counts_by_reason": {SAMPLE_RATE_MISMATCH_SKIP_REASON: 1},
        "localization_excluded_sensor_recent_counts_by_reason": {},
        "localization_excluded_sensor_recent_counts_by_sensor": {},
        "localization_excluded_sensor_recent_window_seconds": 300.0,
    }
    solve_stage = _solve_status(metrics, fusion_available=True)
    assert solve_stage.health == "ok" and "excluded" not in solve_stage.summary
