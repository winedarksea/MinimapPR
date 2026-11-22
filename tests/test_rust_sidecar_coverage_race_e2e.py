"""Coverage race through the real Rust sidecar binary (Mode 2 ingest).

Two tetrahedral nodes share one explicit cluster. Node b's frame k is posted
after node a's frame k+1 (one frame of network skew). Before the fix every
cooldown-gated localization ran on node a's channels alone ("0 cross-node
pairs" on the live box); with the coverage wait on (default) every one of them
must contain cross-node TDOA pairs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tests.helpers import SIRITH_TETRA_SENSOR_OFFSETS_M
from tests.test_rust_manifest_handoff_e2e import (
    _http_json,
    _running_rust_sidecar,
    _store_forward_payload,
    _wait_for_live_dsp_results,
)

SAMPLE_RATE_HZ = 16_000
FRAME_SAMPLES = 512
FRAME_DURATION_NS = FRAME_SAMPLES * 1_000_000_000 // SAMPLE_RATE_HZ
TETRA_CHANNEL_COUNT = 4
NODE_IDS = ("node-a", "node-b")
NODE_B_EAST_OFFSET_M = 3.0
CLUSTER_ID = "coverage-race-live"
# Two default trigger cooldowns (0.8 s) after the first solve at frame 0.
FRAME_COUNT = 60
SIGNAL_SEED = 20261122
SIGNAL_AMPLITUDE = 0.3
RESULTS_POLL_LIMIT = 200


def _cluster_sensor_positions() -> dict[str, list[float]]:
    positions: dict[str, list[float]] = {}
    for node_index, node_id in enumerate(NODE_IDS):
        east_offset_m = node_index * NODE_B_EAST_OFFSET_M
        for channel_index, offset_m in enumerate(SIRITH_TETRA_SENSOR_OFFSETS_M):
            positions[f"{node_id}:ch{channel_index}"] = [
                float(offset_m[0]) + east_offset_m,
                float(offset_m[1]),
                float(offset_m[2]),
            ]
    return positions


def _frame_payload(node_id: str, frame_index: int, start_time_ns: int, frame_channels: np.ndarray) -> bytes:
    payload = _store_forward_payload(
        node_id=node_id,
        node_type="sirith_tetra",
        sensor_offsets_m=[[float(axis) for axis in offset] for offset in SIRITH_TETRA_SENSOR_OFFSETS_M],
        channels=frame_channels,
        start_time_ns=start_time_ns + frame_index * FRAME_DURATION_NS,
        sequence=frame_index + 1,
        cluster_id=CLUSTER_ID,
        cluster_sensor_positions=_cluster_sensor_positions(),
        sample_rate_hz=SAMPLE_RATE_HZ,
    )
    frame = payload["buffered_frames"][0]["frame"]
    frame["start_sample_index"] = frame_index * FRAME_SAMPLES
    frame["end_sample_index"] = (frame_index + 1) * FRAME_SAMPLES
    return json.dumps(payload).encode("utf-8")


def _post_frame(port: int, body: bytes) -> None:
    response = _http_json(
        port,
        "/api/v1/ingest/store-forward",
        method="POST",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    assert response["accepted"] is True


def _is_cross_node_pair(pair: dict) -> bool:
    return (int(pair["ch_a"]) < TETRA_CHANNEL_COUNT) != (int(pair["ch_b"]) < TETRA_CHANNEL_COUNT)


def _cadence_localizations(payloads: list[dict]) -> list[dict]:
    """Cadence-triggered SRP solves (frame 0's solve rides the classifier render)."""
    return [
        payload
        for payload in payloads
        if payload.get("manifest_type") == "localization_result"
        and payload.get("cluster_id") == CLUSTER_ID
        and isinstance(payload.get("localization"), dict)
        and payload["localization"].get("pair_tdoas")
        and "window_type" not in (payload.get("coverage_stats") or {})
    ]


def _replay_with_node_b_one_frame_late(coverage_wait_enabled: bool, tmp_path: Path) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(SIGNAL_SEED)
    channels_by_node = {
        node_id: (SIGNAL_AMPLITUDE * rng.standard_normal((TETRA_CHANNEL_COUNT, FRAME_COUNT * FRAME_SAMPLES))).astype(
            np.float32
        )
        for node_id in NODE_IDS
    }
    start_time_ns = time.time_ns()

    def frame_body(node_id: str, frame_index: int) -> bytes:
        samples = slice(frame_index * FRAME_SAMPLES, (frame_index + 1) * FRAME_SAMPLES)
        return _frame_payload(node_id, frame_index, start_time_ns, channels_by_node[node_id][:, samples])

    with _running_rust_sidecar(
        tmp_path,
        extra_env={
            "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS": "15.0",
            "MINIMAPPR_CLASSIFIER_RENDER_MIN_INTERVAL_SECONDS": "1000000.0",
            "MINIMAPPR_DSP_LOCALIZATION_RMS_GATE": "0.0",
            "MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M": "0.25",
            "MINIMAPPR_LOCALIZATION_COVERAGE_WAIT_ALL_STREAMING_NODES_ENABLED": str(coverage_wait_enabled).lower(),
        },
    ) as (port, _spool_dir):
        # /api/v1/dsp/results is a 50-entry ring that raw_audio_frame events
        # flood, so collect localizations as they appear instead of at the end.
        localizations_by_manifest_id: dict[str, dict] = {}

        def collect_localizations(results: list[dict]) -> bool:
            for localization in _cadence_localizations(results):
                localizations_by_manifest_id[localization["manifest_id"]] = localization
            # Triggers at frames 25 and 50 (0.8 s audio cooldown) must both land.
            return len(localizations_by_manifest_id) >= 2

        for frame_index in range(FRAME_COUNT):
            _post_frame(port, frame_body(NODE_IDS[0], frame_index))
            if frame_index > 0:
                _post_frame(port, frame_body(NODE_IDS[1], frame_index - 1))
            collect_localizations(_http_json(port, f"/api/v1/dsp/results?limit={RESULTS_POLL_LIMIT}"))
        _post_frame(port, frame_body(NODE_IDS[1], FRAME_COUNT - 1))
        _wait_for_live_dsp_results(
            port,
            description="two cadence-triggered cluster localizations",
            predicate=collect_localizations,
        )
        status = _http_json(port, "/api/v1/dsp/status")
    assert isinstance(status, dict)
    return list(localizations_by_manifest_id.values()), status


def test_live_sidecar_coverage_wait_uses_cross_node_pairs_when_a_node_lags(tmp_path: Path) -> None:
    localizations, status = _replay_with_node_b_one_frame_late(True, tmp_path)
    for localization_manifest in localizations:
        pairs = localization_manifest["localization"]["pair_tdoas"]
        assert any(_is_cross_node_pair(pair) for pair in pairs), (
            "coverage wait must bring the late node into every cluster solve; "
            f"got {len(pairs)} pairs: {pairs}"
        )
    stop_reasons = status["localization_coverage_wait_stop_reasons"]
    assert stop_reasons.get("all_covered", 0) >= 2
    assert stop_reasons.get("total_cap", 0) == 0
    # Bounded by the 1 s cap even on a loaded CI box.
    assert status["localization_coverage_wait_max_ms"] <= 1_000.0


def test_live_sidecar_without_coverage_wait_reproduces_single_node_solves(tmp_path: Path) -> None:
    # Flag off == pre-fix dispatch: the bug as seen on the live box.
    localizations, _status = _replay_with_node_b_one_frame_late(False, tmp_path)
    cross_node_pair_counts = [
        sum(_is_cross_node_pair(pair) for pair in manifest["localization"]["pair_tdoas"])
        for manifest in localizations
    ]
    assert cross_node_pair_counts and all(count == 0 for count in cross_node_pair_counts), cross_node_pair_counts
