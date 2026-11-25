"""Run one range-matrix cell through the real sidecar and the full app, then score it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.range_matrix_cells import (
    COORDINATE_MODE,
    LOCALIZATION_WINDOW_SECONDS,
    NEAREST_SENSOR_IN_BAND_SNR_DB,
    SCENE_HUMIDITY_FRACTION,
    SCENE_TEMPERATURE_C,
    SITE_ORIGIN,
    SOURCE_BAND_HZ,
    RangeMatrixCell,
    realize_cell,
)
from tests.range_matrix_scoring import CellOutcome, detections_overlapping_span, event_arrival_span_ns, score_cell
from tests.realistic_capture_synthesis import Emission, SourceBurstSpec, render_calibrated_captures
from tests.realistic_scene_full_app_run import StreamingNode, run_streaming_scene_through_full_app

# Suburban ambient over the whole 0-24 kHz band (-80 dBFS).
SENSOR_NOISE_RMS = 1.0e-4
SENSOR_NOISE_SEED = 5050
SOURCE_BURST = SourceBurstSpec(duration_seconds=0.5, band_hz=SOURCE_BAND_HZ, edge_ramp_seconds=0.005, seed=20261124)
PRE_ROLL_SECONDS = 1.0
TAIL_SECONDS = 2.0
DETECTION_WAIT_SECONDS = 20.0

RANGE_MATRIX_APP_ENVIRONMENT = {
    "MINIMAPPR_SITE_ORIGIN_SOURCE": "manual",
    "MINIMAPPR_SITE_ORIGIN_LAT": repr(SITE_ORIGIN.lat),
    "MINIMAPPR_SITE_ORIGIN_LON": repr(SITE_ORIGIN.lon),
    "MINIMAPPR_SITE_ORIGIN_ALT_M": repr(SITE_ORIGIN.alt_m),
    "MINIMAPPR_COORDINATE_MODE": COORDINATE_MODE,
    "MINIMAPPR_DEFAULT_TEMPERATURE_C": repr(SCENE_TEMPERATURE_C),
    "MINIMAPPR_DEFAULT_HUMIDITY": repr(SCENE_HUMIDITY_FRACTION),
    # Absolute gate 12 dB over the ambient floor: every cell's nearest sensor
    # is 30 dB over it in band, so E5 never measures trigger admission (E7 does).
    "MINIMAPPR_TRIGGER_RMS": repr(4.0 * SENSOR_NOISE_RMS),
    "MINIMAPPR_TRIGGER_COOLDOWN_SECONDS": "5.0",
    # Render interval = classification window - 2 s BirdNET overlap = 1 s.
    "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS": "3.0",
    "MINIMAPPR_REPORTING_WINDOW_SECONDS": "0.25",
    "MINIMAPPR_LOCALIZATION_WINDOW_SECONDS": repr(LOCALIZATION_WINDOW_SECONDS),
    # Settings defaults (the shared e2e base narrows them for small scenes).
    "MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M": "0.5",
    "MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M": "2.0",
}


@dataclass
class LiveCellResult:
    outcome: CellOutcome
    detection_count: int
    manifest_count: int


def run_range_matrix_cell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cell: RangeMatrixCell) -> LiveCellResult:
    realized = realize_cell(cell)
    verdict = realized.oracle_verdict()
    _first_arrival_s, last_arrival_end_s = realized.scene.emission_to_arrival_span_seconds(SOURCE_BURST.duration_seconds)
    captures = render_calibrated_captures(
        realized.scene,
        [Emission(PRE_ROLL_SECONDS, NEAREST_SENSOR_IN_BAND_SNR_DB, SOURCE_BURST)],
        sensor_gain_relative_to_reference=realized.sensor_gain_relative_to_nearest,
        capture_seconds=PRE_ROLL_SECONDS + last_arrival_end_s + TAIL_SECONDS,
        sensor_noise_rms=SENSOR_NOISE_RMS,
        noise_seed=SENSOR_NOISE_SEED,
    )
    streaming_nodes = [
        StreamingNode(
            scene_node=reported.scene_node,
            time_quality=reported.time_quality,
            packet_clock_error_ns=reported.packet_clock_error_ns,
            entered_position_enu_m=reported.entered_position_enu_m,
            entered_position_std_m=reported.stated_position_std_m if reported.entered_position_enu_m else None,
        )
        for reported in realized.reported_nodes
    ]
    emission_offset_ns = int(round(PRE_ROLL_SECONDS * 1e9))

    def event_detected(result) -> bool:
        span = event_arrival_span_ns(
            realized, result.capture_start_packet_time_ns + emission_offset_ns, SOURCE_BURST.duration_seconds
        )
        return bool(detections_overlapping_span(result.detections, span))

    run = run_streaming_scene_through_full_app(
        monkeypatch,
        tmp_path,
        scene=realized.scene,
        nodes=streaming_nodes,
        captures_by_node=captures,
        app_environment=RANGE_MATRIX_APP_ENVIRONMENT,
        detection_wait_seconds=DETECTION_WAIT_SECONDS,
        detections_complete=event_detected,
    )
    outcome = score_cell(
        realized,
        verdict,
        run.detections,
        emission_packet_time_ns=run.capture_start_packet_time_ns + emission_offset_ns,
        burst_seconds=SOURCE_BURST.duration_seconds,
    )
    return LiveCellResult(outcome=outcome, detection_count=len(run.detections), manifest_count=len(run.manifests_by_id))
