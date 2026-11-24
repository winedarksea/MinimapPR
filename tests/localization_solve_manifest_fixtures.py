"""Manifest fragments built per ``docs/localization_solve_contract.md``.

The sidecar does not emit ``solve_provenance`` yet (milestone 1 / R6), so the
Python side is built and tested against payloads constructed from the contract
doc. ``contract_doc_example_solve_provenance`` is the doc's section-9 example
verbatim; ``mixed_cluster_solve_localization_payload`` builds a planar + point
cluster solve in either node arrival order.
"""

from __future__ import annotations

import base64
import copy
from itertools import combinations
from typing import Any

import numpy as np

CONTRACT_DOC_EXAMPLE_SOLVE_PROVENANCE: dict[str, Any] = {
    "schema_version": 1,
    "solve_id": "solve-3f2a",
    "attempt_kind": "event_trigger_absolute_rms",
    "buffer_scope": "automatic_site_cluster",
    "cluster_id": "auto-gps",
    "reporting_node_id": "planar-a",
    "coordinate_frame": {
        "frame_id": "site-7",
        "kind": "site_enu",
        "origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0},
        "projection": "flat_wgs84_series",
    },
    "layout_revision": 12,
    "geometry_push_revision": 3,
    "sensors": [
        {"sensor_id": "planar-a:ch0", "node_id": "planar-a", "node_channel_index": 0,
         "position_m": [1.0177, 2.0177, 1.5], "sample_rate_hz": 16000,
         "position_source": "operator_override", "position_std_m": 0.05,
         "clock_source": "gps_locked", "clock_domain": "utc", "clock_offset_std_s": 1e-6},
        {"sensor_id": "planar-a:ch4", "node_id": "planar-a", "node_channel_index": 4,
         "position_m": [1.0, 2.0, 1.5], "sample_rate_hz": 16000,
         "position_source": "operator_override", "position_std_m": 0.05,
         "clock_source": "gps_locked", "clock_domain": "utc", "clock_offset_std_s": 1e-6},
        {"sensor_id": "point-b:ch0", "node_id": "point-b", "node_channel_index": 0,
         "position_m": [6.0, 0.0, 1.5], "sample_rate_hz": 16000,
         "position_source": "reported_gps", "position_std_m": None,
         "clock_source": "ntp_disciplined", "clock_domain": "utc", "clock_offset_std_s": 0.002},
    ],
    "active_sensor_indices": [0, 1, 2],
    "excluded_sensors": [],
    "excluded_pairs": [
        {"sensor_index_a": 0, "sensor_index_b": 2, "reason": "timeline_residual_over_limit"},
        {"sensor_index_a": 1, "sensor_index_b": 2, "reason": "timeline_residual_over_limit"},
    ],
    "node_timing": [
        {"node_id": "planar-a", "time_quality": "gps_locked", "clock_offset_std_s": 1e-6,
         "timeline_residual_ns": 1200, "last_timestamp_step_ns": None, "timestamp_step_count": 0},
        {"node_id": "point-b", "time_quality": "ntp_disciplined", "clock_offset_std_s": 0.002,
         "timeline_residual_ns": 15000000, "last_timestamp_step_ns": -21000000, "timestamp_step_count": 1},
    ],
    "window_center_time_ns": 1764000000016000000,
    "window_seconds": 0.032,
    "solve_sample_rate_hz": 16000,
}

PLANAR_NODE_ID = "planar-a"
POINT_NODE_ID = "point-b"
MIXED_CLUSTER_ID = "auto-gps"
MIXED_WINDOW_CENTER_TIME_NS = 1_764_000_000_016_000_000
MIXED_SOUND_SPEED_MPS = 343.0
MIXED_SAMPLE_RATE_HZ = 16_000
# Planar array body offsets (5 mics: a 4-mic square plus centre).
PLANAR_BODY_OFFSETS_M = [
    (0.05, 0.0, 0.0),
    (0.0, 0.05, 0.0),
    (-0.05, 0.0, 0.0),
    (0.0, -0.05, 0.0),
    (0.0, 0.0, 0.0),
]
PLANAR_NODE_POSITION_M = (30.0, 25.0, 1.5)
POINT_NODE_POSITION_M = (33.0, 21.0, 4.0)


def contract_doc_example_solve_provenance() -> dict[str, Any]:
    return copy.deepcopy(CONTRACT_DOC_EXAMPLE_SOLVE_PROVENANCE)


def mixed_cluster_sensor_positions_m() -> dict[str, tuple[float, float, float]]:
    positions = {
        f"{PLANAR_NODE_ID}:ch{index}": tuple(np.add(PLANAR_NODE_POSITION_M, offset).tolist())
        for index, offset in enumerate(PLANAR_BODY_OFFSETS_M)
    }
    positions[f"{POINT_NODE_ID}:ch0"] = POINT_NODE_POSITION_M
    return positions


def mixed_cluster_ordered_sensor_ids(arrival_order: tuple[str, ...]) -> list[str]:
    """Buffer order follows node arrival order, as the sidecar's cluster registry does."""
    sensor_ids = list(mixed_cluster_sensor_positions_m())
    return [sensor_id for node_id in arrival_order for sensor_id in sensor_ids if sensor_id.startswith(f"{node_id}:")]


def arrival_lag_seconds(source_m, sensor_a_m, sensor_b_m, sound_speed_mps: float = MIXED_SOUND_SPEED_MPS) -> float:
    """arrival(a) - arrival(b), the Rust ``lag_seconds`` convention."""
    source = np.asarray(source_m, dtype=np.float64)
    return float(
        (np.linalg.norm(source - np.asarray(sensor_a_m)) - np.linalg.norm(source - np.asarray(sensor_b_m)))
        / sound_speed_mps
    )


def mixed_cluster_solve_localization_payload(
    *,
    arrival_order: tuple[str, ...],
    source_position_m,
    frame_id: str,
    reporting_node_id: str,
    frame_kind: str = "site_enu",
    solve_id: str = "solve-mixed-1",
    sidecar_position_m=None,
) -> dict[str, Any]:
    """A ``localization`` payload for the six-sensor planar + point solve."""
    positions = mixed_cluster_sensor_positions_m()
    ordered_ids = mixed_cluster_ordered_sensor_ids(arrival_order)
    sensors = [
        {
            "sensor_id": sensor_id,
            "node_id": sensor_id.split(":ch")[0],
            "node_channel_index": int(sensor_id.split(":ch")[1]),
            "position_m": list(positions[sensor_id]),
            "sample_rate_hz": MIXED_SAMPLE_RATE_HZ,
            "position_source": "reported_gps",
            "position_std_m": None,
            "clock_source": "gps_locked",
            "clock_domain": "utc",
            "clock_offset_std_s": 1e-6,
        }
        for sensor_id in ordered_ids
    ]
    pair_tdoas = [
        {
            "ch_a": index_a,
            "ch_b": index_b,
            "delay_samples": 0.0,
            "lag_seconds": arrival_lag_seconds(source_position_m, positions[sensor_a], positions[sensor_b]),
            "confidence": 0.9,
            "is_cross_node": sensor_a.split(":ch")[0] != sensor_b.split(":ch")[0],
            "timing_uncertainty_seconds": 1.4e-6,
        }
        for (index_a, sensor_a), (index_b, sensor_b) in combinations(enumerate(ordered_ids), 2)
    ]
    provenance = {
        "schema_version": 1,
        "solve_id": solve_id,
        "attempt_kind": "event_trigger_absolute_rms",
        "buffer_scope": "automatic_site_cluster",
        "cluster_id": MIXED_CLUSTER_ID,
        "reporting_node_id": reporting_node_id,
        "coordinate_frame": {
            "frame_id": frame_id,
            "kind": frame_kind,
            "origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0},
            "projection": "flat_wgs84_series",
        },
        "layout_revision": 4,
        "geometry_push_revision": 2,
        "sensors": sensors,
        "active_sensor_indices": list(range(len(sensors))),
        "excluded_sensors": [],
        "excluded_pairs": [],
        "node_timing": [
            {"node_id": node_id, "time_quality": "gps_locked", "clock_offset_std_s": 1e-6,
             "timeline_residual_ns": 900, "last_timestamp_step_ns": None, "timestamp_step_count": 0}
            for node_id in arrival_order
        ],
        "window_center_time_ns": MIXED_WINDOW_CENTER_TIME_NS,
        "window_seconds": 0.032,
        "solve_sample_rate_hz": MIXED_SAMPLE_RATE_HZ,
    }
    return {
        "attempted_algorithm": "srp_phat",
        "resolved_algorithm": "srp_phat",
        "position_m": list(sidecar_position_m if sidecar_position_m is not None else source_position_m),
        "confidence": 0.9,
        "range_projection_mode": "range_refined",
        "sound_speed_mps": MIXED_SOUND_SPEED_MPS,
        "pair_tdoas": pair_tdoas,
        "solve_provenance": provenance,
    }


def planar_node_payload(position_m=PLANAR_NODE_POSITION_M, yaw_deg: float = 0.0) -> dict[str, Any]:
    return {
        "id": PLANAR_NODE_ID,
        "node_type": "sirith_planar",
        "position_m": list(position_m),
        "orientation": {"yaw_deg": yaw_deg, "pitch_deg": 0.0, "roll_deg": 0.0},
        "sensor_offsets_m": [list(offset) for offset in PLANAR_BODY_OFFSETS_M],
        "capabilities": ["audio"],
        "metadata": {},
    }


def point_node_payload(position_m=POINT_NODE_POSITION_M) -> dict[str, Any]:
    return {
        "id": POINT_NODE_ID,
        "node_type": "point",
        "position_m": list(position_m),
        "sensor_offsets_m": [[0.0, 0.0, 0.0]],
        "capabilities": ["audio"],
        "metadata": {},
    }


def classifier_render_manifest(
    *,
    node_payload: dict[str, Any],
    localization: dict[str, Any] | None,
    toa_ns: int | None,
    manifest_id: str = "render-1",
    sample_count: int = 1600,
    audio_seed: int = 7,
) -> dict[str, Any]:
    """A memory-path ``classifier_render`` manifest with inline PCM."""
    rng = np.random.default_rng(audio_seed)
    pcm = (rng.normal(0.0, 0.3, size=sample_count).clip(-1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    node_context: dict[str, Any] = {"node": node_payload, "time_quality": "gps_locked"}
    if toa_ns is not None:
        node_context["toa_ns"] = toa_ns
    stream_key = str(node_payload["id"])
    manifest: dict[str, Any] = {
        "manifest_id": manifest_id,
        "manifest_type": "classifier_render",
        "source_handles": [
            {"journal_epoch": 1, "stream_key": stream_key,
             "segment_id": "seg-mem-00000000000000000001-00000000000000000002",
             "payload_offset_bytes": 0, "payload_length_bytes": 0, "segment_path": "/nonexistent"}
        ],
        "derived_handle": {"journal_epoch": 1, "stream_key": stream_key, "segment_id": "seg-mem-1-3",
                           "payload_offset_bytes": 0, "payload_length_bytes": len(pcm),
                           "segment_path": "/nonexistent"},
        "classifier_render": {"sample_format": "pcm16le", "channels": 1, "sample_rate_hz": MIXED_SAMPLE_RATE_HZ,
                              "render_kind": "birdnet_hybrid_spatial_blend"},
        "raw_render_bytes": base64.b64encode(pcm).decode("ascii"),
        "node_context": node_context,
    }
    if localization is not None:
        manifest["localization"] = localization
    return manifest
