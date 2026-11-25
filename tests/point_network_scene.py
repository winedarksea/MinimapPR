"""A four-point-node network and its callers, for P7 point-network clip tests.

Captures carry spherical spreading (amplitude / distance) as well as exact
fractional propagation delays, so the node nearest a caller really is the
clearest, and several callers simply add. Rust render manifests are built per
``docs/localization_solve_contract.md`` (as in
``tests/localization_solve_manifest_fixtures.py``) for a point-only cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from minimappr.core.audio_buffer import MultiSensorBuffer
from tests.acoustic_scene_synthesis import band_limited_noise_burst, fractionally_delayed, sensor_id_for
from tests.localization_solve_manifest_fixtures import classifier_render_manifest

NANOSECONDS_PER_SECOND = 1_000_000_000
SAMPLE_RATE_HZ = 16_000
SOUND_SPEED_MPS = 343.0
POINT_NETWORK_CLUSTER_ID = "point-net"
POINT_NODE_POSITIONS_M: dict[str, tuple[float, float, float]] = {
    "pt-a": (0.0, 0.0, 1.5),
    "pt-b": (60.0, 0.0, 1.5),
    "pt-c": (0.0, 60.0, 1.5),
    "pt-d": (60.0, 60.0, 1.5),
}
# A whole number of samples after the epoch, so buffer sample indices are exact.
CAPTURE_START_NS = 1_764_000_000_000_000_000
CAPTURE_SECONDS = 2.0
SENSOR_NOISE_RMS = 0.002
# A bird-band burst (the heuristic classifier calls it ``bird_like``).
CALL_BAND_HZ = (2500.0, 6000.0)
CALL_DURATION_SECONDS = 0.15
CALL_EDGE_RAMP_SECONDS = 0.002
# Source level: peak amplitude at 1 m.
CALL_PEAK_AMPLITUDE_AT_ONE_METRE = 1.0


@dataclass(frozen=True, slots=True)
class Caller:
    position_m: tuple[float, float, float]
    emission_offset_seconds: float  # after CAPTURE_START_NS
    amplitude_scale: float = 1.0
    seed: int = 11
    # The Python-ingest solve uses the environment's speed of sound; its scene must too.
    sound_speed_mps: float = SOUND_SPEED_MPS

    def waveform(self) -> np.ndarray:
        return band_limited_noise_burst(
            sample_rate_hz=SAMPLE_RATE_HZ,
            duration_seconds=CALL_DURATION_SECONDS,
            band_hz=CALL_BAND_HZ,
            peak_amplitude=CALL_PEAK_AMPLITUDE_AT_ONE_METRE * self.amplitude_scale,
            edge_ramp_seconds=CALL_EDGE_RAMP_SECONDS,
            seed=self.seed,
        )

    def distance_m(self, node_id: str) -> float:
        return float(np.linalg.norm(np.subtract(self.position_m, POINT_NODE_POSITIONS_M[node_id])))

    def arrival_time_ns(self, node_id: str) -> int:
        seconds = self.emission_offset_seconds + self.distance_m(node_id) / self.sound_speed_mps
        return CAPTURE_START_NS + int(round(seconds * NANOSECONDS_PER_SECOND))


def point_sensor_id(node_id: str) -> str:
    return sensor_id_for(node_id, 0)


def render_point_node_captures(callers: list[Caller], *, noise_seed: int = 3) -> dict[str, np.ndarray]:
    capture_samples = int(round(CAPTURE_SECONDS * SAMPLE_RATE_HZ))
    rng = np.random.default_rng(noise_seed)
    captures: dict[str, np.ndarray] = {}
    for node_id in POINT_NODE_POSITIONS_M:
        mixed = SENSOR_NOISE_RMS * rng.standard_normal(capture_samples)
        for caller in callers:
            distance_m = caller.distance_m(node_id)
            delay_samples = (caller.emission_offset_seconds + distance_m / caller.sound_speed_mps) * SAMPLE_RATE_HZ
            mixed += fractionally_delayed(caller.waveform(), delay_samples, capture_samples) / distance_m
        captures[point_sensor_id(node_id)] = mixed.astype(np.float32)
    return captures


async def append_captures(buffer: MultiSensorBuffer, captures: dict[str, np.ndarray]) -> None:
    for sensor_id, samples in captures.items():
        await buffer.append(
            sensor_id=sensor_id, sample_rate_hz=SAMPLE_RATE_HZ, start_time_ns=CAPTURE_START_NS, samples=samples
        )


def point_node_payload(node_id: str) -> dict[str, Any]:
    return {
        "id": node_id,
        "node_type": "point",
        "position_m": list(POINT_NODE_POSITIONS_M[node_id]),
        "sensor_offsets_m": [[0.0, 0.0, 0.0]],
        "capabilities": ["audio"],
        "metadata": {},
    }


def point_network_render_manifest(
    *,
    caller: Caller,
    reporting_node_id: str,
    frame_id: str,
    solve_id: str,
    manifest_id: str,
    solve_window_seconds: float,
    render_half_span_seconds: float,
) -> dict[str, Any]:
    """A ``classifier_render`` for a four-point-node cluster solve of ``caller``.

    The solve window is centred on the call's arrival at the reporting node
    (the sidecar centres it on the reporter's trigger), and the render spans
    ``render_half_span_seconds`` either side of that centre. The render's own
    audio is unrelated noise: a classification of it cannot pass for the call.
    """
    window_center_time_ns = caller.arrival_time_ns(reporting_node_id)
    sensor_ids = [point_sensor_id(node_id) for node_id in POINT_NODE_POSITIONS_M]
    positions = {point_sensor_id(node_id): position for node_id, position in POINT_NODE_POSITIONS_M.items()}
    sensors = [
        {
            "sensor_id": sensor_id,
            "node_id": sensor_id.split(":ch")[0],
            "node_channel_index": 0,
            "position_m": list(positions[sensor_id]),
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "position_source": "operator_override",
            "position_std_m": 0.05,
            "clock_source": "gps_locked",
            "clock_domain": "utc",
            "clock_offset_std_s": 1e-6,
        }
        for sensor_id in sensor_ids
    ]
    pair_tdoas = [
        {
            "ch_a": index_a,
            "ch_b": index_b,
            "delay_samples": 0.0,
            "lag_seconds": (caller.distance_m(sensor_a.split(":ch")[0]) - caller.distance_m(sensor_b.split(":ch")[0]))
            / SOUND_SPEED_MPS,
            "confidence": 0.9,
            "is_cross_node": True,
            "timing_uncertainty_seconds": 1.4e-6,
        }
        for (index_a, sensor_a), (index_b, sensor_b) in combinations(enumerate(sensor_ids), 2)
    ]
    provenance = {
        "schema_version": 1,
        "solve_id": solve_id,
        "attempt_kind": "event_trigger_absolute_rms",
        "buffer_scope": "automatic_site_cluster",
        "cluster_id": POINT_NETWORK_CLUSTER_ID,
        "reporting_node_id": reporting_node_id,
        "node_local_routing_reason": None,
        "coordinate_frame": {
            "frame_id": frame_id,
            "kind": "site_enu",
            "origin": {"lat": 37.0, "lon": -122.0, "alt_m": 0.0},
            "projection": "flat_wgs84_series",
        },
        "layout_revision": 1,
        "geometry_push_revision": 1,
        "sensors": sensors,
        "active_sensor_indices": list(range(len(sensors))),
        "excluded_sensors": [],
        "excluded_pairs": [],
        "node_timing": [
            {"node_id": node_id, "time_quality": "gps_locked", "clock_offset_std_s": 1e-6,
             "timeline_residual_ns": 900, "last_timestamp_step_ns": None, "timestamp_step_count": 0}
            for node_id in POINT_NODE_POSITIONS_M
        ],
        "window_center_time_ns": window_center_time_ns,
        "window_seconds": solve_window_seconds,
        "solve_sample_rate_hz": SAMPLE_RATE_HZ,
    }
    localization = {
        "attempted_algorithm": "srp_phat",
        "resolved_algorithm": "srp_phat",
        "position_m": list(caller.position_m),
        "confidence": 0.9,
        "range_projection_mode": "range_refined",
        "sound_speed_mps": SOUND_SPEED_MPS,
        "pair_tdoas": pair_tdoas,
        "solve_provenance": provenance,
    }
    render_half_span_ns = int(round(render_half_span_seconds * NANOSECONDS_PER_SECOND))
    manifest = classifier_render_manifest(
        node_payload=point_node_payload(reporting_node_id),
        localization=localization,
        toa_ns=window_center_time_ns,
        manifest_id=manifest_id,
        sample_count=int(round(2 * render_half_span_seconds * SAMPLE_RATE_HZ)),
    )
    manifest["classifier_render"]["sample_rate_hz"] = SAMPLE_RATE_HZ
    manifest["classifier_render"]["render_start_ns"] = window_center_time_ns - render_half_span_ns
    manifest["classifier_render"]["render_end_ns"] = window_center_time_ns + render_half_span_ns
    return manifest


def clip_onset_sample(clip: np.ndarray, caller: Caller) -> int:
    """Sample index in ``clip`` where ``caller``'s burst starts (cross-correlation peak)."""
    waveform = caller.waveform()
    correlation = np.correlate(np.asarray(clip, dtype=np.float64), waveform, mode="valid")
    return int(np.argmax(np.abs(correlation)))
