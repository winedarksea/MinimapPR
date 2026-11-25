"""Synthetic calibration bundles for the replay-realism e2e (plan item E4).

A scene (``tests/acoustic_scene_synthesis.py``) is rendered per node at that
node's own sample rate, cut into firmware frames whose packet timing can carry
a timestamp step, and written as a real calibration bundle (WAV per node plus
``frame_timing/{node_id}.json``), so the e2e exercises bundle -> per-frame
uploads (``minimappr/sim/replay_frames.py``) -> sidecar
(``minimappr/sim/replay_sidecar.py``) exactly as a field bundle would.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from minimappr.calibration.bundle import build_ground_truth_payload, write_bundle_zip
from minimappr.calibration.frame_timing import NodeFrameTimingTable, frame_timing_bundle_path
from minimappr.calibration.pipeline import BUNDLE_KIND, SCHEMA_VERSION, write_multichannel_wav
from minimappr.core.capture_audio_buffer import CapturedFrameTiming
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import speed_of_sound_mps
from minimappr.models import NodeType, TimeQuality, Vec3
from minimappr.sim.binary_ingest_encoder import ContinuousCaptureFraming, frames_from_continuous_capture
from tests.acoustic_scene_synthesis import AcousticScene, SceneNode, band_limited_noise_burst, scene_node_near_enu
from tests.helpers import SIRITH_TETRA_SENSOR_OFFSETS_M
from tests.mixed_cluster_acceptance_scene import (
    COORDINATE_MODE,
    PLANAR_SENSOR_OFFSETS_BODY_M,
    SCENE_HUMIDITY_FRACTION,
    SCENE_TEMPERATURE_C,
    SENSOR_NOISE_RMS,
    SITE_ORIGIN,
    SOURCE_BAND_HZ,
    SOURCE_DURATION_SECONDS,
    SOURCE_EDGE_RAMP_SECONDS,
    SOURCE_PEAK_AMPLITUDE,
    SOURCE_SEED,
)

# --- Rates and framing ---
SAMPLE_RATE_48_KHZ = 48_000
SAMPLE_RATE_44_1_KHZ = 44_100
# The source is synthesised at the higher rate and resampled exactly (147/160)
# for 44.1 kHz nodes, so every node hears the same waveform.
SOURCE_SYNTHESIS_RATE_HZ = SAMPLE_RATE_48_KHZ
RESAMPLE_UP_BY_RATE = {SAMPLE_RATE_48_KHZ: 1, SAMPLE_RATE_44_1_KHZ: 147}
RESAMPLE_DOWN_BY_RATE = {SAMPLE_RATE_48_KHZ: 1, SAMPLE_RATE_44_1_KHZ: 160}
# 20 ms firmware frames at either rate (960 / 882 samples), 5 per upload (100 ms).
FRAME_DURATION_SECONDS = 0.02
FRAMES_PER_UPLOAD = 5
PRE_ROLL_SECONDS = 1.0
CAPTURE_SECONDS = 2.0
SENSOR_NOISE_SEED_BY_RATE = {SAMPLE_RATE_48_KHZ: 1124, SAMPLE_RATE_44_1_KHZ: 4411}
# Arbitrary recording epoch; replay shifts it next to the server clock.
BUNDLE_CAPTURE_START_NS = 1_764_000_000_000_000_000
SESSION_ID = "e4-replay-realism"

# --- Geometry (site ENU, relative to the planar node's GPS fix) ---
PLANAR_NODE_ID = "planar-replay"
TETRA_NODE_ID = "tetra-replay"
POINT_NODE_ID = "point-replay"
PLANAR_APPROXIMATE_ENU_M: Vec3 = (37.0, -22.0, 1.5)
# Tetra up a 2.5 m mast past the source, so the SRP search box (sensor bounds
# plus padding) contains the source.
TETRA_OFFSET_FROM_PLANAR_M: Vec3 = (2.0, 3.5, 2.5)
# Point on a 4 m pole, as in the E2 scene.
POINT_OFFSET_FROM_PLANAR_M: Vec3 = (1.0, 2.0, 4.0)
SOURCE_OFFSET_FROM_PLANAR_M: Vec3 = (1.0, 2.5, 2.0)
POINT_SENSOR_OFFSETS_M: tuple[Vec3, ...] = ((0.0, 0.0, 0.0),)


@dataclass(frozen=True, slots=True)
class ReplayNodePlan:
    node_id: str
    node_type: NodeType
    sensor_offsets_m: tuple[Vec3, ...]
    offset_from_planar_m: Vec3
    sample_rate_hz: int


@dataclass(frozen=True, slots=True)
class PacketTimestampStep:
    """At ``first_stepped_frame_index`` the node's packet clock jumps by
    ``step_ns``; samples stay continuous (no sequence or sample gap).

    By default the clock steps *off* true time (later packets read ``step_ns``
    late). With ``corrects_earlier_offset`` it steps *onto* true time: earlier
    packets read ``step_ns`` early, as before an SNTP/GPS re-sync."""

    node_id: str
    first_stepped_frame_index: int
    step_ns: int
    corrects_earlier_offset: bool = False

    def offset_ns(self, frame_index: int) -> int:
        if self.corrects_earlier_offset:
            return -self.step_ns if frame_index < self.first_stepped_frame_index else 0
        return self.step_ns if frame_index >= self.first_stepped_frame_index else 0


@dataclass(frozen=True, slots=True)
class PacketClockDrift:
    """The node's packet clock runs ``ns_per_frame`` fast of its sample clock
    per frame (time-compressed crystal drift); samples stay continuous."""

    node_id: str
    ns_per_frame: int

    def offset_ns(self, frame_index: int) -> int:
        return frame_index * self.ns_per_frame


def planar_plan(sample_rate_hz: int) -> ReplayNodePlan:
    return ReplayNodePlan(PLANAR_NODE_ID, NodeType.SIRITH_PLANAR, PLANAR_SENSOR_OFFSETS_BODY_M, (0.0, 0.0, 0.0), sample_rate_hz)


def tetra_plan(sample_rate_hz: int) -> ReplayNodePlan:
    return ReplayNodePlan(
        TETRA_NODE_ID, NodeType.SIRITH_TETRA, SIRITH_TETRA_SENSOR_OFFSETS_M, TETRA_OFFSET_FROM_PLANAR_M, sample_rate_hz
    )


def point_plan(sample_rate_hz: int) -> ReplayNodePlan:
    return ReplayNodePlan(POINT_NODE_ID, NodeType.POINT, POINT_SENSOR_OFFSETS_M, POINT_OFFSET_FROM_PLANAR_M, sample_rate_hz)


def _offset(base: Sequence[float], delta: Sequence[float]) -> Vec3:
    return (float(base[0] + delta[0]), float(base[1] + delta[1]), float(base[2] + delta[2]))


def frame_samples_for(sample_rate_hz: int) -> int:
    return int(round(sample_rate_hz * FRAME_DURATION_SECONDS))


def build_replay_scene(node_plans: Sequence[ReplayNodePlan]) -> AcousticScene:
    """Scene truth at the source synthesis rate; nodes in plan order (planar first)."""
    frame = LocalCoordinateFrame(origin=SITE_ORIGIN, mode=COORDINATE_MODE)
    nodes: list[SceneNode] = []
    planar_position_m: Vec3 | None = None
    for plan in node_plans:
        approximate = _offset(PLANAR_APPROXIMATE_ENU_M, plan.offset_from_planar_m)
        node = scene_node_near_enu(
            frame,
            node_id=plan.node_id,
            node_type=plan.node_type,
            approximate_position_enu_m=approximate,
            sensor_offsets_body_m=plan.sensor_offsets_m,
        )
        nodes.append(node)
        if plan.node_id == PLANAR_NODE_ID:
            planar_position_m = tuple(frame.geo_to_local(node.position_geo))
    if planar_position_m is None:
        raise ValueError("every replay scene is placed relative to the planar node")
    return AcousticScene(
        site_origin=SITE_ORIGIN,
        coordinate_mode=COORDINATE_MODE,
        nodes=tuple(nodes),
        source_position_enu_m=_offset(planar_position_m, SOURCE_OFFSET_FROM_PLANAR_M),
        sample_rate_hz=SOURCE_SYNTHESIS_RATE_HZ,
        sound_speed_mps=speed_of_sound_mps(SCENE_TEMPERATURE_C, SCENE_HUMIDITY_FRACTION),
    )


def render_captures_at_node_rates(scene: AcousticScene, node_plans: Sequence[ReplayNodePlan]) -> dict[str, np.ndarray]:
    source_at_synthesis_rate = band_limited_noise_burst(
        sample_rate_hz=SOURCE_SYNTHESIS_RATE_HZ,
        duration_seconds=SOURCE_DURATION_SECONDS,
        band_hz=SOURCE_BAND_HZ,
        peak_amplitude=SOURCE_PEAK_AMPLITUDE,
        edge_ramp_seconds=SOURCE_EDGE_RAMP_SECONDS,
        seed=SOURCE_SEED,
    )
    captures: dict[str, np.ndarray] = {}
    for sample_rate_hz in sorted({plan.sample_rate_hz for plan in node_plans}):
        rate_scene = dataclasses.replace(scene, sample_rate_hz=sample_rate_hz)
        source = resample_poly(
            source_at_synthesis_rate, RESAMPLE_UP_BY_RATE[sample_rate_hz], RESAMPLE_DOWN_BY_RATE[sample_rate_hz]
        )
        rendered = rate_scene.render_node_captures(
            source,
            emission_offset_seconds=PRE_ROLL_SECONDS,
            capture_seconds=CAPTURE_SECONDS,
            sensor_noise_rms=SENSOR_NOISE_RMS,
            noise_seed=SENSOR_NOISE_SEED_BY_RATE[sample_rate_hz],
        )
        for plan in node_plans:
            if plan.sample_rate_hz == sample_rate_hz:
                captures[plan.node_id] = rendered[plan.node_id]
    return captures


def node_frame_timing_table(
    plan: ReplayNodePlan, capture_samples: int, *, step: PacketTimestampStep | PacketClockDrift | None
) -> NodeFrameTimingTable:
    """GPS-locked 20 ms frames on the capture timeline, with an optional
    packet-clock step or drift."""
    frame_samples = frame_samples_for(plan.sample_rate_hz)
    frame_count = -(-capture_samples // frame_samples)
    offsets_ns_by_frame_index = (
        {index: step.offset_ns(index) for index in range(frame_count) if step.offset_ns(index) != 0}
        if step is not None and step.node_id == plan.node_id
        else {}
    )
    frames = frames_from_continuous_capture(
        np.zeros((1, capture_samples), dtype=np.float32),
        ContinuousCaptureFraming(
            sample_rate_hz=plan.sample_rate_hz,
            first_frame_start_time_ns=BUNDLE_CAPTURE_START_NS,
            frame_samples=frame_samples,
            timestamp_offset_ns_by_frame_index=offsets_ns_by_frame_index,
        ),
    )
    return NodeFrameTimingTable(
        node_id=plan.node_id,
        sample_rate_hz=plan.sample_rate_hz,
        frames=tuple(
            CapturedFrameTiming(
                sample_offset=frame.start_sample_index,
                sample_count=frame.samples_per_channel,
                start_time_ns=frame.start_time_ns,
                time_quality=TimeQuality.GPS_LOCKED.value,
                sequence=frame.sequence,
            )
            for frame in frames
        ),
    )


def write_replay_bundle(
    tmp_path: Path,
    *,
    scene: AcousticScene,
    node_plans: Sequence[ReplayNodePlan],
    captures: Mapping[str, np.ndarray],
    step: PacketTimestampStep | PacketClockDrift | None = None,
) -> Path:
    """A calibration bundle zip with per-node WAVs and frame timing tables."""
    work_dir = tmp_path / "bundle_work"
    coordinate_frame = scene.coordinate_frame()
    node_entries = []
    for plan in node_plans:
        node = scene.node(plan.node_id)
        channels_first = captures[plan.node_id]
        write_multichannel_wav(work_dir / "audio" / f"{plan.node_id}.wav", channels_first, plan.sample_rate_hz)
        table = node_frame_timing_table(plan, int(channels_first.shape[1]), step=step)
        timing_path = work_dir / frame_timing_bundle_path(plan.node_id)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(table.to_json_payload()))
        node_entries.append(
            {
                "node_id": plan.node_id,
                "node_type": plan.node_type.value,
                "audio_file": f"audio/{plan.node_id}.wav",
                "frame_timing_file": frame_timing_bundle_path(plan.node_id),
                "sample_rate_hz": plan.sample_rate_hz,
                "audio_start_time_ns": BUNDLE_CAPTURE_START_NS,
                "channel_sensor_ids": list(node.sensor_ids),
                "position_geo": node.position_geo.model_dump(mode="json"),
                "position_m": list(scene.node_position_enu_m(plan.node_id)),
                "sensor_offsets_m": [list(offset) for offset in plan.sensor_offsets_m],
                "orientation": {},
            }
        )
    capture_end_ns = BUNDLE_CAPTURE_START_NS + int(CAPTURE_SECONDS * 1e9)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "session_id": SESSION_ID,
        "time_window": {"start_ns": BUNDLE_CAPTURE_START_NS, "end_ns": capture_end_ns},
        "site": {"origin": SITE_ORIGIN.model_dump(mode="json"), "coordinate_mode": COORDINATE_MODE},
        "environment": {
            "temperature_c": SCENE_TEMPERATURE_C,
            "humidity_fraction": SCENE_HUMIDITY_FRACTION,
            "speed_of_sound_mps": scene.sound_speed_mps,
            "source": "fallback",
        },
        "nodes": node_entries,
        "reference_audio": None,
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    source_geo = coordinate_frame.local_to_geo(scene.source_position_enu_m)
    emission_ns = BUNDLE_CAPTURE_START_NS + int(PRE_ROLL_SECONDS * 1e9)
    ground_truth = build_ground_truth_payload(
        [
            {
                "id": "e4-source",
                "label": "synthetic burst",
                "geometry_kind": "static",
                "lat": source_geo.lat,
                "lon": source_geo.lon,
                "alt_m": source_geo.alt_m,
                "start_ns": emission_ns,
                "end_ns": emission_ns + int(SOURCE_DURATION_SECONDS * 1e9),
            }
        ]
    )
    return write_bundle_zip(work_dir, ground_truth, tmp_path / f"{SESSION_ID}.zip")
