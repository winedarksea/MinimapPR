"""Deterministic input for the E10 hot-path release gate.

One seeded scene: tetrahedral nodes in one explicit cluster hear periodic
chirps from a fixed source over a low noise floor. Nodes join and leave on a
fixed schedule, so cluster membership changes while solves are in flight.
Every post is pre-rendered before the run starts, so the only thing that varies
between a baseline and a candidate checkout is the server under test.

Wire format is the store-forward JSON envelope (``/api/v1/ingest/store-forward``)
with ``cluster_sensor_positions``: it carries explicit geometry, which keeps the
input identical across commits that change how GPS geometry is resolved.
This module deliberately imports nothing from ``minimappr`` so the same file
can drive an older checkout's server.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field

import numpy as np

SPEED_OF_SOUND_MPS = 343.2
NANOSECONDS_PER_SECOND = 1_000_000_000
PCM16_FULL_SCALE = 32767.0
# Mirrored from firmware/nodes/sirith_tetrahedral/include/node_config.h (see tests/helpers.py).
SIRITH_TETRA_SENSOR_OFFSETS_M: tuple[tuple[float, float, float], ...] = (
    (-0.016238, 0.025000, -0.010205),
    (0.027063, 0.000000, -0.010205),
    (-0.016238, -0.025000, -0.010205),
    (0.005413, 0.000000, 0.030615),
)


@dataclass(frozen=True)
class NodeMembershipWindow:
    node_id: str
    position_m: tuple[float, float, float]
    join_seconds: float
    leave_seconds: float


@dataclass(frozen=True)
class HotPathScenario:
    name: str
    duration_seconds: float
    nodes: tuple[NodeMembershipWindow, ...]
    sample_rate_hz: int = 16_000
    frame_samples: int = 512
    frames_per_post: int = 4
    cluster_id: str = "hot-path-gate"
    source_position_m: tuple[float, float, float] = (1.5, 7.0, 1.2)
    event_interval_seconds: float = 1.5
    event_duration_seconds: float = 0.35
    event_amplitude: float = 0.3
    noise_rms: float = 0.004
    signal_seed: int = 20261124
    # Each node's uploads are offset by this much per node index, so posts from
    # different nodes interleave the way independent radios do.
    per_node_post_skew_seconds: float = 0.007
    # Slower than the ~1 render/s the gate's short classification window
    # produces, so the classifier consumer falls behind for the whole run.
    slow_classifier_delay_seconds: float = 2.5
    settle_seconds: float = 6.0

    def to_json_dict(self) -> dict:
        # Round-trip through JSON so tuples become lists: a freshly built dict
        # must compare equal to one loaded from a result file.
        return json.loads(json.dumps(asdict(self)))


@dataclass(frozen=True)
class ScheduledPost:
    send_offset_seconds: float
    node_id: str
    first_sequence: int
    last_sequence: int
    first_sample_index: int
    end_sample_index: int
    body: bytes = field(repr=False)


SMOKE_SCENARIO = HotPathScenario(
    name="smoke",
    duration_seconds=8.0,
    nodes=(
        NodeMembershipWindow("gate-node-a", (0.0, 0.0, 0.0), 0.0, 8.0),
        NodeMembershipWindow("gate-node-b", (3.0, 0.0, 0.0), 0.0, 5.0),
        NodeMembershipWindow("gate-node-c", (0.0, 3.0, 0.0), 2.0, 8.0),
    ),
    settle_seconds=4.0,
)

FULL_SCENARIO = HotPathScenario(
    name="full",
    duration_seconds=40.0,
    nodes=(
        NodeMembershipWindow("gate-node-a", (0.0, 0.0, 0.0), 0.0, 40.0),
        NodeMembershipWindow("gate-node-b", (3.0, 0.0, 0.0), 0.0, 25.0),
        NodeMembershipWindow("gate-node-c", (0.0, 3.0, 0.0), 10.0, 40.0),
        NodeMembershipWindow("gate-node-d", (3.0, 3.0, 0.0), 18.0, 32.0),
    ),
)

SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in (SMOKE_SCENARIO, FULL_SCENARIO)}


def cluster_sensor_positions(scenario: HotPathScenario) -> dict[str, list[float]]:
    return {
        f"{node.node_id}:ch{channel_index}": [float(node.position_m[axis] + offset[axis]) for axis in range(3)]
        for node in scenario.nodes
        for channel_index, offset in enumerate(SIRITH_TETRA_SENSOR_OFFSETS_M)
    }


def _chirp_burst(scenario: HotPathScenario) -> np.ndarray:
    burst_samples = int(round(scenario.event_duration_seconds * scenario.sample_rate_hz))
    times_s = np.arange(burst_samples) / scenario.sample_rate_hz
    # 700 Hz -> ~1.3 kHz sweep plus a 1.7 kHz tone: inside the default localization band.
    chirp = np.sin(2.0 * np.pi * (700.0 * times_s + 850.0 * np.square(times_s)))
    tone = 0.35 * np.sin(2.0 * np.pi * 1700.0 * times_s)
    return (scenario.event_amplitude * (0.65 * chirp + tone) * np.hanning(burst_samples)).astype(np.float64)


def call_emission_offsets_seconds(scenario: HotPathScenario) -> np.ndarray:
    """When the source emits each known call, relative to the scenario's first sample."""
    return np.arange(scenario.event_interval_seconds / 2.0, scenario.duration_seconds, scenario.event_interval_seconds)


def call_arrival_windows_seconds(scenario: HotPathScenario) -> list[tuple[float, float]]:
    """Per known call: (first arrival at any sensor, last sample at the farthest sensor), scenario-relative.

    Every call is heard by at least one member node (membership never empties),
    so each window is a call the server could have detected.
    """
    source = np.asarray(scenario.source_position_m, dtype=np.float64)
    propagation_delays_s = [
        float(np.linalg.norm(source - (np.asarray(node.position_m) + np.asarray(offset)))) / SPEED_OF_SOUND_MPS
        for node in scenario.nodes
        for offset in SIRITH_TETRA_SENSOR_OFFSETS_M
    ]
    earliest_delay_s, latest_delay_s = min(propagation_delays_s), max(propagation_delays_s)
    return [
        (float(emission_s) + earliest_delay_s, float(emission_s) + latest_delay_s + scenario.event_duration_seconds)
        for emission_s in call_emission_offsets_seconds(scenario)
    ]


def render_node_channels(scenario: HotPathScenario) -> dict[str, np.ndarray]:
    """Channels-first float32 audio for each node over the whole scenario (silence outside membership)."""
    total_samples = int(round(scenario.duration_seconds * scenario.sample_rate_hz))
    burst = _chirp_burst(scenario)
    event_starts_s = call_emission_offsets_seconds(scenario)
    source = np.asarray(scenario.source_position_m, dtype=np.float64)
    rng = np.random.default_rng(scenario.signal_seed)
    channels_by_node: dict[str, np.ndarray] = {}
    for node in scenario.nodes:
        channels = rng.normal(0.0, scenario.noise_rms, size=(len(SIRITH_TETRA_SENSOR_OFFSETS_M), total_samples))
        for channel_index, offset in enumerate(SIRITH_TETRA_SENSOR_OFFSETS_M):
            sensor_position = np.asarray(node.position_m, dtype=np.float64) + np.asarray(offset, dtype=np.float64)
            delay_s = float(np.linalg.norm(source - sensor_position)) / SPEED_OF_SOUND_MPS
            for event_start_s in event_starts_s:
                # Fractional delay via linear interpolation keeps sub-sample TDOAs.
                start_exact = (event_start_s + delay_s) * scenario.sample_rate_hz
                start_index = int(np.floor(start_exact))
                fraction = start_exact - start_index
                shifted = np.interp(np.arange(burst.size) - fraction, np.arange(burst.size), burst, left=0.0, right=0.0)
                stop_index = min(total_samples, start_index + burst.size)
                if start_index < total_samples:
                    channels[channel_index, start_index:stop_index] += shifted[: stop_index - start_index]
        channels_by_node[node.node_id] = channels.astype(np.float32)
    return channels_by_node


def _encode_pcm16le_b64(channels_first: np.ndarray) -> str:
    interleaved = np.clip(channels_first.T, -1.0, 1.0)
    return base64.b64encode((interleaved * PCM16_FULL_SCALE).astype("<i2").tobytes()).decode("ascii")


def _store_forward_body(
    scenario: HotPathScenario,
    node: NodeMembershipWindow,
    frames: list[tuple[int, int, np.ndarray]],
    start_time_ns: int,
    sensor_positions: dict[str, list[float]],
) -> bytes:
    buffered_frames = []
    for sequence, start_sample_index, frame_channels in frames:
        frame_start_ns = start_time_ns + start_sample_index * NANOSECONDS_PER_SECOND // scenario.sample_rate_hz
        frame_end_ns = frame_start_ns + scenario.frame_samples * NANOSECONDS_PER_SECOND // scenario.sample_rate_hz
        buffered_frames.append(
            {
                "frame": {
                    "start_time_ns": frame_start_ns,
                    "utc_start_ns": frame_start_ns,
                    "utc_end_ns": frame_end_ns,
                    "start_sample_index": start_sample_index,
                    "end_sample_index": start_sample_index + scenario.frame_samples,
                    "sample_rate_hz": scenario.sample_rate_hz,
                    "channels": int(frame_channels.shape[0]),
                    "encoding": "pcm16le",
                    "samples_per_channel": scenario.frame_samples,
                    "samples_b64": _encode_pcm16le_b64(frame_channels),
                    "sequence": sequence,
                    "time_quality": "gps_locked",
                    "toa_ns": frame_start_ns,
                    "tor_ns": frame_start_ns + NANOSECONDS_PER_SECOND // 1000,
                    "source_type": "raw_sensor",
                }
            }
        )
    payload = {
        "node": {
            "id": node.node_id,
            "node_type": "sirith_tetra",
            "position_m": list(node.position_m),
            "sensor_offsets_m": [list(offset) for offset in SIRITH_TETRA_SENSOR_OFFSETS_M],
            "capabilities": ["audio", "array_localization"],
            "metadata": {},
            "properties": {"cluster_sensor_positions": sensor_positions},
            "cluster_id": scenario.cluster_id,
        },
        "sort_by_toa": True,
        "buffered_frames": buffered_frames,
    }
    return json.dumps(payload).encode("utf-8")


def build_post_schedule(scenario: HotPathScenario, start_time_ns: int) -> list[ScheduledPost]:
    """Every upload of the run in send order. A post is sent once its last sample exists."""
    channels_by_node = render_node_channels(scenario)
    sensor_positions = cluster_sensor_positions(scenario)
    schedule: list[ScheduledPost] = []
    for node_index, node in enumerate(scenario.nodes):
        channels = channels_by_node[node.node_id]
        first_frame = int(np.ceil(node.join_seconds * scenario.sample_rate_hz / scenario.frame_samples))
        end_frame = int(np.floor(node.leave_seconds * scenario.sample_rate_hz / scenario.frame_samples))
        sequence = 1
        for post_first_frame in range(first_frame, end_frame, scenario.frames_per_post):
            frames = []
            for frame_index in range(post_first_frame, min(end_frame, post_first_frame + scenario.frames_per_post)):
                start_sample = frame_index * scenario.frame_samples
                frames.append((sequence, start_sample, channels[:, start_sample : start_sample + scenario.frame_samples]))
                sequence += 1
            end_sample = frames[-1][1] + scenario.frame_samples
            schedule.append(
                ScheduledPost(
                    send_offset_seconds=end_sample / scenario.sample_rate_hz
                    + node_index * scenario.per_node_post_skew_seconds,
                    node_id=node.node_id,
                    first_sequence=frames[0][0],
                    last_sequence=frames[-1][0],
                    first_sample_index=frames[0][1],
                    end_sample_index=end_sample,
                    body=_store_forward_body(scenario, node, frames, start_time_ns, sensor_positions),
                )
            )
    schedule.sort(key=lambda post: (post.send_offset_seconds, post.node_id))
    return schedule
