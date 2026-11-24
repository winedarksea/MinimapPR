"""Deterministic acoustic scenes for end-to-end localization tests.

A scene is one point source at a known site-frame (ENU) position heard by
nodes at known GPS positions, each with its own body-frame sensor offsets and
orientation. Captures are rendered with exact fractional propagation delays
(frequency-domain shift of a band-limited source) plus seeded independent
sensor noise, so the expected per-sensor arrival times and pairwise lags are
known in closed form.

Conventions shared with the stored-detection contract:
- sensor id = ``"{node_id}:ch{channel_index}"``;
- sensor world position = node ENU position + R(body-to-world) @ offset, with R
  from ``minimappr.spatial_audio.geometry.orientation_rotation_matrix`` (the
  convention ``core/cross_node_beamform.py`` uses);
- the lag for an ordered pair ``(a, b)`` is ``arrival(b) - arrival(a)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from minimappr.core.geo import LocalCoordinateFrame
from minimappr.models import GeoPoint, NodeOrientation, NodeType, Vec3
from minimappr.spatial_audio.geometry import orientation_rotation_matrix

NANOSECONDS_PER_SECOND = 1_000_000_000


def sensor_id_for(node_id: str, channel_index: int) -> str:
    return f"{node_id}:ch{channel_index}"


def float32_representable_geo(position_geo: GeoPoint) -> GeoPoint:
    """The binary upload carries lat/lon/alt as float32; truth must use the same values."""
    return GeoPoint(
        lat=float(np.float32(position_geo.lat)),
        lon=float(np.float32(position_geo.lon)),
        alt_m=float(np.float32(position_geo.alt_m)),
    )


@dataclass(frozen=True, slots=True)
class SceneNode:
    node_id: str
    node_type: NodeType
    # Already float32-representable (see ``scene_node_near_enu``).
    position_geo: GeoPoint
    sensor_offsets_body_m: tuple[Vec3, ...]
    orientation: NodeOrientation | None = None

    @property
    def sensor_ids(self) -> tuple[str, ...]:
        return tuple(sensor_id_for(self.node_id, index) for index in range(len(self.sensor_offsets_body_m)))


def scene_node_near_enu(
    coordinate_frame: LocalCoordinateFrame,
    *,
    node_id: str,
    node_type: NodeType,
    approximate_position_enu_m: Vec3,
    sensor_offsets_body_m: Sequence[Vec3],
    orientation: NodeOrientation | None = None,
) -> SceneNode:
    """Place a node at the float32 GPS fix nearest to ``approximate_position_enu_m``."""
    exact_geo = coordinate_frame.local_to_geo(approximate_position_enu_m)
    return SceneNode(
        node_id=node_id,
        node_type=node_type,
        position_geo=float32_representable_geo(exact_geo),
        sensor_offsets_body_m=tuple(tuple(float(axis) for axis in offset) for offset in sensor_offsets_body_m),
        orientation=orientation,
    )


@dataclass(frozen=True, slots=True)
class AcousticScene:
    site_origin: GeoPoint
    coordinate_mode: str
    nodes: tuple[SceneNode, ...]
    source_position_enu_m: Vec3
    sample_rate_hz: int
    sound_speed_mps: float

    def coordinate_frame(self) -> LocalCoordinateFrame:
        return LocalCoordinateFrame(origin=self.site_origin, mode=self.coordinate_mode)

    def node(self, node_id: str) -> SceneNode:
        return next(node for node in self.nodes if node.node_id == node_id)

    def node_position_enu_m(self, node_id: str) -> np.ndarray:
        return np.asarray(self.coordinate_frame().geo_to_local(self.node(node_id).position_geo), dtype=np.float64)

    def sensor_positions_enu_m(self) -> dict[str, np.ndarray]:
        positions: dict[str, np.ndarray] = {}
        for node in self.nodes:
            node_position_m = self.node_position_enu_m(node.node_id)
            body_to_world = orientation_rotation_matrix(node.orientation)
            for sensor_id, offset_m in zip(node.sensor_ids, node.sensor_offsets_body_m):
                positions[sensor_id] = node_position_m + body_to_world @ np.asarray(offset_m, dtype=np.float64)
        return positions

    def propagation_delays_s(self) -> dict[str, float]:
        source_m = np.asarray(self.source_position_enu_m, dtype=np.float64)
        return {
            sensor_id: float(np.linalg.norm(source_m - position_m) / self.sound_speed_mps)
            for sensor_id, position_m in self.sensor_positions_enu_m().items()
        }

    def arrival_lag_seconds(self, sensor_a: str, sensor_b: str) -> float:
        delays_s = self.propagation_delays_s()
        return delays_s[sensor_b] - delays_s[sensor_a]

    def expected_lags_by_sorted_sensor_pair(self, sensor_ids: Sequence[str]) -> dict[tuple[str, str], float]:
        ordered = sorted(sensor_ids)
        return {
            (sensor_a, sensor_b): self.arrival_lag_seconds(sensor_a, sensor_b)
            for index, sensor_a in enumerate(ordered)
            for sensor_b in ordered[index + 1 :]
        }

    def render_node_captures(
        self,
        source_waveform: np.ndarray,
        *,
        emission_offset_seconds: float,
        capture_seconds: float,
        sensor_noise_rms: float,
        noise_seed: int,
    ) -> dict[str, np.ndarray]:
        """Channels-first float32 capture per node; sample 0 of every node is the
        same instant (capture start), and the source starts emitting
        ``emission_offset_seconds`` after it."""
        capture_samples = int(round(capture_seconds * self.sample_rate_hz))
        delays_s = self.propagation_delays_s()
        rng = np.random.default_rng(noise_seed)
        captures: dict[str, np.ndarray] = {}
        for node in self.nodes:
            channels = []
            for sensor_id in node.sensor_ids:
                delay_samples = (emission_offset_seconds + delays_s[sensor_id]) * self.sample_rate_hz
                delayed = fractionally_delayed(source_waveform, delay_samples, capture_samples)
                channels.append(delayed + sensor_noise_rms * rng.standard_normal(capture_samples))
            captures[node.node_id] = np.stack(channels, axis=0).astype(np.float32)
        return captures

    def emission_to_arrival_span_seconds(self, source_duration_seconds: float) -> tuple[float, float]:
        """(first arrival, last arrival end) relative to emission start, over all sensors."""
        delays_s = self.propagation_delays_s().values()
        return min(delays_s), max(delays_s) + source_duration_seconds


def band_limited_noise_burst(
    *,
    sample_rate_hz: int,
    duration_seconds: float,
    band_hz: tuple[float, float],
    peak_amplitude: float,
    edge_ramp_seconds: float,
    seed: int,
) -> np.ndarray:
    """Seeded white noise brick-wall filtered to ``band_hz`` with raised-cosine edges.

    Broadband content gives GCC-PHAT one sharp peak per pair; the short ramps
    keep the onset abrupt so an RMS trigger fires on the first event frame.
    """
    sample_count = int(round(duration_seconds * sample_rate_hz))
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(sample_count))
    frequencies_hz = np.fft.rfftfreq(sample_count, d=1.0 / sample_rate_hz)
    spectrum[(frequencies_hz < band_hz[0]) | (frequencies_hz > band_hz[1])] = 0.0
    burst = np.fft.irfft(spectrum, n=sample_count)
    ramp_samples = max(1, int(round(edge_ramp_seconds * sample_rate_hz)))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, ramp_samples))
    burst[:ramp_samples] *= ramp
    burst[-ramp_samples:] *= ramp[::-1]
    return (peak_amplitude * burst / np.max(np.abs(burst))).astype(np.float64)


def fractionally_delayed(signal: np.ndarray, delay_samples: float, output_samples: int) -> np.ndarray:
    """``signal`` delayed by a fractional sample count into a zero buffer of ``output_samples``.

    The integer part is a placement; the fractional part is a linear phase
    ramp, exact for a band-limited signal (unlike linear interpolation, which
    low-pass filters and biases high-frequency phase).
    """
    integer_delay = int(np.floor(delay_samples))
    fractional_delay = float(delay_samples - integer_delay)
    padded_length = signal.size + 2
    spectrum = np.fft.rfft(signal, n=padded_length)
    frequencies_cycles_per_sample = np.fft.rfftfreq(padded_length)
    shifted = np.fft.irfft(
        spectrum * np.exp(-2j * np.pi * frequencies_cycles_per_sample * fractional_delay), n=padded_length
    )
    output = np.zeros(output_samples, dtype=np.float64)
    end = min(output_samples, integer_delay + padded_length)
    if end > integer_delay >= 0:
        output[integer_delay:end] = shifted[: end - integer_delay]
    return output
