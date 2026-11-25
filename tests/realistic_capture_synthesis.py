"""SNR-calibrated captures for the range matrix (E5) and faint funnel (E7).

Extends ``tests/acoustic_scene_synthesis.py`` with what a realistic range
sweep needs: a per-sensor gain (spherical spreading and air absorption, set
by the caller), a source level stated as the in-band SNR at a reference
sensor over a white sensor-noise floor, and any number of emissions in one
capture. Exact fractional delays come from ``fractionally_delayed``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from tests.acoustic_scene_synthesis import AcousticScene, band_limited_noise_burst, fractionally_delayed

DECIBELS_PER_AMPLITUDE_DECADE = 20.0


@dataclass(frozen=True)
class SourceBurstSpec:
    duration_seconds: float
    band_hz: tuple[float, float]
    edge_ramp_seconds: float
    seed: int


@dataclass(frozen=True)
class Emission:
    """One burst: when it leaves the source (capture time) and how loud it is
    at the reference sensor (in-band SNR over the noise floor)."""

    emission_offset_seconds: float
    reference_in_band_snr_db: float
    burst: SourceBurstSpec


def white_noise_in_band_rms(noise_rms: float, band_hz: tuple[float, float], sample_rate_hz: int) -> float:
    """RMS of the part of full-band white noise that falls inside ``band_hz``."""
    nyquist_hz = sample_rate_hz / 2.0
    return float(noise_rms * np.sqrt((band_hz[1] - band_hz[0]) / nyquist_hz))


def unit_rms_burst(spec: SourceBurstSpec, sample_rate_hz: int) -> np.ndarray:
    burst = band_limited_noise_burst(
        sample_rate_hz=sample_rate_hz,
        duration_seconds=spec.duration_seconds,
        band_hz=spec.band_hz,
        peak_amplitude=1.0,
        edge_ramp_seconds=spec.edge_ramp_seconds,
        seed=spec.seed,
    )
    return burst / float(np.sqrt(np.mean(burst**2)))


def render_calibrated_captures(
    scene: AcousticScene,
    emissions: Sequence[Emission],
    *,
    sensor_gain_relative_to_reference: Mapping[str, float],
    capture_seconds: float,
    sensor_noise_rms: float,
    noise_seed: int,
) -> dict[str, np.ndarray]:
    """Channels-first float32 capture per node; sample 0 is the same instant at
    every node (each node's packet clock error is applied at framing, not here)."""
    capture_samples = int(round(capture_seconds * scene.sample_rate_hz))
    delays_s = scene.propagation_delays_s()
    rng = np.random.default_rng(noise_seed)
    rendered_bursts = []
    for emission in emissions:
        noise_in_band_rms = white_noise_in_band_rms(sensor_noise_rms, emission.burst.band_hz, scene.sample_rate_hz)
        reference_rms = noise_in_band_rms * 10.0 ** (emission.reference_in_band_snr_db / DECIBELS_PER_AMPLITUDE_DECADE)
        rendered_bursts.append((emission, reference_rms * unit_rms_burst(emission.burst, scene.sample_rate_hz)))
    captures: dict[str, np.ndarray] = {}
    for node in scene.nodes:
        channels = []
        for sensor_id in node.sensor_ids:
            channel = sensor_noise_rms * rng.standard_normal(capture_samples)
            for emission, burst in rendered_bursts:
                delay_samples = (emission.emission_offset_seconds + delays_s[sensor_id]) * scene.sample_rate_hz
                channel += sensor_gain_relative_to_reference[sensor_id] * fractionally_delayed(
                    burst, delay_samples, capture_samples
                )
            channels.append(channel)
        captures[node.node_id] = np.stack(channels, axis=0).astype(np.float32)
    return captures


def spreading_gains_relative_to_nearest(
    scene: AcousticScene, air_absorption_db_per_m: float
) -> dict[str, float]:
    """Spherical spreading plus broadband air absorption, 1.0 at the nearest sensor."""
    source_m = np.asarray(scene.source_position_enu_m, dtype=np.float64)
    distances_m = {
        sensor_id: float(np.linalg.norm(source_m - position_m))
        for sensor_id, position_m in scene.sensor_positions_enu_m().items()
    }
    nearest_m = min(distances_m.values())
    return {
        sensor_id: (nearest_m / distance_m)
        * 10.0 ** (-air_absorption_db_per_m * (distance_m - nearest_m) / DECIBELS_PER_AMPLITUDE_DECADE)
        for sensor_id, distance_m in distances_m.items()
    }
