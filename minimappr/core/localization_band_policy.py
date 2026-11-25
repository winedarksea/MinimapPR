"""Which GCC-PHAT band, and which window taper, each microphone pair uses.

Mirrors ``minimappr-core/src/localization_band_policy.rs`` (parity values in
``tests/test_localization_band_policy.py``).

PHAT whitens every bin, so a bin with no coherent signal adds random phase;
the band ceiling is what keeps a compact array's bearing from reversing. That
ceiling is a property of the array's own spacing, so it is derived per array:
c / (2 * d_max) over the array's solving mics (owner decision 2026-11-25).
The SIRITH tetra's widest pair (52.5 mm) gives ~3266 Hz, the planar's 50 mm
diagonal ~3432 Hz. Cross-node pairs have long baselines where c / 2d means
nothing for a broadband TDOA, so they get a wide band plus a Tukey taper that
removes the zero-lag peak a rectangular cut through a loud call leaves on
every channel (E8 bake-off). Intra-array pairs are not tapered: on tonal calls
the taper measurably worsened tetra and planar bearings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

# Below this widest-pair spacing an array cannot alias: its ceiling is Nyquist.
COINCIDENT_ARRAY_SPACING_M = 1.0e-6


@dataclass(frozen=True, slots=True)
class LocalizationBandPolicy:
    """``0`` ceilings mean: intra-array -> derive from geometry; cross-node -> Nyquist."""

    band_min_hz: float = 50.0
    intra_array_band_max_hz: float = 0.0
    cross_node_band_max_hz: float = 11_000.0
    cross_node_window_taper_fraction: float = 0.2


@dataclass(frozen=True, slots=True)
class PairFrontEnd:
    band_hz: tuple[float, float]
    taper_fraction: float | None


def array_alias_ceiling_hz(mic_positions_m: Iterable[np.ndarray], sound_speed_mps: float) -> float:
    """c / (2 * widest pair) of one rigid array; 0 (Nyquist) when it cannot alias."""
    positions = [np.asarray(position, dtype=np.float64) for position in mic_positions_m]
    widest_pair_m = 0.0
    for index, first in enumerate(positions):
        for second in positions[index + 1 :]:
            widest_pair_m = max(widest_pair_m, float(np.linalg.norm(first - second)))
    if widest_pair_m < COINCIDENT_ARRAY_SPACING_M:
        return 0.0
    return float(sound_speed_mps) / (2.0 * widest_pair_m)


def normalize_band_hz(band_hz: tuple[float, float] | None, sample_rate_hz: int) -> tuple[float, float] | None:
    """Nyquist-clamped band; a non-positive ceiling means Nyquist. ``None`` = every bin.

    Mirrors the Rust ``gcc_phat::normalize_band_hz``, including its inverted
    band -> no mask behaviour.
    """
    if band_hz is None:
        return None
    low_hz, high_hz = band_hz
    nyquist_hz = 0.5 * float(sample_rate_hz)
    clamped_low_hz = min(max(float(low_hz), 0.0), nyquist_hz)
    clamped_high_hz = nyquist_hz if high_hz <= 0.0 else min(float(high_hz), nyquist_hz)
    if clamped_high_hz <= clamped_low_hz:
        return None
    return clamped_low_hz, clamped_high_hz


class SensorPairFrontEnds:
    """Per-array ceilings of one solve, looked up per sensor pair.

    ``sensor_node_ids`` says which rigid array (node) each sensor is on;
    ``None`` puts every sensor on one array (a single-node solve). Only the
    given (solving) sensors set an array's geometry.
    """

    def __init__(
        self,
        policy: LocalizationBandPolicy,
        sensor_positions: Mapping[str, np.ndarray],
        sensor_node_ids: Mapping[str, str] | None,
        sound_speed_mps: float,
    ) -> None:
        self._policy = policy
        self._array_by_sensor = {
            sensor_id: (sensor_node_ids.get(sensor_id, sensor_id) if sensor_node_ids is not None else "")
            for sensor_id in sensor_positions
        }
        self._intra_array_ceiling_hz: dict[str, float] = {}
        for array_id in set(self._array_by_sensor.values()):
            if policy.intra_array_band_max_hz > 0.0:
                self._intra_array_ceiling_hz[array_id] = float(policy.intra_array_band_max_hz)
                continue
            members = [sensor_positions[sensor] for sensor, array in self._array_by_sensor.items() if array == array_id]
            self._intra_array_ceiling_hz[array_id] = array_alias_ceiling_hz(members, sound_speed_mps)

    def is_cross_node_pair(self, sensor_a: str, sensor_b: str) -> bool:
        return self._array_by_sensor.get(sensor_a, sensor_a) != self._array_by_sensor.get(sensor_b, sensor_b)

    def pair_front_end(self, sensor_a: str, sensor_b: str) -> PairFrontEnd:
        if self.is_cross_node_pair(sensor_a, sensor_b):
            taper = float(self._policy.cross_node_window_taper_fraction)
            return PairFrontEnd(
                band_hz=(float(self._policy.band_min_hz), float(self._policy.cross_node_band_max_hz)),
                taper_fraction=taper if taper > 0.0 else None,
            )
        ceiling_hz = self._intra_array_ceiling_hz.get(self._array_by_sensor.get(sensor_a, sensor_a), 0.0)
        return PairFrontEnd(band_hz=(float(self._policy.band_min_hz), ceiling_hz), taper_fraction=None)


def solve_pair_front_ends(
    policy: LocalizationBandPolicy | None,
    sensor_positions: Mapping[str, np.ndarray],
    sensor_ids: Iterable[str],
    sensor_node_ids: Mapping[str, str] | None,
    sound_speed_mps: float,
) -> SensorPairFrontEnds | None:
    """Front ends of one solve over ``sensor_ids``; ``None`` without a policy (every bin, no taper)."""
    if policy is None:
        return None
    positions = {sensor_id: sensor_positions[sensor_id] for sensor_id in sensor_ids}
    return SensorPairFrontEnds(policy, positions, sensor_node_ids, sound_speed_mps)


def tukey_taper(window: np.ndarray, taper_fraction: float | None) -> np.ndarray:
    """Copy of ``window`` with a raised-cosine ramp over ``taper_fraction`` of it, half at each end."""
    samples = np.asarray(window, dtype=np.float64)
    if not taper_fraction or taper_fraction <= 0.0:
        return samples
    length = samples.size
    ramp_samples = min(int(0.5 * min(float(taper_fraction), 1.0) * length), length // 2)
    if ramp_samples == 0:
        return samples
    tapered = samples.copy()
    gains = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples))
    tapered[:ramp_samples] *= gains
    tapered[length - ramp_samples :] *= gains[::-1]
    return tapered


def band_policy_from_config(config: object) -> LocalizationBandPolicy:
    """The policy a ``LocalizationConfig`` / ``Settings`` carries (shipped defaults for missing keys)."""
    defaults = LocalizationBandPolicy()
    return LocalizationBandPolicy(
        band_min_hz=float(getattr(config, "localization_band_min_hz", defaults.band_min_hz)),
        intra_array_band_max_hz=float(getattr(config, "localization_band_max_hz", defaults.intra_array_band_max_hz)),
        cross_node_band_max_hz=float(
            getattr(config, "localization_cross_node_band_max_hz", defaults.cross_node_band_max_hz)
        ),
        cross_node_window_taper_fraction=float(
            getattr(config, "localization_cross_node_window_taper_fraction", defaults.cross_node_window_taper_fraction)
        ),
    )
