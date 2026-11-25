"""In-band level measurements on a point node's aligned clip (plan item P7).

Pure numpy, run off the event loop by the caller. Levels are framed in-band
energies in dB, so a short call inside a long clip is measured at its own
level instead of being diluted by the clip's silence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minimappr.core.point_network_clip_config import PointNetworkClipConfig

# Keeps log10 finite on digital silence; far below any real sensor noise.
ENERGY_FLOOR = 1e-20


@dataclass(frozen=True, slots=True)
class FramedInBandEnergy:
    """Per-frame in-band energy of one clip; frame ``k`` starts at ``k * frame_samples``."""

    energies: np.ndarray
    frame_samples: int

    def level_db(self, *, start_sample: int, end_sample: int, quantile: float) -> float | None:
        """Quantile energy (dB) of the frames overlapping ``[start_sample, end_sample)``."""
        if end_sample <= start_sample or self.energies.size == 0:
            return None
        first_frame = max(0, start_sample // self.frame_samples)
        last_frame = min(self.energies.size, -(-end_sample // self.frame_samples))
        if last_frame <= first_frame:
            return None
        return _energy_db(float(np.quantile(self.energies[first_frame:last_frame], quantile)))

    def noise_floor_db(self, quantile: float) -> float | None:
        if self.energies.size == 0:
            return None
        return _energy_db(float(np.quantile(self.energies, quantile)))


def framed_in_band_energy(
    samples: np.ndarray, sample_rate_hz: int, config: PointNetworkClipConfig
) -> FramedInBandEnergy:
    frame_samples = max(1, int(round(config.energy_frame_seconds * sample_rate_hz)))
    frame_count = samples.size // frame_samples
    if frame_count == 0:
        return FramedInBandEnergy(energies=np.zeros(0, dtype=np.float64), frame_samples=frame_samples)
    frames = np.asarray(samples[: frame_count * frame_samples], dtype=np.float64).reshape(frame_count, frame_samples)
    spectra = np.fft.rfft(frames * np.hanning(frame_samples), axis=1)
    frequencies_hz = np.fft.rfftfreq(frame_samples, d=1.0 / sample_rate_hz)
    band_high_hz = min(config.snr_band_high_hz, sample_rate_hz / 2.0)
    in_band = (frequencies_hz >= config.snr_band_low_hz) & (frequencies_hz <= band_high_hz)
    energies = np.sum(np.abs(spectra[:, in_band]) ** 2, axis=1) / frame_samples
    return FramedInBandEnergy(energies=energies, frame_samples=frame_samples)


def _energy_db(energy: float) -> float:
    return 10.0 * float(np.log10(max(energy, ENERGY_FLOOR)))


__all__ = ["FramedInBandEnergy", "framed_in_band_energy"]
