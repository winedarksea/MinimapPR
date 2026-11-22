"""Delay-compensated, band-limited, soft-SNR-weighted GCC-PHAT for onset frames.

For each onset the reference sensor's short template [t0, t0 + L] is correlated
against a search window [t0 - tau_max, t0 + L + tau_max] on the other sensor, so a
50 ms note still lands inside the other sensor's window however far apart the
two are (asymmetric template-vs-search). PHAT whitening is restricted to the
onset's active band and each bin is softly weighted by the lower of the two
sensors' SNR (from each sensor's own noise floor), so noise-only bins on the
faint sensor count little and wind/insect bins outside the call count nothing.
Narrowband frames (steady whistles) correlate the envelope of the cross-
correlation instead of its fine structure, which cycle-skips under multipath.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOISE_FLOOR_PERCENTILE = 20.0
# A single bin's periodogram power is exponentially distributed, so its p-th
# percentile is -ln(1 - p) x the mean. Dividing by this turns the robust low
# percentile back into the mean noise density (else every noise bin reads ~3.5x
# "SNR" and swamps the call).
NOISE_PERCENTILE_TO_MEAN = -np.log(1.0 - NOISE_FLOOR_PERCENTILE / 100.0)
POWER_FLOOR = 1e-20
# Fraction of a segment tapered at each end before correlating (Tukey window).
EDGE_TAPER_FRACTION = 0.1


@dataclass(frozen=True, slots=True)
class SensorNoiseSpectrum:
    frequencies_hz: np.ndarray
    power_density: np.ndarray

    def at(self, frequencies_hz: np.ndarray) -> np.ndarray:
        return np.interp(frequencies_hz, self.frequencies_hz, self.power_density)


@dataclass(frozen=True, slots=True)
class PairCorrelationCurve:
    """Correlation of sensor_b against sensor_a's template; positive lag = b later."""

    sensor_a: str
    sensor_b: str
    lags_s: np.ndarray
    values: np.ndarray
    effective_bandwidth_hz: float
    used_envelope: bool
    weight_sum: float
    # Fine structure, kept even when SRP uses the envelope: once SRP has picked
    # the cycle, LM refinement votes on the fine peak nearest the seed.
    fine_values: np.ndarray | None = None
    centre_frequency_hz: float = 0.0

    def value_at(self, lag_s: np.ndarray) -> np.ndarray:
        return np.interp(lag_s, self.lags_s, self.values, left=0.0, right=0.0)

    def peak_lag_near(self, lag_s: float, half_width_s: float) -> tuple[float, float]:
        """Lag and value of the SRP curve maximum within +-half_width_s of lag_s."""
        return self._peak_near(self.values, lag_s, half_width_s)

    def fine_peak_lag_near(self, lag_s: float, half_width_s: float) -> tuple[float, float]:
        """Fine-structure peak near lag_s. For envelope curves the window is capped
        at half a carrier period so the vote cannot jump to an adjacent cycle."""
        if self.fine_values is None or not self.used_envelope:
            return self.peak_lag_near(lag_s, half_width_s)
        if self.centre_frequency_hz > 0.0:
            half_width_s = min(half_width_s, 0.5 / self.centre_frequency_hz)
        return self._peak_near(self.fine_values, lag_s, half_width_s)

    def _peak_near(self, curve: np.ndarray, lag_s: float, half_width_s: float) -> tuple[float, float]:
        mask = np.abs(self.lags_s - lag_s) <= half_width_s
        if not np.any(mask):
            return float(lag_s), 0.0
        index = int(np.flatnonzero(mask)[np.argmax(curve[mask])])
        return float(self.lags_s[index]), float(curve[index])


def _hann_density(segment: np.ndarray, n_fft: int) -> np.ndarray:
    window = np.hanning(segment.size)
    spectrum = np.fft.rfft(segment * window, n_fft)
    return (np.abs(spectrum) ** 2) / max(float(np.sum(window**2)), POWER_FLOOR)


def estimate_noise_spectrum(
    signal: np.ndarray, *, sample_rate_hz: int, segment_samples: int
) -> SensorNoiseSpectrum:
    """Low-percentile power density over non-overlapping Hann segments."""
    samples = np.asarray(signal, dtype=np.float64)
    segment_samples = max(16, min(int(segment_samples), samples.size))
    count = max(1, samples.size // segment_samples)
    densities = np.vstack(
        [
            _hann_density(samples[index * segment_samples : (index + 1) * segment_samples], segment_samples)
            for index in range(count)
        ]
    )
    return SensorNoiseSpectrum(
        frequencies_hz=np.fft.rfftfreq(segment_samples, 1.0 / sample_rate_hz),
        power_density=np.percentile(densities, NOISE_FLOOR_PERCENTILE, axis=0) / NOISE_PERCENTILE_TO_MEAN
        + POWER_FLOOR,
    )


def _tukey(length: int) -> np.ndarray:
    taper = max(1, int(length * EDGE_TAPER_FRACTION))
    window = np.ones(length)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, taper))
    window[:taper] = ramp
    window[-taper:] = ramp[::-1]
    return window


def joint_band_snr(
    *,
    frequencies_hz: np.ndarray,
    template_density: np.ndarray,
    search_density: np.ndarray,
    template_noise: SensorNoiseSpectrum,
    search_noise: SensorNoiseSpectrum,
    band_min_hz: float,
    band_max_hz: float,
) -> np.ndarray:
    """Per-bin q = min SNR of the two sensors (each over its own noise floor), 0 outside the band."""
    template_snr = np.maximum(template_density / template_noise.at(frequencies_hz) - 1.0, 0.0)
    search_snr = np.maximum(search_density / search_noise.at(frequencies_hz) - 1.0, 0.0)
    in_band = (frequencies_hz >= band_min_hz) & (frequencies_hz <= band_max_hz)
    return np.where(in_band, np.minimum(template_snr, search_snr), 0.0)


def maximum_likelihood_weights(joint_snr: np.ndarray) -> np.ndarray:
    """q^2 / (1 + 2q): the Knapp-Carter ML GCC weight for two channels at SNR q,
    applied on top of PHAT whitening. Noise-only bins fall off quadratically."""
    return joint_snr**2 / (1.0 + 2.0 * joint_snr)


def presence_weights(joint_snr: np.ndarray) -> np.ndarray:
    """q / (1 + q): how much of each bin is call rather than noise (for bandwidth)."""
    return joint_snr / (1.0 + joint_snr)


def rms_bandwidth_hz(frequencies_hz: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    mean = float(np.sum(weights * frequencies_hz) / total)
    return 2.0 * float(np.sqrt(np.sum(weights * (frequencies_hz - mean) ** 2) / total))


def correlate_template_against_search(
    *,
    sensor_a: str,
    sensor_b: str,
    template: np.ndarray,
    search: np.ndarray,
    max_lag_samples: int,
    sample_rate_hz: int,
    band_min_hz: float,
    band_max_hz: float,
    template_noise: SensorNoiseSpectrum,
    search_noise: SensorNoiseSpectrum,
    interpolation_factor: int,
    narrowband_bandwidth_hz: float,
    allow_envelope: bool = True,
) -> PairCorrelationCurve | None:
    """`search` must start `max_lag_samples` before the template (length L + 2D).

    `allow_envelope=False` keeps fine structure for compact pairs: their lag span
    is under a carrier period at bird frequencies and the envelope would be flat
    across it, while SRP over several compact pairs resolves spatial aliasing.
    """
    template = np.asarray(template, dtype=np.float64)
    search = np.asarray(search, dtype=np.float64)
    if template.size < 8 or search.size < template.size + 2 * max_lag_samples:
        return None
    n_fft = int(2 ** np.ceil(np.log2(search.size + template.size)))
    frequencies_hz = np.fft.rfftfreq(n_fft, 1.0 / sample_rate_hz)
    joint_snr = joint_band_snr(
        frequencies_hz=frequencies_hz,
        template_density=_hann_density(template, n_fft),
        search_density=_hann_density(search, n_fft),
        template_noise=template_noise,
        search_noise=search_noise,
        band_min_hz=band_min_hz,
        band_max_hz=band_max_hz,
    )
    weights = maximum_likelihood_weights(joint_snr)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return None
    template_spectrum = np.fft.rfft(template * _tukey(template.size), n_fft)
    search_spectrum = np.fft.rfft(search * _tukey(search.size), n_fft)
    cross = search_spectrum * np.conj(template_spectrum)
    whitened = weights * cross / np.maximum(np.abs(cross), POWER_FLOOR)
    effective_bandwidth_hz = rms_bandwidth_hz(frequencies_hz, presence_weights(joint_snr))
    use_envelope = allow_envelope and effective_bandwidth_hz < narrowband_bandwidth_hz
    interpolated_length = n_fft * max(1, int(interpolation_factor))
    # irfft normalisation: a perfectly coherent frame peaks at sum(weights) * 2 / n_fft.
    normalisation = interpolated_length / (2.0 * weight_sum)
    fine = np.fft.irfft(whitened, interpolated_length) * normalisation
    last_index = 2 * max_lag_samples * max(1, int(interpolation_factor))
    fine_values = fine[: last_index + 1]
    if use_envelope:
        # Analytic correlation: one-sided spectrum -> complex IFFT; |.| is the envelope.
        full = np.zeros(interpolated_length, dtype=np.complex128)
        full[: whitened.size] = whitened
        full[1 : whitened.size] *= 2.0
        values = (np.abs(np.fft.ifft(full)) * normalisation)[: last_index + 1]
    else:
        values = fine_values
    lags_s = (np.arange(values.size) / max(1, int(interpolation_factor)) - max_lag_samples) / sample_rate_hz
    in_band_weights = presence_weights(joint_snr)
    centre_frequency_hz = float(np.sum(in_band_weights * frequencies_hz) / max(float(np.sum(in_band_weights)), POWER_FLOOR))
    return PairCorrelationCurve(
        sensor_a=sensor_a,
        sensor_b=sensor_b,
        lags_s=lags_s,
        values=values,
        effective_bandwidth_hz=effective_bandwidth_hz,
        used_envelope=use_envelope,
        weight_sum=weight_sum,
        fine_values=fine_values,
        centre_frequency_hz=centre_frequency_hz,
    )
