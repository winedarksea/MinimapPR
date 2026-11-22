"""Template-vs-search band-limited soft-SNR GCC-PHAT (Field Mode W2-onset-srp)."""

from __future__ import annotations

import numpy as np

from minimappr.core.onset_frame_correlation import (
    correlate_template_against_search,
    estimate_noise_spectrum,
    joint_band_snr,
    maximum_likelihood_weights,
)

SAMPLE_RATE_HZ = 16_000
FRAME = int(0.08 * SAMPLE_RATE_HZ)


def _delayed(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    n_fft = 2 * signal.size
    spectrum = np.fft.rfft(signal, n_fft)
    frequencies = np.fft.rfftfreq(n_fft)
    return np.fft.irfft(spectrum * np.exp(-2j * np.pi * frequencies * delay_samples), n_fft)[: signal.size]


def _pair(source: np.ndarray, delay_samples: float, noise_rms: float, seed: int):
    rng = np.random.default_rng(seed)
    a = source + rng.normal(0, noise_rms, source.size)
    b = _delayed(source, delay_samples) + rng.normal(0, noise_rms, source.size)
    return a, b


def _correlate(a, b, onset, max_lag, band, allow_envelope=True):
    noise_a = estimate_noise_spectrum(a, sample_rate_hz=SAMPLE_RATE_HZ, segment_samples=FRAME)
    noise_b = estimate_noise_spectrum(b, sample_rate_hz=SAMPLE_RATE_HZ, segment_samples=FRAME)
    return correlate_template_against_search(
        sensor_a="a",
        sensor_b="b",
        template=a[onset : onset + FRAME],
        search=b[onset - max_lag : onset + FRAME + max_lag],
        max_lag_samples=max_lag,
        sample_rate_hz=SAMPLE_RATE_HZ,
        band_min_hz=band[0],
        band_max_hz=band[1],
        template_noise=noise_a,
        search_noise=noise_b,
        interpolation_factor=4,
        narrowband_bandwidth_hz=300.0,
        allow_envelope=allow_envelope,
    )


def _sweep(start_s, duration_s, f0, f1, total):
    out = np.zeros(total)
    t = np.arange(int(duration_s * SAMPLE_RATE_HZ)) / SAMPLE_RATE_HZ
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t * t)
    start = int(start_s * SAMPLE_RATE_HZ)
    out[start : start + t.size] = 0.1 * np.sin(phase)
    return out


def test_cross_node_delay_far_beyond_the_frame_is_recovered() -> None:
    total = int(1.5 * SAMPLE_RATE_HZ)
    source = _sweep(0.4, 0.3, 2000.0, 4500.0, total)
    delay = 37.25 * SAMPLE_RATE_HZ / 1000.0  # 37 ms: half the 80 ms frame
    a, b = _pair(source, delay, 0.002, 4)
    onset = int(0.4 * SAMPLE_RATE_HZ)
    curve = _correlate(a, b, onset, max_lag=int(0.06 * SAMPLE_RATE_HZ), band=(1800.0, 4700.0))
    lag, _ = curve.peak_lag_near(0.0, 0.06)
    assert abs(lag - delay / SAMPLE_RATE_HZ) < 2.0 / (4 * SAMPLE_RATE_HZ)


def test_compact_pair_fine_delay_is_sub_sample_accurate() -> None:
    total = int(1.0 * SAMPLE_RATE_HZ)
    source = _sweep(0.3, 0.4, 2000.0, 2600.0, total)
    a, b = _pair(source, -1.37, 0.002, 5)
    curve = _correlate(a, b, int(0.3 * SAMPLE_RATE_HZ), max_lag=5, band=(1900.0, 2700.0), allow_envelope=False)
    lag = curve.lags_s[int(np.argmax(curve.values))]
    assert abs(lag * SAMPLE_RATE_HZ + 1.37) < 0.3
    assert not curve.used_envelope


def test_narrowband_cross_node_frame_uses_the_envelope_and_keeps_fine_structure() -> None:
    total = int(1.0 * SAMPLE_RATE_HZ)
    t = np.arange(total) / SAMPLE_RATE_HZ
    source = np.where((t > 0.3) & (t < 0.7), 0.1 * np.sin(2 * np.pi * 3000.0 * t), 0.0)
    a, b = _pair(source, 50.3, 0.002, 6)
    curve = _correlate(a, b, int(0.3 * SAMPLE_RATE_HZ) - 40, max_lag=200, band=(2900.0, 3100.0))
    assert curve.used_envelope and curve.fine_values is not None
    # Fine peak near the true lag stays within half a carrier period of it.
    lag, _ = curve.fine_peak_lag_near(50.3 / SAMPLE_RATE_HZ, 1.0)
    assert abs(lag - 50.3 / SAMPLE_RATE_HZ) <= 0.5 / 3000.0


def test_noise_only_and_out_of_band_bins_get_no_weight() -> None:
    frequencies = np.linspace(0, 8000, 9)
    noise = estimate_noise_spectrum(np.random.default_rng(7).normal(0, 1, 4096), sample_rate_hz=16000, segment_samples=256)
    flat = noise.at(frequencies)
    joint = joint_band_snr(
        frequencies_hz=frequencies,
        template_density=flat * np.where(frequencies == 3000.0, 100.0, 1.0),
        search_density=flat * np.where(frequencies == 3000.0, 100.0, 1.0),
        template_noise=noise,
        search_noise=noise,
        band_min_hz=2000.0,
        band_max_hz=5000.0,
    )
    weights = maximum_likelihood_weights(joint)
    assert weights[frequencies == 3000.0][0] > 10.0
    assert np.all(weights[frequencies != 3000.0] == 0.0)
