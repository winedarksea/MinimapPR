"""Note-onset detection from a sub-band spectral-flux history (Field Mode S2).

A short onset frame, not a whole clip, is the unit of correlation: the direct
path leads reflections at a note's onset, and two birds rarely start a note at
the same instant in the same band. Everything here is relative to the window's
own noise floor (per sub-band low percentile), so a faint distant bird yields
onsets exactly like a loud near one -- there is no absolute level gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Percentile of each sub-band's energy over the window taken as its noise floor.
NOISE_FLOOR_PERCENTILE = 20.0
# Scale from MAD to a Gaussian standard deviation.
MAD_TO_STD = 1.4826
POWER_FLOOR = 1e-20
# Hops before an onset that define the "before" level for the onset's band.
ONSET_BASELINE_HOPS = 5
# Analysis frames are long enough that the narrowest sub-band spans this many
# FFT bins; single-bin bands flicker by several dB on pure noise.
MIN_BINS_PER_SUBBAND = 2
# A band's hop-to-hop rise counts towards flux only beyond this many of its own
# noise-rise standard deviations (MAD of its dB differences over the window).
FLUX_RISE_NOISE_MULTIPLIER = 2.0


@dataclass(frozen=True, slots=True)
class SubbandHistory:
    """Per-hop log energy (dB) in log-spaced sub-bands, plus per-band noise floor."""

    energy_db: np.ndarray  # (hops, bands)
    noise_floor_db: np.ndarray  # (bands,)
    band_edges_hz: np.ndarray  # (bands + 1,)
    hop_samples: int
    frame_samples: int

    @property
    def band_centers_hz(self) -> np.ndarray:
        return np.sqrt(self.band_edges_hz[:-1] * self.band_edges_hz[1:])

    def snr_db(self) -> np.ndarray:
        return self.energy_db - self.noise_floor_db[None, :]

    def hop_start_sample(self, hop_index: int) -> int:
        return hop_index * self.hop_samples


@dataclass(frozen=True, slots=True)
class DetectedOnset:
    sample_index: int  # onset position in the analysed signal
    strength: float  # flux in noise-normalised units (MADs above median)
    band_min_hz: float
    band_max_hz: float


def _band_edges_hz(band_min_hz: float, band_max_hz: float, band_count: int) -> np.ndarray:
    return np.geomspace(band_min_hz, band_max_hz, band_count + 1)


def build_subband_history(
    signal: np.ndarray,
    *,
    sample_rate_hz: int,
    hop_seconds: float,
    band_min_hz: float,
    band_max_hz: float,
    band_count: int,
) -> SubbandHistory:
    """Short-time sub-band energies (Hann frames of >= 2 hops, sized so every band
    holds MIN_BINS_PER_SUBBAND bins)."""
    samples = np.asarray(signal, dtype=np.float64)
    hop_samples = max(1, int(round(hop_seconds * sample_rate_hz)))
    nyquist_hz = 0.5 * sample_rate_hz
    upper_hz = min(band_max_hz, nyquist_hz * 0.98)
    lower_hz = min(band_min_hz, upper_hz * 0.5)
    edges = _band_edges_hz(lower_hz, upper_hz, band_count)
    narrowest_band_hz = float(np.min(np.diff(edges)))
    frame_samples = max(
        2 * hop_samples, int(np.ceil(MIN_BINS_PER_SUBBAND * sample_rate_hz / narrowest_band_hz))
    )
    hop_count = max(0, 1 + (samples.size - frame_samples) // hop_samples)
    if hop_count == 0:
        empty = np.zeros((0, band_count))
        return SubbandHistory(empty, np.zeros(band_count), edges, hop_samples, frame_samples)
    starts = np.arange(hop_count) * hop_samples
    frames = samples[starts[:, None] + np.arange(frame_samples)[None, :]] * np.hanning(frame_samples)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(frame_samples, 1.0 / sample_rate_hz)
    band_index = np.digitize(frequencies, edges) - 1
    band_power = np.zeros((hop_count, band_count))
    for band in range(band_count):
        mask = band_index == band
        if np.any(mask):
            band_power[:, band] = power[:, mask].sum(axis=1)
        else:
            # Band narrower than a bin: take the nearest bin so it is not empty.
            nearest = int(np.argmin(np.abs(frequencies - np.sqrt(edges[band] * edges[band + 1]))))
            band_power[:, band] = power[:, nearest]
    energy_db = 10.0 * np.log10(band_power + POWER_FLOOR)
    noise_floor_db = np.percentile(energy_db, NOISE_FLOOR_PERCENTILE, axis=0)
    return SubbandHistory(energy_db, noise_floor_db, edges, hop_samples, frame_samples)


def positive_spectral_flux(history: SubbandHistory, active_band_snr_db: float) -> np.ndarray:
    """Sum over active sub-bands of the positive dB rise from the previous hop.

    Bands sitting at their noise floor flicker randomly; counting only bands
    that are active in the current hop keeps flux a note-onset measure.
    """
    if history.energy_db.shape[0] < 2:
        return np.zeros(history.energy_db.shape[0])
    rise_db = np.diff(history.energy_db, axis=0, prepend=history.energy_db[:1])
    rise_noise_db = MAD_TO_STD * np.median(np.abs(rise_db - np.median(rise_db, axis=0)), axis=0)
    excess_db = np.maximum(rise_db - FLUX_RISE_NOISE_MULTIPLIER * rise_noise_db[None, :], 0.0)
    active = history.snr_db() >= active_band_snr_db
    return np.sum(np.where(active, excess_db, 0.0), axis=1)


def _onset_band_range(
    history: SubbandHistory, hop_index: int, frame_hops: int, active_band_snr_db: float
) -> tuple[float, float] | None:
    """Frequency span of the sub-bands that gained energy at this onset.

    Only bands that are active over the frame AND rose by at least
    `active_band_snr_db` over the level just before the onset count, so a new
    note starting under a still-singing bird gets its own band, not the union.
    """
    stop = min(history.energy_db.shape[0], hop_index + max(1, frame_hops))
    snr = history.snr_db()[hop_index:stop]
    # Means (not max vs min) so pure-noise bands show ~0 dB rise.
    frame_mean_db = np.mean(history.energy_db[hop_index:stop], axis=0)
    before_db = history.energy_db[max(0, hop_index - ONSET_BASELINE_HOPS) : max(1, hop_index)]
    rise_db = frame_mean_db - np.mean(before_db, axis=0)
    active = np.max(snr, axis=0) >= active_band_snr_db
    rising = np.flatnonzero(active & (rise_db >= active_band_snr_db))
    bands = rising if rising.size else np.flatnonzero(active)
    if bands.size == 0:
        return None
    return float(history.band_edges_hz[bands[0]]), float(history.band_edges_hz[bands[-1] + 1])


def detect_onsets(
    history: SubbandHistory,
    *,
    sample_rate_hz: int,
    flux_threshold_mad_multiplier: float,
    min_separation_seconds: float,
    max_onsets: int,
    active_band_snr_db: float,
    frame_seconds: float,
) -> list[DetectedOnset]:
    """Peaks of positive spectral flux above median + k * MAD, strongest first."""
    flux = positive_spectral_flux(history, active_band_snr_db)
    if flux.size < 3:
        return []
    median = float(np.median(flux))
    mad_std = MAD_TO_STD * float(np.median(np.abs(flux - median)))
    # A window that is mostly silent has MAD 0; fall back to the flux std so a
    # single note still stands out rather than every flicker counting.
    spread = mad_std if mad_std > 0.0 else float(np.std(flux))
    if spread <= 0.0:
        return []
    normalized = (flux - median) / spread
    is_peak = (normalized[1:-1] >= normalized[:-2]) & (normalized[1:-1] > normalized[2:])
    candidates = np.flatnonzero(is_peak) + 1
    candidates = candidates[normalized[candidates] >= flux_threshold_mad_multiplier]
    min_separation_hops = max(1, int(round(min_separation_seconds * sample_rate_hz / history.hop_samples)))
    frame_hops = max(1, int(round(frame_seconds * sample_rate_hz / history.hop_samples)))
    chosen: list[int] = []
    for hop in sorted(candidates, key=lambda index: -normalized[index]):
        if all(abs(hop - other) >= min_separation_hops for other in chosen):
            chosen.append(int(hop))
        if len(chosen) >= max_onsets:
            break
    onsets: list[DetectedOnset] = []
    for hop in chosen:
        band = _onset_band_range(history, hop, frame_hops, active_band_snr_db)
        if band is None:
            continue
        # The flux hop marks the frame whose energy rose; the note starts within
        # that frame's leading half, i.e. near the hop start.
        onsets.append(
            DetectedOnset(
                sample_index=history.hop_start_sample(hop),
                strength=float(normalized[hop]),
                band_min_hz=band[0],
                band_max_hz=band[1],
            )
        )
    return onsets
