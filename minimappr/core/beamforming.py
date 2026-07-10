"""Beamforming primitives for localization and classification pipelines.

Provides delay-and-sum and MVDR beamformers in both time-domain and
frequency-domain variants.  The frequency-domain implementations are
preferred for the classification pipeline because they apply exact
fractional delays (no interpolation artifacts) and run faster via
vectorised numpy operations.

Recall-biased beamforming: for classifier use, a wider main lobe
captures more of the target energy at the cost of slightly more noise.
MVDR diagonal loading directly controls this trade-off — higher loading
widens the beam toward delay-and-sum behaviour.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from minimappr.spatial_audio.geometry import alias_cutoff_from_positions

logger = logging.getLogger(__name__)

EPSILON = 1e-12


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ordered_sensor_ids(
    sensor_positions: dict[str, np.ndarray],
    sensor_windows: dict[str, np.ndarray],
) -> list[str]:
    return sorted(sensor_id for sensor_id in sensor_positions if sensor_id in sensor_windows)


def _fractional_shift(samples: np.ndarray, sample_rate_hz: int, shift_s: float) -> np.ndarray:
    """Time-domain fractional sample shift via linear interpolation."""
    if samples.size <= 1:
        return samples.astype(np.float32, copy=True)
    t = np.arange(samples.size, dtype=np.float64) / float(sample_rate_hz)
    shifted_t = t - shift_s
    shifted = np.interp(shifted_t, t, samples, left=0.0, right=0.0)
    return shifted.astype(np.float32)


def _steering_delays_s(
    *,
    sensor_positions: dict[str, np.ndarray],
    sensor_ids: list[str],
    steer_position_m: tuple[float, float, float],
    sound_speed_mps: float,
) -> np.ndarray:
    """Relative propagation delays from *steer_position_m* to each sensor.

    Returns an array of shape ``(M,)`` with the minimum delay subtracted
    so the closest sensor has delay 0.
    """
    target = np.asarray(steer_position_m, dtype=np.float64)
    distances = np.asarray(
        [np.linalg.norm(sensor_positions[sensor_id] - target) for sensor_id in sensor_ids],
        dtype=np.float64,
    )
    delays = distances / max(sound_speed_mps, 1.0)
    delays -= np.min(delays)
    return delays


# ---------------------------------------------------------------------------
# Time-domain delay-and-sum (original, kept for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DelayAndSumBeamformer:
    """Classic time-domain delay-and-sum via linear-interpolation alignment."""

    def beamform(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        steer_position_m: tuple[float, float, float],
        sound_speed_mps: float = 343.2,
        frequency_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        del frequency_weights
        sensor_ids = _ordered_sensor_ids(sensor_positions, sensor_windows)
        if not sensor_ids:
            return np.zeros(0, dtype=np.float32)
        if len(sensor_ids) == 1:
            return sensor_windows[sensor_ids[0]].astype(np.float32, copy=True)

        delays = _steering_delays_s(
            sensor_positions=sensor_positions,
            sensor_ids=sensor_ids,
            steer_position_m=steer_position_m,
            sound_speed_mps=sound_speed_mps,
        )
        aligned = []
        for sensor_id, delay_s in zip(sensor_ids, delays, strict=True):
            aligned.append(_fractional_shift(sensor_windows[sensor_id], sample_rate_hz, shift_s=-float(delay_s)))
        stacked = np.vstack(aligned)
        weights = np.ones(stacked.shape[0], dtype=np.float64)
        weights /= np.sum(weights)
        output = np.sum(stacked * weights[:, None], axis=0)
        return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Frequency-domain delay-and-sum (fast, exact fractional delays)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FrequencyDomainDelayAndSumBeamformer:
    """FFT-based delay-and-sum with exact fractional delays via phase shifts.

    Preferred over :class:`DelayAndSumBeamformer` for classification because
    it avoids interpolation artifacts and naturally supports frequency
    weighting.  Inherently recall-biased — it sums all sensors equally,
    producing a wide main lobe.
    """

    def beamform(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        steer_position_m: tuple[float, float, float],
        sound_speed_mps: float = 343.2,
        frequency_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        sensor_ids = _ordered_sensor_ids(sensor_positions, sensor_windows)
        if not sensor_ids:
            return np.zeros(0, dtype=np.float32)
        if len(sensor_ids) == 1:
            return sensor_windows[sensor_ids[0]].astype(np.float32, copy=True)

        delays = _steering_delays_s(
            sensor_positions=sensor_positions,
            sensor_ids=sensor_ids,
            steer_position_m=steer_position_m,
            sound_speed_mps=sound_speed_mps,
        )

        x = np.vstack([sensor_windows[sid] for sid in sensor_ids]).astype(np.float64)
        n_samples = x.shape[1]
        if n_samples == 0:
            return np.zeros(0, dtype=np.float32)

        spectra = np.fft.rfft(x, axis=1)  # (M, F)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)  # (F,)

        # Phase-shift each sensor to compensate for propagation delay.
        # Advancing by τ in time ↔ multiply spectrum by exp(+j2πfτ).
        phase_shifts = np.exp(1j * 2.0 * np.pi * delays[:, None] * freqs[None, :])  # (M, F)
        aligned_spectra = spectra * phase_shifts
        summed_spectrum = np.mean(aligned_spectra, axis=0)

        if frequency_weights is not None and frequency_weights.size == freqs.size:
            summed_spectrum *= np.asarray(frequency_weights, dtype=np.float64)

        output = np.fft.irfft(summed_spectrum, n=n_samples)
        return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Band-split delay-and-sum renderer (see BEAMFORMED_RENDER_CONTRACT.md)
# ---------------------------------------------------------------------------

def raised_cosine_band_weights(
    freqs: np.ndarray,
    *,
    low_center_hz: float,
    low_width_hz: float,
    high_center_hz: float,
    high_width_hz: float,
) -> np.ndarray:
    """Steered-band weight w(f) in [0, 1] with raised-cosine crossovers.

    w ramps 0→1 across ``low_center_hz ± low_width_hz/2`` and 1→0 across
    ``high_center_hz ± high_width_hz/2``. The high ramp is centered at the
    alias cutoff per the contract (recall-biased). Shared by classification,
    IAMF, and the cross-language parity expectations.
    """
    f = np.asarray(freqs, dtype=np.float64)
    weights = np.ones_like(f)

    lo_start = low_center_hz - low_width_hz / 2.0
    lo_end = low_center_hz + low_width_hz / 2.0
    if lo_end > lo_start:
        x = np.clip((f - lo_start) / (lo_end - lo_start), 0.0, 1.0)
        weights *= 0.5 * (1.0 - np.cos(np.pi * x))
    else:
        weights *= (f >= low_center_hz).astype(np.float64)

    hi_start = high_center_hz - high_width_hz / 2.0
    hi_end = high_center_hz + high_width_hz / 2.0
    if hi_end > hi_start:
        x = np.clip((f - hi_start) / (hi_end - hi_start), 0.0, 1.0)
        weights *= 0.5 * (1.0 + np.cos(np.pi * x))
    else:
        weights *= (f <= high_center_hz).astype(np.float64)

    return weights


@dataclass(slots=True)
class BandSplitRenderConfig:
    """Crossover parameters for :class:`BandSplitDasRenderer`."""

    highpass_hz: float = 100.0
    low_crossover_width_hz: float = 100.0
    high_crossover_width_min_hz: float = 400.0
    high_crossover_width_fraction: float = 0.15
    sound_speed_mps: float = 343.2


@dataclass(slots=True)
class BandSplitDasRenderer:
    """Band-split delay-and-sum: steered below the alias cutoff, omni elsewhere.

    Implements the normative algorithm of BEAMFORMED_RENDER_CONTRACT.md: one
    rFFT per channel; omni = mean spectrum; steered = mean of phase-advanced
    spectra; raised-cosine blend with the high ramp centered at the
    geometry-derived alias cutoff. Recall-biased: everything outside the
    physically-steerable band passes through as omni (including below the
    highpass — low rumble is kept for the classifier).

    ``alias_cutoff_hz = None`` computes the cutoff from the sensor positions
    at render time (c / (2·max_baseline)).
    """

    config: BandSplitRenderConfig = None  # type: ignore[assignment]
    alias_cutoff_hz: float | None = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = BandSplitRenderConfig()

    def beamform(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        steer_position_m: tuple[float, float, float],
        sound_speed_mps: float = 343.2,
        frequency_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        sensor_ids = _ordered_sensor_ids(sensor_positions, sensor_windows)
        if not sensor_ids:
            return np.zeros(0, dtype=np.float32)
        if len(sensor_ids) == 1:
            return sensor_windows[sensor_ids[0]].astype(np.float32, copy=True)

        x = np.vstack([sensor_windows[sid] for sid in sensor_ids]).astype(np.float64)
        n_samples = x.shape[1]
        if n_samples == 0:
            return np.zeros(0, dtype=np.float32)

        cutoff_hz = self.alias_cutoff_hz
        if cutoff_hz is None or not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
            positions = np.vstack([sensor_positions[sid] for sid in sensor_ids])
            cutoff_hz = alias_cutoff_from_positions(positions, sound_speed_mps)

        delays = _steering_delays_s(
            sensor_positions=sensor_positions,
            sensor_ids=sensor_ids,
            steer_position_m=steer_position_m,
            sound_speed_mps=sound_speed_mps,
        )

        spectra = np.fft.rfft(x, axis=1)  # (M, F)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)

        omni_spectrum = np.mean(spectra, axis=0)
        phase_shifts = np.exp(1j * 2.0 * np.pi * delays[:, None] * freqs[None, :])
        steered_spectrum = np.mean(spectra * phase_shifts, axis=0)

        weights = self.band_weights(freqs, cutoff_hz)
        blended = weights * steered_spectrum + (1.0 - weights) * omni_spectrum

        if frequency_weights is not None and frequency_weights.size == freqs.size:
            blended *= np.asarray(frequency_weights, dtype=np.float64)

        output = np.fft.irfft(blended, n=n_samples)
        return output.astype(np.float32)

    def band_weights(self, freqs: np.ndarray, alias_cutoff_hz: float) -> np.ndarray:
        """Contract crossover weights for a given cutoff."""
        cfg = self.config
        high_width = max(
            cfg.high_crossover_width_min_hz,
            cfg.high_crossover_width_fraction * alias_cutoff_hz,
        )
        return raised_cosine_band_weights(
            freqs,
            low_center_hz=cfg.highpass_hz,
            low_width_hz=cfg.low_crossover_width_hz,
            high_center_hz=alias_cutoff_hz,
            high_width_hz=high_width,
        )


# ---------------------------------------------------------------------------
# Block trajectory rendering (Hann overlap-add over a moving steer position)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BlockTrajectoryRenderer:
    """Hann overlap-add block loop driving any duck-typed beamformer along a
    per-block steer trajectory.

    Extracted from the IAMF pipeline's offline MVDR loop; this is also the
    per-track continuous-stream primitive (contract §8) — a live stream is the
    same loop driven by a track trajectory, so no separate streaming plumbing
    is built until needed.

    ``omni_blend_above_cutoff`` applies the contract's raised-cosine omni
    blend above the geometry-derived alias cutoff so rendered objects keep
    full bandwidth (MVDR/DAS output is not physically steerable up there).
    """

    beamformer: object  # duck-typed: anything with .beamform(...)
    block_size: int = 512
    fade_seconds: float = 0.012
    omni_blend_above_cutoff: bool = False
    high_crossover_width_min_hz: float = 400.0
    high_crossover_width_fraction: float = 0.15
    sound_speed_mps: float = 343.2

    def render(
        self,
        channels: np.ndarray,
        mic_positions_m: np.ndarray,
        sample_rate_hz: int,
        steer_for_sample,
        active_range: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Render ``channels`` (C, N) to a mono object signal.

        ``steer_for_sample(sample_index) -> (x, y, z)`` supplies the steer
        position for each block (evaluated at the block midpoint).
        ``active_range=(first, last)`` silences blocks outside the range and
        applies linear fades at its edges.
        """
        n_channels = int(channels.shape[0])
        n_samples = int(channels.shape[1])
        positions = np.asarray(mic_positions_m, dtype=np.float64)
        sensor_ids = [f"ch{i}" for i in range(n_channels)]
        sensor_positions = {
            sid: positions[i] for i, sid in enumerate(sensor_ids) if i < len(positions)
        }

        output = np.zeros(n_samples, dtype=np.float64)
        norm = np.zeros(n_samples, dtype=np.float64)
        first_active, last_active = active_range if active_range is not None else (0, n_samples)
        if last_active <= first_active:
            return output.astype(np.float32)

        block_size = self.block_size
        hop = max(1, block_size // 2)
        window = np.hanning(block_size).astype(np.float64)

        blend_weights: np.ndarray | None = None
        if self.omni_blend_above_cutoff and n_channels > 1:
            cutoff_hz = alias_cutoff_from_positions(
                positions[:n_channels], self.sound_speed_mps
            )
            freqs = np.fft.rfftfreq(block_size, d=1.0 / sample_rate_hz)
            high_width = max(
                self.high_crossover_width_min_hz,
                self.high_crossover_width_fraction * cutoff_hz,
            )
            # Low crossover degenerate (0 Hz / 0 width) keeps the steered
            # output untouched below the cutoff; only the high ramp blends.
            blend_weights = raised_cosine_band_weights(
                freqs,
                low_center_hz=0.0,
                low_width_hz=0.0,
                high_center_hz=cutoff_hz,
                high_width_hz=high_width,
            )

        for block_start in range(0, n_samples, hop):
            block_end = min(block_start + block_size, n_samples)
            block_len = block_end - block_start

            # Silence blocks entirely outside the active range.
            if block_end <= first_active or block_start >= last_active:
                continue

            sample_mid = block_start + block_len // 2
            steer_pos = steer_for_sample(sample_mid)

            frame = np.zeros((n_channels, block_size), dtype=np.float32)
            frame[:, :block_len] = channels[:, block_start:block_end]
            frame *= window[np.newaxis, :].astype(np.float32)
            sensor_windows = {
                sid: frame[i] for i, sid in enumerate(sensor_ids) if i < n_channels
            }
            block_out = self.beamformer.beamform(
                sensor_positions=sensor_positions,
                sensor_windows=sensor_windows,
                sample_rate_hz=sample_rate_hz,
                steer_position_m=steer_pos,
                sound_speed_mps=self.sound_speed_mps,
            )
            if block_out.size < block_size:
                block_out = np.pad(block_out, (0, block_size - block_out.size))

            if blend_weights is not None:
                omni_block = np.mean(frame.astype(np.float64), axis=0)
                steered_spec = np.fft.rfft(block_out[:block_size].astype(np.float64))
                omni_spec = np.fft.rfft(omni_block)
                blended = blend_weights * steered_spec + (1.0 - blend_weights) * omni_spec
                block_out = np.fft.irfft(blended, n=block_size)

            output[block_start:block_end] += (
                block_out[:block_len].astype(np.float64) * window[:block_len]
            )
            norm[block_start:block_end] += window[:block_len] ** 2

        safe_norm = np.where(norm > 1e-12, norm, 1.0)
        output /= safe_norm

        fade_samples = max(512, int(round(self.fade_seconds * sample_rate_hz)))
        fade_in_start = max(0, first_active)
        fade_in_end = min(n_samples, first_active + fade_samples)
        if fade_in_end > fade_in_start and active_range is not None:
            ramp = np.linspace(0.0, 1.0, fade_in_end - fade_in_start, dtype=np.float64)
            output[fade_in_start:fade_in_end] *= ramp
        fade_out_start = max(0, last_active - fade_samples)
        fade_out_end = min(n_samples, last_active)
        if fade_out_end > fade_out_start and active_range is not None:
            ramp = np.linspace(1.0, 0.0, fade_out_end - fade_out_start, dtype=np.float64)
            output[fade_out_start:fade_out_end] *= ramp

        return output.astype(np.float32)


# ---------------------------------------------------------------------------
# MVDR (Minimum Variance Distortionless Response) — vectorised
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MVDRBeamformer:
    """Vectorised narrowband MVDR beamformer.

    Computes per-frequency-bin covariance matrices and MVDR weights using
    batched :func:`numpy.linalg.solve` — typically 10–50× faster than the
    previous per-bin Python loop for typical array sizes (4–8 sensors).

    *diagonal_loading* controls the regularisation added to each covariance
    matrix.  Higher values widen the main lobe and improve robustness,
    trading spatial selectivity for recall — useful when the beamformed
    output feeds a classifier that benefits from capturing more of the
    target signal.
    """

    diagonal_loading: float = 1e-3
    freq_min_hz: float = 200.0
    freq_max_hz: float = 5000.0

    def beamform(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        steer_position_m: tuple[float, float, float],
        sound_speed_mps: float = 343.2,
        frequency_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        sensor_ids = _ordered_sensor_ids(sensor_positions, sensor_windows)
        if not sensor_ids:
            return np.zeros(0, dtype=np.float32)
        if len(sensor_ids) == 1:
            return sensor_windows[sensor_ids[0]].astype(np.float32, copy=True)

        num_sensors = len(sensor_ids)
        x = np.vstack([sensor_windows[sid] for sid in sensor_ids]).astype(np.float64)
        n_samples = x.shape[1]
        if n_samples == 0:
            return np.zeros(0, dtype=np.float32)

        delays = _steering_delays_s(
            sensor_positions=sensor_positions,
            sensor_ids=sensor_ids,
            steer_position_m=steer_position_m,
            sound_speed_mps=sound_speed_mps,
        )

        spectra = np.fft.rfft(x, axis=1)  # (M, F)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)
        y_spec = np.zeros(freqs.size, dtype=np.complex128)

        # Identify in-band frequency bins for MVDR processing.
        in_band_mask = (freqs > 0.0) & (freqs >= self.freq_min_hz) & (freqs <= self.freq_max_hz)
        passthrough_mask = ~in_band_mask
        in_band_indices = np.where(in_band_mask)[0]

        # Pass-through bins: simple mean across sensors (DC, out-of-band).
        y_spec[passthrough_mask] = np.mean(spectra[:, passthrough_mask], axis=0)

        if in_band_indices.size > 0:
            y_spec[in_band_indices] = self._mvdr_vectorised(
                spectra=spectra,
                freqs=freqs,
                in_band_indices=in_band_indices,
                delays=delays,
                num_sensors=num_sensors,
                frequency_weights=frequency_weights,
            )

        output = np.fft.irfft(y_spec, n=n_samples)
        return output.astype(np.float32)

    # -- internal ------------------------------------------------------------

    def _mvdr_vectorised(
        self,
        *,
        spectra: np.ndarray,
        freqs: np.ndarray,
        in_band_indices: np.ndarray,
        delays: np.ndarray,
        num_sensors: int,
        frequency_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Batch MVDR across all in-band frequency bins using numpy broadcast ops."""

        f_rel = freqs[in_band_indices]  # (F_rel,)
        x_rel = spectra[:, in_band_indices]  # (M, F_rel)
        n_bins = f_rel.size

        # Steering vectors: a[f, m] = exp(-j2πf·τ_m)  — standard convention.
        steering = np.exp(-1j * 2.0 * np.pi * f_rel[:, None] * delays[None, :])  # (F_rel, M)

        # Per-frequency covariance matrices: R[f] = x[:,f] x[:,f]^H + λI
        # Using einsum: R[f, i, j] = x[i, f]·conj(x[j, f])
        cov = np.einsum("if,jf->fij", x_rel, np.conj(x_rel))  # (F_rel, M, M)
        loading = max(self.diagonal_loading, 1e-9)
        cov += loading * np.eye(num_sensors, dtype=np.complex128)[None, :, :]

        # Solve R @ w_raw = a for each frequency bin (batched).
        try:
            w_raw = np.linalg.solve(cov, steering[:, :, None]).squeeze(-1)  # (F_rel, M)
        except np.linalg.LinAlgError:
            # Graceful fallback to mean — singular matrices with loading are very rare.
            logger.warning("MVDR batch solve encountered singular matrix; falling back to mean.")
            return np.mean(x_rel, axis=0)

        # Denominator: a^H R^{-1} a  per frequency.
        denom = np.sum(np.conj(steering) * w_raw, axis=1)  # (F_rel,)
        safe = np.abs(denom) >= EPSILON

        # MVDR weights: w = (R^{-1} a) / (a^H R^{-1} a).
        weights = np.zeros_like(w_raw)
        weights[safe] = w_raw[safe] / denom[safe, None]
        weights[~safe] = 1.0 / num_sensors  # uniform fallback for degenerate bins

        # Output: y[f] = w^H x = sum_m conj(w[f,m])·x[m,f]
        y_rel = np.sum(np.conj(weights) * x_rel.T, axis=1)  # (F_rel,)

        if frequency_weights is not None and frequency_weights.size == freqs.size:
            y_rel *= np.asarray(frequency_weights, dtype=np.float64)[in_band_indices]

        return y_rel


# ---------------------------------------------------------------------------
# Superdirective beamformer (analytical diffuse-noise model)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SuperdirectiveBeamformer:
    """Superdirective beamformer using an analytical diffuse-noise model.

    Instead of estimating the noise covariance from data (like MVDR),
    this uses the theoretical coherence matrix for a diffuse (isotropic)
    noise field::

        Γ[i,j](f) = sinc(2πf·d_ij / c)

    where *d_ij* is the distance between sensors *i* and *j*.  This is
    optimal when background noise is spatially uniform — a good
    approximation for wind, rain, and distant urban noise.

    **Advantages over MVDR for small arrays:**

    * More stable — no risk of poor covariance estimates from short data
    * Better low-frequency directivity for closely-spaced sensors
    * Deterministic weights — same geometry always produces same beam

    *diagonal_loading* regularises the coherence matrix (higher values
    widen the beam toward DAS, improving robustness at the cost of
    directivity).
    """

    diagonal_loading: float = 1e-2
    freq_min_hz: float = 200.0
    freq_max_hz: float = 5000.0

    def beamform(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        steer_position_m: tuple[float, float, float],
        sound_speed_mps: float = 343.2,
        frequency_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        sensor_ids = _ordered_sensor_ids(sensor_positions, sensor_windows)
        if not sensor_ids:
            return np.zeros(0, dtype=np.float32)
        if len(sensor_ids) == 1:
            return sensor_windows[sensor_ids[0]].astype(np.float32, copy=True)

        num_sensors = len(sensor_ids)
        x = np.vstack([sensor_windows[sid] for sid in sensor_ids]).astype(np.float64)
        n_samples = x.shape[1]
        if n_samples == 0:
            return np.zeros(0, dtype=np.float32)

        delays = _steering_delays_s(
            sensor_positions=sensor_positions,
            sensor_ids=sensor_ids,
            steer_position_m=steer_position_m,
            sound_speed_mps=sound_speed_mps,
        )

        spectra = np.fft.rfft(x, axis=1)  # (M, F)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)
        y_spec = np.zeros(freqs.size, dtype=np.complex128)

        in_band_mask = (freqs > 0.0) & (freqs >= self.freq_min_hz) & (freqs <= self.freq_max_hz)
        passthrough_mask = ~in_band_mask
        in_band_indices = np.where(in_band_mask)[0]

        y_spec[passthrough_mask] = np.mean(spectra[:, passthrough_mask], axis=0)

        if in_band_indices.size > 0:
            y_spec[in_band_indices] = self._superdirective_vectorised(
                spectra=spectra,
                freqs=freqs,
                in_band_indices=in_band_indices,
                delays=delays,
                sensor_positions=sensor_positions,
                sensor_ids=sensor_ids,
                num_sensors=num_sensors,
                sound_speed_mps=sound_speed_mps,
                frequency_weights=frequency_weights,
            )

        output = np.fft.irfft(y_spec, n=n_samples)
        return output.astype(np.float32)

    def _superdirective_vectorised(
        self,
        *,
        spectra: np.ndarray,
        freqs: np.ndarray,
        in_band_indices: np.ndarray,
        delays: np.ndarray,
        sensor_positions: dict[str, np.ndarray],
        sensor_ids: list[str],
        num_sensors: int,
        sound_speed_mps: float,
        frequency_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Compute superdirective weights using diffuse-noise coherence."""

        f_rel = freqs[in_band_indices]  # (F_rel,)
        x_rel = spectra[:, in_band_indices]  # (M, F_rel)

        # Inter-sensor distance matrix.
        positions = np.array([sensor_positions[sid] for sid in sensor_ids], dtype=np.float64)
        diff = positions[:, None, :] - positions[None, :, :]  # (M, M, 3)
        dist_matrix = np.linalg.norm(diff, axis=2)  # (M, M)

        # Diffuse-noise coherence: Γ[i,j](f) = sinc(2πf·d_ij / c)
        # np.sinc(x) = sin(πx)/(πx), so sinc(2πf·d/c) = np.sinc(2f·d/c)
        arg = 2.0 * f_rel[:, None, None] * dist_matrix[None, :, :] / max(sound_speed_mps, 1.0)
        gamma = np.sinc(arg).astype(np.complex128)  # (F_rel, M, M)

        # Add diagonal loading for stability.
        loading = max(self.diagonal_loading, 1e-9)
        gamma += loading * np.eye(num_sensors, dtype=np.complex128)[None, :, :]

        # Steering vectors.
        steering = np.exp(-1j * 2.0 * np.pi * f_rel[:, None] * delays[None, :])  # (F_rel, M)

        # Solve Γ @ w_raw = a for each frequency bin.
        try:
            w_raw = np.linalg.solve(gamma, steering[:, :, None]).squeeze(-1)  # (F_rel, M)
        except np.linalg.LinAlgError:
            logger.warning("Superdirective solve encountered singular matrix; falling back to mean.")
            return np.mean(x_rel, axis=0)

        denom = np.sum(np.conj(steering) * w_raw, axis=1)
        safe = np.abs(denom) >= EPSILON

        weights = np.zeros_like(w_raw)
        weights[safe] = w_raw[safe] / denom[safe, None]
        weights[~safe] = 1.0 / num_sensors

        y_rel = np.sum(np.conj(weights) * x_rel.T, axis=1)

        if frequency_weights is not None and frequency_weights.size == freqs.size:
            y_rel *= np.asarray(frequency_weights, dtype=np.float64)[in_band_indices]

        return y_rel


# ---------------------------------------------------------------------------
# GEVD / MaxSNR beamformer
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GEVDBeamformer:
    """Generalized Eigenvalue Decomposition (GEVD) / MaxSNR beamformer.

    Computes weights that maximize the signal-to-noise ratio by solving
    the generalized eigenvalue problem::

        R_signal · w = λ · R_noise · w

    The eigenvector corresponding to the *largest* eigenvalue is the
    optimal beamforming weight vector.

    **Noise covariance estimation:** When no external noise reference is
    available (the common case), the noise covariance is estimated from
    the *spatial* structure: the signal covariance is rank-1 steered toward
    the target, and the noise covariance uses the diffuse-field model
    (same sinc coherence as :class:`SuperdirectiveBeamformer`).

    This combines the adaptivity of MVDR with the robustness of the
    superdirective approach — generally the best choice when the noise
    field is approximately diffuse.

    *diagonal_loading* regularises both covariance matrices.
    """

    diagonal_loading: float = 1e-2
    freq_min_hz: float = 200.0
    freq_max_hz: float = 5000.0

    def beamform(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        steer_position_m: tuple[float, float, float],
        sound_speed_mps: float = 343.2,
        frequency_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        sensor_ids = _ordered_sensor_ids(sensor_positions, sensor_windows)
        if not sensor_ids:
            return np.zeros(0, dtype=np.float32)
        if len(sensor_ids) == 1:
            return sensor_windows[sensor_ids[0]].astype(np.float32, copy=True)

        num_sensors = len(sensor_ids)
        x = np.vstack([sensor_windows[sid] for sid in sensor_ids]).astype(np.float64)
        n_samples = x.shape[1]
        if n_samples == 0:
            return np.zeros(0, dtype=np.float32)

        delays = _steering_delays_s(
            sensor_positions=sensor_positions,
            sensor_ids=sensor_ids,
            steer_position_m=steer_position_m,
            sound_speed_mps=sound_speed_mps,
        )

        spectra = np.fft.rfft(x, axis=1)  # (M, F)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)
        y_spec = np.zeros(freqs.size, dtype=np.complex128)

        in_band_mask = (freqs > 0.0) & (freqs >= self.freq_min_hz) & (freqs <= self.freq_max_hz)
        passthrough_mask = ~in_band_mask
        in_band_indices = np.where(in_band_mask)[0]

        y_spec[passthrough_mask] = np.mean(spectra[:, passthrough_mask], axis=0)

        if in_band_indices.size > 0:
            y_spec[in_band_indices] = self._gevd_vectorised(
                spectra=spectra,
                freqs=freqs,
                in_band_indices=in_band_indices,
                delays=delays,
                sensor_positions=sensor_positions,
                sensor_ids=sensor_ids,
                num_sensors=num_sensors,
                sound_speed_mps=sound_speed_mps,
                frequency_weights=frequency_weights,
            )

        output = np.fft.irfft(y_spec, n=n_samples)
        return output.astype(np.float32)

    def _gevd_vectorised(
        self,
        *,
        spectra: np.ndarray,
        freqs: np.ndarray,
        in_band_indices: np.ndarray,
        delays: np.ndarray,
        sensor_positions: dict[str, np.ndarray],
        sensor_ids: list[str],
        num_sensors: int,
        sound_speed_mps: float,
        frequency_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Batch GEVD across in-band bins."""

        f_rel = freqs[in_band_indices]  # (F_rel,)
        x_rel = spectra[:, in_band_indices]  # (M, F_rel)
        n_bins = f_rel.size

        # Steering vectors.
        steering = np.exp(-1j * 2.0 * np.pi * f_rel[:, None] * delays[None, :])  # (F_rel, M)

        # Signal covariance: data-estimated R_xx = x·x^H.
        r_signal = np.einsum("if,jf->fij", x_rel, np.conj(x_rel))  # (F_rel, M, M)

        # Noise covariance: diffuse-field model (sinc coherence).
        positions = np.array([sensor_positions[sid] for sid in sensor_ids], dtype=np.float64)
        diff = positions[:, None, :] - positions[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        arg = 2.0 * f_rel[:, None, None] * dist_matrix[None, :, :] / max(sound_speed_mps, 1.0)
        r_noise = np.sinc(arg).astype(np.complex128)

        loading = max(self.diagonal_loading, 1e-9)
        eye = np.eye(num_sensors, dtype=np.complex128)[None, :, :]
        r_signal += loading * eye
        r_noise += loading * eye

        # Per-bin GEVD: find eigenvector with largest eigenvalue of
        # R_noise^{-1} R_signal.  We invert R_noise and multiply to get
        # a standard eigenvalue problem.
        y_rel = np.zeros(n_bins, dtype=np.complex128)
        try:
            r_noise_inv = np.linalg.inv(r_noise)  # (F_rel, M, M)
        except np.linalg.LinAlgError:
            logger.warning("GEVD noise covariance inversion failed; falling back to mean.")
            return np.mean(x_rel, axis=0)

        # Product matrix: P = R_noise^{-1} R_signal   (F_rel, M, M)
        product = np.einsum("fij,fjk->fik", r_noise_inv, r_signal)

        # Batch eigendecomposition — numpy eigh requires Hermitian input,
        # but P is not generally Hermitian.  Use np.linalg.eig.
        try:
            eigenvalues, eigenvectors = np.linalg.eig(product)  # (F_rel, M), (F_rel, M, M)
        except np.linalg.LinAlgError:
            logger.warning("GEVD eigendecomposition failed; falling back to mean.")
            return np.mean(x_rel, axis=0)

        # Select the eigenvector with the largest eigenvalue per bin.
        max_idx = np.argmax(np.abs(eigenvalues), axis=1)  # (F_rel,)
        # Extract the dominant eigenvector for each frequency bin.
        w_gevd = eigenvectors[np.arange(n_bins), :, max_idx]  # (F_rel, M)

        # Normalise weights so that w^H a = 1 (distortionless constraint).
        inner = np.sum(np.conj(w_gevd) * steering, axis=1)  # (F_rel,)
        safe = np.abs(inner) >= EPSILON
        w_gevd[safe] = w_gevd[safe] / inner[safe, None]
        w_gevd[~safe] = 1.0 / num_sensors

        y_rel = np.sum(np.conj(w_gevd) * x_rel.T, axis=1)

        if frequency_weights is not None and frequency_weights.size == freqs.size:
            y_rel *= np.asarray(frequency_weights, dtype=np.float64)[in_band_indices]

        return y_rel


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BEAMFORMER_TYPES = {
    "delay_and_sum": "Time-domain delay-and-sum (linear interpolation)",
    "das": "Time-domain delay-and-sum (linear interpolation)",
    "freq_domain_das": "FFT-based delay-and-sum (exact fractional delays)",
    "band_split_das": "Band-split delay-and-sum (steered below alias cutoff, omni elsewhere)",
    "mvdr": "Minimum Variance Distortionless Response",
    "superdirective": "Superdirective (analytical diffuse-noise model)",
    "gevd": "Generalized Eigenvalue Decomposition / MaxSNR",
}

_BEAMFORMER_TYPE_ALIASES = {
    "": "delay_and_sum",
    "das": "delay_and_sum",
    "delay_and_sum": "delay_and_sum",
    "freq_domain_das": "freq_domain_das",
    "band_split_das": "band_split_das",
    "band_split": "band_split_das",
    "mvdr": "mvdr",
    "superdirective": "superdirective",
    "gevd": "gevd",
}


def available_beamformers() -> dict[str, str]:
    """Return a mapping of beamformer type names to descriptions."""
    return dict(_BEAMFORMER_TYPES)


def create_beamformer(
    beamformer_type: str,
    *,
    diagonal_loading: float = 1e-3,
    classifier_diagonal_loading_scale: float = 1.0,
    freq_min_hz: float = 200.0,
    freq_max_hz: float = 5000.0,
) -> "DelayAndSumBeamformer | FrequencyDomainDelayAndSumBeamformer | BandSplitDasRenderer | MVDRBeamformer | SuperdirectiveBeamformer | GEVDBeamformer":
    """Instantiate a beamformer by name.

    When *classifier_diagonal_loading_scale* > 1 the effective diagonal
    loading for MVDR / superdirective / GEVD is increased, widening the
    main lobe for higher recall — preferable when the output feeds a
    classifier.

    Available types: ``delay_and_sum``/``das``, ``freq_domain_das``,
    ``mvdr``, ``superdirective``, ``gevd``.
    """
    name = _BEAMFORMER_TYPE_ALIASES.get(beamformer_type.strip().lower())
    effective_loading = diagonal_loading * max(classifier_diagonal_loading_scale, 1.0)

    if name == "freq_domain_das":
        return FrequencyDomainDelayAndSumBeamformer()
    if name == "band_split_das":
        return BandSplitDasRenderer()
    if name == "mvdr":
        return MVDRBeamformer(
            diagonal_loading=effective_loading,
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
        )
    if name == "superdirective":
        return SuperdirectiveBeamformer(
            diagonal_loading=effective_loading,
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
        )
    if name == "gevd":
        return GEVDBeamformer(
            diagonal_loading=effective_loading,
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
        )
    if name == "delay_and_sum":
        return DelayAndSumBeamformer()
    if name is None:
        logger.warning(
            "Unknown beamformer type %r; falling back to time-domain DAS. "
            "Available: %s",
            beamformer_type,
            ", ".join(sorted(_BEAMFORMER_TYPES)),
        )
    return DelayAndSumBeamformer()
