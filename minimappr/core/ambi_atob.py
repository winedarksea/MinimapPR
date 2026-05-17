"""Frequency-domain A-format to B-format (AmbiX) converter for the Sirith
tetrahedral microphone array.

The four capsules are omnidirectional pressure sensors, so a static gain
matrix cannot recover useful velocity components. Instead each FFT bin uses a
first-order pressure-gradient model:

    p(r, f) ~= sqrt(2) * W(f) - j * k(f) * dot(r, [X(f), Y(f), Z(f)])

where r is the centroid-relative capsule position and k = 2*pi*f/c. The
frequency-dependent Tikhonov pseudoinverse solves that model per bin, then
X/Y/Z are low-passed above the array's spatial-aliasing limit.

Output: 4-channel B-format in ACN/SN3D normalisation (AmbiX) — W, X, Y, Z.

Reference geometry (centroid-relative, metres):
    MK1: [ 0.0,    0.050, 0.0   ]
    MK2: [ 0.0433, 0.025, 0.0   ]
    MK3: [ 0.0,    0.0,   0.0   ]  ← raw hardware origin, NOT the centroid
    MK4: [ 0.02165,0.025, 0.04082]

The centroid is approximately [0.016, 0.025, 0.010], so MK3 is NOT at the
centroid. We correct for this by computing the centroid and shifting all
positions before solving the pressure-gradient model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ── Geometry ──────────────────────────────────────────────────────────────────

SIRITH_MIC_POSITIONS_M: NDArray[np.float64] = np.array(
    [
        [0.0, 0.050, 0.0],
        [0.0433, 0.025, 0.0],
        [0.0, 0.0, 0.0],
        [0.02165, 0.025, 0.04082],
    ],
    dtype=np.float64,
)

SPEED_OF_SOUND_MPS: float = 343.2

# Spatial-aliasing cutoff for the default Sirith geometry (edge ≈ 0.05 m).
ALIAS_CUTOFF_HZ: float = SPEED_OF_SOUND_MPS / (2.0 * 0.05)

# Tikhonov regularisation floor at DC (dimensionless noise floor).
TIKHONOV_LAMBDA_0: float = 1e-3
# Reference frequency for frequency-dependent regularisation.
TIKHONOV_F_REF_HZ: float = 120.0

_INV_SQRT2 = 1.0 / np.sqrt(2.0)


# ── Public API ────────────────────────────────────────────────────────────────


def atob_foa(
    channels: NDArray[np.float32 | np.float64],
    sample_rate_hz: int,
    *,
    block_size: int = 4096,
    hop: int | None = None,
    mic_positions_m: NDArray[np.float64] | None = None,
) -> NDArray[np.float32]:
    """Convert M-channel A-format (raw capsule PCM) to first-order B-format.

    Parameters
    ----------
    channels:
        Shape (M, N), float, normalised ±1.0.  M ≥ 4 mics in the order defined
        by *mic_positions_m* (or SIRITH_MIC_POSITIONS_M when omitted).
    sample_rate_hz:
        Sample rate in Hz.
    block_size:
        FFT size.  Must be a power of two.  Defaults to 4096.
    hop:
        Overlap-add hop in samples.  Defaults to block_size // 2 (50 %).
    mic_positions_m:
        Optional override for mic geometry (shape (M, 3)).  Defaults to the
        Sirith 4-mic array.  Must match channels.shape[0].

    Returns
    -------
    NDArray[np.float32]
        Shape (4, N), B-format channels W, X, Y, Z in ACN/SN3D normalisation.
    """
    positions = (
        SIRITH_MIC_POSITIONS_M if mic_positions_m is None else np.asarray(mic_positions_m, dtype=np.float64)
    )
    M = channels.shape[0] if channels.ndim == 2 else 0
    if channels.ndim != 2 or M < 4:
        raise ValueError(f"channels must have shape (M≥4, N), got {channels.shape}")
    if M != positions.shape[0]:
        raise ValueError(
            f"channels has {M} rows but mic_positions_m has {positions.shape[0]} rows"
        )

    n_samples = channels.shape[1]
    if hop is None:
        hop = block_size // 2

    corrected_positions, _ = centroid_corrected_positions(positions)
    freqs = np.fft.rfftfreq(block_size, d=1.0 / sample_rate_hz)

    # Compute spatial-aliasing cutoff from the actual array geometry.
    alias_cutoff_hz = _alias_cutoff_from_positions(positions)

    output = np.zeros((4, n_samples), dtype=np.float64)
    norm = np.zeros(n_samples, dtype=np.float64)
    window = np.hanning(block_size)

    n_blocks = (n_samples + hop - 1) // hop
    for b in range(n_blocks):
        start = b * hop
        end = min(start + block_size, n_samples)
        actual = end - start

        frame = np.zeros((M, block_size), dtype=np.float64)
        frame[:, :actual] = channels[:, start:end]
        frame *= window[np.newaxis, :]

        # Forward FFT of each capsule channel: (M, n_bins)
        A_freq = np.fft.rfft(frame, axis=1)

        # Frequency-dependent regularised pseudoinverse per bin → (4, n_bins)
        B_freq = _apply_atob_matrix(A_freq, corrected_positions, freqs)

        # Alias LP filter on X/Y/Z above array's spatial-aliasing cutoff.
        _apply_alias_lp(B_freq, freqs, cutoff_hz=alias_cutoff_hz)

        # IFFT back to time domain: (4, block_size)
        B_time = np.fft.irfft(B_freq, n=block_size, axis=1)

        # Overlap-add.
        output[:, start:end] += B_time[:, :actual] * window[:actual][np.newaxis, :]
        norm[start:end] += window[:actual] ** 2

    # Normalise overlap-add.
    safe_norm = np.where(norm > 1e-12, norm, 1.0)
    output /= safe_norm[np.newaxis, :]

    return np.clip(output, -1.0, 1.0).astype(np.float32)


def foa_geometry_suitable(
    positions: NDArray[np.float64],
    max_baseline_m: float,
) -> tuple[bool, str]:
    """Return (suitable, reason) indicating whether *positions* is suitable for FOA.

    FOA synthesis requires ≥ 4 mics and an array baseline ≤ *max_baseline_m*
    (above which spatial aliasing overwhelms the velocity channels).
    """
    n_mics = positions.shape[0]
    if n_mics < 4:
        return False, f"FOA requires ≥4 mics; array has {n_mics}"
    baseline = _max_baseline_m(positions)
    if baseline > max_baseline_m:
        return False, (
            f"array baseline {baseline:.3f} m exceeds FOA limit {max_baseline_m:.3f} m"
        )
    return True, "ok"


def centroid_corrected_positions(
    positions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (corrected_positions, centroid).

    Shifts all capsule positions so the centroid is at the origin.
    """
    centroid = positions.mean(axis=0)
    return positions - centroid, centroid


def encode_mono_to_bformat(
    mono: NDArray[np.float32 | np.float64],
    direction_xyz: tuple[float, float, float],
) -> NDArray[np.float32]:
    """Re-encode a mono signal into B-format for the given steering direction.

    Used by the spatial subtraction step:
        B_clean = B_full − Σᵢ Y_obj_i · O_i

    Parameters
    ----------
    mono:
        Shape (N,) mono signal.
    direction_xyz:
        Unit steering direction (x, y, z).

    Returns
    -------
    NDArray[np.float32]
        Shape (4, N), B-format W/X/Y/Z.
    """
    d = np.asarray(direction_xyz, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm > 1e-9:
        d /= norm

    n = len(mono)
    bformat = np.zeros((4, n), dtype=np.float64)
    sig = np.asarray(mono, dtype=np.float64)
    bformat[0] = _INV_SQRT2 * sig       # W
    bformat[1] = d[0] * sig             # X
    bformat[2] = d[1] * sig             # Y
    bformat[3] = d[2] * sig             # Z
    return bformat.astype(np.float32)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _max_baseline_m(positions: NDArray[np.float64]) -> float:
    """Return the maximum pairwise distance between any two mic positions."""
    n = positions.shape[0]
    if n < 2:
        return 0.0
    max_d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d > max_d:
                max_d = d
    return max_d


def _alias_cutoff_from_positions(positions: NDArray[np.float64]) -> float:
    """Spatial-aliasing cutoff for the given mic geometry: c / (2 * max_edge)."""
    baseline = _max_baseline_m(positions)
    if baseline < 1e-6:
        return ALIAS_CUTOFF_HZ  # degenerate geometry: fall back to default
    return SPEED_OF_SOUND_MPS / (2.0 * baseline)


def _build_encoding_matrix(
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build the M×4 capsule encoding matrix E (capsule × B-format component).

    Row i of E: [1/sqrt(2), d_x_i, d_y_i, d_z_i]
    where (d_x, d_y, d_z) is the unit vector from the array centroid to mic i.
    Works for arbitrary M ≥ 1 mics.
    """
    corrected, _ = centroid_corrected_positions(positions)
    norms = np.linalg.norm(corrected, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    unit_dirs = corrected / norms  # (M, 3)

    M = positions.shape[0]
    E = np.zeros((M, 4), dtype=np.float64)
    E[:, 0] = _INV_SQRT2          # W component (omnidirectional)
    E[:, 1] = unit_dirs[:, 0]     # X
    E[:, 2] = unit_dirs[:, 1]     # Y
    E[:, 3] = unit_dirs[:, 2]     # Z
    return E


def _tikhonov_lambda(freqs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Frequency-dependent Tikhonov regularisation strength.

    λ(f) = λ₀ · (f_ref / max(f, f_ref))²

    Adapts dynamically: at DC the array has no directional information so
    regularisation is maximal (λ₀); above f_ref it decays as 1/f².
    """
    f_clamped = np.maximum(np.abs(freqs), TIKHONOV_F_REF_HZ)
    return TIKHONOV_LAMBDA_0 * (TIKHONOV_F_REF_HZ / f_clamped) ** 2


def _apply_atob_matrix(
    A_freq: NDArray[np.complex128],
    corrected_positions_m: NDArray[np.float64],
    freqs: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Apply a physical, frequency-dependent pressure-gradient A-to-B solve.

    Accepts M mics (corrected_positions_m has shape (M, 3)).
    Always returns (4, n_bins) B-format output.
    """
    M = corrected_positions_m.shape[0]
    n_bins = A_freq.shape[1]
    B_freq = np.zeros((4, n_bins), dtype=np.complex128)
    lambdas = _tikhonov_lambda(freqs)

    for bin_index, freq_hz in enumerate(freqs):
        E = _build_frequency_domain_pressure_gradient_matrix(
            corrected_positions_m,
            float(freq_hz),
        )
        lam = float(lambdas[bin_index])
        reg_matrix = E @ E.conj().T + lam * np.eye(M)  # (M, M)
        try:
            B_freq[:, bin_index] = E.conj().T @ np.linalg.solve(
                reg_matrix,
                A_freq[:, bin_index],
            )
        except np.linalg.LinAlgError:
            B_freq[:, bin_index] = (
                E.conj().T @ np.linalg.pinv(reg_matrix) @ A_freq[:, bin_index]
            )

    return B_freq


def _build_frequency_domain_pressure_gradient_matrix(
    corrected_positions_m: NDArray[np.float64],
    freq_hz: float,
) -> NDArray[np.complex128]:
    """Return capsule pressure response for AmbiX W/X/Y/Z at one frequency.

    Returns shape (M, 4) where M = corrected_positions_m.shape[0].
    """
    M = corrected_positions_m.shape[0]
    wavenumber = 2.0 * np.pi * max(freq_hz, 0.0) / SPEED_OF_SOUND_MPS
    E = np.zeros((M, 4), dtype=np.complex128)
    E[:, 0] = np.sqrt(2.0)
    E[:, 1:] = -1j * wavenumber * corrected_positions_m
    return E


def _apply_alias_lp(
    B_freq: NDArray[np.complex128],
    freqs: NDArray[np.float64],
    *,
    cutoff_hz: float = ALIAS_CUTOFF_HZ,
    order: int = 4,
) -> None:
    """Apply Butterworth-style spatial-aliasing LP filter to X/Y/Z channels
    (B_freq[1:4]) above `cutoff_hz`.  W is left full-bandwidth.

    Modifies `B_freq` in-place.
    """
    # Butterworth magnitude response: H(f) = 1 / sqrt(1 + (f/fc)^(2*order))
    f_ratio = np.abs(freqs) / cutoff_hz
    gain = 1.0 / np.sqrt(1.0 + f_ratio ** (2 * order))
    # Apply only to X, Y, Z.
    B_freq[1:4, :] *= gain[np.newaxis, :]
