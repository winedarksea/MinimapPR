"""Frequency-domain A-format to B-format (AmbiX) converter for the Sirith
tetrahedral microphone array.

The four capsules are omnidirectional (pressure sensors), so a simple
time-domain matrix gives incorrect results. Instead we build the encoding
matrix E from the capsule positions, apply a frequency-dependent Tikhonov-
regularised pseudoinverse per FFT bin, and post-filter the X/Y/Z velocity
components with a spatial-aliasing low-pass above ~3.4 kHz.

Output: 4-channel B-format in ACN/SN3D normalisation (AmbiX) — W, X, Y, Z.

Reference geometry (centroid-relative, metres):
    MK1: [ 0.0,    0.050, 0.0   ]
    MK2: [ 0.0433, 0.025, 0.0   ]
    MK3: [ 0.0,    0.0,   0.0   ]  ← raw hardware origin, NOT the centroid
    MK4: [ 0.02165,0.025, 0.04082]

The centroid is approximately [0.016, 0.025, 0.010], so MK3 is NOT at the
centroid.  We correct for this by computing the centroid and shifting all
positions before deriving unit direction vectors — otherwise the matrix
produces incorrect X/Y/Z steering.
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

# Spatial-aliasing cutoff: c / (2 * edge_length).
# Tetrahedron edge ≈ 0.05 m → cutoff ≈ 3 430 Hz.
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
    """Convert 4-channel A-format (raw capsule PCM) to first-order B-format.

    Parameters
    ----------
    channels:
        Shape (4, N), float, normalised ±1.0.  Channels correspond to
        MK1, MK2, MK3, MK4 in the order defined by SIRITH_MIC_POSITIONS_M.
    sample_rate_hz:
        Sample rate in Hz.
    block_size:
        FFT size.  Must be a power of two.  Defaults to 4096.
    hop:
        Overlap-add hop in samples.  Defaults to block_size // 2 (50 %).
    mic_positions_m:
        Optional override for mic geometry.  Defaults to the Sirith array.

    Returns
    -------
    NDArray[np.float32]
        Shape (4, N), B-format channels W, X, Y, Z in ACN/SN3D normalisation.
    """
    if channels.ndim != 2 or channels.shape[0] != 4:
        raise ValueError("channels must have shape (4, N)")

    n_samples = channels.shape[1]
    if hop is None:
        hop = block_size // 2

    positions = (
        SIRITH_MIC_POSITIONS_M if mic_positions_m is None else np.asarray(mic_positions_m)
    )
    E = _build_encoding_matrix(positions)
    freqs = np.fft.rfftfreq(block_size, d=1.0 / sample_rate_hz)

    output = np.zeros((4, n_samples), dtype=np.float64)
    norm = np.zeros(n_samples, dtype=np.float64)
    window = np.hanning(block_size)

    n_blocks = (n_samples + hop - 1) // hop
    for b in range(n_blocks):
        start = b * hop
        end = min(start + block_size, n_samples)
        actual = end - start

        frame = np.zeros((4, block_size), dtype=np.float64)
        frame[:, :actual] = channels[:, start:end]
        frame *= window[np.newaxis, :]

        # Forward FFT of each capsule channel.
        A_freq = np.fft.rfft(frame, axis=1)  # (4, n_bins)

        # Apply frequency-dependent regularised pseudoinverse per bin.
        B_freq = _apply_atob_matrix(A_freq, E, freqs)

        # Alias LP filter on X/Y/Z (channels 1–3) above ALIAS_CUTOFF_HZ.
        _apply_alias_lp(B_freq, freqs)

        # IFFT back to time domain.
        B_time = np.fft.irfft(B_freq, n=block_size, axis=1)  # (4, block_size)

        # Overlap-add.
        output[:, start:end] += B_time[:, :actual] * window[:actual][np.newaxis, :]
        norm[start:end] += window[:actual] ** 2

    # Normalise overlap-add.
    safe_norm = np.where(norm > 1e-12, norm, 1.0)
    output /= safe_norm[np.newaxis, :]

    return np.clip(output, -1.0, 1.0).astype(np.float32)


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


def _build_encoding_matrix(
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build the 4×4 capsule encoding matrix E (capsule × B-format component).

    Row i of E: [1/sqrt(2), d_x_i, d_y_i, d_z_i]
    where (d_x, d_y, d_z) is the unit vector from the array centroid to mic i.
    """
    corrected, _ = centroid_corrected_positions(positions)
    norms = np.linalg.norm(corrected, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    unit_dirs = corrected / norms  # (4, 3)

    E = np.zeros((4, 4), dtype=np.float64)
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
    E: NDArray[np.float64],
    freqs: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Apply the frequency-dependent Tikhonov-regularised A-to-B matrix.

    For each bin k:
        B(k) = E⁺_reg(k) · A(k)
        E⁺_reg = Eᵀ (E Eᵀ + λ(f) I)⁻¹

    This is the standard Tikhonov solution for underdetermined least squares.

    Parameters
    ----------
    A_freq:
        Shape (4, n_bins), complex RFFT of the four capsule channels.
    E:
        Shape (4, 4), encoding matrix.
    freqs:
        Shape (n_bins,), frequency axis from np.fft.rfftfreq.

    Returns
    -------
    NDArray[np.complex128]
        Shape (4, n_bins), B-format in the frequency domain.
    """
    n_bins = A_freq.shape[1]
    B_freq = np.zeros((4, n_bins), dtype=np.complex128)
    lambdas = _tikhonov_lambda(freqs)

    # EEᵀ is constant (real, 4×4).
    EEt = E @ E.T

    for k in range(n_bins):
        lam = float(lambdas[k])
        reg_matrix = EEt + lam * np.eye(4)
        # E⁺_reg = Eᵀ (EEᵀ + λI)⁻¹
        # B(k) = E⁺_reg · A(k)  →  equivalent to solving (EEᵀ + λI) x = E A(k)
        #                                                               then x = E⁺_reg A(k).
        # Numerically: E⁺_reg = E.T @ inv(reg_matrix)
        # B(k) = E.T @ solve(reg_matrix, A(k))
        B_freq[:, k] = E.T @ np.linalg.solve(reg_matrix, A_freq[:, k])

    return B_freq


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
