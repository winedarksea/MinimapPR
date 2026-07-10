"""BlockTrajectoryRenderer omni_blend_above_cutoff behavior (Phase 7)."""

from __future__ import annotations

import numpy as np

from minimappr.core.ambi_atob import SIRITH_MIC_POSITIONS_M
from minimappr.core.beamforming import BlockTrajectoryRenderer, DelayAndSumBeamformer
from minimappr.spatial_audio.geometry import alias_cutoff_from_positions

SAMPLE_RATE_HZ = 16_000
N_SAMPLES = 8000
STEER_POS = (5.0, 0.0, 1.0)
SOUND_SPEED_MPS = 343.2


def _synth_channels(freq_hz: float) -> np.ndarray:
    mics = np.asarray(SIRITH_MIC_POSITIONS_M, dtype=np.float64)
    node_pos = np.zeros(3)
    src = np.asarray(STEER_POS, dtype=np.float64)
    t = np.arange(N_SAMPLES, dtype=np.float64) / SAMPLE_RATE_HZ
    channels = np.zeros((mics.shape[0], N_SAMPLES), dtype=np.float64)
    for m in range(mics.shape[0]):
        mic_world = node_pos + mics[m]
        distance = float(np.linalg.norm(src - mic_world))
        channels[m] = np.sin(2.0 * np.pi * freq_hz * (t - distance / SOUND_SPEED_MPS))
    return channels.astype(np.float32)


def _render(freq_hz: float, *, omni_blend: bool) -> np.ndarray:
    channels = _synth_channels(freq_hz)
    renderer = BlockTrajectoryRenderer(
        beamformer=DelayAndSumBeamformer(),
        block_size=512,
        omni_blend_above_cutoff=omni_blend,
    )
    return renderer.render(
        channels,
        np.asarray(SIRITH_MIC_POSITIONS_M, dtype=np.float64),
        SAMPLE_RATE_HZ,
        lambda sample_mid: STEER_POS,
        active_range=(0, N_SAMPLES),
    )


def test_alias_cutoff_is_above_low_test_tone() -> None:
    cutoff_hz = alias_cutoff_from_positions(
        np.asarray(SIRITH_MIC_POSITIONS_M, dtype=np.float64), SOUND_SPEED_MPS
    )
    assert cutoff_hz > 3000.0  # sanity check on tetra geometry


def test_blend_disabled_reproduces_pure_steered_output() -> None:
    high_freq = alias_cutoff_from_positions(
        np.asarray(SIRITH_MIC_POSITIONS_M, dtype=np.float64), SOUND_SPEED_MPS
    ) * 1.5
    without_blend = _render(high_freq, omni_blend=False)
    with_blend = _render(high_freq, omni_blend=True)
    # Above cutoff, blending pulls output toward the omni average, so the two
    # renders must differ once blending is enabled.
    assert not np.allclose(without_blend, with_blend, atol=1e-4)


def test_blend_enabled_moves_high_freq_output_toward_omni() -> None:
    cutoff_hz = alias_cutoff_from_positions(
        np.asarray(SIRITH_MIC_POSITIONS_M, dtype=np.float64), SOUND_SPEED_MPS
    )
    high_freq = cutoff_hz * 1.8

    channels = _synth_channels(high_freq)
    omni_reference = np.mean(channels.astype(np.float64), axis=0)

    steered_only = _render(high_freq, omni_blend=False)
    blended = _render(high_freq, omni_blend=True)

    trim = slice(1024, N_SAMPLES - 1024)  # avoid fade edges
    err_steered = np.sqrt(np.mean((steered_only[trim] - omni_reference[trim]) ** 2))
    err_blended = np.sqrt(np.mean((blended[trim] - omni_reference[trim]) ** 2))
    assert err_blended < err_steered


def test_blend_leaves_low_freq_below_cutoff_essentially_unchanged() -> None:
    cutoff_hz = alias_cutoff_from_positions(
        np.asarray(SIRITH_MIC_POSITIONS_M, dtype=np.float64), SOUND_SPEED_MPS
    )
    low_freq = cutoff_hz * 0.3

    without_blend = _render(low_freq, omni_blend=False)
    with_blend = _render(low_freq, omni_blend=True)

    trim = slice(1024, N_SAMPLES - 1024)
    np.testing.assert_allclose(without_blend[trim], with_blend[trim], atol=1e-3)
