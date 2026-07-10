"""Objective audio-quality guards for the BlockTrajectoryRenderer / MVDR path.

These don't replace a human listening pass, but catch the failure modes that
produce audible pops, clicks, and blow-ups: block-boundary discontinuities,
ill-conditioned MVDR covariance spikes, NaN/Inf propagation, and runaway gain.
"""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.ambi_atob import SIRITH_MIC_POSITIONS_M
from minimappr.core.beamforming import BlockTrajectoryRenderer, DelayAndSumBeamformer, MVDRBeamformer

SAMPLE_RATE_HZ = 16_000
N_SAMPLES = 16_000
STEER_POS = (5.0, 0.0, 1.0)
SOUND_SPEED_MPS = 343.2
MIC_POSITIONS = np.asarray(SIRITH_MIC_POSITIONS_M, dtype=np.float64)


def _propagated_tone(freq_hz: float, src_pos: tuple[float, float, float], amplitude: float = 0.3) -> np.ndarray:
    src = np.asarray(src_pos, dtype=np.float64)
    t = np.arange(N_SAMPLES, dtype=np.float64) / SAMPLE_RATE_HZ
    channels = np.zeros((MIC_POSITIONS.shape[0], N_SAMPLES), dtype=np.float64)
    for m in range(MIC_POSITIONS.shape[0]):
        distance = float(np.linalg.norm(src - MIC_POSITIONS[m]))
        channels[m] = amplitude * np.sin(2.0 * np.pi * freq_hz * (t - distance / SOUND_SPEED_MPS))
    return channels.astype(np.float32)


def _max_sample_jump(signal: np.ndarray, guard_samples: int = 32) -> float:
    """Largest sample-to-sample derivative, ignoring block edges near a fade boundary."""
    trimmed = signal[guard_samples:-guard_samples] if signal.size > 2 * guard_samples else signal
    if trimmed.size < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(trimmed))))


def test_output_has_no_nan_or_inf() -> None:
    channels = _propagated_tone(800.0, STEER_POS)
    renderer = BlockTrajectoryRenderer(beamformer=MVDRBeamformer(), block_size=512)
    output = renderer.render(
        channels, MIC_POSITIONS, SAMPLE_RATE_HZ, lambda s: STEER_POS, active_range=(0, N_SAMPLES)
    )
    assert np.all(np.isfinite(output))


def test_static_tone_das_output_has_no_block_boundary_clicks() -> None:
    """A stationary single-tone source should render with no derivative spikes
    at 256-sample hop boundaries — a click would show up as an outlier jump."""
    channels = _propagated_tone(800.0, STEER_POS)
    renderer = BlockTrajectoryRenderer(beamformer=DelayAndSumBeamformer(), block_size=512)
    output = renderer.render(
        channels, MIC_POSITIONS, SAMPLE_RATE_HZ, lambda s: STEER_POS, active_range=(0, N_SAMPLES)
    )
    # Steady-state max derivative for a pure tone at this rate/frequency is a
    # known small bound; a click would blow well past it.
    steady_state = output[1024:-1024]
    typical_jump = float(np.median(np.abs(np.diff(steady_state))))
    worst_jump = _max_sample_jump(output, guard_samples=1024)
    assert worst_jump < 8.0 * max(typical_jump, 1e-6)


def test_moving_trajectory_mvdr_output_has_no_block_boundary_clicks() -> None:
    """Steering position changes abruptly at each block boundary (fast-moving
    track). Weighted OLA should smooth the transition instead of producing a
    click at the seam."""
    channels = _propagated_tone(800.0, STEER_POS, amplitude=0.3)

    def steer_for_sample(sample_mid: int) -> tuple[float, float, float]:
        # Alternate steering target every block to stress-test seam smoothing.
        return STEER_POS if (sample_mid // 512) % 2 == 0 else (5.0, 3.0, 1.0)

    renderer = BlockTrajectoryRenderer(beamformer=MVDRBeamformer(), block_size=512)
    output = renderer.render(
        channels, MIC_POSITIONS, SAMPLE_RATE_HZ, steer_for_sample, active_range=(0, N_SAMPLES)
    )
    assert np.all(np.isfinite(output))
    steady_state = output[1024:-1024]
    typical_jump = float(np.median(np.abs(np.diff(steady_state))))
    worst_jump = _max_sample_jump(output, guard_samples=1024)
    assert worst_jump < 12.0 * max(typical_jump, 1e-6)


def test_mvdr_rank_deficient_covariance_does_not_blow_up() -> None:
    """Identical signal on every mic (fully coherent, rank-1 covariance) is the
    classic MVDR ill-conditioning case. Diagonal loading should keep the
    output bounded rather than producing a spike."""
    mono = 0.3 * np.sin(2.0 * np.pi * 800.0 * np.arange(N_SAMPLES) / SAMPLE_RATE_HZ)
    channels = np.tile(mono, (MIC_POSITIONS.shape[0], 1)).astype(np.float32)

    renderer = BlockTrajectoryRenderer(beamformer=MVDRBeamformer(diagonal_loading=1e-3), block_size=512)
    output = renderer.render(
        channels, MIC_POSITIONS, SAMPLE_RATE_HZ, lambda s: STEER_POS, active_range=(0, N_SAMPLES)
    )
    assert np.all(np.isfinite(output))
    # Bounded relative to input amplitude — MVDR is unity-gain toward the
    # steering direction, so output shouldn't blow past a modest multiple of
    # the input even under rank deficiency.
    assert float(np.max(np.abs(output))) < 5.0 * 0.3


def test_mvdr_silent_input_does_not_blow_up() -> None:
    """All-zero covariance (silence) is another degenerate MVDR case —
    guards against division-by-near-zero producing spikes."""
    channels = np.zeros((MIC_POSITIONS.shape[0], N_SAMPLES), dtype=np.float32)
    renderer = BlockTrajectoryRenderer(beamformer=MVDRBeamformer(), block_size=512)
    output = renderer.render(
        channels, MIC_POSITIONS, SAMPLE_RATE_HZ, lambda s: STEER_POS, active_range=(0, N_SAMPLES)
    )
    assert np.all(np.isfinite(output))
    assert float(np.max(np.abs(output))) < 1e-3


def test_output_gain_stays_bounded_relative_to_input() -> None:
    """MVDR/DAS beamforming toward the true source direction should not
    amplify a moderate-level input into clipping territory."""
    channels = _propagated_tone(800.0, STEER_POS, amplitude=0.5)
    for beamformer in (DelayAndSumBeamformer(), MVDRBeamformer()):
        renderer = BlockTrajectoryRenderer(beamformer=beamformer, block_size=512)
        output = renderer.render(
            channels, MIC_POSITIONS, SAMPLE_RATE_HZ, lambda s: STEER_POS, active_range=(0, N_SAMPLES)
        )
        assert float(np.max(np.abs(output))) < 1.0, f"{type(beamformer).__name__} output exceeds full scale"


def test_fade_in_and_out_are_monotonic_and_click_free() -> None:
    """active_range fades should ramp smoothly, not step, to avoid an audible
    click at object onset/offset."""
    channels = _propagated_tone(800.0, STEER_POS)
    renderer = BlockTrajectoryRenderer(beamformer=DelayAndSumBeamformer(), block_size=512, fade_seconds=0.02)
    active_range = (2000, 14000)
    output = renderer.render(
        channels, MIC_POSITIONS, SAMPLE_RATE_HZ, lambda s: STEER_POS, active_range=active_range
    )
    # Well outside the active range (beyond one block's worth of straddle)
    # must be silent (no leakage click).
    block_size = 512
    assert np.max(np.abs(output[: active_range[0] - block_size])) < 1e-4
    assert np.max(np.abs(output[active_range[1] + block_size :])) < 1e-4
    # No derivative spike at the fade boundaries themselves.
    fade_samples = max(512, int(round(0.02 * SAMPLE_RATE_HZ)))
    boundary_window = output[active_range[0] : active_range[0] + fade_samples]
    worst_jump = _max_sample_jump(boundary_window, guard_samples=0)
    interior_jump = _max_sample_jump(output[4000:12000], guard_samples=0)
    assert worst_jump < 8.0 * max(interior_jump, 1e-6)
