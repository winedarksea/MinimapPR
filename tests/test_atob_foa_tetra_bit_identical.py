"""Phase E: regression guard — tetra atob_foa output must stay bit-identical.

The generalized M-mic FOA path must reproduce the legacy 4-mic Sirith tetra
output exactly. The first time this test runs (no fixture yet) it captures a
snapshot; subsequent runs assert equality.

If a deliberate FOA algorithm change is made, delete the fixture to
re-baseline, but only after confirming the change is intended.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minimappr.core.ambi_atob import SIRITH_MIC_POSITIONS_M, atob_foa


_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "atob_foa_tetra_golden.npz"

# Synthesis parameters — keep these constants stable to preserve the fixture.
_SAMPLE_RATE_HZ = 16_000
_DURATION_S = 0.2
_BLOCK_SIZE = 1024
_SEED = 1234


def _build_deterministic_tetra_input() -> np.ndarray:
    """Generate a 4-channel signal with reproducible spectral content.

    Uses a fixed-seed RNG plus a 1 kHz tone to give the FOA encoder both a
    diffuse and a localizable component.
    """
    n = int(_SAMPLE_RATE_HZ * _DURATION_S)
    t = np.arange(n, dtype=np.float64) / _SAMPLE_RATE_HZ
    rng = np.random.default_rng(_SEED)

    noise = rng.standard_normal((4, n)).astype(np.float32) * 0.05
    tone = (0.3 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

    # Per-mic constant delay to simulate a plane wave (no real geometric calc
    # needed for a regression snapshot — just need stable, non-degenerate input).
    delays_samples = [0, 1, 2, 3]
    channels = np.zeros((4, n), dtype=np.float32)
    for idx, d in enumerate(delays_samples):
        channels[idx, d:] += tone[: n - d]
    channels += noise
    return channels


def test_atob_foa_tetra_bit_identical() -> None:
    channels = _build_deterministic_tetra_input()
    bed = atob_foa(
        channels,
        _SAMPLE_RATE_HZ,
        block_size=_BLOCK_SIZE,
        mic_positions_m=SIRITH_MIC_POSITIONS_M,
    )

    assert bed.shape == (4, channels.shape[1])
    assert bed.dtype == np.float32

    if not _FIXTURE_PATH.exists():
        _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(_FIXTURE_PATH, bed=bed)
        pytest.skip(
            f"Captured new FOA golden fixture at {_FIXTURE_PATH}. "
            "Re-run the test to enforce bit-identity going forward."
        )

    with np.load(_FIXTURE_PATH) as data:
        golden = data["bed"]

    assert golden.shape == bed.shape, (
        f"Golden bed shape {golden.shape} != current {bed.shape}"
    )
    # float32 bit-equality is the regression contract.
    np.testing.assert_array_equal(
        bed,
        golden,
        err_msg=(
            "atob_foa output diverged from the golden fixture. "
            "If this change is intentional, delete the fixture file "
            f"({_FIXTURE_PATH.relative_to(Path(__file__).parent.parent)}) "
            "to re-baseline."
        ),
    )


def test_atob_foa_default_geometry_path_matches_explicit() -> None:
    """Passing mic_positions_m=None must equal passing SIRITH_MIC_POSITIONS_M."""
    channels = _build_deterministic_tetra_input()
    bed_default = atob_foa(channels, _SAMPLE_RATE_HZ, block_size=_BLOCK_SIZE)
    bed_explicit = atob_foa(
        channels,
        _SAMPLE_RATE_HZ,
        block_size=_BLOCK_SIZE,
        mic_positions_m=SIRITH_MIC_POSITIONS_M,
    )
    np.testing.assert_array_equal(bed_default, bed_explicit)
