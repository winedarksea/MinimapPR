"""Cross-language agreement for the Rust and Python ambisonics encoders."""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from minimappr.core.ambi_atob import SIRITH_MIC_POSITIONS_M
from minimappr.spatial_audio import encode_ambisonics


MINIMAPPR_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_DIR = MINIMAPPR_ROOT / "minimappr-ingest-sidecar"
SIDECAR_BINARY_PATH = SIDECAR_DIR / "target" / "debug" / "minimappr-ingest-sidecar"

SAMPLE_RATE_HZ = 16_000
BLOCK_SIZE = 1024


pytestmark = pytest.mark.skipif(
    shutil.which("cargo") is None,
    reason="cargo toolchain required to build the Rust sidecar oracle",
)


@functools.lru_cache(maxsize=1)
def _sidecar_binary() -> Path:
    subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "minimappr-ingest-sidecar"],
        cwd=str(SIDECAR_DIR),
        check=True,
    )
    return SIDECAR_BINARY_PATH


def _run_rust_ambi_oracle(channels: np.ndarray, profile: str) -> np.ndarray:
    payload = {
        "profile": profile,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "block_size": BLOCK_SIZE,
        "hop_size": BLOCK_SIZE // 2,
        "mic_positions_m": SIRITH_MIC_POSITIONS_M.tolist(),
        "channels": [channel.astype(float).tolist() for channel in channels],
    }
    completed = subprocess.run(
        [str(_sidecar_binary()), "ambi-oracle"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(completed.stdout)
    assert output["profile"] == profile
    assert output["sample_rate_hz"] == SAMPLE_RATE_HZ
    return np.asarray(output["bformat"], dtype=np.float32)


def _build_deterministic_input() -> np.ndarray:
    n_samples = int(SAMPLE_RATE_HZ * 0.2)
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE_HZ
    rng = np.random.default_rng(98765)
    tone = (0.28 * np.sin(2.0 * np.pi * 930.0 * t)).astype(np.float32)
    noise = (0.03 * rng.standard_normal((4, n_samples))).astype(np.float32)
    channels = np.zeros((4, n_samples), dtype=np.float32)
    for channel_index, delay_samples in enumerate([0, 1, 2, 3]):
        channels[channel_index, delay_samples:] = tone[: n_samples - delay_samples]
    return channels + noise


@pytest.mark.parametrize("profile", ["linear_v1", "parametric_v2"])
def test_rust_ambisonics_matches_python(profile: str) -> None:
    channels = _build_deterministic_input()
    python_bformat = encode_ambisonics(
        channels,
        SAMPLE_RATE_HZ,
        profile=profile,
        mic_positions_m=SIRITH_MIC_POSITIONS_M,
    )
    rust_bformat = _run_rust_ambi_oracle(channels, profile)

    assert rust_bformat.shape == python_bformat.shape
    np.testing.assert_allclose(rust_bformat, python_bformat, rtol=1e-4, atol=1e-4)


def test_rust_parametric_v2_matches_checked_in_golden_fixture() -> None:
    with np.load(Path(__file__).parent / "fixtures" / "atob_foa_tetra_golden_v2.npz") as fixture:
        channels = fixture["channels"]
        golden = fixture["bed"]
    rust_bformat = _run_rust_ambi_oracle(channels, "parametric_v2")
    np.testing.assert_allclose(
        rust_bformat,
        golden,
        rtol=1e-5,
        atol=2e-5,
    )
