"""Cross-language agreement for the Rust and Python band-split DAS renders.

Compares the Rust live render (``beamform-oracle`` CLI mode of the ingest
sidecar) against the Python :class:`BandSplitDasRenderer` per
BEAMFORMED_RENDER_CONTRACT.md §5: sample-level RMS error ≤ 1e-3 of full scale
and spectral magnitude agreement within 0.5 dB on occupied bins.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from minimappr.core.ambi_atob import SIRITH_MIC_POSITIONS_M
from minimappr.core.beamforming import BandSplitDasRenderer, BandSplitRenderConfig
from minimappr.spatial_audio.geometry import alias_cutoff_from_positions

MINIMAPPR_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_DIR = MINIMAPPR_ROOT / "minimappr-ingest-sidecar"
SIDECAR_BINARY_PATH = SIDECAR_DIR / "target" / "debug" / "minimappr-ingest-sidecar"

SAMPLE_RATE_HZ = 16_000
SOUND_SPEED_MPS = 343.2


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


def _run_rust_beamform_oracle(
    channels: np.ndarray,
    mic_positions_m: np.ndarray,
    steer_position_m: tuple[float, float, float],
) -> dict:
    payload = {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "mic_positions_m": mic_positions_m.tolist(),
        "steer_position_m": list(steer_position_m),
        "channels": [channel.astype(float).tolist() for channel in channels],
        "sound_speed_mps": SOUND_SPEED_MPS,
    }
    completed = subprocess.run(
        [str(_sidecar_binary()), "beamform-oracle"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(completed.stdout)
    output["samples"] = np.asarray(output["samples"], dtype=np.float32)
    return output


def _run_python_render(
    channels: np.ndarray,
    mic_positions_m: np.ndarray,
    steer_position_m: tuple[float, float, float],
) -> np.ndarray:
    renderer = BandSplitDasRenderer(config=BandSplitRenderConfig())
    sensor_ids = [f"m{i}" for i in range(channels.shape[0])]
    positions = {sid: np.asarray(mic_positions_m[i], dtype=np.float64) for i, sid in enumerate(sensor_ids)}
    windows = {sid: channels[i] for i, sid in enumerate(sensor_ids)}
    return renderer.beamform(
        sensor_positions=positions,
        sensor_windows=windows,
        sample_rate_hz=SAMPLE_RATE_HZ,
        steer_position_m=steer_position_m,
        sound_speed_mps=SOUND_SPEED_MPS,
    )


def _propagated_channels(
    mic_positions_m: np.ndarray,
    sources: list[tuple[tuple[float, float, float], list[tuple[float, float]]]],
    n_samples: int,
    noise_scale: float = 0.005,
) -> np.ndarray:
    """Synthesize per-mic signals from point sources with true propagation delays."""
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE_HZ
    rng = np.random.default_rng(424242)
    channels = np.zeros((len(mic_positions_m), n_samples), dtype=np.float64)
    for src_pos, tones in sources:
        src = np.asarray(src_pos, dtype=np.float64)
        for mic_index, mic in enumerate(mic_positions_m):
            tau = float(np.linalg.norm(src - np.asarray(mic, dtype=np.float64))) / SOUND_SPEED_MPS
            for amp, freq in tones:
                channels[mic_index] += amp * np.sin(2.0 * np.pi * freq * (t - tau))
    channels += noise_scale * rng.standard_normal(channels.shape)
    return channels.astype(np.float32)


def _assert_contract_parity(python_out: np.ndarray, rust_out: np.ndarray) -> None:
    assert rust_out.shape == python_out.shape
    # §5: RMS of the sample-level difference ≤ 1e-3 of full scale.
    rms_err = float(np.sqrt(np.mean((rust_out.astype(np.float64) - python_out.astype(np.float64)) ** 2)))
    assert rms_err <= 1e-3, f"sample RMS error {rms_err} exceeds contract tolerance"

    # §5: spectral magnitude within 0.5 dB on occupied bins.
    py_mag = np.abs(np.fft.rfft(python_out.astype(np.float64)))
    rs_mag = np.abs(np.fft.rfft(rust_out.astype(np.float64)))
    occupied = py_mag > (py_mag.max() * 1e-2)
    ratio_db = 20.0 * np.log10((rs_mag[occupied] + 1e-12) / (py_mag[occupied] + 1e-12))
    assert float(np.max(np.abs(ratio_db))) <= 0.5, (
        f"max spectral deviation {float(np.max(np.abs(ratio_db))):.3f} dB exceeds 0.5 dB"
    )


def test_tetra_multi_tone_with_interferer_matches() -> None:
    """Seeded multi-tone target + off-axis interferer on the Sirith tetra."""
    mic = np.asarray(SIRITH_MIC_POSITIONS_M, dtype=np.float64)
    target = (4.0, 2.0, 1.0)
    channels = _propagated_channels(
        mic,
        sources=[
            (target, [(0.25, 800.0), (0.15, 1900.0), (0.1, 5200.0)]),
            ((-3.0, -4.0, 0.5), [(0.2, 1200.0), (0.1, 6400.0)]),
        ],
        n_samples=int(SAMPLE_RATE_HZ * 0.25),
    )
    python_out = _run_python_render(channels, mic, target)
    rust = _run_rust_beamform_oracle(channels, mic, target)
    _assert_contract_parity(python_out, rust["samples"])

    expected_cutoff = alias_cutoff_from_positions(mic, SOUND_SPEED_MPS)
    assert abs(rust["alias_cutoff_hz"] - expected_cutoff) < 1.0
    assert abs(rust["effective_spatial_band"][1] - expected_cutoff) < 1.0


def test_two_mic_non_tetra_geometry_matches() -> None:
    """2-mic array exercises the geometry-derived cutoff on non-tetra geometry."""
    mic = np.asarray([[0.0, 0.0, 0.0], [0.12, 0.0, 0.0]], dtype=np.float64)
    target = (2.0, 1.0, 0.0)
    channels = _propagated_channels(
        mic,
        sources=[(target, [(0.3, 600.0), (0.12, 1100.0), (0.08, 3000.0)])],
        n_samples=int(SAMPLE_RATE_HZ * 0.2),
    )
    python_out = _run_python_render(channels, mic, target)
    rust = _run_rust_beamform_oracle(channels, mic, target)
    _assert_contract_parity(python_out, rust["samples"])

    # 0.12 m baseline → c / (2·0.12) = 1430 Hz cutoff.
    assert abs(rust["alias_cutoff_hz"] - SOUND_SPEED_MPS / 0.24) < 1.0
