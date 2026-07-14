"""Golden-vector agreement between Python and the real Rust ingest DSP."""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from minimappr.audio_processing.chain import build_chain_from_rust_stages

ROOT = Path(__file__).resolve().parents[1]
SIDECAR = ROOT / "minimappr-ingest-sidecar"
BINARY = SIDECAR / "target" / "debug" / "minimappr-ingest-sidecar"
SAMPLE_RATE_HZ = 16_000

pytestmark = pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is required")


@functools.lru_cache(maxsize=1)
def _binary() -> Path:
    subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "minimappr-ingest-sidecar"],
        cwd=SIDECAR,
        check=True,
    )
    return BINARY


def _rust(frames: list[np.ndarray], stages: list[dict]) -> list[np.ndarray]:
    payload = {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "stages": stages,
        "frames": [frame.astype(float).tolist() for frame in frames],
    }
    completed = subprocess.run(
        [str(_binary()), "preprocess-oracle"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return [np.asarray(frame, dtype=np.float32) for frame in json.loads(completed.stdout)["frames"]]


@pytest.mark.parametrize(
    "stages",
    [
        [{"type": "gain", "db": 6.0}],
        [{"type": "channel_gain", "db_by_channel": [3.0, -6.0]}],
        [{"type": "highpass", "cutoff_hz": 200.0, "order": 4}],
        [{"type": "lowpass", "cutoff_hz": 3_000.0, "order": 4}],
        [{"type": "bandpass", "low_hz": 200.0, "high_hz": 3_000.0, "order": 4}],
        [{"type": "dc_block"}],
        [{"type": "gain", "db": -3.0}, {"type": "dc_block"}],
    ],
)
def test_real_rust_matches_python_across_frames(stages: list[dict]) -> None:
    rng = np.random.default_rng(20260714)
    frames = [
        rng.normal(0.08, 0.02, size=(2, 257)).astype(np.float32),
        rng.normal(-0.03, 0.04, size=(2, 199)).astype(np.float32),
    ]
    chain = build_chain_from_rust_stages(stages)
    python_frames = [
        np.stack(
            [
                chain.process(channel.copy(), SAMPLE_RATE_HZ, channel_idx=channel_index)
                for channel_index, channel in enumerate(frame)
            ]
        )
        for frame in frames
    ]
    rust_frames = _rust(frames, stages)
    for python_frame, rust_frame in zip(python_frames, rust_frames, strict=True):
        np.testing.assert_allclose(rust_frame, python_frame, rtol=2e-5, atol=2e-6)


def test_rust_rejects_invalid_profile_like_python() -> None:
    invalid = [{"type": "gain", "db": 100.0}]
    with pytest.raises(ValueError):
        build_chain_from_rust_stages(invalid)
    completed = subprocess.run(
        [str(_binary()), "preprocess-oracle"],
        input=json.dumps({"sample_rate_hz": SAMPLE_RATE_HZ, "stages": invalid, "frames": []}),
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
