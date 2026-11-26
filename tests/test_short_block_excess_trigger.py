"""Short-block excess trigger candidate (review item F): default off on both ingest paths.

Python reference here; the Rust port is held to it by the
``relative_trigger_short_block`` / ``peak_short_block_rms`` goldens in
``tests/test_attribution_trigger_cross_language_parity.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from minimappr.config import Settings
from minimappr.core.ingest_relative_trigger import (
    NoiseFloorRelativeTrigger,
    peak_short_block_rms,
    short_block_samples,
)
from minimappr.core.localization_overhaul_config import (
    DEFAULT_TRIGGER_SHORT_BLOCK_SECONDS,
    RUST_FORWARDED_LOCALIZATION_SETTING_KEYS,
    sidecar_environment_variable_name,
)

SAMPLE_RATE_HZ = 48_000
FRAME_SAMPLES = 960
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE_HZ
FLOOR_RMS = 2.5e-5
CHIP_SECONDS = 0.005
RUST_SOLVE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "minimappr-ingest-sidecar" / "src" / "localization_solve_config.rs"
SIDECAR_SHORT_BLOCK_KEYS = (
    "localization_trigger_short_block_excess_enabled",
    "localization_trigger_short_block_seconds",
)


def _relative_trigger() -> NoiseFloorRelativeTrigger:
    return NoiseFloorRelativeTrigger(Settings(trigger_noise_floor_relative_enabled=True).relative_trigger_config())


def _noise_frame(rng: np.random.Generator, channel_count: int = 4) -> np.ndarray:
    return (FLOOR_RMS * rng.standard_normal((channel_count, FRAME_SAMPLES))).astype(np.float32)


def _strongest_channel_rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64), axis=1)).max())


def test_block_tiling_covers_the_remainder_with_a_full_length_block() -> None:
    channel = np.zeros((1, 10), dtype=np.float32)
    channel[0, 9] = 2.0
    # 4-sample blocks: [0, 4), [4, 8) and the tail block [6, 10).
    assert peak_short_block_rms(channel, 4.0, 1.0) == pytest.approx(1.0)
    # A block no shorter than the frame is the strongest channel's whole-frame RMS.
    assert peak_short_block_rms(channel, 4.0, 5.0) == pytest.approx(np.sqrt(0.4))
    assert peak_short_block_rms(np.zeros((0, 0), dtype=np.float32), 48_000.0, 0.005) == 0.0
    # Half rounds away from zero, as in Rust (Python's round() would give 2).
    assert short_block_samples(1.0, 2.5) == 3


def test_chip_diluted_by_its_frame_is_admitted_only_with_short_blocks() -> None:
    rng = np.random.default_rng(5)
    frame_only, with_blocks = _relative_trigger(), _relative_trigger()
    for _ in range(int(10.0 / FRAME_SECONDS)):
        frame = _noise_frame(rng)
        assert not frame_only.observe("n", _strongest_channel_rms(frame), FRAME_SECONDS)
        assert not with_blocks.observe(
            "n", _strongest_channel_rms(frame), FRAME_SECONDS, peak_short_block_rms(frame, SAMPLE_RATE_HZ, CHIP_SECONDS)
        )
    chip_frame = _noise_frame(rng)
    chip_samples = int(CHIP_SECONDS * SAMPLE_RATE_HZ)
    chip_frame[:, -chip_samples:] += (FLOOR_RMS * 10 ** (11.0 / 20.0) * rng.standard_normal((4, chip_samples))).astype(np.float32)
    frame_rms = _strongest_channel_rms(chip_frame)
    assert not frame_only.observe("n", frame_rms, FRAME_SECONDS)
    assert with_blocks.observe("n", frame_rms, FRAME_SECONDS, peak_short_block_rms(chip_frame, SAMPLE_RATE_HZ, CHIP_SECONDS))
    # The floor follows the frame level either way.
    assert with_blocks.floor_rms("n") == pytest.approx(frame_only.floor_rms("n"), rel=1e-12)


def test_without_a_block_level_the_trigger_is_unchanged() -> None:
    rng = np.random.default_rng(9)
    levels = FLOOR_RMS * np.exp(rng.normal(scale=0.3, size=800))
    reference, candidate = _relative_trigger(), _relative_trigger()
    for level in levels:
        assert reference.observe("n", float(level), FRAME_SECONDS) == candidate.observe("n", float(level), FRAME_SECONDS, None)
    assert reference.floor_rms("n") == candidate.floor_rms("n")


def test_short_block_candidates_default_off() -> None:
    settings = Settings()
    assert settings.trigger_short_block_excess_enabled is False
    assert settings.localization_trigger_short_block_excess_enabled is False
    assert settings.omni_scan_clearest_channel_gate_enabled is False
    assert settings.trigger_short_block_seconds == DEFAULT_TRIGGER_SHORT_BLOCK_SECONDS
    assert settings.localization_trigger_short_block_seconds == DEFAULT_TRIGGER_SHORT_BLOCK_SECONDS


def test_short_block_seconds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="trigger_short_block_seconds"):
        Settings(trigger_short_block_seconds=0.0)


@pytest.mark.parametrize("setting_key", SIDECAR_SHORT_BLOCK_KEYS)
def test_sidecar_short_block_defaults_match_the_clap_literals(setting_key: str) -> None:
    match = re.search(
        rf'env = "{sidecar_environment_variable_name(setting_key)}",\s*default_value_t = ([A-Za-z0-9.eE+-]+)',
        RUST_SOLVE_CONFIG_PATH.read_text(),
    )
    assert match is not None, setting_key
    python_default = getattr(Settings(), setting_key)
    if isinstance(python_default, bool):
        assert match.group(1) == str(python_default).lower()
    else:
        assert float(match.group(1)) == pytest.approx(float(python_default))
    assert setting_key in RUST_FORWARDED_LOCALIZATION_SETTING_KEYS
