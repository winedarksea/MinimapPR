"""Scoped gain, immutable profile, and listening-WAV contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from minimappr.audio_processing.levels import apply_bounded_rms_gain, apply_listening_level
from minimappr.audio_processing.profiles import (
    DEFAULT_AUDIO_PROCESSING_CONFIGURATION,
    LISTENING_PROFILE_NAME,
    profile_fingerprint,
)
from minimappr.audio_processing.wav_serving import listening_wav_bytes
from minimappr.audio_processing.chain import (
    NodePreprocessorFactory,
    create_localization_preprocessor,
)
from minimappr.audio_processing.stages import HighpassFilterStage
from minimappr.config import Settings
from minimappr.utils.audio import read_wav_mono, write_wav_mono


def test_listening_gain_reaches_target_without_exceeding_peak_ceiling() -> None:
    time = np.arange(16_000, dtype=np.float32) / 16_000.0
    quiet_tone = 0.001 * np.sin(2.0 * np.pi * 440.0 * time)
    output, report = apply_listening_level(quiet_tone)

    assert report.applied_gain_db == pytest.approx(24.0, abs=0.01)
    assert float(np.max(np.abs(output))) <= 10.0 ** (-1.0 / 20.0) + 1e-6
    assert report.clipping_risk_sample_count == 0


def test_transient_limits_listening_gain_before_clipping() -> None:
    signal = np.full(16_000, 0.001, dtype=np.float32)
    signal[8000] = 0.5
    output, report = apply_listening_level(signal)

    assert report.applied_gain_db < 6.0
    assert float(np.max(np.abs(output))) <= 10.0 ** (-1.0 / 20.0) + 1e-6


def test_multichannel_listening_uses_one_common_scalar() -> None:
    channels = np.vstack(
        (np.linspace(-0.01, 0.01, 2048), np.linspace(-0.005, 0.005, 2048))
    ).astype(np.float32)
    output, _ = apply_listening_level(channels)

    nonzero = np.abs(channels[1]) > 1e-6
    ratio_zero = output[0, nonzero] / channels[0, nonzero]
    ratio_one = output[1, nonzero] / channels[1, nonzero]
    np.testing.assert_allclose(ratio_zero, ratio_one, rtol=1e-5, atol=1e-5)


def test_empty_and_nonfinite_level_inputs() -> None:
    empty, report = apply_listening_level(np.zeros(0, dtype=np.float32))
    assert empty.size == 0
    assert report.applied_gain_db == 0.0
    with pytest.raises(ValueError, match="finite"):
        apply_bounded_rms_gain(
            np.array([np.nan], dtype=np.float32),
            target_rms=0.1,
            max_gain=2.0,
            peak_ceiling=0.9,
        )


def test_silence_remains_silent_and_dc_offset_is_removed() -> None:
    silence = np.zeros(2_048, dtype=np.float32)
    silent_output, silent_report = apply_listening_level(silence)
    np.testing.assert_array_equal(silent_output, silence)
    assert silent_report.clipping_risk_sample_count == 0

    dc_offset = np.full(2_048, 0.25, dtype=np.float32)
    centered_output, _ = apply_listening_level(dc_offset)
    assert float(np.max(np.abs(centered_output))) == pytest.approx(0.0, abs=1e-7)


def test_default_profiles_are_immutable_and_fingerprinted() -> None:
    profile = DEFAULT_AUDIO_PROCESSING_CONFIGURATION.profile(LISTENING_PROFILE_NAME)
    with pytest.raises(TypeError):
        profile.stages[0]["type"] = "gain"
    assert profile_fingerprint(profile) == profile_fingerprint(profile)


def test_digital_trim_lookup_supports_single_and_multichannel_sensor_ids() -> None:
    factory = NodePreprocessorFactory(Settings(ingest_gain_multiplier=1.0))
    factory.set_node_override(
        "node-a",
        {"stages": [{"type": "channel_gain", "db_by_channel": [3.0, -6.0]}]},
    )
    assert factory.fixed_gain_db_for_sensor("node-a") == pytest.approx(3.0)
    assert factory.fixed_gain_db_for_sensor("node-a:ch1") == pytest.approx(-6.0)


def test_listening_wav_is_in_memory_and_does_not_modify_canonical_file(tmp_path: Path) -> None:
    canonical_path = tmp_path / "canonical.wav"
    signal = np.sin(np.linspace(0.0, 20.0, 4096)).astype(np.float32) * 0.002
    write_wav_mono(canonical_path, signal, 16_000)
    before_hash = hashlib.sha256(canonical_path.read_bytes()).hexdigest()

    listening_bytes, report = listening_wav_bytes(canonical_path)

    assert listening_bytes.startswith(b"RIFF")
    assert report.applied_gain_db > 0.0
    assert hashlib.sha256(canonical_path.read_bytes()).hexdigest() == before_hash
    canonical, _ = read_wav_mono(canonical_path)
    assert float(np.max(np.abs(canonical))) < 0.01


# ---------------------------------------------------------------------------
# Localization band: the window filter is the band floor only; ceilings are
# per pair in the GCC-PHAT bin mask (core/localization_band_policy.py)
# ---------------------------------------------------------------------------


def _localization_stages(*, min_hz: float, max_hz: float) -> list:
    config = Settings(
        localization_band_min_hz=min_hz,
        localization_band_max_hz=max_hz,
    ).localization_config()
    preprocessor = create_localization_preprocessor(config)
    return [] if preprocessor is None else list(preprocessor.stages)


@pytest.mark.parametrize("max_hz", [0.0, 3500.0])
def test_localization_band_floor_builds_a_highpass_whatever_the_ceiling(max_hz: float) -> None:
    """An explicit ceiling is the intra-array GCC-PHAT ceiling; cross-node pairs
    must still see the wide band, so no window filter may cap it."""
    stages = _localization_stages(min_hz=300.0, max_hz=max_hz)

    assert len(stages) == 1
    assert isinstance(stages[0], HighpassFilterStage)
    assert stages[0].cutoff_hz == pytest.approx(300.0)


def test_fully_disabled_localization_band_builds_nothing() -> None:
    assert _localization_stages(min_hz=0.0, max_hz=0.0) == []


def test_default_localization_band_rejects_rumble_and_keeps_the_bird_band() -> None:
    """The shipped default: a 50 Hz floor, each array's own c / 2d ceiling
    (max 0) applied per pair in the PHAT mask, so the window filter keeps the
    3.4-7 kHz calls that cross-node pairs correlate over."""
    sample_rate_hz = 16_000
    settings = Settings()
    assert settings.localization_band_min_hz == pytest.approx(50.0)
    assert settings.localization_band_max_hz == pytest.approx(0.0)

    preprocessor = create_localization_preprocessor(settings.localization_config())
    assert preprocessor is not None

    t = np.linspace(0.0, 0.5, int(0.5 * sample_rate_hz), endpoint=False, dtype=np.float32)
    rumble = np.sin(2.0 * np.pi * 10.0 * t).astype(np.float32)
    in_band = np.sin(2.0 * np.pi * 1_000.0 * t).astype(np.float32)
    bird_call = np.sin(2.0 * np.pi * 6_000.0 * t).astype(np.float32)

    def _rms(signal: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.asarray(signal, dtype=np.float32) ** 2)))

    assert _rms(preprocessor.process(rumble, sample_rate_hz)) < 0.10 * _rms(rumble)
    assert _rms(preprocessor.process(in_band, sample_rate_hz)) > 0.70 * _rms(in_band)
    assert _rms(preprocessor.process(bird_call, sample_rate_hz)) > 0.70 * _rms(bird_call)
