"""In-memory WAV conditioning for human-listening endpoints."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from minimappr.audio_processing.levels import AudioLevelReport, apply_level_profile
from minimappr.audio_processing.profiles import (
    LISTENING_PROFILE_NAME,
    load_audio_processing_configuration,
)


def listening_wav_bytes(
    path: Path,
    processing_config_path: Path | str | None = None,
) -> tuple[bytes, AudioLevelReport]:
    with wave.open(str(path), "rb") as source:
        channel_count = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate_hz = source.getframerate()
        frame_count = source.getnframes()
        raw = source.readframes(frame_count)
    if sample_width != 2:
        raise ValueError("Listening conditioning currently requires PCM16 WAV input")
    interleaved = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channel_count <= 0 or interleaved.size % channel_count:
        raise ValueError("WAV payload does not align to its channel count")
    channels_first = interleaved.reshape(-1, channel_count).T
    profile = load_audio_processing_configuration(processing_config_path).profile(
        LISTENING_PROFILE_NAME
    )
    conditioned, report = apply_level_profile(channels_first, profile)
    pcm = (np.clip(conditioned.T, -1.0, 1.0) * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as destination:
        destination.setnchannels(channel_count)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate_hz)
        destination.writeframes(pcm.tobytes())
    return output.getvalue(), report


def level_report_headers(report: AudioLevelReport) -> dict[str, str]:
    return {
        "X-Minimappr-Audio-Level-Profile": "listening",
        "X-Minimappr-Applied-Gain-Db": f"{report.applied_gain_db:.3f}",
        "X-Minimappr-Input-Rms-Dbfs": f"{report.input_rms_dbfs:.3f}",
        "X-Minimappr-Input-Peak-Dbfs": f"{report.input_peak_dbfs:.3f}",
        "X-Minimappr-Output-Rms-Dbfs": f"{report.output_rms_dbfs:.3f}",
        "X-Minimappr-Output-Peak-Dbfs": f"{report.output_peak_dbfs:.3f}",
    }
