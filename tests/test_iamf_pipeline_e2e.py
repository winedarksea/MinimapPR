"""End-to-end test for the IAMF pipeline with Python ingest path.

Exercises the full IamfPipeline.run() with a MultiSensorBuffer containing
synthetic 4-channel audio, verifying:
  1. Audio extraction from the Python buffer
  2. A-to-B ambisonics conversion
  3. IAMF bitstream encoding (ipcm)
  4. AmbiX WAV output validity
  5. Coverage diagnostics
  6. Ambisonics-only path (include_iamf=False)
"""

from __future__ import annotations

import math
import tempfile
import time
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.capture_session import CaptureSessionRecord, CaptureState
from minimappr.core.iamf_pipeline import IamfPipeline, OUTPUT_RATE_HZ

SAMPLE_RATE = 16_000
DURATION_S = 2.0
N_CHANNELS = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sine(freq_hz: float, duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return (np.sin(2 * math.pi * freq_hz * t) * 0.3).astype(np.float32)


def _synthetic_4ch(duration_s: float = DURATION_S, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a (4, N) float32 array with coherent 440 Hz source."""
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    channels = np.zeros((4, n), dtype=np.float32)
    for i in range(4):
        delay = 0.00005 * i
        phase = 2 * math.pi * 440.0 * (t - delay)
        channels[i] = (np.sin(phase) * 0.25).astype(np.float32)
    return channels


async def _populate_buffer(
    buffer: MultiSensorBuffer,
    sensor_ids: list[str],
    start_ns: int,
    duration_s: float = DURATION_S,
    sr: int = SAMPLE_RATE,
) -> None:
    """Fill the buffer with synthetic 4-channel audio in 80 ms frames."""
    frame_duration_s = 0.080
    frame_samples = int(sr * frame_duration_s)
    n_frames = int(duration_s / frame_duration_s)
    current_ns = start_ns

    for frame_idx in range(n_frames):
        t_offset = frame_idx * frame_duration_s
        n = frame_samples if frame_idx < n_frames - 1 else int(sr * duration_s) - frame_idx * frame_samples
        t = np.arange(n) / sr + t_offset

        for ch_idx, sensor_id in enumerate(sensor_ids):
            delay = 0.00005 * ch_idx
            phase = 2 * math.pi * 440.0 * (t - delay)
            samples = (np.sin(phase) * 0.25).astype(np.float32)

            await buffer.append(
                sensor_id=sensor_id,
                sample_rate_hz=sr,
                start_time_ns=current_ns,
                samples=samples,
            )

        current_ns += int(frame_duration_s * 1_000_000_000)


def _make_session_record(
    session_id: str,
    work_dir: Path,
    start_ns: int,
    end_ns: int,
    sensor_ids: list[str],
    *,
    include_iamf: bool = True,
    include_video: bool = False,
) -> CaptureSessionRecord:
    return CaptureSessionRecord(
        session_id=session_id,
        state=CaptureState.PROCESSING,
        stream_key="test_node",
        range_lease_id=None,
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        first_frame_pts_ns=start_ns,
        work_dir=work_dir,
        video_path=None,
        iamf_path=None,
        youtube_path=None,
        error=None,
        use_python_ingest=True,
        channel_sensor_ids=sensor_ids,
        include_iamf=include_iamf,
        include_video=include_video,
    )


class _StubStorage:
    """Minimal storage stub that satisfies IamfPipeline's DB queries."""

    async def get_tracks_in_range(self, start_ns: int, end_ns: int):
        return []

    async def get_detections_for_track(self, track_id: str):
        return []

    async def upsert_capture_session(self, record):
        pass

    async def insert_large_artifact_for_session(self, *args, **kwargs):
        pass


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIamfPipelineE2E:
    """Full pipeline E2E with Python ingest buffer."""

    @pytest.fixture
    def work_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "capture_work"
        d.mkdir()
        return d

    @pytest.fixture
    def artifacts_dir(self, tmp_path: Path) -> Path:
        """Override the module-level ARTIFACTS_DIR for the test."""
        d = tmp_path / "artifacts"
        d.mkdir()
        return d

    @pytest.mark.asyncio
    async def test_full_pipeline_with_python_buffer(self, work_dir: Path, artifacts_dir: Path):
        """Run IamfPipeline.run() end-to-end with synthetic audio in a MultiSensorBuffer."""
        sensor_ids = [f"test_node:ch{i}" for i in range(N_CHANNELS)]
        buffer = MultiSensorBuffer(max_duration_seconds=30.0)

        start_ns = time.time_ns()
        await _populate_buffer(buffer, sensor_ids, start_ns)
        end_ns = start_ns + int(DURATION_S * 1_000_000_000)

        # Pin the buffer so extraction works.
        buffer.pin("test_session", start_ns)

        record = _make_session_record(
            session_id="e2e_test_session",
            work_dir=work_dir,
            start_ns=start_ns,
            end_ns=end_ns,
            sensor_ids=sensor_ids,
        )

        # Patch ARTIFACTS_DIR so the pipeline writes to our temp dir.
        import minimappr.core.iamf_pipeline as iamf_mod
        original_artifacts_dir = iamf_mod.ARTIFACTS_DIR
        iamf_mod.ARTIFACTS_DIR = artifacts_dir

        storage = _StubStorage()
        pipeline = IamfPipeline(
            sidecar_url=None,
            db_storage=storage,
            multi_sensor_buffer=buffer,
        )

        try:
            await pipeline.run(record)
        finally:
            iamf_mod.ARTIFACTS_DIR = original_artifacts_dir
            buffer.unpin("test_session")

        # Verify the pipeline completed successfully.
        assert record.error is None, f"Pipeline failed: {record.error}"

        # Verify ambix.wav was written (intermediate, may be cleaned up).
        # The final artifact should be in the artifacts dir.
        iamf_files = list(artifacts_dir.glob("*_audio.iamf"))
        assert len(iamf_files) >= 1, f"Expected IAMF file in {artifacts_dir}, found: {list(artifacts_dir.iterdir())}"

        # Verify the IAMF file is non-empty.
        iamf_path = iamf_files[0]
        assert iamf_path.stat().st_size > 0, "IAMF file is empty"

    @pytest.mark.asyncio
    async def test_ambisonics_only_pipeline(self, work_dir: Path, artifacts_dir: Path):
        """When include_iamf=False, the pipeline should produce ambisonics WAV only."""
        sensor_ids = [f"test_node:ch{i}" for i in range(N_CHANNELS)]
        buffer = MultiSensorBuffer(max_duration_seconds=30.0)

        start_ns = time.time_ns()
        await _populate_buffer(buffer, sensor_ids, start_ns)
        end_ns = start_ns + int(DURATION_S * 1_000_000_000)

        buffer.pin("test_session_ambix", start_ns)

        record = _make_session_record(
            session_id="e2e_ambix_only",
            work_dir=work_dir,
            start_ns=start_ns,
            end_ns=end_ns,
            sensor_ids=sensor_ids,
            include_iamf=False,
        )

        import minimappr.core.iamf_pipeline as iamf_mod
        original_artifacts_dir = iamf_mod.ARTIFACTS_DIR
        iamf_mod.ARTIFACTS_DIR = artifacts_dir

        storage = _StubStorage()
        pipeline = IamfPipeline(
            sidecar_url=None,
            db_storage=storage,
            multi_sensor_buffer=buffer,
        )

        try:
            await pipeline.run(record)
        finally:
            iamf_mod.ARTIFACTS_DIR = original_artifacts_dir
            buffer.unpin("test_session_ambix")

        assert record.error is None, f"Ambisonics-only pipeline failed: {record.error}"

        # Verify ambix WAV was written to artifacts dir.
        ambix_files = list(artifacts_dir.glob("*_ambix.wav"))
        assert len(ambix_files) >= 1, f"Expected ambix WAV in {artifacts_dir}, found: {list(artifacts_dir.iterdir())}"

        # Verify the WAV is valid.
        ambix_path = ambix_files[0]
        with wave.open(str(ambix_path), "rb") as w:
            assert w.getnchannels() == 4, f"Expected 4 channels, got {w.getnchannels()}"
            assert w.getsampwidth() == 2, "Expected 16-bit PCM"
            # Sample rate should be OUTPUT_RATE_HZ (48000) after resampling.
            assert w.getframerate() == OUTPUT_RATE_HZ, f"Expected {OUTPUT_RATE_HZ} Hz, got {w.getframerate()}"

        # No IAMF file should be produced.
        iamf_files = list(artifacts_dir.glob("*_audio.iamf"))
        assert len(iamf_files) == 0, "IAMF file should not be produced when include_iamf=False"

    @pytest.mark.asyncio
    async def test_extract_range_coverage_diagnostics(self):
        """Verify extract_range returns coverage diagnostics with no gaps for synthetic input."""
        sensor_ids = [f"test_node:ch{i}" for i in range(N_CHANNELS)]
        buffer = MultiSensorBuffer(max_duration_seconds=30.0)

        start_ns = time.time_ns()
        await _populate_buffer(buffer, sensor_ids, start_ns)
        end_ns = start_ns + int(DURATION_S * 1_000_000_000)

        buffer.pin("test_diag", start_ns)

        channels, sr, diag = await buffer.extract_range(sensor_ids, start_ns, end_ns)

        buffer.unpin("test_diag")

        # Verify basic shape.
        assert channels.shape[0] == N_CHANNELS, f"Expected {N_CHANNELS} channels, got {channels.shape[0]}"
        assert channels.shape[1] > 0, "Expected non-zero samples"

        # Verify coverage diagnostics.
        assert "channel_coverage_ratios" in diag, "Missing channel_coverage_ratios in sync_diag"
        assert "coverage_warnings" in diag, "Missing coverage_warnings in sync_diag"

        # Synthetic input should have near-perfect coverage.
        for i, ratio in enumerate(diag["channel_coverage_ratios"]):
            assert ratio >= 0.99, f"Channel {i} coverage ratio {ratio:.2%} is below 99%"

        # No warnings expected for complete synthetic input.
        assert len(diag["coverage_warnings"]) == 0, f"Unexpected warnings: {diag['coverage_warnings']}"

    @pytest.mark.asyncio
    async def test_extract_range_missing_channel_warning(self):
        """Verify extract_range warns when a channel has <50% coverage."""
        sensor_ids = [f"test_node:ch{i}" for i in range(N_CHANNELS)]
        buffer = MultiSensorBuffer(max_duration_seconds=30.0)

        start_ns = time.time_ns()
        # Only populate 2 of 4 channels.
        for ch_idx in range(2):
            sensor_id = sensor_ids[ch_idx]
            samples = _sine(440.0, DURATION_S)
            await buffer.append(
                sensor_id=sensor_id,
                sample_rate_hz=SAMPLE_RATE,
                start_time_ns=start_ns,
                samples=samples,
            )

        end_ns = start_ns + int(DURATION_S * 1_000_000_000)

        channels, sr, diag = await buffer.extract_range(sensor_ids, start_ns, end_ns)

        # Missing channels should produce warnings.
        assert len(diag["coverage_warnings"]) >= 2, (
            f"Expected warnings for 2 missing channels, got {len(diag['coverage_warnings'])}: {diag['coverage_warnings']}"
        )

        # Missing channels should have 0% coverage.
        for i in [2, 3]:
            assert diag["channel_coverage_ratios"][i] < 0.5, (
                f"Channel {i} should have <50% coverage but has {diag['channel_coverage_ratios'][i]:.2%}"
            )
