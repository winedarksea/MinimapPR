"""Capture pipeline tests using synthetic audio input.

Tests are self-contained: they generate synthetic 4-channel PCM, exercise the
pure-Python DSP components (A-to-B conversion, spatial subtraction, loudness
measurement, and IAMF bitstream encoding via the Rust writer shim), and clean
up all temporary files on exit.

The Rust sidecar and ffmpeg are NOT required for these tests.  Components that
call external processes (VideoCapture, ffmpeg mux) are exercised via unit-level
mocks only.
"""

from __future__ import annotations

import asyncio
import math
import struct
import tempfile
import time
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from minimappr.core.ambi_atob import (
    SIRITH_MIC_POSITIONS_M,
    atob_foa,
    centroid_corrected_positions,
    encode_mono_to_bformat,
    _build_encoding_matrix,
)
from minimappr.core.capture_session import (
    CaptureSessionManager,
    CaptureStartRequest,
    CaptureState,
)
from minimappr.core.iamf_pipeline import (
    LoudnessMeasurement,
    TrackTrajectory,
    _measure_loudness,
    _conform_channels_to_sample_count,
    _subtract_object_slot,
    _subtract_objects,
    _build_positions_per_unit,
    _xyz_to_spherical,
    _write_wav,
    _write_wav_mono,
)
from minimappr.core.iamf_object_slot import IamfObjectSlot, select_iamf_object_slot

SAMPLE_RATE = 16_000


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sine(freq_hz: float, duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return (np.sin(2 * math.pi * freq_hz * t) * 0.3).astype(np.float32)


def _synthetic_4ch(duration_s: float = 1.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a (4, N) float32 array simulating a coherent 440 Hz source at
    different mic positions.  Each channel has slight level and phase offsets
    to simulate a real tetrahedral array.
    """
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    channels = np.zeros((4, n), dtype=np.float32)
    for i in range(4):
        # Small phase shift per channel simulates TDOA.
        delay = 0.00005 * i
        phase = 2 * math.pi * 440.0 * (t - delay)
        channels[i] = (np.sin(phase) * 0.25).astype(np.float32)
    return channels


# ── ambi_atob tests ───────────────────────────────────────────────────────────

class TestAmbiAtob:
    def test_encoding_matrix_shape(self):
        E = _build_encoding_matrix(SIRITH_MIC_POSITIONS_M)
        assert E.shape == (4, 4)

    def test_centroid_correction_zero_mean(self):
        corrected, centroid = centroid_corrected_positions(SIRITH_MIC_POSITIONS_M)
        mean = corrected.mean(axis=0)
        np.testing.assert_allclose(mean, 0.0, atol=1e-9)

    def test_centroid_not_at_mk3(self):
        # MK3 is at [0,0,0] but centroid is NOT [0,0,0] — verify correction.
        _, centroid = centroid_corrected_positions(SIRITH_MIC_POSITIONS_M)
        assert np.linalg.norm(centroid) > 1e-4

    def test_atob_output_shape(self):
        channels = _synthetic_4ch(duration_s=0.5)
        bformat = atob_foa(channels, SAMPLE_RATE)
        assert bformat.shape == (4, channels.shape[1])

    def test_atob_output_dtype(self):
        channels = _synthetic_4ch()
        bformat = atob_foa(channels, SAMPLE_RATE)
        assert bformat.dtype == np.float32

    def test_atob_silence_in_silence_out(self):
        silence = np.zeros((4, SAMPLE_RATE), dtype=np.float32)
        bformat = atob_foa(silence, SAMPLE_RATE)
        rms = float(np.sqrt(np.mean(bformat ** 2)))
        assert rms < 1e-5, f"silence in → near-silence out expected, got RMS {rms}"

    def test_atob_energy_preserved(self):
        channels = _synthetic_4ch()
        bformat = atob_foa(channels, SAMPLE_RATE)
        # B-format W channel should have non-negligible energy.
        w_rms = float(np.sqrt(np.mean(bformat[0] ** 2)))
        assert w_rms > 0.01, f"W-channel RMS too low: {w_rms}"

    def test_atob_values_bounded(self):
        channels = _synthetic_4ch()
        bformat = atob_foa(channels, SAMPLE_RATE)
        assert bformat.min() >= -1.0
        assert bformat.max() <= 1.0

    def test_encode_mono_to_bformat_shape(self):
        mono = _sine(440.0, 0.5)
        bfmt = encode_mono_to_bformat(mono, (1.0, 0.0, 0.0))
        assert bfmt.shape == (4, len(mono))

    def test_encode_mono_w_equals_signal_scaled(self):
        mono = _sine(1000.0, 0.1)
        bfmt = encode_mono_to_bformat(mono, (1.0, 0.0, 0.0))
        expected_w = mono / math.sqrt(2)
        np.testing.assert_allclose(bfmt[0], expected_w, atol=1e-5)


# ── iamf_pipeline DSP tests ───────────────────────────────────────────────────

class TestSpatialSubtraction:
    def test_subtract_reduces_w_energy(self):
        channels = _synthetic_4ch(duration_s=0.5)
        bed_full = atob_foa(channels, SAMPLE_RATE)
        mono = _sine(440.0, 0.5)
        n = bed_full.shape[1]
        mono_padded = np.zeros(n, dtype=np.float32)
        mono_padded[:len(mono)] = mono

        traj = TrackTrajectory(
            track_id="trk-001",
            waypoints=[(0, (1.0, 0.0, 0.0)), (n - 1, (1.0, 0.0, 0.0))],
        )
        object_tracks = {"trk-001": mono_padded}
        bed_clean = _subtract_objects(bed_full, object_tracks, [traj], n)

        assert bed_clean.shape == bed_full.shape
        assert bed_clean.dtype == np.float32
        assert bed_clean.min() >= -1.0
        assert bed_clean.max() <= 1.0


class TestLoudnessMeasurement:
    def test_silence_gives_low_lufs(self):
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        result = _measure_loudness(silence, SAMPLE_RATE)
        assert result.integrated_lufs < -80.0

    def test_full_scale_sine_lufs_range(self):
        sig = _sine(1000.0, 3.0)
        result = _measure_loudness(sig, SAMPLE_RATE)
        # Full-scale sine ≈ −3 dBFS → roughly −3 LUFS (before K-weighting).
        assert result.integrated_lufs > -40.0
        assert result.integrated_lufs < 0.0

    def test_true_peak_full_scale(self):
        sig = np.ones(SAMPLE_RATE, dtype=np.float32)
        result = _measure_loudness(sig, SAMPLE_RATE)
        # True peak of a DC offset at 1.0 should be ≈ 0 dBFS.
        assert result.true_peak_dbfs > -3.0

    def test_returns_loudness_measurement_type(self):
        sig = _sine(440.0, 1.0)
        result = _measure_loudness(sig, SAMPLE_RATE)
        assert isinstance(result, LoudnessMeasurement)
        assert isinstance(result.integrated_lufs, float)
        assert isinstance(result.true_peak_dbfs, float)


class TestPositionMetadata:
    def test_build_positions_per_unit_length(self):
        n_samples = 4096
        spf = 512
        traj = TrackTrajectory(
            track_id="t1",
            waypoints=[(0, (1.0, 0.0, 0.0)), (n_samples - 1, (0.0, 1.0, 0.0))],
        )
        result = _build_positions_per_unit([traj], n_samples, spf)
        expected_frames = math.ceil(n_samples / spf)
        assert len(result) == expected_frames

    def test_xyz_to_spherical_x_axis(self):
        az, el, dist = _xyz_to_spherical((1.0, 0.0, 0.0))
        assert abs(az) < 1e-6
        assert abs(el) < 1e-6
        assert abs(dist - 1.0) < 1e-6

    def test_xyz_to_spherical_elevation(self):
        az, el, dist = _xyz_to_spherical((0.0, 0.0, 1.0))
        assert abs(el - 90.0) < 1e-5

    def test_xyz_to_spherical_origin(self):
        az, el, dist = _xyz_to_spherical((0.0, 0.0, 0.0))
        assert dist == 0.0


class TestObjectSlotSelector:
    def test_rejects_low_quality_tracks(self):
        n = 2048
        traj = TrackTrajectory(
            track_id="low",
            waypoints=[(0, (1.0, 0.0, 0.0)), (n - 1, (1.0, 0.0, 0.0))],
            tqi=0.2,
            label_confidence=0.9,
            localization_confidence=0.9,
        )
        slot = select_iamf_object_slot(
            [traj],
            {"low": np.ones(n, dtype=np.float32) * 0.2},
            n,
            512,
            sample_rate_hz=SAMPLE_RATE,
        )
        assert slot is None

    def test_selects_louder_higher_confidence_candidate(self):
        n = 4096
        quiet = TrackTrajectory(
            track_id="quiet",
            waypoints=[(0, (1.0, 0.0, 0.0)), (n - 1, (1.0, 0.0, 0.0))],
            tqi=0.5,
            label_confidence=0.5,
            localization_confidence=0.5,
        )
        loud = TrackTrajectory(
            track_id="loud",
            waypoints=[(0, (0.0, 1.0, 0.0)), (n - 1, (0.0, 1.0, 0.0))],
            tqi=0.9,
            label_confidence=0.9,
            localization_confidence=0.9,
        )
        slot = select_iamf_object_slot(
            [quiet, loud],
            {
                "quiet": np.ones(n, dtype=np.float32) * 0.1,
                "loud": np.ones(n, dtype=np.float32) * 0.5,
            },
            n,
            512,
            sample_rate_hz=SAMPLE_RATE,
        )
        assert isinstance(slot, IamfObjectSlot)
        assert slot.track_id == "loud"
        assert slot.positions_per_unit[0][0]["azimuth_deg"] == pytest.approx(90.0)

    def test_stable_winner_survives_short_challenger(self):
        n = 4096
        incumbent = TrackTrajectory(
            track_id="incumbent",
            waypoints=[(0, (1.0, 0.0, 0.0)), (n - 1, (1.0, 0.0, 0.0))],
            tqi=0.7,
            label_confidence=0.7,
            localization_confidence=0.7,
        )
        challenger = TrackTrajectory(
            track_id="challenger",
            waypoints=[(512, (0.0, 1.0, 0.0)), (1024, (0.0, 1.0, 0.0))],
            tqi=1.0,
            label_confidence=1.0,
            localization_confidence=1.0,
        )
        slot = select_iamf_object_slot(
            [incumbent, challenger],
            {
                "incumbent": np.ones(n, dtype=np.float32) * 0.2,
                "challenger": np.ones(n, dtype=np.float32) * 0.8,
            },
            n,
            512,
            sample_rate_hz=SAMPLE_RATE,
        )
        assert slot is not None
        assert slot.track_id == "incumbent"

    def test_strong_challenger_takes_over_after_hold_time(self):
        n = 16_384
        incumbent = TrackTrajectory(
            track_id="incumbent",
            waypoints=[(0, (1.0, 0.0, 0.0)), (n - 1, (1.0, 0.0, 0.0))],
            tqi=0.7,
            label_confidence=0.7,
            localization_confidence=0.7,
        )
        challenger = TrackTrajectory(
            track_id="challenger",
            waypoints=[(512, (0.0, 1.0, 0.0)), (n - 1, (0.0, 1.0, 0.0))],
            tqi=1.0,
            label_confidence=1.0,
            localization_confidence=1.0,
        )
        slot = select_iamf_object_slot(
            [incumbent, challenger],
            {
                "incumbent": np.ones(n, dtype=np.float32) * 0.2,
                "challenger": np.ones(n, dtype=np.float32) * 0.8,
            },
            n,
            512,
            sample_rate_hz=SAMPLE_RATE,
        )
        assert slot is not None
        assert len(slot.active_ranges) == 2
        assert slot.handoff_gap_ranges
        assert slot.positions_per_unit[-1][0]["azimuth_deg"] == pytest.approx(90.0)

    def test_handoff_gap_silences_object_and_omits_position_metadata(self):
        n = 16_384
        incumbent = TrackTrajectory(
            track_id="incumbent",
            waypoints=[(0, (1.0, 0.0, 0.0)), (n - 1, (1.0, 0.0, 0.0))],
            tqi=0.7,
            label_confidence=0.7,
            localization_confidence=0.7,
        )
        challenger = TrackTrajectory(
            track_id="challenger",
            waypoints=[(512, (0.0, 1.0, 0.0)), (n - 1, (0.0, 1.0, 0.0))],
            tqi=1.0,
            label_confidence=1.0,
            localization_confidence=1.0,
        )

        slot = select_iamf_object_slot(
            [incumbent, challenger],
            {
                "incumbent": np.ones(n, dtype=np.float32) * 0.2,
                "challenger": np.ones(n, dtype=np.float32) * 0.8,
            },
            n,
            512,
            sample_rate_hz=SAMPLE_RATE,
            challenger_hold_units=2,
            handoff_gap_seconds=0.25,
        )

        assert slot is not None
        assert slot.handoff_gap_ranges
        gap_start, gap_end = slot.handoff_gap_ranges[0]
        assert np.max(np.abs(slot.samples[gap_start:gap_end])) == 0.0
        for unit_idx in range(gap_start // 512, math.ceil(gap_end / 512)):
            assert slot.positions_per_unit[unit_idx] == {}
            assert slot.unit_track_ids[unit_idx] is None

    def test_subtract_object_slot_leaves_unselected_source_in_bed(self):
        n = 1024
        selected = _sine(440.0, n / SAMPLE_RATE)
        unselected = _sine(880.0, n / SAMPLE_RATE)
        selected_bed = encode_mono_to_bformat(selected, (1.0, 0.0, 0.0))
        unselected_bed = encode_mono_to_bformat(unselected, (0.0, 1.0, 0.0))
        bed_full = selected_bed + unselected_bed
        slot = IamfObjectSlot(
            slot_id=0,
            track_id="selected",
            samples=selected,
            positions_per_unit=[
                {
                    0: {
                        "azimuth_deg": 0.0,
                        "elevation_deg": 0.0,
                        "distance_norm": 0.2,
                        "end_azimuth_deg": 0.0,
                        "end_elevation_deg": 0.0,
                        "end_distance_norm": 0.2,
                    }
                },
                {
                    0: {
                        "azimuth_deg": 0.0,
                        "elevation_deg": 0.0,
                        "distance_norm": 0.2,
                        "end_azimuth_deg": 0.0,
                        "end_elevation_deg": 0.0,
                        "end_distance_norm": 0.2,
                    }
                },
            ],
            unit_track_ids=["selected", "selected"],
            active_ranges=[(0, n)],
            handoff_gap_ranges=[],
            score=1.0,
        )

        cleaned = _subtract_object_slot(bed_full, slot, 512, n)

        assert np.sqrt(np.mean((cleaned - unselected_bed) ** 2)) < 1e-6

    def test_subtract_object_slot_skips_handoff_gap(self):
        n = 1024
        selected = _sine(440.0, n / SAMPLE_RATE)
        bed_full = encode_mono_to_bformat(selected, (1.0, 0.0, 0.0))
        slot = IamfObjectSlot(
            slot_id=0,
            track_id="selected",
            samples=selected.copy(),
            positions_per_unit=[
                {
                    0: {
                        "azimuth_deg": 0.0,
                        "elevation_deg": 0.0,
                        "distance_norm": 0.2,
                    }
                },
                {},
            ],
            unit_track_ids=["selected", None],
            active_ranges=[(0, 512)],
            handoff_gap_ranges=[(512, n)],
            score=1.0,
        )

        cleaned = _subtract_object_slot(bed_full, slot, 512, n)

        assert np.sqrt(np.mean(cleaned[:, :512] ** 2)) < 1e-6
        assert np.sqrt(np.mean((cleaned[:, 512:] - bed_full[:, 512:]) ** 2)) < 1e-6


class TestDurationConformance:
    def test_conform_channels_to_sample_count_exact_target(self):
        channels = np.stack([
            _sine(440.0, 0.2),
            _sine(550.0, 0.2),
        ])

        conformed = _conform_channels_to_sample_count(channels, channels.shape[1] + 37)

        assert conformed.shape == (2, channels.shape[1] + 37)
        assert np.all(np.isfinite(conformed))


# ── WAV I/O helpers ───────────────────────────────────────────────────────────

class TestWavHelpers:
    def test_write_and_read_wav(self, tmp_path):
        channels = _synthetic_4ch(duration_s=0.25)
        out = tmp_path / "test.wav"
        _write_wav(out, channels, SAMPLE_RATE)
        assert out.exists()
        assert out.stat().st_size > 0

        with wave.open(str(out), "rb") as w:
            assert w.getnchannels() == 4
            assert w.getframerate() == SAMPLE_RATE
            assert w.getsampwidth() == 2

    def test_write_mono_wav(self, tmp_path):
        mono = _sine(440.0, 0.25)
        out = tmp_path / "mono.wav"
        _write_wav_mono(out, mono, SAMPLE_RATE)
        with wave.open(str(out), "rb") as w:
            assert w.getnchannels() == 1


# ── CaptureSessionManager unit tests ─────────────────────────────────────────

class TestCaptureSessionManager:
    @pytest.mark.asyncio
    async def test_start_failed_on_bad_sidecar(self, tmp_path):
        """If the Rust sidecar is unreachable, start() returns FAILED."""
        manager = CaptureSessionManager()
        req = CaptureStartRequest(
            stream_key="test-stream",
            sidecar_url="http://127.0.0.1:19999",  # nothing listening here
            work_dir=tmp_path,
            max_duration_s=30.0,
        )
        record = await manager.start(req)
        assert record.state == CaptureState.FAILED
        assert record.error is not None

    @pytest.mark.asyncio
    async def test_get_returns_none_for_unknown(self):
        manager = CaptureSessionManager()
        assert manager.get("nonexistent-session-id") is None

    @pytest.mark.asyncio
    async def test_stop_raises_key_error_for_unknown(self):
        manager = CaptureSessionManager()
        with pytest.raises(KeyError):
            await manager.stop("nonexistent-id", "http://localhost:8081")

    @pytest.mark.asyncio
    async def test_successful_start_and_stop(self, tmp_path):
        """Full start → stop cycle with mocked sidecar and VideoCapture."""
        manager = CaptureSessionManager()

        # Mock the range-lease HTTP calls and VideoCapture.
        fake_lease = {
            "lease_id": "srl-test-lease-001",
            "start_ns": time.time_ns(),
            "end_ns": time.time_ns() + 60_000_000_000,
            "heartbeat_deadline_ns": time.time_ns() + 30_000_000_000,
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=fake_lease)

        from minimappr.core.video_capture import VideoCaptureResult

        fake_capture_result = VideoCaptureResult(
            output_path=tmp_path / "output_raw.mp4",
            first_frame_pts_ns=time.time_ns(),
            process_start_ns=time.time_ns(),
        )

        with (
            patch(
                "minimappr.core.capture_session.httpx.AsyncClient.post",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch(
                "minimappr.core.capture_session.httpx.AsyncClient.delete",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch(
                "minimappr.core.video_capture.VideoCapture.start",
                new_callable=AsyncMock,
            ),
            patch(
                "minimappr.core.video_capture.VideoCapture.stop",
                new_callable=AsyncMock,
                return_value=fake_capture_result,
            ),
        ):
            req = CaptureStartRequest(
                stream_key="mic-array-1",
                sidecar_url="http://localhost:8081",
                work_dir=tmp_path,
                max_duration_s=30.0,
            )
            record = await manager.start(req)
            assert record.state == CaptureState.RECORDING
            assert record.range_lease_id == "srl-test-lease-001"

            stop_record = await manager.stop(record.session_id, "http://localhost:8081")
            assert stop_record.state == CaptureState.PROCESSING
            assert stop_record.end_time_ns is not None

    @pytest.mark.asyncio
    async def test_final_state_update_callback_runs_after_completion(self, tmp_path):
        """The durable row must see COMPLETED/FAILED, not only PROCESSING."""
        manager = CaptureSessionManager()
        persisted_states: list[CaptureState] = []

        async def persist(record):
            persisted_states.append(record.state)

        manager.set_session_update_callback(persist)

        fake_lease = {
            "lease_id": "srl-test-lease-002",
            "start_ns": time.time_ns(),
        }
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=fake_lease)

        from minimappr.core.video_capture import VideoCaptureResult

        fake_capture_result = VideoCaptureResult(
            output_path=tmp_path / "output_raw.mp4",
            first_frame_pts_ns=time.time_ns(),
            process_start_ns=time.time_ns(),
        )

        with (
            patch(
                "minimappr.core.capture_session.httpx.AsyncClient.post",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch(
                "minimappr.core.capture_session.httpx.AsyncClient.delete",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch(
                "minimappr.core.video_capture.VideoCapture.start",
                new_callable=AsyncMock,
            ),
            patch(
                "minimappr.core.video_capture.VideoCapture.stop",
                new_callable=AsyncMock,
                return_value=fake_capture_result,
            ),
        ):
            req = CaptureStartRequest(
                stream_key="mic-array-1",
                sidecar_url="http://localhost:8081",
                work_dir=tmp_path,
                max_duration_s=30.0,
            )
            record = await manager.start(req)
            await manager.stop(record.session_id, "http://localhost:8081")

            for _ in range(20):
                if persisted_states:
                    break
                await asyncio.sleep(0.01)

        assert persisted_states == [CaptureState.COMPLETED]


# ── Round-trip: A-to-B → subtract → measure → write ─────────────────────────

class TestCapturePipelineRoundTrip:
    """Integration-level test exercising the full pure-Python DSP path."""

    def test_full_dsp_path_produces_clean_bed_wav(self, tmp_path):
        duration_s = 0.5
        channels = _synthetic_4ch(duration_s=duration_s)
        n = channels.shape[1]

        # Step 1: A-to-B.
        bed_full = atob_foa(channels, SAMPLE_RATE)
        assert bed_full.shape == (4, n)

        # Step 2: Synthesise one object track.
        obj_mono = _sine(800.0, duration_s)[:n]
        traj = TrackTrajectory(
            track_id="bird-001",
            waypoints=[(0, (2.0, 0.0, 0.0)), (n - 1, (2.5, 0.5, 0.0))],
        )

        # Step 3: Spatial subtraction.
        bed_clean = _subtract_objects(bed_full, {"bird-001": obj_mono}, [traj], n)
        assert bed_clean.shape == (4, n)
        assert np.all(np.isfinite(bed_clean))

        # Step 4: Loudness measurement on W channel.
        lm = _measure_loudness(bed_clean[0], SAMPLE_RATE)
        assert np.isfinite(lm.integrated_lufs)
        assert np.isfinite(lm.true_peak_dbfs)

        # Step 5: Write all files and verify they exist / are non-empty.
        bed_path = tmp_path / "bed.wav"
        obj_path = tmp_path / "object_bird-001.wav"
        _write_wav(bed_path, bed_clean, SAMPLE_RATE)
        _write_wav_mono(obj_path, obj_mono, SAMPLE_RATE)

        assert bed_path.exists() and bed_path.stat().st_size > 100
        assert obj_path.exists() and obj_path.stat().st_size > 100

        # Step 6: Verify WAV metadata.
        with wave.open(str(bed_path), "rb") as w:
            assert w.getnchannels() == 4
            assert w.getframerate() == SAMPLE_RATE
        with wave.open(str(obj_path), "rb") as w:
            assert w.getnchannels() == 1

        # Cleanup (tmp_path is cleaned up by pytest automatically, but be explicit).
        bed_path.unlink()
        obj_path.unlink()
