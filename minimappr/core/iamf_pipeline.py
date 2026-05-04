"""IAMF post-processing pipeline orchestrator.

Executes the "studio render" background job for a completed capture session:

  1. Extract  — read the active IAMF recording buffer for [start_ns/end_ns];
                trim by TOA; zero-pad gaps; capture rate is read from the
                recording source (dynamic, not hardcoded).
  2. Matrix   — frequency-domain A-to-B conversion → bed_full.wav
  3. Isolate  — query DB for tracks; MVDR beamform each → object_{id}.wav
  4. Subtract — B_clean = B_full − Σ Y_obj·O (block-based, time-varying
                steering per SAMPLES_PER_FRAME-block) → bed.wav
  5. Resample — upsample bed + objects to OUTPUT_RATE_HZ with polyphase filter
  6. Measure  — BS.1770-4 integrated loudness + True Peak at OUTPUT_RATE_HZ
  7. Metadata — map MinimapPR position_m → listener-relative spherical coords
  8. Encode   — POST to Rust /api/v1/capture/encode/iamf → audio.iamf
  9. Multiplex— primary: ffmpeg mux video + audio.iamf → youtube_export.mp4
                (IAMF/Eclipsa); fallback: AmbiX WAV + AAC if FFmpeg lacks
                IAMF support.
 10. Cleanup  — remove intermediate files on success
 11. Register — move final files to artifacts dir, insert large_artifacts row
                via insert_large_artifact_for_session with sync diagnostics.

TODO (requires Rust sidecar endpoint changes): Replace inline JSON channel
arrays in _mvdr_beamform and _encode_iamf with file-path-based IPC (write WAV
to work_dir, POST path). Full multiminute arrays saturate Pi RAM at ~150 MB
of JSON for a 5-minute 16 kHz capture.
Note that we never want the raw real-time audio path for MinimapPR writing to disk.
Only this record method is allowed to write to disk raw audio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import wave
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from minimappr.core.ambi_atob import SIRITH_MIC_POSITIONS_M, atob_foa, encode_mono_to_bformat
from minimappr.core.capture_session import CaptureSessionRecord
from minimappr.core.iamf_object_slot import IamfObjectSlot, select_iamf_object_slot

logger = logging.getLogger(__name__)

SAMPLE_RATE_HZ = 16_000      # Default fallback; actual rate read from segment headers
OUTPUT_RATE_HZ = 48_000      # Target rate for IAMF encode, loudness, and WAV output
SAMPLES_PER_FRAME = 512      # IAMF codec frame size at OUTPUT_RATE_HZ
SUBTRACT_BLOCK = 512         # Block size (at capture rate) for time-varying subtraction
ARTIFACTS_DIR = Path("data/artifacts")


# ── Data shapes ───────────────────────────────────────────────────────────────

@dataclass
class TrackTrajectory:
    track_id: str
    waypoints: list[tuple[int, tuple[float, float, float]]]
    """List of (sample_offset, (x, y, z)) waypoints at the capture sample rate."""
    tqi: float = 0.0
    label_confidence: float = 0.0
    localization_confidence: float = 0.0


@dataclass
class LoudnessMeasurement:
    integrated_lufs: float
    true_peak_dbfs: float


@dataclass(frozen=True)
class RecordingSource:
    stream_key: str
    channel_count: int
    sample_rate_hz: int
    role: str
    coverage_ratio: float


# ── Pipeline ──────────────────────────────────────────────────────────────────

class IamfPipeline:
    """Runs the full studio render for one capture session."""

    def __init__(
        self,
        sidecar_url: Optional[str],
        db_storage: "Any",  # minimappr.storage.db.Storage
        *,
        multi_sensor_buffer: Optional[Any] = None,  # minimappr.core.audio_buffer.MultiSensorBuffer
        executor: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._sidecar_url = sidecar_url
        self._db = db_storage
        self._multi_sensor_buffer = multi_sensor_buffer
        self._http = httpx.AsyncClient(timeout=60.0) if sidecar_url else None

    async def run(self, record: CaptureSessionRecord) -> None:
        """Execute all pipeline steps for the given session record."""
        work_dir = record.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

        assert record.start_time_ns is not None
        assert record.end_time_ns is not None
        assert record.first_frame_pts_ns is not None

        start_ns = record.first_frame_pts_ns
        end_ns = record.end_time_ns

        loop = asyncio.get_event_loop()

        # ── 1. Extract ────────────────────────────────────────────────────────
        logger.info("[%s] step 1: extracting audio", record.session_id[:8])
        raw_channels, capture_rate_hz, sync_diag = await self._extract_audio(
            record, start_ns, end_ns
        )
        # Log coverage warnings from the buffer extraction step.
        for warning in sync_diag.get("coverage_warnings", []):
            logger.warning("[%s] audio extraction: %s", record.session_id[:8], warning)
        source_inventory = _build_recording_source_inventory(
            record.stream_key,
            channel_count=int(raw_channels.shape[0]),
            sample_rate_hz=capture_rate_hz,
            n_samples=int(raw_channels.shape[1]),
            start_ns=start_ns,
            end_ns=end_ns,
        )
        sync_diag["recording_sources"] = [
            {
                "stream_key": source.stream_key,
                "channel_count": source.channel_count,
                "sample_rate_hz": source.sample_rate_hz,
                "role": source.role,
                "coverage_ratio": source.coverage_ratio,
            }
            for source in source_inventory
        ]

        # ── 2. A-to-B matrix ─────────────────────────────────────────────────
        logger.info("[%s] step 2: A-to-B ambisonics matrix", record.session_id[:8])
        bed_full = await loop.run_in_executor(
            None, atob_foa, raw_channels, capture_rate_hz
        )

        # When IAMF encoding is disabled, produce ambisonics-only output.
        if not record.include_iamf:
            logger.info("[%s] IAMF disabled; writing ambisonics-only output", record.session_id[:8])
            if capture_rate_hz != OUTPUT_RATE_HZ:
                bed_full = await loop.run_in_executor(
                    None, _resample, bed_full, capture_rate_hz, OUTPUT_RATE_HZ
                )
            bed_full = _normalize_channels_for_encode(bed_full)
            ambix_path = work_dir / "ambix.wav"
            _write_wav(ambix_path, bed_full, OUTPUT_RATE_HZ)

            # Mux with video if available.
            if record.video_path and record.video_path.exists():
                logger.info("[%s] mux: ambisonics + video", record.session_id[:8])
                youtube_path = work_dir / "youtube_export.mp4"
                await _ambix_aac_mux(record.video_path, ambix_path, youtube_path, offset_s=0.0)
                record.youtube_path = youtube_path

            # Register artifact.
            artifacts_dir = ARTIFACTS_DIR
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            final_ambix = artifacts_dir / f"{record.session_id}_ambix.wav"
            ambix_path.replace(final_ambix)
            if record.youtube_path and record.youtube_path.exists():
                final_mp4 = artifacts_dir / f"{record.session_id}_youtube.mp4"
                record.youtube_path.replace(final_mp4)
                record.youtube_path = final_mp4

            await self._register_artifact(record, sync_diag)
            logger.info("[%s] ambisonics-only pipeline complete", record.session_id[:8])
            return

        bed_full_path = work_dir / "bed_full.wav"
        _write_wav(bed_full_path, bed_full, capture_rate_hz)

        # ── 3. Object isolation (MVDR) ────────────────────────────────────────
        logger.info("[%s] step 3: MVDR object isolation", record.session_id[:8])
        n_capture_samples = raw_channels.shape[1]
        trajectories = await self._query_trajectories(
            start_ns, end_ns, n_capture_samples, capture_rate_hz
        )
        object_tracks: dict[str, NDArray[np.float32]] = {}
        for traj in trajectories:
            mono = await self._mvdr_beamform(raw_channels, traj, capture_rate_hz)
            obj_path = work_dir / f"object_{traj.track_id}.wav"
            _write_wav_mono(obj_path, mono, capture_rate_hz)
            object_tracks[traj.track_id] = mono

        # ── 4. Select one YouTube-compatible IAMF object slot ─────────────────
        logger.info("[%s] step 4: selecting IAMF object slot", record.session_id[:8])
        capture_samples_per_unit = max(
            1, int(round(capture_rate_hz * SAMPLES_PER_FRAME / OUTPUT_RATE_HZ))
        )
        capture_slot = select_iamf_object_slot(
            trajectories,
            object_tracks,
            n_capture_samples,
            capture_samples_per_unit,
            sample_rate_hz=capture_rate_hz,
        )

        selected_owner_ids = (
            {tid for tid in capture_slot.unit_track_ids if tid is not None}
            if capture_slot is not None
            else set()
        )
        selected_trajectories = (
            [traj for traj in trajectories if traj.track_id in selected_owner_ids]
            if capture_slot is not None
            else []
        )

        # ── 5. Spatial subtraction of only the active IAMF object slot ─────────
        logger.info("[%s] step 4: spatial subtraction", record.session_id[:8])
        bed_clean_capture = await loop.run_in_executor(
            None,
            _subtract_object_slot,
            bed_full,
            capture_slot,
            capture_samples_per_unit,
            n_capture_samples,
        )
        bed_full_path.unlink(missing_ok=True)

        # ── 6. Resample bed + selected object candidates to OUTPUT_RATE_HZ ────
        logger.info("[%s] step 5: resampling to %d Hz", record.session_id[:8], OUTPUT_RATE_HZ)
        if capture_rate_hz != OUTPUT_RATE_HZ:
            bed_clean = await loop.run_in_executor(
                None, _resample, bed_clean_capture, capture_rate_hz, OUTPUT_RATE_HZ
            )
            resampled_objects: dict[str, NDArray[np.float32]] = {}
            for tid in selected_owner_ids:
                mono = object_tracks[tid]
                resampled_objects[tid] = await loop.run_in_executor(
                    None, _resample, mono[np.newaxis, :], capture_rate_hz, OUTPUT_RATE_HZ
                )
                resampled_objects[tid] = resampled_objects[tid][0]
        else:
            bed_clean = bed_clean_capture
            resampled_objects = {tid: object_tracks[tid] for tid in selected_owner_ids}

        # ── 7. Match exact video duration before metadata and encode ──────────
        logger.info("[%s] step 6: conforming audio duration", record.session_id[:8])
        output_trajectories = _scale_trajectory_waypoints(
            selected_trajectories, capture_rate_hz, OUTPUT_RATE_HZ
        )
        original_output_samples = int(bed_clean.shape[1])
        if record.video_path and record.video_path.exists():
            target_samples = await _probe_video_target_sample_count(
                record.video_path, OUTPUT_RATE_HZ
            )
            if target_samples is not None and target_samples > 0:
                bed_clean = await loop.run_in_executor(
                    None, _conform_channels_to_sample_count, bed_clean, target_samples
                )
                resampled_objects = {
                    tid: await loop.run_in_executor(
                        None,
                        _conform_mono_to_sample_count,
                        mono,
                        target_samples,
                    )
                    for tid, mono in resampled_objects.items()
                }
                output_trajectories = _scale_trajectory_sample_offsets(
                    output_trajectories,
                    original_output_samples,
                    target_samples,
                )

        bed_clean = _normalize_channels_for_encode(bed_clean)

        bed_path = work_dir / "bed.wav"
        _write_wav(bed_path, bed_clean, OUTPUT_RATE_HZ)

        # ── 8. IAMF metadata (waypoints in final OUTPUT_RATE_HZ domain) ───────
        logger.info("[%s] step 7: building IAMF metadata", record.session_id[:8])
        n_output_samples = bed_clean.shape[1]
        output_slot = select_iamf_object_slot(
            output_trajectories,
            resampled_objects,
            n_output_samples,
            SAMPLES_PER_FRAME,
            sample_rate_hz=OUTPUT_RATE_HZ,
        )
        positions_per_unit = output_slot.positions_per_unit if output_slot is not None else []
        object_path: Path | None = None
        if output_slot is not None:
            object_path = work_dir / "object_slot.wav"
            _write_wav_mono(object_path, output_slot.samples, OUTPUT_RATE_HZ)
        positions_path = work_dir / "iamf_positions.json"
        _write_iamf_positions_sidecar(
            positions_path,
            positions_per_unit,
            output_slot,
            SAMPLES_PER_FRAME,
            OUTPUT_RATE_HZ,
        )

        # ── 9. BS.1770-4 loudness measurement at OUTPUT_RATE_HZ ──────────────
        logger.info("[%s] step 6: loudness measurement", record.session_id[:8])
        bed_loudness = await loop.run_in_executor(
            None, _measure_loudness, bed_clean[0], OUTPUT_RATE_HZ
        )
        object_loudness = []
        if output_slot is not None:
            lm = await loop.run_in_executor(
                None, _measure_loudness, output_slot.samples, OUTPUT_RATE_HZ
            )
            object_loudness.append(lm)

        # ── 10. Encode IAMF ───────────────────────────────────────────────────
        logger.info("[%s] step 8: IAMF encode", record.session_id[:8])
        object_track_list = [output_slot.samples] if output_slot is not None else []
        iamf_path = work_dir / "audio.iamf"
        iamf_bytes = await self._encode_iamf(
            bed_clean,
            object_track_list,
            positions_per_unit,
            bed_loudness,
            object_loudness,
            bed_path=bed_path,
            object_path=object_path,
            positions_path=positions_path,
            output_iamf_path=iamf_path,
        )
        if iamf_bytes is not None:
            iamf_path.write_bytes(iamf_bytes)
        record.iamf_path = iamf_path

        # AmbiX WAV: 4-channel W/X/Y/Z at OUTPUT_RATE_HZ for fallback mux.
        ambix_path = work_dir / "ambix.wav"
        _write_wav(ambix_path, bed_clean, OUTPUT_RATE_HZ)

        # ── 10. Multiplex video + IAMF (primary) or AmbiX+AAC (fallback) ──────
        if record.video_path and record.video_path.exists():
            logger.info("[%s] step 9: ffmpeg mux", record.session_id[:8])
            youtube_path = work_dir / "youtube_export.mp4"
            video_audio_offset_s = (
                sync_diag.get("actual_audio_start_ns", start_ns) - start_ns
            ) / 1e9
            await _ffmpeg_mux(
                record.video_path,
                iamf_path,
                ambix_path,
                youtube_path,
                video_audio_offset_s=video_audio_offset_s,
            )
            record.youtube_path = youtube_path
        else:
            logger.warning("[%s] step 9: no video file; skipping mux", record.session_id[:8])

        # ── 11. Cleanup intermediates ─────────────────────────────────────────
        logger.info("[%s] step 10: cleanup intermediates", record.session_id[:8])
        for f in [bed_path, ambix_path, object_path, positions_path] + [
            work_dir / f"object_{tid}.wav" for tid in object_tracks
        ]:
            if f is not None:
                f.unlink(missing_ok=True)

        # ── 12. Register artifacts ────────────────────────────────────────────
        logger.info("[%s] step 11: registering artifacts", record.session_id[:8])
        artifacts_dir = ARTIFACTS_DIR
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        final_iamf = artifacts_dir / f"{record.session_id}_audio.iamf"
        iamf_path.replace(final_iamf)
        record.iamf_path = final_iamf

        if record.youtube_path and record.youtube_path.exists():
            final_mp4 = artifacts_dir / f"{record.session_id}_youtube.mp4"
            record.youtube_path.replace(final_mp4)
            record.youtube_path = final_mp4

        await self._register_artifact(record, sync_diag)
        logger.info("[%s] pipeline complete", record.session_id[:8])

    # ── private helpers ────────────────────────────────────────────────────────

    async def _extract_audio(
        self, record: CaptureSessionRecord, start_ns: int, end_ns: int
    ) -> tuple[NDArray[np.float32], int, dict]:
        """Return (channels, capture_rate_hz, sync_diagnostics) for [start_ns, end_ns].

        IAMF recording is the only raw-audio disk exception. The real-time Rust
        ingest/DSP path remains memory-only, so Rust journal extraction is not
        available here.
        """
        if record.capture_audio_buffer is not None:
            return record.capture_audio_buffer.extract_range(
                record.channel_sensor_ids,
                start_ns,
                end_ns,
            )
        if record.use_python_ingest and self._multi_sensor_buffer is not None:
            return await self._multi_sensor_buffer.extract_range(
                record.channel_sensor_ids, start_ns, end_ns
            )
        raise RuntimeError(
            "IAMF capture requires the dedicated recording buffer; Rust raw journal extraction is disabled"
        )

    async def _query_trajectories(
        self,
        start_ns: int,
        end_ns: int,
        total_samples: int,
        capture_rate_hz: int,
    ) -> list[TrackTrajectory]:
        """Query the DB for confirmed tracks within the recording window."""
        if self._db is None:
            return []
        try:
            rows = await self._db.query_tracks_in_window(start_ns, end_ns)
        except Exception as exc:
            logger.warning("trajectory query failed: %s", exc)
            return []

        trajectories: list[TrackTrajectory] = []
        for row in rows:
            track_id = row["track_id"]
            detections = await self._db.query_detections_for_track(track_id, start_ns, end_ns)
            waypoints: list[tuple[int, tuple[float, float, float]]] = []
            label_confidences: list[float] = []
            localization_confidences: list[float] = []
            for det in detections:
                det_ns = det.get("toa_ns", det.get("timestamp_ns", start_ns))
                sample_offset = int((det_ns - start_ns) * capture_rate_hz / 1_000_000_000)
                sample_offset = max(0, min(total_samples - 1, sample_offset))
                pos = (det.get("x", 0.0), det.get("y", 0.0), det.get("z", 0.0))
                waypoints.append((sample_offset, pos))
                label_confidences.append(float(det.get("label_confidence") or 0.0))
                localization_confidences.append(float(det.get("confidence") or 0.0))
            if waypoints:
                trajectories.append(
                    TrackTrajectory(
                        track_id=track_id,
                        waypoints=waypoints,
                        tqi=float(row.get("tqi") or 0.0),
                        label_confidence=(
                            float(np.mean(label_confidences)) if label_confidences else 0.0
                        ),
                        localization_confidence=(
                            float(np.mean(localization_confidences))
                            if localization_confidences
                            else 0.0
                        ),
                    )
                )

        return trajectories

    async def _mvdr_beamform(
        self,
        channels: NDArray[np.float32],
        traj: TrackTrajectory,
        capture_rate_hz: int,
    ) -> NDArray[np.float32]:
        """Beamform one track trajectory to a mono object signal.

        Uses the Python MVDRBeamformer when no sidecar is configured, otherwise
        calls the Rust /api/v1/capture/render/mvdr endpoint.
        """
        if self._multi_sensor_buffer is None and self._http is None:
            raise RuntimeError(
                "No beamforming backend available (need multi_sensor_buffer or sidecar_url)"
            )
        if self._multi_sensor_buffer is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._mvdr_beamform_python, channels, traj, capture_rate_hz
            )
        return await self._mvdr_beamform_rust(channels, traj, capture_rate_hz)

    def _mvdr_beamform_python(
        self,
        channels: NDArray[np.float32],
        traj: TrackTrajectory,
        capture_rate_hz: int,
    ) -> NDArray[np.float32]:
        """Pure-Python block-by-block MVDR beamformer for offline rendering."""
        from minimappr.core.beamforming import MVDRBeamformer

        n_channels = channels.shape[0]
        n_samples = channels.shape[1]

        # Build sensor_id → position mapping from the fixed Sirith geometry.
        # Channels arrive in order ch0…ch{N-1}, matching SIRITH_MIC_POSITIONS_M rows.
        sensor_ids = [f"ch{i}" for i in range(n_channels)]
        sensor_positions: dict[str, np.ndarray] = {
            sid: SIRITH_MIC_POSITIONS_M[i]
            for i, sid in enumerate(sensor_ids)
            if i < len(SIRITH_MIC_POSITIONS_M)
        }

        beamformer = MVDRBeamformer(diagonal_loading=1e-3)
        output = np.zeros(n_samples, dtype=np.float32)

        if not traj.waypoints:
            return output

        first_active = traj.waypoints[0][0]
        last_active = traj.waypoints[-1][0]
        fade_samples = 100

        for block_start in range(0, n_samples, SUBTRACT_BLOCK):
            block_end = min(block_start + SUBTRACT_BLOCK, n_samples)
            block_len = block_end - block_start

            # Silence blocks entirely outside the track's active range.
            if block_end <= first_active or block_start >= last_active:
                continue

            sample_mid = block_start + block_len // 2
            steer_pos = _interpolate_waypoints(traj.waypoints, sample_mid)

            sensor_windows: dict[str, np.ndarray] = {
                sid: channels[i, block_start:block_end]
                for i, sid in enumerate(sensor_ids)
                if i < channels.shape[0]
            }
            block_out = beamformer.beamform(
                sensor_positions=sensor_positions,
                sensor_windows=sensor_windows,
                sample_rate_hz=capture_rate_hz,
                steer_position_m=steer_pos,
            )
            if block_out.size < block_len:
                block_out = np.pad(block_out, (0, block_len - block_out.size))
            output[block_start:block_end] = block_out[:block_len]

        # Linear fade-in at the start of the active range.
        fade_in_start = max(0, first_active)
        fade_in_end = min(n_samples, first_active + fade_samples)
        if fade_in_end > fade_in_start:
            ramp = np.linspace(0.0, 1.0, fade_in_end - fade_in_start, dtype=np.float32)
            output[fade_in_start:fade_in_end] *= ramp

        # Linear fade-out at the end of the active range.
        fade_out_start = max(0, last_active - fade_samples)
        fade_out_end = min(n_samples, last_active)
        if fade_out_end > fade_out_start:
            ramp = np.linspace(1.0, 0.0, fade_out_end - fade_out_start, dtype=np.float32)
            output[fade_out_start:fade_out_end] *= ramp

        return output

    async def _mvdr_beamform_rust(
        self,
        channels: NDArray[np.float32],
        traj: TrackTrajectory,
        capture_rate_hz: int,
    ) -> NDArray[np.float32]:
        """Call the Rust MVDR endpoint for one track trajectory.

        NOTE: Channels are serialized as JSON here, which is RAM-intensive for
        long recordings (~150 MB JSON for 5 min × 16 kHz × 4 ch). A future
        Rust endpoint change to accept WAV file paths is needed for Pi safety.
        """
        assert self._http is not None
        n_samples = channels.shape[1]
        estimated_json_mb = n_samples * 4 * 8 / 1_000_000  # ~8 bytes per float in JSON
        if estimated_json_mb > 50:
            logger.warning(
                "MVDR payload ~%.0f MB; consider file-based IPC for long sessions",
                estimated_json_mb,
            )

        waypoints_dto = [
            {"sample_offset": s, "position_m": list(pos)}
            for s, pos in traj.waypoints
        ]
        payload = {
            "channels": [ch.tolist() for ch in channels],
            "sample_rate_hz": capture_rate_hz,
            "trajectory": waypoints_dto,
        }
        resp = await self._http.post(
            f"{self._sidecar_url}/api/v1/capture/render/mvdr",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return np.asarray(data["samples"], dtype=np.float32)

    async def _encode_iamf(
        self,
        bed: NDArray[np.float32],
        objects: list[NDArray[np.float32]],
        positions_per_unit: list[dict[int, dict]],
        bed_loudness: LoudnessMeasurement,
        object_loudness: list[LoudnessMeasurement],
        *,
        bed_path: Path,
        object_path: Path | None,
        positions_path: Path,
        output_iamf_path: Path,
    ) -> bytes | None:
        """Encode bed + objects as IAMF.

        Uses the pure-Python ipcm writer when no sidecar is configured,
        otherwise POSTs to the Rust /api/v1/capture/encode/iamf endpoint.
        """
        if self._multi_sensor_buffer is not None:
            from minimappr.core.iamf_writer import write_iamf
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                write_iamf,
                bed,
                objects,
                positions_per_unit,
                bed_loudness,
                object_loudness,
            )
        return await self._encode_iamf_rust(
            bed_path,
            object_path,
            positions_path,
            output_iamf_path,
            bed_loudness,
            object_loudness,
        )

    async def _encode_iamf_rust(
        self,
        bed_path: Path,
        object_path: Path | None,
        positions_path: Path,
        output_iamf_path: Path,
        bed_loudness: LoudnessMeasurement,
        object_loudness: list[LoudnessMeasurement],
    ) -> bytes | None:
        """POST file paths to the Rust IAMF Opus encoder."""
        assert self._http is not None
        if len(object_loudness) > 1:
            raise ValueError("IAMF base export supports at most one object track")
        payload = {
            "sample_rate_hz": OUTPUT_RATE_HZ,
            "samples_per_frame": SAMPLES_PER_FRAME,
            "bed_wav_path": str(bed_path),
            "object_wav_path": str(object_path) if object_path is not None else None,
            "positions_json_path": str(positions_path),
            "output_iamf_path": str(output_iamf_path),
            "bitrate_per_channel_bps": 128_000,
            "bed_loudness": {
                "integrated_loudness_lufs": bed_loudness.integrated_lufs,
                "true_peak_dbfs": bed_loudness.true_peak_dbfs,
            },
            "object_loudness": [
                {
                    "integrated_loudness_lufs": lm.integrated_lufs,
                    "true_peak_dbfs": lm.true_peak_dbfs,
                }
                for lm in object_loudness
            ],
        }
        resp = await self._http.post(
            f"{self._sidecar_url}/api/v1/capture/encode/iamf",
            json=payload,
        )
        resp.raise_for_status()
        return resp.content or None

    async def _register_artifact(
        self, record: CaptureSessionRecord, sync_diag: dict
    ) -> None:
        if self._db is None:
            return
        try:
            await self._db.insert_large_artifact_for_session(
                session_id=record.session_id,
                artifact_type="iamf_video",
                iamf_path=str(record.iamf_path) if record.iamf_path else None,
                youtube_path=str(record.youtube_path) if record.youtube_path else None,
                created_ns=time.time_ns(),
            )
        except Exception as exc:
            logger.error("artifact registration failed: %s", exc)


# ── Pure-Python DSP helpers ───────────────────────────────────────────────────

def _subtract_object_slot(
    bed_full: NDArray[np.float32],
    slot: IamfObjectSlot | None,
    samples_per_unit: int,
    n_samples: int,
) -> NDArray[np.float32]:
    """Remove only the selected IAMF object slot from the Ambisonic bed."""
    if slot is None:
        return bed_full.astype(np.float32).copy()

    bed = bed_full.astype(np.float64).copy()
    samples_per_unit = max(1, samples_per_unit)
    n = min(n_samples, slot.samples.shape[0])

    for unit_idx, unit_positions in enumerate(slot.positions_per_unit):
        pos = unit_positions.get(slot.slot_id)
        if not pos:
            continue
        start = unit_idx * samples_per_unit
        end = min(start + samples_per_unit, n)
        if end <= start:
            continue
        azimuth_deg = 0.5 * (
            float(pos["azimuth_deg"]) + float(pos.get("end_azimuth_deg", pos["azimuth_deg"]))
        )
        elevation_deg = 0.5 * (
            float(pos["elevation_deg"]) + float(pos.get("end_elevation_deg", pos["elevation_deg"]))
        )
        block_bfmt = encode_mono_to_bformat(
            slot.samples[start:end],
            _spherical_to_unit_xyz(azimuth_deg, elevation_deg),
        ).astype(np.float64)
        bed[:, start:end] -= block_bfmt

    return bed.astype(np.float32)


def _subtract_objects(
    bed_full: NDArray[np.float32],
    object_tracks: dict[str, NDArray[np.float32]],
    trajectories: list[TrackTrajectory],
    n_samples: int,
) -> NDArray[np.float32]:
    """B_clean(t) = B_full(t) − Σᵢ Y_obj_i(t) · O_i(t).

    Processes in SUBTRACT_BLOCK-sized blocks.  Within each block the steering
    direction is interpolated from the track's waypoints, so the subtraction
    tracks the actual acoustic trajectory rather than a static mean position.
    """
    traj_by_id = {t.track_id: t for t in trajectories}
    bed = bed_full.astype(np.float64).copy()

    for track_id, mono in object_tracks.items():
        traj = traj_by_id.get(track_id)
        if traj is None or not traj.waypoints:
            continue
        n = min(n_samples, mono.shape[0])
        for block_start in range(0, n, SUBTRACT_BLOCK):
            block_end = min(block_start + SUBTRACT_BLOCK, n)
            sample_mid = block_start + (block_end - block_start) // 2
            pos = _interpolate_waypoints(traj.waypoints, sample_mid)
            block_bfmt = encode_mono_to_bformat(
                mono[block_start:block_end], pos
            ).astype(np.float64)
            bed[:, block_start:block_end] -= block_bfmt

    return bed.astype(np.float32)


def _build_recording_source_inventory(
    stream_key: str,
    *,
    channel_count: int,
    sample_rate_hz: int,
    n_samples: int,
    start_ns: int,
    end_ns: int,
) -> list[RecordingSource]:
    """Describe timestamped sources available to the offline render.

    v1 renders the Ambisonic scene from the primary Sirith stream. The inventory
    keeps the interface explicit so cross-node object beamforming can add more
    entries without changing the IAMF export contract.
    """
    requested_duration_s = max((end_ns - start_ns) / 1_000_000_000.0, 1e-9)
    observed_duration_s = n_samples / max(sample_rate_hz, 1)
    return [
        RecordingSource(
            stream_key=stream_key,
            channel_count=channel_count,
            sample_rate_hz=sample_rate_hz,
            role="primary_scene_and_object_fallback",
            coverage_ratio=float(np.clip(observed_duration_s / requested_duration_s, 0.0, 1.0)),
        )
    ]

def _resample(
    channels: NDArray[np.float32], from_hz: int, to_hz: int
) -> NDArray[np.float32]:
    """Polyphase resample a (C, N) or (1, N) array along axis 1."""
    if from_hz == to_hz:
        return channels
    g = gcd(to_hz, from_hz)
    up, down = to_hz // g, from_hz // g
    return resample_poly(channels, up, down, axis=1).astype(np.float32)


def _scale_trajectory_waypoints(
    trajectories: list[TrackTrajectory],
    from_hz: int,
    to_hz: int,
) -> list[TrackTrajectory]:
    """Scale sample_offset waypoints from one sample rate domain to another."""
    if from_hz == to_hz:
        return trajectories
    scale = to_hz / from_hz
    return [
        TrackTrajectory(
            track_id=t.track_id,
            waypoints=[(int(s * scale), pos) for s, pos in t.waypoints],
            tqi=t.tqi,
            label_confidence=t.label_confidence,
            localization_confidence=t.localization_confidence,
        )
        for t in trajectories
    ]


def _scale_trajectory_sample_offsets(
    trajectories: list[TrackTrajectory],
    from_samples: int,
    to_samples: int,
) -> list[TrackTrajectory]:
    if from_samples <= 0 or from_samples == to_samples:
        return trajectories
    scale = to_samples / from_samples
    return [
        TrackTrajectory(
            track_id=t.track_id,
            waypoints=[(max(0, min(to_samples - 1, int(round(s * scale)))), pos) for s, pos in t.waypoints],
            tqi=t.tqi,
            label_confidence=t.label_confidence,
            localization_confidence=t.localization_confidence,
        )
        for t in trajectories
    ]


async def _probe_video_target_sample_count(
    video_path: Path,
    sample_rate_hz: int,
) -> int | None:
    """Return the exact 48 kHz audio sample count implied by the video duration."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,duration_ts,time_base,nb_frames,r_frame_rate:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning(
            "ffprobe duration probe failed (rc=%d): %s",
            proc.returncode,
            stderr.decode(errors="replace")[-300:],
        )
        return None

    try:
        data = json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("ffprobe returned invalid JSON: %s", exc)
        return None

    duration_s = _duration_seconds_from_ffprobe_json(data)
    if duration_s is None or duration_s <= 0.0:
        return None
    return max(1, int(round(duration_s * sample_rate_hz)))


def _duration_seconds_from_ffprobe_json(data: dict[str, Any]) -> float | None:
    streams = data.get("streams") or []
    stream = streams[0] if streams else {}

    duration_ts = stream.get("duration_ts")
    time_base = stream.get("time_base")
    if duration_ts is not None and time_base:
        num, den = _parse_rational(str(time_base))
        if num > 0 and den > 0:
            return float(duration_ts) * num / den

    duration = stream.get("duration") or (data.get("format") or {}).get("duration")
    if duration is not None:
        try:
            return float(duration)
        except (TypeError, ValueError):
            return None

    nb_frames = stream.get("nb_frames")
    frame_rate = stream.get("r_frame_rate")
    if nb_frames is not None and frame_rate:
        num, den = _parse_rational(str(frame_rate))
        if num > 0 and den > 0:
            return float(nb_frames) * den / num
    return None


def _parse_rational(value: str) -> tuple[int, int]:
    if "/" not in value:
        try:
            return int(value), 1
        except ValueError:
            return 0, 0
    left, right = value.split("/", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return 0, 0


def _conform_channels_to_sample_count(
    channels: NDArray[np.float32],
    target_samples: int,
) -> NDArray[np.float32]:
    """Stretch/squeeze channels to exactly target_samples with polyphase resampling."""
    target_samples = max(1, int(target_samples))
    current_samples = int(channels.shape[1])
    if current_samples == target_samples:
        return channels.astype(np.float32, copy=False)
    if abs(current_samples - target_samples) <= 1:
        return _trim_or_pad_channels(channels, target_samples)

    g = gcd(current_samples, target_samples)
    up = target_samples // g
    down = current_samples // g
    conformed = resample_poly(channels, up, down, axis=1).astype(np.float32)
    return _trim_or_pad_channels(conformed, target_samples)


def _conform_mono_to_sample_count(
    mono: NDArray[np.float32],
    target_samples: int,
) -> NDArray[np.float32]:
    conformed = _conform_channels_to_sample_count(mono[np.newaxis, :], target_samples)
    return conformed[0]


def _trim_or_pad_channels(
    channels: NDArray[np.float32],
    target_samples: int,
) -> NDArray[np.float32]:
    target_samples = max(1, int(target_samples))
    current_samples = int(channels.shape[1])
    if current_samples > target_samples:
        return channels[:, :target_samples].astype(np.float32, copy=False)
    if current_samples < target_samples:
        pad = target_samples - current_samples
        return np.pad(channels, ((0, 0), (0, pad))).astype(np.float32)
    return channels.astype(np.float32, copy=False)


def _normalize_channels_for_encode(channels: NDArray[np.float32]) -> NDArray[np.float32]:
    peak = float(np.max(np.abs(channels))) if channels.size else 0.0
    if peak <= 0.98:
        return channels.astype(np.float32, copy=False)
    return (channels.astype(np.float32) * (0.98 / peak)).astype(np.float32)


def _write_iamf_positions_sidecar(
    path: Path,
    positions_per_unit: list[dict[int, dict]],
    slot: IamfObjectSlot | None,
    samples_per_frame: int,
    sample_rate_hz: int,
) -> None:
    payload = {
        "sample_rate_hz": sample_rate_hz,
        "samples_per_frame": samples_per_frame,
        "positions_per_unit": [
            {str(obj_id): pos for obj_id, pos in unit.items()}
            for unit in positions_per_unit
        ],
        "unit_track_ids": slot.unit_track_ids if slot is not None else [],
        "active_ranges": slot.active_ranges if slot is not None else [],
        "handoff_gap_ranges": slot.handoff_gap_ranges if slot is not None else [],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _measure_loudness(signal: NDArray, sample_rate_hz: int) -> LoudnessMeasurement:
    """Simplified BS.1770-4 integrated loudness and true peak.

    Uses K-weighting (two-stage filter chain) and a 400 ms gating window.
    Full broadcast-grade accuracy is not required here; results within ±1 LU
    of the spec are sufficient for IAMF metadata.
    """
    sig = np.asarray(signal, dtype=np.float64)

    sig_kw = _k_weight(sig, sample_rate_hz)

    block_len = max(1, int(0.4 * sample_rate_hz))
    hop = max(1, block_len // 4)
    n = len(sig_kw)
    mean_sq_blocks = []
    for start in range(0, n, hop):
        blk = sig_kw[start : start + block_len]
        if len(blk) == 0:
            continue
        ms = float(np.mean(blk ** 2))
        mean_sq_blocks.append(ms)

    if not mean_sq_blocks:
        return LoudnessMeasurement(integrated_lufs=-120.0, true_peak_dbfs=-120.0)

    abs_thresh_ms = 10 ** ((-70.0 - 0.691) / 10.0)
    gated = [ms for ms in mean_sq_blocks if ms > abs_thresh_ms]
    if not gated:
        gated = mean_sq_blocks

    integrated_ms = np.mean(gated)
    if integrated_ms <= 1e-12:
        lufs = -120.0
    else:
        lufs = float(-0.691 + 10.0 * math.log10(integrated_ms))

    true_peak_linear = float(np.max(np.abs(np.interp(
        np.linspace(0, len(sig) - 1, len(sig) * 4),
        np.arange(len(sig)),
        sig,
    ))))
    true_peak_dbfs = float(20.0 * math.log10(max(true_peak_linear, 1e-12)))

    return LoudnessMeasurement(integrated_lufs=lufs, true_peak_dbfs=true_peak_dbfs)


def _k_weight(sig: NDArray[np.float64], sr: int) -> NDArray[np.float64]:
    """Apply ITU-R BS.1770-4 K-weighting (two-stage filter, coefficients spec-derived)."""
    from scipy.signal import lfilter

    Vh = 1.58489319
    Vb = 1.25892541
    f0 = 1681.97441
    Q = 0.7071752
    K = math.tan(math.pi * f0 / sr)
    b0 = (Vh + Vb * K / Q + K ** 2) / (1 + K / Q + K ** 2)
    b1 = 2 * (K ** 2 - Vh) / (1 + K / Q + K ** 2)
    b2 = (Vh - Vb * K / Q + K ** 2) / (1 + K / Q + K ** 2)
    a1 = 2 * (K ** 2 - 1) / (1 + K / Q + K ** 2)
    a2 = (1 - K / Q + K ** 2) / (1 + K / Q + K ** 2)
    stage1 = lfilter([b0, b1, b2], [1.0, a1, a2], sig)

    f0_hp = 38.13547
    Q_hp = 0.5003270
    K2 = math.tan(math.pi * f0_hp / sr)
    b0h = 1.0 / (1 + K2 / Q_hp + K2 ** 2)
    a1h = 2 * (K2 ** 2 - 1) * b0h
    a2h = (1 - K2 / Q_hp + K2 ** 2) * b0h
    stage2 = lfilter([b0h, -2 * b0h, b0h], [1.0, a1h, a2h], stage1)

    return stage2


def _build_positions_per_unit(
    trajectories: list[TrackTrajectory],
    n_samples: int,
    samples_per_frame: int,
) -> list[dict[int, dict]]:
    """Build per-temporal-unit object position dictionaries (in OUTPUT_RATE_HZ domain)."""
    n_frames = (n_samples + samples_per_frame - 1) // samples_per_frame
    positions_per_unit: list[dict[int, dict]] = [{} for _ in range(n_frames)]

    for obj_idx, traj in enumerate(trajectories):
        if not traj.waypoints:
            continue
        for fi in range(n_frames):
            sample_start = fi * samples_per_frame
            sample_end = min(n_samples - 1, sample_start + samples_per_frame - 1)
            start_xyz = _interpolate_waypoints(traj.waypoints, sample_start)
            end_xyz = _interpolate_waypoints(traj.waypoints, sample_end)
            az, el, dist = _xyz_to_spherical(start_xyz)
            end_az, end_el, end_dist = _xyz_to_spherical(end_xyz)
            positions_per_unit[fi][obj_idx] = {
                "azimuth_deg": az,
                "elevation_deg": el,
                "distance_norm": min(1.0, max(0.0, dist / 30.0)),
                "end_azimuth_deg": end_az,
                "end_elevation_deg": end_el,
                "end_distance_norm": min(1.0, max(0.0, end_dist / 30.0)),
            }

    return positions_per_unit


def _interpolate_waypoints(
    waypoints: list[tuple[int, tuple[float, float, float]]], sample: int
) -> tuple[float, float, float]:
    if not waypoints:
        return (1.0, 0.0, 0.0)
    if len(waypoints) == 1 or sample <= waypoints[0][0]:
        return waypoints[0][1]
    if sample >= waypoints[-1][0]:
        return waypoints[-1][1]
    for i in range(len(waypoints) - 1):
        s0, p0 = waypoints[i]
        s1, p1 = waypoints[i + 1]
        if s0 <= sample <= s1:
            t = (sample - s0) / max(s1 - s0, 1)
            return (
                p0[0] + t * (p1[0] - p0[0]),
                p0[1] + t * (p1[1] - p0[1]),
                p0[2] + t * (p1[2] - p0[2]),
            )
    return waypoints[-1][1]


def _xyz_to_spherical(
    pos: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert Cartesian room coords to (azimuth_deg, elevation_deg, distance_m)."""
    x, y, z = pos
    dist = math.sqrt(x * x + y * y + z * z)
    if dist < 1e-9:
        return 0.0, 0.0, 0.0
    az = math.degrees(math.atan2(y, x))
    el = math.degrees(math.asin(z / dist))
    return az, el, dist


def _spherical_to_unit_xyz(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cos_el = math.cos(el)
    return (
        cos_el * math.cos(az),
        cos_el * math.sin(az),
        math.sin(el),
    )


# ── WAV I/O helpers ───────────────────────────────────────────────────────────

def _write_wav(path: Path, channels_first: NDArray, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    interleaved = np.clip(channels_first.T, -1.0, 1.0)
    pcm = (interleaved * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(int(channels_first.shape[0]))
        w.setsampwidth(2)
        w.setframerate(int(sample_rate_hz))
        w.writeframes(pcm.tobytes())


def _write_wav_mono(path: Path, samples: NDArray, sample_rate_hz: int) -> None:
    _write_wav(path, samples[np.newaxis, :], sample_rate_hz)


def _decode_pcm16le_4ch(raw: bytes) -> NDArray[np.float32]:
    """Decode raw 4-channel interleaved PCM16LE bytes into float (4, N)."""
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if len(samples) % 4 != 0:
        samples = samples[: len(samples) - len(samples) % 4]
    # Fortran order: first index (channel) varies fastest → correct for interleaved.
    return samples.reshape(4, -1, order="F")


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

async def _ffmpeg_mux(
    video_path: Path,
    iamf_path: Path,
    ambix_wav: Path,
    output_path: Path,
    *,
    video_audio_offset_s: float = 0.0,
) -> None:
    """Mux video with audio into a YouTube-ready MP4.

    Primary path: embed the .iamf file directly (IAMF/Eclipsa codec string
    iamf.001.001.* as required by YouTube Eclipsa verification). Requires
    FFmpeg 6.1+. Falls back to AmbiX WAV + AAC if FFmpeg lacks IAMF support.

    For full YouTube Eclipsa compliance the Rust IAMF encoder must produce
    Opus-coded frames (codec string iamf.001.001.Opus). If it uses ipcm the
    copy here will still produce a structurally valid IAMF-in-MP4, but
    YouTube may reject it at the Eclipsa validation step.

    The video_audio_offset_s corrects for the gap between the video
    first-frame PTS anchor and the actual start of the extracted audio.
    A positive value means audio starts after the video clock origin and
    the audio stream is delayed by that amount in the output container.
    """
    iamf_success = await _try_iamf_mux(
        video_path, iamf_path, output_path, video_audio_offset_s
    )
    if iamf_success:
        logger.info("mux: IAMF-in-MP4 succeeded → %s", output_path.name)
        return

    logger.warning(
        "mux: IAMF path failed; falling back to AmbiX+AAC → %s",
        output_path.name,
    )
    await _ambix_aac_mux(video_path, ambix_wav, output_path, video_audio_offset_s)


async def _try_iamf_mux(
    video_path: Path,
    iamf_path: Path,
    output_path: Path,
    offset_s: float,
) -> bool:
    """Attempt IAMF-in-MP4 mux; returns True on success."""
    itsoffset_args: list[str] = []
    if abs(offset_s) > 0.001:
        itsoffset_args = ["-itsoffset", f"{offset_s:.6f}"]

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        *itsoffset_args,
        "-i", str(iamf_path),
        "-map", "0:v", "-map", "1",
        "-c:v", "copy", "-c:a", "copy",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.debug(
            "IAMF mux failed (rc=%d): %s",
            proc.returncode,
            stderr.decode(errors="replace")[-300:],
        )
        output_path.unlink(missing_ok=True)
        return False
    return True


async def _ambix_aac_mux(
    video_path: Path,
    ambix_wav: Path,
    output_path: Path,
    offset_s: float,
) -> None:
    """Fallback mux: 4-channel AmbiX WAV encoded as AAC into MP4."""
    itsoffset_args: list[str] = []
    if abs(offset_s) > 0.001:
        itsoffset_args = ["-itsoffset", f"{offset_s:.6f}"]

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        *itsoffset_args,
        "-i", str(ambix_wav),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "320k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-metadata:s:a:0", "channel_layout=4.0",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg AmbiX mux failed (rc={proc.returncode}): "
            f"{stderr.decode(errors='replace')[-500:]}"
        )
