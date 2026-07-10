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
  8. Encode   — FFmpeg/Rust IAMF Opus stream groups → audio.iamf
  9. Multiplex— ffmpeg mux video + audio.iamf → youtube_export.mp4
                with IAMF stream groups preserved.
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
import tempfile
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

from minimappr.core.ambi_atob import (
    SIRITH_MIC_POSITIONS_M,
    atob_foa,
    encode_mono_to_bformat,
    foa_geometry_suitable,
)
from minimappr.core.capture_session import CaptureSessionRecord
from minimappr.core.iamf_object_slot import IamfObjectSlot, select_iamf_object_slot
from minimappr.core.recording_visual import render_recording_visual_mp4
from minimappr.spatial_audio import encode_ambisonics
from minimappr.spatial_audio.objects import (
    subtract_object_slot_from_bed,
    subtract_objects_from_bed,
)
from minimappr.utils.audio import read_wav_mono

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
    label: str = ""
    tqi: float = 0.0
    label_confidence: float = 0.0
    localization_confidence: float = 0.0
    rendered_object_samples: NDArray[np.float32] | None = None


@dataclass
class LoudnessMeasurement:
    integrated_lufs: float
    true_peak_dbfs: float


@dataclass(frozen=True)
class IamfStreamGroupLayout:
    group_index: int
    group_type: str
    stream_indices: tuple[int, ...]


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
        artifact_dir: Optional[Path] = None,
        iamf_ambi_profile: str = "parametric_v2",
        mvdr_diagonal_loading: float = 1e-3,
        iamf_object_band_split_enabled: bool = True,
    ) -> None:
        self._sidecar_url = sidecar_url
        self._db = db_storage
        self._multi_sensor_buffer = multi_sensor_buffer
        self._artifact_dir = artifact_dir or ARTIFACTS_DIR
        self._http = httpx.AsyncClient(timeout=60.0) if sidecar_url else None
        self._iamf_ambi_profile = iamf_ambi_profile
        # Object rendering intentionally uses the configured loading WITHOUT
        # the classifier ×10 recall widening — objects want selectivity.
        self._mvdr_diagonal_loading = mvdr_diagonal_loading
        self._iamf_object_band_split_enabled = iamf_object_band_split_enabled

    async def run(self, record: CaptureSessionRecord) -> None:
        """Execute all pipeline steps for the given session record."""
        work_dir = record.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

        assert record.start_time_ns is not None
        assert record.end_time_ns is not None

        # Audio-only captures have no video first-frame timestamp. In that
        # case the capture buffer start is the correct render origin.
        start_ns = record.first_frame_pts_ns or record.start_time_ns
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
        logger.info(
            "[%s] step 2: A-to-B ambisonics profile=%s",
            record.session_id[:8],
            self._iamf_ambi_profile,
        )
        bed_full = await self._encode_ambisonic_bed(
            raw_channels,
            capture_rate_hz,
            work_dir,
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
            artifacts_dir = self._artifact_dir
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            final_ambix = artifacts_dir / f"{record.session_id}_ambix.wav"
            ambix_path.replace(final_ambix)
            record.ambix_path = final_ambix
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
        registered_mic_positions = await self._registered_mic_positions(
            record.stream_key, int(raw_channels.shape[0])
        )
        for traj in trajectories:
            if traj.rendered_object_samples is not None:
                mono = traj.rendered_object_samples
            else:
                mono = await self._mvdr_beamform(
                    raw_channels,
                    traj,
                    capture_rate_hz,
                    mic_positions_m=registered_mic_positions,
                    work_dir=work_dir,
                )
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
            capture_rate_hz,
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
            record.object_path = object_path
        positions_path = work_dir / "iamf_positions.json"
        _write_iamf_positions_sidecar(
            positions_path,
            positions_per_unit,
            output_slot,
            SAMPLES_PER_FRAME,
            OUTPUT_RATE_HZ,
        )
        visual_path = work_dir / "recording_visual.mp4"
        visual_ready = await render_recording_visual_mp4(
            visual_path,
            output_slot,
            output_trajectories,
            n_samples=n_output_samples,
            sample_rate_hz=OUTPUT_RATE_HZ,
            samples_per_unit=SAMPLES_PER_FRAME,
        )
        if visual_ready:
            record.visual_path = visual_path

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

        # AmbiX WAV: 4-channel W/X/Y/Z at OUTPUT_RATE_HZ for review exports.
        ambix_path = work_dir / "ambix.wav"
        _write_wav(ambix_path, bed_clean, OUTPUT_RATE_HZ)

        # ── 10. Multiplex video + IAMF when video exists ──────────────────────
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
        for f in [bed_path] + [
            work_dir / f"object_{tid}.wav" for tid in object_tracks
        ]:
            if f is not None:
                f.unlink(missing_ok=True)

        # ── 12. Register artifacts ────────────────────────────────────────────
        logger.info("[%s] step 11: registering artifacts", record.session_id[:8])
        artifacts_dir = self._artifact_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        final_iamf = artifacts_dir / f"{record.session_id}_audio.iamf"
        iamf_path.replace(final_iamf)
        record.iamf_path = final_iamf

        final_ambix = artifacts_dir / f"{record.session_id}_ambix.wav"
        if ambix_path.exists():
            ambix_path.replace(final_ambix)
            record.ambix_path = final_ambix

        if object_path is not None and object_path.exists():
            final_object = artifacts_dir / f"{record.session_id}_object.wav"
            object_path.replace(final_object)
            record.object_path = final_object

        if record.visual_path and record.visual_path.exists():
            final_visual = artifacts_dir / f"{record.session_id}_visual.mp4"
            record.visual_path.replace(final_visual)
            record.visual_path = final_visual

        if positions_path.exists():
            final_positions = artifacts_dir / f"{record.session_id}_object_positions.json"
            positions_path.replace(final_positions)
            record.positions_path = final_positions

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
            capture_buf = record.capture_audio_buffer
            sensor_ids = record.channel_sensor_ids
            return await asyncio.to_thread(
                capture_buf.extract_range, sensor_ids, start_ns, end_ns
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
        """Query the DB for tracks eligible for object rendering."""
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
            rendered_object_samples = np.zeros(total_samples, dtype=np.float32)
            rendered_object_sample_count = 0
            track_confidence = float(row.get("confidence") or 0.0)
            for det in detections:
                pos = _detection_position_m(det)
                rendered_span = _try_place_birdnet_rendered_object_audio(
                    det,
                    rendered_object_samples,
                    start_ns,
                    end_ns,
                    capture_rate_hz,
                )
                if rendered_span is not None:
                    span_start_sample, span_end_sample, placed_samples = rendered_span
                    if placed_samples > 0:
                        rendered_object_sample_count += placed_samples
                        waypoints.append((span_start_sample, pos))
                        waypoints.append((max(span_start_sample, span_end_sample - 1), pos))
                else:
                    waypoints.extend(
                        _detection_waypoints_for_export(
                            det,
                            start_ns,
                            end_ns,
                            total_samples,
                            capture_rate_hz,
                            pos,
                        )
                    )
                label_confidences.append(float(det.get("label_confidence") or 0.0))
                localization_confidences.append(float(det.get("confidence") or 0.0))
            if waypoints:
                waypoints = _dedupe_sorted_waypoints(waypoints)
                detection_localization_confidence = (
                    float(np.mean(localization_confidences))
                    if localization_confidences
                    else 0.0
                )
                trajectories.append(
                    TrackTrajectory(
                        track_id=track_id,
                        waypoints=waypoints,
                        label=str(row.get("label") or ""),
                        tqi=float(row.get("tqi") or 0.0),
                        label_confidence=(
                            float(np.mean(label_confidences))
                            if label_confidences
                            else track_confidence
                        ),
                        localization_confidence=max(
                            detection_localization_confidence,
                            track_confidence,
                        ),
                        rendered_object_samples=(
                            rendered_object_samples
                            if rendered_object_sample_count > 0
                            else None
                        ),
                    )
                )

        return trajectories

    async def _registered_mic_positions(
        self,
        stream_key: str,
        channel_count: int,
    ) -> NDArray[np.float64] | None:
        """Registered node geometry for the recording's node, when available.

        Falls back to None (Sirith default downstream) when the node is
        unknown or its sensor offsets don't match the channel count.
        """
        node_id = _node_id_from_sensor_id(stream_key)
        lookup = getattr(self._db, "get_node_by_id", None)
        if not callable(lookup):
            return None
        try:
            node = await lookup(node_id)
        except Exception:  # pragma: no cover - storage backends vary in tests
            return None
        if not isinstance(node, dict):
            return None
        offsets = node.get("sensor_offsets_m")
        if not isinstance(offsets, list) or len(offsets) < channel_count:
            return None
        try:
            positions = np.asarray(offsets[:channel_count], dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if positions.shape != (channel_count, 3):
            return None
        return positions

    async def _mvdr_beamform(
        self,
        channels: NDArray[np.float32],
        traj: TrackTrajectory,
        capture_rate_hz: int,
        mic_positions_m: NDArray[np.float64] | None = None,
        work_dir: Path | None = None,
    ) -> NDArray[np.float32]:
        """Beamform one track trajectory to a mono object signal.

        Prefer the Rust /api/v1/capture/render/mvdr endpoint when configured,
        but fall back to the in-process Python MVDR beamformer if that RPC
        fails. The offline renderer already has the extracted channel windows,
        so a transient sidecar read error should degrade performance, not
        convert the whole recording into a failed session.
        """
        if self._http is not None:
            try:
                return await self._mvdr_beamform_rust(channels, traj, capture_rate_hz, work_dir)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                logger.warning(
                    "MVDR rust beamform failed for track %s; falling back to Python beamformer: %s",
                    traj.track_id,
                    str(exc) or exc.__class__.__name__,
                )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._mvdr_beamform_python, channels, traj, capture_rate_hz, mic_positions_m
        )

    async def _encode_ambisonic_bed(
        self,
        raw_channels: NDArray[np.float32],
        capture_rate_hz: int,
        work_dir: Path,
    ) -> NDArray[np.float32]:
        if self._http is not None:
            try:
                return await self._encode_ambisonic_bed_rust(
                    raw_channels,
                    capture_rate_hz,
                    work_dir,
                )
            except (httpx.HTTPError, KeyError, ValueError, OSError) as exc:
                logger.warning(
                    "ambisonics rust render failed; falling back to Python encoder: %s",
                    str(exc) or exc.__class__.__name__,
                )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            _encode_ambisonic_bed,
            raw_channels,
            capture_rate_hz,
            self._iamf_ambi_profile,
        )

    async def _encode_ambisonic_bed_rust(
        self,
        raw_channels: NDArray[np.float32],
        capture_rate_hz: int,
        work_dir: Path,
    ) -> NDArray[np.float32]:
        assert self._http is not None
        input_path = work_dir / "ambisonics_input.wav"
        output_path = work_dir / "ambisonics_sidecar.wav"
        _write_wav(input_path, raw_channels, capture_rate_hz)
        payload = {
            "input_wav_path": str(input_path),
            "output_wav_path": str(output_path),
            "mic_positions_m": SIRITH_MIC_POSITIONS_M.tolist(),
            "profile": self._iamf_ambi_profile,
        }
        response = await self._http.post(
            f"{self._sidecar_url}/api/v1/capture/render/ambisonics",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        if int(data.get("sample_rate_hz", 0)) != int(capture_rate_hz):
            raise ValueError("ambisonics sidecar returned a mismatched sample rate")
        channels, sample_rate_hz = _read_wav_channels(output_path, expected_channels=4)
        if sample_rate_hz != capture_rate_hz:
            raise ValueError("ambisonics sidecar WAV has a mismatched sample rate")
        return channels

    def _mvdr_beamform_python(
        self,
        channels: NDArray[np.float32],
        traj: TrackTrajectory,
        capture_rate_hz: int,
        mic_positions_m: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float32]:
        """Pure-Python MVDR beamformer for offline rendering.

        Delegates to :class:`BlockTrajectoryRenderer` (Hann overlap-add) so
        independently estimated MVDR weights change smoothly instead of
        stepping at block boundaries. Trajectory interpolation stays here.
        """
        from minimappr.core.beamforming import BlockTrajectoryRenderer, MVDRBeamformer

        if not traj.waypoints:
            return np.zeros(channels.shape[1], dtype=np.float32)

        positions = mic_positions_m if mic_positions_m is not None else SIRITH_MIC_POSITIONS_M
        renderer = BlockTrajectoryRenderer(
            beamformer=MVDRBeamformer(diagonal_loading=self._mvdr_diagonal_loading),
            block_size=SUBTRACT_BLOCK,
            omni_blend_above_cutoff=self._iamf_object_band_split_enabled,
        )
        return renderer.render(
            channels,
            np.asarray(positions, dtype=np.float64),
            capture_rate_hz,
            lambda sample_mid: _interpolate_waypoints(traj.waypoints, sample_mid),
            active_range=(traj.waypoints[0][0], traj.waypoints[-1][0]),
        )

    async def _mvdr_beamform_rust(
        self,
        channels: NDArray[np.float32],
        traj: TrackTrajectory,
        capture_rate_hz: int,
        work_dir: Path | None = None,
    ) -> NDArray[np.float32]:
        """Call the Rust MVDR endpoint for one track trajectory."""
        assert self._http is not None
        if work_dir is not None:
            try:
                return await self._mvdr_beamform_rust_file(
                    channels,
                    traj,
                    capture_rate_hz,
                    work_dir,
                )
            except (httpx.HTTPError, KeyError, ValueError, OSError) as exc:
                logger.warning(
                    "MVDR rust file render failed for track %s; falling back to JSON RPC: %s",
                    traj.track_id,
                    str(exc) or exc.__class__.__name__,
                )

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

    async def _mvdr_beamform_rust_file(
        self,
        channels: NDArray[np.float32],
        traj: TrackTrajectory,
        capture_rate_hz: int,
        work_dir: Path,
    ) -> NDArray[np.float32]:
        assert self._http is not None
        safe_track_id = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_"
            for ch in traj.track_id
        )[:80] or "track"
        input_path = work_dir / f"mvdr_input_{safe_track_id}.wav"
        output_path = work_dir / f"mvdr_object_{safe_track_id}.wav"
        _write_wav(input_path, channels, capture_rate_hz)
        waypoints_dto = [
            {"sample_offset": s, "position_m": list(pos)}
            for s, pos in traj.waypoints
        ]
        response = await self._http.post(
            f"{self._sidecar_url}/api/v1/capture/render/mvdr-file",
            json={
                "input_wav_path": str(input_path),
                "output_wav_path": str(output_path),
                "trajectory": waypoints_dto,
            },
        )
        response.raise_for_status()
        data = response.json()
        if int(data.get("sample_rate_hz", 0)) != int(capture_rate_hz):
            raise ValueError("MVDR sidecar returned a mismatched sample rate")
        mono, sample_rate_hz = read_wav_mono(output_path)
        if sample_rate_hz != capture_rate_hz:
            raise ValueError("MVDR sidecar WAV has a mismatched sample rate")
        return mono.astype(np.float32, copy=False)

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
        """Encode bed + objects as IAMF with FFmpeg/libopus.

        Object coordinates stay in the JSON sidecar until the custom writer has
        decoder coverage. FFmpeg's muxer reliably remuxes its own IAMF stream
        groups into MP4; the experimental coordinate-bearing writer does not.
        """
        await _encode_iamf_ffmpeg(
            bed_path,
            object_path,
            output_iamf_path,
            bed_loudness,
            object_loudness,
        )

        # Groundwork for a future IAMF spec version: OBJECT_BASED audio
        # elements with per-frame SINGLE_POSITION automation. No-op today
        # (ffmpeg has no support for either as of 8.0.1); once ffmpeg adds
        # `-stream_group audio_element_type=object` support, this starts
        # producing a second, positional export alongside the v1.0 file
        # above without affecting it.
        try:
            if object_path is not None and await _ffmpeg_supports_iamf_object_based():
                v11_path = output_iamf_path.with_name(
                    output_iamf_path.stem + "_objects_v11" + output_iamf_path.suffix
                )
                await _encode_iamf_ffmpeg_objects_v11(
                    bed_path,
                    object_path,
                    positions_path,
                    v11_path,
                    bed_loudness,
                    object_loudness,
                )
        except Exception:
            logger.debug("v1.1 OBJECT_BASED IAMF export skipped", exc_info=True)

        return None

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
                ambix_path=str(record.ambix_path) if record.ambix_path else None,
                iamf_path=str(record.iamf_path) if record.iamf_path else None,
                object_path=str(record.object_path) if record.object_path else None,
                visual_path=str(record.visual_path) if record.visual_path else None,
                youtube_path=str(record.youtube_path) if record.youtube_path else None,
                positions_path=str(record.positions_path) if record.positions_path else None,
                created_ns=time.time_ns(),
            )
        except Exception as exc:
            logger.error("artifact registration failed: %s", exc)


# ── Pure-Python DSP helpers ───────────────────────────────────────────────────

def _encode_ambisonic_bed(
    raw_channels: NDArray[np.float32],
    capture_rate_hz: int,
    iamf_ambi_profile: str,
) -> NDArray[np.float32]:
    if iamf_ambi_profile == "linear_v1":
        return atob_foa(raw_channels, capture_rate_hz)
    return encode_ambisonics(
        raw_channels,
        capture_rate_hz,
        profile=iamf_ambi_profile,
    )


def _subtract_object_slot(
    bed_full: NDArray[np.float32],
    slot: IamfObjectSlot | None,
    samples_per_unit: int,
    n_samples: int,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> NDArray[np.float32]:
    """Remove only the selected IAMF object slot from the Ambisonic bed."""
    return subtract_object_slot_from_bed(
        bed_full,
        slot,
        samples_per_unit,
        n_samples,
        sample_rate_hz=sample_rate_hz,
    )


def _subtract_objects(
    bed_full: NDArray[np.float32],
    object_tracks: dict[str, NDArray[np.float32]],
    trajectories: list[TrackTrajectory],
    n_samples: int,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> NDArray[np.float32]:
    """B_clean(t) = B_full(t) − Σᵢ Y_obj_i(t) · O_i(t).

    Processes in SUBTRACT_BLOCK-sized blocks.  Within each block the steering
    direction is interpolated from the track's waypoints, so the subtraction
    tracks the actual acoustic trajectory rather than a static mean position.
    """
    return subtract_objects_from_bed(
        bed_full,
        object_tracks,
        trajectories,
        n_samples,
        sample_rate_hz=sample_rate_hz,
    )


def render_cluster_audio(
    cluster_spec: "Any",  # minimappr.models.ClusterSpec
    channels: dict[str, NDArray[np.float32]],
    sensor_positions: dict[str, NDArray],
    sensor_grades: dict[str, "Any"],  # minimappr.models.SyncGrade
    capture_rate_hz: int,
    node_specs: dict[str, "Any"] | None = None,  # minimappr.models.NodeSpec
) -> "ClusterRenderResult":
    """DSP-layer render for a node cluster.

    Accepts pre-extracted per-sensor audio windows and the cluster topology,
    selects the render mode (FOA_BED vs N_MONO_OBJECTS per AUTO logic), and
    returns a ``ClusterRenderResult`` with the rendered audio + metadata.

    This function does no I/O.  The full pipeline (loudness measurement, IAMF
    encode, resampling, artifact registration) is performed by IamfPipeline.run().

    AUTO mode partition rules:
    * Prefer one intact high-grade tetra node as the FOA anchor bed.
    * Never build a coherent FOA bed across distributed nodes.
    * Non-anchor sensors are emitted as mono side streams for object/remote
      context rather than blended coherently into the bed.
    * If no suitable anchor tetra exists, fall back to all-N-mono.
    """
    from minimappr.models import ClusterSpec, IamfRenderMode, SyncGrade
    from minimappr.core.iamf_writer import MonoSubstreamMeta, NMonoBundle

    assert isinstance(cluster_spec, ClusterSpec)

    _HIGH_GRADES = {SyncGrade.GPS_PPS, SyncGrade.PTP}
    mode = cluster_spec.iamf_render_mode

    # Resolve ordered sensor lists.
    sensor_ids = sorted(channels.keys())
    node_specs = node_specs or {}

    if mode == IamfRenderMode.AUTO:
        high_ids = [sid for sid in sensor_ids if sensor_grades.get(sid) in _HIGH_GRADES]
        anchor_node_id, anchor_sensor_ids = _select_anchor_tetra_sensor_ids(
            high_ids,
            sensor_positions,
            cluster_spec.max_baseline_m_for_foa,
        )
        if anchor_sensor_ids:
            high_ids = anchor_sensor_ids
            ntp_ids = [sid for sid in sensor_ids if sid not in set(anchor_sensor_ids)]
            suitable = True
        else:
            suitable = False
            ntp_ids = sensor_ids
            high_ids = []
            anchor_node_id = None
        mode = IamfRenderMode.FOA_BED if suitable else IamfRenderMode.N_MONO_OBJECTS
    else:
        high_ids = sensor_ids
        ntp_ids = []
        anchor_node_id = _node_id_from_sensor_id(sensor_ids[0]) if sensor_ids else None

    if mode == IamfRenderMode.FOA_BED:
        # Stack channels in sensor_id order; build matching position array.
        foa_ids = high_ids if high_ids else sensor_ids
        stacked = np.array([channels[sid] for sid in foa_ids], dtype=np.float32)
        mic_pos = _mic_positions_for_sensor_ids(
            foa_ids,
            sensor_positions,
            node_specs,
            anchor_node_id,
        )
        bed = encode_ambisonics(
            stacked,
            capture_rate_hz,
            profile="parametric_v2",
            mic_positions_m=mic_pos,
        )

        ntp_bundle: NMonoBundle | None = None
        if ntp_ids:
            ntp_streams = [
                (
                    MonoSubstreamMeta(
                        sensor_id=sid,
                        position_m=tuple(float(v) for v in sensor_positions[sid]),
                        sync_grade=getattr(sensor_grades.get(sid), "value", "free"),
                    ),
                    channels[sid],
                )
                for sid in ntp_ids
            ]
            ntp_bundle = NMonoBundle(streams=ntp_streams, sample_rate_hz=capture_rate_hz)

        return ClusterRenderResult(
            render_mode=IamfRenderMode.FOA_BED,
            foa_bed=bed,
            foa_sensor_ids=foa_ids,
            anchor_node_id=anchor_node_id,
            ntp_mono_bundle=ntp_bundle,
            capture_rate_hz=capture_rate_hz,
        )

    else:  # N_MONO_OBJECTS
        streams = [
            (
                MonoSubstreamMeta(
                    sensor_id=sid,
                    position_m=tuple(float(v) for v in sensor_positions[sid]),
                    sync_grade=getattr(sensor_grades.get(sid), "value", "free"),
                ),
                channels[sid],
            )
            for sid in sensor_ids
        ]
        bundle = NMonoBundle(streams=streams, sample_rate_hz=capture_rate_hz)
        return ClusterRenderResult(
            render_mode=IamfRenderMode.N_MONO_OBJECTS,
            mono_bundle=bundle,
            capture_rate_hz=capture_rate_hz,
        )


@dataclass
class ClusterRenderResult:
    """Output of ``render_cluster_audio()``.

    Exactly one of ``foa_bed`` or ``mono_bundle`` is populated depending on
    which render mode was chosen.
    """
    render_mode: "Any"  # IamfRenderMode
    capture_rate_hz: int
    foa_bed: NDArray[np.float32] | None = None
    foa_sensor_ids: list[str] | None = None
    anchor_node_id: str | None = None
    ntp_mono_bundle: "Any | None" = None  # NMonoBundle for NTP sensors in AUTO+FOA mode
    mono_bundle: "Any | None" = None       # NMonoBundle for full N-mono mode


def _select_anchor_tetra_sensor_ids(
    high_grade_sensor_ids: list[str],
    sensor_positions: dict[str, NDArray],
    max_baseline_m_for_foa: float,
) -> tuple[str | None, list[str]]:
    grouped: dict[str, list[str]] = {}
    for sensor_id in high_grade_sensor_ids:
        grouped.setdefault(_node_id_from_sensor_id(sensor_id), []).append(sensor_id)

    candidates: list[tuple[float, str, list[str]]] = []
    for node_id, ids in grouped.items():
        if len(ids) < 4:
            continue
        ordered = sorted(ids, key=_sensor_channel_sort_key)[:4]
        positions = np.array([sensor_positions[sid] for sid in ordered], dtype=np.float64)
        suitable, _ = foa_geometry_suitable(positions, max_baseline_m_for_foa)
        if not suitable:
            continue
        baseline = _max_pairwise_baseline_m(positions)
        candidates.append((baseline, node_id, ordered))

    if not candidates:
        return None, []
    # Prefer the tightest suitable tetra geometry; tie-break by node id for determinism.
    _, node_id, ordered = min(candidates, key=lambda item: (item[0], item[1]))
    return node_id, ordered


def _mic_positions_for_sensor_ids(
    sensor_ids: list[str],
    sensor_positions: dict[str, NDArray],
    node_specs: dict[str, "Any"],
    anchor_node_id: str | None,
) -> NDArray[np.float64]:
    from minimappr.spatial_audio.geometry import rotate_positions

    if anchor_node_id and anchor_node_id in node_specs:
        node = node_specs[anchor_node_id]
        offsets = np.asarray(getattr(node, "sensor_offsets_m", []), dtype=np.float64)
        if offsets.shape[0] >= len(sensor_ids) and offsets.ndim == 2 and offsets.shape[1] == 3:
            channel_indices = [_sensor_channel_index(sensor_id) for sensor_id in sensor_ids]
            if all(index is not None and index < offsets.shape[0] for index in channel_indices):
                selected = np.asarray([offsets[int(index)] for index in channel_indices], dtype=np.float64)
                rotated = rotate_positions(selected, getattr(node, "orientation", None))
                position_m = getattr(node, "position_m", None)
                if position_m is not None:
                    rotated = rotated + np.asarray(position_m, dtype=np.float64)
                return rotated.astype(np.float64)

    return np.array([sensor_positions[sid] for sid in sensor_ids], dtype=np.float64)


def _node_id_from_sensor_id(sensor_id: str) -> str:
    return sensor_id.split(":", 1)[0]


def _sensor_channel_sort_key(sensor_id: str) -> tuple[int, str]:
    index = _sensor_channel_index(sensor_id)
    return (index if index is not None else 10_000, sensor_id)


def _sensor_channel_index(sensor_id: str) -> int | None:
    suffix = sensor_id.rsplit(":", 1)[-1]
    if suffix.startswith("ch") and suffix[2:].isdigit():
        return int(suffix[2:])
    return None


def _max_pairwise_baseline_m(positions: NDArray[np.float64]) -> float:
    max_distance = 0.0
    for first_index in range(positions.shape[0]):
        for second_index in range(first_index + 1, positions.shape[0]):
            max_distance = max(
                max_distance,
                float(np.linalg.norm(positions[first_index] - positions[second_index])),
            )
    return max_distance


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


def _detection_position_m(det: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(det.get("x") or 0.0),
        float(det.get("y") or 0.0),
        float(det.get("z") or 0.0),
    )


def _detection_waypoints_for_export(
    det: dict[str, Any],
    start_ns: int,
    end_ns: int,
    total_samples: int,
    capture_rate_hz: int,
    pos: tuple[float, float, float],
) -> list[tuple[int, tuple[float, float, float]]]:
    report_start_ns = _optional_detection_int(det.get("report_window_start_ns"))
    report_end_ns = _optional_detection_int(det.get("report_window_end_ns"))
    if (
        report_start_ns is not None
        and report_end_ns is not None
        and report_end_ns >= start_ns
        and report_start_ns <= end_ns
    ):
        span_start_sample = _ns_to_capture_sample(
            max(start_ns, report_start_ns), start_ns, total_samples, capture_rate_hz
        )
        span_end_sample = _ns_to_capture_sample_exclusive(
            min(end_ns, report_end_ns), start_ns, total_samples, capture_rate_hz
        )
        if span_end_sample > span_start_sample:
            return [
                (span_start_sample, pos),
                (max(span_start_sample, span_end_sample - 1), pos),
            ]

    det_ns = int(det.get("toa_ns") or det.get("timestamp_ns") or start_ns)
    return [(_ns_to_capture_sample(det_ns, start_ns, total_samples, capture_rate_hz), pos)]


def _try_place_birdnet_rendered_object_audio(
    det: dict[str, Any],
    destination: NDArray[np.float32],
    start_ns: int,
    end_ns: int,
    capture_rate_hz: int,
) -> tuple[int, int, int] | None:
    feature_summary = _detection_feature_summary(det)
    render_kind = str(feature_summary.get("rust_render_kind") or "").strip()
    if not render_kind.startswith("birdnet_hybrid_"):
        return None

    render_start_ns = _optional_detection_int(
        feature_summary.get("rust_render_start_ns")
        or feature_summary.get("render_start_ns")
    )
    render_end_ns = _optional_detection_int(
        feature_summary.get("rust_render_end_ns")
        or feature_summary.get("render_end_ns")
    )
    snippet_path = det.get("snippet_path")
    if render_start_ns is None or render_end_ns is None or render_end_ns <= render_start_ns:
        return None
    if render_end_ns < start_ns or render_start_ns > end_ns:
        return None
    if not snippet_path:
        return None

    path = Path(str(snippet_path))
    if not path.exists():
        return None

    try:
        mono, snippet_rate_hz = read_wav_mono(path)
    except Exception as exc:
        logger.warning("IAMF object snippet read failed for %s: %s", path, exc)
        return None
    if mono.size == 0:
        return None

    mono = mono.astype(np.float32, copy=False)
    if snippet_rate_hz != capture_rate_hz:
        mono = _resample(mono[np.newaxis, :], snippet_rate_hz, capture_rate_hz)[0]

    target_samples = max(
        1,
        int(round((render_end_ns - render_start_ns) * capture_rate_hz / 1_000_000_000)),
    )
    mono = _conform_mono_to_sample_count(mono, target_samples)

    unclipped_start = int(round((render_start_ns - start_ns) * capture_rate_hz / 1_000_000_000))
    unclipped_end = unclipped_start + mono.shape[0]
    dest_start = max(0, unclipped_start)
    dest_end = min(destination.shape[0], unclipped_end)
    if dest_end <= dest_start:
        return None

    source_start = dest_start - unclipped_start
    source_end = source_start + (dest_end - dest_start)
    destination[dest_start:dest_end] = np.clip(
        destination[dest_start:dest_end] + mono[source_start:source_end],
        -1.0,
        1.0,
    )
    return dest_start, dest_end, dest_end - dest_start


def _detection_feature_summary(det: dict[str, Any]) -> dict[str, Any]:
    feature_summary = det.get("feature_summary")
    if isinstance(feature_summary, dict):
        return feature_summary
    raw = det.get("feature_summary_json")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _optional_detection_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ns_to_capture_sample(
    value_ns: int,
    start_ns: int,
    total_samples: int,
    capture_rate_hz: int,
) -> int:
    if total_samples <= 0:
        return 0
    sample = int(round((value_ns - start_ns) * capture_rate_hz / 1_000_000_000))
    return max(0, min(total_samples - 1, sample))


def _ns_to_capture_sample_exclusive(
    value_ns: int,
    start_ns: int,
    total_samples: int,
    capture_rate_hz: int,
) -> int:
    if total_samples <= 0:
        return 0
    sample = int(round((value_ns - start_ns) * capture_rate_hz / 1_000_000_000))
    return max(0, min(total_samples, sample))


def _dedupe_sorted_waypoints(
    waypoints: list[tuple[int, tuple[float, float, float]]]
) -> list[tuple[int, tuple[float, float, float]]]:
    deduped: dict[int, tuple[float, float, float]] = {}
    for sample_offset, pos in waypoints:
        deduped[int(sample_offset)] = pos
    return sorted(deduped.items(), key=lambda item: item[0])


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
            label=t.label,
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
            label=t.label,
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


def _read_wav_channels(
    path: Path,
    *,
    expected_channels: int | None = None,
) -> tuple[NDArray[np.float32], int]:
    with wave.open(str(path), "rb") as wav_file:
        channel_count = int(wav_file.getnchannels())
        sample_width = int(wav_file.getsampwidth())
        sample_rate_hz = int(wav_file.getframerate())
        frame_count = int(wav_file.getnframes())
        raw = wav_file.readframes(frame_count)

    if expected_channels is not None and channel_count != expected_channels:
        raise ValueError(
            f"{path} has {channel_count} channels, expected {expected_channels}"
        )
    if sample_width != 2:
        raise ValueError(f"{path} must be 16-bit PCM, got {sample_width * 8} bits")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    if data.size % channel_count != 0:
        raise ValueError(f"{path} sample count is not divisible by channel count")
    return data.reshape(-1, channel_count).T.astype(np.float32), sample_rate_hz


def _decode_pcm16le_4ch(raw: bytes) -> NDArray[np.float32]:
    """Decode raw 4-channel interleaved PCM16LE bytes into float (4, N)."""
    return _decode_pcm16le(raw, channel_count=4)


def _decode_pcm16le(raw: bytes, channel_count: int) -> NDArray[np.float32]:
    """Decode raw N-channel interleaved PCM16LE bytes into float (N, samples)."""
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    remainder = len(samples) % channel_count
    if remainder:
        samples = samples[: len(samples) - remainder]
    # Fortran order: first index (channel) varies fastest → correct for interleaved.
    return samples.reshape(channel_count, -1, order="F")


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

_ffmpeg_object_based_support: bool | None = None


async def _ffmpeg_supports_iamf_object_based() -> bool:
    """Probe whether the installed ffmpeg accepts audio_element_type=object.

    IAMF's OBJECT_BASED audio element type (and the per-frame
    SINGLE_POSITION parameter used to animate an object's position) were
    added to the spec after v1.1.0 and have no `-stream_group` support in
    ffmpeg as of 8.0.1 -- it rejects `audio_element_type=object` with
    "Unable to parse option value". This runs a throwaway 1-frame encode to
    detect support so a future ffmpeg with OBJECT_BASED support is picked up
    automatically. The result is cached for the process lifetime.
    """
    global _ffmpeg_object_based_support
    if _ffmpeg_object_based_support is not None:
        return _ffmpeg_object_based_support

    with tempfile.TemporaryDirectory() as tmp:
        probe_path = Path(tmp) / "probe.iamf"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=0.02",
            "-c:a", "libopus",
            "-stream_group",
            "type=iamf_audio_element:id=1:st=0:audio_element_type=object,"
            "layer=ch_layout=mono",
            "-stream_group",
            "type=iamf_mix_presentation:id=2:stg=0:"
            "submix=parameter_id=100:parameter_rate=48000:default_mix_gain=0.0|"
            "element=stg=0:headphones_rendering_mode=binaural:"
            "parameter_id=101:parameter_rate=48000:default_mix_gain=0.0|"
            "layout=sound_system=stereo:integrated_loudness=0:digital_peak=0",
            str(probe_path),
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            supported = proc.returncode == 0 and probe_path.exists()
        except FileNotFoundError:
            supported = False

    _ffmpeg_object_based_support = supported
    return supported


async def _encode_iamf_ffmpeg_objects_v11(
    bed_path: Path,
    object_path: Path,
    positions_path: Path,
    output_iamf_path: Path,
    bed_loudness: LoudnessMeasurement,
    object_loudness: list[LoudnessMeasurement],
) -> None:
    """Encode an OBJECT_BASED export with per-frame SINGLE_POSITION automation.

    Stub: only reached once `_ffmpeg_supports_iamf_object_based()` returns
    True, which it does not for any ffmpeg release as of 8.0.1. When ffmpeg
    adds `-stream_group` support for `audio_element_type=object` and a
    per-frame position parameter, implement the encode here using
    `positions_path` (`*_object_positions.json`, written by
    `_write_iamf_positions_sidecar` -- `positions_per_unit` gives
    azimuth_deg/elevation_deg/distance_norm per temporal unit) as the
    parameter source.
    """
    raise NotImplementedError("ffmpeg OBJECT_BASED IAMF export not yet available")


async def _ffmpeg_mux(
    video_path: Path,
    iamf_path: Path,
    ambix_wav: Path,
    output_path: Path,
    *,
    video_audio_offset_s: float = 0.0,
) -> None:
    """Mux video with audio into a YouTube-ready MP4.

    Prefer embedding the .iamf file directly and recreating IAMF stream groups
    in the MP4. If the host ffmpeg build cannot mux IAMF into MP4, fail the
    export instead of silently switching to a different audio format.

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

    raise RuntimeError(f"IAMF-in-MP4 mux failed for {output_path.name}")


async def _encode_iamf_ffmpeg(
    bed_path: Path,
    object_path: Path | None,
    output_iamf_path: Path,
    bed_loudness: LoudnessMeasurement,
    object_loudness: list[LoudnessMeasurement],
) -> None:
    """Encode AmbiX bed plus optional mono object as standalone IAMF/Opus."""
    if object_path is not None and len(object_loudness) != 1:
        raise ValueError("IAMF object export requires exactly one object loudness measurement")

    filter_complex = (
        "[0:a]channelmap=0:mono[bed0];"
        "[0:a]channelmap=1:mono[bed1];"
        "[0:a]channelmap=2:mono[bed2];"
        "[0:a]channelmap=3:mono[bed3]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(bed_path),
    ]
    if object_path is not None:
        cmd.extend(["-i", str(object_path)])
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[bed0]",
        "-map", "[bed1]",
        "-map", "[bed2]",
        "-map", "[bed3]",
    ])
    if object_path is not None:
        cmd.extend(["-map", "1:a:0"])

    output_stream_count = 5 if object_path is not None else 4
    for stream_index in range(output_stream_count):
        cmd.extend(["-streamid", f"{stream_index}:{stream_index}"])

    bed_lufs = _format_iamf_loudness(bed_loudness.integrated_lufs)
    bed_peak = _format_iamf_loudness(bed_loudness.true_peak_dbfs)
    cmd.extend([
        "-c:a", "libopus",
        "-ar", str(OUTPUT_RATE_HZ),
        "-b:a", "128000",
        "-stream_group",
        "type=iamf_audio_element:id=1:st=0:st=1:st=2:st=3:"
        "audio_element_type=scene,layer=ch_layout=ambisonic 1:ambisonics_mode=mono",
    ])

    if object_path is not None:
        mix_lufs = _format_iamf_loudness(
            min(bed_loudness.integrated_lufs, object_loudness[0].integrated_lufs)
        )
        mix_peak = _format_iamf_loudness(
            max(bed_loudness.true_peak_dbfs, object_loudness[0].true_peak_dbfs)
        )
        cmd.extend([
            "-stream_group",
            "type=iamf_audio_element:id=2:st=4,layer=ch_layout=mono",
            "-stream_group",
            "type=iamf_mix_presentation:id=3:stg=0:stg=1:"
            "annotations=en-us=MinimapPR IAMF,"
            "submix=parameter_id=100:parameter_rate=48000:default_mix_gain=0.0|"
            "element=stg=0:headphones_rendering_mode=binaural:"
            "annotations=en-us=Ambisonics:parameter_id=101:parameter_rate=48000:"
            "default_mix_gain=0.0|"
            "element=stg=1:headphones_rendering_mode=binaural:"
            "annotations=en-us=Bird Object:parameter_id=102:parameter_rate=48000:"
            "default_mix_gain=0.0|"
            f"layout=sound_system=stereo:integrated_loudness={mix_lufs}:"
            f"digital_peak={mix_peak}",
        ])
    else:
        cmd.extend([
            "-stream_group",
            "type=iamf_mix_presentation:id=3:stg=0:"
            "annotations=en-us=MinimapPR IAMF,"
            "submix=parameter_id=100:parameter_rate=48000:default_mix_gain=0.0|"
            "element=stg=0:headphones_rendering_mode=binaural:"
            "annotations=en-us=Ambisonics:parameter_id=101:parameter_rate=48000:"
            "default_mix_gain=0.0|"
            f"layout=sound_system=stereo:integrated_loudness={bed_lufs}:"
            f"digital_peak={bed_peak}",
        ])
    cmd.append(str(output_iamf_path))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        output_iamf_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg IAMF encode failed (rc={proc.returncode}): "
            f"{stderr.decode(errors='replace')[-800:]}"
        )


async def _try_iamf_mux(
    video_path: Path,
    iamf_path: Path,
    output_path: Path,
    offset_s: float,
) -> bool:
    """Attempt IAMF-in-MP4 mux; returns True on success."""
    try:
        stream_groups = await _probe_iamf_stream_groups(iamf_path)
    except RuntimeError as exc:
        logger.debug("IAMF mux probe failed: %s", exc)
        return False

    input_stream_indices = sorted({
        stream_index
        for group in stream_groups
        for stream_index in group.stream_indices
    })
    if not input_stream_indices:
        logger.debug("IAMF mux failed: no IAMF audio streams in %s", iamf_path)
        return False
    input_to_output_index = {
        input_stream_index: output_index
        for output_index, input_stream_index in enumerate(input_stream_indices, start=1)
    }
    audio_element_output_group_indices: list[int] = []
    stream_group_args: list[str] = []
    for group in stream_groups:
        if group.group_type == "IAMF Audio Element":
            stream_parts = [
                f"st={input_to_output_index[input_stream_index]}"
                for input_stream_index in group.stream_indices
                if input_stream_index in input_to_output_index
            ]
            if not stream_parts:
                logger.debug("IAMF mux failed: audio element group has no streams")
                return False
            stream_group_args.extend([
                "-stream_group",
                f"map=1={group.group_index}:" + ":".join(stream_parts),
            ])
            audio_element_output_group_indices.append(len(audio_element_output_group_indices))
        elif group.group_type == "IAMF Mix Presentation":
            if not audio_element_output_group_indices:
                logger.debug("IAMF mux failed: mix presentation before audio elements")
                return False
            stg_parts = [f"stg={idx}" for idx in audio_element_output_group_indices]
            stream_group_args.extend([
                "-stream_group",
                f"map=1={group.group_index}:" + ":".join(stg_parts),
            ])

    if not stream_group_args:
        logger.debug("IAMF mux failed: no mappable IAMF stream groups")
        return False

    itsoffset_args: list[str] = []
    if abs(offset_s) > 0.001:
        itsoffset_args = ["-itsoffset", f"{offset_s:.6f}"]

    stream_map_args: list[str] = []
    for input_stream_index in input_stream_indices:
        stream_map_args.extend(["-map", f"1:{input_stream_index}"])
    stream_id_args = ["-streamid", "0:1"]
    for output_stream_index in range(1, len(input_stream_indices) + 1):
        stream_id_args.extend([
            "-streamid",
            f"{output_stream_index}:{output_stream_index + 1}",
        ])

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        *itsoffset_args,
        "-i", str(iamf_path),
        "-map", "0:v:0",
        *stream_map_args,
        "-c:v", "copy", "-c:a", "copy",
        *stream_id_args,
        *stream_group_args,
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


async def _probe_iamf_stream_groups(iamf_path: Path) -> list[IamfStreamGroupLayout]:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v", "error",
        "-show_stream_groups",
        "-of", "json",
        str(iamf_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe IAMF stream-group probe failed (rc={proc.returncode}): "
            f"{stderr.decode(errors='replace')[-500:]}"
        )
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe IAMF stream-group JSON parse failed: {exc}") from exc

    groups: list[IamfStreamGroupLayout] = []
    for group in payload.get("stream_groups", []):
        group_type = str(group.get("type") or "")
        if group_type not in {"IAMF Audio Element", "IAMF Mix Presentation"}:
            continue
        stream_indices = tuple(
            int(stream["index"])
            for stream in group.get("streams", [])
            if "index" in stream
        )
        groups.append(
            IamfStreamGroupLayout(
                group_index=int(group.get("index", len(groups))),
                group_type=group_type,
                stream_indices=stream_indices,
            )
        )
    if not groups:
        raise RuntimeError(f"no IAMF stream groups found in {iamf_path}")
    return groups


def _format_iamf_loudness(value: float) -> str:
    if not math.isfinite(value):
        return "0.0"
    return f"{value:.1f}"


async def _ambix_aac_mux(
    video_path: Path,
    ambix_wav: Path,
    output_path: Path,
    offset_s: float,
) -> None:
    """Mux 4-channel AmbiX WAV as AAC into MP4 for review exports."""
    itsoffset_args: list[str] = []
    if abs(offset_s) > 0.001:
        itsoffset_args = ["-itsoffset", f"{offset_s:.6f}"]

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        *itsoffset_args,
        "-i", str(ambix_wav),
        "-filter:a", "pan=stereo|c0=0.707*c0+0.5*c1+0.5*c2|c1=0.707*c0-0.5*c1+0.5*c2",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "320k",
        "-map", "0:v:0", "-map", "1:a:0",
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
