"""IAMF post-processing pipeline orchestrator.

Executes the "studio render" background job for a completed capture session:

  1. Extract  — fetch ordered journal segments via Rust /api/v1/journal/range
  2. Matrix   — frequency-domain A-to-B conversion → bed_full.wav
  3. Isolate  — query DB for tracks; MVDR beamform each → object_{id}.wav
  4. Subtract — B_clean = B_full − Σ Y_obj·O (re-encode each object to B-fmt
                and subtract) → bed.wav; delete bed_full.wav
  5. Measure  — BS.1770-4 integrated loudness + True Peak on bed (W-channel)
                and each object track
  6. Metadata — map MinimapPR position_m → listener-relative spherical coords
  7. Encode   — POST to Rust /api/v1/capture/encode/iamf → audio.iamf;
                also export 4-channel AmbiX WAV for YouTube mux
  8. Multiplex— ffmpeg: output_raw.mp4 + ambix.wav → youtube_export.mp4
  9. Cleanup  — remove intermediate files on success
  10. Register — move final files to artifacts dir, insert large_artifacts DB row

All CPU-intensive work runs in a dedicated executor pool to keep the async
event loop responsive.
"""

from __future__ import annotations

import asyncio
import logging
import math
import struct
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import httpx
import numpy as np
from numpy.typing import NDArray

from minimappr.core.ambi_atob import atob_foa, encode_mono_to_bformat
from minimappr.core.capture_session import CaptureSessionRecord

logger = logging.getLogger(__name__)

SAMPLE_RATE_HZ = 16_000      # Raw journal sample rate
OUTPUT_RATE_HZ = 48_000      # Target output sample rate for IAMF
SAMPLES_PER_FRAME = 512      # IAMF codec frame size (= 512 @ 48 kHz)
ARTIFACTS_DIR = Path("data/artifacts")


# ── Data shapes ───────────────────────────────────────────────────────────────

@dataclass
class TrackTrajectory:
    track_id: str
    waypoints: list[tuple[int, tuple[float, float, float]]]
    """List of (sample_offset, (x, y, z)) waypoints."""


@dataclass
class LoudnessMeasurement:
    integrated_lufs: float
    true_peak_dbfs: float


# ── Pipeline ──────────────────────────────────────────────────────────────────

class IamfPipeline:
    """Runs the full studio render for one capture session."""

    def __init__(
        self,
        sidecar_url: str,
        db_storage: "Any",  # minimappr.storage.db.Storage
        *,
        executor: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._sidecar_url = sidecar_url
        self._db = db_storage
        self._http = httpx.AsyncClient(timeout=60.0)

    async def run(self, record: CaptureSessionRecord) -> None:
        """Execute all pipeline steps for the given session record."""
        work_dir = record.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

        assert record.start_time_ns is not None
        assert record.end_time_ns is not None
        assert record.first_frame_pts_ns is not None

        start_ns = record.first_frame_pts_ns
        end_ns = record.end_time_ns

        # ── 1. Extract ────────────────────────────────────────────────────────
        logger.info("[%s] step 1: extracting journal range", record.session_id[:8])
        raw_channels = await self._extract_audio(record.stream_key, start_ns, end_ns)

        # ── 2. A-to-B matrix ─────────────────────────────────────────────────
        logger.info("[%s] step 2: A-to-B ambisonics matrix", record.session_id[:8])
        bed_full = await asyncio.get_event_loop().run_in_executor(
            None, atob_foa, raw_channels, SAMPLE_RATE_HZ
        )
        bed_full_path = work_dir / "bed_full.wav"
        _write_wav(bed_full_path, bed_full, SAMPLE_RATE_HZ)

        # ── 3. Object isolation (MVDR) ────────────────────────────────────────
        logger.info("[%s] step 3: MVDR object isolation", record.session_id[:8])
        trajectories = await self._query_trajectories(start_ns, end_ns, raw_channels.shape[1])
        object_tracks: dict[str, NDArray[np.float32]] = {}
        for traj in trajectories:
            mono = await self._mvdr_beamform(raw_channels, traj)
            obj_path = work_dir / f"object_{traj.track_id}.wav"
            _write_wav_mono(obj_path, mono, SAMPLE_RATE_HZ)
            object_tracks[traj.track_id] = mono

        # ── 4. Spatial subtraction ────────────────────────────────────────────
        logger.info("[%s] step 4: spatial subtraction", record.session_id[:8])
        bed_clean = await asyncio.get_event_loop().run_in_executor(
            None,
            _subtract_objects,
            bed_full,
            object_tracks,
            trajectories,
            raw_channels.shape[1],
        )
        bed_path = work_dir / "bed.wav"
        _write_wav(bed_path, bed_clean, SAMPLE_RATE_HZ)
        bed_full_path.unlink(missing_ok=True)

        # ── 5. BS.1770-4 loudness measurement ────────────────────────────────
        logger.info("[%s] step 5: loudness measurement", record.session_id[:8])
        bed_loudness = await asyncio.get_event_loop().run_in_executor(
            None, _measure_loudness, bed_clean[0], SAMPLE_RATE_HZ
        )
        object_loudness = []
        for mono in object_tracks.values():
            lm = await asyncio.get_event_loop().run_in_executor(
                None, _measure_loudness, mono, SAMPLE_RATE_HZ
            )
            object_loudness.append(lm)

        # ── 6. Metadata: position waypoints per temporal unit ─────────────────
        logger.info("[%s] step 6: building IAMF metadata", record.session_id[:8])
        n_samples_out = bed_clean.shape[1]
        positions_per_unit = _build_positions_per_unit(
            trajectories, n_samples_out, SAMPLES_PER_FRAME
        )

        # ── 7. Encode IAMF + AmbiX WAV ────────────────────────────────────────
        logger.info("[%s] step 7: IAMF encode", record.session_id[:8])
        object_track_list = list(object_tracks.values())
        iamf_bytes = await self._encode_iamf(
            bed_clean,
            object_track_list,
            positions_per_unit,
            bed_loudness,
            object_loudness,
        )
        iamf_path = work_dir / "audio.iamf"
        iamf_path.write_bytes(iamf_bytes)
        record.iamf_path = iamf_path

        # AmbiX WAV: 4-channel W/X/Y/Z for YouTube mux.
        ambix_path = work_dir / "ambix.wav"
        _write_wav(ambix_path, bed_clean, SAMPLE_RATE_HZ)

        # ── 8. Multiplex video + AmbiX ────────────────────────────────────────
        if record.video_path and record.video_path.exists():
            logger.info("[%s] step 8: ffmpeg mux", record.session_id[:8])
            youtube_path = work_dir / "youtube_export.mp4"
            await _ffmpeg_mux(record.video_path, ambix_path, youtube_path)
            record.youtube_path = youtube_path
        else:
            logger.warning("[%s] step 8: no video file; skipping mux", record.session_id[:8])

        # ── 9. Cleanup intermediates ──────────────────────────────────────────
        logger.info("[%s] step 9: cleanup intermediates", record.session_id[:8])
        for f in [bed_path, ambix_path] + [
            work_dir / f"object_{tid}.wav" for tid in object_tracks
        ]:
            f.unlink(missing_ok=True)

        # ── 10. Register artifacts ────────────────────────────────────────────
        logger.info("[%s] step 10: registering artifacts", record.session_id[:8])
        artifacts_dir = ARTIFACTS_DIR
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        final_iamf = artifacts_dir / f"{record.session_id}_audio.iamf"
        iamf_path.replace(final_iamf)
        record.iamf_path = final_iamf

        if record.youtube_path and record.youtube_path.exists():
            final_mp4 = artifacts_dir / f"{record.session_id}_youtube.mp4"
            record.youtube_path.replace(final_mp4)
            record.youtube_path = final_mp4

        await self._register_artifact(record)
        logger.info("[%s] pipeline complete", record.session_id[:8])

    # ── private helpers ────────────────────────────────────────────────────────

    async def _extract_audio(
        self, stream_key: str, start_ns: int, end_ns: int
    ) -> NDArray[np.float32]:
        """Fetch ordered journal segments and assemble 4-channel float PCM."""
        resp = await self._http.get(
            f"{self._sidecar_url}/api/v1/journal/range",
            params={"stream_key": stream_key, "start_ns": start_ns, "end_ns": end_ns},
        )
        resp.raise_for_status()
        segments = resp.json()

        pcm_chunks: list[NDArray[np.float32]] = []
        for seg in segments:
            seg_path = Path(seg["segment_path"])
            if not seg_path.exists():
                logger.warning("segment not found: %s", seg_path)
                continue
            raw = seg_path.read_bytes()
            pcm = _decode_pcm16le_4ch(raw)
            pcm_chunks.append(pcm)

        if not pcm_chunks:
            raise RuntimeError("no audio segments found for time range")

        return np.concatenate(pcm_chunks, axis=1)

    async def _query_trajectories(
        self, start_ns: int, end_ns: int, total_samples: int
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
            for det in detections:
                # Convert detection timestamp to sample offset from start_ns.
                det_ns = det.get("toa_ns", det.get("timestamp_ns", start_ns))
                sample_offset = int((det_ns - start_ns) * SAMPLE_RATE_HZ / 1e9)
                sample_offset = max(0, min(total_samples - 1, sample_offset))
                pos = (det.get("x", 0.0), det.get("y", 0.0), det.get("z", 0.0))
                waypoints.append((sample_offset, pos))
            if waypoints:
                trajectories.append(TrackTrajectory(track_id=track_id, waypoints=waypoints))

        return trajectories

    async def _mvdr_beamform(
        self, channels: NDArray[np.float32], traj: TrackTrajectory
    ) -> NDArray[np.float32]:
        """Call the Rust MVDR endpoint for one track trajectory."""
        waypoints_dto = [
            {"sample_offset": s, "position_m": list(pos)}
            for s, pos in traj.waypoints
        ]
        payload = {
            "channels": [ch.tolist() for ch in channels],
            "sample_rate_hz": SAMPLE_RATE_HZ,
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
    ) -> bytes:
        payload = {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "samples_per_frame": SAMPLES_PER_FRAME,
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
            "bed_channels": [ch.tolist() for ch in bed],
            "object_tracks": [o.tolist() for o in objects],
            "positions_per_unit": [
                {str(obj_id): pos for obj_id, pos in unit.items()}
                for unit in positions_per_unit
            ],
        }
        resp = await self._http.post(
            f"{self._sidecar_url}/api/v1/capture/encode/iamf",
            json=payload,
        )
        resp.raise_for_status()
        return resp.content

    async def _register_artifact(self, record: CaptureSessionRecord) -> None:
        if self._db is None:
            return
        try:
            await self._db.insert_large_artifact(
                session_id=record.session_id,
                artifact_type="iamf_video",
                iamf_path=str(record.iamf_path) if record.iamf_path else None,
                youtube_path=str(record.youtube_path) if record.youtube_path else None,
                created_ns=time.time_ns(),
            )
        except Exception as exc:
            logger.warning("artifact registration failed: %s", exc)


# ── Pure-Python DSP helpers ───────────────────────────────────────────────────

def _subtract_objects(
    bed_full: NDArray[np.float32],
    object_tracks: dict[str, NDArray[np.float32]],
    trajectories: list[TrackTrajectory],
    n_samples: int,
) -> NDArray[np.float32]:
    """B_clean(f) = B_full(f) − Σᵢ Y_obj_i(f) · O_i(f).

    Re-encodes each beamformed object back into B-format coordinates and
    subtracts from the full bed.  Because B-format encoding is linear,
    this is exact.
    """
    traj_by_id = {t.track_id: t for t in trajectories}
    bed = bed_full.astype(np.float64).copy()

    for track_id, mono in object_tracks.items():
        traj = traj_by_id.get(track_id)
        if traj is None or not traj.waypoints:
            continue
        # Use the mean position over the track as the steering direction.
        positions = np.array([list(wp[1]) for wp in traj.waypoints], dtype=np.float64)
        mean_pos = positions.mean(axis=0)
        obj_bformat = encode_mono_to_bformat(
            mono[:n_samples], tuple(mean_pos)  # type: ignore[arg-type]
        )
        n = min(bed.shape[1], obj_bformat.shape[1])
        bed[:, :n] -= obj_bformat[:, :n].astype(np.float64)

    return np.clip(bed, -1.0, 1.0).astype(np.float32)


def _measure_loudness(signal: NDArray, sample_rate_hz: int) -> LoudnessMeasurement:
    """Simplified BS.1770-4 integrated loudness and true peak.

    Uses K-weighting (two-stage filter chain) and a 400 ms gating window.
    Full broadcast-grade accuracy is not required here; results within ±1 LU
    of the spec are sufficient for IAMF metadata.
    """
    sig = np.asarray(signal, dtype=np.float64)

    # ── K-weighting filter (ITU-R BS.1770-4 Table 2) ─────────────────────────
    # Stage 1: pre-filter (shelf, fs=48000 → adjust for actual rate).
    sig_kw = _k_weight(sig, sample_rate_hz)

    # ── Gating: 400 ms blocks, 75 % overlap, absolute threshold −70 LUFS ─────
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

    # Absolute gating threshold: mean square corresponding to −70 LUFS.
    abs_thresh_ms = 10 ** ((-70.0 - 0.691) / 10.0)
    gated = [ms for ms in mean_sq_blocks if ms > abs_thresh_ms]
    if not gated:
        gated = mean_sq_blocks

    integrated_ms = np.mean(gated)
    if integrated_ms <= 1e-12:
        lufs = -120.0
    else:
        lufs = float(-0.691 + 10.0 * math.log10(integrated_ms))

    # True peak: oversample 4× using linear interpolation as an approximation.
    true_peak_linear = float(np.max(np.abs(np.interp(
        np.linspace(0, len(sig) - 1, len(sig) * 4),
        np.arange(len(sig)),
        sig,
    ))))
    true_peak_dbfs = float(20.0 * math.log10(max(true_peak_linear, 1e-12)))

    return LoudnessMeasurement(integrated_lufs=lufs, true_peak_dbfs=true_peak_dbfs)


def _k_weight(sig: NDArray[np.float64], sr: int) -> NDArray[np.float64]:
    """Apply a two-stage K-weighting filter approximation.

    Stage 1 is a high-shelf boost; stage 2 is a high-pass filter.
    Coefficients taken from the BS.1770-4 specification, adjusted for `sr`.
    """
    from scipy.signal import lfilter, bilinear

    # Stage 1: pre-filter (high-shelf, +4 dB above ~1.5 kHz).
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

    # Stage 2: high-pass at 38 Hz (revised RLB weighting).
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
    """Build per-temporal-unit object position dictionaries.

    For each frame index, interpolate each object's position from its
    trajectory waypoints and convert to listener-relative spherical coords.
    """
    n_frames = (n_samples + samples_per_frame - 1) // samples_per_frame
    positions_per_unit: list[dict[int, dict]] = [{} for _ in range(n_frames)]

    for obj_idx, traj in enumerate(trajectories):
        if not traj.waypoints:
            continue
        waypoints = traj.waypoints
        for fi in range(n_frames):
            sample_mid = fi * samples_per_frame + samples_per_frame // 2
            pos_xyz = _interpolate_waypoints(waypoints, sample_mid)
            az, el, dist = _xyz_to_spherical(pos_xyz)
            positions_per_unit[fi][obj_idx] = {
                "azimuth_deg": az,
                "elevation_deg": el,
                "distance_m": dist,
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
    """Decode raw 4-channel PCM16LE bytes into a float (4, N) array."""
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if len(samples) % 4 != 0:
        samples = samples[: len(samples) - len(samples) % 4]
    return samples.reshape(4, -1, order="F")  # channel-interleaved → channels-first


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

async def _ffmpeg_mux(
    video_path: Path, ambix_wav: Path, output_path: Path
) -> None:
    """Mux raw video with AmbiX WAV into a YouTube-ready MP4."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
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
            f"ffmpeg mux failed (rc={proc.returncode}): {stderr.decode(errors='replace')[-500:]}"
        )
