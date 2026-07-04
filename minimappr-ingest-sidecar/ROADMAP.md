# IAMF and Ambisonics Design Plan
Here is the technical solution architecture for integrating studio-grade Ambisonics, IAMF (object-based audio), and synchronized video capture into MinimapPR.

This plan leverages the existing Rust ingest journal for high-fidelity raw audio access, the Python backend for spatial metadata reasoning, and ffmpeg for video capture and final multiplexing.

## High-Level Architecture Overview
To achieve "studio-grade" outputs while keeping the live pipeline lightweight, the recording subsystem will operate completely asynchronously from the live tracking pipeline. It uses a Session-Based Capture Model:

Live Capture Phase: Pin the Rust journal and record raw video to disk.
Post-Processing Phase: Extract high-fidelity audio, generate spatial metadata, encode to IAMF/Ambisonics, and multiplex with the video.

## Core Components

### A. Capture Session Manager (Python)
A new state machine manager in Python that coordinates recordings.

State: Maintains CaptureSession records in SQLite (States: PENDING, RECORDING, PROCESSING, COMPLETED, FAILED).
Rust Journal Pinning: When a session starts, it issues a `StreamRangeLease` request to the Rust sidecar for the relevant array node's `stream_key` starting at `start_time_ns`. This range pin prevents the tmpfs rolling ring from evicting any newly written segments whose `toa_ns` falls within the active recording window.
Lease Hard Cap: The Rust side enforces an absolute, non-renewable `end_ns = start_ns + MAX_DURATION` (e.g., 5 minutes) on every `StreamRangeLease` at creation time. Heartbeat signals from Python are liveness checks only — receiving them cannot extend `end_ns` and any attempt to do so must be rejected by the Rust sidecar. Missing a heartbeat triggers immediate GC of the lease on the Rust side, instantly freeing tmpfs memory. The Python session manager may not override this cap under any circumstances.
Release: Once the session stops and audio is extracted, the pin is released.

### B. Video Capture Subsystem
A lightweight, headless subprocess runner for ffmpeg.

Source / Execution: Configurable per deployment. On Raspberry Pi (Bookworm), this uses `libcamera-vid -o - --codec h264 | ffmpeg -f h264 -i pipe:0 -c:v copy output_raw.mp4` to leverage the modern libcamera stack. On macOS, this uses `AVFoundation` with no live-view UI overhead.
Hardware-Accelerated Codec: The video encoder is selected from a deployment profile to prevent video encoding from competing with the audio ingest pipeline. Defaults: `h264_v4l2m2m` on Linux/Pi (V4L2 M2M kernel encoder), `h264_videotoolbox` on macOS (VideoToolbox), `libx264 -preset veryfast -crf 18` as a software fallback. The ffmpeg subprocess is fully process-isolated and shares no thread pool or memory with the Rust audio worker.
Synchronization: Pipes ffmpeg's progress output (`-progress pipe:2 -stats_period 0.1`) and parses the first `pts_time=` line to capture the exact timestamp of the first actual video frame, avoiding the 50–500ms driver init latency. Audio is extracted strictly aligned to this first-frame timestamp rather than the process spawn time.

### C. Studio-Grade Audio Extraction
To get studio-grade audio, we bypass the real-time classification downsampling and use the raw, uncompressed PCM data from the pinned Rust journal.

Raw Ambisonic Bed (Frequency-Domain A-to-B): For the ambient "bed", we pull the 4-channel Sirith array data. Because the capsules are omnidirectional (measuring pressure, not direction), a simple time-domain matrix fails. Instead, we derive the encoding matrix E at runtime from `SIRITH_MIC_POSITIONS_M`. Because MK3 sits at [0,0,0] in that constant — not the geometric centroid (approximately [0.016, 0.025, 0.010]) — the centroid shift must be applied to all capsule positions before computing unit direction vectors, otherwise the matrix produces incorrect steering. The frequency-domain pseudoinverse of E is applied per FFT bin with frequency-dependent Tikhonov regularization (λ(f) = λ₀·(f_ref/max(f, f_ref))², with f_ref ≈ 120 Hz, λ₀ ≈ 1e-3) to prevent blowup near DC where velocity components are physically near zero. The X, Y, and Z velocity components are low-passed above ~3.4 kHz (= c / (2 × 0.05 m edge length)) to prevent spatial aliasing, while W remains full-bandwidth. The output is 4-channel B-format in ACN/SN3D (AmbiX) normalization.

Instead of a fixed formula, the regularization strength will be dynamically adapted based on the condition number of the encoding matrix at each frequency bin. This is more physically accurate, applying regularization only as much as is needed to counteract the ill-conditioning inherent at low frequencies, resulting in a more natural-sounding bass response

Object Extraction: Object isolation uses a shared `render_mvdr.rs` module placed alongside `gcc_phat.rs` and `srp_phat.rs` in the DSP worker, not as IAMF-only code. This ensures its minimum variance distortionless response (MVDR) calculations are consistent with the live localization pipeline, independently testable, and available to future live beamforming paths. It accepts a trajectory of `(sample_offset, [x,y,z])` waypoints from track history, processes 50% overlapping blocks, and crossfades between steering directions to output a clean mono track per object.

Steering Slew-Rate Limiter: A first-order IIR smoother is applied to the cartesian steering unit vector before computing delays per block: `current = α·target + (1−α)·current`, renormalized to unit length. The smoothing constant α is chosen to limit angular slew to approximately 180°/second, preventing audible phase tearing when the Kalman tracker makes a sudden large position jump between frames.

Track Handoff and Gaps: Inactive tracks fade to silence over ~100 ms; new tracks fade in over ~100 ms. No processing bridges silence gaps between separate acoustic events — a brief silence between two calls is always the correct output. Attempting to steer through the gap would smear unrelated sounds together.

### D. Spatial Subtraction
After each object track is beamformed, its acoustic contribution is still present in the full B-format bed, causing it to be rendered twice at playback: once as a positioned point-source object element, and once diffusely from the ambient bed. To prevent this, each extracted object is re-encoded back into B-format coordinates and subtracted from the full bed before the bed is finalized:

  B_clean(f) = B_full(f) − Σᵢ [ Y_obj_i(f) · O_i(f) ]

where Y_obj_i(f) is the ambisonic encoding vector for the i-th object's steering direction and O_i(f) is the beamformed mono object in the frequency domain. Because B-format encoding is a linear operation, this subtraction is exact rather than approximate. This step runs after all objects have been isolated and before the bed WAV is written to disk.

### E. IAMF Scene Builder & Exporter
Translates MinimapPR tracks into IAMF (Immersive Audio Model and Formats).

Audio Elements:
- A 4-channel Ambisonic B-format bed track for the environment (after spatial subtraction).
- N isolated mono tracks for distinct acoustic objects.

Loudness Measurement (BS.1770-4): Before writing the Mix_Presentation_OBU, a final offline pass computes ITU-R BS.1770-4 integrated loudness (LUFS) and True Peak for the FOA bed (measured on the W channel) and each object mono track independently. These measurements are injected into the `loudness_info` field of Mix_Presentation_OBU before the file is finalized, ensuring YouTube playback compliance (−14 LUFS normalization target) and correct IAMF metadata.

Metadata Generation: Queries the TrackManager / SQLite history for the exact time window. The listener origin is fixed at the mic array centroid. The transform maps `room_xyz` → `listener_relative_xyz` → `spherical(azimuth, elevation, distance)`. Object trajectory waypoints are sampled at the track update rate and written as Parameter_Block_OBUs in each temporal unit.

Before multiplexing, the final extracted audio (both the ambisonic bed and the object tracks) is optionally passed through a high-quality polyphase resampler (e.g., using libsoxr).

Compilation (Custom Rust Writer): Writing IAMF v1.0.0 directly avoids the heavy `iamf-tools` C++ build chain on Raspberry Pi. We use a purpose-built Rust writer (~800 lines) implementing the OBU (Open Bitstream Unit) framing: 1-byte header, LEB128 size, and payload for each OBU type (IA_Sequence_Header, Codec_Config with `ipcm`, Audio_Element for the FOA bed and each object, Mix_Presentation with loudness_info, Temporal_Delimiter, Parameter_Block, and Audio_Frame). The static descriptor OBUs are written once; temporal units repeat at the codec frame rate.

https://github.com/AOMediaCodec/iamf-tools/blob/main/docs/external/encoding_with_external_tools.md#encode-wav-files-to-iamf-with-ffmpeg

YouTube AmbiX Mux: As a backup, the processor simultaneously outputs a derived 4-channel AmbiX (W/X/Y/Z in ACN/SN3D) WAV from the bed. `ffmpeg` multiplexes this standard AmbiX track with the raw video to produce a bulletproof, YouTube-ready spatial audio MP4.

## Data Flow & Execution Sequence

### START Recording (POST /api/v1/capture/start)
1. The API receives the request (with optional video source and deployment profile).
2. Generates a session_id.
3. Sends a `StreamRangeLease` to the Rust sidecar. The Rust side immediately sets the absolute `end_ns` cap; no subsequent call can extend it.
4. Spawns ffmpeg using the deployment-profile hardware codec and parses the first-frame `pts_time=` for the exact `start_time_ns`.

### STOP Recording (POST /api/v1/capture/{id}/stop)
1. Sends SIGTERM (`process.terminate()`) to the ffmpeg process to cleanly finalize the raw MP4 `moov` atom (required on macOS).
2. Records `end_time_ns`.
3. Enqueues the session into a background asyncio task for post-processing so it doesn't block the API.

### Background Post-Processing (The "Studio" Render)
1. **Extract**: Calls a new Rust endpoint `GET /api/v1/journal/range?stream_key=X&start_ns=Y&end_ns=Z` to fetch the ordered segments spanning the time window. Unpins the journal immediately after reading to free tmpfs memory.
2. **Matrix**: Applies frequency-domain A-to-B conversion (centroid-corrected E matrix, Tikhonov-regularized pseudoinverse) to render the full Ambisonic background bed to `bed_full.wav`.
3. **Isolate**: Queries the database for all confirmed tracks within the time window. Runs `render_tvds` with the slew-rate limiter and track handoff rules on each track's trajectory to generate `object_{track_id}.wav`.
4. **Subtract**: Re-encodes each beamformed object into B-format and subtracts all contributions from `bed_full` (B_clean = B_full − Σ Y_obj · O). Writes the result to `bed.wav`. Deletes `bed_full.wav`.
5. **Measure**: Runs a BS.1770-4 integrated loudness and True Peak pass over `bed.wav` (W channel) and each `object_{track_id}.wav`. Holds these values for injection into the IAMF writer.
6. **Metadata**: Maps MinimapPR's `position_m` coordinates to IAMF listener-relative spherical coordinates for the writer.
7. **Encode IAMF & AmbiX**: The Rust IAMF writer emits the custom `.iamf` OBU bitstream with the measured `loudness_info` injected into Mix_Presentation_OBU. Simultaneously exports a 4-channel AmbiX WAV from `bed.wav`.
8. **Multiplex**: Executes `ffmpeg` to combine `output_raw.mp4` with the AmbiX WAV into `youtube_export.mp4`.
9. **Cleanup**: Deletes all intermediate files (`bed.wav`, `object_*.wav`, AmbiX WAV) on success. On FAILED state, actively sweeps the session working directory to remove any leftover multi-gigabyte intermediate files.
10. **Register**: Moves `youtube_export.mp4` and the archival `audio.iamf` into `MinimapPR/data/artifacts` and inserts a `large_artifacts` DB record with `artifact_type="iamf_video"`.

## Safety & Performance Constraints

**Non-Blocking Operation**: Audio processing, beamforming, loudness measurement, and video multiplexing are highly CPU-intensive. All post-processing runs strictly in a dedicated background thread/process pool. The real-time TDOA and ingestion pipelines must never be affected.

**Memory Budgets**: The Rust `StreamRangeLease` hard cap is the sole OOM defense and is enforced by the Rust sidecar, not the Python session manager. The absolute `end_ns` is set at creation and cannot be extended. Missing heartbeats on the Python side trigger immediate Rust-side GC, freeing tmpfs. No secondary Python-side duration enforcement is required or trusted. A UI element (so users control the end runtime) may make sense.

**Hardware Codec Isolation**: The ffmpeg video subprocess must use hardware-accelerated encoding in all production deployments (`h264_v4l2m2m` on Pi, `h264_videotoolbox` on macOS). Software `libx264` at 1080p on a Pi 4 consumes two CPU cores and directly competes with audio ingest. The subprocess is fully process-isolated from the Rust audio worker and must not share any execution resources with it.

**Audio-Video Synchronization**: The raw audio journal operates on absolute ns timestamps (aligned to GPS/NTP). The video capture uses the host OS clock, anchored to the first actual ffmpeg `pts_time`. The post-processor aligns the audio start exactly to this frame, applying a fractional-sample offset if `timing_diagnostics` indicate clock skew between the audio hardware clock and the host wall clock.

**Tests** a short capture unittest should exist that confirms capture works and produces correctly formated files, then cleans up after itself. This will use a synthetic input as in test_birdnet_hybrid_production.py 

## Acceptance Gates:
Validate .iamf with iamf-tools or libiamf, not just byte-shape tests.
Produce a final MP4 carrying IAMF audio, preferably Opus IAMF for YouTube, not AAC AmbiX.
Decide profile limits. IAMF v1.0 Base has only 2 audio elements according to Google’s Eclipsa plugin post, so “FOA bed plus N mono objects” may require v1.1 Enhanced, object flattening, or a different representation. Note only IAMF v1.0 is supported by YouTube at the current time supporting 2 audio elements and 18 audio channels (v1.1 expands support to 28 audio elements and 28 audio channels).
Define exact timestamp semantics for journal segment trimming, video first-frame clock anchoring, drift correction, and sample padding.
Replace JSON bulk transport with files/shared artifacts/streaming IPC.
Add end-to-end synthetic capture tests that run ffprobe plus IAMF decoder validation.

# Future Extension
Allow recording the COP map as part of the saved video (perhaps just as a screen recording channel in ffmpeg)
