# Phase 1 sync spike (two Android phones)

Plan: `~/.claude/plans/let-s-make-a-plan-silly-kernighan-…md`, Stage A. Phase 1 gate:
`docs/field_mode_phone_mesh_plan.md`, "Phase 1 — sync spike".

## Tooling

| Piece | Where |
|---|---|
| Waveforms and picks over FFI (surface v3) | `minimappr-ffi/src/chirp_waveforms.rs`, `chirp_detector.rs` |
| Chirp lab on the phone | `field-app` Chirp lab screen / `mode=chirp_lab` intent |
| Laptop driver | `scripts/chirp_lab/run_chirp_lab.py` (`prep`, `start`, `mark`, `lock`, `unlock`, `status`, `stop`, `pull`, `analyse`), `adb_wifi.sh`, `devices.json` |
| Offline analyser | `minimappr-sync-spike` (`sync_spike validate / analyse / analyse-single / summarize`) |
| Raw bundles | `data/field_phase1/<date>_<label>/{pixel7a,sm_a037f}/` (gitignored) |

The analyser's synthetic recovery test (`minimappr-sync-spike/tests/synthetic_pair_recovery.rs`) renders
a 7.3 m pair with ±60 ppm crystals, unrelated monotonic epochs, a body precursor, timestamp jitter and
NTP-level wall-clock error. It recovers the range to 1.5 mm (double-sided) and 1.9 mm (tracked rate),
and θ to a linear model at 0.9 µs RMS. Uncompensated ranging is 55 mm off, which shows why rate
compensation is needed at a 2 s gap. A constant ~1.5 mm comes from the different self and remote pick
timing (envelope peak against leading edge); the d_ii calibration absorbs it.

## Running one matrix cell

```bash
cd scripts/chirp_lab
./run_chirp_lab.py prep                       # volume, rotation lock, sound effects off, dumpsys audio
./run_chirp_lab.py start --label a3_1m_pos1 --band 13000 16000 --source-a unprocessed --source-b voice_recognition
./run_chirp_lab.py mark foam                  # A2: foam → hand → table
./run_chirp_lab.py lock                       # A6: screens off (+ hardening dump)
./run_chirp_lab.py stop
./run_chirp_lab.py pull --matrix-run A3 --tape-distance-m 1.000 --temperature-c 21.5 --pose-a portrait --pose-b portrait
./run_chirp_lab.py analyse ../../data/field_phase1/<date>_a3_1m_pos1
cargo run --release -p minimappr-sync-spike --bin sync_spike -- summarize docs/field_mode_progress/phase1_summary.md data/field_phase1/*/analysis/report.json
```

For the 5–20 m runs, run `./adb_wifi.sh` once over USB, then pass `--wifi` to every command.

## Run matrix (two phones)

| Run | Setup | Gate item | Status |
|---|---|---|---|
| A0 | Survey, each phone alone | Mic info, UNPROCESSED and near-ultrasound support | session.json from the single-phone loopbacks (see below) |
| A1 | L / R / both × portrait / landscape, listener 1 m from the bottom edge | Speaker channel map | not run |
| A2 | foam → hand → table markers, 20 chirps each | Self onset within 10 µs; limiter headroom | not run |
| A3 | Tape 1.000 m, 3 repositionings × 30 rounds | d_ii sum, speaker vectors | not run |
| A4 | 1/5/10/20 m × 13–16 / 15–18 / 18–20 kHz, LFM vs HFM at 5 m | Range against band | not run |
| A5 | Hallway or outdoor, 1–20 m × 50 rounds, temperature logged | 10 cm at 20 m (PARTIAL: 2 models) | not run |
| A6 | 30 min on foam at ~1.5 m, P = 4 s, screens locked | θ residual < 50 µs over 30 min; locked playback | not run |
| A7 | Capture source sweep | Capture path | not run |
| A8 | Volume 0 | Muted phone → `own_chirp_inaudible` | not run |
| A9 | Leakage from the A4 1–2 m runs | Below-11 kHz leakage | not run |

A1–A9 need someone to place and tape-measure the phones, so they were not run in the implementation
session. The per-model tables in `minimappr-core/src/mesh/config.rs`
(`self_path_length_by_platform_m`, `speaker_offset_body_frame_by_platform_m`,
`chirp_output_channel_by_platform`) stay empty until A1 and A3 have run.

## Desk pipeline check (2026-11-25; not a gate run)

`data/field_phase1/20261125_desk_pipeline_check`. Setup: phones about 0.65 m apart on a desk, not
tape-measured. P = 4 s, 13–16 kHz up/down HFM, both phones on VOICE_RECOGNITION, 193 s. The run
exercised the whole chain (`prep → start → mark ×2 → stop → pull → validate → analyse`), and it
surfaced real device behaviour:

- **Bundles:** both are valid, with 48/48 self chirps detected on each phone and no capture gaps or
  writer overflows. On the Pixel, `unprocessed_supported` is false; its near-ultrasound mic and speaker
  are both true. The Pixel plays in LOW_LATENCY, the A03s in POWER_SAVING.
- **Self onset, Pixel 7a:** σ 3.3 µs, max segment shift 7.4 µs. That is inside the 10 µs item, though
  only on one surface. A smooth ~50 µs drift over 3 min remains, plus one 468 µs single-chirp outlier.
- **Self onset, SM-A037F:** the residual of output frame against capture is a sawtooth. It ramps about
  +13.5 µs per chirp (about 3.4 ppm), then snaps back about 350 µs every 10–15 chirps, with occasional
  +250–300 µs single-chirp outliers. This looks like a sample-rate converter slipping samples in the
  POWER_SAVING output path. Ranging measures the self chirp, so playback slips alone don't bias a
  range. They do rule out using output frame indices as the A03s's emission time.
- **Remote picks:** at 0.65 m on a desk the remote SNR was only 20–25 dB. Picks flip between direct
  and reflected paths, so double-sided ranges jump between about 0.42 and 1.2 m around a 0.67 m
  median. One stretch held 640–643 mm. θ residuals (about 0.5 ms) follow the same flips. A3/A4 need a
  clear line of sight off reflective surfaces, or a raised level.

## Live LAN session (Stage B, 2026-11-25)

The Pixel hosts and the A03s joins by NSD, and also by typed IP. The session clock locks in about 6 s
(best RTT 25–35 ms over Wi-Fi). Over two minutes, 23 rounds completed. The rate converges smoothly to
−12.4 ppm, and the raw range sits at 0.664–0.696 m: two clusters, likely multipath, with no
speaker/mic correction. Run it with `scripts/chirp_lab/two_phone_session_e2e.py`, which now also fails
if the peer diagnostics stop advancing.

The first live runs exposed two core bugs, both fixed with regression tests:

1. `host_sync_pipeline.rs`: when a node in a **two-phone** island was flagged with a silent gap, the
   re-solve without it had too few members, and the island was skipped. Its filter never re-seeded, so
   every later round was flagged again and diagnostics froze. Live, they froze at round 4 while 20
   rounds completed. The fix keeps the round's offsets (re-seed) and drops the stalled node's ranges.
   The test is `minimappr-ffi/tests/two_device_session.rs::silent_capture_stall_in_a_two_phone_island_recovers`.
2. `clock_filter.rs`: a silent gap kept the rate even when it rested on two noisy rounds 2 s apart,
   which locked in 281 ppm live. The rate is now reset to the prior unless it is validated (baseline
   ≥ 20 s).

## Gate table

| Item | Status | Evidence |
|---|---|---|
| Range 10 cm at 20 m | not run | |
| θ residual < 50 µs over 30 min | not run | |
| Self onset stable within 10 µs | not run | |
| Locked-screen playback (Android 17) | not run | |
| Muted phone flagged inaudible | not run | |
| Bird-band leakage | not run | |
| Three phone models | PARTIAL by construction | Two models (Pixel 7a, SM-A037F) |

## Bundle schema

The schema of record is `minimappr-sync-spike/src/bundle_schema.rs`. Per phone:

- `capture.wav`: PCM16 mono 48 kHz. Sample 0 is AudioRecord frame 0, on the same timeline as the
  capture timestamps. The header may be incomplete; the file length is trusted.
- `session.json`: `run_id`, `role`, `mode`, `platform_name`, `device{manufacturer, model, sdk_int,
  fingerprint}`, `schedule{period_ms, offset_ms, lead_ms, waveform{kind, band_low_hz, band_high_hz,
  duration_s}, channel, level_dbfs}`, `capture{requested_source, actual_source, buffer_size_frames,
  sample_rate_hz}`, `playback{…}`, `active_microphones`, `unprocessed_supported`,
  `near_ultrasound{mic, speaker}`, `routed_input`, `routed_output`, `volume`, `started_ns`,
  `started_wall_ms`.
- `events.jsonl`: each event carries `t_ns` (System.nanoTime) and `type`. The types are:
  - `emission` (`emission_index`, `output_frame_index`, `target_ns`, `played`, `channel`, `level_dbfs`)
  - `capture_timestamp` (`frame_position`, `nano_time`, `wall_ms`) and `playback_timestamp`
  - `attitude` (`quaternion` w, x, y, z)
  - `screen`, `route_change`, `capture_gap`, `marker`, `thermal`, `volume`, `writer_overflow`

Per run, the laptop writes `run_manifest.json` at pull time. It holds `label`, `matrix_run`,
`tape_distance_m`, `temperature_c`, `surface` and `marker_surfaces`. It also has
`roles{A|B: {serial, dir, self_path_m, speaker_offset_world_m, pose}}`, plus `notes`.

How rounds are paired: A chirps at k·P and B at k·P + P/2, each on its own output frames. A round is
(A_k, B_j, A_{k+1}), where B_j is the B chirp that A heard between its k-th and (k+1)-th chirps. The
first remote matches are found by mapping across the phones by wall clock (NTP accuracy, ±0.4 s
window). After that, the median offset narrows each window to ±50 ms. The two directions get separate
medians, because their flight times enter with opposite signs.
