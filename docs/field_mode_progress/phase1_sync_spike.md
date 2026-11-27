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

A phone that chirps from its earpiece (Pixel 10 Pro XL, Right) needs its d_ii at pull time
(`--self-path-b 0.16`) and its speaker vector in the run frame (`--speaker-offset-b 0.16 0 0` with the
bottom edges facing).
Otherwise the self pick can land on the body precursor (see 2026-11-26).

## Run matrix (two phones)

| Run | Setup | Gate item | Status |
|---|---|---|---|
| A0 | Survey, each phone alone | Mic info, UNPROCESSED and near-ultrasound support | session.json from the single-phone loopbacks (see below) |
| A1 | L / R / both × portrait / landscape, listener 1 m from the bottom edge | Speaker channel map | Pixel 7a: right = earpiece (by ear, 2026-11-25), so Left. Pixel 10 Pro XL: right = earpiece (body precursor, then confirmed by ear, 2026-11-26), and Right is used (Left ranges badly, see below) |
| A2 | foam → hand → table markers, 20 chirps each | Self onset within 10 µs; limiter headroom | not run |
| A3 | Tape 1.000 m, 3 repositionings × 30 rounds | d_ii sum, speaker vectors | not run |
| A4 | 1/5/10/20 m × 13–16 / 15–18 / 18–20 kHz, LFM vs HFM at 5 m | Range against band | not run |
| A5 | Hallway or outdoor, 1–20 m × 50 rounds, temperature logged | 10 cm at 20 m (PARTIAL: 2 models) | not run |
| A6 | 30 min on foam at ~1.5 m, P = 4 s, screens locked | θ residual < 50 µs over 30 min; locked playback | run at 1.000 m, flat on backs: Pixel 7a + A03s (2026-11-25), Pixel 7a + Pixel 10 Pro XL (2026-11-26) |
| A7 | Capture source sweep | Capture path | not run |
| A8 | Volume 0 | Muted phone → `own_chirp_inaudible` | not run |
| A9 | Leakage from the A4 1–2 m runs | Below-11 kHz leakage | not run |

The per-model tables in `minimappr-core/src/mesh/config.rs` (`self_path_length_by_platform_m`,
`speaker_offset_body_frame_by_platform_m`, `chirp_output_channel_by_platform`) hold the Pixel 7a
channel and the Pixel 10 Pro XL channel, d_ii and speaker vector. The rest wait on A1 and A3.

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

## 30-minute locked soak at 1.000 m (2026-11-25, A6 + A3 calibration attempt)

`data/field_phase1/20261125_soak_1m`. Setup: both phones flat on their backs, screens up, bottom edges
1.000 m apart. The tape is accurate to about a centimetre (a few mm either way). 19.89 °C, P = 4 s,
13–16 kHz up/down HFM, VOICE_RECOGNITION on both, right channel, screens locked for the whole
1858 s run.

- **Run health:** 464/465 emissions, no capture gaps, writer overflows or playback underruns. Both
  report the bottom mic as active (Pixel `builtin_mic_1` at y = 5.8 mm). The Pixel's capture
  timestamp fit residual is 1.29 ms; ranging works in the sample domain, so this does not affect it.
- **Locked playback:** chirps continued with the screens off for 30 min. Pixel: 0 of 464 self chirps
  missed. A03s: 5 of 465 missed. The analyser marks the A03s FAIL for those 5.
- **Range is multimodal:** the levels are 1.029 / 1.072 / 1.109 / **1.148** (253 rounds) /
  **1.189** (154 rounds) m. They are spaced 41.5 mm, which is one 240 µs (11.5-sample) hop of a
  single pick. Within a level the spread is about 1 mm. The hops are in the **A03s self pick only**
  (92/457 rounds deviate ±240 µs from the local median). The Pixel self pick and both remote picks
  are clean.
- **Absolute offset:** the raw range is about 13–15 cm longer than the tape, and the implied
  (d_AA + d_BB)/2 is −0.128 m, which is not physically possible. With both bottom mics active,
  two-way ranging should read close to 1.0 m regardless of where the speakers are. The likely cause
  is self picks arriving early. **Confirmed after the run: the Pixel's right channel drives the
  earpiece**, about 14 cm from its bottom mic. Its self chirp most likely reaches the mic through the
  body before the airborne path arrives. `chirp_output_channel_by_platform` now maps `pixel_7a` to
  Left, and so does `devices.json`. **The run cannot be used as a d_ii calibration.**
- **Clock:** the rate is −12.24 ppm. Round to round, θ moves by a median of 2.6 µs. All rounds
  under one linear fit give an RMS of 92 µs, because each A03s self-pick hop shifts θ by ~120 µs.
  **Within one pick mode, a 30-minute linear fit leaves an RMS of 18–29 µs and a p95 of 33–49 µs**,
  which is inside the 50 µs item. Whether the gate passes depends on fixing the self pick. The
  remaining wander is about ±80 µs over 30 min, a slow drift of under 0.1 ppm.
- **Leakage:** 2–11 kHz is at the noise floor (−0.2 dB median) on both. PASS.

Next: make the self pick robust. Try the first-arrival detector for self chirps, or a lower level
or volume so the self chirp does not limit. Then repeat the soak with the phones raised and upright.

## Two Pixels at 1.000 m (2026-11-26)

Pixel 7a (A) and Pixel 10 Pro XL (B, SDK 37, `pixel_10_pro_xl`). Both flat on their backs, screens
up, bottom edges facing and 1.000 m apart (cm-level tape). P = 4 s, 13–16 kHz up/down HFM,
VOICE_RECOGNITION on both, temperature not measured (20 °C assumed). Runs are in
`data/field_phase1/20261126_*`.

| Run | B channel | Volume | Range (double-sided) | Robust σ | θ linear-fit RMS |
|---|---|---|---|---|---|
| `pixels_check` (160 s) | left | 12/25 | 0.948 m | 8.8 mm | 70 µs (one 527 ms silent drop) |
| `pixels_v18_left` (170 s) | left | 18/25 | 0.913 m | 11.6 mm | 25.8 µs |
| `pixels_v18_b_right` (180 s) | right | 18/25 | **0.9863 m** | **0.1 mm** | **0.89 µs** |
| `pixels_soak_1m` (30 min, locked) | right | 18/25 | **0.9861 m** | **0.6 mm** | **5.7 µs** |

The right-channel rows use d_BB = 0.16 m and B's speaker 0.16 m beyond its mic along the axis (the
manifest's `self_path_m` and `speaker_offset_world_m`). With the 2 cm default, the soak reads 1.037 m
over three levels (see below).

- **The A03s problems are gone.** Self σ is 0.2–1.9 µs on both Pixels, with no self-pick hops,
  locked playback misses 0 of 458 chirps on each phone, and there are no writer overflows. The clock
  rate is −7.4 ppm.
- **Clock gate passes.** Over 30 min, the 90 s heartbeat prediction residual has a p95 of 9.0 µs and
  an RMS of 3.4 µs. The A03s soak measured 92 µs RMS, from self-pick hops.
- **The Pixel 10's right channel is the earpiece** (confirmed by ear after the runs). Its self recording shows a body-borne precursor
  (−8 dB) 22.3 samples before the airborne arrival. That puts the speaker about 0.16 m from the bottom
  mic. With the 2 cm default d_ii, the precursor-rejection window is about 21.5 samples, so the
  precursor sits right at its edge. Over the soak the self pick moved from the airborne peak, to
  halfway, to the precursor, at about 30 s and about 17 min. That gave range levels of
  0.996 / 1.036 / 1.074 m, each step half the 22-sample lead in range. With d_BB = 0.16 m the
  window covers it, and the soak holds one level at σ 0.6 mm. The 3-minute run and the soak agree to
  0.2 mm. The remaining −14 mm against the tape is within the tape, the unmeasured temperature
  (~1.7 mm per °C) and the 7a's unmeasured lateral speaker offset. `config.rs` now carries
  `pixel_10_pro_xl` → Right, d_ii 0.16 m, speaker [0, 0.16, 0] m.
- **On the left channel (presumably the bottom speaker), the Pixel 10 hears the 7a as three copies**:
  0 dB at about +45 samples, −6 dB at about +20 and −9 to −12 dB at 0. The first copy is right at
  the detector's −10 dB first-arrival acceptance, so the B←A pick flips between copies with level and
  from round to round. The range moved from 0.948 m at volume 12 to 0.913 m at volume 18. On the
  right channel the same B←A reception is a single clean arrival, even though the 7a's emission did
  not change. The cause is not known. Logging `activeMicrophones` while playback runs would show
  whether the Pixel switches capture mics when its bottom speaker is in use.
- **Silent capture drop (Pixel 10, `pixels_check`).** At 46.6 s the audio server stalled both
  directions: playback lost about 350 ms and capture lost 527 ms. Every read succeeded, so the app
  logged no `capture_gap`. Only the capture timestamps show it (frames advanced 22 848 in 1.0 s of
  monotonic time). It coincided with a laptop `status` (dumpsys) call and a system memory trim. It did
  not recur in the 30-minute soak, which had no polling. The Pixel 10's capture timestamps also wander
  by one 256-frame HAL burst (5.33 ms).
- **Prep bug.** `cmd media_session volume --set` does nothing on Android 17. The 2026-11-25 soak ran
  the 7a at 19/25, not 12. `prep` now uses `cmd audio set-volume` and checks the readback.
- **The Pixel 10 charges as "AC" over this USB link,** so `svc power stayon usb` does not keep it
  awake. Use `svc power stayon true`, and unlock its secure keyguard by hand before `start`. `start`
  now checks for the keyguard.

Fixes from this session:

- Analyser: `capture_timeline.rs` fits the capture timeline in segments, splitting at timestamp steps
  over 15 ms. A single fit had put every self prediction after the drop 0.5 s off: 33 of 40 missed.
  The wall-clock mapping goes through it too. Remote windows are centred on the mapped sample, so a
  drop no longer shifts them. `capture_continuity.rs` finds breaks in the self-pick interval against
  the output-frame interval (tolerance 2 samples). It separates single-pick hops (outliers) from
  lasting steps, leaves out rounds that overlap a break, and numbers capture segments. θ fits,
  tracked-rate ratio and onset σ remove the steps between segments (`stitch_segments`). The report
  lists silent drops, timeline breaks and self-pick hops for each phone.
- App: `SilentCaptureDropDetector` logs a `capture_gap` with `source: timestamp_step` and passes it
  to the session as a gap. Failed reads log `source: failed_read`.
- Core FFI: the live `CaptureClockAnchor` jump threshold is raised from 5 ms to 15 ms. The Pixel 10's
  5.33 ms burst wander would otherwise restart its fit every few seconds.

Re-analysing the 2026-11-25 A03s soak with the segmented analyser classifies 40 of its ±240 µs hops
as self-pick hops. 35 longer pick-mode switches count as timeline steps. 167 rounds are left out,
and the 291 kept range at σ 1.7 mm.

## Two Pixels, rounds 2–4: channel map, picks and speaker geometry (2026-11-26 evening)

Same phones, flat and screen up, tape bottom edge to bottom edge, 66–68 °F (passed as
`--temperature-c`). Runs: `20261126_{b1*,m1_*,pads_*,topmic_*,d180_*,b4*,b5_*,b6_*,r3_*,r4_*}`.
This supersedes the channel conclusions of the section above.

**Channel map.**
- **Pixel 7a:** Right is the earpiece (by ear). Its bottom speaker (Left) reaches the other phone
  14 dB weaker, with a second copy ~23 samples later, and ranged 0.93–1.04 m between identical runs.
  The first soak's 0.986 m used it, so that number was luck.
- **Pixel 10 Pro XL:** Left is the bottom speaker: one clean arrival in every pose, level-independent
  within 2 mm at volumes 8–18. Its earpiece (Right) self pick moved 8–26 cm between runs.
- **Capture:** both phones record from the bottom mic. Covering the 7a's top changed levels by
  under 1 dB.
- **Config:** 7a → Right, Pixel 10 → Left.

**Pose hazards.** A phone whose own body lies between its mic and the other phone gets remote picks
that are wrong. This happened twice:
- Pixel 10 reversed, on the table: ~2 ms early, through the table.
- 7a reversed, on pads: lead 840 samples, bimodal ranges.

Bottoms facing is the only clean pose measured.

**Pick fixes (`first_arrival.rs`).** Both failures came from envelope sidelobes the 7a earpiece
channel raises:
- **Self picks.** The 7a's self recording has a second peak 24 samples (13–16 kHz) or 12 samples
  (12–18 kHz) before the main one. It sits on the template's first autocorrelation sidelobe, raised
  3–10 dB by the near-field channel. It is not an up/down split and not a body path.
  - Its level depends on placement: −9 dB in round 3, the strongest peak in round 4.
  - The old body-precursor logic picked whichever side of it the d_ii window allowed. The range
    moved 40 mm with volume (d_ii 0.02) and was 87 mm off in round 4 (d_ii 0.20).
  - Self picks now take the strongest peak (`SelfRecordingPick::StrongestPeak`), and d_ii only
    enters the range equation.
- **Remote picks.** The Pixel 10's down-half saw the raised sidelobe at −9.9 dB, 0.2 dB over the old
  1.6× margin. The pick jumped 11 samples in 2–3 of 20 rounds (and 1 of 49 live).
  - `sidelobe_rejection_margin` is now 2.0. Real early arrivals clear their sidelobe by 27–30 dB.
- **Effect on old runs.** Re-analysed, Pixel 10 Left picks are byte-identical. Its earpiece runs no
  longer break into 20 excluded rounds.

**Band 12–18 kHz.** At 13–16 kHz a real second arrival ~18 samples after the first merges into the
main lobe. The 7a earpiece range then depends on placement by ~13 mm. At 12–18 kHz the two
placements and both distances (1.0 and 1.8 m) agree within 0.2 mm, and leakage into the 2–11 kHz
bird band passes. The core scheduler, the Kotlin defaults and the chirp-lab driver now default to
12–18 kHz.

**d_ii and speaker vector.** The 7a's speaker vector is [0, 0.145, 0] m, estimated from the phone's
dimensions (earpiece slot to bottom mic). d_ii is 0.095 m, an effective value chosen so that
bottoms-facing runs read tape + v/2. The leftover pick-timing constant goes into d_ii; the
orientation-dependent speaker term stays separate. The orientation split itself was not measured,
because the reversed pose fails (above).

**Live speaker geometry (`mesh/speaker_geometry.rs`).**
- **Attitude:** phones report it per round (`ChirpArrivalReport.reporter_attitude`: rotation-vector
  quaternion and heading accuracy). The protocol minor version is now 1, so both phones need the
  same build.
- **Correction:** the host applies the exact `correct_for_speaker_mic_offsets` only when the pair
  bearing is known from a GPS-anchored layout, chirality is settled, and both headings are within
  30°.
- **Otherwise:** range σ widens by ½(|v_A|+|v_B|) and clock-offset σ by (|v_A|+|v_B|)/(2c), with no
  guessed correction.
- **Today it always widens:** the live path sends no GNSS fixes, and indoors both Pixels report 60°
  heading accuracy. Their yaws are also 74° apart instead of the true ~180°.

**Final check (round 4, 1.00 m, 68 °F, final build).**

| Run | Range | σ |
|---|---|---|
| `r4_final_vol18` chirp lab, corrected | 1.0026 m | 0.5 mm |
| `r4_final_vol12` chirp lab, corrected | 1.0029 m | 0.3 mm |
| `r4_live_final` live LAN, 50 rounds (no correction live) | 1.0766 m (predicted 1.0751 = corrected + v/2) | 0.6 mm, 0 jumps |
| Before the fixes: `r4_base_vol18_b13_16` (old detector) | 1.0371 m raw; the 12–18 kHz run read 1.1250 m | |

**Other results.**
- **Volume 0:** flagged inaudible on both phones (B5).
- **Gaps:** the live sessions had no capture gaps.

**Open.**
- **GNSS in the live path.** Needed before the live correction can ever apply.
- **Earpiece placements.** The second arrival is only 2.6 dB below the first in round 4. A placement
  where it wins would move the 7a earpiece pick by ~18 samples (not observed).
- **12–18 kHz checks.** A volume sweep beyond 12/18, and a 30-minute soak on the final build.
- **Commit.** Nothing is committed.

## Gate table

| Item | Status | Evidence |
|---|---|---|
| Range 10 cm at 20 m | not run | |
| θ residual < 50 µs over 30 min | PASS (Pixel 7a + Pixel 10 Pro XL); FAIL with the A03s | pixels_soak_1m: heartbeat p95 9.0 µs, linear-fit RMS 5.7 µs over 1824 s. soak_1m (A03s): self-pick hops |
| Self onset stable within 10 µs | PASS on one surface (both Pixels); FAIL (A03s) | pixels_soak_1m: self σ 0.3 / 1.0 µs, given the right d_ii for an earpiece. soak_1m: A03s hops ±240 µs. A2 (surface changes) not run |
| Locked-screen playback (Android 17) | PASS (Pixel 7a, Pixel 10 Pro XL, SDK 37); A03s 5/465 missed | pixels_soak_1m: 0/458 missed on each Pixel. soak_1m |
| Muted phone flagged inaudible | PASS (Pixel 7a, Pixel 10 Pro XL) | b5_vol0: `muted_phone_inaudible` on both |
| Bird-band leakage | PASS (13–16 and 12–18 kHz) | soak_1m: −0.2 dB over floor, 2–11 kHz; r4_final: ≤ 1.1 dB at 12–18 kHz |
| Three phone models | Three models measured (Pixel 7a, SM-A037F, Pixel 10 Pro XL), never three at once | |

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
  - `capture_gap` (`source`: `failed_read`, or `timestamp_step` for a silent drop the capture
    timestamps show; `expected_frame`, `got_frame`, `missing_frames`)
  - `screen`, `route_change`, `marker`, `thermal`, `volume`, `writer_overflow`

Per run, the laptop writes `run_manifest.json` at pull time. It holds `label`, `matrix_run`,
`tape_distance_m`, `temperature_c`, `surface` and `marker_surfaces`. It also has
`roles{A|B: {serial, dir, self_path_m, speaker_offset_world_m, pose}}`, plus `notes`.

How rounds are paired: A chirps at k·P and B at k·P + P/2, each on its own output frames. A round is
(A_k, B_j, A_{k+1}), where B_j is the B chirp that A heard between its k-th and (k+1)-th chirps. The
first remote matches are found by mapping across the phones by wall clock (NTP accuracy, ±0.4 s
window). After that, the median offset narrows each window to ±50 ms. The two directions get separate
medians, because their flight times enter with opposite signs.
