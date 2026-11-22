# W1-sim: Layer 1 acoustic world simulator (`minimappr-sim`)

Stream notes for the Field Mode non-phone build-out (`build_out_plan.md`, row W1-sim; `field_mode_test_strategy.md`, Layer 1). Branch `fm/sim`.

## What exists

New standalone crate `minimappr-sim/`, a library plus a binary. It does not depend on `minimappr-core` yet.

```
minimappr-sim render-bundle --scenario x.toml --out bundle.zip [--seed N] [--stored]
minimappr-sim bench --scenario x.toml [--repeats 5]
```

Pipeline: `Scenario` (TOML, serde, `deny_unknown_fields`) → `world::plan_world` (every random draw and chaos knob resolved once, with seeded per-purpose ChaCha streams) → `render::render_world` → `bundle::write_bundle`.

| Module | Responsibility |
| --- | --- |
| `physics/constants.rs` | Named physical constants. The speed-of-sound formula mirrors `minimappr/core/localization.py`, so simulated TDOAs carry no bias relative to what the server assumes |
| `physics/atmosphere.rs` | ISO 9613-1 pure-tone absorption α(f, T, RH, p), plus a per-FFT-bin table |
| `physics/fractional_delay.rs` | Kaiser-windowed sinc, 32 taps, 4096 tabulated phases (quantisation ≤ 1/8192 sample), 8-lane vectorised dot product |
| `physics/biquad.rs` | RBJ biquads, Butterworth cascades, streaming state |
| `physics/geometry.rs` | ENU vectors and piecewise-linear trajectories |
| `scenario/*` | Config structs: site, environment, ground, reflectors, ambient noise, devices, sources, chirps, chaos knobs, render tunables |
| `signals/*` | Call generators (tonal whistle, trill, FM sweep, harsh), WAV file sources (any rate, int or float, mixed to mono and resampled), LFM/HFM chirps, speech babble, clap, fast seeded noise |
| `world/*` | Device planning (clocks, mic geometry, d_ii, motion, GPS error, gated filters, dropouts), emission planning (scripted calls, chirp rounds, knob-generated sources), reflector placement |
| `render/*` | Exact time-varying propagation, per-path absorption filtering, fused receive chain, dropouts |
| `bundle/*` | Zip writer, manifest, WAV (PCM16), flat geo conversion (coefficients mirror `geo.py`). `ground_truth.rs` is the **only** module that knows the ground-truth schema |

## Physics model

- **Propagation.** Receiver i takes sample n at true time t = θᵢ + n / (fs (1 + αᵢ)). Each path contributes g(t) · s(t_e), where the retarded emission time t_e solves t − t_e = L(x_src(t_e), x_rx(t)) / c by fixed-point iteration (tolerance 1e-13 s). Doppler, clock offset and in-clip drift stretch all come out of this one formula. The solve runs every 32 output samples and is linearly interpolated between, which is exact for static geometry with drift.
- **Paths.** Direct; ground image source (plane z = h, broadband reflection coefficient); point reflectors (trunks) with 1/(r₁ + r₂) spreading × coefficient; and, for a phone's own chirp only, a body-conduction precursor at 5000 m/s, with level set by grip (foam −40, hand −26, hard table −14 dB relative to the airborne self path). Random trunks: 3–6 of them, 2–8 m from the device centroid, −18 to −6 dB.
- **Absorption.** Zero-phase, applied in the frequency domain: one forward FFT per emission, then one multiply and inverse FFT per path-length bucket. A bucket's width is set so the error at the emission's band top (the 99.9 % energy frequency) stays within ±`absorption_sharing_tolerance_db`/2 (default 0.1 dB, so ±0.05 dB). Setting it to 0 gives exact per-path filters, which the physics tests use.
- **Receive chain**, fused into one pass per channel: ambient noise (one-pole tilt at 800 Hz) plus white mic self-noise → pocket gate (acoustic, high shelf) → second-mic body shadow + mic HPF/LPF → route-change gate (band limit) → soft-knee tanh limiter on phones, hard clip on tetra nodes → dropouts (reported ones zero-filled, silent ones removed). Each channel runs as 4 interleaved segments, each warmed up for 0.1 s, so the latency-bound IIR recursions overlap. The residual from the warm-up split is below 1e-7.
- **Devices.** `sirith_tetra` uses `SIRITH_MIC_POSITIONS_M`, copied from `minimappr-ingest-sidecar/src/dsp_worker.rs:41` and re-centred on the centroid (this equals `SIRITH_TETRA_SENSOR_OFFSETS_M` in `tests/helpers.py`); its clock is disciplined. `phone`: primary mic at the device origin, speaker at −d_ii along y (d_ii drawn from 1–15 cm), optional second mic at +15 cm with a −3 dB shelf above 4 kHz, clock random within ±20 ms and ±60 ppm.

## Chaos knobs (`src/scenario/chaos.rs`; all off by default)

`sample_drops`, `walking`, `pocket`, `canopy_gps` (canopy heights for knob-generated birds, plus an Ornstein–Uhlenbeck GPS error with σ = 6 m and τ = 60 s on reported positions), `route_change`, `thermal_stall`, `micro_motion`, `muted_speaker`, `speech` (synthetic formant babble or a WAV), `clap`, `intermittent_flock`, `same_band_pair`, `two_speaker_emission`. `scenarios/all_chaos_knobs.toml` turns every one on. Each knob draws from its own named RNG stream, and a test checks that toggling one knob leaves the other draws unchanged.

## Bundle output

The output is a calibration bundle (manifest schema 1, ground truth v2 per `bundle_format_v2.md`): `manifest.json`, `ground_truth.json`, `detections.json` (empty), an optional `expectations.json` (passed through from the scenario's `[expectations]`), `audio/{device}.wav` (PCM16, channel i = `channel_sensor_ids[i]`), and `reference_audio/README.txt`.

- Each node records `audio_start_time_ns` as the session start, i.e. what the device believes. The true offset and drift go in `ground_truth.simulator.devices[]`.
- Each call event carries the v2 keys: `individual_id`, `species` entries (a scenario `label` that differs from `species` becomes the `true` entry, the species a `candidate`, because the loader requires `label` to be among the species labels), and `geometry` = `static` or, for moving sources, `trajectory` with `t_ns` waypoints on the session epoch. Simulator extras: `position_m` (mid-call), `trajectory_m`, `level_db_spl_at_1m`, `generator`, `origin`.
- Every call also gets one echo event per reflector (trunks, ground plane): `is_echo_source`, `echo_of_event_id`, static geometry at the image position seen from the device centroid (point scatterer: along centroid→trunk at r₁ + r₂; ground: mirror image), and `reflector` {`kind`, `position_m`, `extra_delay_ms`, `attenuation_db`}. Event ids must be unique; `plan_world` rejects duplicates before rendering.
- Top-level additive keys: `reflectors` (trunks and the ground plane), `chirps` (schedule, including muted slots and earpiece emissions), `distractors` (speech, clap; these are not scored events), and `simulator` (seed, c, devices: clock truth, true trajectory, reported position, GPS error track, d_ii, speakers, dropouts, gated filters).
- I verified that the current `load_bundle` and `replay.py` accept all of these.

## Results (2026-11-22, Apple M2, 4P + 4E cores)

- `cargo test` (in `minimappr-sim/`): **20 passed** across 8 test binaries.
  - Delay vs analytic band-limited delay, 7 ranges and offsets: worst **1.1e-4 samples**, waveform error ≤ 1.2e-4 relative.
  - Drift ±60 / 23.7 / 0 ppm: measured to < 0.0001 ppm; arrivals within 5e-5 samples; tone frequency scaled by exactly 1/(1 + α).
  - ISO 9613-1 vs the 48 published table values (6 climates × 8 octave bands): all within rounding. Rendered level: 1/r gives 20.0001 dB per decade; absorption over 100 m at 8 kHz is within 0.0001 dB of α·r.
  - Reflection arrivals (direct, ground, trunk) within 8e-5 samples; amplitudes match coefficient/L.
  - Doppler: approaching source at 20 m/s, 2123.4415 vs 2123.4416 Hz; receding receiver, 1883.7345 vs 1883.7344 Hz.
  - Kernel flat within 0.1 dB to 20 kHz at all fractional phases.
  - Self-chirp: precursor present, limiter caps the output at its ceiling.
  - Scenario parsing, determinism, knob expansion, dropout lengths and zero-fill, pocket (−25 dB) and route change (> 40 dB at 12 kHz).
  - Bundle layout and JSON.
  - A 44.1 kHz stereo 24-bit WAV source resamples to unit-RMS mono at 48 kHz with the tone frequency preserved.
- `cargo clippy --all-targets`: clean.
- Render speed (plan + render, release build, `minimappr-sim bench`):

  | Scenario | Best | Median | Single thread |
  | --- | --- | --- | --- |
  | 60 s × 6 phones (8 channels, 4 chirp rounds, 10 calls, ground + 5 trunks) | 162 ms | 167 ms | ≈ 490 ms |
  | 60 s × 6 tetra (24 channels) | 300 ms | 337 ms | |

  The 6-phone case meets the target (< 0.5 s). Writing the bundle zip adds about 0.26 s at deflate level 1, or 0.05 s with `--stored`.
- Python: `MINIMAPPR_SIM_BINARY=<release binary> python -m pytest tests/test_sim_bundle_roundtrip.py -q` gives **2 passed**. The bundle loads, and the replay produces detections with recall 1.0.

## Deviations and open issues

- ~~**The in-tree cargo build is blocked until merge.**~~ Resolved in review: `minimappr-sim` is a root workspace member and its profiles live in the root `Cargo.toml` as `[profile.*.package.minimappr-sim]`. Original note: A root `Cargo.toml` workspace (from W1-core-extract) appeared in the main checkout during this stream. This worktree sits inside that checkout, so cargo sees `minimappr-sim` as an unlisted workspace member and refuses to build it. I did not add a `[workspace]` table, as instructed. Verification ran on an rsync copy outside the tree. **At merge, add `"minimappr-sim"` to `[workspace].members`.** After that, the Python test's own `cargo build` path works unchanged, because it asks `cargo metadata` for the target directory.
- **Profiles must move to the root.** `[profile.test] opt-level = 3` and the release profile (`lto = "thin"`, `codegen-units = 1`) live in the crate's `Cargo.toml` for now. Cargo ignores them in members, so move them to the root, e.g. as `[profile.test.package.minimappr-sim] opt-level = 3`. I did not commit the crate-local `Cargo.lock`; the workspace lock supersedes it.
- **Synthetic motion and sensor streams are not emitted** (accelerometer, proximity/light, thermal headroom, orientation with heading error). Ground truth records everything needed to derive them (trajectories, gate windows, stall times). Layer 2 can add them.
- **Not modelled:** a vertical temperature gradient (listed as optional in the strategy), frequency-dependent ground impedance, speaker directivity, AGC, and "chirp contamination", "handling noise", "GPS jump" and "asymmetric bouts". The last of these needs a stub classifier, which belongs to Layer 2. GPS error is a continuous OU walk rather than a 30 s step jump.
- **Ambient noise is independent per mic.** A real diffuse field is coherent at low frequency across a 5 cm tetra.
- **Server observation, not a simulator bug:** replaying `roundtrip_small` places both detections at the tetra node positions (gcc_phat, position error 7–10 m). An independent numpy cross-correlation check confirmed that the bundle's inter-mic and inter-node TDOAs match geometry for the trill. The pure tonal whistle is ambiguous under multipath (cycle skipping), which is exactly what the strategy wants exposed. Worth a look in the S-track.
- **Ground-truth schema.** Done (v2, see Bundle output). Flock bouts are one event per bout with `individual_id`; echoes are emitted as v2 echo events.

## Review notes (review/sim)

- Fixed: v2 loader rejected scenarios whose `label` differed from `species`; duplicate event ids now fail at planning instead of at load; echo events added; chaos knobs whose planning loops could not advance (zero walk legs / stall interval / bout gap, sub-10 ms GPS step) and degenerate render settings (zero kernel width, cutoff outside (0, 1], NaN duration) are rejected by `Scenario::validate`; absorption guard is at least two kernel half-widths; WAV writer errors instead of wrapping the 32-bit RIFF size.
- Clock convention (unchanged, for the record): sample n is at true time θ + n / (fs (1 + α)) while the manifest stamps it `start_ns + n / fs`, so a server reading the manifest sees arrivals shifted by −θ (and stretched by α). `ground_truth.simulator.devices[].clock_offset_s` is θ.
- Approximations to keep in mind: pocket muffling also filters mic self-noise; absorption uses the mid-call path length; echo images for moving sources use the mid-call position.
