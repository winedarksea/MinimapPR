# MinimapPR Field Mode — Test Strategy

As of 2026-11-21. Companion to [field_mode_phone_mesh_plan.md](field_mode_phone_mesh_plan.md), which keeps only a summary of this document.

## Summary

Clean unit tests of acoustic localization ("four mics at the corners, shift a sine wave by 12 samples, check the TDOA") almost always pass, and the system still fails the first time it meets real woods. Driving out with six phones and a tape measure after every change to GCC-PHAT or clock sync is far too slow. The way through is four test layers. Most validation runs in seconds under `cargo test` and `pytest`, and each layer exists to catch a class of failure the layer below cannot.

| Layer | What it is | Speed | Catches | Cannot catch |
| --- | --- | --- | --- | --- |
| 1 · Acoustic world simulator | Rust crate rendering realistic 48 kHz streams in virtual time | Seconds | DSP, sync and solver bugs under drift, multipath, Doppler, dropped samples | OS audio drivers, real forest scattering, real mic quirks |
| 2 · Deterministic mesh simulation | Node and host state machines wired over a seeded simulated network | Seconds | Scheduling, matching, handover, backpressure, network faults | Acoustic realism beyond Layer 1 |
| 3 · Golden bundles and measured impulse responses | One careful field recording, replayed forever; measured forest and phone responses applied to new audio | Under 5 s per bundle once cached | Real hardware and forest acoustics | OS lifecycle, phone models that weren't recorded |
| 4 · Hardware in the loop | Real phones on a desk, in a foam box or in a backyard | Minutes to hours | OS audio, background capture, battery, heat, transports | Geometry and signal levels at field scale |

Field trips are kept for the phase gates in the main plan.

## What MinimapPR already has

- **Replay gate.** `minimappr/sim/replay.py` rebuilds a `FusionNode` from a calibration bundle (`minimappr/calibration/bundle.py`), replays each node's raw audio and scores it against ground truth through an expectations schema (`min_event_recall`, `max_median_position_error_m`, `max_p90_position_error_m`). Field Mode bundles should reuse that expectations format and report style.
- **Synthetic soak.** `minimappr/sim/soak.py` and `run_demo.render_sensor_frame` generate 16 kHz synthetic frames for API and runtime stability checks. They are not sample-accurate and have no clock drift, so they can't test sub-sample delay or clock-rate effects; Layer 1 is new code.
- **Parity tests.** `tests/test_localization_cross_language_parity.py` feeds identical synthetic tetrahedral windows to Rust and Python and checks bearing agreement loosely (dot product above 0.9, about 25°). Golden `.npz` fixtures live in `tests/fixtures/` (about 0.5 MB, plain git).
- **No channel simulator.** Nothing in the repo models fractional delays, clock drift, air absorption or reflections.

## Layer 1 · Acoustic world simulator

A dependency-light `minimappr-sim` crate, used as a dev-dependency of `minimappr-core`, renders 48 kHz PCM per device in virtual time and feeds it straight into the core. It never sleeps on the wall clock and never touches audio hardware. It needs five cheap physical ingredients, following the ForestIR formulation without becoming a ray tracer.

**1. Exact fractional propagation and clock drift.** Never delay by whole samples; that hides interpolation bugs and the in-clip stretch from clock-rate offset. Receiver $i$ takes sample $n$ at true time

```math
t_i[n] = \theta_i + \frac{n}{f_s\,(1 + \alpha_i)}
```

with a random clock offset $\theta_i$ and drift $\alpha_i$ drawn from ±60 ppm. Each path contributes the source waveform evaluated at $t_i[n] - r(t)/c$, using windowed-sinc or Farrow fractional interpolation. Because the path length $r(t)$ is recomputed as nodes move, Doppler comes out of the same formula with no separate model.

**2. A cheap forest channel per source–receiver pair.**

- Direct path with $1/r$ spreading and ISO 9613-1 air absorption from temperature and humidity. This is what makes the 12–16 vs 18–20 kHz chirp comparison meaningful.
- One ground reflection (image source).
- 3–6 random trunk reflections, 3–25 ms late, 6–18 dB down, to stress multipath rejection.
- Per-device mic response, for example a steep roll-off above 16 kHz on weak phones.
- Self-recording: a phone's own chirp feeds its mic at a per-model speaker-to-mic distance $d_{ii}$ drawn from 1–15 cm, through hard clipping or `tanh` limiting.
- Optional second mic on stereo-capable phones, about 15 cm from the first, with a mild level shadow above 4 kHz from the phone body, plus a synthetic orientation stream with heading error.
- Optional: a vertical temperature gradient, giving an effective speed of sound that differs by path.

**3. Real dry bird calls and real noise, not synthetic pings.** Keep a curated fixture set of about 10 MB:

- one pure tonal whistle (for example White-throated Sparrow), one fast trill, one broadband call and one harsh call;
- 30 s of real wind and forest ambience.

Tonal whistles expose GCC-PHAT cycle skipping that white-noise bursts never trigger. Take close, high-SNR recordings, since field recordings carry their own reverb. Xeno-canto recordings are mostly CC BY-NC-SA or similar, which suits a free non-commercial project. Keep an `ATTRIBUTION` file listing each recording's ID, recordist and licence.

**4. Chaos knobs.** Each is a scenario parameter, off by default:

- **Sample drops:** delete 256–4096 samples from a phone's stream, both reported gaps and silent ones.
- **Walking:** piecewise paths (walk at 1.2 m/s, stop for 6 s) with a matching synthetic accelerometer stream, so stillness-triggered sync rounds and Doppler during chirps both get exercised.
- **Pocket:** cut everything above 1.5 kHz by 25 dB on one phone mid-session.
- **Canopy and bad GPS:** birds at $z$ = 5–30 m; each phone's GPS gets a correlated random-walk error with σ = 6 m.
- **Audio route change:** one phone's input switches to a 16 kHz band-limited stream (a Bluetooth headset) mid-session.
- **Chirp contamination:** chirps land inside detection windows, to check they are masked out.
- **Handling noise:** broadband transients on one phone, tied to spikes in its synthetic accelerometer stream.
- **GPS jump:** one phone's fix jumps 15–20 m for 30 s while the group stands still.

**5. Speed.** The reviewer estimates 200–500 ms to render 60 s with 6 phones, 4 chirp rounds and 10 overlapping calls; treat that as a target to measure, not a given. Two things keep it achievable:

- Render paths only around events (chirps and calls) and add noise cheaply, rather than interpolating every path at every sample.
- Build the simulator optimized (`opt-level = 3` for `minimappr-sim` in `[profile.test]`, or `cargo test --release`). Debug builds are 10–50 times slower.

PR scenarios use 10–20 s windows.

## Layer 2 · Deterministic mesh simulation

**What it requires of `minimappr-core`.** The node and host logic — sync scheduler, event matcher, clip fetcher, solver, tracker, event log — must be sans-IO state machines. They take inputs (audio blocks, messages, timer ticks) and return outputs (messages to send, timers to set). The clock, random-number source and transport are injected; the core uses no tokio and never reads the wall clock. The Android shell, the WASM PWA, a Linux array-node host and the test harness all drive the same state machines. This is the FoundationDB / TigerBeetle pattern.

**Simulated network.** A seeded PRNG controls latency, loss, reordering and disconnects on a virtual clock. The same seed also controls geometry, clock drift, reflections and call timing, so a failing run prints one seed and `MINIMAPPR_SIM_SEED=<seed> cargo test <scenario>` reproduces it exactly on a laptop.

| Scenario | Setup | Assertion |
| --- | --- | --- |
| Dawn chorus overload | 5 species calling every second, 6 phones | Species-keyed matching stays correct; memory bounded; every drop counted, none silent |
| Host death | Kill the host mid chirp round, then mid clip fetch | Next eligible node takes over from its mirrored log and localizes again within one chirp round |
| Mixed topology | 1 PPS-locked array node (DOA, passive chirp listening) + 2–3 chirp-synced phones | Joint TDOA + DOA beats phones alone on the same seeds |
| Network faults | 10% packet loss; one phone's clip 800 ms late; a joiner disconnects mid-call | Positions still produced or cleanly abandoned; no stale clip used against a new clock model |
| iOS PWA node | Stream stops at "screen lock"; occasional silent block drops | Node is down-weighted or demoted to detection-only; other nodes unaffected |
| Route change | One phone switches to a band-limited input mid-session | Its clock model is invalidated at once and its clips are excluded until the next good round |
| Couple with one stereo phone | 2 chirp-synced phones, one a 2-mic `SubArray` with 20° heading error | One bearing with the mirror removed on every fixed seed; the reported bearing covariance covers the truth |

**Seeds.** Every PR runs a fixed list of 10 seeds, so the gate is deterministic and never flaky. Nightly runs use 200 fresh seeds. When a nightly seed fails, it joins the fixed list after the fix as a regression seed.

## Layer 3 · Golden bundles and measured impulse responses

Simulation misses hardware quirks: Android resampling, speaker distortion, real canopy scattering. Layer 3 closes that gap with **one** careful field recording session at the start of Phase 3, turned into permanent fixtures. Phase 1's raw recordings are kept as early bundles too.

**Session setup.**

- 4–5 phones of different models plus one array node, at laser-measured positions and heights.
- Continuous raw 48 kHz on every device, with each device's own sample count.
- Temperature and humidity logged.

**What to record.**

- The chirp sweep: 12–16, 15–18 and 18–20 kHz, linear and hyperbolic FM, from every phone.
- A bird-call playlist from a portable speaker at about 15 marked points: inside the layout, outside it, and raised on a pole (4–6 m, higher if possible, since the canopy prior assumes 10–30 m).
- One phone in a pocket, and a short walk.
- Exponential sine sweeps (Farina method, 200 Hz–22 kHz, about 5 s) from the speaker at 10–15 spots, and from each phone's own speaker, recorded on every device.

**Bundle format.** Extend the calibration bundle with:

- a per-device stream on that device's own sample clock, with no trusted shared timestamps;
- the chirp schedule;
- device models;
- ground-truth positions with their measurement uncertainty.

A phone bundle replays into `minimappr-core` directly under `cargo test`, and through the existing `pytest` replay gate for server parity.

**Measured impulse responses.** Deconvolving the sweeps gives real speaker-to-mic responses that include the forest and each phone's mic. Convolve them with new dry calls, dawn-chorus mixes or candidate chirp waveforms to get near-field-trial fidelity across hundreds of variations without leaving the desk. Limits to keep in mind:

- each response includes the playback speaker and is valid only for that source spot and that receiver;
- responses are linear, so they miss AGC, limiters and OS resampling (the raw bundles cover those);
- correct each phone's clock-rate offset before deconvolving, or the response smears;
- chirp tests need the phone-to-phone sweeps, not the speaker ones.

**Semi-synthetic faults.** Replay real bundles with injected faults: drop 512 samples on one phone, add real wind noise at a set SNR, mute a phone.

**Storage.** Raw multitrack audio runs to gigabytes and stays out of git. Store bundles in R2 (Cloudflare is already in the stack), addressed by SHA-256, with a manifest in the repo (`tests/fixtures/field_bundles.toml`: hash, size, licence, description) and a fetch script that caches locally. When fixtures are missing, a local run skips with a loud message naming the fetch command, and CI fails; a silent skip would hide regressions. Field recordings can pick up passers-by, so listen before uploading and keep the bucket private.

## Layer 4 · Hardware in the loop

Real Android (`AudioRecord`, `LocalOnlyHotspot`, foreground service) and iOS (PWA now, `AVAudioEngine` and Network.framework later) still need testing on devices, because no simulator models OS audio drivers or thermal throttling. None of it needs a field trip.

- **Isolation-box soak.** 2–3 phones in a cardboard box lined with foam, run for 3 h or overnight, screen-locked (screen-on for the iOS PWA). Log chirp offset residuals, drift smoothness, gaps, battery and thermal state. This tests sample continuity and drift, not ranging: reflections inside a box make distances meaningless. Foam absorbs 12–20 kHz well, so the chirps stay quiet.
- **Backyard mini-array (2 m × 2 m).** End-to-end smoke test before each release: start a session, scan the QR code, play a bird call from a speaker, check the dot on the map. Ranging resolves centimetres, so the geometry is valid; signal levels are not representative of the field.
- **Android 17 audio hardening.** Run screen-locked chirp rounds with `adb shell cmd audio set-enable-hardening throw`, so a blocked playback or audio-focus call throws instead of failing silently.
- **Transport checks.** `LocalOnlyHotspot` joins from Android and iOS, the iOS PWA's routes to the host, and later Network.framework peer-to-peer.
- **Wired loopback (optional).** A multi-output USB audio interface feeds Layer 1 or Layer 3 audio straight into phones running the real app. Wired input is an audio route change, which the production app pins away from and treats as clock invalidation, so this needs a debug-build test-input override.

## What to assert

Use statistical gates and physical invariants rather than one `error < 3.0 m` case, so subtle bugs can't hide.

| Check | Gate | Bug it catches |
| --- | --- | --- |
| Accuracy, simulated Monte Carlo | Inside the layout: horizontal median < 1.5 m, P95 < 3 m. Outside, at 50 m: bearing error < 5° | Solver divergence, multipath lock-on, 2D canopy bias |
| Accuracy, golden bundles | Per-bundle gates in the expectations file (replay.py style), tightened as results improve | Real hardware and forest effects the simulator misses |
| Uncertainty calibration | Reported 95% ellipse contains the truth in 90–98% of trials; NEES chi-square check on the 3D covariance | Overconfident ellipses, mis-scaled covariance |
| Clock sync | Simulated residual < 10 µs clean and < 50 µs with chaos knobs on; recovers within one round after a reported gap; a silent gap is caught (residual jump) within one round, before it corrupts a fix | Missing in-clip rate correction, stale models after gaps |
| Geometric invariance | Applying any rotation and translation to all nodes and birds transforms the final estimates identically within 1 mm | Axis flips, ENU and sign errors, grid bias |
| Loop consistency | Rejected triples violate $\lvert\tau_{AB}+\tau_{BC}+\tau_{CA}\rvert > \tau_{\text{thresh}}$; accepted ones close; tonal fixtures never cycle-skip | Cycle skipping, cross-species correlation |
| Graceful degradation | Muting or pocketing 1 of 5 phones grows the ellipse but moves the estimate no further than its old ellipse | Missing robust loss (Huber or Cauchy), brittle outlier handling |
| Cross-language parity | Ported functions match Python within 10⁻⁵ relative on shared `.npz` fixtures; end-to-end results compared in centimetres and degrees | Drift between the server and the phone app |
| Map anchor stability | A 20 m GPS jump on one phone while the group is still moves the anchor less than 1 m | Non-robust fit, missing stillness clamp |
| Resource bounds | Dawn-chorus scenario: memory bounded, every drop counted | Unbounded queues, silent loss |
| Faint detections | Stage-1 trigger miss rate on faint-call fixtures stays under its gate | Gates that block distant birds (AGENTS.md) |

Notes on these gates:

- **Ellipse coverage.** Over 100 trials at a true 95%, the binomial spread is about ±2.2 points, so the 90–98% band sits about 2σ below and 1.4σ above. Fixed PR seeds make that deterministic, but nightly random seeds would flake; nightly runs use 400+ trials or a proper binomial test.
- **Geometric invariance.** The coarse SRP grid is not rotation-invariant, so compare final estimates after LM refinement, and align the grid to the layout frame. MinimapPR has been bitten by frame bugs before (the per-process site-origin ENU split), so this check is cheap insurance.
- **Parity tolerance.** The current tetrahedral bearing parity test (about 25°) is far looser than Field Mode needs; ported functions get the tight check, and end-to-end results get physical-unit tolerances.

## Rollout by phase

| Phase (main plan) | Test work |
| --- | --- |
| 1 · Sync spike | Minimal Layer 1 (fractional delay, drift, air absorption, mic response, self-clipping) to compare chirp bands and linear vs hyperbolic FM in milliseconds before touching phones. Isolation-box soak for the 30-minute residual gate, then 3 h. Android 17 hardening test in `throw` mode. `MicrophoneInfo` and audio-source survey. Keep all raw recordings as early bundles |
| 2 · Shared core | Sans-IO state machines in `minimappr-core`. Full Layer 1 and Layer 2 under `cargo test`: 10 fixed seeds per PR (target under 5 s total), 200 fresh seeds nightly. Parity suite for every ported function |
| 3 · Android MVP + mixed nodes | One laser-measured recording afternoon at the start: golden bundles and measured responses into R2; Layer 3 in CI |
| 3–5 | Field trips only for phase gates. Isolation-box soak for each new phone model. Backyard smoke test before each release |
