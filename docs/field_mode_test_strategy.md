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

- Direct path with $1/r$ spreading and ISO 9613-1 air absorption from temperature and humidity. This is what makes the 13–16 vs 18–20 kHz chirp comparison meaningful.
- One ground reflection (image source).
- 3–6 random trunk reflections, 3–25 ms late, 6–18 dB down, to stress multipath rejection.
- Per-device mic response, for example a steep roll-off above 16 kHz on weak phones.
- Self-recording: a phone's own chirp feeds its mic at a per-model speaker-to-mic distance $d_{ii}$ drawn from 1–15 cm, through hard clipping or `tanh` limiting, plus an earlier, weaker structural precursor (sound through the phone body at about 5,000 m/s) whose level depends on a grip parameter (foam, hand, hard table).
- Optional second mic on stereo-capable phones, about 15 cm from the first, with a mild level shadow above 4 kHz from the phone body, plus a synthetic orientation stream with heading error.
- Optional: a vertical temperature gradient, giving an effective speed of sound that differs by path.

**3. Real dry bird calls and real noise, not synthetic pings.** Keep a curated fixture set of about 10 MB:

- one pure tonal whistle (for example White-throated Sparrow), one fast trill, one broadband call and one harsh call;
- 30 s of real wind and forest ambience.

Tonal whistles expose GCC-PHAT cycle skipping that white-noise bursts never trigger. Take close, high-SNR recordings, since field recordings carry their own reverb. Xeno-canto recordings are mostly CC BY-NC-SA or similar, which suits a free non-commercial project. Keep an `ATTRIBUTION` file listing each recording's ID, recordist and licence.

**4. Chaos knobs.** Each is a scenario parameter, off by default:

- **Sample drops:** delete 256–4096 samples from a phone's stream, both reported gaps and silent ones.
- **Walking:** piecewise paths (walk at 1.2 m/s, stop for 6 s) with a matching synthetic accelerometer stream, so stillness-triggered sync rounds and Doppler during chirps both get exercised.
- **Pocket:** cut everything above 1.5 kHz by 25 dB on one phone mid-session, with a matching proximity and light stream.
- **Canopy and bad GPS:** birds at $z$ = 5–30 m; each phone's GPS gets a correlated random-walk error with σ = 6 m.
- **Audio route change:** one phone's input switches to a 16 kHz band-limited stream (a Bluetooth headset) mid-session.
- **Thermal throttle:** a phone's synthetic thermal headroom ramps up; past 0.95 its capture thread stalls for 50–300 ms, with the stall either reported by an overrun counter or silent.
- **Chirp contamination:** chirps land inside detection windows, including a phone 1–2 m away, to check the detection low-pass removes them and own-slot blanking (with noise fill) leaves no clicks in Stage 1 or the classifier input.
- **Two-speaker emission:** a phone's chirp plays from both its bottom speaker and its earpiece, 15 cm apart, to check that double arrivals are flagged rather than ranged.
- **Micro-motion:** 10–30 cm hand and body shifts during a chirp round with no steps, and a matching linear-acceleration blip, to check the clock graph's slot-gap and motion weighting.
- **Handling noise:** broadband transients on one phone, tied to spikes in its synthetic accelerometer stream.
- **GPS jump:** one phone's fix jumps 15–20 m for 30 s while the group stands still.
- **Muted speaker:** one phone's chirps play at zero level, so its self-recording shows nothing.
- **Speech:** recorded human conversation near one or two phones, overlapping bird calls.
- **Clap:** a broadband impulse from a point inside the group, for the clap-test check.
- **Intermittent flock:** N sources of one species at fixed perches, each singing 40–250 ms notes in bouts with 4–12 s gaps and occasional 10–30 m hops, so the perched tracker, clustering and echo checks are exercised.
- **Same-band species pair:** two species with overlapping bands at different spots and levels, overlapping inside one classification window, for attribution.
- **Asymmetric bouts:** per-bout overrides for BirdNET score (a stub classifier in Layer 2) and per-phone noise, so a weak-ID bout and a weak-fix bout of the same bird can be scripted.

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
| Mono couple | 2 chirp-synced mono phones walking a bending trail, magnetometer 30° off; bird sings twice | Mirror resolved once the line has turned or moved sideways; before that, both candidates reported, never a confident wrong side |
| Collinear trail | 5 phones in single file along a straight trail, bird 15 m to one side | Collinearity flagged; both mirror candidates reported until a DOA node or an off-trail step breaks the tie; never a confident wrong side |
| Countersinging | Two sources of the same species 20–40 m apart, alternating and overlapping within 1 s | Two separate positions from per-onset SRP clustering; neither pulled toward the midpoint |
| Intermittent flock + strong echoes | 4 birds of one species 20–40 m apart, singing 100–250 ms bursts with 4–12 s silent gaps; strong trunk reflections 5–25 ms late, 6–10 dB down | A separate perched track for each active bird; no echo promoted to an extra bird; tracks survive 10 s gaps without drifting away |
| Asymmetric bouts | One bird sings twice 8 s apart. Bout 1: clean 4-phone TDOA, BirdNET score 0.25. Bout 2: score 0.85, but 2 phones have speech or handling noise | Both bouts fuse into one entity that is localized *and* classified; no dropped fix |
| Same-band two species | A loud robin and a quieter tanager in overlapping 2–5 kHz bands, within one 2 s window, from trees 25 m apart | One shared clip gives two spatial clusters; attribution labels robin at tree A and tanager at tree B with no swaps across fixed seeds; LOCO reruns stay under their cap |
| Omni-to-localized dedup | A localized bird sings 5 times; 2 of the 5 bouts fail to localize | Zero duplicate unlocalized entries for that species; the existing entity's recency and belief update instead; the map never gets an unlocalized pin; one notification, not two |
| Faint bird, one classifier | Bird 10 m from one phone and 35 m from three; only the near phone's classifier fires, and the far phones' Stage-1 triggers do not | Readiness query brings in the far phones' clips anyway; fix produced within the accuracy gate |
| Acoustic split and merge | 6 phones on one host split into two 3-phone clumps about 70 m apart, then walk back together | Two islands, each with its own layout and clock, localize independently; no TDOA pair crosses islands; they merge into one 6-phone layout within one heartbeat of hearing each other's chirps |
| Offline host loss | 2 phones walk out of the host's hotspot range for 60 s, detect birds, then return | Both drop to solo after the timeout, queue detections, and upload them on rejoin with no duplicates (idempotent on `(node_id, local_seq)`); chirp sync resumes |
| Walking at call time | Group walks while a bird sings, stops 2 s later, post-detection round 4 s after the call | Clock offset applied backward; geometry from the round not applied to phones that moved; no fix worse than its reported region |
| Mirrored layout | MDS returns the reflected layout; then a stereo phone or walking track gives a direction | GPS fit picks the right chirality on fixed seeds or reports low chirality confidence; once a direction cue arrives, no bird is ever mirrored |
| Readiness cap | 8 synced phones all pass readiness for one call, cap set to 6 | Exactly 6 clips fetched, best in-band SNR and spread kept; the rest counted, not silently dropped |
| Mic-only and round-robin | 2 Mic-only phones; a clump of 3 phones within 3 m | Mic-only phones contribute clips; the clump runs about one classification per call, not three |
| Transport fallback | The relay drops 2 of 5 phones for 20 s; a session reaches its message cap; later the host loses cell | Dropped phones reconnect and catch up from the event log with no duplicates; past the cap the session moves to local transport or detection-only and says so, with no silent loss; on cell loss the host starts its local transport and phones that don't join run solo and catch up |
| Thermal shedding | One phone and the host heat up over a 30 min session | The phone moves to Mic-only before its first stall; online, the host hands over before throttling; on the hotspot path, the host sheds classification instead; any stall is caught (overrun counter or residual jump) and the clock model is rebuilt within one round |
| Earbuds connect | One phone connects Bluetooth earbuds mid-session (output route change only) | Chirps stay on the built-in speaker; the clock model is kept; a simulated iOS phone becomes listen-only |
| Constrained link | The host's cell link degrades to a satellite or bandwidth-constrained link | Auto treats it as no signal and starts the hotspot; no clip is sent over the constrained link |
| People nearby | A second group of 4 simulated hikers with Bluetooth advertisers and speech passes 40 m away | Their speech cluster gets a "people" tag in the right sector; no address appears in any message; nothing older than 60 s is held |
| Muted phone | One phone's chirps silent | Muted-volume flag raised on its first slot; the phone is excluded from ranging, not mis-ranged |

**Seeds.** Every PR runs a fixed list of 10 seeds, so the gate is deterministic and never flaky. Nightly runs use 200 fresh seeds. When a nightly seed fails, it joins the fixed list after the fix as a regression seed.

## Layer 3 · Golden bundles and measured impulse responses

Simulation misses hardware quirks: Android resampling, speaker distortion, real canopy scattering. Layer 3 closes that gap with **one** careful field recording session at the start of Phase 3, turned into permanent fixtures. Phase 1's raw recordings are kept as early bundles too.

**Session setup.**

- 4–5 phones of different models plus one array node, at laser-measured positions and heights.
- Continuous raw 48 kHz on every device, with each device's own sample count.
- Temperature and humidity logged.

**What to record.**

- The chirp sweep: 13–16, 15–18 and 18–20 kHz, linear and hyperbolic FM, from every phone.
- A bird-call playlist from a portable speaker at about 15 marked points: inside the layout, outside it, and raised on a pole (4–6 m, higher if possible, since the canopy prior assumes 10–30 m).
- Two speakers at once at marked points 20–30 m apart: a same-band species pair overlapping within 2 s, and same-species notes alternating with 4–12 s gaps, for clustering and attribution.
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
- **Android 17 audio hardening.** Run screen-locked chirp rounds with `adb shell cmd audio set-enable-hardening throw`, so a blocked playback or audio-focus call throws instead of failing silently. Confirm the mode actually took effect in `adb shell dumpsys audio` rather than trusting the command's exit code.
- **Transport checks.** `LocalOnlyHotspot` joins from Android and iOS, Nearby Connections against the hotspot, the online relay over cell on 2+ carriers, the iOS PWA's routes to the host (online relay, Linux HTTPS), and later Network.framework peer-to-peer.
- **Self-chirp surfaces.** Each phone model's self-recording onset on foam, in a hand and on a hard table, matched-filter vs threshold, near vs far mic.
- **Capture path and battery.** Phase 1 compares `AudioRecord` against AAudio (none and power-saving modes) for overrun reporting, hidden resampling and battery. A 3 h battery A/B runs with sensor-hub batching on and off.
- **Lock screen.** The Live Update card stays pinned and updates in place with the screen locked through a 30 min session.
- **Wired loopback (optional). A multi-output USB audio interface feeds Layer 1 or Layer 3 audio straight into phones running the real app. Wired input is an audio route change, which the production app pins away from and treats as clock invalidation, so this needs a debug-build test-input override.

## What to assert

Use statistical gates and physical invariants rather than one `error < 3.0 m` case, so subtle bugs can't hide.

| Check | Gate | Bug it catches |
| --- | --- | --- |
| Accuracy, simulated Monte Carlo | Inside the layout: horizontal median < 1.5 m, P95 < 3 m. Outside, at 50 m: bearing error < 5° | Solver divergence, multipath lock-on, 2D canopy bias |
| Accuracy, golden bundles | Per-bundle gates in the expectations file (replay.py style), tightened as results improve | Real hardware and forest effects the simulator misses |
| Uncertainty calibration | Reported 95% likelihood contour (the region drawn on the map) contains the truth in 90–98% of trials, inside and outside the layout; NEES chi-square check on the LM covariance kept for the tracker | Overconfident regions, ellipses pointing the wrong way outside the layout, mis-scaled covariance |
| Clock sync | Simulated residual < 10 µs clean and < 50 µs with chaos knobs on; recovers within one round after a reported gap; a silent gap is caught (residual jump) within one round, before it corrupts a fix | Missing in-clip rate correction, stale models after gaps |
| Geometric invariance | Applying any rotation and translation to all nodes and birds transforms the final estimates identically within 1 mm | Axis flips, ENU and sign errors, grid bias |
| Chirality | Mirrored-layout seeds: once any direction cue exists, no localized bird is mirrored; before that, chirality confidence is reported low | Proper-rotation-only GPS fit on a reflected MDS layout |
| Island isolation | No TDOA pair is ever formed across islands that share no clock domain; a merge never reuses a clock model from before it | Cross-domain TDOA, stale clocks after a merge |
| Loop consistency | Rejected triples violate $\lvert\tau_{AB}+\tau_{BC}+\tau_{CA}\rvert > \tau_{\text{thresh}}$; accepted ones close; tonal fixtures never cycle-skip | Cycle skipping, cross-species correlation |
| Graceful degradation | Muting or pocketing 1 of 5 phones grows the uncertainty region but moves the estimate no further than its old region | Missing robust loss (Huber or Cauchy), brittle outlier handling |
| Cross-language parity | Ported functions match Python within 10⁻⁵ relative on shared `.npz` fixtures; end-to-end results compared in centimetres and degrees | Drift between the server and the phone app |
| Map anchor stability | A 20 m GPS jump on one phone while the group is still moves the anchor less than 1 m | Non-robust fit, missing stillness clamp |
| Resource bounds | Dawn-chorus scenario: memory bounded, every drop counted | Unbounded queues, silent loss |
| Faint detections | Stage-1 trigger miss rate on faint-call fixtures stays under its gate | Gates that block distant birds (AGENTS.md) |
| Localized and classified | `min_localized_and_classified_recall` on golden bundles and Layer 2 bout scenarios; set from the server track's baseline, then tightened | Bouts split across tracks; single-bout coupling of ID and fix |
| Label swaps | Same-band two-species seeds: zero swaps on the fixed list; nightly swap rate under its gate | Attribution picking the louder bird; stale species priors |
| Echo promotions | Flock-and-echo seeds: no reflection or sidelobe shown as an extra bird | A fixed-threshold second peak; a missing onset or level check |
| Duplicate entities | Omni-to-localized seeds: zero duplicate entities or double notifications per bird | Parallel event streams; fixed-bucket reporting windows |
| Tracker continuity | A bird silent for 10 s keeps its track within its gate; a 15 m hop with a matching species stays one track | Constant-velocity coasting; relocation gate too tight |

Notes on these gates:

- **Ellipse coverage.** Over 100 trials at a true 95%, the binomial spread is about ±2.2 points, so the 90–98% band sits about 2σ below and 1.4σ above. Fixed PR seeds make that deterministic, but nightly random seeds would flake; nightly runs use 400+ trials or a proper binomial test.
- **Geometric invariance.** The coarse SRP grid is not exactly rotation-invariant, even in polar form, so compare final estimates after LM refinement, and centre and align the grid to the island's layout frame. MinimapPR has been bitten by frame bugs before (the per-process site-origin ENU split), so this check is cheap insurance.
- **Parity tolerance.** The current tetrahedral bearing parity test (about 25°) is far looser than Field Mode needs; ported functions get the tight check, and end-to-end results get physical-unit tolerances.

## Rollout by phase

| Phase (main plan) | Test work |
| --- | --- |
| S · Server track (alongside Phase 1) | Extend the replay expectations schema with localized-and-classified recall, label-swap, duplicate-entity and echo-promotion gates; measure today's baseline. Record speaker-playback bundles with the existing calibration capture (same-band species pair, two same-species speakers alternating, a weak and a strong take of one call) and build Python synthetic flock-and-echo mixes. Live dawn-chorus captures scored on duplicates and label stability only |
| 1 · Sync spike | Minimal Layer 1 (fractional delay, drift, air absorption, mic response, self-clipping, body-conduction precursor) to compare chirp bands, linear vs hyperbolic FM, slot lengths, and rate-compensated vs uncompensated ranging in milliseconds before touching phones. Self-chirp surface tests; bottom-speaker channel map (and rotation); chirp leakage below 11 kHz at 1–2 m; Safari chirp TX level; clap test. Isolation-box soak for the 30-minute residual gate, then 3 h. `AudioRecord` vs AAudio capture comparison; proximity and light readings with the screen locked. Android 17 hardening test in `throw` mode. `MicrophoneInfo` and audio-source survey. Keep all raw recordings as early bundles |
| 2 · Shared core | Sans-IO state machines in `minimappr-core`. Full Layer 1 and Layer 2 under `cargo test`: 10 fixed seeds per PR (target under 5 s total), 200 fresh seeds nightly. Parity suite for every ported function, including the server track's clustering, perched tracker, species fusion and entity model |
| 3 · Android MVP + mixed nodes | One laser-measured recording afternoon at the start: golden bundles and measured responses into R2; Layer 3 in CI. Thermal, earbuds and constrained-link scenarios in Layer 2; lock-screen card check |
| 3–5 | Field trips only for phase gates. Isolation-box soak for each new phone model. Backyard smoke test before each release |
