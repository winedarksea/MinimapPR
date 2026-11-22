# W2-core-sync: chirp sync, clock model, islands, layout and anchor

Stream notes for the Field Mode non-phone build-out (`build_out_plan.md`, row W2-core-sync). Branch `fm/core-sync`, off `field-mode`. The pre-study results and recommendation are in [phase1_sim_prestudy.md](phase1_sim_prestudy.md).

## What exists

There are two new module trees in `minimappr-core`. `lib.rs` gains only `pub mod layout; pub mod sync;`. All code is sans-IO: it never reads the clock, does no I/O and uses no tokio or rayon. Inputs are sample counts on each phone's own clock, distances and sensor values.

| Module | Responsibility |
|---|---|
| `sync/chirp_waveform.rs` | LFM and HFM sweeps (Tukey-tapered, unit peak), band presets 13–16, 15–18 and 18–20 kHz, the up-down "V" chirp (`ChirpWaveform` with per-direction `parts`), the stereo single-channel buffer, active-RMS levelling |
| `sync/chirp_codes.rs` | Session-seeded per-phone codes (up half and down half of hopped sub-band segments, 576 codes for 4 hops), SplitMix64, FNV-1a session seed |
| `sync/matched_filter.rs` | FFT analytic correlation (envelope and real part), parabolic peak interpolation |
| `sync/first_arrival.rs` | Earliest-peak pick (noise floor, relative level, sidelobe-of-strongest rejection), self-recording precursor rejection, double-emission flag (resolved second peak or a broadened main lobe), leading-edge or envelope-peak timing, raw-threshold baseline |
| `sync/split_part_detector.rs` | Picks each part of a multi-part waveform separately and averages them, which cancels the range–Doppler coupling of the up and down halves; reports `doppler_skew_samples` |
| `sync/two_way_ranging.rs` | BeepBeep distance and clock offset with d_ii; `RateCompensation::{None, TrackedRateRatio, DoubleSided}` (UWB SDS-TWR formula); `correct_for_speaker_mic_offsets` |
| `sync/chirp_round.rs` | Round arrival matrix → all pairs; closing-chirp rate ratios; `clock_offset_constraints` (carries each pair's offset to one reference time using rates and coarse offsets) |
| `sync/clock_graph.rs` | Weighted robust LS of the offsets relative to a reference phone. The variance grows with slot gap and motion energy. Huber IRLS plus worst-first rejection, per-node σ |
| `sync/clock_filter.rs` | Per-phone two-state Kalman filter (offset and rate) fed by rounds: rate validity after a minimum baseline, backward prediction, outlier gate, silent-gap detection (re-seeds the offset, keeps the rate), invalidation |
| `sync/acoustic_islands.rs` | Union-find components over two-way-closed hearing edges, `AcousticIslandTracker` (stable ids, merge, split-off, dissolve; merges and split-offs get a fresh clock-domain epoch), `share_clock_domain`, map capability (full fix, bearing/couple, detection only) |
| `layout/classical_mds.rs` | Floyd–Warshall gap fill plus classical MDS (nalgebra `SymmetricEigen`) |
| `layout/robust_layout_refine.rs` | LM on 2D positions with fixed heights and 3D residuals, Huber IRLS, worst-first rejection that warm-starts from the Huber solution, gauge-free σ from the SVD pseudo-inverse, rigidity check |
| `layout/barometric_height.rs` | Hypsometric relative heights; pressure-bias lock from co-location (QR join, or a chirp range under 2 m); GPS-altitude and unknown fallbacks with wide σ |
| `layout/gps_anchor_fit.rs` | Closed-form weighted 2D Kabsch, Huber IRLS plus a final hard outlier drop, a fit on both layout and mirror, chirality confidence 1/(1+e^{−ΔL}), `ChiralityCue::Locked`, `chirality_from_bearing`, GNSS-quality σ inflation |
| `layout/anchor_filter.rs` | Three scalar Kalman filters (east, north, rotation with wrap); process noise scaled by `group_is_still`, a soft innovation gate, and a chirality flip only at high confidence |

The ranging, clock, graph, island and layout APIs take arrival times or distances, never waveforms, so clap or audible-band arrivals use them unchanged. For an impulsive source, `FirstArrivalDetector` works with any template, including a short click.

Dependencies: **nalgebra 0.33** (`default-features = false, features = ["std"]`) in a labelled block of `minimappr-core/Cargo.toml`, and `minimappr-sim` as a **dev-dependency only** (the sim does not depend on core).

## Tests and builds (2026-11-22, M2)

- `cargo test -p minimappr-core`: **135 lib tests** (68 of them new in `sync` and `layout`) and **7 sim scenario tests** (`tests/sync_sim_scenarios.rs`, 3 fixed seeds each, ~11 s in debug).
- Scenario tests: a still round meets the clock gate (graph offset p95 < 50 µs) and gives cm ranges and a layout p95 < 5 cm. Raw offsets carry the speaker–mic bias until corrected. Rate compensation beats none. A hard-table precursor never captures self picks on the near or far mic, while the raw threshold does. Up-down cancels the walking Doppler bias that a single HFM sweep shows. The sim layout anchors to 3 m-noise GPS with the right chirality (confidence > 0.95). Clumps 400 m apart form two acoustic islands.
- `cargo test --workspace`: all green (core 135 + 7, sidecar 116, sim 20).
- `cargo build -p minimappr-core --target wasm32-unknown-unknown` and `--target aarch64-linux-android`: both pass.
- `cargo clippy -p minimappr-core --all-targets`: no warnings in the new files (the remaining warnings are older, in `srp_phat.rs` and `dsp_math.rs`).
- Pre-study: `cargo run --release -p minimappr-core --example phase1_sync_prestudy` (~40 s).

## Findings worth carrying into the plan

1. **The speaker–mic asymmetry biases every BeepBeep clock offset**, by (L_AB − L_BA)/(2c) ≈ −(v_A + v_B)·û/(2c). That is up to d_ii/c: about 295 µs p95 on the near mic and about 700 µs on the far mic in the sim, several times the 50 µs gate. The fix (`correct_for_speaker_mic_offsets`) needs the per-model speaker→mic **vector** (not just |d_ii|), the phone's attitude and the layout. The plan's per-model table should store the vector.
2. **Walking Doppler biases any single sweep by about 30 cm**, LFM and HFM alike. An up-down pair picked per half removes the bias.
3. **Leading-edge timing** halves the remote pick spread against the envelope peak, because ground bounces fall inside the main lobe.
4. **Precursor rejection must ignore level.** The limiter compresses the airborne self path but not the precursor, and interference moves the apparent precursor peak up to one main-lobe width earlier. So the rejection window is the body/air lead plus 60 µs plus one main-lobe width. If the precursor ends up as the strongest peak, a comparable peak just after it is taken as the airborne path.
5. **Each pair's two-way offset refers to its exchange centre**, which is the emission midpoint plus half a flight time. Pairs must be carried to one reference time with the rates before the graph solve: at ±60 ppm the offset moves about 60 µs across a 0.5 s spread.

## Deviations

- **Codes are up-half/down-half hops, not a free mix of directions.** The first version used balanced random directions per segment; the Doppler study showed that only a fixed up-then-down split lets the two halves cancel. Even so, hopped codes fail under walking Doppler (40–49% wrong locks; see the pre-study), so the recommended waveform is the plain up-down HFM. Codes stay available for island disambiguation while still.
- **Clock-graph motion weighting is implemented but not exercised in the sim scenarios.** The harness passes zero motion energy, because the sim emits no accelerometer stream yet (sim.md, "Synthetic motion and sensor streams are not emitted").
- **The clock filter is not driven by multi-round sim scenes.** The harness renders one round per trial; the filter's heartbeat, backward and silent-gap behaviour is unit-tested with synthetic offsets.
- **Rigid layout is 2D with fixed heights**, as the plan says. Heights are inputs, and their σ does not enter the solve.

## Open issues

- Per-phone codes that survive walking Doppler: longer, continuous per-phone up-down chirps with different band offsets or sweep rates, rather than short hops. Needed only when islands overlap.
- A 4.6% wrong-lock rate at 45 m spacing (15–18 kHz up-down) in the sim. A per-half consistency check against `doppler_skew_samples` (reject a pair whose skew implies more than walking speed) would catch most of these.
- Wire in motion energy once the sim (Layer 2) emits accelerometer streams, then rerun table C with the micro-motion knob.
- The layout uses the true heights and the anchor fit uses synthetic GPS. The end-to-end barometer → layout → anchor → anchor-filter chain is exercised only in unit tests.
- Mixed-model receive group delays (absorbed into the clock offsets, which is consistent for TDOA): confirm in Phase 1.
- Coordinate nalgebra with the parallel streams (core-solver may add it too); the version and feature set here are `0.33`, `std` only.

## Review fixes (2026-11-22, promotion review)

- `first_arrival`: the self-recording "precursor became strongest" search picked the *latest* comparable peak up to one lead window after the strongest, so a close hand/table reflection (−3 dB, 0.4 ms late, d_ii 5 cm) was taken as the airborne path (≈ 450 µs error). It now runs only when no earlier precursor was already rejected, starts at the known body/air lead, is skipped when that lead is under half a main lobe (the precursor cannot form its own peak), and takes the first qualifying peak. `lead_before_strongest_samples` is now signed (it underflowed `usize` on that path, a debug-build panic). Pre-study tables reproduce unchanged with the fix.
- `clock_graph`: constraints in components not connected to the reference were judged against the unsolved zero placeholders and reported as rejected outliers; they are now skipped (residual NaN).
- `acoustic_islands`: a fragment that left island A and joined island B reported A as "absorbed" although A's main body survives; only unclaimed islands are absorbed now.
- `chirp_round`: a short `self_path_lengths_m` skips pairs instead of panicking. `robust_layout_refine`: non-finite ranges are inactive. `first_arrival`: a leading-edge fraction > 1 no longer panics.
- `anchor_filter`: documented that every fit must share one local layout frame (align each new MDS layout to the last one first).
