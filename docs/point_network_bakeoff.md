# Point-network localizer bake-off (E8)

Should the sidecar run onset clustering live for point-only networks, or
keep today's localizer? Point-only means several single-mic or omni nodes
(phones, point sensors) spread over a site, where no node can give a bearing
on its own. Live onset clustering is shadow-only until this bake-off says
otherwise (localization overhaul plan, "Not in this round").

Harness: `minimappr-mesh-sim/src/bakeoff_localizers.rs` and its `bakeoff_*`
helper modules. The adoption rule lives in `bakeoff_decision.rs`. It was
committed (cadbec7) before any held-out seed was scored.

## How to run

| What | Command | Cost |
|---|---|---|
| Nightly / full | `cargo run --release -p minimappr-mesh-sim --example point_network_bakeoff -- --markdown out.md` | 3 min on 8 threads (measured, see Results) |
| Fresh held-out seeds | add `--held-out-base <N>`; a nightly can pass the date | same |
| One scene, every candidate | `... --example point_network_bakeoff -- --seed 7` | about 1 s |
| Opt-in test | `cargo test --release -p minimappr-mesh-sim --test point_network_bakeoff -- --ignored --nocapture` | same as the full run |
| Default-suite smoke | `cargo test -p minimappr-mesh-sim --test point_network_bakeoff` | under 1 s |

The smoke tests check one easy scene, which the Cartesian solve (behind a
repaired front-end) and onset clustering must both localize, and run the
whole protocol on one tuning seed and one held-out seed. An `#[ignore]`d test
records the production front-end finding below; remove its marker when the
front-end is repaired.

## Scenes

Each seed gives one scene. Every scene is rendered by the Layer 1 simulator
(`minimappr-sim`), which models spherical spreading, ISO 9613-1 absorption, a
ground reflection, and independent ambient and self-noise per device. Audio
is 24 kHz mono per node. Ranges are in `bakeoff_config.rs::SceneDrawConfig`.

- **Layout:** 3–8 nodes. The site spread is log-uniform from 20 m to 1 km,
  and nodes keep a minimum separation. Nodes stand 1.2–1.8 m above the ground.
- **Event:**
  - one bird (50 %);
  - two simultaneous birds, the second starting 0–250 ms after the first and
    at least 20 % of the spread away (30 %);
  - one continuous harmonic source with a soft 0.4 s fade-in, standing in
    for a drone (20 %).
- **Bird calls:** FM sweep, tonal whistle, trill or harsh, centred at
  2–5.5 kHz, 0.2–0.5 s long, 85–100 dB SPL at 1 m.
- **Clocks:** all GPS-locked (40 %), all NTP (40 %), or mixed GPS/NTP with
  one free-running node half the time (20 %).
  - Each node's true offset is drawn at its class's declared sigma, so the
    declared uncertainty is honest: GPS 1 µs, NTP 2 ms, free-running 50 ms
    (the server's `localization_clock_class_uncertainty_seconds` defaults).
  - NTP clocks drift up to ±10 ppm and free-running clocks up to ±50 ppm.
  - A node stamps sample n as n / fs, so a window cut on the shared stamped
    timeline carries the real clock error.
- **Positions:** 40 % of sites have hand-measured nodes; at such a site each
  node is hand-measured with probability 0.6.
  - A hand-measured node states a 0.3–1.5 m accuracy, and its entered
    position is off by a draw at that sigma.
  - Other nodes are surveyed at 0.1 m.
- **Trigger:** the first node whose 32 ms frame RMS stands 8 dB over its
  noise floor starts a solve, then a 0.8 s cooldown. These are the
  `localization_trigger_noise_floor_relative_db` and
  `trigger_cooldown_seconds` defaults.

## Candidates

| Candidate | What it is |
|---|---|
| `production_gcc_phat_lsq` | Today's point-network path. (1) The sidecar's event window: an 80 ms node-local window, grown by the cross-node lag beyond half of it on each side, with the lag capped at `localization_coverage_total_cap_seconds` = 1 s. (2) GCC-PHAT pairs over 50–3430 Hz. (3) The server's default `python_cartesian` re-solve: pairs weighted by inverse variance on pair clock uncertainty (P9), the SRP steering direction as a bearing prior, and the SciPy-parity Cartesian solver. With no re-solve it falls back to the sidecar estimate. |
| `production_sidecar_srp` | The sidecar's own SRP estimate from the same call, with clock-offset covariance (`localization_single_node_solver = rust`). Diagnostic. |
| `production_uncapped_window` | Production with the 1 s lag cap lifted. Diagnostic. |
| `production_repaired_front_end` | Production with the lag cap lifted, a 20 % Tukey taper on the window, and a 50–11 000 Hz band. Diagnostic: it separates the front-end from the solver. |
| `onset_clustering_shadow` | `onset_clustering::localize_onset_clusters`, as the sidecar would run it live. It gets one window per trigger, from one layout lag plus 0.5 s before the trigger to two layout lags plus 1 s after. Accepted clusters with at least 3 TDOA nodes that pass the Field Mode noise-onset gate become fixes. Its covariance carries the R5 policy: the cluster covariance plus each node's clock sigma and position sigma / c, propagated through the reference-TDOA Jacobian (`bakeoff_covariance_propagation.rs`). The SRP grid reach is the only tuned knob: the core default, or 1.5× aperture. |

Every candidate's fixes pass the server's handoff gates (`gate_handoff_localization`):

- an unobservable range mode makes a bearing ray, not a fix;
- a fix more than 1200 m from the contributing centroid is dropped;
- the range-proportional covariance ceiling applies.

The confidence and GDOP gates ship disabled and are not modelled. Neither is
the low-frequency lateral widening, which never fires at these baselines.

## Metrics (held-out seeds)

- **Attribution:** each fix's cues pick the scripted call that is sounding
  nearest that moment at that node. The cues are the trigger for production
  and the led onsets for onset clustering. No position is consulted.
- **Recall:** the fraction of scripted calls whose primary fix lands within
  max(5 m, 10 % of the aperture) horizontally. The primary fix is the first
  fix attributed to the call.
- **Median and p90 error:** horizontal error of the primary fix, over calls
  that have one.
- **95 % coverage:** the fraction of primary fixes whose stated horizontal
  covariance contains the truth, meaning e^T S^-1 e ≤ chi-square₂(0.95).
  This is E3's scoring restricted to the plane, since point layouts rarely
  observe height. A singular or missing covariance counts as a miss.
- **Two-caller attribution error:** in two-bird scenes, the fraction of fixes
  that are not within the gate of the call their cues point to. These are
  merged, swapped or ghost fixes.
- **Solve cost:** wall time per candidate invocation.

## Decision rule (pre-registered)

The seeds are split. Seeds 1–40 are tuning seeds: they only choose the onset
grid reach, and production runs at its shipped defaults. Held-out seeds start
at 2 000 001. Seeds 1 000 001–1 000 012 were looked at while the harness was
being built, so they are excluded.

Onset clustering is recommended for live adoption only if, on the held-out
seeds, all of the following hold against `production_gcc_phat_lsq`:

1. Recall gain is at least +0.05 absolute, and the one-sided 95 % paired
   bootstrap lower bound is above 0. The bootstrap resamples seeds 2000 times.
2. On calls both candidates fixed, the ratio of p90 errors (onset over
   production) is at most 1.10.
3. 95 % coverage is at least production's minus 0.02.
4. The two-caller attribution error rate is at most production's plus 0.05.
5. Guard: criterion 1 also holds against the stronger of the two production
   front-end diagnostics, judged by held-out recall.

If 1–4 pass and 5 fails, the gain comes from production's front-end, not
from its solver. The verdict is then "keep production; repair its front-end".

## Results

The full run was done once, on 2026-11-25, at commit cadbec7. It used 40
tuning seeds and 200 held-out seeds (2 000 001–2 000 200), giving 265 calls
in 65 two-caller scenes, 103 single-bird scenes and 32 continuous-source
scenes. No seed failed. Wall time was 182 s on 8 threads (680 CPU-s), built
with `--profile test`, where mesh-sim, sim and core run at opt-level 3 but
other dependencies (rustfft, nalgebra) do not. Release solve times will be
lower; the ratios between candidates are what matter.

The onset grid reach was tuned on the tuning seeds. The core default scored
recall 0.347 (p90 205 m) and 1.5× aperture scored recall 0.327 (p90 108 m),
so the core default was chosen.

### Held-out results

| candidate | calls | recall | calls fixed | median err (m) | p90 err (m) | 95 % coverage | 2-caller attribution err | median solve (ms) |
|---|---|---|---|---|---|---|---|---|
| production_gcc_phat_lsq | 265 | 0.302 | 126 | 2.3 | 91.7 | 0.540 | 0.545 | 90 |
| production_sidecar_srp | 265 | 0.147 | 74 | 7.6 | 141.5 | 0.649 | 0.586 | 90 |
| production_uncapped_window | 265 | 0.313 | 128 | 1.8 | 77.4 | 0.523 | 0.523 | 91 |
| production_repaired_front_end | 265 | **0.453** | 149 | **1.1** | **67.0** | 0.557 | **0.400** | 122 |
| onset_clustering_shadow | 265 | 0.298 | 164 | 13.8 | 133.5 | 0.378 | 0.602 | 1588 |

### Decision

| criterion | passed | evidence |
|---|---|---|
| recall gain over production | no | gain −0.004; one-sided 95 % lower bound −0.065 |
| matched-call p90 error ratio | no | 1.241 over 105 calls both fixed (limit 1.10) |
| coverage not worse | no | onset 0.378 vs production 0.540 |
| two-caller attribution not worse | no | onset 0.602 vs production 0.545 (limit +0.05) |
| guard vs `production_repaired_front_end` | no | gain −0.155; lower bound −0.213 |

**Verdict: keep production; onset clustering stays shadow-only.** It fails
every criterion. It publishes more fixes (164 calls fixed against 126), but
they are less accurate: median 13.8 m against 2.3 m. It is the only candidate
that localizes anything on a 3-node site (recall 0.256, 43 calls), and the
only one that beats production on tonal whistles (0.321 against 0.125).
Neither win offsets the losses elsewhere. It also costs about 18× more per
solve, rising to about 12 s at spreads of 343 m or more, so a live version
would have to run off the audio hot path.

### Slices worth reading (recall / p90 m / coverage)

| slice | production | repaired front-end | onset |
|---|---|---|---|
| spread < 100 m (114 calls) | 0.316 / 32.8 / 0.50 | 0.544 / 10.3 / 0.54 | 0.333 / 36.3 / 0.51 |
| spread 100–343 m (101) | 0.277 / 91.7 / 0.55 | 0.396 / 69.2 / 0.54 | 0.287 / 101.7 / 0.30 |
| spread ≥ 343 m (50) | 0.320 / 205.5 / 0.60 | 0.360 / 200.6 / 0.65 | 0.240 / 235.1 / 0.24 |
| single bird (103) | 0.398 / 66.6 / 0.58 | 0.650 / 7.8 / 0.58 | 0.437 / 113.4 / 0.43 |
| two birds (130) | 0.146 / 141.8 / 0.49 | 0.246 / 121.6 / 0.52 | 0.192 / 133.5 / 0.29 |
| continuous source (32) | 0.625 / 51.9 / 0.52 | 0.656 / 60.3 / 0.58 | 0.281 / 150.3 / 0.44 |
| FM sweep first call (67) | 0.179 / 189.7 / 0.48 | 0.403 / 90.3 / 0.51 | 0.343 / 108.6 / 0.51 |
| tonal whistle (56) | 0.125 / 128.8 / 0.65 | 0.268 / 121.6 / 0.60 | 0.321 / 59.5 / 0.27 |
| all GPS clocks (93) | 0.366 / 54.3 / 0.57 | 0.527 / 13.7 / 0.56 | 0.312 / 123.6 / 0.47 |
| all NTP clocks (113) | 0.239 / 128.8 / 0.58 | 0.389 / 121.6 / 0.58 | 0.283 / 140.8 / 0.30 |
| 3 nodes (43) | 0 / – / – | 0 / – / – | 0.256 / 38.8 / 0.31 |
| all surveyed (168) | 0.321 / 71.9 / 0.63 | 0.500 / 57.1 / 0.67 | 0.304 / 137.0 / 0.34 |
| has hand-measured node (97) | 0.268 / 124.9 / 0.37 | 0.371 / 107.6 / 0.33 | 0.289 / 123.6 / 0.46 |

The every-slice table for all five candidates comes out of the CLI
(`--markdown`).

## Production findings (reported, not fixed)

The bake-off changed no production behaviour. These findings come from it:

1. **The GCC-PHAT front-end loses most point-network calls.** The repaired
   front-end raises recall from 0.302 to 0.453 and cuts single-bird p90 error
   from 66.6 m to 7.8 m with the same solver. Lifting the lag cap alone does
   little (0.313), so the gain comes from the two causes below. A
   default-suite `#[ignore]`d test,
   `easy_scene_is_localized_by_production`, reproduces both on six surveyed
   GPS nodes over a 60 m site with one loud FM sweep.
   - **Band ceiling.** The band is 50–3430 Hz (`localization_band_hz`, chosen
     to stop PHAT reversing bearings on the compact tetra). It discards most
     of the energy of a 3.4–7 kHz call. On that scene every pair lag came out
     wrong. With the band widened to 11 kHz and the window tapered, the
     Cartesian solve landed 0.4 m from the truth.
   - **Zero-lag bias from a rectangular cut.** The event window is centred on
     the trigger, so it often ends while a call is still loud at every node.
     That cut is the same broadband step at the same sample index on every
     channel, and PHAT whitening makes it the strongest feature. On the easy
     scene with the full band, every cross-node pair read a lag of under
     0.2 ms against true lags of 3–70 ms. A plain time-domain
     cross-correlation found the true lags, and so did a 20 % Tukey taper.
     A compact array never shows this, because its true lags are near zero
     anyway. The cause is `gcc_phat.rs::phat_correlation` with no taper (the
     Python `localization.gcc_phat` has none either), fed by
     `localization_event_span.rs` windows.
2. **Three-node point networks never localize.** The sidecar SRP needs 4
   channels (`srp_phat_degraded_partial_array`) and `rust_tdoa_solve.py`
   needs 4 sensors. That is 35 of the 200 held-out sites, so 3 nodes and a
   2-D fix are unsupported.
3. **Hand-measured position accuracy never reaches the published
   covariance on the default handoff path.** Coverage drops from 0.63 at
   all-surveyed sites to 0.37 at sites with a hand-measured node.
   - `position_std_m` reaches the sidecar provenance and the Python-ingest
     pair weights (`tdoa_measurements.py`).
   - Neither the sidecar SRP covariance nor the `rust_tdoa_solve.py`
     re-solve adds it.
   - The owner decision ("stated accuracy widening the error") is therefore
     not met on this path. Onset clustering with the propagation added here
     holds 0.46 on those sites.
4. **Production covariance under-covers even when clocks and positions are
   honest.** Coverage is 0.57 with all-GPS clocks and 0.63 at all-surveyed
   sites, against a stated 0.95. The pair-weight clock policy (P9) does not
   make up for multipath and two-caller confusion.
5. **The re-solve can diverge to 10⁸–10⁹ m on bad pairs.** The solver labels
   these `range_bearing_projected`, so the handoff gate demotes them. In
   the held-out run, 28 production solves were gated for an unobservable
   range, and the divergent ones are among them. They are harmless today,
   but only because of the gate.

## What would change the verdict

- A repaired production front-end (band and taper) is the obvious next
  candidate for a decision of its own. It would need re-validating on
  tetra and planar arrays, where the 3430 Hz ceiling came from.
- Onset clustering is worth another look only for 3-node sites and tonal
  calls, and only after its covariance is honest: 0.38 coverage, with the
  clock and position propagation already included.
